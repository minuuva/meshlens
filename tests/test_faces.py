"""Tests for the token/face indexing every downstream statistic rests on.

The `1 + 9f` offset is the single most load-bearing piece of arithmetic in this
project and was never covered before, so these tests are deliberately built to
fail loudly on an off-by-one rather than to merely exercise the code.
"""

import numpy as np
import pytest

from meshlens.faces import (
    BOS,
    EOS,
    TOKENS_PER_FACE,
    causal_pairs,
    centroids_from_tokens,
    face_attention,
    face_span,
    n_faces,
    undiscretize,
)


def make_sequence(face_tokens):
    """`<bos> ...faces... <eos>` from a list of 9-token faces."""
    return np.array([BOS, *np.ravel(face_tokens), EOS])


def test_n_faces_and_span_are_consistent():
    for F in (1, 2, 17, 468):
        seq = make_sequence(np.zeros((F, TOKENS_PER_FACE), dtype=int))
        assert n_faces(len(seq)) == F
    assert face_span(0) == (1, 10)
    assert face_span(1) == (10, 19)
    # spans must tile the body exactly, leaving BOS at 0 and nothing overlapping
    ends = [face_span(f)[1] for f in range(5)]
    starts = [face_span(f)[0] for f in range(5)]
    assert starts[1:] == ends[:-1]
    assert starts[0] == 1, "face 0 must start after BOS, not at it"


def test_undiscretize_matches_meshxl_not_the_notebook_variant():
    # MeshXL: (t + 0.5)/128*2 - 1. The notebooks used t/127*2 - 1.
    assert undiscretize(0) == pytest.approx(-1 + 1 / 128)
    assert undiscretize(127) == pytest.approx(1 - 1 / 128)
    assert undiscretize(64) == pytest.approx(0.0078125)
    assert undiscretize(0) != pytest.approx(-1.0), "this is the off-by-a-bin variant"


def test_centroids_decode_the_right_tokens():
    # face 0 sits at the low corner, face 1 at the high corner
    face0 = np.zeros(TOKENS_PER_FACE, dtype=int)
    face1 = np.full(TOKENS_PER_FACE, 127, dtype=int)
    seq = make_sequence([face0, face1])
    c = centroids_from_tokens(seq)
    assert c.shape == (2, 3)
    assert np.allclose(c[0], undiscretize(0))
    assert np.allclose(c[1], undiscretize(127))
    # an off-by-one read would pull BOS(128) or EOS into a face and break this
    assert c[0].max() < 0, "face 0 decoded high; indexing is shifted"
    assert c[1].min() > 0, "face 1 decoded low; indexing is shifted"


def test_centroid_averages_the_three_vertices():
    face = np.array([0, 0, 0, 127, 127, 127, 0, 0, 0])  # verts low, high, low
    c = centroids_from_tokens(make_sequence([face]))
    expected = (undiscretize(0) * 2 + undiscretize(127)) / 3
    assert np.allclose(c[0], expected)


def _attn_with_block(F, q, k, value):
    T = 2 + TOKENS_PER_FACE * F
    a = np.zeros((T, T))
    qs, qe = face_span(q)
    ks, ke = face_span(k)
    a[qs:qe, ks:ke] = value
    return a


def test_face_attention_locates_the_right_block():
    a = _attn_with_block(F=5, q=3, k=1, value=0.5)
    raw = face_attention(a, 5, exclude_sink=False)
    assert raw[3, 1] == pytest.approx(0.5)
    assert raw.sum() == pytest.approx(0.5), "mass leaked into other face blocks"


def test_face_attention_renormalizes_over_causal_non_sink_keys():
    a = _attn_with_block(F=5, q=3, k=1, value=0.5)
    out = face_attention(a, 5)
    assert out[3, 1] == pytest.approx(1.0), "sole causal key should absorb the row"
    assert out[3].sum() == pytest.approx(1.0)


def test_face_attention_excludes_sink_and_non_causal_keys():
    F = 6
    a = np.zeros((2 + TOKENS_PER_FACE * F, 2 + TOKENS_PER_FACE * F))
    for k in (0, 2, 4, 5):  # face 0 is sink; 4 and 5 are not causal for q=3
        qs, qe = face_span(3)
        ks, ke = face_span(k)
        a[qs:qe, ks:ke] = 1.0
    out = face_attention(a, F)
    assert out[3, 0] == 0.0, "face 0 is the sink and must be dropped"
    assert out[3, 4] == 0.0 and out[3, 5] == 0.0, "future faces must be dropped"
    assert out[3, 2] == pytest.approx(1.0)
    # queries 0 and 1 have no admissible keys at all
    assert out[0].sum() == 0.0 and out[1].sum() == 0.0


def test_face_attention_rows_are_distributions():
    rng = np.random.default_rng(0)
    F = 8
    T = 2 + TOKENS_PER_FACE * F
    a = rng.random((T, T))
    out = face_attention(a, F)
    sums = out.sum(axis=1)
    assert np.allclose(sums[2:], 1.0)
    assert np.allclose(sums[:2], 0.0)


def test_causal_pairs_match_face_attention_support():
    centroids = np.random.default_rng(1).random((7, 3))
    q, k, d_seq, d_3d = causal_pairs(centroids)
    assert (k >= 1).all(), "face 0 must never appear as a key"
    assert (q >= 2).all()
    assert (k < q).all(), "pairs must be strictly causal"
    assert len(q) == sum(len(range(1, qq)) for qq in range(2, 7))
    assert np.allclose(d_seq, q - k)
    assert d_3d[0] == pytest.approx(np.linalg.norm(centroids[q[0]] - centroids[k[0]]))


def test_causal_pairs_degenerate_mesh():
    q, k, d_seq, d_3d = causal_pairs(np.zeros((2, 3)))
    assert len(q) == 0 and len(d_3d) == 0


def test_batched_face_attention_matches_the_single_head_version():
    # face_attention now delegates to the batched path; this pins the batched
    # result against an independent, obviously-correct computation of the same
    # thing, so the vectorization cannot silently change semantics.
    from meshlens.faces import causal_keep_mask, face_attention_batch

    rng = np.random.default_rng(21)
    F, H = 9, 4
    T = 2 + TOKENS_PER_FACE * F
    attn = rng.random((H, T, T))

    batched = face_attention_batch(attn, F)
    for h in range(H):
        naive = np.zeros((F, F))
        for q in range(F):
            for k in range(F):
                qs, qe = face_span(q)
                ks, ke = face_span(k)
                naive[q, k] = attn[h, qs:qe, ks:ke].mean()
        keep = causal_keep_mask(F)
        naive = np.where(keep, naive, 0.0)
        total = naive.sum(axis=1, keepdims=True)
        naive = np.divide(naive, total, out=np.zeros_like(naive), where=total > 0)
        assert np.allclose(batched[h], naive)


def test_causal_keep_mask_agrees_with_causal_pairs():
    from meshlens.faces import causal_keep_mask

    F = 11
    keep = causal_keep_mask(F)
    q, k, _, _ = causal_pairs(np.zeros((F, 3)))
    expected = np.zeros((F, F), dtype=bool)
    expected[q, k] = True
    assert np.array_equal(keep, expected), "mask and pair enumeration must select the same cells"
