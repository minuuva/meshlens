"""The torch reduction must agree exactly with the tested numpy reference.

`extract.block_mean_torch` exists purely for memory reasons. If it ever diverges
from `faces.face_attention_batch` the experiments silently measure something
else, so the equivalence is pinned here rather than assumed.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from meshlens.extract import block_mean_torch, normalize_faces, sink_per_head
from meshlens.faces import TOKENS_PER_FACE, causal_keep_mask, face_attention_batch


def test_torch_block_mean_matches_numpy_reference():
    rng = np.random.default_rng(31)
    F, H = 12, 6
    T = 2 + TOKENS_PER_FACE * F
    attn = rng.random((H, T, T)).astype(np.float32)

    ref = face_attention_batch(attn, F, exclude_sink=False)
    got = block_mean_torch(torch.from_numpy(attn), F)
    assert got.shape == (H, F, F)
    assert np.allclose(got, ref, atol=1e-6)


def test_chunking_does_not_change_the_result():
    rng = np.random.default_rng(32)
    F, H = 7, 9  # head count deliberately not a multiple of the chunk size
    T = 2 + TOKENS_PER_FACE * F
    attn = torch.from_numpy(rng.random((H, T, T)).astype(np.float32))
    whole = block_mean_torch(attn, F, head_chunk=H)
    chunked = block_mean_torch(attn, F, head_chunk=4)
    assert np.allclose(whole, chunked)


def test_normalize_matches_the_reference_end_to_end():
    rng = np.random.default_rng(33)
    F, H = 10, 5
    T = 2 + TOKENS_PER_FACE * F
    attn = rng.random((H, T, T)).astype(np.float32)

    ref = face_attention_batch(attn, F)  # masked and renormalized
    got = normalize_faces(block_mean_torch(torch.from_numpy(attn), F), causal_keep_mask(F))
    assert np.allclose(got, ref, atol=1e-6)
    assert np.allclose(got.sum(axis=2)[:, 2:], 1.0)


def test_sink_reads_bos_plus_face_zero_from_the_final_face_rows():
    F, H = 6, 3
    T = 2 + TOKENS_PER_FACE * F
    attn = torch.zeros((H, T, T))
    q0 = 1 + TOKENS_PER_FACE * (F - 1)
    # put all of the final face's mass on BOS and face 0, i.e. a pure sink head
    attn[0, q0 : q0 + TOKENS_PER_FACE, : 1 + TOKENS_PER_FACE] = 1.0 / (1 + TOKENS_PER_FACE)
    # and a head that puts everything on a content face instead
    ks = 1 + TOKENS_PER_FACE * 3
    attn[1, q0 : q0 + TOKENS_PER_FACE, ks : ks + TOKENS_PER_FACE] = 1.0 / TOKENS_PER_FACE

    sink = sink_per_head(attn, F)
    assert sink[0] == pytest.approx(1.0)
    assert sink[1] == pytest.approx(0.0)
    assert sink[2] == pytest.approx(0.0)
