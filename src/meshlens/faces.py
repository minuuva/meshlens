"""Token/face bookkeeping for MeshXL sequences.

Sequence layout is `<bos> f0 f1 ... f_{F-1} <eos>`, where each face occupies
exactly 9 consecutive tokens: three vertices, each x then y then z. So face `f`
spans tokens `[1 + 9f, 10 + 9f)`.

The sink region is token 0 (BOS) together with face 0, following the definition
frozen in docs/prereg_round1.md. Face 0 is therefore never a key in any
statistic, and never a query.
"""

import numpy as np

N_DISCRETE = 128
BOS, EOS, PAD = 128, 129, 130
TOKENS_PER_FACE = 9
VERTICAL_AXIS = 1  # ShapeNet is y-up; confirmed by scripts/confound_check.py


def n_faces(seq_len):
    """Number of whole faces in a `<bos> ... <eos>` sequence of this length."""
    return (seq_len - 2) // TOKENS_PER_FACE


def face_span(f):
    """Token slice occupied by face `f`."""
    start = 1 + TOKENS_PER_FACE * f
    return start, start + TOKENS_PER_FACE


def undiscretize(t, num_discrete=N_DISCRETE, lo=-1.0, hi=1.0):
    """MeshXL's own inverse discretization, from models/mesh_xl/tokenizer.py.

    Deliberately not the `t / 127 * 2 - 1` variant used in some of the earlier
    notebooks: that one is off by a bin and disagrees with what the model was
    trained on. Both are affine in `t`, so correlations are unaffected, but
    absolute distances and areas are not.
    """
    return (np.asarray(t, dtype=np.float64) + 0.5) / num_discrete * (hi - lo) + lo


def centroids_from_tokens(tokens):
    """Face centroids in the coordinate space the model actually sees.

    Decodes the token stream rather than reading the source mesh, so the
    geometry here is exactly the quantized geometry the network was given.
    Returns an (F, 3) array in XYZ order.
    """
    tokens = np.asarray(tokens)
    F = n_faces(len(tokens))
    body = tokens[1 : 1 + TOKENS_PER_FACE * F].reshape(F, 3, 3)
    return undiscretize(body).mean(axis=1)


def face_attention(attn, n_face, exclude_sink=True):
    """Reduce a token-level attention matrix to faces by 9x9 block mean.

    `attn` is (T, T) for a single head. Returns (F, F) where entry (q, k) is the
    mean attention from face q's nine query rows onto face k's nine key columns.

    With `exclude_sink`, every row is renormalized over its causal non-sink keys
    (faces 1..q-1), so the returned row sums to 1 where any such key exists and
    to 0 for q <= 1. That renormalization is what makes the statistic a question
    about *where the non-sink attention goes*, rather than a question about how
    much attention the sink took.
    """
    return face_attention_batch(np.asarray(attn)[None], n_face, exclude_sink)[0]


def causal_keep_mask(n_face):
    """(F, F) mask of admissible (query, key) face pairs: strictly causal, face 0 excluded."""
    keep = np.tril(np.ones((n_face, n_face), dtype=bool), k=-1)
    keep[:, 0] = False  # face 0 is sink
    keep[:2, :] = False  # queries 0 and 1 have no admissible keys
    return keep


def face_attention_batch(attn, n_face, exclude_sink=True, keep=None):
    """`face_attention` for all heads of a layer at once.

    `attn` is (H, T, T). Vectorized over heads because the per-head Python loop
    it replaces ran 24 x 32 times per mesh and dominated runtime. `keep` may be
    passed in to reuse the mask across layers.
    """
    attn = np.asarray(attn, dtype=np.float64)
    H = attn.shape[0]
    end = 1 + TOKENS_PER_FACE * n_face
    faces = attn[:, 1:end, 1:end].reshape(
        H, n_face, TOKENS_PER_FACE, n_face, TOKENS_PER_FACE
    ).mean(axis=(2, 4))

    if not exclude_sink:
        return faces

    if keep is None:
        keep = causal_keep_mask(n_face)
    faces = np.where(keep[None], faces, 0.0)
    total = faces.sum(axis=2, keepdims=True)
    return np.divide(faces, total, out=np.zeros_like(faces), where=total > 0)


def causal_pairs(centroids, min_query_face=2):
    """Flatten the causal non-sink (q, k) pairs into aligned index and distance arrays.

    Returns (q_idx, k_idx, d_seq, d_3d). Face 0 is excluded as sink and q must
    exceed 1, matching `face_attention`.
    """
    F = len(centroids)
    qs, ks = [], []
    for q in range(min_query_face, F):
        k = np.arange(1, q)
        qs.append(np.full(len(k), q))
        ks.append(k)
    if not qs:
        empty = np.zeros(0)
        return empty.astype(int), empty.astype(int), empty, empty
    q_idx = np.concatenate(qs)
    k_idx = np.concatenate(ks)
    d_seq = (q_idx - k_idx).astype(np.float64)
    d_3d = np.linalg.norm(centroids[q_idx] - centroids[k_idx], axis=1)
    return q_idx, k_idx, d_seq, d_3d
