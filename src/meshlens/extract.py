"""Torch-side reduction of attention to face level.

Kept separate from `faces.py`, which stays numpy-only so the tests and the
dataset analyses never need torch.

Why this exists: for an 800-face mesh the token attention of one layer is
(32, 7202, 7202). Converting that to float64 in numpy, as the reference path
does, allocates about 11 GB per layer and stalls the machine. Reducing in torch
at float32 and converting only the (32, F, F) result keeps peak allocation to a
few hundred megabytes, and `avg_pool2d` is exactly a 9x9 block mean.
"""

import numpy as np
import torch
import torch.nn.functional as F_nn

from .faces import TOKENS_PER_FACE


def block_mean_torch(attn, n_face, head_chunk=8):
    """(H, T, T) float32 torch -> (H, F, F) float64 numpy, by 9x9 block mean.

    Heads are processed in chunks so that no single allocation is proportional
    to H * T^2.
    """
    end = 1 + TOKENS_PER_FACE * n_face
    out = np.empty((attn.shape[0], n_face, n_face), dtype=np.float64)
    for start in range(0, attn.shape[0], head_chunk):
        stop = min(start + head_chunk, attn.shape[0])
        block = attn[start:stop, 1:end, 1:end].unsqueeze(1)  # (c, 1, 9F, 9F)
        pooled = F_nn.avg_pool2d(block, TOKENS_PER_FACE).squeeze(1)  # (c, F, F)
        out[start:stop] = pooled.double().numpy()
    return out


def sink_per_head(attn, n_face):
    """Mean attention onto BOS plus face 0, from the final face's query rows.

    The sink definition frozen in docs/prereg_round1.md.
    """
    q0 = 1 + TOKENS_PER_FACE * (n_face - 1)
    rows = attn[:, q0 : q0 + TOKENS_PER_FACE, : 1 + TOKENS_PER_FACE]
    return rows.sum(dim=2).mean(dim=1).double().numpy()


def normalize_faces(faces, keep):
    """Apply the causal non-sink mask and renormalize rows, in place-ish."""
    faces = np.where(keep[None], faces, 0.0)
    total = faces.sum(axis=2, keepdims=True)
    return np.divide(faces, total, out=np.zeros_like(faces), where=total > 0)
