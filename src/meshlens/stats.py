"""Partial correlation and chair-level bootstrap.

The estimator is computed per chair and then aggregated across chairs, rather
than by pooling every (q, k) pair into one sample. Pair count grows with the
square of face count, so pooling would let an 800-face chair contribute sixteen
times the weight of a 200-face chair. Aggregating per chair gives every mesh one
vote and makes the bootstrap over chairs exact. See the amendment note in
docs/prereg_round1.md.
"""

import numpy as np
from scipy.stats import rankdata


def _pearson(x, y):
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / denom) if denom > 0 else np.nan


def partial_spearman(x, y, z):
    """Spearman correlation of x and y with z partialled out of both.

    Ranks first, then applies the standard first-order partial correlation
    identity to the ranked variables.
    """
    if len(x) < 4:
        return np.nan
    rx, ry, rz = (rankdata(v).astype(np.float64) for v in (x, y, z))
    r_xy, r_xz, r_yz = _pearson(rx, ry), _pearson(rx, rz), _pearson(ry, rz)
    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if not np.isfinite(denom) or denom < 1e-12:
        return np.nan
    return float((r_xy - r_xz * r_yz) / denom)


def head_partials(a, d_seq, d_3d):
    """The two numbers Experiment 1 is about, for one head on one chair.

    Distances are negated so that both correlations read as "attends more to
    closer", making a positive value mean proximity-seeking on either axis.
    """
    return (
        partial_spearman(a, -d_3d, -d_seq),  # spatial, controlling for sequence
        partial_spearman(a, -d_seq, -d_3d),  # sequence, controlling for space
    )


def head_partials_batch(a_rows, d_seq, d_3d):
    """`head_partials` for many heads sharing one chair's pair geometry.

    The nuisance ranks and their mutual correlation depend only on the mesh, not
    on the head, so they are computed once here rather than 768 times. `a_rows`
    is (n_heads, n_pairs); returns two (n_heads,) arrays.
    """
    a_rows = np.asarray(a_rows, dtype=np.float64)
    n_heads, n_pairs = a_rows.shape
    if n_pairs < 4:
        return np.full(n_heads, np.nan), np.full(n_heads, np.nan)

    r_space = rankdata(-np.asarray(d_3d, dtype=np.float64))
    r_seq = rankdata(-np.asarray(d_seq, dtype=np.float64))
    r_ss = _pearson(r_space, r_seq)
    one_minus_ss = 1 - r_ss**2

    spatial = np.full(n_heads, np.nan)
    sequence = np.full(n_heads, np.nan)
    for h in range(n_heads):
        ra = rankdata(a_rows[h])
        r_a_space = _pearson(ra, r_space)
        r_a_seq = _pearson(ra, r_seq)
        d_sp = np.sqrt((1 - r_a_seq**2) * one_minus_ss)
        d_sq = np.sqrt((1 - r_a_space**2) * one_minus_ss)
        if np.isfinite(d_sp) and d_sp > 1e-12:
            spatial[h] = (r_a_space - r_a_seq * r_ss) / d_sp
        if np.isfinite(d_sq) and d_sq > 1e-12:
            sequence[h] = (r_a_seq - r_a_space * r_ss) / d_sq
    return spatial, sequence


def bootstrap_ci(per_chair, statistic=np.nanmedian, n_boot=1000, seed=0, alpha=0.05):
    """Percentile CI for a statistic, resampling whole chairs with replacement."""
    per_chair = np.asarray(per_chair, dtype=np.float64)
    per_chair = per_chair[np.isfinite(per_chair)]
    if len(per_chair) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_chair), size=(n_boot, len(per_chair)))
    draws = statistic(per_chair[idx], axis=1)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(statistic(per_chair)), float(lo), float(hi)
