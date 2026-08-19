"""Tests for the partial correlation and bootstrap, against known answers.

The whole of Experiment 1 is one number produced by `partial_spearman`, so it is
checked here on cases where the right answer is known by construction --
especially the case the experiment exists for: two variables that look strongly
related until a shared driver is removed.
"""

import numpy as np
import pytest

from meshlens.stats import bootstrap_ci, head_partials, partial_spearman


def test_spurious_correlation_vanishes_when_the_driver_is_partialled_out():
    # This is the confound in miniature: x and y are correlated only because
    # both track z. Height and sequence position stand in for x and z.
    rng = np.random.default_rng(0)
    z = np.arange(500.0)
    x = z + rng.normal(0, 20, 500)
    y = z + rng.normal(0, 20, 500)
    raw = np.corrcoef(x, y)[0, 1]
    assert raw > 0.8, "setup should look strongly correlated before control"
    assert abs(partial_spearman(x, y, z)) < 0.2


def test_genuine_correlation_survives_the_control():
    rng = np.random.default_rng(1)
    w = rng.normal(size=500)
    z = rng.normal(size=500)  # independent nuisance
    x = w + rng.normal(0, 0.1, 500)
    y = w + rng.normal(0, 0.1, 500)
    assert partial_spearman(x, y, z) > 0.9


def test_is_monotone_invariant_as_a_rank_statistic():
    rng = np.random.default_rng(2)
    x, y, z = (rng.normal(size=300) for _ in range(3))
    base = partial_spearman(x, y, z)
    # any strictly increasing transform must leave a rank statistic alone
    assert partial_spearman(np.exp(x), np.arctan(y), z**3) == pytest.approx(base)


def test_degenerate_inputs_return_nan_not_garbage():
    assert np.isnan(partial_spearman([1.0, 2.0], [1.0, 2.0], [1.0, 2.0]))
    z = np.arange(50.0)
    assert np.isnan(partial_spearman(z, np.arange(50.0), z))  # x identical to z


def test_partial_is_undefined_when_x_is_a_deterministic_function_of_z():
    # Not a defensive nicety: a head whose attention were an exact function of
    # distance would carry no residual variation to attribute, and reporting a
    # number there would be inventing one. nan is the honest answer, and
    # bootstrap_ci drops it.
    d = np.random.default_rng(9).random(200)
    assert np.isnan(partial_spearman(1.0 / (1.0 + d), np.arange(200.0), -d))


def test_head_partials_sign_convention_is_closer_is_positive():
    # Attention decaying with 3D distance, with enough noise that it is not a
    # deterministic function of it -- as real attention never is.
    rng = np.random.default_rng(3)
    d_3d = rng.random(400) * 2
    d_seq = rng.random(400) * 100
    a = 1.0 / (1.0 + d_3d) + rng.normal(0, 0.05, 400)
    rho_spatial, rho_seq = head_partials(a, d_seq, d_3d)
    assert rho_spatial > 0.9, "proximity-seeking in space must read positive"
    assert abs(rho_seq) < 0.2

    # and the mirror image: attention that decays with recency only
    a = 1.0 / (1.0 + d_seq) + rng.normal(0, 0.002, 400)
    rho_spatial, rho_seq = head_partials(a, d_seq, d_3d)
    assert rho_seq > 0.9
    assert abs(rho_spatial) < 0.2


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(4)
    per_chair = rng.normal(0.3, 0.1, 100)
    point, lo, hi = bootstrap_ci(per_chair, seed=0)
    assert lo < point < hi
    assert lo > 0.2 and hi < 0.4


def test_bootstrap_ci_is_deterministic_under_a_fixed_seed():
    v = np.random.default_rng(5).normal(size=60)
    assert bootstrap_ci(v, seed=7) == bootstrap_ci(v, seed=7)


def test_bootstrap_ci_ignores_nan_heads():
    v = np.array([0.5, 0.5, np.nan, 0.5, 0.5])
    point, lo, hi = bootstrap_ci(v, seed=0)
    assert point == pytest.approx(0.5)
    assert np.isfinite(lo) and np.isfinite(hi)


def test_batched_partials_match_the_reference_implementation():
    # head_partials_batch hoists the nuisance ranks out of the head loop for
    # speed. It must agree with the straightforward version to floating point.
    from meshlens.stats import head_partials_batch

    rng = np.random.default_rng(11)
    n_pairs = 800
    d_3d = rng.random(n_pairs) * 2
    d_seq = rng.random(n_pairs) * 120
    a_rows = np.stack([
        1.0 / (1.0 + d_3d) + rng.normal(0, 0.05, n_pairs),
        1.0 / (1.0 + d_seq) + rng.normal(0, 0.002, n_pairs),
        rng.normal(size=n_pairs),
    ])
    batch_sp, batch_sq = head_partials_batch(a_rows, d_seq, d_3d)
    for h in range(len(a_rows)):
        ref_sp, ref_sq = head_partials(a_rows[h], d_seq, d_3d)
        assert batch_sp[h] == pytest.approx(ref_sp, abs=1e-9)
        assert batch_sq[h] == pytest.approx(ref_sq, abs=1e-9)


def test_batched_partials_handle_a_degenerate_head():
    from meshlens.stats import head_partials_batch

    rng = np.random.default_rng(12)
    d_3d = rng.random(300)
    d_seq = rng.random(300) * 50
    a_rows = np.stack([rng.normal(size=300), np.zeros(300)])  # second head is flat
    sp, sq = head_partials_batch(a_rows, d_seq, d_3d)
    assert np.isfinite(sp[0]) and np.isfinite(sq[0])
    assert np.isnan(sp[1]) and np.isnan(sq[1]), "a constant head has no signal to attribute"
