"""Tests for the sampling and the Experiment 2 manipulation.

`resort_faces` *is* Experiment 2. If it reordered vertices, dropped faces, or
failed to decorrelate height from position, E2 would measure something other
than what the preregistration claims, and nothing downstream would notice.
"""

import numpy as np
import pytest
from scipy.stats import spearmanr

from meshlens.data import MAX_FACES, MIN_FACES, resort_faces, sample_splits


class FakeChair(dict):
    """Minimal stand-in for a dataset item: plain arrays, no torch."""


def make_chair(n_faces, seed=0):
    rng = np.random.default_rng(seed)
    vertices = rng.random((n_faces * 3, 3)) * 2 - 1
    faces = np.arange(n_faces * 3).reshape(n_faces, 3)
    # sort faces by the vertical axis, as MeshXL's preprocessing does
    order = np.argsort(vertices[faces].mean(axis=1)[:, 1])
    return FakeChair(vertices=vertices, faces=faces[order])


def test_splits_are_disjoint_deterministic_and_within_bounds():
    data = [make_chair(n, seed=i) for i, n in enumerate([10, 60, 200, 500, 900, 300] * 40)]
    primary, holdout = sample_splits(data, n_primary=20, n_holdout=20)

    assert len(primary) == 20 and len(holdout) == 20
    assert not set(primary) & set(holdout), "held-out set must not overlap the primary set"
    again = sample_splits(data, n_primary=20, n_holdout=20)
    assert np.array_equal(primary, again[0]) and np.array_equal(holdout, again[1])

    for i in np.concatenate([primary, holdout]):
        assert MIN_FACES <= len(data[i]["faces"]) <= MAX_FACES


def test_holdout_does_not_shift_when_the_primary_size_changes():
    # The held-out set is drawn from one permutation, so asking for a different
    # primary size must not silently reshuffle which chairs were set aside.
    data = [make_chair(n, seed=i) for i, n in enumerate([100, 200, 300, 400] * 60)]
    perm_a, _ = sample_splits(data, n_primary=10, n_holdout=10)
    perm_b, _ = sample_splits(data, n_primary=30, n_holdout=10)
    assert np.array_equal(perm_a, perm_b[:10]), "the draw order must be stable"


def test_resort_preserves_geometry_exactly():
    chair = make_chair(40, seed=3)
    out = resort_faces(chair, primary_axis=0)

    assert np.array_equal(out["vertices"], chair["vertices"]), "vertices must not move"
    assert out["faces"].shape == chair["faces"].shape
    # same multiset of faces, just reordered; each face's own vertex order intact
    before = sorted(map(tuple, np.asarray(chair["faces"])))
    after = sorted(map(tuple, np.asarray(out["faces"])))
    assert before == after, "re-sorting must permute faces, not rewrite them"


def test_resort_moves_the_confound_from_height_to_x():
    # The whole point of Experiment 2. Built on a chair sorted by height, as the
    # real data is; after re-sorting, index must track x and stop tracking y.
    chair = make_chair(300, seed=4)
    v, f = np.asarray(chair["vertices"]), np.asarray(chair["faces"])
    before = v[f].mean(axis=1)
    idx = np.arange(len(before))
    assert spearmanr(idx, before[:, 1]).statistic > 0.95, "fixture should start height-sorted"

    after = v[np.asarray(resort_faces(chair, primary_axis=0)["faces"])].mean(axis=1)
    assert spearmanr(idx, after[:, 0]).statistic == pytest.approx(1.0, abs=1e-9)
    assert abs(spearmanr(idx, after[:, 1]).statistic) < 0.2, "height must be decorrelated"


def test_resort_is_a_total_order_with_no_ties_left_to_chance():
    # lexsort must break ties on the secondary and tertiary axes, otherwise the
    # ordering depends on the input order and the manipulation is not reproducible
    vertices = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] * 4)
    faces = np.arange(12).reshape(4, 3)
    chair = FakeChair(vertices=vertices, faces=faces)
    a = resort_faces(chair, primary_axis=0)["faces"]
    shuffled = FakeChair(vertices=vertices, faces=faces[::-1])
    b = resort_faces(shuffled, primary_axis=0)["faces"]
    assert sorted(map(tuple, a)) == sorted(map(tuple, b))


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_resort_on_any_axis_makes_that_axis_monotone(axis):
    chair = make_chair(120, seed=5)
    v = np.asarray(chair["vertices"])
    out = v[np.asarray(resort_faces(chair, primary_axis=axis)["faces"])].mean(axis=1)
    assert np.all(np.diff(out[:, axis]) >= -1e-12)
