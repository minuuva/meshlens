"""Tests for the round 2 probe: topology, and the multi-control partial.

Experiment 4 asks whether attention tracks adjacency *after* both sequence
distance and 3D distance are removed. Controlling for one and not the other
would leave the uncontrolled one free to explain the whole result, so the
multi-control path is the load-bearing piece and is checked against a case built
to fail exactly that way.
"""

import numpy as np
import pytest

from meshlens.faces import (
    N_DISCRETE,
    TOKENS_PER_FACE,
    adjacency_matrix,
    causal_pairs,
    shared_vertex_counts,
    vertex_keys,
)
from meshlens.stats import (
    head_partials_adjacency_batch,
    partial_spearman,
    partial_spearman_multi,
)
from meshlens.verdict import ADJACENCY_RATE, E4_SUPPORT
from tests.test_faces import make_sequence


class TestMultiControlPartial:
    def test_reduces_to_the_first_order_version_with_one_control(self):
        rng = np.random.default_rng(41)
        x, y, z = (rng.normal(size=400) for _ in range(3))
        assert partial_spearman_multi(x, y, [z]) == pytest.approx(
            partial_spearman(x, y, z), abs=1e-9
        )

    def test_one_control_is_not_enough_when_two_drivers_are_present(self):
        rng = np.random.default_rng(42)
        z1, z2 = rng.normal(size=600), rng.normal(size=600)
        x = z1 + z2 + rng.normal(0, 0.3, 600)
        y = z1 + z2 + rng.normal(0, 0.3, 600)
        one = abs(partial_spearman(x, y, z1))
        both = abs(partial_spearman_multi(x, y, [z1, z2]))
        assert both < one, "adding the second control must reduce the spurious effect"

    def test_known_limitation_additively_shared_continuous_drivers(self):
        """Rank-linear residualization does not fully remove this case.

        With x and y both equal to z1 + z2 plus noise, the residuals stay
        correlated at roughly 0.46. The confound is additive in raw space, but
        the statistic works in rank space, and rank(z1 + z2) is not a linear
        function of rank(z1) and rank(z2). Pinned here so the limitation is a
        recorded property rather than a surprise.

        Experiment 4 is not this case -- see the calibration test below -- but
        any future use of this function on two continuous variables should be.
        """
        rng = np.random.default_rng(42)
        z1, z2 = rng.normal(size=600), rng.normal(size=600)
        x = z1 + z2 + rng.normal(0, 0.3, 600)
        y = z1 + z2 + rng.normal(0, 0.3, 600)
        assert abs(partial_spearman_multi(x, y, [z1, z2])) > 0.3

    @staticmethod
    def _e4_shaped(rng, n=20000, rate=ADJACENCY_RATE, multiplier=1.0):
        """Synthetic pairs matching E4: sparse binary adjacency, correlated controls.

        `rate` is the measured fraction of causal face pairs that are adjacent.
        Getting it right matters: an earlier version of this fixture used a rate
        near 0.005%, which made the probe look hopelessly underpowered when the
        method was fine.
        """
        d_seq = rng.integers(1, 400, n).astype(float)
        d_3d = np.abs(rng.normal(0, 0.4, n)) + 0.02 * d_seq
        adjacency = (rng.random(n) < rate).astype(float)
        attention = np.exp(-3 * d_3d) / (1 + d_seq) * np.exp(rng.normal(0, 0.4, n))
        return attention * (1 + (multiplier - 1) * adjacency), adjacency, [d_seq, d_3d]

    def test_calibrated_on_experiment_4_data_shape(self):
        """The property E4 actually depends on: no false positive.

        Attention here depends on the controls alone. The estimate must stay
        near zero, because anything else would manufacture topological structure
        out of recency and proximity.
        """
        rng = np.random.default_rng(7)
        worst = max(
            abs(partial_spearman_multi(*self._e4_shaped(rng, multiplier=1.0)))
            for _ in range(10)
        )
        assert worst < 0.05, f"null inflated to {worst:.3f}; E4 would read noise as topology"

    def test_the_support_threshold_corresponds_to_a_reachable_effect(self):
        """Power check: rho_adj >= 0.15 must not require an absurd effect.

        A rare binary predictor attenuates rank correlation, so the threshold
        could in principle have been unreachable. At the measured adjacency rate
        a doubling of attention on adjacent pairs clears it, which makes the
        preregistered band an interpretable effect size rather than a wish.
        """
        rng = np.random.default_rng(11)
        doubled = np.median([
            partial_spearman_multi(*self._e4_shaped(rng, multiplier=2.0)) for _ in range(5)
        ])
        assert doubled >= E4_SUPPORT, f"2x attention gives only rho={doubled:.3f}"

    def test_batched_matches_the_reference(self):
        rng = np.random.default_rng(44)
        n = 500
        c1, c2 = rng.normal(size=n), rng.normal(size=n)
        adj = (rng.random(n) < 0.1).astype(float)
        rows = np.stack([adj + rng.normal(0, 0.5, n), rng.normal(size=n), c1 + adj])
        got = head_partials_adjacency_batch(rows, adj, [c1, c2])
        for h in range(len(rows)):
            assert got[h] == pytest.approx(
                partial_spearman_multi(rows[h], adj, [c1, c2]), abs=1e-8
            )


class TestAdjacency:
    def test_matrix_is_symmetric_with_no_self_pairs(self):
        # the dataset ships both directions and self-pairs; both are normalized
        edges = [[0, 0], [0, 1], [1, 0], [1, 2], [3, 3]]
        adj = adjacency_matrix(edges, 5)
        assert adj[0, 1] and adj[1, 0]
        assert adj[1, 2] and adj[2, 1]
        assert not adj.diagonal().any(), "self-pairs must be dropped"
        assert np.array_equal(adj, adj.T)

    def test_out_of_range_face_indices_are_dropped_not_wrapped(self):
        # a truncated sequence yields fewer faces than the edge list references
        adj = adjacency_matrix([[0, 1], [0, 9]], 3)
        assert adj[0, 1]
        assert adj.sum() == 2, "the out-of-range edge must vanish, not alias onto face 0"


class TestSharedVertices:
    def _seq(self, faces):
        return make_sequence(faces)

    def test_counts_exact_shared_vertex_positions(self):
        v_a = [1, 2, 3]
        v_b = [4, 5, 6]
        v_c = [7, 8, 9]
        v_d = [10, 11, 12]
        # face 0 = a,b,c ; face 1 = a,b,d (shares two) ; face 2 = d,d,d (shares none with 0)
        seq = self._seq([v_a + v_b + v_c, v_a + v_b + v_d, v_d + v_d + v_d])
        keys = vertex_keys(seq)
        assert keys.shape == (3, 3)
        got = shared_vertex_counts(keys, [1, 2], [0, 0])
        assert got[0] == 2, "faces sharing an edge share two vertices"
        assert got[1] == 0

    def test_identical_faces_share_all_three(self):
        v = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        keys = vertex_keys(self._seq([v, v]))
        assert shared_vertex_counts(keys, [1], [0])[0] == 3

    def test_keys_separate_vertices_that_differ_in_any_coordinate(self):
        keys = vertex_keys(self._seq([[0, 0, 0, 0, 0, 1, 0, 1, 0]]))
        assert len(set(keys[0].tolist())) == 3, "distinct grid cells must not collide"

    def test_key_encoding_cannot_alias_across_coordinates(self):
        # a naive encoding like x+y+z would make (0,0,1) and (1,0,0) identical
        keys = vertex_keys(self._seq([[0, 0, 1, 1, 0, 0, 0, 1, 0]]))
        assert len(set(keys[0].tolist())) == 3
        assert keys.max() < N_DISCRETE**3


def test_adjacency_and_causal_pairs_line_up_on_a_real_shaped_mesh():
    rng = np.random.default_rng(45)
    F = 12
    seq = make_sequence(rng.integers(0, N_DISCRETE, (F, TOKENS_PER_FACE)))
    centroids = rng.random((F, 3))
    q, k, _, _ = causal_pairs(centroids)
    adj = adjacency_matrix([[i, i + 1] for i in range(F - 1)], F)
    values = adj[q, k]
    # consecutive faces are adjacent, so every pair one apart in sequence counts,
    # except those involving face 0, which is excluded as sink
    expected = sum(1 for a, b in zip(q, k) if abs(a - b) == 1 and b >= 1)
    assert values.sum() == expected
