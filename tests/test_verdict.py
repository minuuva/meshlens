"""The preregistered decision rules must fire correctly on known inputs.

Checked here against synthetic values rather than against a slice of the real
run, so the decision path is verified without previewing the result. Every
threshold is exercised on both sides and exactly at the boundary, because a
rule that is off by one comparison would quietly convert a null into a finding.
"""

import pytest

from meshlens.verdict import (
    E1_REFUTE_SEQ,
    E1_REFUTE_SPATIAL,
    E1_SUPPORT,
    E2_LOSS_GATE,
    E2_RETENTION,
    E3_INDEPENDENT,
    E3_INHERITED,
    e1_verdict,
    e2_verdict,
    e3_verdict,
)


class TestE1:
    def test_supported_needs_both_the_threshold_and_a_ci_clear_of_zero(self):
        assert e1_verdict(0.35, ci_lo=0.20, med_seq=0.10)[0] == "SUPPORTED"
        # same point estimate, but the interval touches zero
        assert e1_verdict(0.35, ci_lo=-0.02, med_seq=0.10)[0] != "SUPPORTED"

    def test_refuted_needs_both_a_weak_spatial_and_a_strong_sequence_effect(self):
        assert e1_verdict(0.03, ci_lo=-0.05, med_seq=0.55)[0] == "REFUTED"
        # weak spatial but no sequence effect either is not a refutation
        assert e1_verdict(0.03, ci_lo=-0.05, med_seq=0.05)[0] == "INCONCLUSIVE"

    def test_the_gap_between_thresholds_is_inconclusive_not_rounded_away(self):
        for med in (0.10, 0.15, 0.19):
            assert e1_verdict(med, ci_lo=0.05, med_seq=0.50)[0] == "INCONCLUSIVE"

    def test_boundaries_are_inclusive_exactly_as_written(self):
        assert e1_verdict(E1_SUPPORT, ci_lo=0.01, med_seq=0.0)[0] == "SUPPORTED"
        assert e1_verdict(E1_REFUTE_SPATIAL - 1e-9, ci_lo=0.0, med_seq=E1_REFUTE_SEQ)[0] == "REFUTED"
        # at the refute-spatial threshold itself the rule does not fire
        assert e1_verdict(E1_REFUTE_SPATIAL, ci_lo=0.0, med_seq=0.9)[0] == "INCONCLUSIVE"


class TestE2:
    def test_the_loss_gate_overrides_any_head_level_reading(self):
        # a large, clean-looking effect is still not reportable as head function
        assert e2_verdict(delta_median=-0.9, loss_ratio=5.0)[0] == "GATE_TRIPPED"
        assert e2_verdict(delta_median=0.0, loss_ratio=5.0)[0] == "GATE_TRIPPED"

    def test_stable_and_follows_sort_key_split_at_the_retention_band(self):
        assert e2_verdict(0.05, loss_ratio=1.1)[0] == "STABLE"
        assert e2_verdict(-0.05, loss_ratio=1.1)[0] == "STABLE"
        assert e2_verdict(-0.40, loss_ratio=1.1)[0] == "FOLLOWS_SORT_KEY"

    def test_retention_is_symmetric_and_inclusive(self):
        assert e2_verdict(E2_RETENTION, loss_ratio=1.0)[0] == "STABLE"
        assert e2_verdict(-E2_RETENTION, loss_ratio=1.0)[0] == "STABLE"
        assert e2_verdict(E2_RETENTION + 1e-9, loss_ratio=1.0)[0] == "FOLLOWS_SORT_KEY"

    def test_gate_is_strict_so_exactly_double_still_passes(self):
        assert e2_verdict(0.0, loss_ratio=E2_LOSS_GATE)[0] == "STABLE"
        assert e2_verdict(0.0, loss_ratio=E2_LOSS_GATE + 1e-9)[0] == "GATE_TRIPPED"


class TestE3:
    @pytest.mark.parametrize("rho,expected", [
        (0.85, "INHERITED"),
        (E3_INHERITED, "INHERITED"),
        (0.35, "PARTIAL"),
        (E3_INDEPENDENT, "INDEPENDENT"),
        (-0.10, "INDEPENDENT"),
    ])
    def test_three_bands(self, rho, expected):
        assert e3_verdict(rho)[0] == expected

    def test_the_middle_band_is_not_silently_collapsed_to_a_side(self):
        assert e3_verdict((E3_INDEPENDENT + E3_INHERITED) / 2)[0] == "PARTIAL"
