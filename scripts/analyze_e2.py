"""Apply the Experiment 2 interpretation rule from docs/prereg_round1.md.

E2 re-sorts faces so sequence index tracks x instead of height. A genuinely
spatial head keeps its rho_spatial; a recency head's apparent height selectivity
follows the sort key instead.

  reported statistic  per-head change in rho_spatial between orderings,
                      bootstrap CI over chairs
  spatial head        retains rho_spatial within 0.10 of its canonical value

  VALIDITY GATE       if mean cross-entropy under re-sorted input exceeds twice
                      the canonical-order loss, E2 is reported as a
                      distribution-shift result rather than as evidence about
                      head function, and E1 stands as the primary evidence.

    python scripts/analyze_e2.py results/e_canonical_primary.npz results/e_xsort_primary.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meshlens.stats import bootstrap_ci

SINK_THRESHOLD = 0.5
RETENTION = 0.10  # a spatial head keeps rho_spatial within this
GATE = 2.0  # loss ratio above which E2 becomes a distribution-shift result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("canonical")
    ap.add_argument("xsort")
    ap.add_argument("--layer", type=int, default=-1)
    args = ap.parse_args()

    a = np.load(args.canonical, allow_pickle=True)
    b = np.load(args.xsort, allow_pickle=True)
    n_layers = a["rho_spatial"].shape[1]
    layer = args.layer % n_layers

    # ---- validity gate, before anything about heads ----
    la, lb = float(a["loss"].mean()), float(b["loss"].mean())
    ratio = lb / la if la > 0 else np.inf
    print("VALIDITY GATE")
    print(f"  canonical mean cross-entropy = {la:.4f}")
    print(f"  x-sorted  mean cross-entropy = {lb:.4f}")
    print(f"  ratio = {ratio:.2f}x  (gate at {GATE:.0f}x)")
    gate_tripped = ratio > GATE
    if gate_tripped:
        print("\n  GATE TRIPPED: re-sorted input is far enough off-distribution that")
        print("  the head-function reading is not supported. Reporting E2 as a")
        print("  distribution-shift result; E1 stands as the primary evidence.")
    else:
        print("\n  gate clear: re-sorted input stays close enough to distribution")

    # chairs must line up cell for cell before differencing
    ca, cb = a["chair_idx"], b["chair_idx"]
    common = np.intersect1d(ca, cb)
    ia = np.array([np.where(ca == c)[0][0] for c in common])
    ib = np.array([np.where(cb == c)[0][0] for c in common])
    print(f"\n{len(common)} chairs present in both runs")

    active = a["sink"][ia][:, layer, :].mean(axis=0) <= SINK_THRESHOLD
    print(f"layer {layer}: {active.sum()} active heads (classified on canonical order)")
    if active.sum() == 0:
        return

    sp_a = a["rho_spatial"][ia][:, layer, :][:, active]
    sp_b = b["rho_spatial"][ib][:, layer, :][:, active]

    med_a, lo_a, hi_a = bootstrap_ci(np.nanmedian(sp_a, axis=1), seed=0)
    med_b, lo_b, hi_b = bootstrap_ci(np.nanmedian(sp_b, axis=1), seed=0)
    delta = np.nanmedian(sp_b, axis=1) - np.nanmedian(sp_a, axis=1)
    med_d, lo_d, hi_d = bootstrap_ci(delta, seed=0)

    print(f"\n  rho_spatial canonical = {med_a:+.3f}  95% CI [{lo_a:+.3f}, {hi_a:+.3f}]")
    print(f"  rho_spatial x-sorted  = {med_b:+.3f}  95% CI [{lo_b:+.3f}, {hi_b:+.3f}]")
    print(f"  change                = {med_d:+.3f}  95% CI [{lo_d:+.3f}, {hi_d:+.3f}]")

    per_head = np.nanmedian(sp_b, axis=0) - np.nanmedian(sp_a, axis=0)
    retained = np.abs(per_head) <= RETENTION
    print(f"\n  heads retaining rho_spatial within {RETENTION}: "
          f"{np.nansum(retained)}/{active.sum()}")

    if gate_tripped:
        print("\n  (interpreted as distribution shift, per the gate above)")
    elif abs(med_d) <= RETENTION:
        print("\n  SPATIAL SELECTIVITY IS STABLE under the re-sort")
    else:
        print("\n  APPARENT SPATIAL SELECTIVITY FOLLOWS THE SORT KEY")


if __name__ == "__main__":
    main()
