"""Apply the Experiment 1 interpretation rule from docs/prereg_round1.md.

The rule, fixed before any attention was extracted:

  median rho_spatial >= +0.20, CI excluding 0   -> spatial selectivity survives
  median rho_spatial <  +0.10 and rho_seq >= +0.30 -> refuted; attention is recency
  otherwise                                      -> inconclusive at this n

The primary statistic is over ACTIVE heads in the FINAL layer, where a head is
active when its mean sink attention over the final face's query rows is at most
0.5. Classification depends only on `sink`, never on the correlations.

    python scripts/analyze_e1.py results/e_canonical_primary.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meshlens.stats import bootstrap_ci
from meshlens.verdict import SINK_THRESHOLD, e1_verdict


def replicate(path, layer, med_a, lo_a, hi_a):
    """Report a second split against the first, without pooling them.

    The held-out set is a replication, not a top-up. Pooling to double n and
    push a borderline estimate over a threshold is exactly what the
    preregistration exists to prevent, so the two estimates are shown side by
    side and the question asked is only whether they agree.
    """
    d = np.load(path, allow_pickle=True)
    sink = d["sink"]
    active = sink[:, layer, :].mean(axis=0) <= SINK_THRESHOLD
    if active.sum() == 0:
        print(f"\nREPLICATION {path}: no active heads")
        return
    per_chair = np.nanmedian(d["rho_spatial"][:, layer, :][:, active], axis=1)
    med_b, lo_b, hi_b = bootstrap_ci(per_chair, seed=0)
    seq_b = np.nanmedian(d["rho_seq"][:, layer, :][:, active], axis=1)
    med_sq_b, lo_sq_b, hi_sq_b = bootstrap_ci(seq_b, seed=0)

    print(f"\nREPLICATION on {d['split']} ({d['rho_spatial'].shape[0]} chairs, "
          f"{active.sum()} active heads)")
    print(f"  rho_spatial = {med_b:+.3f}  95% CI [{lo_b:+.3f}, {hi_b:+.3f}]")
    print(f"  rho_seq     = {med_sq_b:+.3f}  95% CI [{lo_sq_b:+.3f}, {hi_sq_b:+.3f}]")
    tag, why = e1_verdict(med_b, lo_b, med_sq_b)
    print(f"  verdict on this split alone: {tag}")

    overlap = not (hi_a < lo_b or hi_b < lo_a)
    print(f"\n  primary   {med_a:+.3f} [{lo_a:+.3f}, {hi_a:+.3f}]")
    print(f"  held-out  {med_b:+.3f} [{lo_b:+.3f}, {hi_b:+.3f}]")
    print(f"  intervals {'overlap: the estimate replicates' if overlap else 'DISJOINT: the splits disagree'}")
    print("  (reported separately by design; the splits are not pooled)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--compare", help="a second run to report as a replication")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    spatial, seq, sink = d["rho_spatial"], d["rho_seq"], d["sink"]
    n_chairs, n_layers, n_heads = spatial.shape
    layer = args.layer % n_layers

    print(f"{args.npz}: {n_chairs} chairs, {n_layers} layers, {n_heads} heads")
    print(f"ordering={d['ordering']} split={d['split']} mean loss={d['loss'].mean():.4f}\n")

    # Head classification first, from sink alone, averaged over chairs.
    mean_sink = sink[:, layer, :].mean(axis=0)
    active = mean_sink <= SINK_THRESHOLD
    print(f"layer {layer}: {active.sum()} active / {(~active).sum()} dormant "
          f"(sink threshold {SINK_THRESHOLD})")
    print(f"  sink attention: min={mean_sink.min():.3f} median={np.median(mean_sink):.3f} "
          f"max={mean_sink.max():.3f}")

    if active.sum() == 0:
        print("\nno active heads; primary statistic undefined")
        return

    # Per chair, the median over active heads. Bootstrap resamples chairs.
    per_chair_spatial = np.nanmedian(spatial[:, layer, :][:, active], axis=1)
    per_chair_seq = np.nanmedian(seq[:, layer, :][:, active], axis=1)

    med_sp, lo_sp, hi_sp = bootstrap_ci(per_chair_spatial, seed=0)
    med_sq, lo_sq, hi_sq = bootstrap_ci(per_chair_seq, seed=0)

    print(f"\nPRIMARY (median over {active.sum()} active heads, final layer)")
    print(f"  rho_spatial = {med_sp:+.3f}  95% CI [{lo_sp:+.3f}, {hi_sp:+.3f}]")
    print(f"  rho_seq     = {med_sq:+.3f}  95% CI [{lo_sq:+.3f}, {hi_sq:+.3f}]")
    tag, why = e1_verdict(med_sp, lo_sp, med_sq)
    print(f"\n  {tag}: {why}")

    if args.compare:
        replicate(args.compare, layer, med_sp, lo_sp, hi_sp)

    print("\nby layer (median over that layer's active heads):")
    for L in range(n_layers):
        act = sink[:, L, :].mean(axis=0) <= SINK_THRESHOLD
        if act.sum() == 0:
            print(f"  layer {L:2d}: no active heads")
            continue
        sp = np.nanmedian(np.nanmedian(spatial[:, L, :][:, act], axis=1))
        sq = np.nanmedian(np.nanmedian(seq[:, L, :][:, act], axis=1))
        print(f"  layer {L:2d}: active={act.sum():2d}  rho_spatial={sp:+.3f}  rho_seq={sq:+.3f}")


if __name__ == "__main__":
    main()
