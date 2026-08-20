"""Apply the Experiment 4 and 5 interpretation rules from docs/prereg_round2.md.

E4  median rho_adj >= +0.15, CI excluding 0     -> topology survives the controls
    median rho_adj <  +0.05, CI excluding 0.15  -> no topological structure
    otherwise                                    -> inconclusive

E5  is the adjacency effect just token matching? Adjacent faces share two
    vertices and therefore six of eighteen coordinate tokens. rho_shared measures
    the same partial correlation against shared-vertex count over NON-ADJACENT
    pairs only. If it is comparable to rho_adj, the E4 result is token matching.

    python scripts/analyze_round2.py results/round2_holdout.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meshlens.stats import bootstrap_ci
from meshlens.verdict import SINK_THRESHOLD, e4_verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=-1)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    adj, shared, sink = d["rho_adj"], d["rho_shared"], d["sink"]
    n_chairs, n_layers, n_heads = adj.shape
    layer = args.layer % n_layers

    print(f"{args.npz}: {n_chairs} chairs, {n_layers} layers, {n_heads} heads")
    print(f"split={d['split']}  median adjacency rate="
          f"{100 * np.median(d['adjacency_rate']):.2f}% of causal pairs\n")

    active = sink[:, layer, :].mean(axis=0) <= SINK_THRESHOLD
    print(f"layer {layer}: {active.sum()} active / {(~active).sum()} dormant")
    if active.sum() == 0:
        print("no active heads; primary statistic undefined")
        return

    per_chair_adj = np.nanmedian(adj[:, layer, :][:, active], axis=1)
    med_a, lo_a, hi_a = bootstrap_ci(per_chair_adj, seed=0)

    print(f"\nE4 PRIMARY (median over {active.sum()} active heads, final layer)")
    print(f"  rho_adj = {med_a:+.3f}  95% CI [{lo_a:+.3f}, {hi_a:+.3f}]")
    tag, why = e4_verdict(med_a, lo_a, hi_a)
    print(f"\n  {tag}: {why}")

    per_chair_sh = np.nanmedian(shared[:, layer, :][:, active], axis=1)
    if np.isfinite(per_chair_sh).sum() >= 2:
        med_s, lo_s, hi_s = bootstrap_ci(per_chair_sh, seed=0)
        print(f"\nE5 CONTROL (shared vertex positions, non-adjacent pairs only)")
        print(f"  rho_shared = {med_s:+.3f}  95% CI [{lo_s:+.3f}, {hi_s:+.3f}]")
        # Exploratory, and reported whichever way E4 landed. The preregistered
        # E5 rule only asks whether an adjacency effect is really token
        # matching; it has nothing to say when there is no adjacency effect.
        # But the two quantities are still worth putting side by side, because
        # a model that tracks repeated coordinate values more strongly than it
        # tracks mesh structure is doing something lexical, not geometric.
        all_active = sink.mean(axis=0) <= SINK_THRESHOLD  # (layers, heads)
        act_adj = np.nanmedian(adj, axis=0)[all_active]
        act_sh = np.nanmedian(shared, axis=0)[all_active]
        ok = np.isfinite(act_adj) & np.isfinite(act_sh)
        if ok.sum():
            a, sh_ = act_adj[ok], act_sh[ok]
            print(f"\n  exploratory, over all {ok.sum()} active head slots in every layer:")
            print(f"    shared-token sensitivity exceeds adjacency in "
                  f"{100 * float((sh_ > a).mean()):.1f}% of them")
            print(f"    medians {np.median(a):+.3f} adjacency vs {np.median(sh_):+.3f} "
                  f"shared tokens ({np.median(sh_) / np.median(a):.2f}x)")
            print(f"    heads above +0.15: {int((a > 0.15).sum())} for adjacency, "
                  f"{int((sh_ > 0.15).sum())} for shared tokens")

        if med_a <= 0.05:
            print("\n  (the preregistered E5 rule does not apply: E4 found no "
                  "adjacency effect for it to explain)")
        elif med_s >= med_a * 0.5:
            print("\n  TOKEN MATCHING: the adjacency effect is comparable to the "
                  "shared-token effect, so it is not specifically topological")
        else:
            print("\n  TOPOLOGY SURVIVES: the adjacency effect clearly exceeds "
                  "shared-token matching")
    else:
        print("\nE5 CONTROL: not computable (too few usable non-adjacent pairs)")

    print("\nby layer (median over that layer's active heads):")
    for L in range(n_layers):
        act = sink[:, L, :].mean(axis=0) <= SINK_THRESHOLD
        if act.sum() == 0:
            print(f"  layer {L:2d}: no active heads")
            continue
        a = np.nanmedian(np.nanmedian(adj[:, L, :][:, act], axis=1))
        s = np.nanmedian(np.nanmedian(shared[:, L, :][:, act], axis=1))
        print(f"  layer {L:2d}: active={act.sum():2d}  rho_adj={a:+.3f}  rho_shared={s:+.3f}")


if __name__ == "__main__":
    main()
