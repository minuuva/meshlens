"""Exploratory head-level summary. Not confirmatory.

Nothing here was preregistered, so nothing here decides anything. It exists
because the primary statistic -- one median at one layer -- throws away most of
what the run measured, and the head-level distribution turns out to say
something much less equivocal than a borderline median does.

    python scripts/explore_heads.py results/e_canonical_primary.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meshlens.verdict import SINK_THRESHOLD


def summarize(name, spatial, sequence):
    ok = np.isfinite(spatial) & np.isfinite(sequence)
    s, q = spatial[ok], sequence[ok]
    print(f"\n{name}  (n={len(s)})")
    print(f"  more spatial than recency : {int((s > q).sum())} ({100 * (s > q).mean():.1f}%)")
    print(f"  rho_spatial > 0.30        : {int((s > 0.30).sum())}")
    print(f"  rho_spatial > 0.50        : {int((s > 0.50).sum())}")
    print(f"  max rho_spatial           : {s.max():+.3f}")
    print(f"  max rho_seq               : {q.max():+.3f}")
    print(f"  median spatial / recency  : {np.median(s):+.3f} / {np.median(q):+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--split-layer", type=int, default=12)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    spatial = np.nanmedian(d["rho_spatial"], axis=0)  # (layers, heads)
    sequence = np.nanmedian(d["rho_seq"], axis=0)
    active = d["sink"].mean(axis=0) <= SINK_THRESHOLD
    n_layers, n_heads = spatial.shape

    print(f"{args.npz}: {n_layers * n_heads} head slots, {active.sum()} active")
    summarize("ALL head slots", spatial.ravel(), sequence.ravel())
    summarize("ACTIVE heads", spatial[active], sequence[active])

    early, late = active.copy(), active.copy()
    early[args.split_layer :, :] = False
    late[: args.split_layer, :] = False
    print()
    for label, mask in ((f"layers 0-{args.split_layer - 1}", early),
                        (f"layers {args.split_layer}-{n_layers - 1}", late)):
        s = spatial[mask]
        s = s[np.isfinite(s)]
        print(f"  {label} active: median rho_spatial = {np.median(s):+.3f}, "
              f"{int((s <= 0).sum())}/{len(s)} at or below zero")


if __name__ == "__main__":
    main()
