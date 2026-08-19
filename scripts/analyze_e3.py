"""Apply the Experiment 3 interpretation rule from docs/prereg_round1.md.

  rho >= 0.5  -> sink head identity is inherited through the modality swap
  rho <= 0.2  -> sinks re-form independently under the new modality
  otherwise   -> partial inheritance, reported with the interval

rho is Spearman between the two 24x32 per-head sink grids, over all 768 heads.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

INHERITED, INDEPENDENT = 0.5, 0.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="?", default="results/e3_sink_transfer.npz")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    d = np.load(args.npz)
    mesh, text = d["mesh"], d["text"]
    n_layers, n_heads = mesh.shape
    m, t = mesh.ravel(), text.ravel()

    rho = spearmanr(m, t).statistic
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(m), size=(args.n_boot, len(m)))
    draws = np.array([spearmanr(m[i], t[i]).statistic for i in idx])
    lo, hi = np.percentile(draws, [2.5, 97.5])

    print(f"{args.npz}: {n_layers} layers x {n_heads} heads = {len(m)} cells\n")
    print(f"mesh sink: mean={m.mean():.3f} median={np.median(m):.3f} max={m.max():.3f}")
    print(f"text sink: mean={t.mean():.3f} median={np.median(t):.3f} max={t.max():.3f}")
    print(f"\nSpearman(mesh, text) over all heads = {rho:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")

    if rho >= INHERITED:
        print("\n  INHERITED: sink head identity survives the modality swap")
    elif rho <= INDEPENDENT:
        print("\n  INDEPENDENT: sinks re-form under the new modality")
    else:
        print("\n  PARTIAL inheritance")

    print("\nper layer:")
    for L in range(n_layers):
        r = spearmanr(mesh[L], text[L]).statistic
        print(f"  layer {L:2d}: mesh={mesh[L].mean():.3f} text={text[L].mean():.3f} rho={r:+.3f}")


if __name__ == "__main__":
    main()
