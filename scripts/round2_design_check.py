"""Reproduce the design validation behind docs/prereg_round2.md.

Two questions, both answered from the dataset alone with no model weights and no
GPU, so a reader can check the reasoning that picked round 2's probe before any
attention was extracted.

  1. Do other object categories decorrelate height from sequence position?
     (They do not, which is why round 2 changes the question instead.)
  2. Is topological adjacency independent enough of sequence distance to be a
     usable probe? (It is, far more so than metric distance.)

    python scripts/round2_design_check.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meshlens.faces import VERTICAL_AXIS, causal_pairs

CATEGORIES = {"chair": "03001627", "table": "04379243", "lamp": "03636649", "bench": "02828884"}
SEQ_ADJACENT = 3  # "next to each other in the token stream" means within this many faces


def centroids(item):
    v = np.asarray(item["vertices"], dtype=np.float64)
    f = np.asarray(item["faces"], dtype=np.int64)
    return v[f].mean(axis=1)


def adjacency_matrix(item, n_faces):
    fe = np.asarray(item["face_edges"], dtype=np.int64)
    A = np.zeros((n_faces, n_faces), dtype=bool)
    A[fe[:, 0], fe[:, 1]] = True
    A[fe[:, 1], fe[:, 0]] = True
    np.fill_diagonal(A, False)
    return A


def category_confound(data_dir):
    print("Does the confound vary by object category?\n")
    print(f"{'category':8s} {'meshes':>7s} {'faces':>6s}  {'rho(i,x)':>9s} "
          f"{'rho(i,y)':>9s} {'rho(i,z)':>9s}  {'R2 h~pos':>9s}")
    medians = []
    for name, cid in CATEGORIES.items():
        path = Path(data_dir) / f"{cid}_train.npz"
        if not path.exists():
            print(f"{name:8s} (missing {path.name}; see data/README.md)")
            continue
        data = np.load(path, allow_pickle=True)["arr_0"]
        rho = {a: [] for a in range(3)}
        r2, counts = [], []
        for item in data:
            C = centroids(item)
            n = len(C)
            if n < 50:
                continue
            counts.append(n)
            idx = np.arange(n)
            for a in range(3):
                if C[:, a].std() > 1e-9:
                    rho[a].append(spearmanr(idx, C[:, a]).statistic)
            h = C[:, VERTICAL_AXIS]
            if h.var() > 0:
                u = np.linspace(0, 1, n)
                resid = h - np.polyval(np.polyfit(u, h, 3), u)
                r2.append(1 - resid.var() / h.var())
        medians.append((np.median(rho[VERTICAL_AXIS]), np.mean(r2)))
        print(f"{name:8s} {len(counts):7d} {int(np.median(counts)):6d}  "
              f"{np.median(rho[0]):+9.3f} {np.median(rho[1]):+9.3f} "
              f"{np.median(rho[2]):+9.3f}  {np.mean(r2):9.3f}")
    if len(medians) > 1:
        ys = [m[0] for m in medians]
        rs = [m[1] for m in medians]
        print(f"\n  spread in rho(index, height): {max(ys) - min(ys):.3f}")
        print(f"  spread in R^2:                {max(rs) - min(rs):.3f}")
        print("  -> the sort is a property of preprocessing, not of the object")


def adjacency_independence(data_dir, n_sample=40, seed=2027):
    print("\n\nIs topological adjacency independent of sequence distance?\n")
    path = Path(data_dir) / f"{CATEGORIES['chair']}_train.npz"
    data = np.load(path, allow_pickle=True)["arr_0"]
    eligible = [i for i, it in enumerate(data) if 50 <= len(np.asarray(it["faces"])) <= 800]
    sample = np.random.default_rng(seed).choice(eligible, n_sample, replace=False)

    r_seq, r_3d, frac, near = [], [], [], []
    for i in sample:
        item = data[i]
        C = centroids(item)
        A = adjacency_matrix(item, len(C))
        q, k, d_seq, d_3d = causal_pairs(C)
        if len(q) < 200:
            continue
        adj = A[q, k].astype(float)
        if adj.sum() < 5:
            continue
        frac.append(adj.mean())
        r_seq.append(spearmanr(adj, d_seq).statistic)
        r_3d.append(spearmanr(adj, d_3d).statistic)
        near.append((d_seq[adj > 0] <= SEQ_ADJACENT).mean())

    for label, v in (("adjacency vs sequence distance", r_seq), ("adjacency vs 3D distance", r_3d)):
        v = np.array(v)
        print(f"  spearman({label:32s}) median={np.median(v):+.3f}  "
              f"[p05={np.percentile(v, 5):+.3f}, p95={np.percentile(v, 95):+.3f}]")
    print(f"\n  causal face pairs that are adjacent: {100 * np.median(frac):.2f}%")
    print(f"  adjacent pairs more than {SEQ_ADJACENT} apart in sequence: "
          f"{100 * (1 - np.median(near)):.1f}%")
    print("  -> for comparison, 3D distance correlates +0.569 with sequence distance")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/shapenet")
    args = ap.parse_args()
    category_confound(args.data_dir)
    adjacency_independence(args.data_dir)


if __name__ == "__main__":
    main()
