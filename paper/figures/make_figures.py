"""Regenerate every figure in the paper from committed artifacts.

    python paper/figures/make_figures.py --npz data/shapenet/03001627_train.npz

Figures whose inputs are missing are skipped with a note, so the confound figure
can be built from the dataset alone before any model has run.

Palette: slots 1-3 of the reference categorical theme (blue / orange / aqua),
validated all-pairs in light mode -- worst CVD dE 9.2, worst normal-vision dE
24.0. Aqua sits below 3:1 on white, so every series is direct-labeled rather
than identified by color alone.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#8a8a86"
COLUMN = 3.4  # inches; \linewidth of the two-column layout

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "legend.frameon": False,
    "lines.linewidth": 1.2,
    "grid.color": "#e2e2df",
    "grid.linewidth": 0.5,
})


def _tidy(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, alpha=0.9)
    ax.set_axisbelow(True)


def face_centroids(item):
    v = np.asarray(item["vertices"], dtype=np.float64)
    f = np.asarray(item["faces"], dtype=np.int64)
    return v[f].mean(axis=1)


def fig_confound(npz_path, out):
    """Sequence position and vertical height are the same variable."""
    from meshlens.data import load_chairs

    data = load_chairs(npz_path)
    meshes = [face_centroids(it) for it in data]
    meshes = [c for c in meshes if len(c) >= 50]

    rho = {a: [] for a in range(3)}
    for c in meshes:
        idx = np.arange(len(c))
        for a in range(3):
            if c[:, a].std() > 1e-9:
                rho[a].append(spearmanr(idx, c[:, a]).statistic)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COLUMN * 2, 2.1),
                                   gridspec_kw={"wspace": 0.28})

    bins = np.linspace(-1, 1, 61)
    order = [(1, ORANGE, "y (vertical)"), (0, BLUE, "x"), (2, AQUA, "z")]
    peaks = {}
    for a, color, label in order:
        counts, edges = np.histogram(rho[a], bins=bins)
        ax1.hist(rho[a], bins=bins, color=color, alpha=0.85, histtype="stepfilled", linewidth=0)
        top = int(np.argmax(counts))
        peaks[a] = ((edges[top] + edges[top + 1]) / 2, counts[top])
    ax1.set_xlabel("Spearman(face index, centroid coordinate)")
    ax1.set_ylabel("chairs")
    ax1.set_title(f"Every chair in the training set (n={len(meshes)})", color=INK)
    ax1.set_xlim(-1.05, 1.05)
    _tidy(ax1)

    # Direct labels rather than a legend box: aqua sits under 3:1 on white, so
    # identity must not rest on color alone. Each label is anchored just above
    # its own peak, and pulled inward at the axis edges so nothing overflows.
    ymax = ax1.get_ylim()[1]
    for a, color, label in order:
        x, count = peaks[a]
        inset = 0.03 * (ax1.get_xlim()[1] - ax1.get_xlim()[0])
        if x > 0.6:
            xt, ha = x - inset, "right"
        elif x < -0.6:
            xt, ha = x + inset, "left"
        else:
            xt, ha = x, "center"
        ax1.annotate(label, xy=(xt, min(count + 0.05 * ymax, 0.93 * ymax)), color=color,
                     ha=ha, va="bottom", fontsize=7, fontweight="bold")

    # one representative chair, closest to the median correlation
    med = np.median(rho[1])
    pick = int(np.argmin([abs(spearmanr(np.arange(len(c)), c[:, 1]).statistic - med)
                          for c in meshes]))
    c = meshes[pick]
    ax2.scatter(np.arange(len(c)), c[:, 1], s=3, color=ORANGE, alpha=0.55, linewidths=0)
    r = spearmanr(np.arange(len(c)), c[:, 1]).statistic
    ax2.set_xlabel("face index in token sequence")
    ax2.set_ylabel("centroid height")
    ax2.set_title(f"A representative chair (ρ = {r:+.3f})", color=INK)
    _tidy(ax2, grid_axis="both")

    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}  (median y-correlation {med:+.3f})")


def fig_head_taxonomy(res_path, out, layer=-1):
    """Every head placed by what its attention actually tracks."""
    d = np.load(res_path, allow_pickle=True)
    sp, sq = d["rho_spatial"], d["rho_seq"]
    n_layers = sp.shape[1]

    fig, ax = plt.subplots(figsize=(COLUMN, COLUMN))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("depth", ["#cfe0f5", "#123f74"])
    for L in range(n_layers):
        ax.scatter(np.nanmedian(sq[:, L, :], axis=0), np.nanmedian(sp[:, L, :], axis=0),
                   s=9, color=cmap(L / (n_layers - 1)), alpha=0.85, linewidths=0)
    lim = 1.0
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, linewidth=0.6, linestyle=(0, (3, 3)), zorder=0)
    ax.axhline(0, color=MUTED, linewidth=0.6, zorder=0)
    ax.axvline(0, color=MUTED, linewidth=0.6, zorder=0)
    ax.set_xlabel("$\\rho_{seq}$  (recency, space controlled)")
    ax.set_ylabel("$\\rho_{spatial}$  (geometry, sequence controlled)")
    ax.set_title("Each point is one head, median over chairs", color=INK)
    ax.set_xlim(-0.4, 1.0)
    ax.set_ylim(-0.4, 1.0)
    _tidy(ax, grid_axis="both")
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_layers - 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label("layer", color=INK_2)
    cb.outline.set_visible(False)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_layer_profile(res_path, out, sink_threshold=0.5):
    """How the two effects trade off with depth."""
    from meshlens.stats import bootstrap_ci

    d = np.load(res_path, allow_pickle=True)
    sp, sq, sink = d["rho_spatial"], d["rho_seq"], d["sink"]
    n_layers = sp.shape[1]

    curves = {"spatial": [], "seq": []}
    for L in range(n_layers):
        act = sink[:, L, :].mean(axis=0) <= sink_threshold
        if act.sum() == 0:
            for k in curves:
                curves[k].append((np.nan, np.nan, np.nan))
            continue
        for key, arr in (("spatial", sp), ("seq", sq)):
            curves[key].append(bootstrap_ci(np.nanmedian(arr[:, L, :][:, act], axis=1), seed=0))

    fig, ax = plt.subplots(figsize=(COLUMN * 1.5, 2.0))
    x = np.arange(n_layers)
    for key, color, label in (("seq", ORANGE, "recency"), ("spatial", BLUE, "geometry")):
        v = np.array(curves[key])
        ax.fill_between(x, v[:, 1], v[:, 2], color=color, alpha=0.16, linewidth=0)
        ax.plot(x, v[:, 0], color=color)
        last = np.where(np.isfinite(v[:, 0]))[0]
        if len(last):
            ax.annotate(label, xy=(x[last[-1]] + 0.3, v[last[-1], 0]), color=color,
                        va="center", fontsize=7, fontweight="bold")
    ax.axhline(0, color=MUTED, linewidth=0.6, zorder=0)
    ax.set_xlabel("layer")
    ax.set_ylabel("median partial $\\rho$ over active heads")
    ax.set_xlim(-0.5, n_layers + 2.5)
    _tidy(ax)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_resort(canon_path, xsort_path, out, layer=-1):
    """Experiment 2: does apparent geometry follow the sort key?"""
    a = np.load(canon_path, allow_pickle=True)
    b = np.load(xsort_path, allow_pickle=True)
    L = layer % a["rho_spatial"].shape[1]
    ca = np.nanmedian(a["rho_spatial"][:, L, :], axis=0)
    cb = np.nanmedian(b["rho_spatial"][:, L, :], axis=0)

    fig, ax = plt.subplots(figsize=(COLUMN, COLUMN))
    ax.scatter(ca, cb, s=12, color=BLUE, alpha=0.8, linewidths=0)
    lo = float(np.nanmin([ca.min(), cb.min(), 0]) - 0.05)
    hi = float(np.nanmax([ca.max(), cb.max()]) + 0.05)
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=0.6, linestyle=(0, (3, 3)), zorder=0)
    ax.set_xlabel("$\\rho_{spatial}$, canonical order")
    ax.set_ylabel("$\\rho_{spatial}$, x-sorted")
    ax.set_title("A head on the diagonal is genuinely spatial", color=INK)
    _tidy(ax, grid_axis="both")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/shapenet/03001627_train.npz")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/figures")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = Path(args.results)

    if Path(args.npz).exists():
        fig_confound(args.npz, out / "confound.pdf")
    else:
        print(f"skip confound: {args.npz} missing (see data/README.md)")

    canon = res / "e_canonical_primary.npz"
    xsort = res / "e_xsort_primary.npz"
    if canon.exists():
        fig_head_taxonomy(canon, out / "head_taxonomy.pdf")
        fig_layer_profile(canon, out / "layer_profile.pdf")
    else:
        print(f"skip head figures: {canon} missing (run scripts/run_experiment.py)")
    if canon.exists() and xsort.exists():
        fig_resort(canon, xsort, out / "resort.pdf")
    else:
        print("skip resort figure: needs both canonical and xsort runs")


if __name__ == "__main__":
    main()
