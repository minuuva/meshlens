"""Run Experiments 4 and 5 from docs/prereg_round2.md.

Per chair and per (layer, head):

  rho_adj     partial Spearman of attention with topological adjacency,
              controlling for BOTH sequence distance and 3D distance
  rho_shared  the E5 control: partial Spearman of attention with the number of
              shared vertex positions, computed over NON-ADJACENT pairs only, so
              it isolates token matching from topology
  sink        as in round 1, for the active/dormant split

Runs on the held-out split, which round 1 never touched.

    python scripts/run_round2.py --split holdout
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meshlens.data import load_chairs, make_tokenizer, sample_splits, tokenize
from meshlens.extract import block_mean_torch, normalize_faces, sink_per_head
from meshlens.faces import (
    adjacency_matrix,
    causal_keep_mask,
    causal_pairs,
    centroids_from_tokens,
    n_faces,
    shared_vertex_counts,
    vertex_keys,
)
from meshlens.stats import head_partials_adjacency_batch

MIN_ADJACENT_PAIRS = 20  # inclusion rule, fixed in the preregistration


def run_chair(model, ids, centroids, adjacency, keys):
    """Returns (rho_adj, rho_shared, sink), each (n_layers, n_heads), or None."""
    F = n_faces(len(ids))
    q_idx, k_idx, d_seq, d_3d = causal_pairs(centroids)
    if len(q_idx) < 64:
        return None

    adj = adjacency[q_idx, k_idx].astype(np.float64)
    if adj.sum() < MIN_ADJACENT_PAIRS:
        return None

    shared = shared_vertex_counts(keys, q_idx, k_idx)
    non_adj = adj == 0
    controls = [d_seq, d_3d]
    controls_na = [d_seq[non_adj], d_3d[non_adj]]
    shared_na = shared[non_adj]
    # if every non-adjacent pair shares the same number of vertices there is
    # nothing for E5 to correlate against
    e5_usable = non_adj.sum() >= 64 and np.ptp(shared_na) > 0

    keep = causal_keep_mask(F)
    out = {"adj": [], "shared": [], "sink": []}

    def make_hook():
        def hook(module, inputs, output):
            attn = output[1]
            if attn is None:
                return output
            a = attn[0].detach()
            out["sink"].append(sink_per_head(a, F))
            faces = normalize_faces(block_mean_torch(a, F), keep)
            rows = faces[:, q_idx, k_idx]
            out["adj"].append(head_partials_adjacency_batch(rows, adj, controls))
            if e5_usable:
                out["shared"].append(
                    head_partials_adjacency_batch(rows[:, non_adj], shared_na, controls_na)
                )
            else:
                out["shared"].append(np.full(rows.shape[0], np.nan))
            return (output[0], None) + tuple(output[2:])

        return hook

    layers = model.model.decoder.layers
    handles = [layer.self_attn.register_forward_hook(make_hook()) for layer in layers]
    try:
        with torch.no_grad():
            model(input_ids=ids.unsqueeze(0), output_attentions=True)
    finally:
        for h in handles:
            h.remove()

    return (np.stack(out["adj"]), np.stack(out["shared"]), np.stack(out["sink"]),
            float(adj.mean()), int(adj.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["primary", "holdout"], default="holdout")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--npz", default="data/shapenet/03001627_train.npz")
    ap.add_argument("--ckpt", default="ckpts/meshxl-1.3b-chair.pth")
    ap.add_argument("--meshxl-root", default=".")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from meshlens.model import load_meshxl

    data = load_chairs(args.npz)
    primary, holdout = sample_splits(data)
    idx = primary if args.split == "primary" else holdout
    if args.limit:
        idx = idx[: args.limit]

    model, missing, unexpected = load_meshxl(args.ckpt)
    assert not missing and not unexpected, f"bad load: {len(missing)}/{len(unexpected)}"
    tokenizer = make_tokenizer(args.meshxl_root)

    adj_r, shared_r, sink_r, kept, rates = [], [], [], [], []
    t_start = time.time()
    for n, i in enumerate(idx):
        item = data[i]
        ids = tokenize(item, tokenizer)
        F = n_faces(len(ids))
        centroids = centroids_from_tokens(ids.numpy())
        adjacency = adjacency_matrix(item["face_edges"], F)
        keys = vertex_keys(ids.numpy())

        t0 = time.time()
        result = run_chair(model, ids, centroids, adjacency, keys)
        if result is None:
            print(f"[{n + 1}/{len(idx)}] chair {i}  skipped "
                  f"(under {MIN_ADJACENT_PAIRS} adjacent pairs)", flush=True)
            continue
        ra, rs, sk, rate, count = result
        adj_r.append(ra)
        shared_r.append(rs)
        sink_r.append(sk)
        kept.append(int(i))
        rates.append(rate)
        print(f"[{n + 1}/{len(idx)}] chair {i}  faces={F}  adj={count} ({100 * rate:.2f}%)  "
              f"{time.time() - t0:.1f}s  elapsed={(time.time() - t_start) / 60:.1f}m", flush=True)

    out = args.out or f"results/round2_{args.split}.npz"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        rho_adj=np.stack(adj_r),
        rho_shared=np.stack(shared_r),
        sink=np.stack(sink_r),
        chair_idx=np.array(kept),
        adjacency_rate=np.array(rates),
        split=args.split,
    )
    print(f"\nwrote {out}  shape={np.stack(adj_r).shape}  "
          f"median adjacency rate={100 * np.median(rates):.2f}%  "
          f"total={(time.time() - t_start) / 60:.1f}m")


if __name__ == "__main__":
    main()
