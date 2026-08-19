"""Run Experiment 1 / 2 from docs/prereg_round1.md and write per-chair statistics.

Per chair and per (layer, head) this records three numbers:

  rho_spatial  partial Spearman of attention with 3D proximity, sequence controlled
  rho_seq      partial Spearman of attention with sequence proximity, space controlled
  sink         mean attention onto BOS plus face 0, over the final face's query rows

The active/dormant split is a function of `sink` alone and is applied later, so
no head is classified after its correlations have been seen.

    python scripts/run_experiment.py --ordering canonical --split primary
    python scripts/run_experiment.py --ordering xsort     --split primary
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meshlens.data import load_chairs, make_tokenizer, resort_faces, sample_splits, tokenize
from meshlens.extract import block_mean_torch, normalize_faces, sink_per_head
from meshlens.faces import causal_keep_mask, causal_pairs, centroids_from_tokens, n_faces
from meshlens.model import load_meshxl
from meshlens.stats import head_partials_batch


def run_chair(model, ids, centroids):
    """Returns (rho_spatial, rho_seq, sink), each (n_layers, n_heads)."""
    F = n_faces(len(ids))
    q_idx, k_idx, d_seq, d_3d = causal_pairs(centroids)
    if len(q_idx) < 64:
        return None

    keep = causal_keep_mask(F)  # reused across all 24 layers
    layers = model.model.decoder.layers
    out = {"spatial": [], "seq": [], "sink": []}

    def make_hook():
        def hook(module, inputs, output):
            attn = output[1]
            if attn is None:
                return output
            # Reduce inside torch at float32: a float64 numpy copy of this
            # tensor is ~11 GB for an 800-face mesh and stalls the machine.
            a = attn[0].detach()  # (H, T, T) torch float32, no copy
            out["sink"].append(sink_per_head(a, F))
            faces = normalize_faces(block_mean_torch(a, F), keep)  # (H, F, F)
            rows = faces[:, q_idx, k_idx]  # (H, n_pairs)
            sp, sq = head_partials_batch(rows, d_seq, d_3d)
            out["spatial"].append(sp)
            out["seq"].append(sq)
            # Drop the tensor so the decoder cannot accumulate 24 of them.
            return (output[0], None) + tuple(output[2:])

        return hook

    handles = [layer.self_attn.register_forward_hook(make_hook()) for layer in layers]
    try:
        with torch.no_grad():
            model(input_ids=ids.unsqueeze(0), output_attentions=True)
    finally:
        for h in handles:
            h.remove()

    return tuple(np.stack(out[k]) for k in ("spatial", "seq", "sink"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ordering", choices=["canonical", "xsort"], default="canonical")
    ap.add_argument("--split", choices=["primary", "holdout"], default="primary")
    ap.add_argument("--limit", type=int, default=None, help="for timing runs only")
    ap.add_argument("--npz", default="data/shapenet/03001627_train.npz")
    ap.add_argument("--ckpt", default="ckpts/meshxl-1.3b-chair.pth")
    ap.add_argument("--meshxl-root", default=".", help="checkout containing models/mesh_xl/")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = load_chairs(args.npz)
    primary, holdout = sample_splits(data)
    idx = primary if args.split == "primary" else holdout
    if args.limit:
        idx = idx[: args.limit]

    model, missing, unexpected = load_meshxl(args.ckpt)
    assert not missing and not unexpected, f"bad load: {len(missing)}/{len(unexpected)}"
    tokenizer = make_tokenizer(args.meshxl_root)

    spatial, seq, sink, kept, losses = [], [], [], [], []
    t_start = time.time()
    for n, i in enumerate(idx):
        item = data[i]
        if args.ordering == "xsort":
            item = resort_faces(item, primary_axis=0)
        ids = tokenize(item, tokenizer)
        centroids = centroids_from_tokens(ids.numpy())
        t0 = time.time()
        result = run_chair(model, ids, centroids)
        if result is None:
            continue
        # validity gate for Experiment 2: how far off-distribution is this input?
        with torch.no_grad():
            losses.append(float(model(input_ids=ids.unsqueeze(0), labels=ids.unsqueeze(0)).loss))
        sp, sq, sk = result
        spatial.append(sp)
        seq.append(sq)
        sink.append(sk)
        kept.append(int(i))
        print(
            f"[{n + 1}/{len(idx)}] chair {i}  faces={n_faces(len(ids))}  "
            f"{time.time() - t0:.1f}s  loss={losses[-1]:.4f}  "
            f"elapsed={(time.time() - t_start) / 60:.1f}m",
            flush=True,
        )

    out = args.out or f"results/e_{args.ordering}_{args.split}.npz"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        rho_spatial=np.stack(spatial),
        rho_seq=np.stack(seq),
        sink=np.stack(sink),
        chair_idx=np.array(kept),
        loss=np.array(losses),
        ordering=args.ordering,
        split=args.split,
    )
    print(f"\nwrote {out}  shape={np.stack(spatial).shape}  "
          f"mean loss={np.mean(losses):.4f}  total={(time.time() - t_start) / 60:.1f}m")


if __name__ == "__main__":
    main()
