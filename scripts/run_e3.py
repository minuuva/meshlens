"""Experiment 3: does sink head identity survive a modality transplant?

MeshXL loads pretrained OPT weights and reinitializes only the word and
positional embeddings, so the transformer body is inherited across a complete
vocabulary and modality swap. Both models are OPT-1.3b shaped: 24 layers, 32
heads, hidden 2048. That makes per-head sink attention directly comparable cell
by cell, which is a comparison the language-only literature has no way to set up.

The sink definition is the one frozen in docs/prereg_round1.md and transfers
without change: attention onto the first ten tokens, from the final nine query
rows. For a mesh those ten tokens are BOS plus face 0; for text they are BOS plus
the first nine tokens.

Lengths are matched at OPT's native 2048-token context, so meshes are restricted
to those near 227 faces rather than using the full-length sequences from E1.

    python scripts/run_e3.py --text data/text/corpus.txt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meshlens.data import load_chairs, make_tokenizer, tokenize
from meshlens.extract import sink_per_head
from meshlens.faces import TOKENS_PER_FACE, n_faces
from meshlens.model import load_meshxl

CONTEXT = 2048  # OPT-1.3b's native max_position_embeddings
SINK_WIDTH = 1 + TOKENS_PER_FACE  # BOS plus the first face, i.e. ten tokens


def sink_grid(model, sequences, layers, describe):
    """Mean per-head sink attention over sequences, as (n_layers, n_heads)."""
    per_seq = []
    for n, ids in enumerate(sequences):
        rows = {}

        def make_hook(i):
            def hook(module, inputs, output):
                if output[1] is not None:
                    a = output[1][0].detach()
                    tail = a[:, -TOKENS_PER_FACE:, :SINK_WIDTH]
                    rows[i] = tail.sum(dim=2).mean(dim=1).double().numpy()
                return (output[0], None) + tuple(output[2:])

            return hook

        handles = [l.self_attn.register_forward_hook(make_hook(i)) for i, l in enumerate(layers)]
        try:
            with torch.no_grad():
                model(input_ids=ids.unsqueeze(0), output_attentions=True)
        finally:
            for h in handles:
                h.remove()
        per_seq.append(np.stack([rows[i] for i in sorted(rows)]))
        print(f"  {describe} [{n + 1}/{len(sequences)}] len={len(ids)}", flush=True)
    return np.stack(per_seq).mean(axis=0)


def mesh_sequences(npz, ckpt, meshxl_root, n_seq):
    data = load_chairs(npz)
    tk = make_tokenizer(meshxl_root)
    target = (CONTEXT - 2) // TOKENS_PER_FACE  # ~227 faces
    order = sorted(range(len(data)), key=lambda i: abs(len(np.asarray(data[i]["faces"])) - target))
    seqs = []
    for i in order[:n_seq]:
        ids = tokenize(data[i], tk)
        seqs.append(ids[:CONTEXT])
    model, missing, unexpected = load_meshxl(ckpt)
    assert not missing and not unexpected
    print(f"mesh: {n_seq} chairs near {target} faces "
          f"(mean {np.mean([n_faces(len(s)) for s in seqs]):.0f})", flush=True)
    return model, seqs


def text_sequences(text_path, n_seq):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
    raw = Path(text_path).read_text(errors="ignore")
    ids = tok(raw, return_tensors="pt").input_ids[0]
    usable = len(ids) // CONTEXT
    assert usable >= n_seq, f"corpus gives only {usable} windows of {CONTEXT}"
    seqs = [ids[i * CONTEXT : (i + 1) * CONTEXT].clone() for i in range(n_seq)]
    for s in seqs:
        s[0] = tok.bos_token_id  # every window starts from BOS, as a mesh does
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-1.3b", attn_implementation="eager", torch_dtype=torch.float32
    )
    model.eval()
    print(f"text: {n_seq} windows of {CONTEXT} tokens", flush=True)
    return model, seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="data/text/corpus.txt")
    ap.add_argument("--npz", default="data/shapenet/03001627_train.npz")
    ap.add_argument("--ckpt", default="ckpts/meshxl-1.3b-chair.pth")
    ap.add_argument("--meshxl-root", default=".")
    ap.add_argument("--n-seq", type=int, default=20)
    ap.add_argument("--out", default="results/e3_sink_transfer.npz")
    args = ap.parse_args()

    mesh_model, mesh_seqs = mesh_sequences(args.npz, args.ckpt, args.meshxl_root, args.n_seq)
    mesh_grid = sink_grid(mesh_model, mesh_seqs, mesh_model.model.decoder.layers, "mesh")
    del mesh_model

    text_model, text_seqs = text_sequences(args.text, args.n_seq)
    assert len(text_model.model.decoder.layers) == mesh_grid.shape[0], "layer counts differ"
    text_grid = sink_grid(text_model, text_seqs, text_model.model.decoder.layers, "text")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, mesh=mesh_grid, text=text_grid)
    print(f"\nwrote {args.out}  grids {mesh_grid.shape}")


if __name__ == "__main__":
    main()
