# Data

Nothing in this directory is committed except this file. Everything below is
downloaded locally and gitignored.

## ShapeNet chairs (required)

Pre-processed by the MeshXL authors; category `03001627`. 21.9 MB.

```sh
mkdir -p data/shapenet
curl -L -o data/shapenet/03001627_train.npz \
  https://huggingface.co/datasets/CH3COOK/MeshXL-shapenet-data/resolve/main/03001627_train.npz
```

Round 2's design check also uses table, lamp, and bench, which the sft
checkpoint likewise covers:

```sh
for cat in 04379243 03636649 02828884; do
  curl -L -o data/shapenet/${cat}_train.npz \
    https://huggingface.co/datasets/CH3COOK/MeshXL-shapenet-data/resolve/main/${cat}_train.npz
done
```

Full set of categories: https://huggingface.co/datasets/CH3COOK/MeshXL-shapenet-data

Each `.npz` holds `arr_0`, an array of dicts with keys `vertices` (torch float
tensor, `nv x 3`), `faces` (torch long tensor, `nf x 3`), `face_edges`, `texts`,
`cat_id`, `object_id`, `ext_info`. 2822 chairs, median 468 faces, max 800.

`face_edges` is an (n_edges, 2) list of face-index pairs that share an edge:
exact topological ground truth, and the probe round 2 is built on.

**Faces arrive pre-sorted.** The MeshXL tokenizer does not sort; it flattens
faces in the order the array already has. Face index therefore *is* token
sequence order. See `scripts/confound_check.py`.

## Model checkpoint (required for attention work, not for the confound check)

The chair fine-tune is public, so none of this depends on the original authors'
machine. `CH3COOK/MeshXL-1.3b-sft` holds `meshxl-1.3b-chair.pth` (2.45 GB)
alongside table, lamp, and bench fine-tunes.

```sh
python -c "from huggingface_hub import hf_hub_download; \
  print(hf_hub_download('CH3COOK/MeshXL-1.3b-sft','meshxl-1.3b-chair.pth'))"
```

Note that `mesh-xl/mesh-xl-1.3b`, the repo MeshXL's constructor names, supplies
only config and a `dummy` tensor: every real weight comes from the sft checkpoint
above. A correct load reports zero missing and zero unexpected keys and scores
cross-entropy near 0.06 on a training chair. Random init would score
ln(131) = 4.875, so that number is the check that the weights actually landed.

`scripts/setup_vm.sh` provisions all of the above on a fresh CPU box.

## Licensing

ShapeNet is distributed under a research-only license and may not be
redistributed. This repository ships neither the meshes nor the weights.
