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

Full set of categories: https://huggingface.co/datasets/CH3COOK/MeshXL-shapenet-data

Each `.npz` holds `arr_0`, an array of dicts with keys `vertices` (torch float
tensor, `nv x 3`), `faces` (torch long tensor, `nf x 3`), `face_edges`, `texts`,
`cat_id`, `object_id`, `ext_info`. 2822 chairs, median 468 faces, max 800.

**Faces arrive pre-sorted.** The MeshXL tokenizer does not sort; it flattens
faces in the order the array already has. Face index therefore *is* token
sequence order. See `scripts/confound_check.py`.

## Model checkpoint (required for attention work, not for the confound check)

`ckpts/meshxl-1.3b-chair.pth` — the chair-finetuned 1.3B checkpoint. Base models
are on HuggingFace under `CH3COOK/mesh-xl-{125m,350m,1.3b}`.

## Licensing

ShapeNet is distributed under a research-only license and may not be
redistributed. This repository ships neither the meshes nor the weights.
