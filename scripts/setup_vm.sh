#!/usr/bin/env bash
# Provision a CPU box for the attention experiments. No GPU is required: a
# teacher-forced 1.3B forward pass over ~1800 tokens with attention streaming
# takes about 4.5s on 100 vCPU, so the full 200-chair sample runs in ~15 min.
set -euo pipefail

ROOT=${ROOT:-/root/meshlens}
mkdir -p "$ROOT" && cd "$ROOT"
[ -d MeshXL ] || git clone -q --depth 1 https://github.com/OpenMeshLab/MeshXL.git
cd MeshXL

[ -d .venv ] || uv venv --python 3.11 -q
. .venv/bin/activate
uv pip install -q torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -q "transformers==4.44.2" "numpy<2.2" scipy einops huggingface_hub accelerate

mkdir -p ckpts data/shapenet
export HF_HUB_DISABLE_PROGRESS_BARS=1
[ -f ckpts/meshxl-1.3b-chair.pth ] || python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil
shutil.copy(hf_hub_download("CH3COOK/MeshXL-1.3b-sft", "meshxl-1.3b-chair.pth"),
            "ckpts/meshxl-1.3b-chair.pth")
PY
[ -f data/shapenet/03001627_train.npz ] || curl -sL -o data/shapenet/03001627_train.npz \
  https://huggingface.co/datasets/CH3COOK/MeshXL-shapenet-data/resolve/main/03001627_train.npz

# Experiment 3 needs a natural-text corpus for the OPT arm. Public domain,
# fetched rather than vendored so no copyrighted text enters the repo.
mkdir -p data/text
[ -f data/text/corpus.txt ] || curl -sL https://www.gutenberg.org/files/1342/1342-0.txt \
  -o data/text/corpus.txt

echo "provisioned: $(du -sh ckpts data/shapenet data/text | tr '\n' ' ')"
