# meshlens

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22149960.svg)](https://doi.org/10.5281/zenodo.22149960)

Does attention in an autoregressive 3D mesh transformer track the geometry of the
shape, or the order in which that shape was serialized?

It tracks the order. This repository holds the code, preregistrations, and
artifacts behind that finding.

**Paper:** [paper/main.pdf](paper/main.pdf), "Serialization Order Confounds
Attention Analysis in MeshXL"

## The problem

[MeshXL](https://arxiv.org/abs/2405.20853) emits a mesh as a token stream: nine
tokens per triangle, three vertices of three coordinates each. Because the
sequence *is* the geometry, attention can be painted directly onto the chair,
which makes a head look geometrically specialized.

It is a trap. Faces arrive sorted by height, so face index and vertical position
are nearly the same variable:

| axis | Spearman(face index, coordinate) |
|---|---|
| x | +0.097 |
| **y (vertical)** | **+0.951 median** |
| z | −0.277 |

Measured on all 2820 training chairs; 82% exceed 0.9. So "early layers attend to
chair legs, late layers to chair backs" is what plain recency looks like. The
confound is structural: it comes from preprocessing, holds in every mesh, and
varies only from +0.914 to +0.968 across four object categories.

Reproduce with `python scripts/confound_check.py`. No GPU, no model weights.

## Findings

Every confirmatory result follows a rule fixed before any attention was
extracted. Inconclusive results are reported as inconclusive.

| result | |
|---|---|
| Attention is recency, not geometry | ρ_seq +0.430 vs ρ_spatial +0.194 at the final layer |
| The spatial effect replicates | +0.180 on a disjoint held-out 100 chairs, intervals overlap |
| No head is primarily geometric | none of 768 reaches ρ_spatial 0.50; ρ_seq reaches +0.951 |
| Early layers have no geometry at all | layers 0 to 11 median −0.043, 65 of 84 active heads at or below zero |
| The model encodes sequence, not shape | re-sorting faces changes no geometry and costs 40× cross-entropy |
| Mesh topology is not tracked | ρ_adj +0.041, no head above +0.15 |
| Token repetition beats topology 2× | ρ_shared +0.095 vs +0.047, in all 24 layers |
| Sinks form under mesh training | MeshXL vs OPT-1.3b per-head sink maps correlate −0.146 |

The last one is a rare controlled comparison: MeshXL inherits OPT-1.3b's
transformer body and reinitializes only the embeddings, so per-head sink
attention is comparable cell by cell across a full modality swap.

## Layout

```
src/meshlens/   faces, extraction, statistics, data, model loader, decision rules
scripts/        confound check, experiment runners, analyses
tests/          62 tests
docs/           frozen preregistrations, with results appended
paper/          main.tex, main.pdf, and figures/make_figures.py
results/        committed run artifacts
```

Figures regenerate from committed artifacts via `make_figures.py`. No figure is
hand-placed.

## Two gotchas worth knowing

MeshXL's constructor calls `to_bettertransformer()`, which installs a fused
attention kernel that never materializes the softmax matrix. Attention weights
cannot be read through it. `src/meshlens/model.py` forces an eager implementation.

MeshXL's inverse discretization is `(t + 0.5)/128 * 2 - 1`, not `t/127 * 2 - 1`.
Both are affine in `t`, so correlations are unaffected, but absolute distances are
not.

## Setup

```sh
uv sync                 # analyses and tests, no torch
uv run pytest
uv sync --extra model   # adds torch and transformers for attention work
```

Data and checkpoints stay local; see [data/README.md](data/README.md). The chair
fine-tune is public, so nothing depends on private artifacts.
`scripts/setup_vm.sh` provisions a fresh box.

No GPU required. The 100-chair experiment takes about 80 minutes on 100 vCPU. The
original work needed an A100 only because it stored complete `(32, T, T)`
attention per layer.

## Citation

```bibtex
@software{choi2026meshlens,
  author    = {Choi, Minu},
  title     = {meshlens: Serialization Order Confounds Attention
               Analysis in {MeshXL}},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22149960},
  url       = {https://doi.org/10.5281/zenodo.22149960}
}
```

## License

Code is MIT (see [LICENSE](LICENSE)). The paper and figures are CC BY 4.0.

Neither ShapeNet nor the MeshXL checkpoints are redistributed here. ShapeNet is
research-license; see [data/README.md](data/README.md) for how to obtain both.

## Provenance

This grew out of an earlier collaboration with Ethan Cao and Patrick Ho, who are
not involved in the current version. That draft reported layer-wise geometric
specialization, which does not survive the recency control above, and several of
its figures reported quantities no code computed. The present version re-derives
everything under preregistered rules; the reasoning is in [docs/](docs/) and the
commit history.

MeshXL is by Chen et al. (NeurIPS 2024); the chair fine-tune is the authors' own
public checkpoint.
