# meshlens

Mechanistic interpretability for 3D mesh transformers. This repository asks
whether attention in an autoregressive mesh generator tracks the *geometry* of
the shape it is building, or merely the *order* in which that shape was
serialized — and finds that separating the two is harder than the existing
literature assumes.

Work in progress. Round 1 is complete and recorded in
[docs/prereg_round1.md](docs/prereg_round1.md); round 2 is frozen in
[docs/prereg_round2.md](docs/prereg_round2.md) and running.

## The problem

[MeshXL](https://arxiv.org/abs/2405.20853) generates a mesh as a token stream:
nine tokens per triangle, three vertices of three coordinates each. Because the
sequence *is* the geometry, every attention weight has a literal spatial
address, and you can paint attention onto the chair. That makes it tempting to
read a head's attention map as evidence of geometric specialization.

The temptation is a trap. In MeshXL's canonical serialization, faces arrive
sorted by height, so **face index and vertical position are very nearly the same
variable**:

| | Spearman(face index, coordinate) |
|---|---|
| x | +0.097 |
| **y (vertical)** | **+0.935 mean, +0.951 median** |
| z | −0.277 |

Measured across all 2820 training chairs; 82% of chairs exceed 0.9. A cubic in
normalized sequence position explains 86% of height variance.

So "early layers attend to chair legs, late layers attend to chair backs" is not
evidence of geometric structure. Plain recency produces exactly that picture.
The confound is structural — a property of the canonical face sort, present in
every training mesh, and not escapable by choosing different chairs.

Reproduce with `python scripts/confound_check.py` (no GPU, no model weights).

## Findings

All decisions follow rules fixed before any attention was extracted. Where a
verdict is inconclusive it is reported as inconclusive; no threshold has been
moved after the fact.

- **Attention is predominantly recency, at every depth.** Over 100 chairs, the
  median partial correlation with sequence proximity is +0.430 (95% CI
  [+0.412, +0.472]) at the final layer, versus +0.194 ([+0.165, +0.206]) with 3D
  proximity once sequence distance is controlled.

- **The preregistered verdict is INCONCLUSIVE, and stays that way.** Support
  required ≥ 0.20; the estimate landed six thousandths under.

- **It replicates, which settles what that means.** A disjoint held-out 100
  chairs, untouched until the primary analysis was complete, gives +0.180
  ([+0.167, +0.199]) — overlapping intervals, and inconclusive on its own terms
  too. Two independent samples at +0.194 and +0.180 means the effect is
  precisely and reproducibly about 0.19, not noisily estimated. More data would
  narrow the interval without moving it. The splits are reported separately and
  never pooled.

- **No head in the model is primarily geometric** (exploratory). Across all 768
  head slots, none reaches ρ_spatial 0.50; the maximum anywhere is +0.469, while
  ρ_seq reaches +0.951. Of 263 active heads, 10 are more spatial than
  recency-driven.

- **The first half of the network has no geometric selectivity at all.** Layers
  0–11 median ρ_spatial −0.043, with 65 of 84 active heads at or below zero.
  Geometry appears only past layer 10 and stays modest (layers 12–23 median
  +0.260). This is the opposite of the "legs early, backs late" reading.

- **Sink head identity does not survive a modality transplant.** MeshXL inherits
  OPT-1.3b's transformer body and reinitializes only the embeddings, so the two
  are comparable head by head. Their sink maps are unrelated (Spearman −0.146,
  [−0.216, −0.078]) and the magnitudes differ sevenfold: 72% of MeshXL's 768
  heads sink above 0.5 versus 4% of OPT's. Whatever sink structure MeshXL has,
  it formed under mesh training rather than arriving with the weights.

- **The model is modelling the sequence, not the shape.** Re-sorting faces so
  index tracks x instead of height changes no geometry — same vertices, same
  triangles, same shape, only presentation order — and raises cross-entropy
  **40×** (0.055 → 2.203, higher in 37/37 chairs). That tripped the experiment's
  own validity gate, so it is reported as a distribution-shift result rather than
  as evidence about heads. It also means the confound may not be breakable by
  reordering at all.

## What is here

```
src/meshlens/
  faces.py     token/face indexing, MeshXL's undiscretize, 9x9 block reduction
  extract.py   torch-side reduction (a float64 numpy copy is ~11 GB per layer)
  stats.py     partial Spearman, batched variant, chair-level bootstrap
  data.py      seeded primary/holdout split, the x-resort manipulation
  model.py     loader forcing eager attention (see below)
  verdict.py   the preregistered decision rules, as tested code
scripts/       confound check, experiment runners, analyses
tests/         62 tests
docs/          the frozen preregistrations, with results appended
paper/         LaTeX source and figures/make_figures.py
results/       committed run artifacts
```

Two things worth knowing before reusing any of this:

**MeshXL's constructor calls `to_bettertransformer()`,** which swaps in a fused
SDPA kernel that never materializes the softmax matrix. Attention weights cannot
be read through it. `src/meshlens/model.py` reproduces MeshXL's construction
exactly but forces eager attention.

**MeshXL's `undiscretize` is `(t + 0.5)/128 * 2 − 1`,** not `t/127 * 2 − 1`. Both
are affine in `t`, so correlations are unaffected, but absolute distances and
areas are not.

## Setup

```sh
uv sync                      # analyses and tests; no torch
uv run pytest
uv sync --extra model        # adds torch + transformers for attention work
```

Data and checkpoints stay local — see [data/README.md](data/README.md). The chair
fine-tune is public, so nothing depends on private artifacts.
`scripts/setup_vm.sh` provisions a fresh box.

No GPU is required. A teacher-forced 1.3B forward pass with attention streaming
runs in seconds on CPU; the 100-chair experiment takes about 80 minutes on 100
vCPU. The original work needed an A100 only because it stored complete
`(32, T, T)` attention per layer.

## Provenance

Built on a earlier collaboration by Ethan Cao, Minu Choi, and Patrick Ho (prior work
University of Virginia), whose paper source is in `paper/`. The other two authors
have since left the work. This repository reworks it: the earlier draft's claim
of layer-wise geometric specialization does not survive the recency control
above, and several of its numbers had no artifact behind them. The audit is
tracked in the preregistration and commit history rather than quietly patched.

MeshXL is by Chen et al. (NeurIPS 2024). ShapeNet is research-license and is not
redistributed here.
