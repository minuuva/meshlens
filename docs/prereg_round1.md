# Preregistration round 1: separating spatial attention from sequence recency

Committed before any attention is extracted under this design, and before any
statistic below is computed on any model. Written after, and only after, the
dataset-level confound measurement in `scripts/confound_check.py`, which used no
model weights.

## Why this exists

In MeshXL's serialization, face index tracks the vertical axis at Spearman
rho = +0.935 across all 2822 training chairs (82% of chairs above 0.9). A cubic
in normalized sequence position explains 86% of height variance.

So an observation of the form "this head attends to low triangles when querying
from a late position" is not evidence of spatial selectivity. Recency alone
produces it. Every claim of geometric head specialization in this model class is
subject to that confound, and it is structural: it is a property of the
canonical face sort, present in every training mesh, and not escapable by
choosing different chairs.

36% of the height standard deviation survives the regression, so the two are
separable in principle. This round separates them.

## Definitions fixed in advance

These are fixed here because the earlier version of this work used two
incompatible definitions of "dormant" (prose said total sink attention, code
thresholded BOS-only attention).

- **Sink attention** of a head at query row q: the summed attention from q onto
  token 0 (BOS) plus tokens 1-9 (the first face). One number per (head, query).
- **Dormant head**: mean sink attention over the query rows of the final face
  exceeds 0.5. **Active head**: everything else. No third category.
- **Face-level attention** a(q,k): mean of the 9x9 token block from face q to
  face k, sink columns excluded, renormalized over non-sink faces.
- **Sequence distance** d_seq(q,k) = q - k, in faces.
- **Spatial distance** d_3d(q,k) = Euclidean distance between face centroids,
  decoded with MeshXL's own `undiscretize`, `(t + 0.5)/128 * 2 - 1`. Not the
  `t/127 * 2 - 1` variant used in two of the prior notebooks.
- **Causal restriction**: only k < q pairs enter any statistic.

## Sample

100 chairs drawn from `03001627_train.npz` with `numpy.random.default_rng(2027)`
from those with 50-800 faces, fixed before any extraction. A disjoint 100-chair
replication set is drawn with the same call and held out; it is not touched
until the primary analysis is complete and is reported whatever it shows.

Statistics are streamed per forward pass and the attention tensors discarded;
nothing requires holding a (32, T, T) tensor per layer. Chair count is a
deliberate increase over the 10 used previously.

## Design validation (dataset only, no model involved)

Both confirmatory designs were checked for feasibility before this document was
frozen, using face centroids alone on a 60-chair seeded sample. No model weights
were loaded and no attention was extracted.

**Experiment 1 is identifiable.** Over causal (q,k) face pairs, sequence
distance and 3D distance correlate at Spearman +0.569 median (p05 +0.218,
p95 +0.801). They share roughly a third of their variance, which leaves ample
independent variation for a partial correlation to resolve. Had this been above
about 0.9 the experiment would have had no power and the design would have been
abandoned here.

**Experiment 2 is a clean swap, not a degradation.** Re-sorting faces
lexicographically with x as the primary key moves the confound completely:

| ordering | index vs height | index vs x |
|---|---|---|
| canonical | +0.953 | -- |
| x-sorted | +0.015 | +1.000 |

Height becomes independent of sequence position while the mesh remains fully
sorted, which is the same kind of object the model was trained on. The
manipulation relocates the confounded axis rather than destroying the ordering
structure, which is why it is preferred here over a random face shuffle. A
shuffle remains available as a fallback if the Experiment 2 validity gate trips.

**Correction, recorded with the results below.** That last sentence is wrong and
the run showed why. The x-sorted input raised cross-entropy roughly fortyfold,
far past the gate. A random shuffle destroys more ordering structure than a
re-sort does, so it can only score worse; there is no fallback in that
direction. The confound may not be breakable by reordering at all.

## Experiment 1 (confirmatory): does spatial proximity survive the control?

Primary test, on unmodified canonically-sorted meshes, fully in distribution.

For each head at each layer, over all causal (q,k) face pairs pooled across the
100 chairs, compute two partial Spearman correlations:

- `rho_spatial` = corr(a, -d_3d) controlling for d_seq
- `rho_seq` = corr(a, -d_seq) controlling for d_3d

Primary statistic: the median `rho_spatial` across **active heads in the final
layer**. Confidence intervals by bootstrap over chairs, 1000 resamples.

**Interpretation rule, fixed now:**
- median `rho_spatial` >= +0.20 with CI excluding 0 -> spatial selectivity
  survives the control. The prior claim is supported and reported as supported.
- median `rho_spatial` < +0.10 while median `rho_seq` >= +0.30 -> the prior
  claim is refuted; attention in this model is recency, not geometry, and the
  paper reports the refutation as its result.
- Anything between -> reported as inconclusive at this n, with the interval.

No head is dropped after seeing its value. The active/dormant split is computed
before any correlation is looked at.

## Experiment 2 (confirmatory): axis re-sort

The cleanest decorrelation available that keeps the input *sorted*, and so stays
far closer to the training distribution than a shuffle. Faces are re-sorted so
that sequence index tracks the x axis instead of the vertical axis, using the
same lexicographic scheme. Height and index are then near-independent while the
input remains a sorted mesh of the same geometry.

For each head, `rho_spatial` and `rho_seq` are recomputed under x-sorted input.

**Interpretation rule:** a genuinely spatial head retains `rho_spatial` within
0.10 of its canonical-order value. A recency head's apparent height selectivity
transfers to x. The reported statistic is the per-head change in `rho_spatial`
between orderings, with a bootstrap CI over chairs.

**Validity gate:** mean next-token cross-entropy under re-sorted input is
recorded and reported. If it exceeds twice the canonical-order loss, the model
is off-distribution enough that Experiment 2 is reported as a distribution-shift
result rather than as evidence about head function, and Experiment 1 stands as
the primary evidence.

## Experiment 3 (confirmatory): is the sink inherited from OPT?

MeshXL loads pretrained OPT weights and reinitializes only the word and
positional embeddings. The transformer body is inherited across a total
vocabulary and modality swap. This is a natural experiment the language-only
literature cannot run.

Per-head sink attention is measured on a 24x32 grid for MeshXL-1.3b on meshes,
and for `facebook/opt-1.3b` on a matched-length sample of text. The two grids
are compared by Spearman correlation over all 768 heads.

**Interpretation rule:**
- rho >= 0.5 -> sink head identity is inherited through the modality swap; sinks
  live in the QK circuits of the body, not in embeddings or data distribution.
- rho <= 0.2 -> sinks re-form independently under the new modality.
- in between -> reported as partial inheritance with the interval.

Reported whichever way it lands.

## Not allowed

Changing the active/dormant threshold, the sink definition, the chair sample,
or the correlation estimator after seeing any result. Dropping heads or layers
post hoc. Substituting the replication set for the primary set. Any bug fix
forced by the data is committed, documented here, and triggers a full rerun of
every affected arm.

## Exploratory, explicitly not confirmatory

Reported as exploratory and not subject to the rules above:

- Per-layer, per-feature linear probes for the ground-truth feature set that
  this domain supplies for free: coordinate slot (`i mod 9`), axis (`i mod 3`),
  vertex (`(i mod 9) // 3`), coordinate value (128 ordered bins), face index,
  and per-face centroid height, area, normal direction, and step distance.
  Categorical features use classification probes and metric features use
  regression probes; the contrast between the two is the point.
- The head taxonomy plot: every head as a point in (`rho_seq`, `rho_spatial`).
- Re-running the GradSAM raw-vs-gradient correlation that previously failed on a
  chair-index mismatch.

## Amendments

**A1, recorded before any model was run on any chair.** Experiment 1 above says
the partial correlations are computed "over all causal (q,k) face pairs pooled
across the 100 chairs". They are instead computed **per chair**, and the primary
statistic is the median over chairs of the per-head value.

Reason: pair count grows as the square of face count. The sample spans 50 to 800
faces, so pooling raw pairs would give an 800-face chair roughly 250 times the
weight of a 50-face chair, and the pooled number would mostly describe the
largest meshes. Computing per chair gives every mesh one vote and makes the
bootstrap over chairs, which the design already specified, exact rather than
approximate.

Nothing about the interpretation rules, thresholds, sample, or held-out set
changes. No attention had been extracted under this design when this amendment
was written; the change is a property of the estimator, not a response to a
result.

## Results

Recorded 2026-08-19, after the tests ran as specified.

### Experiment 1: INCONCLUSIVE

100 chairs, canonical order, final layer, 19 of 32 heads active.

    rho_spatial = +0.194   95% CI [+0.165, +0.206]
    rho_seq     = +0.430   95% CI [+0.412, +0.472]

Support required rho_spatial >= 0.20 with a CI excluding zero; refutation
required < 0.10 alongside rho_seq >= 0.30. The estimate falls six thousandths
under the support threshold, so the verdict is INCONCLUSIVE. The threshold was
not moved, no other layer was substituted, and the sink cutoff was not adjusted.

This is not a power problem. The interval is 0.04 wide, so additional chairs
will tighten it around 0.19 rather than resolve it; the value sits inside the
dead band the thresholds carved out.

### Experiment 2: GATE TRIPPED, halted at 37 of 100 chairs

    canonical mean cross-entropy = 0.055
    x-sorted  mean cross-entropy = 2.203
    ratio = 40.0x                     (gate at 2x)
    per-chair ratio: min 12.2x, median 48.7x, max 124.2x, higher in 37/37

Per the gate, E2 is reported as a distribution-shift result and not as evidence
about head function; E1 stands as the primary evidence. Head-level statistics
from this arm are not reported.

The run was stopped at 37 chairs rather than completed. The gate outcome was not
in doubt: the smallest per-chair ratio observed was still six times the gate, and
every chair moved the same way. Completing it would have produced head-level
numbers the gate already forbids reporting. `results/e2_validity_gate.npz` holds
the per-chair losses and `results/run_e1_e2.log` the full run log.

The measurement means something on its own, even though the arm failed. A
re-sort changes no geometry whatsoever: same vertices, same triangles, same
shape, only the order of presentation. Losing that much predictive power to a
permutation says the model is modelling the sequence rather than the shape,
which is an argument for the same conclusion E1 points toward and does not
depend on where the E1 threshold was placed.

### Experiment 3: not run

Coded and provisioned, not yet executed.

### Exploratory, not confirmatory

Per-head medians over the 100 chairs, all 768 head slots:

  - no head reaches rho_spatial 0.50 anywhere; the maximum is +0.469, while
    rho_seq reaches +0.951
  - of the 263 active heads, 10 are more spatial than recency-driven, 3.8%
  - median over active heads: +0.147 spatial against +0.591 recency
  - layers 0-11: median rho_spatial -0.043, 65 of 84 active heads at or below zero
  - layers 12-23: median rho_spatial +0.260, 6 of 179 at or below zero

So no head in this model is primarily geometric, and the first half of the
network carries no geometric selectivity once sequence position is controlled.

### Carried forward

The held-out 100 chairs remain untouched. Reordering appears to be a dead end
for decorrelation, so the natural round-2 design is other object categories --
the sft checkpoint ships table, lamp, and bench -- where the coupling between
height and sequence index should vary by shape without any manipulation and
without leaving the training distribution.
