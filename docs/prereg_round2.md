# Preregistration round 2: does attention track mesh topology?

Committed before any attention is extracted under this design. Written after the
dataset-level design validation below, which used no model weights, and after
round 1's results were recorded in `prereg_round1.md`.

## Why this exists

Round 1 left one question open and closed two routes to answering it.

Open: at the final layer, attention's partial correlation with 3D proximity is
+0.194 (95% CI [+0.165, +0.206]) once sequence distance is controlled. That is a
real effect with a tight interval, and it sits inside the dead band the round 1
thresholds carved out. It is not nothing, and it is not much.

Closed, first route: re-ordering faces. Round 1's Experiment 2 raised
cross-entropy roughly fortyfold and tripped its own validity gate. Any
reordering appears to destroy the model, so the confound cannot be broken that
way.

Closed, second route: other object categories. The idea was that a lamp and a
table would couple height to sequence position differently, giving natural
variation with no manipulation. Measured across all four categories the sft
checkpoint ships, they do not:

| category | meshes | median rho(face index, height) | R^2 height ~ position |
|---|---|---|---|
| chair | 2820 | +0.951 | 0.858 |
| table | 4632 | +0.914 | 0.778 |
| lamp | 564 | +0.968 | 0.844 |
| bench | 504 | +0.939 | 0.826 |

The spread is 0.054 in correlation and 0.081 in R^2. That is because the sort is
a property of MeshXL's preprocessing, not of the object: every mesh in every
category is sorted the same way. There is no natural variation to exploit.

So this round changes the question rather than the manipulation. Instead of
asking whether attention tracks *metric proximity*, which is badly entangled
with sequence position, it asks whether attention tracks *mesh topology*, which
is not.

## The probe

Two faces are topologically adjacent when they share an edge. The dataset ships
this directly as `face_edges`, so it is exact ground truth requiring no
annotation, no threshold, and no metric choice.

Adjacency is the thing a mesh generator would actually need in order to build a
closed surface, and it is much less entangled with the confound than distance
is. Design validation on a 40-chair seeded sample, centroids and topology only,
no model:

| quantity | value |
|---|---|
| Spearman(adjacency, sequence distance) | −0.153 median, p05 −0.244, p95 −0.099 |
| Spearman(adjacency, 3D distance) | −0.165 median, p05 −0.319, p95 −0.104 |
| causal face pairs that are adjacent | 1.15% |
| adjacent pairs more than 3 apart in sequence | **52.0%** |

Compare the entanglement round 1 had to work against: 3D distance correlates
+0.569 with sequence distance. Adjacency correlates −0.153. Slightly over half
of all adjacency is invisible to a recency account, which is what makes this
worth running.

## Definitions fixed in advance

Carried unchanged from `prereg_round1.md`: the sink region, the active/dormant
threshold of 0.5, face-level attention with sink columns excluded and causal
rows renormalized, MeshXL's own `undiscretize`, and the per-chair-then-aggregate
estimator from amendment A1.

New here:

- **Adjacency** A(q,k) = 1 when faces q and k share an edge per `face_edges`,
  else 0. Symmetric; self-pairs dropped.
- **rho_adj** = partial Spearman between face attention and adjacency,
  controlling for **both** sequence distance and 3D distance. Second-order
  partial, computed by regressing the ranks of attention and of adjacency on the
  ranks of both controls and correlating the residuals.
- Only chairs with at least 20 adjacent causal pairs enter, fixed here so that
  the inclusion rule cannot be tuned later.

## Sample

The 100-chair held-out split from round 1, drawn by
`numpy.random.default_rng(2027)` and recorded in `meshlens/data.py`. Round 1's
primary split is not reused, so this round's estimate is not contaminated by the
meshes that produced the +0.194.

## Experiment 4 (confirmatory): does attention track adjacency?

Primary statistic: median `rho_adj` across active heads in the final layer,
bootstrap CI over chairs, 1000 resamples.

**Interpretation rule, fixed now:**

- median `rho_adj` >= +0.15 with CI excluding zero -> attention carries
  genuine topological structure beyond recency and metric proximity. Reported as
  evidence that something geometric survives the controls.
- median `rho_adj` < +0.05, CI excluding +0.15 -> no topological structure.
  Combined with round 1 this is a strong recency account, and is reported as one.
- anything between -> reported as inconclusive with the interval, as round 1's
  E1 was, with no threshold adjustment.

Secondary, exploratory: `rho_adj` by layer, and its relationship to the
`rho_spatial` head taxonomy from round 1.

## Experiment 5 (confirmatory): is adjacency preference explained by shared vertices?

Adjacent faces share two vertices, so they share six of their eighteen
coordinate tokens. A head that appeared to track adjacency might only be
matching repeated token values.

Control: among **non-adjacent** causal pairs, count shared vertex positions
(0, 1, or 2 shared vertices is possible for non-adjacent faces in a
non-manifold mesh; exact token-value matches are counted directly). Compute
`rho_shared`, the partial Spearman between attention and shared-token count,
controlling for sequence and 3D distance.

**Interpretation rule:** if `rho_adj` is positive and `rho_shared` is
comparable, the adjacency result is token matching rather than topology, and is
reported as such. If `rho_adj` clearly exceeds `rho_shared`, the topological
reading survives this control.

## Not allowed

Changing the adjacency definition, the inclusion rule, the thresholds, or the
split after seeing any result. Substituting round 1's primary split for the
held-out split. Dropping layers or heads post hoc. Any bug fix forced by the
data is committed, documented here, and triggers a full rerun.

## Results

(To be appended after the tests run as specified, whatever they show.)
