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

## Power, checked before running

A rare binary predictor attenuates rank correlation, so a threshold that sounds
modest can be unreachable in practice. At the measured adjacency rate of 1.15%
of causal pairs, with roughly 20k pairs per chair, simulated attention that is
multiplied on adjacent pairs gives:

| attention multiplier on adjacent pairs | rho_adj |
|---|---|
| 1x (null) | 0.001 |
| 2x | 0.172 |
| 5x | 0.379 |
| 25x | 0.616 |

So the +0.15 support threshold corresponds to roughly a doubling of attention on
adjacent pairs, which is an interpretable effect size rather than a wish, and the
null is well behaved. Both properties are pinned in `tests/test_round2.py` so
they cannot drift.

Recorded because the first version of that simulation used an adjacency rate near
0.005% by mistake and made the probe look hopelessly underpowered. The rate is
the parameter this design lives or dies on.

**Realized rate, noted while the run was in progress and before any result was
seen.** The held-out chairs are coming in nearer 0.70% adjacent pairs than the
1.15% measured on the design sample, so the probe is somewhat more attenuated
than the table above assumes:

| attention multiplier | rho at 1.15% | rho at 0.70% |
|---|---|---|
| 1x (null) | 0.001 | 0.006 |
| 2x | 0.175 | 0.146 |
| 5x | 0.369 | 0.308 |

Reaching +0.15 takes roughly 2.2x attention on adjacent pairs at the realized
rate, against 1.9x at the assumed one. The threshold stays reachable and the null
stays clean, so the rule is unchanged. The realized rate is reported with the
result rather than compared silently against a threshold calibrated on a rate the
data did not deliver.

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
- median `rho_adj` < +0.05, with the CI excluding +0.15 -> no topological structure.
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

Recorded 2026-08-20, after the tests ran as specified, on the 100 held-out chairs.

### Experiment 4: REFUTED

Final layer, 20 of 32 heads active.

    rho_adj = +0.041   95% CI [+0.037, +0.044]

The rule required +0.15 with a CI excluding zero to call topology present, and
below +0.05 with the CI excluding +0.15 to call it absent. The estimate is
+0.041 with an upper bound of +0.044, so it is refuted: attention carries no
topological structure once sequence distance and 3D distance are controlled.

Not a single one of the 261 active head slots in the whole network exceeds the
+0.15 band, and the maximum anywhere is +0.074. The realized adjacency rate was
0.73% of causal pairs against the 1.15% the power table assumed, which raises the
effect needed to clear +0.15 from roughly 1.9x to 2.2x attention on adjacent
pairs. That penalty is far too small to account for a result this flat.

### Experiment 5: rule does not apply, but the control is the interesting number

    rho_shared = +0.086   95% CI [+0.082, +0.093]

The preregistered E5 rule asks whether an adjacency effect is really token
matching. There is no adjacency effect for it to explain, so the rule is silent.

The comparison is worth reporting anyway, as exploratory. Over all 261 active
head slots, shared-token sensitivity exceeds adjacency in 93.5% of them, with
medians of +0.095 against +0.047, almost exactly a factor of two. Ten heads clear
+0.15 on shared tokens; none do on adjacency. The ordering holds in all 24
layers.

Shared vertex positions here are counted over non-adjacent pairs only, so this
is not adjacency leaking in under another name. It is sensitivity to repeated
coordinate values between faces that do not touch.

That is a lexical mechanism, not a geometric one. What little structure survives
the controls looks like a model matching repeated token values, which is what a
language model does, rather than tracking the surface it is building.

### What round 2 settles

Round 1 left the metric-proximity effect at about 0.19, precisely estimated and
sitting in a dead band, with the question of whether anything geometric survives
the recency control unresolved. Round 2 answers it for the sharpest available
geometric feature, chosen precisely because its entanglement with the confound is
-0.153 rather than +0.569: mesh topology is not tracked, and the nearest thing to
a positive result is token repetition.
