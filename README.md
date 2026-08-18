# 3D Mesh Transformer Mechanistic Interpretation

This repo is ongoing work on mechanistic interpretation for 3D mesh transformers. The main goal is to understand what an autoregressive mesh generation model attends to when it predicts the next spatial coordinate token.

The model represents each triangle as nine coordinate tokens. It predicts these tokens one at a time to build a mesh. This makes it possible to study how attention changes across triangles, token positions, layers, and attention heads during generation.

## Environment setup

This project uses Python 3.11 and uv for dependency and environment management.

Install uv if it is not already available.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create or update the local environment from the lockfile.

```bash
uv sync --locked
```

Activate the environment if you want to use it directly in a terminal.

```bash
source .venv/bin/activate
```

The `.venv` directory stays local and should not be uploaded. Commit `pyproject.toml`, `uv.lock`, and `.python-version` so the environment can be reproduced on another machine.

## Research focus

This work explores the following questions.

- Which parts of a mesh does the model use when predicting the next coordinate token?
- Do different attention heads learn different spatial roles?
- Which heads are active, and which heads are dormant?
- Why do the BOS token and early triangle tokens act as attention sinks?
- What purpose do sink tokens serve during mesh generation?
- Does removing or replacing sink tokens affect attention, entropy, mesh quality, or generation length?
- Can several spatial sink tokens work better than one sink at the start of the sequence?

## Notebooks

### [Attention Head Analysis](Attention_Head_Analysis.ipynb)

This is the broader attention head analysis. It compares several ways of measuring attention.

- Raw attention shows where each head places attention weight.
- GradSAM keeps both helpful and harmful gradient contributions.
- Attention weighted by value norm gives a rough measure of information flow.
- Head removal estimates which heads help or hurt the prediction.

The notebook also compares heads across layers and chair meshes. It separates active heads from dormant heads based on how much attention they place on the BOS token.

### [Attention Head GradCAM](Attention_Head_GradCAM.ipynb)

This notebook compares raw attention with GradCAM attention. Raw attention shows where the model looks. GradCAM estimates which attended regions affect the prediction.

It compares attention across heads, layers, query triangles, and different chair meshes. The goal is to see whether attention head behavior is consistent and whether the regions receiving attention are the regions that matter for prediction.

### [Sink Removal Experiments](sink_removal_experiments.ipynb)

This notebook tests what happens when attention sink tokens are changed or removed during generation.

- Generate without the BOS token.
- Replace the first triangle with neutral coordinate tokens.
- Compare attention entropy and concentration with and without the sink.
- Repeat the comparison across several random seeds.
- Measure how strongly each query position attends to the BOS token and the first triangle.

### [Spatial Sink Analysis](spatial_sink_analysis_v2.ipynb)

This notebook studies whether sink attention is related to spatial position.

- Compare sink attention with triangle position.
- Measure distance from the mesh center and the first triangle.
- Control for sequence position when measuring spatial effects.
- Test whether nearby triangles can take the place of a sink.
- Use spatial clusters to find possible locations for additional sink tokens.

### [Spatial Scaffold Experiment](spatial_scaffold_experiment_v2.ipynb)

This notebook tests spatial scaffold sink tokens. It adds eight tokens at fixed positions around the mesh and initializes them from the BOS embedding.

The experiment records how much attention goes to BOS, the scaffold tokens, and earlier triangles. It compares the generated mesh with the baseline using mesh measurements, visual comparison, and the distance between consecutive triangles.

## Supporting files

- `baseline_tokens_1000.pkl` stores baseline generation tokens used by the sink removal work.
- `exp1_remove_bos_entropy.png` shows the entropy comparison from the BOS removal experiment.
- `exp1_remove_bos_meshes.png` shows the generated meshes from the BOS removal experiment.

## Current state

This is an active research repo. The notebooks are exploratory and some depend on local model weights, ShapeNet data, and saved attention files from earlier runs.
