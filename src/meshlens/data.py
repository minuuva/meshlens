"""Chair sampling and tokenization, fixed by the preregistration.

The primary and held-out samples come from one seeded permutation so that the
held-out set is determined before the primary set is ever looked at.
"""

import numpy as np

from .faces import BOS, EOS, N_DISCRETE

SEED = 2027
MIN_FACES, MAX_FACES = 50, 800
N_PRIMARY = N_HOLDOUT = 100


def load_chairs(npz_path):
    return np.load(npz_path, allow_pickle=True)["arr_0"]


def sample_splits(data, seed=SEED, n_primary=N_PRIMARY, n_holdout=N_HOLDOUT):
    """Disjoint primary and held-out chair indices from a single seeded draw."""
    eligible = np.array(
        [i for i, it in enumerate(data) if MIN_FACES <= len(np.asarray(it["faces"])) <= MAX_FACES]
    )
    perm = np.random.default_rng(seed).permutation(eligible)
    return perm[:n_primary], perm[n_primary : n_primary + n_holdout]


def resort_faces(item, primary_axis=0):
    """Re-sort faces lexicographically with `primary_axis` as the leading key.

    Experiment 2's manipulation. Only face order changes; vertex order within
    each face is untouched, so the geometry is identical and only the position
    of each face in the token stream moves. With `primary_axis=0` the sequence
    index tracks x instead of height (measured: index-vs-height +0.953 -> +0.015,
    index-vs-x -> +1.000).
    """
    vertices = np.asarray(item["vertices"], dtype=np.float64)
    faces = np.asarray(item["faces"], dtype=np.int64)
    centroids = vertices[faces].mean(axis=1)
    others = [a for a in range(3) if a != primary_axis]
    # np.lexsort takes the primary key last
    order = np.lexsort((centroids[:, others[1]], centroids[:, others[0]], centroids[:, primary_axis]))
    out = dict(item)
    out["faces"] = faces[order]
    return out


def tokenize(item, tokenizer):
    """Token ids for one mesh, with the placeholder ends replaced by BOS/EOS.

    The tokenizer marks the first and last positions with its pad id (-1), which
    the embedding layer cannot index.
    """
    import torch

    vertices = item["vertices"]
    faces = item["faces"]
    if not isinstance(vertices, torch.Tensor):
        vertices = torch.as_tensor(np.asarray(vertices))
    if not isinstance(faces, torch.Tensor):
        faces = torch.as_tensor(np.asarray(faces))
    out = tokenizer.tokenize(
        {"vertices": vertices.float().unsqueeze(0), "faces": faces.long().unsqueeze(0)}
    )
    ids = out["input_ids"].squeeze(0).clone()
    ids[0], ids[-1] = BOS, EOS
    return ids


def make_tokenizer(meshxl_root="."):
    """MeshXL's own tokenizer, imported from a checkout of the upstream repo.

    `meshxl_root` must contain `models/mesh_xl/`. Passed explicitly rather than
    inferred from the working directory, since running a script puts the
    script's own directory on the path and not the caller's cwd.
    """
    import os
    import sys
    from types import SimpleNamespace

    root = os.path.abspath(meshxl_root)
    if not os.path.isdir(os.path.join(root, "models", "mesh_xl")):
        raise FileNotFoundError(
            f"{root} does not look like a MeshXL checkout (no models/mesh_xl/). "
            "Clone https://github.com/OpenMeshLab/MeshXL and pass --meshxl-root."
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    from models.mesh_xl.tokenizer import MeshTokenizer

    return MeshTokenizer(SimpleNamespace(n_discrete_size=N_DISCRETE))
