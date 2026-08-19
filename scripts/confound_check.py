"""Does token sequence position encode vertical height in MeshXL's mesh serialization?

MeshXL's tokenizer (models/mesh_xl/tokenizer.py) performs no sorting: it gathers
face coordinates in the order they already appear in the data and flattens them
to 9 tokens per face. Face index in the stored array is therefore identical to
token sequence order, so this question can be answered from the dataset alone --
no model weights and no GPU.

This matters because any claim of the form "early layers attend to chair legs,
late layers attend to chair backs" is only meaningful if height and sequence
position are separable. Run against the full 2822-chair training set.

    python scripts/confound_check.py
"""

import numpy as np
from scipy.stats import spearmanr

NPZ = "data/shapenet/03001627_train.npz"
VERTICAL_AXIS = 1  # ShapeNet is y-up; confirmed empirically by the axis sweep below
AXES = "xyz"


def face_centroids(item):
    """Centroid of every face, in stored XYZ order and in face-index order."""
    vertices = np.asarray(item["vertices"], dtype=np.float64)
    faces = np.asarray(item["faces"], dtype=np.int64)
    return vertices[faces].mean(axis=1)


def axis_sweep(meshes):
    """Which axis does face index track? Reported per axis so it is not assumed."""
    rho = {a: [] for a in range(3)}
    for centroids in meshes:
        index = np.arange(len(centroids))
        for a in range(3):
            if centroids[:, a].std() > 1e-9:
                rho[a].append(spearmanr(index, centroids[:, a]).statistic)
    return {AXES[a]: np.array(v) for a, v in rho.items()}


def residual_variance(meshes):
    """How much height variance survives after regressing out sequence position?

    A cubic in normalized position gives position every reasonable chance to
    explain height, so the residual is a conservative estimate of what is left
    to identify a genuine spatial effect with.
    """
    r_squared, residual_ratio = [], []
    for centroids in meshes:
        height = centroids[:, VERTICAL_AXIS]
        if height.var() <= 0:
            continue
        position = np.linspace(0.0, 1.0, len(centroids))
        residual = height - np.polyval(np.polyfit(position, height, 3), position)
        r_squared.append(1 - residual.var() / height.var())
        residual_ratio.append(residual.std() / height.std())
    return np.array(r_squared), np.array(residual_ratio)


def main():
    data = np.load(NPZ, allow_pickle=True)["arr_0"]
    meshes = [c for c in map(face_centroids, data) if len(c) >= 50]
    print(f"chairs: {len(meshes)} of {len(data)} (>= 50 faces)\n")

    print("Spearman(face index, centroid axis):")
    for name, rho in axis_sweep(meshes).items():
        print(
            f"  {name}: mean={rho.mean():+.3f}  median={np.median(rho):+.3f}  "
            f"p05={np.percentile(rho, 5):+.3f}  frac|r|>0.9={np.mean(np.abs(rho) > 0.9):.3f}"
        )

    r_squared, residual_ratio = residual_variance(meshes)
    print("\nheight ~ cubic(normalized sequence position):")
    print(f"  R^2                      mean={r_squared.mean():.3f}  median={np.median(r_squared):.3f}")
    print(f"  residual std / total std mean={residual_ratio.mean():.3f}")
    print(f"  chairs with R^2 > 0.95   {np.mean(r_squared > 0.95):.3f}")


if __name__ == "__main__":
    main()
