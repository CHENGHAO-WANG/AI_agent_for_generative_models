"""Build an artificially imbalanced PathMNIST variant for the class-balancing appendix demo.

Keeps 10% of the training images of three classes (seeded); validation and test splits are
unchanged so evaluation stays on the original distribution. Writes a MedMNIST-style .npz with
`class_names`, loadable by `genagent init --data <file>`.

Usage: uv run python scripts/make_imbalanced_pathmnist.py [out.npz]
"""

import sys
from pathlib import Path

import numpy as np

REDUCED = {"mucus": 0.10, "normal colon mucosa": 0.10, "cancer-associated stroma": 0.10}
SEED = 0


def main(out: str = "data_cache/pathmnist_imbalanced.npz") -> None:
    from medmnist import INFO, PathMNIST

    Path("data_cache").mkdir(exist_ok=True)
    PathMNIST(split="train", download=True, size=28, root="data_cache")
    z = np.load("data_cache/pathmnist.npz")
    names = [INFO["pathmnist"]["label"][str(i)] for i in range(len(INFO["pathmnist"]["label"]))]
    y = z["train_labels"].reshape(-1)
    rng = np.random.default_rng(SEED)
    keep = np.ones(len(y), bool)
    for cname, frac in REDUCED.items():
        idx = np.flatnonzero(y == names.index(cname))
        drop = rng.choice(idx, len(idx) - int(round(frac * len(idx))), replace=False)
        keep[drop] = False
    arrays = {k: z[k] for k in z.files}
    arrays["train_images"], arrays["train_labels"] = z["train_images"][keep], z["train_labels"][keep]
    np.savez_compressed(out, class_names=np.array(names), **arrays)
    counts = np.bincount(arrays["train_labels"].reshape(-1), minlength=len(names))
    print(out, dict(zip(names, counts.tolist())))


if __name__ == "__main__":
    main(*sys.argv[1:])
