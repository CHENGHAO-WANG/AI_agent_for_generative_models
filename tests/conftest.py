"""Tiny synthetic fixtures for each modality, written to temporary files."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def image_npz(tmp_path):
    """240 RGB 16x16 images in 3 classes; class determines mean brightness of one channel."""
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], [120, 80, 40])
    X = rng.uniform(0, 0.4, (len(y), 16, 16, 3))
    X[np.arange(len(y)), :, :, y] += 0.5
    p = tmp_path / "imgs.npz"
    np.savez(p, images=(X * 255).astype(np.uint8), label=y)
    return p


@pytest.fixture
def table_csv(tmp_path):
    """300 rows, mixed numeric/categorical columns, a label, and some missing values."""
    rng = np.random.default_rng(0)
    n = 300
    y = rng.choice(["a", "b"], n, p=[0.75, 0.25])
    df = pd.DataFrame({
        "age": rng.normal(40, 10, n).round(),
        "income": rng.lognormal(10, 0.5, n) + (y == "b") * 5000,
        "city": rng.choice(["x", "y", "z"], n),
        "score": rng.normal(0, 1, n),
        "label": y,
    })
    df.loc[rng.choice(n, 10, replace=False), "score"] = np.nan
    p = tmp_path / "table.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def scrna_h5ad(tmp_path):
    """300 cells x 300 genes of Poisson counts with two expression programs."""
    import anndata as ad

    rng = np.random.default_rng(0)
    n, g = 300, 300
    grp = rng.integers(0, 2, n)
    rate = np.full((n, g), 2.0)
    rate[grp == 1, :50] = 8.0
    X = rng.poisson(rate).astype(np.float32)
    a = ad.AnnData(X)
    a.var_names = [f"G{i}" for i in range(g)]
    a.obs_names = [f"c{i}" for i in range(n)]
    p = tmp_path / "cells.h5ad"
    a.write_h5ad(p)
    return p
