"""Vector representations of each modality for models that work on flat float vectors.

- image:   flattened pixels in [0, 1], optionally reduced by PCA
- scrna:   log-normalized expression, optionally reduced by PCA; decoded back to counts by Poisson
           sampling at an empirical library size
- tabular: TabularCodec vectors (standardized numeric columns, then one-hot blocks); decoded by
           un-standardizing and argmax per block
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from ..embed import scrna_counts_from_lognorm, scrna_lognorm


def sample_libsizes(ctx, y: np.ndarray | None, n: int, rng: np.random.Generator) -> np.ndarray:
    """Library sizes for generated cells: drawn from the training cells of the same class when known."""
    if y is not None and ctx.libsizes_by_class:
        return np.array([rng.choice(ctx.libsizes_by_class[int(c)]) for c in y], dtype=np.float32)
    return rng.choice(ctx.libsizes, size=n).astype(np.float32)


class VectorSpace:
    """Fit on training data with `fit`; `encode` maps native samples to float32 vectors, `decode`
    maps vectors back to native samples. With `pca`, vectors are the top principal components (not
    available for tabular data, whose one-hot blocks must stay identifiable)."""

    def __init__(self, ctx, pca: int | None = None, standardize: bool = True):
        if pca and ctx.modality == "tabular":
            raise ValueError("PCA is not supported for tabular vectors")
        self.ctx, self.n_pca, self.standardize = ctx, pca, standardize
        self.pca = None

    # -- layout ------------------------------------------------------------------
    @property
    def num_dim(self) -> int:
        """Number of leading continuous coordinates (all of them except tabular one-hot blocks)."""
        return len(self.ctx.codec.num) if self.ctx.modality == "tabular" else self.dim

    @property
    def cat_blocks(self) -> list[slice]:
        """Slices of the one-hot blocks (tabular only)."""
        if self.ctx.modality != "tabular":
            return []
        out, j = [], self.num_dim
        for c in self.ctx.codec.cat:
            k = len(self.ctx.codec.levels[c])
            out.append(slice(j, j + k))
            j += k
        return out

    # -- conversion --------------------------------------------------------------
    def _raw(self, X) -> np.ndarray:
        m = self.ctx.modality
        if m == "scrna":
            return scrna_lognorm(X)
        if m == "tabular":
            return self.ctx.codec.encode(X)
        return np.asarray(X, dtype=np.float32).reshape(len(X), -1)

    def fit(self, X) -> np.ndarray:
        E = self._raw(X)
        if self.n_pca:
            self.pca = PCA(min(self.n_pca, *E.shape), random_state=self.ctx.seed).fit(E)
            E = self.pca.transform(E)
        self.dim = E.shape[1]
        if self.standardize:
            self.mu, self.sd = E.mean(0), E.std(0) + 1e-6
        else:
            self.mu, self.sd = np.zeros(self.dim, np.float32), np.ones(self.dim, np.float32)
        return ((E - self.mu) / self.sd).astype(np.float32)

    def encode(self, X) -> np.ndarray:
        E = self._raw(X)
        if self.pca is not None:
            E = self.pca.transform(E)
        return ((E - self.mu) / self.sd).astype(np.float32)

    def decode(self, E: np.ndarray, y: np.ndarray | None, rng: np.random.Generator):
        E = E * self.sd + self.mu
        if self.pca is not None:
            E = self.pca.inverse_transform(E)
        m = self.ctx.modality
        if m == "scrna":
            return scrna_counts_from_lognorm(E, sample_libsizes(self.ctx, y, len(E), rng), rng)
        if m == "tabular":
            return self.ctx.codec.decode(E)
        return np.clip(E, 0, 1).reshape(len(E), *self.ctx.shape).astype(np.float32)


def concat_samples(parts: list, modality: str):
    """Join per-batch samples in the modality's native container."""
    if modality == "tabular":
        import pandas as pd

        return pd.concat(parts, ignore_index=True)
    return np.concatenate(parts).astype(np.float32)
