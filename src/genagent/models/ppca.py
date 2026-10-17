"""Probabilistic PCA (Tipping & Bishop, 1999), fitted in closed form.

x = mu_c + W z + sigma * eps,  z ~ N(0, I_q),  eps ~ N(0, I_d)

The maximum-likelihood solution comes from the eigendecomposition of the sample covariance S:
sigma^2 is the mean of the d - q discarded eigenvalues and W = U_q (Lambda_q - sigma^2 I)^(1/2).
With labels, each class has its own mean mu_c while W and sigma^2 are shared and fitted to the
pooled within-class covariance (well-posed even for classes with a handful of samples).
Vectors: pixels (images), log-normalized expression (scRNA), codec vectors (tabular)."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from ..utils import batches
from .base import GenerativeModel, register
from .spaces import VectorSpace, concat_samples


@register
class PPCA(GenerativeModel):
    name = "ppca"
    family = "latent_variable"
    modalities = ("image", "scrna", "tabular")
    library = "from scratch (numpy, closed-form maximum likelihood)"
    conditional = "class-specific means with a shared loading matrix and noise variance"
    capabilities = ("sample", "encode", "anomaly")
    summary = ("Probabilistic PCA: a linear-Gaussian latent variable model fitted in closed form in seconds. "
               "Exact log-likelihood (anomaly score) and a linear latent space. Samples are Gaussian around a "
               "linear subspace, so non-linear structure, skewed or discrete features and sparsity are smoothed "
               "away: a fast baseline, not a high-fidelity generator.")

    @classmethod
    def default_config(cls, ctx):
        return {"n_components": {"image": 32, "scrna": 20}.get(ctx.modality, 8)}

    @classmethod
    def search_space(cls, ctx):
        choices = [4, 8, 16] if ctx.modality == "tabular" else [10, 20, 50]
        return {"n_components": {"type": "categorical", "choices": choices}}

    def fit(self, X, y, X_val, y_val, deadline):
        # Pixels and codec vectors are already on a common scale; log-normalized genes keep their variances
        self.space = VectorSpace(self.ctx, standardize=False)
        E = self.space.fit(X)
        n, d = E.shape
        q = int(min(self.config["n_components"], d - 1, n - 1))
        self.mu = E.mean(0, dtype=np.float64)
        if y is not None:
            # Class means; a class absent from training (anomaly hold-out) falls back to the overall mean
            self.mu_c = np.stack([E[y == c].mean(0, dtype=np.float64) if np.any(y == c) else self.mu
                                  for c in range(self.ctx.n_classes)])
        else:
            self.mu_c = None

        def resid(sl):
            return E[sl].astype(np.float64) - (self.mu_c[y[sl]] if self.mu_c is not None else self.mu)

        # Leading eigenpairs of S = R^T R / n (R = residuals), from the smaller of the two Gram matrices
        if n >= d:
            S = np.zeros((d, d))
            for sl in batches(n, 8192):  # chunked, so large image sets need no float64 copy of the data
                R = resid(sl)
                S += R.T @ R
            S /= n
            trace = float(np.trace(S))
            lam, U = np.linalg.eigh(S)
            lam, U = lam[::-1][:q], U[:, ::-1][:, :q]
        else:
            R = resid(slice(0, n))
            trace = float((R ** 2).sum() / n)
            _, s, Vt = np.linalg.svd(R / np.sqrt(n), full_matrices=False)
            lam, U = s[:q] ** 2, Vt[:q].T
        # sigma^2 = mean of the discarded eigenvalues = (tr S - sum of the kept ones) / (d - q)
        self.sigma2 = max((trace - lam.sum()) / (d - q), 1e-8)
        self.lam = np.maximum(lam, self.sigma2)
        self.U = U
        self.W = U * np.sqrt(self.lam - self.sigma2)
        self.q, self.d = q, d
        self.history = {"n_components": q, "sigma2": self.sigma2,
                        "explained_variance_ratio": float(lam.sum() / trace) if trace > 0 else 0.0,
                        "epochs_run": 1, "stopped_by_budget": False}
        return self.history

    def sample(self, n, y=None):
        rng = np.random.default_rng(self.ctx.seed + 29)
        y = self._labels_for(n, y)
        out = []
        for sl in batches(n, 4096):
            m = sl.stop - sl.start
            ys = y[sl] if y is not None and self.mu_c is not None else None
            mean = self.mu_c[ys] if ys is not None else self.mu
            E = mean + rng.standard_normal((m, self.q)) @ self.W.T + np.sqrt(self.sigma2) * rng.standard_normal((m, self.d))
            out.append(self.space.decode(E.astype(np.float32), ys, rng))
        return concat_samples(out, self.ctx.modality)

    def encode(self, X):
        # Posterior mean M^-1 W^T (x - mu) with M = W^T W + sigma^2 I; overall mean (no label at encoding)
        R = self.space.encode(X).astype(np.float64) - self.mu
        M = self.W.T @ self.W + self.sigma2 * np.eye(self.q)
        return np.linalg.solve(M, self.W.T @ R.T).T.astype(np.float32)

    def _log_density(self, E, mean):
        """log N(x; mean, W W^T + sigma^2 I). The covariance has eigenvalues lam (on U) and sigma^2
        elsewhere, so the inverse and log-determinant follow without forming the d x d matrix."""
        r = E - mean
        proj = r @ self.U
        maha = ((r ** 2).sum(1) - (proj ** 2 * (1 - self.sigma2 / self.lam)).sum(1)) / self.sigma2
        logdet = np.log(self.lam).sum() + (self.d - self.q) * np.log(self.sigma2)
        return -0.5 * (self.d * np.log(2 * np.pi) + logdet + maha)

    def log_likelihood(self, X) -> np.ndarray:
        E = self.space.encode(X).astype(np.float64)
        if self.mu_c is None:
            return self._log_density(E, self.mu)
        # Mixture over classes weighted by their training frequencies
        logp = np.stack([self._log_density(E, self.mu_c[c]) + np.log(max(self.ctx.class_freq[c], 1e-12))
                         for c in range(self.ctx.n_classes)])
        return logsumexp(logp, axis=0)

    def anomaly_score(self, X):
        return -self.log_likelihood(X)
