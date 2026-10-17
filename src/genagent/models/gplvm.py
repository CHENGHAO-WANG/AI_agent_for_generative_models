"""Bayesian GPLVM (Titsias & Lawrence, 2010) with GPyTorch.

Each output dimension is an independent sparse variational GP f_j(z) of a low-dimensional latent z
(shared ARD-RBF kernel, inducing points per output), with Gaussian noise. Every training point has
a variational posterior q(z_n) = N(m_n, s_n^2) and a N(0, I) prior; training maximizes the ELBO
on minibatches. Vectors: PCA-reduced, standardized pixels (images) or log-normalized expression
(scRNA); standardized codec vectors (tabular).

Labels are not used in training. To generate, latents are drawn from a Gaussian fitted to the
learned posteriors (per class when labeled, shrunk toward the pooled covariance) and passed
through the GP predictive, with output noise. For new data, `encode` optimizes a point estimate of
z under the fixed GP (no amortized encoder), and the anomaly score is the negative log joint
density at that point; both are approximations."""

from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.decomposition import PCA

from ..utils import batches, device
from .base import GenerativeModel, register
from .spaces import VectorSpace, concat_samples


def _subsample(n: int, y: np.ndarray | None, cap: int, rng: np.random.Generator, floor: int = 20) -> np.ndarray:
    """At most `cap` indices, proportional per class but keeping at least `floor` of each class."""
    if n <= cap:
        return np.arange(n)
    if y is None:
        return np.sort(rng.choice(n, cap, replace=False))
    out = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        k = min(len(idx), max(floor, int(round(cap * len(idx) / n))))
        out.append(rng.choice(idx, k, replace=False))
    return np.sort(np.concatenate(out))


_MODEL_CLASS = None


def _model_class():
    """The GPyTorch model class, built on first use so that importing the registry does not import
    gpytorch. Exposed as the module attribute `_BGPLVM` so saved models can be unpickled."""
    global _MODEL_CLASS
    if _MODEL_CLASS is not None:
        return _MODEL_CLASS
    import gpytorch
    from gpytorch.models.gplvm import BayesianGPLVM, VariationalLatentVariable
    from gpytorch.priors import NormalPrior
    from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

    class _BGPLVM(BayesianGPLVM):
        def __init__(self, n: int, d: int, q: int, m: int, X_init: torch.Tensor):
            batch = torch.Size([d])
            # Inducing inputs start at latent positions of random training points
            Z = X_init[torch.randperm(n)[:m]].clone()
            inducing = Z.unsqueeze(0).repeat(d, 1, 1)
            q_u = CholeskyVariationalDistribution(len(Z), batch_shape=batch)
            q_f = VariationalStrategy(self, inducing, q_u, learn_inducing_locations=True)
            prior = NormalPrior(torch.zeros(n, q), torch.ones(n, q))
            latent = VariationalLatentVariable(n, d, q, X_init.clone(), prior)
            super().__init__(latent, q_f)
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=q))

        def forward(self, z):
            return gpytorch.distributions.MultivariateNormal(self.mean_module(z), self.covar_module(z))

    _BGPLVM.__module__, _BGPLVM.__qualname__ = __name__, "_BGPLVM"
    _MODEL_CLASS = _BGPLVM
    return _BGPLVM


def __getattr__(name):
    if name == "_BGPLVM":
        return _model_class()
    raise AttributeError(name)


@register
class GPLVM(GenerativeModel):
    name = "gplvm"
    family = "latent_variable"
    modalities = ("image", "scrna", "tabular")
    library = "GPyTorch (Bayesian GPLVM, sparse variational GP)"
    conditional = ("none in the model: per-class Gaussian over the learned latent positions of that class "
                   "(shrunk toward the pooled covariance), decoded through the GP")
    capabilities = ("sample", "encode", "anomaly")
    summary = ("Bayesian Gaussian-process latent variable model: a smooth non-linear map from a low-dimensional "
               "latent space with calibrated uncertainty; strong on small data (hundreds to a few thousand "
               "points), trained on at most `max_points`. scRNA and images are modeled on their top principal "
               "components (variation outside them is lost). Class samples come from latent regions of that "
               "class, not a label-conditioned model. Encoding/anomaly scores of new data optimize the latent "
               "point under the fixed GP (approximate).")

    @classmethod
    def default_config(cls, ctx):
        return {"epochs": 300 if ctx.n_train < 5000 else 100, "batch_size": 256, "lr": 0.01, "latent_dim": 5,
                "n_inducing": 64, "max_points": 5000, "output_pca": 0 if ctx.modality == "tabular" else 50,
                "encode_steps": 200}

    @classmethod
    def search_space(cls, ctx):
        return {"latent_dim": {"type": "categorical", "choices": [2, 5, 10]},
                "n_inducing": {"type": "categorical", "choices": [32, 64, 128]},
                "lr": {"type": "float", "low": 3e-3, "high": 3e-2, "log": True}}

    def fit(self, X, y, X_val, y_val, deadline):
        import gpytorch

        dev = device()
        c = self.config
        rng = np.random.default_rng(self.ctx.seed)
        sub = _subsample(len(X), y, c["max_points"], rng)
        Xs = X.iloc[sub].reset_index(drop=True) if hasattr(X, "iloc") else X[sub]
        self._y = y[sub] if y is not None else None
        self.space = VectorSpace(self.ctx, pca=c["output_pca"] or None)
        E = self.space.fit(Xs)
        n, d = E.shape
        q = int(min(c["latent_dim"], d))
        # Latents start at the principal components, scaled to the N(0, I) prior
        P = PCA(q, random_state=self.ctx.seed).fit_transform(E)
        X_init = torch.as_tensor(P / (P.std(0) + 1e-6), dtype=torch.float32)
        torch.manual_seed(self.ctx.seed)
        self.gp = _model_class()(n, d, q, min(c["n_inducing"], n), X_init).to(dev)
        self.lik = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([d])).to(dev)
        mll = gpytorch.mlls.VariationalELBO(self.lik, self.gp, num_data=n)
        opt = torch.optim.Adam([{"params": self.gp.parameters()}, {"params": self.lik.parameters()}], lr=c["lr"])
        Y = torch.as_tensor(E, device=dev)
        self.gp.train()
        self.lik.train()
        losses, ep = [], 0
        for ep in range(1, c["epochs"] + 1):
            perm = torch.randperm(n, device=dev)
            tot = 0.0
            for sl in batches(n, c["batch_size"]):
                idx = perm[sl]
                z = self.gp.sample_latent_variable()[idx]  # reparameterized draw from q(z) (adds the KL term)
                loss = -mll(self.gp(z), Y[idx].T).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(idx)
                if deadline.expired():
                    break
            losses.append(tot / n)
            if deadline.expired():
                break
        self.gp.eval()
        self.lik.eval()
        self._E_train = Y.detach()
        self._fit_latent_prior()
        self.q = q
        self.history = {"neg_elbo": losses, "epochs_run": ep, "points_used": n, "stopped_by_budget": deadline.expired()}
        return self.history

    def _fit_latent_prior(self, shrink: float = 10.0):
        """Gaussian approximation of the aggregate posterior: covariance of the posterior means plus the
        mean posterior variance; per class, shrunk toward the pooled covariance with weight `shrink`."""
        lv = self.gp.X
        M = lv.q_mu.detach().cpu().numpy().astype(np.float64)
        # GPyTorch parameterizes the posterior standard deviation as softplus(q_log_sigma)
        V = np.diag((torch.nn.functional.softplus(lv.q_log_sigma) ** 2).mean(0).detach().cpu().numpy())
        self.z_mean, self.z_cov = M.mean(0), np.cov(M.T).reshape(M.shape[1], M.shape[1]) + V
        self.z_class = {}
        if self._y is not None:
            for cl in range(self.ctx.n_classes):
                Mc = M[self._y == cl]
                if len(Mc) == 0:
                    continue
                Sc = np.cov(Mc.T).reshape(M.shape[1], M.shape[1]) if len(Mc) > 1 else np.zeros_like(self.z_cov)
                w = len(Mc) / (len(Mc) + shrink)
                self.z_class[cl] = (Mc.mean(0), w * Sc + (1 - w) * (self.z_cov - V) + V)

    @torch.no_grad()
    def _predict(self, z: torch.Tensor):
        """Predictive mean and variance (including output noise) of every output at latents z; (m, d)."""
        f = self.gp(z)
        noise = self.lik.noise.reshape(-1, 1)
        return f.mean.T, (f.variance + noise).T

    @torch.no_grad()
    def sample(self, n, y=None):
        dev = device()
        rng = np.random.default_rng(self.ctx.seed + 31)
        y = self._labels_for(n, y)
        Z = np.empty((n, self.q))
        groups = [(np.arange(n), None)] if y is None else [(np.flatnonzero(y == c), int(c)) for c in np.unique(y)]
        for idx, c in groups:
            mean, cov = self.z_class.get(c, (self.z_mean, self.z_cov))
            Z[idx] = rng.multivariate_normal(mean, cov, size=len(idx))
        out = []
        for sl in batches(n, 1024):
            mu, var = self._predict(torch.as_tensor(Z[sl], dtype=torch.float32, device=dev))
            E = mu.cpu().numpy() + np.sqrt(var.cpu().numpy()) * rng.standard_normal(mu.shape)
            out.append(self.space.decode(E.astype(np.float32), y[sl] if y is not None else None, rng))
        return concat_samples(out, self.ctx.modality)

    def _optimize_latents(self, X):
        """MAP latent points of new data under the fixed GP; returns (z, negative log joint density)."""
        dev = device()
        E = torch.as_tensor(self.space.encode(X), device=dev)
        M = self.gp.X.q_mu.detach()
        zs, nll = [], []
        for sl in batches(len(E), 2048):
            e = E[sl]
            # Start from the latent of the nearest training point in output space
            z = M[torch.cdist(e, self._E_train).argmin(1)].clone().requires_grad_(True)
            opt = torch.optim.Adam([z], lr=0.05)
            for _ in range(self.config["encode_steps"]):
                mu, var = self._predict_grad(z)
                loss = self._neg_log_joint(e, z, mu, var).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                mu, var = self._predict(z)
                nll.append(self._neg_log_joint(e, z, mu, var).cpu().numpy())
            zs.append(z.detach().cpu().numpy())
        return np.concatenate(zs), np.concatenate(nll)

    def _predict_grad(self, z):
        f = self.gp(z)
        return f.mean.T, (f.variance + self.lik.noise.reshape(-1, 1)).T

    @staticmethod
    def _neg_log_joint(e, z, mu, var):
        # -log N(e; mu, var) - log N(z; 0, I), per point
        ll = -0.5 * (((e - mu) ** 2) / var + torch.log(2 * math.pi * var)).sum(1)
        lp = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(1)
        return -(ll + lp)

    def encode(self, X):
        return self._optimize_latents(X)[0].astype(np.float32)

    def anomaly_score(self, X):
        return self._optimize_latents(X)[1]
