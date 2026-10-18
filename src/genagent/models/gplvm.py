"""Bayesian GPLVM (Titsias & Lawrence, 2010) with amortized inference, using GPyTorch.

Each output dimension is an independent sparse variational GP f_j(z) of a low-dimensional latent z
(shared ARD-RBF kernel, inducing points per output), with Gaussian noise per output. The latent
posterior is amortized: an MLP encoder gives q(z | x) = N(m(x), diag(s(x)^2)) (a back-constrained
GPLVM), trained jointly with the GP by maximizing the minibatch ELBO

    sum_j E_q[log p(x_j | f_j(z))] - KL(q(f) || p(f)) / n - KL(q(z | x) || N(0, I)).

Vectors: PCA-reduced, standardized pixels (images) or log-normalized expression (scRNA);
standardized codec vectors (tabular).

Labels are not used in training. To generate, latents are drawn from a Gaussian fitted to the
encoded training data (per class when labeled, shrunk toward the pooled covariance) and passed
through the GP predictive, with output noise. `encode` returns m(x); the anomaly score is a
Monte Carlo estimate of the negative ELBO of a point."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from ..utils import batches, device
from .base import GenerativeModel, register
from .nets import MLP
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
    """The GPyTorch GP class, built on first use so that importing the registry does not import
    gpytorch. Exposed as the module attribute `_SVGP` so saved models can be unpickled."""
    global _MODEL_CLASS
    if _MODEL_CLASS is not None:
        return _MODEL_CLASS
    import gpytorch
    from gpytorch.models import ApproximateGP
    from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

    class _SVGP(ApproximateGP):
        """d independent sparse variational GPs over a q-dimensional input, one set of inducing
        points per output, shared kernel hyperparameters."""

        def __init__(self, Z: torch.Tensor, d: int):
            batch = torch.Size([d])
            q_u = CholeskyVariationalDistribution(len(Z), batch_shape=batch)
            strategy = VariationalStrategy(self, Z.unsqueeze(0).repeat(d, 1, 1), q_u, learn_inducing_locations=True)
            super().__init__(strategy)
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=Z.shape[1]))

        def forward(self, z):
            return gpytorch.distributions.MultivariateNormal(self.mean_module(z), self.covar_module(z))

    _SVGP.__module__, _SVGP.__qualname__ = __name__, "_SVGP"
    _MODEL_CLASS = _SVGP
    return _SVGP


def __getattr__(name):
    if name == "_SVGP":
        return _model_class()
    raise AttributeError(name)


@register
class GPLVM(GenerativeModel):
    name = "gplvm"
    family = "latent_variable"
    modalities = ("image", "scrna", "tabular")
    library = "GPyTorch (sparse variational GPs) + MLP encoder (amortized Bayesian GPLVM)"
    conditional = ("none in the model: per-class Gaussian over the encoded latent positions of that class "
                   "(shrunk toward the pooled covariance), decoded through the GP")
    capabilities = ("sample", "encode", "anomaly")
    summary = ("Bayesian Gaussian-process latent variable model with an amortized encoder: a smooth non-linear map "
               "from a low-dimensional latent space with calibrated uncertainty; well suited to small data. scRNA "
               "and images are modeled on their top principal components (variation outside them is lost). Class "
               "samples come from latent regions of that class, not a label-conditioned model. Anomaly score = "
               "negative ELBO.")

    @classmethod
    def default_config(cls, ctx):
        return {"epochs": 300 if ctx.n_train < 5000 else 60, "batch_size": 256, "lr": 0.01, "latent_dim": 5,
                "n_inducing": 64, "enc_width": 256, "max_points": 20000,
                "output_pca": 0 if ctx.modality == "tabular" else 50}

    @classmethod
    def search_space(cls, ctx):
        return {"latent_dim": {"type": "categorical", "choices": [2, 5, 10]},
                "n_inducing": {"type": "categorical", "choices": [32, 64, 128]},
                "lr": {"type": "float", "low": 3e-3, "high": 3e-2, "log": True}}

    def _posterior(self, Y: torch.Tensor):
        """Encoder output: mean and standard deviation of q(z | x)."""
        mu, s = self.enc(Y).chunk(2, -1)
        return mu, F.softplus(s) + 1e-4

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
        self.q = q
        torch.manual_seed(self.ctx.seed)
        Y = torch.as_tensor(E, device=dev)
        # Principal components, scaled to the N(0, I) prior, place the inducing points and warm-start the encoder
        P = PCA(q, random_state=self.ctx.seed).fit_transform(E)
        P = torch.as_tensor(P / (P.std(0) + 1e-6), dtype=torch.float32, device=dev)
        self.enc = MLP(d, 2 * q, [c["enc_width"], c["enc_width"]], layernorm=True).to(dev)
        self._warm_start(Y, P)
        Z = P[torch.randperm(n, device=dev)[:min(c["n_inducing"], n)]].clone()
        self.gp = _model_class()(Z.cpu(), d).to(dev)
        self.lik = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([d])).to(dev)
        mll = gpytorch.mlls.VariationalELBO(self.lik, self.gp, num_data=n)
        params = list(self.gp.parameters()) + list(self.lik.parameters()) + list(self.enc.parameters())
        opt = torch.optim.Adam(params, lr=c["lr"])
        self.gp.train()
        self.lik.train()
        self.enc.train()
        losses, ep = [], 0
        for ep in range(1, c["epochs"] + 1):
            perm = torch.randperm(n, device=dev)
            tot = 0.0
            for sl in batches(n, c["batch_size"]):
                Yb = Y[perm[sl]]
                mu, s = self._posterior(Yb)
                z = mu + s * torch.randn_like(s)
                # Per-datapoint ELBO: the mll sums the d outputs' expected log-likelihoods (each minus its
                # inducing KL / n); the latent KL enters once per point
                kl_z = (0.5 * (mu ** 2 + s ** 2 - 1) - torch.log(s)).sum(1).mean()
                loss = -mll(self.gp(z), Yb.T).sum() + kl_z
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(Yb)
                if deadline.expired():
                    break
            losses.append(tot / n)
            if deadline.expired():
                break
        self.gp.eval()
        self.lik.eval()
        self.enc.eval()
        self._fit_latent_prior(Y)
        self.history = {"neg_elbo_per_point": losses, "epochs_run": ep, "points_used": n,
                        "stopped_by_budget": deadline.expired()}
        return self.history

    def _warm_start(self, Y: torch.Tensor, P: torch.Tensor, steps: int = 300):
        """Regress the encoder mean onto the principal components so training starts from a spread-out,
        PCA-like latent layout (the GP inducing points are placed there too)."""
        opt = torch.optim.Adam(self.enc.parameters(), lr=1e-3)
        g = torch.Generator(device=Y.device).manual_seed(self.ctx.seed)
        for _ in range(steps):
            idx = torch.randint(0, len(Y), (min(256, len(Y)),), device=Y.device, generator=g)
            mu, _ = self._posterior(Y[idx])
            loss = F.mse_loss(mu, P[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    @torch.no_grad()
    def _fit_latent_prior(self, Y: torch.Tensor, shrink: float = 10.0):
        """Gaussian approximation of the aggregate posterior: covariance of the encoded means plus the
        mean posterior variance; per class, shrunk toward the pooled covariance with weight `shrink`."""
        parts = [self._posterior(Y[sl]) for sl in batches(len(Y), 4096)]
        M = torch.cat([p[0] for p in parts]).cpu().numpy().astype(np.float64)
        V = np.diag((torch.cat([p[1] for p in parts]) ** 2).mean(0).cpu().numpy())
        q = M.shape[1]
        cov_m = np.cov(M.T).reshape(q, q)
        self.z_mean, self.z_cov = M.mean(0), cov_m + V
        self.z_class = {}
        if self._y is not None:
            for cl in range(self.ctx.n_classes):
                Mc = M[self._y == cl]
                if len(Mc) == 0:
                    continue
                Sc = np.cov(Mc.T).reshape(q, q) if len(Mc) > 1 else np.zeros((q, q))
                w = len(Mc) / (len(Mc) + shrink)
                self.z_class[cl] = (Mc.mean(0), w * Sc + (1 - w) * cov_m + V)

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
            f = self.gp(torch.as_tensor(Z[sl], dtype=torch.float32, device=dev))
            # Predictive marginal per point and output: GP mean, GP variance plus output noise
            var = f.variance + self.lik.noise.reshape(-1, 1)
            E = (f.mean + var.sqrt() * torch.randn_like(var)).T.cpu().numpy()
            out.append(self.space.decode(E, y[sl] if y is not None else None, rng))
        return concat_samples(out, self.ctx.modality)

    @torch.no_grad()
    def encode(self, X):
        Y = torch.as_tensor(self.space.encode(X), device=device())
        return np.concatenate([self._posterior(Y[sl])[0].cpu().numpy() for sl in batches(len(Y), 4096)])

    @torch.no_grad()
    def anomaly_score(self, X, n_samples: int = 8):
        """Negative ELBO of each point: -E_q[log p(x | z)] + KL(q(z | x) || N(0, I)), with the
        expectation over z estimated from `n_samples` draws (the GP part is exact given z)."""
        Y = torch.as_tensor(self.space.encode(X), device=device())
        out = []
        for sl in batches(len(Y), 1024):
            Yb = Y[sl]
            mu, s = self._posterior(Yb)
            ell = 0
            for _ in range(n_samples):
                z = mu + s * torch.randn_like(s)
                ell = ell + self.lik.expected_log_prob(Yb.T, self.gp(z)).sum(0)
            kl = (0.5 * (mu ** 2 + s ** 2 - 1) - torch.log(s)).sum(1)
            out.append((-ell / n_samples + kl).cpu().numpy())
        return np.concatenate(out)
