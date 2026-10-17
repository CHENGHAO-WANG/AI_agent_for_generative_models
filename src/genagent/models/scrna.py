"""scRNA-seq generators on raw counts built on scVI (scvi-tools): scVI itself and latent diffusion on scVI latents.

All return synthetic raw counts (N, G) for the dataset's highly variable genes."""

from __future__ import annotations

import datetime
import math
from pathlib import Path

import numpy as np
import torch

from ..utils import batches, device
from .base import GenerativeModel, label_tensor, n_embed, register
from .nets import MLPDenoiser


def _adata(X: np.ndarray):
    import anndata as ad

    a = ad.AnnData(np.asarray(X, dtype=np.float32))
    a.obs_names = [f"c{i}" for i in range(len(X))]
    return a


# ============================================================ scVI

@register
class SCVIModel(GenerativeModel):
    name = "scvi"
    family = "vae"
    modalities = ("scrna",)
    library = "scvi-tools (SCVI, negative-binomial likelihood)"
    conditional = ("none in the model: replays posteriors of real cells of the class (encodes real cells of "
                   "the requested class and decodes them with fresh noise); weak for very rare classes")
    capabilities = ("sample", "encode", "anomaly")
    summary = ("scVI: VAE with a negative-binomial count likelihood and library-size handling; the standard "
               "model for scRNA counts. Gives a low-dimensional latent space (representation learning) and "
               "per-cell reconstruction likelihood (anomaly score). Unconditional samples come from the N(0, I) "
               "prior, which over-represents the dominant cell population; class-specific samples replay real "
               "cells of that class, so they cannot add states the class's few real cells lack.")

    @classmethod
    def default_config(cls, ctx):
        return {"epochs": 400 if ctx.n_train < 5000 else 200, "batch_size": 128, "lr": 1e-3,
                "n_latent": 10, "n_hidden": 128, "n_layers": 1, "dropout_rate": 0.1, "gene_likelihood": "nb"}

    @classmethod
    def search_space(cls, ctx):
        return {"lr": {"type": "float", "low": 3e-4, "high": 3e-3, "log": True},
                "n_latent": {"type": "categorical", "choices": [10, 20, 30]},
                "n_hidden": {"type": "categorical", "choices": [128, 256]},
                "n_layers": {"type": "int", "low": 1, "high": 2}}

    def _fit_scvi(self, X, deadline, frac: float = 1.0):
        import scvi

        scvi.settings.seed = self.ctx.seed
        a = _adata(X)
        scvi.model.SCVI.setup_anndata(a)
        c = self.config
        self.model = scvi.model.SCVI(a, n_latent=c["n_latent"], n_hidden=c["n_hidden"], n_layers=c["n_layers"],
                                     dropout_rate=c["dropout_rate"], gene_likelihood=c["gene_likelihood"])
        secs = max(10.0, deadline.remaining() * frac) if math.isfinite(deadline.remaining()) else None
        kw = {"max_time": datetime.timedelta(seconds=secs)} if secs else {}
        self.model.train(max_epochs=c["epochs"], batch_size=c["batch_size"], plan_kwargs={"lr": c["lr"]},
                         enable_progress_bar=False, train_size=0.9, early_stopping=True,
                         early_stopping_patience=30, **kw)
        h = self.model.history
        self.history = {"elbo_train": h["elbo_train"].values.ravel().tolist(),
                        "elbo_validation": h["elbo_validation"].values.ravel().tolist() if "elbo_validation" in h else [],
                        "epochs_run": len(h["elbo_train"]), "stopped_by_budget": deadline.expired()}
        self._X_train = X

    def fit(self, X, y, X_val, y_val, deadline):
        self._y_train = y
        self._fit_scvi(X, deadline)
        return self.history

    @torch.no_grad()
    def _decode(self, z: torch.Tensor, lib: np.ndarray, sample: bool = True):
        mod = self.model.module
        dev = mod.device
        library = torch.log(torch.as_tensor(lib, dtype=torch.float32, device=dev)).unsqueeze(1)
        batch = torch.zeros(len(z), 1, device=dev)
        px = mod.generative(z=z.to(dev), library=library, batch_index=batch)["px"]
        return px.sample() if sample else px

    def _latent_samples(self, n, y, rng):
        """Prior samples; for labeled data, posterior samples of random cells from the requested class."""
        if y is None or self._y_train is None:
            return torch.randn(n, self.config["n_latent"], generator=torch.Generator().manual_seed(int(rng.integers(1e9))))
        idx = np.array([rng.choice(np.flatnonzero(self._y_train == c)) for c in y])
        z = self.model.get_latent_representation(_adata(self._X_train[idx]), give_mean=False)
        return torch.as_tensor(z, dtype=torch.float32)

    def sample(self, n, y=None):
        rng = np.random.default_rng(self.ctx.seed + 11)
        y = self._labels_for(n, y)
        out = []
        for sl in batches(n, 1024):
            ys = y[sl] if y is not None else None
            z = self._latent_samples(sl.stop - sl.start, ys, rng)
            lib = self._sample_libsizes(ys, len(z), rng)
            out.append(self._decode(z, lib).cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    def encode(self, X):
        return self.model.get_latent_representation(_adata(X), give_mean=True)

    @torch.no_grad()
    def anomaly_score(self, X):
        z = torch.as_tensor(self.encode(X), dtype=torch.float32)
        out = []
        for sl in batches(len(X), 1024):
            px = self._decode(z[sl], X[sl].sum(1), sample=False)
            x = torch.as_tensor(X[sl], device=self.model.module.device)
            out.append((-px.log_prob(x).sum(-1)).cpu().numpy())
        return np.concatenate(out)

    def save(self, path):
        path = Path(path)
        self.model.save(str(path / "scvi"), overwrite=True, save_anndata=True)
        m = self.model
        self.model = None
        torch.save(self, path / "scvi_wrapper.pt")
        self.model = m


def load_scvi_based(path: Path):
    import scvi

    w = torch.load(Path(path) / "scvi_wrapper.pt", weights_only=False)
    w.model = scvi.model.SCVI.load(str(Path(path) / "scvi"))
    return w


# ============================================================ latent diffusion on scVI latents

@register
class SCVILatentDiffusion(SCVIModel):
    name = "latent_diffusion_scvi"
    family = "diffusion"
    modalities = ("scrna",)
    library = "scvi-tools encoder/decoder + diffusers DDPM scheduler + residual MLP denoiser (from scratch)"
    conditional = "class embedding with classifier-free guidance in latent space"
    capabilities = ("sample", "encode", "anomaly")
    summary = ("Two-stage model: scVI autoencoder, then a DDPM learns the distribution of scVI latents. "
               "Replaces scVI's N(0, I) prior with a learned one, which usually matches cell-type structure "
               "better than plain scVI sampling; decoding keeps the negative-binomial count model.")

    @classmethod
    def default_config(cls, ctx):
        return {**SCVIModel.default_config(ctx), "diff_steps": 8000, "diff_lr": 1e-3, "diff_width": 256,
                "diff_depth": 4, "sampling_steps": 1000, "guidance_scale": 1.0, "p_uncond": 0.1,
                "ema_decay": 0.999}

    @classmethod
    def search_space(cls, ctx):
        return {"n_latent": {"type": "categorical", "choices": [10, 20]},
                "diff_lr": {"type": "float", "low": 2e-4, "high": 3e-3, "log": True},
                "diff_width": {"type": "categorical", "choices": [256, 512]}}

    def fit(self, X, y, X_val, y_val, deadline):
        from diffusers import DDPMScheduler

        from .ddpm import random_batches, train_denoiser

        self._y_train = y
        self._fit_scvi(X, deadline, frac=0.6)
        scvi_hist = self.history
        dev = device()
        c = self.config
        Z = self.model.get_latent_representation(_adata(X), give_mean=True).astype(np.float32)
        self.z_mu, self.z_sd = Z.mean(0), Z.std(0) + 1e-6
        Zt = torch.from_numpy((Z - self.z_mu) / self.z_sd).to(dev)
        lab_all = label_tensor(self.ctx, y, len(Z), dev)
        self.den = MLPDenoiser(Z.shape[1], n_embed(self.ctx), c["diff_width"], c["diff_depth"]).to(dev)
        # Latents are standardized, not bounded: the scheduler must not clip predictions to [-1, 1]
        self.sched = DDPMScheduler(num_train_timesteps=1000, clip_sample=False)
        g = torch.Generator(device=dev).manual_seed(self.ctx.seed)
        it = random_batches(Zt, lab_all, 256, c["diff_steps"], g)
        self.den, losses, steps = train_denoiser(self.den, self.sched, it, self.ctx.n_classes, c["diff_lr"],
                                                 c["p_uncond"], c["ema_decay"], deadline, total_steps=c["diff_steps"])
        self.history = {**scvi_hist, "diffusion_loss_every_100_steps": losses, "diffusion_steps_run": steps,
                        "stopped_by_budget": deadline.expired()}
        return self.history

    @torch.no_grad()
    def _latent_samples(self, n, y, rng):
        """Ancestral (stochastic) DDPM sampling; it matched the latents' spread better than DDIM."""
        from .ddpm import sample_denoiser

        dev = device()
        gen = torch.Generator(device=dev).manual_seed(int(rng.integers(1e9)))
        lab = label_tensor(self.ctx, y, n, dev)
        gs = self.config["guidance_scale"] if y is not None else 0.0
        x = sample_denoiser(self.den, self.sched, (n, len(self.z_mu)), lab, self.ctx.n_classes, gs,
                            self.config["sampling_steps"], "ddpm", gen)
        z = x.cpu().numpy() * self.z_sd + self.z_mu
        return torch.as_tensor(z, dtype=torch.float32)
