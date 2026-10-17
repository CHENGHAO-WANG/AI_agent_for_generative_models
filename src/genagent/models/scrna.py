"""scRNA-seq generators on raw counts: scVI (scvi-tools), latent diffusion on scVI latents, WGAN-GP.

All return synthetic raw counts (N, G) for the dataset's highly variable genes."""

from __future__ import annotations

import copy
import datetime
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..embed import scrna_counts_from_lognorm, scrna_lognorm
from ..utils import batches, device
from .base import GenerativeModel, register


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

class _ResBlock(nn.Module):
    """Pre-norm residual MLP block; the time+label embedding modulates every block (FiLM)."""

    def __init__(self, width, cdim):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1, self.fc2 = nn.Linear(width, width), nn.Linear(width, width)
        self.cond = nn.Linear(cdim, 2 * width)

    def forward(self, h, c):
        scale, shift = self.cond(c).chunk(2, -1)
        x = self.norm(h) * (1 + scale) + shift
        return h + self.fc2(F.silu(self.fc1(F.silu(x))))


class _MLPDenoiser(nn.Module):
    """Noise predictor for low-dimensional latents. Conditioning only at the input (the earlier design)
    left samples at ~70% of the latents' spread, concentrated on the dominant population."""

    def __init__(self, dim, n_embed, width=256, depth=4, cdim=128):
        super().__init__()
        self.cdim = cdim
        self.inp = nn.Linear(dim, width)
        self.lab = nn.Embedding(n_embed, cdim)
        self.tmlp = nn.Sequential(nn.Linear(cdim, cdim), nn.SiLU(), nn.Linear(cdim, cdim))
        self.blocks = nn.ModuleList([_ResBlock(width, cdim) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, dim))

    def forward(self, x, t, lab):
        half = self.cdim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=x.device) / half)
        a = t.float()[:, None] * freqs[None]
        c = self.tmlp(torch.cat([a.sin(), a.cos()], 1)) + self.lab(lab)
        h = self.inp(x)
        for b in self.blocks:
            h = b(h, c)
        return self.out(h)


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

        self._y_train = y
        self._fit_scvi(X, deadline, frac=0.6)
        scvi_hist = self.history
        dev = device()
        Z = self.model.get_latent_representation(_adata(X), give_mean=True).astype(np.float32)
        self.z_mu, self.z_sd = Z.mean(0), Z.std(0) + 1e-6
        Zt = torch.from_numpy((Z - self.z_mu) / self.z_sd).to(dev)
        n_embed = self.ctx.n_classes + 1
        lab_all = torch.as_tensor(y if y is not None else np.full(len(Z), self.ctx.n_classes), device=dev)
        self.den = _MLPDenoiser(Z.shape[1], n_embed, self.config["diff_width"], self.config["diff_depth"]).to(dev)
        # Latents are standardized, not bounded: the scheduler must not clip predictions to [-1, 1]
        self.sched = DDPMScheduler(num_train_timesteps=1000, clip_sample=False)
        opt = torch.optim.AdamW(self.den.parameters(), lr=self.config["diff_lr"])
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.config["diff_steps"])
        # EMA of weights, used for sampling; warm-up keeps short runs from being dominated by the init
        ema = copy.deepcopy(self.den).eval()
        losses = []
        g = torch.Generator(device=dev).manual_seed(self.ctx.seed)
        for step in range(self.config["diff_steps"]):
            idx = torch.randint(0, len(Zt), (256,), device=dev, generator=g)
            x0, lab = Zt[idx], lab_all[idx]
            drop = torch.rand(len(idx), device=dev) < self.config["p_uncond"]
            lab = torch.where(drop, torch.full_like(lab, self.ctx.n_classes), lab)
            noise = torch.randn_like(x0)
            t = torch.randint(0, 1000, (len(idx),), device=dev)
            loss = F.mse_loss(self.den(self.sched.add_noise(x0, noise, t), t, lab), noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_sched.step()
            decay = min(self.config["ema_decay"], (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), self.den.parameters()):
                    pe.lerp_(p, 1 - decay)
            if step % 100 == 0:
                losses.append(loss.item())
            if deadline.expired():
                break
        self.den = ema
        self.history = {**scvi_hist, "diffusion_loss_every_100_steps": losses, "diffusion_steps_run": step + 1,
                        "stopped_by_budget": deadline.expired()}
        return self.history

    @torch.no_grad()
    def _latent_samples(self, n, y, rng):
        """Ancestral (stochastic) DDPM sampling; it matched the latents' spread better than DDIM."""
        from diffusers import DDPMScheduler

        dev = device()
        sched = DDPMScheduler.from_config(self.sched.config)
        sched.set_timesteps(self.config["sampling_steps"])
        gen = torch.Generator(device=dev).manual_seed(int(rng.integers(1e9)))
        x = torch.randn(n, len(self.z_mu), device=dev, generator=gen)
        null = torch.full((n,), self.ctx.n_classes, device=dev, dtype=torch.long)
        lab = torch.as_tensor(y, device=dev) if y is not None else null
        gs = self.config["guidance_scale"] if y is not None else 0.0
        self.den.eval()
        for t in sched.timesteps:
            tt = torch.full((n,), int(t), device=dev, dtype=torch.long)
            eps = self.den(x, tt, lab)
            if gs > 0:
                eps_u = self.den(x, tt, null)
                eps = eps_u + gs * (eps - eps_u)
            x = sched.step(eps, t, x, generator=gen).prev_sample
        z = x.cpu().numpy() * self.z_sd + self.z_mu
        return torch.as_tensor(z, dtype=torch.float32)


# ============================================================ WGAN-GP on log-normalized expression

class _MLP(nn.Module):
    def __init__(self, d_in, d_out, widths, act_out=None, layernorm=False):
        super().__init__()
        layers, d = [], d_in
        for w in widths:
            layers += [nn.Linear(d, w)] + ([nn.LayerNorm(w)] if layernorm else []) + [nn.LeakyReLU(0.2)]
            d = w
        layers.append(nn.Linear(d, d_out))
        if act_out is not None:
            layers.append(act_out)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@register
class SCRNAWGANGP(GenerativeModel):
    name = "wgan_gp_scrna"
    family = "gan"
    modalities = ("scrna",)
    library = "from scratch (PyTorch), scGAN-style"
    conditional = "one-hot label concatenated to generator and critic inputs"
    capabilities = ("sample",)
    summary = ("Wasserstein GAN with gradient penalty generating log-normalized expression (scGAN-style); "
               "counts are recovered by Poisson sampling at an empirical library size. Can produce sharp "
               "marginals but is data-hungry and unstable on small datasets; no likelihood or encoder.")

    @classmethod
    def default_config(cls, ctx):
        return {"epochs": 300 if ctx.n_train < 5000 else 100, "batch_size": 128, "lr": 1e-4, "z_dim": 64,
                "width": 512, "n_critic": 5, "gp_lambda": 10.0}

    @classmethod
    def search_space(cls, ctx):
        return {"lr": {"type": "float", "low": 3e-5, "high": 5e-4, "log": True},
                "width": {"type": "categorical", "choices": [256, 512, 1024]},
                "n_critic": {"type": "int", "low": 3, "high": 6}}

    def _onehot(self, y, n, dev):
        if not self.ctx.labeled:
            return torch.zeros(n, 0, device=dev)
        return F.one_hot(torch.as_tensor(y, device=dev), self.ctx.n_classes).float()

    def fit(self, X, y, X_val, y_val, deadline):
        dev = device()
        c = self.config
        L = torch.from_numpy(scrna_lognorm(X))
        k = self.ctx.n_classes
        G_dim = X.shape[1]
        self.G = _MLP(c["z_dim"] + k, G_dim, [c["width"] // 2, c["width"]], nn.ReLU()).to(dev)
        D = _MLP(G_dim + k, 1, [c["width"], c["width"] // 2], layernorm=True).to(dev)
        og = torch.optim.Adam(self.G.parameters(), lr=c["lr"], betas=(0.5, 0.9))
        od = torch.optim.Adam(D.parameters(), lr=c["lr"], betas=(0.5, 0.9))
        wd, ep = [], 0
        for ep in range(1, c["epochs"] + 1):
            perm = torch.randperm(len(L))
            tot, nb = 0.0, 0
            for i, sl in enumerate(batches(len(L), c["batch_size"])):
                idx = perm[sl]
                real = L[idx].to(dev)
                oh = self._onehot(y[idx.numpy()] if y is not None else None, len(idx), dev)
                fake = self.G(torch.cat([torch.randn(len(idx), c["z_dim"], device=dev), oh], 1))
                # Critic update with gradient penalty on interpolates
                eps = torch.rand(len(idx), 1, device=dev)
                inter = (eps * real + (1 - eps) * fake.detach()).requires_grad_(True)
                gr = torch.autograd.grad(D(torch.cat([inter, oh], 1)).sum(), inter, create_graph=True)[0]
                gp = ((gr.norm(dim=1) - 1) ** 2).mean()
                d_real, d_fake = D(torch.cat([real, oh], 1)).mean(), D(torch.cat([fake.detach(), oh], 1)).mean()
                ld = d_fake - d_real + c["gp_lambda"] * gp
                od.zero_grad()
                ld.backward()
                od.step()
                if i % c["n_critic"] == 0:
                    lg = -D(torch.cat([fake, oh], 1)).mean()
                    og.zero_grad()
                    lg.backward()
                    og.step()
                tot += (d_real - d_fake).item()
                nb += 1
            wd.append(tot / nb)
            if deadline.expired():
                break
        self.G.eval()
        self.history = {"wasserstein_estimate": wd, "epochs_run": ep, "stopped_by_budget": deadline.expired()}
        return self.history

    @torch.no_grad()
    def sample(self, n, y=None):
        dev = device()
        rng = np.random.default_rng(self.ctx.seed + 13)
        y = self._labels_for(n, y)
        out = []
        for sl in batches(n, 2048):
            m = sl.stop - sl.start
            ys = y[sl] if y is not None else None
            z = torch.randn(m, self.config["z_dim"], device=dev)
            Lg = self.G(torch.cat([z, self._onehot(ys, m, dev)], 1)).cpu().numpy()
            out.append(scrna_counts_from_lognorm(Lg, self._sample_libsizes(ys, m, rng), rng))
        return np.concatenate(out).astype(np.float32)
