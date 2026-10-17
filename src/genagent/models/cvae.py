"""Conditional VAE for every modality; the label embedding is fed to encoder and decoder.

Likelihood by modality:
- image:   Bernoulli per pixel on a conv encoder/decoder
- scrna:   negative binomial on raw counts (mean = decoded gene fractions x observed library size,
           learned per-gene inverse dispersion), MLP on log1p counts, as in scVI but label-conditioned
- tabular: Gaussian (learned per-column scale) on standardized numeric columns plus a categorical
           likelihood per one-hot block, MLP on the codec vectors
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import batches, device
from .base import GenerativeModel, label_tensor, n_embed, register
from .nets import ConvVAENet, VectorVAENet
from .spaces import VectorSpace, concat_samples, sample_libsizes


def _nb_log_prob(x, mu, theta):
    """Negative binomial log-likelihood with mean `mu` and inverse dispersion `theta`."""
    eps = 1e-8
    log_t_mu = torch.log(theta + mu + eps)
    return (torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1)
            + theta * (torch.log(theta + eps) - log_t_mu) + x * (torch.log(mu + eps) - log_t_mu))


@register
class CVAE(GenerativeModel):
    name = "cvae"
    family = "vae"
    modalities = ("image", "scrna", "tabular")
    library = "from scratch (PyTorch)"
    conditional = "label embedding fed to encoder and decoder"
    capabilities = ("sample", "encode", "anomaly")
    summary = ("Conditional VAE. Fast, stable, gives a latent space and a likelihood bound (anomaly score = "
               "negative ELBO). Images: conv net, Bernoulli pixels; samples blurrier than diffusion/GAN. scRNA: "
               "negative-binomial count likelihood like scVI, but truly label-conditioned (samples of a class come "
               "from the decoder given that label, not from replayed real cells). Tabular: Gaussian numeric + "
               "categorical likelihoods; label-conditioned, unlike TVAE.")

    @classmethod
    def default_config(cls, ctx):
        if ctx.modality == "image":
            return {"epochs": 15 if ctx.n_train > 20000 else 50, "batch_size": 256, "lr": 1e-3,
                    "latent_dim": 32, "channels": 64, "beta": 1.0}
        if ctx.modality == "scrna":
            return {"epochs": 400 if ctx.n_train < 5000 else 200, "batch_size": 128, "lr": 1e-3,
                    "latent_dim": 10, "width": 128, "depth": 2, "beta": 1.0, "kl_warmup": 0.25}
        return {"epochs": 300 if ctx.n_train < 20000 else 100, "batch_size": 256, "lr": 1e-3,
                "latent_dim": 16, "width": 256, "depth": 2, "beta": 1.0, "kl_warmup": 0.25}

    @classmethod
    def search_space(cls, ctx):
        sp = {"lr": {"type": "float", "low": 2e-4, "high": 3e-3, "log": True},
              "beta": {"type": "float", "low": 0.25, "high": 2.0, "log": True}}
        if ctx.modality == "image":
            sp["latent_dim"] = {"type": "categorical", "choices": [16, 32, 64]}
        else:
            sp["latent_dim"] = {"type": "categorical", "choices": [8, 16, 32]}
            sp["width"] = {"type": "categorical", "choices": [128, 256, 512]}
        return sp

    # -- data and likelihood ---------------------------------------------------
    def _inputs(self, X, fit=False) -> torch.Tensor:
        m = self.ctx.modality
        if m == "image":
            return torch.from_numpy(np.asarray(X, dtype=np.float32))
        if m == "scrna":
            return torch.from_numpy(np.asarray(X, dtype=np.float32))  # raw counts; log1p is taken in the net
        if fit:
            # The codec already standardizes numeric columns; one-hot blocks stay 0/1 as targets
            self.space = VectorSpace(self.ctx, standardize=False)
            return torch.from_numpy(self.space.fit(X))
        return torch.from_numpy(self.space.encode(X))

    def _build(self, d: int):
        c = self.config
        m = self.ctx.modality
        if m == "image":
            C, H, _ = self.ctx.shape
            self.net = ConvVAENet(C, H, n_embed(self.ctx), c["latent_dim"], c["channels"])
        elif m == "scrna":
            self.net = VectorVAENet(d, d, n_embed(self.ctx), c["latent_dim"], c["width"], c["depth"])
            self.net.log_theta = nn.Parameter(torch.zeros(d))
        else:
            self.blocks = self.space.cat_blocks
            self.nnum = self.space.num_dim
            self.net = VectorVAENet(d, d, n_embed(self.ctx), c["latent_dim"], c["width"], c["depth"])
            self.net.log_sigma = nn.Parameter(torch.zeros(self.nnum))
        self.net.to(device())

    def _enc_input(self, x):
        return torch.log1p(x) if self.ctx.modality == "scrna" else x

    def _rec_nll(self, x, out):
        """Per-sample negative log-likelihood of x given decoder outputs."""
        m = self.ctx.modality
        if m == "image":
            return F.binary_cross_entropy_with_logits(out, x, reduction="none").sum((1, 2, 3))
        if m == "scrna":
            mu = F.softmax(out, -1) * x.sum(1, keepdim=True)
            return -_nb_log_prob(x, mu, self.net.log_theta.exp()).sum(1)
        ls = self.net.log_sigma
        nll = (0.5 * ((x[:, :self.nnum] - out[:, :self.nnum]) / ls.exp()) ** 2 + ls).sum(1)
        for b in self.blocks:
            nll = nll + F.cross_entropy(out[:, b], x[:, b].argmax(1), reduction="none")
        return nll

    def _loss(self, x, lab):
        mu, lv = self.net.encode(self._enc_input(x), lab)
        z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        rec = self._rec_nll(x, self.net.decode(z, lab))
        kl = -0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)
        return rec, kl

    # -- lifecycle ---------------------------------------------------------------
    def fit(self, X, y, X_val, y_val, deadline):
        dev = device()
        c = self.config
        Xt = self._inputs(X, fit=True)
        self._build(int(np.prod(Xt.shape[1:])))
        opt = torch.optim.Adam(self.net.parameters(), lr=c["lr"])
        vector = self.ctx.modality != "image"
        if vector:
            Xv = self._inputs(X_val).to(dev)
            lab_v = label_tensor(self.ctx, y_val, len(Xv), dev)
        losses, val_losses, ep = [], [], 0
        best, best_state = float("inf"), None
        warm = max(1, int(c.get("kl_warmup", 0) * c["epochs"]))
        for ep in range(1, c["epochs"] + 1):
            self.net.train()
            # Linear KL warm-up for vector models (avoids early posterior collapse on small data)
            beta = c["beta"] * (min(1.0, ep / warm) if vector else 1.0)
            perm = torch.randperm(len(Xt))
            tot = 0.0
            for sl in batches(len(Xt), c["batch_size"]):
                idx = perm[sl]
                x = Xt[idx].to(dev)
                lab = label_tensor(self.ctx, y[idx.numpy()] if y is not None else None, len(idx), dev)
                # Occasionally use the null label so label-free encoding/scoring stays in-distribution
                drop = torch.rand(len(idx), device=dev) < 0.1
                lab = torch.where(drop, torch.full_like(lab, self.ctx.n_classes), lab)
                rec, kl = self._loss(x, lab)
                loss = (rec + beta * kl).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(idx)
                if deadline.expired():
                    break
            losses.append(tot / len(Xt))
            if vector:
                # Keep the best validation checkpoint (small tables and cell sets overfit quickly)
                self.net.eval()
                with torch.no_grad():
                    rec, kl = self._loss(Xv, lab_v)
                    v = (rec + kl).mean().item()
                val_losses.append(v)
                if v < best and ep >= warm:
                    best, best_state = v, copy.deepcopy(self.net.state_dict())
            if deadline.expired():
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        self.history = {"train_loss": losses, "epochs_run": ep, "stopped_by_budget": deadline.expired()}
        if vector:
            self.history |= {"val_neg_elbo": val_losses, "best_val_neg_elbo": best}
        return self.history

    @torch.no_grad()
    def _decode_sample(self, out, ys, rng):
        """Draw native samples from decoder outputs."""
        m = self.ctx.modality
        if m == "image":
            return torch.sigmoid(out).cpu().numpy()
        if m == "scrna":
            lib = sample_libsizes(self.ctx, ys, len(out), rng)
            mu = (F.softmax(out, -1).cpu().numpy() * lib[:, None]).astype(np.float64)
            theta = self.net.log_theta.exp().cpu().numpy().astype(np.float64)
            # Negative binomial as a Gamma-Poisson mixture
            rate = rng.gamma(theta[None, :], mu / theta[None, :])
            return rng.poisson(rate).astype(np.float32)
        E = out.cpu().numpy().copy()
        sig = self.net.log_sigma.exp().cpu().numpy()
        E[:, :self.nnum] += rng.standard_normal((len(E), self.nnum)) * sig
        for b in self.blocks:
            p = F.softmax(out[:, b], -1).cpu().numpy()
            k = (p.cumsum(1) > rng.random((len(p), 1))).argmax(1)
            E[:, b] = np.eye(p.shape[1])[k]
        return self.space.decode(E, ys, rng)

    @torch.no_grad()
    def sample(self, n, y=None):
        dev = device()
        rng = np.random.default_rng(self.ctx.seed + 23)
        y = self._labels_for(n, y)
        out = []
        for sl in batches(n, 2048):
            m = sl.stop - sl.start
            ys = y[sl] if y is not None else None
            lab = label_tensor(self.ctx, ys, m, dev)
            z = torch.randn(m, self.config["latent_dim"], device=dev)
            out.append(self._decode_sample(self.net.decode(z, lab), ys, rng))
        return concat_samples(out, self.ctx.modality)

    @torch.no_grad()
    def encode(self, X):
        dev = device()
        Xt = self._inputs(X)
        out = []
        for sl in batches(len(Xt), 2048):
            x = Xt[sl].to(dev)
            mu, _ = self.net.encode(self._enc_input(x), label_tensor(self.ctx, None, len(x), dev))
            out.append(mu.cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def anomaly_score(self, X):
        dev = device()
        Xt = self._inputs(X)
        out = []
        for sl in batches(len(Xt), 2048):
            x = Xt[sl].to(dev)
            rec, kl = self._loss(x, label_tensor(self.ctx, None, len(x), dev))
            out.append((rec + kl).cpu().numpy())
        return np.concatenate(out)
