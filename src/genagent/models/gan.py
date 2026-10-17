"""Conditional GAN for every modality with a choice of adversarial loss.

Losses (`loss` config):
- vanilla: the original GAN objective with the non-saturating generator loss
- hinge:   hinge loss with a spectrally normalized discriminator
- wgan_gp: Wasserstein critic with gradient penalty (no spectral normalization)

Networks by modality:
- image:   DCGAN-style conv generator, projection discriminator; pixels scaled to [-1, 1]
- scrna:   MLPs on log-normalized expression (ReLU output), one-hot label concatenated (scGAN-style)
- tabular: MLPs on the codec vectors; Gumbel-softmax on each one-hot block so generated categories
           look like real one-hots to the discriminator
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..embed import scrna_counts_from_lognorm, scrna_lognorm
from ..utils import batches, device
from .base import GenerativeModel, label_tensor, n_embed, register
from .nets import MLP, ConvDiscriminator, ConvGenerator
from .spaces import VectorSpace, concat_samples

LOSSES = ("vanilla", "hinge", "wgan_gp")


def d_loss(kind: str, d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    if kind == "vanilla":
        return F.softplus(-d_real).mean() + F.softplus(d_fake).mean()
    if kind == "hinge":
        return F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean()
    return d_fake.mean() - d_real.mean()


def g_loss(kind: str, d_fake: torch.Tensor) -> torch.Tensor:
    # Non-saturating loss for the vanilla objective; -D(fake) for hinge and Wasserstein
    return F.softplus(-d_fake).mean() if kind == "vanilla" else -d_fake.mean()


class _VecGen(nn.Module):
    """MLP generator; output activation depends on the modality."""

    def __init__(self, zdim, k, d, width, blocks: list[slice], nnum: int, relu_out: bool):
        super().__init__()
        self.k, self.blocks, self.nnum = k, blocks, nnum
        self.net = MLP(zdim + k, d, [width // 2, width], nn.ReLU() if relu_out else None)
        self.tau = 0.2

    def forward(self, z, oh):
        out = self.net(torch.cat([z, oh], 1))
        if not self.blocks:
            return out
        parts = [out[:, :self.nnum]] + [F.gumbel_softmax(out[:, b], tau=self.tau) for b in self.blocks]
        return torch.cat(parts, 1)


class _VecDisc(nn.Module):
    def __init__(self, d, k, width, spectral: bool):
        super().__init__()
        # Layer norm for the WGAN-GP critic (batch-independent, compatible with the gradient penalty)
        self.net = MLP(d + k, 1, [width, width // 2], layernorm=not spectral, spectral=spectral)

    def forward(self, x, oh):
        return self.net(torch.cat([x, oh], 1)).squeeze(1)


@register
class GAN(GenerativeModel):
    name = "gan"
    family = "gan"
    modalities = ("image", "scrna", "tabular")
    library = "from scratch (PyTorch)"
    conditional = ("images: label embedding in generator, projection discriminator; scRNA/tabular: one-hot label "
                   "concatenated to generator and discriminator inputs")
    capabilities = ("sample",)
    summary = ("Conditional GAN with loss vanilla | hinge | wgan_gp. Sharp samples and very fast sampling; risk of "
               "mode collapse; no likelihood or encoder. Images: DCGAN (default hinge). scRNA: scGAN-style MLP on "
               "log-normalized expression (default wgan_gp), counts by Poisson sampling; data-hungry. Tabular: MLP "
               "with Gumbel-softmax categories (default wgan_gp); CTGAN handles imbalanced categories better.")

    @classmethod
    def default_config(cls, ctx):
        if ctx.modality == "image":
            return {"loss": "hinge", "epochs": 15 if ctx.n_train > 20000 else 60, "batch_size": 128, "lr_g": 2e-4,
                    "lr_d": 2e-4, "z_dim": 100, "channels": 64, "n_critic": 1, "gp_lambda": 10.0}
        many = ctx.n_train >= (5000 if ctx.modality == "scrna" else 20000)
        return {"loss": "wgan_gp", "epochs": 100 if many else 300, "batch_size": 128, "lr": 1e-4, "z_dim": 64,
                "width": 512 if ctx.modality == "scrna" else 256, "n_critic": 5, "gp_lambda": 10.0}

    @classmethod
    def search_space(cls, ctx):
        if ctx.modality == "image":
            return {"lr_g": {"type": "float", "low": 5e-5, "high": 5e-4, "log": True},
                    "lr_d": {"type": "float", "low": 5e-5, "high": 5e-4, "log": True},
                    "channels": {"type": "categorical", "choices": [32, 64]}}
        return {"lr": {"type": "float", "low": 3e-5, "high": 5e-4, "log": True},
                "width": {"type": "categorical", "choices": [256, 512, 1024]},
                "n_critic": {"type": "int", "low": 3, "high": 6}}

    @property
    def _image(self) -> bool:
        return self.ctx.modality == "image"

    def _cond(self, y, n, dev):
        """Label input: class indices for the conv nets, one-hot (possibly empty) for the MLPs."""
        if self._image:
            return label_tensor(self.ctx, y, n, dev)
        if not self.ctx.labeled:
            return torch.zeros(n, 0, device=dev)
        return F.one_hot(torch.as_tensor(y, device=dev), self.ctx.n_classes).float()

    def _data(self, X) -> torch.Tensor:
        m = self.ctx.modality
        if m == "image":
            return torch.from_numpy(np.asarray(X, dtype=np.float32)) * 2 - 1
        if m == "scrna":
            return torch.from_numpy(scrna_lognorm(X))
        # Codec vectors: numeric columns already standardized, one-hot blocks exactly 0/1
        self.space = VectorSpace(self.ctx, standardize=False)
        return torch.from_numpy(self.space.fit(X))

    def _build(self, d: int):
        c = self.config
        if c["loss"] not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}")
        dev = device()
        spectral = c["loss"] != "wgan_gp"
        if self._image:
            C, H, _ = self.ctx.shape
            ne = n_embed(self.ctx)
            self.G = ConvGenerator(C, H, ne, c["z_dim"], c["channels"]).to(dev)
            D = ConvDiscriminator(C, H, ne, c["channels"], spectral=spectral).to(dev)
            lr_g, lr_d, betas = c["lr_g"], c["lr_d"], (0.0, 0.9)
        else:
            k = self.ctx.n_classes
            tab = self.ctx.modality == "tabular"
            self.G = _VecGen(c["z_dim"], k, d, c["width"], self.space.cat_blocks if tab else [],
                             self.space.num_dim if tab else d, relu_out=not tab).to(dev)
            D = _VecDisc(d, k, c["width"], spectral).to(dev)
            lr_g = lr_d = c["lr"]
            betas = (0.5, 0.9)
        og = torch.optim.Adam(self.G.parameters(), lr=lr_g, betas=betas)
        od = torch.optim.Adam(D.parameters(), lr=lr_d, betas=betas)
        return D, og, od

    def _gradient_penalty(self, D, real, fake, cond):
        eps = torch.rand(len(real), *([1] * (real.dim() - 1)), device=real.device)
        inter = (eps * real + (1 - eps) * fake.detach()).requires_grad_(True)
        gr = torch.autograd.grad(D(inter, cond).sum(), inter, create_graph=True)[0]
        return ((gr.flatten(1).norm(dim=1) - 1) ** 2).mean()

    def fit(self, X, y, X_val, y_val, deadline):
        dev = device()
        c = self.config
        Xt = self._data(X)
        D, og, od = self._build(Xt.shape[1] if not self._image else 0)
        kind = c["loss"]
        hist_d, hist_g, ep = [], [], 0
        for ep in range(1, c["epochs"] + 1):
            self.G.train()
            perm = torch.randperm(len(Xt))
            sd = sg = 0.0
            nd = ng = 0
            for i, sl in enumerate(batches(len(Xt), c["batch_size"])):
                idx = perm[sl]
                real = Xt[idx].to(dev)
                cond = self._cond(y[idx.numpy()] if y is not None else None, len(idx), dev)
                fake = self.G(torch.randn(len(idx), c["z_dim"], device=dev), cond)
                ld = d_loss(kind, D(real, cond), D(fake.detach(), cond))
                if kind == "wgan_gp":
                    ld = ld + c["gp_lambda"] * self._gradient_penalty(D, real, fake, cond)
                od.zero_grad()
                ld.backward()
                od.step()
                sd += ld.item()
                nd += 1
                # The generator steps once per n_critic discriminator steps
                if i % c["n_critic"] == 0:
                    lg = g_loss(kind, D(fake, cond))
                    og.zero_grad()
                    lg.backward()
                    og.step()
                    sg += lg.item()
                    ng += 1
                if deadline.expired():
                    break
            hist_d.append(sd / max(nd, 1))
            hist_g.append(sg / max(ng, 1))
            if deadline.expired():
                break
        self.G.eval()
        self.history = {"loss": kind, "d_loss": hist_d, "g_loss": hist_g, "epochs_run": ep,
                        "stopped_by_budget": deadline.expired()}
        return self.history

    @torch.no_grad()
    def sample(self, n, y=None):
        dev = device()
        rng = np.random.default_rng(self.ctx.seed + 13)
        y = self._labels_for(n, y)
        z_dim = self.config["z_dim"]
        out = []
        for sl in batches(n, 2048):
            m = sl.stop - sl.start
            ys = y[sl] if y is not None else None
            x = self.G(torch.randn(m, z_dim, device=dev), self._cond(ys, m, dev)).cpu().numpy()
            if self._image:
                out.append((x + 1) / 2)
            elif self.ctx.modality == "scrna":
                out.append(scrna_counts_from_lognorm(x, self._sample_libsizes(ys, m, rng), rng))
            else:
                out.append(self.space.decode(x, ys, rng))
        return concat_samples(out, self.ctx.modality)
