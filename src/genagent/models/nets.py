"""Network backbones shared by the generative models: conv nets for images, MLPs for vectors."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sn(module: nn.Module, on: bool) -> nn.Module:
    return nn.utils.spectral_norm(module) if on else module


# ============================================================ vector backbones

class MLP(nn.Module):
    def __init__(self, d_in, d_out, widths, act_out=None, layernorm=False, spectral=False):
        super().__init__()
        layers, d = [], d_in
        for w in widths:
            layers += [_sn(nn.Linear(d, w), spectral)] + ([nn.LayerNorm(w)] if layernorm else []) + [nn.LeakyReLU(0.2)]
            d = w
        layers.append(_sn(nn.Linear(d, d_out), spectral))
        if act_out is not None:
            layers.append(act_out)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
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


class MLPDenoiser(nn.Module):
    """Noise predictor for vectors. Conditioning only at the input (an earlier design) left samples
    at ~70% of the scVI latents' spread, concentrated on the dominant population."""

    def __init__(self, dim, n_embed, width=256, depth=4, cdim=128):
        super().__init__()
        self.cdim = cdim
        self.inp = nn.Linear(dim, width)
        self.lab = nn.Embedding(n_embed, cdim)
        self.tmlp = nn.Sequential(nn.Linear(cdim, cdim), nn.SiLU(), nn.Linear(cdim, cdim))
        self.blocks = nn.ModuleList([ResBlock(width, cdim) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, dim))

    def forward(self, x, t, lab):
        # Sinusoidal timestep features, added to the label embedding
        half = self.cdim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=x.device) / half)
        a = t.float()[:, None] * freqs[None]
        c = self.tmlp(torch.cat([a.sin(), a.cos()], 1)) + self.lab(lab)
        h = self.inp(x)
        for b in self.blocks:
            h = b(h, c)
        return self.out(h)


class VectorVAENet(nn.Module):
    """MLP encoder/decoder with a label embedding fed to both. The decoder returns `d_out` raw
    outputs; the model interprets them according to its likelihood."""

    def __init__(self, d_in, d_out, n_embed, latent, width, depth=2):
        super().__init__()
        widths = [width] * depth
        self.emb = nn.Embedding(n_embed, 16)
        self.enc = MLP(d_in + 16, 2 * latent, widths, layernorm=True)
        self.dec = MLP(latent + 16, d_out, widths[::-1], layernorm=True)

    def encode(self, x, lab):
        mu, lv = self.enc(torch.cat([x, self.emb(lab)], 1)).chunk(2, -1)
        return mu, lv.clamp(-10, 10)

    def decode(self, z, lab):
        return self.dec(torch.cat([z, self.emb(lab)], 1))


# ============================================================ image backbones

class ConvVAENet(nn.Module):
    def __init__(self, C, H, n_embed, latent, ch):
        super().__init__()
        s = H // 4
        self.s, self.ch = s, ch
        self.emb = nn.Embedding(n_embed, 16)
        self.enc = nn.Sequential(
            nn.Conv2d(C + 1, ch, 4, 2, 1), nn.GroupNorm(8, ch), nn.LeakyReLU(0.2),
            nn.Conv2d(ch, 2 * ch, 4, 2, 1), nn.GroupNorm(8, 2 * ch), nn.LeakyReLU(0.2), nn.Flatten())
        self.lab_map = nn.Linear(16, H * H)
        self.fc_mu = nn.Linear(2 * ch * s * s + 16, latent)
        self.fc_lv = nn.Linear(2 * ch * s * s + 16, latent)
        self.dec_fc = nn.Linear(latent + 16, 2 * ch * s * s)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(2 * ch, ch, 4, 2, 1), nn.GroupNorm(8, ch), nn.ReLU(),
            nn.ConvTranspose2d(ch, C, 4, 2, 1))
        self.H = H

    def encode(self, x, lab):
        e = self.emb(lab)
        lm = self.lab_map(e).view(-1, 1, self.H, self.H)
        h = torch.cat([self.enc(torch.cat([x, lm], 1)), e], 1)
        return self.fc_mu(h), self.fc_lv(h)

    def decode(self, z, lab):
        h = self.dec_fc(torch.cat([z, self.emb(lab)], 1)).view(-1, 2 * self.ch, self.s, self.s)
        return self.dec(h)


class ConvGenerator(nn.Module):
    def __init__(self, C, H, n_embed, zdim, ch):
        super().__init__()
        s = H // 4
        self.s, self.ch = s, ch
        self.emb = nn.Embedding(n_embed, 32)
        self.fc = nn.Sequential(nn.Linear(zdim + 32, 4 * ch * s * s), nn.LayerNorm(4 * ch * s * s), nn.ReLU())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(4 * ch, 2 * ch, 4, 2, 1), nn.GroupNorm(8, 2 * ch), nn.ReLU(),
            nn.ConvTranspose2d(2 * ch, ch, 4, 2, 1), nn.GroupNorm(8, ch), nn.ReLU(),
            nn.Conv2d(ch, C, 3, 1, 1), nn.Tanh())

    def forward(self, z, lab):
        h = self.fc(torch.cat([z, self.emb(lab)], 1)).view(-1, 4 * self.ch, self.s, self.s)
        return self.net(h)


class ConvDiscriminator(nn.Module):
    """Projection discriminator; spectral normalization unless the loss regularizes the critic
    itself (WGAN-GP's gradient penalty)."""

    def __init__(self, C, H, n_embed, ch, spectral=True):
        super().__init__()
        sn = lambda m: _sn(m, spectral)  # noqa: E731
        self.body = nn.Sequential(
            sn(nn.Conv2d(C, ch, 4, 2, 1)), nn.LeakyReLU(0.2),
            sn(nn.Conv2d(ch, 2 * ch, 4, 2, 1)), nn.LeakyReLU(0.2),
            sn(nn.Conv2d(2 * ch, 4 * ch, 3, 1, 1)), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.out = sn(nn.Linear(4 * ch, 1))
        self.emb = sn(nn.Embedding(n_embed, 4 * ch))

    def forward(self, x, lab):
        h = self.body(x)
        return self.out(h).squeeze(1) + (self.emb(lab) * h).sum(1)
