"""Image generators: class-conditional DDPM (diffusers), conditional conv VAE, conditional DCGAN.

All operate on (N, C, H, W) float images in [0, 1] with H, W divisible by 4. Class conditioning
uses an extra "null" class index so the same networks also work for unlabeled data."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import batches, device
from .base import GenerativeModel, ModelContext, register


def _n_embed(ctx: ModelContext) -> int:
    # class indices 0..K-1 plus a null index K (unconditional / classifier-free guidance)
    return ctx.n_classes + 1


def _label_tensor(ctx: ModelContext, y: np.ndarray | None, n: int, dev) -> torch.Tensor:
    if y is None:
        return torch.full((n,), ctx.n_classes, dtype=torch.long, device=dev)
    return torch.as_tensor(y, dtype=torch.long, device=dev)


# ============================================================ DDPM (diffusers)

@register
class ImageDDPM(GenerativeModel):
    name = "ddpm_image"
    family = "diffusion"
    modalities = ("image",)
    library = "diffusers (UNet2DModel + DDPM/DDIM schedulers)"
    conditional = "class embedding with classifier-free guidance"
    capabilities = ("sample", "anomaly")
    summary = ("Denoising diffusion with a small U-Net. Best sample fidelity and diversity for images; "
               "slowest to train and sample. Anomaly score = denoising error averaged over noise levels.")

    @classmethod
    def default_config(cls, ctx):
        big = ctx.n_train > 20000
        return {"epochs": 15 if big else 60, "batch_size": 128, "lr": 3e-4, "base_channels": 64,
                "layers_per_block": 1, "attention": False, "num_train_timesteps": 1000, "sampling_steps": 50,
                "guidance_scale": 2.0, "p_uncond": 0.1, "ema_decay": 0.999}

    @classmethod
    def search_space(cls, ctx):
        return {"lr": {"type": "float", "low": 5e-5, "high": 1e-3, "log": True},
                "base_channels": {"type": "categorical", "choices": [32, 64]},
                "layers_per_block": {"type": "int", "low": 1, "high": 2},
                "guidance_scale": {"type": "float", "low": 1.0, "high": 4.0}}

    def _build(self):
        from diffusers import DDPMScheduler, UNet2DModel

        c = self.config["base_channels"]
        C, H, _ = self.ctx.shape
        down = ("DownBlock2D", "AttnDownBlock2D" if self.config["attention"] else "DownBlock2D", "DownBlock2D")
        up = ("UpBlock2D", "AttnUpBlock2D" if self.config["attention"] else "UpBlock2D", "UpBlock2D")
        self.unet = UNet2DModel(sample_size=H, in_channels=C, out_channels=C,
                                layers_per_block=self.config["layers_per_block"],
                                block_out_channels=(c, 2 * c, 2 * c), down_block_types=down,
                                up_block_types=up, num_class_embeds=_n_embed(self.ctx)).to(device())
        self.sched = DDPMScheduler(num_train_timesteps=self.config["num_train_timesteps"])

    def _ema_update(self, ema, decay, step):
        # Warm-up so that short runs (e.g. search trials) are not dominated by the initial weights
        decay = min(decay, (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), self.unet.parameters()):
                pe.lerp_(p, 1 - decay)

    def fit(self, X, y, X_val, y_val, deadline):
        self._build()
        dev = device()
        opt = torch.optim.AdamW(self.unet.parameters(), lr=self.config["lr"])
        # Exponential moving average of weights: used for sampling, smooths the noisy late iterates
        ema = copy.deepcopy(self.unet).eval()
        Xt = torch.from_numpy(X) * 2 - 1
        yt = torch.from_numpy(y) if y is not None else None
        losses, ep, step = [], 0, 0
        for ep in range(1, self.config["epochs"] + 1):
            self.unet.train()
            perm = torch.randperm(len(Xt))
            tot = 0.0
            for sl in batches(len(Xt), self.config["batch_size"]):
                idx = perm[sl]
                x0 = Xt[idx].to(dev)
                lab = _label_tensor(self.ctx, yt[idx].numpy() if yt is not None else None, len(idx), dev)
                # Classifier-free guidance training: randomly drop labels to the null class
                drop = torch.rand(len(idx), device=dev) < self.config["p_uncond"]
                lab = torch.where(drop, torch.full_like(lab, self.ctx.n_classes), lab)
                noise = torch.randn_like(x0)
                t = torch.randint(0, self.sched.config.num_train_timesteps, (len(idx),), device=dev)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    pred = self.unet(self.sched.add_noise(x0, noise, t), t, class_labels=lab).sample
                loss = F.mse_loss(pred.float(), noise)
                opt.zero_grad()
                loss.backward()
                opt.step()
                self._ema_update(ema, self.config["ema_decay"], step)
                step += 1
                tot += loss.item() * len(idx)
                if deadline.expired():
                    break
            losses.append(tot / len(Xt))
            if deadline.expired():
                break
        self.unet = ema
        self.history = {"train_loss": losses, "epochs_run": ep, "stopped_by_budget": deadline.expired()}
        return self.history

    @torch.no_grad()
    def sample(self, n, y=None):
        from diffusers import DDIMScheduler

        dev = device()
        y = self._labels_for(n, y)
        sched = DDIMScheduler.from_config(self.sched.config)
        sched.set_timesteps(self.config["sampling_steps"])
        self.unet.eval()
        g = self.config["guidance_scale"] if self.ctx.labeled else 0.0
        out = []
        for sl in batches(n, 512):
            m = sl.stop - sl.start
            x = torch.randn((m, *self.ctx.shape), device=dev)
            lab = _label_tensor(self.ctx, y[sl] if y is not None else None, m, dev)
            null = torch.full_like(lab, self.ctx.n_classes)
            for t in sched.timesteps:
                tt = torch.full((m,), int(t), device=dev, dtype=torch.long)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    eps = self.unet(x, tt, class_labels=lab).sample.float()
                    if g > 0:
                        eps_u = self.unet(x, tt, class_labels=null).sample.float()
                        eps = eps_u + g * (eps - eps_u)
                x = sched.step(eps, t, x).prev_sample
            out.append(((x.clamp(-1, 1) + 1) / 2).cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    @torch.no_grad()
    def anomaly_score(self, X):
        dev = device()
        self.unet.eval()
        ts = [50, 200, 400]
        scores = []
        for sl in batches(len(X), 512):
            x0 = torch.from_numpy(X[sl]).to(dev) * 2 - 1
            lab = _label_tensor(self.ctx, None, len(x0), dev)
            err = 0
            for t in ts:
                noise = torch.randn_like(x0)
                tt = torch.full((len(x0),), t, device=dev, dtype=torch.long)
                pred = self.unet(self.sched.add_noise(x0, noise, tt), tt, class_labels=lab).sample
                err = err + ((pred - noise) ** 2).mean((1, 2, 3))
            scores.append((err / len(ts)).cpu().numpy())
        return np.concatenate(scores)


# ============================================================ conditional VAE

class _CVAENet(nn.Module):
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

    def decode_logits(self, z, lab):
        h = self.dec_fc(torch.cat([z, self.emb(lab)], 1)).view(-1, 2 * self.ch, self.s, self.s)
        return self.dec(h)


@register
class ImageCVAE(GenerativeModel):
    name = "cvae_image"
    family = "vae"
    modalities = ("image",)
    library = "from scratch (PyTorch)"
    conditional = "label embedding fed to encoder and decoder"
    capabilities = ("sample", "encode", "anomaly")
    summary = ("Conditional convolutional VAE. Fast, stable, gives a latent space and a likelihood bound "
               "(anomaly score = negative ELBO); samples are typically blurrier than diffusion/GAN.")

    @classmethod
    def default_config(cls, ctx):
        return {"epochs": 15 if ctx.n_train > 20000 else 50, "batch_size": 256, "lr": 1e-3,
                "latent_dim": 32, "channels": 64, "beta": 1.0}

    @classmethod
    def search_space(cls, ctx):
        return {"lr": {"type": "float", "low": 2e-4, "high": 3e-3, "log": True},
                "latent_dim": {"type": "categorical", "choices": [16, 32, 64]},
                "beta": {"type": "float", "low": 0.25, "high": 2.0, "log": True}}

    def _loss(self, x, lab):
        mu, lv = self.net.encode(x, lab)
        z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        logits = self.net.decode_logits(z, lab)
        rec = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum((1, 2, 3))
        kl = -0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)
        return rec, kl

    def fit(self, X, y, X_val, y_val, deadline):
        C, H, _ = self.ctx.shape
        dev = device()
        self.net = _CVAENet(C, H, _n_embed(self.ctx), self.config["latent_dim"], self.config["channels"]).to(dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.config["lr"])
        Xt = torch.from_numpy(X)
        losses, ep = [], 0
        for ep in range(1, self.config["epochs"] + 1):
            self.net.train()
            perm = torch.randperm(len(Xt))
            tot = 0.0
            for sl in batches(len(Xt), self.config["batch_size"]):
                idx = perm[sl]
                x = Xt[idx].to(dev)
                lab = _label_tensor(self.ctx, y[idx.numpy()] if y is not None else None, len(idx), dev)
                # Occasionally use the null label so label-free encoding/scoring stays in-distribution
                drop = torch.rand(len(idx), device=dev) < 0.1
                lab = torch.where(drop, torch.full_like(lab, self.ctx.n_classes), lab)
                rec, kl = self._loss(x, lab)
                loss = (rec + self.config["beta"] * kl).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(idx)
                if deadline.expired():
                    break
            losses.append(tot / len(Xt))
            if deadline.expired():
                break
        self.net.eval()
        self.history = {"train_loss": losses, "epochs_run": ep, "stopped_by_budget": deadline.expired()}
        return self.history

    @torch.no_grad()
    def sample(self, n, y=None):
        dev = device()
        y = self._labels_for(n, y)
        out = []
        for sl in batches(n, 2048):
            m = sl.stop - sl.start
            lab = _label_tensor(self.ctx, y[sl] if y is not None else None, m, dev)
            z = torch.randn(m, self.config["latent_dim"], device=dev)
            out.append(torch.sigmoid(self.net.decode_logits(z, lab)).cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    @torch.no_grad()
    def encode(self, X, y=None):
        dev = device()
        out = []
        for sl in batches(len(X), 2048):
            lab = _label_tensor(self.ctx, y[sl] if y is not None else None, sl.stop - sl.start, dev)
            mu, _ = self.net.encode(torch.from_numpy(X[sl]).to(dev), lab)
            out.append(mu.cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def anomaly_score(self, X):
        dev = device()
        out = []
        for sl in batches(len(X), 2048):
            x = torch.from_numpy(X[sl]).to(dev)
            lab = _label_tensor(self.ctx, None, len(x), dev)
            rec, kl = self._loss(x, lab)
            out.append((rec + kl).cpu().numpy())
        return np.concatenate(out)


# ============================================================ conditional DCGAN

class _Gen(nn.Module):
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


class _Disc(nn.Module):
    """Projection discriminator with spectral normalization."""

    def __init__(self, C, H, n_embed, ch):
        super().__init__()
        sn = nn.utils.spectral_norm
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


@register
class ImageCDCGAN(GenerativeModel):
    name = "dcgan_image"
    family = "gan"
    modalities = ("image",)
    library = "from scratch (PyTorch)"
    conditional = "label embedding in generator, projection discriminator"
    capabilities = ("sample",)
    summary = ("Conditional DCGAN (hinge loss, spectral norm). Sharp samples and very fast sampling; "
               "risk of mode collapse and no likelihood or encoder.")

    @classmethod
    def default_config(cls, ctx):
        return {"epochs": 15 if ctx.n_train > 20000 else 60, "batch_size": 128, "lr_g": 2e-4, "lr_d": 2e-4,
                "z_dim": 100, "channels": 64}

    @classmethod
    def search_space(cls, ctx):
        return {"lr_g": {"type": "float", "low": 5e-5, "high": 5e-4, "log": True},
                "lr_d": {"type": "float", "low": 5e-5, "high": 5e-4, "log": True},
                "channels": {"type": "categorical", "choices": [32, 64]}}

    def fit(self, X, y, X_val, y_val, deadline):
        C, H, _ = self.ctx.shape
        dev = device()
        ne = _n_embed(self.ctx)
        self.G = _Gen(C, H, ne, self.config["z_dim"], self.config["channels"]).to(dev)
        D = _Disc(C, H, ne, self.config["channels"]).to(dev)
        og = torch.optim.Adam(self.G.parameters(), lr=self.config["lr_g"], betas=(0.0, 0.9))
        od = torch.optim.Adam(D.parameters(), lr=self.config["lr_d"], betas=(0.0, 0.9))
        Xt = torch.from_numpy(X) * 2 - 1
        hist_d, hist_g, ep = [], [], 0
        for ep in range(1, self.config["epochs"] + 1):
            self.G.train()
            perm = torch.randperm(len(Xt))
            sd = sg = 0.0
            nb = 0
            for sl in batches(len(Xt), self.config["batch_size"]):
                idx = perm[sl]
                x = Xt[idx].to(dev)
                lab = _label_tensor(self.ctx, y[idx.numpy()] if y is not None else None, len(idx), dev)
                z = torch.randn(len(idx), self.config["z_dim"], device=dev)
                fake = self.G(z, lab)
                ld = F.relu(1 - D(x, lab)).mean() + F.relu(1 + D(fake.detach(), lab)).mean()
                od.zero_grad()
                ld.backward()
                od.step()
                lg = -D(fake, lab).mean()
                og.zero_grad()
                lg.backward()
                og.step()
                sd += ld.item()
                sg += lg.item()
                nb += 1
                if deadline.expired():
                    break
            hist_d.append(sd / nb)
            hist_g.append(sg / nb)
            if deadline.expired():
                break
        self.G.eval()
        self.history = {"d_loss": hist_d, "g_loss": hist_g, "epochs_run": ep, "stopped_by_budget": deadline.expired()}
        return self.history

    @torch.no_grad()
    def sample(self, n, y=None):
        dev = device()
        y = self._labels_for(n, y)
        out = []
        for sl in batches(n, 2048):
            m = sl.stop - sl.start
            lab = _label_tensor(self.ctx, y[sl] if y is not None else None, m, dev)
            out.append(((self.G(torch.randn(m, self.config["z_dim"], device=dev), lab) + 1) / 2).cpu().numpy())
        return np.concatenate(out).astype(np.float32)
