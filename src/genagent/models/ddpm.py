"""Class-conditional DDPM for every modality, and the diffusion core shared with latent diffusion.

Images use a diffusers U-Net on pixels scaled to [-1, 1]; scRNA and tabular data use a residual MLP
denoiser on standardized vectors (`VectorSpace`). Labels enter as a class embedding trained with
classifier-free guidance (labels randomly dropped to a null class)."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import batches, device
from .base import GenerativeModel, label_tensor, n_embed, register
from .nets import MLPDenoiser
from .spaces import VectorSpace, concat_samples


# ============================================================ diffusion core

def ema_update(ema: nn.Module, net: nn.Module, decay: float, step: int) -> None:
    # Warm-up so that short runs (e.g. search trials) are not dominated by the initial weights
    decay = min(decay, (1 + step) / (10 + step))
    with torch.no_grad():
        for pe, p in zip(ema.parameters(), net.parameters()):
            pe.lerp_(p, 1 - decay)


def epoch_batches(X: torch.Tensor, lab: torch.Tensor, bs: int, epochs: int, dev, state: dict):
    """Shuffled passes over the data; records the current epoch in `state`."""
    for ep in range(1, epochs + 1):
        state["epochs_run"] = ep
        perm = torch.randperm(len(X))
        for sl in batches(len(X), bs):
            idx = perm[sl]
            yield X[idx].to(dev), lab[idx].to(dev)


def random_batches(X: torch.Tensor, lab: torch.Tensor, bs: int, steps: int, gen: torch.Generator):
    """`steps` batches drawn with replacement (X and lab already on the device)."""
    for _ in range(steps):
        idx = torch.randint(0, len(X), (bs,), device=X.device, generator=gen)
        yield X[idx], lab[idx]


def train_denoiser(net: nn.Module, sched, batch_iter, null: int, lr: float, p_uncond: float, ema_decay: float,
                   deadline, total_steps: int | None = None, autocast: bool = False, log_every: int = 100):
    """Epsilon-prediction training with classifier-free label dropout and an EMA of the weights.

    `net(x_t, t, lab)` predicts the noise. With `total_steps`, the learning rate follows a cosine
    decay. Returns (EMA network, mean loss per `log_every` steps, steps run)."""
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_steps) if total_steps else None
    ema = copy.deepcopy(net).eval()
    losses, run, step = [], 0.0, 0
    T = sched.config.num_train_timesteps
    for x0, lab in batch_iter:
        drop = torch.rand(len(lab), device=lab.device) < p_uncond
        lab = torch.where(drop, torch.full_like(lab, null), lab)
        noise = torch.randn_like(x0)
        t = torch.randint(0, T, (len(x0),), device=x0.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            pred = net(sched.add_noise(x0, noise, t), t, lab)
        loss = F.mse_loss(pred.float(), noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if lr_sched is not None:
            lr_sched.step()
        ema_update(ema, net, ema_decay, step)
        step += 1
        run += loss.item()
        if step % log_every == 0:
            losses.append(run / log_every)
            run = 0.0
        if deadline.expired():
            break
    return ema, losses, step


@torch.no_grad()
def sample_denoiser(net: nn.Module, train_sched, shape: tuple, lab: torch.Tensor, null: int, guidance: float,
                    steps: int, sampler: str = "ddim", gen: torch.Generator | None = None,
                    autocast: bool = False) -> torch.Tensor:
    """Reverse diffusion from Gaussian noise. `sampler`: "ddim" (deterministic, few steps) or
    "ddpm" (ancestral, stochastic). Classifier-free guidance when `guidance` > 0."""
    from diffusers import DDIMScheduler, DDPMScheduler

    S = DDIMScheduler if sampler == "ddim" else DDPMScheduler
    sched = S.from_config(train_sched.config)
    sched.set_timesteps(steps)
    net.eval()
    x = torch.randn(shape, device=lab.device, generator=gen)
    nl = torch.full_like(lab, null)
    for t in sched.timesteps:
        tt = torch.full((len(x),), int(t), device=x.device, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            eps = net(x, tt, lab).float()
            if guidance > 0:
                eps_u = net(x, tt, nl).float()
                eps = eps_u + guidance * (eps - eps_u)
        x = sched.step(eps, t, x, generator=gen).prev_sample
    return x


@torch.no_grad()
def denoising_error(net: nn.Module, sched, x0: torch.Tensor, null: int, ts=(50, 200, 400)) -> torch.Tensor:
    """Per-sample noise-prediction error averaged over a few noise levels (unconditional)."""
    lab = torch.full((len(x0),), null, device=x0.device, dtype=torch.long)
    err = 0
    for t in ts:
        noise = torch.randn_like(x0)
        tt = torch.full((len(x0),), t, device=x0.device, dtype=torch.long)
        err = err + ((net(sched.add_noise(x0, noise, tt), tt, lab) - noise) ** 2).flatten(1).mean(1)
    return err / len(ts)


class _UNet(nn.Module):
    """diffusers U-Net behind the (x, t, lab) -> noise interface of the core."""

    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x, t, lab):
        return self.unet(x, t, class_labels=lab).sample


# ============================================================ DDPM model

@register
class DDPM(GenerativeModel):
    name = "ddpm"
    family = "diffusion"
    modalities = ("image", "scrna", "tabular")
    library = ("diffusers (UNet2DModel + DDPM/DDIM schedulers) for images; diffusers scheduler + residual MLP "
               "denoiser (from scratch) for scRNA and tabular")
    conditional = "class embedding with classifier-free guidance"
    capabilities = ("sample", "anomaly")
    summary = ("Denoising diffusion. Images: small U-Net; best sample fidelity and diversity, slowest to train and "
               "sample. scRNA: MLP denoiser directly on standardized log-normalized genes (data-hungry; counts by "
               "Poisson sampling). Tabular: MLP denoiser on the standardized/one-hot encoding, categories by argmax "
               "(a continuous relaxation, cruder than CTGAN's handling of categories). Anomaly score = denoising "
               "error averaged over noise levels.")

    @classmethod
    def default_config(cls, ctx):
        if ctx.modality == "image":
            big = ctx.n_train > 20000
            return {"epochs": 15 if big else 60, "batch_size": 128, "lr": 3e-4, "base_channels": 64,
                    "layers_per_block": 1, "attention": False, "num_train_timesteps": 1000, "sampling_steps": 50,
                    "guidance_scale": 2.0, "p_uncond": 0.1, "ema_decay": 0.999}
        return {"diff_steps": 10000, "batch_size": 256, "lr": 1e-3, "width": 512 if ctx.modality == "scrna" else 256,
                "depth": 4, "sampling_steps": 1000, "guidance_scale": 1.0, "p_uncond": 0.1, "ema_decay": 0.999}

    @classmethod
    def search_space(cls, ctx):
        if ctx.modality == "image":
            return {"lr": {"type": "float", "low": 5e-5, "high": 1e-3, "log": True},
                    "base_channels": {"type": "categorical", "choices": [32, 64]},
                    "layers_per_block": {"type": "int", "low": 1, "high": 2},
                    "guidance_scale": {"type": "float", "low": 1.0, "high": 4.0}}
        return {"lr": {"type": "float", "low": 2e-4, "high": 3e-3, "log": True},
                "width": {"type": "categorical", "choices": [256, 512, 1024]},
                "guidance_scale": {"type": "float", "low": 0.0, "high": 3.0}}

    @property
    def _image(self) -> bool:
        return self.ctx.modality == "image"

    def _build(self, dim: int | None = None):
        from diffusers import DDPMScheduler, UNet2DModel

        c = self.config
        if self._image:
            ch = c["base_channels"]
            C, H, _ = self.ctx.shape
            down = ("DownBlock2D", "AttnDownBlock2D" if c["attention"] else "DownBlock2D", "DownBlock2D")
            up = ("UpBlock2D", "AttnUpBlock2D" if c["attention"] else "UpBlock2D", "UpBlock2D")
            self.net = _UNet(UNet2DModel(sample_size=H, in_channels=C, out_channels=C,
                                         layers_per_block=c["layers_per_block"], block_out_channels=(ch, 2 * ch, 2 * ch),
                                         down_block_types=down, up_block_types=up,
                                         num_class_embeds=n_embed(self.ctx))).to(device())
            self.sched = DDPMScheduler(num_train_timesteps=c["num_train_timesteps"])
        else:
            self.net = MLPDenoiser(dim, n_embed(self.ctx), c["width"], c["depth"]).to(device())
            # Standardized vectors are not bounded: the scheduler must not clip predictions to [-1, 1]
            self.sched = DDPMScheduler(num_train_timesteps=1000, clip_sample=False)

    def _vectors(self, X, fit=False) -> torch.Tensor:
        if self._image:
            return torch.from_numpy(np.asarray(X, dtype=np.float32)) * 2 - 1
        if fit:
            self.space = VectorSpace(self.ctx)
            return torch.from_numpy(self.space.fit(X))
        return torch.from_numpy(self.space.encode(X))

    def fit(self, X, y, X_val, y_val, deadline):
        dev = device()
        c = self.config
        Xt = self._vectors(X, fit=True)
        self._build(Xt.shape[1] if not self._image else None)
        lab = label_tensor(self.ctx, y, len(Xt), "cpu")
        state = {"epochs_run": 0}
        if self._image:
            it = epoch_batches(Xt, lab, c["batch_size"], c["epochs"], dev, state)
            per_epoch = -(-len(Xt) // c["batch_size"])
            self.net, losses, steps = train_denoiser(self.net, self.sched, it, self.ctx.n_classes, c["lr"],
                                                     c["p_uncond"], c["ema_decay"], deadline,
                                                     autocast=dev.type == "cuda", log_every=per_epoch)
            self.history = {"train_loss": losses, "epochs_run": state["epochs_run"]}
        else:
            g = torch.Generator(device=dev).manual_seed(self.ctx.seed)
            it = random_batches(Xt.to(dev), lab.to(dev), c["batch_size"], c["diff_steps"], g)
            self.net, losses, steps = train_denoiser(self.net, self.sched, it, self.ctx.n_classes, c["lr"],
                                                     c["p_uncond"], c["ema_decay"], deadline,
                                                     total_steps=c["diff_steps"])
            self.history = {"loss_every_100_steps": losses, "diffusion_steps_run": steps,
                            "epochs_run": round(steps * c["batch_size"] / len(Xt), 1)}
        self.history["stopped_by_budget"] = deadline.expired()
        return self.history

    @torch.no_grad()
    def sample(self, n, y=None):
        dev = device()
        c = self.config
        y = self._labels_for(n, y)
        rng = np.random.default_rng(self.ctx.seed + 19)
        gen = torch.Generator(device=dev).manual_seed(self.ctx.seed + 19)
        g = c["guidance_scale"] if y is not None else 0.0
        out = []
        for sl in batches(n, 512 if self._image else 4096):
            m = sl.stop - sl.start
            ys = y[sl] if y is not None else None
            lab = label_tensor(self.ctx, ys, m, dev)
            if self._image:
                x = sample_denoiser(self.net, self.sched, (m, *self.ctx.shape), lab, self.ctx.n_classes, g,
                                    c["sampling_steps"], "ddim", gen, autocast=dev.type == "cuda")
                out.append(((x.clamp(-1, 1) + 1) / 2).cpu().numpy())
            else:
                # Ancestral sampling: matched the spread of the data better than DDIM for vectors
                x = sample_denoiser(self.net, self.sched, (m, self.space.dim), lab, self.ctx.n_classes, g,
                                    c["sampling_steps"], "ddpm", gen)
                out.append(self.space.decode(x.cpu().numpy().clip(-8, 8), ys, rng))
        return concat_samples(out, self.ctx.modality)

    @torch.no_grad()
    def anomaly_score(self, X):
        dev = device()
        self.net.eval()
        Xt = self._vectors(X)
        return np.concatenate([denoising_error(self.net, self.sched, Xt[sl].to(dev), self.ctx.n_classes).cpu().numpy()
                               for sl in batches(len(Xt), 512)])
