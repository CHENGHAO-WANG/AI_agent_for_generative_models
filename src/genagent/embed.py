"""Shared feature spaces used by the general metrics, plus modality codecs.

- scRNA:   log-normalized expression -> PCA(50), fit on real train data
- tabular: TabularCodec (standardize / one-hot) -> PCA(<=20)
- image:   penultimate features of a CNN classifier trained on real train data
           (labeled) or the bottleneck of a conv autoencoder (unlabeled)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

from .utils import batches, device


def scrna_lognorm(X: np.ndarray, target: float = 1e4) -> np.ndarray:
    lib = X.sum(1, keepdims=True)
    return np.log1p(X / np.maximum(lib, 1) * target).astype(np.float32)


def scrna_counts_from_lognorm(L: np.ndarray, libsizes: np.ndarray, rng: np.random.Generator,
                              target: float = 1e4) -> np.ndarray:
    """Map generated log-normalized expression back to counts: rescale to a sampled library
    size and draw Poisson counts around the implied means."""
    expr = np.clip(np.expm1(np.clip(L, 0, None)), 0, None)
    frac = expr / np.maximum(expr.sum(1, keepdims=True), 1e-8)
    mu = frac * libsizes[:, None]
    return rng.poisson(mu).astype(np.float32)


class TabularCodec:
    """Numeric: median-impute + standardize. Categorical: mode-impute + one-hot.
    `decode` inverts (argmax for categorical blocks)."""

    def __init__(self, df: pd.DataFrame, types: dict[str, str]):
        self.columns = list(df.columns)
        self.types = types
        self.num = [c for c in self.columns if types[c] == "numeric"]
        self.cat = [c for c in self.columns if types[c] == "categorical"]
        self.med = {c: float(df[c].median()) for c in self.num}
        self.mu = {c: float(df[c].fillna(self.med[c]).mean()) for c in self.num}
        self.sd = {c: float(df[c].fillna(self.med[c]).std() or 1.0) for c in self.num}
        self.is_int = {c: bool(np.allclose(df[c].dropna(), np.round(df[c].dropna()))) for c in self.num}
        self.levels = {c: list(pd.Series(df[c].dropna().unique()).astype(str).sort_values()) for c in self.cat}
        self.mode = {c: str(df[c].astype(str).mode().iloc[0]) for c in self.cat}
        # Integer-coded categorical columns are decoded back to numbers
        self.cat_numeric = {c: bool(pd.api.types.is_numeric_dtype(df[c])) for c in self.cat}

    @property
    def dim(self) -> int:
        return len(self.num) + sum(len(v) for v in self.levels.values())

    def encode(self, df: pd.DataFrame) -> np.ndarray:
        parts = []
        for c in self.num:
            parts.append(((df[c].fillna(self.med[c]).astype(float) - self.mu[c]) / self.sd[c]).to_numpy()[:, None])
        for c in self.cat:
            v = df[c].astype(str).where(df[c].notna(), self.mode[c]).to_numpy(dtype=object)
            parts.append((v[:, None] == np.array(self.levels[c])[None, :]).astype(float))
        return np.concatenate(parts, 1).astype(np.float32) if parts else np.zeros((len(df), 0), np.float32)

    def decode(self, A: np.ndarray) -> pd.DataFrame:
        out, j = {}, 0
        for c in self.num:
            v = A[:, j] * self.sd[c] + self.mu[c]
            out[c] = np.round(v) if self.is_int[c] else v
            j += 1
        for c in self.cat:
            k = len(self.levels[c])
            v = np.array(self.levels[c])[A[:, j:j + k].argmax(1)]
            out[c] = pd.to_numeric(v) if self.cat_numeric[c] else v
            j += k
        return pd.DataFrame(out)[self.columns]


# ---------------------------------------------------------------- image networks

class SmallCNN(nn.Module):
    """Compact classifier; `features` is the 128-d penultimate representation."""

    def __init__(self, in_ch: int, n_classes: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.GroupNorm(8, 128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Linear(128, n_classes)

    def features(self, x):
        return self.body(x)

    def forward(self, x):
        return self.head(self.body(x))


class ConvAE(nn.Module):
    def __init__(self, in_ch: int, side: int, latent: int = 64):
        super().__init__()
        s = side // 4
        self.s = s
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * s * s, latent))
        self.dec_fc = nn.Linear(latent, 64 * s * s)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(32, in_ch, 4, 2, 1), nn.Sigmoid())

    def features(self, x):
        return self.enc(x)

    def forward(self, x):
        h = self.dec_fc(self.enc(x)).view(-1, 64, self.s, self.s)
        return self.dec(h)


def train_classifier(X: np.ndarray, y: np.ndarray, n_classes: int, epochs: int = 5, seed: int = 0,
                     bs: int = 256, lr: float = 1e-3, class_weight: bool = False,
                     steps: int | None = None) -> SmallCNN:
    """Train SmallCNN on images with light flip augmentation and a cosine learning-rate decay.

    `steps` fixes the number of optimizer steps (overriding `epochs`), so classifiers trained on
    datasets of different sizes get the same training effort."""
    torch.manual_seed(seed)
    dev = device()
    net = SmallCNN(X.shape[1], n_classes).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    steps = steps or epochs * ((len(X) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    w = None
    if class_weight:
        cnt = np.bincount(y, minlength=n_classes).astype(np.float32)
        w = torch.tensor(cnt.sum() / np.maximum(cnt, 1) / n_classes, device=dev)
    g = torch.Generator().manual_seed(seed)
    done = 0
    while done < steps:
        net.train()
        perm = torch.randperm(len(Xt), generator=g)
        for sl in batches(len(Xt), bs):
            if done >= steps:
                break
            idx = perm[sl]
            xb, yb = Xt[idx].to(dev), yt[idx].to(dev)
            if torch.rand(1, generator=g).item() < 0.5:
                xb = xb.flip(3)
            loss = F.cross_entropy(net(xb), yb, weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            done += 1
    net.eval()
    return net


@torch.no_grad()
def predict_proba(net: nn.Module, X: np.ndarray, bs: int = 1024) -> np.ndarray:
    dev = device()
    net.eval()
    out = [F.softmax(net(torch.from_numpy(X[sl]).to(dev)), 1).cpu().numpy() for sl in batches(len(X), bs)]
    return np.concatenate(out)


@torch.no_grad()
def net_features(net: nn.Module, X: np.ndarray, bs: int = 1024) -> np.ndarray:
    dev = device()
    net.eval()
    return np.concatenate([net.features(torch.from_numpy(X[sl]).to(dev)).cpu().numpy()
                           for sl in batches(len(X), bs)])


def train_autoencoder(X: np.ndarray, epochs: int = 5, seed: int = 0, bs: int = 256) -> ConvAE:
    torch.manual_seed(seed)
    dev = device()
    net = ConvAE(X.shape[1], X.shape[2]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    Xt = torch.from_numpy(X)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for sl in batches(len(Xt), bs):
            xb = Xt[perm[sl]].to(dev)
            loss = F.mse_loss(net(xb), xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    return net


# ---------------------------------------------------------------- shared embedder

class Embedder:
    """Fit on real training data; maps any sample set of the same modality to feature vectors."""

    def __init__(self, ds, seed: int = 0, max_fit: int = 100000):
        self.modality = ds.modality
        tr = ds.splits["train"]
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(tr))[:max_fit]
        self.classifier = None
        if ds.modality == "image":
            X = tr.X[idx]
            if ds.labeled:
                self.net = train_classifier(X, tr.y[idx], len(ds.class_names), epochs=8, seed=seed)
                self.classifier = self.net
                self.kind = "cnn-classifier-features"
            else:
                self.net = train_autoencoder(X, epochs=5, seed=seed)
                self.kind = "conv-autoencoder-features"
        elif ds.modality == "scrna":
            L = scrna_lognorm(tr.X)
            self.pca = PCA(min(50, L.shape[1] - 1), random_state=seed).fit(L)
            self.kind = "lognorm-pca50"
        else:
            from .data import tabular_column_types

            self.codec = TabularCodec(tr.X, tabular_column_types(tr.X))
            E = self.codec.encode(tr.X)
            self.pca = PCA(min(20, E.shape[1]), random_state=seed).fit(E) if E.shape[1] > 20 else None
            self.kind = "tabular-codec" + ("-pca20" if self.pca is not None else "")

    def transform(self, X) -> np.ndarray:
        if self.modality == "image":
            return net_features(self.net, np.asarray(X, dtype=np.float32))
        if self.modality == "scrna":
            return self.pca.transform(scrna_lognorm(X)).astype(np.float32)
        E = self.codec.encode(X)
        return (self.pca.transform(E) if self.pca is not None else E).astype(np.float32)
