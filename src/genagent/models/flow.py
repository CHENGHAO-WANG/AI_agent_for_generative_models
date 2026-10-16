"""Neural spline flow (zuko) for scRNA (on PCA of log-normalized expression) and tabular data
(on the standardized / one-hot codec space)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from ..embed import scrna_counts_from_lognorm, scrna_lognorm
from ..utils import batches, device
from .base import GenerativeModel, register


@register
class NSFFlow(GenerativeModel):
    name = "nsf_flow"
    family = "flow"
    modalities = ("scrna", "tabular")
    library = "zuko (neural spline flow)"
    conditional = "one-hot label as flow context"
    capabilities = ("sample", "anomaly")
    summary = ("Neural spline flow with exact log-likelihood (anomaly score = negative log-likelihood). "
               "scRNA: models the top principal components of log-normalized expression, then maps back to "
               "counts (loses variation outside the PCA subspace). Tabular: models the standardized/one-hot "
               "encoding with dequantization noise on categorical columns.")

    @classmethod
    def default_config(cls, ctx):
        return {"epochs": 300 if ctx.n_train < 5000 else 60, "batch_size": 256, "lr": 1e-3, "transforms": 4,
                "hidden": 128, "n_components": 30, "cat_noise": 0.15}

    @classmethod
    def search_space(cls, ctx):
        sp = {"lr": {"type": "float", "low": 2e-4, "high": 3e-3, "log": True},
              "transforms": {"type": "int", "low": 2, "high": 6},
              "hidden": {"type": "categorical", "choices": [64, 128, 256]}}
        if ctx.modality == "scrna":
            sp["n_components"] = {"type": "categorical", "choices": [10, 20, 30, 50]}
        return sp

    # -- data space ------------------------------------------------------------
    def _to_space(self, X, fit=False):
        if self.ctx.modality == "scrna":
            L = scrna_lognorm(X)
            if fit:
                self.pca = PCA(self.config["n_components"], random_state=self.ctx.seed).fit(L)
            E = self.pca.transform(L)
        else:
            E = self.ctx.codec.encode(X)
        if fit:
            self.mu, self.sd = E.mean(0), E.std(0) + 1e-6
        return ((E - self.mu) / self.sd).astype(np.float32)

    def _from_space(self, E, y, rng):
        E = E * self.sd + self.mu
        if self.ctx.modality == "scrna":
            L = self.pca.inverse_transform(E)
            return scrna_counts_from_lognorm(L, self._sample_libsizes(y, len(E), rng), rng)
        return self.ctx.codec.decode(E)

    def _ctx(self, y, n, dev):
        if not self.ctx.labeled or y is None:
            return None
        return F.one_hot(torch.as_tensor(y, device=dev), self.ctx.n_classes).float()

    def fit(self, X, y, X_val, y_val, deadline):
        import zuko

        dev = device()
        c = self.config
        E = torch.from_numpy(self._to_space(X, fit=True))
        cat_mask = torch.zeros(E.shape[1], dtype=torch.bool)
        if self.ctx.modality == "tabular":
            cat_mask[len(self.ctx.codec.num):] = True
        self.flow = zuko.flows.NSF(E.shape[1], context=self.ctx.n_classes if self.ctx.labeled else 0,
                                   transforms=c["transforms"], hidden_features=[c["hidden"]] * 2).to(dev)
        opt = torch.optim.Adam(self.flow.parameters(), lr=c["lr"])
        Ev = torch.from_numpy(self._to_space(X_val))
        tr_loss, va_loss, ep = [], [], 0
        best, best_state = float("inf"), None
        for ep in range(1, c["epochs"] + 1):
            self.flow.train()
            perm = torch.randperm(len(E))
            tot = 0.0
            for sl in batches(len(E), c["batch_size"]):
                idx = perm[sl]
                x = E[idx].to(dev)
                # Dequantization noise on one-hot blocks makes the discrete part continuous
                x = x + cat_mask.to(dev) * torch.randn_like(x) * c["cat_noise"]
                cx = self._ctx(y[idx.numpy()] if y is not None else None, len(idx), dev)
                loss = -self.flow(cx).log_prob(x).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.flow.parameters(), 5.0)
                opt.step()
                tot += loss.item() * len(idx)
            tr_loss.append(tot / len(E))
            with torch.no_grad():
                self.flow.eval()
                v = -self.flow(self._ctx(y_val, len(Ev), dev)).log_prob(Ev.to(dev)).mean().item()
            va_loss.append(v)
            # Keep the best validation checkpoint (flows overfit small data quickly)
            if v < best:
                best, best_state = v, {k: t.detach().clone() for k, t in self.flow.state_dict().items()}
            if deadline.expired():
                break
        if best_state is not None:
            self.flow.load_state_dict(best_state)
        self.flow.eval()
        self.history = {"train_nll": tr_loss, "val_nll": va_loss, "best_val_nll": best, "epochs_run": ep,
                        "stopped_by_budget": deadline.expired()}
        return self.history

    @torch.no_grad()
    def sample(self, n, y=None):
        dev = device()
        rng = np.random.default_rng(self.ctx.seed + 17)
        torch.manual_seed(self.ctx.seed + 17)
        y = self._labels_for(n, y)
        parts = []
        for sl in batches(n, 4096):
            m = sl.stop - sl.start
            cx = self._ctx(y[sl] if y is not None else None, m, dev)
            E = self.flow(cx).sample() if cx is not None else self.flow().sample((m,))
            E = E.cpu().numpy().clip(-8, 8)  # guard against rare tail explosions
            parts.append(self._from_space(E, y[sl] if y is not None else None, rng))
        if self.ctx.modality == "tabular":
            import pandas as pd

            return pd.concat(parts, ignore_index=True)
        return np.concatenate(parts).astype(np.float32)

    @torch.no_grad()
    def anomaly_score(self, X):
        dev = device()
        E = torch.from_numpy(self._to_space(X))
        return np.concatenate([(-self.flow().log_prob(E[sl].to(dev)) if not self.ctx.labeled else
                                -self._marginal_logprob(E[sl].to(dev))).cpu().numpy()
                               for sl in batches(len(E), 4096)])

    def _marginal_logprob(self, x):
        # log p(x) = logsumexp_c [log p(x|c) + log p(c)]
        lps = []
        for c in range(self.ctx.n_classes):
            cx = F.one_hot(torch.full((len(x),), c, device=x.device), self.ctx.n_classes).float()
            lps.append(self.flow(cx).log_prob(x) + float(np.log(max(self.ctx.class_freq[c], 1e-12))))
        return torch.logsumexp(torch.stack(lps, 0), 0)
