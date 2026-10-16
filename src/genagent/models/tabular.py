"""Tabular generators from the `ctgan` library: CTGAN and TVAE.

The label (if any) is modeled as one more discrete column; CTGAN samples it conditionally,
TVAE uses rejection sampling to hit requested class counts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import GenerativeModel, register

LABEL_COL = "__label__"


class _CTGANBase(GenerativeModel):
    modalities = ("tabular",)
    library = "ctgan"

    def _frame(self, X: pd.DataFrame, y) -> tuple[pd.DataFrame, list[str]]:
        """Impute missing values (the library cannot model NaN) and append the label column."""
        codec = self.ctx.codec
        df = X.copy()
        for c in codec.num:
            df[c] = df[c].fillna(codec.med[c]).astype(float)
        for c in codec.cat:
            df[c] = df[c].astype(str).where(X[c].notna(), codec.mode[c])
        discrete = list(codec.cat)
        if y is not None:
            df[LABEL_COL] = np.asarray(y).astype(str)
            discrete.append(LABEL_COL)
        return df, discrete

    def _post(self, df: pd.DataFrame) -> pd.DataFrame:
        # Restore integer-valued numeric columns and the original column order
        codec = self.ctx.codec
        for c in codec.num:
            if codec.is_int[c]:
                df[c] = np.round(df[c].astype(float))
        return df[codec.columns].reset_index(drop=True)

    def fit(self, X, y, X_val, y_val, deadline):
        df, discrete = self._frame(X, y)
        self.model = self._make()
        self.model.fit(df, discrete_columns=discrete)
        loss = getattr(self.model, "loss_values", None)
        self.history = {"epochs_run": self.config["epochs"],
                        "loss_tail": loss.tail(5).to_dict("records") if loss is not None else None}
        return self.history

    def _sample_raw(self, n, cls_value=None):
        raise NotImplementedError

    def sample(self, n, y=None):
        y = self._labels_for(n, y)
        if y is None:
            return self._post(self._sample_raw(n))
        parts, order = [], []
        for c in np.unique(y):
            k = int((y == c).sum())
            parts.append(self._sample_raw(k, str(c)).head(k))
            order.append(np.flatnonzero(y == c))
        out = pd.concat(parts, ignore_index=True)
        # Restore the requested order of labels
        out.index = np.concatenate(order)
        return self._post(out.sort_index())


@register
class CTGANModel(_CTGANBase):
    name = "ctgan"
    family = "gan"
    conditional = "native conditional vector (training-by-sampling)"
    capabilities = ("sample",)
    summary = ("CTGAN: conditional GAN for mixed-type tables with mode-specific normalization of numeric "
               "columns; handles imbalanced categorical columns well; needs several hundred epochs.")

    @classmethod
    def default_config(cls, ctx):
        return {"epochs": 300 if ctx.n_train < 20000 else 100, "batch_size": 500, "generator_lr": 2e-4,
                "discriminator_lr": 2e-4, "embedding_dim": 128}

    @classmethod
    def search_space(cls, ctx):
        return {"generator_lr": {"type": "float", "low": 5e-5, "high": 1e-3, "log": True},
                "discriminator_lr": {"type": "float", "low": 5e-5, "high": 1e-3, "log": True},
                "embedding_dim": {"type": "categorical", "choices": [64, 128, 256]}}

    def _make(self):
        from ctgan import CTGAN

        c = self.config
        bs = max(10, (min(c["batch_size"], self.ctx.n_train) // 10) * 10)  # must be divisible by pac=10
        return CTGAN(epochs=c["epochs"], batch_size=bs, generator_lr=c["generator_lr"],
                     discriminator_lr=c["discriminator_lr"], embedding_dim=c["embedding_dim"])

    def _sample_raw(self, n, cls_value=None):
        if cls_value is None:
            return self.model.sample(n)
        return self.model.sample(n, condition_column=LABEL_COL, condition_value=cls_value)


@register
class TVAEModel(_CTGANBase):
    name = "tvae"
    family = "vae"
    conditional = "rejection sampling on the generated label column"
    capabilities = ("sample",)
    summary = ("TVAE: VAE for mixed-type tables (same preprocessing as CTGAN). Usually more stable and "
               "faster to train than CTGAN on small tables; weaker on rare categories.")

    @classmethod
    def default_config(cls, ctx):
        return {"epochs": 300 if ctx.n_train < 20000 else 100, "batch_size": 500, "embedding_dim": 128,
                "loss_factor": 2}

    @classmethod
    def search_space(cls, ctx):
        return {"embedding_dim": {"type": "categorical", "choices": [32, 64, 128]},
                "loss_factor": {"type": "float", "low": 1.0, "high": 4.0}}

    def _make(self):
        from ctgan import TVAE

        c = self.config
        return TVAE(epochs=c["epochs"], batch_size=min(c["batch_size"], self.ctx.n_train),
                    embedding_dim=c["embedding_dim"], loss_factor=c["loss_factor"])

    def _sample_raw(self, n, cls_value=None):
        if cls_value is None:
            return self.model.sample(n)
        got, tries = [], 0
        need = n
        while need > 0 and tries < 30:
            s = self.model.sample(max(4 * need, 100))
            s = s[s[LABEL_COL] == cls_value]
            got.append(s)
            need -= len(s)
            tries += 1
        out = pd.concat(got, ignore_index=True)
        if len(out) < n:  # class too rare to hit by rejection; pad by resampling what we have
            out = pd.concat([out, out.sample(n - len(out), replace=True, random_state=0)], ignore_index=True) \
                if len(out) else self.model.sample(n)
        return out
