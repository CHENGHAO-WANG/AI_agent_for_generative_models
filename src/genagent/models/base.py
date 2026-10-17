"""Common interface and registry for generative models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch

REGISTRY: dict[str, type["GenerativeModel"]] = {}

# Former per-modality names -> (registry name, fixed config); kept so older plans and runs still work
ALIASES: dict[str, tuple[str, dict]] = {
    "ddpm_image": ("ddpm", {}),
    "cvae_image": ("cvae", {}),
    "dcgan_image": ("gan", {"loss": "hinge"}),
    "wgan_gp_scrna": ("gan", {"loss": "wgan_gp"}),
}


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


def resolve_model(name: str) -> tuple[type["GenerativeModel"], dict]:
    """Registry class for a model name or alias, plus the config the alias fixes. KeyError if unknown."""
    if name in ALIASES:
        name, fixed = ALIASES[name]
        return REGISTRY[name], dict(fixed)
    return REGISTRY[name], {}


@dataclass
class ModelContext:
    """What a model needs to know about the data, independent of the samples themselves."""
    modality: str
    n_train: int
    n_classes: int = 0
    class_names: list[str] | None = None
    shape: tuple[int, ...] = ()           # image (C, H, W)
    n_features: int = 0                   # scRNA genes / tabular raw columns
    codec: Any = None                     # TabularCodec for tabular
    col_types: dict[str, str] | None = None
    libsizes: np.ndarray | None = None    # scRNA training library sizes
    libsizes_by_class: dict[int, np.ndarray] = field(default_factory=dict)
    class_freq: np.ndarray | None = None
    seed: int = 0

    @property
    def labeled(self) -> bool:
        return self.n_classes > 0

    @staticmethod
    def from_dataset(ds, seed: int = 0) -> "ModelContext":
        from ..data import tabular_column_types
        from ..embed import TabularCodec

        tr = ds.splits["train"]
        ctx = ModelContext(modality=ds.modality, n_train=len(tr), seed=seed)
        if ds.labeled:
            ctx.n_classes = len(ds.class_names)
            ctx.class_names = ds.class_names
            ctx.class_freq = np.bincount(tr.y, minlength=ctx.n_classes) / len(tr.y)
        if ds.modality == "image":
            ctx.shape = tuple(tr.X.shape[1:])
            ctx.n_features = int(np.prod(ctx.shape))
        elif ds.modality == "scrna":
            ctx.n_features = tr.X.shape[1]
            ctx.libsizes = tr.X.sum(1)
            if ds.labeled:
                ctx.libsizes_by_class = {c: tr.X[tr.y == c].sum(1) for c in range(ctx.n_classes)}
        else:
            ctx.col_types = tabular_column_types(tr.X)
            ctx.codec = TabularCodec(tr.X, ctx.col_types)
            ctx.n_features = tr.X.shape[1]
        return ctx


class GenerativeModel:
    """Subclasses implement fit/sample and optionally encode/anomaly_score.

    `fit` receives real data splits (X, y) and a Deadline; `sample(n, y)` returns n samples in
    the dataset's native representation (images, counts, or DataFrame). When the model was
    trained with labels, `y` gives the class of each requested sample."""

    name: ClassVar[str]
    family: ClassVar[str]               # vae | gan | diffusion | flow | latent_variable
    modalities: ClassVar[tuple[str, ...]]
    library: ClassVar[str]
    conditional: ClassVar[str]          # how class-conditioning works
    capabilities: ClassVar[tuple[str, ...]] = ("sample",)
    requires_labels: ClassVar[bool] = False  # cannot be trained without class labels
    summary: ClassVar[str] = ""

    def __init__(self, config: dict, ctx: ModelContext):
        self.config = {**self.default_config(ctx), **(config or {})}
        self.ctx = ctx
        self.history: dict[str, Any] = {}

    # -- configuration -------------------------------------------------------
    @classmethod
    def default_config(cls, ctx: ModelContext) -> dict:
        return {}

    @classmethod
    def search_space(cls, ctx: ModelContext) -> dict:
        """{param: {"type": "float"|"int"|"categorical", "low", "high", "log", "choices"}}"""
        return {}

    @classmethod
    def describe(cls, ctx: ModelContext | None = None) -> dict:
        d = {"name": cls.name, "family": cls.family, "modalities": list(cls.modalities),
             "library": cls.library, "conditional": cls.conditional,
             "capabilities": list(cls.capabilities), "requires_labels": cls.requires_labels,
             "summary": cls.summary}
        if ctx is not None:
            d["default_config"] = cls.default_config(ctx)
            d["search_space"] = cls.search_space(ctx)
        return d

    # -- lifecycle -------------------------------------------------------------
    def fit(self, X, y, X_val, y_val, deadline) -> dict:
        raise NotImplementedError

    def sample(self, n: int, y: np.ndarray | None = None):
        raise NotImplementedError

    def encode(self, X) -> np.ndarray:
        raise NotImplementedError(f"{self.name} has no latent representation")

    def anomaly_score(self, X) -> np.ndarray:
        raise NotImplementedError(f"{self.name} has no anomaly score")

    def save(self, path: Path) -> None:
        torch.save(self, Path(path) / "model.pt")

    @staticmethod
    def load(path: Path) -> "GenerativeModel":
        path = Path(path)
        if (path / "scvi_wrapper.pt").exists():
            from .scrna import load_scvi_based

            return load_scvi_based(path)
        return torch.load(path / "model.pt", weights_only=False)

    # -- helpers ----------------------------------------------------------------
    def _labels_for(self, n: int, y: np.ndarray | None) -> np.ndarray | None:
        """Default labels: draw from the training class distribution."""
        if not self.ctx.labeled:
            return None
        if y is not None:
            return np.asarray(y, dtype=np.int64)
        rng = np.random.default_rng(self.ctx.seed + 7)
        return rng.choice(self.ctx.n_classes, size=n, p=self.ctx.class_freq)

    def _sample_libsizes(self, y: np.ndarray | None, n: int, rng: np.random.Generator) -> np.ndarray:
        from .spaces import sample_libsizes

        return sample_libsizes(self.ctx, y, n, rng)


def n_embed(ctx: ModelContext) -> int:
    # class indices 0..K-1 plus a null index K (unconditional / classifier-free guidance)
    return ctx.n_classes + 1


def label_tensor(ctx: ModelContext, y: np.ndarray | None, n: int, dev) -> torch.Tensor:
    """Class indices for a batch; the null index when no labels are given."""
    if y is None:
        return torch.full((n,), ctx.n_classes, dtype=torch.long, device=dev)
    return torch.as_tensor(y, dtype=torch.long, device=dev)


def models_for(modality: str) -> list[type[GenerativeModel]]:
    return [m for m in REGISTRY.values() if modality in m.modalities]
