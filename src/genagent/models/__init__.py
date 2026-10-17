"""Model registry. Importing this package registers every model."""

from . import cvae, ddpm, flow, gan, gplvm, ppca, scrna, tabular  # noqa: F401  (registration side effects)
from .base import ALIASES, REGISTRY, GenerativeModel, ModelContext, models_for, resolve_model

__all__ = ["ALIASES", "REGISTRY", "GenerativeModel", "ModelContext", "models_for", "resolve_model"]
