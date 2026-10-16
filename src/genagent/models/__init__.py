"""Model registry. Importing this package registers every model."""

from . import flow, image, scrna, tabular  # noqa: F401  (registration side effects)
from .base import REGISTRY, GenerativeModel, ModelContext, models_for

__all__ = ["REGISTRY", "GenerativeModel", "ModelContext", "models_for"]
