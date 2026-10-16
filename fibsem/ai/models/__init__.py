"""Registered models: what this install offers, what each one is, and loading
them once.

    from fibsem.ai import models

    model = models.get("fake-segmentation")
    result = model.segment(image)
    result.fraction("Cell"), result.save(folder)

A caller names a model; the registry (`registry`) says where it is and how it is
set up, its card (`card`) says what it is, and the manager (`manager`) loads it
once and runs it through its backend (`backends`). Only the backend knows the
framework.
"""

from fibsem.ai.models.card import ClassInfo, InputSpec, ModelCard
from fibsem.ai.models.manager import (
    InputMismatch,
    Model,
    ModelManager,
    ModelNotFound,
    ModelUnavailable,
    default_manager,
)
from fibsem.ai.models.registry import ModelEntry, ModelRegistry
from fibsem.ai.models.results import SegmentationResult


def get(name: str, **config) -> Model:
    """The model registered as *name* (``name@version`` pins one), loaded once by
    the default manager. *config* overrides its registered settings."""
    return default_manager().get(name, **config)


__all__ = [
    "ClassInfo",
    "InputMismatch",
    "InputSpec",
    "Model",
    "ModelCard",
    "ModelEntry",
    "ModelManager",
    "ModelNotFound",
    "ModelRegistry",
    "ModelUnavailable",
    "SegmentationResult",
    "default_manager",
    "get",
]
