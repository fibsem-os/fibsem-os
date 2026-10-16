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


def load_checkpoint(checkpoint: str):
    """The model a protocol or the app names by its checkpoint, as the
    framework object the detection code calls itself (``.inference``), loaded
    once by the default manager and shared by every caller after."""
    return default_manager().for_checkpoint(checkpoint).backend.model


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
    "load_checkpoint",
]
