"""The ``legacy`` backend: a checkpoint as `fibsem.segmentation.model.load_model`
has always read it, its backend picked from its name.

It is how the checkpoints protocols and tasks name keep working, loaded once
through the manager instead of on every detection. Its card is made from the
checkpoint's name and fibsem's shared class labels, since these checkpoints
carry none of their own.
"""

from dataclasses import dataclass
from typing import Any, ClassVar, List

import numpy as np

from fibsem.ai.models.backends._packages import missing
from fibsem.ai.models.card import ClassInfo, ModelCard


@dataclass
class LegacyConfig:
    """No settings: the checkpoint is the model."""


class LegacyModel:
    """The ``legacy`` backend. ``model`` is the loaded model itself, for the
    detection code that calls it directly."""

    name: ClassVar[str] = "legacy"
    Config: ClassVar[type] = LegacyConfig

    def __init__(self) -> None:
        self.model: Any = None
        self.device = "cpu"

    @staticmethod
    def default_card(entry: Any) -> ModelCard:
        import matplotlib.colors as mcolors

        from fibsem.segmentation.config import CLASS_COLORS, CLASS_LABELS

        return ModelCard(
            id=entry.name,
            version="",
            backend="legacy",
            classes={
                int(k): ClassInfo(
                    str(label), mcolors.to_hex(CLASS_COLORS.get(k, "white"))
                )
                for k, label in CLASS_LABELS.items()
            },
        )

    @staticmethod
    def requirements(card: ModelCard) -> List[str]:
        return missing(["torch", *card.packages])

    def load(self, source: str, card: ModelCard, config: Any, device: str) -> None:
        from fibsem.segmentation.model import load_model

        checkpoint = source.split("/", 3)[-1]
        self.model = load_model(checkpoint=checkpoint)
        self.device = str(getattr(self.model, "device", device))

    def predict(self, image: np.ndarray) -> np.ndarray:
        """A class id per pixel."""
        mask = np.asarray(self.model.inference(image, rgb=False))
        return mask[0] if mask.ndim == 3 else mask

    def unload(self) -> None:
        self.model = None
