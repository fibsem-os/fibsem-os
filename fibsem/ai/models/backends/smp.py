"""The ``smp`` backend: the segmentation-models-pytorch U-Nets AutoLamella's
detection has always run (``fibsem.segmentation.model.SegmentationModel``).

The model is a checkpoint file, *ref*, or a checkpoint name in fibsem's
Hugging Face repository, downloaded once on first load as before.
"""

from dataclasses import dataclass
from typing import Any, ClassVar, List

import numpy as np

from fibsem.ai.models.backends._packages import missing
from fibsem.ai.models.card import ModelCard

PACKAGES = ("torch", "segmentation_models_pytorch")


@dataclass
class SmpConfig:
    """*checkpoint* is a different checkpoint than *ref* ("" keeps *ref*)."""

    checkpoint: str = ""


class SmpModel:
    """The ``smp`` backend."""

    name: ClassVar[str] = "smp"
    Config: ClassVar[type] = SmpConfig

    def __init__(self) -> None:
        self._model: Any = None
        self.device = "cpu"

    @staticmethod
    def requirements(card: ModelCard) -> List[str]:
        return missing([*PACKAGES, *card.packages])

    def load(self, source: str, card: ModelCard, config: Any, device: str) -> None:
        from fibsem.segmentation.model import SegmentationModel

        self._model = SegmentationModel(checkpoint=config.checkpoint or source)
        self.device = str(self._model.device)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """A class id per pixel."""
        mask = self._model.inference(image, rgb=False)
        return np.asarray(mask)[0] if np.ndim(mask) == 3 else np.asarray(mask)

    def unload(self) -> None:
        self._model = None
