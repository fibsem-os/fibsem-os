"""The ``nnunet`` backend: nnU-Net v2 models
(``fibsem.segmentation.nnunet_model_v2.SegmentationModelNNUnetV2``).

The model is an nnU-Net trained-model folder, *ref*; the checkpoint inside it
is a setting, ``checkpoint_best.pth`` unless set.
"""

import os
from dataclasses import dataclass
from typing import Any, ClassVar, List

import numpy as np

from fibsem.ai.models.backends._packages import missing
from fibsem.ai.models.card import ModelCard

PACKAGES = ("torch", "nnunetv2")


@dataclass
class NnunetConfig:
    """*checkpoint* is the checkpoint file inside the model folder."""

    checkpoint: str = "checkpoint_best.pth"


class NnunetModel:
    """The ``nnunet`` backend. ``model`` is the loaded model itself, for the
    detection code that calls it directly."""

    name: ClassVar[str] = "nnunet"
    Config: ClassVar[type] = NnunetConfig

    def __init__(self) -> None:
        self.model: Any = None
        self.device = "cpu"

    @staticmethod
    def requirements(card: ModelCard) -> List[str]:
        return missing([*PACKAGES, *card.packages])

    def load(self, source: str, card: ModelCard, config: Any, device: str) -> None:
        from fibsem.segmentation.nnunet_model_v2 import SegmentationModelNNUnetV2

        checkpoint = os.path.join(source, config.checkpoint)
        self.model = SegmentationModelNNUnetV2(checkpoint=checkpoint)
        self.device = str(self.model.device)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """A class id per pixel."""
        return np.asarray(self.model.inference(image, rgb=False))

    def unload(self) -> None:
        self.model = None
