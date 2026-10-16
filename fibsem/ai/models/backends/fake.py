"""``FakeSegmentationModel``: a model that needs no framework, for tests and the
worked example.

It labels the grid screening classes without looking at the pixels: grid bars
on a lattice, a few cells in the squares between them, and a patch of
contamination, placed by its *seed*. The same image size and seed give the same
mask.
"""

from dataclasses import dataclass
from typing import Any, ClassVar, List

import numpy as np

from fibsem.ai.models.card import ClassInfo, ModelCard

BACKGROUND, CELL, GRID_BAR, BEADS, CONTAMINATION, CRACK = range(6)


@dataclass
class FakeConfig:
    """The fake model's settings: *seed* places its cells and contamination."""

    seed: int = 0


class FakeSegmentationModel:
    """The ``fake`` backend."""

    name: ClassVar[str] = "fake"
    Config: ClassVar[type] = FakeConfig

    def __init__(self) -> None:
        self.config = FakeConfig()

    @staticmethod
    def default_card() -> ModelCard:
        return ModelCard(
            id="fake-segmentation",
            version="0.1.0",
            backend="fake",
            task="segmentation",
            classes={
                BACKGROUND: ClassInfo("background", "#000000"),
                CELL: ClassInfo("Cell", "#ff0000"),
                GRID_BAR: ClassInfo("Grid Bar", "#00ff00"),
                BEADS: ClassInfo("Beads", "#00ffff"),
                CONTAMINATION: ClassInfo("Contamination", "#ff9900"),
                CRACK: ClassInfo("Crack", "#ff00ff"),
            },
            provenance={"trained_on": "none: draws bars and blobs, for tests"},
        )

    @staticmethod
    def requirements(card: ModelCard) -> List[str]:
        """The packages it needs that are missing: none."""
        return []

    def load(self, source: str, card: ModelCard, config: Any, device: str) -> None:
        self.config = config

    def predict(self, image: np.ndarray) -> np.ndarray:
        """A class id per pixel, the size of *image*."""
        height, width = image.shape[:2]
        rng = np.random.default_rng(self.config.seed)
        mask = np.zeros((height, width), dtype=np.uint8)
        yy, xx = np.mgrid[0:height, 0:width]

        # A lattice of bars, three squares across the short side.
        pitch = max(min(height, width) // 3, 1)
        bar = max(pitch // 6, 1)
        on_bar = ((xx + pitch // 2) % pitch < bar) | ((yy + pitch // 2) % pitch < bar)
        mask[on_bar] = GRID_BAR

        # A cell in some of the squares, and one patch of contamination.
        radius = max(pitch // 10, 1)
        for _ in range(4):
            cy, cx = rng.integers(0, height), rng.integers(0, width)
            blob = (yy - cy) ** 2 + (xx - cx) ** 2 < radius**2
            mask[blob & ~on_bar] = CELL
        cy, cx = rng.integers(0, height), rng.integers(0, width)
        patch = (yy - cy) ** 2 / 4 + (xx - cx) ** 2 < (2 * radius) ** 2
        mask[patch] = CONTAMINATION
        return mask

    def unload(self) -> None:
        pass
