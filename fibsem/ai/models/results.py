"""Results, one type per task: what a model answered, with which model made it.

Every result records the model's name, its pinned version, the backend and the
settings it ran with, so a result read years later still says where it came from.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage

from fibsem.ai.models.card import ClassInfo

# How much of a labelled pixel's colour is the class's, over the image.
OVERLAY_ALPHA = 0.45


@dataclass
class SegmentationResult:
    """A class id per pixel (*mask*), the classes by id, the pixel size in
    metres (None when unknown), and which model made it (*provenance*)."""

    mask: np.ndarray
    classes: Dict[int, ClassInfo]
    pixel_size: Optional[float]
    provenance: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)

    def _id(self, name: str) -> int:
        for class_id, info in self.classes.items():
            if info.name == name:
                return class_id
        names = ", ".join(info.name for info in self.classes.values())
        raise ValueError(f"no class '{name}'; the classes: {names}")

    def fraction(self, name: str) -> float:
        """The share of the image labelled *name*, 0 to 1."""
        return float(np.mean(self.mask == self._id(name)))

    def area_um2(self, name: str) -> Optional[float]:
        """The area labelled *name* in µm², or None without a pixel size."""
        if self.pixel_size is None:
            return None
        pixels = int(np.count_nonzero(self.mask == self._id(name)))
        return pixels * (self.pixel_size * 1e6) ** 2

    def count(self, name: str) -> int:
        """How many separate regions are labelled *name*."""
        _, found = ndimage.label(self.mask == self._id(name))
        return int(found)

    def overlay(self, image: np.ndarray) -> np.ndarray:
        """*image* in grey with each class drawn over it in its colour, RGB
        uint8. Background (class 0) is left as it is."""
        grey = np.asarray(image, dtype=np.float64)
        if grey.ndim == 3:
            grey = grey.mean(axis=2)
        low, high = float(grey.min()), float(grey.max())
        grey = (grey - low) / (high - low) * 255 if high > low else grey * 0
        rgb = np.repeat(grey[..., None], 3, axis=2)
        for class_id, info in self.classes.items():
            if class_id == 0 or not info.colour:
                continue
            colour = np.array(_rgb(info.colour), dtype=np.float64)
            where = self.mask == class_id
            rgb[where] = (1 - OVERLAY_ALPHA) * rgb[where] + OVERLAY_ALPHA * colour
        return rgb.astype(np.uint8)

    def to_dict(self) -> Dict[str, Any]:
        """The result without the mask: each labelled class's share, area and
        region count, the image's pixel size, and the provenance."""
        classes = {}
        for class_id, info in self.classes.items():
            if class_id == 0:
                continue
            classes[info.name] = {
                "fraction": round(self.fraction(info.name), 4),
                "area_um2": _rounded(self.area_um2(info.name)),
                "count": self.count(info.name),
            }
        return {
            "model": self.provenance,
            "pixel_size_m": self.pixel_size,
            "shape": list(self.mask.shape),
            "classes": classes,
            "warnings": list(self.warnings),
        }

    def save(
        self, folder: str, stem: str, image: Optional[np.ndarray] = None
    ) -> Dict[str, str]:
        """Save the mask (``<stem>-mask.tif``), the result (``<stem>-result.json``)
        and, given the *image*, the overlay (``<stem>-overlay.png``) in *folder*.
        Answers their paths."""
        os.makedirs(folder, exist_ok=True)
        paths = {
            "mask_path": os.path.join(folder, f"{stem}-mask.tif"),
            "result_path": os.path.join(folder, f"{stem}-result.json"),
        }
        tifffile.imwrite(paths["mask_path"], self.mask)
        if image is not None:
            paths["overlay_path"] = os.path.join(folder, f"{stem}-overlay.png")
            Image.fromarray(self.overlay(image)).save(paths["overlay_path"])
        with open(paths["result_path"], "w") as f:
            json.dump({**self.to_dict(), **paths}, f, indent=2)
        return paths


def _rgb(colour: str) -> List[int]:
    colour = colour.lstrip("#")
    return [int(colour[i : i + 2], 16) for i in (0, 2, 4)]


def _rounded(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value, 1)
