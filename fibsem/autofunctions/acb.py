"""Auto-contrast / brightness (ACB) for FIB-SEM images."""
from __future__ import annotations

import dataclasses
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from fibsem.structures import BeamType, FibsemRectangle

if TYPE_CHECKING:
    from fibsem.structures import FibsemImage, ImageStats

logger = logging.getLogger(__name__)


@dataclass
class AutoContrastBrightnessSettings:
    """Parameters controlling the hardware ACB loop.

    Convergence is declared when both hard criteria are satisfied:
      - |mean - mean_target| <= mean_tolerance
      - saturation_hi <= saturation_limit

    Soft targets (range_utilisation, contrast_ratio) are reported but do not
    block convergence.
    """
    n_iterations: int = 5
    brightness_step: float = 0.05
    contrast_step: float = 0.05
    mean_target: float = 0.5
    mean_tolerance: float = 0.05
    saturation_limit: float = 0.005
    min_range_utilisation: float = 0.6
    probe_resolution: tuple = (512, 384)
    probe_dwell_time: float = 0.2e-6

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AutoContrastBrightnessSettings":
        return cls(**d)


@dataclass
class AutoContrastBrightnessIteration:
    """State captured at a single step of the hardware ACB loop."""
    brightness: float
    contrast: float
    stats: ImageStats
    image: FibsemImage

    def to_dict(self, index: int) -> dict:
        return {
            "index": index,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "image": f"iter_{index:02d}.tif",
            "stats": dataclasses.asdict(self.stats),
        }

    @classmethod
    def from_dict(cls, d: dict, result_dir: Path) -> "AutoContrastBrightnessIteration":
        from fibsem.structures import FibsemImage, ImageStats
        return cls(
            brightness=d["brightness"],
            contrast=d["contrast"],
            stats=ImageStats(**d["stats"]),
            image=FibsemImage.load(str(result_dir / d["image"])),
        )


@dataclass
class AutoContrastBrightnessResult:
    """Result of a hardware ACB run, containing all per-iteration data."""
    image: FibsemImage
    stats: ImageStats
    converged: bool
    iterations: list[AutoContrastBrightnessIteration]
    settings: AutoContrastBrightnessSettings = None

    @property
    def n_iterations(self) -> int:
        return len(self.iterations)

    def plot(self, save_path: str = "acb.png") -> None:
        from fibsem.autofunctions.plotting import plot_acb_result
        plot_acb_result(self, save_path=save_path)

    def save(self, path: str = ".", name: str = "acb") -> Path:
        """Save the result to a timestamped directory.

        Creates ``<path>/<name>-<timestamp>/`` containing:
          - ``data.json``  — stats and controls for every iteration
          - ``iter_00.tif``, ``iter_01.tif``, … — probe image per iteration
          - ``final.tif`` — the final probe image

        Args:
            path: Parent directory (created if needed).
            name: Prefix for the result directory.

        Returns:
            Path to the created directory.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = Path(path) / f"{name}-{ts}"
        result_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "converged": self.converged,
            "n_iterations": self.n_iterations,
            "final_stats": dataclasses.asdict(self.stats),
            "settings": self.settings.to_dict() if self.settings is not None else None,
            "iterations": [it.to_dict(i) for i, it in enumerate(self.iterations)],
        }

        with open(result_dir / "data.json", "w") as f:
            json.dump(data, f, indent=2)

        for i, it in enumerate(self.iterations):
            it.image.save(path=str(result_dir / f"iter_{i:02d}"))

        self.image.save(path=str(result_dir / "final"))

        logger.info("ACB result saved to %s", result_dir)
        return result_dir

    @classmethod
    def load(cls, result_dir: str) -> "AutoContrastBrightnessResult":
        """Load a result previously saved with :meth:`save`.

        Args:
            result_dir: Path to the directory created by :meth:`save`.

        Returns:
            Reconstructed ``AutoContrastBrightnessResult``.
        """
        result_dir = Path(result_dir)

        with open(result_dir / "data.json") as f:
            data = json.load(f)

        iterations = [
            AutoContrastBrightnessIteration.from_dict(it_data, result_dir)
            for it_data in data["iterations"]
        ]

        from fibsem.structures import FibsemImage, ImageStats
        final_image = FibsemImage.load(str(result_dir / "final.tif"))
        final_stats = ImageStats(**data["final_stats"])
        settings_data = data.get("settings")
        settings = AutoContrastBrightnessSettings.from_dict(settings_data) if settings_data is not None else None

        return cls(
            image=final_image,
            stats=final_stats,
            converged=data["converged"],
            iterations=iterations,
            settings=settings,
        )


def run_auto_contrast_brightness(
    microscope,
    beam_type: BeamType = BeamType.ELECTRON,
    hfw: float = 150e-6,
    settings: AutoContrastBrightnessSettings = None,
) -> AutoContrastBrightnessResult:
    """Iteratively adjust detector brightness/contrast until histogram targets are met.

    Acquires fast probe images and steps the microscope detector controls until
    the mean intensity and saturation limits converge.  The caller should then
    acquire a full-resolution image with the adjusted detector settings.

    Args:
        microscope: FibsemMicroscope instance.
        beam_type: Which beam the detector belongs to.
        hfw: Horizontal field width for the probe images (metres).
        settings: ``AutoContrastBrightnessSettings``; defaults constructed if ``None``.

    Returns:
        ``AutoContrastBrightnessResult`` with the final probe image, stats, convergence
        flag, and the full per-iteration history.
    """
    from fibsem.structures import ImageSettings

    if settings is None:
        settings = AutoContrastBrightnessSettings()

    probe_settings = ImageSettings(
        resolution=settings.probe_resolution,
        dwell_time=settings.probe_dwell_time,
        hfw=hfw,
        beam_type=beam_type,
        autocontrast=False,
        autogamma=False,
        save=False,
        reduced_area=FibsemRectangle(0.25, 0.25, 0.5, 0.5)
    )

    brightness = microscope.get_detector_brightness(beam_type)
    contrast = microscope.get_detector_contrast(beam_type)
    iterations: list[AutoContrastBrightnessIteration] = []

    probe = microscope.acquire_image(image_settings=probe_settings)
    stats = probe.compute_stats()
    iterations.append(AutoContrastBrightnessIteration(brightness=brightness, contrast=contrast, stats=stats, image=probe))
    logger.debug("ACB iter 0: brightness=%.3f contrast=%.3f stats=[%s]", brightness, contrast, stats)

    for i in range(settings.n_iterations):
        if stats.converged(settings.mean_target, settings.mean_tolerance, settings.saturation_limit):
            logger.debug("ACB converged at iteration %d", i)
            break

        mean_error = stats.mean - settings.mean_target
        brightness -= np.sign(mean_error) * settings.brightness_step
        brightness = float(np.clip(brightness, 0.0, 1.0))

        if stats.saturation_hi > settings.saturation_limit:
            contrast -= settings.contrast_step
        elif stats.range_utilisation < settings.min_range_utilisation:
            contrast += settings.contrast_step
        contrast = float(np.clip(contrast, 0.0, 1.0))

        microscope.set_detector_brightness(brightness, beam_type)
        microscope.set_detector_contrast(contrast, beam_type)

        probe = microscope.acquire_image(image_settings=probe_settings)
        stats = probe.compute_stats()
        iterations.append(AutoContrastBrightnessIteration(brightness=brightness, contrast=contrast, stats=stats, image=probe))
        logger.debug(
            "ACB iter %d: brightness=%.3f contrast=%.3f stats=[%s]",
            i + 1, brightness, contrast, stats,
        )

    return AutoContrastBrightnessResult(
        image=probe,
        stats=stats,
        converged=stats.converged(settings.mean_target, settings.mean_tolerance, settings.saturation_limit),
        iterations=iterations,
        settings=settings,
    )
