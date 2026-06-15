"""Image-based auto-focus for FIB-SEM images."""
from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from fibsem.structures import BeamType

import numpy as np
if TYPE_CHECKING:
    from fibsem.structures import BeamType, FibsemImage, FibsemRectangle
    from fibsem.microscope import FibsemMicroscope

logger = logging.getLogger(__name__)


@dataclass
class FocusSweepPass:
    """A single pass in a multi-pass focus sweep."""
    n_steps: int = 10
    step_size: float = 0.5e-3   # metres


@dataclass
class AutoFocusSettings:
    """Parameters controlling the image-based auto-focus sweep.

    Each entry in *passes* is one sweep centred on the best WD from the
    previous pass (or the current WD for pass 0).  A single-pass sweep is
    expressed as one FocusSweepPass in the list.

    Example — coarse/fine two-pass:
        passes = [FocusSweepPass(10, 2e-3), FocusSweepPass(10, 0.2e-3)]

    Example — three-pass:
        passes = [FocusSweepPass(10, 4e-3), FocusSweepPass(10, 0.5e-3), FocusSweepPass(10, 0.05e-3)]
    """
    method: str = "laplacian"       # "laplacian" | "sobel" | "variance" | "tenengrad"
    passes: list = None             # list[FocusSweepPass]; None → single default pass
    probe_resolution: tuple = (768, 512)
    probe_dwell_time: float = 0.5e-6
    reduced_area: 'FibsemRectangle' = None

    def __post_init__(self):
        if self.passes is None:
            self.passes = [FocusSweepPass()]

    def to_dict(self) -> dict:
        d = {
            "method": self.method,
            "passes": [dataclasses.asdict(p) for p in self.passes],
            "probe_resolution": list(self.probe_resolution),
            "probe_dwell_time": self.probe_dwell_time,
            "reduced_area": dataclasses.asdict(self.reduced_area) if self.reduced_area is not None else None,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AutoFocusSettings":
        from fibsem.structures import FibsemRectangle
        d = dict(d)
        passes = [FocusSweepPass(**p) for p in d.pop("passes", [])] or [FocusSweepPass()]
        ra = d.pop("reduced_area", None)
        obj = cls(passes=passes, **d)
        if ra is not None:
            obj.reduced_area = FibsemRectangle(**ra)
        return obj


@dataclass
class AutoFocusIteration:
    """State captured at a single working-distance position in the sweep."""
    working_distance: float
    focus_score: float
    pass_index: int
    image: 'FibsemImage'

    def to_dict(self, index: int) -> dict:
        return {
            "index": index,
            "pass_index": self.pass_index,
            "working_distance": self.working_distance,
            "focus_score": self.focus_score,
            "image": f"iter_{index:02d}.tif",
        }

    @classmethod
    def from_dict(cls, d: dict, result_dir: Path) -> "AutoFocusIteration":
        from fibsem.structures import FibsemImage
        return cls(
            pass_index=d.get("pass_index", 0),
            working_distance=d["working_distance"],
            focus_score=d["focus_score"],
            image=FibsemImage.load(str(result_dir / d["image"])),
        )


@dataclass
class AutoFocusResult:
    """Result of an image-based auto-focus run."""
    image: FibsemImage
    working_distance: float
    initial_working_distance: float
    focus_score: float
    iterations: list[AutoFocusIteration]
    settings: AutoFocusSettings = None

    @property
    def n_iterations(self) -> int:
        return len(self.iterations)

    def plot(self, save_path: str = None) -> None:
        from fibsem.autofunctions.autofocus_plotting import plot_autofocus_result
        plot_autofocus_result(self, save_path=save_path)

    def save(self, path: str = ".", name: str = "autofocus") -> Path:
        """Save the result to a timestamped directory.

        Creates ``<path>/<name>-<timestamp>/`` containing:
          - ``data.json``  — settings and per-iteration working distance + score
          - ``iter_00.tif``, … — probe image per iteration
          - ``best.tif`` — the sharpest probe image
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = Path(path) / f"{name}-{ts}"
        result_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "working_distance": self.working_distance,
            "initial_working_distance": self.initial_working_distance,
            "focus_score": self.focus_score,
            "n_iterations": self.n_iterations,
            "settings": self.settings.to_dict() if self.settings is not None else None,
            "iterations": [it.to_dict(i) for i, it in enumerate(self.iterations)],
        }

        with open(result_dir / "data.json", "w") as f:
            json.dump(data, f, indent=2)

        for i, it in enumerate(self.iterations):
            it.image.save(path=str(result_dir / f"iter_{i:02d}"))

        self.image.save(path=str(result_dir / "best"))

        logger.info("AutoFocus result saved to %s", result_dir)
        return result_dir

    @classmethod
    def load(cls, result_dir: str) -> "AutoFocusResult":
        """Load a result previously saved with :meth:`save`."""
        result_dir = Path(result_dir)

        with open(result_dir / "data.json") as f:
            data = json.load(f)

        iterations = [
            AutoFocusIteration.from_dict(it_data, result_dir)
            for it_data in data["iterations"]
        ]

        from fibsem.structures import FibsemImage
        best_image = FibsemImage.load(str(result_dir / "best.tif"))
        settings_data = data.get("settings")
        settings = AutoFocusSettings.from_dict(settings_data) if settings_data is not None else None

        return cls(
            image=best_image,
            working_distance=data["working_distance"],
            initial_working_distance=data.get("initial_working_distance", data["working_distance"]),
            focus_score=data["focus_score"],
            iterations=iterations,
            settings=settings,
        )


def run_auto_focus(
    microscope: 'FibsemMicroscope',
    beam_type: BeamType = BeamType.ELECTRON,
    hfw: float = 150e-6,
    settings: AutoFocusSettings = None,
) -> AutoFocusResult:
    """Multi-pass image-based auto-focus sweep.

    Each pass sweeps working distance over a range centred on the best WD
    found by the previous pass (or the current WD for pass 0), then scores
    each position with the chosen focus metric.  Passes with a narrower
    step_size progressively refine the focus position.

    Args:
        microscope: FibsemMicroscope instance.
        beam_type: Which beam to focus.
        hfw: Horizontal field width for probe images (metres).
        settings: ``AutoFocusSettings``; defaults constructed if ``None``.

    Returns:
        ``AutoFocusResult`` with the best probe image, winning working
        distance, focus score, and full per-pass iteration history.
    """
    from fibsem.autofunctions.metrics import get_focus_measure_function
    from fibsem.structures import BeamType, ImageSettings

    if beam_type is None:
        beam_type = BeamType.ELECTRON
    if settings is None:
        settings = AutoFocusSettings()

    focus_fn = get_focus_measure_function(settings.method)

    probe_settings = ImageSettings(
        resolution=settings.probe_resolution,
        dwell_time=settings.probe_dwell_time,
        hfw=hfw,
        beam_type=beam_type,
        autocontrast=False,
        autogamma=False,
        save=False,
        reduced_area=settings.reduced_area,
    )

    # warn if any pass has a wider range than the one before it
    for i in range(1, len(settings.passes)):
        prev = settings.passes[i - 1]
        curr = settings.passes[i]
        if curr.n_steps * curr.step_size >= prev.n_steps * prev.step_size:
            logger.warning(
                "Pass %d range (%.2e m) >= pass %d range (%.2e m) — "
                "later passes should be narrower than earlier ones",
                i, curr.n_steps * curr.step_size,
                i - 1, prev.n_steps * prev.step_size,
            )

    initial_wd = microscope.get_working_distance(beam_type)
    centre_wd = initial_wd
    iterations: list[AutoFocusIteration] = []

    try:
        for pass_index, sweep_pass in enumerate(settings.passes):
            half_range = sweep_pass.n_steps / 2 * sweep_pass.step_size
            wds = np.linspace(centre_wd - half_range, centre_wd + half_range, sweep_pass.n_steps + 1)

            pass_scores = []
            for i, wd in enumerate(wds):
                microscope.set_working_distance(wd, beam_type)
                probe = microscope.acquire_image(image_settings=probe_settings)
                score = float(np.mean(focus_fn(probe.data.astype(np.float32))))
                iterations.append(AutoFocusIteration(
                    pass_index=pass_index,
                    working_distance=float(wd),
                    focus_score=score,
                    image=probe,
                ))
                pass_scores.append(score)
                logger.debug(
                    "AutoFocus pass %d step %d/%d: wd=%.4e score=%.4f",
                    pass_index, i, sweep_pass.n_steps, wd, score,
                )

            best_in_pass = int(np.argmax(pass_scores))
            centre_wd = float(wds[best_in_pass])
            logger.info(
                "AutoFocus pass %d complete: best WD=%.4e score=%.4f",
                pass_index, centre_wd, pass_scores[best_in_pass],
            )

    except Exception:
        logger.exception("AutoFocus failed — restoring initial WD %.4e", initial_wd)
        microscope.set_working_distance(initial_wd, beam_type)
        raise

    best_idx = int(np.argmax([it.focus_score for it in iterations]))
    best = iterations[best_idx]

    microscope.set_working_distance(best.working_distance, beam_type)
    logger.info(
        "AutoFocus complete: best WD=%.4e score=%.4f (%d passes, %d steps total)",
        best.working_distance, best.focus_score, len(settings.passes), len(iterations),
    )

    return AutoFocusResult(
        image=best.image,
        working_distance=best.working_distance,
        initial_working_distance=initial_wd,
        focus_score=best.focus_score,
        iterations=iterations,
        settings=settings,
    )
