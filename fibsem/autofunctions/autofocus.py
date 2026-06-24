"""Image-based auto-focus for FIB-SEM images."""
from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from fibsem.structures import BeamType

import numpy as np
if TYPE_CHECKING:
    from fibsem.structures import BeamType, FibsemRectangle, ImageSettings
    from fibsem.microscope import FibsemMicroscope

logger = logging.getLogger(__name__)


class FocusMethod(str, Enum):
    """Focus measurement methods for autofocus algorithms."""
    LAPLACIAN = "laplacian"
    SOBEL = "sobel"
    VARIANCE = "variance"
    TENENGRAD = "tenengrad"


@dataclass
class FocusSweepPass:
    """A single pass in a multi-pass focus sweep.

    A pass sweeps ``search_range`` metres (±range/2) about its centre, sampling
    one image every ``step_size`` metres.
    """
    search_range: float = 5e-3      # metres (total span; positions cover ±range/2)
    step_size: float = 0.5e-3       # metres
    enabled: bool = True

    @property
    def n_steps(self) -> int:
        """Number of steps spanning the range (derived from range / step)."""
        if self.step_size <= 0:
            return 1
        return max(1, round(self.search_range / self.step_size))


# default coarse/fine passes (FM scale); FIB-SEM callers pass explicit passes
def _default_passes() -> "list[FocusSweepPass]":
    return [
        FocusSweepPass(search_range=50e-6, step_size=5e-6),   # coarse
        FocusSweepPass(search_range=10e-6, step_size=1e-6),   # fine
    ]


def _sweep_pass_from_dict(p: dict) -> FocusSweepPass:
    """Build a FocusSweepPass from a serialized dict, tolerating the legacy
    ``n_steps`` field (converted to ``search_range = n_steps * step_size``)
    and ignoring unknown keys."""
    p = dict(p)
    step_size = p.get("step_size", 0.5e-3)
    if "search_range" in p:
        search_range = p["search_range"]
    elif "n_steps" in p:
        search_range = p["n_steps"] * step_size
    else:
        search_range = 5e-3
    return FocusSweepPass(
        search_range=search_range,
        step_size=step_size,
        enabled=p.get("enabled", True),
    )


@dataclass
class AutoFocusSettings:
    """Parameters controlling an image-based auto-focus sweep.

    The sweep is expressed as a list of *passes*; each pass is one sweep centred
    on the best position found by the previous pass (or the starting position
    for pass 0). A single-pass sweep is just one ``FocusSweepPass``. Use the
    ``from_coarse_fine`` factory to build a two-pass (coarse + fine) config.

    Domain-specific fields are ignored by the domain that does not use them:
    FM uses ``channel_name``; FIB-SEM uses ``probe_resolution`` /
    ``probe_dwell_time`` / ``use_autocontrast``.
    """
    method: "FocusMethod" = FocusMethod.TENENGRAD
    passes: list = None             # list[FocusSweepPass]; None → default coarse/fine
    probe_resolution: tuple = (768, 512)
    probe_dwell_time: float = 0.5e-6
    reduced_area: 'FibsemRectangle' = None
    use_autocontrast: bool = True
    channel_name: Optional[str] = None

    def __post_init__(self):
        if self.passes is None:
            self.passes = _default_passes()
        if not isinstance(self.method, FocusMethod):
            self.method = FocusMethod(self.method)

    @property
    def enabled(self) -> bool:
        """True if any pass is enabled (i.e. autofocus should run)."""
        return any(p.enabled for p in self.passes)

    @classmethod
    def from_coarse_fine(
        cls,
        coarse_range: float = 50e-6,
        coarse_step: float = 5e-6,
        coarse_enabled: bool = True,
        fine_range: float = 10e-6,
        fine_step: float = 1e-6,
        fine_enabled: bool = True,
        method: "FocusMethod" = FocusMethod.TENENGRAD,
        channel_name: Optional[str] = None,
        **kwargs,
    ) -> "AutoFocusSettings":
        """Build a two-pass (coarse + fine) AutoFocusSettings."""
        passes = [
            FocusSweepPass(search_range=coarse_range, step_size=coarse_step, enabled=coarse_enabled),
            FocusSweepPass(search_range=fine_range, step_size=fine_step, enabled=fine_enabled),
        ]
        return cls(method=method, passes=passes, channel_name=channel_name, **kwargs)

    def to_dict(self) -> dict:
        return {
            "method": self.method.value,
            "passes": [dataclasses.asdict(p) for p in self.passes],
            "probe_resolution": list(self.probe_resolution),
            "probe_dwell_time": self.probe_dwell_time,
            "reduced_area": dataclasses.asdict(self.reduced_area) if self.reduced_area is not None else None,
            "use_autocontrast": self.use_autocontrast,
            "channel_name": self.channel_name,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AutoFocusSettings":
        from fibsem.structures import FibsemRectangle
        d = dict(d)

        # Legacy coarse/fine schema → passes
        if "passes" not in d and ("coarse_range" in d or "fine_range" in d):
            return cls.from_coarse_fine(
                coarse_range=d.get("coarse_range", 50e-6),
                coarse_step=d.get("coarse_step", 5e-6),
                coarse_enabled=d.get("coarse_enabled", True),
                fine_range=d.get("fine_range", 10e-6),
                fine_step=d.get("fine_step", 1e-6),
                fine_enabled=d.get("fine_enabled", True),
                method=FocusMethod(d.get("method", FocusMethod.TENENGRAD.value)),
                channel_name=d.get("channel_name"),
            )

        passes = [_sweep_pass_from_dict(p) for p in d.pop("passes", [])] or _default_passes()
        ra = d.pop("reduced_area", None)
        method = d.pop("method", FocusMethod.TENENGRAD.value)
        obj = cls(method=FocusMethod(method), passes=passes, **d)
        if ra is not None:
            obj.reduced_area = FibsemRectangle(**ra)
        return obj


@dataclass
class AutoFocusIteration:
    """State captured at a single working-distance position in the sweep."""
    working_distance: float
    focus_score: float
    pass_index: int
    image: np.ndarray

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
        from fibsem.structures import load_tiff
        return cls(
            pass_index=d.get("pass_index", 0),
            working_distance=d["working_distance"],
            focus_score=d["focus_score"],
            image=load_tiff(result_dir / d["image"]),
        )


@dataclass
class AutoFocusResult:
    """Result of an image-based auto-focus run."""
    image: np.ndarray
    working_distance: float
    initial_working_distance: float
    focus_score: float
    iterations: list[AutoFocusIteration]
    settings: AutoFocusSettings = None
    method: Optional[str] = None

    @property
    def n_iterations(self) -> int:
        return len(self.iterations)

    def plot(self, save_path: str = "autofocus.png") -> None:
        from fibsem.autofunctions.plotting import plot_autofocus_result
        plot_autofocus_result(self, save_path=save_path)

    def save(self, path: str = ".", name: str = "autofocus", save_plot: bool = True) -> Path:
        """Save the result to ``<path>/<name>/``.

        The directory contains:
          - ``data.json``  — settings and per-iteration working distance + score
          - ``iter_00.tif``, … — probe image per iteration
          - ``best.tif`` — the sharpest probe image
          - ``plot.png`` — diagnostic plot (when ``save_plot`` is True)
        """
        result_dir = Path(path) / name
        result_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "working_distance": self.working_distance,
            "initial_working_distance": self.initial_working_distance,
            "focus_score": self.focus_score,
            "n_iterations": self.n_iterations,
            "method": self.method,
            "settings": self.settings.to_dict() if self.settings is not None else None,
            "iterations": [it.to_dict(i) for i, it in enumerate(self.iterations)],
        }

        with open(result_dir / "data.json", "w") as f:
            json.dump(data, f, indent=2)

        from fibsem.structures import save_tiff
        for i, it in enumerate(self.iterations):
            save_tiff(it.image, result_dir / f"iter_{i:02d}.tif")

        save_tiff(self.image, result_dir / "best.tif")

        if save_plot:
            self.plot(save_path=str(result_dir / "plot.png"))

        logger.info("AutoFocus result saved to %s", result_dir)
        return result_dir

    @classmethod
    def load(cls, result_dir: str) -> "AutoFocusResult":
        """Load a result previously saved with :meth:`save`."""
        from fibsem.structures import load_tiff
        result_dir = Path(result_dir)

        with open(result_dir / "data.json") as f:
            data = json.load(f)

        iterations = [
            AutoFocusIteration.from_dict(it_data, result_dir)
            for it_data in data["iterations"]
        ]

        best_image = load_tiff(result_dir / "best.tif")
        settings_data = data.get("settings")
        settings = AutoFocusSettings.from_dict(settings_data) if settings_data is not None else None

        return cls(
            image=best_image,
            working_distance=data["working_distance"],
            initial_working_distance=data.get("initial_working_distance", data["working_distance"]),
            focus_score=data["focus_score"],
            iterations=iterations,
            settings=settings,
            method=data.get("method"),
        )


def _run_sweep(
    microscope: "FibsemMicroscope",
    probe_settings: "ImageSettings",
    focus_fn: "Callable[[np.ndarray], np.ndarray]",
    centre_wd: float,
    sweep_pass: FocusSweepPass,
    pass_index: int,
    beam_type: BeamType,
) -> "tuple[list[AutoFocusIteration], float]":
    """Acquire images across a WD range and return iterations + best WD."""
    half_range = sweep_pass.search_range / 2
    wds = np.linspace(centre_wd - half_range, centre_wd + half_range, sweep_pass.n_steps + 1)

    iterations = []
    for i, wd in enumerate(wds):
        microscope.set_working_distance(wd, beam_type)
        probe = microscope.acquire_image(image_settings=probe_settings)
        score = float(np.mean(focus_fn(probe.filtered_data.astype(np.float32))))
        iterations.append(AutoFocusIteration(
            pass_index=pass_index,
            working_distance=float(wd),
            focus_score=score,
            image=probe.data,
        ))
        logger.debug("AutoFocus pass %d step %d/%d: wd=%.4e score=%.4f",
                     pass_index, i, sweep_pass.n_steps, wd, score)

    best_idx = int(np.argmax([it.focus_score for it in iterations]))
    best_wd = iterations[best_idx].working_distance
    logger.info("AutoFocus pass %d complete: best WD=%.4e score=%.4f",
                pass_index, best_wd, iterations[best_idx].focus_score)
    return iterations, best_wd


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

    focus_fn = get_focus_measure_function(settings.method.value)

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

    active_passes = [p for p in settings.passes if p.enabled]
    if not active_passes:
        raise ValueError("AutoFocusSettings has no enabled passes")

    # warn if any pass has a wider range than the one before it
    for i in range(1, len(active_passes)):
        prev = active_passes[i - 1]
        curr = active_passes[i]
        if curr.search_range >= prev.search_range:
            logger.warning(
                "Pass %d range (%.2e m) >= pass %d range (%.2e m) — "
                "later passes should be narrower than earlier ones",
                i, curr.search_range, i - 1, prev.search_range,
            )

    initial_wd = microscope.get_working_distance(beam_type)
    centre_wd = initial_wd
    iterations: list[AutoFocusIteration] = []

    if settings.use_autocontrast:
        microscope.autocontrast(beam_type, settings.reduced_area)

    try:
        for pass_index, sweep_pass in enumerate(active_passes):
            pass_iters, centre_wd = _run_sweep(
                microscope, probe_settings, focus_fn,
                centre_wd, sweep_pass, pass_index, beam_type,
            )
            iterations.extend(pass_iters)
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
        method=settings.method.value,
    )
