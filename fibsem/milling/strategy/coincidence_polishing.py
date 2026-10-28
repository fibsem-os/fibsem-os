"""Coincidence Polishing: one polishing pass toward a fluorescent target.

A pass mills the stage exactly as it is given, then looks: a FIB image of the
band, the objective put back where it should be, a reflection stack and a
target-channel stack, and the measurements an operator or an agent judges the
pass by. Where to mill, how hard, and when to stop are the caller's to decide;
the strategy decides nothing and answers with what it measured.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import threading
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from fibsem import acquire
from fibsem.cancellation import raise_if_cancelled
from fibsem.conversions import image_to_microscope_image_coordinates
from fibsem.fm.acquisition import acquire_z_stack
from fibsem.fm.structures import ChannelSettings, FluorescenceImage, ZParameters
from fibsem.imaging import localise
from fibsem.imaging.analysis import (
    Planes,
    intensity_figure,
    load_planes,
    measure_intensity,
)
from fibsem.imaging.marks import rgb_of
from fibsem.imaging.thickness import LamellaThicknessResult, measure_lamella_band
from fibsem.microscope import FibsemMicroscope
from fibsem.milling.base import (
    FibsemMillingStage,
    MillingStrategy,
    MillingStrategyConfig,
    MillingStrategyResult,
)
from fibsem.milling.strategy.standard import StandardMillingStrategy
from fibsem.structures.image import FibsemImage
from fibsem.structures.params import Display, param
from fibsem.structures.settings import BeamType, ImageSettings, Point

FACES = ("top", "bottom")
PASS_FILE = "pass.json"


@dataclass
class CoincidencePolishingConfig(MillingStrategyConfig):
    """One polishing pass: the face it mills, and how it looks afterwards. The
    pattern and the milling settings are the stage's own."""

    face: str = param(
        "top",
        choices=list(FACES),
        doc="The face this pass mills, top or bottom. The pass is recorded under it.",
        display=Display("Face"),
    )
    acquire_fib_image: bool = param(
        True,
        doc="Take a FIB image after the mill and measure the band on it.",
        display=Display("Acquire FIB Image"),
    )
    fib_image: Optional[ImageSettings] = param(
        None,
        doc="The FIB image's settings (resolution, dwell, field of view). None takes it with the ion beam as it is set.",
        display=Display("FIB Image", advanced=True),
    )
    band_centre_px: Optional[List[float]] = param(
        None,
        doc="Where the band is measured, [x, y] in the FIB image's pixels; None is the image's centre.",
        display=Display("Band Centre", advanced=True),
    )
    band_window: List[float] = param(
        default_factory=lambda: [10e-6, 4e-6],
        unit="m",
        doc="The window the band is measured in: [width along the band, height across it].",
        display=Display("Band Window", advanced=True),
    )
    objective_position: Optional[float] = param(
        None,
        unit="m",
        doc="Move the objective here before the stacks, as it drifts while milling; None leaves it.",
        display=Display("Objective Position", scale=1e3, unit="mm"),
    )
    acquire_reflection_stack: bool = param(
        True,
        doc="Take the reflection stack, which shows the lamella's faces and quality.",
        display=Display("Acquire Reflection Stack"),
    )
    reflection_channel: ChannelSettings = param(
        default_factory=lambda: ChannelSettings(name="reflection"),
        doc="The reflection channel.",
        display=Display("Reflection Channel", advanced=True),
    )
    target_channel: Optional[ChannelSettings] = param(
        None,
        doc="The channel the target is seen in; None takes no target stack.",
        display=Display("Target Channel", advanced=True),
    )
    zparams: ZParameters = param(
        default_factory=lambda: ZParameters(zmin=-1e-6, zmax=1e-6, zstep=0.5e-6),
        doc="Both stacks' planes, about the objective's position.",
        display=Display("Z Stack", advanced=True),
    )
    targets: List[List[float]] = param(
        default_factory=list,
        doc="The targets, [x, y] in the target stack's pixels, as the last pass refitted them.",
        display=Display("Targets", hidden=True),
    )
    control: Optional[List[float]] = param(
        None,
        doc="A control punctum away from the mill, [x, y] in the target stack's pixels; the targets' signal is read against it.",
        display=Display("Control", hidden=True),
    )
    radius_px: float = param(
        3.0,
        doc="The radius a target's signal is read in, in pixels; its background is the ring 2 to 3 radii out.",
        display=Display("Signal Radius", advanced=True),
    )
    fit_cutout_px: int = param(
        10,
        doc="Half the size of the box each target is refitted in, in pixels.",
        display=Display("Fit Cutout", advanced=True),
    )
    baseline: Optional[str] = param(
        None,
        doc=f"An earlier pass's {PASS_FILE}; each target's change is measured from it.",
        display=Display("Baseline", hidden=True),
    )


class CoincidencePolishingStrategy(MillingStrategy[CoincidencePolishingConfig]):
    """Mill one pass as given, then image and measure the band and the targets."""

    name: str = "Coincidence Polishing"
    fullname: str = "Coincidence Polishing (one pass)"
    config_class = CoincidencePolishingConfig
    # Its settings are a pass's, chosen pass by pass: not a protocol's stage.
    selectable: bool = False

    def run(
        self,
        microscope: FibsemMicroscope,
        stage: FibsemMillingStage,
        stop_event: Optional[threading.Event] = None,
    ) -> MillingStrategyResult:
        """Mill, image, measure. Says what it measured, where it saved the
        images and ``pass.json``, and draws the pass."""
        config = self.config
        if config.face not in FACES:
            raise ValueError(f"face is top or bottom, not {config.face!r}")
        if microscope.fm is None and (
            config.acquire_reflection_stack
            or config.target_channel is not None
            or config.objective_position is not None
        ):
            raise ValueError(
                "Coincidence Polishing needs a fluorescence module for its stacks"
            )
        folder = _pass_folder(stage)
        logging.info(f"{self.name}: {stage.name} ({config.face}) into {folder}")

        StandardMillingStrategy().run(microscope, stage, stop_event)
        raise_if_cancelled(stop_event)

        record: Dict[str, Any] = {
            "stage": stage.name,
            "face": config.face,
            "pattern": stage.pattern.name,
            "milling_current": stage.milling.milling_current,
            "files": {},
        }
        facts: Dict[str, Any] = {"Face": config.face}

        fib = None
        if config.acquire_fib_image:
            fib = _fib_image(microscope, config.fib_image, folder)
            record["files"]["fib_image"] = fib.filepath
            facts["FIB image"] = str(fib.filepath)
            raise_if_cancelled(stop_event)

        if config.objective_position is not None:
            before = microscope.fm.objective.position
            microscope.fm.objective.move_absolute(config.objective_position)
            after = microscope.fm.objective.position
            record["objective"] = {"before": before, "after": after}
            facts["Objective moved (nm)"] = round((after - before) * 1e9, 1)

        stacks: Dict[str, str] = {}
        if config.acquire_reflection_stack:
            stacks["reflection"] = _stack(
                microscope, config.reflection_channel, config, folder, stop_event
            )
            facts["Reflection stack"] = stacks["reflection"]
        if config.target_channel is not None:
            stacks["target"] = _stack(
                microscope, config.target_channel, config, folder, stop_event
            )
            facts["Target stack"] = stacks["target"]
        record["files"].update(stacks)

        band = None
        if fib is not None:
            band = _measure_band(fib, config, folder, record, facts)
        planes, intensity = None, None
        if "target" in stacks and config.targets:
            planes = load_planes(stacks["target"])
            intensity = _measure_targets(planes, config, record, facts)

        pass_file = os.path.join(folder, PASS_FILE)
        with open(pass_file, "w") as f:
            json.dump(record, f, indent=2, default=_plain)
        record["files"]["pass"] = pass_file
        facts["Pass data"] = pass_file

        figure = _figure(band, planes, intensity, stacks.get("reflection"), folder)
        return MillingStrategyResult(facts=facts, figure=figure)


def _pass_folder(stage: FibsemMillingStage) -> str:
    """A new folder for this pass, under the stage's output folder."""
    root = stage.output_dir or stage.imaging.path or os.getcwd()
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", stage.name).strip("-") or "pass"
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    folder = os.path.join(str(root), f"polishing-{slug}-{stamp}")
    os.makedirs(folder, exist_ok=True)
    return folder


def _fib_image(
    microscope: FibsemMicroscope, settings: Optional[ImageSettings], folder: str
) -> FibsemImage:
    if settings is None:
        return acquire.acquire_current_image(
            microscope, BeamType.ION, path=os.path.join(folder, "fib.tif")
        )
    settings = deepcopy(settings)
    settings.beam_type = BeamType.ION
    settings.save = True
    settings.path = folder
    settings.filename = "fib"
    return acquire.acquire_image(microscope, settings)


def _stack(
    microscope: FibsemMicroscope,
    channel: ChannelSettings,
    config: CoincidencePolishingConfig,
    folder: str,
    stop_event: Optional[threading.Event],
) -> str:
    image: Optional[FluorescenceImage] = acquire_z_stack(
        microscope.fm, channel, config.zparams, stop_event
    )
    raise_if_cancelled(stop_event)
    if image is None:
        raise RuntimeError(f"The {channel.name} stack came back empty")
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", channel.name).strip("-") or "channel"
    return image.save(os.path.join(folder, f"{slug}-stack.ome.tiff"))


def _measure_band(
    fib: FibsemImage,
    config: CoincidencePolishingConfig,
    folder: str,
    record: Dict[str, Any],
    facts: Dict[str, Any],
) -> Optional[LamellaThicknessResult]:
    """The band's edges and width on the FIB image. The edges are given as the
    microscope's image coordinates (µm from the image centre, +y up), the frame
    a pattern's point is placed in."""
    pixel_size = fib.metadata.pixel_size.x
    centre = (
        image_to_microscope_image_coordinates(
            Point(*config.band_centre_px), fib.data, pixel_size, subpixel_precision=True
        )
        if config.band_centre_px is not None
        else Point(0, 0)
    )
    try:
        band = measure_lamella_band(fib, centre, tuple(config.band_window))
    except ValueError as exc:
        record["band"] = {"error": str(exc)}
        facts["Band"] = f"not measured: {exc}"
        return None
    answer = band.to_dict()
    record["band"] = answer
    record["files"]["band"] = str(band.save(folder, "thickness"))
    if answer.get("thickness_nm") is None:
        facts["Band"] = "not found"
        return band
    ref_x = band.reference[0]
    for name, edges in (("Top", band.top), ("Bottom", band.bottom)):
        y_px = float(np.nanmedian(edges[band.measured]))
        edge = image_to_microscope_image_coordinates(
            Point(ref_x, y_px), fib.data, pixel_size, subpixel_precision=True
        )
        facts[f"{name} edge y (µm)"] = round(edge.y * 1e6, 3)
        record["band"][f"{name.lower()}_edge_y_m"] = edge.y
    facts["Band FWHM (nm)"] = answer["thickness_nm"]["median"]
    facts["Band centre offset (nm)"] = answer.get("centre_offset_nm")
    return band


def _measure_targets(
    planes: Planes,
    config: CoincidencePolishingConfig,
    record: Dict[str, Any],
    facts: Dict[str, Any],
) -> Dict[str, Any]:
    """Each target refitted in the stack, then its signal read, against the
    control when there is one, and its change from the baseline pass."""
    _, stack = planes.layer(None)
    fits = []
    for i, (x, y) in enumerate(config.targets, start=1):
        try:
            xr, yr, zr, _ = localise.target_fitting_fluorescence(
                stack,
                x,
                y,
                planes.planes // 2,
                cutout=config.fit_cutout_px,
                use_xy_fitting=True,
            )
            fit = {"point": [float(xr), float(yr)], "z": float(zr), "fitted": True}
        except Exception as exc:  # noqa: BLE001 - a failed fit is a measurement
            fit = {"point": [float(x), float(y)], "fitted": False, "error": str(exc)}
        fit["moved_px"] = round(
            float(np.hypot(fit["point"][0] - x, fit["point"][1] - y)), 2
        )
        fits.append(fit)
        facts[f"Target {i} fit"] = "ok" if fit["fitted"] else "failed"
        facts[f"Target {i} moved (px)"] = fit["moved_px"]

    points = [fit["point"] for fit in fits]
    reference = None
    if config.control is not None:
        points.append([float(v) for v in config.control])
        reference = len(points) - 1
    intensity = measure_intensity(
        planes, points, radius_px=config.radius_px, reference=reference
    )
    baseline = _baseline(config.baseline)
    rows = intensity["points"]
    for i, (fit, row) in enumerate(zip(fits, rows), start=1):
        fit["signal"] = row["signal"]
        fit["ratio"] = row.get("ratio")
        facts[f"Target {i} signal"] = row["signal"]
        if row.get("ratio") is not None:
            facts[f"Target {i} ratio to control"] = row["ratio"]
        level = row.get("ratio") if reference is not None else row["signal"]
        was = baseline[i - 1] if baseline and i - 1 < len(baseline) else None
        if level is not None and was:
            change = round(100.0 * (level - was) / abs(was), 1)
            fit["change_from_baseline_pct"] = change
            facts[f"Target {i} change from baseline (%)"] = change
    if reference is not None:
        facts["Control signal"] = rows[reference]["signal"]
    record["targets"] = fits
    record["control"] = config.control
    record["intensity"] = intensity
    return intensity


def _baseline(path: Optional[str]) -> Optional[List[Optional[float]]]:
    """The level each target had in an earlier pass: its ratio to the control,
    or its signal when that pass read none."""
    if not path:
        return None
    try:
        with open(path) as f:
            targets = json.load(f).get("targets") or []
    except (OSError, ValueError) as exc:
        logging.warning(f"The baseline {path} could not be read: {exc}")
        return None
    return [
        t.get("ratio") if t.get("ratio") is not None else t.get("signal")
        for t in targets
    ]


def _figure(
    band: Optional[LamellaThicknessResult],
    planes: Optional[Planes],
    intensity: Optional[Dict[str, Any]],
    reflection: Optional[str],
    folder: str,
) -> Optional[str]:
    """The pass on one image: the band with its edges, each target's strip,
    and the reflection stack's sharpest plane."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from PIL import Image

    parts: List[np.ndarray] = []
    if band is not None:
        figure = band.plot()
        FigureCanvasAgg(figure)
        parts.append(rgb_of(figure))
    if planes is not None and intensity is not None:
        parts.append(intensity_figure(planes, intensity))
    if reflection is not None:
        _, stack = load_planes(reflection).layer(None)
        sharpness = [float(np.var(plane)) for plane in stack]
        best = int(np.argmax(sharpness))
        figure = Figure(figsize=(6, 4.2), facecolor="#16181d")
        FigureCanvasAgg(figure)
        axes = figure.add_subplot(1, 1, 1)
        axes.imshow(stack[best], cmap="gray")
        axes.set_title(
            f"Reflection, plane {best} of {len(stack)}", color="#d8dce4", fontsize=10
        )
        axes.axis("off")
        parts.append(rgb_of(figure))
    if not parts:
        return None
    width = max(part.shape[1] for part in parts)
    padded = [
        np.pad(
            part,
            ((0, 0), (0, width - part.shape[1]), (0, 0)),
            constant_values=0x16,
        )
        for part in parts
    ]
    path = os.path.join(folder, "pass.png")
    Image.fromarray(np.concatenate(padded, axis=0)).save(path)
    return path


def _plain(value: Any) -> Any:
    """For json: numpy numbers as plain numbers, anything else as text."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
