"""The Demo's services.

`DemoMilling` mills on the simulator's ``milling_system`` (``DemoSimulator``, the
Demo's connection): the patterns, the milling state and the application files. The beams
change only through the microscope's beam methods, so through the beam devices.

Nothing on the Demo ends a mill but the clock, so a ``run`` is timed by the estimate
it starts with, on simulated time: each wait is ``sim_sleep``, which the test suite
turns off, and the time counts as it would have passed. A ``start`` alone runs until
something stops it. Each pattern takes a few seconds, or, with
``FIBSEM_SIM_MILL_SCALE`` set, that fraction of the sputter-rate estimate for what
is drawn, so a stop can be tried mid-mill (FIB-1496).

`DemoSpotBurn` is the shared point-by-point burn, also on simulated time, and
`DemoCoating` the shared coating, whose valve stays open for simulated time.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from fibsem._timing import sim_sleep
from fibsem.devices.core import ParameterMetadata
from fibsem.drivers.demo.simulator import (
    SIM_ASYNC_MILLING_EXTRA_TIME,
    SIMULATOR_SCAN_DIRECTIONS,
)
from fibsem.milling.base import estimate_volume_time
from fibsem.milling.progress import MillingProgress
from fibsem.milling.rtm import RTM_MODES
from fibsem.services.coating import (
    ARCTIS_DEPOSITION_POSITION,
    Coating,
    bind_coating,
)
from fibsem.services.milling import BeamMilling, bind_milling, progress_update
from fibsem.services.spot_burn import SpotBurn, bind_spot_burn
from fibsem.structures import (
    ACTIVE_MILLING_STATES,
    FibsemCircleSettings,
    FibsemLineSettings,
    FibsemMillingSettings,
    FibsemPatternSettings,
    FibsemPolygonSettings,
    MillingState,
)

if TYPE_CHECKING:
    from fibsem.drivers.demo.microscope import DemoMicroscope
    from fibsem.drivers.demo.simulator import DemoSimulator

# simulated seconds each pattern takes
_PATTERN_TIME = 5
# set to mill for this fraction of the sputter-rate estimate instead
SIM_MILL_SCALE = "FIBSEM_SIM_MILL_SCALE"


class DemoMilling(BeamMilling):
    """The Demo's milling, on the simulator's ``milling_system``."""

    parent: DemoMicroscope

    scan_directions = tuple(SIMULATOR_SCAN_DIRECTIONS)
    rtm_modes = RTM_MODES

    def __init__(self, connection: DemoSimulator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.connection = connection
        # The run in progress, which the clock ends: its length and the simulated
        # time it has run for. None outside a run (`start` alone runs until stopped).
        self._run_total: Optional[float] = None
        self._run_elapsed = 0.0
        # Whether the mill running now was started by `start`, which never ends on
        # its own here; its estimate adds `SIM_ASYNC_MILLING_EXTRA_TIME`.
        self._open_ended = False
        # The RTM mode reading now, None when RTM is off.
        self._rtm_mode: Optional[str] = None
        self._rtm_rng = np.random.default_rng(0)

    def _setting_metadata(self, name: str) -> ParameterMetadata:
        if name == "application_file":
            files = self.connection.milling_system.application_files
            return ParameterMetadata(choices=tuple(files))
        return super()._setting_metadata(name)

    def _select(self, settings: FibsemMillingSettings) -> None:
        system = self.connection.milling_system
        self.connection.set_channel(settings.milling_channel)
        system.default_beam_type = settings.milling_channel
        system.default_application_file = settings.application_file
        system.patterning_mode = settings.patterning_mode

    def _add(self, pattern: FibsemPatternSettings) -> None:
        logging.debug({"msg": "draw_pattern", "pattern_settings": pattern.to_dict()})
        self.connection.milling_system.patterns.append(pattern)

    # The Demo draws every pattern type, by adding it to the simulated list.
    _draw_rectangle = _draw_line = _draw_circle = _draw_bitmap = _draw_polygon = _add

    def read_state(self) -> MillingState:
        return self.connection.milling_system.state

    def _set_state(self, state: MillingState) -> None:
        self.connection.milling_system.state = state

    def _start(self) -> None:
        if self._run_total is not None:
            # a timed run, which ends by itself: not an open-ended start
            self._open_ended = False
            self._set_state(MillingState.RUNNING)
            return
        if self.read_state() is MillingState.IDLE:
            self._set_state(MillingState.RUNNING)
            self._open_ended = True
            logging.info("Milling started.")

    def _before_run(self) -> None:
        # Into the simulated sample from the start, so the scene shows the mill
        # however the run ends.
        microscope = self.parent
        current = microscope.get_beam_current(microscope.milling_channel)
        self.connection.mill_into_sample_scene(current)
        # a stale `start` mustn't stretch a timed run's estimate
        self._open_ended = False
        self._run_total = self._estimate()
        self._run_elapsed = 0.0

    def _poll(self) -> MillingProgress:
        state = self.read_state()
        if self._run_total is None:
            return progress_update(state=state)
        if state is MillingState.RUNNING and self._run_elapsed >= self._run_total:
            self._set_state(MillingState.IDLE)
            state = MillingState.IDLE
        return progress_update(
            state=state,
            total=self._run_total,
            remaining=max(0.0, self._run_total - self._run_elapsed),
        )

    def _wait(self, seconds: float) -> None:
        sim_sleep(seconds)
        if self.read_state() is MillingState.RUNNING:
            self._run_elapsed += seconds

    def _after_run(self) -> None:
        self._run_total = None
        self._set_state(MillingState.IDLE)

    def _stop_patterning(self) -> None:
        self._set_state(MillingState.IDLE)
        self._open_ended = False

    def _pause_patterning(self) -> None:
        self._set_state(MillingState.PAUSED)

    def _resume_patterning(self) -> None:
        self._set_state(MillingState.RUNNING)

    def _estimate(self) -> float:
        estimate = _PATTERN_TIME * len(self.connection.milling_system.patterns)
        scale = _mill_scale()
        if scale:
            microscope = self.parent
            current = microscope.get_beam_current(microscope.milling_channel)
            volume = sum(p.volume for p in self.drawn_patterns)
            estimate = scale * estimate_volume_time(volume, current)
        if self._open_ended and self.read_state() in ACTIVE_MILLING_STATES:
            estimate += SIM_ASYNC_MILLING_EXTRA_TIME
        return estimate

    def _clear(self) -> None:
        self.connection.milling_system.patterns = []

    def _start_rtm(self, mode: str) -> None:
        self._rtm_mode = mode

    def _read_rtm(self):
        """Points over each monitored pattern's box, in the milling beam's field,
        whose values fall as the run mills."""
        if self._rtm_mode is None:
            return None
        beam = self.beam(self.parent.milling_channel)
        width, height = _resolution(beam)
        pixel = float(beam.hfw.get_value()) / width
        side = _RTM_SIDE[self._rtm_mode]
        milled = self._run_elapsed / self._run_total if self._run_total else 0.0
        values, positions = {}, {}
        for index, pattern in self.rtm_patterns().items():
            cx, cy, w, h = _box(pattern)
            xs = np.linspace(cx - w / 2, cx + w / 2, side)
            ys = np.linspace(cy - h / 2, cy + h / 2, side)
            x, y = np.meshgrid(xs, ys)
            points = np.stack(
                [width / 2 + x.ravel() / pixel, height / 2 - y.ravel() / pixel], axis=1
            )
            positions[index] = np.round(points).astype(int)
            noise = self._rtm_rng.normal(0.0, 5.0, side * side)
            values[index] = 200.0 * (1.0 - 0.7 * min(milled, 1.0)) + noise
        return (width, height), values, positions

    def _stop_rtm(self) -> None:
        self._rtm_mode = None


# Points along each side of a pattern's box, per RTM mode.
_RTM_SIDE = {"low": 8, "high": 24}


def _resolution(beam) -> Tuple[int, int]:
    try:
        width, height = beam.resolution.get_value()
        return int(width), int(height)
    except Exception:  # noqa: BLE001 - a beam without a resolution
        return 1536, 1024


def _box(pattern: FibsemPatternSettings) -> Tuple[float, float, float, float]:
    """The centre and size of *pattern*'s bounding box, in metres."""
    if isinstance(pattern, FibsemCircleSettings):
        return (
            pattern.centre_x,
            pattern.centre_y,
            2 * pattern.radius,
            2 * pattern.radius,
        )
    if isinstance(pattern, FibsemLineSettings):
        xs, ys = (pattern.start_x, pattern.end_x), (pattern.start_y, pattern.end_y)
    elif isinstance(pattern, FibsemPolygonSettings):
        vertices = np.asarray(pattern.vertices, dtype=float).reshape(-1, 2)
        xs, ys = vertices[:, 0], vertices[:, 1]
    else:  # a rectangle or a bitmap
        return pattern.centre_x, pattern.centre_y, pattern.width, pattern.height
    return (
        (min(xs) + max(xs)) / 2,
        (min(ys) + max(ys)) / 2,
        max(xs) - min(xs),
        max(ys) - min(ys),
    )


def _mill_scale() -> float:
    try:
        return max(0.0, float(os.environ.get(SIM_MILL_SCALE) or 0))
    except ValueError:
        return 0.0


def bind_demo_milling(microscope: DemoMicroscope) -> Optional[DemoMilling]:
    """Build ``milling`` for a Demo microscope whose beams are built."""
    return bind_milling(DemoMilling, microscope, connection=microscope._connection)


class DemoSpotBurn(SpotBurn):
    """The Demo's spot burn: the point-by-point burn on the simulated beams, where
    unblanking a parked beam burns the spot into the sample scene. Its waits are
    simulated time."""

    def _wait(self, seconds: float) -> None:
        sim_sleep(seconds)


def bind_demo_spot_burn(microscope: DemoMicroscope) -> Optional[DemoSpotBurn]:
    """Build ``spot_burn`` for a Demo microscope whose beams are built."""
    return bind_spot_burn(DemoSpotBurn, microscope)


class DemoCoating(Coating):
    """The Demo's coating: the shared one, its valve open for simulated time."""

    def _wait(self, seconds: float) -> None:
        sim_sleep(seconds)


def bind_demo_coating(microscope: DemoMicroscope) -> Optional[DemoCoating]:
    """Build ``coating`` for a Demo microscope whose devices are built. A simulated
    Arctis (a compustage) deposits from the Arctis position."""
    from fibsem.drivers.demo.simulator import sim_is_compustage

    arctis = sim_is_compustage(microscope.system)
    return bind_coating(
        microscope, DemoCoating, ARCTIS_DEPOSITION_POSITION if arctis else None
    )
