"""The Demo's services.

`DemoMilling` mills on the microscope's simulated ``milling_system``: the patterns, the
milling state and the application files the demo sets up at construction. The beams
change only through the microscope's beam methods, so through the beam devices.

Nothing on the Demo ends a mill but the clock, so a ``run`` is timed by the estimate
it starts with, on simulated time: each wait is ``sim_sleep``, which the test suite
turns off, and the time counts as it would have passed. A ``start`` alone runs until
something stops it.

`DemoSpotBurn` is the shared point-by-point burn, also on simulated time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from fibsem._timing import sim_sleep
from fibsem.devices.core import ParameterMetadata
from fibsem.drivers.demo.simulator import (
    SIM_ASYNC_MILLING_EXTRA_TIME,
    SIMULATOR_SCAN_DIRECTIONS,
)
from fibsem.milling.progress import MillingProgress
from fibsem.services.milling import Milling, bind_milling, progress_update
from fibsem.services.spot_burn import SpotBurn, bind_spot_burn
from fibsem.structures import (
    ACTIVE_MILLING_STATES,
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemLineSettings,
    FibsemMillingSettings,
    FibsemPatternSettings,
    FibsemPolygonSettings,
    FibsemRectangleSettings,
    MillingState,
)

if TYPE_CHECKING:
    from fibsem.drivers.demo.microscope import DemoMicroscope

# The patterns the Demo draws, in the order `FibsemMicroscope.draw_pattern` checks them.
_DRAWN = (
    FibsemRectangleSettings,
    FibsemLineSettings,
    FibsemCircleSettings,
    FibsemBitmapSettings,
    FibsemPolygonSettings,
)
_PATTERNING_MODES = ("Serial", "Parallel")
# simulated seconds each pattern takes
_PATTERN_TIME = 5


class DemoMilling(Milling):
    """The Demo's milling, on the microscope's simulated ``milling_system``."""

    parent: DemoMicroscope

    setting_names = (
        "milling_channel",
        "hfw",
        "milling_current",
        "milling_voltage",
        "application_file",
        "patterning_mode",
    )
    scan_directions = tuple(SIMULATOR_SCAN_DIRECTIONS)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The run in progress, which the clock ends: its length and the simulated
        # time it has run for. None outside a run (`start` alone runs until stopped).
        self._run_total: Optional[float] = None
        self._run_elapsed = 0.0
        # Whether the mill running now was started by `start`, which never ends on
        # its own here; its estimate adds `SIM_ASYNC_MILLING_EXTRA_TIME`.
        self._open_ended = False

    def _setting_metadata(self, name: str) -> ParameterMetadata:
        if name == "application_file":
            files = self.parent.milling_system.application_files
            return ParameterMetadata(choices=tuple(files))
        return super()._setting_metadata(name)

    def _setup(self, settings: FibsemMillingSettings, name: Optional[str]) -> None:
        if settings.patterning_mode not in _PATTERNING_MODES:
            raise ValueError(
                f"Invalid patterning mode: {settings.patterning_mode}. "
                f"Must be one of {_PATTERNING_MODES}."
            )
        microscope = self.parent
        system = microscope.milling_system
        channel = settings.milling_channel
        microscope.milling_channel = channel
        microscope.set_channel(channel)
        system.default_beam_type = channel
        system.default_application_file = settings.application_file
        system.patterning_mode = settings.patterning_mode
        # the recipe's beam conditions go through the beam devices
        microscope._write_beam("hfw", settings.hfw, channel)
        microscope._write_beam("current", settings.milling_current, channel)
        microscope._write_beam("voltage", settings.milling_voltage, channel)
        self._clear()
        logging.debug({"msg": "setup_milling", "mill_settings": settings.to_dict()})

    def _draw(self, pattern: FibsemPatternSettings) -> None:
        if not isinstance(pattern, _DRAWN):
            logging.warning(f"The Demo does not draw {type(pattern).__name__}.")
            return
        logging.debug({"msg": "draw_pattern", "pattern_settings": pattern.to_dict()})
        self.parent.milling_system.patterns.append(pattern)

    def read_state(self) -> MillingState:
        return self.parent.milling_system.state

    def _set_state(self, state: MillingState) -> None:
        self.parent.milling_system.state = state

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
        microscope._mill_into_sample_scene(current)
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

    def _stop(self) -> None:
        self._set_state(MillingState.IDLE)
        self._open_ended = False

    def _pause(self) -> None:
        self._set_state(MillingState.PAUSED)

    def _resume(self) -> None:
        self._set_state(MillingState.RUNNING)

    def _estimate(self) -> float:
        estimate = _PATTERN_TIME * len(self.parent.milling_system.patterns)
        if self._open_ended and self.read_state() in ACTIVE_MILLING_STATES:
            estimate += SIM_ASYNC_MILLING_EXTRA_TIME
        return estimate

    def _clear(self) -> None:
        self.parent.milling_system.patterns = []


def bind_demo_milling(microscope: DemoMicroscope) -> Optional[DemoMilling]:
    """Build ``milling`` for a Demo microscope whose beams are built."""
    return bind_milling(DemoMilling, microscope)


class DemoSpotBurn(SpotBurn):
    """The Demo's spot burn: the point-by-point burn on the simulated beams, where
    unblanking a parked beam burns the spot into the sample scene. Its waits are
    simulated time."""

    def _wait(self, seconds: float) -> None:
        sim_sleep(seconds)


def bind_demo_spot_burn(microscope: DemoMicroscope) -> Optional[DemoSpotBurn]:
    """Build ``spot_burn`` for a Demo microscope whose beams are built."""
    return bind_spot_burn(DemoSpotBurn, microscope)
