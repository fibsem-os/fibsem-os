"""The Demo's services.

`DemoMilling` mills with the demo code (`fibsem.microscopes.simulator.DemoMilling`),
on the microscope's ``milling_system``, so the Demo mills as it did before the
service. Each hook calls that code's
method for the step, by its class: the microscope's own method of the same name goes
to this service, so calling it would come straight back here.

Nothing on the Demo ends a mill but the clock, so a ``run`` is timed by the estimate
it starts with, on simulated time: each wait is ``sim_sleep``, which the test suite
turns off, and the time counts as it would have passed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional, Tuple

from fibsem._timing import sim_sleep
from fibsem.devices.core import ParameterMetadata
from fibsem.microscopes.simulator import DemoMilling as DemoMillingCode
from fibsem.services.milling import Milling, MillingPoll, bind_milling
from fibsem.structures import (
    BeamType,
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
    from fibsem.microscopes.device_demo import DemoMicroscope

# Each pattern type and the demo code that draws it, in the order
# `FibsemMicroscope.draw_pattern` checks them.
_DRAW: Tuple[Tuple[type, Callable], ...] = (
    (FibsemRectangleSettings, DemoMillingCode.draw_rectangle),
    (FibsemLineSettings, DemoMillingCode.draw_line),
    (FibsemCircleSettings, DemoMillingCode.draw_circle),
    (FibsemBitmapSettings, DemoMillingCode.draw_bitmap_pattern),
    (FibsemPolygonSettings, DemoMillingCode.draw_polygon),
)


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The run in progress, which the clock ends: its length and the simulated
        # time it has run for. None outside a run (`start` alone runs until stopped).
        self._run_total: Optional[float] = None
        self._run_elapsed = 0.0

    def _setting_metadata(self, name: str) -> ParameterMetadata:
        if name == "application_file":
            files = self.parent.milling_system.application_files
            return ParameterMetadata(choices=tuple(files))
        return super()._setting_metadata(name)

    def _setup(self, settings: FibsemMillingSettings, name: Optional[str]) -> None:
        # The recipe's beam conditions go through the beam devices (`set`).
        DemoMillingCode.setup_milling(self.parent, settings)

    def _draw(self, pattern: FibsemPatternSettings) -> None:
        for kind, draw in _DRAW:
            if isinstance(pattern, kind):
                draw(self.parent, pattern)
                return
        logging.warning(f"The Demo does not draw {type(pattern).__name__}.")

    def read_state(self) -> MillingState:
        return DemoMillingCode.get_milling_state(self.parent)

    def _start(self) -> None:
        if self._run_total is None:
            DemoMillingCode.start_milling(self.parent)
            return
        # a timed run, which ends by itself: not an open-ended start
        self.parent._async_milling = False
        self.parent.milling_system.state = MillingState.RUNNING

    def _before_run(self) -> None:
        # Into the simulated sample from the start, so the scene shows the mill
        # however the run ends.
        microscope = self.parent
        current = microscope.get_beam_current(microscope.milling_channel)
        microscope._mill_into_sample_scene(current)
        # a stale `start` mustn't stretch a timed run's estimate
        microscope._async_milling = False
        self._run_total = DemoMillingCode.estimate_milling_time(microscope)
        self._run_elapsed = 0.0

    def _poll(self) -> MillingPoll:
        state = self.read_state()
        if self._run_total is None:
            return MillingPoll(state=state)
        if state is MillingState.RUNNING and self._run_elapsed >= self._run_total:
            self.parent.milling_system.state = state = MillingState.IDLE
        return MillingPoll(
            state=state, elapsed=self._run_elapsed, total=self._run_total
        )

    def _wait(self, seconds: float) -> None:
        sim_sleep(seconds)
        if self.read_state() is MillingState.RUNNING:
            self._run_elapsed += seconds

    def _after_run(self) -> None:
        self._run_total = None
        self.parent.milling_system.state = MillingState.IDLE

    def _stop(self) -> None:
        DemoMillingCode.stop_milling(self.parent)

    def _pause(self) -> None:
        DemoMillingCode.pause_milling(self.parent)

    def _resume(self) -> None:
        DemoMillingCode.resume_milling(self.parent)

    def _estimate(self) -> float:
        return DemoMillingCode.estimate_milling_time(self.parent)

    def _clear(self) -> None:
        DemoMillingCode.clear_patterns(self.parent)


def bind_demo_milling(microscope: DemoMicroscope) -> Optional[DemoMilling]:
    """Build ``milling`` for a Demo microscope whose beams are built."""
    return bind_milling(DemoMilling, microscope)
