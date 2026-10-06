"""The Demo's services.

`DemoMilling` mills with the demo code both demos share
(`fibsem.microscopes.simulator.DemoMilling`), on the microscope's ``milling_system``,
so the device-built Demo mills as the legacy Demo does. Each hook calls that code's
method for the step, by its class: the microscope's own method of the same name goes
to this service, so calling it would come straight back here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional, Tuple

from fibsem.devices.core import ParameterMetadata
from fibsem.microscopes.simulator import DemoMilling as DemoMillingCode
from fibsem.services.milling import Milling, bind_milling
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
        DemoMillingCode.start_milling(self.parent)

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
