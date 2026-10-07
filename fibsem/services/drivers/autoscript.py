"""ThermoFisher's services.

`AutoScriptMilling` mills on AutoScript's ``connection.patterning`` with the code
``ThermoMicroscope`` has always milled with (`fibsem.microscopes.autoscript.ThermoMilling`):
the per-pattern application file, the Serial mode cross-sections need, and the
patterning state read in the milling view, under the imaging channel's lock. Each
hook calls that code's method for the step, by its class: the microscope's own
method of the same name goes to this service, so calling it would come straight
back here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Tuple

from fibsem.devices.core import ParameterMetadata
from fibsem.microscopes.autoscript import TFS_SCAN_DIRECTIONS, ThermoMilling
from fibsem.services.milling import Milling, bind_milling
from fibsem.structures import (
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
    from fibsem.microscopes.autoscript import ThermoMicroscope

# Each pattern type and the code that draws it, in the order
# `FibsemMicroscope.draw_pattern` checks them.
_DRAW: Tuple[Tuple[type, Callable], ...] = (
    (FibsemRectangleSettings, ThermoMilling.draw_rectangle),
    (FibsemLineSettings, ThermoMilling.draw_line),
    (FibsemCircleSettings, ThermoMilling.draw_circle),
    (FibsemBitmapSettings, ThermoMilling.draw_bitmap_pattern),
    (FibsemPolygonSettings, ThermoMilling.draw_polygon),
)


class AutoScriptMilling(Milling):
    """ThermoFisher milling, on ``connection.patterning``."""

    parent: ThermoMicroscope

    setting_names = (
        "milling_channel",
        "hfw",
        "milling_current",
        "milling_voltage",
        "application_file",
        "patterning_mode",
    )
    scan_directions = tuple(TFS_SCAN_DIRECTIONS)

    def _setting_metadata(self, name: str) -> ParameterMetadata:
        if name == "application_file":
            files = self.parent.connection.patterning.list_all_application_files()
            return ParameterMetadata(choices=tuple(files))
        return super()._setting_metadata(name)

    def _setup(self, settings: FibsemMillingSettings, name: Optional[str]) -> None:
        # the milling view, the application file, the patterning mode, then hfw,
        # voltage and current on the milling beam
        ThermoMilling.setup_milling(self.parent, settings)

    def _draw(self, pattern: FibsemPatternSettings) -> None:
        for kind, draw in _DRAW:
            if isinstance(pattern, kind):
                draw(self.parent, pattern)
                return

    def read_state(self) -> MillingState:
        return ThermoMilling.get_milling_state(self.parent)

    def _start(self) -> None:
        ThermoMilling.start_milling(self.parent)

    def _stop(self) -> None:
        ThermoMilling.stop_milling(self.parent)

    def _pause(self) -> None:
        ThermoMilling.pause_milling(self.parent)

    def _resume(self) -> None:
        ThermoMilling.resume_milling(self.parent)

    def _estimate(self) -> float:
        return ThermoMilling.estimate_milling_time(self.parent)

    def _clear(self) -> None:
        ThermoMilling.clear_patterns(self.parent)

    def _restore(self) -> None:
        # The patterning mode persists in xT, so a stage left in Parallel would carry
        # over to the next one unless reset.
        self.parent.set_patterning_mode("Serial")


def bind_autoscript_milling(
    microscope: ThermoMicroscope,
) -> Optional[AutoScriptMilling]:
    """Build ``milling`` for a connected Thermo microscope whose beams are built."""
    return bind_milling(AutoScriptMilling, microscope)
