"""Odemis's services.

`OdemisMilling` mills on the Delmic AutoScript adapter with the code
``OdemisThermoMicroscope`` has always milled with
(`fibsem.microscopes.odemis_microscope.OdemisPatterning`): the per-pattern application
file, and the patterning state read on the milling channel. Each hook calls that
code's method for the step, by its class: the microscope's own method of the same name
goes to this service, so calling it would come straight back here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Tuple

from fibsem.devices.core import ParameterMetadata
from fibsem.microscopes.odemis_microscope import OdemisPatterning
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
    from fibsem.microscopes.odemis_microscope import OdemisThermoMicroscope

# Each pattern type and the code that draws it, in the order
# `FibsemMicroscope.draw_pattern` checks them.
_DRAW: Tuple[Tuple[type, Callable], ...] = (
    (FibsemRectangleSettings, OdemisPatterning.draw_rectangle),
    (FibsemLineSettings, OdemisPatterning.draw_line),
    (FibsemCircleSettings, OdemisPatterning.draw_circle),
    (FibsemBitmapSettings, OdemisPatterning.draw_bitmap_pattern),
    (FibsemPolygonSettings, OdemisPatterning.draw_polygon),
)


class OdemisMilling(Milling):
    """Odemis milling, on the AutoScript adapter's patterning."""

    parent: OdemisThermoMicroscope

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
            files = self.parent.connection.get_available_application_files()
            return ParameterMetadata(choices=tuple(files))
        return super()._setting_metadata(name)

    def _setup(self, settings: FibsemMillingSettings, name: Optional[str]) -> None:
        # the milling view, the application file, the patterning mode, then hfw,
        # current and voltage on the milling beam
        OdemisPatterning.setup_milling(self.parent, settings)

    def _draw(self, pattern: FibsemPatternSettings) -> None:
        for kind, draw in _DRAW:
            if isinstance(pattern, kind):
                draw(self.parent, pattern)
                return

    def read_state(self) -> MillingState:
        return OdemisPatterning.get_milling_state(self.parent)

    def _start(self) -> None:
        OdemisPatterning.start_milling(self.parent)

    def _stop(self) -> None:
        OdemisPatterning.stop_milling(self.parent)

    def _pause(self) -> None:
        OdemisPatterning.pause_milling(self.parent)

    def _resume(self) -> None:
        OdemisPatterning.resume_milling(self.parent)

    def _estimate(self) -> float:
        return OdemisPatterning.estimate_milling_time(self.parent)

    def _clear(self) -> None:
        OdemisPatterning.clear_patterns(self.parent)

    def _restore(self) -> None:
        # The patterning mode persists in xT, as on ThermoFisher.
        OdemisPatterning.set_patterning_mode(self.parent, "Serial")


def bind_odemis_milling(
    microscope: OdemisThermoMicroscope,
) -> Optional[OdemisMilling]:
    """Build ``milling`` for an Odemis microscope whose beams are built."""
    return bind_milling(OdemisMilling, microscope)
