"""Tescan's services.

`TescanMilling` mills on DrawBeam with the code ``TescanMicroscope`` has always milled
with (`fibsem.microscopes.tescan.TescanDrawBeam`): a layer made from the milling
preset on the ion column, and a second connection to stop it from another thread.
Each hook calls that code's method for the step, by its class: the microscope's own
method of the same name goes to this service, so calling it would come straight back
here. ``TescanMicroscope.run_milling`` keeps its own loop and progress bar.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

from fibsem.devices.beam import Beam
from fibsem.devices.core import ParameterMetadata
from fibsem.microscopes.tescan import (
    DEFAULT_IMAGING_PRESET,
    TESCAN_SCAN_DIRECTIONS,
    TescanDrawBeam,
)
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
    from fibsem.microscopes.tescan import TescanMicroscope

# Each pattern type and the code that draws it, in the order
# `FibsemMicroscope.draw_pattern` checks them.
_DRAW: Tuple[Tuple[type, Callable], ...] = (
    (FibsemRectangleSettings, TescanDrawBeam.draw_rectangle),
    (FibsemLineSettings, TescanDrawBeam.draw_line),
    (FibsemCircleSettings, TescanDrawBeam.draw_circle),
    (FibsemBitmapSettings, TescanDrawBeam.draw_bitmap_pattern),
    (FibsemPolygonSettings, TescanDrawBeam.draw_polygon),
)


class TescanMilling(Milling):
    """Tescan milling, on a DrawBeam layer."""

    parent: TescanMicroscope

    # The preset sets the current and voltage; the rest go into the DrawBeam layer.
    setting_names = (
        "milling_channel",
        "hfw",
        "preset",
        "spot_size",
        "rate",
        "dwell_time",
        "spacing",
        "patterning_mode",
    )
    scan_directions = TESCAN_SCAN_DIRECTIONS

    def _setting_metadata(self, name: str) -> ParameterMetadata:
        if name == "milling_channel":
            # the DrawBeam layer is on the ion column only
            return ParameterMetadata(choices=(BeamType.ION,))
        return super()._setting_metadata(name)

    def _setup(self, settings: FibsemMillingSettings, name: Optional[str]) -> None:
        # ion only: the milling preset, then a DrawBeam layer from the recipe
        TescanDrawBeam.setup_milling(self.parent, settings)

    def _draw(self, pattern: FibsemPatternSettings) -> None:
        for kind, draw in _DRAW:
            if isinstance(pattern, kind):
                draw(self.parent, pattern)
                return

    def read_state(self) -> MillingState:
        return TescanDrawBeam.get_milling_state(self.parent)

    def _start(self) -> None:
        TescanDrawBeam.start_milling(self.parent)

    def _stop(self) -> None:
        TescanDrawBeam.stop_milling(self.parent)

    def _pause(self) -> None:
        TescanDrawBeam.pause_milling(self.parent)

    def _resume(self) -> None:
        TescanDrawBeam.resume_milling(self.parent)

    def _estimate(self) -> float:
        return TescanDrawBeam.estimate_milling_time(self.parent)

    def _clear(self) -> None:
        TescanDrawBeam.clear_patterns(self.parent)

    def _save(self, beam: Beam) -> None:
        super()._save(beam)
        # The column reports no preset until one is activated this session; put it
        # back on the imaging preset then, as milling always has.
        if self._saved is not None and "preset" in beam.parameters:
            if self._saved.get("preset") is None and beam.preset.settable:
                self._saved["preset"] = DEFAULT_IMAGING_PRESET

    def _write_back(self, beam: Beam, name: str, value: Any) -> None:
        # Activating a preset is the fragile step; a failure leaves the column where
        # it is, with a warning, rather than failing the end of milling.
        try:
            super()._write_back(beam, name, value)
        except Exception as e:
            logging.warning(f"Error restoring {name} {value!r} after milling: {e}")

    def _restore(self) -> None:
        # setup_milling's own snapshot, which this service does not use
        self.parent._preset_before_milling = None


def bind_tescan_milling(microscope: TescanMicroscope) -> Optional[TescanMilling]:
    """Build ``milling`` for a connected Tescan microscope whose beams are built."""
    return bind_milling(TescanMilling, microscope)
