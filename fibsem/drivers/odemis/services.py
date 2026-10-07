"""Odemis's services.

`OdemisMilling` mills on the Delmic AutoScript adapter (the microscope's
``connection``): the per-pattern application file, the patterning state read on the
milling channel, and the Serial mode it resets after.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from fibsem.devices.core import ParameterMetadata
from fibsem.drivers.odemis.microscope import (
    ODEMIS_DROPPED_PATTERN_SETTINGS,
    ODEMIS_SCAN_DIRECTIONS,
    beam_type_to_odemis,
)
from fibsem.services.milling import Milling, bind_milling
from fibsem.structures import (
    ACTIVE_MILLING_STATES,
    CrossSectionPattern,
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
    from fibsem.drivers.odemis.microscope import OdemisThermoMicroscope

_PATTERNING_MODES = ("Serial", "Parallel")


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
    scan_directions = ODEMIS_SCAN_DIRECTIONS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # the recipe's application file, which each draw puts back after its own
        self._application_file = "Si"

    @property
    def _connection(self):
        return self.parent.connection

    def _setting_metadata(self, name: str) -> ParameterMetadata:
        if name == "application_file":
            files = self._connection.get_available_application_files()
            return ParameterMetadata(choices=tuple(files))
        return super()._setting_metadata(name)

    def _setup(self, settings: FibsemMillingSettings, name: Optional[str]) -> None:
        # the milling view, the application file, the patterning mode, then hfw,
        # current and voltage on the milling beam
        microscope = self.parent
        channel = settings.milling_channel
        self._application_file = settings.application_file
        microscope.milling_channel = channel
        microscope.set_channel(channel)
        odemis_channel = beam_type_to_odemis[channel]
        self._connection.set_default_patterning_beam_type(odemis_channel)
        logging.info(f"Patterning beam type set to {channel} - {odemis_channel} .")
        self._connection.set_default_application_file(settings.application_file)
        logging.info(f"Default application file set to {settings.application_file}.")
        self._set_patterning_mode(settings.patterning_mode)
        microscope._write_beam("hfw", settings.hfw, channel)
        microscope._write_beam("current", settings.milling_current, channel)
        microscope._write_beam("voltage", settings.milling_voltage, channel)
        self._clear()
        logging.debug({"msg": "setup_milling", "mill_settings": settings.to_dict()})

    def _set_patterning_mode(self, mode: str) -> None:
        if mode not in _PATTERNING_MODES:
            raise ValueError(
                f"Patterning mode {mode} not supported. Supported modes: Serial, Parallel"
            )
        self._connection.set_patterning_mode(mode)
        logging.debug({"msg": "set_patterning_mode", "mode": mode})

    def _draw(self, pattern: FibsemPatternSettings) -> None:
        if isinstance(pattern, FibsemRectangleSettings):
            self._draw_rectangle(pattern)
        elif isinstance(pattern, FibsemLineSettings):
            self._draw_line(pattern)
        elif isinstance(pattern, FibsemCircleSettings):
            self._draw_circle(pattern)
        # Raised rather than skipped. Drawing nothing left a stage with no patterns,
        # which never leaves IDLE, so the milling run waited on it indefinitely;
        # raising inside the milling task fails the task with this message and still
        # restores the beams.
        elif isinstance(pattern, FibsemBitmapSettings):
            raise NotImplementedError(
                f"{type(self.parent).__name__} cannot draw bitmap patterns: the "
                "Delmic AutoScript adapter has no bitmap patterning."
            )
        elif isinstance(pattern, FibsemPolygonSettings):
            raise NotImplementedError(
                f"{type(self.parent).__name__} cannot draw polygon patterns: the "
                "Delmic AutoScript adapter has no polygon patterning."
            )

    def _warn_dropped_pattern_settings(self, pattern: FibsemPatternSettings) -> None:
        """Warn about the settings the Delmic adapter ignores (xtadapter 1.16.0).

        It creates each pattern from its geometry and depth only: a pattern meant as
        an exclusion zone is milled like any other, and a pass count or milling time
        is replaced by the microscope's own.
        """
        dropped = [
            name
            for name in ODEMIS_DROPPED_PATTERN_SETTINGS
            if getattr(pattern, name, None)
        ]
        if dropped:
            logging.warning(
                f"{type(self.parent).__name__} cannot apply {', '.join(dropped)} to "
                f"{type(pattern).__name__}: the Delmic AutoScript adapter "
                "ignores them, and the pattern is drawn without."
            )

    def _create(self, create, pdict: dict, application_file: str) -> dict:
        """Create one pattern with *application_file*, then put the recipe's back."""
        self._connection.set_default_application_file(application_file)
        pinfo = create(pdict)
        self._connection.set_default_application_file(self._application_file)
        return pinfo

    def _draw_rectangle(self, pattern: FibsemRectangleSettings) -> None:
        self._warn_dropped_pattern_settings(pattern)
        pdict = pattern.to_dict()
        pdict["center_x"] = pdict.pop("centre_x")
        pdict["center_y"] = pdict.pop("centre_y")
        create, application_file = self._connection.create_rectangle, "Si"
        if pattern.cross_section is CrossSectionPattern.CleaningCrossSection:
            create = self._connection.create_cleaning_cross_section
            application_file = "Si-ccs"
        if pattern.cross_section is CrossSectionPattern.RegularCrossSection:
            create = self._connection.create_regular_cross_section
            application_file = "Si-multipass"
        pinfo = self._create(create, pdict, application_file)
        logging.debug(
            {
                "msg": "draw_rectangle",
                "pattern_settings": pattern.to_dict(),
                "pinfo": pinfo,
            }
        )

    def _draw_line(self, pattern: FibsemLineSettings) -> None:
        pinfo = self._create(self._connection.create_line, pattern.to_dict(), "Si")
        logging.debug(
            {"msg": "draw_line", "pattern_settings": pattern.to_dict(), "pinfo": pinfo}
        )

    def _draw_circle(self, pattern: FibsemCircleSettings) -> None:
        self._warn_dropped_pattern_settings(pattern)
        pdict = pattern.to_dict()
        pdict["outer_diameter"] = 2 * pattern.radius
        # an annulus, as ThermoFisher draws one: the adapter takes the inner
        # diameter, and 0 mills the whole disc
        pdict["inner_diameter"] = 0
        if pattern.thickness != 0:
            pdict["inner_diameter"] = pdict["outer_diameter"] - 2 * pattern.thickness
        pdict["center_x"] = pattern.centre_x
        pdict["center_y"] = pattern.centre_y
        pinfo = self._create(self._connection.create_circle, pdict, "Si")
        logging.debug(
            {
                "msg": "draw_circle",
                "pattern_settings": pattern.to_dict(),
                "pinfo": pinfo,
            }
        )

    def read_state(self) -> MillingState:
        # The patterning state is that of the active view, so the milling channel is
        # selected first, under the microscope's lock.
        microscope = self.parent
        with microscope._threading_lock:
            microscope.set_channel(microscope.milling_channel)
            return MillingState[self._connection.get_patterning_state().upper()]

    def _start(self) -> None:
        if self.read_state() is MillingState.IDLE:
            self._connection.start_milling()
            logging.info("Starting milling...")

    def _stop(self) -> None:
        if self.read_state() in ACTIVE_MILLING_STATES:
            logging.info("Stopping milling...")
            self._connection.stop_milling()
            logging.info("Milling stopped.")

    def _pause(self) -> None:
        if self.read_state() is MillingState.RUNNING:
            logging.info("Pausing milling...")
            self._connection.pause_milling()
            logging.info("Milling paused.")

    def _resume(self) -> None:
        if self.read_state() is MillingState.PAUSED:
            logging.info("Resuming milling...")
            self._connection.resume_milling()
            logging.info("Milling resumed.")

    def _estimate(self) -> float:
        return self._connection.estimate_milling_time()

    def _clear(self) -> None:
        self._connection.clear_patterns()

    def _restore(self) -> None:
        # The patterning mode persists in xT, as on ThermoFisher.
        self._set_patterning_mode("Serial")


def bind_odemis_milling(
    microscope: OdemisThermoMicroscope,
) -> Optional[OdemisMilling]:
    """Build ``milling`` for an Odemis microscope whose beams are built."""
    return bind_milling(OdemisMilling, microscope)
