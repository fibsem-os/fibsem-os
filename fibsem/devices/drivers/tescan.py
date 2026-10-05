"""The Tescan (SharkSEM) beams as devices.

``TescanBeam`` implements the ``Beam`` device with the beam branches of
``TescanMicroscope._get``/``_set``, moved as they are, so the old call and the device
make the same SDK calls in the same order and log the same messages.
``TescanMicroscope`` builds one per enabled column at connect and routes its beam keys
to them; its old branches stay until a session on an instrument confirms the devices.

The vendor beams are ``connection.SEM`` and ``connection.FIB``. SharkSEM is one socket,
so every read and write holds the microscope's ``_connection_lock``, as ``_get`` and
``_set`` do. This module imports nothing from the SDK; the Tescan backend's guarded
import is the only one.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

import fibsem.constants as constants
from fibsem.devices.beam import Beam
from fibsem.devices.core import ParameterMetadata, Resources
from fibsem.structures import BeamType, Point, RangeLimit

if TYPE_CHECKING:
    from fibsem.microscopes.tescan import TescanMicroscope

# What the old get_available_values lists for "current".
_CURRENT_CHOICES: Dict[BeamType, List[float]] = {
    BeamType.ELECTRON: [1.0e-12],
    BeamType.ION: [20e-12, 60e-12, 0.2e-9, 0.74e-9, 2.0e-9, 7.6e-9, 28.0e-9, 120e-9],
}

_NOT_SETTABLE = ParameterMetadata(settable=False)


class TescanBeam(Beam):
    """A Tescan column: ``connection.SEM`` or ``connection.FIB``.

    Each parameter is the matching branch of ``TescanMicroscope._get``/``_set``. A
    write prepares the beam first (turns it on, stops the scan, waits until it is not
    busy), as ``_set`` does for every beam key. The choices are
    ``get_available_values``'s.

    What the Tescan API refuses is still written, so the old API keeps its message,
    but reads as not settable: the ion column's current and voltage (set by preset),
    and the resolution, dwell time and stigmation, which are read from the last image.

    Not here, so absent on the new API and still answered by the old branches: the ion
    column's working distance (the old read warns of an unknown key), the electron
    column's preset (it is set directly, ``beam_uses_presets``), ``detector_mode``
    (not in the API), ``blanked``, ``plasma_gas`` and the scan modes.
    """

    def __init__(
        self,
        beam_type: BeamType,
        parent: TescanMicroscope,
        resources: Optional[Resources] = None,
    ):
        super().__init__(beam_type, parent=parent, resources=resources)

    @property
    def _lock(self):
        return self.parent._connection_lock

    @property
    def _beam(self) -> Any:
        """The vendor beam, looked up on every call, as the old branches do."""
        return self.parent._get_beam(self.beam_type)

    def _prepared(self) -> Any:
        """The vendor beam, prepared as ``_set`` prepares it before every beam key."""
        beam = self._beam
        self.parent._prepare_beam(self.beam_type)
        return beam

    @property
    def _cache(self) -> Any:
        """The beam settings the last image reported, for what the API cannot read."""
        return self.parent._beam_parameters[self.beam_type]

    def _not_supported(self, key: str) -> None:
        with self._lock:
            self._prepared()
            logging.info(f"Setting {key} directly is not supported by Tescan API.")

    # -- on --------------------------------------------------------------------------

    def read_on(self) -> bool:
        with self._lock:
            beam = self._beam
            return beam.Beam.GetStatus() == beam.Beam.Status.BeamOn

    def write_on(self, value: bool) -> None:
        with self._lock:
            beam = self._prepared()
            beam.Beam.On() if value else beam.Beam.Off()
            logging.info(
                f"{self.beam_type.name} beam turned {'on' if value else 'off'}."
            )

    # -- working distance: the electron column only ----------------------------------

    def available_working_distance(self) -> bool:
        return self.beam_type is BeamType.ELECTRON

    def read_working_distance(self) -> float:
        with self._lock:
            return self._beam.Optics.GetWD() * constants.MILLIMETRE_TO_METRE

    def write_working_distance(self, value: float) -> None:
        with self._lock:
            beam = self._prepared()
            beam.Optics.SetWD(value * constants.METRE_TO_MILLIMETRE)
            logging.info(f"Electron beam working distance set to {value} m.")

    # -- current and voltage: set by preset on the ion column ------------------------

    def read_current(self) -> float:
        with self._lock:
            beam = self._beam
            if self.beam_type is BeamType.ELECTRON:
                return beam.Beam.GetCurrent() * constants.PICO_TO_SI
            return beam.Beam.ReadProbeCurrent() * constants.PICO_TO_SI

    def write_current(self, value: float) -> None:
        with self._lock:
            beam = self._prepared()
            if self.beam_type is BeamType.ION:
                logging.info(
                    f"Setting current directly for {self.beam_type} is not supported by Tescan API, please use presets instead."
                )
                return
            beam.Beam.SetCurrent(value * constants.SI_TO_PICO)
            logging.info(f"Electron beam current set to {value} A.")

    def metadata_current(self) -> ParameterMetadata:
        return ParameterMetadata(
            choices=list(_CURRENT_CHOICES[self.beam_type]),
            settable=self.beam_type is BeamType.ELECTRON,
        )

    def read_voltage(self) -> float:
        with self._lock:
            return self._beam.Beam.GetVoltage()

    def write_voltage(self, value: float) -> None:
        with self._lock:
            beam = self._prepared()
            if self.beam_type is BeamType.ION:
                logging.warning(
                    f"Setting voltage directly for {self.beam_type} is not supported by Tescan API, please use presets instead."
                )
                return
            beam.Beam.SetVoltage(value)
            logging.info(f"Electron beam voltage set to {value} V.")

    def metadata_voltage(self) -> ParameterMetadata:
        return ParameterMetadata(settable=self.beam_type is BeamType.ELECTRON)

    # -- field of view, scan rotation, shift -----------------------------------------

    def read_hfw(self) -> float:
        with self._lock:
            return self._beam.Optics.GetViewfield() * constants.MILLIMETRE_TO_METRE

    def write_hfw(self, value: float) -> None:
        from fibsem.microscopes.tescan import LIMITS

        with self._lock:
            beam = self._prepared()
            limits = LIMITS[self.beam_type]["hfw"]
            value = np.clip(value, limits[0], limits[1])
            beam.Optics.SetViewfield(value * constants.METRE_TO_MILLIMETRE)
            logging.info(f"{self.beam_type.name} HFW set to {value} m.")

    def metadata_hfw(self) -> ParameterMetadata:
        from fibsem.microscopes.tescan import LIMITS

        low, high = LIMITS[self.beam_type]["hfw"]
        return ParameterMetadata(limits=RangeLimit(min=low, max=high))

    def read_scan_rotation(self) -> float:
        with self._lock:
            # degrees, and nan on the simulator
            scan_rotation = self._beam.Optics.GetImageRotation()
        if np.isnan(scan_rotation):
            scan_rotation = 0.0
        return scan_rotation * constants.DEGREES_TO_RADIANS

    def write_scan_rotation(self, value: float) -> None:
        with self._lock:
            beam = self._prepared()
            beam.Optics.SetImageRotation(value * constants.RADIANS_TO_DEGREES)
            logging.info(f"{self.beam_type.name} scan rotation set to {value} radians.")

    def read_shift(self) -> Point:
        with self._lock:
            values = self._beam.Optics.GetImageShift()
        return Point(
            x=values[0] * constants.MILLIMETRE_TO_METRE,
            y=values[1] * constants.MILLIMETRE_TO_METRE,
        )

    def write_shift(self, value: Point) -> None:
        with self._lock:
            beam = self._prepared()
            point = Point(
                value.x * constants.METRE_TO_MILLIMETRE,
                value.y * constants.METRE_TO_MILLIMETRE,
            )
            beam.Optics.SetImageShift(point.x, point.y)
            logging.info(f"{self.beam_type.name} beam shift set to {value}.")

    # -- read from the last image: the API cannot read or set these ------------------

    def read_resolution(self) -> Any:
        return self._cache.resolution

    def write_resolution(self, value: Any) -> None:
        self._not_supported("resolution")

    def metadata_resolution(self) -> ParameterMetadata:
        return _NOT_SETTABLE

    def read_dwell_time(self) -> Optional[float]:
        return self._cache.dwell_time

    def write_dwell_time(self, value: float) -> None:
        self._not_supported("dwell_time")

    def metadata_dwell_time(self) -> ParameterMetadata:
        return _NOT_SETTABLE

    def read_stigmation(self) -> Optional[Point]:
        return self._cache.stigmation

    def write_stigmation(self, value: Point) -> None:
        self._not_supported("stigmation")

    def metadata_stigmation(self) -> ParameterMetadata:
        return _NOT_SETTABLE

    # -- presets: the ion column's; the last one activated, and activation ----------

    def available_preset(self) -> bool:
        return self.parent.beam_uses_presets(self.beam_type)

    def read_preset(self) -> Optional[str]:
        return self._cache.preset

    def write_preset(self, value: str) -> None:
        with self._lock:
            beam = self._prepared()
            self.parent._activate_preset(beam, self.beam_type, value)

    def metadata_preset(self) -> ParameterMetadata:
        return ParameterMetadata(choices=self.parent._get_presets(self.beam_type))

    # -- the detector: each column has its own, so no channel to claim ---------------

    def read_detector_type(self) -> Optional[str]:
        with self._lock:
            detector = self._beam.Detector.Get(Channel=0)
        return None if detector is None else detector.name

    def write_detector_type(self, value: str) -> None:
        with self._lock:
            beam = self._prepared()
            detector = self.parent._get_detector(value, self.beam_type)
            if detector is None:
                logging.warning(f"Detector {value} not found for {self.beam_type}.")
                return
            beam.Detector.Set(Channel=0, Detector=detector)
            self.parent._active_detector[self.beam_type] = detector
            logging.debug(f"{self.beam_type.name} detector type set to {value}.")

    def metadata_detector_type(self) -> ParameterMetadata:
        detectors = self.parent._get_available_detectors(self.beam_type)
        return ParameterMetadata(choices=[d.name for d in detectors])

    def _gain_black(self) -> Any:
        return self._beam.Detector.GetGainBlack(
            Detector=self.parent._active_detector[self.beam_type]
        )

    def read_detector_contrast(self) -> float:
        with self._lock:
            contrast, _ = self._gain_black()
        return contrast / 100

    def write_detector_contrast(self, value: float) -> None:
        self._write_gain_black("detector_contrast", value)

    def read_detector_brightness(self) -> float:
        with self._lock:
            _, brightness = self._gain_black()
        return brightness / 100

    def write_detector_brightness(self, value: float) -> None:
        self._write_gain_black("detector_brightness", value)

    def _write_gain_black(self, key: str, value: float) -> None:
        with self._lock:
            beam = self._prepared()
            if not (0 <= value <= 1):
                logging.warning(
                    f"Invalid value for {self.beam_type} {key}: {value}. Must be between 0 and 1."
                )
                return
            active_detector = self.parent._active_detector[self.beam_type]
            if active_detector is None:
                logging.warning(
                    f"No active detector for {self.beam_type}. Please set detector type first."
                )
                return
            contrast, brightness = beam.Detector.GetGainBlack(Detector=active_detector)
            if key == "detector_contrast":
                contrast = value * 100
            if key == "detector_brightness":
                brightness = value * 100
            beam.Detector.SetGainBlack(
                Detector=active_detector, Gain=contrast, Black=brightness
            )
            logging.info(f"{self.beam_type.name} {key} set to {value}.")


def bind_tescan_beams(
    microscope: TescanMicroscope, resources: Optional[Resources] = None
) -> Dict[BeamType, TescanBeam]:
    """Build ``beams[BeamType]`` for a connected Tescan microscope: one per enabled
    column, so a disabled one is never touched."""
    enabled = {
        BeamType.ELECTRON: microscope.system.electron.enabled,
        BeamType.ION: microscope.system.ion.enabled,
    }
    return {
        beam_type: TescanBeam(beam_type, microscope, resources).connect()
        for beam_type, on in enabled.items()
        if on
    }
