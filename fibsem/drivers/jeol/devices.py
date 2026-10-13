"""The JEOL beams and stage as devices.

``JeolElectronBeam`` and ``JeolIonBeam`` are the two columns: nearly every parameter is
a different JEOL call on each (the SEM's on the SEM client, the FIB's on the FIB
client), so each column is its own class over a shared `JeolBeam`. ``JeolStage`` is
the stage in fibsem's frame: JEOL's axes differ from fibsem's by the signs in
`STAGE_AXIS_SIGNS`, and that is all, so the shared movement and projection maths hold
as they are.

Each device is handed the microscope's `JeolConnection` and makes every JEOL call
through it. What JEOL can set is listed here as parameter choices and limits (beam
currents, scan resolutions, dwell times), so a request for something it doesn't have
is snapped to the nearest it does, and the UI offers only what it has.

Ported from the old-API driver (``fibsem/microscopes/cryolameller.py``); the measured
values (signs, calibrations, the spot area) are kept as they were measured.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import fibsem.constants as constants
from fibsem.devices.beam import Beam, nearest_resolution
from fibsem.devices.core import ParameterMetadata, Resources
from fibsem.devices.stage import Stage, axis_limits_from_degrees, rotating_stage_poses
from fibsem.drivers.jeol import images
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemImageMetadata,
    FibsemRectangle,
    FibsemStagePosition,
    ImageSettings,
    Point,
    RangeLimit,
)

if TYPE_CHECKING:
    from fibsem.drivers.jeol.connection import JeolConnection
    from fibsem.drivers.jeol.microscope import JeolMicroscope
    from fibsem.drivers.registry import BuildContext
    from fibsem.structures import DeviceEntry

# -- what the columns have ----------------------------------------------------------

# The beam currents, in beam mode (FIB) or probe number (SEM) order: mode n is the
# n-th current, counted from 1 as JEOL counts. The electron list is to be confirmed on
# the instrument.
BEAM_CURRENTS: Dict[BeamType, List[float]] = {
    BeamType.ELECTRON: [
        1.0e-12,
        3.0e-12,
        10e-12,
        30e-12,
        0.1e-9,
        0.3e-9,
        1e-9,
        4e-9,
        15e-9,
        16e-9,
        17e-9,
        18e-9,
        20e-9,
        21e-9,
        22e-9,
        23e-9,
        24e-9,
        300e-9,
    ],
    BeamType.ION: [
        90e-9,
        50e-9,
        30e-9,
        10e-9,
        3e-9,
        1e-9,
        500e-12,
        300e-12,
        100e-12,
        30e-12,
        10e-12,
        3e-12,
        1e-12,
    ],
}

BEAM_VOLTAGES: Dict[BeamType, List[float]] = {
    BeamType.ELECTRON: [1000, 3000, 5000, 10000, 15000],
    BeamType.ION: [5000, 10000, 15000, 20000, 25000, 30000],
}

# Field of view: JEOL sets a magnification, and the field is 128 mm over it.
HFW_AT_UNIT_MAGNIFICATION = 128e-3
HFW_LIMITS: Dict[BeamType, RangeLimit] = {
    BeamType.ELECTRON: RangeLimit(min=1.0e-6, max=2580.0e-6),
    BeamType.ION: RangeLimit(min=1.0e-6, max=1280.0e-6),
}
# setMagnification answers 2 when it set the nearest magnification it has.
MAGNIFICATION_NEAREST = 2

# The FIB scan types: (resolution, dwell time in s, scan memory), by scan type number.
FIB_SCAN_TYPES: Dict[int, Tuple[Tuple[int, int], float, int]] = {
    1: ((640, 480), 0.49e-6, 0),
    2: ((640, 480), 1.6e-6, 0),
    4: ((1280, 960), 0.41e-6, 0),
    5: ((640, 480), 3.3e-6, 0),
    6: ((1280, 960), 0.81e-6, 0),
    7: ((640, 480), 8.1e-6, 1),
    8: ((640, 480), 32.5e-6, 1),
    9: ((1280, 960), 8.1e-6, 1),
    10: ((1280, 960), 16.3e-6, 2),
    11: ((1280, 960), 32.6e-6, 2),
    12: ((2560, 1920), 81.4e-6, 2),
    13: ((1280, 960), 65.1e-6, 2),
    14: ((2560, 1920), 16.3e-6, 2),
    15: ((5120, 3840), 4.1e-6, 2),
    16: ((1280, 960), 130e-6, 2),
    17: ((2560, 1920), 32.6e-6, 2),
    18: ((5120, 3840), 8.1e-6, 2),
}

# The SEM scan speeds JEOL images at (Fine2, scan mode 3), and the dwell time of each.
SEM_SCAN_MODE = 3
SEM_SCAN_SPEEDS: Dict[int, float] = {
    3: 0.167e-6,
    4: 0.333e-6,
    5: 0.667e-6,
    6: 1.333e-6,
    7: 2.667e-6,
    8: 5.333e-6,
    9: 14.08e-6,
    10: 29.25e-6,
    11: 59.67e-6,
    12: 90.16e-6,
}

# The SEM photo sizes, and the index setPhotoImageSize takes for each. 800x600 (index
# 1) is left out, as it always was.
SEM_PHOTO_SIZES: Dict[Tuple[int, int], int] = {
    (640, 480): 0,
    (1280, 960): 2,
    (2560, 1920): 3,
    (5120, 3840): 4,
}
# The photo size before one is set this session: the configuration's default.
DEFAULT_SEM_RESOLUTION = (1280, 960)

# SEM image shift: one DAC step moves the beam 18.6 nm; the DAC runs -2048 to 2047.
SEM_SHIFT_METRES_PER_DAC = 18.6e-9
SEM_SHIFT_DAC_RANGE = (-2048, 2047)

# Working distance: linear in the SEM objective lens DAC and in the FIB OL voltage.
SEM_WORKING_DISTANCE_MM_REF = 8.544141
SEM_WORKING_DISTANCE_DAC_REF = 552960
SEM_WORKING_DISTANCE_DAC_PER_MM = -25151.6
FIB_WORKING_DISTANCE_MM_REF = 19.00
FIB_WORKING_DISTANCE_VOLTAGE_REF = 16864.5
FIB_WORKING_DISTANCE_VOLTAGE_PER_MM = 246

# Detectors: JEOL's view signals. The FIB has one.
SEM_DETECTOR_SIGNALS = range(388, 398 + 1)
FIB_DETECTOR_SIGNAL = 381

# The sign of each beam's shift against fibsem's (+x moves the content right, +y up).
# Measured 2026-09-29, both beams alike: at scan rotation 0, JEOL's +x moves the
# content left and +y down. The shift is not fixed to the column: at 180 degrees the
# direction turns with the scan, so no further flip is needed.
BEAM_SHIFT_SIGNS: Dict[BeamType, Tuple[float, float]] = {
    BeamType.ELECTRON: (-1.0, -1.0),
    BeamType.ION: (-1.0, -1.0),
}

# How long the ion column takes to switch on, and autocontrast to run, in seconds.
ION_BEAM_ON_WAIT = 5.0
AUTOCONTRAST_WAIT = 5.0
# Added to a FIB scan's own time before its frame is read back.
FIB_SCAN_MARGIN = 0.2


def nearest_beam_current(current: float, beam_type: BeamType) -> Tuple[int, float]:
    """The beam mode (1-based, as JEOL counts) and the current nearest *current*. The
    currents span five decades, so nearest is by ratio."""
    currents = np.asarray(BEAM_CURRENTS[beam_type])
    index = int(np.argmin(np.abs(np.log(currents) - np.log(current))))
    if not np.isclose(currents[index], current, rtol=1e-6):
        logging.warning(
            f"{beam_type.name} current {current:.3e} A is not available; "
            f"using {currents[index]:.3e} A"
        )
    return index + 1, float(currents[index])


def fib_scan_type(resolution: Tuple[int, int], dwell_time: float) -> int:
    """The FIB scan type with *resolution* whose dwell time is nearest *dwell_time*."""
    candidates = [
        number
        for number, (size, _, _) in FIB_SCAN_TYPES.items()
        if size == tuple(resolution)
    ]
    if not candidates:
        raise ValueError(f"No FIB scan type for resolution {resolution}.")
    return min(
        candidates, key=lambda number: abs(FIB_SCAN_TYPES[number][1] - dwell_time)
    )


def magnification_for(hfw: float) -> int:
    return max(1, int(round(HFW_AT_UNIT_MAGNIFICATION / hfw)))


def _hfw_of(magnification: float) -> float:
    return HFW_AT_UNIT_MAGNIFICATION / magnification


def _degrees(radians: float) -> int:
    """A scan rotation in whole degrees, 0 to 359, as JEOL takes it."""
    return int(round(np.degrees(radians))) % 360


class JeolBeam(Beam):
    """What both JEOL columns share: the connection, the image they build, live view.

    A column implements ``_grab`` (one frame as an array, at the settings it is set
    to), and its parameters.
    """

    def __init__(
        self,
        beam_type: BeamType,
        parent: JeolMicroscope,
        connection: JeolConnection,
        resources: Optional[Resources] = None,
    ):
        super().__init__(beam_type, parent=parent, resources=resources)
        self.connection = connection

    @property
    def _lock(self):
        return self.connection.lock

    def _fib(self, method: str, *args: Any, **kwargs: Any) -> Any:
        return self.connection.fib_call(method, *args, **kwargs)

    def _sem(self, method: str, *args: Any, **kwargs: Any) -> Any:
        return self.connection.sem_call(method, *args, **kwargs)

    # -- shared metadata -------------------------------------------------------------

    def metadata_current(self) -> ParameterMetadata:
        return ParameterMetadata(choices=list(BEAM_CURRENTS[self.beam_type]))

    def metadata_voltage(self) -> ParameterMetadata:
        return ParameterMetadata(choices=list(BEAM_VOLTAGES[self.beam_type]))

    def metadata_hfw(self) -> ParameterMetadata:
        return ParameterMetadata(limits=HFW_LIMITS[self.beam_type])

    def _clip_hfw(self, value: float) -> float:
        limits = HFW_LIMITS[self.beam_type]
        return float(np.clip(value, limits.min, limits.max))

    def _signed_shift(self, value: Point) -> Point:
        """A shift between fibsem's convention and JEOL's: a sign is its own inverse."""
        sx, sy = BEAM_SHIFT_SIGNS[self.beam_type]
        return Point(sx * value.x, sy * value.y)

    # -- imaging ---------------------------------------------------------------------

    # Seconds to wait after the resolution is set, before the frame (the old driver's).
    RESOLUTION_SETTLE = 0.0

    def _grab(self) -> np.ndarray:
        """One frame at the column's current settings."""
        raise NotImplementedError

    def _acquire(self, image_settings: Optional[ImageSettings]) -> FibsemImage:
        # In the old driver's order, which JEOL needs: the view selected first, so
        # what follows acts on this column's view; then the field of view, the state
        # for the metadata, the resolution (and the SEM's settle), and the frame.
        # Not the dwell time: on JEOL it sets the scan mode, which sets the resolution
        # too, and the two can conflict. The image records the dwell time read back.
        microscope = self.parent
        with self._lock:
            self.connection.select_view(self.beam_type)
            if image_settings is not None:
                self.hfw.set_value(image_settings.hfw)
            state = microscope.get_microscope_state(beam_type=self.beam_type)
            if image_settings is not None:
                self.resolution.set_value(
                    nearest_resolution(
                        image_settings.resolution, self.resolution.choices
                    )
                )
                if self.RESOLUTION_SETTLE:
                    self.connection.wait(self.RESOLUTION_SETTLE)
            data = self._grab()
        return self._image(data, state, image_settings)

    def _image(
        self, data: np.ndarray, state: Any, image_settings: Optional[ImageSettings]
    ) -> FibsemImage:
        """The FibsemImage for a frame: its settings are what the column read back."""
        hfw = self.hfw.get_value()
        height, width = data.shape[:2]
        # the path and filename to save under: these settings', else the last given
        last = image_settings or self._last_image_settings or ImageSettings()
        settings = ImageSettings(
            beam_type=self.beam_type,
            resolution=(width, height),
            dwell_time=self.dwell_time.get_value(),
            hfw=hfw,
            path=last.path,
            filename=last.filename,
        )
        image = FibsemImage(
            data=data,
            metadata=FibsemImageMetadata(
                image_settings=settings,
                microscope_state=state,
                pixel_size=Point(x=hfw / width, y=hfw / width),
            ),
        )
        return image

    def _live(self, stop: threading.Event) -> None:
        # re-acquire with the current settings until stopped
        while not stop.is_set():
            image = self._live_frame()
            if stop.is_set():
                break
            self.live_frame.emit(image)

    def _live_frame(self) -> FibsemImage:
        return self.acquire()


class JeolElectronBeam(JeolBeam):
    """The SEM, on the SEM client.

    Not here, because the API has no write for them: stigmation (read only) and the
    detector mode. ``on`` reads the gun valve and writes the HT switch, as the old
    driver did; whether they agree is to be checked on the instrument.
    """

    RESOLUTION_SETTLE = 0.2

    def __init__(
        self,
        parent: JeolMicroscope,
        connection: JeolConnection,
        resources: Optional[Resources] = None,
    ):
        super().__init__(BeamType.ELECTRON, parent, connection, resources)
        # setPhotoImageSize has no read: the size last set
        self._resolution = DEFAULT_SEM_RESOLUTION

    def read_on(self) -> bool:
        return bool(self._sem("getGunValve"))

    def write_on(self, value: bool) -> None:
        self._sem("setHtOnOff", 1 if value else 0)

    def read_blanked(self) -> bool:
        return bool(self._sem("getBeamBlanking"))

    def write_blanked(self, value: bool) -> None:
        self._sem("setBeamBlanking", 1 if value else 0)

    def read_working_distance(self) -> float:
        dac = self._sem("getObjectLens")
        wd_mm = (
            SEM_WORKING_DISTANCE_MM_REF
            - (SEM_WORKING_DISTANCE_DAC_REF - dac) / SEM_WORKING_DISTANCE_DAC_PER_MM
        )
        return wd_mm * constants.MILLIMETRE_TO_METRE

    def write_working_distance(self, value: float) -> None:
        wd_mm = value * constants.METRE_TO_MILLIMETRE
        dac = (wd_mm - SEM_WORKING_DISTANCE_MM_REF) * SEM_WORKING_DISTANCE_DAC_PER_MM
        self._sem("setObjectLensAbs", int(round(dac + SEM_WORKING_DISTANCE_DAC_REF)))

    def read_current(self) -> float:
        return BEAM_CURRENTS[BeamType.ELECTRON][self._sem("getProbeNo") - 1]

    def write_current(self, value: float) -> None:
        probe, _ = nearest_beam_current(value, BeamType.ELECTRON)
        self._sem("setProbeNo", probe)

    def read_voltage(self) -> float:
        return self._sem("getAcceleratingVoltage") * 1000  # kV

    def write_voltage(self, value: float) -> None:
        self._sem("setAcceleratingVoltage", value / 1000)

    def read_hfw(self) -> float:
        return _hfw_of(self._sem("getMagnification"))

    def write_hfw(self, value: float) -> None:
        self._sem(
            "setMagnification",
            magnification_for(self._clip_hfw(value)),
            ok=(MAGNIFICATION_NEAREST,),
        )

    def read_scan_rotation(self) -> float:
        return np.radians(self._sem("getScanRotationAngle"))

    def write_scan_rotation(self, value: float) -> None:
        degrees = _degrees(value)
        if degrees:
            self._sem("setScanRotationSwitch", 1)
            self._sem("setScanRotationAngle", degrees)
        else:
            # zero, then the rotation switched off
            self._sem("setScanRotationAngle", 0)
            self._sem("setScanRotationSwitch", 0)

    def read_shift(self) -> Point:
        x = self._sem("getImageShiftX") * SEM_SHIFT_METRES_PER_DAC
        y = self._sem("getImageShiftY") * SEM_SHIFT_METRES_PER_DAC
        return self._signed_shift(Point(x, y))

    def write_shift(self, value: Point) -> None:
        value = self._signed_shift(value)
        low, high = SEM_SHIFT_DAC_RANGE
        steps = [round(v / SEM_SHIFT_METRES_PER_DAC) for v in (value.x, value.y)]
        x, y = (int(np.clip(s, low, high)) for s in steps)
        if [x, y] != steps:
            logging.warning(f"SEM beam shift {value} clipped to DAC ({x}, {y})")
        self._sem("setAbsoluteImageShiftX", x)
        self._sem("setAbsoluteImageShiftY", y)

    def metadata_shift(self) -> ParameterMetadata:
        reach = SEM_SHIFT_DAC_RANGE[1] * SEM_SHIFT_METRES_PER_DAC
        limit = RangeLimit(min=-reach, max=reach)
        return ParameterMetadata(limits={"x": limit, "y": limit})

    def read_stigmation(self) -> Point:
        return Point(self._sem("getOlStigmatorX"), self._sem("getOlStigmatorY"))

    def read_resolution(self) -> Tuple[int, int]:
        return self._resolution

    def write_resolution(self, value: Sequence[int]) -> None:
        frame = nearest_resolution(tuple(value), list(SEM_PHOTO_SIZES))
        if frame != tuple(value):
            logging.warning(
                f"SEM resolution {tuple(value)} is not available; using {frame}."
            )
        self._sem("setPhotoScanMode2", 0, SEM_SCAN_MODE)  # photo type NORMAL, on Fine2
        self._sem("setPhotoImageSize", SEM_PHOTO_SIZES[frame])
        self._resolution = frame

    def metadata_resolution(self) -> ParameterMetadata:
        return ParameterMetadata(choices=list(SEM_PHOTO_SIZES))

    def read_dwell_time(self) -> float:
        self._sem("setScanMode", SEM_SCAN_MODE)
        status = self._sem("getImageStatus")
        return SEM_SCAN_SPEEDS[int(status["RATE_NO"])]

    def write_dwell_time(self, value: float) -> None:
        speed = min(SEM_SCAN_SPEEDS, key=lambda s: abs(SEM_SCAN_SPEEDS[s] - value))
        self._sem("setScanSpeed", SEM_SCAN_MODE, speed, 1)

    def metadata_dwell_time(self) -> ParameterMetadata:
        dwells = sorted(SEM_SCAN_SPEEDS.values())
        return ParameterMetadata(
            limits=RangeLimit(min=dwells[0], max=dwells[-1]), choices=dwells
        )

    def read_detector_type(self) -> Optional[str]:
        # the layout first, as the old driver did: the views are where it expects
        self.connection.set_view_layout()
        signal = self._fib("getViewSignal", self.connection.view(BeamType.ELECTRON))
        return self.connection.sdk.command.EViewSignal(signal).name

    def write_detector_type(self, value: str) -> None:
        signal = self._detector_signal(value)
        if signal is None:
            logging.warning(f"No SEM detector {value!r}; not set.")
            return
        self.connection.set_view_layout()
        self._fib("setViewSignal", self.connection.view(BeamType.ELECTRON), signal)

    def metadata_detector_type(self) -> ParameterMetadata:
        return ParameterMetadata(choices=[s.name for s in self._detector_signals()])

    def _detector_signals(self) -> List[Any]:
        signals = self.connection.sdk.command.EViewSignal
        return [signals(index) for index in SEM_DETECTOR_SIGNALS]

    def _detector_signal(self, name: str) -> Optional[Any]:
        return next((s for s in self._detector_signals() if s.name == name), None)

    def _signal_settings(self) -> Dict[str, Any]:
        return self._sem("getSignalContrastBrightness", 1)

    def read_detector_contrast(self) -> float:
        return int(self._signal_settings()["CONTRAST"]) / 4095

    def write_detector_contrast(self, value: float) -> None:
        self._sem("setContrast", int(4095 * value))

    def read_detector_brightness(self) -> float:
        return 1.0 - int(self._signal_settings()["BRIGHTNESS"]) / 4095

    def write_detector_brightness(self, value: float) -> None:
        self._sem("setBrightness", int(4095 * (1 - value)))

    # -- imaging ---------------------------------------------------------------------

    def _grab(self) -> np.ndarray:
        """A photo: JEOL writes it to a TIFF file, which is read and deleted."""
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            path = tmp.name
        try:
            self._sem("setScanMode", SEM_SCAN_MODE)
            status = self._sem("getImageStatus")
            self._sem("setPhotoIntegration", 0)
            if int(status["LIVE_FREEZE"]) == 0:
                self._sem("setLive")
            self._sem("execExternalPhoto2", os.path.splitext(path)[0], 2, 0)
            return images.read_photo(path, wait=self.connection.wait)
        finally:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def _autocontrast(self, reduced_area: Optional[FibsemRectangle]) -> None:
        # JEOL's ACB works on the whole frame: the area is not used
        with self._lock:
            self.connection.select_view(BeamType.ELECTRON)
            self._sem("setLive")
            self._sem("execACB")
            self.connection.wait(AUTOCONTRAST_WAIT)
            self._sem("setFreeze")

    def _auto_focus(self, reduced_area: Optional[FibsemRectangle]) -> None:
        self._sem("execAutoFocus", 0)  # fine mode

    def _live_frame(self) -> FibsemImage:
        """A frame of the SEM's own live scan, read from its image buffer (Windows
        only: ``sem_live`` reads the imaging process's memory)."""
        # the old driver's order: the SEM's view selected and the state read first
        self.connection.select_view(self.beam_type)
        state = self.parent.get_microscope_state(beam_type=self.beam_type)
        sem_live = self.connection.sdk.sem_live
        pid = sem_live.find_pid()
        if pid is None:
            raise RuntimeError("The SEM imaging process was not found.")
        handle = sem_live.get_handle(pid)
        if not handle:
            raise RuntimeError("The SEM imaging process could not be opened.")
        buffers = sem_live.locate_buffers(handle)
        if not buffers:
            raise RuntimeError("No SEM image buffers were found.")
        fast, slow = buffers[0], buffers[1] if len(buffers) > 1 else buffers[0]
        self._sem("setScanMode", SEM_SCAN_MODE)
        speed = int(self._sem("getImageStatus")["RATE_NO"])
        self._sem("setLive")
        address, size = fast if speed > 2 else slow
        data = sem_live.read_image(handle, address, size)
        if data is None:
            raise RuntimeError("The SEM live frame could not be read.")
        return self._image(np.asarray(data), state, None)


class JeolIonBeam(JeolBeam):
    """The FIB, on the FIB client.

    Not here, because the API has none: blanking, stigmation, a focus routine, the
    detector mode and presets. Its resolution and dwell time are one setting on JEOL,
    the scan type: each write picks the scan type that keeps the other.
    """

    def __init__(
        self,
        parent: JeolMicroscope,
        connection: JeolConnection,
        resources: Optional[Resources] = None,
    ):
        super().__init__(BeamType.ION, parent, connection, resources)

    def read_on(self) -> bool:
        return bool(self._fib("getFibGunHtSw"))

    def write_on(self, value: bool) -> None:
        self._fib("setFibGunHtSw", bool(value))
        if value:
            self.connection.wait(ION_BEAM_ON_WAIT)

    def read_working_distance(self) -> float:
        voltage = self._fib("getFibOlVoltage")
        wd_mm = (
            FIB_WORKING_DISTANCE_MM_REF
            + (voltage - FIB_WORKING_DISTANCE_VOLTAGE_REF)
            / FIB_WORKING_DISTANCE_VOLTAGE_PER_MM
        )
        return wd_mm * constants.MILLIMETRE_TO_METRE

    def write_working_distance(self, value: float) -> None:
        wd_mm = value * constants.METRE_TO_MILLIMETRE
        voltage = (
            wd_mm - FIB_WORKING_DISTANCE_MM_REF
        ) * FIB_WORKING_DISTANCE_VOLTAGE_PER_MM
        self._fib(
            "setFibOlVoltage", int(round(voltage + FIB_WORKING_DISTANCE_VOLTAGE_REF))
        )

    def read_current(self) -> float:
        return BEAM_CURRENTS[BeamType.ION][self._fib("getFibBeamMode") - 1]

    def write_current(self, value: float) -> None:
        mode, _ = nearest_beam_current(value, BeamType.ION)
        self._fib("setFibBeamMode", mode)

    def read_voltage(self) -> float:
        return float(self._fib("getFibColumnVoltage"))

    def write_voltage(self, value: float) -> None:
        self._fib("setFibColumnVoltage", int(value))

    def read_hfw(self) -> float:
        return _hfw_of(self._fib("getFibMagnification"))

    def write_hfw(self, value: float) -> None:
        self._fib(
            "setFibMagnification",
            magnification_for(self._clip_hfw(value)),
            ok=(MAGNIFICATION_NEAREST,),
        )

    def read_scan_rotation(self) -> float:
        return np.radians(self._fib("getFibScanRotation"))

    def write_scan_rotation(self, value: float) -> None:
        self._fib("setFibScanRotation", _degrees(value))

    def read_shift(self) -> Point:
        x = self._fib("getFibFineShiftX") * constants.MILLI_TO_SI
        y = self._fib("getFibFineShiftY") * constants.MILLI_TO_SI
        return self._signed_shift(Point(x, y))

    def write_shift(self, value: Point) -> None:
        value = self._signed_shift(value)
        self._fib("setFibFineShiftXAbs", value.x * constants.SI_TO_MILLI)
        self._fib("setFibFineShiftYAbs", value.y * constants.SI_TO_MILLI)

    # -- the scan type: resolution and dwell time ------------------------------------

    def _scan(self) -> Tuple[int, float, int, int]:
        """The scan type, frame time, width and height of the scan memory in use."""
        return self._fib("getFibScanType", self.connection.fib_scan_mode)

    def _set_scan_type(self, number: int) -> None:
        mode = FIB_SCAN_TYPES[number][2]
        self._fib("setFibScanType", mode, number)
        self.connection.fib_scan_mode = mode

    def read_resolution(self) -> Tuple[int, int]:
        _, _, width, height = self._scan()
        return (int(width), int(height))

    def write_resolution(self, value: Sequence[int]) -> None:
        frame = nearest_resolution(tuple(value), self._resolutions())
        if frame != tuple(value):
            logging.info(
                f"FIB resolution {tuple(value)} is not available; using {frame}."
            )
        self._set_scan_type(fib_scan_type(frame, self.read_dwell_time()))
        # the dwell times listed are the scan types at this resolution
        self.dwell_time.refresh_metadata()

    def metadata_resolution(self) -> ParameterMetadata:
        return ParameterMetadata(choices=self._resolutions())

    @staticmethod
    def _resolutions() -> List[Tuple[int, int]]:
        return sorted({size for size, _, _ in FIB_SCAN_TYPES.values()})

    def read_dwell_time(self) -> float:
        number = self._scan()[0]
        return FIB_SCAN_TYPES[number][1]

    def write_dwell_time(self, value: float) -> None:
        self._set_scan_type(fib_scan_type(self.read_resolution(), value))

    def metadata_dwell_time(self) -> ParameterMetadata:
        # the scan types at the resolution set now (write_resolution refreshes it)
        dwells = [dwell for _, dwell, _ in FIB_SCAN_TYPES.values()]
        resolution = self.read_resolution()
        here = sorted(d for size, d, _ in FIB_SCAN_TYPES.values() if size == resolution)
        return ParameterMetadata(
            limits=RangeLimit(min=min(dwells), max=max(dwells)), choices=here
        )

    # -- the detector: one ------------------------------------------------------------

    def read_detector_type(self) -> Optional[str]:
        self.connection.set_view_layout()
        signal = self._fib("getViewSignal", self.connection.view(BeamType.ION))
        return self.connection.sdk.command.EViewSignal(signal).name

    def write_detector_type(self, value: str) -> None:
        signal = self.connection.sdk.command.EViewSignal(FIB_DETECTOR_SIGNAL)
        if value != signal.name:
            logging.warning(
                f"No FIB detector {value!r}; the FIB has {signal.name!r}. Not set."
            )
            return
        self.connection.set_view_layout()
        self._fib("setViewSignal", self.connection.view(BeamType.ION), signal)

    def metadata_detector_type(self) -> ParameterMetadata:
        signal = self.connection.sdk.command.EViewSignal(FIB_DETECTOR_SIGNAL)
        return ParameterMetadata(choices=[signal.name])

    def read_detector_contrast(self) -> float:
        return (self._fib("getFibContrast") + 2048) / (2048 + 2047)

    def write_detector_contrast(self, value: float) -> None:
        self._fib("setFibContrast", int(value * (2047 + 2048)) - 2048)

    def read_detector_brightness(self) -> float:
        return 1.0 - (self._fib("getFibBrightness") + 512) / (4095 + 512)

    def write_detector_brightness(self, value: float) -> None:
        self._fib("setFibBrightness", int((1 - value) * (4095 + 512)) - 512)

    # -- imaging ---------------------------------------------------------------------

    def _grab(self) -> np.ndarray:
        """One scan of the scan memory in use, read back from the display."""
        mode = self.connection.fib_scan_mode
        _, frame_time, _, _ = self._scan()
        self._fib("startFibOneShotScan", mode)
        self.connection.wait(frame_time + FIB_SCAN_MARGIN)
        payload = self._fib("getDispImage", False)
        return images.decode_display_image(payload)

    def _autocontrast(self, reduced_area: Optional[FibsemRectangle]) -> None:
        # JEOL's ACB is on the SEM client and acts on the selected view, here the FIB's
        with self._lock:
            self.connection.select_view(BeamType.ION)
            self._fib("setFibFreezeMode", 0)  # live
            self._sem("execACB")
            self.connection.wait(AUTOCONTRAST_WAIT)
            self._fib("setFibFreezeMode", 1)  # frozen


# -- the stage ----------------------------------------------------------------------

# The sign of each JEOL stage axis against fibsem's, which is ThermoFisher's: the
# movement and projection maths are ThermoFisher's, so a -1 here turns that axis round
# where positions cross between JEOL and fibsem, and nothing above has to know. A sign
# is its own inverse. x and y measured 2026-09-29; z, r and t are not measured yet.
STAGE_AXIS_SIGNS: Dict[str, float] = {
    "x": -1.0,
    "y": -1.0,
    "z": -1.0,
    "r": 1.0,
    "t": 1.0,
}

# The stage limits in fibsem's frame, metres and degrees.
STAGE_LIMITS: Dict[str, RangeLimit] = {
    "x": RangeLimit(min=-55e-3, max=10e-3),
    "y": RangeLimit(min=-15e-3, max=15e-3),
    "z": RangeLimit(min=-3.0e-3, max=3.0e-3),
    "r": RangeLimit(min=-190, max=190),
    "t": RangeLimit(min=-40, max=55),
}

# How long a stage move may take, and how often it is looked at, in seconds.
STAGE_MOVE_TIMEOUT = 120.0
STAGE_MOVE_POLL_INTERVAL = 0.5


def _signed(value: Optional[float], axis: str) -> Optional[float]:
    return None if value is None else STAGE_AXIS_SIGNS[axis] * value


def from_jeol_stage_position(position: Dict[str, float]) -> FibsemStagePosition:
    """A JEOL stage position (``getStagePosition``: metres and radians) in fibsem's."""
    return FibsemStagePosition(
        x=_signed(position["x"], "x"),
        y=_signed(position["y"], "y"),
        z=_signed(position["fz"], "z"),
        r=_signed(position["rz"], "r"),
        t=_signed(position["rx"], "t"),
        coordinate_system="RAW",
    )


def to_jeol_stage_position(position: FibsemStagePosition) -> Dict[str, float]:
    """A position with every axis set, as ``setMultiAxisStagePositionAbs`` takes it:
    millimetres and degrees."""
    return {
        "x": _signed(position.x, "x") * constants.METRE_TO_MILLIMETRE,
        "y": _signed(position.y, "y") * constants.METRE_TO_MILLIMETRE,
        "fz": _signed(position.z, "z") * constants.METRE_TO_MILLIMETRE,
        "rz": _signed(position.r, "r") * constants.RADIANS_TO_DEGREES,
        "rx": _signed(position.t, "t") * constants.RADIANS_TO_DEGREES,
    }


class JeolStage(Stage):
    """The JEOL stage (BSS), in fibsem's frame.

    - ``read_position``: ``getStagePosition``, signed into fibsem's axes.
    - ``_move_absolute``: one ``setMultiAxisStagePositionAbs`` of every axis, an axis
      the move leaves None taken from where the stage is; then waits for the stage to
      stop (the move is asynchronous).
    - ``_move_relative``: where the stage is plus the offset, moved to absolutely.

    The stage has a rotation axis, but reaches the FIB by tilting alone, at the SEM
    rotation: ``poses`` says so. It has no homing (the native UI does it) and no link
    of z to the working distance, so neither is here.
    """

    def __init__(
        self,
        parent: JeolMicroscope,
        connection: JeolConnection,
        resources: Optional[Resources] = None,
    ):
        super().__init__(parent=parent, resources=resources)
        self.connection = connection

    def read_position(self) -> FibsemStagePosition:
        return from_jeol_stage_position(self.connection.fib_call("getStagePosition"))

    def metadata_position(self) -> ParameterMetadata:
        return ParameterMetadata(limits=axis_limits_from_degrees(STAGE_LIMITS))

    def poses(
        self, rotation_reference: float, shuttle_pre_tilt: float, fib_column_tilt: float
    ) -> Dict[str, FibsemStagePosition]:
        """SEM and FIB at the reference rotation: the FIB pose is a tilt, not a turn."""
        return rotating_stage_poses(
            rotation_reference, shuttle_pre_tilt, fib_column_tilt, rotates=False
        )

    def _move_absolute(self, position: FibsemStagePosition) -> None:
        current = self.position.get_value()
        full = FibsemStagePosition(
            **{
                axis: getattr(
                    current if getattr(position, axis) is None else position, axis
                )
                for axis in ("x", "y", "z", "r", "t")
            }
        )
        target = to_jeol_stage_position(full)
        command = self.connection.sdk.command
        axes = command.EStageAxisFlags
        logging.info(f"Moving stage to {full}.")
        self.connection.fib_call(
            "setMultiAxisStagePositionAbs",
            command.EStageType.BSS,
            axes.X | axes.Y | axes.T | axes.R | axes.FineZ,
            x=target["x"],
            y=target["y"],
            t=target["rx"],
            r=target["rz"],
            fz=target["fz"],
        )
        self._wait_until_stopped()

    def _wait_until_stopped(self) -> None:
        start = time.monotonic()
        while self.connection.fib_call("getStageMoveStatus"):
            if time.monotonic() - start > STAGE_MOVE_TIMEOUT:
                raise TimeoutError(
                    f"The stage was still moving after {STAGE_MOVE_TIMEOUT} s."
                )
            self.connection.wait(STAGE_MOVE_POLL_INTERVAL)

    def _move_relative(self, delta: FibsemStagePosition) -> None:
        current = self.position.get_value()
        self._move_absolute(current + delta)


# -- builders -----------------------------------------------------------------------


def build_jeol_beam(entry: DeviceEntry, context: BuildContext) -> JeolBeam:
    """The column an ``electron`` or ``ion`` entry names."""
    microscope = context.microscope
    if entry.name == "electron":
        return JeolElectronBeam(microscope, microscope._connection).connect()
    if entry.name == "ion":
        return JeolIonBeam(microscope, microscope._connection).connect()
    raise ValueError("a JEOL beam is named 'electron' or 'ion'")


def build_jeol_stage(entry: DeviceEntry, context: BuildContext) -> JeolStage:
    microscope = context.microscope
    stage = JeolStage(microscope, microscope._connection)
    stage.name = entry.name
    return stage.connect()
