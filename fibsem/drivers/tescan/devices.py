"""The Tescan (SharkSEM) beams and stage as devices.

``TescanBeam`` implements the ``Beam`` device with what were the beam branches of
``TescanMicroscope._get``/``_set``, moved as they were: the device makes the SDK calls
those branches made, in the same order, and logs the same messages
(``tests/fixtures/tescan_beam_calls.json``). ``TescanMicroscope`` builds one per enabled
column at connect and routes its beam keys to them, and builds a ``TescanStage``, which
converts between fibsem's stage frame and Tescan's, and routes its stage keys and moves
to it. It builds both from its device entries (``fibsem.devices.entries``) with the
builders at the end of this module, which the Tescan driver record names.

Each device is handed the microscope's `TescanConnection` and makes its SDK calls
through it. The vendor beams are ``SEM`` and ``FIB`` on it. SharkSEM is one socket, so
every read and write holds the connection's ``lock``. This module imports nothing from
the SDK; the Tescan backend's guarded import is the only one.
"""

from __future__ import annotations

import logging
import threading
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np

import fibsem.constants as constants
from fibsem.devices.beam import UNBOUNDED_SHIFT_LIMITS, Beam
from fibsem.devices.core import ParameterMetadata, Resources
from fibsem.devices.entries import beam_type_of
from fibsem.devices.stage import AXIS_UNITS, UNLIMITED, Stage
from fibsem.drivers.tescan import images
from fibsem.structures import (
    BeamSettings,
    BeamType,
    FibsemImage,
    FibsemRectangle,
    FibsemStagePosition,
    ImageSettings,
    Point,
    RangeLimit,
)

if TYPE_CHECKING:
    from fibsem.drivers.registry import BuildContext
    from fibsem.drivers.tescan.connection import TescanConnection
    from fibsem.drivers.tescan.microscope import TescanMicroscope
    from fibsem.structures import DeviceEntry

# The "current" choices, a fixed list.
_CURRENT_CHOICES: Dict[BeamType, List[float]] = {
    BeamType.ELECTRON: [1.0e-12],
    BeamType.ION: [20e-12, 60e-12, 0.2e-9, 0.74e-9, 2.0e-9, 7.6e-9, 28.0e-9, 120e-9],
}

_NOT_SETTABLE = ParameterMetadata(settable=False)


class TescanBeam(Beam):
    """A Tescan column: ``SEM`` or ``FIB`` on the connection.

    A write prepares the beam first (turns it on, stops the scan, waits until it is
    not busy).

    What the Tescan API refuses is still written, so the caller gets its message, but
    reads as not settable: the current on both columns, the ion column's voltage
    (set by preset), and the resolution, dwell time and stigmation, which are read
    from the last image.

    Not here, because the Tescan API does not have them: the ion column's working
    distance and ``detector_mode`` (the microscope's wrappers return None for
    them), ``blanked``, ``plasma_gas`` and the scan modes.

    ``preserve_settings_on_preset_change``: restore the rotation, field of view and
    shift across a preset change; when False, the preset's own values stand.
    """

    def __init__(
        self,
        beam_type: BeamType,
        parent: TescanMicroscope,
        connection: TescanConnection,
        resources: Optional[Resources] = None,
    ):
        from fibsem.drivers.tescan.microscope import (
            TESCAN_PRESERVE_SETTINGS_ON_PRESET_CHANGE,
        )

        super().__init__(beam_type, parent=parent, resources=resources)
        self.connection = connection
        self.preserve_settings_on_preset_change: bool = (
            TESCAN_PRESERVE_SETTINGS_ON_PRESET_CHANGE
        )
        # The detector the column images with, set by write_detector_type.
        self._detector: Any = None
        # The beam settings the last image reported, for what the API cannot read.
        self._cache = BeamSettings(beam_type)

    @property
    def _lock(self):
        return self.connection.lock

    @property
    def _beam(self) -> Any:
        """The vendor beam, looked up on every call."""
        return self.connection.beam(self.beam_type)

    def _prepared(self) -> Any:
        """The vendor beam, prepared for a write (`TescanConnection.prepare_beam`)."""
        beam = self._beam
        self.connection.prepare_beam(self.beam_type)
        return beam

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
            logging.debug(f"Electron beam working distance set to {value} m.")

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
            logging.debug(f"Electron beam current set to {value} A.")

    def metadata_current(self) -> ParameterMetadata:
        # Not settable on either column: the ion current is the preset's, and the
        # electron current is not set directly on a Tescan either.
        return ParameterMetadata(
            choices=list(_CURRENT_CHOICES[self.beam_type]), settable=False
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
            logging.debug(f"Electron beam voltage set to {value} V.")

    def metadata_voltage(self) -> ParameterMetadata:
        return ParameterMetadata(settable=self.beam_type is BeamType.ELECTRON)

    # -- field of view, scan rotation, shift -----------------------------------------

    def read_hfw(self) -> float:
        with self._lock:
            return self._beam.Optics.GetViewfield() * constants.MILLIMETRE_TO_METRE

    def write_hfw(self, value: float) -> None:
        from fibsem.drivers.tescan.microscope import LIMITS

        with self._lock:
            beam = self._prepared()
            limits = LIMITS[self.beam_type]["hfw"]
            value = np.clip(value, limits[0], limits[1])
            beam.Optics.SetViewfield(value * constants.METRE_TO_MILLIMETRE)
            logging.debug(f"{self.beam_type.name} HFW set to {value} m.")

    def metadata_hfw(self) -> ParameterMetadata:
        from fibsem.drivers.tescan.microscope import LIMITS

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
            logging.debug(
                f"{self.beam_type.name} scan rotation set to {value} radians."
            )

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
            point = value * constants.METRE_TO_MILLIMETRE
            beam.Optics.SetImageShift(point.x, point.y)
            logging.debug(f"{self.beam_type.name} beam shift set to {value}.")

    def metadata_shift(self) -> ParameterMetadata:
        # the TESCAN API doesn't report an image-shift range
        return ParameterMetadata(limits=UNBOUNDED_SHIFT_LIMITS)

    def _shift_by(self, dx: float, dy: float) -> None:
        # invert direction for scan rotated images...
        if np.isclose(self.scan_rotation.get_value(), np.pi):
            dx, dy = -dx, -dy
        logging.info(f"{self.beam_type.name} shifting by ({dx}, {dy})")
        super()._shift_by(dx, dy)

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

    # -- presets: the last one activated, and activation, on either column ----------
    # The ion column is set by preset (``beam_uses_presets``); the electron column's
    # can be activated too, as set_beam_settings does when it restores one.

    def read_preset(self) -> Optional[str]:
        return self._cache.preset

    def write_preset(self, value: str) -> None:
        with self._lock:
            beam = self._prepared()
            self._activate_preset(beam, value)

    def metadata_preset(self) -> ParameterMetadata:
        return ParameterMetadata(choices=self._get_presets())

    # -- the detector: each column has its own, so no channel to claim ---------------

    def read_detector_type(self) -> Optional[str]:
        with self._lock:
            detector = self._beam.Detector.Get(Channel=0)
        return None if detector is None else detector.name

    def write_detector_type(self, value: str) -> None:
        with self._lock:
            beam = self._prepared()
            detector = self._get_detector(value)
            if detector is None:
                logging.warning(f"Detector {value} not found for {self.beam_type}.")
                return
            beam.Detector.Set(Channel=0, Detector=detector)
            self._detector = detector
            logging.debug(f"{self.beam_type.name} detector type set to {value}.")

    def metadata_detector_type(self) -> ParameterMetadata:
        detectors = self.list_detectors()
        return ParameterMetadata(choices=[d.name for d in detectors])

    def _gain_black(self) -> Any:
        with self._lock:
            return self._beam.Detector.GetGainBlack(Detector=self._detector)

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
            active_detector = self._detector
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
            logging.debug(f"{self.beam_type.name} {key} set to {value}.")

    # -- imaging and the autofunctions -----------------------------------------------
    # TescanMicroscope's acquire_image, autocontrast, auto_focus and live
    # view worker, moved as they are: the image is built from the frame's header, as
    # before, and the SDK call holds the connection lock for the whole frame.

    def _acquire(self, image_settings: Optional[ImageSettings]) -> FibsemImage:
        from fibsem.drivers.tescan import microscope as tescan

        microscope = self.parent
        connection = self.connection
        beam_type = self.beam_type
        settings = (
            image_settings if image_settings is not None else self.image_settings()
        )
        logging.debug(f"acquiring new {beam_type.name} image.")

        if image_settings is not None:
            connection.settle_after_electron_image(beam_type)

        # prepare the beam (turn on, stop scanning)
        beam = connection.prepare_beam(beam_type)

        dwell_time_ns = settings.dwell_time * constants.SI_TO_NANO
        image_width, image_height = settings.resolution

        # Only apply settings if image_settings was provided
        if image_settings is not None:
            hfw = microscope.get_field_of_view(beam_type=beam_type)
            if not np.isclose(hfw, settings.hfw, atol=1e-6):
                microscope.set_field_of_view(settings.hfw, beam_type)

        image_roi = settings.reduced_area
        detector = self._detector
        with self._lock:
            if image_roi is not None:
                left, top, right, bottom = images.to_tescan_image_roi(
                    rect=image_roi, image_shape=(image_width, image_height)
                )
                image = beam.Scan.AcquireROI(
                    Detector=detector,
                    Width=image_width,
                    Height=image_height,
                    Left=left,
                    Top=top,
                    Right=right,
                    Bottom=bottom,
                    DwellTime=dwell_time_ns,
                )
            else:
                image = beam.Scan.AcquireImage(
                    Detector=detector,
                    Bpp=tescan.Bpp.Grayscale_8_bit,
                    Width=image_width,
                    Height=image_height,
                    DwellTime=dwell_time_ns,
                )

        if image is None:
            raise ValueError("Failed to acquire image from microscope.")

        fibsem_image = images.image_from_tescan(image, settings, microscope.stage)
        fibsem_image.metadata.image_settings.beam_type = deepcopy(beam_type)

        # the image's beam state, for what the API cannot read
        state = fibsem_image.metadata.microscope_state
        if beam_type is BeamType.ELECTRON:
            beam_state = state.electron_beam
        else:
            beam_state = state.ion_beam
        cache = self._cache
        cache.dwell_time = settings.dwell_time
        cache.resolution = settings.resolution
        cache.stigmation = beam_state.stigmation
        cache.preset = beam_state.preset

        # the manufacturer's details are only in the image header
        info = microscope.system.info
        if info.model == "Unknown":
            info.model = image.Header["MAIN"]["DeviceModel"]
            info.serial_number = image.Header["MAIN"]["SerialNumber"]
            info.software_version = image.Header["MAIN"]["SoftwareVersion"]

        return fibsem_image

    def _autocontrast(self, reduced_area: Optional[FibsemRectangle]) -> None:
        # The SDK's AutoSignal works on the whole frame: the area is not used.
        beam = self.connection.prepare_beam(beam_type=self.beam_type)
        logging.debug(f"Running autocontrast on {self.beam_type.name}.")
        with self._lock:
            beam.Detector.AutoSignal(Detector=self._detector)

    def _has_auto_focus(self) -> bool:
        # The ion column has no working distance or focus control: no auto_focus,
        # so microscope.auto_focus(ION) falls to the focus sweep, which warns and
        # skips (the working distance can't be set).
        return self.beam_type is BeamType.ELECTRON

    def _auto_focus(self, reduced_area: Optional[FibsemRectangle]) -> None:
        # AutoWDFine, on the electron column only.
        beam = self.connection.prepare_beam(beam_type=self.beam_type)
        with self._lock:
            beam.AutoWDFine(self._detector)

    def _live(self, stop: threading.Event) -> None:
        # Tescan has no streaming API: re-acquire with the current settings until
        # stopped. acquire holds the connection lock per frame, so other threads'
        # calls queue between frames.
        try:
            while not stop.is_set():
                image = self.acquire()
                if stop.is_set():
                    break
                self.live_frame.emit(image)
        except Exception as e:
            logging.error(f"Error in TESCAN acquisition worker: {e}")

    # -- the column's presets and detectors ------------------------------------------

    def _get_presets(self) -> List[str]:
        with self._lock:
            presets = self._beam.Preset.Enum()
        return sorted(presets)

    def list_detectors(self) -> List[Any]:
        """The column's detectors, as the SDK lists them."""
        with self._lock:
            return self._beam.Detector.Enum()

    def _get_detector(self, detector_type: Union[Any, str]) -> Optional[Any]:
        """The SDK detector named ``detector_type``, or None if the column has none."""
        from fibsem.drivers.tescan import microscope as tescan

        if isinstance(detector_type, tescan.Detector):
            detector_type = detector_type.name

        for detector in self.list_detectors():
            if detector_type == detector.name:
                logging.debug(f"Found detector {detector.name}, index {detector.index}")
                return detector
        return None

    def _activate_preset(self, beam: Any, preset_name: str) -> None:
        """Activate a preset, preserving rotation/FOV/shift when
        ``preserve_settings_on_preset_change`` is set."""
        beam_type = self.beam_type
        # Check if preset is available
        with self._lock:
            available = beam.Preset.IsAvailable(preset_name)
        if not available:
            logging.warning(f"Preset {preset_name} not available for {beam_type}.")
            return

        preserve_settings = self.preserve_settings_on_preset_change

        # Save current settings if they should be preserved
        image_rotation = None
        view_field = None
        image_shift_x = None
        image_shift_y = None

        if preserve_settings:
            with self._lock:
                image_rotation = beam.Optics.GetImageRotation()
                view_field = beam.Optics.GetViewfield()
                image_shift_x, image_shift_y = beam.Optics.GetImageShift()
            logging.debug(f"Image rotation before changing preset: {image_rotation}.")
            logging.debug(f"FOV before changing preset: {view_field}.")
            logging.debug(
                f"XY shift before changing preset: {image_shift_x}, {image_shift_y}."
            )

        try:
            # Activate preset
            with self._lock:
                beam.Preset.Activate(preset_name)
            logging.info(f"Preset {preset_name} activated for {beam_type}.")

            # Wait for preset to fully apply
            self.connection.wait_for_beam_ready(
                beam, beam_type, operation="preset activation"
            )

            # Update internal state
            self._cache.preset = preset_name

        except Exception as e:
            logging.error(
                f"Failed to activate preset {preset_name} for {beam_type}: {e}"
            )
            raise  # Re-raise to ensure finally block runs and then propagate error

        finally:
            # Restore settings if requested
            if preserve_settings:
                try:
                    with self._lock:
                        beam.Optics.SetImageRotation(image_rotation)
                        beam.Optics.SetViewfield(view_field)
                        beam.Optics.SetImageShift(image_shift_x, image_shift_y)
                    logging.debug(
                        f"Restored image rotation after changing preset to {image_rotation}."
                    )
                    logging.debug(
                        f"Restored FOV after changing preset to {view_field}."
                    )
                    logging.debug(
                        f"Restored XY shift after changing preset to {image_shift_x}, {image_shift_y}."
                    )
                except Exception as restore_error:
                    logging.error(f"Failed to restore beam settings: {restore_error}")


STAGE_FRAME_TESCAN = "tescan_native"
"""Tescan's own stage frame, stamped on an image from a Tescan with no stage device.

x and y run opposite the image, y rides on the tilt module and z is chamber-vertical,
+z down. A stage device converts into fibsem's frame and stamps that.
"""


TILT_AXIS_Z = 29.8e-3
"""z′₀, in metres: the Tescan z reading at which the converted frame puts the tilt axis.

The working z′ in a hardware session log (29.69 to 30.04 mm over the session), until
the hardware session measures the eucentric z′. Not Tescan's
``eucentric_height``, which is the FIB working distance.
"""


def from_tescan_frame(
    native: FibsemStagePosition, tilt_axis_z: float = TILT_AXIS_Z
) -> FibsemStagePosition:
    """A position in Tescan's frame (metres, radians) in fibsem's frame.

    Tescan's x and y run opposite fibsem's; its y rides the tilt module and its z is
    chamber-vertical, +z down, while fibsem's y and z both ride the tilt:

        x = -x′,  y = -y′ - (z′ - z′₀)·sin t,  z = -(z′ - z′₀)·cos t
    """
    h = native.z - tilt_axis_z
    t = native.t
    return FibsemStagePosition(
        x=-native.x,
        y=-native.y - h * np.sin(t),
        z=-h * np.cos(t),
        r=native.r,
        t=t,
        coordinate_system=native.coordinate_system,
    )


def to_tescan_frame(
    position: FibsemStagePosition, tilt_axis_z: float = TILT_AXIS_Z
) -> FibsemStagePosition:
    """The inverse of :func:`from_tescan_frame`, for a position with every axis set."""
    t = position.t
    return FibsemStagePosition(
        x=-position.x,
        y=-(position.y - position.z * np.tan(t)),
        z=tilt_axis_z - position.z / np.cos(t),
        r=position.r,
        t=t,
        coordinate_system=position.coordinate_system,
    )


class TescanStage(Stage):
    """The Tescan stage, in fibsem's frame, converting to Tescan's inside.

    ``position`` is in fibsem's frame (:func:`from_tescan_frame`); every
    ``Stage.MoveTo`` is in Tescan's. Through the conversion a position puts the sample
    where a ThermoFisher stage at that position would, so the shared movement and
    projection maths hold unchanged.

    - ``read_position``: ``Stage.GetPosition`` in mm and degrees, converted.
    - ``_move_absolute``: one ``Stage.MoveTo``. y and z convert together, so when one
      is None it is taken from where the stage is. When both are, y′ and z′ are left
      where they are too: a pose change alone is a pure tilt and rotation on the
      instrument, as it was before the conversion.
    - ``_move_relative``: reads where the stage is and moves to that plus the offset
      (SharkSEM moves are absolute); x, y or z the offset leaves None is left None, so a
      tilt alone is a pure tilt here too.

    ``tilt_axis_z`` is z′₀ (:data:`TILT_AXIS_Z`). Relative moves and anything at one
    pose don't depend on it; positions compared across tilts do.

    Fibsem has never read the Tescan stage's limits, so every axis is unlimited and
    the instrument refuses what it cannot reach. ``home()`` refers to the native UI, so
    ``homed`` is absent (its key reads None). z is not linked to
    the working distance, so ``linked`` reads False and ``link`` changes nothing.
    """

    def __init__(
        self,
        parent: TescanMicroscope,
        connection: TescanConnection,
        resources: Optional[Resources] = None,
        tilt_axis_z: float = TILT_AXIS_Z,
    ):
        super().__init__(parent=parent, resources=resources)
        self.connection = connection
        self.tilt_axis_z = tilt_axis_z

    @property
    def _lock(self):
        return self.connection.lock

    def from_native(self, native: FibsemStagePosition) -> FibsemStagePosition:
        """A position Tescan reported (an image header's, say) in fibsem's frame."""
        return from_tescan_frame(native, self.tilt_axis_z)

    def native_delta(
        self, native: FibsemStagePosition, tilt: float
    ) -> FibsemStagePosition:
        """A move in Tescan's frame at ``tilt`` as the same move in fibsem's."""
        return FibsemStagePosition(
            x=-native.x,
            y=-native.y - native.z * np.sin(tilt),
            z=-native.z * np.cos(tilt),
            r=native.r,
            t=native.t,
        )

    def read_position(self) -> FibsemStagePosition:
        from fibsem.drivers.tescan.microscope import from_tescan_stage_position

        with self._lock:
            position = self.connection.sdk.Stage.GetPosition()
        return self.from_native(from_tescan_stage_position(position))

    def metadata_position(self) -> ParameterMetadata:
        return ParameterMetadata(limits={axis: UNLIMITED for axis in AXIS_UNITS})

    def read_linked(self) -> bool:
        return False

    def _link(self) -> None:
        # Nothing to link: ``link`` leaves the stage as it is and reports False, as
        # ``link_stage`` did before the stage was a device.
        logging.info("The Tescan stage is not linked to the working distance.")

    def _native(self, position: FibsemStagePosition) -> FibsemStagePosition:
        """``position`` in Tescan's frame, with None where the move leaves an axis."""
        x = None if position.x is None else -position.x
        y = z = None
        if position.y is not None or position.z is not None:
            # where the stage is, only for what the move leaves out
            given = (position.y, position.z, position.t)
            current = self.position.get_value() if None in given else position
            full = FibsemStagePosition(
                x=0.0,
                y=current.y if position.y is None else position.y,
                z=current.z if position.z is None else position.z,
                r=0.0,
                t=current.t if position.t is None else position.t,
            )
            native = to_tescan_frame(full, self.tilt_axis_z)
            y, z = native.y, native.z
        return FibsemStagePosition(x=x, y=y, z=z, r=position.r, t=position.t)

    def _move_absolute(self, position: FibsemStagePosition) -> None:
        from fibsem.drivers.tescan.microscope import to_tescan_stage_position

        logging.info(f"Moving stage to {position}.")
        x, y, z, r, t = to_tescan_stage_position(position=self._native(position))
        with self._lock:
            self.connection.sdk.Stage.MoveTo(x=x, y=y, z=z, rot=r, tiltx=t)
        logging.debug({"msg": "move_stage_absolute", "position": position.to_dict()})

    def _move_relative(self, delta: FibsemStagePosition) -> None:
        logging.info(f"Moving stage by {delta}.")
        current = self.position.get_value()
        # x, y or z the delta leaves None stays None, so a tilt alone stays a pure tilt
        target = current + delta
        for axis in ("x", "y", "z"):
            if getattr(delta, axis) is None:
                setattr(target, axis, None)
        logging.debug(f"Moving stage to {target}")
        self._move_absolute(target)
        logging.debug({"msg": "move_stage_relative", "position": delta.to_dict()})


def build_tescan_beam(entry: DeviceEntry, context: BuildContext) -> TescanBeam:
    """The column an ``electron`` or ``ion`` entry names."""
    beam_type = beam_type_of(entry)
    microscope = context.microscope
    return TescanBeam(beam_type, microscope, microscope._connection).connect()


def build_tescan_stage(entry: DeviceEntry, context: BuildContext) -> TescanStage:
    microscope = context.microscope
    stage = TescanStage(microscope, microscope._connection)
    stage.name = entry.name
    return stage.connect()
