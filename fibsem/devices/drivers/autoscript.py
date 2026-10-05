"""The AutoScript (Thermo Fisher) stage, beams, chamber, manipulator and gas injectors
as devices.

``AutoscriptStage`` implements the ``Stage`` device with what ``ThermoMicroscope``
does today, moved as-is, so the old call and the device make the same SDK calls in
the same order. ``AutoscriptCompustage`` is the same for a compustage (Arctis,
Hydra), ``AutoscriptBeam`` for the beam keys, and ``AutoscriptChamber``,
``AutoscriptManipulator`` and ``AutoscriptGasInjector`` for the vacuum, the needle and
the GIS. ``ThermoMicroscope`` builds them at connect and routes its keys and moves to
them; its old code stays until a session on an instrument confirms the devices.

The vendor stage is ``microscope._vendor_stage``, which the Thermo backend sets at
connect to ``specimen.stage`` or ``specimen.compustage`` (``microscope.stage`` is the
stage device); the vendor beams are under
``microscope.connection.beams``. This module imports the SDK only through
``fibsem.microscopes.autoscript``, which is where the guarded import lives, apart from
the SDK's ``Point`` inside a write, which the old branch imports there too.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import numpy as np

from fibsem.devices.beam import Beam
from fibsem.devices.chamber import Chamber
from fibsem.devices.core import ParameterMetadata, Resources
from fibsem.devices.gis import GasInjector
from fibsem.devices.manipulator import Manipulator
from fibsem.devices.stage import Stage, axis_limits_from_degrees, compustage_poses
from fibsem.structures import (
    BeamType,
    ChamberState,
    FibsemImage,
    FibsemManipulatorPosition,
    FibsemRectangle,
    FibsemStagePosition,
    ImageSettings,
    InsertableDeviceState,
    Point,
    RangeLimit,
    ScanMode,
)

if TYPE_CHECKING:
    from fibsem.microscopes.autoscript import ThermoMicroscope


class AutoscriptStage(Stage):
    """The AutoScript stage on an offset (non-compustage) mount.

    Each method is what the matching part of ``ThermoMicroscope`` does today:

    - ``read_position``: the ``stage_position`` branch of ``_get``, including setting the
      default coordinate system before every read;
    - ``read_homed`` / ``read_linked``: the ``stage_homed`` / ``stage_linked`` branches;
    - ``metadata_position``: ``_get_axis_limits``, which also says which axes exist;
    - ``_move_absolute``: ``move_stage_absolute``, with its working-distance restore
      and its axis restrictions;
    - ``_move_relative``: ``move_stage_relative``;
    - ``_home`` / ``_link``: the ``stage_home`` / ``stage_link`` branches of ``_set``.

    The old moves end by reading the position back; ``Stage.move_through`` does the
    same read, so a move here is the same calls end to end. ``_get_axis_limits`` gives
    r and t in degrees while positions are in radians; the metadata converts them, so
    limits and values share one unit.
    """

    compustage = False

    def __init__(self, parent: ThermoMicroscope, resources: Optional[Resources] = None):
        super().__init__(parent=parent, resources=resources)

    @property
    def _stage(self):
        """The vendor stage, looked up on each call as the old methods do."""
        return self.parent._vendor_stage

    def _to_autoscript(self, position: FibsemStagePosition):
        from fibsem.microscopes.autoscript import stage_position_to_autoscript

        return stage_position_to_autoscript(position, compustage=self.compustage)

    # -- position ----------------------------------------------------------------

    def read_position(self) -> FibsemStagePosition:
        from fibsem.microscopes.autoscript import stage_position_from_autoscript

        self._stage.set_default_coordinate_system(
            self.parent._default_stage_coordinate_system
        )
        return stage_position_from_autoscript(self._stage.current_position)

    def metadata_position(self) -> ParameterMetadata:
        return ParameterMetadata(
            limits=axis_limits_from_degrees(self.parent._get_axis_limits())
        )

    # -- homing and linking -----------------------------------------------------------

    def read_homed(self) -> bool:
        return self._stage.is_homed

    def read_linked(self) -> bool:
        return self._stage.is_linked

    # -- commands ---------------------------------------------------------------------

    def _move_absolute(self, position: FibsemStagePosition) -> None:
        from fibsem.microscopes.autoscript import MoveSettings

        # get current working distance, to be restored later
        wd = self.parent.get_working_distance(BeamType.ELECTRON)

        autoscript_position = self._to_autoscript(position)

        if self.parent._axis_restrictions_apply(
            position
        ):  # ONLY when restrictions are on
            autoscript_position.z = None
            autoscript_position.r = None

        logging.info(f"Moving stage to {position}.")
        self._stage.absolute_move(
            autoscript_position, MoveSettings(rotate_compucentric=True)
        )

        # restore working distance to adjust for microscope compensation
        if not self.compustage:
            self.parent.set_working_distance(wd, BeamType.ELECTRON)

        logging.debug({"msg": "move_stage_absolute", "position": position.to_dict()})

    def _move_relative(self, delta: FibsemStagePosition) -> None:
        logging.info(f"Moving stage by {delta}.")
        self._stage.relative_move(self._to_autoscript(delta))
        logging.debug({"msg": "move_stage_relative", "position": delta.to_dict()})

    def _home(self) -> None:
        logging.info("Homing stage...")
        self._stage.home()
        logging.info("Stage homed.")

    def _link(self) -> None:
        logging.info("Linking stage...")
        self._stage.link()
        logging.info("Stage linked.")


class AutoscriptCompustage(AutoscriptStage):
    """The AutoScript compustage: x, y, z and a tilt ``a`` in specimen coordinates.

    It differs from the offset stage in three places, each as the old code has it:
    positions convert to and from ``CompustagePosition`` (no r), an absolute move does
    not restore the working distance (it still reads it first, as today), and there
    is no linking: the old ``set("stage_link")`` logs and does nothing, so ``linked``
    is absent here and ``link()`` is unavailable. Its limits are the fixed compustage
    table ``_get_axis_limits`` returns, which has no r.
    """

    compustage = True

    def available_linked(self) -> bool:
        return False

    def poses(
        self, rotation_reference: float, shuttle_pre_tilt: float, fib_column_tilt: float
    ) -> Dict[str, FibsemStagePosition]:
        return compustage_poses(rotation_reference, shuttle_pre_tilt, fib_column_tilt)


def autoscript_stage_class(microscope: ThermoMicroscope) -> Type[AutoscriptStage]:
    """The driver class for the stage the Thermo backend found at connect."""
    return AutoscriptCompustage if microscope.stage_is_compustage else AutoscriptStage


def bind_autoscript_stage(
    microscope: ThermoMicroscope, resources: Optional[Resources] = None
) -> AutoscriptStage:
    """Build ``stage`` for a connected Thermo microscope."""
    return autoscript_stage_class(microscope)(microscope, resources).connect()


class AutoscriptBeam(Beam):
    """An AutoScript beam: ``connection.beams.electron_beam`` or ``.ion_beam``.

    Each parameter is the matching branch of ``ThermoMicroscope._get``/``_set`` moved
    as it is, so the old call and the device make the same SDK calls and log the same
    messages. The choices are ``ThermoMicroscope.get_available_values``'s.

    The detector is the active device's, so the detector parameters claim the imaging
    channel and select this beam's (``needs_channel``), as the old branches do under
    the lock. The scan commands are the old ``spot_mode``/``reduced_area``/
    ``full_frame`` keys; ``scanning_mode`` reads the vendor's scan mode, which nothing
    read before. The electron beam's ``angular_correction`` and ``tilt_correction``
    are the old ``angular_correction_angle`` and ``angular_correction_tilt_correction``
    keys; the tilt correction could only be set before, and now reads too.

    ``acquire``, ``last_image``, ``autocontrast`` and ``auto_focus`` are the old
    methods, claiming the imaging channel for the vendor call.

    Not here, so absent on the new API and still answered by the old branches:
    ``preset`` (Thermo has none).
    """

    needs_channel = frozenset(
        {"detector_type", "detector_mode", "detector_brightness", "detector_contrast"}
    )

    def __init__(
        self,
        beam_type: BeamType,
        parent: ThermoMicroscope,
        resources: Optional[Resources] = None,
    ):
        super().__init__(beam_type, parent=parent, resources=resources)
        self.bind_channel(lambda: parent.set_channel(beam_type))

    @property
    def _beam(self) -> Any:
        """The vendor beam, looked up on every call, as the old branches do."""
        return self.parent._get_beam(self.beam_type)

    def _set_log(self, what: str, value: Any, unit: str) -> None:
        logging.info(f"{self.beam_type.name} {what} set to {value}{unit}.")

    def read_on(self) -> bool:
        return self._beam.is_on

    def write_on(self, value: bool) -> None:
        beam = self._beam
        beam.turn_on() if value else beam.turn_off()
        logging.info(f"{self.beam_type.name} beam turned {'on' if value else 'off'}.")

    def read_blanked(self) -> bool:
        return self._beam.is_blanked

    def write_blanked(self, value: bool) -> None:
        beam = self._beam
        beam.blank() if value else beam.unblank()
        logging.info(
            f"{self.beam_type.name} beam {'blanked' if value else 'unblanked'}."
        )

    def read_working_distance(self) -> float:
        return self._beam.working_distance.value

    def write_working_distance(self, value: float) -> None:
        self._beam.working_distance.value = value
        self._set_log("working distance", value, " m")

    def read_current(self) -> float:
        return self._beam.beam_current.value

    def write_current(self, value: float) -> None:
        self._beam.beam_current.value = value
        self._set_log("current", value, " A")

    def metadata_current(self) -> ParameterMetadata:
        beam = self._beam
        if self.beam_type is BeamType.ION:
            return ParameterMetadata(choices=list(beam.beam_current.available_values))
        # the electron currents double from the minimum, as the microscope lists them
        limits = beam.beam_current.limits
        choices, current = [], limits.min
        while current <= limits.max:
            choices.append(current)
            current *= 2.0
        return ParameterMetadata(choices=choices)

    def read_voltage(self) -> float:
        return self._beam.high_voltage.value

    def write_voltage(self, value: float) -> None:
        self._beam.high_voltage.value = value
        self._set_log("voltage", value, " V")

    def metadata_voltage(self) -> ParameterMetadata:
        from fibsem.microscopes.autoscript import THERMO_VOLTAGE_CHOICES

        limits = self._beam.high_voltage.limits
        return ParameterMetadata(
            choices=[
                v
                for v in THERMO_VOLTAGE_CHOICES[self.beam_type]
                if limits.min <= v <= limits.max
            ]
        )

    def read_hfw(self) -> float:
        return self._beam.horizontal_field_width.value

    def write_hfw(self, value: float) -> None:
        # clipped just inside the vendor's maximum, as the old branch does
        beam = self._beam
        limits = beam.horizontal_field_width.limits
        value = np.clip(value, limits.min, limits.max - 10e-6)
        beam.horizontal_field_width.value = value
        logging.info(f"{self.beam_type.name} HFW set to {value} m.")

    def metadata_hfw(self) -> ParameterMetadata:
        limits = self._beam.horizontal_field_width.limits
        return ParameterMetadata(
            limits=RangeLimit(min=limits.min, max=limits.max - 10e-6)
        )

    def read_dwell_time(self) -> float:
        return self._beam.scanning.dwell_time.value

    def write_dwell_time(self, value: float) -> None:
        self._beam.scanning.dwell_time.value = value
        self._set_log("dwell time", value, " s")

    def read_scan_rotation(self) -> float:
        return self._beam.scanning.rotation.value

    def write_scan_rotation(self, value: float) -> None:
        self._beam.scanning.rotation.value = value
        self._set_log("scan rotation", value, " radians")

    def read_shift(self) -> Point:
        beam = self._beam
        return Point(beam.beam_shift.value.x, beam.beam_shift.value.y)

    def write_shift(self, value: Point) -> None:
        from autoscript_sdb_microscope_client.structures import Point as ThermoPoint

        self._beam.beam_shift.value = ThermoPoint(value.x, value.y)
        self._set_log("shift", value, "")

    def read_stigmation(self) -> Point:
        beam = self._beam
        return Point(beam.stigmator.value.x, beam.stigmator.value.y)

    def write_stigmation(self, value: Point) -> None:
        from autoscript_sdb_microscope_client.structures import Point as ThermoPoint

        self._beam.stigmator.value = ThermoPoint(value.x, value.y)
        self._set_log("stigmation", value, "")

    def read_resolution(self) -> List[int]:
        # a list, as the old get returns it
        resolution = self._beam.scanning.resolution.value
        return [int(resolution.split("x")[0]), int(resolution.split("x")[-1])]

    def write_resolution(self, value: Tuple[int, int]) -> None:
        self._beam.scanning.resolution.value = f"{value[0]}x{value[1]}"

    # The detector: the active device's, so these run with this beam's channel
    # selected (needs_channel). The writes check as the old branches do.

    @property
    def _detector(self) -> Any:
        return self.parent.connection.detector

    def read_detector_type(self) -> str:
        return self._detector.type.value

    def write_detector_type(self, value: str) -> None:
        detector = self._detector
        if value in detector.type.available_values:
            detector.type.value = value
            logging.info(f"Detector type set to {value}.")
        else:
            logging.warning(f"Detector type {value} not available.")

    def metadata_detector_type(self) -> ParameterMetadata:
        # read outside a parameter's claim, so it selects the channel itself
        with self.parent._threading_lock:
            self.parent.set_channel(self.beam_type)
            return ParameterMetadata(choices=list(self._detector.type.available_values))

    # No choices for the mode: they are the detector type's, which can change.
    def read_detector_mode(self) -> str:
        return self._detector.mode.value

    def write_detector_mode(self, value: str) -> None:
        detector = self._detector
        if value in detector.mode.available_values:
            detector.mode.value = value
            logging.info(f"Detector mode set to {value}.")
        else:
            logging.warning(f"Detector mode {value} not available.")

    def read_detector_brightness(self) -> float:
        return self._detector.brightness.value

    def write_detector_brightness(self, value: float) -> None:
        if 0 < value <= 1:
            self._detector.brightness.value = value
            logging.info(f"Detector brightness set to {value}.")
        else:
            logging.warning(
                f"Detector brightness {value} not available, must be between 0 and 1."
            )

    def read_detector_contrast(self) -> float:
        return self._detector.contrast.value

    def write_detector_contrast(self, value: float) -> None:
        if 0 < value <= 1:
            self._detector.contrast.value = value
            logging.info(f"Detector contrast set to {value}.")
        else:
            logging.warning(
                f"Detector contrast {value} not available, mut be between 0 and 1."
            )

    # The scan area: the old spot_mode, reduced_area and full_frame keys.

    def read_scanning_mode(self) -> Optional[ScanMode]:
        # New: nothing read the vendor's scan mode before. A mode with no ScanMode
        # (line, external) or a failed read warns and reads None, so a scan command
        # that has already been made does not fail on its read-back.
        try:
            mode = str(self._beam.scanning.mode.value)
        except Exception as e:
            logging.warning(f"{self.beam_type.name} scan mode could not be read: {e}")
            return None
        found = _SCAN_MODES.get(mode.replace("_", "").replace(" ", "").lower())
        if found is None:
            logging.warning(
                f"{self.beam_type.name} scan mode {mode} is not one of ours."
            )
        return found

    def _spot(self, point: Point) -> None:
        self._beam.scanning.mode.set_spot(x=point.x, y=point.y)

    def _reduced_area(self, area: FibsemRectangle) -> None:
        self._beam.scanning.mode.set_reduced_area(
            left=area.left, top=area.top, width=area.width, height=area.height
        )

    def _full_frame(self) -> None:
        self._beam.scanning.mode.set_full_frame()

    # The angular correction: the electron column's only.

    def available_angular_correction(self) -> bool:
        return self.beam_type is BeamType.ELECTRON

    def read_angular_correction(self) -> float:
        return self._beam.angular_correction.angle.value

    def write_angular_correction(self, value: float) -> None:
        self._beam.angular_correction.angle.value = value
        logging.info(f"Angular correction angle set to {value} radians.")

    def available_tilt_correction(self) -> bool:
        return self.beam_type is BeamType.ELECTRON

    def read_tilt_correction(self) -> Optional[bool]:
        # New: the old key could only be set. A failed read warns and reads None, so
        # a write that has already been made does not fail on its read-back.
        try:
            return bool(self._beam.angular_correction.tilt_correction.is_on)
        except Exception as e:
            logging.warning(f"Tilt correction could not be read: {e}")
            return None

    def write_tilt_correction(self, value: bool) -> None:
        tilt_correction = self._beam.angular_correction.tilt_correction
        tilt_correction.turn_on() if value else tilt_correction.turn_off()

    # Only a plasma ion column has a gas.
    def available_plasma_gas(self) -> bool:
        return self.beam_type is BeamType.ION and bool(self.parent.system.ion.plasma)

    def read_plasma_gas(self) -> str:
        return self._beam.source.plasma_gas.value

    def write_plasma_gas(self, value: str) -> None:
        # An unlisted gas warns and is still set, as the old branch does.
        gases = self._beam.source.plasma_gas.available_values
        if value not in gases:
            logging.warning(
                f"Plasma gas {value} not available. Available values: {gases}"
            )
        logging.info(f"Setting plasma gas to {value}... this may take some time...")
        self._beam.source.plasma_gas.value = value
        logging.info(f"Plasma gas set to {value}.")

    def metadata_plasma_gas(self) -> ParameterMetadata:
        return ParameterMetadata(
            choices=list(self._beam.source.plasma_gas.available_values)
        )

    # Imaging and the autofunctions: ThermoMicroscope's acquire_image (and
    # acquire_image3's current-settings path), last_image, autocontrast and auto_focus,
    # moved as they are. The vendor call runs with this beam's channel claimed
    # (claim_channel), which is the old `_threading_lock` + `set_channel` pair, and the
    # FibsemImage is built from get_microscope_state as before.

    def _acquire(self, image_settings: Optional[ImageSettings]) -> FibsemImage:
        from fibsem.microscopes import autoscript as thermo

        microscope = self.parent
        name = self.beam_type.name
        if image_settings is None:
            # acquire_image(beam_type=...): the beam's current settings
            settings = microscope.get_imaging_settings(beam_type=self.beam_type)
            logging.info(f"acquiring new {name} image.")
            with self.claim_channel():
                adorned = microscope.connection.imaging.grab_frame(None)
            logging.info(f"acquiring new {name} image.")
        else:
            settings = image_settings
            if settings.reduced_area is not None:
                rect = settings.reduced_area
                reduced_area = thermo.Rectangle(
                    rect.left, rect.top, rect.width, rect.height
                )
                logging.debug(
                    f"Set reduced are: {reduced_area} for beam type {settings.beam_type}"
                )
            else:
                reduced_area = None
                self.full_frame()
            microscope.set_field_of_view(hfw=settings.hfw, beam_type=self.beam_type)
            logging.info(f"acquiring new {name} image.")
            frame_settings = thermo.GrabFrameSettings(
                resolution=f"{settings.resolution[0]}x{settings.resolution[1]}",
                dwell_time=settings.dwell_time,
                reduced_area=reduced_area,
                line_integration=settings.line_integration,
                scan_interlacing=settings.scan_interlacing,
                frame_integration=settings.frame_integration,
                drift_correction=settings.drift_correction,
            )
            with self.claim_channel():
                adorned = microscope.connection.imaging.grab_frame(frame_settings)
            if settings.reduced_area is not None:
                self.full_frame()

        state = microscope.get_microscope_state(beam_type=self.beam_type)
        image = thermo.fibsem_image_from_adorned_image(
            copy.deepcopy(adorned), copy.deepcopy(settings), copy.deepcopy(state)
        )
        microscope._set_additional_metadata(image)
        if image_settings is not None:
            microscope._last_imaging_settings = image_settings
        logging.debug({"msg": "acquire_image", "metadata": image.metadata.to_dict()})
        return image

    def _last_image(self) -> FibsemImage:
        from fibsem.microscopes import autoscript as thermo

        microscope = self.parent
        with self.claim_channel():
            image = microscope.connection.imaging.get_image()
        image = thermo.AdornedImage(
            data=image.data.astype(np.uint8), metadata=image.metadata
        )
        state = microscope.get_microscope_state(beam_type=self.beam_type)
        fibsem_image = thermo.fibsem_image_from_adorned_image(
            adorned=image, image_settings=None, state=state, beam_type=self.beam_type
        )
        microscope._set_additional_metadata(fibsem_image)
        logging.debug(
            {"msg": "acquire_image", "metadata": fibsem_image.metadata.to_dict()}
        )
        return fibsem_image

    def _autocontrast(self, reduced_area: Optional[FibsemRectangle]) -> None:
        # The routine optimises the active detector in the active view, so the channel
        # is held for all of it, with the reduced area set inside (FIB-569).
        logging.debug(f"Running autocontrast on {self.beam_type.name}.")
        with self.claim_channel():
            if reduced_area is not None:
                self.reduced_area(reduced_area)
            self.parent.connection.auto_functions.run_auto_cb()
        if reduced_area is not None:
            self.full_frame()
        logging.debug({"msg": "autocontrast", "beam_type": self.beam_type.name})

    def _auto_focus(self, reduced_area: Optional[FibsemRectangle]) -> None:
        # Held for the whole routine, as autocontrast is: it runs in the active view.
        logging.debug(f"Running auto-focus on {self.beam_type.name}.")
        with self.claim_channel():
            if reduced_area is not None:
                self.reduced_area(reduced_area)
            self.parent.connection.auto_functions.run_auto_focus()
        if reduced_area is not None:
            self.full_frame()
        logging.debug({"msg": "auto_focus", "beam_type": self.beam_type.name})

    # Live view: ThermoMicroscope's _acquisition_worker and _fast_acquisition_worker,
    # moved as they are, with this beam's live_frame in place of the microscope's
    # signal (which forwards it) and the stop event the beam gives.

    def _live(self, stop: threading.Event) -> None:
        self.parent.set_channel(channel=self.beam_type)
        try:
            while True:
                if stop.is_set():
                    break
                # fast continuous acquisition
                self._live_fast(stop)
                if stop.is_set():
                    break
                # acquire an image with the current beam settings
                self.live_frame.emit(self.acquire())
        except Exception as e:
            logging.error(f"Error in acquisition worker: {e}")

    def _live_fast(self, stop: threading.Event) -> None:
        from fibsem.microscopes import autoscript as thermo

        imaging = self.parent.connection.imaging
        try:
            with self.claim_channel():
                imaging.start_acquisition()
            while imaging.state == thermo.ImagingState.ACQUIRING:
                if stop.is_set():
                    imaging.stop_acquisition()
                    break
                with self.claim_channel():
                    adorned_image = imaging.get_image(
                        thermo.GetImageSettings(wait_for_frame=True)
                    )
                    image = self.parent._construct_image(
                        adorned_image, beam_type=self.beam_type
                    )
                    logging.info(f"Acquired Image: {image.data.shape}")
                    self.live_frame.emit(image)
        except Exception as e:
            logging.error(f"Exception occurred during fast acquisition: {e}")
        finally:
            imaging.stop_acquisition()


# The vendor's scan mode names (FullFrame, ReducedArea, Spot), lower-cased.
_SCAN_MODES: Dict[str, ScanMode] = {
    "fullframe": ScanMode.FULL_FRAME,
    "reducedarea": ScanMode.REDUCED_AREA,
    "spot": ScanMode.SPOT,
}


def bind_autoscript_beams(
    microscope: ThermoMicroscope, resources: Optional[Resources] = None
) -> Dict[BeamType, AutoscriptBeam]:
    """Build ``beams[BeamType]`` for a connected Thermo microscope: one per enabled
    column, so a disabled one is never touched."""
    enabled = {
        BeamType.ELECTRON: microscope.system.electron.enabled,
        BeamType.ION: microscope.system.ion.enabled,
    }
    return {
        beam_type: AutoscriptBeam(beam_type, microscope, resources).connect()
        for beam_type, on in enabled.items()
        if on
    }


# -- the chamber, the manipulator and the gas injectors -------------------------------


class AutoscriptChamber(Chamber):
    """The AutoScript vacuum, ``connection.vacuum``.

    ``read_state``/``read_pressure`` are the ``chamber_state``/``chamber_pressure``
    branches of ``ThermoMicroscope._get``, and ``_pump``/``_vent`` the
    ``pump_chamber``/``vent_chamber`` branches of ``_set``. The old key returns the
    vendor's name for the state ("Pumped"); the device reads it as a ``ChamberState``.
    """

    def __init__(self, parent: ThermoMicroscope, resources: Optional[Resources] = None):
        super().__init__(parent=parent, resources=resources)

    @property
    def _vacuum(self) -> Any:
        return self.parent.connection.vacuum

    def read_state(self) -> ChamberState:
        return ChamberState.from_name(self._vacuum.chamber_state)

    def read_pressure(self) -> float:
        return self._vacuum.chamber_pressure.value

    def _pump(self) -> None:
        logging.info("Pumping chamber...")
        self._vacuum.pump()
        logging.info("Chamber pumped.")

    def _vent(self) -> None:
        logging.info("Venting chamber...")
        self._vacuum.vent()
        logging.info("Chamber vented.")


def bind_autoscript_chamber(
    microscope: ThermoMicroscope, resources: Optional[Resources] = None
) -> AutoscriptChamber:
    """Build ``chamber`` for a connected Thermo microscope."""
    return AutoscriptChamber(microscope, resources).connect()


_MANIPULATOR_NAMES = ("PARK", "EUCENTRIC")


class AutoscriptManipulator(Manipulator):
    """The AutoScript needle, ``connection.specimen.manipulator``.

    Each method is what the matching ``ThermoMicroscope`` method does today, without
    the read-back, which the device's commands make: ``_insert``
    (``insert_manipulator``), ``_retract`` (``retract_manipulator``),
    ``_move_relative``/``_move_absolute`` (``move_manipulator_relative``/
    ``_absolute``) and ``saved_position`` (``_get_saved_manipulator_position``).
    The corrected and offset moves depend on the stage tilt, so they stay
    ``ThermoMicroscope``'s and move through this device.
    """

    def __init__(self, parent: ThermoMicroscope, resources: Optional[Resources] = None):
        super().__init__(parent=parent, resources=resources)

    @property
    def _needle(self) -> Any:
        return self.parent.connection.specimen.manipulator

    def read_position(self) -> FibsemManipulatorPosition:
        from fibsem.microscopes.autoscript import manipulator_position_from_autoscript

        return manipulator_position_from_autoscript(self._needle.current_position)

    def read_state(self) -> InsertableDeviceState:
        from fibsem.microscopes.autoscript import ManipulatorState

        # the old key is True only when inserted; anything else is not inserted
        state = self._needle.state
        if state == ManipulatorState.INSERTED:
            return InsertableDeviceState.INSERTED
        if state == getattr(ManipulatorState, "RETRACTED", None):
            return InsertableDeviceState.RETRACTED
        return InsertableDeviceState.UNKNOWN

    def named_positions(self) -> List[str]:
        return list(_MANIPULATOR_NAMES)

    @staticmethod
    def _saved(name: str) -> Any:
        from fibsem.microscopes.autoscript import ManipulatorSavedPosition

        return (
            ManipulatorSavedPosition.PARK
            if name == "PARK"
            else ManipulatorSavedPosition.EUCENTRIC
        )

    def saved_position(self, name: str = "PARK") -> FibsemManipulatorPosition:
        from fibsem.microscopes.autoscript import (
            ManipulatorCoordinateSystem,
            manipulator_position_from_autoscript,
        )

        if name not in _MANIPULATOR_NAMES:
            raise ValueError(f"saved position {name} not supported.")
        autoscript_position = self._needle.get_saved_position(
            self._saved(name),
            ManipulatorCoordinateSystem.STAGE,  # as the old method reads it
        )
        position = manipulator_position_from_autoscript(autoscript_position)
        logging.debug(
            {
                "msg": "get_saved_manipulator_position",
                "name": name,
                "position": position.to_dict(),
            }
        )
        return position

    def _insert(self, name: str) -> None:
        from fibsem.microscopes.autoscript import ManipulatorCoordinateSystem

        if name not in _MANIPULATOR_NAMES:
            raise ValueError(f"insert position {name} not supported.")
        saved_position = self._saved(name)
        insert_position = self._needle.get_saved_position(
            saved_position, ManipulatorCoordinateSystem.RAW
        )
        # not an f-string in the old method either; kept so the messages match
        logging.info("inserting manipulator to {saved_position}: {insert_position}.")
        self._needle.insert(insert_position)
        logging.info("insert manipulator complete.")

    def _retract(self) -> None:
        from fibsem.microscopes.autoscript import (
            ManipulatorCoordinateSystem,
            ManipulatorSavedPosition,
        )

        needle = self._needle
        park_position = needle.get_saved_position(
            ManipulatorSavedPosition.PARK, ManipulatorCoordinateSystem.RAW
        )
        logging.info(f"retracting needle to {park_position}")
        needle.absolute_move(park_position)
        time.sleep(1)  # AutoScript sometimes throws errors if you retract too quick?
        logging.info("retracting needle...")
        needle.retract()
        logging.info("retract needle complete")

    def _move_relative(self, delta: FibsemManipulatorPosition) -> None:
        from fibsem.microscopes.autoscript import manipulator_position_to_autoscript

        logging.info(f"moving manipulator by {delta}")
        self._needle.relative_move(manipulator_position_to_autoscript(delta))
        logging.debug({"msg": "move_manipulator_relative", "position": delta.to_dict()})

    def _move_absolute(self, position: FibsemManipulatorPosition) -> None:
        from fibsem.microscopes.autoscript import manipulator_position_to_autoscript

        logging.info(f"moving manipulator to {position}")
        self._needle.absolute_move(manipulator_position_to_autoscript(position))
        logging.debug(
            {"msg": "move_manipulator_absolute", "position": position.to_dict()}
        )


def bind_autoscript_manipulator(
    microscope: ThermoMicroscope, resources: Optional[Resources] = None
) -> AutoscriptManipulator:
    """Build ``manipulator`` for a connected Thermo microscope that has one."""
    return AutoscriptManipulator(microscope, resources).connect()


MULTICHEM = "Multichem"


class AutoscriptGasInjector(GasInjector):
    """One AutoScript gas injector: a GIS port (``gas.get_gis_port(port)``), or the
    multichem (``gas.get_multichem()``), looked up on every call as
    ``ThermoMicroscope.get_gis`` does.

    The hooks are ``ThermoMicroscope``'s GIS methods: ``_insert`` (``insert_gis``),
    ``_heater_on`` (``gis_turn_heater_on``, including its wait for temperature),
    ``_retract`` (``retract_gis``), and the open, close and heater-off calls of
    ``cryo_deposition_v2``.

    fibsem has never read a gas injector's state back from AutoScript, so ``state``,
    ``heated``, ``opened`` and ``gas`` report what this device's own commands last
    did (retracted, off and closed until then). They do not see a change made in the
    microscope's own UI.
    """

    def __init__(
        self,
        parent: ThermoMicroscope,
        port: str,
        resources: Optional[Resources] = None,
    ):
        super().__init__(parent=parent, resources=resources)
        self.port = port
        self._inserted = False
        self._heated = False
        self._opened = False
        self._gas: str = "" if port == MULTICHEM else port

    @property
    def multichem(self) -> bool:
        return self.port == MULTICHEM

    @property
    def _gis(self) -> Any:
        gas = self.parent.connection.gas
        return gas.get_multichem() if self.multichem else gas.get_gis_port(self.port)

    def read_gas(self) -> str:
        return self._gas

    def read_state(self) -> InsertableDeviceState:
        if self._inserted:
            return InsertableDeviceState.INSERTED
        return InsertableDeviceState.RETRACTED

    def read_heated(self) -> bool:
        return self._heated

    def read_opened(self) -> bool:
        return self._opened

    def _insert(self, position: Optional[str]) -> None:
        gis = self._gis
        if position:
            logging.info(f"Inserting Multichem GIS to {position}")
            gis.insert(position)
        else:
            logging.info("Inserting Gas Injection System")
            gis.insert()
        self._inserted = True
        logging.debug({"msg": "insert_gis", "insert_position": position})

    def _retract(self) -> None:
        self._gis.retract()
        self._inserted = False
        logging.debug({"msg": "retract_gis", "use_multichem": self.multichem})

    def _heater_on(self, gas: Optional[str]) -> None:
        gis = self._gis
        logging.info(f"Turning on heater for {gas}")
        if gas is not None:
            gis.turn_heater_on(gas)
            self._gas = gas
        else:
            gis.turn_heater_on()
        self._heated = True

        logging.info("Waiting for heater to get to temperature...")
        time.sleep(3)  # we need to wait a bit

        wait_time = 0
        max_wait_time = 15
        target_temp = 300  # validate this somehow?
        while True:
            # a multichem needs the gas name
            temp = (
                gis.get_temperature(gas) if gas is not None else gis.get_temperature()
            )
            logging.info(
                f"Waiting for heater: {temp}K, target={target_temp}, wait_time={wait_time}/{max_wait_time} sec"
            )
            if temp >= target_temp:
                break
            time.sleep(1)  # wait for the heat
            wait_time += 1
            if wait_time > max_wait_time:
                raise TimeoutError("Gas Injection Failed to heat within time...")

        logging.debug(
            {
                "msg": "gis_turn_heater_on",
                "temp": temp,
                "target_temp": target_temp,
                "wait_time": wait_time,
                "max_wait_time": max_wait_time,
            }
        )

    def _heater_off(self) -> None:
        self._gis.turn_heater_off()
        self._heated = False

    def _open(self) -> None:
        self._gis.open()
        self._opened = True

    def _close(self) -> None:
        self._gis.close()
        self._opened = False


def bind_autoscript_gis(
    microscope: ThermoMicroscope, resources: Optional[Resources] = None
) -> Dict[str, AutoscriptGasInjector]:
    """Build one gas injector per GIS port the instrument lists, and the multichem
    (keyed ``MULTICHEM``) when it has one, for a connected Thermo microscope."""
    devices: Dict[str, AutoscriptGasInjector] = {}
    gas = microscope.connection.gas
    if microscope.is_available("gis"):
        for port in gas.list_all_gis_ports():
            devices[port] = AutoscriptGasInjector(microscope, port, resources).connect()
    if microscope.is_available("gis_multichem"):
        devices[MULTICHEM] = AutoscriptGasInjector(
            microscope, MULTICHEM, resources
        ).connect()
    return devices
