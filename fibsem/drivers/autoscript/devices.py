"""The AutoScript (Thermo Fisher) stage, beams, chamber and manipulator as devices.

``AutoscriptStage`` implements the ``Stage`` device with what ``ThermoMicroscope``
does today, moved as-is, so the old call and the device make the same SDK calls in
the same order. ``AutoscriptCompustage`` is the same for a compustage (Arctis,
Hydra), ``AutoscriptBeam`` for the beam keys, and ``AutoscriptChamber`` and
``AutoscriptManipulator`` for the vacuum and the needle. ``ThermoMicroscope`` builds
them at connect and routes its keys and moves to them; the old code they replaced is
deleted.

The vendor stage is ``microscope._vendor_stage``, which the Thermo backend sets at
connect to ``specimen.stage`` or ``specimen.compustage`` (``microscope.stage`` is the
stage device); the vendor beams are under
``microscope.connection.beams``. This module imports the SDK only through
``fibsem.drivers.autoscript.microscope``, which is where the guarded import lives, apart from
the SDK's ``Point`` inside a write, which the old branch imports there too.

The FM.

The Thermo Fisher FM (Arctis, Hydra) as devices.

``AutoscriptFMCamera``, ``AutoscriptFMLightSource``, ``AutoscriptFMFilterSet``,
``AutoscriptFMObjective`` and the ``AutoscriptFM`` group are the old
``ThermoFisherFluorescenceMicroscope``'s parts moved onto the FM devices: each read,
write and command makes the SDK calls the old property or method made, in the same
order (``tests/fixtures/autoscript_fm_old_pins.json`` holds those calls).
``ThermoMicroscope.fm`` is the FM API over them
(``DeviceThermoFisherFluorescenceMicroscope``).

The channel. The FM and the beams are one AutoScript connection with one active view,
so whoever sets the view last owns it (FIB-517). ``AutoscriptFMChannel`` is the old
class's ``active_channel()`` as it was: point the connection at the FM for a block, put
the view back after the outermost block, count the depth so a run isn't undone by each
step inside it, and take the microscope's ``imaging_channel`` lock (its
``_threading_lock``) for that bookkeeping only. A parameter in ``needs_channel`` runs in
that scope. The generic claim isn't used because it holds the lock through the read and
always takes it; the old scope skips the lock when the FM already has the view, which is
what keeps the objective movable while live view re-takes the lock every frame.

Live view keeps what the old fast acquisition does, with each frame pulled rather than
pushed: `AutoscriptFM.start_live` switches the light on and starts the acquisition, each
``acquire_frame`` while live is the next ``imaging.get_image()``, and stopping switches
the light off and stops the acquisition.

The FM parts need the SDK, through ``fibsem.fm.autoscript``, which each imports in the
method that uses it, so the module still loads where AutoScript is not installed.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
)

import numpy as np

from fibsem.devices.beam import ACQUISITION_RESOLUTIONS, Beam
from fibsem.devices.chamber import Chamber
from fibsem.devices.core import (
    IMAGING_CHANNEL,
    BoundParameter,
    Device,
    ParameterMetadata,
    Resources,
    resources_of,
)
from fibsem.devices.fm import FM, Camera, FilterSet, LightSource, Objective
from fibsem.devices.manipulator import Manipulator
from fibsem.devices.sample_loader import (
    GridExchangeError,
    Magazine,
    MagazineSlot,
    MagazineSlotState,
    SampleLoader,
    StageSample,
)
from fibsem.devices.stage import (
    Stage,
    axis_limits_from_degrees,
    compustage_device_at_pose,
    compustage_poses,
)
from fibsem.devices.wire import Frame
from fibsem.fm.structures import (
    REFLECTION,
    ChannelSettings,
    EmissionFilter,
    emission_filter_for,
    objective_device_state,
)
from fibsem.structures import (
    BeamType,
    ChamberState,
    DeviceEntry,
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
    from fibsem.drivers.autoscript.microscope import ThermoMicroscope
    from fibsem.drivers.registry import BuildContext


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
        from fibsem.drivers.autoscript.microscope import stage_position_to_autoscript

        return stage_position_to_autoscript(position, compustage=self.compustage)

    # -- position ----------------------------------------------------------------

    def read_position(self) -> FibsemStagePosition:
        from fibsem.drivers.autoscript.microscope import stage_position_from_autoscript

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
        from fibsem.drivers.autoscript.microscope import MoveSettings

        # get current working distance, to be restored later
        wd = self.parent.get_working_distance(BeamType.ELECTRON)

        # leaving alone the axes the microscope would refuse (the objective inserted,
        # or entering the FM pose)
        autoscript_position = self._to_autoscript(
            self.parent._without_blocked_axes(position)
        )

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

    def has_builtin_shuttle(self) -> bool:
        return True

    def poses(
        self, rotation_reference: float, shuttle_pre_tilt: float, fib_column_tilt: float
    ) -> Dict[str, FibsemStagePosition]:
        return compustage_poses(rotation_reference, shuttle_pre_tilt, fib_column_tilt)

    def device_at_pose(self, orientation: str) -> Optional[str]:
        return compustage_device_at_pose(orientation)


def autoscript_stage_class(microscope: ThermoMicroscope) -> Type[AutoscriptStage]:
    """The driver class for the stage the Thermo backend found at connect."""
    return AutoscriptCompustage if microscope._compustage_installed else AutoscriptStage


def bind_autoscript_stage(
    microscope: ThermoMicroscope, resources: Optional[Resources] = None
) -> AutoscriptStage:
    """Build ``stage`` for a connected Thermo microscope."""
    return autoscript_stage_class(microscope)(microscope, resources).connect()


class AutoscriptBeam(Beam):
    """An AutoScript beam: ``connection.beams.electron_beam`` or ``.ion_beam``.

    Each parameter is the matching branch of the old ``ThermoMicroscope._get``/``_set``
    (now removed) moved as it is, so the device makes the same SDK calls and logs the
    same messages. The choices are what the old ``get_available_values`` answered for
    a beam key.

    The detector is the active device's, so the detector parameters claim the imaging
    channel and select this beam's (``needs_channel``), as the old branches do under
    the lock. The scan commands are the old ``spot_mode``/``reduced_area``/
    ``full_frame`` keys; ``scanning_mode`` reads the vendor's scan mode, which nothing
    read before. The electron beam's ``angular_correction`` and ``tilt_correction``
    are the old ``angular_correction_angle`` and ``angular_correction_tilt_correction``
    keys; the tilt correction could only be set before, and now reads too.

    ``acquire``, ``last_image``, ``autocontrast`` and ``auto_focus`` are the old
    methods, claiming the imaging channel for the vendor call.

    Not here: ``preset`` (Thermo has none), whose key reads None.
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
        from fibsem.drivers.autoscript.microscope import THERMO_VOLTAGE_CHOICES

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

    def _scan_sizes(self) -> List[Tuple[int, int]]:
        return [
            tuple(int(px) for px in r.split("x"))
            for r in self._beam.scanning.resolution.available_values
        ]

    def scan_resolutions(self) -> list:
        return self._scan_sizes()

    def metadata_dwell_time(self) -> ParameterMetadata:
        limits = self._beam.scanning.dwell_time.limits
        return ParameterMetadata(limits=RangeLimit(min=limits.min, max=limits.max))

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

    def metadata_resolution(self) -> ParameterMetadata:
        # The scan's own sizes, then the frames a grab is taken at that it doesn't
        # list (the squares): which of those it takes is not yet checked on hardware.
        listed = self._scan_sizes()
        extra = [r for r in ACQUISITION_RESOLUTIONS if r not in listed]
        return ParameterMetadata(choices=listed + extra)

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

    def read_detector_mode(self) -> str:
        return self._detector.mode.value

    def write_detector_mode(self, value: str) -> None:
        detector = self._detector
        if value in detector.mode.available_values:
            detector.mode.value = value
            logging.info(f"Detector mode set to {value}.")
        else:
            logging.warning(f"Detector mode {value} not available.")

    def metadata_detector_mode(self) -> ParameterMetadata:
        # the detector type's modes, read again when the type changes; read outside a
        # parameter's claim, so it selects the channel itself
        with self.parent._threading_lock:
            self.parent.set_channel(self.beam_type)
            return ParameterMetadata(choices=list(self._detector.mode.available_values))

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
        from fibsem.drivers.autoscript import microscope as thermo

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
        from fibsem.drivers.autoscript import microscope as thermo

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
        from fibsem.drivers.autoscript import microscope as thermo

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


# -- the chamber and the manipulator --------------------------------------------------


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


class AutoscriptSampleLoader(SampleLoader):
    """The AutoScript autoloader (Arctis, xT 28.x, AutoScript >= 4.10),
    ``connection.specimen.autoloader``.

    The magazine is ``get_slots(False)``, the autoloader's last-known record (it may
    read ``Unknown`` throughout before any scan), and ``scan`` is ``get_slots(True)``,
    a physical scan. Slots are the 1-based ``AutoloaderSlot.id``; ``load(id)`` blocks
    until the exchange is done and ``unload()`` takes nothing. A slot's description is
    its ``sample_description``. ``on_stage`` is ``autoloader.stage``. States arrive as
    enum names in either case and are read case-blind; from AutoScript 4.14 the home
    slot of the grid on the stage reads ``Loaded``, before that ``Empty``, which the
    grid model absorbs.

    Confirmed from operator code: the ``get_slots(bool)`` shape, the state strings,
    ``load(id)``/``unload()`` and ``autoloader.stage``. Measured on an Arctis
    (FIB-893, 2026-10-02): an unload and a load take about 3 minutes. Description
    writes seen to stick on an Arctis (2026-10-02); read back anyway.
    """

    # How many slots until the magazine is first read; the Arctis magazine has 12.
    DEFAULT_CAPACITY = 12

    def __init__(self, parent: ThermoMicroscope, resources: Optional[Resources] = None):
        super().__init__(parent=parent, resources=resources)
        self._capacity = self.DEFAULT_CAPACITY

    @property
    def _autoloader(self) -> Any:
        return self.parent.connection.specimen.autoloader

    def read_magazine(self) -> Magazine:
        return self._magazine(list(self._autoloader.get_slots(False)))

    def read_on_stage(self) -> StageSample:
        stage = getattr(self._autoloader, "stage", None)
        if stage is None:
            return StageSample()
        state = MagazineSlotState.from_name(getattr(stage, "state", "Unknown"))
        present = {
            MagazineSlotState.OCCUPIED: True,
            MagazineSlotState.LOADED: True,
            MagazineSlotState.EMPTY: False,
        }.get(state)
        return StageSample(present, _description(stage))

    def read_capacity(self) -> int:
        """The slots the magazine reported at its last read; nothing is asked."""
        return self._capacity

    def read_exchange_time(self) -> float:
        """An unload and a load, measured on an Arctis; every exchange is charged
        the full figure, a run's first load included, which errs long."""
        return 180.0

    def _load(self, slot: int) -> None:
        try:
            self._autoloader.load(slot)
        except Exception as e:
            raise GridExchangeError(
                f"Autoloader could not load slot {slot}: {e}"
            ) from e

    def _unload(self) -> None:
        try:
            self._autoloader.unload()
        except Exception as e:
            raise GridExchangeError(f"Autoloader could not unload: {e}") from e

    def _scan(self) -> Magazine:
        return self._magazine(list(self._autoloader.get_slots(True)))

    def _set_description(self, slot: int, text: str) -> None:
        try:
            for hw in self._autoloader.get_slots(False):
                if int(hw.id) == slot:
                    hw.sample_description = text
                    return
        except Exception as e:  # noqa: BLE001 - whatever AutoScript raised, as one error
            raise GridExchangeError(
                f"Could not write the autoloader slot description: {e}"
            ) from e
        raise GridExchangeError(f"Autoloader reported no slot {slot} to name.")

    def _magazine(self, hw_slots: list) -> Magazine:
        # The rows as the hardware gave them: what a bench session needs from the log
        # when every slot shows unknown or empty and the question is what AutoScript
        # actually said.
        logging.info(
            "Autoloader slots: "
            + ", ".join(f"{hw.id}={_describe_hw(hw)}" for hw in hw_slots)
        )
        stage = getattr(self._autoloader, "stage", None)
        if stage is not None:
            logging.info(f"Autoloader stage: {_describe_hw(stage)}")
        if hw_slots:
            self._capacity = len(hw_slots)
        return Magazine(
            tuple(
                MagazineSlot(
                    int(hw.id),
                    MagazineSlotState.from_name(getattr(hw, "state", "Unknown")),
                    _description(hw),
                )
                for hw in hw_slots
            )
        )


def _description(hw: Any) -> str:
    return (getattr(hw, "sample_description", "") or "").strip()


def _describe_hw(hw: Any) -> str:
    """``State 'description'``, as the vendor gave the state."""
    state = str(getattr(hw, "state", "Unknown")).rsplit(".", 1)[-1].capitalize()
    described = _description(hw)
    return state + (f" '{described}'" if described else "")


def autoloader_installed(microscope: ThermoMicroscope) -> bool:
    """Whether the instrument has an autoloader; False when AutoScript can't say."""
    try:
        return bool(microscope.connection.specimen.autoloader.is_installed)
    except Exception:  # noqa: BLE001 - device absent, or not ready
        return False


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
        from fibsem.drivers.autoscript.microscope import (
            manipulator_position_from_autoscript,
        )

        return manipulator_position_from_autoscript(self._needle.current_position)

    def read_state(self) -> InsertableDeviceState:
        from fibsem.drivers.autoscript.microscope import ManipulatorState

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
        from fibsem.drivers.autoscript.microscope import ManipulatorSavedPosition

        return (
            ManipulatorSavedPosition.PARK
            if name == "PARK"
            else ManipulatorSavedPosition.EUCENTRIC
        )

    def saved_position(self, name: str = "PARK") -> FibsemManipulatorPosition:
        from fibsem.drivers.autoscript.microscope import (
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
        from fibsem.drivers.autoscript.microscope import ManipulatorCoordinateSystem

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
        from fibsem.drivers.autoscript.microscope import (
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
        from fibsem.drivers.autoscript.microscope import (
            manipulator_position_to_autoscript,
        )

        logging.info(f"moving manipulator by {delta}")
        self._needle.relative_move(manipulator_position_to_autoscript(delta))
        logging.debug({"msg": "move_manipulator_relative", "position": delta.to_dict()})

    def _move_absolute(self, position: FibsemManipulatorPosition) -> None:
        from fibsem.drivers.autoscript.microscope import (
            manipulator_position_to_autoscript,
        )

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


# -- Builders by entry --------------------------------------------------------------
#
# The Thermo driver's device builders (``DRIVER.devices`` in ``autoscript``): each
# builds one device from its ``hardware.devices`` entry, on the microscope's
# connection, and names it after its entry. ``ThermoMicroscope`` builds them in steps,
# as connect learns what is fitted (``_build_beams``, ``_build_stage``,
# ``_build_parts``).


def _named(device: Device, entry: DeviceEntry) -> Device:
    device.name = entry.name
    return device.connect()


def build_autoscript_beam(entry: DeviceEntry, context: BuildContext) -> AutoscriptBeam:
    """The column an ``electron`` or ``ion`` entry names."""
    if entry.name not in ("electron", "ion"):
        raise ValueError("a Thermo beam is named 'electron' or 'ion'")
    beam_type = BeamType.ELECTRON if entry.name == "electron" else BeamType.ION
    return AutoscriptBeam(beam_type, context.microscope).connect()


def build_autoscript_stage(
    entry: DeviceEntry, context: BuildContext
) -> AutoscriptStage:
    microscope = context.microscope
    return _named(autoscript_stage_class(microscope)(microscope), entry)


def build_autoscript_chamber(
    entry: DeviceEntry, context: BuildContext
) -> AutoscriptChamber:
    return _named(AutoscriptChamber(context.microscope), entry)


def build_autoscript_manipulator(
    entry: DeviceEntry, context: BuildContext
) -> AutoscriptManipulator:
    return _named(AutoscriptManipulator(context.microscope), entry)


def build_autoscript_sample_loader(
    entry: DeviceEntry, context: BuildContext
) -> AutoscriptSampleLoader:
    return _named(AutoscriptSampleLoader(context.microscope), entry)


# The FM.

FM_ACTIVE_VIEW = 3
"""The FM's view on Arctis, as the old class sets it."""

MULTI_BAND = emission_filter_for("Fluorescence", {})
"""Thermo's one fluorescence filter, multi-band; the other choice is reflection."""


class AutoscriptFMChannel:
    """The FM's share of the microscope's one AutoScript connection, shared by the FM
    devices: the old ``ThermoFisherFluorescenceMicroscope.active_channel()``, moved.
    ``scope()`` says why it works as it does."""

    def __init__(self, microscope: ThermoMicroscope, lock: Any):
        self._microscope = microscope
        self.lock = lock
        self._depth = 0
        self._restore_view: Optional[int] = None

    @property
    def connection(self) -> Any:
        """The microscope's AutoScript client, looked up on each call."""
        return self._microscope.connection

    def set_active_channel(self) -> None:
        """Point the connection at the FM and leave it there (the old method)."""
        from fibsem.fm.autoscript import ImagingDevice

        self.connection.imaging.set_active_view(FM_ACTIVE_VIEW)
        self.connection.imaging.set_active_device(
            ImagingDevice.FLUORESCENCE_LIGHT_MICROSCOPE
        )

    def settings(self) -> Any:
        """The FM camera's settings, selecting the FM first, as ``fm_settings`` does."""
        self.set_active_channel()
        return self.connection.detector.camera_settings

    def _is_ours(self) -> bool:
        """Whether the connection is already pointed at the FM.

        The view alone answers it, for the same reason only the view is captured and
        restored: AutoScript documents ``set_active_device`` as changing the device
        *in the active view*, so the device follows the view rather than varying under
        it. One read, and deliberately not under the lock: taking the lock to find out
        whether the lock is needed would defeat the point.
        """
        return self.connection.imaging.get_active_view() == FM_ACTIVE_VIEW

    @contextmanager
    def scope(self) -> Iterator[None]:
        """Hold the connection on the FM for the block, then put the view back.

        The FM and the beams are one connection with one active view and one active
        device, so whoever sets it last owns it. A read that sets it and walks away
        steals the microscope from whatever else is using it: the objective's state,
        read on every stage poll for the overview's info bar, once left the connection
        on the FM under a running beam acquisition (FIB-517). The view is what is
        captured and put back, and that is enough: the device belongs to the view and
        comes back with it.

        The lock covers the bookkeeping only, and deliberately not the body. A scope
        can span a whole tileset, and the lock is the microscope's ``_threading_lock``,
        which every caller on the microscope shares, devices claiming
        ``imaging_channel`` included; held for minutes it would block them all.

        A depth count rather than a captured local, so the view is put back once, by
        the outermost scope: a tileset holds the channel for the whole run, and each
        tile's acquisition opens a scope inside it that must not restore the beam view
        between tiles.

        Nothing to change means nothing to lock. When the connection is already on the
        FM there is no view to set and none to put back, so the scope does no work and
        takes no lock. Live view re-takes this lock every frame with nothing between
        iterations, and Python locks are not fair, so a waiter would be starved rather
        than delayed: moving the objective while streaming was unusable for exactly
        this reason. The fast path makes no writes to the channel, so it cannot leave
        the connection where the next beam operation does not expect it.
        """
        if self._depth == 0 and self._is_ours():
            yield
            return

        with self.lock:
            if self._depth == 0:
                self._restore_view = self.connection.imaging.get_active_view()
            self.set_active_channel()
            # Counted only once the channel is ours. A raise here comes out of
            # ``__enter__``, so neither the block nor the ``finally`` runs; a depth
            # left too high would mean no later scope restored the view again.
            self._depth += 1
        try:
            yield
        finally:
            with self.lock:
                self._depth -= 1
                if self._depth == 0:
                    self.connection.imaging.set_active_view(self._restore_view)


class _OnTheFMChannel(Device):
    """A part whose ``needs_channel`` parameters run in the FM channel's scope."""

    def __init__(
        self,
        name: str,
        channel: AutoscriptFMChannel,
        parent: Any = None,
        resources: Optional[Resources] = None,
    ):
        super().__init__(name=name, parent=parent, resources=resources)
        self._channel = channel

    @contextmanager
    def _claim(self, param: BoundParameter) -> Iterator[None]:
        if not param.needs_channel:
            yield
            return
        with self._channel.scope():
            yield


class AutoscriptFMFilterSet(_OnTheFMChannel, FilterSet):
    """The excitation is held here, as the old filter set holds it: AutoScript's
    emission type is read-only, so the colour is applied when the light starts. The
    emission filter is the camera's filter type, reflection or fluorescence."""

    needs_channel = frozenset({"emission_filter"})

    def __init__(self, channel: AutoscriptFMChannel, **kwargs: Any):
        from fibsem.fm.autoscript import CameraEmissionType

        super().__init__("filter_set", channel, **kwargs)
        self.emission_type = CameraEmissionType.RED

    def read_excitation_wavelength(self) -> float:
        from fibsem.fm.autoscript import COLOR_TO_WAVELENGTH

        color = self.emission_type
        if color not in COLOR_TO_WAVELENGTH:
            raise ValueError(
                f"Invalid excitation color: {color}: must be one of "
                f"{list(COLOR_TO_WAVELENGTH.keys())}"
            )
        return COLOR_TO_WAVELENGTH[color]

    def write_excitation_wavelength(self, value: float) -> None:
        from fibsem.fm.autoscript import COLOR_TO_WAVELENGTH, WAVELENGTH_TO_COLOR

        color = WAVELENGTH_TO_COLOR.get(value, None)
        if color is None:
            closest = min(COLOR_TO_WAVELENGTH.values(), key=lambda x: abs(x - value))
            color = WAVELENGTH_TO_COLOR[closest]
        self.emission_type = color

    def metadata_excitation_wavelength(self) -> ParameterMetadata:
        from fibsem.fm.autoscript import AVAILABLE_FM_WAVELENGTHS

        return ParameterMetadata(choices=sorted(AVAILABLE_FM_WAVELENGTHS))

    def read_emission_filter(self) -> EmissionFilter:
        from fibsem.fm.autoscript import CameraFilterType

        mode = self._channel.settings().filter.type.value
        if mode is CameraFilterType.FLUORESCENCE:
            return MULTI_BAND
        return REFLECTION

    def write_emission_filter(self, value: EmissionFilter) -> None:
        from fibsem.fm.autoscript import CameraFilterType

        if value not in (REFLECTION, MULTI_BAND):
            raise ValueError(f"filter_set has no emission filter {value}")
        settings = self._channel.settings()
        if value == REFLECTION:
            settings.filter.type.value = CameraFilterType.REFLECTION
        else:
            settings.filter.type.value = CameraFilterType.FLUORESCENCE

    def metadata_emission_filter(self) -> ParameterMetadata:
        return ParameterMetadata(choices=[REFLECTION, MULTI_BAND])


class AutoscriptFMCamera(_OnTheFMChannel, Camera):
    """The FM camera's settings. Gain is the detector's contrast; offset is a setting
    kept here, as the old camera keeps it; pixel size and resolution are the objective's
    configuration over the binning."""

    needs_channel = frozenset({"exposure_time", "binning", "gain"})

    def __init__(
        self,
        channel: AutoscriptFMChannel,
        filter_set: AutoscriptFMFilterSet,
        **kwargs: Any,
    ):
        from fibsem.fm.autoscript import DEFAULT_CONFIGURATION

        super().__init__("camera", channel, **kwargs)
        self._filter_set = filter_set
        # AutoScript has no camera offset to set, so it is kept here, from zero.
        self._offset = 0.0
        self._pixel_size: Tuple[float, float] = DEFAULT_CONFIGURATION["pixel_size"]
        self._resolution: Tuple[int, int] = DEFAULT_CONFIGURATION["resolution"]

    def _exposure_time_limits(self) -> Tuple[float, float]:
        with self._channel.scope():
            limits = self._channel.settings().exposure_time.limits
            return (limits.min, limits.max)

    def read_exposure_time(self) -> float:
        return self._channel.settings().exposure_time.value

    def write_exposure_time(self, value: float) -> None:
        limits = self._exposure_time_limits()
        if not limits[0] <= value <= limits[1]:
            raise ValueError(
                f"Exposure time must be between {limits[0]} and {limits[1]}, got {value}"
            )
        self._channel.settings().exposure_time.value = value

    def metadata_exposure_time(self) -> ParameterMetadata:
        low, high = self._exposure_time_limits()
        return ParameterMetadata(limits=RangeLimit(min=low, max=high))

    def _available_binnings(self) -> Tuple[int, ...]:
        with self._channel.scope():
            return self._channel.settings().binning.available_values

    def _binning(self) -> int:
        with self._channel.scope():
            return self._channel.settings().binning.value

    def read_binning(self) -> int:
        return self._channel.settings().binning.value

    def write_binning(self, value: int) -> None:
        # The choices are read again for the message, as the old setter reads them.
        if value not in self._available_binnings():
            raise ValueError(
                f"Binning must be one of {self._available_binnings()}, got {value}"
            )
        self._channel.settings().binning.value = value

    def metadata_binning(self) -> ParameterMetadata:
        return ParameterMetadata(choices=list(self._available_binnings()))

    def read_gain(self) -> float:
        return self._channel.connection.detector.contrast.value

    def write_gain(self, value: float) -> None:
        self._channel.connection.detector.contrast.value = value

    def metadata_gain(self) -> ParameterMetadata:
        # The FM detector's contrast, which AutoScript keeps as a fraction already.
        with self._channel.scope():
            limits = self._channel.connection.detector.contrast.limits
            return ParameterMetadata(limits=RangeLimit(min=limits.min, max=limits.max))

    def read_offset(self) -> float:
        return self._offset

    def write_offset(self, value: float) -> None:
        if value < 0:
            raise ValueError("Offset must be non-negative.")
        self._offset = value

    # Binning read once per axis, as the old properties read it.

    def read_pixel_size(self) -> tuple:
        return (
            self._pixel_size[0] * self._binning(),
            self._pixel_size[1] * self._binning(),
        )

    def read_resolution(self) -> tuple:
        return (
            self._resolution[0] // self._binning(),
            self._resolution[1] // self._binning(),
        )

    def _acquire(self) -> np.ndarray:
        # The old camera grabs unscoped and leaves the FM selected, relying on the
        # acquisition around it to hold the channel. Here the grab holds it itself;
        # inside an acquisition that is the same calls, and a bare grab can't leave
        # the connection on the FM (FIB-517).
        from fibsem.fm.autoscript import GrabFrameSettings

        with self._channel.scope():
            frame_settings = GrabFrameSettings(
                emission_type=self._filter_set.emission_type
            )
            self._channel.set_active_channel()
            image = self._channel.connection.imaging.grab_frame(frame_settings)
            return image.data

    # The acquisition live view runs, as the old camera starts and stops it.

    def _start_acquisition(self) -> None:
        self._channel.set_active_channel()
        self._channel.connection.imaging.start_acquisition()

    def _stop_acquisition(self) -> None:
        self._channel.set_active_channel()
        self._channel.connection.imaging.stop_acquisition()


class AutoscriptFMLightSource(_OnTheFMChannel, LightSource):
    """The light's brightness. AutoScript sets it for the colour that is emitting, so
    a write switches the light on in the filter set's colour, sets it, and switches it
    off again, as the old light source does."""

    needs_channel = frozenset({"power"})

    def __init__(
        self,
        channel: AutoscriptFMChannel,
        filter_set: AutoscriptFMFilterSet,
        **kwargs: Any,
    ):
        super().__init__("light_source", channel, **kwargs)
        self._filter_set = filter_set

    def read_power(self) -> float:
        return self._channel.connection.detector.brightness.value

    def write_power(self, value: float) -> None:
        self.start_emission(self._filter_set.emission_type)
        self._channel.connection.detector.brightness.value = value
        self.stop_emission()

    def metadata_power(self) -> ParameterMetadata:
        with self._channel.scope():
            limits = self._channel.connection.detector.brightness.limits
            return ParameterMetadata(limits=RangeLimit(min=limits.min, max=limits.max))

    def start_emission(self, emission_type: Any) -> None:
        with self._channel.scope():
            self._channel.connection.detector.camera_settings.emission.start(
                emission_type=emission_type
            )

    def stop_emission(self) -> None:
        with self._channel.scope():
            self._channel.connection.detector.camera_settings.emission.stop()


class AutoscriptFMObjective(_OnTheFMChannel, Objective):
    """The objective: its focus is the camera's focus setting, and inserting or
    retracting it is the detector's. Magnification and numerical aperture are the
    configuration's; ``limit_position`` is the furthest a move may go in."""

    needs_channel = frozenset({"position", "state"})

    def __init__(self, channel: AutoscriptFMChannel, **kwargs: Any):
        from fibsem.fm.autoscript import DEFAULT_CONFIGURATION

        super().__init__("objective", channel, **kwargs)
        self._magnification = DEFAULT_CONFIGURATION["magnification"]
        self._numerical_aperture = DEFAULT_CONFIGURATION["numerical_aperture"]
        self._limit_position: float = DEFAULT_CONFIGURATION.get(
            "limit_position", 8.6e-3
        )

    def _focus_limits(self) -> Tuple[float, float]:
        with self._channel.scope():
            limits = self._channel.settings().focus.limits
            return (limits.min, limits.max)

    def read_position(self) -> float:
        return self._channel.settings().focus.value

    def metadata_position(self) -> ParameterMetadata:
        low, high = self._focus_limits()
        return ParameterMetadata(limits=RangeLimit(min=low, max=high))

    def read_state(self) -> InsertableDeviceState:
        return objective_device_state(self._channel.connection.detector.state)

    def read_magnification(self) -> float:
        return self._magnification

    def read_numerical_aperture(self) -> float:
        return self._numerical_aperture

    def read_limit_position(self) -> float:
        return self._limit_position

    def write_limit_position(self, value: float) -> None:
        self._limit_position = value
        logging.info(
            f"Objective user-defined position limit set to: "
            f"{self._limit_position * 1e3:.3f} mm"
        )

    def _move_absolute(self, position: float) -> None:
        # The hardware limits are read for each bound, as the old move reads them.
        if not self._focus_limits()[0] <= position <= self._focus_limits()[1]:
            raise ValueError(
                f"Position {position} out of limits {self._focus_limits()}"
            )
        if not position <= self._limit_position:
            logging.warning(
                f"Clipping position {position} to user-defined limits "
                f"{self._limit_position}"
            )
            position = np.clip(position, 0, self._limit_position)
        with self._channel.scope():
            self._channel.settings().focus.value = position

    def _move_relative(self, delta: float) -> None:
        self._move_absolute(self.position.get_value() + delta)

    def _insert(self) -> bool:
        with self._channel.scope():
            if self.state.get_value() is InsertableDeviceState.INSERTED:
                logging.warning("Objective lens is already inserted.")
                return False
            self._channel.connection.detector.insert()
            return True

    def _retract(self) -> bool:
        with self._channel.scope():
            if self.state.get_value() is InsertableDeviceState.RETRACTED:
                logging.warning("Objective lens is already retracted.")
                return False
            self._channel.connection.detector.retract()
            return True


class AutoscriptFM(FM):
    """The Thermo FM group: a channel set up on its parts, then a frame, holding the
    FM channel throughout; and live view, as the old fast acquisition runs it."""

    def __init__(
        self,
        channel: AutoscriptFMChannel,
        parent: Any = None,
        resources: Optional[Resources] = None,
    ):
        super().__init__(name="fm", parent=parent, resources=resources)
        self._channel = channel

    def check_health(self) -> Optional[str]:
        """Whether the FM answers: one live read, the camera's exposure time."""
        self.camera.exposure_time.get_value()
        return None

    def _apply_channel(self, channel: Optional[Dict[str, Any]]) -> None:
        """The old ``set_channel``: excitation, emission filter, power, exposure, gain."""
        if channel is None:
            return
        from fibsem.fm.microscope import emission_filter_named

        settings = ChannelSettings.from_dict(channel)
        filters = self.filter_set
        filters.excitation_wavelength.write_through(settings.excitation_wavelength)
        filters.emission_filter.write_through(
            emission_filter_named(
                settings.emission_wavelength, filters.emission_filter.choices
            )
        )
        self.light_source.power.write_through(settings.power)
        camera = self.camera
        camera.exposure_time.write_through(settings.exposure_time)
        if settings.gain is not None:
            camera.gain.write_through(settings.gain)

    def _frame(self, channel: Optional[Dict[str, Any]]) -> np.ndarray:
        self._apply_channel(channel)
        if self.is_live:
            return self._live_frame()
        return self.camera.acquire()

    def _acquire_channel(self, channel: Optional[Dict[str, Any]]) -> np.ndarray:
        with self._channel.scope():
            return self._frame(channel)

    def _acquire_frame(self, channel: Optional[Dict[str, Any]]) -> Frame:
        # The metadata is read inside the scope too, as the old acquisition reads it,
        # so it describes the state the frame was taken in.
        with self._channel.scope():
            acquisition_date = datetime.now().isoformat()
            data = self._frame(channel)
            metadata = {"acquisition_date": acquisition_date, **self._frame_metadata()}
            return Frame(data, metadata)

    # -- live view: the old fast acquisition, pulled ------------------------------------

    def _start_live(self, channel: Optional[Dict[str, Any]]) -> None:
        self._apply_channel(channel)
        connection = self._channel.connection
        with self._channel.lock:
            self._channel.set_active_channel()
            # The colour the hardware has, as the old fast acquisition starts it.
            emission_color = connection.detector.camera_settings.emission.type.value
            self.light_source.start_emission(emission_type=emission_color)
            self.camera._start_acquisition()

    def _live_frame(self) -> np.ndarray:
        from fibsem.fm.autoscript import ImagingState

        if self._channel.connection.imaging.state != ImagingState.ACQUIRING:
            # Stopped from outside (the microscope's own UI): end live view as the old
            # loop did, light off and acquisition stopped.
            self.stop_live()
            raise RuntimeError("The FM acquisition stopped; live view has ended.")
        with self._channel.lock:
            # Re-forced each frame, as the old loop does: something may have taken it.
            self._channel.set_active_channel()
            return self._channel.connection.imaging.get_image().data

    def _stop_live(self) -> None:
        self.light_source.stop_emission()
        self.camera._stop_acquisition()


def bind_autoscript_fm(
    microscope: ThermoMicroscope,
    resources: Optional[Resources] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Device]:
    """The Thermo FM's parts and group for a connected Thermo microscope, by device
    name. They share the microscope's ``imaging_channel`` lock with the beams.
    *config* is the fm entry's own keys (``mount_transform``)."""
    resources = resources if resources is not None else resources_of(microscope)
    channel = AutoscriptFMChannel(microscope, resources.lock(IMAGING_CHANNEL))
    common = {"parent": microscope, "resources": resources}
    filter_set = AutoscriptFMFilterSet(channel, **common)
    parts: Dict[str, Device] = {
        "camera": AutoscriptFMCamera(channel, filter_set, **common),
        "light_source": AutoscriptFMLightSource(channel, filter_set, **common),
        "filter_set": filter_set,
        "objective": AutoscriptFMObjective(channel, **common),
    }
    parts["camera"].configure(config)
    group = AutoscriptFM(channel, **common).fill_roles(**parts)
    return {device.name: device.connect() for device in [group, *parts.values()]}
