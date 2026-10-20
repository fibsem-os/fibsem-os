"""The AutoScript (Thermo Fisher) stage, beams, chamber and manipulator as devices.

``AutoscriptStage`` implements the ``Stage`` device with what ``ThermoMicroscope``
does today, moved as-is, so the old call and the device make the same SDK calls in
the same order. ``AutoscriptCompustage`` is the same for a compustage (Arctis,
Hydra), ``AutoscriptBeam`` for the beam keys, and ``AutoscriptChamber`` and
``AutoscriptManipulator`` for the vacuum and the needle. ``ThermoMicroscope`` builds
them at connect and routes its keys and moves to them; the old code they replaced is
deleted.

Each device is handed the microscope's `AutoScriptConnection` and makes its SDK calls
through it. The vendor stage is ``connection.vendor_stage``, which connect sets to
``specimen.stage`` or ``specimen.compustage`` (``microscope.stage`` is the stage
device); the vendor beams are under ``connection.sdk.beams``. This module imports the SDK only through
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
(``ThermoFisherFluorescenceMicroscope``, below).

The channel. The FM and the beams are one AutoScript connection with one active view,
so whoever sets the view last owns it (FIB-517). ``AutoscriptFMChannel`` is the old
class's ``active_channel()`` as it was: point the connection at the FM for a block, put
the view back after the outermost block, count the depth so a run isn't undone by each
step inside it, and take the microscope's ``imaging_channel`` lock (its
``_threading_lock``, which is the connection's ``lock``) for that bookkeeping only. A parameter in ``needs_channel`` runs in
that scope. The generic claim isn't used because it holds the lock through the read and
always takes it; the old scope skips the lock when the FM already has the view, which is
what keeps the objective movable while live view re-takes the lock every frame.

Live view keeps what the old fast acquisition does, with each frame pulled rather than
pushed: `AutoscriptFM.start_live` switches the light on and starts the acquisition, each
``acquire_frame`` while live is the next ``imaging.get_image()``, and stopping switches
the light off and stops the acquisition.

The FM parts import the SDK names they need in the method that uses them, so the module
still loads where AutoScript is not installed.
"""

from __future__ import annotations

import copy
import logging
import math
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
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
    Union,
)

import numpy as np

from fibsem.devices.beam import Beam, nearest_resolution
from fibsem.devices.chamber import Chamber
from fibsem.devices.chamber_camera import ChamberCamera
from fibsem.devices.core import (
    IMAGING_CHANNEL,
    BoundParameter,
    Device,
    ParameterMetadata,
    Resources,
    resources_of,
)
from fibsem.devices.fm import FM, Camera, FilterSet, LightSource, Objective
from fibsem.devices.gis import GasInjector
from fibsem.devices.manipulator import Manipulator
from fibsem.devices.sample_loader import (
    GridExchangeError,
    Magazine,
    MagazineSlot,
    MagazineSlotState,
    SampleLoader,
    StageSample,
)
from fibsem.devices.sputter_coater import SputterCoater
from fibsem.devices.stage import (
    Stage,
    axis_limits_from_degrees,
    flip_stage_device_at_pose,
    flip_stage_poses,
)
from fibsem.devices.wire import Frame
from fibsem.drivers.autoscript import images
from fibsem.fm import microscope as fm_api
from fibsem.fm.structures import (
    REFLECTION,
    ChannelSettings,
    EmissionFilter,
    matching_emission_filter,
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
    Resolution,
    ScanMode,
)

if TYPE_CHECKING:
    from fibsem.drivers.autoscript.connection import AutoScriptConnection
    from fibsem.drivers.autoscript.microscope import ThermoMicroscope
    from fibsem.drivers.registry import BuildContext
    from fibsem.fm.structures import CameraSettings
    from fibsem.microscope import FibsemMicroscope


class AutoscriptStage(Stage):
    """The AutoScript stage on an offset (non-compustage) mount.

    Each method is what the matching part of ``ThermoMicroscope`` does today:

    - ``read_position``: the ``stage_position`` branch of ``_get``, including setting the
      default coordinate system before every read;
    - ``read_homed`` / ``read_linked``: the ``stage_homed`` / ``stage_linked`` branches;
    - ``metadata_position``: the SDK's axis limits, which also say which axes exist;
    - ``_move_absolute``: ``move_stage_absolute``, with its working-distance restore
      and its axis restrictions;
    - ``_move_relative``: ``move_stage_relative``;
    - ``_home`` / ``_link``: the ``stage_home`` / ``stage_link`` branches of ``_set``.

    The old moves end by reading the position back; ``Stage.move_absolute`` and
    ``move_relative`` do the same read, so a move here is the same calls end to end.
    Limits and positions share one unit: radians for r and t.
    """

    compustage = False

    def __init__(
        self,
        parent: ThermoMicroscope,
        connection: AutoScriptConnection,
        resources: Optional[Resources] = None,
    ):
        super().__init__(parent=parent, resources=resources)
        self.connection = connection

    @property
    def _stage(self):
        """The vendor stage, looked up on each call as the old methods do."""
        return self.connection.vendor_stage

    def _to_autoscript(self, position: FibsemStagePosition):
        from fibsem.drivers.autoscript.microscope import stage_position_to_autoscript

        return stage_position_to_autoscript(position, compustage=self.compustage)

    # -- position ----------------------------------------------------------------

    def read_position(self) -> FibsemStagePosition:
        from fibsem.drivers.autoscript.microscope import stage_position_from_autoscript

        self._stage.set_default_coordinate_system(
            self.connection.default_stage_coordinate_system
        )
        return stage_position_from_autoscript(self._stage.current_position)

    def metadata_position(self) -> ParameterMetadata:
        # The SDK gives x, y, z in metres and t in radians, the axes' own units. It has
        # no limit for r, which turns freely: one full turn either way.
        if not hasattr(self._stage, "get_axis_limits"):
            from fibsem.drivers.demo.simulator import STAGE_LIMITS_DEFAULT

            return ParameterMetadata(
                limits=axis_limits_from_degrees(STAGE_LIMITS_DEFAULT)
            )
        limits: Dict[str, RangeLimit] = {}
        for axis in ("x", "y", "z", "t"):
            axis_limit = self._stage.get_axis_limits(axis)
            limits[axis] = RangeLimit(min=axis_limit.min, max=axis_limit.max)
        limits["r"] = RangeLimit(min=-2 * math.pi, max=2 * math.pi)
        return ParameterMetadata(limits=limits)

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
            self.parent.stage_movement.without_blocked_axes(position)
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
    table, which has no r.
    """

    compustage = True

    def metadata_position(self) -> ParameterMetadata:
        from fibsem.drivers.demo.simulator import STAGE_LIMITS_COMPUSTAGE

        return ParameterMetadata(
            limits=axis_limits_from_degrees(STAGE_LIMITS_COMPUSTAGE)
        )

    def available_linked(self) -> bool:
        return False

    def has_builtin_shuttle(self) -> bool:
        return True

    def mirrors_back_view(self) -> bool:
        # xT presents an image of the back of the grid mirrored, as if from the front
        return True

    def poses(
        self, rotation_reference: float, shuttle_pre_tilt: float, fib_column_tilt: float
    ) -> Dict[str, FibsemStagePosition]:
        return flip_stage_poses(rotation_reference, shuttle_pre_tilt, fib_column_tilt)

    def device_at_pose(self, orientation: str) -> Optional[str]:
        return flip_stage_device_at_pose(orientation)


def autoscript_stage_class(microscope: ThermoMicroscope) -> Type[AutoscriptStage]:
    """The driver class for the stage the Thermo backend found at connect."""
    compustage = microscope._connection.compustage_installed
    return AutoscriptCompustage if compustage else AutoscriptStage


def bind_autoscript_stage(
    microscope: ThermoMicroscope, resources: Optional[Resources] = None
) -> AutoscriptStage:
    """Build ``stage`` for a connected Thermo microscope."""
    stage = autoscript_stage_class(microscope)
    return stage(microscope, microscope._connection, resources).connect()


def _point(value: Any) -> Point:
    """An AutoScript Point as a fibsem Point."""
    return Point(float(value.x), float(value.y))


def _point_limits(limits: Any) -> Dict[str, RangeLimit]:
    """An AutoScript Limits2d as a Point parameter's limits."""
    return {
        "x": RangeLimit(min=limits.limits_x.min, max=limits.limits_x.max),
        "y": RangeLimit(min=limits.limits_y.min, max=limits.limits_y.max),
    }


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

    ``acquire``, ``autocontrast`` and ``auto_focus`` are the old
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
        connection: AutoScriptConnection,
        resources: Optional[Resources] = None,
    ):
        super().__init__(beam_type, parent=parent, resources=resources)
        self.connection = connection
        self.bind_channel(lambda: connection.set_channel(beam_type))

    @property
    def _beam(self) -> Any:
        """The vendor beam, looked up on every call, as the old branches do."""
        return self.connection.beam(self.beam_type)

    def _set_log(self, what: str, value: Any, unit: str) -> None:
        logging.debug(
            f"{self.beam_type.name} {what} set to {value}{unit}.",
            stacklevel=2,  # name the write_* method, not this helper
        )

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
        logging.debug(
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
        logging.debug(f"{self.beam_type.name} HFW set to {value} m.")

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

    def metadata_shift(self) -> ParameterMetadata:
        # the limits xT reports; Beam.shift_by clips to them
        limits = self._beam.beam_shift.limits
        return ParameterMetadata(
            limits={
                "x": RangeLimit(min=limits.limits_x.min, max=limits.limits_x.max),
                "y": RangeLimit(min=limits.limits_y.min, max=limits.limits_y.max),
            }
        )

    def read_stigmation(self) -> Point:
        beam = self._beam
        return Point(beam.stigmator.value.x, beam.stigmator.value.y)

    def write_stigmation(self, value: Point) -> None:
        from autoscript_sdb_microscope_client.structures import Point as ThermoPoint

        self._beam.stigmator.value = ThermoPoint(value.x, value.y)
        self._set_log("stigmation", value, "")

    # The electron column's source tilt (radians) and lens alignment (metres), each
    # an AutoScript Point with Limits2d. A column without one answers unavailable.
    def available_source_tilt(self) -> bool:
        return self._has_electron("source_tilt")

    def read_source_tilt(self) -> Point:
        return _point(self._beam.source_tilt.value)

    def write_source_tilt(self, value: Point) -> None:
        self._beam.source_tilt.value = [value.x, value.y]
        self._set_log("source tilt", value, " rad")

    def metadata_source_tilt(self) -> ParameterMetadata:
        return ParameterMetadata(limits=_point_limits(self._beam.source_tilt.limits))

    def available_lens_alignment(self) -> bool:
        return self._has_electron("lens_alignment")

    def read_lens_alignment(self) -> Point:
        return _point(self._beam.lens_alignment.value)

    def write_lens_alignment(self, value: Point) -> None:
        self._beam.lens_alignment.value = [value.x, value.y]
        self._set_log("lens alignment", value, " m")

    def metadata_lens_alignment(self) -> ParameterMetadata:
        return ParameterMetadata(limits=_point_limits(self._beam.lens_alignment.limits))

    def _has_electron(self, name: str) -> bool:
        """Whether this is the electron column and it has *name*; an AutoScript or
        an instrument without it is "no"."""
        if self.beam_type is not BeamType.ELECTRON:
            return False
        try:
            return getattr(self._beam, name) is not None
        except Exception:  # noqa: BLE001 - an older SDK, or a column without it
            return False

    def read_resolution(self) -> List[int]:
        # a list, as the old get returns it
        return list(Resolution.parse(self._beam.scanning.resolution.value))

    def write_resolution(self, value: Tuple[int, int]) -> None:
        self._beam.scanning.resolution.value = str(Resolution(*value))

    def metadata_resolution(self) -> ParameterMetadata:
        return ParameterMetadata(
            choices=[
                tuple(Resolution.parse(r))
                for r in self._beam.scanning.resolution.available_values
            ]
        )

    # The detector: the active device's, so these run with this beam's channel
    # selected (needs_channel). The writes check as the old branches do.

    @property
    def _detector(self) -> Any:
        return self.connection.sdk.detector

    def read_detector_type(self) -> str:
        return self._detector.type.value

    def write_detector_type(self, value: str) -> None:
        detector = self._detector
        if value in detector.type.available_values:
            detector.type.value = value
            logging.debug(f"Detector type set to {value}.")
        else:
            logging.warning(f"Detector type {value} not available.")

    def metadata_detector_type(self) -> ParameterMetadata:
        # read outside a parameter's claim, so it selects the channel itself
        with self.connection.lock:
            self.connection.set_channel(self.beam_type)
            return ParameterMetadata(choices=list(self._detector.type.available_values))

    def read_detector_mode(self) -> str:
        return self._detector.mode.value

    def write_detector_mode(self, value: str) -> None:
        detector = self._detector
        if value in detector.mode.available_values:
            detector.mode.value = value
            logging.debug(f"Detector mode set to {value}.")
        else:
            logging.warning(f"Detector mode {value} not available.")

    def metadata_detector_mode(self) -> ParameterMetadata:
        # the detector type's modes, read again when the type changes; read outside a
        # parameter's claim, so it selects the channel itself
        with self.connection.lock:
            self.connection.set_channel(self.beam_type)
            return ParameterMetadata(choices=list(self._detector.mode.available_values))

    def read_detector_brightness(self) -> float:
        return self._detector.brightness.value

    def write_detector_brightness(self, value: float) -> None:
        if 0 < value <= 1:
            self._detector.brightness.value = value
            logging.debug(f"Detector brightness set to {value}.")
        else:
            logging.warning(
                f"Detector brightness {value} not available, must be between 0 and 1."
            )

    def read_detector_contrast(self) -> float:
        return self._detector.contrast.value

    def write_detector_contrast(self, value: float) -> None:
        if 0 < value <= 1:
            self._detector.contrast.value = value
            logging.debug(f"Detector contrast set to {value}.")
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
            mode = self._beam.scanning.mode.value
        except Exception as e:
            logging.warning(f"{self.beam_type.name} scan mode could not be read: {e}")
            return None
        # AutoScript reads an int from its ScanningMode enumeration; a name is read
        # too, in case a version reports one.
        if isinstance(mode, int):
            found = _SCAN_MODE_VALUES.get(mode)
        else:
            found = _SCAN_MODES.get(str(mode).replace("_", "").replace(" ", "").lower())
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
        logging.debug(f"Angular correction angle set to {value} radians.")

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

    def installed_plasma_gas(self) -> Optional[str]:
        """``source.plasma_gas.value``, the read ``plasma_gas`` makes. Only a plasma
        source has one; on a Ga column the attribute is missing or the read raises,
        and either is "cannot say"."""
        if self.beam_type is not BeamType.ION:
            return None
        plasma_gas = getattr(self._beam.source, "plasma_gas", None)
        if plasma_gas is None:
            return None
        return plasma_gas.value or None

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
    # acquire_image3's current-settings path), autocontrast and auto_focus,
    # moved as they are. The vendor call runs with this beam's channel claimed
    # (claim_channel), which is the old `_threading_lock` + `set_channel` pair, and the
    # FibsemImage is built from get_microscope_state as before.

    def scan_resolution(self, resolution: Tuple[int, int]) -> Tuple[int, int]:
        """GrabFrameSettings takes only a listed resolution: the nearest one."""
        return nearest_resolution(resolution, self.resolution.choices)

    def _acquire(self, image_settings: Optional[ImageSettings]) -> FibsemImage:
        from fibsem.drivers.autoscript import microscope as thermo

        microscope = self.parent
        name = self.beam_type.name
        if image_settings is None:
            # acquire_image(beam_type=...): the beam's current settings
            settings = self.image_settings()
            logging.debug(f"acquiring new {name} image.")
            with self.claim_channel():
                adorned = self.connection.sdk.imaging.grab_frame(None)
            logging.debug(f"acquiring new {name} image.")
        else:
            settings = image_settings
            resolution = self.scan_resolution(settings.resolution)
            if resolution != tuple(settings.resolution):
                logging.warning(
                    f"Resolution {settings.resolution} not available, using {resolution}"
                )
                settings = copy.deepcopy(settings)
                settings.resolution = resolution
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
            logging.debug(f"acquiring new {name} image.")
            frame_settings = thermo.GrabFrameSettings(
                resolution=str(Resolution(*settings.resolution)),
                dwell_time=settings.dwell_time,
                reduced_area=reduced_area,
                line_integration=settings.line_integration,
                scan_interlacing=settings.scan_interlacing,
                frame_integration=settings.frame_integration,
                drift_correction=settings.drift_correction,
            )
            with self.claim_channel():
                adorned = self.connection.sdk.imaging.grab_frame(frame_settings)
            if settings.reduced_area is not None:
                self.full_frame()

        state = microscope.get_microscope_state(beam_type=self.beam_type)
        image = images.fibsem_image_from_adorned_image(
            copy.deepcopy(adorned), copy.deepcopy(settings), copy.deepcopy(state)
        )
        logging.debug({"msg": "acquire_image", "metadata": image.metadata.to_dict()})
        return image

    def _autocontrast(self, reduced_area: Optional[FibsemRectangle]) -> None:
        # The routine optimises the active detector in the active view, so the channel
        # is held for all of it, with the reduced area set inside (FIB-569).
        logging.debug(f"Running autocontrast on {self.beam_type.name}.")
        with self.claim_channel():
            if reduced_area is not None:
                self.reduced_area(reduced_area)
            self.connection.sdk.auto_functions.run_auto_cb()
        if reduced_area is not None:
            self.full_frame()
        logging.debug({"msg": "autocontrast", "beam_type": self.beam_type.name})

    def _auto_focus(self, reduced_area: Optional[FibsemRectangle]) -> None:
        # Held for the whole routine, as autocontrast is: it runs in the active view.
        logging.debug(f"Running auto-focus on {self.beam_type.name}.")
        with self.claim_channel():
            if reduced_area is not None:
                self.reduced_area(reduced_area)
            self.connection.sdk.auto_functions.run_auto_focus()
        if reduced_area is not None:
            self.full_frame()
        logging.debug({"msg": "auto_focus", "beam_type": self.beam_type.name})

    # Live view: ThermoMicroscope's _acquisition_worker and _fast_acquisition_worker,
    # moved as they are, with this beam's live_frame in place of the microscope's
    # signal (which forwards it) and the stop event the beam gives.

    def _live(self, stop: threading.Event) -> None:
        self.connection.set_channel(self.beam_type)
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

        imaging = self.connection.sdk.imaging
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
                    image = self._image_from_adorned(adorned_image)
                    logging.debug(f"Acquired Image: {image.data.shape}")
                    self.live_frame.emit(image)
        except Exception as e:
            logging.error(f"Exception occurred during fast acquisition: {e}")
        finally:
            imaging.stop_acquisition()

    def _image_from_adorned(self, adorned_image: Any) -> FibsemImage:
        """A live frame as a FibsemImage, with the beam's current state."""
        microscope = self.parent
        state = microscope.get_microscope_state(beam_type=self.beam_type)
        image_settings = self.image_settings()

        image = images.fibsem_image_from_adorned_image(
            copy.deepcopy(adorned_image),
            copy.deepcopy(image_settings),
            copy.deepcopy(state),
        )

        self.session.stamp(image)

        return image


# The vendor's scan mode names (FullFrame, ReducedArea, Spot), lower-cased.
_SCAN_MODES: Dict[str, ScanMode] = {
    "fullframe": ScanMode.FULL_FRAME,
    "reducedarea": ScanMode.REDUCED_AREA,
    "spot": ScanMode.SPOT,
}

# AutoScript's ScanningMode values (FULL_FRAME 1, LINE 2, SPOT 3, REDUCED_AREA 4,
# EXTERNAL 5, OTHER 6, CROSSOVER 7); the ones with no ScanMode read None.
_SCAN_MODE_VALUES: Dict[int, ScanMode] = {
    1: ScanMode.FULL_FRAME,
    3: ScanMode.SPOT,
    4: ScanMode.REDUCED_AREA,
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
        beam_type: AutoscriptBeam(
            beam_type, microscope, microscope._connection, resources
        ).connect()
        for beam_type, on in enabled.items()
        if on
    }


# -- the chamber and the manipulator --------------------------------------------------


# The chamber camera's place on the channel the beams and the FM share.
CHAMBER_CAMERA_VIEW = 4
CHAMBER_CAMERA_DEVICE = 3


class AutoscriptChamberCamera(ChamberCamera):
    """The chamber camera: view 4, device 3 on the shared imaging channel.

    A glance at it takes the channel away from whatever had it, so the frame is grabbed
    under the channel lock and the view put back in a ``finally``: leaving the
    connection on the chamber camera strands whoever was mid-operation, which is
    FIB-517 with a different thief (FIB-545). It restores the view alone, because
    ``set_active_device`` changes the device *in the active view*, so the device comes
    back with it.
    """

    def __init__(
        self,
        parent: ThermoMicroscope,
        connection: AutoScriptConnection,
        resources: Optional[Resources] = None,
    ):
        super().__init__(parent=parent, resources=resources)
        self.connection = connection

    def _acquire(self) -> np.ndarray:
        imaging = self.connection.sdk.imaging
        with self.connection.lock:
            restore_view = imaging.get_active_view()
            imaging.set_active_view(CHAMBER_CAMERA_VIEW)
            imaging.set_active_device(CHAMBER_CAMERA_DEVICE)
            try:
                image = imaging.get_image()
            finally:
                imaging.set_active_view(restore_view)
        logging.debug({"msg": "acquire_chamber_image"})
        return image.data


class AutoscriptChamber(Chamber):
    """The AutoScript vacuum, ``connection.vacuum``.

    ``read_state``/``read_pressure`` are the ``chamber_state``/``chamber_pressure``
    branches of ``ThermoMicroscope._get``, and ``_pump``/``_vent`` the
    ``pump_chamber``/``vent_chamber`` branches of ``_set``. The old key returns the
    vendor's name for the state ("Pumped"); the device reads it as a ``ChamberState``.
    """

    def __init__(
        self,
        parent: ThermoMicroscope,
        connection: AutoScriptConnection,
        resources: Optional[Resources] = None,
    ):
        super().__init__(parent=parent, resources=resources)
        self.connection = connection

    @property
    def _vacuum(self) -> Any:
        return self.connection.sdk.vacuum

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
    return AutoscriptChamber(microscope, microscope._connection, resources).connect()


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

    def __init__(
        self,
        parent: ThermoMicroscope,
        connection: AutoScriptConnection,
        resources: Optional[Resources] = None,
    ):
        super().__init__(parent=parent, resources=resources)
        self.connection = connection
        self._capacity = self.DEFAULT_CAPACITY

    @property
    def _autoloader(self) -> Any:
        return self.connection.sdk.specimen.autoloader

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
        return bool(microscope._connection.sdk.specimen.autoloader.is_installed)
    except Exception:  # noqa: BLE001 - device absent, or not ready
        return False


_MANIPULATOR_NAMES = ("PARK", "EUCENTRIC")


def _in_raw_unless_named(
    position: FibsemManipulatorPosition,
) -> FibsemManipulatorPosition:
    """A position that names no coordinate system moves in the needle's raw frame."""
    if position.coordinate_system is not None:
        return position
    return replace(position, coordinate_system="RAW")


class AutoscriptManipulator(Manipulator):
    """The AutoScript needle, ``connection.specimen.manipulator``.

    Each method is what the matching ``ThermoMicroscope`` method does today, without
    the read-back, which the device's commands make: ``_insert``
    (``insert_manipulator``), ``_retract`` (``retract_manipulator``),
    ``_move_relative``/``_move_absolute`` (``move_manipulator_relative``/
    ``_absolute``) and ``saved_position`` (``_get_saved_manipulator_position``).
    The needle's axes tilt with the stage, so the corrected and offset moves
    (``_move_corrected``, ``_offset_target``) correct for the tilt of the stage in its
    ``stage`` role.
    """

    def __init__(
        self,
        parent: ThermoMicroscope,
        connection: AutoScriptConnection,
        resources: Optional[Resources] = None,
    ):
        super().__init__(parent=parent, resources=resources)
        self.connection = connection

    @property
    def _needle(self) -> Any:
        return self.connection.sdk.specimen.manipulator

    def is_installed(self) -> Optional[bool]:
        """``specimen.manipulator.is_installed``: documented, read-only, a bool."""
        return bool(self._needle.is_installed)

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

        delta = _in_raw_unless_named(delta)
        logging.info(f"moving manipulator by {delta}")
        self._needle.relative_move(manipulator_position_to_autoscript(delta))
        logging.debug({"msg": "move_manipulator_relative", "position": delta.to_dict()})

    def _move_absolute(self, position: FibsemManipulatorPosition) -> None:
        from fibsem.drivers.autoscript.microscope import (
            manipulator_position_to_autoscript,
        )

        position = _in_raw_unless_named(position)
        logging.info(f"moving manipulator to {position}")
        self._needle.absolute_move(manipulator_position_to_autoscript(position))
        logging.debug(
            {"msg": "move_manipulator_absolute", "position": position.to_dict()}
        )

    # -- corrected moves -----------------------------------------------------------
    #
    # The needle moves in stage coordinates, whose y and z tilt with the stage. In the
    # electron view an image dy is along the tilted y; in the ion view it is along the
    # tilted z. x needs no correction.

    @staticmethod
    def _y_corrected(expected_y: float, stage_tilt: float) -> FibsemManipulatorPosition:
        """The needle move for an image dy in the electron view."""
        y_move = +np.cos(stage_tilt) * expected_y
        z_move = +np.sin(stage_tilt) * expected_y
        return FibsemManipulatorPosition(x=0, y=y_move, z=z_move)

    @staticmethod
    def _z_corrected(expected_z: float, stage_tilt: float) -> FibsemManipulatorPosition:
        """The needle move for an image dy in the ion view."""
        y_move = -np.sin(stage_tilt) * expected_z
        z_move = +np.cos(stage_tilt) * expected_z
        return FibsemManipulatorPosition(x=0, y=y_move, z=z_move)

    def _stage_tilt(self) -> float:
        return self.stage.position.get_value().t

    def _move_corrected(self, dx: float, dy: float, beam_type: BeamType) -> None:
        """BeamType.ELECTRON moves in x, y and BeamType.ION in x, z (raw coordinates),
        corrected for the stage tilt."""
        from fibsem.drivers.autoscript.microscope import ManipulatorCoordinateSystem

        stage_tilt = self._stage_tilt()
        if beam_type is BeamType.ELECTRON:
            yz_move = self._y_corrected(dy, stage_tilt)
        elif beam_type is BeamType.ION:
            yz_move = self._z_corrected(dy, stage_tilt)
        else:
            raise ValueError(f"Unknown beam type: {beam_type}")

        # explicitly set the coordinate system
        self._needle.set_default_coordinate_system(ManipulatorCoordinateSystem.STAGE)
        self._move_relative(
            FibsemManipulatorPosition(
                x=dx, y=yz_move.y, z=yz_move.z, r=0.0, coordinate_system="STAGE"
            )
        )

    def _offset_target(
        self, offset: FibsemManipulatorPosition, name: str
    ) -> FibsemManipulatorPosition:
        saved_position = self.saved_position(name)

        # calculate corrected manipulator movement
        yz_move = self._z_corrected(offset.z, self._stage_tilt())

        # adjust for offset
        saved_position.x += offset.x
        saved_position.y += yz_move.y + offset.y
        saved_position.z += yz_move.z  # RAW, up = negative, STAGE: down = negative
        saved_position.r = None  # rotation is not supported

        logging.debug(
            {
                "msg": "move_manipulator_to_position_offset",
                "name": name,
                "offset": offset.to_dict(),
                "saved_position": saved_position.to_dict(),
            }
        )
        return saved_position


def bind_autoscript_manipulator(
    microscope: ThermoMicroscope, resources: Optional[Resources] = None
) -> AutoscriptManipulator:
    """Build ``manipulator`` for a connected Thermo microscope that has one."""
    return AutoscriptManipulator(
        microscope, microscope._connection, resources
    ).connect()


# -- The gas injector and the sputter coater ----------------------------------------
#
# Built on an Arctis and a Hydra Bio only (``thermo_variant``), the Thermo instruments
# fibsem-os drives them on. Anything else gets neither device, and a configuration
# that names one there is refused. An entry's ``variant`` forces one on a bench.

ARCTIS = "arctis"
HYDRA_BIO = "hydra_bio"
THERMO_VARIANTS = (ARCTIS, HYDRA_BIO)

MULTICHEM = "Multichem"
"""The ``port`` a ``gis`` entry names to use the multichem rather than a GIS port."""

GIS_READ_BACK_VERSION = "4.13.1"
"""From this AutoScript, ``GisPort`` reads back ``state``, ``is_heated`` and
``is_open``. Before it, the device reports what its own commands last did."""

GIS_HEATED_TEMPERATURE = 300.0
"""Kelvin. ``heater_on`` waits for the gas to reach it, as the old driver did."""

GIS_HEATER_TIMEOUT = 15.0
"""Seconds ``heater_on`` waits for ``GIS_HEATED_TEMPERATURE`` before it fails."""

HYDRA_BIO_GIS_PORT = "Pt dep Cryo"
"""The port a Hydra Bio deposits from, when the entry names none and it is listed:
the cryo platinum port the bench runs used."""

HYDRA_BIO_SPUTTER_CURRENT = 1.2e-7
"""Amperes (0.12 uA). A Hydra Bio coater runs at 1 nA unless set, so each run sets
the device's current, this until a caller sets another."""


def thermo_variant(
    microscope: ThermoMicroscope, entry: Optional[DeviceEntry] = None
) -> Optional[str]:
    """``ARCTIS`` or ``HYDRA_BIO`` by the model name xT reports, or None for any other
    instrument. An entry's ``variant`` wins. A Hydra is taken for a Hydra Bio: the
    devices' own ``is_installed`` keeps one without a coater or GIS from building."""
    forced = entry.options.get("variant") if entry is not None else None
    if forced is not None:
        if forced not in THERMO_VARIANTS:
            raise ValueError(f"variant is one of {THERMO_VARIANTS}, not {forced!r}")
        return forced
    model = str(getattr(microscope.system.info, "model", None) or "").lower()
    if "arctis" in model:
        return ARCTIS
    if "hydra" in model:
        return HYDRA_BIO
    return None


def _supported_variant(
    microscope: ThermoMicroscope, entry: DeviceEntry, what: str
) -> str:
    variant = thermo_variant(microscope, entry)
    if variant is None:
        raise ValueError(
            f"a Thermo {what} is supported on an Arctis or a Hydra Bio only, not on "
            f"{microscope.system.info.model!r}"
        )
    return variant


def gis_read_back_supported(connection: AutoScriptConnection) -> bool:
    """Whether both AutoScript ends are at least ``GIS_READ_BACK_VERSION``; False when
    either version can't be read."""
    from packaging.version import parse as parse_version

    try:
        autoscript = connection.sdk.service.autoscript
        versions = (autoscript.client.version, autoscript.server.version)
        return all(
            parse_version(str(v)) >= parse_version(GIS_READ_BACK_VERSION)
            for v in versions
        )
    except Exception:  # noqa: BLE001 - an old server, or a version string we can't parse
        return False


def _retractable_state(state: Any) -> InsertableDeviceState:
    """A vendor ``RetractableDeviceState`` ("Inserted", "Retracted", ...)."""
    name = str(getattr(state, "name", state)).upper()
    if name == "INSERTED":
        return InsertableDeviceState.INSERTED
    if name == "RETRACTED":
        return InsertableDeviceState.RETRACTED
    return InsertableDeviceState.UNKNOWN


class _AutoscriptGasInjector(GasInjector):
    """What a GIS port and the multichem share: the vendor object, looked up on every
    call, the heater's wait for temperature, and the state the device tracks where
    AutoScript can't read it back.

    The old driver's insertion guard (raw stage z at most 4 mm) isn't kept: it was set
    for a bulk stage, and an Arctis compustage's raw z is another frame.

    The heater is left on between grids: turning it off and on again leaves the valve
    reading "GIS port valve is not controllable" until the gas is back at temperature.
    """

    def __init__(
        self,
        parent: ThermoMicroscope,
        connection: AutoScriptConnection,
        resources: Optional[Resources] = None,
    ):
        super().__init__(parent=parent, resources=resources)
        self.connection = connection
        self._inserted = False
        self._heated = False
        self._opened = False

    @property
    def _vendor(self) -> Any:
        raise NotImplementedError

    def _insert(self, position: Optional[str]) -> None:
        vendor = self._vendor
        if position:
            logging.info(f"Inserting {self.name} to {position}")
            vendor.insert(position)
        else:
            logging.info(f"Inserting {self.name}")
            vendor.insert()
        self._inserted = True

    def _retract(self) -> None:
        logging.info(f"Retracting {self.name}")
        self._vendor.retract()
        self._inserted = False

    def _open(self) -> None:
        logging.info(f"Opening the {self.name} valve")
        self._vendor.open()
        self._opened = True

    def _close(self) -> None:
        logging.info(f"Closing the {self.name} valve")
        self._vendor.close()
        self._opened = False

    def _wait_for_temperature(self) -> None:
        """Wait for the heating to finish (``_heating_done``) or the gas to reach
        ``GIS_HEATED_TEMPERATURE``; raise ``TimeoutError`` after
        ``GIS_HEATER_TIMEOUT``. The vendor's ``turn_heater_on`` returns before the
        heating is done."""
        deadline = time.monotonic() + GIS_HEATER_TIMEOUT
        while True:
            temperature = self.read_temperature()
            logging.info(
                f"{self.name} heater: {temperature:.1f} K, "
                f"waiting for {GIS_HEATED_TEMPERATURE:.0f} K"
            )
            if self._heating_done() or temperature >= GIS_HEATED_TEMPERATURE:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"{self.name}: the gas reached {temperature:.1f} K, not "
                    f"{GIS_HEATED_TEMPERATURE:.0f} K, within {GIS_HEATER_TIMEOUT:.0f} s."
                )
            time.sleep(1.0)

    def _heating_done(self) -> bool:
        """Whether the instrument reports the heating finished; False where it
        can't say."""
        return False


class AutoscriptGisPort(_AutoscriptGasInjector):
    """One GIS port, ``gas.get_gis_port(port)``; ``port`` is one of
    ``gas.list_all_gis_ports()``, which names each port by its gas.

    From AutoScript 4.13.1 (both ends, ``gis_read_back_supported``) ``state``,
    ``heated`` and ``opened`` are the port's ``state``, ``is_heated`` and ``is_open``,
    so a change made in xT shows. Before it they are what this device's own commands
    last did: retracted, off and closed until then. A purge is the base class's open,
    wait, close.
    """

    def __init__(
        self,
        parent: ThermoMicroscope,
        connection: AutoScriptConnection,
        port: str,
        read_back: bool = False,
        resources: Optional[Resources] = None,
    ):
        super().__init__(parent, connection, resources)
        self.port = port
        self.read_back = read_back

    @property
    def _vendor(self) -> Any:
        return self.connection.sdk.gas.get_gis_port(self.port)

    def is_installed(self) -> Optional[bool]:
        return self.port in self.connection.sdk.gas.list_all_gis_ports()

    def read_gas(self) -> str:
        return self.port

    def read_state(self) -> InsertableDeviceState:
        if self.read_back:
            return _retractable_state(self._vendor.state)
        if self._inserted:
            return InsertableDeviceState.INSERTED
        return InsertableDeviceState.RETRACTED

    def read_heated(self) -> bool:
        # with read-back, is_heated: "True if GIS heating has completed"
        return bool(self._vendor.is_heated) if self.read_back else self._heated

    def _heating_done(self) -> bool:
        return self.read_back and bool(self._vendor.is_heated)

    def read_opened(self) -> bool:
        return bool(self._vendor.is_open) if self.read_back else self._opened

    def read_temperature(self) -> float:
        return float(self._vendor.get_temperature())

    def _heater_on(self, gas: Optional[str]) -> None:
        if gas is not None and gas != self.port:
            raise ValueError(
                f"{self.name}: the {self.port} port has no gas {gas!r}; a gas is "
                "chosen on a multichem only."
            )
        logging.info(f"Turning on the {self.name} heater ({self.port})")
        self._vendor.turn_heater_on()
        self._heated = True
        self._wait_for_temperature()

    def _heater_off(self) -> None:
        logging.info(f"Turning off the {self.name} heater")
        self._vendor.turn_heater_off()
        self._heated = False


class AutoscriptMultichem(_AutoscriptGasInjector):
    """The multichem, ``gas.get_multichem()``.

    ``state`` is the multichem's ``state``. Its heater and valve can't be read back
    with the calls documented for AutoScript 4.9, so ``heated`` and ``opened`` are
    what this device's own commands last did. The heater and ``temperature`` take a
    gas name: ``gas`` is the one ``heater_on`` was last given, and reads "" (with no
    temperature) until then. ``insert`` takes a ``MultiChemInsertPosition`` name.

    ``Multichem.purge()`` (AutoScript 4.11) isn't used: its arguments aren't
    documented in the reference this was written from. A purge is the base class's
    open, wait, close.
    """

    def __init__(
        self,
        parent: ThermoMicroscope,
        connection: AutoScriptConnection,
        resources: Optional[Resources] = None,
    ):
        super().__init__(parent, connection, resources)
        self._gas = ""

    @property
    def _vendor(self) -> Any:
        return self.connection.sdk.gas.get_multichem()

    def is_installed(self) -> Optional[bool]:
        return bool(self.connection.sdk.gas.list_all_multichem_ports())

    def read_gas(self) -> str:
        return self._gas

    def read_state(self) -> InsertableDeviceState:
        return _retractable_state(self._vendor.state)

    def read_heated(self) -> bool:
        return self._heated

    def read_opened(self) -> bool:
        return self._opened

    def read_temperature(self) -> float:
        if not self._gas:
            return math.nan
        return float(self._vendor.get_temperature(self._gas))

    def _heater_on(self, gas: Optional[str]) -> None:
        gas = gas or self._gas
        if not gas:
            raise ValueError(f"{self.name}: name the gas to heat on the multichem.")
        logging.info(f"Turning on the {self.name} heater ({gas})")
        self._vendor.turn_heater_on(gas)
        self._gas = gas
        self._heated = True
        self._wait_for_temperature()

    def _heater_off(self) -> None:
        logging.info(f"Turning off the {self.name} heater ({self._gas})")
        self._vendor.turn_heater_off(self._gas)
        self._heated = False


class _AutoscriptSputterCoater(SputterCoater):
    """What the Arctis and Hydra Bio coaters share, ``specimen.sputter_coater``: the
    current and its limits, and around a run both beams on, the ion voltage put back
    and the stage position read back.

    Both beams on, because a Hydra Bio refused with "MicroSputter operation cannot
    start: IonBeam is off". The ion voltage is put back after the run, in a
    ``finally``, because the coater changes it (seen on a Hydra Bio, where the bench
    runs set it back to 30 kV); this puts back what it was before the run. The run
    moves the stage, so its position is read back.
    """

    def __init__(
        self,
        parent: ThermoMicroscope,
        connection: AutoScriptConnection,
        resources: Optional[Resources] = None,
    ):
        super().__init__(parent=parent, resources=resources)
        self.connection = connection
        self._running = False

    @property
    def _coater(self) -> Any:
        return self.connection.sdk.specimen.sputter_coater

    def is_installed(self) -> Optional[bool]:
        return bool(self._coater.is_installed)

    def read_current(self) -> float:
        return float(self._coater.current.value)

    def write_current(self, value: float) -> None:
        self._coater.current.value = value
        logging.debug(f"Sputter coater current set to {value} A.")

    def metadata_current(self) -> ParameterMetadata:
        # Arctis and Hydra Bio select the current from presets; current.limits
        # raises on both (AutoScript reference)
        return ParameterMetadata(
            choices=[float(v) for v in self._coater.current.available_values]
        )

    def read_inserted(self) -> bool:
        # Neither coater is insertable: is_inserted raises on both (AutoScript
        # reference), so this is True only while this device's own run is on.
        return self._running

    def _run(self, seconds: float, grids: Optional[Tuple[int, ...]]) -> None:
        whole = int(round(seconds))  # the vendor takes whole seconds
        if whole < 1:
            raise ValueError(
                f"{self.name}: a run lasts whole seconds, not {seconds} s."
            )
        sdk = self.connection.sdk
        ion = sdk.beams.ion_beam
        voltage = ion.high_voltage.value
        for beam in (sdk.beams.electron_beam, ion):
            if not beam.is_on:
                beam.turn_on()
        self._before_run()
        on = f" on grids {list(grids)}" if grids else ""
        logging.info(f"Sputtering for {whole} s{on}...")
        self._running = True
        try:
            self._vendor_run(whole, grids)
        finally:
            self._running = False
            ion.high_voltage.value = voltage
            stage = self.parent.devices.get("stage")
            if stage is not None:
                stage.position.get_value()

    def _before_run(self) -> None:
        """What the instrument needs after the beams are on and before the run."""

    def _vendor_run(self, seconds: int, grids: Optional[Tuple[int, ...]]) -> None:
        raise NotImplementedError


class AutoscriptArctisSputterCoater(_AutoscriptSputterCoater):
    """The Arctis sputter coater (xT 28, AutoScript 4.8 and later).

    On an Arctis, ``run(time)`` homes the FLM, moves the compustage to the sputter
    position and back, and sputters whatever is on the stage, so ``coats_grids`` is
    False. ``prepare()`` and ``recover()`` aren't supported there.

    The compustage isn't homed first: the run returns it to where it started, so a
    home before the run would leave it at home. AutoScript known issue AST-2132 reads
    "Arctis - MicroSputter - Sputtering from AS fails under certain conditions"; if
    the bench finds a home is one of them, it goes back here.
    """

    def _vendor_run(self, seconds: int, grids: Optional[Tuple[int, ...]]) -> None:
        self._coater.run(seconds)


class AutoscriptHydraBioSputterCoater(_AutoscriptSputterCoater):
    """The Hydra Bio MicroSputter (AutoScript 4.12 and later).

    ``run(seconds, grids)`` moves the stage to each numbered grid itself, so
    ``coats_grids`` is True and a run names its grids: the coater numbers the two
    grids on the shuttle 1 and 2.

    Before each run, as the AutoScript reference's Hydra Bio example does, the coater
    is homed unless ``is_homed``. The coater runs at 1 nA unless its current is set,
    so each run then sets the device's current: ``HYDRA_BIO_SPUTTER_CURRENT`` (0.12
    uA, the bench value) until a caller sets another. The current is one of the
    coater's presets (``current.available_values``); a run refuses one that isn't. Whether ``seconds`` is per grid or for the whole run
    is not known yet; the bench runs coated both grids with one 120 s run.
    """

    coats_grids = True

    def __init__(
        self,
        parent: ThermoMicroscope,
        connection: AutoScriptConnection,
        current: float = HYDRA_BIO_SPUTTER_CURRENT,
        resources: Optional[Resources] = None,
    ):
        super().__init__(parent, connection, resources)
        self._run_current = float(current)

    def write_current(self, value: float) -> None:
        super().write_current(value)
        self._run_current = float(value)

    def _before_run(self) -> None:
        coater = self._coater
        presets = [float(v) for v in coater.current.available_values]
        if presets and not any(
            math.isclose(p, self._run_current, rel_tol=1e-3) for p in presets
        ):
            raise ValueError(
                f"{self.name}: {self._run_current:.2e} A is not one of the coater's "
                f"currents {presets}."
            )
        if not coater.is_homed:
            logging.info(f"Homing {self.name}...")
            coater.home()
        logging.info(f"Setting the sputter current to {self._run_current:.2e} A.")
        coater.current.value = self._run_current

    def _vendor_run(self, seconds: int, grids: Optional[Tuple[int, ...]]) -> None:
        if not grids:
            raise ValueError(f"{self.name}: name the grids to coat (1, 2).")
        self._coater.run(seconds, list(grids))


def gis_fitted(microscope: ThermoMicroscope) -> bool:
    """Whether to build a ``gis`` by default: an Arctis or a Hydra Bio that lists a
    GIS port or a multichem. False when AutoScript can't say."""
    if thermo_variant(microscope) is None:
        return False
    try:
        gas = microscope._connection.sdk.gas
        return bool(gas.list_all_gis_ports()) or bool(gas.list_all_multichem_ports())
    except Exception:  # noqa: BLE001 - no gas API on this instrument, or not ready
        return False


def sputter_coater_fitted(microscope: ThermoMicroscope) -> bool:
    """Whether to build a ``sputter_coater`` by default: an Arctis or a Hydra Bio
    whose coater reports installed. False when AutoScript can't say."""
    if thermo_variant(microscope) is None:
        return False
    try:
        return bool(microscope._connection.sdk.specimen.sputter_coater.is_installed)
    except Exception:  # noqa: BLE001 - no coater API on this instrument, or not ready
        return False


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
    microscope = context.microscope
    return AutoscriptBeam(beam_type, microscope, microscope._connection).connect()


def build_autoscript_stage(
    entry: DeviceEntry, context: BuildContext
) -> AutoscriptStage:
    microscope = context.microscope
    stage = autoscript_stage_class(microscope)
    return _named(stage(microscope, microscope._connection), entry)


def build_autoscript_chamber(
    entry: DeviceEntry, context: BuildContext
) -> AutoscriptChamber:
    microscope = context.microscope
    return _named(AutoscriptChamber(microscope, microscope._connection), entry)


def build_autoscript_chamber_camera(
    entry: DeviceEntry, context: BuildContext
) -> AutoscriptChamberCamera:
    microscope = context.microscope
    return _named(AutoscriptChamberCamera(microscope, microscope._connection), entry)


def build_autoscript_manipulator(
    entry: DeviceEntry, context: BuildContext
) -> AutoscriptManipulator:
    microscope = context.microscope
    return _named(AutoscriptManipulator(microscope, microscope._connection), entry)


def build_autoscript_sample_loader(
    entry: DeviceEntry, context: BuildContext
) -> AutoscriptSampleLoader:
    microscope = context.microscope
    return _named(AutoscriptSampleLoader(microscope, microscope._connection), entry)


def build_autoscript_sputter_coater(
    entry: DeviceEntry, context: BuildContext
) -> _AutoscriptSputterCoater:
    """The sputter coater, on an Arctis or a Hydra Bio. A Hydra Bio entry's
    ``current`` (amperes) replaces ``HYDRA_BIO_SPUTTER_CURRENT``."""
    microscope = context.microscope
    connection = microscope._connection
    if _supported_variant(microscope, entry, "sputter coater") == HYDRA_BIO:
        current = float(entry.options.get("current", HYDRA_BIO_SPUTTER_CURRENT))
        device = AutoscriptHydraBioSputterCoater(microscope, connection, current)
    else:
        device = AutoscriptArctisSputterCoater(microscope, connection)
    if not device.is_installed():
        raise ValueError("the instrument reports no sputter coater")
    return _named(device, entry)


def build_autoscript_gis(
    entry: DeviceEntry, context: BuildContext
) -> _AutoscriptGasInjector:
    """The gas injector, on an Arctis or a Hydra Bio: the port an entry's ``port``
    names (``MULTICHEM`` for the multichem), else on a Hydra Bio
    ``HYDRA_BIO_GIS_PORT`` where it is listed, else the multichem where there is one,
    else the first GIS port."""
    microscope = context.microscope
    variant = _supported_variant(microscope, entry, "GIS")
    connection = microscope._connection
    gas = connection.sdk.gas
    port = entry.options.get("port")
    if port is None and variant == HYDRA_BIO:
        if HYDRA_BIO_GIS_PORT in gas.list_all_gis_ports():
            port = HYDRA_BIO_GIS_PORT
    if port is None:
        if gas.list_all_multichem_ports():
            port = MULTICHEM
        else:
            ports = list(gas.list_all_gis_ports())
            if not ports:
                raise ValueError("the instrument lists no GIS port and no multichem")
            port = ports[0]
    if port == MULTICHEM:
        device = AutoscriptMultichem(microscope, connection)
    else:
        device = AutoscriptGisPort(
            microscope,
            connection,
            str(port),
            read_back=gis_read_back_supported(connection),
        )
    if device.is_installed() is False:
        raise ValueError(f"the instrument has no GIS port {port!r}")
    return _named(device, entry)


# The FM.

FM_ACTIVE_VIEW = 3
"""The FM's view on Arctis, as the old class sets it."""

MULTI_BAND = EmissionFilter(
    "Quad-band", bands=((414, 450), (499.5, 530), (580, 611), (661, 800))
)
"""Thermo's one fluorescence filter; the other choice is reflection. The Arctis and
iFLM datasheets name the Semrock LED-DA/FI/TR/Cy5-B-000 quad-band set; these are
its emitter's (FF01-432/515/595/730) bands, for the 365, 450, 550 and 635 nm LEDs.
AutoScript doesn't report them, so a scope with another set shows these."""

AVAILABLE_FM_WAVELENGTHS = (365, 450, 550, 635)
"""The FM light's excitation wavelengths, in nm, one per AutoScript emission colour."""


def _color_to_wavelength() -> Dict[Any, float]:
    """Each AutoScript emission colour's excitation wavelength, in nm."""
    from autoscript_sdb_microscope_client.enumerations import CameraEmissionType

    # TODO: migrate to using the enumeration from autoscript_sdb_microscope_client
    return dict(
        zip(
            (
                CameraEmissionType.BLUE,
                CameraEmissionType.GREEN_YELLOW,
                CameraEmissionType.RED,
                CameraEmissionType.VIOLET,
            ),
            AVAILABLE_FM_WAVELENGTHS,
        )
    )


# specs:
# arctis: https://assets.thermofisher.com/TFS-Assets/MSD/Datasheets/arctis-cryo-plasma-fib-ds0384-en.pdf
# - 100x magnification
# - 0.75 NA
# - 150 um fov
# - 4mm working distance
# - light source: 365 nm, 450 nm, 550 nm, 635 nm
# iflm: https://assets.thermofisher.com/TFS-Assets/MSD/Datasheets/iflm-correlative-system-ds0499.pdf
# - 20x magnification
# - 0.7 NA
# - 500 um fov
# - 1.3 mm working distance
# - light source: 365 nm, 450 nm, 550 nm, 635 nm

ARCTIS_CONFIGURATION = {
    "name": "ARCTIS",
    "magnification": 100.0,
    "numerical_aperture": 0.75,
    "working_distance": 4e-3,
    "pixel_size": (2.74822695035461e-08, 2.74822695035461e-08),
    "resolution": (4512, 4512),
    "focus_position": 8.0e-3,  # 8000 microns
    "limit_position": 8.6e-3,  # 8600 microns
}

ARCTIS_LWD_CONFIGURATION = {
    "name": "ARCTIS_LWD",
    "magnification": 50.0,
    "numerical_aperture": 0.75,
    "working_distance": 4e-3,
    "pixel_size": (2.74822695035461e-08 * 2, 2.74822695035461e-08 * 2),
    "resolution": (4512, 4512),
    "focus_position": 7.6e-3,  # 7600 microns
    "limit_position": 8.6e-3,  # 8600 microns
}

IFLM_CONFIGURATION = {
    "magnification": 20.0,
    "numerical_aperture": 0.7,
    "working_distance": 1.3e-3,
    "pixel_size": (2.74822695035461e-08, 2.74822695035461e-08),
    "resolution": (4512, 4512),
}

# 2.74822695035461e-08
DEFAULT_CONFIGURATION = ARCTIS_CONFIGURATION  # Default to ARCTIS configuration

HFW = 150e-6  # Horizontal field width for ARCTIS (diagonal)
IFLM_HFW = 500e-6  # Horizontal field width for iFlm


class AutoscriptFMChannel:
    """The FM's share of the microscope's one AutoScript connection, shared by the FM
    devices: the old ``ThermoFisherFluorescenceMicroscope.active_channel()``, moved.
    ``scope()`` says why it works as it does."""

    def __init__(self, connection: AutoScriptConnection, lock: Any):
        self._autoscript = connection
        self.lock = lock
        self._depth = 0
        self._restore_view: Optional[int] = None

    @property
    def connection(self) -> Any:
        """The AutoScript client (``SdbMicroscopeClient``), looked up on each call."""
        return self._autoscript.sdk

    def set_active_channel(self) -> None:
        """Point the connection at the FM and leave it there (the old method)."""
        from autoscript_sdb_microscope_client.enumerations import ImagingDevice

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
        from autoscript_sdb_microscope_client.enumerations import CameraEmissionType

        super().__init__("filter_set", channel, **kwargs)
        self.emission_type = CameraEmissionType.RED

    def read_excitation_wavelength(self) -> float:
        color_to_wavelength = _color_to_wavelength()
        color = self.emission_type
        if color not in color_to_wavelength:
            raise ValueError(
                f"Invalid excitation color: {color}: must be one of "
                f"{list(color_to_wavelength.keys())}"
            )
        return color_to_wavelength[color]

    def write_excitation_wavelength(self, value: float) -> None:
        wavelength_to_color = {v: k for k, v in _color_to_wavelength().items()}
        color = wavelength_to_color.get(value, None)
        if color is None:
            closest = min(wavelength_to_color, key=lambda x: abs(x - value))
            color = wavelength_to_color[closest]
        self.emission_type = color

    def metadata_excitation_wavelength(self) -> ParameterMetadata:
        return ParameterMetadata(choices=sorted(AVAILABLE_FM_WAVELENGTHS))

    def read_emission_filter(self) -> EmissionFilter:
        from autoscript_sdb_microscope_client.enumerations import CameraFilterType

        mode = self._channel.settings().filter.type.value
        if mode is CameraFilterType.FLUORESCENCE:
            return MULTI_BAND
        return REFLECTION

    def write_emission_filter(self, value: EmissionFilter) -> None:
        from autoscript_sdb_microscope_client.enumerations import CameraFilterType

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
        from autoscript_sdb_microscope_client.structures import GrabFrameSettings

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


# The objective's states as AutoScript names them (RetractableDeviceState). A name
# not here reads as UNKNOWN.
_OBJECTIVE_STATES = {
    "Inserted": InsertableDeviceState.INSERTED,
    "Retracted": InsertableDeviceState.RETRACTED,
    "Busy": InsertableDeviceState.MOVING,
    "Error": InsertableDeviceState.ERROR,
}


class AutoscriptFMObjective(_OnTheFMChannel, Objective):
    """The objective: its focus is the camera's focus setting, and inserting or
    retracting it is the detector's. Magnification and numerical aperture are the
    configuration's; ``limit_position`` is the furthest a move may go in."""

    needs_channel = frozenset({"position", "state"})

    def __init__(self, channel: AutoscriptFMChannel, **kwargs: Any):
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
        return _OBJECTIVE_STATES.get(
            self._channel.connection.detector.state, InsertableDeviceState.UNKNOWN
        )

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
        settings = ChannelSettings.from_dict(channel)
        filters = self.filter_set
        filters.excitation_wavelength.write_through(settings.excitation_wavelength)
        filters.emission_filter.write_through(
            matching_emission_filter(
                settings.emission_filter, filters.emission_filter.choices
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
        # so it describes the state the frame was taken in. The time is
        # `FM.acquire_frame`'s, taken before this runs (FIB-1190).
        with self._channel.scope():
            data = self._frame(channel)
            return Frame(data, self._frame_metadata())

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
        from autoscript_sdb_microscope_client.enumerations import ImagingState

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


class ThermoFisherObjectiveLens(fm_api.ObjectiveLens):
    """The FM API's objective over the Thermo objective device, with what the old
    Thermo objective added: its configured focus position, and homing."""

    def __init__(self, device, channel, parent=None):
        super().__init__(device, parent=parent)
        self._channel = channel
        self._focus_position = DEFAULT_CONFIGURATION["focus_position"]

    def is_homed(self) -> bool:
        with self._channel.scope():
            return self._channel.connection.detector.is_homed

    def home(self) -> None:
        with self._channel.scope():
            self._channel.connection.detector.home()
        self._notify_moved()


class ThermoFisherFluorescenceMicroscope(fm_api.FluorescenceMicroscope):
    """The FM API over the Thermo FM devices (Arctis, Hydra, iFLM).

    What the old ``ThermoFisherFluorescenceMicroscope`` did, through its devices: the channel
    scope is the devices' own (``AutoscriptFMChannel``), so a tileset that holds the
    FM's view holds it for every device inside it, and live view is the old fast
    acquisition, pulled.
    """

    objective: ThermoFisherObjectiveLens
    filter_set: fm_api.FilterSet

    def __init__(self, devices, parent: Optional["FibsemMicroscope"] = None):
        super().__init__(devices, parent=parent)
        self._channel = devices["fm"]._channel
        self.objective = ThermoFisherObjectiveLens(
            devices["objective"], self._channel, parent=self
        )
        self.filter_set = fm_api.FilterSet(devices["filter_set"], parent=self)

    @property
    def connection(self) -> Any:
        """The microscope's AutoScript client (``SdbMicroscopeClient``)."""
        return self._channel.connection

    def set_active_channel(self) -> None:
        self._channel.set_active_channel()

    @contextmanager
    def active_channel(self):
        with self._channel.scope():
            yield

    @property
    def fm_settings(self) -> "CameraSettings":
        return self._channel.settings()


def bind_autoscript_fm(
    microscope: ThermoMicroscope,
    resources: Optional[Resources] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Device]:
    """The Thermo FM's parts and group for a connected Thermo microscope, by device
    name. They share the microscope's ``imaging_channel`` lock with the beams.
    *config* is the fm entry's own keys (``mount_transform``)."""
    resources = resources if resources is not None else resources_of(microscope)
    channel = AutoscriptFMChannel(
        microscope._connection, resources.lock(IMAGING_CHANNEL)
    )
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


def build_autoscript_fm(
    entry: DeviceEntry, context: BuildContext
) -> ThermoFisherFluorescenceMicroscope:
    """The FM API over the Thermo FM devices, sharing the microscope's connection and
    its imaging channel lock with the beams.

    Selecting the FM's channel is the only test AutoScript has for whether an FM is
    fitted, so a microscope without one fails here and connects without an FM.
    Live view is pulled by this process's own worker, which never stops asking while
    it runs, so it has no watchdog: a slow frame handler must not end it. A served FM
    keeps the default."""
    microscope = context.microscope
    devices = bind_autoscript_fm(microscope, config=entry.as_block())
    devices["fm"].live_timeout = None
    fm = ThermoFisherFluorescenceMicroscope(devices, parent=microscope)
    fm.set_active_channel()
    return fm
