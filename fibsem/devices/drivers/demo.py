"""The Demo backend's parts as devices: what ``DemoMicroscope`` is built from.

Each device is the matching part of ``LegacyDemoMicroscope``, the Demo before
devices: every parameter and command does what the matching branch of its ``_get``
and ``_set`` (or method) does, and each FM device what the matching part of the
simulated FM does.
Each keeps its own simulated part, copied when it is built from the starting parts
it is given (``start``, a ``DemoParts``) or else from the microscope's own, and
never touches the microscope's again; ``DemoMicroscope`` routes the old keys to
them.
"""

from __future__ import annotations

import logging
import threading
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from fibsem._timing import sim_sleep
from fibsem.devices.beam import Beam
from fibsem.devices.chamber import Chamber
from fibsem.devices.core import Device, ParameterMetadata, Resources, resources_of
from fibsem.devices.fm import FM, Camera, FilterSet, LightSource, Objective
from fibsem.devices.gis import GasInjector
from fibsem.devices.manipulator import Manipulator
from fibsem.devices.stage import Stage, axis_limits_from_degrees, compustage_poses
from fibsem.fm.api import emission_filter_named
from fibsem.fm.microscope import (
    BINNING_VALUES,
    EMISSION_WAVELENGTHS,
    EXCITATION_WAVELENGTHS,
    SIM_CAMERA_EXPOSURE_LIMITS,
    SIM_OBJECTIVE_POSITION_LIMITS,
    SIM_OBJECTIVE_TRAVEL_SECONDS,
    UINT16_MAX,
    UINT16_MIN,
)
from fibsem.fm.structures import ChannelSettings, EmissionFilter, emission_filter_for
from fibsem.structures import (
    BeamSettings,
    BeamType,
    ChamberState,
    FibsemDetectorSettings,
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
from fibsem.util.draw_numbers import draw_text

if TYPE_CHECKING:
    from fibsem.fm.microscope import Camera as FMClassCamera
    from fibsem.fm.microscope import FilterSet as FMClassFilterSet
    from fibsem.fm.microscope import FluorescenceMicroscope
    from fibsem.fm.microscope import LightSource as FMClassLightSource
    from fibsem.fm.microscope import ObjectiveLens as FMClassObjectiveLens
    from fibsem.microscopes.device_demo import DemoMicroscope
    from fibsem.microscopes.simulator import DemoParts


class DemoBeam(Beam):
    """A Demo beam.

    It keeps its own simulated column: ``sim_beam`` and ``sim_detector`` (the beam's
    and detector's settings), ``sim_on``, ``sim_blanked``, ``sim_scanning_mode`` and
    ``sim_scanning_mode_value``, copied when it is built from the starting
    ``electron_system`` or ``ion_system`` (``start``, else the microscope's own), and
    it never touches the microscope's again. The plasma gas is configuration
    (``system.ion``), not beam state, so it is still read and written there.
    """

    def __init__(
        self,
        beam_type: BeamType,
        parent: DemoMicroscope,
        resources: Optional[Resources] = None,
        start: Optional[DemoParts] = None,
    ):
        super().__init__(beam_type, parent=parent, resources=resources)
        source = parent if start is None else start
        start = (
            source.electron_system
            if beam_type is BeamType.ELECTRON
            else source.ion_system
        )
        self.sim_beam: BeamSettings = deepcopy(start.beam)
        self.sim_detector: FibsemDetectorSettings = deepcopy(start.detector)
        self.sim_on: bool = start.on
        self.sim_blanked: bool = start.blanked
        self.sim_scanning_mode: str = start.scanning_mode
        self.sim_scanning_mode_value = deepcopy(start.scanning_mode_value)

    def _choices(self, key: str) -> ParameterMetadata:
        """The simulator's values for a key: Demo's ``get_available_values``."""
        from fibsem.microscopes import simulator as sim

        if key == "current":
            if self.beam_type is BeamType.ION:
                gas = self.read_plasma_gas() if self.available_plasma_gas() else None
                choices = sim.SIMULATOR_BEAM_CURRENTS[BeamType.ION][gas]
            else:
                choices = sim.SIMULATOR_BEAM_CURRENTS[BeamType.ELECTRON]
        elif key == "voltage":
            choices = (
                [2000, 5000, 10000, 20000, 30000]
                if self.beam_type is BeamType.ELECTRON
                else [500, 1000, 2000, 8000, 16000, 30000]
            )
        elif key == "detector_type":
            choices = ["ETD", "TLD", "EDS"]
        elif key == "detector_mode":
            choices = ["SecondaryElectrons", "BackscatteredElectrons", "EDS"]
        elif key == "plasma_gas":
            choices = sim.SIMULATOR_PLASMA_GASES
        else:
            raise KeyError(key)
        return ParameterMetadata(choices=list(choices))

    # Each parameter is the matching Demo branch as it stands.

    def read_voltage(self) -> float:
        return self.sim_beam.voltage

    def write_voltage(self, value: float) -> None:
        self.sim_beam.voltage = value

    def metadata_voltage(self) -> ParameterMetadata:
        return self._choices("voltage")

    def read_current(self) -> float:
        return self.sim_beam.beam_current

    def write_current(self, value: float) -> None:
        self.sim_beam.beam_current = value

    def metadata_current(self) -> ParameterMetadata:
        return self._choices("current")

    def read_working_distance(self) -> float:
        return self.sim_beam.working_distance

    def write_working_distance(self, value: float) -> None:
        self.sim_beam.working_distance = value

    def read_hfw(self) -> float:
        return self.sim_beam.hfw

    def write_hfw(self, value: float) -> None:
        self.sim_beam.hfw = value

    def read_scan_rotation(self) -> float:
        return float(self.sim_beam.scan_rotation)

    def write_scan_rotation(self, value: float) -> None:
        self.sim_beam.scan_rotation = float(value)

    def read_blanked(self) -> bool:
        return self.sim_blanked

    def write_blanked(self, value: bool) -> None:
        self.sim_blanked = value
        if not value and self.sim_scanning_mode == "spot":
            self.parent._burn_into_sample_scene(self.beam_type)  # the spot burn

    def read_detector_type(self) -> str:
        return self.sim_detector.type

    def write_detector_type(self, value: str) -> None:
        self.sim_detector.type = value

    def metadata_detector_type(self) -> ParameterMetadata:
        return self._choices("detector_type")

    def read_detector_mode(self) -> str:
        return self.sim_detector.mode

    def write_detector_mode(self, value: str) -> None:
        self.sim_detector.mode = value

    def metadata_detector_mode(self) -> ParameterMetadata:
        return self._choices("detector_mode")

    def read_detector_contrast(self) -> float:
        return self.sim_detector.contrast

    def write_detector_contrast(self, value: float) -> None:
        self.sim_detector.contrast = value

    def read_detector_brightness(self) -> float:
        return self.sim_detector.brightness

    def write_detector_brightness(self, value: float) -> None:
        self.sim_detector.brightness = value

    def read_resolution(self) -> tuple:
        return self.sim_beam.resolution

    def write_resolution(self, value: tuple) -> None:
        self.sim_beam.resolution = value

    def read_dwell_time(self) -> float:
        return self.sim_beam.dwell_time

    def write_dwell_time(self, value: float) -> None:
        self.sim_beam.dwell_time = value

    # Reads hand back a new Point, as the Demo branch does, so a caller can't change
    # the simulator's state through the value it was given.
    def read_stigmation(self) -> Point:
        return Point(self.sim_beam.stigmation.x, self.sim_beam.stigmation.y)

    def write_stigmation(self, value: Point) -> None:
        self.sim_beam.stigmation = value

    def read_shift(self) -> Point:
        return Point(self.sim_beam.shift.x, self.sim_beam.shift.y)

    def write_shift(self, value: Point) -> None:
        self.sim_beam.shift = value

    def read_on(self) -> bool:
        return self.sim_on

    def write_on(self, value: bool) -> None:
        self.sim_on = value

    def read_scanning_mode(self) -> ScanMode:
        return ScanMode(self.sim_scanning_mode)

    # The scan commands: the spot_mode, reduced_area and full_frame branches of _set.

    def _spot(self, point: Point) -> None:
        self.sim_scanning_mode = "spot"
        self.sim_scanning_mode_value = point

    def _reduced_area(self, area: FibsemRectangle) -> None:
        self.sim_scanning_mode = "reduced_area"
        self.sim_scanning_mode_value = area

    def _full_frame(self) -> None:
        self.sim_scanning_mode = "full_frame"
        self.sim_scanning_mode_value = None

    # Only a plasma ion column has a gas.
    def available_plasma_gas(self) -> bool:
        return self.beam_type is BeamType.ION and self.parent.system.ion.plasma

    def read_plasma_gas(self) -> str:
        return self.parent.system.ion.plasma_gas

    def write_plasma_gas(self, value: str) -> None:
        # An unavailable gas logs and is ignored, as the Demo branch does.
        gases = self._choices("plasma_gas").choices
        logging.info(f"Checking if plasma_gas={value} is available ({BeamType.ION})")
        if value not in gases:
            logging.warning(
                f"Plasma gas {value} not available. Available values: {gases}"
            )
            return
        logging.info(f"Setting plasma gas to {value}... this may take some time...")
        self.parent.system.ion.plasma_gas = value
        logging.info(f"Plasma gas set to {value}.")

    def metadata_plasma_gas(self) -> ParameterMetadata:
        return self._choices("plasma_gas")

    # Imaging, the autofunctions and live view run the Demo's own code (the
    # ``_demo_*`` methods of ``DemoImaging``), so both demos image alike and the
    # contract suite still compares like with like.

    def _acquire(self, image_settings: Optional[ImageSettings]) -> FibsemImage:
        return self.parent._demo_acquire(image_settings, self.beam_type)

    def _last_image(self) -> FibsemImage:
        return self.parent._demo_last_image(self.beam_type)

    def _autocontrast(self, reduced_area: Optional[FibsemRectangle]) -> None:
        self.parent._demo_autocontrast(self.beam_type, reduced_area)

    def _auto_focus(self, reduced_area: Optional[FibsemRectangle]) -> None:
        self.parent._demo_auto_focus(self.beam_type, reduced_area)

    def _live(self, stop: threading.Event) -> None:
        self.parent._demo_live(self.beam_type, stop, self.live_frame.emit)

    # "preset" is not implemented: Demo has no presets, so it is absent on the new
    # API while the old set("preset", ...) stays a logged no-op.


def bind_demo_beams(
    microscope: DemoMicroscope,
    resources: Optional[Resources] = None,
    start: Optional[DemoParts] = None,
) -> Dict[BeamType, Beam]:
    """Build ``beams[BeamType]`` for a connected Demo microscope."""
    resources = resources if resources is not None else resources_of(microscope)
    return {
        beam_type: DemoBeam(beam_type, microscope, resources, start).connect()
        for beam_type in (BeamType.ELECTRON, BeamType.ION)
    }


class DemoStage(Stage):
    """The Demo stage.

    It keeps its own simulated stage in ``sim_position``, ``sim_homed`` and
    ``sim_linked``, copied when it is built from the starting ``stage_system``
    (``start``, else the microscope's own), and it never touches the microscope's
    again. A microscope that builds it routes the stage keys to it
    (``DemoMicroscope``). Each method is what the matching part of
    ``LegacyDemoMicroscope`` does, on that copy:

    - ``read_position``: the ``stage_position`` branch of ``_get``;
    - ``read_homed`` / ``read_linked``: the ``stage_homed`` / ``stage_linked`` branches;
    - ``metadata_position``: ``_get_axis_limits``, which also says which axes exist;
    - ``_move_absolute`` / ``_move_relative``: ``move_stage_absolute`` /
      ``move_stage_relative``;
    - ``_home`` / ``_link``: the ``stage_home`` / ``stage_link`` branches of ``_set``.

    Two things differ from the old calls, both on purpose. Positions come back as a
    copy, never the simulator's live object, so the cached value can't change under a
    reader. And ``_get_axis_limits`` gives r and t in degrees while positions are in
    radians; the metadata here converts them, so limits and values share one unit.
    """

    def __init__(
        self,
        parent: DemoMicroscope,
        resources: Optional[Resources] = None,
        start: Optional[DemoParts] = None,
    ):
        super().__init__(parent=parent, resources=resources)
        start = (parent if start is None else start).stage_system
        self.sim_position: FibsemStagePosition = deepcopy(start.position)
        self.sim_homed: bool = start.is_homed
        self.sim_linked: bool = start.is_linked
        # Read once when built, as a vendor's limits would be at connect.
        self._axis_limits = parent._get_axis_limits()

    # -- position ----------------------------------------------------------------

    def read_position(self) -> FibsemStagePosition:
        sim_sleep(0.1)  # the read delay the Demo branch has
        return deepcopy(self.sim_position)

    def metadata_position(self) -> ParameterMetadata:
        # The axes are the ones the simulator gives limits for: a compustage has no r.
        limits = axis_limits_from_degrees(self._axis_limits)
        return ParameterMetadata(limits=limits)

    # -- homing and linking -----------------------------------------------------------

    def read_homed(self) -> bool:
        return self.sim_homed

    # A compustage can't link: the old set("stage_link") logs and does nothing there,
    # so on the new API "linked" is absent and link() is unavailable.
    def available_linked(self) -> bool:
        return not self.parent.stage_is_compustage

    # The simulator is a compustage or an offset stage by its configuration.
    def poses(
        self, rotation_reference: float, shuttle_pre_tilt: float, fib_column_tilt: float
    ) -> Dict[str, FibsemStagePosition]:
        if self.parent.stage_is_compustage:
            return compustage_poses(
                rotation_reference, shuttle_pre_tilt, fib_column_tilt
            )
        return super().poses(rotation_reference, shuttle_pre_tilt, fib_column_tilt)

    def read_linked(self) -> bool:
        return self.sim_linked

    # -- commands ---------------------------------------------------------------------

    def _move_absolute(self, position: FibsemStagePosition) -> None:
        from fibsem.microscopes.simulator import STAGE_MOVEMENT_SLEEP_TIME

        sim_sleep(STAGE_MOVEMENT_SLEEP_TIME)
        for axis in ("x", "y", "z", "r", "t"):
            value = getattr(position, axis)
            if value is not None:
                setattr(self.sim_position, axis, value)
        logging.debug({"msg": "move_stage_absolute", "position": position.to_dict()})

    def _move_relative(self, delta: FibsemStagePosition) -> None:
        from fibsem.microscopes.simulator import STAGE_MOVEMENT_SLEEP_TIME

        sim_sleep(STAGE_MOVEMENT_SLEEP_TIME)
        self.sim_position += delta
        logging.debug({"msg": "move_stage_relative", "position": delta.to_dict()})

    def _home(self) -> None:
        logging.info("Homing stage...")
        self.sim_homed = True
        logging.info("Stage homed.")

    def _link(self) -> None:
        logging.info("Linking stage...")
        self.sim_linked = True
        logging.info("Stage linked.")


def bind_demo_stage(
    microscope: DemoMicroscope,
    resources: Optional[Resources] = None,
    start: Optional[DemoParts] = None,
) -> DemoStage:
    """Build ``stage`` for a connected Demo microscope."""
    return DemoStage(microscope, resources, start).connect()


def _insertable_state(inserted: bool) -> InsertableDeviceState:
    return (
        InsertableDeviceState.INSERTED if inserted else InsertableDeviceState.RETRACTED
    )


class DemoChamber(Chamber):
    """The Demo chamber.

    It keeps its own simulated chamber in ``sim_state`` and ``sim_pressure``, copied
    when it is built from the starting ``chamber`` (``start``, else the microscope's
    own), and it never touches the microscope's again. A microscope that builds it
    routes the chamber keys to it (``DemoMicroscope``). Each method is what the
    matching part of ``LegacyDemoMicroscope`` does, on that copy:

    - ``read_state`` / ``read_pressure``: the ``chamber_state`` / ``chamber_pressure``
      branches of ``_get``;
    - ``_pump`` / ``_vent``: the ``pump_chamber`` / ``vent_chamber`` branches of
      ``_set``.
    """

    def __init__(
        self,
        parent: DemoMicroscope,
        resources: Optional[Resources] = None,
        start: Optional[DemoParts] = None,
    ):
        super().__init__(parent=parent, resources=resources)
        chamber = (parent if start is None else start).chamber
        self.sim_state = ChamberState.from_name(chamber.state)
        self.sim_pressure: float = chamber.pressure

    def read_state(self) -> ChamberState:
        return self.sim_state

    def read_pressure(self) -> float:
        return self.sim_pressure

    def _pump(self) -> None:
        logging.info("Pumping chamber...")
        self.sim_state = ChamberState.PUMPED
        self.sim_pressure = 1e-6
        logging.info("Chamber pumped.")

    def _vent(self) -> None:
        logging.info("Venting chamber...")
        self.sim_state = ChamberState.VENTED
        self.sim_pressure = 1e5
        logging.info("Chamber vented.")


def bind_demo_chamber(
    microscope: DemoMicroscope,
    resources: Optional[Resources] = None,
    start: Optional[DemoParts] = None,
) -> DemoChamber:
    """Build ``chamber`` for a connected Demo microscope."""
    return DemoChamber(microscope, resources, start).connect()


# Demo's two saved positions. insert_manipulator goes to PARK and
# retract_manipulator to the origin, which is also EUCENTRIC.
_DEMO_PARK = FibsemManipulatorPosition(x=0, y=0, z=180e-6, r=0, t=0)
_DEMO_ORIGIN = FibsemManipulatorPosition(x=0, y=0, z=0, r=0, t=0)


class DemoManipulator(Manipulator):
    """The Demo manipulator.

    It keeps its own simulated needle in ``sim_position`` and ``sim_inserted``,
    copied when it is built from the starting ``manipulator_system`` (``start``,
    else the microscope's own), and it never touches the microscope's again. A microscope that builds it
    routes the manipulator keys to it (``DemoMicroscope``). Each method is what
    the matching part of ``LegacyDemoMicroscope`` does, on that copy:

    - ``read_position`` / ``read_state``: the ``manipulator_position`` /
      ``manipulator_state`` branches of ``_get``;
    - ``saved_position``: ``_get_saved_manipulator_position``;
    - ``_insert`` / ``_retract``: ``insert_manipulator`` / ``retract_manipulator``,
      which go to one fixed position each, whatever name is asked for;
    - ``_move_absolute`` / ``_move_relative``: ``move_manipulator_absolute`` /
      ``move_manipulator_relative``.

    As with the stage, positions come back as a copy, and a move keeps a copy of
    the position it was given, so neither side can change the other's object.
    """

    def __init__(
        self,
        parent: DemoMicroscope,
        resources: Optional[Resources] = None,
        start: Optional[DemoParts] = None,
    ):
        super().__init__(parent=parent, resources=resources)
        start = (parent if start is None else start).manipulator_system
        self.sim_position: FibsemManipulatorPosition = deepcopy(start.position)
        self.sim_inserted: bool = start.inserted

    def read_position(self) -> FibsemManipulatorPosition:
        return deepcopy(self.sim_position)

    def read_state(self) -> InsertableDeviceState:
        return _insertable_state(self.sim_inserted)

    def named_positions(self) -> List[str]:
        return ["PARK", "EUCENTRIC"]

    def saved_position(self, name: str = "PARK") -> FibsemManipulatorPosition:
        if name == "PARK":
            return deepcopy(_DEMO_PARK)
        if name == "EUCENTRIC":
            return deepcopy(_DEMO_ORIGIN)
        raise ValueError(f"Unknown manipulator position: {name}")

    def _insert(self, name: str) -> None:
        logging.info(f"Inserting manipulator to {name}...")
        self._move_absolute(_DEMO_PARK)
        self.sim_inserted = True
        logging.debug({"msg": "insert_manipulator", "name": name})

    def _retract(self) -> None:
        logging.info("Retracting manipulator...")
        self._move_absolute(_DEMO_ORIGIN)
        self.sim_inserted = False
        logging.debug({"msg": "retract_manipulator"})

    def _move_absolute(self, position: FibsemManipulatorPosition) -> None:
        logging.info(f"Moving manipulator: {position} (Absolute)")
        self.sim_position = deepcopy(position)
        logging.debug(
            {"msg": "move_manipulator_absolute", "position": position.to_dict()}
        )

    def _move_relative(self, delta: FibsemManipulatorPosition) -> None:
        logging.info(f"Moving manipulator: {delta} (Relative)")
        self.sim_position += delta
        logging.debug({"msg": "move_manipulator_relative", "position": delta.to_dict()})


def bind_demo_manipulator(
    microscope: DemoMicroscope,
    resources: Optional[Resources] = None,
    start: Optional[DemoParts] = None,
) -> DemoManipulator:
    """Build ``manipulator`` for a connected Demo microscope."""
    return DemoManipulator(microscope, resources, start).connect()


class DemoGasInjector(GasInjector):
    """The Demo gas injection system.

    It keeps its own simulated GIS in ``sim_gas``, ``sim_inserted``, ``sim_heated``
    and ``sim_opened``, copied when it is built from the starting ``gis_system``
    (``start``, else the microscope's own), and it never touches the microscope's
    again. Each hook is what the
    matching method of Demo's ``GasInjectionSystem`` does, on that copy. Demo's GIS
    takes no insert position or gas, so those arguments are only logged.
    """

    def __init__(
        self,
        parent: DemoMicroscope,
        resources: Optional[Resources] = None,
        start: Optional[DemoParts] = None,
    ):
        super().__init__(parent=parent, resources=resources)
        start = (parent if start is None else start).gis_system
        self.sim_gas: str = start.gas
        self.sim_inserted: bool = start.inserted
        self.sim_heated: bool = start.heated
        self.sim_opened: bool = start.opened

    def read_gas(self) -> str:
        return self.sim_gas

    def read_state(self) -> InsertableDeviceState:
        return _insertable_state(self.sim_inserted)

    def read_heated(self) -> bool:
        return self.sim_heated

    def read_opened(self) -> bool:
        return self.sim_opened

    def _insert(self, position: Optional[str]) -> None:
        self.sim_inserted = True
        logging.debug("GIS inserted")

    def _retract(self) -> None:
        self.sim_inserted = False
        logging.debug("GIS retracted")

    def _heater_on(self, gas: Optional[str]) -> None:
        self.sim_heated = True
        logging.debug("GIS heater on")
        sim_sleep(3)  # Demo's heater takes a moment, as its deposition waits for

    def _heater_off(self) -> None:
        self.sim_heated = False
        logging.debug("GIS heater off")

    def _open(self) -> None:
        self.sim_opened = True
        logging.debug("GIS opened")

    def _close(self) -> None:
        self.sim_opened = False
        logging.debug("GIS closed")


def bind_demo_gis(
    microscope: DemoMicroscope,
    resources: Optional[Resources] = None,
    start: Optional[DemoParts] = None,
) -> DemoGasInjector:
    """Build ``gis`` for a connected Demo microscope."""
    return DemoGasInjector(microscope, resources, start).connect()


# -- The FM -------------------------------------------------------------------------
#
# The simulated FM's parts as devices. Each keeps its own simulated part in sim_*
# fields, copied when built from the part the simulated FM (``fibsem.fm.microscope``)
# built, and does what that part does, on the copy. the Demo's ``fm`` is the FM API
# over them (``fibsem.fm.api``).


class DemoCamera(Camera):
    """The Demo FM camera: what ``SceneCamera`` does, on its own simulated camera.

    It images the sample scene when the microscope has one (``render_fm_scene``,
    reading the channel and focus through the microscope's ``fm``), and the stock
    noise frame with a numbered "FM<n>" otherwise.
    """

    def __init__(
        self,
        camera: FMClassCamera,
        parent: DemoMicroscope,
        resources: Optional[Resources] = None,
    ):
        super().__init__(name="camera", parent=parent, resources=resources)
        self.sim_exposure_time: float = camera._exposure_time
        self.sim_binning: int = camera._binning
        self.sim_gain: float = camera._gain
        self.sim_offset: float = camera._offset
        self.sim_sensor_pixel_size: Tuple[float, float] = camera._pixel_size
        self.sim_sensor_resolution: Tuple[int, int] = camera._resolution
        self.sim_index: int = camera._index
        self._frames: Dict[Tuple[int, Tuple[int, int]], np.ndarray] = {}

    def read_exposure_time(self) -> float:
        return self.sim_exposure_time

    def write_exposure_time(self, value: float) -> None:
        self.sim_exposure_time = value

    def metadata_exposure_time(self) -> ParameterMetadata:
        low, high = SIM_CAMERA_EXPOSURE_LIMITS
        return ParameterMetadata(limits=RangeLimit(min=low, max=high))

    def read_binning(self) -> int:
        return self.sim_binning

    def write_binning(self, value: int) -> None:
        if value not in BINNING_VALUES:
            raise ValueError(
                f"Binning must be one of {tuple(BINNING_VALUES)}, got {value}"
            )
        self.sim_binning = value

    def metadata_binning(self) -> ParameterMetadata:
        return ParameterMetadata(choices=list(BINNING_VALUES))

    def read_gain(self) -> float:
        return self.sim_gain

    def write_gain(self, value: float) -> None:
        if value < 0:
            raise ValueError("Gain must be non-negative.")
        self.sim_gain = value

    def read_offset(self) -> float:
        return self.sim_offset

    def write_offset(self, value: float) -> None:
        if value < 0:
            raise ValueError("Offset must be non-negative.")
        self.sim_offset = value

    def read_pixel_size(self) -> tuple:
        x, y = self.sim_sensor_pixel_size
        return (x * self.sim_binning, y * self.sim_binning)

    def read_resolution(self) -> tuple:
        width, height = self.sim_sensor_resolution
        return (width // self.sim_binning, height // self.sim_binning)

    def _acquire(self) -> np.ndarray:
        from fibsem.microscopes.simulator import render_fm_scene

        resolution = self.read_resolution()
        fm = getattr(self.parent, "fm", None)
        if fm is not None:
            frame = render_fm_scene(
                fm, self.sim_exposure_time, self.read_pixel_size()[0], resolution
            )
            if frame is not None:
                self.sim_index += 1
                return frame
        sim_sleep(self.sim_exposure_time)
        noise = np.random.randint(
            UINT16_MIN, UINT16_MAX, size=resolution[::-1], dtype=np.uint16
        )
        key = (self.sim_index % 10, resolution)
        if key not in self._frames:
            self._frames[key] = draw_text(
                f"FM{key[0]}",
                size=(resolution[0] // 4, resolution[1] // 4),
                thickness=min(64, resolution[0] // 16),
                image_shape=resolution[::-1],
            )
        self.sim_index += 1
        number = self._frames[key]
        return np.where(number > 0, number, noise)


class DemoLightSource(LightSource):
    """The Demo FM light source: one power, a fraction of the maximum, unchecked."""

    def __init__(
        self,
        light_source: FMClassLightSource,
        parent: DemoMicroscope,
        resources: Optional[Resources] = None,
    ):
        super().__init__(name="light_source", parent=parent, resources=resources)
        self.sim_power: float = light_source._power

    def read_power(self) -> float:
        return self.sim_power

    def write_power(self, value: float) -> None:
        self.sim_power = value

    def metadata_power(self) -> ParameterMetadata:
        return ParameterMetadata(limits=RangeLimit(min=0.0, max=1.0))


class DemoFilterSet(FilterSet):
    """The Demo FM filter set: Thermo's, simulated. Its excitation bands are
    ``EXCITATION_WAVELENGTHS``, and a wavelength between them selects the nearest, as
    on the hardware. Its emission filters are reflection and one multi-band filter
    whose bands aren't reported.

    Both are settings for the next exposure, as on every driver: the simulated FM has
    no wheel to read, and the camera renders whatever they say.
    """

    def __init__(
        self,
        filter_set: FMClassFilterSet,
        parent: DemoMicroscope,
        resources: Optional[Resources] = None,
    ):
        super().__init__(name="filter_set", parent=parent, resources=resources)
        self.sim_excitation_wavelength: float = filter_set._excitation_wavelength
        filters = self._emission_filters()
        self.sim_emission_filter: EmissionFilter = emission_filter_named(
            filter_set._emission_wavelength, filters
        )

    @staticmethod
    def _emission_filters() -> List[EmissionFilter]:
        return [emission_filter_for(value, {}) for value in EMISSION_WAVELENGTHS]

    def read_excitation_wavelength(self) -> float:
        return self.sim_excitation_wavelength

    def write_excitation_wavelength(self, value: float) -> None:
        self.sim_excitation_wavelength = value

    def metadata_excitation_wavelength(self) -> ParameterMetadata:
        return ParameterMetadata(choices=list(EXCITATION_WAVELENGTHS))

    def read_emission_filter(self) -> EmissionFilter:
        return self.sim_emission_filter

    def write_emission_filter(self, value: EmissionFilter) -> None:
        if value not in self._emission_filters():
            raise ValueError(f"filter_set has no emission filter {value}")
        self.sim_emission_filter = value

    def metadata_emission_filter(self) -> ParameterMetadata:
        return ParameterMetadata(choices=self._emission_filters())


class DemoObjective(Objective):
    """The Demo FM objective: the simulated objective's moves, on its own copy.

    A move past ``limit_position`` is clipped to it, and insert and retract take the
    simulated travel time, as the simulated objective's do.
    """

    def __init__(
        self,
        objective: FMClassObjectiveLens,
        parent: DemoMicroscope,
        resources: Optional[Resources] = None,
    ):
        super().__init__(name="objective", parent=parent, resources=resources)
        self.sim_position: float = objective._position
        self.sim_magnification: float = objective._magnification
        self.sim_numerical_aperture: float = objective._numerical_aperture
        self.sim_insert_position: float = objective._insert_position
        self.sim_retract_position: float = objective._retract_position
        self.sim_limit_position: float = objective._limit_position

    def read_position(self) -> float:
        sim_sleep(0.1)
        return self.sim_position

    def metadata_position(self) -> ParameterMetadata:
        low, high = SIM_OBJECTIVE_POSITION_LIMITS
        return ParameterMetadata(limits=RangeLimit(min=low, max=high))

    def read_state(self) -> InsertableDeviceState:
        return _insertable_state(self.read_position() >= self.sim_insert_position)

    def read_magnification(self) -> float:
        return self.sim_magnification

    def read_numerical_aperture(self) -> float:
        return self.sim_numerical_aperture

    def read_limit_position(self) -> float:
        return self.sim_limit_position

    def write_limit_position(self, value: float) -> None:
        self.sim_limit_position = value
        logging.info(
            f"Objective user-defined position limit set to: {value * 1e3:.3f} mm"
        )

    # A move changes position and state; read them back so both are signalled.
    def _moved(self) -> None:
        for name in ("position", "state"):
            self.parameters[name].get_value()

    def _move_relative(self, delta: float) -> None:
        self.sim_position += delta
        logging.info(
            f"Objective moved to new position: {self.sim_position * 1e3:.3f} mm "
            f"(delta: {delta * 1e3:.3f} mm)"
        )
        self._moved()

    def _move_absolute(self, position: float) -> None:
        if not position <= self.sim_limit_position:
            logging.warning(
                f"Clipping position {position} to user-defined limits "
                f"{self.sim_limit_position}"
            )
            position = float(np.clip(position, 0, self.sim_limit_position))
        sim_sleep(0.5)
        self.sim_position = position
        logging.info(
            f"Objective moved to absolute position: {self.sim_position * 1e3:.3f} mm"
        )
        self._moved()

    def _insert(self) -> None:
        sim_sleep(SIM_OBJECTIVE_TRAVEL_SECONDS)
        self._move_absolute(self.sim_insert_position)

    def _retract(self) -> None:
        sim_sleep(SIM_OBJECTIVE_TRAVEL_SECONDS)
        self._move_absolute(self.sim_retract_position)


class DemoFM(FM):
    """The Demo FM group: sets up a channel on its parts, then takes a frame."""

    def __init__(
        self,
        parts: Dict[str, Device],
        parent: DemoMicroscope,
        resources: Optional[Resources] = None,
    ):
        super().__init__(name="fm", parent=parent, resources=resources)
        self.parts = parts

    def _acquire_channel(self, channel: Optional[Dict[str, Any]]) -> np.ndarray:
        self._apply_channel(channel)
        return self.parts["camera"].acquire()

    def _start_live(self, channel: Optional[Dict[str, Any]]) -> None:
        # The simulated camera renders a frame when asked: nothing runs between.
        self._apply_channel(channel)

    def _apply_channel(self, channel: Optional[Dict[str, Any]]) -> None:
        if channel is not None:
            settings = ChannelSettings.from_dict(channel)
            filters = self.parts["filter_set"]
            filters.excitation_wavelength.write_through(settings.excitation_wavelength)
            filters.emission_filter.write_through(
                emission_filter_named(
                    settings.emission_wavelength, filters.emission_filter.choices
                )
            )
            self.parts["light_source"].power.write_through(settings.power)
            camera = self.parts["camera"]
            camera.exposure_time.write_through(settings.exposure_time)
            if settings.gain is not None:
                camera.gain.write_through(settings.gain)


def bind_demo_fm(
    microscope: DemoMicroscope,
    fm: FluorescenceMicroscope,
    resources: Optional[Resources] = None,
) -> Dict[str, Device]:
    """Build the FM's parts and group for a connected Demo microscope, each starting
    where the simulated FM ``fm``'s part is, by device name."""
    resources = resources if resources is not None else resources_of(microscope)
    parts: Dict[str, Device] = {
        "camera": DemoCamera(fm.camera, microscope, resources),
        "light_source": DemoLightSource(fm.light_source, microscope, resources),
        "filter_set": DemoFilterSet(fm.filter_set, microscope, resources),
        "objective": DemoObjective(fm.objective, microscope, resources),
    }
    group = DemoFM(parts, microscope, resources)
    return {device.name: device.connect() for device in [group, *parts.values()]}
