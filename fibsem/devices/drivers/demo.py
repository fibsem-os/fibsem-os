"""The Demo backend's parts as devices: what ``DemoMicroscope`` is built from.

Each device does what the matching part of the Demo did before devices: every
parameter and command what the matching branch of its ``_get`` and ``_set`` (or
method) did, and each FM device what the matching part of the simulated FM did
before devices.
Each keeps its own simulated part, copied when it is built from the starting parts
it is given (``start``, a ``DemoParts``), and never touches the microscope's again;
``DemoMicroscope`` routes the old keys to them.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import numpy as np

from fibsem._timing import sim_sleep
from fibsem.devices.beam import Beam
from fibsem.devices.chamber import Chamber
from fibsem.devices.core import Device, ParameterMetadata, Resources, resources_of
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
from fibsem.devices.stage import Stage, axis_limits_from_degrees, compustage_poses
from fibsem.fm.microscope import emission_filter_named
from fibsem.fm.structures import (
    REFLECTION,
    ChannelSettings,
    EmissionFilter,
    emission_filter_for,
)
from fibsem.microscopes.simulator import (
    BINNING_VALUES,
    EMISSION_WAVELENGTHS,
    EXCITATION_WAVELENGTHS,
    FM_ACTIVE_DEVICE,
    FM_ACTIVE_VIEW,
    SIM_CAMERA_BINNING,
    SIM_CAMERA_EXPOSURE_LIMITS,
    SIM_CAMERA_EXPOSURE_TIME,
    SIM_CAMERA_GAIN,
    SIM_CAMERA_OFFSET,
    SIM_CAMERA_PIXEL_SIZE,
    SIM_CAMERA_RESOLUTION,
    SIM_LIGHT_SOURCE_POWER,
    SIM_OBJECTIVE_INSERT_POSITION,
    SIM_OBJECTIVE_MAGNIFICATION,
    SIM_OBJECTIVE_NA,
    SIM_OBJECTIVE_POSITION_LIMITS,
    SIM_OBJECTIVE_RETRACT_POSITION,
    SIM_OBJECTIVE_TRAVEL_SECONDS,
    SIM_OBJECTIVE_USER_POSITION_LIMIT,
    UINT16_MAX,
    UINT16_MIN,
)
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
    from fibsem.microscopes.device_demo import DemoMicroscope
    from fibsem.microscopes.registry import BuildContext
    from fibsem.microscopes.simulator import DemoParts
    from fibsem.structures import DeviceEntry


class DemoBeam(Beam):
    """A Demo beam.

    It keeps its own simulated column: ``sim_beam`` and ``sim_detector`` (the beam's
    and detector's settings), ``sim_on``, ``sim_blanked``, ``sim_scanning_mode`` and
    ``sim_scanning_mode_value``, copied when it is built from the starting
    ``electron_system`` or ``ion_system`` (``start``), and it never touches the microscope's again. The plasma gas is configuration
    (``system.ion``), not beam state, so it is still read and written there.
    """

    def __init__(
        self,
        beam_type: BeamType,
        parent: DemoMicroscope,
        resources: Optional[Resources],
        start: DemoParts,
    ):
        super().__init__(beam_type, parent=parent, resources=resources)
        start = (
            start.electron_system
            if beam_type is BeamType.ELECTRON
            else start.ion_system
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


class DemoStage(Stage):
    """The Demo stage.

    It keeps its own simulated stage in ``sim_position``, ``sim_homed`` and
    ``sim_linked``, copied when it is built from the starting ``stage_system``
    (``start``), and it never touches the microscope's again. A microscope that
    builds it routes the stage keys to it (``DemoMicroscope``). Each method does what
    the matching part of the Demo did before devices, on that copy:

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
        resources: Optional[Resources],
        start: DemoParts,
    ):
        super().__init__(parent=parent, resources=resources)
        start = start.stage_system
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


def _insertable_state(inserted: bool) -> InsertableDeviceState:
    return (
        InsertableDeviceState.INSERTED if inserted else InsertableDeviceState.RETRACTED
    )


class DemoChamber(Chamber):
    """The Demo chamber.

    It keeps its own simulated chamber in ``sim_state`` and ``sim_pressure``, copied
    when it is built from the starting ``chamber`` (``start``), and it never touches
    the microscope's again. A microscope that builds it routes the chamber keys to it
    (``DemoMicroscope``). Each method does what the matching part of the Demo did
    before devices, on that copy:

    - ``read_state`` / ``read_pressure``: the ``chamber_state`` / ``chamber_pressure``
      branches of ``_get``;
    - ``_pump`` / ``_vent``: the ``pump_chamber`` / ``vent_chamber`` branches of
      ``_set``.
    """

    def __init__(
        self,
        parent: DemoMicroscope,
        resources: Optional[Resources],
        start: DemoParts,
    ):
        super().__init__(parent=parent, resources=resources)
        chamber = start.chamber
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


# Demo's two saved positions. insert_manipulator goes to PARK and
# retract_manipulator to the origin, which is also EUCENTRIC.
_DEMO_PARK = FibsemManipulatorPosition(x=0, y=0, z=180e-6, r=0, t=0)
_DEMO_ORIGIN = FibsemManipulatorPosition(x=0, y=0, z=0, r=0, t=0)


class DemoManipulator(Manipulator):
    """The Demo manipulator.

    It keeps its own simulated needle in ``sim_position`` and ``sim_inserted``,
    copied when it is built from the starting ``manipulator_system`` (``start``), and
    it never touches the microscope's again. A microscope that builds it routes the
    manipulator keys to it (``DemoMicroscope``). Each method does what the matching
    part of the Demo did before devices, on that copy:

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
        resources: Optional[Resources],
        start: DemoParts,
    ):
        super().__init__(parent=parent, resources=resources)
        start = start.manipulator_system
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

    def _stop(self) -> None:
        # Its moves finish before they return, so there is never one to stop.
        logging.info("Stopping manipulator.")


# -- The sample loader --------------------------------------------------------------


class DemoSampleLoader(SampleLoader):
    """An in-memory autoloader for a simulated compustage system.

    ``occupied`` lists the 1-based magazine slots that hold a grid, as printed on a
    real magazine; ``names`` maps a slot number to the description its grid carries
    (an occupied slot without one reads blank, as on the hardware, and the grid model
    calls it ``Grid-NN``). The slot of the grid on the stage reads ``LOADED``, as
    AutoScript 4.14 reports it.

    ``fail_next_exchange`` makes the next load or unload raise ``GridExchangeError``
    and change nothing, so a run's load-failure path can be exercised.
    ``exchange_delay`` is how long each load and each unload pretends to take, and
    ``scan_delay`` a scan, through ``sim_sleep`` (a no-op under
    ``FIBSEM_SIM_NO_DELAY=1``, as the tests run). ``start_unscanned`` starts the
    magazine as a real one reads after it has been undocked: every slot ``UNKNOWN``
    until a scan; a read does not change that.

    ``sim_grid_position`` is where this autoloader really puts a grid, (x, y, z) in
    metres from the stage origin, as a real Arctis does (FIB-1144); the simulated
    scene draws a loaded grid there. None puts it at the origin.
    """

    def __init__(
        self,
        parent: Any = None,
        resources: Optional[Resources] = None,
        capacity: int = 12,
        occupied: Iterable[int] = (),
        names: Optional[Mapping[Union[int, str], str]] = None,
        exchange_delay: float = 0.0,
        start_unscanned: bool = False,
        scan_delay: float = 0.0,
        grid_position: Optional[Tuple[float, float, float]] = None,
    ):
        super().__init__(parent=parent, resources=resources)
        self.sim_capacity = int(capacity)
        self.exchange_delay = float(exchange_delay)
        self.scan_delay = float(scan_delay)
        self.sim_grid_position = (
            tuple(float(v) for v in grid_position) if grid_position else None
        )
        self.fail_next_exchange = False
        names = names or {}
        # slot number -> description, for each slot holding a grid
        self.sim_grids: Dict[int, str] = {}
        for number in occupied:
            number = int(number)
            if not 1 <= number <= self.sim_capacity:
                raise ValueError(
                    f"Magazine slot {number} is outside capacity {self.sim_capacity}."
                )
            self.sim_grids[number] = str(
                names.get(number, names.get(str(number))) or ""
            )
        self.sim_on_stage: Optional[int] = None  # the slot whose grid is on the stage
        self.sim_scanned = not start_unscanned

    def read_magazine(self) -> Magazine:
        slots = []
        for number in range(1, self.sim_capacity + 1):
            if not self.sim_scanned:
                slots.append(MagazineSlot(number, MagazineSlotState.UNKNOWN))
            elif number == self.sim_on_stage:
                slots.append(
                    MagazineSlot(
                        number, MagazineSlotState.LOADED, self.sim_grids[number]
                    )
                )
            elif number in self.sim_grids:
                slots.append(
                    MagazineSlot(
                        number, MagazineSlotState.OCCUPIED, self.sim_grids[number]
                    )
                )
            else:
                slots.append(MagazineSlot(number, MagazineSlotState.EMPTY))
        return Magazine(tuple(slots))

    def read_on_stage(self) -> StageSample:
        if self.sim_on_stage is None:
            return StageSample(present=False)
        return StageSample(present=True, description=self.sim_grids[self.sim_on_stage])

    def read_capacity(self) -> int:
        return self.sim_capacity

    def read_exchange_time(self) -> float:
        """An unload and a load, each ``exchange_delay``."""
        return 2 * self.exchange_delay

    def _load(self, slot: int) -> None:
        if slot not in self.sim_grids:
            raise GridExchangeError(f"Magazine slot {slot} holds no grid.")
        if self.sim_on_stage is not None:
            raise GridExchangeError("A grid is already on the stage; unload it first.")
        self._exchange()
        self.sim_on_stage = slot

    def _unload(self) -> None:
        if self.sim_on_stage is None:
            return
        self._exchange()
        self.sim_on_stage = None

    def _scan(self) -> None:
        sim_sleep(self.scan_delay)
        self.sim_scanned = True

    def _set_description(self, slot: int, text: str) -> None:
        if not 1 <= slot <= self.sim_capacity:
            raise GridExchangeError(f"The sample loader has no slot {slot}.")
        if slot in self.sim_grids:
            self.sim_grids[slot] = text

    def _exchange(self) -> None:
        if self.fail_next_exchange:
            self.fail_next_exchange = False
            raise GridExchangeError("Simulated autoloader exchange failure.")
        sim_sleep(self.exchange_delay)


# The keys a Demo sample loader entry takes, and their defaults. A simulator
# configuration from before the entry had them under `sim.loader`, which is still
# read for any key the entry leaves out.
DEMO_SAMPLE_LOADER_KEYS: Dict[str, Any] = {
    "capacity": 12,
    "occupied": (),
    "names": None,
    "exchange_delay": 0.0,
    "start_unscanned": False,
    "scan_delay": 0.0,
    "grid_position": None,
}


# -- Builders by entry --------------------------------------------------------------
#
# The Demo driver's device builders (``DRIVER.devices`` in ``device_demo``): each
# builds one device from its ``hardware.devices`` entry. The devices of one connect
# share their starting parts and resources through the build context, and each device is named after its entry.


def _demo_start(context: "BuildContext") -> Tuple[Resources, "DemoParts"]:
    from fibsem.microscopes.simulator import initial_demo_parts

    if "Demo" not in context.shared:
        microscope = context.microscope
        context.shared["Demo"] = (
            resources_of(microscope),
            initial_demo_parts(microscope.system),
        )
    return context.shared["Demo"]


def _named(device: Device, entry: "DeviceEntry") -> Device:
    device.name = entry.name
    return device.connect()


def build_demo_beam(entry: "DeviceEntry", context: "BuildContext") -> DemoBeam:
    """The beam an ``electron`` or ``ion`` entry names."""
    if entry.name not in ("electron", "ion"):
        raise ValueError("a Demo beam is named 'electron' or 'ion'")
    beam_type = BeamType.ELECTRON if entry.name == "electron" else BeamType.ION
    resources, start = _demo_start(context)
    return DemoBeam(beam_type, context.microscope, resources, start).connect()


def build_demo_stage(entry: "DeviceEntry", context: "BuildContext") -> DemoStage:
    return _named(DemoStage(context.microscope, *_demo_start(context)), entry)


def build_demo_chamber(entry: "DeviceEntry", context: "BuildContext") -> DemoChamber:
    return _named(DemoChamber(context.microscope, *_demo_start(context)), entry)


def build_demo_manipulator(
    entry: "DeviceEntry", context: "BuildContext"
) -> DemoManipulator:
    return _named(DemoManipulator(context.microscope, *_demo_start(context)), entry)


def build_demo_sample_loader(
    entry: "DeviceEntry", context: "BuildContext"
) -> DemoSampleLoader:
    """The simulated autoloader, from its entry's keys (``DEMO_SAMPLE_LOADER_KEYS``),
    each falling back to the old ``sim.loader`` block, then the default."""
    microscope = context.microscope
    system = getattr(microscope, "system", None)
    legacy = (getattr(system, "sim", None) or {}).get("loader") or {}
    keys = {
        key: entry.options.get(key, legacy.get(key, default))
        for key, default in DEMO_SAMPLE_LOADER_KEYS.items()
    }
    resources, _ = _demo_start(context)
    return _named(
        DemoSampleLoader(
            microscope,
            resources,
            capacity=int(keys["capacity"]),
            occupied=keys["occupied"] or (),
            names=keys["names"] or {},
            exchange_delay=float(keys["exchange_delay"]),
            start_unscanned=bool(keys["start_unscanned"]),
            scan_delay=float(keys["scan_delay"]),
            grid_position=keys["grid_position"] or None,
        ),
        entry,
    )


# -- The FM -------------------------------------------------------------------------
#
# The Demo FM's parts as devices. Each keeps its own simulated part in sim_*
# fields, starting from the ``SIM_*`` values. The Demo's ``fm`` is the FM API over them
# (``fibsem.fm.microscope``), and the group holds the FM's share of the Demo's imaging
# channel (``DemoFMChannel``).


class DemoCamera(Camera):
    """The Demo FM camera, on its own simulated camera.

    It images the sample scene when the microscope has one (``render_fm_scene``,
    reading the channel and focus through the microscope's ``fm``), and the stock
    noise frame with a numbered "FM<n>" otherwise.
    """

    def __init__(
        self,
        parent: Optional[DemoMicroscope] = None,
        resources: Optional[Resources] = None,
    ):
        super().__init__(name="camera", parent=parent, resources=resources)
        self.sim_exposure_time: float = SIM_CAMERA_EXPOSURE_TIME
        self.sim_binning: int = SIM_CAMERA_BINNING
        self.sim_gain: float = SIM_CAMERA_GAIN
        self.sim_offset: float = SIM_CAMERA_OFFSET
        self.sim_sensor_pixel_size: Tuple[float, float] = SIM_CAMERA_PIXEL_SIZE
        self.sim_sensor_resolution: Tuple[int, int] = SIM_CAMERA_RESOLUTION
        self.sim_index: int = 0
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
        parent: Optional[DemoMicroscope] = None,
        resources: Optional[Resources] = None,
    ):
        super().__init__(name="light_source", parent=parent, resources=resources)
        self.sim_power: float = SIM_LIGHT_SOURCE_POWER

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

    Both are settings for the next exposure, as on every driver: the Demo FM has no
    wheel to read, and the camera renders whatever they say.
    """

    def __init__(
        self,
        parent: Optional[DemoMicroscope] = None,
        resources: Optional[Resources] = None,
    ):
        super().__init__(name="filter_set", parent=parent, resources=resources)
        self.sim_excitation_wavelength: float = EXCITATION_WAVELENGTHS[0]
        self.sim_emission_filter: EmissionFilter = REFLECTION

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
        parent: Optional[DemoMicroscope] = None,
        resources: Optional[Resources] = None,
    ):
        super().__init__(name="objective", parent=parent, resources=resources)
        self.sim_position: float = SIM_OBJECTIVE_RETRACT_POSITION
        self.sim_magnification: float = SIM_OBJECTIVE_MAGNIFICATION
        self.sim_numerical_aperture: float = SIM_OBJECTIVE_NA
        self.sim_insert_position: float = SIM_OBJECTIVE_INSERT_POSITION
        self.sim_retract_position: float = SIM_OBJECTIVE_RETRACT_POSITION
        self.sim_limit_position: float = SIM_OBJECTIVE_USER_POSITION_LIMIT

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


class DemoFMChannel:
    """The FM's share of the Demo's one imaging channel, which the FM and the beams
    share as on a TFS system (FIB-518): ``AutoscriptFMChannel``, simulated.

    The Demo has one active view and one active device (``imaging_system``), and
    whoever sets them last owns the microscope. This takes them the way the Thermo
    FM's channel takes the AutoScript connection's: the same depth count, the same
    lock (the microscope's ``_threading_lock``), the same restore and the same fast
    path, so a test written against the Demo says something about the hardware.
    Without a microscope (an FM served on its own) there is no channel to share.
    """

    def __init__(self, microscope: Optional[DemoMicroscope] = None):
        # A microscope with no imaging system (not a demo) has no channel to share.
        if getattr(microscope, "imaging_system", None) is None:
            microscope = None
        self._microscope = microscope
        # The microscope's lock, as the Thermo channel takes it: an FM scope and a beam
        # acquisition then queue against each other rather than interleaving.
        self.lock = getattr(microscope, "_threading_lock", None) or threading.RLock()
        self._depth = 0
        self._restore_view: Optional[int] = None
        self._restore_device: Optional[int] = None

    def set_active_channel(self) -> None:
        """Point the channel at the FM and leave it there, as
        ``AutoscriptFMChannel.set_active_channel`` does."""
        if self._microscope is None:
            return
        imaging = self._microscope.imaging_system
        imaging.active_view = FM_ACTIVE_VIEW
        imaging.active_device = FM_ACTIVE_DEVICE

    def _is_ours(self) -> bool:
        """Whether the channel is already on the FM: the view alone answers it, as on
        the driver, where the device follows the view. Read outside the lock on
        purpose (``AutoscriptFMChannel._is_ours``)."""
        return self._microscope.imaging_system.active_view == FM_ACTIVE_VIEW

    @contextmanager
    def scope(self) -> Iterator[None]:
        """Hold the channel on the FM for the block, then put it back.

        ``AutoscriptFMChannel.scope`` says why: depth counted, so a tileset holding it
        for a whole run is not undone by each tile; the lock covers the bookkeeping and
        never the body; nothing taken when the channel is already the FM, which is what
        keeps the objective usable while streaming.

        Puts the device back with the view, where the driver puts back the view alone.
        On hardware the device belongs to the view and comes back with it; here they
        are two fields, and restoring only the view would leave ``active_device`` on
        the FM for the rest of the session.
        """
        if self._microscope is None:
            yield
            return

        if self._depth == 0 and self._is_ours():
            yield
            return

        imaging = self._microscope.imaging_system
        with self.lock:
            if self._depth == 0:
                self._restore_view = imaging.active_view
                self._restore_device = imaging.active_device
            self.set_active_channel()
            self._depth += 1
        try:
            yield
        finally:
            with self.lock:
                self._depth -= 1
                if self._depth == 0:
                    imaging.active_view = self._restore_view
                    imaging.active_device = self._restore_device


class DemoFM(FM):
    """The Demo FM group: sets up a channel on its parts, then takes a frame. It holds
    the FM's share of the imaging channel (``channel``), for the FM API to take."""

    def __init__(
        self,
        parent: Optional[DemoMicroscope] = None,
        resources: Optional[Resources] = None,
    ):
        super().__init__(name="fm", parent=parent, resources=resources)
        self.channel = DemoFMChannel(parent)

    def _acquire_channel(self, channel: Optional[Dict[str, Any]]) -> np.ndarray:
        self._apply_channel(channel)
        return self.camera.acquire()

    def _start_live(self, channel: Optional[Dict[str, Any]]) -> None:
        # The simulated camera renders a frame when asked: nothing runs between.
        self._apply_channel(channel)

    def _apply_channel(self, channel: Optional[Dict[str, Any]]) -> None:
        if channel is not None:
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


def bind_demo_fm(
    microscope: Optional[DemoMicroscope] = None,
    resources: Optional[Resources] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Device]:
    """Build the Demo FM's parts and group, by device name, for a Demo microscope,
    or on their own without one (an FM served by itself, imaging no scene). *config*
    is the fm entry's own keys (``mount_transform``)."""
    resources = resources if resources is not None else resources_of(microscope)
    parts: Dict[str, Device] = {
        "camera": DemoCamera(microscope, resources),
        "light_source": DemoLightSource(microscope, resources),
        "filter_set": DemoFilterSet(microscope, resources),
        "objective": DemoObjective(microscope, resources),
    }
    parts["camera"].configure(config)
    group = DemoFM(microscope, resources).fill_roles(**parts)
    return {device.name: device.connect() for device in [group, *parts.values()]}
