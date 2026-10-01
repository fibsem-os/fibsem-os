"""The Demo backend's beams and stage as devices, beside the untouched Demo microscope.

``DemoBeam`` implements each parameter with what the matching branch of
``DemoMicroscope._get`` and ``_set`` reads and writes, so the old call and the new
parameter touch the same state. This is step 1 of moving a key ("declare and
implement"); the Demo chain itself is unchanged. ``DemoStage``, ``DemoChamber``
and ``DemoManipulator`` have gone further: each keeps its own copy of its part, and
the microscope routes that part's keys to it.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from fibsem._timing import sim_sleep
from fibsem.devices.beam import Beam
from fibsem.devices.chamber import Chamber
from fibsem.devices.core import ParameterMetadata, Resources
from fibsem.devices.gis import GasInjector
from fibsem.devices.manipulator import Manipulator
from fibsem.devices.stage import Stage, axis_limits_from_degrees
from fibsem.structures import (
    BeamType,
    ChamberState,
    FibsemManipulatorPosition,
    FibsemRectangle,
    FibsemStagePosition,
    InsertableDeviceState,
    Point,
    RangeLimit,
    ScanMode,
)

if TYPE_CHECKING:
    from fibsem.microscopes.simulator import DemoMicroscope


class DemoBeam(Beam):
    def __init__(
        self,
        beam_type: BeamType,
        parent: DemoMicroscope,
        resources: Optional[Resources] = None,
    ):
        super().__init__(beam_type, parent=parent, resources=resources)
        self._system = (
            parent.electron_system
            if beam_type is BeamType.ELECTRON
            else parent.ion_system
        )

    def _choices(self, key: str) -> ParameterMetadata:
        return ParameterMetadata(
            choices=self.parent.get_available_values(key, self.beam_type)
        )

    # Each parameter is the matching Demo branch as it stands.

    def read_voltage(self) -> float:
        return self._system.beam.voltage

    def write_voltage(self, value: float) -> None:
        self._system.beam.voltage = value

    def metadata_voltage(self) -> ParameterMetadata:
        return self._choices("voltage")

    def read_current(self) -> float:
        return self._system.beam.beam_current

    def write_current(self, value: float) -> None:
        self._system.beam.beam_current = value

    def metadata_current(self) -> ParameterMetadata:
        return self._choices("current")

    def read_working_distance(self) -> float:
        return self._system.beam.working_distance

    def write_working_distance(self, value: float) -> None:
        self._system.beam.working_distance = value

    def read_hfw(self) -> float:
        return self._system.beam.hfw

    def write_hfw(self, value: float) -> None:
        self._system.beam.hfw = value

    def read_scan_rotation(self) -> float:
        return float(self._system.beam.scan_rotation)

    def write_scan_rotation(self, value: float) -> None:
        self._system.beam.scan_rotation = float(value)

    def read_blanked(self) -> bool:
        return self._system.blanked

    def write_blanked(self, value: bool) -> None:
        self._system.blanked = value
        if not value and self._system.scanning_mode == "spot":
            self.parent._burn_into_sample_scene(self.beam_type)  # the spot burn

    def read_detector_type(self) -> str:
        return self._system.detector.type

    def write_detector_type(self, value: str) -> None:
        self._system.detector.type = value

    def metadata_detector_type(self) -> ParameterMetadata:
        return self._choices("detector_type")

    def read_detector_mode(self) -> str:
        return self._system.detector.mode

    def write_detector_mode(self, value: str) -> None:
        self._system.detector.mode = value

    def metadata_detector_mode(self) -> ParameterMetadata:
        return self._choices("detector_mode")

    def read_detector_contrast(self) -> float:
        return self._system.detector.contrast

    def write_detector_contrast(self, value: float) -> None:
        self._system.detector.contrast = value

    def read_detector_brightness(self) -> float:
        return self._system.detector.brightness

    def write_detector_brightness(self, value: float) -> None:
        self._system.detector.brightness = value

    def read_resolution(self) -> tuple:
        return self._system.beam.resolution

    def write_resolution(self, value: tuple) -> None:
        self._system.beam.resolution = value

    def read_dwell_time(self) -> float:
        return self._system.beam.dwell_time

    def write_dwell_time(self, value: float) -> None:
        self._system.beam.dwell_time = value

    # Reads hand back a new Point, as the Demo branch does, so a caller can't change
    # the simulator's state through the value it was given.
    def read_stigmation(self) -> Point:
        return Point(self._system.beam.stigmation.x, self._system.beam.stigmation.y)

    def write_stigmation(self, value: Point) -> None:
        self._system.beam.stigmation = value

    def read_shift(self) -> Point:
        return Point(self._system.beam.shift.x, self._system.beam.shift.y)

    def write_shift(self, value: Point) -> None:
        self._system.beam.shift = value

    def read_on(self) -> bool:
        return self._system.on

    def write_on(self, value: bool) -> None:
        self._system.on = value

    def read_scanning_mode(self) -> ScanMode:
        return ScanMode(self._system.scanning_mode)

    # The scan commands: the spot_mode, reduced_area and full_frame branches of _set.

    def _spot(self, point: Point) -> None:
        self._system.scanning_mode = "spot"
        self._system.scanning_mode_value = point

    def _reduced_area(self, area: FibsemRectangle) -> None:
        self._system.scanning_mode = "reduced_area"
        self._system.scanning_mode_value = area

    def _full_frame(self) -> None:
        self._system.scanning_mode = "full_frame"
        self._system.scanning_mode_value = None

    # Only a plasma ion column has a gas.
    def available_plasma_gas(self) -> bool:
        return self.beam_type is BeamType.ION and self.parent.system.ion.plasma

    def read_plasma_gas(self) -> str:
        return self.parent.system.ion.plasma_gas

    def write_plasma_gas(self, value: str) -> None:
        # An unavailable gas logs and is ignored, as the Demo branch does.
        microscope = self.parent
        if not microscope.check_available_values("plasma_gas", value, BeamType.ION):
            logging.warning(
                f"Plasma gas {value} not available. Available values: "
                f"{microscope.get_available_values('plasma_gas', BeamType.ION)}"
            )
            return
        logging.info(f"Setting plasma gas to {value}... this may take some time...")
        microscope.system.ion.plasma_gas = value
        logging.info(f"Plasma gas set to {value}.")

    def metadata_plasma_gas(self) -> ParameterMetadata:
        return self._choices("plasma_gas")

    # "preset" is not implemented: Demo has no presets, so it is absent on the new
    # API while the old set("preset", ...) keeps its no-op through the Demo chain.


def bind_demo_beams(
    microscope: DemoMicroscope, resources: Optional[Resources] = None
) -> Dict[BeamType, Beam]:
    """Build ``beams[BeamType]`` for a connected Demo microscope."""
    resources = resources if resources is not None else Resources()
    return {
        beam_type: DemoBeam(beam_type, microscope, resources).connect()
        for beam_type in (BeamType.ELECTRON, BeamType.ION)
    }


class DemoStage(Stage):
    """The Demo stage.

    It keeps its own simulated stage in ``sim_position``, ``sim_homed`` and
    ``sim_linked``, copied from the microscope's ``stage_system`` at connect, so it
    starts where Demo's stage is and never touches Demo's again. A microscope that builds it routes the stage keys to it
    (``DeviceDemoMicroscope``). Each method is what the matching part of
    ``DemoMicroscope`` does, on that copy:

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

    def __init__(self, parent: DemoMicroscope, resources: Optional[Resources] = None):
        super().__init__(parent=parent, resources=resources)
        start = parent.stage_system
        self.sim_position: FibsemStagePosition = deepcopy(start.position)
        self.sim_homed: bool = start.is_homed
        self.sim_linked: bool = start.is_linked
        # Read once at connect, as a vendor's limits would be.
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
    microscope: DemoMicroscope, resources: Optional[Resources] = None
) -> DemoStage:
    """Build ``stage`` for a connected Demo microscope."""
    return DemoStage(microscope, resources).connect()


def _insertable_state(inserted: bool) -> InsertableDeviceState:
    return (
        InsertableDeviceState.INSERTED if inserted else InsertableDeviceState.RETRACTED
    )


class DemoChamber(Chamber):
    """The Demo chamber.

    It keeps its own simulated chamber in ``sim_state`` and ``sim_pressure``, copied
    from the microscope's ``chamber`` at connect, so it starts where Demo's chamber
    is and never touches Demo's again. A microscope that builds it routes the chamber keys to it
    (``DeviceDemoMicroscope``). Each method is what the matching part of
    ``DemoMicroscope`` does, on that copy:

    - ``read_state`` / ``read_pressure``: the ``chamber_state`` / ``chamber_pressure``
      branches of ``_get``;
    - ``_pump`` / ``_vent``: the ``pump_chamber`` / ``vent_chamber`` branches of
      ``_set``.
    """

    def __init__(self, parent: DemoMicroscope, resources: Optional[Resources] = None):
        super().__init__(parent=parent, resources=resources)
        self.sim_state = ChamberState.from_name(parent.chamber.state)
        self.sim_pressure: float = parent.chamber.pressure

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
    microscope: DemoMicroscope, resources: Optional[Resources] = None
) -> DemoChamber:
    """Build ``chamber`` for a connected Demo microscope."""
    return DemoChamber(microscope, resources).connect()


# Demo's two saved positions. insert_manipulator goes to PARK and
# retract_manipulator to the origin, which is also EUCENTRIC.
_DEMO_PARK = FibsemManipulatorPosition(x=0, y=0, z=180e-6, r=0, t=0)
_DEMO_ORIGIN = FibsemManipulatorPosition(x=0, y=0, z=0, r=0, t=0)


class DemoManipulator(Manipulator):
    """The Demo manipulator.

    It keeps its own simulated needle in ``sim_position`` and ``sim_inserted``,
    copied from the microscope's ``manipulator_system`` at connect, so it starts
    where Demo's is and never touches Demo's again. A microscope that builds it
    routes the manipulator keys to it (``DeviceDemoMicroscope``). Each method is what
    the matching part of ``DemoMicroscope`` does, on that copy:

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

    def __init__(self, parent: DemoMicroscope, resources: Optional[Resources] = None):
        super().__init__(parent=parent, resources=resources)
        start = parent.manipulator_system
        self.sim_position: FibsemManipulatorPosition = deepcopy(start.position)
        self.sim_inserted: bool = start.inserted

    def read_position(self) -> FibsemManipulatorPosition:
        return deepcopy(self.sim_position)

    def read_state(self) -> InsertableDeviceState:
        return _insertable_state(self.sim_inserted)

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
    microscope: DemoMicroscope, resources: Optional[Resources] = None
) -> DemoManipulator:
    """Build ``manipulator`` for a connected Demo microscope."""
    return DemoManipulator(microscope, resources).connect()


class DemoGasInjector(GasInjector):
    """The Demo gas injection system.

    Each read and hook is the matching field or method of Demo's ``gis_system``,
    the ``GasInjectionSystem`` that ``cryo_deposition_v2`` drives. Demo's GIS
    takes no insert position or gas, so those arguments are only logged.
    """

    def __init__(self, parent: DemoMicroscope, resources: Optional[Resources] = None):
        super().__init__(parent=parent, resources=resources)
        self._system = parent.gis_system

    def read_gas(self) -> str:
        return self._system.gas

    def read_state(self) -> InsertableDeviceState:
        return _insertable_state(self._system.inserted)

    def read_heated(self) -> bool:
        return self._system.heated

    def read_opened(self) -> bool:
        return self._system.opened

    def _insert(self, position: Optional[str]) -> None:
        self._system.insert()

    def _retract(self) -> None:
        self._system.retract()

    def _heater_on(self, gas: Optional[str]) -> None:
        self._system.turn_heater_on()
        sim_sleep(3)  # Demo's heater takes a moment, as its deposition waits for

    def _heater_off(self) -> None:
        self._system.turn_heater_off()

    def _open(self) -> None:
        self._system.open()

    def _close(self) -> None:
        self._system.close()


def bind_demo_gis(
    microscope: DemoMicroscope, resources: Optional[Resources] = None
) -> DemoGasInjector:
    """Build ``gis`` for a connected Demo microscope."""
    return DemoGasInjector(microscope, resources).connect()
