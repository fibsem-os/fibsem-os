"""The Demo backend's beams and stage as devices, beside the untouched Demo microscope.

``DemoBeam`` implements each parameter with what the matching branch of
``DemoMicroscope._get`` and ``_set`` reads and writes, so the old call and the new
parameter touch the same state. This is step 1 of moving a key ("declare and
implement"); the Demo chain itself is unchanged.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from fibsem._timing import sim_sleep
from fibsem.devices.beam import Beam
from fibsem.devices.core import ParameterMetadata, Resources
from fibsem.devices.stage import Stage
from fibsem.structures import (
    BeamType,
    FibsemRectangle,
    FibsemStagePosition,
    Point,
    RangeLimit,
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

    def read_scanning_mode(self) -> str:
        return self._system.scanning_mode

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


# ``_get_axis_limits`` gives these in degrees.
_DEGREE_AXES = ("r", "t")


class DemoStage(Stage):
    """The Demo stage.

    Each method is what the matching part of ``DemoMicroscope`` does today, reading
    and writing the same ``stage_system``, so the old call and the device touch the
    same state:

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
        self._system = parent.stage_system
        # Read once at connect, as a vendor's limits would be.
        self._axis_limits = parent._get_axis_limits()

    # -- position ----------------------------------------------------------------

    def read_position(self) -> FibsemStagePosition:
        sim_sleep(0.1)  # the read delay the Demo branch has
        return deepcopy(self._system.position)

    def metadata_position(self) -> ParameterMetadata:
        # The axes are the ones the simulator gives limits for: a compustage has no r.
        limits = {}
        for axis, limit in self._axis_limits.items():
            low, high = limit.min, limit.max
            if axis in _DEGREE_AXES:
                low, high = float(np.radians(low)), float(np.radians(high))
            limits[axis] = RangeLimit(min=low, max=high)
        return ParameterMetadata(limits=limits)

    # -- homing and linking -----------------------------------------------------------

    def read_homed(self) -> bool:
        return self._system.is_homed

    # A compustage can't link: the old set("stage_link") logs and does nothing there,
    # so on the new API "linked" is absent and link() is unavailable.
    def available_linked(self) -> bool:
        return not self.parent.stage_is_compustage

    def read_linked(self) -> bool:
        return self._system.is_linked

    # -- commands ---------------------------------------------------------------------

    def _move_absolute(self, position: FibsemStagePosition) -> None:
        from fibsem.microscopes.simulator import STAGE_MOVEMENT_SLEEP_TIME

        sim_sleep(STAGE_MOVEMENT_SLEEP_TIME)
        for axis in ("x", "y", "z", "r", "t"):
            value = getattr(position, axis)
            if value is not None:
                setattr(self._system.position, axis, value)
        logging.debug({"msg": "move_stage_absolute", "position": position.to_dict()})

    def _move_relative(self, delta: FibsemStagePosition) -> None:
        from fibsem.microscopes.simulator import STAGE_MOVEMENT_SLEEP_TIME

        sim_sleep(STAGE_MOVEMENT_SLEEP_TIME)
        self._system.position += delta
        logging.debug({"msg": "move_stage_relative", "position": delta.to_dict()})

    def _home(self) -> None:
        logging.info("Homing stage...")
        self._system.is_homed = True
        logging.info("Stage homed.")

    def _link(self) -> None:
        logging.info("Linking stage...")
        self._system.is_linked = True
        logging.info("Stage linked.")


def bind_demo_stage(
    microscope: DemoMicroscope, resources: Optional[Resources] = None
) -> DemoStage:
    """Build ``stage`` for a connected Demo microscope."""
    return DemoStage(microscope, resources).connect()
