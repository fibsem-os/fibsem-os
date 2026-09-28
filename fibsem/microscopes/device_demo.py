"""The Demo backend, rebuilt from devices.

``DeviceDemoMicroscope`` is the demo microscope the device migration grows, one
device at a time, beside the untouched ``DemoMicroscope``. Each device it builds
takes over its keys through ``FibsemMicroscope.get``/``set``; every key that has
not moved yet still goes to the Demo chain, which this class inherits for now.
When every key and method has moved, the inheritance goes, and this class takes
the ``DemoMicroscope`` name.

``tests/test_microscope_contract.py`` runs the same contract against both, and
compares them call by call, so the two can't drift while both exist.

Select it with ``sim: {devices: true}`` in a Demo configuration.

Devices so far: the beams (``DemoBeam``), the stage (``DemoStage``), the chamber
(``DemoChamber``) and the manipulator (``DemoManipulator``), from
``fibsem.devices.drivers.demo``. The stage is
``stage_device``, a temporary name until the stage redesign settles it; its keys
are not routed, the stage methods use it directly. The chamber is
``chamber_device``; its state and pressure keys are routed, and ``pump``/``vent``
call its commands. The manipulator is ``manipulator_device``; its position and
state keys are routed, and the manipulator methods call its commands.
"""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import Optional

from fibsem.devices.beam import BEAM_ROUTES
from fibsem.devices.chamber import CHAMBER_ROUTES
from fibsem.devices.drivers.demo import (
    bind_demo_beams,
    bind_demo_chamber,
    bind_demo_manipulator,
    bind_demo_stage,
)
from fibsem.devices.manipulator import MANIPULATOR_ROUTES
from fibsem.microscope import _records_stage_move
from fibsem.microscopes.simulator import DemoMicroscope
from fibsem.structures import (
    BeamType,
    FibsemManipulatorPosition,
    FibsemStagePosition,
)


class DeviceDemoMicroscope(DemoMicroscope):
    def connect_to_microscope(
        self, ip_address: str, port: int = 8080, reset_beam_shift: bool = True
    ) -> None:
        super().connect_to_microscope(ip_address, port, reset_beam_shift)
        self._connect_devices()

    def _connect_devices(self) -> None:
        """Build the devices and route their keys to them."""
        self.beams = MappingProxyType(bind_demo_beams(self))
        self._beam_routes = MappingProxyType(dict(BEAM_ROUTES))
        self.stage_device = bind_demo_stage(self)
        self.chamber_device = bind_demo_chamber(self)
        self.manipulator_device = bind_demo_manipulator(self)
        routes = {key: ("chamber_device", name) for key, name in CHAMBER_ROUTES.items()}
        routes.update(
            (key, ("manipulator_device", name))
            for key, name in MANIPULATOR_ROUTES.items()
        )
        self._device_routes = MappingProxyType(routes)

    # The old moves go through the device without its limit check, as today.

    @_records_stage_move
    def move_stage_absolute(self, position: FibsemStagePosition) -> FibsemStagePosition:
        self.stage_device.move_through(position)
        return self.get_stage_position()

    def home(self) -> None:
        # Demo's own override returns None, and the old API keeps that.
        self.stage_device.home()

    @_records_stage_move
    def move_stage_relative(self, position: FibsemStagePosition) -> FibsemStagePosition:
        self.stage_device.move_through(position, relative=True)
        return self.get_stage_position()

    # The manipulator methods, through the device. The old API returns what Demo's
    # returned: the position, or None from retract_manipulator.

    def insert_manipulator(self, name: str = "PARK") -> FibsemManipulatorPosition:
        return self.manipulator_device.insert(name)

    def retract_manipulator(self) -> None:
        self.manipulator_device.retract()

    def move_manipulator_absolute(
        self, position: FibsemManipulatorPosition
    ) -> FibsemManipulatorPosition:
        return self.manipulator_device.move_absolute(position)

    def move_manipulator_relative(
        self, position: FibsemManipulatorPosition
    ) -> FibsemManipulatorPosition:
        return self.manipulator_device.move_relative(position)

    def move_manipulator_corrected(
        self, dx: float, dy: float, beam_type: BeamType
    ) -> FibsemManipulatorPosition:
        # Demo applies no correction: dx and dy go straight onto x and y.
        logging.info(
            f"Moving manipulator: dx={dx:.2e}, dy={dy:.2e}, "
            f"beam_type = {beam_type.name} (Corrected)"
        )
        return self.manipulator_device.move_relative(
            FibsemManipulatorPosition(x=dx, y=dy)
        )

    def move_manipulator_to_position_offset(
        self, offset: FibsemManipulatorPosition, name: Optional[str] = None
    ) -> FibsemManipulatorPosition:
        return self.manipulator_device.move_to_offset(offset, name or "EUCENTRIC")

    def _get_saved_manipulator_position(
        self, name: str = "PARK"
    ) -> FibsemManipulatorPosition:
        return self.manipulator_device.saved_position(name)
