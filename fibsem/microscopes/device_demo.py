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

Devices so far, from ``fibsem.devices.drivers.demo``: the beams (``DemoBeam``),
the stage (``DemoStage``), the chamber (``DemoChamber``), the manipulator
(``DemoManipulator``) and the gas injection system (``DemoGasInjector``).
The beams' keys are routed. The others are ``stage_device``, ``chamber_device``,
``manipulator_device`` and ``gis_device`` (temporary names until the stage
redesign settles them); their keys are not routed, because only the base class's
own methods read them, and those methods use the devices directly. The GIS has no
keys; ``cryo_deposition_v2`` runs its sequence through the device's commands.

The FM's parts are ``fm_devices``, built over the same objects ``fm`` holds, so the
FM API and the devices share one state; ``fm`` itself is unchanged.
"""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import Dict, Optional

from fibsem._timing import sim_sleep
from fibsem.devices.beam import BEAM_ROUTES
from fibsem.devices.core import Device
from fibsem.devices.drivers.demo import (
    bind_demo_beams,
    bind_demo_chamber,
    bind_demo_gis,
    bind_demo_manipulator,
    bind_demo_stage,
)
from fibsem.devices.drivers.fm import bind_fm_devices
from fibsem.microscope import _records_stage_move
from fibsem.microscopes.simulator import DemoMicroscope
from fibsem.structures import (
    BeamType,
    FibsemGasInjectionSettings,
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
        self.gis_device = bind_demo_gis(self)
        self.fm_devices = MappingProxyType(self._fm_devices())

    def _fm_devices(self) -> Dict[str, Device]:
        """Devices over the FM ``fm`` already holds, so both drive the same parts."""
        if self.fm is None:
            return {}
        # A remote FM (``fm.driver: remote``) is already built from devices.
        devices = getattr(self.fm, "devices", None)
        if devices is not None:
            return dict(devices)
        return bind_fm_devices(self.fm)

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

    def cryo_deposition_v2(self, gis_settings: FibsemGasInjectionSettings) -> None:
        """Demo's deposition, step for step, through the GIS device.

        Demo never opens the valve (its ``gis.open()`` is commented out) but closes
        it after the wait; this keeps that.
        """
        gis = self.gis_device
        logging.info({"msg": "inserting gis", "settings": gis_settings.to_dict()})
        logging.info(
            f"Inserting Gas Injection System at {gis_settings.insert_position}"
        )
        gis.insert(gis_settings.insert_position)
        logging.info(f"Turning on heater for {gis_settings.gas}")
        gis.heater_on(gis_settings.gas)
        sim_sleep(3)  # wait for the heat
        logging.info(f"Running deposition for {gis_settings.duration} seconds")
        sim_sleep(gis_settings.duration)
        gis.close()
        logging.info(f"Turning off heater for {gis_settings.gas}")
        gis.heater_off()
        logging.info("Retracting Gas Injection System")
        gis.retract()
