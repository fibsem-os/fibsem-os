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
redesign settles them). The stage keeps its own state, so its keys are routed too:
``stage_position``, ``stage_homed`` and ``stage_linked`` read the device, and
``stage_home`` and ``stage_link`` run its commands. Demo's ``stage_system`` is left
where connect found it; only a compustage's ``stage_linked``, which its device
doesn't have, still reads it, and nothing changes that value on a compustage.
The chamber and manipulator still share Demo's state, so their keys are not routed
yet; the base class's methods that read them use the devices directly. The GIS has
no keys; ``cryo_deposition_v2`` runs its sequence through the device's commands.

The FM's parts are ``fm_devices``, built over the same objects ``fm`` holds, so the
FM API and the devices share one state; ``fm`` itself is unchanged.
"""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import Dict, Optional

from fibsem._timing import sim_sleep
from fibsem.devices.beam import BEAM_ROUTES, STAGE_COMMAND_ROUTES, STAGE_ROUTES
from fibsem.devices.core import Device
from fibsem.devices.drivers.demo import (
    bind_demo_beams,
    bind_demo_chamber,
    bind_demo_gis,
    bind_demo_manipulator,
    bind_demo_stage,
)
from fibsem.devices.drivers.fm import bind_fm_devices
from fibsem.microscope import FibsemMicroscope
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
        self._device_routes = MappingProxyType(
            {key: ("stage_device", name) for key, name in STAGE_ROUTES.items()}
        )
        self._command_routes = MappingProxyType(
            {key: ("stage_device", name) for key, name in STAGE_COMMAND_ROUTES.items()}
        )
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

    # The raw stage and manipulator moves, `home`, the saved manipulator positions
    # and the deposition are the base class's, through the devices. Demo overrides them, so
    # while this class inherits Demo it names the base class's versions explicitly.
    # What stays here is Demo's own: its corrected and offset needle moves.
    move_stage_absolute = FibsemMicroscope.move_stage_absolute
    move_stage_relative = FibsemMicroscope.move_stage_relative
    home = FibsemMicroscope.home
    insert_manipulator = FibsemMicroscope.insert_manipulator
    retract_manipulator = FibsemMicroscope.retract_manipulator
    move_manipulator_absolute = FibsemMicroscope.move_manipulator_absolute
    move_manipulator_relative = FibsemMicroscope.move_manipulator_relative
    _get_saved_manipulator_position = FibsemMicroscope._get_saved_manipulator_position
    cryo_deposition_v2 = FibsemMicroscope.cryo_deposition_v2

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
