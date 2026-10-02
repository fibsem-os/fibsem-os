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

Devices, from ``fibsem.devices.drivers.demo``: the beams (``DemoBeam``), the stage
(``DemoStage``), the chamber (``DemoChamber``), the manipulator
(``DemoManipulator``) and the gas injection system (``DemoGasInjector``), as
``beams``, ``stage_device``, ``chamber_device``, ``manipulator_device`` and
``gis_device`` (temporary names until the stage redesign settles them). Each keeps
its own simulated part, copied from Demo's at connect, and Demo's own parts are
never touched again. So every key and method that reads or changes a part goes to
its device:

- the beam keys (``_beam_routes``), and ``beam_shift``, the scan-mode keys
  (``spot_mode``, ``reduced_area``, ``full_frame``) and the spot burn's read of
  where the beam is parked;
- ``stage_position``, ``stage_homed`` and ``stage_linked``, and the
  ``stage_home`` and ``stage_link`` commands; a compustage has no ``linked``, so
  its ``stage_linked`` still reads Demo's, which nothing changes there;
- ``chamber_state`` and ``chamber_pressure``, and the ``pump_chamber`` and
  ``vent_chamber`` commands (a false value still goes to Demo's branch, which does
  nothing);
- ``manipulator_position`` and ``manipulator_state`` (a bool, as Demo returns it);
  the old API moves the needle with methods, which use the device;
- the GIS has no keys; ``cryo_deposition_v2`` runs its sequence through the
  device's commands.

What still goes to the Demo chain is what is not a device's: configuration
(``plasma``), imaging, milling, the FM and the sample scene.

The FM's parts are ``fm_devices``, built over the same objects ``fm`` holds, so the
FM API and the devices share one state; ``fm`` itself is unchanged.
"""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import Any, Callable, Dict, Optional, Tuple, Union

from fibsem._timing import sim_sleep
from fibsem.devices.beam import BEAM_ROUTES, STAGE_COMMAND_ROUTES, STAGE_ROUTES
from fibsem.devices.chamber import CHAMBER_COMMAND_ROUTES, CHAMBER_ROUTES
from fibsem.devices.core import Device
from fibsem.devices.drivers.demo import (
    bind_demo_beams,
    bind_demo_chamber,
    bind_demo_gis,
    bind_demo_manipulator,
    bind_demo_stage,
)
from fibsem.devices.drivers.fm import bind_fm_devices
from fibsem.devices.manipulator import MANIPULATOR_ROUTES
from fibsem.microscope import FibsemMicroscope, _records_beam_shift
from fibsem.microscopes.simulator import DemoMicroscope
from fibsem.structures import (
    BeamSettings,
    BeamType,
    FibsemGasInjectionSettings,
    FibsemManipulatorPosition,
    FibsemRectangle,
    FibsemStagePosition,
    Point,
)

# Today's scan-mode set keys, and the methods that run the beam commands for them.
_SCAN_MODE_KEYS: Dict[str, Callable[[Any, Any, BeamType], None]] = {
    "spot_mode": lambda m, point, bt: m.set_spot_scanning_mode(point, bt),
    "reduced_area": lambda m, area, bt: m.set_reduced_area_scanning_mode(area, bt),
    "full_frame": lambda m, _, bt: m.set_full_frame_scanning_mode(bt),
}


def _routes(device: str, routes: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
    """Old keys -> (the microscope's device attribute, the device's name for them)."""
    return {key: (device, name) for key, name in routes.items()}


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
        self._device_routes = MappingProxyType(
            {
                **_routes("stage_device", STAGE_ROUTES),
                **_routes("chamber_device", CHAMBER_ROUTES),
                **_routes("manipulator_device", MANIPULATOR_ROUTES),
            }
        )
        self._command_routes = MappingProxyType(
            {
                **_routes("stage_device", STAGE_COMMAND_ROUTES),
                **_routes("chamber_device", CHAMBER_COMMAND_ROUTES),
            }
        )
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

    # Demo's beam methods that touch its beam state directly, on the beam devices.

    @_records_beam_shift
    def beam_shift(self, dx: float, dy: float, beam_type: BeamType) -> None:
        logging.debug(
            {"msg": "beam_shift", "dx": dx, "dy": dy, "beam_type": beam_type.name}
        )
        shift = self.beams[beam_type].shift
        shift.write_through(shift.get_value() + Point(float(dx), float(dy)))

    def _spot_and_beam(
        self, beam_type: BeamType
    ) -> Tuple[Union[None, Point, FibsemRectangle], BeamSettings]:
        beam = self.beams[beam_type]
        return beam.sim_scanning_mode_value, beam.sim_beam

    def _set(self, key: str, value, beam_type: Optional[BeamType] = None) -> None:
        # The scan-mode keys are the beam's commands; the methods use them already.
        if beam_type is not None and key in _SCAN_MODE_KEYS:
            _SCAN_MODE_KEYS[key](self, value, beam_type)
            return
        super()._set(key, value, beam_type)

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
