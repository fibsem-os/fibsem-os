"""The Demo backend, built from devices.

``DemoMicroscope`` is the simulated microscope every Demo session gets. It is its
devices plus the demo code it shares with ``LegacyDemoMicroscope``, the Demo before
devices, from ``fibsem.microscopes.simulator`` (``DemoSession``,
``DemoConfiguration``, ``DemoImaging``, ``DemoScene`` and ``DemoMilling``).
``fibsem.microscopes.simulator.DemoMicroscope`` is this class too.

``tests/test_microscope_contract.py`` runs the same contract against both, and
compares them call by call against the legacy one, so the two can't drift.

Devices, from ``fibsem.devices.drivers.demo``: the beams (``DemoBeam``), the stage
(``DemoStage``), the chamber (``DemoChamber``), the manipulator
(``DemoManipulator``) and the gas injection system (``DemoGasInjector``), as
``beams``, ``stage``, ``chamber_device``, ``manipulator_device`` and
``gis_device``. They are
built at construction from the parts a demo starts with (``initial_demo_parts``),
and each keeps its own simulated part. Every key and method that reads or changes a
part goes to its device:

- the beam keys (``_beam_routes``), and ``beam_shift``, the scan-mode keys
  (``spot_mode``, ``reduced_area``, ``full_frame``) and the spot burn's read of
  where the beam is parked;
- ``stage_position``, ``stage_homed`` and ``stage_linked``, and the
  ``stage_home`` and ``stage_link`` commands; a compustage has no ``linked``, so
  its ``stage_linked`` reads the stage's own flag, which nothing changes there;
- ``chamber_state`` and ``chamber_pressure``, and the ``pump_chamber`` and
  ``vent_chamber`` commands (a false value does nothing, as on the legacy Demo);
- ``manipulator_position`` and ``manipulator_state`` (a bool, as the legacy Demo
  returns it); the old API moves the needle with methods, which use the device;
- the GIS has no keys; ``cryo_deposition_v2`` runs its sequence through the
  device's commands.

The shared code answers what the configuration alone does (the fitted parts, the
stage's limits, the grid loader, ``plasma`` and the constant value lists), and runs
imaging, the sample scene and milling, changing the beams only through
``get``/``set`` and so, here, through the beam devices.

The FM is devices too: ``fm_devices`` are the Demo FM devices (``DemoCamera`` and
the rest), each copied when it is built from the part the simulated FM built, and
``fm`` is the FM API over them (``DemoFluorescenceMicroscope``), so today's
FM API drives the devices. It keeps the simulated FM's share of the imaging
channel with the beams, and the session's own state: the objective's saved focus,
the channel name and colour, and the image transform. A remote FM
(``fm.driver: remote``) is already the FM API over devices, remote ones.
"""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from fibsem import manufacturers
from fibsem._timing import sim_sleep
from fibsem.devices.beam import BEAM_ROUTES, STAGE_COMMAND_ROUTES, STAGE_ROUTES
from fibsem.devices.chamber import CHAMBER_COMMAND_ROUTES, CHAMBER_ROUTES
from fibsem.devices.core import Device
from fibsem.devices.drivers.demo import (
    bind_demo_beams,
    bind_demo_chamber,
    bind_demo_fm,
    bind_demo_gis,
    bind_demo_manipulator,
    bind_demo_stage,
)
from fibsem.devices.manipulator import MANIPULATOR_ROUTES
from fibsem.fm.api import DeviceFluorescenceMicroscope
from fibsem.microscope import FibsemMicroscope, _records_beam_shift
from fibsem.microscopes.registry import DriverEntry
from fibsem.microscopes.simulator import (
    SIMULATOR_KNOWN_UNKNOWN_KEYS,
    DemoConfiguration,
    DemoImaging,
    DemoMilling,
    DemoParts,
    DemoScene,
    DemoSession,
    SimulatedFluorescenceMicroscope,
    initial_demo_parts,
)
from fibsem.structures import (
    BeamSettings,
    BeamType,
    FibsemGasInjectionSettings,
    FibsemManipulatorPosition,
    FibsemRectangle,
    FibsemStagePosition,
    Point,
    SystemSettings,
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


# The beam keys Demo answers without a beam type: no plasma gas, and no preset.
_BEAM_KEYS_WITHOUT_BEAM = ("plasma_gas", "preset")


class DemoFluorescenceMicroscope(
    DeviceFluorescenceMicroscope, SimulatedFluorescenceMicroscope
):
    """The FM API over the Demo FM devices, sharing the imaging channel with
    the beams as the simulated FM does (``SimulatedFluorescenceMicroscope``)."""


def _needs_beam_type(key: str, beam_type: Optional[BeamType]) -> None:
    """A beam key with no beam type raises, as the legacy Demo's branches do."""
    if beam_type is None and key in BEAM_ROUTES and key not in _BEAM_KEYS_WITHOUT_BEAM:
        raise ValueError(f"{key} needs a beam type")


def _unknown_key(key: str, beam_type: Optional[BeamType]) -> None:
    """Log a key no device or shared code answers, as the legacy Demo does; it
    reads None."""
    if key in SIMULATOR_KNOWN_UNKNOWN_KEYS:
        logging.debug(f"Skipping unknown key: {key} for {beam_type}")
        return
    logging.warning(f"Unknown key: {key} ({beam_type})")


# This driver, as the registry knows it (fibsem.microscopes.registry).
DRIVER = DriverEntry(
    manufacturer=manufacturers.DEMO,
    microscope_class="fibsem.microscopes.device_demo:DemoMicroscope",
    config={"port": 7520, "ion-column-tilt": 52, "electron-column-tilt": 0},
)


class DemoMicroscope(
    DemoSession,
    DemoConfiguration,
    DemoImaging,
    DemoScene,
    DemoMilling,
    FibsemMicroscope,
):
    """The demo microscope built from devices, with the shared demo code."""

    vertical_move_views = (BeamType.ION, BeamType.ELECTRON)
    # The needle moves as the legacy Demo's does, with no correction on a corrected
    # move (``move_manipulator_corrected``); its named positions are the device's.
    manipulator_move_types = ("relative", "corrected")

    def __init__(self, system_settings: SystemSettings):
        self._start_session(system_settings)
        # The ion beam's plasma gas is read before its device is built, which only
        # offers a gas on a plasma column; the legacy Demo reads it at connect.
        self._read_plasma_source()
        self._build_devices(initial_demo_parts(self.system))
        self._setup_fluorescence()
        self.fm_devices = MappingProxyType(self._fm_devices())
        self._finish_session()

    def _build_devices(self, parts: DemoParts) -> None:
        """Build the devices from the starting parts and route their keys to them."""
        self.beams = MappingProxyType(bind_demo_beams(self, start=parts))
        self._beam_routes = MappingProxyType(dict(BEAM_ROUTES))
        self.stage = bind_demo_stage(self, start=parts)
        self.chamber_device = bind_demo_chamber(self, start=parts)
        self.manipulator_device = bind_demo_manipulator(self, start=parts)
        self._device_routes = MappingProxyType(
            {
                **_routes("stage", STAGE_ROUTES),
                **_routes("chamber_device", CHAMBER_ROUTES),
                **_routes("manipulator_device", MANIPULATOR_ROUTES),
            }
        )
        self._command_routes = MappingProxyType(
            {
                **_routes("stage", STAGE_COMMAND_ROUTES),
                **_routes("chamber_device", CHAMBER_COMMAND_ROUTES),
            }
        )
        self.gis_device = bind_demo_gis(self, start=parts)

    def _fm_devices(self) -> Dict[str, Device]:
        """The FM's devices, and ``fm`` as the FM API over them."""
        if self.fm is None:
            return {}
        # A remote FM (``fm.driver: remote``) is already the FM API over devices.
        devices = getattr(self.fm, "devices", None)
        if devices is not None:
            return dict(devices)
        simulated = self.fm
        devices = bind_demo_fm(self, simulated, config=self.system.fm.to_dict())
        self.fm = DemoFluorescenceMicroscope(devices, parent=self)
        # The saved focus is the session's, so the FM API keeps it; the configured
        # one was applied to the simulated FM's objective before the devices existed.
        self.fm.objective.focus_position = simulated.objective.focus_position
        return devices

    # The beam methods the shared demo code leaves to each demo, on the beam devices.

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
        if self._set_imaging_key(key, value) or self._set_milling_key(key, value):
            return
        # The ion beam has a plasma gas only on a plasma column.
        if key == "plasma_gas" and beam_type is BeamType.ION:
            logging.debug("Plasma gas cannot be set on this microscope.")
            return
        _needs_beam_type(key, beam_type)
        # A command its device doesn't run: the legacy Demo's branches log and do
        # nothing.
        if key == "stage_link":
            logging.debug("Compustage does not support linking.")
            return
        if key in ("pump_chamber", "vent_chamber"):
            logging.info(f"Invalid value for {key}: {value}")
            return
        _unknown_key(key, beam_type)

    def get_available_values(
        self, key: str, beam_type: Optional[BeamType] = None
    ) -> List[Any]:
        # A beam key's values are its parameter's choices, then the configured and
        # milling ones; a beam key asked without a beam type gets the shared answer.
        param = self._route(key, beam_type)
        if param is not None and param.choices is not None:
            return list(param.choices)
        configured = self._configured_values(key)
        if configured is None:
            configured = self._milling_values(key)
        if configured is not None:
            return configured
        return super().get_available_values(key, beam_type)

    def _get(self, key: str, beam_type: Optional[BeamType] = None) -> Any:
        if key == "plasma":
            return self._read_plasma(beam_type)
        # A beam has a plasma gas only on a plasma column.
        if key == "plasma_gas":
            return None
        _needs_beam_type(key, beam_type)
        # A compustage has no link; nothing changes the stage's own flag there.
        if key == "stage_linked":
            return self.stage.sim_linked
        return _unknown_key(key, beam_type)

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
