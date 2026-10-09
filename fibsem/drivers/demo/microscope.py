"""The Demo backend, built from devices.

``DemoMicroscope`` is the simulated microscope every Demo session gets. It is its
devices plus the demo code in ``fibsem.drivers.demo.simulator`` (``DemoSession``,
``DemoConfiguration``, ``DemoImaging`` and ``DemoScene``).
``fibsem.drivers.demo.simulator.DemoMicroscope`` is this class too.

``tests/test_microscope_contract.py`` pins its behaviour, which is what the Demo
did before devices.

Devices, from ``fibsem.drivers.demo.devices``: the beams (``DemoBeam``), the stage
(``DemoStage``), the chamber (``DemoChamber``) and the manipulator
(``DemoManipulator``), as ``beams``, ``stage``, ``chamber_device`` and
``manipulator_device``. They are built at construction from the parts a demo starts
with (``initial_demo_parts``), and each keeps its own simulated part. Every key and
method that reads or changes a part goes to its device:

- the beam keys (``_beam_routes``), and ``beam_shift``, the scan-mode keys
  (``spot_mode``, ``reduced_area``, ``full_frame``) and the spot burn's read of
  where the beam is parked;
- ``stage_position``, ``stage_homed`` and ``stage_linked``, and the
  ``stage_home`` and ``stage_link`` commands; a compustage has no ``linked``, so
  its ``stage_linked`` reads the stage's own flag, which nothing changes there;
- ``chamber_state`` and ``chamber_pressure``, and the ``pump_chamber`` and
  ``vent_chamber`` commands (a false value does nothing, as before devices);
- ``manipulator_position`` and ``manipulator_state`` (a bool, as before devices);
  the old API moves the needle with methods, which use the device.

Milling is a service, ``milling`` (``fibsem.drivers.demo.services.DemoMilling``),
which holds the Demo's milling code, and the milling methods go to it
(``FibsemMicroscope``); ``finish_milling`` puts the milling beam back as
``setup_milling`` found it.

The shared code answers what the configuration alone does (the fitted parts, the
stage's limits, the grid loader and the constant value lists), and runs
imaging and the sample scene, changing the beams only through the
microscope's beam methods and so through the beam devices.

The FM is devices too: ``fm_devices`` are the Demo FM devices (``DemoCamera`` and
the rest), and ``fm`` is the FM API over them (``DemoFluorescenceMicroscope``), so
today's FM API drives the devices. The ``fm`` group holds the FM's share of the
imaging channel with the beams (``DemoFMChannel``); the FM API keeps the session's
own state: the objective's saved focus, the channel name and colour, and the image
transform. A remote FM
(``fm.driver: remote``) is already the FM API over devices, remote ones.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from fibsem import manufacturers
from fibsem._timing import sim_sleep
from fibsem.devices.beam import BEAM_ROUTES, STAGE_COMMAND_ROUTES, STAGE_ROUTES
from fibsem.devices.chamber import CHAMBER_COMMAND_ROUTES, CHAMBER_ROUTES
from fibsem.devices.core import Device, resources_of
from fibsem.devices.entries import (
    bind_device_roles,
    build_device_entries,
    configured_device_entries,
    resolve_system_devices,
)
from fibsem.devices.manipulator import MANIPULATOR_ROUTES
from fibsem.drivers.demo.devices import bind_demo_fm
from fibsem.drivers.demo.services import bind_demo_milling, bind_demo_spot_burn
from fibsem.drivers.demo.simulator import (
    SIM_OBJECTIVE_FOCUS_POSITION,
    DemoConfiguration,
    DemoImaging,
    DemoParts,
    DemoScene,
    DemoSession,
    initial_demo_parts,
    sim_is_compustage,
)
from fibsem.fm.microscope import FluorescenceMicroscope
from fibsem.microscope import FibsemMicroscope, _records_beam_shift
from fibsem.microscopes._stage import SampleGridLoader
from fibsem.structures import (
    BeamSettings,
    BeamType,
    DeviceEntry,
    FibsemManipulatorPosition,
    FibsemRectangle,
    FibsemStagePosition,
    Point,
    SystemSettings,
)

# The devices a Demo microscope has, in the order it builds them. The configuration's
# `hardware.devices` switches one off, or adds one (``fibsem.devices.entries``).
DEMO_DEVICES = (
    DeviceEntry(name="electron", type="beam"),
    DeviceEntry(name="ion", type="beam"),
    DeviceEntry(name="stage", type="stage"),
    DeviceEntry(name="chamber", type="chamber"),
    DeviceEntry(name="manipulator", type="manipulator"),
)

# A simulated compustage also has an autoloader, as an Arctis does.
DEMO_SAMPLE_LOADER = DeviceEntry(name="sample_loader", type="sample_loader")

# Today's scan-mode set keys, and the methods that run the beam commands for them.
_SCAN_MODE_KEYS: Dict[str, Callable[[Any, Any, BeamType], None]] = {
    "spot_mode": lambda m, point, bt: m.set_spot_scanning_mode(point, bt),
    "reduced_area": lambda m, area, bt: m.set_reduced_area_scanning_mode(area, bt),
    "full_frame": lambda m, _, bt: m.set_full_frame_scanning_mode(bt),
}


def _routes(device: str, routes: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
    """Old keys -> (the microscope's device attribute, the device's name for them)."""
    return {key: (device, name) for key, name in routes.items()}


class DemoFluorescenceMicroscope(FluorescenceMicroscope):
    """The FM API over the Demo FM devices, sharing the imaging channel with the
    beams through the ``fm`` group's ``DemoFMChannel``, as the Thermo FM shares the
    AutoScript connection through its group's channel.

    Without *devices* it builds the Demo FM's own, for *parent* (``bind_demo_fm``):
    a stand-in FM wherever one is needed without hardware, on a microscope or on its
    own (the FM widgets run by themselves, tests)."""

    def __init__(
        self,
        devices: Optional[Dict[str, Device]] = None,
        parent: Optional[Any] = None,
    ):
        if devices is None:
            devices = bind_demo_fm(parent)
        super().__init__(devices, parent=parent)
        self._channel = devices["fm"].channel
        # The Demo FM starts with a saved focus; a session setting, so kept here.
        self.objective._focus_position = SIM_OBJECTIVE_FOCUS_POSITION

    def set_active_channel(self) -> None:
        self._channel.set_active_channel()

    @contextmanager
    def active_channel(self):
        with self._channel.scope():
            yield


class DemoMicroscope(
    DemoSession,
    DemoConfiguration,
    DemoImaging,
    DemoScene,
    FibsemMicroscope,
):
    """The demo microscope built from devices, with the shared demo code."""

    vertical_move_views = (BeamType.ION, BeamType.ELECTRON)
    # The simulated stage turns about its origin: a move half a turn round reflects x and
    # y through zero, which is what the base class does for a driver that names no
    # centre. Named here so its images record it too. Without it they record the
    # ThermoFisher constant, and the overview draws a position from the other side of
    # the stage about 1.6 mm from where a move there goes (FIB-1081, FIB-655).
    rotation_centre = (0.0, 0.0)
    # The needle moves as the Demo's did before devices, with no correction on a corrected
    # move (``move_manipulator_corrected``); its named positions are the device's.
    manipulator_move_types = ("relative", "corrected")

    def __init__(self, system_settings: SystemSettings):
        self._start_session(system_settings)
        # The ion beam's plasma gas is read before its device is built, which only
        # offers a gas on a plasma column, so it is read at connect.
        self._read_plasma_source()
        self._build_devices(initial_demo_parts(self.system))
        self._setup_fluorescence()
        self._finish_session()

    def _build_devices(self, parts: DemoParts) -> None:
        """Build the devices the configuration asks for, from the starting parts,
        and route their keys to them.

        The Demo has every device in ``DEMO_DEVICES``; ``hardware.devices`` switches
        one off, or adds one (``fibsem.devices.entries``). The FM is built in
        ``_setup_fluorescence``.
        """
        defaults = DEMO_DEVICES
        if sim_is_compustage(self.system):
            defaults = (*DEMO_DEVICES, DEMO_SAMPLE_LOADER)
        resolved = [
            item
            for item in resolve_system_devices(
                self.system, defaults, driver=manufacturers.DEMO
            )
            if item.type != "fm"
        ]
        # The builders start from this microscope's parts (``fibsem.drivers.demo
        # .devices``), so the devices and the shared demo code begin the same.
        shared = {manufacturers.DEMO: (resources_of(self), parts)}
        built = build_device_entries(resolved, self, shared=shared)
        bind_device_roles(
            resolved, built, configured_device_entries(self.system).keys()
        )
        for name, device in built.items():
            self._set_device(name, device)
        self._beam_routes = MappingProxyType(dict(BEAM_ROUTES))
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
        self.milling = bind_demo_milling(self)
        self.spot_burn = bind_demo_spot_burn(self)

    def _create_grid_loader(self) -> Optional[SampleGridLoader]:
        """The grid model over the ``sample_loader`` device, or None when there is
        none (``enabled: false``): grids are then exchanged by hand."""
        from fibsem.microscopes._stage import DeviceSampleLoader

        device = self.devices.get("sample_loader")
        if device is None:
            logging.info("No sample loader: grids are exchanged by hand.")
            return None
        return DeviceSampleLoader(self, device, read_at_connect=True)

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
