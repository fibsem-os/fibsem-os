"""The Demo backend, built from devices.

``DemoMicroscope`` is the simulated microscope every Demo session gets. It is its
devices and services, built over one simulator object (``DemoSimulator`` in
``fibsem.drivers.demo.simulator``), which is the Demo's connection: the imaging
channel, the image sequence, the sample scene and the milling patterns. The devices
and services are handed it at build, as Thermo's are handed ``AutoScriptConnection``.
``fibsem.drivers.demo.simulator.DemoMicroscope`` is this class too.

``tests/test_microscope_contract.py`` pins its behaviour, which is what the Demo
did before devices.

Devices, from ``fibsem.drivers.demo.devices``: the beams (``DemoBeam``), the stage
(``DemoStage``), the chamber (``DemoChamber``) and the manipulator
(``DemoManipulator``), as ``beams``, ``stage``, ``chamber_device`` and
``manipulator_device``. They are built at construction from the parts a demo starts
with (``initial_demo_parts``), and each keeps its own simulated part. Every
method that reads or changes a part goes to its device: the beam wrappers,
``beam_shift``, the scan modes and the spot burn's read of where the beam is parked;
the stage moves, ``home`` and ``link_stage`` (a compustage has no ``linked``);
``pump`` and ``vent``; and the manipulator moves.

Milling is a service, ``milling`` (``fibsem.drivers.demo.services.DemoMilling``),
which holds the Demo's milling code, and the milling methods go to it
(``FibsemMicroscope``); ``finish_milling`` puts the milling beam back as
``setup_milling`` found it.

The simulator runs imaging and the sample scene, changing the beams only through
the microscope's beam methods and so through the beam devices.

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
from typing import Any, Dict, Optional, Tuple, Union

from fibsem.devices.core import Device, resources_of
from fibsem.devices.entries import build_device_entries, resolve_system_devices
from fibsem.drivers.demo.devices import bind_demo_fm
from fibsem.drivers.demo.services import bind_demo_milling, bind_demo_spot_burn
from fibsem.drivers.demo.simulator import (
    SIM_OBJECTIVE_FOCUS_POSITION,
    DemoParts,
    DemoSimulator,
    initial_demo_parts,
    sim_is_compustage,
)
from fibsem.fm.microscope import FluorescenceMicroscope
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (
    BeamSettings,
    BeamType,
    DeviceEntry,
    FibsemExperimentRef,
    FibsemManipulatorPosition,
    FibsemRectangle,
    FibsemUser,
    ImageSettings,
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
    DeviceEntry(name="chamber_camera", type="chamber_camera"),
    DeviceEntry(name="manipulator", type="manipulator"),
)

# A simulated compustage also has an autoloader, as an Arctis does.
DEMO_SAMPLE_LOADER = DeviceEntry(name="sample_loader", type="sample_loader")


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


class DemoMicroscope(FibsemMicroscope):
    """The demo microscope built from devices, over the Demo's simulator."""

    driver_name = "Demo"
    vertical_move_views = (BeamType.ION, BeamType.ELECTRON)
    # The simulated stage turns about its origin: a move half a turn round reflects x and
    # y through zero, which is what the base class does for a driver that names no
    # centre. Named here so its images record it too. Without it they record the
    # ThermoFisher constant, and the overview draws a position from the other side of
    # the stage about 1.6 mm from where a move there goes (FIB-1081, FIB-655).
    rotation_centre = (0.0, 0.0)
    # The simulated sample loader's magazine is free to read, so the sample handling
    # service reads it when it is built.
    _read_magazine_at_connect = True

    _connection: DemoSimulator

    def __init__(self, system_settings: SystemSettings):
        self.system = system_settings
        self._connection = DemoSimulator(self)
        self._build_devices(initial_demo_parts(self.system))
        self._setup_fluorescence()

        # user, experiment metadata
        # TODO: remove once db integrated
        self.user = FibsemUser.from_environment()
        self.experiment = FibsemExperimentRef()

        self.milling_channel: BeamType = BeamType.ION
        self._connection.setup_sample_scene()
        logging.debug(
            {
                "msg": "create_microscope_client",
                "system_settings": self.system.to_dict(),
            }
        )

    def _build_devices(self, parts: DemoParts) -> None:
        """Build the devices the configuration asks for, from the starting parts.

        The Demo has every device in ``DEMO_DEVICES``; ``hardware.devices`` switches
        one off, or adds one (``fibsem.devices.entries``). The FM is built in
        ``_setup_fluorescence``.
        """
        defaults = DEMO_DEVICES
        if sim_is_compustage(self.system):
            defaults = (*defaults, DEMO_SAMPLE_LOADER)
        resolved = [
            item
            for item in resolve_system_devices(
                self.system, defaults, driver=self.driver_name
            )
            if item.type != "fm"
        ]
        # The builders start from this microscope's parts (``fibsem.drivers.demo
        # .devices``), so the devices and the shared demo code begin the same.
        shared = {self.driver_name: (resources_of(self), parts)}
        built = build_device_entries(resolved, self, shared=shared)
        for name, device in built.items():
            self._set_device(name, device)
        # A simulated instrument with no manipulator keeps no manipulator device, so
        # `is_available("manipulator")` says so (`sim.has_manipulator`).
        self._keep_if_fitted("manipulator")
        self._bind_device_roles()
        self.milling = bind_demo_milling(self)
        self._build_stage_movement()
        self.spot_burn = bind_demo_spot_burn(self)
        self._build_imaging()

    def _setup_fluorescence(self) -> None:
        # fluorescence microscope
        #
        # `has_fm` stands in for a capability read, not for configuration. A real
        # Thermo system has no `is_installed` for the FM -- every other subsystem has
        # one -- so the only way to know is to try selecting it and see whether the
        # microscope refuses, which `ThermoMicroscope.__init__` already does. The
        # simulator has nothing to ask, so it is told what the pretend hardware would
        # have answered, and that belongs in `sim:` rather than in a configuration
        # block describing the instrument.
        #
        # Deliberately separate from the fluorescence *geometry*, so that "an FM is
        # present but nothing is configured for it" stays representable -- that is the
        # state an existing site hits on upgrade, and the one worth testing (FIB-830).
        #
        # Defaults to `sim.is_compustage`, which is what this branched on before, so
        # every simulator configuration keeps its current behaviour without the key.
        has_fm = bool(self.system.sim.get("has_fm", sim_is_compustage(self.system)))

        # Two independent questions, and the simulator is the only place both can be
        # posed. `_fluorescence_is_configured` is whether the site said its instrument
        # has an FM; `has_fm` is what the hardware probe would have answered. Both are
        # required, which is what makes the middle row of the table below the sim
        # configuration representable: an FM detected on a system nothing is
        # configured for -- a site upgrading -- gets no FM, and that is the case worth
        # being able to test.
        # The probe only answers for the Demo's own FM; one on another driver (an FM
        # on its own PC) is built as the configuration says.
        if has_fm or not self._fluorescence_uses_own_driver():
            self.fm = self._build_fluorescence(self.driver_name)
        else:
            self.fm = None
        if self.fm is None:
            logging.info("No fluorescence microscope in this simulated system.")

        self._apply_fluorescence_calibration()
        self._warn_on_fluorescence_geometry()

    def connect_to_microscope(
        self, ip_address: str, port: int = 8080, reset_beam_shift: bool = True
    ) -> None:
        """Connect to the microscope server.
        Args:
            ip_address: The IP address of the microscope server.
            port: The port number of the microscope server.
            reset_beam_shift: Whether to reset beam shifts on connect (default: True).
        """
        # connect to microscope
        self._connection.connect(ip_address=ip_address, port=port)

        # system information
        self.system.info.model = "DemoMicroscope"
        self.system.info.serial_number = "123456"
        self.system.info.software_version = "0.1"
        self.system.info.hardware_version = "v0.23"
        self.system.info.ip_address = ip_address

        # reset beam shifts
        if reset_beam_shift:
            self.reset_beam_shifts()

        # user logging
        info = self.system.info
        logging.info(
            f"Microscope client connected to {info.model} with serial number {info.serial_number} and software version {info.software_version}"
        )

        # logging
        logging.debug(
            {
                "msg": "connect_to_microscope",
                "ip_address": ip_address,
                "port": port,
                "system_info": info.to_dict(),
            }
        )

        try:
            self._build_sample_handling()
        except Exception as e:
            logging.warning(f"Could not build the sample handling: {e}")

        return

    def _disconnect(self) -> None:
        """Disconnect from the microscope server."""
        self._connection.disconnect()
        logging.info("Disconnected from Demo Microscope")

    # The beam methods, on the beam devices.

    def _spot_and_beam(
        self, beam_type: BeamType
    ) -> Tuple[Union[None, Point, FibsemRectangle], BeamSettings]:
        beam = self.beams[beam_type]
        return beam.sim_scanning_mode_value, beam.sim_beam
