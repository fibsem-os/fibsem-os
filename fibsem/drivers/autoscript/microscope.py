"""AutoScript (ThermoFisher) microscope backend.

`ThermoMicroscope` and every AutoScript-specific helper live here, so the AutoScript
import shim below is the only place the SDK is loaded.
"""

from __future__ import annotations

import copy
import glob
import logging
import os
import sys
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
from packaging.version import InvalidVersion, Version
from packaging.version import parse as parse_version

from fibsem.devices.entries import build_device_entries, resolve_system_devices
from fibsem.devices.sample_loader import GridExchangeError
from fibsem.drivers.autoscript.connection import AutoScriptConnection
from fibsem.microscope import (
    FibsemMicroscope,
    RequiredDeviceUnavailable,
    _records_beam_shift,
)
from fibsem.structures import (
    BeamType,
    DeviceEntry,
    FibsemDetectorSettings,
    FibsemExperimentRef,
    FibsemImage,
    FibsemManipulatorPosition,
    FibsemRectangle,
    FibsemStagePosition,
    FibsemUser,
    GridSlot,
    ImageSettings,
    MicroscopeState,
    Point,
    RangeLimit,
    SampleGrid,
    SystemSettings,
)
from fibsem.util.timestamps import to_datetime

if TYPE_CHECKING:
    from autoscript_sdb_microscope_client._dynamic_object_proxies import (
        ElectronBeam,
        IonBeam,
    )

    from fibsem.fm.microscope import FluorescenceMicroscope

THERMO_API_AVAILABLE = False
MINIMUM_AUTOSCRIPT_VERSION = parse_version("4.9")
# Set when the guarded import below fails, so the connection error can say why.
THERMO_API_IMPORT_ERROR: Optional[str] = None
# Declared so importers can rely on the name; only meaningful once
# THERMO_API_AVAILABLE is True.
AUTOSCRIPT_VERSION: Optional[Version] = None

# The scan directions xT offers for a pattern.
TFS_SCAN_DIRECTIONS = [
    "BottomToTop",
    "DynamicAllDirections",
    "DynamicInnerToOuter",
    "DynamicLeftToRight",
    "DynamicTopToBottom",
    "InnerToOuter",
    "LeftToRight",
    "OuterToInner",
    "RightToLeft",
    "TopToBottom",
]

# The voltages a ThermoFisher microscope offers, per beam, in volts. The API gives
# only a range, and any value in it can be set, but these are the ones xT lists.
THERMO_VOLTAGE_CHOICES = {
    BeamType.ELECTRON: (1000, 2000, 3000, 5000, 10000, 20000, 30000),
    BeamType.ION: (500, 1000, 2000, 8000, 16000, 30000),
}

# Legacy install locations, kept on sys.path for older machines. Current installs
# copy the AutoScript packages into the active environment's site-packages (see
# INSTALLATION.md), so these rarely match any more.
_LEGACY_AUTOSCRIPT_SYS_PATHS = [
    r"C:\Program Files\Thermo Scientific AutoScript",
    r"C:\Program Files\Enthought\Python\envs\AutoScript\Lib\site-packages",
    r"C:\Program Files\Python36\envs\AutoScript",
    r"C:\Program Files\Python36\envs\AutoScript\Lib\site-packages",
]

# Where a copy of AutoScript might be sitting when the import fails. Searched only
# to build the diagnostic message; never added to sys.path.
_AUTOSCRIPT_SEARCH_GLOBS = [
    r"C:\ProgramData\miniforge3\envs\*\Lib\site-packages",
    r"C:\ProgramData\miniconda3\envs\*\Lib\site-packages",
    r"C:\ProgramData\anaconda3\envs\*\Lib\site-packages",
    os.path.expanduser(r"~\miniforge3\envs\*\Lib\site-packages"),
    os.path.expanduser(r"~\miniconda3\envs\*\Lib\site-packages"),
    os.path.expanduser(r"~\anaconda3\envs\*\Lib\site-packages"),
    os.path.expanduser(r"~\.conda\envs\*\Lib\site-packages"),
    r"C:\Program Files\Python*\envs\AutoScript\Lib\site-packages",
    *_LEGACY_AUTOSCRIPT_SYS_PATHS,
]


class AutoScriptException(Exception):
    pass


try:
    for _legacy_path in _LEGACY_AUTOSCRIPT_SYS_PATHS:
        sys.path.append(_legacy_path)
    import autoscript_sdb_microscope_client
    from autoscript_sdb_microscope_client import SdbMicroscopeClient

    version = autoscript_sdb_microscope_client.build_information.INFO_VERSIONSHORT
    try:
        AUTOSCRIPT_VERSION = parse_version(version)
    except InvalidVersion:
        raise AutoScriptException(f"Failed to parse AutoScript version '{version}'")

    # special case for Monash development environment
    if os.environ.get("COMPUTERNAME", "hostname") == "MU00190108":
        logging.info("Overwriting autoscript version to 4.9, for Monash dev install")
        AUTOSCRIPT_VERSION = MINIMUM_AUTOSCRIPT_VERSION

    if AUTOSCRIPT_VERSION < MINIMUM_AUTOSCRIPT_VERSION:
        raise AutoScriptException(
            f"AutoScript {version} found. Please update your AutoScript version to 4.9 or higher."
        )

    from autoscript_sdb_microscope_client._dynamic_object_proxies import (
        CirclePattern,
        CleaningCrossSectionPattern,
        LinePattern,
        RectanglePattern,
        RegularCrossSectionPattern,
    )
    from autoscript_sdb_microscope_client.enumerations import (
        CoordinateSystem,
        ImagingState,
        ManipulatorCoordinateSystem,
        ManipulatorSavedPosition,
        ManipulatorState,
        PatterningState,
        RegularCrossSectionScanMethod,
    )
    from autoscript_sdb_microscope_client.structures import (
        AdornedImage,
        BitmapPatternDefinition,
        CompustagePosition,
        GetImageSettings,
        GrabFrameSettings,
        Limits,
        Limits2d,
        ManipulatorPosition,
        MoveSettings,
        Rectangle,
        StagePosition,
    )

    THERMO_API_AVAILABLE = True
except AutoScriptException as e:
    THERMO_API_IMPORT_ERROR = str(e)
    logging.warning("Failed to load AutoScript (ThermoFisher): %s", str(e))
except ImportError as e:
    THERMO_API_IMPORT_ERROR = str(e)
    logging.debug("AutoScript (ThermoFisher) not found: %s", str(e))
except Exception as e:
    THERMO_API_IMPORT_ERROR = str(e)
    logging.error(
        "Failed to load AutoScript (ThermoFisher) due to unexpected error",
        exc_info=True,
    )


def find_autoscript_install_candidates() -> List[str]:
    """site-packages directories holding an AutoScript client package.

    A best-effort search of the common conda and venv roots, used only to explain
    a failed import. It does not change what gets imported.
    """
    candidates: List[str] = []
    for pattern in _AUTOSCRIPT_SEARCH_GLOBS:
        for site_packages_dir in glob.glob(pattern):
            marker = os.path.join(site_packages_dir, "autoscript_sdb_microscope_client")
            if os.path.isdir(marker) and site_packages_dir not in candidates:
                candidates.append(site_packages_dir)
    return candidates


def _is_active_environment(site_packages_dir: str) -> bool:
    """Whether `site_packages_dir` is on this interpreter's sys.path.

    Separates a broken copy in the environment that is running from a complete
    copy in some other environment that was never activated.
    """
    target = os.path.normcase(os.path.abspath(site_packages_dir))
    return any(os.path.normcase(os.path.abspath(p)) == target for p in sys.path if p)


def _format_candidate_list(paths: List[str], limit: int = 5) -> str:
    shown = paths[:limit]
    text = "; ".join(shown)
    remaining = len(paths) - len(shown)
    if remaining > 0:
        text += f"; and {remaining} more"
    return text


def autoscript_unavailable_message() -> str:
    """Why AutoScript could not be used, with the most likely fix.

    Three different situations used to share one "not installed" message: never
    installed, too old, or a partial copy. The import error is quoted when there
    is one, and any copies found on disk are sorted by whether they sit in the
    environment that is running.
    """
    parts = ["Autoscript (ThermoFisher) is not available."]

    if THERMO_API_IMPORT_ERROR:
        parts.append(f"Reason: {THERMO_API_IMPORT_ERROR}.")

    candidates = find_autoscript_install_candidates()
    active_env_candidates = [c for c in candidates if _is_active_environment(c)]
    other_env_candidates = [c for c in candidates if c not in active_env_candidates]

    if active_env_candidates:
        parts.append(
            "Found AutoScript packages in the currently active environment ("
            + _format_candidate_list(active_env_candidates)
            + ") but they did not import successfully -- check that ALL required "
            "packages were copied (autoscript_core, autoscript_sdb_microscope_client, "
            "autoscript_sdb_microscope_client_tests, autoscript_toolkit, "
            "thermoscientific_logging) and that their versions match."
        )
    if other_env_candidates:
        parts.append(
            "Found AutoScript packages in a different environment than the one "
            "currently running: "
            + _format_candidate_list(other_env_candidates)
            + ". Make sure you activated the environment AutoScript was copied "
            "into, or copy the AutoScript packages into this environment's "
            "site-packages instead."
        )
    if not candidates:
        parts.append(
            f"No AutoScript installation was found in {len(_AUTOSCRIPT_SEARCH_GLOBS)} "
            "common locations."
        )

    parts.append("Please see INSTALLATION.md for installation instructions.")
    return " ".join(parts)


def stage_position_to_autoscript(
    position: "FibsemStagePosition", compustage: bool = False
) -> Union["StagePosition", "CompustagePosition"]:
    """Convert a FibsemStagePosition to an AutoScript StagePosition or CompustagePosition.

    Args:
        position: The FibsemStagePosition to convert.
        compustage: Whether the stage is a compustage.

    Returns:
        StagePosition or CompustagePosition compatible with AutoScript.

    Raises:
        ImportError: If AutoScript libraries are not available.
    """
    if not THERMO_API_AVAILABLE:
        raise ImportError(
            "AutoScript libraries not available. Cannot convert to AutoScript position."
        )

    if compustage:
        return CompustagePosition(
            x=position.x,
            y=position.y,
            z=position.z,
            a=position.t,
            coordinate_system=CoordinateSystem.SPECIMEN,
        )
    else:
        return StagePosition(
            x=position.x,
            y=position.y,
            z=position.z,
            r=position.r,
            t=position.t,
            coordinate_system=CoordinateSystem.RAW,
        )


def stage_position_from_autoscript(
    position: Union["StagePosition", "CompustagePosition"],
) -> "FibsemStagePosition":
    """Create a FibsemStagePosition from an AutoScript position object.

    Args:
        position: AutoScript StagePosition or CompustagePosition.

    Returns:
        FibsemStagePosition: Converted position.

    Raises:
        ImportError: If AutoScript libraries are not available.
    """
    if not THERMO_API_AVAILABLE:
        raise ImportError(
            "AutoScript libraries not available. Cannot convert from AutoScript position."
        )

    from fibsem.structures import FibsemStagePosition

    if isinstance(position, CompustagePosition):
        return FibsemStagePosition(
            x=position.x,
            y=position.y,
            z=position.z,
            r=0.0,
            t=position.a,
            coordinate_system=CoordinateSystem.SPECIMEN.upper(),
        )

    return FibsemStagePosition(
        x=position.x,
        y=position.y,
        z=position.z,
        r=position.r,
        t=position.t,
        coordinate_system=position.coordinate_system.upper(),
    )


def manipulator_position_to_autoscript(
    position: "FibsemManipulatorPosition",
) -> "ManipulatorPosition":
    """Convert a FibsemManipulatorPosition to an AutoScript ManipulatorPosition.

    Args:
        position: The FibsemManipulatorPosition to convert.

    Returns:
        ManipulatorPosition compatible with AutoScript.

    Raises:
        ImportError: If AutoScript libraries are not available.
    """
    if not THERMO_API_AVAILABLE:
        raise ImportError(
            "AutoScript libraries not available. Cannot convert to AutoScript position."
        )

    if position.coordinate_system == "RAW":
        coordinate_system = "Raw"
    elif position.coordinate_system == "STAGE":
        coordinate_system = "Stage"
    else:
        coordinate_system = position.coordinate_system

    return ManipulatorPosition(
        x=position.x,
        y=position.y,
        z=position.z,
        r=None,
        coordinate_system=coordinate_system,
    )


def manipulator_position_from_autoscript(
    position: "ManipulatorPosition",
) -> "FibsemManipulatorPosition":
    """Create a FibsemManipulatorPosition from an AutoScript ManipulatorPosition.

    Args:
        position: AutoScript ManipulatorPosition.

    Returns:
        FibsemManipulatorPosition: Converted position.

    Raises:
        ImportError: If AutoScript libraries are not available.
    """
    if not THERMO_API_AVAILABLE:
        raise ImportError(
            "AutoScript libraries not available. Cannot convert from AutoScript position."
        )

    from fibsem.structures import FibsemManipulatorPosition

    return FibsemManipulatorPosition(
        x=position.x,
        y=position.y,
        z=position.z,
        coordinate_system=position.coordinate_system.upper(),
    )


# The device types each connect step builds (``ThermoMicroscope._build_devices``).
# The FM is built on its own path; any other type a configuration adds is built last.
_BEAM_TYPES = ("beam",)
_STAGE_TYPES = ("stage",)
_PART_TYPES = ("chamber", "manipulator")
_LOADER_TYPES = ("sample_loader",)
_OWN_TYPES = _BEAM_TYPES + _STAGE_TYPES + _PART_TYPES + _LOADER_TYPES + ("fm",)


class ThermoMicroscope(FibsemMicroscope):
    """
    A class representing a Thermo Fisher FIB-SEM microscope.

    This class inherits from the abstract base class `FibsemMicroscope`, which defines the core functionality of a
    microscope. In addition to the methods defined in the base class, this class provides additional methods specific
    to the Thermo Fisher FIB-SEM microscope.

    Attributes:
        connection (SdbMicroscopeClient): The microscope client connection.

    Inherited Methods:
        connect_to_microscope(self, ip_address: str, port: int = 7520) -> None:
            Connect to a Thermo Fisher microscope at the specified IP address and port.

        disconnect(self) -> None:
            Disconnects the microscope client connection.

        acquire_image(self, image_settings: ImageSettings) -> FibsemImage:
            Acquire a new image with the specified settings.

        last_image(self, beam_type: BeamType = BeamType.ELECTRON) -> FibsemImage:
            Get the last previously acquired image.

        autocontrast(self, beam_type: BeamType) -> None:
            Automatically adjust the microscope image contrast for the specified beam type.

        auto_focus(self, beam_type: BeamType) -> None:
            Automatically adjust the microscope focus for the specified beam type.

        beam_shift(self, dx: float, dy: float,  beam_type: BeamType) -> None:
            Adjusts the beam shift of given beam based on relative values that are provided.

        move_stage_absolute(self, position: FibsemStagePosition):
            Move the stage to the specified coordinates.

        move_stage_relative(self, position: FibsemStagePosition):
            Move the stage by the specified relative move.

        stable_move(self, dx: float, dy: float, beam_type: BeamType,) -> None:
            Calculate the corrected stage movements based on the beam_type, and then move the stage relatively.

        vertical_move(self, dy: float, dx: float = 0, beam_type: BeamType = BeamType.ION) -> None:
            Move the stage to correct the coincidence point, from either beam view.

        get_manipulator_position(self) -> FibsemManipulatorPosition:
            Get the current manipulator position.

        insert_manipulator(self, name: str) -> None:
            Insert the manipulator into the sample.

        retract_manipulator(self) -> None:
            Retract the manipulator from the sample.

        move_manipulator_relative(self, position: FibsemManipulatorPosition) -> None:
            Move the manipulator by the specified relative move.

        move_manipulator_absolute(self, position: FibsemManipulatorPosition) -> None:
            Move the manipulator to the specified coordinates.

        move_manipulator_corrected(self, dx: float, dy: float, beam_type: BeamType) -> None:
            Move the manipulator by the specified relative move, correcting for the beam type.

        move_manipulator_to_position_offset(self, offset: FibsemManipulatorPosition, name: str) -> None:
            Move the manipulator to the specified position offset.

        _get_saved_manipulator_position(self, name: str) -> FibsemManipulatorPosition:
            Get the saved manipulator position with the specified name.

        setup_milling(self, mill_settings: FibsemMillingSettings):
            Configure the microscope for milling using the ion beam.

        run_milling(self, stop_event=None):
            Mill what is drawn and return when the mill ends.

        finish_milling(self, imaging_current: float):
            Finalises the milling process by clearing the microscope of any patterns and returning the current to the imaging current.

        set_microscope_state(self, microscope_state: MicroscopeState) -> None:
            Reset the microscope state to the provided state.

    New methods:
        __init__(self):
            Initializes a new instance of the class.
    """

    driver_name = "ThermoFisher"
    vertical_move_views = (BeamType.ION, BeamType.ELECTRON)

    # The AutoScript client, from construction until disconnect. The devices and
    # services are handed it; nothing else reaches the SDK.
    _connection: Optional[AutoScriptConnection] = None

    # Specimen minus raw coordinates, (x, y) metres: where xT centres its compucentric
    # rotation, as an offset. Read once at connect, since xT does not change it while
    # connected; None on a compustage, which does not rotate, or before connect.
    _compucentric_offset: Optional[Tuple[float, float]] = None

    # The port connect_to_microscope used: info.port, or the driver's default.
    # reconnect uses it again.
    _port: Optional[int] = None

    def __init__(self, system_settings: SystemSettings):
        if not THERMO_API_AVAILABLE:
            raise Exception(autoscript_unavailable_message())

        # create microscope client
        self._connection = self._new_connection()

        # initialise system settings
        self.system: SystemSettings = system_settings

        # user, experiment metadata
        # TODO: remove once db integrated
        self.user = FibsemUser.from_environment()
        self.experiment = FibsemExperimentRef()

        # logging
        logging.debug(
            {
                "msg": "create_microscope_client",
                "system_settings": system_settings.to_dict(),
            }
        )

    def _new_connection(self) -> AutoScriptConnection:
        """A client, sharing the imaging channel lock with the devices."""
        return AutoScriptConnection(SdbMicroscopeClient(), self._threading_lock)

    def reconnect(self):
        """Attempt to reconnect to the microscope client."""
        if self._connection is None:
            raise ConnectionError("Please connect to the microscope first")

        self.disconnect()
        if self._port is None:
            self.connect_to_microscope(self.system.info.ip_address)
        else:
            self.connect_to_microscope(self.system.info.ip_address, port=self._port)

    def disconnect(self):
        """Disconnect from the microscope client."""
        if self._connection is None:
            logging.warning("Microscope client is not connected.")
            return

        self._connection.disconnect()
        self._connection = None

    def connect_to_microscope(
        self, ip_address: str, port: int = 7520, reset_beam_shift: bool = True
    ) -> None:
        """
        Connect to a Thermo Fisher microscope at the specified IP address and port.

        Args:
            ip_address (str): The IP address of the microscope to connect to.
            port (int): The port number of the microscope (default: 7520).
            reset_beam_shift (bool): Whether to reset beam shifts on connect (default: True).

        Returns:
            None: This function doesn't return anything.

        Raises:
            Exception: If there's an error while connecting to the microscope.

        Example:
            To connect to a microscope with IP address 192.168.0.10 and port 7520:

            >>> microscope = ThermoMicroscope()
            >>> microscope.connect_to_microscope("192.168.0.10", 7520)
        """
        if self._connection is None:
            self._connection = self._new_connection()
        sdk = self._connection.sdk

        logging.info(f"Microscope client connecting to [{ip_address}:{port}]")
        self._connection.connect(host=ip_address, port=port)
        self._port = port
        logging.info(f"Microscope client connected to [{ip_address}:{port}]")

        # system information
        self.system.info.model = sdk.service.system.name
        self.system.info.serial_number = sdk.service.system.serial_number
        self.system.info.hardware_version = sdk.service.system.version
        self.system.info.software_version = sdk.service.autoscript.client.version
        info = self.system.info
        logging.info(
            f"Microscope client connected to model {info.model} with serial number {info.serial_number} and software version {info.software_version}."
        )

        # autoscript information
        logging.info(f"Autoscript Client: {sdk.service.autoscript.client.version}")
        logging.info(f"Autoscript Server: {sdk.service.autoscript.server.version}")

        self._build_beams()
        self._build_milling()
        self._build_spot_burn()
        self._build_imaging()

        if reset_beam_shift:
            self.reset_beam_shifts()

        # the stage, in its default coordinate system from now on
        self._connection.find_stage()
        self._compucentric_offset = self._connection.read_compucentric_offset()
        self._build_stage()
        self._build_stage_movement()
        # TODO: set default move settings, is this dependent on the stage type?
        if self.milling is not None:
            # AutoScript needs a valid application file before a pattern is made
            self.milling._set_application_file("Si", default=True)

        self._last_imaging_settings: ImageSettings = ImageSettings()
        self.milling_channel: BeamType = BeamType.ION

        from fibsem.drivers.autoscript.devices import ThermoFisherFluorescenceMicroscope

        # Its own FM leaves the shared imaging channel on the FM, as building it
        # selects the channel; any other outcome leaves it on the electron beam.
        self.fm = self._build_fluorescence(self.driver_name)
        if not isinstance(self.fm, ThermoFisherFluorescenceMicroscope):
            self.set_channel(BeamType.ELECTRON)

        self._apply_fluorescence_calibration()
        self._warn_on_fluorescence_geometry()

        try:
            self._build_sample_handling()
        except Exception as e:
            logging.warning(f"Could not build the sample handling: {e}")

        self._build_parts()
        self._bind_device_roles()

    def _build_devices(
        self,
        defaults: List[DeviceEntry],
        types: Optional[Tuple[str, ...]] = None,
        exclude_types: Tuple[str, ...] = (),
    ) -> Dict[str, Any]:
        """Build one connect step's devices: *defaults*, what the instrument has, with
        the configuration's ``hardware.devices`` entries of *types* over them
        (``fibsem.devices.entries``), and put them in ``devices``."""
        resolved = resolve_system_devices(
            self.system, defaults, types, exclude_types, self.driver_name
        )
        built = build_device_entries(resolved, self)
        for name, device in built.items():
            self._set_device(name, device)
        return built

    def _build_beams(self) -> None:
        """Build the beam devices.

        The scan-mode methods then use the beam's scan commands, and acquire_image,
        last_image, autocontrast and auto_focus its imaging commands. A disabled column
        gets no device: its wrappers read None, and its imaging raises.
        """
        self._build_devices(
            [
                DeviceEntry(name="electron", type="beam"),
                DeviceEntry(name="ion", type="beam"),
            ],
            _BEAM_TYPES,
        )

    def _build_milling(self) -> None:
        """Build the milling service over the beams; the milling methods then go to it
        (`FibsemMicroscope`). Without an ion beam there is none, and they raise."""
        from fibsem.drivers.autoscript.services import bind_autoscript_milling

        self.milling = bind_autoscript_milling(self)

    def _build_spot_burn(self) -> None:
        """Build the spot burn service over the beams; `run_spot_burn` then goes to
        it: the point-by-point burn, with the beams' spot and blank commands."""
        from fibsem.services.spot_burn import SpotBurn, bind_spot_burn

        self.spot_burn = bind_spot_burn(SpotBurn, self)

    def _build_stage(self) -> None:
        """Build the stage device.

        The moves, ``home`` and ``link_stage`` then go through the device. Its
        ``link`` command only links, and a compustage has no ``linked`` (it reads
        None).
        """
        self._build_devices([DeviceEntry(name="stage", type="stage")], _STAGE_TYPES)

    def _build_parts(self) -> None:
        """Build the chamber and the manipulator, and keep the manipulator only when
        it says it is fitted. Then build any other device the configuration adds.

        ``pump``, ``vent`` and the manipulator's raw moves then go through the
        devices. The corrected and offset needle moves stay here and move through
        the device.
        """
        self._build_devices(
            [
                DeviceEntry(name="chamber", type="chamber"),
                DeviceEntry(name="manipulator", type="manipulator"),
            ],
            _PART_TYPES,
        )
        self._keep_if_fitted("manipulator")

        # whatever else the configuration adds, such as a device on its own PC
        self._build_devices([], exclude_types=_OWN_TYPES)

    def _build_sample_handling(self) -> None:
        """The sample handling service, after the ``sample_loader`` device on a
        compustage (the autoloader stage)."""
        stage = self.stage
        if stage is not None and stage.has_builtin_shuttle():
            self._build_sample_loader()
        super()._build_sample_handling()

    def _build_sample_loader(self) -> Optional[Any]:
        """The ``sample_loader`` device, built when the instrument has an autoloader,
        unless the configuration switches it off; None without one.

        A compustage without one gets no loader at all: its grids are exchanged by
        hand, and a phantom twelve-slot magazine would only mislead. The magazine is
        not read here; ``get_inventory`` or ``run_inventory`` reads it.
        """
        from fibsem.drivers.autoscript.devices import autoloader_installed

        device = self.devices.get("sample_loader")
        if device is None:
            fitted = []
            if autoloader_installed(self):
                fitted.append(DeviceEntry(name="sample_loader", type="sample_loader"))
            built = self._build_devices(fitted, _LOADER_TYPES)
            device = next(iter(built.values()), None)
        if device is None:
            logging.info("No sample loader: grids are exchanged by hand.")
        return device

    def get_detector_settings(
        self, beam_type: BeamType = BeamType.ELECTRON
    ) -> FibsemDetectorSettings:
        """The four detector reads under one hold of the imaging channel, so they
        describe one detector and claim the channel once against other callers
        (FIB-544). The lock is re-entrant, so the reads inside take it freely."""
        with self._threading_lock:
            return super().get_detector_settings(beam_type)

    def set_channel(self, channel: BeamType) -> None:
        """
        Set the active channel for the microscope.

        Args:
            channel (BeamType): The beam type to set as the active channel.
        """
        self._connection.set_channel(channel)

    def acquire_chamber_image(self) -> FibsemImage:
        """Acquire an image of the chamber inside."""
        # The chamber camera is a third device on the channel the beams and the FM
        # share, so a glance at it takes the microscope away from whatever had it.
        # Captured and put back in a `finally`: leaving the connection on the chamber
        # camera strands whoever was mid-operation, which is FIB-517 with a different
        # thief (FIB-545). Restores the view alone -- `set_active_device` changes the
        # device *in the active view*, so the device comes back with it.
        imaging = self._connection.sdk.imaging
        with self._threading_lock:
            restore_view = imaging.get_active_view()
            imaging.set_active_view(4)
            imaging.set_active_device(3)
            try:
                image = imaging.get_image()
            finally:
                imaging.set_active_view(restore_view)
        logging.debug({"msg": "acquire_chamber_image"})
        return FibsemImage(data=image.data, metadata=None)

    @_records_beam_shift
    def beam_shift(
        self, dx: float, dy: float, beam_type: BeamType = BeamType.ION
    ) -> Point:
        """
        Adjusts the beam shift based on relative values that are provided.

        Args:
            dx: the relative x term
            dy: the relative y term
            beam_type: the beam to shift
        Return:
            Point: the current beam shift of the requested beam_type, as this can now be clipped.
        """
        # beam shift limits
        beam = self._connection.beam(beam_type)
        limits: Limits2d = beam.beam_shift.limits

        # check if requested shift is outside limits
        current_shift = self.get_beam_shift(beam_type=beam_type)
        new_shift = Point(x=current_shift.x + dx, y=current_shift.y + dy)
        if new_shift.x < limits.limits_x.min or new_shift.x > limits.limits_x.max:
            logging.warning(
                f"Beam shift x value {new_shift.x} is out of bounds: {limits.limits_x}"
            )
        if new_shift.y < limits.limits_y.min or new_shift.y > limits.limits_y.max:
            logging.warning(
                f"Beam shift y value {new_shift.y} is out of bounds: {limits.limits_y}"
            )

        # clip the requested shift to the limits
        new_shift.x = np.clip(new_shift.x, limits.limits_x.min, limits.limits_x.max)
        new_shift.y = np.clip(new_shift.y, limits.limits_y.min, limits.limits_y.max)
        self.set_beam_shift(shift=new_shift, beam_type=beam_type)

        logging.debug(
            {"msg": "beam_shift", "dx": dx, "dy": dy, "beam_type": beam_type.name}
        )

        return self.get_beam_shift(beam_type=beam_type)

    def move_coincident_from_sem(self, dx: float, dy: float) -> FibsemStagePosition:
        """Correct coincident point from SEM to FIB stage position.

        Deprecated: call ``vertical_move(dy, dx, beam_type=BeamType.ELECTRON)``.
        Kept for one release because custom scripts may call it.
        """
        return self.vertical_move(dy=dy, dx=dx, beam_type=BeamType.ELECTRON)

    def _x_corrected_needle_movement(
        self, expected_x: float
    ) -> FibsemManipulatorPosition:
        """Calculate the corrected needle movement to move in the x-axis.

        Args:
            expected_x (float): distance along the x-axis (image coordinates)
        Returns:
            FibsemManipulatorPosition: x-corrected needle movement (relative position)
        """
        return FibsemManipulatorPosition(x=expected_x, y=0, z=0)  # no adjustment needed

    def _y_corrected_needle_movement(
        self, expected_y: float, stage_tilt: float
    ) -> FibsemManipulatorPosition:
        """Calculate the corrected needle movement to move in the y-axis.

        Args:
            expected_y (float): distance along the y-axis (image coordinates)
            stage_tilt (float, optional): stage tilt.

        Returns:
            FibsemManipulatorPosition: y-corrected needle movement (relative position)
        """
        y_move = +np.cos(stage_tilt) * expected_y
        z_move = +np.sin(stage_tilt) * expected_y
        return FibsemManipulatorPosition(x=0, y=y_move, z=z_move)

    def _z_corrected_needle_movement(
        self, expected_z: float, stage_tilt: float
    ) -> FibsemManipulatorPosition:
        """Calculate the corrected needle movement to move in the z-axis.

        Args:
            expected_z (float): distance along the z-axis (image coordinates)
            stage_tilt (float, optional): stage tilt.

        Returns:
            FibsemManipulatorPosition: z-corrected needle movement (relative position)
        """
        y_move = -np.sin(stage_tilt) * expected_z
        z_move = +np.cos(stage_tilt) * expected_z
        return FibsemManipulatorPosition(x=0, y=y_move, z=z_move)

    def move_manipulator_corrected(
        self,
        dx: float = 0,
        dy: float = 0,
        beam_type: BeamType = BeamType.ELECTRON,
    ) -> FibsemManipulatorPosition:
        """Calculate the required corrected needle movements based on the BeamType to move in the desired image coordinates.
        Then move the needle relatively. Manipulator movement axis is based on stage tilt, so we need to adjust for that
        with corrected movements, depending on the stage tilt and imaging perspective.

        BeamType.ELECTRON:  move in x, y (raw coordinates)
        BeamType.ION:       move in x, z (raw coordinates)

        Args:
            microscope (FibsemMicroscope)
            dx (float): distance along the x-axis (image coordinates)
            dy (float): distance along the y-axis (image corodinates)
            beam_type (BeamType, optional): the beam type to move in. Defaults to BeamType.ELECTRON.
        """
        if self.manipulator_device is None:
            raise self._unsupported("move_manipulator_corrected")
        stage_tilt = self.get_stage_position().t

        # xy
        if beam_type is BeamType.ELECTRON:
            x_move = self._x_corrected_needle_movement(expected_x=dx)
            yz_move = self._y_corrected_needle_movement(dy, stage_tilt=stage_tilt)

        # xz,
        if beam_type is BeamType.ION:
            x_move = self._x_corrected_needle_movement(expected_x=dx)
            yz_move = self._z_corrected_needle_movement(
                expected_z=dy, stage_tilt=stage_tilt
            )

        # explicitly set the coordinate system
        self._connection.sdk.specimen.manipulator.set_default_coordinate_system(
            ManipulatorCoordinateSystem.STAGE
        )
        manipulator_position = FibsemManipulatorPosition(
            x=x_move.x, y=yz_move.y, z=yz_move.z, r=0.0, coordinate_system="STAGE"
        )

        # move manipulator
        return self.move_manipulator_relative(manipulator_position)

    def move_manipulator_to_position_offset(
        self, offset: FibsemManipulatorPosition, name: str = None
    ) -> FibsemManipulatorPosition:
        """Move the manipulator to the specified coordinates, offset by the provided offset."""
        saved_position = self._get_saved_manipulator_position(name)

        # calculate corrected manipulator movement
        stage_tilt = self.get_stage_position().t
        yz_move = self._z_corrected_needle_movement(offset.z, stage_tilt)

        # adjust for offset
        saved_position.x += offset.x
        saved_position.y += yz_move.y + offset.y
        saved_position.z += yz_move.z  # RAW, up = negative, STAGE: down = negative
        saved_position.r = None  # rotation is not supported

        logging.debug(
            {
                "msg": "move_manipulator_to_position_offset",
                "name": name,
                "offset": offset.to_dict(),
                "saved_position": saved_position.to_dict(),
            }
        )

        # move manipulator absolute
        return self.move_manipulator_absolute(saved_position)

    manipulator_move_types = ("relative", "corrected")

    @property
    def rotation_centre(self) -> Optional[Tuple[float, float]]:
        """Where a half turn is centred, raw (x, y) in metres (FIB-655).

        xT's compucentric centre, read at connect, plus the instrument's calibrated
        `rotation_centre_correction`. Stage moves and image reprojection both use it,
        so a position reached by rotating and one drawn on the other side's image agree.
        """
        if self._compucentric_offset is None:
            return None
        ox, oy = self._compucentric_offset
        cx, cy = self.system.stage.rotation_centre_correction
        return (-ox + cx, -oy + cy)
