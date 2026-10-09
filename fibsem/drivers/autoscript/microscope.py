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
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
from packaging.version import InvalidVersion, Version
from packaging.version import parse as parse_version

from fibsem import manufacturers
from fibsem.devices.beam import BEAM_ROUTES, STAGE_COMMAND_ROUTES, STAGE_ROUTES
from fibsem.devices.chamber import CHAMBER_COMMAND_ROUTES, CHAMBER_ROUTES
from fibsem.devices.entries import build_device_entries, resolve_system_devices
from fibsem.devices.manipulator import MANIPULATOR_ROUTES
from fibsem.microscope import (
    FibsemMicroscope,
    RequiredDeviceUnavailable,
    _records_beam_shift,
)
from fibsem.microscopes._stage import (
    GridExchangeError,
    GridSlot,
    SampleGrid,
    SampleGridLoader,
    _slot_name,
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
    ImageSettings,
    MicroscopeState,
    Point,
    RangeLimit,
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


def image_settings_from_adorned_image(
    image: "AdornedImage",
    beam_type: Optional["BeamType"] = None,
) -> "ImageSettings":
    """Create ImageSettings from an AutoScript AdornedImage.

    Args:
        image: AutoScript AdornedImage.
        beam_type: Beam type for the image settings.

    Returns:
        ImageSettings: Converted image settings.

    Raises:
        ImportError: If AutoScript libraries are not available.
    """
    if not THERMO_API_AVAILABLE:
        raise ImportError(
            "AutoScript libraries not available. Cannot convert from AdornedImage."
        )

    from fibsem.structures import BeamType, ImageSettings
    from fibsem.utils import current_timestamp

    if beam_type is None:
        beam_type = BeamType.ELECTRON

    return ImageSettings(
        resolution=(image.width, image.height),
        dwell_time=image.metadata.scan_settings.dwell_time,
        hfw=image.width * image.metadata.binary_result.pixel_size.x,
        autocontrast=True,
        beam_type=beam_type,
        save=False,
        path="path",
        filename=current_timestamp(),
        reduced_area=None,
    )


def fibsem_image_from_adorned_image(
    adorned: "AdornedImage",
    image_settings: Optional["ImageSettings"] = None,
    state: Optional["MicroscopeState"] = None,
    beam_type: Optional["BeamType"] = None,
) -> "FibsemImage":
    """Create a FibsemImage from an AutoScript AdornedImage.

    Args:
        adorned: AutoScript AdornedImage.
        image_settings: Image settings. Defaults to None (derived from adorned).
        state: Microscope state. Defaults to None (derived from adorned).
        beam_type: Beam type for the image. Defaults to BeamType.ELECTRON.

    Returns:
        FibsemImage: Converted image.

    Raises:
        ImportError: If AutoScript libraries are not available.
    """
    if not THERMO_API_AVAILABLE:
        raise ImportError(
            "AutoScript libraries not available. Cannot convert from AdornedImage."
        )

    from fibsem.structures import (
        BeamSettings,
        BeamType,
        FibsemImage,
        FibsemImageMetadata,
        FibsemStagePosition,
        MicroscopeState,
        Point,
    )

    if beam_type is None:
        beam_type = BeamType.ELECTRON

    # AutoScript's acquisition_datetime is the instrument PC's clock time with no zone
    # (`07/16/2026 11:07:27` in a saved file). Read as this machine's local time --
    # the instrument PC's, or one in the same building -- and recorded with its offset
    # (FIB-1190). A state passed in keeps its own timestamp, when it was read:
    # overwriting it with this string is what put a string in a float field (FIB-487).
    acquired = to_datetime(adorned.metadata.acquisition.acquisition_datetime)
    if acquired is not None:
        acquired = acquired.astimezone()

    if state is None:
        state = MicroscopeState(
            # Built from the frame's own metadata, so read when the frame was.
            timestamp=acquired.timestamp() if acquired is not None else time.time(),
            stage_position=FibsemStagePosition(
                adorned.metadata.stage_settings.stage_position.x,
                adorned.metadata.stage_settings.stage_position.y,
                adorned.metadata.stage_settings.stage_position.z,
                adorned.metadata.stage_settings.stage_position.r,
                adorned.metadata.stage_settings.stage_position.t,
            ),
            electron_beam=BeamSettings(beam_type=BeamType.ELECTRON),
            ion_beam=BeamSettings(beam_type=BeamType.ION),
        )

    if image_settings is None:
        image_settings = image_settings_from_adorned_image(adorned, beam_type)

    pixel_size = Point(
        adorned.metadata.binary_result.pixel_size.x,
        adorned.metadata.binary_result.pixel_size.y,
    )

    metadata = FibsemImageMetadata(
        image_settings=image_settings,
        pixel_size=pixel_size,
        microscope_state=state,
        acquisition_datetime=acquired,
    )
    return FibsemImage(data=adorned.data, metadata=metadata)


class AutoscriptSampleLoader(SampleGridLoader):
    """The AutoScript autoloader (Arctis, xT 28.x, AutoScript >= 4.10) as a grid loader.

    Not used: ``ThermoMicroscope`` builds the ``sample_loader`` device
    (``fibsem.drivers.autoscript.devices.AutoscriptSampleLoader``) and the grid model
    over it. Kept until the device has run on an Arctis, then deleted.

    Wraps ``connection.specimen.autoloader``. Magazine slots mirror ``get_slots()``
    and are addressed by the 1-based ``AutoloaderSlot.id``; ``load(id)`` blocks until
    the exchange is done and ``unload()`` takes nothing. Grid names live in each
    slot's ``sample_description``, read on inventory and written back on rename.

    Two things the hardware does that the in-memory model must absorb:

    - The home slot of the grid on the stage. From AutoScript 4.14 it reads
      ``Loaded`` (release notes 4.14.0, "Autoloader state change"), which says
      outright that the slot's grid is on the stage, so a fresh connect can put it
      back in both places. Up to 4.13 it read ``Empty``, and the only way to keep
      the grid in its home slot was memory: a rescan keeps it there while our
      working slot holds it. Both paths are kept, so either version reads
      "present, loaded". States are compared case-blind: they arrive as enum names.
    - ``get_slots(False)`` returns the autoloader's last-known states, which may
      all be ``Unknown`` before any scan; ``get_slots(True)`` runs a physical scan.
      ``get_inventory`` is the first, ``run_inventory`` the second, and the caller
      chooses: a read is instant, a scan is not.

    Confirmed from operator code: the ``get_slots(bool)`` shape, the state strings,
    ``load(id)`` / ``unload()``, and ``autoloader.stage`` reporting what is on the
    microscope. Not verified on hardware: whether ``sample_description`` is writable
    (a refusal is logged, not raised). The magazine is not queried on construction;
    call ``get_inventory()`` or ``run_inventory()``.
    """

    @property
    def _autoloader(self):
        return self.parent.connection.specimen.autoloader

    @property
    def exchange_seconds(self) -> float:
        """Measured on an Arctis (FIB-893, 2026-10-02): an unload and a load took
        about 3 minutes, a load alone about 2 (98 s on 2026-09-13). Every exchange
        is charged the full figure, a run's first load included, which errs long."""
        return 180.0

    @property
    def is_installed(self) -> bool:
        try:
            return bool(self._autoloader.is_installed)
        except Exception:  # noqa: BLE001 - device absent, or not ready
            return False

    # -- inventory -----------------------------------------------------------

    def _read_magazine(self) -> None:
        self._apply_hardware_slots(list(self._autoloader.get_slots(False)))

    def _scan_magazine(self) -> None:
        self._apply_hardware_slots(list(self._autoloader.get_slots(True)))

    def _apply_hardware_slots(self, hw_slots: list) -> None:
        # The rows as the hardware gave them, before any reading of ours: what a
        # bench session needs from the log when the Sample view shows every slot
        # as unknown or empty and the question is what AutoScript actually said.
        logging.info(
            "Autoloader slots: " + ", ".join(_describe_hw(hw, hw.id) for hw in hw_slots)
        )
        stage = getattr(self._autoloader, "stage", None)
        if stage is not None:
            logging.info(f"Autoloader stage: {_describe_hw(stage)}")
        if hw_slots:
            self.capacity = len(hw_slots)

        loaded = {s.loaded_grid.name for s in self.holder.occupied_slots}
        slots: dict = {}
        unknown: set = set()
        on_stage: Optional[SampleGrid] = None
        for hw in hw_slots:
            number = int(hw.id)
            name = _slot_name(number - 1)
            previous = self.slots.get(name)
            state = _slot_state(hw)
            grid: Optional[SampleGrid] = None
            if state in ("Occupied", "Loaded"):
                described = (getattr(hw, "sample_description", "") or "").strip()
                grid_name = described or f"Grid-{number:02d}"
                if previous is not None and previous.loaded_grid is not None:
                    if previous.loaded_grid.name == grid_name:
                        grid = previous.loaded_grid  # keep identity across scans
                if grid is None:
                    grid = SampleGrid(name=grid_name)
                if state == "Loaded":
                    on_stage = grid  # 4.14: its home slot, and it is on the stage
            elif (
                previous is not None
                and previous.loaded_grid is not None
                and previous.loaded_grid.name in loaded
            ):
                grid = (
                    previous.loaded_grid
                )  # <= 4.13: its home reads Empty while loaded
            elif state == "Unknown":
                logging.warning(f"Autoloader slot {number} has not been scanned.")
                unknown.add(name)
            slots[name] = GridSlot(name=name, index=number - 1, loaded_grid=grid)
        self.slots = slots
        self.unknown_slots = unknown

        working = self.working_slot
        if on_stage is not None:
            # The same object in both places, as a load through us leaves it.
            working.loaded_grid = on_stage
        # Something may be on the stage that we did not load: reflect the hardware.
        stage = getattr(self._autoloader, "stage", None)
        if stage is not None:
            if working.loaded_grid is None and _slot_state(stage) == "Occupied":
                described = (getattr(stage, "sample_description", "") or "").strip()
                working.loaded_grid = SampleGrid(name=described or "Grid-on-stage")
            elif working.loaded_grid is not None and _slot_state(stage) == "Empty":
                working.loaded_grid = None

    def _write_slot_description(self, slot: GridSlot) -> None:
        """Write the grid's name to its slot, and read it back.

        Every inventory read takes the name from the slot description, so a write
        that did not land would quietly undo a rename at the next read. Seen to
        stick on an Arctis (2026-10-02); read back anyway, and raise
        ``GridExchangeError`` when it did not, so the rename can say so.
        """
        description = slot.loaded_grid.name if slot.loaded_grid is not None else ""
        number = slot.index + 1
        try:
            hw = self._hardware_slot(number)
            if hw is not None:
                hw.sample_description = description
                hw = self._hardware_slot(number)
        except Exception as e:  # noqa: BLE001 - whatever AutoScript raised, as one error
            raise GridExchangeError(
                f"Could not write the autoloader slot description: {e}"
            ) from e
        if hw is None:
            raise GridExchangeError(f"Autoloader reported no slot {number} to name.")
        written = (getattr(hw, "sample_description", "") or "").strip()
        if written != description:
            raise GridExchangeError(
                f"Autoloader slot {number} reads back '{written}', not "
                f"'{description}': the name did not stick."
            )

    def _hardware_slot(self, number: int):
        """AutoScript's record of one magazine slot (1-based), freshly read."""
        for hw in self._autoloader.get_slots(False):
            if int(hw.id) == number:
                return hw
        return None

    # -- exchange ------------------------------------------------------------

    def _do_load(self, slot: GridSlot) -> None:
        try:
            self._autoloader.load(slot.index + 1)
        except Exception as e:
            raise GridExchangeError(
                f"Autoloader could not load {slot.name}: {e}"
            ) from e

    def _do_unload(self, working_slot: GridSlot) -> None:
        try:
            self._autoloader.unload()
        except Exception as e:
            raise GridExchangeError(f"Autoloader could not unload: {e}") from e


def _describe_hw(hw, label=None) -> str:
    """``id=State 'description'`` for a slot, ``State 'description'`` for the stage."""
    described = (getattr(hw, "sample_description", "") or "").strip()
    text = _slot_state(hw) + (f" '{described}'" if described else "")
    return f"{label}={text}" if label is not None else text


def _slot_state(hw_slot) -> str:
    """``AutoloaderSlot.state`` as a plain string; enum members stringify to it."""
    state = getattr(hw_slot, "state", "Unknown")
    text = str(state)
    text = text.rsplit(".", 1)[-1] if "." in text else text
    # Enum names arrive upper-case (``OCCUPIED``); one spelling for the comparisons.
    return text.capitalize()


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

        get(self, key:str, beam_type: BeamType = None):
            Returns the value of the specified key.

        set(self, key: str, value, beam_type: BeamType = None) -> None:
            Sets the value of the specified key.

    New methods:
        __init__(self):
            Initializes a new instance of the class.
    """

    vertical_move_views = (BeamType.ION, BeamType.ELECTRON)

    # What connect found: the compustage rather than the stage. The Thermo paths that
    # depend on the vendor stage read this; everything else asks the stage device.
    _compustage_installed: bool = False

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
        self.connection = SdbMicroscopeClient()

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

    def reconnect(self):
        """Attempt to reconnect to the microscope client."""
        if self.connection is None:
            raise ConnectionError("Please connect to the microscope first")

        self.disconnect()
        if self._port is None:
            self.connect_to_microscope(self.system.info.ip_address)
        else:
            self.connect_to_microscope(self.system.info.ip_address, port=self._port)

    def disconnect(self):
        """Disconnect from the microscope client."""
        if self.connection is None:
            logging.warning("Microscope client is not connected.")
            return

        self.connection.disconnect()
        del self.connection
        self.connection = None

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
        if self.connection is None:
            self.connection = SdbMicroscopeClient()

        logging.info(f"Microscope client connecting to [{ip_address}:{port}]")
        self.connection.connect(host=ip_address, port=port)
        self._port = port
        logging.info(f"Microscope client connected to [{ip_address}:{port}]")

        # system information
        self.system.info.model = self.connection.service.system.name
        self.system.info.serial_number = self.connection.service.system.serial_number
        self.system.info.hardware_version = self.connection.service.system.version
        self.system.info.software_version = (
            self.connection.service.autoscript.client.version
        )
        info = self.system.info
        logging.info(
            f"Microscope client connected to model {info.model} with serial number {info.serial_number} and software version {info.software_version}."
        )

        # autoscript information
        logging.info(
            f"Autoscript Client: {self.connection.service.autoscript.client.version}"
        )
        logging.info(
            f"Autoscript Server: {self.connection.service.autoscript.server.version}"
        )

        self._build_beams()
        self._build_milling()

        if reset_beam_shift:
            self.reset_beam_shifts()

        # assign stage
        if self.connection.specimen.compustage.is_installed:
            self._vendor_stage = self.connection.specimen.compustage
            self._compustage_installed = True
            self._default_stage_coordinate_system = CoordinateSystem.SPECIMEN
        elif self.connection.specimen.stage.is_installed:
            self._vendor_stage = self.connection.specimen.stage
            self._compustage_installed = False
            self._default_stage_coordinate_system = CoordinateSystem.RAW
        else:
            raise Exception(
                "No stage installed. Please check the microscope configuration."
            )

        # set default coordinate system
        self._vendor_stage.set_default_coordinate_system(
            self._default_stage_coordinate_system
        )
        self._compucentric_offset = (
            None if self._compustage_installed else self._read_compucentric_offset()
        )
        self._build_stage()
        # TODO: set default move settings, is this dependent on the stage type?
        if self.milling is not None:
            # AutoScript needs a valid application file before a pattern is made
            self.milling._set_application_file("Si", default=True)

        self._last_imaging_settings: ImageSettings = ImageSettings()
        self.milling_channel: BeamType = BeamType.ION

        from fibsem.fm.autoscript import DeviceThermoFisherFluorescenceMicroscope

        # Its own FM leaves the shared imaging channel on the FM, as building it
        # selects the channel; any other outcome leaves it on the electron beam.
        self.fm = self._build_fluorescence(manufacturers.THERMOFISHER)
        if not isinstance(self.fm, DeviceThermoFisherFluorescenceMicroscope):
            self.set_channel(BeamType.ELECTRON)

        self._apply_fluorescence_calibration()
        self._warn_on_fluorescence_geometry()

        try:
            self._create_sample_stage()
        except Exception as e:
            logging.warning(f"Could not create sample stage: {e}")

        # after the sample stage, which reads which subsystems are fitted
        self._build_parts()

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
            self.system, defaults, types, exclude_types, manufacturers.THERMOFISHER
        )
        built = build_device_entries(resolved, self)
        for name, device in built.items():
            self._set_device(name, device)
        return built

    def _build_beams(self) -> None:
        """Build the beam devices and route the beam keys that have moved to them.

        The scan-mode methods then use the beam's scan commands, and acquire_image,
        last_image, autocontrast and auto_focus its imaging commands. A disabled column
        gets no device: its keys read None, and its imaging raises.
        """
        self._build_devices(
            [
                DeviceEntry(name="electron", type="beam"),
                DeviceEntry(name="ion", type="beam"),
            ],
            _BEAM_TYPES,
        )
        self._beam_routes = MappingProxyType(dict(BEAM_ROUTES))

    def _build_milling(self) -> None:
        """Build the milling service over the beams; the milling methods then go to it
        (`FibsemMicroscope`). Without an ion beam there is none, and they raise."""
        from fibsem.drivers.autoscript.services import bind_autoscript_milling

        self.milling = bind_autoscript_milling(self)

    def _build_stage(self) -> None:
        """Build the stage device and route the stage keys to it.

        The moves, ``home`` and ``link_stage`` then go through the device. Its
        ``link`` command only links: a false ``stage_link`` does nothing, and a
        compustage has no ``linked`` (its ``stage_linked`` reads None).
        """
        self._build_devices([DeviceEntry(name="stage", type="stage")], _STAGE_TYPES)
        self._device_routes = MappingProxyType(
            {key: ("stage", name) for key, name in STAGE_ROUTES.items()}
        )
        self._command_routes = MappingProxyType(
            {key: ("stage", name) for key, name in STAGE_COMMAND_ROUTES.items()}
        )

    def _build_parts(self) -> None:
        """Build the chamber, and the manipulator when it is fitted, and route the
        chamber and manipulator keys to them. Then build any other device the
        configuration adds.

        ``pump``, ``vent`` and the manipulator's raw moves then go through the
        devices. The corrected and offset needle moves stay here and move through
        the device.
        """
        fitted = [DeviceEntry(name="chamber", type="chamber")]
        if self.is_available("manipulator"):
            fitted.append(DeviceEntry(name="manipulator", type="manipulator"))
        self._build_devices(fitted, _PART_TYPES)

        routes = dict(self._device_routes)
        routes.update(
            {key: ("chamber_device", name) for key, name in CHAMBER_ROUTES.items()}
        )
        if self.manipulator_device is not None:
            routes.update(
                {
                    key: ("manipulator_device", name)
                    for key, name in MANIPULATOR_ROUTES.items()
                }
            )
        self._device_routes = MappingProxyType(routes)
        commands = dict(self._command_routes)
        commands.update(
            {
                key: ("chamber_device", name)
                for key, name in CHAMBER_COMMAND_ROUTES.items()
            }
        )
        self._command_routes = MappingProxyType(commands)

        # whatever else the configuration adds, such as a device on its own PC
        self._build_devices([], exclude_types=_OWN_TYPES)

    def _create_grid_loader(self) -> Optional["SampleGridLoader"]:
        """The grid model over the ``sample_loader`` device, built when the
        instrument has an autoloader, unless the configuration switches it off.

        A compustage without one gets no loader at all: its grids are exchanged by
        hand, and a phantom twelve-slot magazine would only mislead. The magazine is
        not read here; ``get_inventory`` or ``run_inventory`` reads it.
        """
        from fibsem.drivers.autoscript.devices import autoloader_installed
        from fibsem.microscopes._stage import DeviceSampleLoader

        device = self.devices.get("sample_loader")
        if device is None:
            fitted = []
            if autoloader_installed(self):
                fitted.append(DeviceEntry(name="sample_loader", type="sample_loader"))
            built = self._build_devices(fitted, _LOADER_TYPES)
            device = next(iter(built.values()), None)
        if device is None:
            logging.info("No sample loader: grids are exchanged by hand.")
            return None
        return DeviceSampleLoader(self, device)

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
        # TODO: create mapping for the other channels/devices
        self.connection.imaging.set_active_view(channel.value)
        self.connection.imaging.set_active_device(channel.value)
        logging.debug(f"Set active channel to {channel.name}")

    def acquire_image(
        self,
        image_settings: Optional[ImageSettings] = None,
        beam_type: Optional[BeamType] = None,
    ) -> FibsemImage:
        """
        Acquire a new image with the specified settings.

            Args:
            image_settings (ImageSettings): The settings for the new image.
            beam_type (BeamType, optional): The beam type to use with current settings.
                Used only if image_settings is not provided.

        Returns:
            FibsemImage: A new FibsemImage object representing the acquired image.
        """
        # The beam's acquire command; a beam_type takes precedence and means the
        # current settings.
        if beam_type is not None:
            return self._beam_device(beam_type).acquire(None)
        if image_settings is None:
            raise ValueError(
                "Must provide image_settings to acquire a new image if beam_type is not specified."
            )
        return self._beam_device(image_settings.beam_type).acquire(image_settings)

    def last_image(self, beam_type: BeamType = BeamType.ELECTRON) -> FibsemImage:
        """
        Get the last previously acquired image.

        Args:
            beam_type (BeamType, optional): The imaging beam type of the last image.
                Defaults to BeamType.ELECTRON.

        Returns:
            FibsemImage: A new FibsemImage object representing the last acquired image.
        """
        return self._beam_device(beam_type).last_image()

    def acquire_chamber_image(self) -> FibsemImage:
        """Acquire an image of the chamber inside."""
        # The chamber camera is a third device on the channel the beams and the FM
        # share, so a glance at it takes the microscope away from whatever had it.
        # Captured and put back in a `finally`: leaving the connection on the chamber
        # camera strands whoever was mid-operation, which is FIB-517 with a different
        # thief (FIB-545). Restores the view alone -- `set_active_device` changes the
        # device *in the active view*, so the device comes back with it.
        with self._threading_lock:
            restore_view = self.connection.imaging.get_active_view()
            self.connection.imaging.set_active_view(4)
            self.connection.imaging.set_active_device(3)
            try:
                image = self.connection.imaging.get_image()
            finally:
                self.connection.imaging.set_active_view(restore_view)
        logging.debug({"msg": "acquire_chamber_image"})
        return FibsemImage(data=image.data, metadata=None)

    def _construct_image(
        self, adorned_image: AdornedImage, beam_type: BeamType
    ) -> FibsemImage:
        """Construct a FibsemImage from an AdornedImage and the current microscope state."""
        # get the required metadata, convert to FibsemImage
        state = self.get_microscope_state(beam_type=beam_type)
        image_settings = self.get_imaging_settings(beam_type=beam_type)

        image = fibsem_image_from_adorned_image(
            copy.deepcopy(adorned_image),
            copy.deepcopy(image_settings),
            copy.deepcopy(state),
        )

        self._set_additional_metadata(image)

        return image

    def autocontrast(
        self, beam_type: BeamType, reduced_area: FibsemRectangle = None
    ) -> None:
        """
        Automatically adjust the microscope image contrast for the specified beam type.

        Args:
            beam_type (BeamType) The imaging beam type for which to adjust the contrast.
        """
        self._beam_device(beam_type).autocontrast(reduced_area)

    def auto_focus(
        self, beam_type: BeamType, reduced_area: Optional[FibsemRectangle] = None
    ) -> None:
        """Automatically focus the specified beam type.

        Args:
            beam_type (BeamType): The imaging beam type for which to focus.
        """
        self._beam_device(beam_type).auto_focus(reduced_area)

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
        beam = self._get_beam(beam_type=beam_type)
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

    # ---- fitted subsystems, as AutoScript reports them --------------------

    #: For a probe that cannot say.
    DEFAULT_FITTED = {
        "manipulator": True,
    }

    def _probe_manipulator_installed(self) -> Optional[bool]:
        """`specimen.manipulator.is_installed` -- documented, read-only, a bool."""
        return bool(self.connection.specimen.manipulator.is_installed)

    def _probe_plasma_gas(self) -> Optional[str]:
        """`ion_beam.source.plasma_gas.value` -- the call `get("plasma_gas")` makes.

        Only a plasma source has one; on a Ga column the attribute is missing or the
        read raises, and either is "cannot say", which leaves the file's answer.
        """
        plasma_gas = getattr(self.connection.beams.ion_beam.source, "plasma_gas", None)
        if plasma_gas is None:
            return None
        return plasma_gas.value or None

    def _get_axis_limits(self) -> Dict[str, RangeLimit]:
        """Get the stage axis limits for x, y, z, t, r."""
        from fibsem.drivers.demo.simulator import (
            STAGE_LIMITS_COMPUSTAGE,
            STAGE_LIMITS_DEFAULT,
        )

        if self._compustage_installed:
            return STAGE_LIMITS_COMPUSTAGE

        if not hasattr(self._vendor_stage, "get_axis_limits"):
            return STAGE_LIMITS_DEFAULT

        limits: Dict[str, RangeLimit] = {}
        for axis in ["x", "y", "z", "t"]:
            axis_limit = self._vendor_stage.get_axis_limits(axis)
            # t is in radians -> degrees
            if axis == "t":
                limits[axis] = RangeLimit(
                    min=np.degrees(axis_limit.min), max=np.degrees(axis_limit.max)
                )
                continue

            limits[axis] = RangeLimit(
                min=axis_limit.min,
                max=axis_limit.max,
            )

        # special case for r (no specified limits, infinite rotation)
        if not self._compustage_installed:
            limits["r"] = RangeLimit(
                min=-360,
                max=360,
            )
        return limits

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
        self.connection.specimen.manipulator.set_default_coordinate_system(
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

    def _beam_device(self, beam_type: BeamType) -> Any:
        """The beam device for ``beam_type``; a column disabled in the config has none."""
        device = self.beams.get(beam_type)
        if device is None:
            raise ValueError(f"The {beam_type.name} beam is not enabled.")
        return device

    def _get_beam(self, beam_type: BeamType) -> Union["ElectronBeam", "IonBeam"]:
        """Get the beam connection api for the given beam type.
        Args:
            beam_type (BeamType): The type of beam to get (ELECTRON or ION).
        Returns:
            Union['ElectronBeam', 'IonBeam']: The autoscript beam connection object for the given beam type."""
        if beam_type is BeamType.ELECTRON:
            return self.connection.beams.electron_beam
        elif beam_type is BeamType.ION:
            return self.connection.beams.ion_beam
        else:
            raise ValueError(f"Unknown beam type: {beam_type}")

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

    def _read_compucentric_offset(self) -> Tuple[float, float]:
        """Specimen minus raw stage coordinates, (x, y) metres, as xT reports them."""
        try:
            self._vendor_stage.set_default_coordinate_system(CoordinateSystem.SPECIMEN)
            specimen = stage_position_from_autoscript(
                self._vendor_stage.current_position
            )
            self._vendor_stage.set_default_coordinate_system(CoordinateSystem.RAW)
            raw = stage_position_from_autoscript(self._vendor_stage.current_position)
        finally:
            self._vendor_stage.set_default_coordinate_system(
                self._default_stage_coordinate_system
            )
        offset = (specimen.x - raw.x, specimen.y - raw.y)
        logging.info(
            f"Compucentric rotation offset (specimen - raw): "
            f"x={offset[0] * 1e6:.2f} um, y={offset[1] * 1e6:.2f} um."
        )
        return offset
