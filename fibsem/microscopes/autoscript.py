"""AutoScript (ThermoFisher) microscope backend.

`ThermoMicroscope` and every AutoScript-specific helper live here, so the AutoScript
import shim below is the only place the SDK is loaded. The other backends that borrow
`ThermoMicroscope`'s stage geometry (the simulator, Odemis) call its methods unbound and
import them from here.
"""

from __future__ import annotations

import copy
import glob
import logging
import os
import sys
import time
from copy import deepcopy
from functools import wraps
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
from packaging.version import InvalidVersion, Version
from packaging.version import parse as parse_version
from skimage import transform

from fibsem.microscope import FibsemMicroscope
from fibsem.microscopes._stage import (
    GridExchangeError,
    GridSlot,
    SampleGrid,
    SampleGridLoader,
    SampleHolder,
    Stage,
    _slot_name,
)
from fibsem.milling.progress import MillingProgress, MillingProgressStatus
from fibsem.structures import (
    ACTIVE_MILLING_STATES,
    BeamType,
    CrossSectionPattern,
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemExperimentRef,
    FibsemGasInjectionSettings,
    FibsemImage,
    FibsemLineSettings,
    FibsemManipulatorPosition,
    FibsemMillingSettings,
    FibsemPolygonSettings,
    FibsemRectangle,
    FibsemRectangleSettings,
    FibsemStagePosition,
    FibsemUser,
    ImageSettings,
    MicroscopeState,
    MillingState,
    Point,
    RangeLimit,
    SystemSettings,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from fibsem.structures import TFibsemPatternSettings

THERMO_API_AVAILABLE = False
MINIMUM_AUTOSCRIPT_VERSION_4_7 = parse_version("4.7")
# Set when the guarded import below fails, so the connection error can say why.
THERMO_API_IMPORT_ERROR: Optional[str] = None
# Declared so importers can rely on the name; only meaningful once
# THERMO_API_AVAILABLE is True.
AUTOSCRIPT_VERSION: Optional[Version] = None

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
        logging.info("Overwriting autoscript version to 4.7, for Monash dev install")
        AUTOSCRIPT_VERSION = MINIMUM_AUTOSCRIPT_VERSION_4_7

    if AUTOSCRIPT_VERSION < MINIMUM_AUTOSCRIPT_VERSION_4_7:
        raise AutoScriptException(
            f"AutoScript {version} found. Please update your AutoScript version to 4.7 or higher."
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
        MultiChemInsertPosition,
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

    if state is None:
        state = MicroscopeState(
            timestamp=adorned.metadata.acquisition.acquisition_datetime,
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
    else:
        state.timestamp = adorned.metadata.acquisition.acquisition_datetime

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
    )
    return FibsemImage(data=adorned.data, metadata=metadata)


class AutoscriptManipulator:
    """Manipulator interface for AutoScript-based microscopes."""

    def __init__(
        self,
        parent: "ThermoMicroscope",
    ) -> None:
        self.parent = parent

    def __repr__(self) -> str:
        return f"<Manipulator: position={self.position}>"

    @property
    def position(self) -> FibsemManipulatorPosition:
        return self.parent.get_manipulator_position()

    def insert(self) -> None:
        """Insert the manipulator."""
        self.parent.insert_manipulator()

    def retract(self) -> None:
        """Retract the manipulator."""
        self.parent.retract_manipulator()

    def move_absolute(
        self, position: FibsemManipulatorPosition
    ) -> FibsemManipulatorPosition:
        pass

    def move_relative(
        self, position: FibsemManipulatorPosition
    ) -> FibsemManipulatorPosition:
        pass

    def move_corrected(
        self, dx: float, dy: float, beam_type: BeamType
    ) -> FibsemManipulatorPosition:
        pass


class AutoscriptStage(Stage):
    """Stage interface for AutoScript-based microscopes."""

    def __init__(
        self,
        parent: "ThermoMicroscope",
        holder: SampleHolder,
        loader: Optional["SampleGridLoader"] = None,
    ) -> None:
        super().__init__(parent, holder, loader)


class AutoscriptCompustage(Stage):
    """Compustage interface for AutoScript-based microscopes."""

    def __init__(
        self,
        parent: "ThermoMicroscope",
        holder: SampleHolder,
        loader: Optional["SampleGridLoader"] = None,
    ) -> None:
        super().__init__(parent, holder, loader)


class AutoscriptSampleLoader(SampleGridLoader):
    """The AutoScript autoloader (Arctis, xT 28.x, AutoScript >= 4.10) as a grid loader.

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
        description = slot.loaded_grid.name if slot.loaded_grid is not None else ""
        try:
            for hw in self._autoloader.get_slots(False):
                if int(hw.id) == slot.index + 1:
                    hw.sample_description = description
                    return
            logging.warning(f"Autoloader reported no slot {slot.index + 1} to name.")
        except Exception as e:  # noqa: BLE001 - not verified writable on hardware
            logging.warning(f"Could not write the autoloader slot description: {e}")

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


def _slot_state(hw_slot) -> str:
    """``AutoloaderSlot.state`` as a plain string; enum members stringify to it."""
    state = getattr(hw_slot, "state", "Unknown")
    text = str(state)
    text = text.rsplit(".", 1)[-1] if "." in text else text
    # Enum names arrive upper-case (``OCCUPIED``); one spelling for the comparisons.
    return text.capitalize()


class AutoscriptSputterCoater:
    pass


class AutoscriptGISPort:
    port_name: str = "Pt dep"
    zlimit: float = 4.0e-3  # RAW_COORDINATES

    def __init__(self, parent: "ThermoMicroscope"):
        self.parent = parent

        available_ports = self.parent.connection.gas.list_all_gis_ports()

        print(f"available gis ports: {available_ports}")
        self._port = self.parent.connection.gas.get_gis_port(self.port_name)

    def insert(self):

        self._run_safety_check()

        self._port.insert()

    def retract(self):
        self._port.retract()

    def _move_to_safe_gis_position(self):

        self.parent.move_stage_absolute(FibsemStagePosition(z=self.zlimit - 500e-6))

    def _run_safety_check(self):

        stage_position = self.parent.get_stage_position()
        if stage_position.z > self.zlimit:
            raise ValueError(
                f"Unable to insert gis at current z-position{stage_position.pretty}, {self.zlimit * 1e3}mm"
            )

    def open(self):
        self._port.open()

    def close(self):
        self._port.close()

    @property
    def temperature(self) -> float:
        return self._port.get_temperature()

    def turn_heater_on(self, target_temp: float = 300, timeout: float = 15):
        self._port.turn_heater_on(target_temp, timeout)

    def turn_heater_off(self):
        self._port.turn_heater_off()

    def run_deposition(self, duration: int) -> None:

        self.insert()

        # QUERY: acquire diagnostic sem image?

        self.open()

        remaining_time = duration
        while True:
            print(f"Depositing: {self.port_name} - {remaining_time}s")
            time.sleep(1)
            remaining_time -= 1

            if remaining_time <= 0:
                break

        self.close()
        self.retract()


def _thermo_application_file_wrapper_for_drawing_functions(
    patterning_function: Callable[["ThermoMicroscope", TFibsemPatternSettings], Any],
) -> Callable[["ThermoMicroscope", TFibsemPatternSettings], Any]:
    @wraps(patterning_function)
    def wrap(self: ThermoMicroscope, pattern_settings: TFibsemPatternSettings) -> Any:
        # Ensure the default is correctly set
        self.set_application_file(self.get_default_application_file())
        try:
            retval = patterning_function(self, pattern_settings)
        finally:
            # Ensure any changes inside patterning_function don't persist
            self.set_application_file(self.get_default_application_file())
        return retval

    return wrap


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

        run_milling(self, milling_current: float, asynch: bool = False):
            Run ion beam milling using the specified milling current.

        finish_milling(self, imaging_current: float):
            Finalises the milling process by clearing the microscope of any patterns and returning the current to the imaging current.

        setup_sputter(self, protocol: dict):
            Set up the sputter coating process on the microscope.

        draw_sputter_pattern(self, hfw: float, line_pattern_length: float, sputter_time: float):
            Draws a line pattern for sputtering with the given parameters.

        run_sputter(self, **kwargs):
            Runs the GIS Platinum Sputter.

        finish_sputter(self, application_file: str) -> None:
            Finish the sputter process by clearing patterns and resetting beam and imaging settings.

        set_microscope_state(self, microscope_state: MicroscopeState) -> None:
            Reset the microscope state to the provided state.

        get(self, key:str, beam_type: BeamType = None):
            Returns the value of the specified key.

        set(self, key: str, value, beam_type: BeamType = None) -> None:
            Sets the value of the specified key.

    New methods:
        __init__(self):
            Initializes a new instance of the class.

        _y_corrected_stage_movement(self, expected_y: float, beam_type: BeamType = BeamType.ELECTRON) -> FibsemStagePosition:
            Calculate the y corrected stage movement, corrected for the additional tilt of the sample holder (pre-tilt angle).
    """

    vertical_move_views = (BeamType.ION, BeamType.ELECTRON)

    def __init__(self, system_settings: SystemSettings):
        if not THERMO_API_AVAILABLE:
            raise Exception(autoscript_unavailable_message())

        # create microscope client
        self.connection = SdbMicroscopeClient()

        # initialise system settings
        self.system: SystemSettings = system_settings
        self._patterns: List = []

        # user, experiment metadata
        # TODO: remove once db integrated
        self.user = FibsemUser.from_environment()
        self.experiment = FibsemExperimentRef()
        self._default_application_file = "Si"
        self._current_application_file = self._default_application_file

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
        self.connect_to_microscope(self.system.info.ip_address)

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

        # TODO: get the port
        logging.info(f"Microscope client connecting to [{ip_address}:{port}]")
        self.connection.connect(host=ip_address, port=port)
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

        if reset_beam_shift:
            self.reset_beam_shifts()

        # assign stage
        if self.connection.specimen.compustage.is_installed:
            self.stage = self.connection.specimen.compustage
            self.stage_is_compustage = True
            self._default_stage_coordinate_system = CoordinateSystem.SPECIMEN
        elif self.connection.specimen.stage.is_installed:
            self.stage = self.connection.specimen.stage
            self.stage_is_compustage = False
            self._default_stage_coordinate_system = CoordinateSystem.RAW
        else:
            raise Exception(
                "No stage installed. Please check the microscope configuration."
            )

        # set default coordinate system
        self.stage.set_default_coordinate_system(self._default_stage_coordinate_system)
        # TODO: set default move settings, is this dependent on the stage type?
        self.set_application_file(self.get_default_application_file(), default=True)

        self._last_imaging_settings: ImageSettings = ImageSettings()
        self.milling_channel: BeamType = BeamType.ION

        try:
            if not self._fluorescence_is_configured():
                logging.info(
                    "No fluorescence microscope configured for this system. Set "
                    "`fm.enabled` in the microscope configuration to enable it."
                )
                self.fm = None
                self.set_channel(BeamType.ELECTRON)
            else:
                from fibsem.fm.autoscript import ThermoFisherFluorescenceMicroscope

                self.fm = ThermoFisherFluorescenceMicroscope(self, self.connection)
                self.fm.set_active_channel()  # this will fail if no fm available
                logging.info(
                    "Thermo Fisher Fluorescence Microscope initialized successfully."
                )
        except Exception as e:
            logging.error(
                f"Failed to initialize Thermo Fisher Fluorescence Microscope: {e}"
            )
            self.fm = None
            self.set_channel(BeamType.ELECTRON)

        self._warn_on_fluorescence_geometry()

        try:
            self._create_sample_stage()
        except Exception as e:
            logging.warning(f"Could not create sample stage: {e}")

    def _create_grid_loader(self) -> Optional["SampleGridLoader"]:
        """The AutoScript autoloader, when the microscope has one.

        A compustage without an autoloader gets no loader at all: its grids are
        exchanged by hand, and a phantom twelve-slot magazine would only mislead.
        """
        loader = AutoscriptSampleLoader(parent=self)
        if not loader.is_installed:
            logging.info(
                "Compustage without an autoloader: grids are exchanged by hand."
            )
            return None
        return loader

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
        if beam_type is not None:
            return self.acquire_image3(image_settings=None, beam_type=beam_type)

        if image_settings is None:
            raise ValueError(
                "Must provide image_settings to acquire a new image if beam_type is not specified."
            )

        # set reduced area settings
        if image_settings.reduced_area is not None:
            rect = image_settings.reduced_area
            reduced_area = Rectangle(rect.left, rect.top, rect.width, rect.height)
            logging.debug(
                f"Set reduced are: {reduced_area} for beam type {image_settings.beam_type}"
            )
        else:
            reduced_area = None
            self.set_full_frame_scanning_mode(image_settings.beam_type)

        # set the imaging hfw
        self.set_field_of_view(
            hfw=image_settings.hfw, beam_type=image_settings.beam_type
        )

        logging.info(f"acquiring new {image_settings.beam_type.name} image.")

        # set the imaging frame settings
        frame_settings = GrabFrameSettings(
            resolution=f"{image_settings.resolution[0]}x{image_settings.resolution[1]}",
            dwell_time=image_settings.dwell_time,
            reduced_area=reduced_area,
            line_integration=image_settings.line_integration,
            scan_interlacing=image_settings.scan_interlacing,
            frame_integration=image_settings.frame_integration,
            drift_correction=image_settings.drift_correction,
        )

        # One lock over both RPCs. `grab_frame` reads the active view's buffer, so a
        # channel that is not still ours when the grab lands returns whoever took it in
        # between -- silently, since the metadata below is built from `image_settings`
        # rather than from what came back. That is FIB-517 on the beam side (FIB-542),
        # and it is the discipline every other set-then-act pair here already keeps.
        #
        # Deliberately just the pair: `_threading_lock` is a class attribute shared by
        # every caller in the process, so holding it over the metadata reads or the
        # state fetch below would block all of them for the length of a frame.
        with self._threading_lock:
            self.set_channel(image_settings.beam_type)
            image = self.connection.imaging.grab_frame(frame_settings)

        # restore to full frame imaging
        if image_settings.reduced_area is not None:
            self.set_full_frame_scanning_mode(image_settings.beam_type)

        # get the microscope state (for metadata)
        # TODO: convert to using fromAdornedImage, we dont need to full state
        # we should just get the 'state' of the image beam, e.g. stage, beam, detector for electron
        # therefore we don't trigger the view to switch
        state = self.get_microscope_state(beam_type=image_settings.beam_type)

        fibsem_image = fibsem_image_from_adorned_image(
            copy.deepcopy(image),
            copy.deepcopy(image_settings),
            copy.deepcopy(state),
        )

        self._set_additional_metadata(fibsem_image)

        # store last imaging settings
        self._last_imaging_settings = image_settings

        logging.debug(
            {"msg": "acquire_image", "metadata": fibsem_image.metadata.to_dict()}
        )

        return fibsem_image

    def acquire_image3(
        self,
        image_settings: Optional[ImageSettings] = None,
        beam_type: Optional[BeamType] = None,
    ) -> FibsemImage:
        """
        Acquire a new image with the specified settings or current settings for the given beam type.

        Args:
            image_settings (ImageSettings, optional): The settings for the new image.
                Takes precedence if both parameters are provided.
            beam_type (BeamType, optional): The beam type to use with current settings.
                Used only if image_settings is not provided.

        Returns:
            FibsemImage: A new FibsemImage representing the acquired image.

        Raises:
            ValueError: If neither image_settings nor beam_type is provided.

        Examples:
            # Acquire with specific settings
            settings = ImageSettings(beam_type=BeamType.ELECTRON, hfw=1e-6, resolution=(1024, 1024))
            image = microscope.acquire_image3(image_settings=settings)

            # Acquire with current settings for a specific beam type
            image = microscope.acquire_image3(beam_type=BeamType.ION)

            # If both provided, image_settings takes precedence
            image = microscope.acquire_image3(image_settings=settings, beam_type=BeamType.ION)  # Uses settings
        """

        # Validate parameters - at least one must be provided
        if image_settings is None and beam_type is None:
            raise ValueError(
                "Must provide either image_settings (to acquire with specific settings) or beam_type (to acquire with current microscope settings for that beam type)."
            )

        if image_settings is not None:
            # Use provided image settings (takes precedence)
            effective_beam_type = image_settings.beam_type
            effective_image_settings = image_settings

            # apply specified image settings, create frame settings
            self._apply_image_settings(image_settings)
            frame_settings = self._create_frame_settings(image_settings)
        else:
            # Use current settings for the specified beam type
            effective_beam_type = beam_type
            effective_image_settings = self.get_imaging_settings(beam_type=beam_type)
            frame_settings = None

        logging.info(f"acquiring new {effective_beam_type.name} image.")

        # Locked for the same reason as `acquire_image`, and just as narrowly: this is
        # the path every `beam_type=`-only call takes, including the live worker's.
        with self._threading_lock:
            self.set_channel(effective_beam_type)
            adorned_image: AdornedImage = self.connection.imaging.grab_frame(
                frame_settings
            )

        # QUERY: is this required, reduced area is only set for the grab_frame?
        # Restore full frame if reduced area was used (same as acquire_image)
        if image_settings is not None and image_settings.reduced_area is not None:
            self.set_full_frame_scanning_mode(image_settings.beam_type)

        logging.info(f"acquiring new {effective_beam_type.name} image.")

        # Create FibsemImage with metadata (common for both paths)
        state = self.get_microscope_state(beam_type=effective_beam_type)
        fibsem_image = fibsem_image_from_adorned_image(
            copy.deepcopy(adorned_image),
            copy.deepcopy(effective_image_settings),
            copy.deepcopy(state),
        )

        # Set additional metadata
        self._set_additional_metadata(fibsem_image)

        # Store last imaging settings if image_settings was provided
        if image_settings is not None:
            self._last_imaging_settings = image_settings

        logging.debug(
            {"msg": "acquire_image", "metadata": fibsem_image.metadata.to_dict()}
        )

        return fibsem_image

    def _apply_image_settings(self, image_settings: ImageSettings) -> None:
        """Apply imaging settings to the microscope."""
        # Set reduced area or full frame
        if image_settings.reduced_area is not None:
            logging.debug(
                f"Set reduced area: {image_settings.reduced_area} for beam type {image_settings.beam_type}"
            )
        else:
            self.set_full_frame_scanning_mode(image_settings.beam_type)

        # Set the imaging hfw
        self.set_field_of_view(
            hfw=image_settings.hfw, beam_type=image_settings.beam_type
        )

    def _create_frame_settings(
        self, image_settings: ImageSettings
    ) -> "GrabFrameSettings":
        """Create GrabFrameSettings from ImageSettings."""
        reduced_area = None
        if image_settings.reduced_area is not None:
            rect = image_settings.reduced_area
            reduced_area = Rectangle(rect.left, rect.top, rect.width, rect.height)

        return GrabFrameSettings(
            resolution=f"{image_settings.resolution[0]}x{image_settings.resolution[1]}",
            dwell_time=image_settings.dwell_time,
            reduced_area=reduced_area,
            line_integration=image_settings.line_integration,
            scan_interlacing=image_settings.scan_interlacing,
            frame_integration=image_settings.frame_integration,
            drift_correction=image_settings.drift_correction,
        )

    def last_image(self, beam_type: BeamType = BeamType.ELECTRON) -> FibsemImage:
        """
        Get the last previously acquired image.

        Args:
            beam_type (BeamType, optional): The imaging beam type of the last image.
                Defaults to BeamType.ELECTRON.

        Returns:
            FibsemImage: A new FibsemImage object representing the last acquired image.

        Raises:
            Exception: If there's an error while getting the last image.
        """
        # One lock over the channel and the read, as `acquire_image` holds it over
        # `set_channel` + `grab_frame` (FIB-542). `get_image` retrieves the image "in the
        # active view", so this is the same pair on the retrieval path: a channel that is
        # not still ours when the read lands returns whoever took it, and the metadata
        # below is built from `beam_type` rather than from what came back, so it is
        # labelled as though nothing happened (FIB-569).
        with self._threading_lock:
            self.set_channel(beam_type)
            image = self.connection.imaging.get_image()
        image = AdornedImage(data=image.data.astype(np.uint8), metadata=image.metadata)

        # get the microscope state (for metadata)
        state = self.get_microscope_state(beam_type=beam_type)

        # create the fibsem image
        fibsem_image = fibsem_image_from_adorned_image(
            adorned=image,
            image_settings=None,
            state=state,
            beam_type=beam_type,
        )

        self._set_additional_metadata(fibsem_image)

        logging.debug(
            {"msg": "acquire_image", "metadata": fibsem_image.metadata.to_dict()}
        )

        return fibsem_image

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

    def _acquisition_worker(self, beam_type: BeamType):
        """Worker thread for image acquisition."""
        # TODO: add lock
        self.set_channel(channel=beam_type)

        try:
            while True:
                if self._stop_acquisition_event.is_set():
                    break

                # fast continuous acquisition
                USE_FAST_ACQUISITION = True
                if USE_FAST_ACQUISITION:
                    self._fast_acquisition_worker(beam_type=beam_type)
                    if self._stop_acquisition_event.is_set():
                        break

                # acquire image using current beam settings # TODO: migrate to start_acquisition while loop
                image = self.acquire_image(beam_type=beam_type, image_settings=None)

                # emit the acquired image
                if beam_type is BeamType.ELECTRON:
                    self.sem_acquisition_signal.emit(image)
                if beam_type is BeamType.ION:
                    self.fib_acquisition_signal.emit(image)

        except Exception as e:
            logging.error(f"Error in acquisition worker: {e}")

    def _fast_acquisition_worker(self, beam_type: BeamType):
        try:
            with self._threading_lock:
                self.set_channel(channel=beam_type)  # re-force active channel...?
                self.connection.imaging.start_acquisition()

            while self.connection.imaging.state == ImagingState.ACQUIRING:
                if self._stop_acquisition_event.is_set():
                    self.connection.imaging.stop_acquisition()
                    break
                with self._threading_lock:
                    self.set_channel(channel=beam_type)  # re-force active channel...?
                    adorned_image = self.connection.imaging.get_image(
                        GetImageSettings(wait_for_frame=True)
                    )
                    image = self._construct_image(adorned_image, beam_type=beam_type)

                    logging.info(f"Acquired Image: {image.data.shape}")
                    # emit the acquired image
                    if beam_type is BeamType.ELECTRON:
                        self.sem_acquisition_signal.emit(image)
                    if beam_type is BeamType.ION:
                        self.fib_acquisition_signal.emit(image)
        except Exception as e:
            logging.error(f"Exception occurred during fast acquisition: {e}")
        finally:
            self.connection.imaging.stop_acquisition()

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
        logging.debug(f"Running autocontrast on {beam_type.name}.")
        # `run_auto_cb` optimises "the active detector in the active view", so the
        # channel has to be ours for the whole routine, not just when it starts. Unlike
        # a stolen grab this is a *write*: a routine that runs on the wrong view tunes
        # the other column's brightness and contrast and leaves it that way (FIB-569).
        #
        # The reduced-area write is inside for the same reason -- outside it, the
        # routine could run on the right view with someone else's scan region.
        #
        # A longer hold than FIB-542's single grab, and deliberately so: the operation
        # itself is what needs the channel, so there is no narrower correct scope. It is
        # seconds, and anything wanting the microscope during an autocontrast conflicts
        # with it physically in any case.
        with self._threading_lock:
            self.set_channel(beam_type)
            if reduced_area is not None:
                self.set_reduced_area_scanning_mode(reduced_area, beam_type)

            self.connection.auto_functions.run_auto_cb()
        if reduced_area is not None:
            self.set_full_frame_scanning_mode(beam_type)

        logging.debug({"msg": "autocontrast", "beam_type": beam_type.name})

    def auto_focus(
        self, beam_type: BeamType, reduced_area: Optional[FibsemRectangle] = None
    ) -> None:
        """Automatically focus the specified beam type.

        Args:
            beam_type (BeamType): The imaging beam type for which to focus.
        """
        logging.debug(f"Running auto-focus on {beam_type.name}.")
        # Held for the same reason as `autocontrast`, and it matters more here:
        # `run_auto_focus` runs "in the active view", and `imaging/tiled.py` calls this
        # once per tile of an unattended tileset -- exactly the long run interleaved
        # with GUI-thread reads that stopped a workflow task in FIB-517. Losing the
        # channel focuses the other column and every later acquisition inherits it.
        with self._threading_lock:
            self.set_channel(beam_type)
            if reduced_area is not None:
                self.set_reduced_area_scanning_mode(reduced_area, beam_type)

            # run the auto focus
            self.connection.auto_functions.run_auto_focus()

        # restore the full frame scanning mode
        if reduced_area is not None:
            self.set_full_frame_scanning_mode(beam_type)
        logging.debug({"msg": "auto_focus", "beam_type": beam_type.name})

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

    def move_stage_absolute(self, position: FibsemStagePosition) -> FibsemStagePosition:
        """
        Move the stage to the specified coordinates.

        Args:
            position: The raw stage position to move to.

        Returns:
            FibsemStagePosition: The stage position after movement.
        """

        # get current working distance, to be restored later
        wd = self.get_working_distance(BeamType.ELECTRON)

        # convert to autoscript position
        autoscript_position = stage_position_to_autoscript(
            position, compustage=self.stage_is_compustage
        )  # TODO: apply compucentric/raw coordinate offset here?

        if self._axis_restrictions_apply():  # ONLY when restrictions are on
            autoscript_position.z = None
            autoscript_position.r = None

        logging.info(f"Moving stage to {position}.")
        self.stage.absolute_move(
            autoscript_position, MoveSettings(rotate_compucentric=True)
        )  # TODO: This needs at least an optional safe move to prevent collision?

        # restore working distance to adjust for microscope compenstation
        if not self.stage_is_compustage:
            self.set_working_distance(wd, BeamType.ELECTRON)

        logging.debug({"msg": "move_stage_absolute", "position": position.to_dict()})

        return self.get_stage_position()

    def move_stage_relative(self, position: FibsemStagePosition) -> FibsemStagePosition:
        """
        Move the stage by the specified relative move.

        Args:
            position: the relative stage position to move by.
        """

        logging.info(f"Moving stage by {position}.")

        # convert to autoscript position
        thermo_position = stage_position_to_autoscript(
            position, self.stage_is_compustage
        )

        # move stage
        self.stage.relative_move(thermo_position)

        logging.debug({"msg": "move_stage_relative", "position": position.to_dict()})

        return self.get_stage_position()

    # TODO: migrate from stable_move vocab to sample_stage
    def stable_move(
        self, dx: float, dy: float, beam_type: BeamType, static_wd: bool = False
    ) -> FibsemStagePosition:
        """
        Calculate the corrected stage movements based on the beam_type stage tilt, shuttle pre-tilt,
        and then move the stage relatively.

        Args:
            dx (float): distance along the x-axis (image coordinates)
            dy (float): distance along the y-axis (image coordinates)
            beam_type (BeamType): beam type to move in
            static_wd (bool, optional): whether to fix the working distance to the eucentric heights. Defaults to False.
        """

        wd = self.get_working_distance(beam_type=BeamType.ELECTRON)

        scan_rotation = self.get_scan_rotation(beam_type=beam_type)
        if np.isclose(scan_rotation, np.pi):
            dx *= -1.0
            dy *= -1.0

        # calculate stable movement
        yz_move = self._y_corrected_stage_movement(
            expected_y=dy,
            beam_type=beam_type,
        )
        stage_position = FibsemStagePosition(
            x=dx, y=yz_move.y, z=yz_move.z, r=0, t=0, coordinate_system="RAW"
        )

        # move stage
        self.move_stage_relative(stage_position)

        # adjust working distance to compensate for stage movement
        if static_wd:
            wd = self.system.electron.eucentric_height

        if not self.stage_is_compustage:  # TODO: can replace with self.stage.is_linked
            self.set_working_distance(wd, BeamType.ELECTRON)

        # logging
        logging.debug(
            {
                "msg": "stable_move",
                "dx": dx,
                "dy": dy,
                "beam_type": beam_type.name,
                "static_wd": static_wd,
                "working_distance": wd,
                "scan_rotation": scan_rotation,
                "position": stage_position.to_dict(),
            }
        )

        return self.get_stage_position()

    def vertical_move(
        self,
        dy: float,
        dx: float = 0.0,
        beam_type: BeamType = BeamType.ION,
        relaxation: float = 1.0,
    ) -> FibsemStagePosition:
        """Restore the coincidence point from an offset measured in one beam view.

        Args:
            dy (float): distance along the y-axis (image coordinates)
            dx (float, optional): distance along the x-axis (image coordinates). Defaults to 0.0.
            beam_type (BeamType, optional): the view the offset was measured in.
                Defaults to ION.
            relaxation (float, optional): under-relaxation of the correction.
                1.0 applies the geometrically exact height change; a value
                below 1.0 deliberately undershoots, which keeps a manual
                look-correct-look loop convergent where a slight model error
                would otherwise make it oscillate. This replaces the old
                hard-coded 0.9, which was found to be absorbing a
                decomposition error rather than correcting perspective
                (FIB-773).
        """
        self._check_vertical_move_supported(beam_type)
        if beam_type is BeamType.ELECTRON:
            return self._vertical_move_from_sem(dx=dx, dy=dy, relaxation=relaxation)
        return self._vertical_move_from_fib(dx=dx, dy=dy, relaxation=relaxation)

    def _vertical_move_from_fib(
        self,
        dy: float,
        dx: float = 0.0,
        relaxation: float = 1.0,
    ) -> FibsemStagePosition:
        """Move the stage vertically to correct coincidence point

        The offset is measured in the FIB view: the feature is already centred in
        the SEM, and a chamber-vertical move is invisible to the electron beam.

        Args:
            dy (float): distance along the y-axis (image coordinates)
            dx (float, optional): distance along the x-axis (image coordinates). Defaults to 0.0.
        """

        # get current working distance, to be restored later
        wd = self.get_working_distance(beam_type=BeamType.ELECTRON)

        # adjust for scan rotation
        scan_rotation = self.get_scan_rotation(beam_type=BeamType.ION)
        if np.isclose(scan_rotation, np.pi):
            dx *= -1.0
            dy *= -1.0

        # TODO: ARCTIS Do we need to reverse the direction of the movement because of the inverted stage tilt?
        if self.stage_is_compustage:
            dy *= -1.0
            stage_tilt = self.get_stage_position().t
            if stage_tilt >= np.deg2rad(-90):
                dy *= -1.0

        # a chamber-vertical displacement of dz appears in the FIB view as
        # dz * sin(column_tilt), independent of stage tilt - so the height
        # change that cancels an observed dy is dy / sin(column_tilt)
        z_move = dy / np.sin(np.deg2rad(self.system.ion.column_tilt)) * relaxation

        # decompose the chamber-vertical into the tilted stage axes:
        # y = m*sin(t), z = m*cos(t). The old form used z = m/cos(t), which
        # is only vertical at t=0 - at every tilted pose it dragged the
        # feature sideways in the SEM view, partially undoing the SEM
        # centring this operation exists to preserve (FIB-773).
        theta = self.get_stage_position().t  # rad
        dy = z_move * np.sin(theta)
        dz = z_move * np.cos(theta)
        stage_position = FibsemStagePosition(x=dx, y=dy, z=dz, coordinate_system="RAW")
        logging.info(f"Vertical movement: {stage_position}")
        self.move_stage_relative(
            stage_position
        )  # NOTE: this seems to be a bit less than previous... -> perspective correction?

        # Vertical moves re-establish the coincidence plane. Always restore the
        # pre-move SEM (electron) working distance so fine corrections keep their
        # focus. For a large correction, snap the FIB (ion) WD to eucentric (the
        # best estimate at the new coincidence plane); small corrections keep the
        # current FIB focus.
        EUCENTRIC_RESET_THRESHOLD = 100e-6  # m (stage-z travel)
        self.set_working_distance(wd=wd, beam_type=BeamType.ELECTRON)
        if abs(dz) > EUCENTRIC_RESET_THRESHOLD:
            self.set_working_distance(
                wd=self.system.ion.eucentric_height, beam_type=BeamType.ION
            )

        # logging
        logging.debug(
            {
                "msg": "vertical_move",
                "dy": dy,
                "dx": dx,
                "wd": wd,
                "scan_rotation": scan_rotation,
                "position": stage_position.to_dict(),
            }
        )

        return self.get_stage_position()

    def move_coincident_from_sem(self, dx: float, dy: float) -> FibsemStagePosition:
        """Correct coincident point from SEM to FIB stage position.

        Deprecated: call ``vertical_move(dy, dx, beam_type=BeamType.ELECTRON)``.
        Kept for one release because custom scripts may call it.
        """
        return self.vertical_move(dy=dy, dx=dx, beam_type=BeamType.ELECTRON)

    def _vertical_move_from_sem(
        self, dx: float, dy: float, relaxation: float = 1.0
    ) -> FibsemStagePosition:
        """Correct the coincidence point from an offset measured in the SEM view.

        Not the mirror image of the FIB path but a superset: a stable move first
        brings the feature to the centre of the SEM, and the height correction
        that follows puts the FIB back.
        """

        # NOTE:
        # inaccurate over longer distances, but works for small movements
        # less accurate for higher tilt angles

        # move to position in SEM
        base_position = self.get_stage_position()
        self.stable_move(dx=dx, dy=dy, beam_type=BeamType.ELECTRON)

        # calculate the difference in position after SEM move
        position_after_sem_move = self.get_stage_position()
        dy = position_after_sem_move.y - base_position.y
        dz = position_after_sem_move.z - base_position.z

        # correct for the stage tilt and milling angle
        if self.get_stage_orientation() in ["SEM", "MILLING"]:
            theta = np.radians(self.get_current_milling_angle())  # deg
            dy = dy * np.sin(theta)

        # NOTE: vertical move also corrects for scan rotation, so we need to adjust dy accordingly
        # if the scan rotation is 0, we need to invert the dy value
        scan_rotation = self.get_scan_rotation(beam_type=BeamType.ION)
        if np.isclose(scan_rotation, 0):
            dy *= -1.0

        # apply the vertical move to correct the position. (The old 1.11
        # here was 1/0.9: it existed to cancel the magic constant inside
        # vertical_move from the outside, and is gone with it - FIB-773.)
        self._vertical_move_from_fib(dx=0, dy=dy, relaxation=relaxation)

        return self.get_stage_position()

    def _y_corrected_stage_movement(
        self,
        expected_y: float,
        beam_type: BeamType = BeamType.ELECTRON,
    ) -> FibsemStagePosition:
        """
        Calculate the y corrected stage movement, corrected for the additional tilt of the sample holder (pre-tilt angle).

        Thin wrapper over :meth:`_view_corrected_stage_movement`: a beam is just a
        view whose axis is tilted from the electron column by its ``column_tilt``.

        Args:
            expected_y (float, optional): distance along y-axis.
            beam_type (BeamType, optional): beam_type to move in. Defaults to BeamType.ELECTRON.

        Returns:
            StagePosition: y corrected stage movement (relative position)
        """
        return self._view_corrected_stage_movement(
            expected_y=expected_y,
            view_tilt=self._beam_view_tilt(beam_type),
        )

    def _inverse_y_corrected_stage_movement(
        self,
        dy: float,
        dz: float,
        beam_type: BeamType = BeamType.ELECTRON,
    ) -> float:
        """
        Calculate the expected_y input from dy, dz stage movements and beam_type.
        This is the inverse of _y_corrected_stage_movement.

        Thin wrapper over :meth:`_inverse_view_corrected_stage_movement`.

        Args:
            dy (float): actual y stage movement
            dz (float): actual z stage movement
            beam_type (BeamType, optional): beam_type used. Defaults to BeamType.ELECTRON.

        Returns:
            float: expected_y input that would produce the given dy, dz movements
        """
        return self._inverse_view_corrected_stage_movement(
            dy=dy,
            dz=dz,
            view_tilt=self._beam_view_tilt(beam_type),
        )

    def _get_axis_limits(self) -> Dict[str, RangeLimit]:
        """Get the stage axis limits for x, y, z, t, r."""
        from fibsem.microscopes.simulator import (
            STAGE_LIMITS_COMPUSTAGE,
            STAGE_LIMITS_DEFAULT,
        )

        if self.stage_is_compustage:
            return STAGE_LIMITS_COMPUSTAGE

        if not hasattr(self.stage, "get_axis_limits"):
            return STAGE_LIMITS_DEFAULT

        limits: Dict[str, RangeLimit] = {}
        for axis in ["x", "y", "z", "t"]:
            axis_limit = self.stage.get_axis_limits(axis)
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
        if not self.stage_is_compustage:
            limits["r"] = RangeLimit(
                min=-360,
                max=360,
            )
        return limits

    def _safe_rotation_movement(self, stage_position: FibsemStagePosition):
        """Tilt the stage flat when performing a large rotation to prevent collision.

        Args:
            stage_position (StagePosition): desired stage position.
        """
        current_position = self.get_stage_position()

        # tilt flat for large rotations to prevent collisions
        from fibsem import movement

        if movement.rotation_angle_is_larger(stage_position.r, current_position.r):
            self.move_stage_absolute(FibsemStagePosition(t=0))
            logging.info("tilting to flat for large rotation.")

        return

    def safe_absolute_stage_movement(self, stage_position: FibsemStagePosition) -> None:
        """Move the stage to the desired position in a safe manner, using compucentric rotation.
        Supports movements in the stage_position coordinate system
        """
        # Before anything moves. The staged move below rotates the stage where it
        # stands, which is the correct order leaving the beams and the wrong one
        # coming back from the FM -- see FIB-841.
        self._refuse_rotation_at_the_fluorescence_microscope(stage_position)

        # safe movements are not required on the compustage, because it doesn't rotate
        if not self.stage_is_compustage:
            # tilt flat for large rotations to prevent collisions
            self._safe_rotation_movement(stage_position)

            # move to compucentric rotation
            self.move_stage_absolute(
                FibsemStagePosition(r=stage_position.r, coordinate_system="RAW")
            )  # TODO: support compucentric rotation directly

        logging.debug(f"safe moving to {stage_position}")
        self.move_stage_absolute(stage_position)

        logging.debug("safe movement complete.")

        return

    def project_stable_move(
        self,
        dx: float,
        dy: float,
        beam_type: BeamType,
        base_position: FibsemStagePosition,
    ) -> FibsemStagePosition:

        scan_rotation = self.get_scan_rotation(beam_type=beam_type)
        if np.isclose(scan_rotation, np.pi):
            dx *= -1.0
            dy *= -1.0

        # stable-move-projection
        point_yz = self._y_corrected_stage_movement(dy, beam_type)
        dy, dz = point_yz.y, point_yz.z

        # calculate the corrected move to reach that point from base-state?
        new_position = deepcopy(base_position)
        new_position.x += dx
        new_position.y += dy
        new_position.z += dz

        return new_position

    def insert_manipulator(self, name: str = "PARK"):
        """Insert the manipulator to the specified position"""

        if not self.is_available("manipulator"):
            raise ValueError("Manipulator not available.")

        if name not in ["PARK", "EUCENTRIC"]:
            raise ValueError(f"insert position {name} not supported.")
        if AUTOSCRIPT_VERSION < MINIMUM_AUTOSCRIPT_VERSION_4_7:
            raise NotImplementedError(
                "Manipulator saved positions not supported in this version. Please upgrade to 4.7 or higher"
            )

        # get the saved position name
        saved_position = (
            ManipulatorSavedPosition.PARK
            if name == "PARK"
            else ManipulatorSavedPosition.EUCENTRIC
        )

        # get the insert position
        insert_position = self.connection.specimen.manipulator.get_saved_position(
            saved_position, ManipulatorCoordinateSystem.RAW
        )
        # insert the manipulator
        logging.info("inserting manipulator to {saved_position}: {insert_position}.")
        self.connection.specimen.manipulator.insert(insert_position)
        logging.info("insert manipulator complete.")

        # return the manipulator position
        manipulator_position = self.get_manipulator_position()
        logging.debug(
            {
                "msg": "insert_manipulator",
                "name": name,
                "position": manipulator_position.to_dict(),
            }
        )
        return manipulator_position

    def retract_manipulator(self):
        """Retract the manipulator"""

        if AUTOSCRIPT_VERSION < MINIMUM_AUTOSCRIPT_VERSION_4_7:
            raise NotImplementedError(
                "Manipulator saved positions not supported in this version. Please upgrade to 4.7 or higher"
            )

        if not self.is_available("manipulator"):
            raise NotImplementedError("Manipulator not available.")

        # Retract the needle, preserving the correct parking postiion
        needle = self.connection.specimen.manipulator
        park_position = needle.get_saved_position(
            ManipulatorSavedPosition.PARK, ManipulatorCoordinateSystem.RAW
        )

        logging.info(f"retracting needle to {park_position}")
        needle.absolute_move(park_position)
        time.sleep(1)  # AutoScript sometimes throws errors if you retract too quick?
        logging.info("retracting needle...")
        needle.retract()
        logging.info("retract needle complete")

    def move_manipulator_relative(self, position: FibsemManipulatorPosition):
        logging.info(f"moving manipulator by {position}")

        # convert to autoscript position
        autoscript_position = manipulator_position_to_autoscript(position)
        # move manipulator relative
        self.connection.specimen.manipulator.relative_move(autoscript_position)
        logging.debug(
            {"msg": "move_manipulator_relative", "position": position.to_dict()}
        )

    def move_manipulator_absolute(self, position: FibsemManipulatorPosition):
        """Move the manipulator to the specified coordinates."""
        logging.info(f"moving manipulator to {position}")

        # convert to autoscript
        autoscript_position = manipulator_position_to_autoscript(position)

        # move manipulator
        self.connection.specimen.manipulator.absolute_move(autoscript_position)
        logging.debug(
            {"msg": "move_manipulator_absolute", "position": position.to_dict()}
        )

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
    ) -> None:
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
        self.move_manipulator_relative(manipulator_position)

        return self.get_manipulator_position()

    def move_manipulator_to_position_offset(
        self, offset: FibsemManipulatorPosition, name: str = None
    ) -> None:
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
        self.move_manipulator_absolute(saved_position)

    def _get_saved_manipulator_position(
        self, name: str = "PARK"
    ) -> FibsemManipulatorPosition:

        if name not in ["PARK", "EUCENTRIC"]:
            raise ValueError(f"saved position {name} not supported.")
        if AUTOSCRIPT_VERSION < MINIMUM_AUTOSCRIPT_VERSION_4_7:
            raise NotImplementedError(
                "Manipulator saved positions not supported in this version. Please upgrade to 4.7 or higher"
            )

        named_position = (
            ManipulatorSavedPosition.PARK
            if name == "PARK"
            else ManipulatorSavedPosition.EUCENTRIC
        )
        autoscript_position = self.connection.specimen.manipulator.get_saved_position(
            named_position,
            ManipulatorCoordinateSystem.STAGE,  # TODO: why is this STAGE not RAW?
        )

        # convert to FibsemManipulatorPosition
        manipulator_position = manipulator_position_from_autoscript(autoscript_position)

        logging.debug(
            {
                "msg": "get_saved_manipulator_position",
                "name": name,
                "position": manipulator_position.to_dict(),
            }
        )

        return manipulator_position

    def setup_milling(
        self,
        mill_settings: FibsemMillingSettings,
    ):
        """
        Configure the microscope for milling using the ion beam.

        Args:
            mill_settings (FibsemMillingSettings): Milling settings.
        """
        self.milling_channel = mill_settings.milling_channel
        self.set_channel(self.milling_channel)
        self.connection.patterning.set_default_beam_type(self.milling_channel.value)
        self.set_application_file(mill_settings.application_file, default=True)
        self.set_patterning_mode(mill_settings.patterning_mode)
        self.clear_patterns()  # clear any existing patterns
        self.set_field_of_view(hfw=mill_settings.hfw, beam_type=self.milling_channel)
        self.set_beam_current(
            current=mill_settings.milling_current, beam_type=self.milling_channel
        )
        self.set_beam_voltage(
            voltage=mill_settings.milling_voltage, beam_type=self.milling_channel
        )

        # TODO: migrate to _set_milling_settings():
        # self.milling_channel = mill_settings.milling_channel
        # self.set_milling_settings(mill_settings)
        # self.clear_patterns()

        logging.debug(
            {"msg": "setup_milling", "mill_settings": mill_settings.to_dict()}
        )

    def run_milling(
        self, milling_current: float, milling_voltage: float, asynch: bool = False
    ):
        """
        Run ion beam milling using the specified milling current.

        Args:
            milling_current (float): The current to use for milling in amps.
            milling_voltage (float): The voltage to use for milling in volts.
            asynch (bool, optional): If True, the milling will be run asynchronously.
                                     Defaults to False, in which case it will run synchronously.
        """
        if not self.is_available("ion_beam"):
            raise ValueError("Ion beam not available.")

        try:
            # change to milling current, voltage # TODO: do this in a more standard way (there are other settings)
            if self.get_beam_voltage(beam_type=self.milling_channel) != milling_voltage:
                self.set_beam_voltage(
                    voltage=milling_voltage, beam_type=self.milling_channel
                )
            if self.get_beam_current(beam_type=self.milling_channel) != milling_current:
                self.set_beam_current(
                    current=milling_current, beam_type=self.milling_channel
                )
        except Exception as e:
            logging.warning(
                f"Failed to set voltage or current: {e}, voltage={milling_voltage}, current={milling_current}"
            )

        # run milling (asynchronously)
        self.set_channel(channel=self.milling_channel)  # the ion beam view
        logging.info(f"running ion beam milling now... asynchronous={asynch}")
        self.start_milling()

        start_time = time.time()
        estimated_time = self.estimate_milling_time()
        remaining_time = estimated_time

        if asynch:
            return  # return immediately, up to the caller to handle the milling process

        MILLING_SLEEP_TIME = 1
        while self.get_milling_state() is MillingState.IDLE:  # giving time to start
            time.sleep(0.5)
        while self.get_milling_state() in ACTIVE_MILLING_STATES:
            # logging.info(f"Patterning State: {self.connection.patterning.state}")
            # TODO: add drift correction support here... generically
            if self.get_milling_state() is MillingState.RUNNING:
                remaining_time -= (
                    MILLING_SLEEP_TIME  # TODO: investigate if this is a good estimate
                )
            time.sleep(MILLING_SLEEP_TIME)
            # TODO: refresh the remaining time by getting the milling time from the patterning API as user can change the patterns on xtUI

            # update milling progress via signal
            self.milling_progress_signal.emit(
                MillingProgress(
                    status=MillingProgressStatus.STAGE_UPDATE,
                    start_time=start_time,
                    milling_state=self.get_milling_state(),
                    estimated_time=estimated_time,
                    remaining_time=remaining_time,
                )
            )

        # milling complete
        self.clear_patterns()

        logging.debug(
            {
                "msg": "run_milling",
                "milling_current": milling_current,
                "milling_voltage": milling_voltage,
                "asynch": asynch,
            }
        )

    def finish_milling(self, imaging_current: float, imaging_voltage: float):
        """
        Finalises the milling process by clearing the microscope of any patterns and returning the current to the imaging current.

        Args:
            imaging_current (float): The current to use for imaging in amps.
        """
        self.clear_patterns()
        self.set_beam_current(current=imaging_current, beam_type=self.milling_channel)
        self.set_beam_voltage(voltage=imaging_voltage, beam_type=self.milling_channel)
        self.set_patterning_mode("Serial")
        # TODO: store initial imaging settings in setup_milling, restore here, rather than hybrid

        logging.debug(
            {
                "msg": "finish_milling",
                "imaging_current": imaging_current,
                "imaging_voltage": imaging_voltage,
            }
        )

    # def setup_milling2(
    #     self,
    #     milling_stage: 'FibsemMillingStage',
    # ):
    #     """
    #     Configure the microscope for milling using the ion beam.

    #     Args:
    #         milling_stage (FibsemMillingStage): Milling stage.
    #     """
    #     self.milling_channel = milling_stage.milling.milling_channel
    #     self.set_channel(self.milling_channel)
    #     self.clear_patterns()  # clear any existing patterns
    #     self.set_default_patterning_beam_type(self.milling_channel)
    #     self.set_application_file(milling_stage.milling.application_file, default=True)
    #     self.set_patterning_mode(milling_stage.milling.patterning_mode)
    #     self.set_field_of_view(hfw=milling_stage.milling.hfw, beam_type=self.milling_channel)
    #     self.set_beam_current(current=milling_stage.milling.milling_current, beam_type=self.milling_channel)
    #     self.set_beam_voltage(voltage=milling_stage.milling.milling_voltage, beam_type=self.milling_channel)

    def set_default_patterning_beam_type(self, beam_type: BeamType):
        """Set the default beam type for patterning."""
        if beam_type not in BeamType:
            raise ValueError(
                f"Beam type {beam_type} not supported. Supported types: {list(BeamType)}"
            )

        self.connection.patterning.set_default_beam_type(beam_type.value)
        return beam_type

    # def finish_milling2(self):
    #     """Clear the patterns and reset the beam settings to the imaging state."""
    #     self.clear_patterns()
    #     self.set_beam_current(current=self.system.ion.beam.beam_current, beam_type=self.milling_channel)
    #     self.set_beam_voltage(voltage=self.system.ion.beam.voltage, beam_type=self.milling_channel)
    #     self.set_patterning_mode(mode="Serial")  # reset to serial mode

    def start_milling(self) -> None:
        """Start the milling process."""
        with self._threading_lock:
            if self.get_milling_state() is MillingState.IDLE:
                self.connection.patterning.start()
                logging.info("Starting milling...")

    def stop_milling(self) -> None:
        """Stop the milling process."""
        with self._threading_lock:
            if self.get_milling_state() in ACTIVE_MILLING_STATES:
                logging.info("Stopping milling...")
                self.connection.patterning.stop()
                logging.info("Milling stopped.")

    def pause_milling(self) -> None:
        """Pause the milling process."""
        with self._threading_lock:
            if self.get_milling_state() == MillingState.RUNNING:
                logging.info("Pausing milling...")
                self.connection.patterning.pause()
                logging.info("Milling paused.")

    def resume_milling(self) -> None:
        """Resume the milling process."""
        with self._threading_lock:
            if self.get_milling_state() == MillingState.PAUSED:
                logging.info("Resuming milling...")
                self.connection.patterning.resume()
                logging.info("Milling resumed.")

    def get_milling_state(self) -> MillingState:
        """Get the current milling state."""
        with self._threading_lock:
            self.set_channel(channel=self.milling_channel)
            return MillingState[self.connection.patterning.state.upper()]

    def clear_patterns(self):
        """Clear all currently drawn milling patterns."""
        self.connection.patterning.clear_patterns()
        self._patterns = []

    def estimate_milling_time(self) -> float:
        """Calculates the estimated milling time for a list of patterns."""
        total_time = 0
        for pattern in self._patterns:
            total_time += pattern.time

        return total_time

    def get_application_file(self, application_file: str, strict: bool = True) -> str:
        """Get a valid application file for the patterning API.
        The api requires setting a valid application file before creating patterns.
        Args:
            application_file (str): The name of the application file to set as default.
            strict (bool): If True, raises an error if the application file is not available.
                If False, tries to find the closest match to the application file.
                Defaults to True.
        Returns:
                str: The name of the application file that was set as default.
        Raises:
            ValueError: If the application file is not available.
        """

        # check if the application file is valid
        application_files = self.get_available_values("application_file")
        if application_file not in application_files:
            if strict:
                raise ValueError(
                    f"Application file {application_file} not available. Available files: {application_files}"
                )
            from difflib import get_close_matches

            closest_match = get_close_matches(application_file, application_files, n=1)
            if not closest_match:
                raise ValueError(
                    f"Application file {application_file} not available. Available files: {application_files}"
                )
            application_file = str(closest_match[0])

        return application_file

    def set_application_file(
        self, application_file: str, default: bool = False, strict: bool = True
    ) -> str:
        """Sets the default application file for the patterning API.
        The api requires setting a valid application file before creating patterns.
        Args:
            application_file (str): The name of the application file to set as default.
        """
        application_file = self.get_application_file(application_file, strict=strict)
        self.connection.patterning.set_default_application_file(application_file)
        self._current_application_file = application_file

        if default:
            self._default_application_file = application_file

        logging.debug(
            {
                "msg": "set_application_file",
                "application_file": application_file,
                "default": default,
            }
        )
        return application_file

    def get_current_application_file(self) -> str:
        return self._current_application_file

    def get_default_application_file(self) -> str:
        return self._default_application_file

    def set_patterning_mode(self, mode: str):
        """Sets the patterning mode for the patterning API.
        The api requires setting a valid patterning mode before creating patterns.
        Args:
            mode (str): The patterning mode to set. Can be "Serial" or "Parallel".
        """
        if mode not in ["Serial", "Parallel"]:
            raise ValueError(
                f"Patterning mode {mode} not supported. Supported modes: Serial, Parallel"
            )

        self.connection.patterning.mode = mode
        logging.debug({"msg": "set_patterning_mode", "mode": mode})
        return mode

    @_thermo_application_file_wrapper_for_drawing_functions
    def draw_rectangle(
        self,
        pattern_settings: FibsemRectangleSettings,
    ):
        """
        Draws a rectangle pattern using the current ion beam.

        Args:
            pattern_settings (FibsemRectangleSettings): the settings for the pattern to draw.

        Returns:
            Pattern: the created pattern.

        Raises:
            AutoscriptError: if an error occurs while creating the pattern.
        """

        # get patterning api
        patterning_api = self.connection.patterning
        if pattern_settings.cross_section is CrossSectionPattern.RegularCrossSection:
            create_pattern_function = patterning_api.create_regular_cross_section
            self.set_patterning_mode(
                "Serial"
            )  # parallel mode not supported for regular cross section
            self.set_application_file("Si-multipass", strict=False)
        elif pattern_settings.cross_section is CrossSectionPattern.CleaningCrossSection:
            create_pattern_function = patterning_api.create_cleaning_cross_section
            self.set_patterning_mode(
                "Serial"
            )  # parallel mode not supported for cleaning cross section
            self.set_application_file("Si-ccs", strict=False)
        else:
            create_pattern_function = patterning_api.create_rectangle
            # ensure a rectangle-compatible application file is set; the stage's
            # application file may be a cross-section-only file (e.g. Si-ccs) that
            # AutoScript rejects for a plain Rectangle pattern.
            self.set_application_file("Si", strict=False)

        # create pattern
        pattern = create_pattern_function(
            center_x=pattern_settings.centre_x,
            center_y=pattern_settings.centre_y,
            width=pattern_settings.width,
            height=pattern_settings.height,
            depth=pattern_settings.depth,
        )

        if not np.isclose(pattern_settings.time, 0.0):
            logging.debug(f"Setting pattern time to {pattern_settings.time}.")
            pattern.time = pattern_settings.time

        # set pattern rotation
        pattern.rotation = pattern_settings.rotation

        # set exclusion
        pattern.is_exclusion_zone = pattern_settings.is_exclusion

        # set scan direction
        available_scan_directions = self.get_available_values("scan_direction")

        if pattern_settings.scan_direction in available_scan_directions:
            pattern.scan_direction = pattern_settings.scan_direction
        else:
            pattern.scan_direction = "TopToBottom"
            logging.warning(
                f"Scan direction {pattern_settings.scan_direction} not supported. Using TopToBottom instead."
            )
            logging.warning(
                f"Supported scan directions are: {available_scan_directions}"
            )

        # set passes
        if pattern_settings.passes:  # not zero
            if isinstance(pattern, RegularCrossSectionPattern):
                pattern.multi_scan_pass_count = pattern_settings.passes
                pattern.scan_method = 1  # multi scan
            else:
                pattern.dwell_time = pattern.dwell_time * (
                    pattern.pass_count / pattern_settings.passes
                )

                # NB: passes, time, dwell time are all interlinked, therefore can only adjust passes indirectly
                # if we adjust passes directly, it just reduces the total time to compensate, rather than increasing the dwell_time
                # NB: the current must be set before doing this, otherwise it will be out of range

        logging.debug(
            {"msg": "draw_rectangle", "pattern_settings": pattern_settings.to_dict()}
        )

        self._patterns.append(pattern)

        return pattern

    @_thermo_application_file_wrapper_for_drawing_functions
    def draw_line(self, pattern_settings: FibsemLineSettings):
        """
        Draws a line pattern on the current imaging view of the microscope.

        Args:
            pattern_settings (FibsemLineSettings): A data class object specifying the pattern parameters,
                including the start and end points, and the depth of the pattern.

        Returns:
            LinePattern: A line pattern object, which can be used to configure further properties or to add the
                pattern to the milling list.

        Raises:
            autoscript.exceptions.InvalidArgumentException: if any of the pattern parameters are invalid.
        """
        pattern = self.connection.patterning.create_line(
            start_x=pattern_settings.start_x,
            start_y=pattern_settings.start_y,
            end_x=pattern_settings.end_x,
            end_y=pattern_settings.end_y,
            depth=pattern_settings.depth,
        )
        logging.debug(
            {"msg": "draw_line", "pattern_settings": pattern_settings.to_dict()}
        )
        self._patterns.append(pattern)
        return pattern

    @_thermo_application_file_wrapper_for_drawing_functions
    def draw_circle(self, pattern_settings: FibsemCircleSettings):
        """
        Draws a circle pattern on the current imaging view of the microscope.

        Args:
            pattern_settings (FibsemCircleSettings): A data class object specifying the pattern parameters,
                including the centre point, radius and depth of the pattern.

        Returns:
            CirclePattern: A circle pattern object, which can be used to configure further properties or to add the
                pattern to the milling list.

        Raises:
            autoscript.exceptions.InvalidArgumentException: if any of the pattern parameters are invalid.
        """

        outer_diameter = 2 * pattern_settings.radius
        inner_diameter = 0
        if pattern_settings.thickness != 0:
            inner_diameter = outer_diameter - 2 * pattern_settings.thickness

        fallback_application_file = "Si"
        try:
            pattern = self.connection.patterning.create_circle(
                center_x=pattern_settings.centre_x,
                center_y=pattern_settings.centre_y,
                outer_diameter=outer_diameter,
                inner_diameter=inner_diameter,
                depth=pattern_settings.depth,
            )
        except Exception:
            if self.get_current_application_file() == fallback_application_file:
                # No need to try again with the same application file
                raise
            logging.warning(
                "Failed to draw circle pattern, falling back on application file %s",
                fallback_application_file,
            )
            self.set_application_file(fallback_application_file)
            pattern = self.connection.patterning.create_circle(
                center_x=pattern_settings.centre_x,
                center_y=pattern_settings.centre_y,
                outer_diameter=outer_diameter,
                inner_diameter=inner_diameter,
                depth=pattern_settings.depth,
            )
        # set exclusion
        pattern.is_exclusion_zone = pattern_settings.is_exclusion

        logging.debug(
            {"msg": "draw_circle", "pattern_settings": pattern_settings.to_dict()}
        )
        self._patterns.append(pattern)
        return pattern

    @_thermo_application_file_wrapper_for_drawing_functions
    def draw_bitmap_pattern(self, pattern_settings: FibsemBitmapSettings):
        # Avoid modifying the original pattern_settings object
        pattern_settings = deepcopy(pattern_settings)

        if pattern_settings.bitmap is None:
            logging.warning("Bitmap pattern will be skipped as no bitmap has been set")
            return None

        # Get bitmap from pattern settings
        bitmap_pattern = BitmapPatternDefinition()

        if pattern_settings.flip_y:
            pattern_settings.bitmap = np.flip(pattern_settings.bitmap, axis=0)

        points = pattern_settings.bitmap

        fallback_application_file = "Si"
        try:
            if pattern_settings.interpolate is not None:
                points = self._resize_bitmap_to_pattern(pattern_settings)
            bitmap_pattern.points = points
            pattern = self.connection.patterning.create_bitmap(
                center_x=pattern_settings.centre_x,
                center_y=pattern_settings.centre_y,
                width=pattern_settings.width,
                height=pattern_settings.height,
                depth=pattern_settings.depth,
                bitmap_pattern_definition=bitmap_pattern,
            )
        except Exception:
            if self.get_current_application_file() == fallback_application_file:
                # No need to try again with the same application file
                raise
            logging.warning(
                "Failed to draw bitmap pattern, falling back on application file %s",
                fallback_application_file,
            )
            self.set_application_file(fallback_application_file)

            if pattern_settings.interpolate is not None:
                points = self._resize_bitmap_to_pattern(pattern_settings)
            bitmap_pattern.points = points
            pattern = self.connection.patterning.create_bitmap(
                center_x=pattern_settings.centre_x,
                center_y=pattern_settings.centre_y,
                width=pattern_settings.width,
                height=pattern_settings.height,
                depth=pattern_settings.depth,
                bitmap_pattern_definition=bitmap_pattern,
            )

        if not np.isclose(pattern_settings.time, 0.0):
            logging.debug("Setting pattern time to %f", pattern_settings.time)
            pattern.time = pattern_settings.time

        # set pattern rotation
        pattern.rotation = pattern_settings.rotation

        # set exclusion
        pattern.is_exclusion_zone = pattern_settings.is_exclusion

        # set scan direction
        available_scan_directions = self.get_available_values("scan_direction")

        if pattern_settings.scan_direction in available_scan_directions:
            pattern.scan_direction = pattern_settings.scan_direction
        else:
            pattern.scan_direction = "TopToBottom"
            logging.warning(
                "Scan direction %s not supported. Using TopToBottom instead.",
                pattern_settings.scan_direction,
            )
            logging.warning(
                "Supported scan directions are: %s", str(available_scan_directions)
            )

        # set passes
        if pattern_settings.passes:  # not zero
            pattern.dwell_time = pattern.dwell_time * (
                pattern.pass_count / pattern_settings.passes
            )

            # NB: passes, time, dwell time are all interlinked, therefore can only adjust passes indirectly
            # if we adjust passes directly, it just reduces the total time to compensate, rather than increasing the dwell_time
            # NB: the current must be set before doing this, otherwise it will be out of range

        logging.debug(
            {
                "msg": "draw_bitmap_pattern",
                "pattern_settings": pattern_settings.to_dict(),
            }
        )
        self._patterns.append(pattern)
        return pattern

    def _resize_bitmap_to_pattern(
        self, pattern_settings: FibsemBitmapSettings
    ) -> NDArray[np.float64 | np.uint8]:
        points = pattern_settings.bitmap

        if points is None:
            raise ValueError(
                "Unable to resize bitmap as FibsemBitmapSettings.bitmap is None"
            )

        # Get pitch to calculate expected pixel size
        rectangle = self.connection.patterning.create_rectangle(
            center_x=pattern_settings.centre_x,
            center_y=pattern_settings.centre_y,
            width=pattern_settings.width,
            height=pattern_settings.height,
            depth=pattern_settings.depth,
        )

        new_shape = (
            int(round(pattern_settings.height / rectangle.pitch_y)),
            int(round(pattern_settings.width / rectangle.pitch_x)),
        )

        # Disable after calculations just in case values are cleared
        rectangle.enabled = False

        if pattern_settings.interpolate == "bicubic":
            order = 3
        elif pattern_settings.interpolate == "bilinear":
            order = 1
        elif pattern_settings.interpolate == "nearest":
            order = 0
        else:
            raise ValueError(
                f"Invalid interpolate option '{pattern_settings.interpolate}'"
            )

        resized_points = np.empty((*new_shape, 2), dtype=object)

        resized_points[:, :, 0] = transform.resize(
            points[:, :, 0]
            .reshape(points.shape[0], points.shape[1])
            .astype(np.float64),
            output_shape=new_shape,
            order=order,
            preserve_range=True,
        ).astype(np.float64)
        resized_points[:, :, 1] = transform.resize(
            points[:, :, 1].reshape(points.shape[0], points.shape[1]).astype(np.uint8),
            output_shape=new_shape,
            order=0,
            preserve_range=True,
        ).astype(np.uint8)

        return resized_points

    @_thermo_application_file_wrapper_for_drawing_functions
    def draw_polygon(self, pattern_settings: FibsemPolygonSettings) -> None:
        """Draw a polygon pattern on the current imaging view of the microscope."""

        if AUTOSCRIPT_VERSION < parse_version("4.12"):
            raise NotImplementedError(
                "Polygon patterning is only supported in Autoscript 4.12 or higher."
            )

        pattern = self.connection.patterning.create_polygon(
            pattern_settings.vertices, depth=pattern_settings.depth
        )
        pattern.is_exclusion_zone = pattern_settings.is_exclusion

        logging.debug(
            {"msg": "draw_polygon", "pattern_settings": pattern_settings.to_dict()}
        )
        self._patterns.append(pattern)
        return pattern

    def get_gis(self, port: str = None):
        use_multichem = self.is_available("gis_multichem")

        if use_multichem:
            gis = self.connection.gas.get_multichem()
        else:
            gis = self.connection.gas.get_gis_port(port)
        logging.debug({"msg": "get_gis", "use_multichem": use_multichem, "port": port})
        self.gis = gis
        return self.gis

    def insert_gis(self, insert_position: str = None) -> None:

        if insert_position:
            logging.info(f"Inserting Multichem GIS to {insert_position}")
            self.gis.insert(insert_position)
        else:
            logging.info("Inserting Gas Injection System")
            self.gis.insert()

        logging.debug({"msg": "insert_gis", "insert_position": insert_position})

    def retract_gis(self):
        """Retract the gis"""
        self.gis.retract()
        logging.debug(
            {"msg": "retract_gis", "use_multichem": self.is_available("gis_multichem")}
        )

    def gis_turn_heater_on(self, gas: str = None) -> None:
        """Turn the heater on and wait for it to get to temperature"""
        logging.info(f"Turning on heater for {gas}")
        if gas is not None:
            self.gis.turn_heater_on(gas)
        else:
            self.gis.turn_heater_on()

        logging.info("Waiting for heater to get to temperature...")
        time.sleep(3)  # we need to wait a bit

        wait_time = 0
        max_wait_time = 15
        target_temp = 300  # validate this somehow?
        while True:
            if gas is not None:
                temp = self.gis.get_temperature(gas)  # multi-chem requires gas name
            else:
                temp = self.gis.get_temperature()
            logging.info(
                f"Waiting for heater: {temp}K, target={target_temp}, wait_time={wait_time}/{max_wait_time} sec"
            )

            if temp >= target_temp:
                break

            time.sleep(1)  # wait for the heat

            wait_time += 1
            if wait_time > max_wait_time:
                raise TimeoutError("Gas Injection Failed to heat within time...")

        logging.debug(
            {
                "msg": "gis_turn_heater_on",
                "temp": temp,
                "target_temp": target_temp,
                "wait_time": wait_time,
                "max_wait_time": max_wait_time,
            }
        )

        return

    def cryo_deposition_v2(self, gis_settings: FibsemGasInjectionSettings) -> None:
        """Run non-specific cryo deposition protocol.

        # TODO: universalise this for demo, tescan
        """

        use_multichem = self.is_available("gis_multichem")
        port = gis_settings.port
        gas = gis_settings.gas
        duration = gis_settings.duration
        insert_position = gis_settings.insert_position

        logging.debug({"msg": "cryo_depositon_v2", "settings": gis_settings.to_dict()})

        # get gis subsystem
        self.get_gis(port)

        # insert gis / multichem
        logging.info(f"Inserting Gas Injection System at {insert_position}")
        if use_multichem is False:
            insert_position = None
        self.insert_gis(insert_position)

        # turn heater on
        gas = gas if use_multichem else None
        self.gis_turn_heater_on(gas)

        # run deposition
        logging.info(f"Running deposition for {duration} seconds")
        self.gis.open()
        time.sleep(duration)
        # TODO: provide more feedback to user
        self.gis.close()

        # turn off heater
        logging.info(f"Turning off heater for {gas}")
        self.gis.turn_heater_off()

        # retract gis / multichem
        logging.info("Retracting Gas Injection System")
        self.retract_gis()

        return

    def setup_sputter(self, protocol: dict):
        """
        Set up the sputter coating process on the microscope.

        Args:
            protocol (dict): Dictionary containing the protocol details for sputter coating.

        Returns:
            None

        Raises:
            None

        Notes:
            This function sets up the sputter coating process on the microscope.
            It sets the active view to the electron beam, clears any existing patterns, and sets the default beam type to the electron beam.
            It then inserts the multichem and turns on the heater for the specified gas according to the given protocol.
            This function also waits for 3 seconds to allow the heater to warm up.
        """
        self.original_active_view = self.connection.imaging.get_active_view()
        self.set_channel(BeamType.ELECTRON)
        self.connection.patterning.clear_patterns()
        self.set_application_file(protocol["application_file"])
        self.connection.patterning.set_default_beam_type(BeamType.ELECTRON.value)
        self.multichem = self.connection.gas.get_multichem()
        self.multichem.insert(protocol["position"])
        self.multichem.turn_heater_on(protocol["gas"])  # "Pt cryo")
        time.sleep(3)

        logging.debug({"msg": "setup_sputter", "protocol": protocol})

    def draw_sputter_pattern(
        self, hfw: float, line_pattern_length: float, sputter_time: float
    ):
        """
        Draws a line pattern for sputtering with the given parameters.

        Args:
            hfw (float): The horizontal field width of the electron beam.
            line_pattern_length (float): The length of the line pattern to draw.
            sputter_time (float): The time to sputter the line pattern.

        Returns:
            None

        Notes:
            Sets the horizontal field width of the electron beam to the given value.
            Draws a line pattern for sputtering with the given length and milling depth.
            Sets the sputter time of the line pattern to the given value.

        """
        self.connection.beams.electron_beam.horizontal_field_width.value = hfw
        pattern = self.connection.patterning.create_line(
            -line_pattern_length / 2,  # x_start
            +line_pattern_length,  # y_start
            +line_pattern_length / 2,  # x_end
            +line_pattern_length,  # y_end
            2e-6,
        )  # milling depth
        pattern.time = sputter_time + 0.1

        logging.debug(
            {
                "msg": "draw_sputter_pattern",
                "hfw": hfw,
                "line_pattern_length": line_pattern_length,
                "sputter_time": sputter_time,
            }
        )

    def run_sputter(self, **kwargs):
        """
        Runs the GIS Platinum Sputter.

        Args:
            **kwargs: Optional keyword arguments for the sputter function. The required argument for
        the Thermo version is "sputter_time" (int), which specifies the time to sputter in seconds.

        Returns:
            None

        Notes:
        - Blanks the electron beam.
        - Starts sputtering with platinum for the specified sputter time, and waits until the sputtering
        is complete before continuing.
        - If the patterning state is not ready, raises a RuntimeError.
        - If the patterning state is running, stops the patterning.
        - If the patterning state is idle, logs a warning message suggesting to adjust the patterning
        line depth.
        """
        sputter_time = kwargs["sputter_time"]

        self.connection.beams.electron_beam.blank()
        if self.connection.patterning.state == "Idle":
            logging.info(
                "Sputtering with platinum for {} seconds...".format(sputter_time)
            )
            self.connection.patterning.start()  # asynchronous patterning
            time.sleep(sputter_time + 5)
        else:
            raise RuntimeError("Can't sputter platinum, patterning state is not ready.")
        if self.connection.patterning.state == "Running":
            self.connection.patterning.stop()
        else:
            logging.warning(
                "Patterning state is {}".format(self.connection.patterning.state)
            )
            logging.warning("Consider adjusting the patterning line depth.")

    def finish_sputter(self, application_file: str) -> None:
        """
        Finish the sputter process by clearing patterns and resetting beam and imaging settings.

        Args:
            application_file (str): The path to the default application file to use.

        Returns:
            None

        Raises:
            None

        Notes:
            This function finishes the sputter process by clearing any remaining patterns and restoring the beam and imaging settings to their
            original state. It sets the beam current back to imaging current and sets the default beam type to ion beam.
            It also retracts the multichem and logs that the sputtering process has finished.
        """
        # Clear any remaining patterns
        self.connection.patterning.clear_patterns()

        # Restore beam and imaging settings to their original state
        self.connection.beams.electron_beam.unblank()
        self.set_application_file(application_file)
        self.connection.imaging.set_active_view(self.original_active_view)
        self.connection.patterning.set_default_beam_type(
            BeamType.ION.value
        )  # set ion beam
        self.multichem.retract()

        # Log that the sputtering process has finished
        logging.info("Platinum sputtering process completed.")

    def get_available_values(
        self, key: str, beam_type: Optional[BeamType] = None
    ) -> Tuple:
        """Get a list of available values for a given key.
        Keys: application_file, plasma_gas, current, detector_type, detector_mode
        """

        values = []
        if key == "application_file":
            values = self.connection.patterning.list_all_application_files()

        if beam_type is BeamType.ION and self.system.ion.plasma:
            if key == "plasma_gas":
                values = (
                    self.connection.beams.ion_beam.source.plasma_gas.available_values
                )

        if key == "current":
            if beam_type is BeamType.ION and self.is_available("ion_beam"):
                values = self.connection.beams.ion_beam.beam_current.available_values
            elif beam_type is BeamType.ELECTRON and self.is_available("electron_beam"):
                # loop through the beam current range, to match the available choices on microscope
                limits: Limits = self.connection.beams.electron_beam.beam_current.limits
                beam_current = limits.min
                while beam_current <= limits.max:
                    values.append(beam_current)
                    beam_current *= 2.0

        if key == "voltage":
            beam = self._get_beam(beam_type)
            limits: Limits = beam.high_voltage.limits
            # QUERY: match what is displayed on microscope, as list[float], or keep as range?
            # technically we can set any value, but primarily people would use what is on microscope
            # SEM: [1000, 2000, 3000, 5000, 10000, 20000, 30000]
            # FIB: [500, 1000, 2000, 8000, 1600, 30000]
            if beam_type is BeamType.ION:
                VALUES = (500, 1000, 2000, 8000, 16000, 30000)
            if beam_type is BeamType.ELECTRON:
                VALUES = (1000, 2000, 3000, 5000, 10000, 20000, 30000)
            # filter values to be within limits
            values = [v for v in VALUES if limits.min <= v <= limits.max]
            return values

        if key == "detector_type":
            values = self.connection.detector.type.available_values

        if key == "detector_mode":
            values = self.connection.detector.mode.available_values

        if key == "scan_direction":
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
            values = TFS_SCAN_DIRECTIONS

        if key == "gis_ports":
            if self.is_available("gis"):
                values = self.connection.gas.list_all_gis_ports()
            elif self.is_available("multichem"):
                values = self.connection.gas.list_all_multichem_ports()
            else:
                values = []

        logging.debug({"msg": "get_available_values", "key": key, "values": values})

        return values

    def _get(
        self, key: str, beam_type: Optional[BeamType] = None
    ) -> Union[
        int,
        float,
        str,
        list,
        Point,
        FibsemStagePosition,
        FibsemManipulatorPosition,
        None,
    ]:
        """Get a property of the microscope."""
        # TODO: make the list of get and set keys available to the user
        if beam_type is not None:
            beam = self._get_beam(beam_type)

        if key == "active_view":
            return self.connection.imaging.get_active_view()
        if key == "active_device":
            return self.connection.imaging.get_active_device()

        # beam properties
        if key == "on":
            return beam.is_on
        if key == "blanked":
            return beam.is_blanked
        if key == "working_distance":
            return beam.working_distance.value
        if key == "current":
            return beam.beam_current.value
        if key == "voltage":
            return beam.high_voltage.value
        if key == "hfw":
            return beam.horizontal_field_width.value
        if key == "dwell_time":
            return beam.scanning.dwell_time.value
        if key == "scan_rotation":
            return beam.scanning.rotation.value
        if key == "voltage_limits":
            return beam.high_voltage.limits
        if key == "voltage_controllable":
            return beam.high_voltage.is_controllable
        if key == "shift":  # beam shift
            return Point(beam.beam_shift.value.x, beam.beam_shift.value.y)
        if key == "stigmation":
            return Point(beam.stigmator.value.x, beam.stigmator.value.y)
        if key == "resolution":
            resolution = beam.scanning.resolution.value
            width, height = (
                int(resolution.split("x")[0]),
                int(resolution.split("x")[-1]),
            )
            return [width, height]

        # system properties
        if key == "eucentric_height":
            if beam_type is BeamType.ELECTRON:
                return self.system.electron.eucentric_height
            elif beam_type is BeamType.ION:
                return self.system.ion.eucentric_height
            else:
                raise ValueError(f"Unknown beam type: {beam_type} for {key}")

        if key == "column_tilt":
            if beam_type is BeamType.ELECTRON:
                return self.system.electron.column_tilt
            elif beam_type is BeamType.ION:
                return self.system.ion.column_tilt
            else:
                raise ValueError(f"Unknown beam type: {beam_type} for {key}")

        # electron beam properties
        if beam_type is BeamType.ELECTRON:
            if key == "angular_correction_angle":
                return beam.angular_correction.angle.value

        # ion beam properties
        if key == "plasma":
            if beam_type is BeamType.ION:
                return self.system.ion.plasma
            else:
                return False

        if key == "plasma_gas":
            if beam_type is BeamType.ION and self.system.ion.plasma:
                return (
                    beam.source.plasma_gas.value
                )  # might need to check if this is available?
            else:
                return None

        # stage properties
        if key == "stage_position":
            # get stage position in raw coordinates
            self.stage.set_default_coordinate_system(
                self._default_stage_coordinate_system
            )  # TODO: remove this once testing is done
            stage_position = stage_position_from_autoscript(
                self.stage.current_position
            )  # TODO: apply compucentric/raw coordinate system conversion here
            return stage_position

        if key == "stage_homed":
            return self.stage.is_homed
        if key == "stage_linked":
            return self.stage.is_linked

        # chamber properties
        if key == "chamber_state":
            return self.connection.vacuum.chamber_state

        if key == "chamber_pressure":
            return self.connection.vacuum.chamber_pressure.value

        # detector mode and type
        if key in [
            "detector_mode",
            "detector_type",
            "detector_brightness",
            "detector_contrast",
        ]:
            # set beam active view and device
            self.set_channel(beam_type)

            if key == "detector_type":
                return self.connection.detector.type.value
            if key == "detector_mode":
                return self.connection.detector.mode.value
            if key == "detector_brightness":
                return self.connection.detector.brightness.value
            if key == "detector_contrast":
                return self.connection.detector.contrast.value

        # manipulator properties
        if key == "manipulator_position":
            position = self.connection.specimen.manipulator.current_position
            return manipulator_position_from_autoscript(position)
        if key == "manipulator_state":
            state = self.connection.specimen.manipulator.state
            return True if state == ManipulatorState.INSERTED else False

        # manufacturer properties
        if key == "manufacturer":
            return self.system.info.manufacturer
        if key == "model":
            return self.system.info.model
        if key == "serial_number":
            return self.system.info.serial_number
        if key == "software_version":
            return self.system.info.software_version
        if key == "hardware_version":
            return self.system.info.hardware_version

        # logging.warning(f"Unknown key: {key} ({beam_type})")
        return None

    def _set(
        self,
        key: str,
        value: Union[str, int, float, BeamType, Point, FibsemRectangle],
        beam_type: Optional[BeamType] = None,
    ) -> None:
        """Set a property of the microscope."""
        # required for setting shift, stigmation
        from autoscript_sdb_microscope_client.structures import Point as ThermoPoint

        # get beam
        if beam_type is not None:
            beam = self._get_beam(beam_type)

        if key == "active_view":
            self.connection.imaging.set_active_view(
                value.value
            )  # the beam type is the active view (in ui)
            return
        if key == "active_device":
            self.connection.imaging.set_active_device(value.value)
            return

        # beam properties
        if key == "working_distance":
            beam.working_distance.value = value
            logging.info(f"{beam_type.name} working distance set to {value} m.")
            return
        if key == "current":
            beam.beam_current.value = value
            logging.info(f"{beam_type.name} current set to {value} A.")
            return
        if key == "voltage":
            beam.high_voltage.value = value
            logging.info(f"{beam_type.name} voltage set to {value} V.")
            return
        if key == "hfw":
            limits = beam.horizontal_field_width.limits
            value = np.clip(value, limits.min, limits.max - 10e-6)
            beam.horizontal_field_width.value = value
            logging.info(f"{beam_type.name} HFW set to {value} m.")
            return
        if key == "dwell_time":
            beam.scanning.dwell_time.value = value
            logging.info(f"{beam_type.name} dwell time set to {value} s.")
            return
        if key == "scan_rotation":
            beam.scanning.rotation.value = value
            logging.info(f"{beam_type.name} scan rotation set to {value} radians.")
            return
        if key == "shift":
            beam.beam_shift.value = ThermoPoint(
                value.x, value.y
            )  # TODO: resolve this coordinate system
            logging.info(f"{beam_type.name} shift set to {value}.")
            return
        if key == "stigmation":
            beam.stigmator.value = ThermoPoint(value.x, value.y)
            logging.info(f"{beam_type.name} stigmation set to {value}.")
            return

        if key == "resolution":
            resolution = f"{value[0]}x{value[1]}"  # WidthxHeight e.g. 1536x1024
            beam.scanning.resolution.value = resolution
            return

        # scanning modes
        if key == "reduced_area":
            beam.scanning.mode.set_reduced_area(
                left=value.left, top=value.top, width=value.width, height=value.height
            )
            return

        if key == "spot_mode":
            # value: Point, image pixels
            beam.scanning.mode.set_spot(x=value.x, y=value.y)
            return

        if key == "full_frame":
            beam.scanning.mode.set_full_frame()
            return

        # beam control
        if key == "on":
            beam.turn_on() if value else beam.turn_off()
            logging.info(f"{beam_type.name} beam turned {'on' if value else 'off'}.")
            return
        if key == "blanked":
            beam.blank() if value else beam.unblank()
            logging.info(
                f"{beam_type.name} beam {'blanked' if value else 'unblanked'}."
            )
            return

        # detector properties
        if key in [
            "detector_mode",
            "detector_type",
            "detector_brightness",
            "detector_contrast",
        ]:
            self.set_channel(beam_type)

            if key == "detector_mode":
                if value in self.connection.detector.mode.available_values:
                    self.connection.detector.mode.value = value
                    logging.info(f"Detector mode set to {value}.")
                else:
                    logging.warning(f"Detector mode {value} not available.")
                return
            if key == "detector_type":
                if value in self.connection.detector.type.available_values:
                    self.connection.detector.type.value = value
                    logging.info(f"Detector type set to {value}.")
                else:
                    logging.warning(f"Detector type {value} not available.")
                return
            if key == "detector_brightness":
                if 0 < value <= 1:
                    self.connection.detector.brightness.value = value
                    logging.info(f"Detector brightness set to {value}.")
                else:
                    logging.warning(
                        f"Detector brightness {value} not available, must be between 0 and 1."
                    )
                return
            if key == "detector_contrast":
                if 0 < value <= 1:
                    self.connection.detector.contrast.value = value
                    logging.info(f"Detector contrast set to {value}.")
                else:
                    logging.warning(
                        f"Detector contrast {value} not available, mut be between 0 and 1."
                    )
                return

        # system properties
        if key == "beam_enabled":
            if beam_type is BeamType.ELECTRON:
                self.system.electron.beam.enabled = value
                return
            elif beam_type is BeamType.ION:
                self.system.ion.beam.enabled = value
                return
            else:
                raise ValueError(f"Unknown beam type: {beam_type} for {key}")
            return

        if key == "eucentric_height":
            if beam_type is BeamType.ELECTRON:
                self.system.electron.eucentric_height = value
                return
            elif beam_type is BeamType.ION:
                self.system.ion.eucentric_height = value
                return
            else:
                raise ValueError(f"Unknown beam type: {beam_type} for {key}")

        if key == "column_tilt":
            if beam_type is BeamType.ELECTRON:
                self.system.electron.column_tilt = value
                return
            elif beam_type is BeamType.ION:
                self.system.ion.column_tilt = value
                return
            else:
                raise ValueError(f"Unknown beam type: {beam_type} for {key}")

        # ion beam properties
        if key == "plasma":
            if beam_type is BeamType.ION:
                self.system.ion.plasma = value
                return

        # electron beam properties
        if beam_type is BeamType.ELECTRON:
            if key == "angular_correction_angle":
                beam.angular_correction.angle.value = value
                logging.info(f"Angular correction angle set to {value} radians.")
                return

            if key == "angular_correction_tilt_correction":
                beam.angular_correction.tilt_correction.turn_on() if value else beam.angular_correction.tilt_correction.turn_off()
                return

        # ion beam properties
        if beam_type is BeamType.ION:
            if key == "plasma_gas":
                if not self.system.ion.plasma:
                    logging.debug("Plasma gas cannot be set on this microscope.")
                    return
                if not self.check_available_values("plasma_gas", [value], beam_type):
                    logging.warning(
                        f"Plasma gas {value} not available. Available values: {self.get_available_values('plasma_gas', beam_type)}"
                    )

                logging.info(
                    f"Setting plasma gas to {value}... this may take some time..."
                )
                beam.source.plasma_gas.value = value
                logging.info(f"Plasma gas set to {value}.")

                return

        # stage properties
        if key == "stage_home":
            logging.info("Homing stage...")
            self.stage.home()
            logging.info("Stage homed.")
            return

        if key == "stage_link":
            if self.stage_is_compustage:
                logging.debug("Compustage does not support linking.")
                return

            logging.info("Linking stage...")
            self.stage.link() if value else self.stage.unlink()
            logging.info(f"Stage {'linked' if value else 'unlinked'}.")
            return

        # chamber properties
        if key == "pump_chamber":
            if value:
                logging.info("Pumping chamber...")
                self.connection.vacuum.pump()
                logging.info("Chamber pumped.")
                return
            else:
                logging.warning(f"Invalid value for pump_chamber: {value}.")
                return

        if key == "vent_chamber":
            if value:
                logging.info("Venting chamber...")
                self.connection.vacuum.vent()
                logging.info("Chamber vented.")
                return
            else:
                logging.warning(f"Invalid value for vent_chamber: {value}.")
                return

        # patterning
        if key == "patterning_mode":
            if value in ["Serial", "Parallel"]:
                self.connection.patterning.mode = value
                logging.info(f"Patterning mode set to {value}.")
                return

        logging.warning(f"Unknown key: {key} ({beam_type})")

        return

    def check_available_values(
        self, key: str, values: list, beam_type: Optional[BeamType] = None
    ) -> bool:
        """Check if the given values are available for the given key."""

        available_values = self.get_available_values(key, beam_type)

        if available_values is None:
            return False

        for value in values:
            if value not in available_values:
                return False

            if isinstance(value, float):
                if value < min(available_values) or value > max(available_values):
                    return False
        return True

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

    def _get_compucentric_rotation_offset(self) -> FibsemStagePosition:
        """Get the difference between the stage position in specimen coordinates and raw coordinates."""
        # no offset for compustage
        if self.stage_is_compustage:
            return FibsemStagePosition(x=0, y=0)

        # get stage position in speciemn coordinates
        self.stage.set_default_coordinate_system(CoordinateSystem.SPECIMEN)
        specimen_stage_position = stage_position_from_autoscript(
            self.stage.current_position
        )

        # get stage position in raw coordinates
        self.stage.set_default_coordinate_system(CoordinateSystem.RAW)
        raw_stage_position = stage_position_from_autoscript(self.stage.current_position)

        # calculate the offset
        offset = specimen_stage_position - raw_stage_position  # XY only

        # restore stage coordinate system
        self.stage.set_default_coordinate_system(self._default_stage_coordinate_system)

        return offset

    def run_sputter_coater(self, time_seconds: int) -> None:
        """Run the sputter coater for a given time in seconds.
        Args:
            time_seconds (int): The time to run the sputter coater in seconds.
        Returns:
            None
        Raises:
            NotImplementedError: If the system is not an Arctis system.
        """

        if not hasattr(self.connection.specimen, "sputter_coater"):
            raise NotImplementedError(
                "Sputter coater not available on this microscope."
            )

        # check if system is Arctis
        if "Arctis" not in self.system.info.model:
            self.connection.specimen.sputter_coater.run(time_seconds)
            return

        # Prepare for sputtering
        self.connection.specimen.sputter_coater.prepare()

        # Change chamber pressure to 20 Pa and sputter current to 10 mA
        # self.connection.vacuum.pump(VacuumSettings(pressure=20))
        # self.connection.specimen.sputter_coater.current.value = 0.01
        # Perform sputtering procedure with 10 second run time
        self.connection.specimen.sputter_coater.run(time_seconds)

        # Recover from sputtering
        self.connection.specimen.sputter_coater.recover()
