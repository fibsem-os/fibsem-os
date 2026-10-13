import logging
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import fibsem.constants as constants
from fibsem.devices.entries import build_device_entries, resolve_system_devices
from fibsem.drivers.tescan.connection import TescanConnection
from fibsem.drivers.tescan.devices import STAGE_FRAME_TESCAN
from fibsem.microscope import (
    FibsemMicroscope,
)

TESCAN_API_AVAILABLE = False
# Read through this rather than importing tescanautomation yourself: the guarded
# import below is also what strips the SDK's PySide6 GUI modules out of
# sys.modules, and that has to have happened first.
TESCAN_API_VERSION: Optional[str] = None
TESCAN_PRESERVE_SETTINGS_ON_PRESET_CHANGE = True  # Restore rotation/FOV/shift across preset changes, if false, use the values stored in the preset
SPOT_BURN_PRESET = "30 keV; 100 pA"  # Beam conditions used for spot burning
DEFAULT_IMAGING_PRESET = (
    "30 keV; 10 pA"  # Fallback for finish_milling when no preset was snapshotted
)
# The DrawBeam scanning paths a pattern can take.
TESCAN_SCAN_DIRECTIONS = (
    "ZigZag",
    "Flyback",
    "RLE",
    "SpiralInsideOut",
    "SpiralOutsideIn",
)

try:
    import tescanautomation
    from tescanautomation import Automation

    # Bpp, Detector, DepthUnit and IEtching are read through this module by devices.py
    # and services.py.
    from tescanautomation.Common import Bpp, Detector, Document  # noqa: F401
    from tescanautomation.DrawBeam import DepthUnit, IEtching  # noqa: F401
    from tescanautomation.DrawBeam import Status as DBStatus

    sys.modules.pop("tescanautomation.GUI")
    sys.modules.pop("tescanautomation.pyside6gui")
    sys.modules.pop("tescanautomation.pyside6gui.imageViewer_private")
    sys.modules.pop("tescanautomation.pyside6gui.infobar_private")
    sys.modules.pop("tescanautomation.pyside6gui.infobar_utils")
    sys.modules.pop("tescanautomation.pyside6gui.rc_GUI")
    sys.modules.pop("tescanautomation.pyside6gui.workflow_private")
    sys.modules.pop("PySide6.QtCore")
    TESCAN_API_AVAILABLE = True
    # getattr, not attribute access: this runs inside the try, so an SDK build
    # without __version__ would otherwise be caught below and disable the whole
    # Tescan backend over a missing version string.
    TESCAN_API_VERSION = getattr(tescanautomation, "__version__", None)
except Exception as e:
    logging.debug(f"Automation (TESCAN) not installed. {e}")

from fibsem.milling.base import FibsemMillingStage
from fibsem.structures import (  # noqa
    ACTIVE_MILLING_STATES,
    BeamSystemSettings,
    BeamType,
    CrossSectionPattern,
    DeviceEntry,
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemExperimentRef,
    FibsemHardwareGeometry,
    FibsemImage,
    FibsemLineSettings,
    FibsemMillingSettings,
    FibsemPolygonSettings,
    FibsemRectangle,
    FibsemRectangleSettings,
    FibsemStagePosition,
    FibsemUser,
    ImageSettings,
    MillingState,
    Point,
    SystemSettings,
)


def from_tescan_stage_position(position: Tuple[float]) -> FibsemStagePosition:
    """Convert a Tescan stage position to a FibsemStagePosition object."""
    x, y, z, r, t = position[:5]  # stage can be up to 6D

    stage_position = FibsemStagePosition(
        x=x * constants.MILLIMETRE_TO_METRE,
        y=y * constants.MILLIMETRE_TO_METRE,
        z=z * constants.MILLIMETRE_TO_METRE,
        r=r * constants.DEGREES_TO_RADIANS,
        t=t * constants.DEGREES_TO_RADIANS,
        coordinate_system="RAW",
    )
    return stage_position


def to_tescan_stage_position(position: FibsemStagePosition) -> Tuple[float]:
    """Convert a FibsemStagePosition object to a Tescan stage position."""
    x = position.x * constants.METRE_TO_MILLIMETRE if position.x is not None else None
    y = position.y * constants.METRE_TO_MILLIMETRE if position.y is not None else None
    z = position.z * constants.METRE_TO_MILLIMETRE if position.z is not None else None
    r = position.r * constants.RADIANS_TO_DEGREES if position.r is not None else None
    t = position.t * constants.RADIANS_TO_DEGREES if position.t is not None else None
    return x, y, z, r, t


def coincident_from_sem_stage_movement_tescan_from_geometry(
    geometry: FibsemHardwareGeometry,
    stage_position: FibsemStagePosition,
    dy: float,
) -> Tuple[float, float]:
    """Tescan coincidence correction from the SEM view — the single source of the math.

    The SEM-view mirror of ``vertical_move``: slide the sample along the FIB line
    of sight, which is invisible in the FIB image, until the offset seen in the
    SEM closes. A feature already positioned in the FIB view therefore lands on
    both beam axes at once — at the coincidence point. Both gestures move the
    sample by dy/sin(angle between the two axes), symmetric in the two
    directions.

    The FIB axis is chamber-fixed, so the sample plane — and with it the shuttle
    pre-tilt — cancels out of this move entirely; only the stage tilt (the
    y-axis rides the tilt module) and the column tilts appear:

        y_move = dy * sin(fib) / (cos(tilt) * sin(fib - sem))
        z_move = dy * cos(fib - tilt) / (cos(tilt) * sin(fib - sem))

    which for a vertical SEM column (sem = 0) reduces to y = dy/cos(tilt),
    z = dy*(tan(tilt) + cot(fib)). See
    https://linear.app/fibsemos/document/tescan-sample-plane-stage-movement-stable-move-derivation-ae56d0f2c414
    for the derivation, the figures, and how the signs are pinned by the two
    hardware-verified moves.

    Args:
        geometry: fixed instrument geometry (only the column tilts are used —
            the pre-tilt and rotation references drop out of this move).
        stage_position: stage pose (tilt t, radians).
        dy: offset along the image y-axis in the SEM view, in metres.

    Returns:
        (y_move, z_move) in metres — before the stage-axis inversion the caller
        applies (``y_stage = -y_move``, z unchanged), in Tescan's own frame.
    """
    stage_tilt = stage_position.t if stage_position.t is not None else 0.0
    sem_column_tilt = np.deg2rad(geometry.column_tilt)
    fib_column_tilt = np.deg2rad(geometry.fib_column_tilt)

    # distance along the FIB axis that closes a SEM-view offset of dy: one
    # beam axis, seen from the other, is foreshortened by sin(angle between them)
    axis_move = dy / np.sin(fib_column_tilt - sem_column_tilt)

    y_move = axis_move * np.sin(fib_column_tilt) / np.cos(stage_tilt)
    z_move = axis_move * np.cos(fib_column_tilt - stage_tilt) / np.cos(stage_tilt)
    return float(y_move), float(z_move)


FibsemStagePosition.from_tescan_stage_position = from_tescan_stage_position

try:
    DrawBeamStatusToPatterningState = {
        DBStatus.ProjectNotLoaded: MillingState.IDLE,
        DBStatus.ProjectLoadedExpositionIdle: MillingState.IDLE,
        DBStatus.ProjectLoadedExpositionInProgress: MillingState.RUNNING,
        DBStatus.ProjectLoadedExpositionPaused: MillingState.PAUSED,
        DBStatus.Unknown: MillingState.ERROR,
    }
except Exception:
    pass

SEM_LIMITS: Dict[str, Tuple] = {
    "hfw": (1.0e-6, 2580.0e-6),
}
FIB_LIMITS: Dict[str, Tuple] = {
    "hfw": (1.0e-6, 450.0e-6),
}

LIMITS = {
    BeamType.ELECTRON: SEM_LIMITS,
    BeamType.ION: FIB_LIMITS,
}


# A current token inside a free-form TESCAN preset name, e.g. "30 keV; 100 pA" or
# "30 keV; 2nA; my cool preset". Only prefixed units (pA/nA/uA/µA): a bare "A" in an
# arbitrary name (e.g. "slot 2A") is far more likely noise than a beam current.
_PRESET_CURRENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([pnuµ])A(?![a-zA-Z])")
_SI_CURRENT_PREFIX = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6}


def parse_current_from_preset(preset: Optional[str]) -> Optional[float]:
    """Parse the beam current (in A) out of a TESCAN preset name, or None.

    Preset names are free-form on the instrument, but conventionally embed the
    beam conditions ("30 keV; 100 pA"). The first current-looking token wins.
    """
    if not preset:
        return None
    match = _PRESET_CURRENT_RE.search(preset)
    if match is None:
        return None
    return float(match.group(1)) * _SI_CURRENT_PREFIX[match.group(2)]


def estimate_preset_milling_time(stage: FibsemMillingStage) -> Optional[float]:
    """Dose-model estimate t = volume / (rate × current) for a preset-driven stage.

    The same inputs DrawBeam computes the real exposure from: the stage's own
    (per-material) etch rate and the current embedded in the preset name. The
    shared sputter-rate table is a silicon calibration keyed on a current field
    TESCAN milling ignores. Returns None (the caller falls back to the table)
    when the preset carries no parseable current or the rate is unusable.
    """
    pattern_time = getattr(stage.pattern, "time", 0)
    if pattern_time:
        return pattern_time

    current = parse_current_from_preset(stage.milling.preset)
    rate = stage.milling.rate  # m³/A/s
    if current is None or current <= 0 or not rate or rate <= 0:
        return None

    volume = stage.pattern.volume  # m³
    if (
        hasattr(stage.pattern, "cross_section")
        and stage.pattern.cross_section is CrossSectionPattern.CleaningCrossSection
    ):
        volume *= 0.66  # ccs is approx 2/3 of the volume of a rectangle
    return volume / (rate * current)


# The device types each connect step builds (``TescanMicroscope._build_devices``).
# The FM is built on its own path; any other type a configuration adds is built last.
_BEAM_TYPES = ("beam",)
_STAGE_TYPES = ("stage",)
_OWN_TYPES = _BEAM_TYPES + _STAGE_TYPES + ("fm",)


class TescanMicroscope(FibsemMicroscope):
    """
    A class representing a TESCAN FIB-SEM microscope.

    This class inherits from the abstract base class `FibsemMicroscope`, which defines the core functionality of a
    microscope. In addition to the methods defined in the base class, this class provides additional methods specific
    to the TESCAN FIB-SEM microscope.
    """

    driver_name = "Tescan"
    vertical_move_views = (BeamType.ION, BeamType.ELECTRON)

    #: fibsem does not drive the Tescan Nanomanipulator (its support was removed), so
    #: the backend reports none and the manipulator methods are the base class's,
    #: which raise.
    DEFAULT_FITTED = {**FibsemMicroscope.DEFAULT_FITTED, "manipulator": False}

    # The SharkSEM connection, from connect until disconnect. The devices and services
    # are handed it; nothing else reaches the SDK.
    _connection: Optional[TescanConnection] = None

    @staticmethod
    def estimate_stage_milling_time(stage: FibsemMillingStage) -> Optional[float]:
        # TESCAN milling is preset-driven: the dose model (stage rate x preset
        # current), not the shared table keyed on the unused milling_current.
        return estimate_preset_milling_time(stage)

    def __init__(self, system_settings: SystemSettings):
        if not TESCAN_API_AVAILABLE:
            raise ImportError(
                "The TESCAN Automation API is not available. Please see the user guide for installation instructions."
            )

        # initialise system settings
        self.system: SystemSettings = system_settings
        self.milling_channel: BeamType = BeamType.ION

        # user, experiment metadata
        # TODO: remove once db integrated
        self.user = FibsemUser.from_environment()
        self.experiment = FibsemExperimentRef()

        # initialise last images
        self.last_image_eb: Optional[FibsemImage] = None
        self.last_image_ib: Optional[FibsemImage] = None

        # logging
        logging.debug(
            {
                "msg": "create_microscope_client",
                "system_settings": system_settings.to_dict(),
            }
        )

    def disconnect(self) -> None:
        self._connection.disconnect()
        self._connection = None

    def connect_to_microscope(
        self,
        ip_address: str = "localhost",
        port: int = 8300,
        reset_beam_shift: bool = True,
    ) -> None:
        """
        Connects to a microscope with the specified IP address and port.

        Args:
            ip_address: ip address of the microscope server (default: localhost).
            port: port of the microscope server (default 8300).
            reset_beam_shift: Whether to reset beam shifts on connect (default: True).
        """
        logging.info(f"Microscope client connecting to [{ip_address}:{port}]")
        self._connection = TescanConnection(
            Automation(ip_address, port), ip_address, port
        )
        logging.info(f"Microscope client connected to [{ip_address}:{port}]")

        # the beams and stage as devices, before anything below goes through them
        self._build_beams()
        self._build_stage()
        self._build_milling()
        self._build_stage_movement()
        self._build_spot_burn()
        self._build_imaging()
        # whatever else the configuration adds, such as a device on its own PC
        self._build_devices([], exclude_types=_OWN_TYPES)
        # Tescan's own driver has no FM; one on its own PC is built from its entry.
        self.fm = self._build_fluorescence(self.driver_name)
        self._apply_fluorescence_calibration()
        self._bind_device_roles()

        # the default detectors: SE, or E-T on an electron column without one
        electron = self.beams.get(BeamType.ELECTRON)
        if electron is not None:
            names = [d.name for d in electron.list_detectors()]
            self.set_detector_type("SE" if "SE" in names else "E-T", BeamType.ELECTRON)
        if BeamType.ION in self.beams:
            self.set_detector_type("SE", BeamType.ION)

        # system info: the hardware reports "TESCAN" in image headers; store the
        # canonical spelling so downstream comparisons never meet the raw form
        self.system.info.manufacturer = self.driver_name
        info = self.system.info
        logging.info(
            f"Microscope client connected to model {info.model} with serial number {info.serial_number} and software version {info.software_version}."
        )

        # reset beam shifts
        if reset_beam_shift:
            self.reset_beam_shifts()

        # create sample stage holder (needed by UI widgets)
        try:
            self._build_sample_handling()
        except Exception as e:
            logging.warning(f"Could not build the sample handling: {e}")

        logging.debug(
            {
                "msg": "connect_to_microscope",
                "ip_address": ip_address,
                "port": port,
                "system_info": self.system.info.to_dict(),
            }
        )

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

        A parameter a beam does not have (the ion column's working distance,
        ``detector_mode``) is unsupported; their wrappers answer for themselves. A
        disabled column gets no device, and its wrappers read None.
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
        from fibsem.drivers.tescan.services import bind_tescan_milling

        self.milling = bind_tescan_milling(self)

    def _build_stage_movement(self, service: Optional[type] = None) -> None:
        """Build the stage movement service, with Tescan's own coincidence move from
        the SEM view (`TescanStageMovement`)."""
        from fibsem.drivers.tescan.services import TescanStageMovement

        super()._build_stage_movement(service or TescanStageMovement)

    def _build_spot_burn(self) -> None:
        """Build the spot burn service over the milling service; `run_spot_burn` then
        goes to it. Without milling there is none, and the method below burns."""
        from fibsem.drivers.tescan.services import bind_tescan_spot_burn

        self.spot_burn = bind_tescan_spot_burn(self)

    def _build_stage(self) -> None:
        """Build the stage device.

        The absolute and relative moves then go through the device, and so do the
        view-corrected moves, which end in them. The device has no ``homed``, so it
        reads None. A disabled stage gets no device, and then
        the stage cannot be read or moved.
        """
        self._build_devices([DeviceEntry(name="stage", type="stage")], _STAGE_TYPES)

    def is_working_distance_settable(self, beam_type: BeamType) -> bool:
        """ION working distance is not settable: the SDK's FIB class has no WD or
        focus control anywhere (FIB.Optics carries only rotation/shift/viewfield) --
        ion focus on TESCAN is preset-driven. Writing the ion working distance is a
        best-effort no-op so state restores keep working; anything that *depends* on
        the write landing (the autofocus sweep) must gate on this instead."""
        return beam_type is not BeamType.ION

    # What the Tescan API has no control for. These are not keys: the wrappers answer
    # for themselves, quietly, because state restores call them every time.

    def get_working_distance(self, beam_type: BeamType) -> Optional[float]:
        if beam_type is BeamType.ION:
            return None  # the FIB has no working distance in the API
        return super().get_working_distance(beam_type)

    def set_working_distance(self, wd: float, beam_type: BeamType) -> Optional[float]:
        if beam_type is BeamType.ION:
            logging.debug("The Tescan API has no ion working distance; not set.")
            return None
        return super().set_working_distance(wd, beam_type)

    def get_detector_mode(self, beam_type: BeamType) -> Optional[str]:
        return None  # not in the Tescan API

    def set_detector_mode(self, mode: str, beam_type: BeamType) -> Optional[str]:
        logging.debug("The Tescan API has no detector mode; not set.")
        return None

    @property
    def stage_frame(self) -> str:
        """fibsem's frame through the stage device; Tescan's own without one."""
        return self.stage.frame if self.stage is not None else STAGE_FRAME_TESCAN

    def move_coincident_from_sem(self, dx: float, dy: float) -> FibsemStagePosition:
        """Correct the coincidence point from the SEM view.

        Deprecated: call ``vertical_move(dy, dx, beam_type=BeamType.ELECTRON)``.
        Kept for one release because custom scripts may call it.
        """
        return self.vertical_move(dy=dy, dx=dx, beam_type=BeamType.ELECTRON)

    def beam_uses_presets(self, beam_type: BeamType) -> bool:
        """The ion column is set by preset: setting its voltage or current directly
        is refused by the Tescan API."""
        return beam_type is BeamType.ION

    def home(self) -> Optional[bool]:
        """A Tescan cannot home from here, so whether it is homed is unknown."""
        logging.warning("No homing available, please use native UI.")
        return None
