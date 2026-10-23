"""The TESCAN SDK, imported once and guarded, and what the driver's modules
share: its presets, limits and stage units.

The driver's modules read the SDK's names as ``sdk.Detector`` and so on, at call
time, so a test can stand a fake SDK in here.
"""

import logging
import sys
from typing import Dict, Optional, Tuple

import fibsem.constants as constants
from fibsem.structures import BeamType, FibsemStagePosition, MillingState

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
