import datetime
import logging
import os
import re
import sys
import threading
import time
from copy import deepcopy
from queue import Queue
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import fibsem.constants as constants
from fibsem import manufacturers
from fibsem.devices.beam import BEAM_ROUTES, STAGE_ROUTES
from fibsem.devices.entries import build_device_entries, resolve_system_devices
from fibsem.microscope import (
    FibsemMicroscope,
    _records_beam_shift,
    _records_stage_move,
)
from fibsem.services.milling import ServiceMilling

TESCAN_API_AVAILABLE = False
# Read through this rather than importing tescanautomation yourself: the guarded
# import below is also what strips the SDK's PySide6 GUI modules out of
# sys.modules, and that has to have happened first.
TESCAN_API_VERSION: Optional[str] = None
TESCAN_BEAM_READY_TIMEOUT = 60  # Max time in seconds to wait for the beam to become ready (busy-wait when using Tescanautomation API)
TESCAN_PRESERVE_SETTINGS_ON_PRESET_CHANGE = True  # Restore rotation/FOV/shift across preset changes, if false, use the values stored in the preset
# Seconds to wait before an ion image that directly follows an electron image. The
# reference pair has always paused here (3 s, then 1 s since #341); why was never
# recorded, so the pause is kept as it was rather than replaced by a status wait.
TESCAN_ELECTRON_TO_ION_SETTLE_TIME = 1
SPOT_BURN_POLL_INTERVAL = (
    1  # Seconds between DrawBeam status polls while a spot is exposing
)
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
    from tescanautomation.Common import Bpp, Detector, Document
    from tescanautomation.DrawBeam import DepthUnit, IEtching
    from tescanautomation.DrawBeam import Status as DBStatus
    from tescanautomation.SEM import HVBeamStatus as SEMStatus

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

from fibsem.imaging.spot import (
    SpotBurnProgress,
    SpotBurnSettings,
    SpotBurnStatus,
)
from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.progress import MillingProgress, MillingProgressStatus
from fibsem.structures import (  # noqa
    ACTIVE_MILLING_STATES,
    STAGE_FRAME_TESCAN,
    BeamSettings,
    BeamSystemSettings,
    BeamType,
    CrossSectionPattern,
    DeviceEntry,
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemDetectorSettings,
    FibsemExperimentRef,
    FibsemHardwareGeometry,
    FibsemImage,
    FibsemImageMetadata,
    FibsemLineSettings,
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
    SystemSettings,
)
from fibsem.util.timestamps import from_posix


def _get_beam_settings_from_tescan_md(md: dict, beam_type: BeamType) -> BeamSettings:
    """Parse metadata from Tescan image header to get beam settings."""
    return BeamSettings(
        beam_type=beam_type,
        working_distance=float(md["WD"]),
        voltage=float(md["HV"]),
        beam_current=float(md["PredictedBeamCurrent"]),
        dwell_time=float(md["DwellTime"]),
        scan_rotation=float(md["ScanRotation"]) * constants.DEGREES_TO_RADIANS,
        stigmation=Point(x=float(md["StigmatorX"]), y=float(md["StigmatorY"])),
        shift=Point(x=float(md["ImageShiftX"]), y=float(md["ImageShiftY"])),
        preset=md.get("LastPreset", None),
    )


def _get_detector_settings_from_tescan_md(md: dict) -> FibsemDetectorSettings:
    """Parse metadata from Tescan image header to get detector settings."""
    return FibsemDetectorSettings(
        type=md["Detector0"],
        brightness=float(md["Detector0Gain"]) / 100.0,
        contrast=float(md["Detector0Offset"]) / 100.0,
    )


def _get_pixel_size_from_tescan_md(md: dict) -> Point:
    """Parse metadata from Tescan image header to get pixel size."""
    pixelsize = Point(float(md["MAIN"]["PixelSizeX"]), float(md["MAIN"]["PixelSizeY"]))
    return pixelsize


def _get_microscope_state_from_tescan_md(
    md: dict, image_shape: Tuple[int, int]
) -> MicroscopeState:
    """Parse metadata from Tescan image header to create a MicroscopeState object."""
    ddict = {}
    SUPPORTED_KEYS = ["MAIN", "FIB", "SEM"]

    for k in md:
        if k in SUPPORTED_KEYS:
            ddict[k] = dict(md[k])

    if "FIB" in ddict:
        beam_type = BeamType.ION
        k = "FIB"
    if "SEM" in ddict:
        beam_type = BeamType.ELECTRON
        k = "SEM"

    # stage position
    stage_position = FibsemStagePosition(
        x=float(ddict[k]["StageX"]),
        y=float(ddict[k]["StageY"]),
        z=float(ddict[k]["StageZ"]),
        r=float(ddict[k]["StageRotation"]) * constants.DEGREES_TO_RADIANS,
        t=float(ddict[k]["StageTilt"]) * constants.DEGREES_TO_RADIANS,
        coordinate_system="RAW",
    )

    # fov must be calc manually from pixelsize * resolution
    pixelsize = _get_pixel_size_from_tescan_md(ddict)
    resolution = image_shape[1], image_shape[0]
    hfw = pixelsize.x * resolution[0]

    # default values
    electron_beam = BeamSettings(beam_type=BeamType.ELECTRON)
    ion_beam = BeamSettings(beam_type=BeamType.ION)
    electron_detector = FibsemDetectorSettings()
    ion_detector = FibsemDetectorSettings()

    # beam settings
    if beam_type is BeamType.ION:
        ion_detector = _get_detector_settings_from_tescan_md(ddict[k])
        ion_beam = _get_beam_settings_from_tescan_md(ddict[k], BeamType.ION)
        ion_beam.hfw = hfw
        ion_beam.resolution = resolution
    if beam_type is BeamType.ELECTRON:
        electron_detector = _get_detector_settings_from_tescan_md(ddict[k])
        electron_beam = _get_beam_settings_from_tescan_md(ddict[k], BeamType.ELECTRON)
        electron_beam.hfw = hfw
        electron_beam.resolution = resolution

    # acquisition timestamp
    acquisition_time = ddict["MAIN"]["Date"] + " " + ddict["MAIN"]["Time"]
    timestamp = datetime.datetime.strptime(
        acquisition_time, "%Y-%m-%d %H:%M:%S"
    ).timestamp()

    ms = MicroscopeState(
        timestamp=timestamp,
        stage_position=stage_position,
        electron_beam=electron_beam,
        ion_beam=ion_beam,
        electron_detector=electron_detector,
        ion_detector=ion_detector,
    )
    return ms


def fromTescanImage(
    image: "Document", image_settings: ImageSettings = None
) -> FibsemImage:
    """Create a FibsemImage object from a Tescan image document."""
    image_data = np.array(image.Image)
    pixelsize = _get_pixel_size_from_tescan_md(image.Header)
    ms = _get_microscope_state_from_tescan_md(
        image.Header, image_shape=image_data.shape
    )
    if image_settings is None:
        image_settings = ImageSettings()

    md = FibsemImageMetadata(
        image_settings=image_settings,
        microscope_state=ms,
        pixel_size=pixelsize,
        # The header's date and time, the instrument's clock, read as this machine's
        # local time when the state was built (FIB-1190).
        acquisition_datetime=from_posix(ms.timestamp),
    )

    return FibsemImage(data=image_data, metadata=deepcopy(md))


def to_tescan_image_roi(
    rect: FibsemRectangle, image_shape: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """Convert a FibsemRectangle to a Tescan image ROI (left, top, right, bottom)."""
    image_width, image_height = image_shape
    left = int(rect.left * image_width)
    top = int(rect.top * image_height)
    right = int(left + rect.width * image_width - 1)
    bottom = int(top + rect.height * image_height - 1)
    return left, top, right, bottom


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
FibsemImage.fromTescanImage = fromTescanImage

try:
    DrawBeamStatusToPatterningState = {
        DBStatus.ProjectNotLoaded: MillingState.IDLE,
        DBStatus.ProjectLoadedExpositionIdle: MillingState.IDLE,
        DBStatus.ProjectLoadedExpositionInProgress: MillingState.RUNNING,
        DBStatus.ProjectLoadedExpositionPaused: MillingState.PAUSED,
        DBStatus.Unknown: MillingState.ERROR,
    }
except Exception as e:
    pass

# def printProgressBar(
#     value, total, prefix="", suffix="", decimals=0, length=100, fill="█"
# ):
#     """
#     terminal progress bar
#     """
#     percent = ("{0:." + str(decimals) + "f}").format(100 * (value / float(total)))
#     filled_length = int(length * value // total)
#     bar = fill * filled_length + "-" * (length - filled_length)
#     print(f"\r{prefix} |{bar}| {percent}% {suffix}", end="\r")


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


class TescanDrawBeam:
    """How a Tescan mills: a DrawBeam layer on the ion column, made from the milling
    preset. `fibsem.drivers.tescan.services.TescanMilling` mills with this code; it
    is the microscope's own milling code when there is no milling service."""

    def setup_milling(
        self,
        mill_settings: FibsemMillingSettings,
    ):
        """
        Configure the microscope for milling using the ion beam.

        Args:
            mill_settings (FibsemMillingSettings): Milling settings.

        """
        if mill_settings.milling_channel is not BeamType.ION:
            raise ValueError("Only FIB milling is currently supported.")

        self._prepare_beam(mill_settings.milling_channel)

        self.clear_patterns()

        self.milling_channel = mill_settings.milling_channel

        # Snapshot the imaging preset so finish_milling can put the column back where the
        # user left it. Only snapshot when one isn't already held: setup_milling sets the
        # milling preset itself, so calling it twice without an intervening finish_milling
        # would otherwise overwrite the snapshot with the milling preset.
        if getattr(self, "_preset_before_milling", None) is None:
            self._preset_before_milling = self.get_preset(BeamType.ION)
            logging.debug(
                f"Snapshot preset before milling: {self._preset_before_milling}"
            )

        self.set(
            "preset", mill_settings.preset, BeamType.ION
        )  # QUERY: do we need to set this here as it is also set in IEtching?

        layer_settings = IEtching(
            syncWriteField=False,
            writeFieldSize=mill_settings.hfw,
            beamCurrent=self.get_beam_current(self.milling_channel),
            spotSize=mill_settings.spot_size,
            rate=mill_settings.rate,
            dwellTime=mill_settings.dwell_time,
            parallel=bool(mill_settings.patterning_mode == "Parallel"),
            preset=mill_settings.preset,
            spacing=mill_settings.spacing,
        )

        # TODO: change the layer name to milling stage name
        with self._connection_lock:
            self.layer = self.connection.DrawBeam.Layer("Layer1", layer_settings)

    def finish_milling(
        self, imaging_current: float = None, imaging_voltage: float = None
    ):
        """
        Finalises the milling process by clearing the microscope of any patterns and returning the current to the imaging current.

        Args:
            imaging_current (float): The current to use for imaging in amps.
        #"""
        # Restore the preset that was active before milling, so the column goes back to the
        # imaging conditions the user was working at. Falls back to the module default when
        # no snapshot was taken (get("preset") can be None if no image has been acquired and
        # no preset set this session).
        preset = self._preset_before_milling or DEFAULT_IMAGING_PRESET
        self._preset_before_milling = None

        # Each cleanup step gets its own try: the preset activation is the fragile one, and
        # a failure there must not skip the (cheap, always-wanted) layer unload.
        try:
            # set_preset (not the raw Preset.Activate) so the rotation/FOV/shift preservation
            # added in #82 applies across the restore.
            self.set_preset(preset, BeamType.ION)
            logging.debug(f"Finished milling, restored preset to {preset}")
        except Exception as e:
            logging.warning(f"Error restoring preset {preset!r} in finish_milling: {e}")

        self.clear_patterns()

    def stop_milling(self):

        # TODO: improve thread safety to stop from another thread
        try:
            thread_connection = Automation(self.system.info.ip_address, port=8300)
            if (
                thread_connection.DrawBeam.GetStatus()[0]
                == DBStatus.ProjectLoadedExpositionInProgress
            ):
                logging.info("Milling is in progress, stopping now...")
                thread_connection.DrawBeam.Stop()
        except Exception as e:
            logging.error(f"Error in stop_milling: {e}")
        finally:
            del thread_connection

    def clear_patterns(self) -> None:
        """Unload the current DrawBeam layer, discarding any patterns it holds.

        Safe to call unconditionally: DrawBeam.UnloadLayer raises when no layer is loaded
        (and when an exposition is still active), which is swallowed here so callers can
        clear patterns without first tracking whether a layer exists.
        """
        try:
            with self._connection_lock:
                self.connection.DrawBeam.UnloadLayer()
        except Exception as e:
            logging.debug(f"Error unloading layer: {e}")

    def start_milling(self) -> None:
        with self._connection_lock:
            self.connection.DrawBeam.Start()

    def pause_milling(self):
        with self._connection_lock:
            self.connection.DrawBeam.Pause()

    def resume_milling(self):
        with self._connection_lock:
            self.connection.DrawBeam.Resume()

    def get_milling_state(self):
        with self._connection_lock:
            state = self.connection.DrawBeam.GetStatus()[0]
        return DrawBeamStatusToPatterningState[state]

    def estimate_milling_time(self) -> float:

        # NOTE: we cannot load the layer again
        # load and unload layer to check time
        # self.connection.DrawBeam.LoadLayer(self.layer)
        est_time = 0
        try:
            with self._connection_lock:
                est_time = self.connection.DrawBeam.EstimateTime()
        except Exception as e:
            logging.error(f"Error in estimating milling time: {e}")

        # self.connection.DrawBeam.UnloadLayer()

        return est_time

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
            AutomationError: if an error occurs while creating the pattern.

        Notes:
            The rectangle pattern will be centered at the specified coordinates (centre_x, centre_y) with the specified
            width, height and depth (in nm). If the cleaning_cross_section attribute of pattern_settings is True, a
            cleaning cross section pattern will be created instead of a rectangle pattern.

            The pattern will be rotated by the angle specified in the rotation attribute of pattern_settings (in degrees)
            and scanned in the direction specified in the scan_direction attribute of pattern_settings.

            The created pattern can be added to the patterning queue and executed using the layer methods in Automation.
        """
        if pattern_settings.scan_direction in TESCAN_SCAN_DIRECTIONS:
            scanning_path = pattern_settings.scan_direction
        else:
            scanning_path = "Flyback"
            logging.warning(
                f"Scan direction {pattern_settings.scan_direction} not supported. Using Flyback instead."
            )
        self.connection.DrawBeam.ScanningPath = scanning_path

        if pattern_settings.cross_section is CrossSectionPattern.CleaningCrossSection:
            add_pattern_fn = self.layer.addRectanglePolish
        else:
            add_pattern_fn = self.layer.addRectangleFilled

        add_pattern_fn(
            CenterX=pattern_settings.centre_x,
            CenterY=pattern_settings.centre_y,
            Depth=pattern_settings.depth,
            DepthUnit="m",
            Width=pattern_settings.width,
            Height=pattern_settings.height,
            Angle=pattern_settings.rotation * constants.RADIANS_TO_DEGREES,
            ScanningPath=scanning_path,
            # ExpositionFactor=passes
        )

        pattern = self.layer

        return pattern

    def draw_line(self, pattern_settings: FibsemLineSettings):
        """
        Draws a line pattern on the current imaging view of the microscope.

        Args:
            pattern_settings (FibsemLineSettings): A data class object specifying the pattern parameters,
                including the start and end points, and the depth of the pattern.

        Returns:
            LinePattern: A line pattern object, which can be used to configure further properties or to add the
                pattern to the milling list.
        """
        start_x = pattern_settings.start_x
        start_y = pattern_settings.start_y
        end_x = pattern_settings.end_x
        end_y = pattern_settings.end_y
        depth = pattern_settings.depth

        self.layer.addLine(
            BeginX=start_x,
            BeginY=start_y,
            EndX=end_x,
            EndY=end_y,
            Depth=depth,
            DepthUnit="m",
        )

        pattern = self.layer
        return pattern

    def draw_circle(self, pattern_settings: FibsemCircleSettings):
        """
        Draws a circle pattern on the current imaging view of the microscope.

        Args:
            pattern_settings (FibsemCircleSettings): A data class object specifying the pattern parameters,
                including the centre point, radius and depth of the pattern.

        Returns:
            CirclePattern: A circle pattern object, which can be used to configure further properties or to add the
                pattern to the milling list.

        """
        pattern = self.layer.addAnnulusFilled(
            CenterX=pattern_settings.centre_x,
            CenterY=pattern_settings.centre_y,
            RadiusA=pattern_settings.radius,
            RadiusB=0,
            Depth=pattern_settings.depth,
            DepthUnit="m",
        )

        return pattern

    def draw_annulus(self, pattern_settings: FibsemCircleSettings):
        """Draws an annulus (donut) pattern on the current imaging view of the microscope.

        Args:
            pattern_settings (FibsemCircleSettings): A data class object specifying the pattern parameters,
            including the centre point, outer radius and thickness of the annulus, and the depth of the pattern.

        Returns:
            annulus pattern object
        """
        outer_radius = pattern_settings.radius
        inner_radius = pattern_settings.radius - pattern_settings.thickness

        pattern = self.layer.addAnnulusFilled(
            CenterX=pattern_settings.centre_x,
            CenterY=pattern_settings.centre_y,
            RadiusA=outer_radius,
            RadiusB=inner_radius,
            Depth=pattern_settings.depth,
            DepthUnit="m",
        )

        return pattern

    def draw_bitmap_pattern(self, pattern_settings: FibsemBitmapSettings):
        return NotImplemented

    def draw_polygon(self, pattern_settings: FibsemPolygonSettings):
        raise NotImplementedError("draw_polygon not implemented for Tescan API")


# The device types each connect step builds (``TescanMicroscope._build_devices``);
# any other type a configuration adds is built last.
_BEAM_TYPES = ("beam",)
_STAGE_TYPES = ("stage",)
_OWN_TYPES = _BEAM_TYPES + _STAGE_TYPES


class TescanMicroscope(ServiceMilling, TescanDrawBeam, FibsemMicroscope):
    """
    A class representing a TESCAN FIB-SEM microscope.

    This class inherits from the abstract base class `FibsemMicroscope`, which defines the core functionality of a
    microscope. In addition to the methods defined in the base class, this class provides additional methods specific
    to the TESCAN FIB-SEM microscope.
    """

    vertical_move_views = (BeamType.ION, BeamType.ELECTRON)

    #: fibsem does not drive the Tescan Nanomanipulator (its support was removed), so
    #: the backend reports none and the manipulator methods are the base class's,
    #: which raise.
    DEFAULT_FITTED = {**FibsemMicroscope.DEFAULT_FITTED, "manipulator": False}

    # The beam of the last requested (non-live) acquisition, for the settle before an
    # ion image that follows an electron one.
    _last_requested_beam_type: Optional[BeamType] = None

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

        # create microscope client
        self.connection: Automation

        # The Tescan SharkSEM connection is a single socket and is NOT thread-safe.
        # UI signal callbacks (e.g. stage_position_changed -> update_ui) run on the
        # main thread while worker threads drive movements/acquisition, so two threads
        # can hit the socket at once and corrupt the byte stream (empty strings ->
        # ValueError, or None -> TypeError). Serialise all socket transactions.
        self._connection_lock = threading.RLock()

        # initialise system settings
        self.system: SystemSettings = system_settings
        self.milling_channel: BeamType = BeamType.ION
        self._preserve_settings_on_preset_change: bool = (
            TESCAN_PRESERVE_SETTINGS_ON_PRESET_CHANGE
        )
        self._last_imaging_settings: ImageSettings = ImageSettings()
        # preset active before milling started, restored by finish_milling
        self._preset_before_milling: Optional[str] = None

        # user, experiment metadata
        # TODO: remove once db integrated
        self.user = FibsemUser.from_environment()
        self.experiment = FibsemExperimentRef()

        # initialise last images
        self.last_image_eb: Optional[FibsemImage] = None
        self.last_image_ib: Optional[FibsemImage] = None

        # fluorescence microscope (not available on Tescan)
        self.fm = None

        # cached beam parameters
        # not all parameters are available via the api, so we cache them after acquiring image
        self._beam_parameters: Dict[BeamType, BeamSettings] = {
            BeamType.ELECTRON: BeamSettings(BeamType.ELECTRON),
            BeamType.ION: BeamSettings(BeamType.ION),
        }

        # logging
        logging.debug(
            {
                "msg": "create_microscope_client",
                "system_settings": system_settings.to_dict(),
            }
        )

    def disconnect(self) -> None:
        with self._connection_lock:
            self.connection.Disconnect()
        del self.connection
        self.connection = None

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
        self.connection = Automation(ip_address, port)
        logging.info(f"Microscope client connected to [{ip_address}:{port}]")

        # set up detectors
        self._default_detector_names = {BeamType.ELECTRON: "SE", BeamType.ION: "SE"}
        self._active_detector: Dict[BeamType, Detector] = {}

        # the beams and stage as devices, before anything below goes through them
        self._build_beams()
        self._build_stage()
        self._build_milling()
        # whatever else the configuration adds, such as a device on its own PC
        self._build_devices([], exclude_types=_OWN_TYPES)

        available_detectors = self._get_available_detectors(BeamType.ELECTRON)
        if self._default_detector_names[BeamType.ELECTRON] not in [
            d.name for d in available_detectors
        ]:
            self._default_detector_names[BeamType.ELECTRON] = "E-T"
        self.set_detector_type(
            self._default_detector_names[BeamType.ELECTRON], BeamType.ELECTRON
        )
        self.set_detector_type(self._default_detector_names[BeamType.ION], BeamType.ION)

        # system info: the hardware reports "TESCAN" in image headers; store the
        # canonical spelling so downstream comparisons never meet the raw form
        self.system.info.manufacturer = manufacturers.TESCAN
        info = self.system.info
        logging.info(
            f"Microscope client connected to model {info.model} with serial number {info.serial_number} and software version {info.software_version}."
        )

        # reset beam shifts
        if reset_beam_shift:
            self.reset_beam_shifts()

        # create sample stage holder (needed by UI widgets)
        try:
            self._create_sample_stage()
        except Exception as e:
            logging.warning(f"Could not create sample stage: {e}")

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
            self.system, defaults, types, exclude_types, manufacturers.TESCAN
        )
        built = build_device_entries(resolved, self)
        for name, device in built.items():
            self._set_device(name, device)
        return built

    def _build_beams(self) -> None:
        """Build the beam devices and route the beam keys to them.

        A key a beam does not have (the ion column's working distance,
        ``detector_mode``) is unknown; their wrappers answer for themselves. Every key
        of a disabled column, which gets no device, goes to ``_get``/``_set``.
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
        (``ServiceMilling``). Without an ion beam there is none, and they raise."""
        from fibsem.drivers.tescan.services import bind_tescan_milling

        self.milling = bind_tescan_milling(self)

    def _build_stage(self) -> None:
        """Build the stage device and route the stage keys to it.

        The absolute and relative moves then go through the device, and so do the
        view-corrected moves, which end in them. ``homed`` is not on the device, so its
        key still goes to ``_get``/``_set``. A disabled stage gets no device, and then
        the stage cannot be read or moved.
        """
        self._build_devices([DeviceEntry(name="stage", type="stage")], _STAGE_TYPES)
        if self.stage is not None:
            self._device_routes = MappingProxyType(
                {key: ("stage", name) for key, name in STAGE_ROUTES.items()}
            )

    @property
    def manufacturer(self) -> str:
        return manufacturers.TESCAN

    def acquire_image(
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
            image = microscope.acquire_image(image_settings=settings)

            # Acquire with current settings for a specific beam type
            image = microscope.acquire_image(beam_type=BeamType.ION)

            # If both provided, image_settings takes precedence
            image = microscope.acquire_image(image_settings=settings, beam_type=BeamType.ION)  # Uses settings
        """

        # Validate parameters - at least one must be provided
        if image_settings is None and beam_type is None:
            raise ValueError(
                "Must provide either image_settings (to acquire with specific settings) or beam_type (to acquire with current microscope settings for that beam type)."
            )
        # The beam's acquire command; image_settings takes precedence.
        target = image_settings.beam_type if image_settings is not None else beam_type
        return self._beam_device(target).acquire(image_settings)

    def last_image(self, beam_type: BeamType) -> FibsemImage:
        """
        Returns the last acquired image for the specified beam type.

        Args:
            beam_type (BeamType.ELECTRON or BeamType.ION): The type of beam used to acquire the image.

        Returns:
            FibsemImage: The last acquired image of the specified beam type.

        """
        return self._beam_device(beam_type).last_image()

    def autocontrast(
        self, beam_type: BeamType, reduced_area: FibsemRectangle = None
    ) -> None:
        """Automatically adjust the microscope image contrast for the specified beam type.

        Args:
            beam_type: The imaging beam type to adjust the contrast for.
        """
        self._beam_device(beam_type).autocontrast(reduced_area)

    def is_working_distance_settable(self, beam_type: BeamType) -> bool:
        """ION working distance is not settable: the SDK's FIB class has no WD or
        focus control anywhere (FIB.Optics carries only rotation/shift/viewfield) --
        ion focus on TESCAN is preset-driven. _set("working_distance", ION) is a
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

    def auto_focus(
        self, beam_type: BeamType, reduced_area: Optional[FibsemRectangle] = None
    ) -> None:
        device = self._beam_device(beam_type)
        if not device.commands["auto_focus"].available:
            logging.warning(
                f"Auto focus is not supported for {beam_type.name} in Tescan API"
            )
            return
        device.auto_focus(reduced_area)

    @_records_beam_shift
    def beam_shift(
        self, dx: float, dy: float, beam_type: BeamType = BeamType.ION
    ) -> None:
        """Adjusts the beam shift based on relative values that are provided.

        Args:
            self (FibsemMicroscope): Fibsem microscope object
            dx (float): the relative x term
            dy (float): the relative y term
        """
        # invert direction for scan rotated images...
        if np.isclose(self.get_scan_rotation(beam_type), np.pi):
            dx *= -1.0
            dy *= -1.0

        logging.info(f"{beam_type.name} shifting by ({dx}, {dy})")
        new_beam_shift = self.get_beam_shift(beam_type) + Point(dx, dy)
        self.set_beam_shift(new_beam_shift, beam_type)
        logging.debug(
            {"msg": "beam_shift", "dx": dx, "dy": dy, "beam_type": beam_type.name}
        )

    def _image_from_tescan(self, image, image_settings: ImageSettings) -> FibsemImage:
        """An acquired Tescan image, with its header's stage position in fibsem's frame.

        The header records the stage in Tescan's frame; the stage device converts it,
        as it converts every position it reads. Without a stage device the position is
        left as Tescan reported it, and ``stage_frame`` says so.
        """
        fibsem_image = fromTescanImage(image, image_settings)
        state = fibsem_image.metadata.microscope_state
        if self.stage is not None and state.stage_position is not None:
            state.stage_position = self.stage.from_native(state.stage_position)
        return fibsem_image

    @property
    def stage_frame(self) -> str:
        """fibsem's frame through the stage device; Tescan's own without one."""
        return self.stage.frame if self.stage is not None else STAGE_FRAME_TESCAN

    @_records_stage_move
    def safe_absolute_stage_movement(self, stage_position: FibsemStagePosition) -> None:
        # Inert until Tescan has a fluorescence microscope at all -- `self.fm` is set
        # to None unconditionally here (FIB-836) -- but the guard belongs on every
        # re-pose path, not only the ones that can reach it today.
        self._refuse_rotation_at_the_fluorescence_microscope(stage_position)

        # TODO: implement if required.
        self.move_stage_absolute(stage_position)

    def move_coincident_from_sem(self, dx: float, dy: float) -> FibsemStagePosition:
        """Correct the coincidence point from the SEM view.

        Deprecated: call ``vertical_move(dy, dx, beam_type=BeamType.ELECTRON)``.
        Kept for one release because custom scripts may call it.
        """
        return self.vertical_move(dy=dy, dx=dx, beam_type=BeamType.ELECTRON)

    def _vertical_move_from_sem(
        self, dx: float, dy: float, relaxation: float = 1.0
    ) -> FibsemStagePosition:
        """Correct the coincidence point from the SEM view.

        Tescan's own move, kept in place of the shared stable move then FIB-vertical
        move because it is the one verified on hardware. ``relaxation`` is not applied,
        as it never was here.

        The mirror of the FIB-view move: the stage slides along the FIB
        line of sight, which is invisible in the FIB image, until the clicked
        feature is centred in the SEM. A feature already positioned in the FIB
        view (e.g. just milled, or just corrected with vertical_move) therefore
        lands on both beam axes at once -- at the coincidence point. The math
        is :func:`coincident_from_sem_stage_movement_tescan_from_geometry`; the sample plane, and with it
        the shuttle pre-tilt, cancels out of this move, so only the stage tilt
        and the column tilts appear. See
        https://linear.app/fibsemos/document/tescan-sample-plane-stage-movement-stable-move-derivation-ae56d0f2c414
        for the derivation and figures.

        Verified on hardware 2026-08-26 (acceptance test: Alt-double-click a
        feature in the SEM view lands it centred in the SEM image with no
        movement in the FIB image). Small focus shifts in both views are
        inherent to the move.

        Args:
            dx (float): distance along the image x-axis (SEM view), in metres.
            dy (float): distance along the image y-axis (SEM view), in metres.
        """
        # adjust for scan rotation (radians, codebase convention)
        scan_rotation = self.get_scan_rotation(BeamType.ELECTRON)
        if np.isclose(scan_rotation, np.pi):
            dx *= -1.0
            dy *= -1.0

        y_move, z_move = coincident_from_sem_stage_movement_tescan_from_geometry(
            geometry=self.hardware_geometry(),
            stage_position=self.get_stage_position(),
            dy=dy,
        )

        # The move in Tescan's frame: x and y run opposite the image, z as computed
        # (+z is down). The stage device takes fibsem's frame, at the current tilt.
        stage_position = FibsemStagePosition(x=-dx, y=-y_move, z=z_move, r=0, t=0)
        if self.stage is not None:
            tilt = self.get_stage_position().t
            stage_position = self.stage.native_delta(stage_position, tilt)
        logging.info(f"coincident move from SEM: {stage_position}")
        self.move_stage_relative(stage_position)

        logging.debug(
            {
                "msg": "move_coincident_from_sem",
                "dx": dx,
                "dy": dy,
                "scan_rotation": scan_rotation,
                "position": stage_position.to_dict(),
            }
        )
        return self.get_stage_position()

    # def run_milling_drift_corrected(self, milling_current: float,
    #     image_settings: ImageSettings,
    #     ref_image: FibsemImage,
    #     reduced_area: FibsemRectangle = None,
    #     asynch: bool = False
    #     ):
    #     """
    #     Run ion beam milling using the specified milling current.

    #     Args:
    #         milling_current (float): The current to use for milling in amps.
    #         asynch (bool, optional): If True, the milling will be run asynchronously.
    #                                  Defaults to False, in which case it will run synchronously.

    #     Returns:
    #         None

    #     Raises:
    #         None
    #     """
    #     status = self.connection.FIB.Beam.GetStatus()
    #     if status != Automation.FIB.Beam.Status.BeamOn:
    #         self.connection.FIB.Beam.On()
    #     self.connection.DrawBeam.LoadLayer(self.layer)
    #     logging.info("running ion beam milling now...")
    #     self.connection.DrawBeam.Start()
    #     self.connection.Progress.Show(
    #         "DrawBeam", "Layer 1 in progress", False, False, 0, 100
    #     )
    #     from fibsem import alignment
    #     while True:
    #         status = self.connection.DrawBeam.GetStatus()
    #         running = status[0] == DBStatus.ProjectLoadedExpositionInProgress
    #         if running:
    #             progress = 0
    #             if status[1] > 0:
    #                 progress = min(100, status[2] / status[1] * 100)
    #             printProgressBar(progress, 100)
    #             self.connection.Progress.SetPercents(progress)
    #             status = self.connection.DrawBeam.GetStatus()
    #             if status[0] == DBStatus.ProjectLoadedExpositionInProgress:
    #                 self.connection.DrawBeam.Pause()
    #             elif status[0] == DBStatus.ProjectLoadedExpositionIdle:
    #                 printProgressBar(100, 100, suffix="Finished")
    #                 self.connection.DrawBeam.Stop()
    #                 self.connection.DrawBeam.UnloadLayer()
    #                 break
    #             logging.info("Drift correction in progress...")
    #             image_settings.beam_type = BeamType.ION
    #             alignment.beam_shift_alignment(
    #                 self,
    #                 image_settings,
    #                 ref_image,
    #                 reduced_area,
    #             )
    #             time.sleep(1)
    #             status = self.connection.DrawBeam.GetStatus()
    #             if status[0] == DBStatus.ProjectLoadedExpositionPaused :
    #                 self.connection.DrawBeam.Resume()
    #             logging.info("Drift correction complete.")
    #             time.sleep(5)
    #         else:
    #             if status[0] == DBStatus.ProjectLoadedExpositionIdle:
    #                 printProgressBar(100, 100, suffix="Finished")
    #                 self.connection.DrawBeam.Stop()
    #                 self.connection.DrawBeam.UnloadLayer()
    #             break

    #     print()  # new line on complete
    #     self.connection.Progress.Hide()

    @staticmethod
    def _spot_burn_point_to_metres(
        point: Point, hfw: float, resolution: Tuple[int, int]
    ) -> Point:
        """Convert a normalised (0-1, top-left origin) image coordinate to DrawBeam coordinates.

        DrawBeam objects are positioned in metres from the image centre with +y up, the same
        convention the draw_* pattern methods already use.

        Args:
            point: normalised image coordinate, (0, 0) top-left to (1, 1) bottom-right.
            hfw: horizontal field width in metres.
            resolution: image resolution as (width, height) in pixels.

        Returns:
            Point: position in metres relative to the image centre.
        """
        from fibsem import conversions

        width, height = resolution
        pixelsize = hfw / width
        pixel_coordinate = Point(x=point.x * width, y=point.y * height)
        return conversions.image_to_microscope_image_coordinates2(
            coord=pixel_coordinate,
            image_shape=(height, width),
            pixelsize=pixelsize,
            subpixel_precision=True,  # spot coordinates are fractional, don't round to a pixel
        )

    def _create_spot_burn_layer(
        self,
        coordinates: List[Point],
        exposure_time: float,
        hfw: float,
        resolution: Tuple[int, int],
    ) -> "Automation.DrawBeam.Layer":
        """Build a DrawBeam layer holding one timed dot per coordinate.

        The layer runs at SPOT_BURN_PRESET; the remaining IEtching fields are mandatory, so
        they come from the configured milling defaults.
        """
        defaults = FibsemMillingSettings()
        layer_settings = IEtching(
            syncWriteField=False,
            writeFieldSize=hfw,
            beamCurrent=self.get_beam_current(BeamType.ION),
            spotSize=defaults.spot_size,
            rate=defaults.rate,
            dwellTime=defaults.dwell_time,
            parallel=False,
            preset=SPOT_BURN_PRESET,
            spacing=defaults.spacing,
        )
        with self._connection_lock:
            layer = self.connection.DrawBeam.Layer("SpotBurn", layer_settings)

        # DepthUnit.Second makes Depth an exposure time rather than a depth, which is
        # exactly a spot burn -- park on the point and expose for this long.
        for point in coordinates:
            centre = self._spot_burn_point_to_metres(
                point, hfw=hfw, resolution=resolution
            )
            logging.info(
                f"spot burn point: {point} -> ({centre.x:.3e}, {centre.y:.3e}) m, "
                f"exposure time: {exposure_time}s"
            )
            layer.addDot(
                CenterX=centre.x,
                CenterY=centre.y,
                Depth=exposure_time,
                DepthUnit=DepthUnit.Second,
            )
        return layer

    def run_spot_burn(
        self,
        settings: SpotBurnSettings,
        beam_type: BeamType = BeamType.ION,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        """Expose each coordinate with the ion beam for the configured exposure time.

        TESCAN cannot implement the blank -> park -> unblank sequence the default
        implementation uses for spot burning: FIB.Scan is a strict subset of SEM.Scan,
        missing exactly SetBlanker, GetBlanker and SetBeamPosition, and no FIB blanker
        exists anywhere in the SDK. This instead uses DrawBeam, which does the same job
        natively -- a Dot object with DepthUnit.Second is a timed exposure at a point.

        All points go into a single layer, so this runs as one DrawBeam exposition.
        Progress is reported via ``spot_burn_progress_signal`` in the shape the spot
        burn widget parses; the point index is derived from elapsed time, since DrawBeam
        runs the dots sequentially at exposure_time each and the SDK has no per-dot
        callback.

        Args:
            settings: coordinates to burn (normalised image coordinates, 0-1) and the
                exposure time per point. ``settings.milling_current`` is ignored on
                TESCAN; the beam conditions come from SPOT_BURN_PRESET.
            beam_type: must be BeamType.ION.
            stop_event: set to cancel the exposition.
        """
        if beam_type is not BeamType.ION:
            raise ValueError(
                f"Spot burn is only supported on the ion beam, got {beam_type.name}."
            )

        exposure_time = float(settings.exposure_time)
        if exposure_time <= 0:
            raise ValueError(f"exposure_time must be positive, got {exposure_time}.")

        if settings.milling_current is not None:
            logging.info(
                f"Spot burn milling_current is ignored on TESCAN; using preset "
                f"{SPOT_BURN_PRESET!r}. (requested: {settings.milling_current})"
            )

        # drop points outside the image bounds, matching the shared implementation
        in_bounds, dropped = [], []
        for pt in settings.coordinates:
            (in_bounds if 0 <= pt.x <= 1 and 0 <= pt.y <= 1 else dropped).append(pt)
        if dropped:
            logging.warning(
                f"Skipping {len(dropped)} spot burn coordinate(s) outside image bounds (0-1): {dropped}"
            )
        coordinates = in_bounds

        if not coordinates:
            logging.warning("No spot burn coordinates to burn.")
            return

        self._prepare_beam(beam_type)

        self.clear_patterns()

        hfw = self.get_field_of_view(beam_type)
        resolution = self.get_resolution(beam_type)
        # milling_current None: TESCAN burns at SPOT_BURN_PRESET, not the request
        self._record_spot_burn_started(
            coordinates, beam_type, exposure_time, None, len(dropped), field_of_view=hfw
        )
        layer = self._create_spot_burn_layer(
            coordinates=coordinates,
            exposure_time=exposure_time,
            hfw=hfw,
            resolution=resolution,
        )

        total_points = len(coordinates)
        estimated_time = total_points * exposure_time
        start_time = time.time()

        with self._connection_lock:
            self.connection.DrawBeam.LoadLayer(layer)
        logging.info(
            f"running spot burn now: {total_points} point(s), "
            f"{exposure_time}s each, {estimated_time}s total..."
        )

        self.spot_burn_progress_signal.emit(
            SpotBurnProgress(
                status=SpotBurnStatus.BURNING,
                current_point=0,
                total_points=total_points,
                remaining_time=exposure_time,
                total_remaining_time=estimated_time,
                total_estimated_time=estimated_time,
            )
        )

        cancelled = False

        with self._connection_lock:
            self.connection.DrawBeam.Start()

        try:
            while self.get_milling_state() in ACTIVE_MILLING_STATES:
                if stop_event is not None and stop_event.is_set():
                    logging.info("Spot burn cancelled.")
                    self.stop_milling()
                    cancelled = True
                    break

                time.sleep(SPOT_BURN_POLL_INTERVAL)

                # DrawBeam exposes no per-dot progress, but it burns the dots in order
                # at exposure_time each, so elapsed time maps onto the point index.
                elapsed = time.time() - start_time
                current_point = min(total_points, int(elapsed // exposure_time) + 1)
                self.spot_burn_progress_signal.emit(
                    SpotBurnProgress(
                        status=SpotBurnStatus.BURNING,
                        # Inferred from elapsed time, not measured: DrawBeam exposes
                        # no per-dot progress. Not authoritative.
                        current_point=current_point,
                        total_points=total_points,
                        remaining_time=max(
                            0.0, current_point * exposure_time - elapsed
                        ),
                        total_remaining_time=max(0.0, estimated_time - elapsed),
                        total_estimated_time=estimated_time,
                    )
                )

            self.spot_burn_progress_signal.emit(
                SpotBurnProgress(
                    status=SpotBurnStatus.CANCELLED
                    if cancelled
                    else SpotBurnStatus.FINISHED,
                    current_point=total_points,
                    total_points=total_points,
                )
            )
        except Exception as e:
            logging.error(f"Error in run_spot_burn: {e}")
            # The failure terminal belongs to the producer. It used to be emitted by
            # FibsemSpotBurnWidget, which only ever sees a burn it started itself --
            # so an unsupervised workflow burn that raised reported nothing at all and
            # left the bar mid-run for the session.
            self.spot_burn_progress_signal.emit(
                SpotBurnProgress(status=SpotBurnStatus.FAILED, error=str(e))
            )
            raise
        finally:
            self.clear_patterns()

    def _beam_device(self, beam_type: BeamType):
        """The beam device for ``beam_type``; a column disabled in the config has none."""
        device = self.beams.get(beam_type)
        if device is None:
            raise ValueError(f"The {beam_type.name} beam is not enabled.")
        return device

    def _get_beam(
        self, beam_type: BeamType
    ) -> Union["Automation.SEM", "Automation.FIB"]:
        """Get the beam object for the given beam type."""
        if not isinstance(beam_type, BeamType):
            raise ValueError(f"Invalid beam type: {beam_type}")

        if beam_type is BeamType.ELECTRON:
            return self.connection.SEM
        if beam_type is BeamType.ION:
            return self.connection.FIB

    def _settle_after_electron_image(self, beam_type: BeamType) -> None:
        """Pause before an ion image that directly follows an electron image.

        Requested acquisitions only: the live re-acquire loop passes a beam type, not
        image settings, and does not reach this. The pause used to sit in the shared
        `acquire.take_reference_images` behind a Tescan check; it is the driver's
        business, and here it also covers any other electron-then-ion pair.
        """
        previous = self._last_requested_beam_type
        self._last_requested_beam_type = beam_type
        if previous is BeamType.ELECTRON and beam_type is BeamType.ION:
            time.sleep(TESCAN_ELECTRON_TO_ION_SETTLE_TIME)

    def _wait_for_beam_ready(
        self,
        beam: Union["Automation.SEM", "Automation.FIB"],
        beam_type: BeamType,
        operation: str = "operation",
    ) -> None:
        """Wait for the beam to become ready (not busy)."""
        start_time = time.monotonic()
        while True:
            with self._connection_lock:
                busy = beam.IsBusy()
            if not busy:
                break
            logging.debug(
                f"Waiting for the {beam_type.name} beam to become ready after {operation}."
            )
            if time.monotonic() - start_time > TESCAN_BEAM_READY_TIMEOUT:
                raise TimeoutError(
                    f"{beam_type.name} beam is not ready after {operation}. "
                    f"Timeout of {TESCAN_BEAM_READY_TIMEOUT} seconds expired."
                )
            time.sleep(1)

    def _prepare_beam(
        self, beam_type: BeamType
    ) -> Union["Automation.SEM", "Automation.FIB"]:
        """Prepare the beam for imaging, milling, or other operations."""
        beam = self._get_beam(beam_type)

        with self._connection_lock:
            # check the beam is on
            status = beam.Beam.GetStatus()
            if status != beam.Beam.Status.BeamOn:
                beam.Beam.On()

            # stop the scanning before we start scanning or before automatic procedures,
            beam.Scan.Stop()

        self._wait_for_beam_ready(beam, beam_type, operation="preparation")

        return beam

    def beam_uses_presets(self, beam_type: BeamType) -> bool:
        """The ion column is set by preset: setting its voltage or current directly
        is refused by the Tescan API."""
        return beam_type is BeamType.ION

    def _get_presets(self, beam_type: BeamType) -> List[str]:
        with self._connection_lock:
            presets = self._get_beam(beam_type=beam_type).Preset.Enum()
        return sorted(presets)

    def _get_available_detectors(self, beam_type: BeamType) -> List[str]:
        """Get a list of available detectors for the given beam type."""
        detectors = []
        with self._connection_lock:
            if beam_type == BeamType.ELECTRON:
                detectors = self.connection.SEM.Detector.Enum()
            elif beam_type == BeamType.ION:
                detectors = self.connection.FIB.Detector.Enum()
        return detectors

    def _get_detector(
        self, detector_type: Union["Detector", str], beam_type: BeamType
    ) -> Optional[str]:
        """Get the detector object for the given detector type and beam type."""
        if isinstance(detector_type, Detector):
            detector_type = detector_type.name

        available_detectors = self._get_available_detectors(beam_type)
        detector: Detector
        for detector in available_detectors:
            if detector_type == detector.name:
                logging.debug(f"Found detector {detector.name}, index {detector.index}")
                return detector
        return None

    def _activate_preset(
        self,
        beam: Union["Automation.SEM", "Automation.FIB"],
        beam_type: BeamType,
        preset_name: str,
    ) -> None:
        """Activate a preset, preserving rotation/FOV/shift when self._preserve_settings_on_preset_change is set."""
        # Check if preset is available
        with self._connection_lock:
            available = beam.Preset.IsAvailable(preset_name)
        if not available:
            logging.warning(f"Preset {preset_name} not available for {beam_type}.")
            return

        preserve_settings = self._preserve_settings_on_preset_change

        # Save current settings if they should be preserved
        image_rotation = None
        view_field = None
        image_shift_x = None
        image_shift_y = None

        if preserve_settings:
            with self._connection_lock:
                image_rotation = beam.Optics.GetImageRotation()
                view_field = beam.Optics.GetViewfield()
                image_shift_x, image_shift_y = beam.Optics.GetImageShift()
            logging.debug(f"Image rotation before changing preset: {image_rotation}.")
            logging.debug(f"FOV before changing preset: {view_field}.")
            logging.debug(
                f"XY shift before changing preset: {image_shift_x}, {image_shift_y}."
            )

        try:
            # Activate preset
            with self._connection_lock:
                beam.Preset.Activate(preset_name)
            logging.info(f"Preset {preset_name} activated for {beam_type}.")

            # Wait for preset to fully apply
            self._wait_for_beam_ready(beam, beam_type, operation="preset activation")

            # Update internal state
            self._beam_parameters[beam_type].preset = preset_name

        except Exception as e:
            logging.error(
                f"Failed to activate preset {preset_name} for {beam_type}: {e}"
            )
            raise  # Re-raise to ensure finally block runs and then propagate error

        finally:
            # Restore settings if requested
            if preserve_settings:
                try:
                    with self._connection_lock:
                        beam.Optics.SetImageRotation(image_rotation)
                        beam.Optics.SetViewfield(view_field)
                        beam.Optics.SetImageShift(image_shift_x, image_shift_y)
                    logging.debug(
                        f"Restored image rotation after changing preset to {image_rotation}."
                    )
                    logging.debug(
                        f"Restored FOV after changing preset to {view_field}."
                    )
                    logging.debug(
                        f"Restored XY shift after changing preset to {image_shift_x}, {image_shift_y}."
                    )
                except Exception as restore_error:
                    logging.error(f"Failed to restore beam settings: {restore_error}")

    def home(self) -> bool:
        logging.warning("No homing available, please use native UI.")
        return False
