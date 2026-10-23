"""Tescan images: an SDK image document as a ``FibsemImage``.

The microscope state comes from the document's header. The stage position in it is in
Tescan's frame; `image_from_tescan` converts it with the stage device, as the device
converts every position it reads.

This module imports nothing from the SDK; the Tescan backend's guarded import is the
only one.
"""

from __future__ import annotations

import datetime
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np

import fibsem.constants as constants
from fibsem.structures import (
    BeamSettings,
    BeamType,
    FibsemDetectorSettings,
    FibsemImage,
    FibsemImageMetadata,
    FibsemRectangle,
    FibsemStagePosition,
    ImageSettings,
    MicroscopeState,
    Point,
)
from fibsem.util.timestamps import from_posix

if TYPE_CHECKING:
    from fibsem.drivers.tescan.devices import TescanStage


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
    image: Any, image_settings: Optional[ImageSettings] = None
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
        # local time when the state was built.
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


def image_from_tescan(
    image: Any, image_settings: ImageSettings, stage: Optional[TescanStage]
) -> FibsemImage:
    """An acquired Tescan image, with its header's stage position in fibsem's frame.

    The header records the stage in Tescan's frame; the stage device converts it,
    as it converts every position it reads. Without a stage device the position is
    left as Tescan reported it, and ``stage_frame`` says so.
    """
    fibsem_image = fromTescanImage(image, image_settings)
    state = fibsem_image.metadata.microscope_state
    if stage is not None and state.stage_position is not None:
        state.stage_position = stage.from_native(state.stage_position)
    return fibsem_image


FibsemImage.fromTescanImage = fromTescanImage
