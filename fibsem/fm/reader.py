"""Reading a fluorescence image from other software, so it can be imported (FIB-1030).

An image to lay over the Overview tab has to say where it was taken and how its
camera saw the sample. Our own acquisitions do; a file from another microscope --
a Zeiss export, a METEOR ImageJ stack, a PNG someone saved -- does not, and it may
not even say which axis is channels and which is z. So importing is two steps:

1. :func:`read_source` reads the pixels as they are stored and everything the file
   offers that can be trusted: tifffile's names for the axes, a pixel size in a
   physical unit, channel names and colours. Nothing is rearranged yet.
2. :func:`build_image` makes a :class:`FluorescenceImage` from those pixels and the
   answers the user confirmed -- which axis is what, the pixel size, the channels,
   whether the image is mirrored -- plus the placement the host supplies.

No Qt, so it is tested on every CI job; the dialog in front of it only asks.

A resolution in inches is ignored: that is a print setting (Zeiss exports say 300
dpi), not a pixel size. Centimetres and ImageJ's units are physical and are read.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

import numpy as np

from fibsem.fm.composite import AVAILABLE_COLORS
from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
    _annotated_metadata,
    open_tiff,
    safe_ome_from_tiff,
    to_czyx,
)

logger = logging.getLogger(__name__)

TIFF_SUFFIXES = (".tif", ".tiff")
RASTER_SUFFIXES = (".png", ".jpg", ".jpeg")
SUFFIXES = TIFF_SUFFIXES + RASTER_SUFFIXES

# The roles an axis can be given. `T` is a time axis: its first point is imported.
ROLES = ("C", "Z", "Y", "X", "T")
ROLE_NAMES = {"C": "Channel", "Z": "Z", "Y": "Y", "X": "X", "T": "Time"}

RGB_CHANNELS = (("Red", "red"), ("Green", "green"), ("Blue", "blue"))

# Metres per unit, for the units a file may state a pixel size in.
_UNITS = {
    "m": 1.0,
    "cm": 1e-2,
    "centimeter": 1e-2,
    "mm": 1e-3,
    "millimeter": 1e-3,
    "um": 1e-6,
    "µm": 1e-6,  # micro sign
    "μm": 1e-6,  # Greek mu, which files also use
    "micron": 1e-6,
    "microns": 1e-6,
    "micrometer": 1e-6,
    "nm": 1e-9,
    "nanometer": 1e-9,
}
# A pixel size outside this is a unit read wrongly, not a microscope.
_PLAUSIBLE = (1e-10, 1e-3)


@dataclass
class ImportSource:
    """An image as stored, and what its file suggests about it."""

    path: str
    data: np.ndarray  # as stored, in `axes` order
    axes: str  # tifffile's letters for the stored axes, e.g. "ZCYX", "YXS", "IYX"
    roles: str  # the suggested role of each axis, from ROLES
    pixel_size: Optional[float] = None  # metres, when the file states one
    channel_names: List[str] = field(default_factory=list)
    channel_colors: List[str] = field(default_factory=list)
    # The file's own channel records, when it has them: wavelengths and exposure
    # carried into the imported copy rather than made up.
    channel_metadata: Optional[List[FluorescenceChannelMetadata]] = None
    described_by: str = "no metadata"  # where the suggestions came from

    @property
    def is_rgb(self) -> bool:
        return "S" in self.axes

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


# ── reading ───────────────────────────────────────────────────────────────────


def read_source(path: str) -> ImportSource:
    """Read *path* for import: its pixels as stored and what it says about them.

    TIFF (plain, ImageJ, OME from any software) through tifffile; PNG and JPEG
    through Pillow. Raises ValueError for any other kind of file.
    """
    suffix = os.path.splitext(path)[1].lower()
    if suffix in TIFF_SUFFIXES:
        source = _read_tiff(path)
    elif suffix in RASTER_SUFFIXES:
        source = _read_raster(path)
    else:
        raise ValueError(
            f"{os.path.basename(path)}: can import {', '.join(SUFFIXES)} files"
        )
    count = channel_count(source.data.shape, source.roles)
    if len(source.channel_names) != count:
        source.channel_names, source.channel_colors = default_channels(count)
        source.channel_metadata = None
    return source


def _read_tiff(path: str) -> ImportSource:
    with open_tiff(path) as tif:
        series = tif.series[0]
        data = series.asarray()
        axes = series.axes.upper()
        page = tif.pages[0]
        imagej = tif.imagej_metadata or {}
        resolution = _tag_resolution(page)
        is_ome = tif.is_ome
    source = ImportSource(path=path, data=data, axes=axes, roles=suggest_roles(axes))
    if is_ome and _read_ome(path, source):
        return source
    if imagej:
        source.described_by = "ImageJ metadata"
        source.pixel_size = _imagej_pixel_size(imagej, resolution)
    else:
        source.pixel_size = _tiff_pixel_size(resolution)
    if source.is_rgb:
        source.channel_names = [n for n, _ in RGB_CHANNELS][: data.shape[-1]]
        source.channel_colors = [c for _, c in RGB_CHANNELS][: data.shape[-1]]
    return source


def _read_ome(path: str, source: ImportSource) -> bool:
    """Fill *source* from the file's OME, ours first. False if it cannot be read."""
    try:
        ome = safe_ome_from_tiff(path)
    except Exception as e:  # noqa: BLE001 - a TIFF whose OME cannot be parsed
        logger.debug(f"No usable OME in {path}: {e}")
        return False
    metadata = _annotated_metadata(ome)
    if metadata is not None:
        source.described_by = "FibsemOS metadata"
    else:
        try:
            metadata = FluorescenceImageMetadata.from_ome(ome)
        except Exception as e:  # noqa: BLE001 - OME without what we need
            logger.debug(f"OME in {path} does not describe the image: {e}")
            return False
        source.described_by = "OME metadata"
    source.pixel_size = _plausible(metadata.pixel_size_x)
    source.channel_metadata = list(metadata.channels)
    source.channel_names = [c.name for c in metadata.channels]
    defaults = default_channels(len(metadata.channels))[1]
    if source.described_by == "FibsemOS metadata":
        stated = [c.color for c in metadata.channels]
    else:
        stated = _ome_colours(ome)
    source.channel_colors = [
        nearest_colour(stated[i]) if i < len(stated) and stated[i] else defaults[i]
        for i in range(len(metadata.channels))
    ]
    return True


def _ome_colours(ome) -> List[Optional[str]]:
    """Each OME channel's colour as a hex string. White is OME's default as well as
    a real choice (a reflection channel): kept, unless every channel is white, which
    says nothing, and then all are None."""
    try:
        channels = ome.images[0].pixels.channels
    except (AttributeError, IndexError):
        return []
    colours: List[Optional[str]] = []
    for channel in channels:
        colour = getattr(channel, "color", None)
        rgb = getattr(colour, "as_rgb_tuple", lambda: None)() if colour else None
        colours.append("#{:02x}{:02x}{:02x}".format(*rgb[:3]) if rgb else None)
    if all(c in (None, "#ffffff") for c in colours):
        return [None] * len(colours)
    return colours


def nearest_colour(colour: str) -> str:
    """The FM canvas colour closest to *colour* -- a name it knows, or a file's hex.

    The channel controls offer named colours; a file that says `#04ff00` means
    green, and should show as green there rather than as an eighth option.
    """
    from fibsem.fm.composite import tint_rgb

    if colour in AVAILABLE_COLORS:
        return colour
    rgb = np.asarray(tint_rgb(colour))
    return min(
        AVAILABLE_COLORS,
        key=lambda name: float(((np.asarray(tint_rgb(name)) - rgb) ** 2).sum()),
    )


def _read_raster(path: str) -> ImportSource:
    from PIL import Image

    with Image.open(path) as image:
        if image.mode in ("L", "I", "I;16", "I;16B", "F"):
            data = np.asarray(image)
            axes = "YX"
        else:
            data = np.asarray(image.convert("RGB"))
            axes = "YXS"
    source = ImportSource(path=path, data=data, axes=axes, roles=suggest_roles(axes))
    if axes == "YXS":
        source.channel_names = [n for n, _ in RGB_CHANNELS]
        source.channel_colors = [c for _, c in RGB_CHANNELS]
    return source


def _tag_resolution(page) -> Optional[Tuple[float, int]]:
    """(pixels per unit, TIFF ResolutionUnit) from a page's tags, or None."""
    tags = page.tags
    if "XResolution" not in tags:
        return None
    value = tags["XResolution"].value
    try:
        per_unit = value[0] / value[1] if isinstance(value, tuple) else float(value)
    except (TypeError, ZeroDivisionError):
        return None
    unit = int(tags["ResolutionUnit"].value) if "ResolutionUnit" in tags else 1
    return (per_unit, unit) if per_unit > 0 else None


def _tiff_pixel_size(resolution) -> Optional[float]:
    """A plain TIFF's pixel size: only when its resolution is in centimetres."""
    if resolution is None:
        return None
    per_unit, unit = resolution
    return _plausible(1e-2 / per_unit) if unit == 3 else None


def _imagej_pixel_size(imagej: dict, resolution) -> Optional[float]:
    if resolution is None:
        return None
    per_unit, tiff_unit = resolution
    unit = str(imagej.get("unit", "")).strip().lower()
    metres = _UNITS.get(unit)
    if metres is None and tiff_unit == 3:
        metres = 1e-2
    return _plausible(metres / per_unit) if metres else None


def _plausible(value) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if _PLAUSIBLE[0] < value < _PLAUSIBLE[1] else None


# ── roles ─────────────────────────────────────────────────────────────────────


def suggest_roles(axes: str) -> str:
    """A role for each of tifffile's *axes*: what the file says, where it says it.

    A colour axis (`S`) is the channels, unless the file also names channels. Axes
    tifffile could not name take the roles still free by the loader's rule: one
    is z, two are channel then z. One more becomes time; beyond that the axis is
    left as `?` for the user to assign.
    """
    roles = [a if a in ROLES or a == "S" else "?" for a in axes.upper()]
    if "S" in roles:
        spare = next((r for r in ("C", "Z", "T") if r not in roles), "?")
        roles[roles.index("S")] = spare
    unknown = [i for i, r in enumerate(roles) if r == "?"]
    free = [r for r in ("C", "Z") if r not in roles]
    if unknown and free:
        for i, role in zip(unknown, free[-len(unknown) :]):
            roles[i] = role
    for i, role in enumerate(roles):
        if role == "?" and "T" not in roles:
            roles[i] = "T"
    return "".join(roles)


def check_roles(roles: str, shape: Sequence[int]) -> None:
    """Raise ValueError, in a sentence the dialog can show, if *roles* cannot be
    used for an array of *shape*."""
    if len(roles) != len(shape):
        raise ValueError(f"{len(shape)} axes need {len(shape)} roles")
    for role in roles:
        if role not in ROLES:
            raise ValueError(f"{role!r} is not a role")
    for role in ("Y", "X"):
        if roles.count(role) != 1:
            raise ValueError(f"exactly one axis must be {ROLE_NAMES[role]}")
    for role in ("C", "Z", "T"):
        if roles.count(role) > 1:
            raise ValueError(f"only one axis can be {ROLE_NAMES[role]}")


def channel_count(shape: Sequence[int], roles: str) -> int:
    return int(shape[roles.index("C")]) if "C" in roles else 1


def default_channels(count: int) -> Tuple[List[str], List[str]]:
    """Names and colours for channels a file does not describe. One channel is
    grey; several take the FM canvas's colours in turn."""
    names = [f"Channel-{i + 1:02d}" for i in range(count)]
    if count == 1:
        return names, ["gray"]
    return names, [AVAILABLE_COLORS[i % len(AVAILABLE_COLORS)] for i in range(count)]


def arrange(data: np.ndarray, roles: str) -> np.ndarray:
    """*data* with its axes in the given *roles*, as (C, Z, Y, X). A time axis
    gives its first point."""
    check_roles(roles, np.shape(data))
    out = to_czyx(data, roles)
    return out[0] if out.ndim == 5 else out


# ── building ──────────────────────────────────────────────────────────────────


def build_image(
    source: ImportSource,
    roles: str,
    pixel_size: float,
    channel_names: Sequence[str],
    channel_colors: Sequence[str],
    flip: bool = False,
    geometry=None,
    stage_position=None,
) -> FluorescenceImage:
    """The image to import: *source*'s pixels in the confirmed *roles*, with the
    confirmed pixel size (metres) and channels, mirrored left to right if *flip*.

    *geometry* and *stage_position* are the placement's: the host decides how the
    imported image is assumed to have been taken, since the file cannot say.
    Wavelengths and exposure come from the file's own channel records when it had
    them; otherwise the defaults a file without metadata has always loaded with.
    """
    data = arrange(source.data, roles)
    if flip:
        data = data[..., ::-1]
    data = np.ascontiguousarray(data)
    nc, nz, height, width = data.shape
    if len(channel_names) != nc or len(channel_colors) != nc:
        raise ValueError(f"{nc} channels need {nc} names and colours")
    if not pixel_size or pixel_size <= 0:
        raise ValueError("a pixel size is needed to place the image")
    known = (
        source.channel_metadata
        if source.channel_metadata and len(source.channel_metadata) == nc
        else None
    )
    channels = []
    for i, (name, colour) in enumerate(zip(channel_names, channel_colors)):
        base = (
            known[i]
            if known is not None
            else FluorescenceChannelMetadata(
                name=name,
                excitation_wavelength=488.0,
                power=1.0,
                exposure_time=0.1,
                gain=1.0,
                offset=0.0,
            )
        )
        channels.append(replace(base, name=str(name), color=str(colour)))
    try:
        stamp = datetime.fromtimestamp(os.path.getmtime(source.path)).isoformat()
    except OSError:
        stamp = datetime.now().isoformat()
    metadata = FluorescenceImageMetadata(
        acquisition_date=stamp,
        pixel_size_x=float(pixel_size),
        pixel_size_y=float(pixel_size),
        resolution=(width, height),
        channels=channels,
        stage_position=stage_position,
        geometry=geometry,
        description=f"Imported from {source.name}"
        + (", mirrored left to right" if flip else ""),
    )
    return FluorescenceImage(data=data, metadata=metadata)


# ── how an imported image is assumed to have been taken ────────────────────────


def assumed_geometry(microscope):
    """The camera an imported image is taken to have come from: this system's FM
    camera when it has one -- so an image exported from it without metadata lands
    as its own acquisitions do -- else one looking straight down the stage, with no
    flips. The file cannot say; Fit from points corrects the turn, and a mirror is
    the import dialog's to set."""
    from fibsem.structures import CameraImageTransform

    try:
        return microscope.fm_image_geometry()
    except (ValueError, AttributeError):
        return replace(
            microscope.hardware_geometry(),
            transform=CameraImageTransform.NONE,
            camera_tilt=0.0,
        )


def assumed_pose(microscope):
    """The stage rotation and tilt an imported image is taken at: the FM
    orientation when the system has one, else the SEM's."""
    for name in ("FM", "SEM"):
        try:
            return microscope.get_orientation(name)
        except ValueError:
            continue
    raise ValueError("the stage has neither an FM nor an SEM orientation")
