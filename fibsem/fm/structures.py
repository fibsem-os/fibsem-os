import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import tifffile as tff
import yaml
from ome_types import from_tiff, from_xml
from ome_types.model import (
    OME as OMEMetadata,
)
from ome_types.model import (
    Binning,
    Channel,
    Channel_ContrastMethod,
    Channel_IlluminationType,
    Color,
    DetectorSettings,
    Filter_Type,
    FilterRef,
    Image,
    Instrument,
    LightEmittingDiode,
    LightPath,
    LightSourceSettings,
    MapAnnotation,
    Objective,
    Pixels,
    Pixels_DimensionOrder,
    Plane,
    StructuredAnnotations,
    TiffData,
    TransmittanceRange,
    UnitsLength,
)
from ome_types.model import (
    Detector as OME_Detector,
)
from ome_types.model import (
    Filter as OMEFilter,
)

# Autofocus types are canonical in fibsem.autofunctions.autofocus; re-export here
# so existing `from fibsem.fm.structures import ...` imports keep working.
from fibsem.autofunctions.autofocus import (  # noqa: E402,F401
    AutoFocusResult,
    AutoFocusSettings,
    FocusMethod,
    FocusSweepPass,
)
from fibsem.codec import AwareTimestamp, NamedEnum, decode, encode

# AutoFocusMode is canonical in fibsem.structures -- there was a second, identical
# enum of the same name here, so the two tilers held different objects for the same
# concept and `is` comparisons across them silently failed. Imported (and so
# re-exported) rather than redefined, to keep that from happening again.
#
# CameraImageTransform, its parser and FibsemHardwareGeometry are canonical there for
# the same reason: the geometry record is one structure for both modalities, it needs
# the enum, and core cannot import from this package -- this module imports from it, so
# the reverse is a cycle. Imported (and so re-exported) here, which is where every
# consumer of them still is. FMImageGeometry was the FM-only half until FIB-481.
from fibsem.structures import (  # noqa: F401
    AutoFocusMode,
    CameraImageTransform,
    Configurable,
    Display,
    FibsemExperimentRef,
    FibsemHardwareGeometry,
    FibsemRectangle,
    FibsemStagePosition,
    InsertableDeviceState,
    Parameter,
    RangeLimit,
    TileOrderStrategy,
    _parse_image_transform,
    distance_param,
    param,
)
from fibsem.util.timestamps import (
    acquisition_datetime_of,
    now,
    now_iso,
    to_aware,
)


class ZStackOrder(NamedEnum, Enum):
    """Acquisition order for z-stack."""

    CHANNEL = "channel"  # default: for each channel, acquire all z-planes
    Z_LEVEL = "z_level"  # for each z-plane, acquire all channels


def _z_stack_order_label(order: ZStackOrder) -> str:
    return {ZStackOrder.CHANNEL: "Channel-wise", ZStackOrder.Z_LEVEL: "Z-level-wise"}[
        order
    ]


BINNING_MAP = {
    1: Binning.ONEBYONE,
    2: Binning.TWOBYTWO,
    4: Binning.FOURBYFOUR,
    8: Binning.EIGHTBYEIGHT,
}
BINNING_TO_INT = {v: k for k, v in BINNING_MAP.items()}
_UNIT_TO_METERS = {
    UnitsLength.NANOMETER: 1e-9,
    UnitsLength.MICROMETER: 1e-6,
    UnitsLength.MILLIMETER: 1e-3,
    UnitsLength.METER: 1.0,
}


def _convert_length_to_meters(
    value: Optional[float], unit: Optional[UnitsLength]
) -> Optional[float]:
    """Convert an OME length value to meters if both value and unit are present."""
    if value is None:
        return None
    return value * _UNIT_TO_METERS.get(unit, 1.0)


def _color_to_hex(color: Optional[Color]) -> str:
    """Convert OME Color to a hex string understood by the FM viewer."""
    if color is None:
        return "gray"
    try:
        r, g, b = color.as_rgb_tuple()
        return f"#{r:02X}{g:02X}{b:02X}"
    except Exception:
        return "gray"


def safe_ome_from_tiff(filename: str) -> OMEMetadata:
    """Parse OME metadata from TIFF file, handling potential known issues with the OME XML.
    Args:
        filename (str): Path to the TIFF file.
    Returns:
        OME: Parsed OME metadata object.
    """
    try:
        # Attempt to read OME metadata directly from the TIFF file
        ome = from_tiff(filename)
    except Exception as e:
        if "Filter" not in str(e):
            raise
        # Some microscopes emit <Channel><Filter .../></Channel>, which the OME
        # schema rejects ("Unknown property Channel:Filter"). Strip every
        # self-closing <Filter .../> element (there is one per channel — a naive
        # single find() drops only the first and re-throws) then re-parse, so the
        # real PhysicalSize/channel metadata is preserved instead of discarded.
        with tff.TiffFile(filename) as tif:
            ome_xml = tif.pages[0].tags["ImageDescription"].value
        ome_xml = re.sub(r"<Filter\b[^>]*/>", "", ome_xml)
        ome = from_xml(ome_xml)
    return ome


Band = Tuple[float, Optional[float]]  # edges in nm; the top may be unknown


@dataclass(frozen=True)
class EmissionFilter(Configurable):
    """One emission filter the filter set can put in the light path.

    ``bands`` are the edges in nm of each band it passes, lowest first: one for a
    single-band filter, two for a dual-band one. Reflection has none. A multi-band
    filter whose bands the driver doesn't know (an FM that reports only a
    fluorescence mode) has ``multi_band`` set and no bands. A driver that knows only a band's
    bottom edge gives ``(low, None)``.
    """

    name: str = param(doc="The filter's name, as the filter set reports it.")
    bands: Tuple[Band, ...] = param(
        (),
        unit="nm",
        doc="The edges of each band it passes, lowest first, as (low, high); high is null when the driver knows only the bottom edge.",
    )
    multi_band: bool = param(
        False,
        doc="True for a filter that passes more than one band, even when its bands are not known.",
    )

    def __post_init__(self):
        edges = tuple(
            (float(lo), None if hi is None else float(hi)) for lo, hi in self.bands
        )
        object.__setattr__(self, "bands", tuple(sorted(edges, key=lambda b: b[0])))
        object.__setattr__(self, "multi_band", bool(self.multi_band or len(edges) > 1))

    @classmethod
    def band(
        cls, name: str, low: float, high: Optional[float] = None
    ) -> "EmissionFilter":
        """A single-band filter."""
        return cls(name, bands=((low, high),))

    def to_dict(self) -> Dict[str, Any]:
        return encode(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmissionFilter":
        return decode(cls, data)

    @property
    def low(self) -> Optional[float]:
        """The bottom edge of the lowest band, or None without bands. Today's FM API
        names a banded filter by this value."""
        return self.bands[0][0] if self.bands else None

    @property
    def high(self) -> Optional[float]:
        """The top edge of the highest band, or None when it isn't known."""
        return self.bands[-1][1] if self.bands else None

    @property
    def centre(self) -> Optional[float]:
        """The middle of a single band in nm, or None otherwise."""
        if len(self.bands) != 1 or self.high is None:
            return None
        return (self.low + self.high) / 2

    @property
    def label(self) -> str:
        """The name to show. A multi-band filter of unknown bands that the FM calls
        "Fluorescence", the name saved channels give it, shows as "Multi-band"."""
        if self.multi_band and not self.bands and self.name.lower() == "fluorescence":
            return "Multi-band"
        return self.name


REFLECTION = EmissionFilter("Reflection")


def band_name(bands: Sequence[Band]) -> str:
    """ "510–560 nm", or "510–540 / 600–650 nm" for a dual-band filter."""
    parts = [f"{lo:.0f}" if hi is None else f"{lo:.0f}–{hi:.0f}" for lo, hi in bands]
    return " / ".join(parts) + " nm"


def emission_filter_for(
    value: Any, edges: Dict[float, Sequence[Sequence[float]]]
) -> EmissionFilter:
    """The filter an emission *value* names: the way channels and images saved
    before FIB-1439 named it (None for reflection, a label for a multi-band filter,
    or the band's bottom edge in nm), and the way some FM classes still report their
    filters. ``edges`` maps a filter's bottom edge to its bands, for FM classes that
    know them; a bare ``(low, high)`` pair is read as one band."""
    if value is None:
        return REFLECTION
    if isinstance(value, str):
        return EmissionFilter(value, multi_band=True)
    low = float(value)
    bands = next((b for bottom, b in edges.items() if math.isclose(bottom, low)), None)
    if bands is None:
        bands = ((low, None),)
    elif bands and not isinstance(bands[0], (tuple, list)):
        bands = (tuple(bands),)
    return EmissionFilter(band_name(bands), bands=bands)


def is_same_filter(a: EmissionFilter, b: EmissionFilter) -> bool:
    """Whether ``a`` and ``b`` name the same filter: equal, the same name, or the
    same bottom edge (a channel saved as "510" is the scope's GFP filter)."""
    if a == b or a.name == b.name:
        return True
    return (
        a.low is not None
        and b.low is not None
        and math.isclose(a.low, b.low, rel_tol=1e-6)
    )


def matching_emission_filter(
    wanted: EmissionFilter, choices: Sequence[EmissionFilter]
) -> EmissionFilter:
    """The filter among ``choices`` that ``wanted`` is, so a channel saved on one
    scope (or before FIB-1439, as a value) finds its filter on another.

    The same filter, or one of the same name. Reflection is reflection. A banded
    filter is the band whose bottom edge is closest; a filter set with no bands has
    only its multi-band filter, which is what such an FM reports a band as. A
    multi-band filter of unknown bands is the multi-band filter.
    """
    found = next((c for c in choices if c == wanted), None) or next(
        (c for c in choices if c.name == wanted.name), None
    )
    if found is not None:
        return found
    multi_band = [c for c in choices if c.multi_band]
    if wanted == REFLECTION:
        matches = []
    elif wanted.low is not None:
        banded = [c for c in choices if c.low is not None]
        matches = (
            sorted(banded, key=lambda c: abs(c.low - wanted.low))[:1] or multi_band
        )
    else:
        matches = multi_band
    if not matches:
        raise ValueError(f"No emission filter for {wanted.label!r}")
    return matches[0]


def emission_filter_from_dict(data: Dict[str, Any]) -> EmissionFilter:
    """A saved channel's filter: ``emission_filter``, or the ``emission_wavelength``
    value of a channel saved before FIB-1439."""
    from fibsem.codec.upgrades.fluorescence import channel

    return decode(EmissionFilter, channel(data)["emission_filter"])


def ome_emission_wavelength(found: EmissionFilter) -> Optional[float]:
    """A channel's OME ``EmissionWavelength``, in nm: the band's centre, or its
    bottom edge when that is all that is known. None for reflection and for a
    multi-band filter, which has no one wavelength."""
    if found.multi_band:
        return None
    return found.centre if found.centre is not None else found.low


def ome_emission_filter(found: EmissionFilter, filter_id: str) -> Optional[OMEFilter]:
    """The channel's emission filter as an OME ``Filter``: its name, ``MultiPass`` or
    ``BandPass``, and its range from the lowest edge to the highest where known
    (OME gives a filter one range). None for reflection, which has no filter."""
    if found == REFLECTION:
        return None
    transmittance = None
    if found.bands:
        transmittance = TransmittanceRange(
            cut_in=found.low,
            cut_in_unit=UnitsLength.NANOMETER,
            cut_out=found.high,
            cut_out_unit=UnitsLength.NANOMETER,
        )
    return OMEFilter(
        id=filter_id,
        model=found.name,
        type=Filter_Type.MULTI_PASS if found.multi_band else Filter_Type.BAND_PASS,
        transmittance_range=transmittance,
    )


def emission_filter_from_ome(
    channel: Channel, filters: Dict[str, OMEFilter]
) -> EmissionFilter:
    """A channel's filter from an OME file: the ``Filter`` its light path names, as
    ``ome_emission_filter`` writes it, or else its ``EmissionWavelength`` alone
    (another program's file, or one written before FIB-1439)."""
    refs = channel.light_path.emission_filters if channel.light_path else []
    found = filters.get(refs[0].id) if refs else None
    if found is None:
        return emission_filter_for(channel.emission_wavelength, {})
    multi_band = found.type == Filter_Type.MULTI_PASS
    span = found.transmittance_range
    bands = ()
    if span is not None and span.cut_in is not None:
        bands = ((span.cut_in, span.cut_out),)
    return EmissionFilter(
        found.model or "Fluorescence", bands=bands, multi_band=multi_band
    )


@dataclass
class ChannelSettings(Configurable):
    name: str = param(
        "Channel-01",
        doc="The channel's name, shown in the image layers and the saved metadata.",
        display=Display("Name"),
    )
    excitation_wavelength: float = param(
        550,
        device_parameter="filter_set.excitation_wavelength",
        doc="The excitation band, by its centre in nm.",
        display=Display("Excitation Wavelength"),
    )  # nm
    emission_filter: EmissionFilter = param(
        REFLECTION,
        doc="The emission filter the channel is acquired through, by its name and bands. Reflection takes none.",
        display=Display("Emission"),
    )
    power: float = param(
        0.01,
        device_parameter="light_source.power",
        doc="The light source's power, as a fraction of its maximum.",
        display=Display("Power"),
    )  # %
    exposure_time: float = param(
        0.001,
        device_parameter="camera.exposure_time",
        doc="The camera's exposure time per frame.",
        display=Display("Exposure Time"),
    )  # seconds
    color: str = param(
        "gray",
        doc="The colour the channel is shown in, by a colour map name (gray, green, magenta, ...).",
        display=Display("Color"),
    )
    gain: float = param(
        0.0,
        device_parameter="camera.gain",
        doc="The camera's gain, as a fraction of its range.",
        display=Display("Gain"),
    )

    def to_dict(self):
        return encode(self)

    @classmethod
    def from_dict(cls, ddict: dict) -> "ChannelSettings":
        """Read a channel saved by any version (`fibsem.codec.upgrades.fluorescence`)."""
        from fibsem.codec.upgrades.fluorescence import channel

        return decode(cls, channel(ddict))

    @property
    def pretty(self) -> str:
        """The channel's name and settings, in one line."""
        return f"{self.name} ({self.excitation_wavelength:.0f}nm → {self.emission_filter.label}) P:{self.power * 100:.1f}%, EXP:{self.exposure_time * 1000:.1f}ms"


@dataclass
class ZParameters(Configurable):
    # Relative to the objective's position when the stack starts.
    zmin: float = param(
        -10e-6,
        distance_param(
            "Z Min",
            "Minimum Z position relative to current position",
            limits=RangeLimit(-100e-6, -0.25e-6),
        ),
    )
    zmax: float = param(
        10e-6,
        distance_param(
            "Z Max",
            "Maximum Z position relative to current position",
            limits=RangeLimit(0.25e-6, 100e-6),
        ),
    )
    zstep: float = param(
        1e-6,
        distance_param(
            "Z Step", "Step size between Z positions", limits=RangeLimit(0.1e-6, 10e-6)
        ),
    )
    order: ZStackOrder = param(
        ZStackOrder.CHANNEL,
        doc="The order the stack is acquired in: channel-wise takes every plane of one channel before the next, z-level-wise every channel at one plane before the next.",
        display=Display("Order", advanced=True, format_fn=_z_stack_order_label),
    )

    def to_dict(self) -> dict:
        return encode(self)

    @classmethod
    def from_dict(cls, ddict: dict) -> "ZParameters":
        return decode(cls, ddict)

    @classmethod
    def from_focus_pass(cls, sweep_pass: "FocusSweepPass") -> "ZParameters":
        """Build a relative sweep centred on zero, spanning ±search_range/2 at step_size."""
        half = sweep_pass.search_range / 2
        return cls(zmin=-half, zmax=half, zstep=sweep_pass.step_size)

    def generate_positions(self, z_init: float) -> List[float]:
        """Generate a list of z positions based on the current z init and relative z parameters.
        Args:
            z_init (float): The initial z position from which to calculate the relative positions.
        Returns:
            List[float]: A list of z positions starting from z_init + zmin to z_init + zmax, with steps of zstep."""

        if self.zstep <= 0:
            raise ValueError("zstep must be a positive value.")

        # Generate z positions from z_init+zmin to z_init+zmax inclusive
        z_positions = np.arange(
            start=z_init + self.zmin,
            stop=z_init + self.zmax + self.zstep * 0.5,
            step=self.zstep,
        )

        return z_positions.tolist()

    @property
    def num_planes(self) -> int:
        """Calculate and return the number of z-stack planes.

        Returns:
            int: Number of planes in the z-stack based on zmin, zmax, and zstep.
                 Returns 0 if zstep is invalid (<=0).
        """
        if self.zstep <= 0:
            return 0

        z_range = self.zmax - self.zmin
        return int(z_range / self.zstep) + 1

    @property
    def pretty_name(self) -> str:
        """Generate a pretty name for the z parameters."""
        if self.zstep <= 0:
            return "Z-Stack: Disabled"

        num_planes = self.num_planes
        if num_planes <= 1:
            return "Z-Stack: Single plane acquisition"

        order_str = (
            "channel-wise" if self.order == ZStackOrder.CHANNEL else "z-level-wise"
        )
        return f"Z-Stack: {num_planes} planes ({self.zmin * 1e6:.1f}μm to {self.zmax * 1e6:.1f}μm, step {self.zstep * 1e6:.1f}μm, {order_str})"


def open_tiff(filename: str) -> "tff.TiffFile":
    """A TiffFile that reads an ImageJ file as ImageJ.

    A METEOR ImageJ export carries OME text in its ImageJ description that is not
    valid OME. tifffile flags such a file as OME too, tries that first, and logs an
    ERROR before falling back to ImageJ -- on every open, for a file that reads
    correctly. So a file that is both is opened with OME off: the ImageJ series is
    the one tifffile would have used.
    """
    tif = tff.TiffFile(filename)
    if tif.is_imagej and tif.is_ome:
        tif.close()
        tif = tff.TiffFile(filename, is_ome=False)
    return tif


# tifffile's names for an axis it cannot place: a page sequence, or unknown.
_UNKNOWN_AXES = "IQ"


def to_czyx(data: np.ndarray, axes: str) -> np.ndarray:
    """*data* with tifffile *axes* arranged as (C, Z, Y, X), or (T, C, Z, Y, X) when
    there is more than one time point.

    Missing channel and z axes are added with length one. A colour axis (`S`, an RGB
    file's samples) is taken as the channels. Axes tifffile could not name are taken
    as they always were here: one is z, two are channel then z.
    """
    data = np.asarray(data)
    axes = axes.upper()
    if len(axes) != data.ndim:
        raise ValueError(
            f"axes {axes!r} do not describe an array of shape {data.shape}"
        )
    unknown = [a for a in axes if a in _UNKNOWN_AXES]
    if unknown:
        fill = [a for a in ("C", "Z") if a not in axes]
        if len(unknown) > len(fill):
            raise ValueError(
                f"cannot tell which axes of {axes!r} {data.shape} are what"
            )
        for letter, name in zip(unknown, fill[-len(unknown) :]):
            axes = axes.replace(letter, name, 1)
    if "S" in axes:
        if "C" in axes:
            raise ValueError(f"both channels and colour samples in {axes!r}")
        axes = axes.replace("S", "C")
    for name in ("Z", "C"):
        if name not in axes:
            data = data[np.newaxis]
            axes = name + axes
    extra = set(axes) - set("TCZYX")
    if extra or len(set(axes)) != len(axes) or "Y" not in axes or "X" not in axes:
        raise ValueError(f"cannot arrange axes {axes!r} as CZYX")
    order = "TCZYX" if "T" in axes else "CZYX"
    data = data.transpose([axes.index(a) for a in order])
    if order == "TCZYX" and data.shape[0] == 1:
        data = data[0]
    return data


def _annotated_metadata(ome: OMEMetadata) -> Optional["FluorescenceImageMetadata"]:
    """Our own metadata, from the map annotation `save` writes; None without one."""
    annotations = ome.structured_annotations
    for annotation in getattr(annotations, "map_annotations", None) or []:
        value = annotation.value
        if not value or "FluorescenceImageMetadata" not in value:
            continue
        text = value["FluorescenceImageMetadata"]
        if not isinstance(text, str):
            continue
        try:
            return FluorescenceImageMetadata.from_dict(json.loads(text))
        except Exception as e:
            logging.warning(f"Failed to load structured annotations: {e}")
            return None
    return None


@dataclass
class FluorescenceImage:
    data: np.ndarray  # TCZYX format (Time, Channels, Z, Y, X)
    metadata: "FluorescenceImageMetadata"
    # the file this image is associated with on disk, set by save() and load(), and
    # inherited by the derived forms -- projection, focus stack, interpolation. A
    # resampled volume is still that file's volume, and it is the only file that
    # describes it; the UI and the correlation record name an image from this
    # (FIB-509), so dropping it on a derived image loses the provenance entirely.
    # Nothing writes back to it: save() takes the filename it is given.
    # excluded from compare/repr: it describes where the image lives, not what it contains,
    # so two images of the same data read from different paths are still equal.
    filepath: Optional[str] = field(default=None, compare=False, repr=False)

    def save(self, filename: str, compression: Optional[str] = None) -> str:
        """
        Save a FMImage to a TIFF file with OME metadata.

        Args:
            filename (str): The filename to save the image to.
            compression (str, optional): A tifffile compression, e.g. "zlib", for a
                copy that should not be larger than its source; with a horizontal
                predictor, which is what makes 16-bit microscopy compress. None, the
                default, writes uncompressed and contiguous, as acquisitions always
                have.
        """
        # if data is 2D, reshape to 4D (CZYX)
        if self.data.ndim == 2:
            self.data = self.data.reshape(1, 1, *self.data.shape)

        ome_md = self.get_ome_metadata()
        # No schema validation here: ome_types builds the XML from typed
        # models, and tff.OmeXml.validate fetches the OME XSD over HTTP on
        # first use. An offline microscope PC (the normal case) blocked on the
        # socket timeout and then raised on every FM save. Outbound network
        # calls are opt-in and must never sit on the acquisition path.
        ome_xml = ome_md.to_xml()

        # Reshape image to 5D for tifffile (CZYX -> TCZYX)
        if self.data.ndim != 5:
            nc, nz, ny, nx = self.data.shape  # C, Z, Y, X
            tifffile_image = self.data.reshape(1, nc, nz, ny, nx)
        else:
            tifffile_image = self.data

        # Mirrors FibsemImage.save: the caller names a file, so the directory it
        # lands in is this method's to create. AcquireFluorescenceTask writes a
        # z-stack into the lamella directory without acquiring a beam image
        # first, so this is the only thing that creates it -- and acquire_image
        # swallows save failures, so a missing directory would lose the stack
        # silently. dirname is empty for a bare filename, which makedirs rejects.
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # TODO: add overwrite protection to prevent overwriting existing files
        with tff.TiffWriter(filename) as tif:
            # Greyscale, said outright: left to guess, tifffile may take a stack
            # of three or four planes for the samples of an RGB image.
            if compression:
                # Compressed pages cannot be contiguous; the per-plane mapping in
                # the OME says where each is, as it does for any file.
                tif.write(
                    data=tifffile_image,
                    photometric="minisblack",
                    compression=compression,
                    predictor=True,
                )
            else:
                tif.write(
                    data=tifffile_image, contiguous=True, photometric="minisblack"
                )
            tif.overwrite_description(ome_xml)

        # set only after a successful write, so a recorded path is always a path that exists
        self.filepath = str(filename)
        return self.filepath

    def get_ome_metadata(self) -> OMEMetadata:
        """Generate OME metadata for the FluorescenceImage."""

        ifd = 0
        planes: List[Plane] = []
        tiff_data_blocks: List[TiffData] = []

        channels_md: List[Channel] = []
        detectors: List[OME_Detector] = []
        light_sources: List[LightEmittingDiode] = []
        objectives: List[Objective] = []
        emission_filters: List[OMEFilter] = []

        nc, nz, ny, nx = self.data.shape  # C, Z, Y, X
        dtype = self.data.dtype.name

        # Use structured metadata directly
        for ch_idx, channel in enumerate(self.metadata.channels):
            id_str = f"{ch_idx + 1:02d}"

            # Extract channel metadata
            light_source_power = channel.power
            camera_binning = BINNING_MAP.get(channel.binning, None)
            camera_gain = channel.gain
            camera_offset = channel.offset

            # wavelengths are in nm
            excitation_wavelength = int(channel.excitation_wavelength)
            emission_wavelength = ome_emission_wavelength(channel.emission_filter)
            emission_filter = ome_emission_filter(
                channel.emission_filter, f"Filter:{id_str}"
            )
            if emission_filter is not None:
                emission_filters.append(emission_filter)

            # Create OME components
            light_source = LightEmittingDiode(
                id=f"LightSource:{id_str}", power=light_source_power
            )

            light_source_settings = LightSourceSettings(
                id=f"LightSource:{id_str}",
            )

            detector = OME_Detector(
                id=f"Detector:{id_str}",
                gain=camera_gain,
                offset=camera_offset,
            )
            detector_settings = DetectorSettings(
                id=f"Detector:{id_str}",
                gain=camera_gain,
                offset=camera_offset,
                binning=camera_binning,
            )

            detectors.append(detector)
            light_sources.append(light_source)

            # Create objective if magnification or NA are available
            objective_id = f"Objective:{id_str}"
            if (
                channel.objective_magnification is not None
                or channel.objective_numerical_aperture is not None
            ):
                objective = Objective(
                    id=objective_id,
                    nominal_magnification=channel.objective_magnification,
                    lens_na=channel.objective_numerical_aperture,
                )
                objectives.append(objective)

            ch_md = Channel(
                name=channel.name,
                id=f"Channel:{id_str}",
                excitation_wavelength=excitation_wavelength,
                excitation_wavelength_unit=UnitsLength.NANOMETER,
                emission_wavelength=emission_wavelength,
                emission_wavelength_unit=UnitsLength.NANOMETER,
                light_path=LightPath(
                    emission_filters=[FilterRef(id=emission_filter.id)]
                )
                if emission_filter is not None
                else None,
                detector_settings=detector_settings,
                illumination_type=Channel_IlluminationType.EPIFLUORESCENCE,
                contrast_method=Channel_ContrastMethod.FLUORESCENCE,
                light_source_settings=light_source_settings,
                color=Color(channel.color),
            )
            channels_md.append(ch_md)

            # Get pixel sizes and z-positions from metadata
            pixel_size_x = self.metadata.pixel_size_x
            pixel_size_y = self.metadata.pixel_size_y

            if nz > 1 and self.metadata.z_positions:
                z_positions = self.metadata.z_positions
                pixel_size_z = self.metadata.pixel_size_z
            else:
                z_positions = [channel.objective_position]
                pixel_size_z = None

            exposure_time = channel.exposure_time

            # Create planes for this channel
            for z_idx in range(nz):
                # Get stage position if available
                stage_x = (
                    self.metadata.stage_position.x
                    if self.metadata.stage_position
                    else None
                )
                stage_y = (
                    self.metadata.stage_position.y
                    if self.metadata.stage_position
                    else None
                )

                plane = Plane(
                    the_c=ch_idx,
                    the_t=0,
                    the_z=z_idx,
                    exposure_time=exposure_time,
                    position_x=stage_x,
                    position_y=stage_y,
                    position_z=z_positions[z_idx]
                    if z_idx < len(z_positions)
                    else z_positions[0],
                    position_x_unit=UnitsLength.METER,
                    position_y_unit=UnitsLength.METER,
                    position_z_unit=UnitsLength.METER,
                )
                tiff_data = TiffData(
                    ifd=ifd, first_c=ch_idx, first_z=z_idx, plane_count=1
                )

                planes.append(plane)
                tiff_data_blocks.append(tiff_data)

                ifd += 1

        # TODO: use metadata.system_info for manufacturer, model, serial number, etc.
        inst = Instrument(
            id="Instrument:01",
            detectors=detectors,
            light_emitting_diodes=light_sources,
            objectives=objectives if objectives else [],
            filters=emission_filters,
        )

        # OME's AcquisitionDate: aware when the image says which zone, naive (the
        # acquiring machine's clock) when it is from before that was recorded.
        acquisition_date = acquisition_datetime_of(self.metadata) or now()

        # Use filename and description from metadata if available
        image_name = self.metadata.filename or "Fluorescence Image"
        image_description = self.metadata.description or "Fluorescence microscopy image"

        image_md = Image(
            id="Image:01",
            name=image_name,
            description=image_description,
            acquisition_date=acquisition_date,
            pixels=Pixels(
                id="Pixels:01",
                channels=channels_md,
                # The planes are written channel by channel, every z of one
                # channel before the next: z varies fastest (FIB-279).
                dimension_order=Pixels_DimensionOrder.XYZCT,
                size_x=nx,
                size_y=ny,
                size_z=nz,
                size_c=nc,
                size_t=1,
                physical_size_x=pixel_size_x,
                physical_size_y=pixel_size_y,
                physical_size_z=pixel_size_z,
                physical_size_x_unit=UnitsLength.METER,
                physical_size_y_unit=UnitsLength.METER,
                physical_size_z_unit=UnitsLength.METER,
                type=dtype,
                planes=planes,
                tiff_data_blocks=tiff_data_blocks,
            ),
        )

        # Create structured annotations with full metadata
        metadata_annotation = MapAnnotation(
            id="Annotation:01",
            value={"FluorescenceImageMetadata": json.dumps(self.metadata.to_dict())},
        )

        structured_annotations = StructuredAnnotations(
            map_annotations=[metadata_annotation]
        )

        ome_md = OMEMetadata(
            images=[image_md],
            instruments=[inst],
            structured_annotations=structured_annotations,
        )

        return ome_md

    # NOTES: objective, detector should be common across all channels... not per channel

    @classmethod
    def load(cls, filename: str) -> "FluorescenceImage":
        """Load an image, with its axes as the file records them and its metadata
        recovered from our annotation, the file's own OME, or defaults, in that order.

        The array is arranged to (C, Z, Y, X) from the axes tifffile reads out of the
        file -- for an OME-TIFF, from its per-plane mapping -- never from comparing
        sizes against the metadata: that comparison cannot tell channels from z-slices
        when there are as many of one as the other, and swapped them (FIB-279).
        """
        with open_tiff(filename) as tif:
            series = tif.series[0]
            data = series.asarray()
            axes = series.axes
        data = to_czyx(data, axes)

        metadata: Optional[FluorescenceImageMetadata] = None
        try:
            ome = safe_ome_from_tiff(filename)
        except Exception as e:
            logging.debug(f"No OME metadata in {filename}: {e}")
            ome = None
        if ome is not None:
            metadata = _annotated_metadata(ome)
            if metadata is None:
                # An OME-TIFF from other software: what its own OME says.
                try:
                    metadata = FluorescenceImageMetadata.from_ome(ome)
                except Exception as e:
                    logging.warning(f"Failed to load OME metadata: {e}")
        if metadata is None:
            metadata = cls._create_basic_metadata(data.shape)

        return cls(data=data, metadata=metadata, filepath=str(filename))

    @classmethod
    def _create_basic_metadata(cls, data_shape: tuple) -> "FluorescenceImageMetadata":
        """Create basic metadata when no structured annotations are available."""
        # Handle different data shapes: (Y, X), (Z, Y, X), (C, Z, Y, X), and
        # (T, C, Z, Y, X), whose channels are those of any one time point.
        if len(data_shape) == 5:
            data_shape = tuple(data_shape[1:])
        if len(data_shape) == 2:
            ny, nx = data_shape
            nc = 1
        elif len(data_shape) == 3:
            _, ny, nx = data_shape
            nc = 1
        elif len(data_shape) == 4:
            nc, _, ny, nx = data_shape
        else:
            raise ValueError(f"Unsupported data shape: {data_shape}")

        channels = []
        for i in range(nc):
            channel = FluorescenceChannelMetadata(
                name=f"Channel_{i + 1:02d}",
                excitation_wavelength=488.0,
                power=1.0,
                exposure_time=0.1,
                gain=1.0,
                offset=0.0,
            )
            channels.append(channel)

        return FluorescenceImageMetadata(
            acquisition_date=now_iso(),
            pixel_size_x=1e-6,  # default 1 micron
            pixel_size_y=1e-6,  # default 1 micron
            resolution=(nx, ny),
            channels=channels,
        )

    def crop(self, roi: FibsemRectangle) -> np.ndarray:
        """Crop image data to a region of interest.

        Args:
            roi: Region of interest with left, top, width, height in 0-1 relative coordinates.

        Returns:
            Cropped image data as numpy array (same number of dimensions as self.data).
        """
        data = self.data
        # Y and X are always the last two dimensions
        h, w = data.shape[-2], data.shape[-1]
        y0 = int(roi.top * h)
        y1 = int((roi.top + roi.height) * h)
        x0 = int(roi.left * w)
        x1 = int((roi.left + roi.width) * w)
        return data[..., y0:y1, x0:x1]

    @staticmethod
    def create_z_stack(ch_images: List["FluorescenceImage"]) -> "FluorescenceImage":
        """Create a z-stack from a list of FluorescenceImage objects."""

        # Ensure all images have the same shape
        shapes = [img.data.shape for img in ch_images]
        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError("All images must have the same shape for z-stacking.")

        # Stack images along Z axis, squeezing singleton dimensions if needed
        data_list = []
        for img in ch_images:
            # If data is (1, Y, X), squeeze to (Y, X) for stacking
            if img.data.ndim == 3 and img.data.shape[0] == 1:
                data_list.append(img.data.squeeze(0))
            else:
                data_list.append(img.data)
        arrs = np.stack(data_list, axis=0)

        # Create z-stack metadata from first image, adding z-positions
        first_metadata = ch_images[0].metadata
        z_positions = [img.metadata.channels[0].objective_position for img in ch_images]

        # Carry the first plane's metadata over and change only what stacking changes.
        # Listing the fields to copy instead means every field added to the dataclass
        # later has to be remembered here too -- which is how `geometry` came to be
        # silently dropped from every z-stacked and multi-channel acquisition.
        md = replace(first_metadata, z_positions=z_positions)

        return FluorescenceImage(data=arrs, metadata=md)

    @staticmethod
    def create_multi_channel_image(
        images: List["FluorescenceImage"],
    ) -> "FluorescenceImage":
        """Create a multi-channel image from a list of FluorescenceImage objects."""

        # Ensure all images have the same shape
        shapes = [img.data.shape for img in images]
        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError(
                "All images must have the same shape for multi-channel stacking."
            )

        if len(images) == 0:
            raise ValueError("No images provided for multi-channel stacking.")

        img_data = []
        for img in images:
            # If the image is 2D, add a singleton z dimension
            if img.data.ndim == 2:
                img.data = img.data[np.newaxis, :, :]
            img_data.append(img.data)

        # Stack images along the channel dimension (CZYX)
        stacked_data = np.stack(img_data, axis=0)  # Shape: (C, Z, Y, X)

        # Combine metadata from all images into FluorescenceImageMetadata
        # Extract common metadata from first image
        first_img = images[0]

        # Combine channels from all images
        channels = []
        for img in images:
            channels.extend(img.metadata.channels)

        # As above: copy and override, so a new metadata field cannot go missing here.
        mds = replace(first_img.metadata, channels=channels)

        return FluorescenceImage(data=stacked_data, metadata=mds)

    def max_intensity_projection(
        self, channel: Optional[int] = None, return_2d: bool = False
    ) -> Union["FluorescenceImage", np.ndarray]:
        """Create a maximum intensity projection along the z-axis.

        Args:
            channel: Optional channel index to project. If None, projects all channels.
                    Zero-indexed (0 = first channel, 1 = second channel, etc.)
            return_2d: If True, return a 2D numpy array suitable for plotting.
                      If False, return a FluorescenceImage with metadata.

        Returns:
            If return_2d=True: A 2D numpy array (Y, X) for single channel or (C, Y, X) for multiple channels
            If return_2d=False: A new FluorescenceImage with the maximum intensity projection.
                               The z-dimension will be reduced to 1.

        Raises:
            ValueError: If the specified channel index is out of range
            ValueError: If the image has no z-dimension to project
        """
        # Check if data has z-dimension to project
        if self.data.ndim < 3:
            raise ValueError(
                "Image must have at least 3 dimensions (C, Z, Y, X) for z-projection"
            )

        nc, nz, ny, nx = self.data.shape

        if nz == 1:
            # No z-projection needed, return a copy
            if channel is not None:
                if channel < 0 or channel >= nc:
                    raise ValueError(
                        f"Channel index {channel} out of range [0, {nc - 1}]"
                    )
                projected_data = self.data[channel : channel + 1, :, :, :]
                projected_channels = [self.metadata.channels[channel]]
            else:
                projected_data = self.data.copy()
                projected_channels = self.metadata.channels.copy()
        else:
            # Perform maximum intensity projection along z-axis (axis=1)
            if channel is not None:
                if channel < 0 or channel >= nc:
                    raise ValueError(
                        f"Channel index {channel} out of range [0, {nc - 1}]"
                    )
                # Project single channel
                channel_data = self.data[channel, :, :, :]  # Shape: (Z, Y, X)
                projected_data = np.max(
                    channel_data, axis=0, keepdims=True
                )  # Shape: (1, Y, X)
                projected_data = projected_data[
                    np.newaxis, :, :, :
                ]  # Shape: (1, 1, Y, X)
                projected_channels = [self.metadata.channels[channel]]
            else:
                # Project all channels
                projected_data = np.max(
                    self.data, axis=1, keepdims=True
                )  # Shape: (C, 1, Y, X)
                projected_channels = self.metadata.channels.copy()

        # Return 2D array if requested
        if return_2d:
            # Remove singleton dimensions for plotting convenience
            return np.squeeze(projected_data)

        # Create new metadata for the projected image
        projected_metadata = FluorescenceImageMetadata(
            acquisition_date=self.metadata.acquisition_date,
            acquisition_datetime=self.metadata.acquisition_datetime,
            pixel_size_x=self.metadata.pixel_size_x,
            pixel_size_y=self.metadata.pixel_size_y,
            pixel_size_z=None,  # No z-dimension after projection
            resolution=self.metadata.resolution,
            channels=projected_channels,
            z_positions=None,  # No z-positions after projection
            stage_position=self.metadata.stage_position,
            filename=self.metadata.filename,
            description=self.metadata.description,
            system_info=self.metadata.system_info,
            # A projection of a lamella's image is still that lamella's. Carried
            # forward so a derived image does not lose what produced it (FIB-466).
            experiment=self.metadata.experiment,
            geometry=self.metadata.geometry,
            dimension_order=self.metadata.dimension_order,
        )

        return FluorescenceImage(
            data=projected_data,
            metadata=projected_metadata,
            # Still this file's image; see the note on the field (FIB-509).
            filepath=self.filepath,
        )

    def focus_stack(
        self,
        channel: Optional[int] = None,
        method: str = "laplacian",
        return_2d: bool = False,
        use_blocks: bool = False,
        block_size: int = 128,
        smooth_transitions: bool = True,
    ) -> Union["FluorescenceImage", np.ndarray]:
        """Create a focus-stacked image with extended depth of field.

        Combines multiple images at different focus positions by selecting the sharpest
        regions from each z-plane to create a single image with maximum sharpness
        throughout the field of view.

        Args:
            channel: Optional channel index to stack. If None, stacks all channels.
                    Zero-indexed (0 = first channel, 1 = second channel, etc.)
            method: Focus measure algorithm to use. Options:
                   'laplacian' - Laplacian variance (default, good general purpose)
                   'sobel' - Sobel gradient magnitude (edge-based)
                   'variance' - Local variance (simple, fast)
                   'tenengrad' - Thresholded Sobel (good for noisy images)
            return_2d: If True, return a 2D numpy array suitable for plotting.
                      If False, return a FluorescenceImage with metadata.
            use_blocks: If True, use block-based focus selection instead of per-pixel.
                       More robust to noise and computationally efficient.
            block_size: Size of square blocks for block-based selection (default: 128)
            smooth_transitions: If True and use_blocks=True, apply Gaussian blur to
                              reduce sharp block boundary artifacts (default: True)

        Returns:
            If return_2d=True: A 2D numpy array (Y, X) for single channel or (C, Y, X) for multiple channels
            If return_2d=False: A new FluorescenceImage with the focus-stacked result.
                               The z-dimension will be reduced to 1.

        Raises:
            ValueError: If the specified channel index is out of range
            ValueError: If the image has insufficient z-planes for stacking
            ValueError: If the specified method is not supported

        Note:
            Focus stacking works best with images that have overlapping depth of field
            across z-planes. Requires at least 2 z-planes to perform stacking.
            Block-based approach (use_blocks=True) is recommended for noisy images.
        """
        from .calibration import (
            create_block_based_focus_stack,
            create_pixel_based_focus_stack,
        )

        # Check if data has sufficient z-planes for stacking
        if self.data.ndim < 3:
            raise ValueError(
                "Image must have at least 3 dimensions (C, Z, Y, X) for focus stacking"
            )

        nc, nz, ny, nx = self.data.shape

        if nz < 2:
            raise ValueError("Focus stacking requires at least 2 z-planes")

        # Validate method by importing function (validation happens in calibration functions)
        from .calibration import get_focus_measure_function

        get_focus_measure_function(method)  # Just for validation

        # Select channels to process
        if channel is not None:
            if channel < 0 or channel >= nc:
                raise ValueError(f"Channel index {channel} out of range [0, {nc - 1}]")
            data_to_process = self.data[
                channel : channel + 1, :, :, :
            ]  # Shape: (1, Z, Y, X)
            selected_channels = [self.metadata.channels[channel]]
        else:
            data_to_process = self.data  # Shape: (C, Z, Y, X)
            selected_channels = self.metadata.channels.copy()

        # Process each channel separately
        stacked_channels = []

        for ch_idx in range(data_to_process.shape[0]):
            channel_data = data_to_process[ch_idx, :, :, :]  # Shape: (Z, Y, X)

            if use_blocks:
                # Use block-based focus selection
                stacked_image = create_block_based_focus_stack(
                    channel_data,
                    method=method,
                    block_size=block_size,
                    smooth_transitions=smooth_transitions,
                )
            else:
                # Use per-pixel focus selection
                stacked_image = create_pixel_based_focus_stack(
                    channel_data, method=method
                )

            stacked_channels.append(stacked_image)

        # Stack the processed channels
        if len(stacked_channels) == 1:
            stacked_data = stacked_channels[0][
                np.newaxis, np.newaxis, :, :
            ]  # Shape: (1, 1, Y, X)
        else:
            stacked_data = np.stack(stacked_channels, axis=0)  # Shape: (C, Y, X)
            stacked_data = stacked_data[:, np.newaxis, :, :]  # Shape: (C, 1, Y, X)

        # Return 2D array if requested
        if return_2d:
            return np.squeeze(stacked_data)

        # Create new metadata for the stacked image
        stacked_metadata = FluorescenceImageMetadata(
            acquisition_date=self.metadata.acquisition_date,
            acquisition_datetime=self.metadata.acquisition_datetime,
            pixel_size_x=self.metadata.pixel_size_x,
            pixel_size_y=self.metadata.pixel_size_y,
            pixel_size_z=None,  # No z-dimension after stacking
            resolution=self.metadata.resolution,
            channels=selected_channels,
            z_positions=None,  # No z-positions after stacking
            stage_position=self.metadata.stage_position,
            filename=self.metadata.filename,
            description=self.metadata.description,
            system_info=self.metadata.system_info,
            # A projection of a lamella's image is still that lamella's. Carried
            # forward so a derived image does not lose what produced it (FIB-466).
            experiment=self.metadata.experiment,
            geometry=self.metadata.geometry,
            dimension_order=self.metadata.dimension_order,
        )

        return FluorescenceImage(
            data=stacked_data,
            metadata=stacked_metadata,
            # Still this file's image; see the note on the field (FIB-509).
            filepath=self.filepath,
        )

    @staticmethod
    def generate_blank_image(
        resolution: Tuple[int, int] = (1024, 1024),
        zlevels: int = 1,
        n_channels: int = 1,
        pixel_size: float = 1e-6,
        random: bool = False,
        dtype: np.dtype = np.uint8,
    ) -> "FluorescenceImage":
        """Generate a blank (or random noise) FluorescenceImage.

        Args:
            resolution: (width, height) in pixels.
            zlevels: Number of z-planes.
            n_channels: Number of fluorescence channels.
            pixel_size: Pixel size in meters (isotropic x/y).
            random: If True, fill with random noise; otherwise zeros.
            dtype: Array dtype. Defaults to np.uint8.

        Returns:
            FluorescenceImage with CZYX data and minimal valid metadata.
        """
        width, height = resolution
        shape = (n_channels, zlevels, height, width)

        if random:
            data = np.random.randint(0, 255, size=shape, dtype=dtype)
        else:
            data = np.zeros(shape=shape, dtype=dtype)

        channels = [
            FluorescenceChannelMetadata(
                name=f"Channel_{i + 1:02d}",
                excitation_wavelength=488.0,
                power=1.0,
                exposure_time=0.1,
                gain=1.0,
                offset=0.0,
            )
            for i in range(n_channels)
        ]

        metadata = FluorescenceImageMetadata(
            acquisition_date=now_iso(),
            pixel_size_x=pixel_size,
            pixel_size_y=pixel_size,
            resolution=resolution,
            channels=channels,
            z_positions=[i * pixel_size for i in range(zlevels)]
            if zlevels > 1
            else None,
        )

        return FluorescenceImage(data=data, metadata=metadata)


@dataclass
class FluorescenceChannelMetadata:
    """Metadata for a single fluorescence channel."""

    # Channel identification
    name: str

    # Optical parameters
    excitation_wavelength: float  # nm

    # Light source
    power: float  # light source power

    # Camera settings
    exposure_time: float  # seconds
    gain: Optional[float]  # camera gain (None when the camera has no gain control)
    offset: float  # camera offset

    # Fields with defaults must come after required fields
    emission_filter: EmissionFilter = REFLECTION
    binning: int = 1  # binning factor (1, 2, 4, 8)
    objective_position: float = 0.0  # z-axis position in meters
    objective_magnification: Optional[float] = None
    objective_numerical_aperture: Optional[float] = None
    color: str = "gray"  # Display color for the channel


@dataclass
class FluorescenceImageMetadata:
    """Complete metadata for fluorescence microscopy images."""

    # Image acquisition settings
    acquisition_date: str  # ISO timestamp string
    pixel_size_x: float  # meters, after binning
    pixel_size_y: float  # meters, after binning
    pixel_size_z: Optional[float] = None  # meters, for z-stacks
    resolution: Tuple[int, int] = (1024, 1024)  # (width, height) after binning

    # Channel metadata - list of individual channel settings
    channels: List[FluorescenceChannelMetadata] = field(default_factory=list)

    # Z-stack parameters (for multi-plane acquisitions)
    z_positions: Optional[List[float]] = None  # meters, objective positions

    # Stage position (optional, for correlative imaging)
    stage_position: Optional[FibsemStagePosition] = None

    # Geometry the image was captured under, for reprojecting stage positions onto it.
    # None on images written before this was recorded; reprojection raises rather than
    # guessing, since a marker drawn in the wrong place looks exactly like one drawn in
    # the right place.
    geometry: Optional[FibsemHardwareGeometry] = None

    # Which experiment, item and task produced this. The same record the beam path
    # carries, for the same reason: a forwarded file should say what produced it
    # without the surrounding directory (FIB-466). None on images written before this
    # was recorded, on ones acquired outside a workflow, and on ones written by other
    # software -- `from_ome` reconstructs from OME's own fields and cannot recover it.
    experiment: Optional[FibsemExperimentRef] = None

    # File information
    filename: Optional[str] = None  # original filename
    description: Optional[str] = None  # image description/notes

    # System information (optional)
    system_info: Optional[dict] = None
    dimension_order: str = "CZYX"  # default dimension order for OME-TIFF

    # When the image was acquired: an aware datetime, written as ISO 8601 with its
    # UTC offset -- the same field the beam images carry (FIB-1190). `FM.acquire_frame`
    # stamps it as the acquisition starts. `acquisition_date` is still written, with
    # the same value as a string, for OME and odemis; before this it held the
    # acquiring machine's clock time with no offset, which is all an older image has.
    # None on those, and on images built rather than acquired -- read the time
    # through `acquisition_datetime_of`, which falls back.
    acquisition_datetime: Optional[AwareTimestamp] = None

    def __post_init__(self):
        """Validate metadata consistency."""
        if not self.channels:
            raise ValueError("At least one channel must be specified")

        # Validate binning factors
        valid_binning = {1, 2, 4, 8}
        for channel in self.channels:
            if channel.binning not in valid_binning:
                raise ValueError(
                    f"Invalid binning factor {channel.binning}, must be one of {valid_binning}"
                )

        # Validate z-stack consistency
        if self.z_positions is not None:
            if len(self.z_positions) < 2:
                raise ValueError("Z-stack must have at least 2 positions")
            if self.pixel_size_z is None:
                # Auto-calculate z pixel size from positions
                self.pixel_size_z = abs(self.z_positions[1] - self.z_positions[0])

    def add_channel(self, channel: FluorescenceChannelMetadata) -> None:
        """Add a new channel to the metadata."""
        self.channels.append(channel)

    def to_dict(self) -> dict:
        return encode(self)

    @classmethod
    def from_dict(cls, metadata_dict: dict) -> "FluorescenceImageMetadata":
        """Read metadata written by any version (`fibsem.codec.upgrades.fluorescence`)."""
        from fibsem.codec.upgrades.fluorescence import image_metadata

        return decode(cls, image_metadata(metadata_dict))

    @classmethod
    def from_ome(cls, ome: OMEMetadata) -> "FluorescenceImageMetadata":
        """Convert OME metadata to FluorescenceImageMetadata with availability checks.

        For importing images from **external software**. `load` prefers the
        `FluorescenceImageMetadata` map annotation, which is what this package writes
        and which round-trips everything; this reconstructs from OME's own structured
        fields, so anything with no OME equivalent -- `system_info`, `geometry`,
        `experiment` -- comes back None. That is the honest answer for a file another
        program produced, not a gap to fill in with defaults.
        """

        # TODO: add test cases for this function
        # TODO: support loading power, gain, offset from OME if available

        if not ome.images:
            raise ValueError("OME metadata contains no images")

        image_md = ome.images[0]
        pixels_md = image_md.pixels

        if pixels_md is None:
            raise ValueError("OME metadata contains no pixel information")

        # Pixel sizes (convert to meters where possible)
        pixel_size_x = _convert_length_to_meters(
            pixels_md.physical_size_x, pixels_md.physical_size_x_unit
        )
        pixel_size_y = _convert_length_to_meters(
            pixels_md.physical_size_y, pixels_md.physical_size_y_unit
        )
        pixel_size_z = _convert_length_to_meters(
            pixels_md.physical_size_z, pixels_md.physical_size_z_unit
        )
        if pixel_size_x is None or pixel_size_y is None:
            raise ValueError("OME metadata missing pixel size information")

        # Image dimensions
        resolution = (pixels_md.size_x, pixels_md.size_y)

        # Collect per-plane data
        exposure_times = {}
        x_positions, y_positions, z_positions = [], [], []
        for plane in pixels_md.planes or []:
            if plane.the_c is not None and plane.exposure_time is not None:
                exposure_times.setdefault(plane.the_c, plane.exposure_time)
            if plane.position_x is not None:
                x_positions.append(
                    _convert_length_to_meters(plane.position_x, plane.position_x_unit)
                    if hasattr(plane, "position_x_unit")
                    else plane.position_x
                )
            if plane.position_y is not None:
                y_positions.append(
                    _convert_length_to_meters(plane.position_y, plane.position_y_unit)
                    if hasattr(plane, "position_y_unit")
                    else plane.position_y
                )
            if plane.position_z is not None:
                z_positions.append(
                    _convert_length_to_meters(plane.position_z, plane.position_z_unit)
                    if hasattr(plane, "position_z_unit")
                    else plane.position_z
                )

        # Stage position from mean plane positions if available
        stage_position = None
        if x_positions or y_positions or z_positions:
            stage_position = FibsemStagePosition(
                x=float(np.mean(x_positions)) if x_positions else 0.0,
                y=float(np.mean(y_positions)) if y_positions else 0.0,
                z=float(np.mean(z_positions)) if z_positions else 0.0,
                r=None,
                t=None,
                coordinate_system="RAW",
            )

        # Use z positions only if we have at least two valid entries
        valid_z_positions = z_positions if len(z_positions) >= 2 else None

        # Build channel metadata with fallbacks
        ome_filters = {
            found.id: found
            for instrument in ome.instruments
            for found in instrument.filters
        }
        channels_md: List[FluorescenceChannelMetadata] = []
        for idx, channel in enumerate(pixels_md.channels or []):
            det_settings = channel.detector_settings
            # ls_settings = channel.light_source_settings

            binning_enum = det_settings.binning if det_settings else None
            binning = BINNING_TO_INT.get(binning_enum, 1)

            objective_position = (
                valid_z_positions[idx]
                if valid_z_positions and idx < len(valid_z_positions)
                else (z_positions[0] if z_positions else 0.0)
            )

            channels_md.append(
                FluorescenceChannelMetadata(
                    name=channel.name or f"Channel_{idx + 1:02d}",
                    color=_color_to_hex(channel.color),
                    excitation_wavelength=channel.excitation_wavelength or 0.0,
                    emission_filter=emission_filter_from_ome(channel, ome_filters),
                    power=0.0,
                    exposure_time=exposure_times.get(idx, 0.0),
                    gain=1.0,
                    offset=0.0,
                    binning=binning,
                    objective_position=objective_position or 0.0,
                    objective_magnification=None,
                    objective_numerical_aperture=None,
                )
            )

        if not channels_md:
            raise ValueError("OME metadata contains no channel information")

        acquired = getattr(image_md, "acquisition_date", None)
        acquisition_date = acquired.isoformat() if acquired else now_iso()

        return cls(
            acquisition_date=acquisition_date,
            # Only when the file says which zone: a naive one is another machine's
            # clock time, and stays in acquisition_date alone.
            acquisition_datetime=to_aware(acquired),
            pixel_size_x=pixel_size_x,
            pixel_size_y=pixel_size_y,
            pixel_size_z=pixel_size_z,
            resolution=resolution,
            channels=channels_md,
            z_positions=valid_z_positions,
            stage_position=stage_position,
            filename=image_md.name,
            description=image_md.description,
            system_info=None,
            dimension_order=pixels_md.dimension_order.value
            if pixels_md.dimension_order is not None
            else "CZYX",
        )


class ObjectiveStartPosition(NamedEnum, Enum):
    """Where the objective is put before an overview begins.

    Only ever a focus value -- neither option inserts or retracts the objective, which
    is a different mechanism and a physical move no start-position setting should imply.
    A run started with the objective retracted has a problem this does not solve.
    """

    # Wherever it is. What every overview did before this existed, and still the
    # default: an overview is usually taken of what you have just been looking at.
    CURRENT = "current"
    # The focus a user saved with "Set Focus Position". Worth choosing when the
    # objective has since been moved by something unrelated -- and worth more with
    # autofocus set to `once`, whose sweep centres on wherever it starts, so beginning
    # from a known-good focus sweeps around the answer rather than around an accident.
    FOCUS = "focus"


@dataclass
class OverviewParameters(Configurable):
    """Parameters for FM overview/tileset acquisition.

    Attributes:
        rows: Number of tile rows.
        cols: Number of tile columns.
        overlap: Fractional overlap between adjacent tiles.
        use_zstack: Acquire a z-stack at each tile.
        autofocus_mode: When to autofocus during the traversal.
        tile_order: Traversal strategy over the grid.
        objective_start: Where the objective is put before the run begins.
        tile_mask: Optional per-tile enable mask, `tile_mask[row][col]`. None acquires
            every tile. Disabled tiles are skipped but keep their place: the mosaic is
            still the full grid size and acquired tiles land at the same canvas
            coordinates they would have in a dense overview.
        register_tiles: Experimental. Place each tile where its picture says it is,
            measured against its neighbours, rather than where the grid planned it.
            Only when the tiles match the stage steps and at least three pairs match;
            otherwise the planned mosaic is kept. As `OverviewAcquisitionSettings`.
    """

    rows: int = param(
        3,
        limits=RangeLimit(1, 100),
        doc="The number of tile rows.",
        display=Display("Rows"),
    )
    cols: int = param(
        3,
        limits=RangeLimit(1, 100),
        doc="The number of tile columns.",
        display=Display("Columns"),
    )
    overlap: float = param(
        0.1,
        limits=RangeLimit(0.0, 0.9),
        doc="How much neighbouring tiles overlap, as a fraction of a tile.",
        display=Display("Overlap", scale=100, unit="%", step=5.0, decimals=0),
    )
    use_zstack: bool = param(
        False,
        doc="Acquire a z-stack at each tile (with the task's z parameters), not a single plane.",
        display=Display("Z-Stack"),
    )
    autofocus_mode: AutoFocusMode = param(
        AutoFocusMode.NONE,
        doc="When to autofocus during the run: never, once, at each row or at each tile.",
        display=Display("Autofocus"),
    )
    tile_order: TileOrderStrategy = param(
        TileOrderStrategy.TYPEWRITER,
        doc="The order the tiles are visited in.",
        display=Display("Tile Order", advanced=True),
    )
    tile_mask: Optional[List[List[bool]]] = param(
        None,
        doc="Which tiles to acquire, as tile_mask[row][col]. Empty acquires every tile; a skipped tile keeps its place in the mosaic.",
        display=Display(hidden=True),
    )
    objective_start: ObjectiveStartPosition = param(
        ObjectiveStartPosition.CURRENT,
        doc="Where the objective is put before the run starts.",
        display=Display("Objective Start", advanced=True),
    )
    register_tiles: bool = param(
        False,
        doc="Experimental. Place each tile where its picture says it is, measured against its neighbours, rather than where the grid planned it. The planned mosaic is kept when the tiles don't match well enough.",
        display=Display("Register Tiles", advanced=True),
    )

    def to_dict(self) -> dict:
        return encode(self)

    @classmethod
    def from_dict(cls, ddict: dict) -> "OverviewParameters":
        return decode(cls, ddict)

    @property
    def n_enabled_tiles(self) -> int:
        """How many tiles will actually be acquired."""
        if self.tile_mask is None:
            return self.rows * self.cols
        return sum(bool(v) for row in self.tile_mask for v in row)


@dataclass
class CameraSettings:
    """Camera settings for fluorescence microscopy acquisition."""

    gain: float = 0.01  # 1%
    offset: float = 0.0
    binning: int = 1
    transform: CameraImageTransform = CameraImageTransform.NONE

    def to_dict(self) -> dict:
        return encode(self)

    @classmethod
    def from_dict(cls, ddict: dict) -> "CameraSettings":
        return decode(cls, ddict)


@dataclass
class FluorescenceConfiguration:
    """Complete FM configuration including all acquisition parameters."""

    channel_settings: List[ChannelSettings]
    z_parameters: ZParameters
    overview_parameters: OverviewParameters
    autofocus_settings: Optional[AutoFocusSettings] = None
    camera_settings: CameraSettings = field(default_factory=CameraSettings)
    focus_position: Optional[float] = None  # meters
    limit_position: Optional[float] = None  # meters
    default_orientation: str = "FM"

    def to_dict(self) -> dict:
        return encode(self)

    @classmethod
    def from_dict(cls, ddict: dict) -> "FluorescenceConfiguration":
        """Read a configuration saved by any version
        (`fibsem.codec.upgrades.fluorescence`)."""
        from fibsem.codec.upgrades.fluorescence import configuration

        return decode(cls, configuration(ddict))

    def export(self, filename: str) -> str:
        """Export FM configuration to YAML file.

        Args:
            filename: Output YAML filename

        Returns:
            str: Path to the exported file

        Example:
            >>> config = FluorescenceConfiguration(channels, z_params, overview_params)
            >>> config.export("my_config.yaml")
        """
        # Convert to dictionary
        config_dict = self.to_dict()

        # Write to YAML file
        with open(filename, "w") as f:
            yaml.dump(
                config_dict, f, default_flow_style=False, indent=4, sort_keys=False
            )

        logging.info(f"FM configuration exported to: {filename}")
        return filename

    @classmethod
    def load(cls, filename: str) -> "FluorescenceConfiguration":
        """Load FM configuration from YAML file.

        Args:
            filename: Path to YAML configuration file

        Returns:
            FluorescenceConfiguration: Loaded configuration object

        Example:
            >>> config = FluorescenceConfiguration.load("config.yaml")
            >>> print(f"Loaded {len(config.channel_settings)} channels")
        """
        with open(filename) as f:
            config_dict = yaml.safe_load(f)

        config = cls.from_dict(config_dict)
        logging.info(f"FM configuration loaded from: {filename}")
        return config
