"""How fibsem writes image files, and how it finds its metadata in any it ever wrote.

Format 1 (fibsem 0.60) is OME-TIFF for every image: FIB/SEM images carry OME-XML
with our full metadata kept inside it as an OME map annotation, as FM stacks
already did. Pages are zlib-compressed. A large image or any z-stack also carries
1/2, 1/4 and 1/8 copies of each page (SubIFDs, which readers that don't know them
skip), and a large image is stored in 256 px tiles, so a viewer can read a small
copy or one area without the whole file.

Files written before 0.60 (format 0) hold a FIB/SEM image's metadata as JSON in
the TIFF description; they are read as they are and never rewritten.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import tifffile as tff

FORMAT_KEY = "fibsem_format"
FORMAT_VERSION = 1
METADATA_KEY = "FibsemImageMetadata"

# A page longer than this on its long side gets the small copies and is tiled.
PYRAMID_ABOVE = 2048
# The small copies, as reduction factors of the full page. Each halves the one before,
# as tiled viewers (deck.gl, Viv) take a pyramid's levels to.
LEVELS: Tuple[int, ...] = (2, 4, 8)
TILE = 256
COMPRESSION = "zlib"
# Image data above this many bytes is saved as BigTIFF: a classic TIFF can't hold
# 4 GB, and the margin leaves room for the OME header. Smaller files stay classic
# TIFF, which more readers open.
BIGTIFF_ABOVE = 2**32 - 2**25


def has_levels(shape: Sequence[int], stack: bool = False) -> bool:
    """Whether a page of (..., Y, X) *shape* gets the small copies: a large page,
    or any page of a z-stack, whose planes add up however small each is."""
    return stack or max(shape[-2:]) > PYRAMID_ABOVE


def write(path: str, data: np.ndarray, description: str, levels: bool) -> None:
    """Write *data* (..., Y, X) as one series of compressed pages, page 0 described
    by *description*; with *levels*, each page carries its small copies, and a page
    above PYRAMID_ABOVE is tiled."""
    from fibsem.imaging.reduce import reduce_by

    options: Dict[str, Any] = {
        "photometric": "minisblack",
        "compression": COMPRESSION,
        # Level 1: within a few % of the default's size on microscopy, at half the time.
        "compressionargs": {"level": 1},
        # the horizontal predictor is what makes 16-bit microscopy compress
        "predictor": np.issubdtype(data.dtype, np.integer),
        "metadata": None,
        # Pages in tiles or strips compress in parallel, which with the smaller
        # file makes a save faster than writing it uncompressed.
        "maxworkers": 4,
    }
    if max(data.shape[-2:]) > PYRAMID_ABOVE:
        options["tile"] = (TILE, TILE)
    else:
        options["rowsperstrip"] = TILE
    with tff.TiffWriter(path, bigtiff=data.nbytes > BIGTIFF_ABOVE, ome=False) as tif:
        tif.write(
            data,
            description=description,
            subifds=len(LEVELS) if levels else 0,
            **options,
        )
        if levels:
            pages, done = data.reshape(-1, *data.shape[-2:]), 1
            # cv2 lets go of the GIL, so a stack's planes reduce in parallel
            with ThreadPoolExecutor(options["maxworkers"]) as pool:
                for factor in LEVELS:
                    step = factor // done
                    pages = np.stack(
                        list(pool.map(lambda page: reduce_by(page, step), pages))
                    )
                    done = factor
                    tif.write(
                        pages.reshape(*data.shape[:-2], *pages.shape[-2:]),
                        subfiletype=1,
                        **options,
                    )


def annotation(ome_xml: str, key: str) -> Optional[str]:
    """The value under *key* in an OME map annotation of *ome_xml*, or None."""
    try:
        # bytes, as an XML declaration naming its encoding is refused in a str
        root = ET.fromstring(ome_xml.encode("utf-8"))
    except ET.ParseError:
        return None
    for m in root.iterfind(".//{*}M"):
        if m.get("K") == key:
            return m.text
    return None


def metadata_dict(description: Optional[str]) -> Optional[dict]:
    """A FIB/SEM image's metadata, as saved, from its page 0 *description*: the OME
    annotation (format 1) or the JSON description (format 0). None for a file that
    records none, such as an FM stack or a plain TIFF."""
    if not description:
        return None
    text = description.lstrip()
    if text.startswith("<"):
        text = annotation(text, METADATA_KEY)
        if text is None:
            return None
    try:
        value = json.loads(text)
    except ValueError:
        logging.debug("Image description is neither OME nor JSON")
        return None
    return value if isinstance(value, dict) else None


def read_metadata_dict(tif: "tff.TiffFile") -> Optional[dict]:
    """`metadata_dict` of an open file's page 0."""
    try:
        description = tif.pages[0].tags["ImageDescription"].value
    except (KeyError, IndexError):
        return None
    return metadata_dict(description)


def beam_image_ome(data: np.ndarray, metadata: Optional[Any]) -> str:
    """The OME-XML of a FIB/SEM image: pixel size, the detector as its one channel,
    the stage position, and the whole *metadata* (a FibsemImageMetadata, or None)
    as saved, under METADATA_KEY."""
    from ome_types.model import (
        OME,
        Channel,
        Image,
        Instrument,
        MapAnnotation,
        Microscope,
        Pixels,
        Pixels_DimensionOrder,
        Plane,
        StructuredAnnotations,
        TiffData,
        UnitsLength,
    )

    from fibsem.structures import BeamType

    ny, nx = data.shape[-2:]
    values = {FORMAT_KEY: str(FORMAT_VERSION)}
    channel_name, plane, pixel_size, name, when, instruments = (
        None,
        {},
        None,
        None,
        None,
        [],
    )
    if metadata is not None:
        values[METADATA_KEY] = json.dumps(metadata.to_dict())
        settings = metadata.image_settings
        state = metadata.microscope_state
        name = settings.filename
        pixel_size = metadata.pixel_size
        when = metadata.acquisition_date
        detector = None
        if state is not None:
            ion = settings.beam_type is BeamType.ION
            detector = state.ion_detector if ion else state.electron_detector
            stage = state.stage_position
            if stage is not None:
                plane = {
                    f"position_{axis}": getattr(stage, axis)
                    for axis in ("x", "y", "z")
                    if getattr(stage, axis, None) is not None
                }
                plane.update({f"{key}_unit": UnitsLength.METER for key in list(plane)})
        beam = getattr(settings.beam_type, "name", None)
        parts = [
            beam.title() if beam else None,
            detector.type if detector else None,
            detector.mode if detector else None,
        ]
        channel_name = " ".join(p for p in parts if p and p != "Unknown") or None
        info = metadata.system_info
        if info is not None and (info.manufacturer or info.model):
            instruments = [
                Instrument(
                    id="Instrument:0",
                    microscope=Microscope(
                        manufacturer=info.manufacturer,
                        model=info.model,
                        serial_number=info.serial_number,
                    ),
                )
            ]
    sizes = {}
    if pixel_size is not None and pixel_size.x and pixel_size.y:
        sizes = {
            "physical_size_x": pixel_size.x,
            "physical_size_y": pixel_size.y,
            "physical_size_x_unit": UnitsLength.METER,
            "physical_size_y_unit": UnitsLength.METER,
        }
    image = Image(
        id="Image:0",
        name=name,
        acquisition_date=when,
        pixels=Pixels(
            id="Pixels:0",
            dimension_order=Pixels_DimensionOrder.XYZCT,
            size_x=nx,
            size_y=ny,
            size_z=1,
            size_c=1,
            size_t=1,
            type=data.dtype.name,
            channels=[
                Channel(id="Channel:0:0", name=channel_name, samples_per_pixel=1)
            ],
            planes=[Plane(the_c=0, the_t=0, the_z=0, **plane)],
            tiff_data_blocks=[TiffData(ifd=0, plane_count=1)],
            **sizes,
        ),
    )
    ome = OME(
        images=[image],
        instruments=instruments,
        structured_annotations=StructuredAnnotations(
            map_annotations=[MapAnnotation(id="Annotation:0", value=values)]
        ),
    )
    return ome.to_xml()
