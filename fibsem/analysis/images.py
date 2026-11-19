"""A saved image as `Planes`, through a reader per format.

`load_planes` asks each reader in `READERS` in turn whether it reads the file,
and the first that does reads it: a fluorescence stack, a beam image (any
TIFF, with or without fibsem's metadata), or a plain picture (PNG, JPEG). A
new format is a new reader, added to `READERS`."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from fibsem.fm.preview import channel_look, czyx, is_fluorescence_image
from fibsem.fm.structures import FluorescenceImage
from fibsem.imaging.export import fibsem_image_fields, saturation_value
from fibsem.structures.image import FibsemImage


@dataclass
class Planes:
    """An image's raw data as (layer, z, y, x), its layer names, and its pixel
    size and z step in metres when known."""

    data: np.ndarray
    names: List[str]
    pixel_size: Optional[float] = None
    z_step: Optional[float] = None
    full_value: Optional[float] = None
    fluorescence: bool = False

    @property
    def planes(self) -> int:
        return self.data.shape[1]

    def layer(self, name: Optional[str]) -> Tuple[str, np.ndarray]:
        """``(name, (z, y, x) data)`` of the layer *name*, the first when None.
        ValueError names the layers there are."""
        if name is None:
            return self.names[0], self.data[0]
        if name not in self.names:
            raise ValueError(
                f"no layer {name}: the image's layers are {', '.join(self.names)}"
            )
        return name, self.data[self.names.index(name)]

    def plane_range(self, planes: Optional[Sequence[int]]) -> List[int]:
        """*planes* checked against the stack, every plane when None."""
        if planes is None:
            return list(range(self.planes))
        bad = [z for z in planes if not 0 <= z < self.planes]
        if bad:
            raise ValueError(
                f"plane {bad[0]} is outside the stack's 0 to {self.planes - 1}"
            )
        return list(planes)


class Reader(NamedTuple):
    """A format: whether it reads the file at a path, and the read."""

    name: str
    reads: Callable[[str], bool]
    read: Callable[[str], Planes]


def _read_fluorescence(path: str) -> Planes:
    image = FluorescenceImage.load(path)
    data = czyx(image)
    channels = list(image.metadata.channels)
    names = [channel_look(channels, i, len(data))[0] for i in range(len(data))]
    md = image.metadata
    return Planes(
        data=data,
        names=names,
        pixel_size=md.pixel_size_x,
        z_step=md.pixel_size_z if data.shape[1] > 1 else None,
        full_value=saturation_value(data),
        fluorescence=True,
    )


def _read_beam(path: str) -> Planes:
    image = FibsemImage.load(path)
    data = np.asarray(image.data)
    if data.ndim != 2:
        raise ValueError(f"{path} is not a single greyscale image")
    md = image.metadata
    return Planes(
        data=data[None, None],
        names=[fibsem_image_fields(image).kind if md is not None else "Image"],
        pixel_size=md.pixel_size.x if md is not None and md.pixel_size else None,
        full_value=saturation_value(data),
    )


PICTURES = (".png", ".jpg", ".jpeg")


def _read_picture(path: str) -> Planes:
    """A PNG or JPEG as one greyscale plane, its scale unknown."""
    from PIL import Image

    with Image.open(path) as picture:
        data = np.asarray(picture.convert("L"))
    return Planes(data=data[None, None], names=["Image"], full_value=255.0)


def _suffix(*suffixes: str) -> Callable[[str], bool]:
    return lambda path: path.lower().endswith(suffixes)


READERS: List[Reader] = [
    Reader("fluorescence stack", is_fluorescence_image, _read_fluorescence),
    Reader("beam image", _suffix(".tif", ".tiff"), _read_beam),
    Reader("picture", _suffix(*PICTURES), _read_picture),
]


def load_planes(path: str) -> Planes:
    """The image at *path* as `Planes`, read by the first reader in `READERS`
    that reads it. ValueError when none does."""
    for reader in READERS:
        if reader.reads(path):
            return reader.read(path)
    raise ValueError(
        f"{os.path.basename(path)} is not an image this can read: "
        f"{', '.join(r.name for r in READERS)}"
    )
