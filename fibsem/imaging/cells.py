"""An image cut into named cells for looking over a large overview: columns
lettered A, B, ... left to right and rows numbered 1, 2, ... top down, so
"B3" is the second column's third cell.

A tiled overview's cells are its tiles, each the share of the mosaic it was
pasted into (the overlap split down the middle), read from the run's record
beside the mosaic (``<stem>/overview-metadata.json``). Any other image is cut
into a plain grid of cells about `CELL_SIDE` source pixels across.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

# A plain grid's cells are about this many source pixels across: a cell
# cropped at native resolution fits a 768 px preview.
CELL_SIDE = 768
MAX_SIDE_CELLS = 12
# At most this many cells cropped in one call.
MAX_CROPS = 8
# A crop's picture is at most this wide (the files route's preview).
CROP_WIDTH = 768
TILES, GRID = "tiles", "grid"


@dataclass
class Cell:
    name: str
    x: int
    y: int
    width: int
    height: int

    @property
    def box(self) -> List[int]:
        return [self.x, self.y, self.width, self.height]

    @property
    def centre(self) -> Tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2


def column_name(index: int) -> str:
    """A, B, ... Z, AA, AB, ..."""
    name = ""
    index += 1
    while index:
        index, rest = divmod(index - 1, 26)
        name = chr(ord("A") + rest) + name
    return name


def _record(path: str) -> Optional[dict]:
    from fibsem.imaging.marks import image_stem
    from fibsem.imaging.tiling.result import METADATA_FILE

    record = os.path.join(os.path.dirname(path), image_stem(path), METADATA_FILE)
    try:
        with open(record) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def tile_layout(path: str) -> Optional[Tuple[int, int, float]]:
    """``(columns, rows, overlap)`` of the tiled overview at *path*, from its
    run's record; None for an image that isn't one, or a single tile."""
    record = _record(path)
    if not record:
        return None
    tiles = record.get("tiles") or []
    settings = record.get("settings") or {}
    overlap = settings.get("overlap", (settings.get("overview") or {}).get("overlap"))
    try:
        columns = max(int(t["col"]) for t in tiles) + 1
        rows = max(int(t["row"]) for t in tiles) + 1
        overlap = float(overlap or 0.0)
    except (KeyError, TypeError, ValueError):
        return None
    if columns * rows < 2 or not 0.0 <= overlap < 1.0:
        return None
    return columns, rows, overlap


def _tile_edges(size: int, count: int, overlap: float) -> List[int]:
    """Where each tile's share of a mosaic *size* pixels long starts, and the
    end: tiles pasted ``(1 - overlap)`` of a tile apart, each overlap split
    down the middle."""
    tile = size / ((count - 1) * (1 - overlap) + 1)
    step = (size - tile) / (count - 1) if count > 1 else 0.0
    edges = [0] + [round(j * step + (tile - step) / 2) for j in range(1, count)]
    return edges + [size]


def _even_edges(size: int, count: int) -> List[int]:
    return [round(size * j / count) for j in range(count + 1)]


def _grid_counts(width: int, height: int) -> Tuple[int, int]:
    columns = min(MAX_SIDE_CELLS, max(2, math.ceil(width / CELL_SIDE)))
    rows = min(MAX_SIDE_CELLS, max(2, round(columns * height / width)))
    return columns, rows


def image_cells(
    width: int, height: int, layout: Optional[Tuple[int, int, float]] = None
) -> Tuple[str, List[Cell]]:
    """An image *width* x *height* source pixels as cells: its tiles when a
    *layout* (`tile_layout`) is given, else a plain grid. ``(kind, cells)``,
    row by row."""
    if layout is not None:
        columns, rows, overlap = layout
        xs = _tile_edges(width, columns, overlap)
        ys = _tile_edges(height, rows, overlap)
        kind = TILES
    else:
        columns, rows = _grid_counts(width, height)
        xs, ys = _even_edges(width, columns), _even_edges(height, rows)
        kind = GRID
    cells = [
        Cell(
            f"{column_name(c)}{r + 1}",
            xs[c],
            ys[r],
            xs[c + 1] - xs[c],
            ys[r + 1] - ys[r],
        )
        for r in range(rows)
        for c in range(columns)
    ]
    return kind, cells


def cells_of(path: str, shape: Sequence[int]) -> Tuple[str, List[Cell]]:
    """The saved image at *path* (its data *shape*) as cells."""
    return image_cells(int(shape[1]), int(shape[0]), tile_layout(path))


def named(cells: Sequence[Cell], names: Sequence[str]) -> List[Cell]:
    """The cells *names* name, in that order. Raises ValueError naming the
    valid range for a name the image doesn't have."""
    found = {cell.name: cell for cell in cells}
    picked = []
    for name in names:
        cell = found.get(str(name).strip().upper())
        if cell is None:
            raise ValueError(
                f"No cell {name!r}: this image's cells are {cells[0].name} "
                f"to {cells[-1].name}."
            )
        picked.append(cell)
    return picked


def crop_box(path: str, crop: Any) -> List[float]:
    """The source-pixel box ``[x, y, width, height]`` a *crop* names on the
    saved image at *path*: a cell's name, or the box itself. Raises
    ValueError."""
    if isinstance(crop, str):
        from fibsem.geometry.saved_images import image_shape

        (cell,) = named(cells_of(path, image_shape(path))[1], [crop])
        return [float(v) for v in cell.box]
    if (
        isinstance(crop, (list, tuple))
        and len(crop) == 4
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in crop)
    ):
        return [float(v) for v in crop]
    raise ValueError("crop is a cell's name (B3) or a box [x, y, width, height].")


def cell_lines(axes, cells: Sequence[Cell], scale: float, width: float) -> None:
    """*cells*' edges on *axes* (in pixels *scale* times the source's): thin
    white lines over a dark halo, so they show on any picture."""
    import matplotlib.patheffects as effects

    from fibsem.imaging.marks import PT

    line = max(1.0, width / 700) * PT
    halo = [effects.withStroke(linewidth=line * 3, foreground=(0, 0, 0, 0.5))]
    for cell in cells:
        x, y = cell.x * scale, cell.y * scale
        w, h = cell.width * scale, cell.height * scale
        axes.plot(
            [x, x + w, x + w, x, x],
            [y, y, y + h, y + h, y],
            color="white",
            linewidth=line,
            alpha=0.8,
            path_effects=halo,
        )


def cells_picture(data: np.ndarray, cells: Sequence[Cell], scale: float) -> np.ndarray:
    """*data* (a preview, uint8, *scale* of the source's pixels) with *cells*
    drawn on it and each named in its top-left corner. RGB, *data*'s size."""
    from fibsem.imaging.labels import LabelPlacer
    from fibsem.imaging.marks import canvas, rgb_of

    figure, axes, _ = canvas(data, data.shape[1])
    cell_lines(axes, cells, scale, data.shape[1])
    placer = LabelPlacer(axes, 1.0)
    inset = placer.size * 0.3
    for cell in cells:
        placer.put(cell.x * scale + inset, cell.y * scale + inset, cell.name, "white")
    return rgb_of(figure)


def from_crop(path: str, crop: Any) -> Tuple[float, float, float, List[float]]:
    """How a point read off the picture of a *crop* (`crop_box`) of the saved
    image at *path* maps to the image's pixels: ``(scale, x0, y0, box)``, the
    image's x being ``x * scale + x0``. The picture is the crop at native
    resolution, or narrowed to `CROP_WIDTH` as the crop answer shows it."""
    box = crop_box(path, crop)
    x0, y0, width, _ = box
    if width <= 0 or box[3] <= 0:
        raise ValueError("A crop's width and height are more than 0.")
    return width / min(width, CROP_WIDTH), x0, y0, box
