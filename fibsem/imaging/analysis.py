"""Measurements on a saved image, in its own values: what the agent's analysis
tools answer, each with a figure to look at and cite as evidence.

An image is read as `Planes`: a beam image is one layer of one plane, named by
its kind (``SEM``, ``FIB``); a fluorescence stack is a layer per channel, a
plane per z. Points and lines are in the image's pixels, as marks are.

- `measure_intensity`: the signal at points, per plane, against the ring
  around each, and as a ratio to a reference point.
- `line_profile`: values along a line, averaged across its width, with the
  full width at half maximum of its peak (or dip).
- `image_quality`: clipping, centre and spread of the histogram (as the
  brightness and contrast routine judges a probe image), and focus.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from fibsem.fm.composite import auto_clim
from fibsem.fm.preview import (
    channel_look,
    czyx,
    is_fluorescence_image,
    plane_sharpness,
)
from fibsem.fm.structures import FluorescenceImage
from fibsem.imaging.export import fibsem_image_fields, saturation_value
from fibsem.imaging.marks import rgb_of
from fibsem.structures.image import FibsemImage

# A ring for a point's background: from this many radii out, to that many.
RING = (2.0, 3.0)
# The most planes a point's strip of crops shows.
STRIP_PLANES = 9
_DPI = 100


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


def load_planes(path: str) -> Planes:
    """The image at *path*, a beam image or a fluorescence stack, as `Planes`."""
    if is_fluorescence_image(path):
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


# --- intensity at points -------------------------------------------------------


def _disc_and_ring(
    shape: Tuple[int, int], x: float, y: float, radius: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Masks of the disc of *radius* about (x, y), and of its background ring."""
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    r = np.hypot(xx - x, yy - y)
    return r <= radius, (r >= RING[0] * radius) & (r <= RING[1] * radius)


def measure_intensity(
    image: Planes,
    points: Sequence[Sequence[float]],
    radius_px: float = 3.0,
    channel: Optional[str] = None,
    reference: Optional[int] = None,
    planes: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Per point and plane of the layer *channel*: the peak and mean in a disc
    of *radius_px*, the median of the ring about it (2 to 3 radii out) as its
    background, and ``signal``, the mean above that background. A point's
    ``best_plane`` is where its signal is highest. With *reference* (a point's
    index), each point's best signal as a ratio to the reference's."""
    if radius_px <= 0:
        raise ValueError("radius_px is above 0.")
    if not points:
        raise ValueError("points is a list of [x, y] in the image's pixels.")
    if reference is not None and not 0 <= reference < len(points):
        raise ValueError(f"reference is a point's index, 0 to {len(points) - 1}.")
    name, stack = image.layer(channel)
    zs = image.plane_range(planes)
    height, width = stack.shape[1:]
    rows = []
    for x, y in points:
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(
                f"point [{x}, {y}] is outside the {width} x {height} image"
            )
        disc, ring = _disc_and_ring((height, width), x, y, radius_px)
        per_plane = []
        for z in zs:
            plane = stack[z]
            inside = plane[disc].astype(np.float64)
            around = plane[ring].astype(np.float64)
            background = float(np.median(around)) if around.size else 0.0
            mean = float(inside.mean())
            per_plane.append(
                {
                    "plane": z,
                    "peak": float(inside.max()),
                    "mean": round(mean, 3),
                    "background": round(background, 3),
                    "signal": round(mean - background, 3),
                }
            )
        best = max(per_plane, key=lambda p: p["signal"])
        saturated = image.full_value is not None and best["peak"] >= image.full_value
        rows.append(
            {
                "point": [float(x), float(y)],
                "best_plane": best["plane"],
                "signal": best["signal"],
                "background": best["background"],
                "peak": best["peak"],
                "saturated": bool(saturated),
                "planes": per_plane,
            }
        )
    if reference is not None:
        ref = rows[reference]["signal"]
        for row in rows:
            row["ratio"] = None if ref == 0 else round(row["signal"] / ref, 3)
    return {
        "channel": name,
        "radius_px": radius_px,
        "ring_px": [RING[0] * radius_px, RING[1] * radius_px],
        "reference": reference,
        "points": rows,
    }


def _strip_planes(best: int, zs: List[int]) -> List[int]:
    """The planes a strip shows: at most `STRIP_PLANES`, centred on *best*."""
    if len(zs) <= STRIP_PLANES:
        return zs
    at = zs.index(best)
    start = min(max(0, at - STRIP_PLANES // 2), len(zs) - STRIP_PLANES)
    return zs[start : start + STRIP_PLANES]


def intensity_figure(image: Planes, result: Dict[str, Any]) -> np.ndarray:
    """A row per point: a crop of each plane about it (one set of limits for
    the row, so its planes compare), the disc and ring drawn on the best, and
    beside them its signal and background against plane."""
    from matplotlib.patches import Circle

    _, stack = image.layer(result["channel"])
    radius = result["radius_px"]
    half = max(8, int(math.ceil(RING[1] * radius * 1.4)))
    rows = result["points"]
    columns = max(
        len(_strip_planes(r["best_plane"], [p["plane"] for p in r["planes"]]))
        for r in rows
    )
    crops = []
    for row in rows:
        x, y = row["point"]
        zs = _strip_planes(row["best_plane"], [p["plane"] for p in row["planes"]])
        crops.append([(z, _crop(stack[z], x, y, half)) for z in zs])
    figure, grid = _figure(len(rows), columns + 3, row_height=1.4)
    for i, (row, strip) in enumerate(zip(rows, crops)):
        # The point's own limits, background to peak: its planes compare, and
        # a bright point doesn't wash out a faint one.
        pixels = np.concatenate([c.ravel() for _, c in strip]).astype(np.float32)
        vmin, vmax = float(np.percentile(pixels, 1)), float(pixels.max())
        for j, (z, crop) in enumerate(strip):
            axes = figure.add_subplot(grid[i, j])
            axes.imshow(
                crop, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest"
            )
            axes.set_xticks([])
            axes.set_yticks([])
            axes.set_title(f"z {z}", fontsize=7, color=_TEXT, pad=2)
            if z == row["best_plane"]:
                centre = (half, half)
                for r, style in (
                    (radius, "-"),
                    (RING[0] * radius, ":"),
                    (RING[1] * radius, ":"),
                ):
                    axes.add_patch(
                        Circle(centre, r, fill=False, lw=0.8, ls=style, color=_MARK)
                    )
                for spine in axes.spines.values():
                    spine.set_edgecolor(_MARK)
        axes = figure.add_subplot(grid[i, columns:])
        zs = [p["plane"] for p in row["planes"]]
        axes.plot(
            zs,
            [p["signal"] for p in row["planes"]],
            color=_MARK,
            lw=1.2,
            label="signal",
        )
        axes.plot(
            zs,
            [p["background"] for p in row["planes"]],
            color=_DIM,
            lw=1,
            ls="--",
            label="background",
        )
        _style(axes)
        label = f"point {i}  ({row['point'][0]:.0f}, {row['point'][1]:.0f})"
        if row.get("ratio") is not None:
            label += f"  ratio {row['ratio']:.2f}"
        axes.set_title(label, fontsize=8, color=_TEXT, loc="left", pad=2)
        if i == len(rows) - 1:
            axes.set_xlabel("plane", fontsize=7, color=_TEXT)
        if i == 0:
            axes.legend(fontsize=6, frameon=False, labelcolor=_TEXT)
    return _rgb(figure)


def _crop(plane: np.ndarray, x: float, y: float, half: int) -> np.ndarray:
    """The square of side 2 *half* + 1 centred on (x, y), padded where the
    image ends so the point stays central."""
    cx, cy = int(round(x)), int(round(y))
    padded = np.pad(plane, half, mode="constant", constant_values=np.median(plane))
    return padded[cy : cy + 2 * half + 1, cx : cx + 2 * half + 1]


# --- a line's profile ----------------------------------------------------------


def line_profile(
    image: Planes,
    start: Sequence[float],
    end: Sequence[float],
    width_px: float = 1.0,
    channel: Optional[str] = None,
    plane: Optional[int] = None,
    dip: bool = False,
) -> Dict[str, Any]:
    """The layer *channel*'s values from *start* to *end* ([x, y] pixels), a
    sample per pixel, each the mean across *width_px*, on *plane* (the max
    projection when None). Answers the samples, their distance in pixels (and
    metres when the pixel size is known), and the full width at half maximum
    of the profile's peak (its dip with *dip*) above the baseline, the median
    of the profile's outer fifths. ``fwhm`` is None when the half maximum is
    not crossed on both sides."""
    from scipy.ndimage import map_coordinates

    name, stack = image.layer(channel)
    if plane is not None:
        image.plane_range([plane])
        data = stack[plane]
    else:
        data = stack.max(axis=0)
    if width_px < 1:
        raise ValueError("width_px is 1 or more.")
    (x0, y0), (x1, y1) = start, end
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 2:
        raise ValueError("a line is at least 2 pixels long.")
    count = int(round(length)) + 1
    t = np.linspace(0.0, 1.0, count)
    xs, ys = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
    nx, ny = -(y1 - y0) / length, (x1 - x0) / length
    offsets = np.arange(int(round(width_px))) - (int(round(width_px)) - 1) / 2
    samples = np.mean(
        [
            map_coordinates(
                data.astype(np.float64),
                [ys + o * ny, xs + o * nx],
                order=1,
                mode="nearest",
            )
            for o in offsets
        ],
        axis=0,
    )
    distance = t * length
    fwhm = _fwhm(distance, samples, dip)
    answer = {
        "channel": name,
        "plane": plane,
        "start": [float(x0), float(y0)],
        "end": [float(x1), float(y1)],
        "width_px": width_px,
        "length_px": round(length, 2),
        "distance_px": [round(float(d), 2) for d in distance],
        "values": [round(float(v), 3) for v in samples],
        "min": float(samples.min()),
        "max": float(samples.max()),
        **fwhm,
    }
    if image.pixel_size:
        answer["pixel_size"] = image.pixel_size
        answer["length_m"] = length * image.pixel_size
        if fwhm["fwhm_px"] is not None:
            answer["fwhm_m"] = fwhm["fwhm_px"] * image.pixel_size
    return answer


def _fwhm(distance: np.ndarray, values: np.ndarray, dip: bool) -> Dict[str, Any]:
    """The extremum, the baseline, the half level between them, and the width
    where the profile crosses it either side of the extremum."""
    fifth = max(1, len(values) // 5)
    baseline = float(np.median(np.concatenate([values[:fifth], values[-fifth:]])))
    at = int(np.argmin(values) if dip else np.argmax(values))
    extreme = float(values[at])
    half = (baseline + extreme) / 2
    sign = -1.0 if dip else 1.0
    above = sign * (values - half) >= 0
    left = right = None
    for i in range(at, 0, -1):
        if not above[i - 1]:
            left = _cross(distance, values, i - 1, i, half)
            break
    for i in range(at, len(values) - 1):
        if not above[i + 1]:
            right = _cross(distance, values, i, i + 1, half)
            break
    width = (
        None if left is None or right is None or extreme == baseline else right - left
    )
    return {
        "kind": "dip" if dip else "peak",
        "extremum_px": round(float(distance[at]), 2),
        "extremum": extreme,
        "baseline": round(baseline, 3),
        "half_level": round(half, 3),
        "half_crossings_px": None
        if width is None
        else [round(left, 2), round(right, 2)],
        "fwhm_px": None if width is None else round(width, 2),
    }


def _cross(distance, values, i: int, j: int, level: float) -> float:
    """Where the segment from sample *i* to *j* meets *level*."""
    a, b = float(values[i]), float(values[j])
    f = 0.5 if a == b else (level - a) / (b - a)
    return float(distance[i] + f * (distance[j] - distance[i]))


def profile_figure(image: Planes, result: Dict[str, Any]) -> np.ndarray:
    """The image about the line, the line and its width drawn, beside the
    profile with its baseline, half level and FWHM marked."""
    _, stack = image.layer(result["channel"])
    plane = result["plane"]
    data = stack[plane] if plane is not None else stack.max(axis=0)
    (x0, y0), (x1, y1) = result["start"], result["end"]
    pad = max(10.0, result["length_px"] * 0.25)
    left, right = (
        max(0, int(min(x0, x1) - pad)),
        min(data.shape[1], int(max(x0, x1) + pad) + 1),
    )
    top, bottom = (
        max(0, int(min(y0, y1) - pad)),
        min(data.shape[0], int(max(y0, y1) + pad) + 1),
    )
    region = data[top:bottom, left:right]
    vmin, vmax = auto_clim(region.astype(np.float32))
    figure, grid = _figure(1, 2, row_height=3.2, width=8.0)
    axes = figure.add_subplot(grid[0, 0])
    axes.imshow(
        region,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        extent=(left, right, bottom, top),
        interpolation="nearest",
    )
    axes.plot(
        [x0, x1],
        [y0, y1],
        color=_MARK,
        lw=max(1.0, result["width_px"] * 0.5),
        alpha=0.8,
    )
    axes.plot([x0], [y0], "o", color=_MARK, ms=3)
    axes.set_xticks([])
    axes.set_yticks([])
    axes = figure.add_subplot(grid[0, 1])
    distance = np.asarray(result["distance_px"])
    unit, scale = ("px", 1.0)
    if result.get("pixel_size"):
        unit, scale = _length_unit(result["length_m"])
        scale = result["pixel_size"] / scale
    axes.plot(distance * scale, result["values"], color=_MARK, lw=1.2)
    axes.axhline(result["baseline"], color=_DIM, lw=0.8, ls=":")
    axes.axhline(result["half_level"], color=_DIM, lw=0.8, ls="--")
    title = f"{result['kind']} at {result['extremum_px'] * scale:.3g} {unit}"
    if result["fwhm_px"] is not None:
        a, b = result["half_crossings_px"]
        axes.axvspan(a * scale, b * scale, color=_MARK, alpha=0.15, lw=0)
        title += f", FWHM {result['fwhm_px'] * scale:.3g} {unit}"
    else:
        title += ", FWHM not crossed"
    axes.set_title(title, fontsize=8, color=_TEXT, loc="left")
    axes.set_xlabel(f"distance ({unit})", fontsize=7, color=_TEXT)
    _style(axes)
    return _rgb(figure)


def _length_unit(metres: float) -> Tuple[str, float]:
    """The unit a length reads best in, and its size in metres."""
    if metres >= 1e-3:
        return "mm", 1e-3
    if metres >= 1e-6:
        return "µm", 1e-6
    return "nm", 1e-9


# --- image quality -------------------------------------------------------------


def image_quality(
    image: Planes, channel: Optional[str] = None, plane: Optional[int] = None
) -> Dict[str, Any]:
    """Whether the layer *channel* is good enough to go on with, on *plane*
    or else the max projection.

    A beam image is judged as the brightness and contrast routine judges one:
    the fraction clipped black and white, the centre and spread of its bulk on
    its type's range, what is wrong in words, and whether it is unusable. A
    fluorescence channel has no such targets (its signal is sparse and its
    camera rarely fills the type's range): it answers its percentiles, the
    percent of pixels at the file's full value, and the 99th percentile as a
    fraction of it. Both answer focus (the mean Laplacian; only comparable
    between images of one scene), and a stack each plane's sharpness relative
    to the sharpest, and which that is."""
    from fibsem.autofunctions.metrics import calculate_focus_quality

    name, stack = image.layer(channel)
    if plane is not None:
        image.plane_range([plane])
        data = stack[plane]
    else:
        data = stack[0] if image.planes == 1 else stack.max(axis=0)
    answer: Dict[str, Any] = {"channel": name, "plane": plane}
    if image.fluorescence:
        answer.update(_signal_range(data, image.full_value))
    else:
        answer.update(_beam_histogram(data))
    answer["focus"] = round(calculate_focus_quality(data.astype(np.float64)), 4)
    if image.planes > 1:
        sharpness = plane_sharpness(stack)
        answer["plane_sharpness"] = [round(s, 3) for s in sharpness]
        answer["sharpest_plane"] = int(np.argmax(sharpness))
    return answer


def _beam_histogram(data: np.ndarray) -> Dict[str, Any]:
    from fibsem.autofunctions.brightness_contrast import (
        BrightnessContrastSettings,
        measure,
        problems,
        unusable,
    )

    settings = BrightnessContrastSettings()
    stats = measure(data, settings)
    return {
        "histogram": {k: round(v, 4) for k, v in stats.to_dict().items()},
        "problems": problems(stats, settings),
        "unusable": unusable(stats, settings),
        "targets": {
            "max_clipped": settings.max_clipped,
            "centre": [settings.min_centre, settings.max_centre],
            "min_spread": settings.min_spread,
        },
    }


def _signal_range(data: np.ndarray, full: Optional[float]) -> Dict[str, Any]:
    p1, median, p99, p999 = np.percentile(data, (1, 50, 99, 99.9))
    answer = {
        "percentiles": {
            "p1": float(p1),
            "median": float(median),
            "p99": float(p99),
            "p99_9": float(p999),
            "max": float(data.max()),
        },
        "saturated": None,
        "range_used": None,
    }
    if full:
        answer["saturated"] = round(
            100.0 * np.count_nonzero(data >= full) / data.size, 3
        )
        answer["range_used"] = round(float(p999) / full, 4)
    return answer


# --- figures -------------------------------------------------------------------

_BACKGROUND = "#16181d"
_TEXT = "#d8dce4"
_DIM = "#8a93a3"
_MARK = "#f2b84b"


def _figure(rows: int, columns: int, row_height: float, width: Optional[float] = None):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    width = width or min(14.0, 1.1 * columns)
    figure = Figure(
        figsize=(width, row_height * rows + 0.4), dpi=_DPI, facecolor=_BACKGROUND
    )
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(rows, columns, wspace=0.15, hspace=0.45)
    return figure, grid


def _style(axes) -> None:
    axes.set_facecolor(_BACKGROUND)
    axes.tick_params(colors=_DIM, labelsize=6)
    for spine in axes.spines.values():
        spine.set_edgecolor(_DIM)


def _rgb(figure) -> np.ndarray:
    return rgb_of(figure)
