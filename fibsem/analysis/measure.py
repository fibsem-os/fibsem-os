"""Measurements on a saved image, in its own values: what the agent's analysis
tools answer, each with a figure to look at and cite as evidence.

An image is read as `Planes` (`fibsem.analysis.images`): a beam image is one
layer of one plane, named by its kind (``SEM``, ``FIB``); a fluorescence stack
is a layer per channel, a plane per z. Points and lines are in the image's pixels, as marks are.

- `measure_intensity`: the signal at points, per plane, against the ring
  around each, and as a ratio to a reference point.
- `line_profile`: values along a line, averaged across its width, with the
  full width at half maximum of its peak (or dip).
- `image_quality`: clipping, centre and spread of the histogram (as the
  brightness and contrast routine judges a probe image), and focus.
- `histogram`: the counts of a layer's values, its percentiles and what is
  clipped at either end of the file's range.
- `power_spectrum`: the radially averaged power of a layer's 2D Fourier
  transform, the finest period with signal above the noise floor, and how
  much stronger it is along x than y.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from fibsem.analysis.images import Planes
from fibsem.fm.composite import auto_clim
from fibsem.fm.preview import plane_sharpness
from fibsem.imaging.marks import rgb_of

# A ring for a point's background: from this many radii out, to that many.
RING = (2.0, 3.0)
# The most planes a point's strip of crops shows.
STRIP_PLANES = 9
_DPI = 100


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


# --- histogram and power spectrum ---------------------------------------------


def _region(
    image: Planes,
    channel: Optional[str],
    plane: Optional[int],
    box: Optional[Sequence[float]],
) -> Tuple[str, np.ndarray]:
    """``(name, 2D data)`` of the layer *channel* on *plane* (else the max
    projection), cut to *box* ``[x, y, width, height]`` when given."""
    name, stack = image.layer(channel)
    if plane is not None:
        image.plane_range([plane])
        data = stack[plane]
    else:
        data = stack[0] if image.planes == 1 else stack.max(axis=0)
    if box is None:
        return name, data
    if len(box) != 4:
        raise ValueError("box is [x, y, width, height]")
    x, y, w, h = (int(round(v)) for v in box)
    height, width = data.shape
    left, top = max(0, x), max(0, y)
    right, bottom = min(width, x + w), min(height, y + h)
    if right - left < 2 or bottom - top < 2:
        raise ValueError(
            f"box {list(box)} holds less than 2x2 pixels of the {width}x{height} image"
        )
    return name, data[top:bottom, left:right]


def histogram(
    image: Planes,
    channel: Optional[str] = None,
    plane: Optional[int] = None,
    bins: int = 64,
    box: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """The counts of the layer *channel*'s values in *bins* equal bins, on
    *plane* (else the max projection) and only in *box* when given.

    The bins span the file's range (0 to its full value) when that is
    known, else its lowest to highest value. Answers the edges and counts,
    the percentiles, and the percent of pixels at the range's ends:
    clipped_low at its lowest value, clipped_high at its highest
    (saturated, when the full value is known)."""
    if not 2 <= bins <= 1024:
        raise ValueError("bins is 2 to 1024")
    name, data = _region(image, channel, plane, box)
    values = data.astype(np.float64).ravel()
    low, high = float(values.min()), float(values.max())
    full = image.full_value
    span = (0.0, float(full)) if full else (low, high if high > low else low + 1.0)
    counts, edges = np.histogram(values, bins=bins, range=span)
    p1, p5, median, p95, p99 = np.percentile(values, (1, 5, 50, 95, 99))
    top = float(full) if full else high
    return {
        "channel": name,
        "plane": plane,
        "box": list(box) if box is not None else None,
        "pixels": int(values.size),
        "range": [span[0], span[1]],
        "edges": [round(float(e), 4) for e in edges],
        "counts": [int(c) for c in counts],
        "percentiles": {
            "min": low,
            "p1": float(p1),
            "p5": float(p5),
            "median": float(median),
            "p95": float(p95),
            "p99": float(p99),
            "max": high,
        },
        "mean": round(float(values.mean()), 4),
        "std": round(float(values.std()), 4),
        "clipped_low": round(
            100.0 * np.count_nonzero(values <= span[0]) / values.size, 3
        ),
        "clipped_high": round(100.0 * np.count_nonzero(values >= top) / values.size, 3),
        "range_used": round((high - low) / (span[1] - span[0]), 4),
    }


def histogram_figure(image: Planes, result: Dict[str, Any]) -> np.ndarray:
    """The counts as bars on a log scale, the percentiles marked."""
    figure, grid = _figure(1, 1, row_height=3.0, width=7.0)
    axes = figure.add_subplot(grid[0, 0])
    edges = np.asarray(result["edges"])
    counts = np.asarray(result["counts"], dtype=float)
    axes.bar(
        edges[:-1],
        np.maximum(counts, 0.8),
        width=np.diff(edges),
        align="edge",
        color=_DIM,
        edgecolor=_BACKGROUND,
        lw=0.3,
        log=True,
    )
    shown = result["percentiles"]
    for key, style in (("p1", ":"), ("median", "-"), ("p99", ":")):
        axes.axvline(shown[key], color=_MARK, lw=1, ls=style)
    axes.set_xlim(edges[0], edges[-1])
    where = f"plane {result['plane']}" if result["plane"] is not None else ""
    if result["box"] is not None:
        where = (where + " box " + ",".join(f"{v:.0f}" for v in result["box"])).strip()
    title = (
        f"{result['channel']} {where}  median {shown['median']:.4g}, "
        f"p1 {shown['p1']:.4g}, p99 {shown['p99']:.4g}  "
        f"clipped {result['clipped_low']:.2g}% / {result['clipped_high']:.2g}%"
    )
    axes.set_title(title, fontsize=8, color=_TEXT, loc="left")
    axes.set_xlabel("value", fontsize=7, color=_TEXT)
    axes.set_ylabel("pixels", fontsize=7, color=_TEXT)
    _style(axes)
    return _rgb(figure)


# A frequency has signal when its power is this many times the noise floor.
SIGNAL_OVER_FLOOR = 2.0
# The noise floor: the median power of the outermost tenth of frequencies.
FLOOR_FRACTION = 0.1


def power_spectrum(
    image: Planes,
    channel: Optional[str] = None,
    plane: Optional[int] = None,
    box: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """The power of the layer *channel*'s 2D Fourier transform, on *plane*
    (else the max projection) and only in *box* when given, the region
    cut to a square and Hann-windowed so its edges add no streaks.

    Answers its radial average against spatial frequency (cycles per pixel,
    0 to 0.5), the noise floor (the median of the outermost tenth), the
    finest period whose power is still SIGNAL_OVER_FLOOR times the floor (in
    pixels, and in metres with the pixel size): how fine the detail is that
    the image holds, which falls with defocus or drift. anisotropy is the
    power along x over the power along y in the middle frequencies: near 1
    for a round probe, far from it for astigmatism, drift or scan lines."""
    name, data = _region(image, channel, plane, box)
    side = min(data.shape)
    if side < 16:
        raise ValueError(f"a power spectrum needs at least 16x16 pixels, not {side}")
    top = (data.shape[0] - side) // 2
    left = (data.shape[1] - side) // 2
    square = data[top : top + side, left : left + side].astype(np.float64)
    square -= square.mean()
    window = np.outer(np.hanning(side), np.hanning(side))
    power = np.abs(np.fft.fftshift(np.fft.fft2(square * window))) ** 2
    centre = side // 2
    yy, xx = np.indices(power.shape)
    dy, dx = yy - centre, xx - centre
    radius = np.hypot(dx, dy)
    shells = np.arange(1, side // 2)
    ring = np.rint(radius).astype(int)
    total = np.bincount(ring.ravel(), power.ravel())
    count = np.bincount(ring.ravel())
    radial = total[shells] / np.maximum(count[shells], 1)
    frequency = shells / side
    outer = max(1, int(round(len(radial) * FLOOR_FRACTION)))
    floor = float(np.median(radial[-outer:]))
    above = np.nonzero(radial > SIGNAL_OVER_FLOOR * floor)[0]
    cutoff = float(frequency[above[-1]]) if len(above) else None
    # The middle frequencies, away from the DC peak and the noise: power in
    # a 30 degree wedge about each axis.
    middle = (radius >= side * 0.05) & (radius <= side * 0.25)
    angle = np.degrees(np.arctan2(np.abs(dy), np.abs(dx)))
    along_x = float(power[middle & (angle <= 15)].mean())
    along_y = float(power[middle & (angle >= 75)].mean())
    pixel = image.pixel_size
    finest_px = 1.0 / cutoff if cutoff else None
    return {
        "channel": name,
        "plane": plane,
        "box": list(box) if box is not None else None,
        "size_px": side,
        "pixel_size": pixel,
        "frequency": [round(float(f), 5) for f in frequency],
        "radial_power": [float(f"{v:.4g}") for v in radial],
        "noise_floor": float(f"{floor:.4g}"),
        "cutoff": cutoff,
        "finest_period_px": round(finest_px, 3) if finest_px else None,
        "finest_period_m": finest_px * pixel if finest_px and pixel else None,
        "anisotropy": round(along_x / along_y, 3) if along_y > 0 else None,
    }


def spectrum_figure(image: Planes, result: Dict[str, Any]) -> np.ndarray:
    """The region's log power, the cutoff drawn as a circle, beside its
    radial average with the noise floor and cutoff."""
    from matplotlib.patches import Circle

    _, data = _region(image, result["channel"], result["plane"], result["box"])
    side = result["size_px"]
    top = (data.shape[0] - side) // 2
    left = (data.shape[1] - side) // 2
    square = data[top : top + side, left : left + side].astype(np.float64)
    square -= square.mean()
    window = np.outer(np.hanning(side), np.hanning(side))
    power = np.log10(np.abs(np.fft.fftshift(np.fft.fft2(square * window))) ** 2 + 1e-12)
    figure, grid = _figure(1, 2, row_height=3.2, width=8.0)
    axes = figure.add_subplot(grid[0, 0])
    vmin, vmax = np.percentile(power, (5, 99.9))
    axes.imshow(power, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    if result["cutoff"]:
        axes.add_patch(
            Circle(
                (side // 2, side // 2),
                result["cutoff"] * side,
                fill=False,
                lw=0.8,
                ls="--",
                color=_MARK,
            )
        )
    axes.set_xticks([])
    axes.set_yticks([])
    axes = figure.add_subplot(grid[0, 1])
    axes.semilogy(result["frequency"], result["radial_power"], color=_MARK, lw=1.2)
    axes.axhline(result["noise_floor"], color=_DIM, lw=0.8, ls=":")
    axes.axhline(SIGNAL_OVER_FLOOR * result["noise_floor"], color=_DIM, lw=0.8, ls="--")
    title = "no detail above the noise floor"
    if result["cutoff"]:
        axes.axvline(result["cutoff"], color=_MARK, lw=0.8, ls="--")
        title = f"finest period {result['finest_period_px']:.3g} px"
        if result["finest_period_m"]:
            unit, scale = _length_unit(result["finest_period_m"])
            title += f" ({result['finest_period_m'] / scale:.3g} {unit})"
    if result["anisotropy"] is not None:
        title += f", x/y {result['anisotropy']:.2f}"
    axes.set_title(title, fontsize=8, color=_TEXT, loc="left")
    axes.set_xlabel("frequency (cycles per pixel)", fontsize=7, color=_TEXT)
    _style(axes)
    return _rgb(figure)


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
    axes.tick_params(which="both", colors=_DIM, labelsize=6)
    for spine in axes.spines.values():
        spine.set_edgecolor(_DIM)


def _rgb(figure) -> np.ndarray:
    return rgb_of(figure)
