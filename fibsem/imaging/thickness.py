"""Lamella thickness from a FIB image: the bright band between the two trenches.

Seen from the ion beam, a lamella is a bright band running across the image with
a dark trench milled on each side. Its width at half its peak brightness (FWHM),
measured column by column, is a top-down estimate of the lamella's thickness: it
cannot see a lamella that is thicker at the bottom than at the top.

Agents measured this by hand with their own scripts during polishing (2026-10-15
Arctis session, FIB-1434); this is the one measurement, on the full-resolution
image and its own pixel size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from fibsem.conversions import microscope_image_to_image_coordinates
from fibsem.structures import FibsemImage, FibsemRectangle, Point

# Under this many pixels across, the FWHM is too coarse to trust (2026-10-15:
# at 39 nm per pixel a 200 nm band was 5 px). The suggested field of view puts
# the band at SUGGESTED_BAND_PX, twice that, so it stays measurable while it thins.
MIN_BAND_PX = 8
SUGGESTED_BAND_PX = 16
# The window searched around a point (the item's POI): along the lamella, and
# across it -- wide enough for a 1 um band and the start of both trenches.
DEFAULT_WINDOW = (10e-6, 4e-6)
# A column whose band is fainter than this fraction of the typical contrast has
# no band: a hole or a break.
MIN_CONTRAST_FRACTION = 0.5

Region = Union[FibsemRectangle, Point, None]


@dataclass
class LamellaBandMeasurement:
    """The band, column by column, and what follows from it.

    Pixel quantities are in the source image's pixels, origin top-left, +y down.
    ``top`` and ``bottom`` are the half-maximum edges of each column of
    ``columns`` (NaN where the column was dropped). ``reference`` is the point
    offsets are measured from: the region's centre, the POI when a point was given.
    """

    pixel_size: float  # metres per pixel
    image_width: int  # pixels
    region: Tuple[int, int, int, int]  # x0, y0, x1, y1 searched (end exclusive)
    reference: Tuple[float, float]  # x, y
    columns: np.ndarray
    top: np.ndarray
    bottom: np.ndarray
    band_level: float  # mean brightness inside the band
    trench_level: float  # mean brightness of the darkest trench pixels

    @property
    def measured(self) -> np.ndarray:
        return np.isfinite(self.top) & np.isfinite(self.bottom)

    @property
    def width_px(self) -> np.ndarray:
        """The band width of each measured column, in pixels."""
        return (self.bottom - self.top)[self.measured]

    @property
    def thickness(self) -> np.ndarray:
        """The band width of each measured column, in metres."""
        return self.width_px * self.pixel_size

    def centre_line(self) -> Tuple[float, float]:
        """The band's centre as a line ``y = slope * x + intercept`` (pixels)."""
        x = self.columns[self.measured]
        centre = ((self.top + self.bottom) / 2)[self.measured]
        if len(x) < 2:
            return 0.0, float(centre[0]) if len(centre) else math.nan
        slope, intercept = np.polyfit(x, centre, 1)
        return float(slope), float(intercept)

    def gaps(self) -> List[Tuple[int, int]]:
        """Runs of dropped columns, as (first, last) image x: holes or breaks."""
        runs: List[Tuple[int, int]] = []
        for x, ok in zip(self.columns, self.measured):
            if ok:
                continue
            if runs and runs[-1][1] == x - 1:
                runs[-1] = (runs[-1][0], int(x))
            else:
                runs.append((int(x), int(x)))
        return runs

    def to_dict(self) -> Dict[str, Any]:
        """The numbers an agent or a report reads, in nm and degrees."""
        nm = self.pixel_size * 1e9
        measured = self.measured
        thick = self.thickness * 1e9
        n_columns = len(self.columns)
        answer: Dict[str, Any] = {
            "pixel_size_nm": round(nm, 3),
            "region_px": list(self.region),
            "continuity": {
                "columns": n_columns,
                "measured": int(measured.sum()),
                "dropped": int(n_columns - measured.sum()),
                "gaps_px": [list(gap) for gap in self.gaps()],
            },
            "uncertainty_nm": round(nm, 1),  # one pixel either way
        }
        if not len(thick):
            answer["thickness_nm"] = None
            return answer

        x = self.columns[measured]
        answer["thickness_nm"] = {
            "mean": _nm(thick.mean()),
            "median": _nm(np.median(thick)),
            "min": _nm(thick.min()),
            "max": _nm(thick.max()),
            "std": _nm(thick.std()),
        }
        # Thirds by position along the region, so a gap does not shift them.
        x0, x1 = self.region[0], self.region[2]
        edges = np.linspace(x0, x1, 4)
        answer["thirds_nm"] = [
            _nm(thick[(x >= lo) & (x < hi)].mean())
            if np.any((x >= lo) & (x < hi))
            else None
            for lo, hi in zip(edges[:-1], edges[1:])
        ]

        # Wedge: how thickness changes along the lamella.
        if len(x) >= 2:
            per_px = float(np.polyfit(x, thick, 1)[0])
            span_px = float(x.max() - x.min())
            answer["wedge"] = {
                "slope_nm_per_um": _nm(per_px / (self.pixel_size * 1e6)),
                "across_band_nm": _nm(per_px * span_px),
            }

        # Tilt and straightness of the centre line. The angle is counter-clockwise
        # on screen, as the band would be rotated: positive rises to the right.
        slope, intercept = self.centre_line()
        centre = ((self.top + self.bottom) / 2)[measured]
        residual = centre - (slope * x + intercept)
        answer["angle_deg"] = round(math.degrees(math.atan(-slope)), 3)
        answer["straightness"] = {
            "centre_spread_px": round(float(residual.std()), 2),
            "centre_spread_nm": _nm(residual.std() * nm),
        }

        # Where the band sits relative to the reference (the POI): positive is
        # above it on screen, as the microscope's image coordinates point.
        ref_x, ref_y = self.reference
        band_y = slope * ref_x + intercept
        answer["centre_offset_nm"] = _nm((ref_y - band_y) * nm)

        answer["contrast"] = {
            "band": round(self.band_level, 2),
            "trench": round(self.trench_level, 2),
            "ratio": round(self.band_level / self.trench_level, 3)
            if self.trench_level > 0
            else None,
        }

        band_px = float(np.median(self.width_px))
        resolution: Dict[str, Any] = {
            "band_px": round(band_px, 1),
            "under_resolved": band_px < MIN_BAND_PX,
            "min_band_px": MIN_BAND_PX,
        }
        if band_px < MIN_BAND_PX:
            target_pixel = np.median(self.thickness) / SUGGESTED_BAND_PX
            resolution["suggested_field_width_um"] = round(
                float(target_pixel * self.image_width * 1e6), 2
            )
        answer["resolution"] = resolution
        return answer


def _nm(value) -> float:
    return round(float(value), 1) + 0.0  # no -0.0


def _region_px(
    region: Region, shape: Tuple[int, int], pixel_size: float, window
) -> Tuple[Tuple[int, int, int, int], Tuple[float, float]]:
    """The pixel box to search, and the reference point offsets are taken from."""
    height, width = shape
    if region is None:
        x0, y0, x1, y1 = 0, 0, width, height
        return (x0, y0, x1, y1), (width / 2, height / 2)
    if isinstance(region, FibsemRectangle):
        x0 = int(round(region.left * width))
        y0 = int(round(region.top * height))
        x1 = int(round((region.left + region.width) * width))
        y1 = int(round((region.top + region.height) * height))
        reference = ((x0 + x1) / 2, (y0 + y1) / 2)
    elif isinstance(region, Point):
        centre = microscope_image_to_image_coordinates(
            region, shape, pixel_size, subpixel_precision=True
        )
        half_w = window[0] / pixel_size / 2
        half_h = window[1] / pixel_size / 2
        x0, x1 = int(round(centre.x - half_w)), int(round(centre.x + half_w))
        y0, y1 = int(round(centre.y - half_h)), int(round(centre.y + half_h))
        reference = (centre.x, centre.y)
    else:
        raise TypeError("region is a FibsemRectangle, a Point or None.")
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 - x0 < 2 or y1 - y0 < 5:
        raise ValueError("The region is outside the image, or too small to measure.")
    return (x0, y0, x1, y1), reference


def _smooth(profile: np.ndarray) -> np.ndarray:
    """A [1, 2, 1] smoothing that leaves a step's half-way crossing in place."""
    padded = np.pad(profile, 1, mode="edge")
    return (padded[:-2] + 2 * padded[1:-1] + padded[2:]) / 4


def _crossing(profile: np.ndarray, start: int, step: int, level: float):
    """Where *profile* first falls below *level* walking from *start* by *step*,
    interpolated between the two pixels; None if it never does."""
    i = start
    while 0 <= i + step < len(profile):
        nxt = i + step
        if profile[nxt] < level:
            a, b = profile[i], profile[nxt]
            frac = (a - level) / (a - b) if a != b else 0.5
            return i + step * frac
        i = nxt
    return None


def _column_band(profile: np.ndarray, guess: float, half_window: float):
    """One column's band: (top, bottom, peak, trench), or None.

    The peak is the brightest point near *guess*; the trench level is the mean
    of the darkest point on each side; the edges are where the profile crosses
    half way between them. A band that does not fall back below half on both
    sides inside the profile is not enclosed by its trenches, and is dropped.
    """
    smooth = _smooth(profile)
    lo = max(0, int(guess - half_window))
    hi = min(len(profile), int(guess + half_window) + 1)
    if hi - lo < 1:
        return None
    peak_at = lo + int(np.argmax(smooth[lo:hi]))
    if peak_at == 0 or peak_at == len(profile) - 1:
        return None
    peak = float(smooth[peak_at])
    trench = float((smooth[:peak_at].min() + smooth[peak_at + 1 :].min()) / 2)
    level = (peak + trench) / 2
    top = _crossing(smooth, peak_at, -1, level)
    bottom = _crossing(smooth, peak_at, 1, level)
    if top is None or bottom is None:
        return None
    return top, bottom, peak, trench


def _measure_columns(crop: np.ndarray, guesses: np.ndarray, half_window: float):
    n = crop.shape[1]
    top = np.full(n, np.nan)
    bottom = np.full(n, np.nan)
    contrast = np.full(n, np.nan)
    trench = np.full(n, np.nan)
    for i in range(n):
        found = _column_band(crop[:, i], guesses[i], half_window)
        if found is not None:
            top[i], bottom[i], peak, trench[i] = found
            contrast[i] = peak - trench[i]
    return top, bottom, contrast, trench


def measure_lamella_band(
    image: FibsemImage,
    region: Region = None,
    window: Tuple[float, float] = DEFAULT_WINDOW,
) -> LamellaBandMeasurement:
    """Measure the bright lamella band of a FIB image, column by column.

    The band runs across the image, as AutoLamella mills it. ``region`` limits the
    search: a :class:`FibsemRectangle` in fractions of the image (as reduced areas
    are), a :class:`Point` in the microscope's image coordinates (metres from the
    centre, +y up, as an item's POI is) with ``window`` (width along the band,
    height across it, in metres) around it, or None for the whole image.

    Raises ValueError for an image without a pixel size or a region off the image.
    """
    if image.metadata is None or image.metadata.pixel_size is None:
        raise ValueError("The image carries no pixel size.")
    pixel_size = float(image.metadata.pixel_size.x)
    data = np.asarray(image.data, dtype=float)
    if data.ndim != 2:
        raise ValueError("Only a single-channel 2-D image can be measured.")
    (x0, y0, x1, y1), reference = _region_px(region, data.shape, pixel_size, window)
    crop = data[y0:y1, x0:x1]
    rows, n = crop.shape

    # First pass: every column searched near the brightest row of the mean
    # profile; then again near the fitted centre line, so a tilted band is
    # followed and a bright speck away from it is not taken for the band.
    first = int(np.argmax(_smooth(crop.mean(axis=1))))
    top, bottom, contrast, trench = _measure_columns(
        crop, np.full(n, float(first)), rows / 4
    )
    ok = np.isfinite(top)
    if ok.sum() >= 2:
        x = np.arange(n)[ok]
        line = np.polyfit(x, ((top + bottom) / 2)[ok], 1)
        width = float(np.median((bottom - top)[ok]))
        top, bottom, contrast, trench = _measure_columns(
            crop, np.polyval(line, np.arange(n)), max(2 * width, 4.0)
        )

    # Drop the columns with no real band (a hole, a break): too faint next to
    # the rest, or off the centre line by more than a band width.
    ok = np.isfinite(top)
    if ok.any():
        ok &= contrast >= MIN_CONTRAST_FRACTION * np.nanmedian(contrast)
    if ok.sum() >= 2:
        x = np.arange(n)[ok]
        centre = (top + bottom) / 2
        line = np.polyfit(x, centre[ok], 1)
        width = float(np.median((bottom - top)[ok]))
        off = np.abs(centre - np.polyval(line, np.arange(n)))
        ok &= off <= max(width, 2.0)
    top = np.where(ok, top, np.nan)
    bottom = np.where(ok, bottom, np.nan)

    band = [
        crop[int(math.ceil(t)) : int(math.floor(b)) + 1, i]
        for i, (t, b) in enumerate(zip(top, bottom))
        if ok[i]
    ]
    band_level = float(np.concatenate(band).mean()) if band else math.nan
    trench_level = float(np.nanmean(trench[ok])) if ok.any() else math.nan
    return LamellaBandMeasurement(
        pixel_size=pixel_size,
        image_width=data.shape[1],
        region=(x0, y0, x1, y1),
        reference=reference,
        columns=np.arange(x0, x1),
        top=top + y0,
        bottom=bottom + y0,
        band_level=band_level,
        trench_level=trench_level,
    )


def draw_band_edges(image: FibsemImage, measurement: LamellaBandMeasurement):
    """The searched region of *image* as RGB, with the band's edges drawn on it:
    what the measurement saw, for a person or an agent to check."""
    from fibsem.imaging.drawing import _ensure_rgb

    x0, y0, x1, y1 = measurement.region
    crop = np.asarray(image.data[y0:y1, x0:x1], dtype=float)
    lo, hi = float(crop.min()), float(crop.max())
    scaled = (crop - lo) / (hi - lo) * 255 if hi > lo else np.zeros_like(crop)
    rgb = _ensure_rgb(scaled.astype(np.uint8)).copy()
    # Thick enough to survive the preview's downscale.
    half = max(0, (x1 - x0) // 512)
    for i, ok in enumerate(measurement.measured):
        if not ok:
            rgb[:, i] = (rgb[:, i] * 0.5 + np.array([128, 0, 0])).astype(np.uint8)
            continue
        for edge in (measurement.top[i], measurement.bottom[i]):
            y = int(round(edge)) - y0
            rgb[max(0, y - half) : y + half + 1, i] = (0, 255, 255)
    return rgb
