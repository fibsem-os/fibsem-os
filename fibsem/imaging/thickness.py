"""Lamella thickness from a FIB image: the bright band between the two trenches.

Seen from the ion beam, a lamella is a bright band running across the image with
a dark trench milled on each side. Its width at half its peak brightness (FWHM),
measured column by column, is a top-down estimate of the lamella's thickness: it
cannot see a lamella that is thicker at the bottom than at the top.

This is the one measurement, on the full-resolution image and its own pixel size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import uniform_filter1d

from fibsem.conversions import microscope_image_to_image_coordinates
from fibsem.structures.image import FibsemImage
from fibsem.structures.settings import FibsemRectangle, Point

# Under this many pixels across, the FWHM is too coarse to trust (on an Arctis run,
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
# Each column is averaged with its neighbours over this length along the band
# before it is measured: a single column of a real FIB image is too noisy for
# its own half-maximum (on one final image, single columns ranged
# 43-368 nm around a 233 nm mean). A hole narrower than this is smeared.
DEFAULT_AVERAGE_ALONG = 100e-9
# A band this close in brightness to its trenches gave misleading numbers on a
# real run (ratios of 1.5 on two lamellae).
LOW_CONTRAST_RATIO = 2.0
# Another bright line in the window (trench residue, a stepped wall) at this
# fraction of the band's contrast or more may be the lamella instead.
OTHER_BAND_FRACTION = 0.5

Region = Union[FibsemRectangle, Point, None]


@dataclass
class LamellaThicknessResult:
    """The band, column by column, and what follows from it: the numbers
    (:meth:`to_dict`), the diagnostic plot (:meth:`plot`), and both on disk
    (:meth:`save`, :meth:`load`), as an autofocus result keeps its run.

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
    # The searched region of the image, as it was measured (before averaging):
    # what the plot draws the edges on, and what save writes as window.tif.
    window: Optional[np.ndarray] = None
    # Other bright lines across the window: (centre y at the reference x, in
    # image pixels; contrast as a fraction of the band's).
    other_bands: Tuple[Tuple[float, float], ...] = ()

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
            answer["warnings"] = [
                {"code": "no_band", "message": "No band was found in the window."}
            ]
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

        ratio = self.band_level / self.trench_level if self.trench_level > 0 else None
        answer["contrast"] = {
            "band": round(self.band_level, 2),
            "trench": round(self.trench_level, 2),
            "ratio": round(ratio, 3) if ratio is not None else None,
        }
        answer["other_bands"] = [
            {"centre_offset_nm": _nm((ref_y - y) * nm), "contrast": round(f, 2)}
            for y, f in self.other_bands
        ]

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

        warnings = []
        if ratio is not None and ratio < LOW_CONTRAST_RATIO:
            warnings.append(
                {
                    "code": "low_contrast",
                    "message": f"The band is only {ratio:.2f}x as bright as its "
                    f"trenches (under {LOW_CONTRAST_RATIO}): the edges are "
                    "uncertain. Take a cleaner image (integration, contrast) first.",
                }
            )
        if self.other_bands:
            warnings.append(
                {
                    "code": "several_bands",
                    "message": f"{len(self.other_bands)} other bright line(s) in "
                    "the window (other_bands): the band measured may not be the "
                    "lamella, and centre_offset_nm is to the band measured. Check "
                    "on the image.",
                }
            )
        if band_px < MIN_BAND_PX:
            warnings.append(
                {
                    "code": "under_resolved",
                    "message": f"The band is {band_px:.0f} px across: image it at "
                    "the suggested field width before trusting the number.",
                }
            )
        answer["warnings"] = warnings
        return answer

    def plot(self, save_path: Optional[str] = None):
        """The diagnostic figure: the window with the band's edges on it, and
        the thickness along the lamella under it. Saved to ``save_path`` when
        given; the figure is returned either way."""
        from fibsem.theme.plots import DARK, plot_theme

        with plot_theme(DARK) as theme:
            fig = _plot_thickness(self, theme)
            if save_path is not None:
                fig.savefig(save_path, dpi=120, bbox_inches="tight")
        return fig

    def plot_jpeg(self) -> bytes:
        """The figure as JPEG bytes, for an answer that carries it."""
        import io

        from fibsem.theme.plots import DARK, plot_theme

        buf = io.BytesIO()
        with plot_theme(DARK) as theme:
            _plot_thickness(self, theme).savefig(
                buf, format="jpeg", dpi=100, pil_kwargs={"quality": 85}
            )
        return buf.getvalue()

    def save(self, path: str = ".", name: str = "thickness"):
        """Save the result to ``<path>/<name>/``: ``data.json`` (the numbers and
        each column's edges), ``window.tif`` (the region measured) and
        ``plot.png``. Returns the folder."""
        import json
        from pathlib import Path

        from fibsem.structures.image import save_tiff

        folder = Path(path) / name
        folder.mkdir(parents=True, exist_ok=True)
        data = {
            **self.to_dict(),
            "data": {
                "pixel_size": self.pixel_size,
                "image_width": self.image_width,
                "region": list(self.region),
                "reference": list(self.reference),
                "top": [None if math.isnan(v) else float(v) for v in self.top],
                "bottom": [None if math.isnan(v) else float(v) for v in self.bottom],
                "band_level": self.band_level,
                "trench_level": self.trench_level,
                "other_bands": [list(band) for band in self.other_bands],
            },
        }
        with open(folder / "data.json", "w") as f:
            json.dump(data, f, indent=2)
        if self.window is not None:
            save_tiff(self.window, folder / "window.tif")
        self.plot(save_path=str(folder / "plot.png"))
        return folder

    @classmethod
    def load(cls, folder: str) -> "LamellaThicknessResult":
        """A result :meth:`save` wrote."""
        import json
        from pathlib import Path

        from fibsem.structures.image import load_tiff

        folder = Path(folder)
        with open(folder / "data.json") as f:
            data = json.load(f)["data"]
        x0, _, x1, _ = data["region"]
        window = folder / "window.tif"
        return cls(
            pixel_size=data["pixel_size"],
            image_width=data["image_width"],
            region=tuple(data["region"]),
            reference=tuple(data["reference"]),
            columns=np.arange(x0, x1),
            top=np.array([np.nan if v is None else v for v in data["top"]]),
            bottom=np.array([np.nan if v is None else v for v in data["bottom"]]),
            band_level=data["band_level"],
            trench_level=data["trench_level"],
            window=load_tiff(window) if window.exists() else None,
            other_bands=tuple(tuple(band) for band in data.get("other_bands", [])),
        )


def _plot_thickness(result: LamellaThicknessResult, theme):
    from matplotlib.figure import Figure

    x0, y0, x1, y1 = result.region
    um = result.pixel_size * 1e6
    along = (result.columns - x0) * um  # um along the window
    measured = result.measured
    answer = result.to_dict()

    fig = Figure(figsize=(7.5, 5.0), layout="constrained")
    top_ax, profile_ax = fig.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [1, 1.2]}
    )
    if result.window is not None:
        top_ax.imshow(
            result.window,
            cmap="gray",
            aspect="auto",
            extent=(0, (x1 - x0) * um, (y1 - y0) * um, 0),
        )
    top_edge = np.where(measured, result.top - y0, np.nan) * um
    bottom_edge = np.where(measured, result.bottom - y0, np.nan) * um
    top_ax.plot(along, top_edge, color=theme.series[0], lw=1)
    top_ax.plot(along, bottom_edge, color=theme.series[1], lw=1)
    if result.other_bands:
        slope, _ = result.centre_line()
        ref_x = result.reference[0]
        for y, _ in result.other_bands:
            line = (y + slope * (result.columns - ref_x) - y0) * um
            top_ax.plot(along, line, color=theme.bad, lw=1, ls="--")
    for first, last in result.gaps():
        for ax in (top_ax, profile_ax):
            ax.axvspan(
                (first - x0) * um, (last + 1 - x0) * um, color=theme.bad, alpha=0.3
            )
    top_ax.set_ylabel("across (um)")
    top_ax.set_title("Band edges at half maximum", fontsize=9)

    thick = np.where(
        measured, (result.bottom - result.top) * result.pixel_size * 1e9, np.nan
    )
    profile_ax.plot(along, thick, color=theme.series[2], lw=1, label="per column")
    stats = answer.get("thickness_nm")
    if stats:
        profile_ax.axhline(
            stats["mean"],
            color=theme.text_muted,
            ls="--",
            lw=1,
            label=f"mean {stats['mean']:.0f} nm",
        )
        edges = np.linspace(0, (x1 - x0) * um, 4)
        for (lo, hi), value in zip(zip(edges[:-1], edges[1:]), answer["thirds_nm"]):
            if value is not None:
                profile_ax.hlines(value, lo, hi, color=theme.series[3], lw=2)
        profile_ax.plot([], [], color=theme.series[3], lw=2, label="thirds")
        title = (
            f"{stats['mean']:.0f} ± {answer['uncertainty_nm']:.0f} nm "
            f"({stats['min']:.0f}-{stats['max']:.0f}), "
            f"wedge {answer.get('wedge', {}).get('across_band_nm', 0):.0f} nm, "
            f"angle {answer['angle_deg']:.2f}°, "
            f"dropped {answer['continuity']['dropped']}/{answer['continuity']['columns']}"
        )
        if answer["warnings"]:
            title += "\n" + ", ".join(
                w["code"].replace("_", " ") for w in answer["warnings"]
            )
        profile_ax.set_title(title, fontsize=9)
        # At least two pixels either side of the mean, so pixel-level noise on
        # an even band does not fill the axis.
        two_px = 2 * answer["uncertainty_nm"]
        profile_ax.set_ylim(
            min(stats["min"], stats["mean"] - two_px) - 5,
            max(stats["max"], stats["mean"] + two_px) + 5,
        )
        profile_ax.legend(fontsize=7, loc="best")
    profile_ax.set_xlabel("along the lamella (um)")
    profile_ax.set_ylabel("thickness (nm)")
    return fig


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
    average_along: float = DEFAULT_AVERAGE_ALONG,
) -> LamellaThicknessResult:
    """Measure the bright lamella band of a FIB image, column by column.

    The band runs across the image, as AutoLamella mills it. ``region`` limits the
    search: a :class:`FibsemRectangle` in fractions of the image (as reduced areas
    are), a :class:`Point` in the microscope's image coordinates (metres from the
    centre, +y up, as an item's POI is) with ``window`` (width along the band,
    height across it, in metres) around it, or None for the whole image.
    ``average_along`` (metres) is how far along the band each column is averaged
    with its neighbours before it is measured.

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
    along = int(round(average_along / pixel_size))
    if along > 1:
        crop = uniform_filter1d(crop, along, axis=1, mode="nearest")
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
    other_bands = ()
    if ok.sum() >= 2:
        x = np.arange(n)[ok]
        slope, intercept = np.polyfit(x, ((top + bottom) / 2)[ok], 1)
        width = float(np.median((bottom - top)[ok]))
        found = _other_bands(crop, slope, intercept, width)
        ref_x = reference[0] - x0
        other_bands = tuple((y + slope * ref_x + y0, fraction) for y, fraction in found)
    return LamellaThicknessResult(
        pixel_size=pixel_size,
        image_width=data.shape[1],
        region=(x0, y0, x1, y1),
        reference=reference,
        columns=np.arange(x0, x1),
        top=top + y0,
        bottom=bottom + y0,
        band_level=band_level,
        trench_level=trench_level,
        window=np.asarray(image.data)[y0:y1, x0:x1].copy(),
        other_bands=other_bands,
    )


def _other_bands(crop: np.ndarray, slope: float, intercept: float, width: float):
    """Bright lines across *crop* other than the band on ``y = slope*x +
    intercept``: (centre y at x = 0, contrast as a fraction of the band's).

    Each column is shifted onto the band's line and the columns averaged, so a
    line parallel to the band stands out of the noise; a peak there counts when
    it is enclosed by darker rows on both sides and at least
    OTHER_BAND_FRACTION of the band's own rise above the trenches.
    """
    from scipy.ndimage import shift as shift_rows
    from scipy.signal import find_peaks

    rows, n = crop.shape
    aligned = np.stack(
        [shift_rows(crop[:, i], -slope * i, order=1, mode="nearest") for i in range(n)],
        axis=1,
    )
    profile = _smooth(_smooth(aligned.mean(axis=1)))
    centre = int(round(intercept))
    if not 0 <= centre < rows:
        return []
    near = slice(max(0, centre - int(width)), min(rows, centre + int(width) + 1))
    peak = float(profile[near].max())
    floor = float(profile.min())
    rise = peak - floor
    if rise <= 0:
        return []
    found, props = find_peaks(profile, prominence=OTHER_BAND_FRACTION * rise / 2)
    bands = []
    for y in found:
        if abs(y - intercept) <= max(1.5 * width, 3.0):
            continue  # the band itself
        fraction = (float(profile[y]) - floor) / rise
        if fraction >= OTHER_BAND_FRACTION:
            bands.append((float(y), round(fraction, 3)))
    return bands
