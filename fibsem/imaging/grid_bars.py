"""Grid bars from an overview: where the squares are, and how far a site is
from the bars around it.

An EM grid is two families of straight, evenly spaced bars. Each family is
found as a peak of the image's power spectrum (its direction and pitch), then
refined by folding the image along that direction: every pixel's distance
along the bars' normal, taken modulo the pitch, and the mean brightness at each
phase. The bars are the band of that profile darker (or brighter) than the
rest. A stitched overview is fitted per tile for the band's phase (tiles
split at the gaps between them, or at their seams), as tiles placed by the stage step are a few microns off one
another (the 2026-10-15 session fitted each tile the same way).

The fit only needs the bars to stand out on average along their length, so it
holds where cells and film are faint (a coated grid): it is the sites, not the
bars, that coating hides. ``snr`` says how well the band stood out.

Two checks the image can confirm keep a wrong fit from passing as good. The
families must be a square mesh seen foreshortened up and down the image (a
tilted view; for the FIB, its grazing angle): the squash that makes them
square is the view's, and clearances up and down are given in the sample's
plane. And where the film has broken, the holes must sit in the openings.
Experimental.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from fibsem.structures import FibsemImage

# Grids from 400 mesh (63 um pitch) to 100 mesh (254 um).
DEFAULT_PITCH_RANGE = (50e-6, 280e-6)
# A site this close to a bar is in the worse zone (Patrick 2026-10-20): only
# when the square has nothing better, and never on the bar itself.
DEFAULT_MARGIN = 10e-6
# Below this the band barely stood out of the noise: the fit is a guess.
GOOD_SNR, WEAK_SNR = 6.0, 4.0
# Bars are the narrower band: a 100 to 400 mesh bar is 20-40% of its pitch.
MAX_BAR_FRACTION = 0.45
# Un-squashed, a square mesh's families are this close to square and equal.
MESH_ANGLE_TOL, MESH_PITCH_TOL = 5.0, 0.06
# Holes on the bars at more than this share of chance: the bars are misplaced.
HOLES_ON_BARS_LIMIT = 0.6
# The view angle the mesh gives against the one the image's stage tilt gives.
VIEW_ANGLE_TOL = 6.0
MIN_HOLES = 3
# Pixels this dark are a stitched overview's gaps (or a masked tile).
GAP_FRACTION = 0.03
_SAMPLES = 30000
_BINS = 96
_FFT_SIZE = 1024
_PAD = 3


@dataclass
class BarFamily:
    """One family of parallel bars, in the image's pixels: ``angle`` is the
    direction of their normal (radians, image x right, y down), ``offsets`` the
    phase where a bar starts, per region of the image (`GridBarFit.regions`)."""

    angle: float
    pitch: float  # px
    width: float  # px
    offsets: List[float]
    dark: bool
    snr: float

    @property
    def normal(self) -> np.ndarray:
        return np.array([math.cos(self.angle), math.sin(self.angle)])

    def phase(self, x: float, y: float, region: int) -> float:
        """Where (x, y) sits in the pitch: 0 to ``width`` is on a bar."""
        t = x * math.cos(self.angle) + y * math.sin(self.angle)
        return (t - self.offsets[region]) % self.pitch


@dataclass
class GridBarFit:
    pixel_size: float  # metres per pixel
    shape: Tuple[int, int]
    families: List[BarFamily]
    regions: List[Tuple[int, int, int, int]]  # y0, y1, x0, x1 (end exclusive)
    data: Optional[np.ndarray] = field(default=None, repr=False)
    squash: float = 1.0  # the view's foreshortening up and down the image
    mesh_error: Tuple[float, float] = (0.0, 1.0)  # degrees off square, pitch ratio
    holes: int = 0
    holes_on_bars: Optional[float] = None  # share of hole area on bars, by chance 1
    expected_squash: Optional[float] = None  # from the image's stage tilt and beam

    @property
    def view_angle(self) -> float:
        """Degrees the view is off the grid's normal, from the mesh."""
        return math.degrees(math.acos(min(1.0, self.squash)))

    @property
    def problems(self) -> List[str]:
        out = []
        off_square, ratio = self.mesh_error
        if off_square > MESH_ANGLE_TOL or abs(ratio - 1) > MESH_PITCH_TOL:
            out.append("not a square mesh")
        if self.holes_on_bars is not None and self.holes_on_bars > HOLES_ON_BARS_LIMIT:
            out.append("holes on the bars")
        if self.expected_squash is not None:
            expected = math.degrees(math.acos(min(1.0, self.expected_squash)))
            if abs(self.view_angle - expected) > VIEW_ANGLE_TOL:
                out.append("view angle off the stage tilt")
        return out

    @property
    def confidence(self) -> str:
        snr = min(f.snr for f in self.families)
        if self.problems or snr < WEAK_SNR:
            return "unreliable"
        return "good" if snr >= GOOD_SNR else "weak"

    def sample_scale(self, family: "BarFamily") -> float:
        """Image px across *family*'s bars per sample-plane px."""
        return math.hypot(family.normal[0], self.squash * family.normal[1])

    def region_of(self, x: float, y: float) -> Optional[int]:
        for i, (y0, y1, x0, x1) in enumerate(self.regions):
            if y0 <= y < y1 and x0 <= x < x1:
                return i
        return None

    def on_bar(self, x: float, y: float, region: int) -> bool:
        return any(f.phase(x, y, region) <= f.width for f in self.families)

    def clearance(
        self,
        x: float,
        y: float,
        region: int,
        direction: Optional[Sequence[float]] = None,
    ) -> float:
        """Sample-plane pixels from (x, y) to the nearest bar edge: all round,
        or along *direction* (image x or y). 0 on a bar."""
        if self.on_bar(x, y, region):
            return 0.0
        best = math.inf
        for f in self.families:
            u = f.phase(x, y, region)
            if direction is None:
                scale = self.sample_scale(f)
                best = min(best, (u - f.width) / scale, (f.pitch - u) / scale)
                continue
            rate = float(f.normal @ np.asarray(direction, dtype=float))
            if abs(rate) < 1e-9:
                continue  # running along this family's bars
            best = min(best, ((f.pitch - u) if rate > 0 else (u - f.width)) / abs(rate))
        return best / (self.squash if direction is not None and direction[1] else 1.0)

    def square_centre(self, x: float, y: float, region: int) -> np.ndarray:
        """The centre of the opening (x, y) is in, in pixels."""
        a, b = self.families
        rhs = []
        for f in (a, b):
            t = x * math.cos(f.angle) + y * math.sin(f.angle)
            k = math.floor((t - f.offsets[region] - f.width) / f.pitch)
            rhs.append(
                f.offsets[region] + f.width + (f.pitch - f.width) / 2 + k * f.pitch
            )
        return np.linalg.solve(np.stack([a.normal, b.normal]), np.array(rhs))

    def to_dict(self) -> Dict[str, Any]:
        um = self.pixel_size * 1e6
        off_square, ratio = self.mesh_error
        return {
            "confidence": self.confidence,
            "problems": self.problems,
            "mesh_pitch_um": round(
                float(np.mean([f.pitch / self.sample_scale(f) for f in self.families]))
                * um,
                1,
            ),
            "view_angle_deg": round(self.view_angle, 1),
            "stage_view_angle_deg": None
            if self.expected_squash is None
            else round(math.degrees(math.acos(min(1.0, self.expected_squash))), 1),
            "mesh_off_square_deg": round(off_square, 1),
            "mesh_pitch_ratio": round(ratio, 3),
            "holes": self.holes,
            "holes_on_bars": None
            if self.holes_on_bars is None
            else round(self.holes_on_bars, 2),
            "families": [
                {
                    "angle_deg": round(math.degrees(f.angle) % 180, 2),
                    "pitch_um": round(f.pitch * um, 1),
                    "bar_width_um": round(f.width * um, 1),
                    "bars": "dark" if f.dark else "bright",
                    "snr": round(f.snr, 1),
                }
                for f in self.families
            ],
            "tiles": len(self.regions),
        }


def fit_grid_bars(
    image: FibsemImage, pitch_range: Tuple[float, float] = DEFAULT_PITCH_RANGE
) -> GridBarFit:
    """The two families of grid bars in *image* (an overview). Raises
    ValueError for an image without a pixel size, or with no lattice in it."""
    if image.metadata is None or image.metadata.pixel_size is None:
        raise ValueError("The image carries no pixel size.")
    pixel_size = float(image.metadata.pixel_size.x)
    data = np.asarray(image.data, dtype=float)
    if data.ndim != 2:
        raise ValueError("Only a single-channel 2-D image can be fitted.")
    valid = data > GAP_FRACTION * data.max()
    if valid.mean() < 0.2:
        raise ValueError("The image is mostly empty.")
    # Bright blobs (ice, contamination) would outweigh the bars' faint contrast.
    lo, hi = np.percentile(data[valid], (2, 90))
    clipped = np.clip(data, lo, hi)
    regions = _regions(valid, data)

    rng = np.random.default_rng(0)
    ys, xs = np.nonzero(valid)
    pick = rng.choice(len(xs), min(_SAMPLES, len(xs)), replace=False)
    xs, ys = xs[pick].astype(float), ys[pick].astype(float)
    values = clipped[ys.astype(int), xs.astype(int)]
    region_of = np.full(len(xs), -1)
    for i, (y0, y1, x0, x1) in enumerate(regions):
        region_of[(ys >= y0) & (ys < y1) & (xs >= x0) & (xs < x1)] = i

    pitch_px = (pitch_range[0] / pixel_size, pitch_range[1] / pixel_size)
    lattice = [
        _refine(xs, ys, values, angle, pitch)
        for angle, pitch in _spectrum_peaks(clipped, valid, pitch_px)
    ]
    # Both families are the same bars: one polarity, the better fit of the two.
    bands = {
        dark: [_band(xs, ys, values, a, p, dark) for a, p in lattice]
        for dark in (True, False)
    }
    dark = min(bands, key=lambda d: sum(b[4] for b in bands[d]))
    families = []
    for (angle, pitch), (offset, width, _, snr, _) in zip(lattice, bands[dark]):
        offsets = []
        for i in range(len(regions)):
            mine = region_of == i
            offsets.append(
                _offset(xs[mine], ys[mine], values[mine], angle, pitch, width, dark)
                if mine.sum() > 2000
                else offset
            )
        families.append(BarFamily(angle, pitch, width, offsets, dark, snr))
    squash, mesh_error = _mesh(families)
    fit = GridBarFit(
        pixel_size, data.shape, families, regions, data, squash, mesh_error
    )
    fit.holes, fit.holes_on_bars = _holes(fit, data, valid)
    fit.expected_squash = _stage_squash(image)
    return fit


def _stage_squash(image) -> Optional[float]:
    """The foreshortening the image's beam and stage tilt give, if it says."""
    from fibsem.geometry.projection import (
        BeamStageProjection,
        surface_foreshortening,
    )

    projection = BeamStageProjection.from_image(image)
    state = image.metadata.microscope_state
    if projection is None or state is None or state.stage_position is None:
        return None
    try:
        return float(surface_foreshortening(projection, state.stage_position))
    except Exception:
        return None


def _fold(xs, ys, values, angle, pitch, bins=_BINS):
    t = xs * math.cos(angle) + ys * math.sin(angle)
    index = np.minimum((np.mod(t, pitch) / pitch * bins).astype(int), bins - 1)
    total = np.bincount(index, values, bins)
    return total / np.maximum(np.bincount(index, None, bins), 1)


def _spectrum_peaks(data, valid, pitch_px):
    """The two strongest periodic directions with a pitch in range, at least
    30 degrees apart: (normal angle, pitch px) each."""
    step = max(1, math.ceil(max(data.shape) / _FFT_SIZE))
    small = data[::step, ::step]
    mask = valid[::step, ::step]
    img = np.where(mask, small, small[mask].mean()) - small[mask].mean()
    h, w = img.shape
    # Zero-padded: a grid spans only a few pitches, so the unpadded spectrum
    # places a peak's direction to several degrees only.
    ph, pw = _PAD * h, _PAD * w
    windowed = img * np.outer(np.hanning(h), np.hanning(w))
    power = np.abs(np.fft.fftshift(np.fft.fft2(windowed, s=(ph, pw)))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(ph))[:, None] / step
    fx = np.fft.fftshift(np.fft.fftfreq(pw))[None, :] / step
    radius = np.hypot(fx, fy)
    power = np.where((radius > 1 / pitch_px[1]) & (radius < 1 / pitch_px[0]), power, 0)
    direction = np.arctan2(fy, fx)
    peaks = []
    for _ in range(2):
        if not power.any():
            raise ValueError("No grid bars found: nothing periodic at a grid's pitch.")
        iy, ix = np.unravel_index(np.argmax(power), power.shape)
        angle = math.atan2(fy[iy, 0], fx[0, ix])
        peaks.append((angle, 1 / math.hypot(fx[0, ix], fy[iy, 0])))
        apart = np.abs(((direction - angle + np.pi / 2) % np.pi) - np.pi / 2)
        power = np.where(apart < math.radians(30), 0, power)
    return peaks


def _refine(xs, ys, values, angle, pitch):
    """The angle and pitch at which the folded profile is sharpest: coarse to
    fine around the spectrum's peak."""
    best = (-1.0, angle, pitch)
    for span, n_angles, stretch, n_pitches in (
        (4, 41, 0.08, 41),
        (0.3, 31, 0.01, 21),
        (0.05, 11, 0.002, 11),
    ):
        _, a0, p0 = best
        for a in a0 + np.radians(np.linspace(-span, span, n_angles)):
            for p in p0 * np.linspace(1 - stretch, 1 + stretch, n_pitches):
                score = _fold(xs, ys, values, a, p, 48).var()
                if score > best[0]:
                    best = (score, a, p)
    return best[1], best[2]


def _band(xs, ys, values, angle, pitch, dark):
    """The bars in the folded profile: the two-level fit (bars, openings) with
    the least residual, bars darker (or brighter) and the narrower band.
    Answers (offset px, width px, dark, snr, residual as a share of the
    profile's variance)."""
    profile = _fold(xs, ys, values, angle, pitch)
    kernel = np.array([1, 4, 6, 4, 1.0]) / 16
    smooth = np.convolve(np.r_[profile[-2:], profile, profile[:2]], kernel, "same")[
        2:-2
    ]
    n = len(smooth)
    best = None
    for sign in (1,) if dark else (-1,):
        s = sign * smooth
        for w in range(int(0.1 * n), int(MAX_BAR_FRACTION * n) + 1):
            means = np.convolve(np.r_[s, s[: w - 1]], np.ones(w) / w, "valid")[:n]
            start = int(np.argmin(means))
            inside = np.zeros(n, bool)
            inside[(np.arange(w) + start) % n] = True
            residual = np.mean(
                (s - np.where(inside, means[start], s[~inside].mean())) ** 2
            )
            if best is None or residual < best[0]:
                best = (residual, start, w, sign, s[~inside].mean() - means[start])
    _, start, w, sign, contrast = best
    noise = math.sqrt(float(np.mean((profile - smooth) ** 2)))
    return (
        start / n * pitch,
        w / n * pitch,
        sign == 1,
        float(contrast / max(noise, 1e-9)),
        float(best[0] / max(smooth.var(), 1e-12)),
    )


def _offset(xs, ys, values, angle, pitch, width, dark):
    """Where the bars start in one region, their width and pitch known."""
    s = _fold(xs, ys, values, angle, pitch) * (1 if dark else -1)
    w = max(1, int(round(width / pitch * _BINS)))
    means = np.convolve(np.r_[s, s[: w - 1]], np.ones(w) / w, "valid")[:_BINS]
    return int(np.argmin(means)) / _BINS * pitch


def _mesh(families) -> Tuple[float, Tuple[float, float]]:
    """The squash up and down the image that makes the two families most
    nearly a square mesh, and how far off square they are then: (squash,
    (degrees off a right angle, pitch ratio))."""
    best = None
    for squash in np.linspace(0.25, 1.0, 151):
        normals, pitches = [], []
        for f in families:
            n = np.array([f.normal[0], squash * f.normal[1]])
            normals.append(n / np.linalg.norm(n))
            pitches.append(f.pitch / np.linalg.norm(n))
        cross = abs(float(normals[0] @ normals[1]))
        off_square = 90 - math.degrees(math.acos(min(1.0, cross)))
        ratio = max(pitches) / min(pitches)
        error = off_square / 90 + math.log(ratio)
        if best is None or error < best[0]:
            best = (error, float(squash), (off_square, ratio))
    return best[1], best[2]


def _holes(fit, data, valid) -> Tuple[int, Optional[float]]:
    """Holes in the film (squares broken open: the darkest compact patches,
    an opening or less across) and the share of their area on the fitted bars,
    against chance: (count, share / bar cover). None with fewer than
    MIN_HOLES holes, or where the bars are the darker band: their crossings
    and shadows are dark compact patches too."""
    from scipy import ndimage as ndi

    if fit.families[0].dark:
        return 0, None
    sigma = max(1.0, 2e-6 / fit.pixel_size)
    fill = float(np.median(data[valid]))
    smooth = ndi.gaussian_filter(np.where(valid, data, fill), sigma)
    h, w = fit.shape
    step = max(1, int(math.ceil(max(h, w) / 1024)))
    yy, xx = np.mgrid[0:h:step, 0:w:step]
    on = np.zeros(yy.shape, bool)
    for i, (y0, y1, x0, x1) in enumerate(fit.regions):
        inside = (yy >= y0) & (yy < y1) & (xx >= x0) & (xx < x1)
        for f in fit.families:
            t = xx * math.cos(f.angle) + yy * math.sin(f.angle)
            on |= inside & (np.mod(t - f.offsets[i], f.pitch) <= f.width)
    seen = valid[0:h:step, 0:w:step]
    threshold = fill - 0.6 * (fill - np.percentile(smooth[valid], 1))
    labels, _ = ndi.label(valid & (smooth < threshold))
    pitch = max(f.pitch / fit.sample_scale(f) for f in fit.families)
    sub = labels[0:h:step, 0:w:step]
    sizes = np.bincount(labels.ravel())
    count, area, on_area = 0, 0, 0
    for i, sl in enumerate(ndi.find_objects(labels), 1):
        extent = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
        if sizes[i] < (0.15 * pitch) ** 2 or extent > 1.3 * pitch:
            continue
        mine = sub == i
        count += 1
        area += int(mine.sum())
        on_area += int((mine & on).sum())
    if count < MIN_HOLES or area == 0:
        return count, None
    return count, float(on_area / area / max(on[seen].mean(), 1e-9))


def _seams(data, axis) -> List[int]:
    """Where tiles stitched edge to edge meet: a line the image jumps across."""
    jump = np.abs(np.diff(data, axis=axis)).mean(axis=1 - axis)
    z = (jump - np.median(jump)) / max(float(jump.std()), 1e-9)
    return [int(i) + 1 for i in np.nonzero(z > 15)[0]]


def _regions(valid, data=None) -> List[Tuple[int, int, int, int]]:
    """The tiles of a stitched overview, split at its empty rows and columns,
    or else at its seams; the whole image when it has neither."""

    def spans(occupied):
        out, start = [], None
        for i, on in enumerate(list(occupied) + [False]):
            if on and start is None:
                start = i
            elif not on and start is not None:
                if i - start > 20:
                    out.append((start, i))
                start = None
        return out

    rows = spans(valid.mean(axis=1) > 0.3)
    cols = spans(valid.mean(axis=0) > 0.3)
    if len(rows) * len(cols) == 1 and data is not None:
        h, w = valid.shape
        ys = [0] + _seams(data, 0) + [h]
        xs = [0] + _seams(data, 1) + [w]
        rows = [(a, b) for a, b in zip(ys, ys[1:]) if b - a > 20]
        cols = [(a, b) for a, b in zip(xs, xs[1:]) if b - a > 20]
    return [(y0, y1, x0, x1) for y0, y1 in rows for x0, x1 in cols]


def measure_sites(
    fit: GridBarFit,
    points: Sequence[Sequence[float]],
    margin: float = DEFAULT_MARGIN,
    radius: float = 0.0,
) -> List[Dict[str, Any]]:
    """Each candidate site (x, y source pixels) against the bars: its square,
    how far it is from the bars (all round, and up and down the image: the FIB
    trench direction, as the FIB image shares the SEM's vertical axis), from
    its square's centre and from the nearest other site, how many sites share
    its square, and its tier: ``middle`` (clear of every bar by more than
    *margin*), ``near_bar`` (within it: only if nothing better) or ``on_bar``
    (never). *radius* (metres) is taken off every clearance: a cell's or a
    lamella's half-size."""
    um = fit.pixel_size * 1e6
    r = radius / fit.pixel_size
    sites = []
    for x, y in points:
        x, y = float(x), float(y)
        region = fit.region_of(x, y)
        if region is None:
            sites.append(
                {
                    "px": [x, y],
                    "tier": None,
                    "reason": "in a gap between tiles or off the image",
                }
            )
            continue
        if fit.on_bar(x, y, region):
            sites.append({"px": [x, y], "tier": "on_bar", "square": None})
            continue
        centre = fit.square_centre(x, y, region)
        around = fit.clearance(x, y, region) - r
        tier = (
            "on_bar"
            if around < 0
            else "middle"
            if around * fit.pixel_size > margin
            else "near_bar"
        )
        sites.append(
            {
                "px": [x, y],
                "tier": tier,
                "square": [round(float(c), 1) for c in centre],
                "clearance_um": round(float(around * um), 1),
                "up_um": round(
                    float((fit.clearance(x, y, region, (0, -1)) - r) * um), 1
                ),
                "down_um": round(
                    float((fit.clearance(x, y, region, (0, 1)) - r) * um), 1
                ),
                "from_centre_um": round(float(np.hypot(*(centre - (x, y)))) * um, 1),
            }
        )
    _neighbours(sites, fit)
    return sites


def _neighbours(sites, fit):
    """Each site's nearest other site and how many sites share its square. A
    square seen from two tiles is one: centres closer than half an opening."""
    um = fit.pixel_size * 1e6
    opening = min(f.pitch - f.width for f in fit.families)
    squares: List[np.ndarray] = []
    for site in sites:
        if site.get("square") is None:
            continue
        centre = np.array(site["square"])
        match = next(
            (i for i, c in enumerate(squares) if np.hypot(*(c - centre)) < opening / 2),
            None,
        )
        if match is None:
            squares.append(centre)
            match = len(squares) - 1
        site["square_index"] = match
    for site in sites:
        others = [
            np.hypot(site["px"][0] - o["px"][0], site["px"][1] - o["px"][1])
            for o in sites
            if o is not site
        ]
        site["nearest_site_um"] = round(float(min(others) * um), 1) if others else None
        if "square_index" in site:
            site["sites_in_square"] = sum(
                o.get("square_index") == site["square_index"] for o in sites
            )


TIER_COLOURS = {
    "middle": (60, 200, 90),
    "near_bar": (245, 160, 40),
    "on_bar": (230, 60, 50),
    None: (160, 160, 160),
}


def sites_picture(
    fit: GridBarFit,
    sites: List[Dict[str, Any]],
    margin: float = DEFAULT_MARGIN,
    max_width: int = 1536,
) -> Tuple[bytes, bytes]:
    """The overview with the fitted bars shaded red, the margin beside them
    orange, and each site numbered in its tier's colour: green clear, orange
    near a bar, red on one. A fit that isn't good says so in the corner.
    Answers (PNG, JPEG preview)."""
    from PIL import Image, ImageDraw

    data = fit.data
    lo, hi = np.percentile(data, (1, 99))
    grey = np.clip((data - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
    rgb = np.stack([grey] * 3, axis=-1).astype(float)
    h, w = fit.shape
    yy, xx = np.mgrid[0:h, 0:w]
    bar = np.zeros((h, w), bool)
    near = np.zeros((h, w), bool)
    m = margin / fit.pixel_size
    for i, (y0, y1, x0, x1) in enumerate(fit.regions):
        sl = (slice(y0, y1), slice(x0, x1))
        for f in fit.families:
            t = xx[sl] * math.cos(f.angle) + yy[sl] * math.sin(f.angle)
            u = np.mod(t - f.offsets[i], f.pitch)
            bar[sl] |= u <= f.width
            near[sl] |= (u <= f.width + m) | (u >= f.pitch - m)
    near &= ~bar
    for mask, colour in (
        (bar, TIER_COLOURS["on_bar"]),
        (near, TIER_COLOURS["near_bar"]),
    ):
        rgb[mask] = 0.65 * rgb[mask] + 0.35 * np.array(colour)
    picture = Image.fromarray(rgb.astype(np.uint8))
    scale = min(1.0, max_width / w)
    if scale < 1:
        picture = picture.resize((int(w * scale), int(h * scale)))
    draw = ImageDraw.Draw(picture)
    radius = max(6, int(0.006 * picture.width))
    for n, site in enumerate(sites, 1):
        x, y = (c * scale for c in site["px"])
        colour = TIER_COLOURS[site.get("tier")]
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius), outline=colour, width=3
        )
        draw.text((x + radius + 2, y - radius - 2), str(n), fill=colour)
    if fit.confidence != "good":
        label = "; ".join([f"bar fit {fit.confidence}"] + fit.problems)
        draw.text((8, 8), label, fill=(255, 255, 255))
    png, jpeg = io.BytesIO(), io.BytesIO()
    picture.save(png, format="PNG")
    picture.save(jpeg, format="JPEG", quality=85)
    return png.getvalue(), jpeg.getvalue()
