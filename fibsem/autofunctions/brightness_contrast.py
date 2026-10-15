"""Detector brightness and contrast from the image, for a beam whose own routine
can't be trusted to leave a usable image.

The maths only: what a probe image's histogram says (`measure`), whether it is good
enough (`problems`) or past saving (`unusable`), and the next brightness and contrast
to try (`BrightnessContrastStepper`). The loop that sets the detector and images is
`Imaging.auto_brightness_contrast` (`fibsem.services.imaging`), which drives the beam.

How the two controls are moved together. Contrast is the detector's gain, so it sets
the histogram's width; brightness is its offset, so it sets where the histogram sits.
Each probe image gives two numbers, the histogram's centre and the log of its spread
(5th to 95th percentile: the bulk, so a few bright edges don't set the gain), and
the stepper aims both at a target with one 2x2 Newton step: how far each control
moves each number is a small matrix it starts from a guess of, and corrects from
every pair of images it measures (Broyden's update). So
the cross terms, a contrast change that also moves the centre, are learned rather
than assumed, and a detector whose controls are more or less sensitive than the
guess costs one extra image, not a slow walk.

What counts as clipped is by area: thin clipped features (bright edges, specks)
don't count, since they cost image matching little and fitting them in costs contrast
everywhere else; saturated areas do, and past a total cap every clipped pixel does.
Once some contrast has been seen to clip, the Newton step stays below it.

A histogram that is clipped can't be measured whole: its percentiles sit on the
limit. Its median can, while less than half the image clips, so the centre's row of
the matrix is learned from the median on every pair of images, clipped or not. When
both ends clip, the gain is too high: contrast comes down, with brightness bringing
the median back to the middle. When one end clips, the median decides: on the
clipped side, the image is too bright (or dark) and brightness moves it; in the
middle, only the tail is too long and the gain comes down. When most of the image
is clipped, brightness searches away from that end, doubling its step until it has
seen both sides and then halving the gap (a bisection); a step that hardly changes
how much is clipped means the histogram is far wider than the range, and the gain
comes down instead.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from fibsem.structures import FibsemRectangle

# How far each control moves (centre, log spread), per unit of its range, before the
# stepper has seen the detector respond: brightness moves the centre one for one,
# and the full contrast range spans a gain of about e^4. Corrected after two images.
DEFAULT_RESPONSE = ((1.0, 0.0), (0.0, 4.0))


class BrightnessContrastError(RuntimeError):
    """The routine ended on an image that is mostly black, mostly white, or flat,
    which nothing after it should use. The detector is left at the best brightness
    and contrast it found; ``result`` is the run, to plot and look into."""

    def __init__(
        self, message: str, result: Optional["BrightnessContrastResult"] = None
    ):
        super().__init__(message)
        self.result = result


@dataclass
class BrightnessContrastSettings:
    """What a good image is, and how the routine may get there.

    The fractions are of the image's pixels; centre and spread are on the image's
    range, 0 (black) to 1 (white).
    """

    max_clipped: float = 0.01
    """At most this fraction of pixels in clipped areas at either limit (black or
    white). Clipped features narrower than ``clipped_feature_width`` don't count."""
    clipped_feature_width: int = 5
    """Clipped features narrower than this, in probe pixels (bright edges, specks),
    are let clip: they cost image matching little, and fitting them in costs
    contrast everywhere else. 0 counts every clipped pixel."""
    max_clipped_total: float = 0.05
    """However thin the features, at most this fraction of all pixels at a limit."""
    min_centre: float = 0.25
    max_centre: float = 0.75
    """The histogram's centre must be between these."""
    spread_percentiles: Tuple[float, float] = (5.0, 95.0)
    """The percentiles that bound the spread: the bulk of the image, so a few
    bright edges don't set the contrast."""
    min_spread: float = 0.15
    """The spread must be at least this much."""
    target_centre: float = 0.5
    target_spread: float = 0.5
    """Where an adjustment aims: the middle of what is allowed, not its edge."""
    max_images: int = 6
    """Probe images, the first included. Each one doses the sample."""
    max_step: float = 0.2
    """The largest change to either control between two images, as a fraction of
    its range."""
    unusable_clipped: float = 0.1
    """An image with more than this fraction at a limit is mostly black or white:
    the routine raises rather than leave it."""
    probe_resolution: Tuple[int, int] = (768, 512)
    probe_dwell_time: float = 0.5e-6
    reduced_area: Optional[FibsemRectangle] = field(
        default_factory=lambda: FibsemRectangle(0.25, 0.25, 0.5, 0.5)
    )
    """Scan and judge only this rectangle of the frame (0 to 1): by default the
    centre half, where the work is, which also doses less of the sample. None for
    the full frame."""
    start_from_last: bool = True
    """Start from the brightness and contrast that last worked at this beam current
    and detector, when there is one."""

    def __post_init__(self) -> None:
        if isinstance(self.reduced_area, dict):
            self.reduced_area = FibsemRectangle.from_dict(self.reduced_area)
        self.probe_resolution = tuple(self.probe_resolution)
        self.spread_percentiles = tuple(self.spread_percentiles)

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["reduced_area"] = (
            self.reduced_area.to_dict() if self.reduced_area is not None else None
        )
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BrightnessContrastSettings":
        return cls(**d)


@dataclass
class HistogramStats:
    """A probe image's histogram, on its range (0 black, 1 white)."""

    black: float
    """Fraction of pixels clipped at the bottom limit that count: those in areas
    wider than the settings' ``clipped_feature_width``, or every one when there
    are more than ``max_clipped_total``."""
    white: float
    """The same, at the top limit."""
    low: float
    """The spread's lower percentile."""
    high: float
    """The spread's upper percentile."""
    median: float
    black_total: float = 0.0
    """Fraction of all pixels at the bottom limit."""
    white_total: float = 0.0
    """Fraction of all pixels at the top limit."""

    @property
    def centre(self) -> float:
        return (self.low + self.high) / 2

    @property
    def spread(self) -> float:
        return self.high - self.low

    def to_dict(self) -> Dict[str, float]:
        return {
            **dataclasses.asdict(self),
            "centre": self.centre,
            "spread": self.spread,
        }


def measure(
    data: np.ndarray, settings: Optional[BrightnessContrastSettings] = None
) -> HistogramStats:
    """The histogram of raw pixel data: integer images on their type's range, float
    images on 0 to 1. Unfiltered, so a clipped pixel counts however it got there,
    but in a 2D image a clipped feature narrower than the settings'
    ``clipped_feature_width`` doesn't count (`HistogramStats.black`)."""
    settings = settings or BrightnessContrastSettings()
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        bottom, top = float(info.min), float(info.max)
    else:
        bottom, top = 0.0, 1.0
    low_q, high_q = settings.spread_percentiles
    low, median, high = (
        (float(v) - bottom) / (top - bottom)
        for v in np.percentile(data, (low_q, 50, high_q))
    )
    counted = []
    totals = []
    for mask in (data <= bottom, data >= top):
        total = float(np.mean(mask))
        totals.append(total)
        if total > settings.max_clipped_total:
            counted.append(total)
        else:
            counted.append(_wide_fraction(mask, settings.clipped_feature_width))
    return HistogramStats(
        black=counted[0],
        white=counted[1],
        low=low,
        high=high,
        median=median,
        black_total=totals[0],
        white_total=totals[1],
    )


def _wide_fraction(mask: np.ndarray, width: int) -> float:
    """The fraction of pixels in ``mask`` that lie in features at least ``width``
    across (a morphological opening), or in all of it for anything but a 2D
    image."""
    return float(np.mean(_wide(mask, width)))


def _wide(mask: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or mask.ndim != 2 or not mask.any():
        return mask
    kernel = np.ones((width, width), np.uint8)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0


def clipped_masks(
    data: np.ndarray, settings: Optional[BrightnessContrastSettings] = None
) -> Dict[str, np.ndarray]:
    """Where ``data`` clips: ``black`` and ``white`` (every clipped pixel) and
    ``black_counted`` and ``white_counted`` (those in areas wide enough to count,
    as `measure` counts them). For a plot to show which clipping mattered."""
    settings = settings or BrightnessContrastSettings()
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        bottom, top = info.min, info.max
    else:
        bottom, top = 0.0, 1.0
    masks = {"black": data <= bottom, "white": data >= top}
    for name in ("black", "white"):
        mask = masks[name]
        masks[f"{name}_counted"] = (
            mask
            if np.mean(mask) > settings.max_clipped_total
            else _wide(mask, settings.clipped_feature_width)
        )
    return masks


def problems(stats: HistogramStats, settings: BrightnessContrastSettings) -> List[str]:
    """What is wrong with the image, in words; empty when it is good enough."""
    found = []
    if stats.black > settings.max_clipped:
        found.append(f"{stats.black:.1%} of pixels are black")
    if stats.white > settings.max_clipped:
        found.append(f"{stats.white:.1%} of pixels are white")
    if not settings.min_centre <= stats.centre <= settings.max_centre:
        found.append(f"centred at {stats.centre:.2f}")
    if stats.spread < settings.min_spread:
        found.append(f"spans only {stats.spread:.2f}")
    return found


def unusable(stats: HistogramStats, settings: BrightnessContrastSettings) -> bool:
    """Whether the image is mostly black, mostly white or flat: worse than failing
    to meet the targets, an image nothing should go on to align or place patterns
    on."""
    return (
        max(stats.black_total, stats.white_total) > settings.unusable_clipped
        or not 0.1 <= stats.centre <= 0.9
        or stats.spread < 0.05
    )


def badness(stats: HistogramStats, settings: BrightnessContrastSettings) -> float:
    """A single number to keep the best image by: clipping first, then distance from
    the targets."""
    clipped = max(stats.black - settings.max_clipped, 0) + max(
        stats.white - settings.max_clipped, 0
    )
    return 10 * clipped + abs(stats.centre - settings.target_centre)


class BrightnessContrastStepper:
    """The next brightness and contrast to try, from the image at the last ones.

    Controls are fractions of their range (0 to 1); the caller converts. Holds what
    it has learned between images: the detector's response, the last measurable
    image, and the brightness bracket at the current contrast.
    """

    def __init__(
        self,
        settings: BrightnessContrastSettings,
        response: Optional[Sequence[Sequence[float]]] = None,
    ):
        self.settings = settings
        self.response = np.array(response or DEFAULT_RESPONSE, dtype=float)
        self._last: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._too_dark: Optional[float] = None
        self._too_bright: Optional[float] = None
        self._search = settings.max_step
        self._search_direction = 0
        self._previous: Optional[Tuple[np.ndarray, HistogramStats]] = None
        # The lowest contrast seen clipping too much: the Newton step, which sizes
        # the gain from the bulk, stays below it rather than climb back into the
        # clipping and oscillate.
        self._clipping_contrast = math.inf

    def next(self, controls: Sequence[float], stats: HistogramStats) -> np.ndarray:
        """The (brightness, contrast) to image at next, each 0 to 1."""
        x = np.array(controls, dtype=float)
        s = self.settings
        previous, self._previous = self._previous, (x, stats)
        if previous is not None:
            self._learn_centre(*previous, x, stats)
        if stats.black > s.max_clipped and stats.white > s.max_clipped:
            return self._lower_gain(x, stats, too_wide=True)
        clipped = max(stats.black, stats.white)
        if clipped > 0.5:
            if previous is not None and _brightness_did_little(*previous, x, stats):
                # The histogram is far wider than the range, not merely off to one
                # side of it.
                return self._lower_gain(x, stats)
            return self._search_brightness(x, stats)
        if clipped > 2 * s.max_clipped:
            # One end clips but most of the image doesn't: the median still says
            # where it sits. Sitting on the clipped side, it is too bright (or
            # dark); sitting where it should, only its tail is too long.
            towards = stats.median - s.target_centre
            if stats.white < stats.black:
                towards = -towards
            if towards > 0.1:
                return self._recentre(x, stats)
            return self._lower_gain(x, stats, too_wide=True)
        return self._newton(x, stats)

    # -- the histogram is measurable: one step for both ---------------------------

    def _newton(self, x: np.ndarray, stats: HistogramStats) -> np.ndarray:
        s = self.settings
        y = np.array([stats.centre, math.log(max(stats.spread, 1e-3))])
        if self._last is not None:
            self._learn(x - self._last[0], y - self._last[1])
        self._last = (x, y)
        # A spread that already passes is kept: changing the gain moves the centre
        # too, so only what fails is aimed at, and the step costs fewer images.
        spread = s.target_spread if stats.spread < s.min_spread else stats.spread
        target = np.array([s.target_centre, math.log(spread)])
        try:
            step = np.linalg.solve(self.response, target - y)
        except np.linalg.LinAlgError:
            self.response = np.array(DEFAULT_RESPONSE)
            step = np.linalg.solve(self.response, target - y)
        new = self._apply(x, step)
        if new[1] > x[1] and self._clipping_contrast < math.inf:
            # Halfway to a contrast seen clipping, at most: a bisection.
            new[1] = min(new[1], (x[1] + self._clipping_contrast) / 2)
        return new

    def _learn(self, dx: np.ndarray, dy: np.ndarray) -> None:
        """Broyden's update of the spread's row: the least change to it that
        explains the last step. Not taken when it says contrast works backwards,
        which noise or a step too small to see can make it say."""
        norm = float(dx @ dx)
        if norm < 1e-6:
            return
        row = self.response[1]
        learned = row + (dy[1] - row @ dx) * dx / norm
        if learned[1] > 0.2:
            self.response[1] = learned
        self._too_dark = self._too_bright = None

    def _learn_centre(
        self,
        x0: np.ndarray,
        before: HistogramStats,
        x1: np.ndarray,
        after: HistogramStats,
    ) -> None:
        """How far each control moves the image, from the medians of the last two
        images (Broyden's update, on the centre's row). The median can be measured
        while up to half the image is clipped, where nothing else can, and it is
        what a contrast change that also shifts the image shows up in."""
        dx = x1 - x0
        norm = float(dx @ dx)
        inside = all(max(st.black, st.white) < 0.45 for st in (before, after))
        if norm < 1e-6 or not inside:
            return
        row = self.response[0]
        learned = row + (after.median - before.median - row @ dx) * dx / norm
        if learned[0] > 0.05:
            self.response[0] = learned

    # -- a histogram that is clipped: one control at a time -----------------------

    def _lower_gain(
        self, x: np.ndarray, stats: HistogramStats, too_wide: bool = False
    ) -> np.ndarray:
        """Halve the gain, and bring the median to the middle. ``too_wide`` when the
        image shows the histogram is wider than the range (it clips at both ends,
        or its tail clips with the median in the middle), rather than this being a
        guess: then this contrast is one the Newton step won't go back to."""
        self._last = None
        self._too_dark = self._too_bright = None
        if too_wide:
            self._clipping_contrast = min(self._clipping_contrast, float(x[1]))
        drop = -math.log(2) / self.response[1, 1]
        centre = (self.settings.target_centre - stats.median) / self.response[0, 0]
        return self._apply(x, np.array([centre, drop]))

    def _recentre(self, x: np.ndarray, stats: HistogramStats) -> np.ndarray:
        """Most of the image is on the clipped side: brightness alone, to bring the
        median to the middle."""
        self._last = None
        centre = (self.settings.target_centre - stats.median) / self.response[0, 0]
        new = self._apply(x, np.array([centre, 0.0]))
        if abs(new[0] - x[0]) < 1e-3:
            # Brightness is at its limit: only less gain can bring it back.
            return self._lower_gain(x, stats)
        return new

    def _search_brightness(self, x: np.ndarray, stats: HistogramStats) -> np.ndarray:
        """One end clips: move brightness away from it until it doesn't, doubling
        the step until both sides are seen, then halving the gap."""
        self._last = None
        brightness = x[0]
        if stats.white > stats.black:
            self._too_bright = (
                brightness
                if self._too_bright is None
                else min(self._too_bright, brightness)
            )
            direction = -1
        else:
            self._too_dark = (
                brightness
                if self._too_dark is None
                else max(self._too_dark, brightness)
            )
            direction = 1
        if self._too_dark is not None and self._too_bright is not None:
            new = (self._too_dark + self._too_bright) / 2
        else:
            if direction == self._search_direction:
                self._search *= 2
            self._search_direction = direction
            new = brightness + direction * self._search
        new = min(max(new, 0.0), 1.0)
        if abs(new - brightness) < 1e-3:
            # Brightness is at its limit and the image still clips: only less gain
            # can bring it back.
            return self._apply(x, np.array([0.0, -math.log(2) / self.response[1, 1]]))
        return np.array([new, x[1]])

    def _apply(self, x: np.ndarray, step: np.ndarray) -> np.ndarray:
        limit = self.settings.max_step
        return np.clip(x + np.clip(step, -limit, limit), 0.0, 1.0)


def _brightness_did_little(
    x0: np.ndarray, before: HistogramStats, x1: np.ndarray, after: HistogramStats
) -> bool:
    """Whether a brightness step away from a clipped end left the image about as
    clipped as it was."""
    db, dc = x1 - x0
    if abs(db) < 1e-3 or abs(dc) > 1e-9:
        return False
    if after.white > after.black:
        return before.white > 0.5 and before.white - after.white < 0.1
    return before.black > 0.5 and before.black - after.black < 0.1


@dataclass
class BrightnessContrastIteration:
    """One probe image: the controls it was taken at, as the detector reports them,
    its histogram, and the image itself (None when not kept)."""

    brightness: float
    contrast: float
    stats: HistogramStats
    image: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self, index: int = 0) -> Dict[str, Any]:
        """Without the image: the file `BrightnessContrastResult.save` writes it
        to is named instead."""
        return {
            "index": index,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "stats": self.stats.to_dict(),
            "image": f"iter_{index:02d}.tif" if self.image is not None else None,
        }

    @classmethod
    def from_dict(
        cls, d: Dict[str, Any], result_dir: Optional[Path] = None
    ) -> "BrightnessContrastIteration":
        from fibsem.structures import load_tiff

        stats = {
            k: v
            for k, v in d["stats"].items()
            if k in {f.name for f in dataclasses.fields(HistogramStats)}
        }
        image = None
        if d.get("image") and result_dir is not None:
            image = load_tiff(Path(result_dir) / d["image"])
        return cls(
            brightness=d["brightness"],
            contrast=d["contrast"],
            stats=HistogramStats(**stats),
            image=image,
        )


@dataclass
class BrightnessContrastResult:
    """Where the routine left the detector, and how it got there: every probe image
    with its controls and histogram, to plot (`plot`) and keep (`save`, `load`)."""

    brightness: float
    contrast: float
    stats: HistogramStats
    converged: bool
    """The image it ended on met every target. False when the routine ran out of
    images on one that is short of them."""
    iterations: List[BrightnessContrastIteration]
    started_from_last: bool = False
    """Whether it started from what last worked at this beam current."""
    initial_brightness: Optional[float] = None
    initial_contrast: Optional[float] = None
    """The detector before the run."""
    usable: bool = True
    """False when the image it ended on is mostly black, mostly white or flat: the
    run raised `BrightnessContrastError`, which carries this result."""
    beam_type: Optional[str] = None
    settings: Optional[BrightnessContrastSettings] = None

    @property
    def n_images(self) -> int:
        return len(self.iterations)

    @property
    def image(self) -> Optional[np.ndarray]:
        """The probe image of the controls it ended on."""
        for it in reversed(self.iterations):
            if (it.brightness, it.contrast) == (self.brightness, self.contrast):
                return it.image
        return None

    def to_dict(self) -> Dict[str, Any]:
        """The result without its images: what ``save`` writes to ``data.json``,
        and what the App Server answers."""
        return {
            "beam_type": self.beam_type,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "initial_brightness": self.initial_brightness,
            "initial_contrast": self.initial_contrast,
            "stats": self.stats.to_dict(),
            "converged": self.converged,
            "usable": self.usable,
            "started_from_last": self.started_from_last,
            "n_images": self.n_images,
            "settings": self.settings.to_dict() if self.settings else None,
            "iterations": [it.to_dict(i) for i, it in enumerate(self.iterations)],
        }

    def plot(self, save_path: Optional[str] = None) -> Any:
        """The diagnostic figure: every probe image, and the controls, clipping,
        centre and spread at each. Written to ``save_path`` when given."""
        from fibsem.autofunctions.plotting import plot_brightness_contrast_result

        return plot_brightness_contrast_result(self, save_path=save_path)

    def save(
        self, path: str = ".", name: str = "brightness_contrast", save_plot: bool = True
    ) -> Path:
        """Save to ``<path>/<name>/``: ``data.json`` (`to_dict`), ``iter_00.tif``
        ... (each probe image), and ``plot.png`` when ``save_plot``."""
        import json

        from fibsem.structures import save_tiff

        result_dir = Path(path) / name
        result_dir.mkdir(parents=True, exist_ok=True)
        with open(result_dir / "data.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        for i, it in enumerate(self.iterations):
            if it.image is not None:
                save_tiff(it.image, result_dir / f"iter_{i:02d}.tif")
        if save_plot:
            self.plot(save_path=str(result_dir / "plot.png"))
        return result_dir

    @classmethod
    def load(cls, result_dir: str) -> "BrightnessContrastResult":
        """A result `save` wrote, images included."""
        import json

        result_dir = Path(result_dir)
        with open(result_dir / "data.json") as f:
            data = json.load(f)
        iterations = [
            BrightnessContrastIteration.from_dict(it, result_dir)
            for it in data["iterations"]
        ]
        final = BrightnessContrastIteration.from_dict({**data, "image": None}).stats
        settings = data.get("settings")
        return cls(
            brightness=data["brightness"],
            contrast=data["contrast"],
            stats=final,
            converged=data["converged"],
            iterations=iterations,
            started_from_last=data.get("started_from_last", False),
            initial_brightness=data.get("initial_brightness"),
            initial_contrast=data.get("initial_contrast"),
            usable=data.get("usable", True),
            beam_type=data.get("beam_type"),
            settings=(
                BrightnessContrastSettings.from_dict(settings) if settings else None
            ),
        )
