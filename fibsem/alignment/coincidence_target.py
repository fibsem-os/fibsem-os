"""Coincidence at a target: an experimental measure, for the agent to try.

On the Arctis the whole-frame measure (`fibsem.alignment.coincidence`) found
almost nothing the two beams share before milling: at 2 kV the SEM sees charge
contrast on a smooth square, the FIB sees topography (FIB-987). By hand, an
agent did better by following one feature: the target it had centred in the
SEM, a raised cell that both beams show (2026-10-15).

This measures that way. It cuts a patch of the SEM image around the target,
squashes it into the FIB view by the foreshortening the pair's geometry gives
(searched only a little either side of it), and finds it anywhere in the FIB
image, so the whole frame is the capture range. Both images go through the
same band-pass filter the whole-frame measure uses, at both of its scales.

It refuses nothing. It answers where the target is in the FIB view, the
offset (dy for `vertical_move`, dz the height error it means), and how far to
trust it: the match score, how well the two scales agree, how close a second
match comes, and an overlay to look at. dx is a reading of its own: a height
error cannot move x, so a steady dx says the beams are shifted (beam shift,
scan rotation), never something a stage move fixes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from fibsem.alignment.coincidence import (
    AGREEMENT_BANDS,
    DEFAULT_AGREEMENT_TOLERANCE,
    DEFAULT_MAX_LATERAL_OFFSET,
    DEFAULT_MAX_RIVAL_RATIO,
    RIVAL_LOBE,
    CoincidenceGeometry,
    _band_pass,
    _local_normalise,
    geometry_from_images,
)
from fibsem.structures import FibsemImage

METHOD = "target-patch"
# Width and height of the SEM patch around the target. About what the agent
# matched on 2026-10-15 (320 x 280 preview px); clipped to the frame.
DEFAULT_PATCH_SIZE: Tuple[float, float] = (60e-6, 50e-6)  # m
# How far either side of the geometry's foreshortening to search, as a
# fraction of it. The geometry knows the number; a wide free search can lock
# onto a wrong match that happens to score better at the wrong squash.
DEFAULT_STRETCH_SEARCH = 0.05
STRETCH_STEP = 0.01
MIN_PATCH_PX = 16
OVERLAY_WIDTH = 1024  # px, the overlay's width (both views side by side)


@dataclass
class TargetCoincidence:
    """Where the SEM target is in the FIB view, and what that means.

    Sign convention as `CoincidenceMeasurement`: dx/dy are the shift that
    brings the FIB-view scene onto the SEM-view scene, in the FIB image plane
    (x right, y down); dy is what ``vertical_move`` (ION) is handed, dz the
    height error it means. dx is never folded into a move.
    """

    dx: float  # m, lateral: beam shift or scan rotation, not height
    dy: float  # m, vertical in the FIB image plane
    dz: float  # m, inferred height error
    score: float  # normalised correlation of the match (lower of the two scales)
    band_disagreement: float  # m, distance between the two scales' answers
    rival_ratio: float  # best match outside the chosen one's lobe, relative to it
    y_stretch: float  # the foreshortening the match was found at
    geometry_y_stretch: float  # the foreshortening the pair's geometry gives
    target: Tuple[float, float]  # m, in the SEM image (origin centre, +y up)
    target_px: Tuple[float, float]  # SEM image px (origin top-left)
    found_px: Tuple[float, float]  # FIB image px where the target was found
    expected_px: Tuple[float, float]  # FIB image px where it would be if coincident
    patch_px: Tuple[int, int, int, int]  # SEM patch: left, top, width, height
    pixel_size: float  # m/px
    warnings: List[str] = field(default_factory=list)
    method: str = METHOD

    @property
    def agrees(self) -> bool:
        """Whether nothing in the measurement itself argues against it."""
        return not self.warnings

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "agrees": self.agrees}


def _pixels(sem: np.ndarray, target: Tuple[float, float], pixel_size: float):
    h, w = sem.shape
    tx = w / 2 + target[0] / pixel_size
    ty = h / 2 - target[1] / pixel_size
    if not (0 <= tx < w and 0 <= ty < h):
        raise ValueError(
            f"target ({target[0]:.3g}, {target[1]:.3g}) m is outside the SEM image"
        )
    return tx, ty


def _patch_box(
    shape: Tuple[int, int], tx: float, ty: float, size: Tuple[float, float], px: float
) -> Tuple[int, int, int, int]:
    h, w = shape
    half_w, half_h = size[0] / 2 / px, size[1] / 2 / px
    x0, x1 = max(0, int(round(tx - half_w))), min(w, int(round(tx + half_w)))
    y0, y1 = max(0, int(round(ty - half_h))), min(h, int(round(ty + half_h)))
    if x1 - x0 < MIN_PATCH_PX or y1 - y0 < MIN_PATCH_PX:
        raise ValueError("the patch around the target is too small to match")
    return x0, y0, x1 - x0, y1 - y0


def _subpixel(scores: np.ndarray, y: int, x: int) -> Tuple[float, float]:
    """The peak at (y, x) refined by a parabola through its neighbours."""

    def offset(minus: float, centre: float, plus: float) -> float:
        denominator = minus - 2 * centre + plus
        return 0.0 if denominator == 0 else 0.5 * (minus - plus) / denominator

    h, w = scores.shape
    oy = offset(*scores[y - 1 : y + 2, x]) if 0 < y < h - 1 else 0.0
    ox = offset(*scores[y, x - 1 : x + 2]) if 0 < x < w - 1 else 0.0
    return y + oy, x + ox


def _rival(scores: np.ndarray, y: int, x: int, lobe_px: float) -> float:
    primary = float(scores[y, x])
    if primary <= 0:
        return 0.0
    yy, xx = np.ogrid[0 : scores.shape[0], 0 : scores.shape[1]]
    outside = (yy - y) ** 2 + (xx - x) ** 2 > lobe_px**2
    if not outside.any():
        return 0.0
    return max(0.0, float(scores[outside].max()) / primary)


def _match(
    sem: np.ndarray,
    fib: np.ndarray,
    box: Tuple[int, int, int, int],
    factors: np.ndarray,
    lobe_px: float,
    bands: Tuple[float, float],
):
    """The patch at *box* of the normalised SEM image found in the normalised
    FIB image, already stretched into the SEM's projection, at one band-pass
    scale. The small search either side of the geometry (*factors*) squashes
    the filtered patch. Answers (score, factor, top-left (y, x) px sub-pixel,
    rival)."""
    sem_p = _band_pass(sem, *bands).astype(np.float32)
    fib_p = _band_pass(fib, *bands).astype(np.float32)
    x0, y0, pw, ph = box
    patch = sem_p[y0 : y0 + ph, x0 : x0 + pw]
    best = None
    for factor in factors:
        height = int(round(ph / factor))
        if height > fib_p.shape[0] or pw > fib_p.shape[1]:
            continue
        squashed = cv2.resize(patch, (pw, height), interpolation=cv2.INTER_LINEAR)
        scores = cv2.matchTemplate(fib_p, squashed, cv2.TM_CCOEFF_NORMED)
        _, score, _, (lx, ly) = cv2.minMaxLoc(scores)
        if best is None or score > best[0]:
            best = (float(score), float(factor), scores, ly, lx)
    if best is None:
        raise ValueError("the FIB image is too small for the patch")
    score, factor, scores, ly, lx = best
    return score, factor, _subpixel(scores, ly, lx), _rival(scores, ly, lx, lobe_px)


def measure_coincidence_at_target(
    sem_image: FibsemImage,
    fib_image: FibsemImage,
    target: Tuple[float, float] = (0.0, 0.0),
    patch_size: Tuple[float, float] = DEFAULT_PATCH_SIZE,
    stretch_search: float = DEFAULT_STRETCH_SEARCH,
    geometry: Optional[CoincidenceGeometry] = None,
) -> TargetCoincidence:
    """Find the SEM image's *target* in the FIB image of the same pair.

    *target* is metres from the SEM image's centre (+y up): the feature you
    centred is (0, 0). The pair must share size and pixel size. The geometry
    comes from the pair's own metadata unless given. Pure image computation:
    no acquisition, no motion, no refusal.
    """
    sem = np.asarray(sem_image.data, dtype=np.float32)
    fib = np.asarray(fib_image.data, dtype=np.float32)
    if sem.shape != fib.shape:
        raise ValueError(f"image shapes differ: {sem.shape} vs {fib.shape}")
    if geometry is None:
        geometry = geometry_from_images(sem_image, fib_image)
    px = geometry.pixel_size
    h, w = sem.shape
    tx, ty = _pixels(sem, target, px)
    box = _patch_box(sem.shape, tx, ty, patch_size, px)
    x0, y0, _, _ = box
    steps = int(round(stretch_search / STRETCH_STEP))
    factors = 1 + STRETCH_STEP * np.arange(-steps, steps + 1)
    stretches = geometry.y_stretch * factors

    # The FIB image stretched in y into the SEM's projection by the geometry,
    # whole rather than cropped, so the whole frame is searched; both are
    # normalised once and band-passed at each scale, as the whole-frame
    # measure filters them.
    height = int(round(h * geometry.y_stretch))
    scale = height / h
    stretched = cv2.resize(fib, (w, height), interpolation=cv2.INTER_LINEAR)
    sem_n, fib_n = _local_normalise(sem), _local_normalise(stretched)

    answers = []
    for bands in AGREEMENT_BANDS:
        score, factor, (top, left), rival = _match(
            sem_n, fib_n, box, factors, RIVAL_LOBE / px, bands
        )
        stretch = geometry.y_stretch * factor
        # the target sits in the patch at (tx - x0, ty - y0), squashed by factor
        found = (left + (tx - x0), (top + (ty - y0) / factor) / scale)
        expected = (tx, h / 2 + (ty - h / 2) / stretch)
        answers.append((score, stretch, found, expected, rival))

    (score_a, stretch_a, found_a, expected_a, rival_a) = answers[0]
    (score_b, stretch_b, found_b, expected_b, rival_b) = answers[1]
    # the shift that brings the FIB scene onto the SEM scene: minus where the
    # target moved to (x right, y down)
    shifts = [(-(f[0] - e[0]) * px, -(f[1] - e[1]) * px) for _, _, f, e, _ in answers]
    dx = float(np.mean([s[0] for s in shifts]))
    dy = float(np.mean([s[1] for s in shifts]))
    disagreement = float(
        np.hypot(shifts[0][0] - shifts[1][0], shifts[0][1] - shifts[1][1])
    )
    score = min(score_a, score_b)
    rival = max(rival_a, rival_b)
    found = tuple(float(v) for v in np.mean([found_a, found_b], axis=0))
    expected = tuple(float(v) for v in np.mean([expected_a, expected_b], axis=0))

    warnings: List[str] = []
    if disagreement > DEFAULT_AGREEMENT_TOLERANCE:
        warnings.append(
            f"the two filter scales disagree by {disagreement * 1e6:.1f} um"
        )
    if rival > DEFAULT_MAX_RIVAL_RATIO:
        warnings.append(f"a second match is nearly as good ({rival:.2f} of it)")
    if abs(dx) > DEFAULT_MAX_LATERAL_OFFSET:
        warnings.append(
            f"dx is {dx * 1e6:+.1f} um: a height error cannot move x, so either "
            "this is the wrong match or the beams are shifted (beam shift, "
            "scan rotation)"
        )
    if (
        max(stretch_a, stretch_b) >= stretches[-1] - 1e-9
        or min(stretch_a, stretch_b) <= stretches[0] + 1e-9
    ):
        warnings.append(
            "the best match is at the edge of the foreshortening search: the "
            "geometry may be off, or the match wrong"
        )

    return TargetCoincidence(
        dx=dx,
        dy=dy,
        dz=float(dy * geometry.dz_per_dy),
        score=float(score),
        band_disagreement=disagreement,
        rival_ratio=float(rival),
        y_stretch=float((stretch_a + stretch_b) / 2),
        geometry_y_stretch=float(geometry.y_stretch),
        target=(float(target[0]), float(target[1])),
        target_px=(float(tx), float(ty)),
        found_px=found,
        expected_px=expected,
        patch_px=box,
        pixel_size=float(px),
        warnings=warnings,
    )


def _to_uint8(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    low, high = np.percentile(data, (0.5, 99.5))
    scaled = (data - low) / max(high - low, 1e-6)
    return (np.clip(scaled, 0, 1) * 255).astype(np.uint8)


def render_target_coincidence(
    sem_image: FibsemImage, fib_image: FibsemImage, result: TargetCoincidence
) -> np.ndarray:
    """The SEM (patch and target) and the FIB (the match, where the target was
    found and where it would be if coincident) side by side: an RGB image
    ``OVERLAY_WIDTH`` px wide."""
    sem = cv2.cvtColor(_to_uint8(sem_image.data), cv2.COLOR_GRAY2RGB)
    fib = cv2.cvtColor(_to_uint8(fib_image.data), cv2.COLOR_GRAY2RGB)
    h, w = sem.shape[:2]
    thickness = max(1, w // 400)
    marker = max(12, w // 40)
    yellow, red, cyan = (255, 220, 0), (255, 60, 40), (0, 200, 255)

    x0, y0, pw, ph = result.patch_px
    cv2.rectangle(sem, (x0, y0), (x0 + pw, y0 + ph), yellow, thickness)
    tx, ty = (int(round(v)) for v in result.target_px)
    cv2.drawMarker(sem, (tx, ty), red, cv2.MARKER_CROSS, marker, thickness + 1)

    fx, fy = result.found_px
    squashed = ph / result.y_stretch
    top = fy - (result.target_px[1] - y0) / result.y_stretch
    left = fx - (result.target_px[0] - x0)
    cv2.rectangle(
        fib,
        (int(round(left)), int(round(top))),
        (int(round(left + pw)), int(round(top + squashed))),
        yellow,
        thickness,
    )
    ex, ey = (int(round(v)) for v in result.expected_px)
    cv2.drawMarker(fib, (ex, ey), cyan, cv2.MARKER_TILTED_CROSS, marker, thickness + 1)
    cv2.drawMarker(
        fib,
        (int(round(fx)), int(round(fy))),
        red,
        cv2.MARKER_CROSS,
        marker,
        thickness + 1,
    )

    font, scale = cv2.FONT_HERSHEY_SIMPLEX, w / 1000
    cv2.putText(
        sem, "SEM: target", (10, int(30 * scale) + 10), font, scale, red, thickness
    )
    label = (
        f"FIB: found (+) vs coincident (x)  dy {result.dy * 1e6:+.1f} um  "
        f"dx {result.dx * 1e6:+.1f} um  score {result.score:.2f}"
    )
    cv2.putText(fib, label, (10, int(30 * scale) + 10), font, scale, red, thickness)

    both = np.concatenate(
        [sem, np.full((h, max(4, w // 100), 3), 255, np.uint8), fib], 1
    )
    height = int(round(both.shape[0] * OVERLAY_WIDTH / both.shape[1]))
    return cv2.resize(both, (OVERLAY_WIDTH, height), interpolation=cv2.INTER_AREA)
