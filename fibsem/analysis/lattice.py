"""The grid-bar lattice on a saved image, and the grid squares it bounds.

The image is reduced to at most WORK_SIDE pixels and flattened; its two
lattice steps (independent: a tilted view foreshortens one) come from its
autocorrelation, accepted when the mean cell folded from one half of the
image explains the other (else kept as a candidate for the agent to judge).
The bars' phase and width are read off the mean
cell, and each square's centre is refined on its own neighbourhood, so a
mosaic's seams show as squares off the one lattice. Answers are in
full-resolution pixels."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from fibsem.analysis.images import Planes
from fibsem.imaging.marks import rgb_of
from fibsem.theme.tokens import (
    ACCENT_COLOR,
    DEFECT_ORANGE_COLOR,
    PANEL_COLOR,
    SAVED_POSITION_COLOUR,
    SELECTED_POSITION_COLOUR,
    SEMANTIC_WARNING_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
)

# The long side the lattice is found on; answers are scaled back up.
WORK_SIDE = 1536
# Bins per lattice direction in the folded mean cell.
CELL_BINS = 48
# The shortest lattice step looked for, in working pixels.
MIN_STEP = 12
# Squares listed by default, nearest the image centre or the first point.
DEFAULT_LIMIT = 9
MAX_LIMIT = 200
# Rounds of folding with each square moved back by its own shift.
REFINE_ROUNDS = 2
# How much of the image's variance the mean cell must explain, scored on
# one half of the image after folding the other (a lattice the image doesn't
# have explains none), for the fit to be accepted.
MIN_EXPLAINED = 0.01
# A square whose bar edges correlate at least this with the mean cell's is
# counted as showing its bars ("matched").
MATCH = 0.3
# The smallest cell scoring at least this fraction of the best is chosen (a
# cell of two squares scores about as well as one).
CLOSE = 0.85
# A pitch hint matches a candidate whose longer step is within this fraction
# of it.
HINT_TOLERANCE = 0.15


class _NoLattice(Exception):
    pass


def find_grid_squares(
    image: Planes,
    points: Optional[Sequence[Sequence[float]]] = None,
    limit: Optional[int] = None,
    box: Optional[Sequence[float]] = None,
    pitch_um: Optional[float] = None,
) -> Dict[str, Any]:
    """`grid_answer` on the lattice fitted to *image*."""
    fit = fit_lattice(image, box, pitch_um)
    return grid_answer(fit, image.pixel_size, points, limit)


def fit_lattice(
    image: Planes,
    box: Optional[Sequence[float]] = None,
    pitch_um: Optional[float] = None,
) -> Dict[str, Any]:
    """The grid-bar lattice on *image* (its first layer; a stack's max
    projection), fitted in *box* ([x, y, width, height] full-resolution
    pixels) when given and preferring a longer step near *pitch_um*, with
    every square in view, for `grid_answer` and `grid_figure`.

    ``accepted`` says whether the fit passed; when it didn't, ``reason``
    says why and the best candidate is kept for the agent to judge. Without
    any candidate (a flat image, no repeat in two directions) there is no
    ``basis``."""
    _, data = image.layer(None)
    full = np.asarray(data).max(axis=0) if data.ndim == 3 else np.asarray(data)
    region = _box(box, full.shape)
    hint = None
    if pitch_um is not None:
        if pitch_um <= 0:
            raise ValueError("pitch_um is a length above 0")
        hint = {"pitch_um": float(pitch_um)}
        if not image.pixel_size:
            hint["note"] = "Not used: the image has no pixel size."
    try:
        fit = _fit(full, region, hint, image.pixel_size)
    except _NoLattice as refusal:
        fit = {"reason": str(refusal), "accepted": False}
    fit.update(shape=full.shape, box=list(region) if box is not None else None)
    fit["hint"] = fit.get("hint", hint)
    return fit


def _box(box, shape) -> Tuple[int, int, int, int]:
    """*box* as whole pixels inside an image of *shape*; the whole image
    when None. ValueError when it holds too little."""
    height, width = shape
    if box is None:
        return 0, 0, width, height
    if len(box) != 4:
        raise ValueError("box is [x, y, width, height] in the image's pixels")
    x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
    x1 = min(width, int(math.ceil(box[0] + box[2])))
    y1 = min(height, int(math.ceil(box[1] + box[3])))
    if x1 - x0 < 4 * MIN_STEP or y1 - y0 < 4 * MIN_STEP:
        raise ValueError(
            f"box {list(box)} holds too little of the {width} x {height} image "
            f"to fit a lattice in (at least {4 * MIN_STEP} px a side)"
        )
    return x0, y0, x1 - x0, y1 - y0


def grid_answer(
    fit: Dict[str, Any],
    pixel: Optional[float],
    points: Optional[Sequence[Sequence[float]]] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """What *fit* says: the lattice, the *limit* squares nearest the first
    of *points* (else the image centre) and how many are in view, and for
    each of *points* ([x, y] full-resolution pixels) the square it is in and
    how far it is from the nearest bar edge, in metres too with the *pixel*
    size. When the fit was not accepted, ``lattice`` is None with
    ``reason``, and the same answer for the best candidate is under
    ``candidate``."""
    limit = DEFAULT_LIMIT if limit is None else int(limit)
    if not 0 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit is 0 to {MAX_LIMIT}")
    points = [_point(p) for p in (points or [])]
    height, width = fit["shape"]
    answer: Dict[str, Any] = {
        "size_px": [width, height],
        "pixel_size": pixel,
        "box": fit["box"],
        "lattice": None,
        "reason": fit["reason"],
        "warnings": [],
    }
    if fit["hint"] is not None:
        answer["hint"] = fit["hint"]
    measured: Dict[str, Any] = {}
    if "basis" in fit:
        measured = _lattice_answer(fit, pixel)
        measured["quality"] = fit["quality"]
        measured["warnings"] = fit["warnings"]
        squares = fit["squares"]
        fitted = [s for s in squares if not s["extrapolated"]]
        measured["squares_in_view"] = len(squares)
        measured["squares_fitted"] = len(fitted)
        measured["squares_matched"] = sum(1 for s in fitted if s["matched"])
        near = points[0] if points else [width / 2, height / 2]
        nearest = sorted(
            squares,
            key=lambda s: math.hypot(s["px"][0] - near[0], s["px"][1] - near[1]),
        )[:limit]
        measured["squares"] = [_square_answer(s) for s in nearest]
        measured["points"] = [_point_answer(fit, p, pixel) for p in points]
    if fit["accepted"]:
        answer["lattice"] = {
            k: measured.pop(k) for k in ("a", "b", "angle_deg", "angle_between_deg")
        }
        answer["lattice"]["bars"] = measured.pop("bars")
        answer.update(measured)
        return answer
    answer["candidate"] = measured or None
    answer["points"] = [{"px": p, "square": None} for p in points]
    return answer


def _point(p) -> List[float]:
    if not (isinstance(p, (list, tuple)) and len(p) == 2):
        raise ValueError("each point is [x, y] in the image's pixels")
    return [float(p[0]), float(p[1])]


# --- the fit -------------------------------------------------------------------


def _flattened(img: np.ndarray) -> np.ndarray:
    """*img* with its slow shading removed and its extremes clipped, at zero
    mean and unit spread."""
    from scipy.ndimage import gaussian_filter, zoom

    lo, hi = np.percentile(img, (1, 99))
    img = np.clip(img, lo, hi)
    # The shading is a blur a quarter of the image wide, taken small.
    small, _ = _block(img, 128)
    shading = gaussian_filter(small, max(small.shape) / 4, mode="nearest")
    shading = zoom(
        shading, (img.shape[0] / small.shape[0], img.shape[1] / small.shape[1]), order=1
    )
    flat = img - shading[: img.shape[0], : img.shape[1]]
    lo, hi = np.percentile(flat, (2, 98))
    flat = np.clip(flat, lo, hi)
    flat = flat - flat.mean()
    spread = flat.std()
    # A thousandth of a grey level: what is left of a uniform image.
    if spread < 1e-3:
        raise _NoLattice("The image is flat: there are no bars to find.")
    return flat / spread


def _autocorrelation(x: np.ndarray) -> np.ndarray:
    """The normalised autocorrelation of *x*, lag (0, 0) at [h, w], each lag
    divided by the pixels it overlaps."""
    h, w = x.shape
    spectrum = np.fft.rfft2(x, s=(2 * h, 2 * w))
    acf = np.fft.fftshift(np.fft.irfft2(np.abs(spectrum) ** 2, s=(2 * h, 2 * w)))
    dy = h - np.abs(np.arange(-h, h))
    dx = w - np.abs(np.arange(-w, w))
    acf /= np.maximum(np.outer(dy, dx), 1)
    return acf / acf[h, w]


def _peaks(acf: np.ndarray, shortest: float) -> List[Tuple[float, np.ndarray]]:
    """The autocorrelation's local maxima in one half plane, strongest
    first, as (value, lag [dx, dy]): lags at least *shortest* long, where at
    least half the image overlaps."""
    from scipy.ndimage import maximum_filter

    h, w = acf.shape[0] // 2, acf.shape[1] // 2
    size = 2 * int(max(2, shortest // 3)) + 1
    ys, xs = np.nonzero((maximum_filter(acf, size=size) == acf) & (acf > 0))
    found = []
    for y, x in zip(ys, xs):
        dy, dx = int(y) - h, int(x) - w
        if dy < 0 or (dy == 0 and dx <= 0) or math.hypot(dx, dy) < shortest:
            continue
        if (h - abs(dy)) * (w - abs(dx)) < h * w / 2:
            continue
        found.append((float(acf[y, x]), np.array([dx, dy], dtype=float)))
    found.sort(key=lambda p: -p[0])
    return found


def _vertex(three: np.ndarray) -> float:
    """The sub-sample offset of the peak of a parabola through *three*."""
    a, b, c = (float(v) for v in three)
    d = a - 2 * b + c
    return 0.0 if d >= 0 else float(np.clip(0.5 * (a - c) / d, -0.5, 0.5))


def _block(img: np.ndarray, side: int) -> Tuple[np.ndarray, int]:
    """*img* block-averaged so its long side is at most *side*, and the factor."""
    h, w = img.shape
    c = max(1, math.ceil(max(h, w) / side))
    h, w = h // c * c, w // c * c
    return img[:h, :w].reshape(h // c, c, w // c, c).mean(axis=(1, 3)), c


def _cell_r2(img: np.ndarray, basis: np.ndarray, bins: int, axis=None) -> float:
    """The fraction of *img*'s variance its mean cell on *basis* explains,
    cross-validated: the cell folded from one checkerboard of blocks is
    scored on the other, so a large cell that only memorises the image
    scores no better than the lattice's own. With *axis* (0 along a, 1
    along b), the mean profile across that direction's bars alone."""
    h, w = img.shape
    ys, xs = np.mgrid[0:h, 0:w]
    side = max(h, w) / 6
    half = ((ys // side + xs // side) % 2).astype(bool).ravel()
    uv = np.linalg.inv(basis) @ np.vstack([xs.ravel(), ys.ravel()])
    index = np.floor((uv % 1.0) * bins).astype(int) % bins
    if axis is None:
        flat_index = index[0] * bins + index[1]
    else:
        flat_index = index[axis]
    values = img.ravel()
    residual = total = 0.0
    for train in (half, ~half):
        test = ~train
        sums = np.bincount(flat_index[train], values[train], bins * bins)
        counts = np.bincount(flat_index[train], minlength=bins * bins)
        cell = np.where(counts > 0, sums / np.maximum(counts, 1), values[train].mean())
        model = cell[flat_index[test]]
        residual += float(((values[test] - model) ** 2).sum())
        total += float(((values[test] - values[test].mean()) ** 2).sum())
    return 1.0 - residual / total


def _basis(flat: np.ndarray, hint_px: Optional[float] = None):
    """The lattice's two shortest steps (columns of a 2x2, working pixels),
    the cross-validated score of the chosen cell and of the best, and what a
    pitch hint did. Raises _NoLattice only when there is no candidate.

    Candidates are pairs of strong autocorrelation peaks of a coarse copy;
    the one kept is the smallest cell scoring at least CLOSE of the best (a
    cell of two squares scores as well as one), among those whose longer
    step is within HINT_TOLERANCE of *hint_px* when any is. Its steps are
    then refined on the working image's autocorrelation."""
    from scipy.ndimage import gaussian_filter

    coarse, c = _block(flat, 512)
    acf = gaussian_filter(_autocorrelation(coarse), 1.0)
    found = _peaks(acf, max(6.0, MIN_STEP / c))
    # The strongest, and the shortest of the strong: on a clean grid every
    # lattice point is about as strong, and the shortest are its steps.
    strong = [p for p in found if p[0] >= 0.5 * found[0][0]] if found else []
    strong.sort(key=lambda p: np.hypot(*p[1]))
    peaks = found[:10] + [p for p in strong[:10] if not any(p is q for q in found[:10])]
    candidates = {}
    for i, (_, p) in enumerate(peaks):
        for _, q in peaks[i + 1 :]:
            sine = abs(p[0] * q[1] - p[1] * q[0]) / (np.hypot(*p) * np.hypot(*q))
            if sine < 0.34:
                continue
            basis = _oriented(_reduced(p, q))
            if min(np.hypot(*basis[:, 0]), np.hypot(*basis[:, 1])) < MIN_STEP / c:
                continue
            candidates[tuple(np.rint(basis).astype(int).ravel())] = basis
    if not candidates:
        raise _NoLattice(
            "The image repeats along no two directions: no lattice is in view."
        )
    refined = {}
    for basis in candidates.values():
        basis = _refined(acf, basis)
        refined[tuple(np.rint(basis * 2).astype(int).ravel())] = basis
    scored = [
        (_cell_r2(coarse, b, 24), abs(np.linalg.det(b)), b) for b in refined.values()
    ]

    def smallest_close(pool):
        top = max(r for r, _, _ in pool)
        return min(
            (s for s in pool if s[0] >= top - (1 - CLOSE) * abs(top)),
            key=lambda s: s[1],
        )

    def longer(b):
        return max(np.hypot(*b[:, 0]), np.hypot(*b[:, 1])) * c

    best = max(r for r, _, _ in scored)
    chosen = default = smallest_close(scored)
    hint = None
    if hint_px:
        # A cell of several of the chosen squares fits as well and says
        # nothing new, so the hint never moves the choice to one.
        near = [
            s
            for s in scored
            if abs(longer(s[2]) - hint_px) <= HINT_TOLERANCE * hint_px
            and not _supercell(s[2], default[2])
        ]
        if near:
            chosen = smallest_close(near)
        hint = {
            "matched": bool(near),
            "changed_choice": chosen is not default,
            "unhinted_longer_step_px": longer(default[2]),
        }
    basis = _refined(gaussian_filter(_autocorrelation(flat), 1.0), chosen[2] * c)
    return basis, chosen[0], best, hint


def _supercell(cell: np.ndarray, of: np.ndarray) -> bool:
    """Whether *cell* is several cells of the lattice *of*: its steps whole
    multiples of *of*'s, within a tenth of a step, and its area at least
    twice."""
    k = np.linalg.inv(of) @ cell
    whole = np.abs(k - np.rint(k)).max() < 0.1
    return bool(whole and abs(np.linalg.det(np.rint(k))) >= 2)


def _refined(acf: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """*basis* refined by least squares on the autocorrelation *acf*'s peaks
    near its lattice points."""
    h, w = acf.shape[0] // 2, acf.shape[1] // 2
    reach = 0.15 * min(np.hypot(*basis[:, 0]), np.hypot(*basis[:, 1]))
    rows, targets, weights = [], [], []
    for n in range(-8, 9):
        for m in range(0, 9):
            if m == 0 and n <= 0:
                continue
            guess = basis @ np.array([n, m])
            if (h - abs(guess[1]) - reach) * (w - abs(guess[0]) - reach) < h * w / 2:
                continue
            x0, y0 = int(round(guess[0])) + w, int(round(guess[1])) + h
            r = int(math.ceil(reach))
            patch = acf[y0 - r : y0 + r + 1, x0 - r : x0 + r + 1]
            if patch.size == 0:
                continue
            y, x = np.unravel_index(int(np.argmax(patch)), patch.shape)
            if not (0 < y < patch.shape[0] - 1 and 0 < x < patch.shape[1] - 1):
                continue  # no peak within reach: on the patch's edge
            rows.append([n, m])
            targets.append(
                [
                    x0 - r + x - w + _vertex(patch[y, x - 1 : x + 2]),
                    y0 - r + y - h + _vertex(patch[y - 1 : y + 2, x]),
                ]
            )
            weights.append(max(float(patch[y, x]), 0.0))
    if len(rows) < 2 or np.linalg.matrix_rank(np.array(rows)) < 2:
        return _oriented(basis)
    sw = np.sqrt(np.array(weights))[:, None]
    solved, *_ = np.linalg.lstsq(
        np.array(rows) * sw, np.array(targets) * sw, rcond=None
    )
    # Kept only when it stays near the guess: peaks of noise can pull a
    # fit anywhere.
    moved = np.hypot(*(solved.T - basis)) / np.hypot(*basis)
    if not np.all(np.isfinite(solved)) or moved.max() > 0.2:
        return _oriented(basis)
    return _oriented(solved.T)


def _reduced(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Lagrange-Gauss reduction: the shortest basis of the lattice a, b span."""
    a, b = a.astype(float), b.astype(float)
    for _ in range(20):
        if a @ a > b @ b:
            a, b = b, a
        k = round((a @ b) / (a @ a))
        if k == 0:
            break
        b = b - k * a
    return np.column_stack([a, b])


def _oriented(basis: np.ndarray) -> np.ndarray:
    """*basis* as (a, b): a the step nearer +x (columns), b nearer +y (rows)."""
    a, b = basis[:, 0], basis[:, 1]
    if abs(a[0]) / np.hypot(*a) < abs(b[0]) / np.hypot(*b):
        a, b = b, a
    if a[0] < 0:
        a = -a
    if b[1] < 0:
        b = -b
    return np.column_stack([a, b])


def _fold(img: np.ndarray, uv: np.ndarray, bins: int = CELL_BINS) -> np.ndarray:
    """*img* folded onto one lattice cell by each pixel's lattice coordinates
    *uv* (2 x pixels): the mean in bins x bins, indexed [u, v]."""
    index = np.floor((uv % 1.0) * bins).astype(int) % bins
    flat = index[0] * bins + index[1]
    total = np.bincount(flat, img.ravel(), bins * bins)
    count = np.bincount(flat, minlength=bins * bins)
    return (total / np.maximum(count, 1)).reshape(bins, bins)


def _bars(cell: np.ndarray) -> Dict[str, Any]:
    """Where the bars sit in the mean cell, per direction: the phase of the
    bar's centre (0 to 1) and its width (fraction of the cell), read where
    the cell is the same all along the bar, and whether bars are darker."""
    from scipy.ndimage import gaussian_filter

    smooth = gaussian_filter(cell, 1.0, mode="wrap")
    bins = cell.shape[0]
    found = []
    for axis in (0, 1):
        # Along u (axis 0), a bar is a column of the cell constant along v.
        spread = smooth.std(axis=1 - axis)
        found.append(_low_run(spread))
    (cu, wu), (cv, wv) = found
    iu = int(round(cu * bins)) % bins
    iv = int(round(cv * bins)) % bins
    fu = int(round((cu + 0.5) * bins)) % bins
    fv = int(round((cv + 0.5) * bins)) % bins
    bar = 0.5 * (smooth[iu, fv] + smooth[fu, iv])
    return {
        "phase": (cu, cv),
        "width": (wu, wv),
        "darker": bool(bar < smooth[fu, fv]),
    }


def _low_run(profile: np.ndarray) -> Tuple[float, float]:
    """The circular run of *profile* below the half level about its minimum:
    its centre and length, as fractions of the profile."""
    n = len(profile)
    lo, hi = float(profile.min()), float(profile.max())
    level = 0.5 * (lo + hi)
    start = int(np.argmin(profile))
    left = right = None
    for k in range(1, n):
        i = (start - k) % n
        if profile[i] >= level:
            j = (i + 1) % n
            left = start - k + 1 - _fraction(profile[j], profile[i], level)
            break
    for k in range(1, n):
        i = (start + k) % n
        if profile[i] >= level:
            j = (i - 1) % n
            right = start + k - 1 + _fraction(profile[j], profile[i], level)
            break
    if left is None or right is None:
        return start / n, 0.0
    return ((left + right) / 2 / n) % 1.0, (right - left) / n


def _fraction(inside: float, outside: float, level: float) -> float:
    """How far from *inside* toward *outside* the profile crosses *level*."""
    d = outside - inside
    return 0.0 if d == 0 else float(np.clip((level - inside) / d, 0.0, 1.0))


def _fit(full: np.ndarray, region, hint, pixel) -> Dict[str, Any]:
    """The lattice fitted in *region* (x, y, width, height) of the image
    *full*, its squares over the whole image, quality and warnings, and
    whether it is accepted. Raises _NoLattice when there is no candidate."""
    x0, y0, bw, bh = region
    img, s = _block(full[y0 : y0 + bh, x0 : x0 + bw].astype(np.float32), WORK_SIDE)
    if min(img.shape) < 4 * MIN_STEP:
        raise _NoLattice("The image is too small to find a lattice in.")
    flat = _flattened(img)
    h, w = flat.shape
    hint_px = None
    if hint is not None and pixel:
        hint_px = hint["pitch_um"] * 1e-6 / pixel / s
    basis, chosen, best, hinted = _basis(flat, hint_px)
    explained = [_cell_r2(flat, basis, CELL_BINS, axis) for axis in (0, 1)]
    periods = [
        _periods_in_view(basis[:, 0], basis[:, 1], h, w),
        _periods_in_view(basis[:, 1], basis[:, 0], h, w),
    ]
    quality = {
        "explained": [round(e, 3) for e in explained],
        "explained_cell": round(chosen, 3),
        "periods_in_view": [round(p, 1) for p in periods],
    }
    reasons = []
    if chosen < MIN_EXPLAINED:
        reasons.append(
            "the best repeat found, folded from one half of the region, "
            f"predicts {max(chosen, 0):.1%} of the other"
        )
    if min(explained) < MIN_EXPLAINED:
        reasons.append(
            "the bars across one direction explain "
            f"{max(min(explained), 0):.1%} of the region"
        )
    fit = {
        "s": s,
        "origin": np.array([x0, y0], dtype=float),
        "basis": basis,
        "quality": quality,
        "warnings": [],
        "squares": [],
        "bars": None,
    }
    if hinted is not None:
        unit = s * pixel * 1e6
        steps = sorted([np.hypot(*basis[:, 0]) * unit, np.hypot(*basis[:, 1]) * unit])
        longer = steps[1]
        hint["longer_step_um"] = round(float(longer), 1)
        # Only the longer step is compared: a tilted view shortens the other.
        hint["shorter_step_um"] = round(float(steps[0]), 1)
        hint["agrees"] = bool(
            abs(longer - hint["pitch_um"]) <= HINT_TOLERANCE * hint["pitch_um"]
        )
        hint["changed_choice"] = hinted["changed_choice"]
        hint["tolerance"] = HINT_TOLERANCE
        if hinted["changed_choice"]:
            hint["note"] = (
                "The hint changed the choice: without it the longer step would "
                f"be {hinted['unhinted_longer_step_px'] * unit:.1f} um."
            )
        elif not hinted["matched"]:
            hint["note"] = (
                "No candidate's longer step is near the hint: the fit disagrees with it."
            )
        fit["hint"] = hint
    try:
        _place_squares(fit, flat)
    except _NoLattice as refusal:
        reasons.append(str(refusal))
    fit["accepted"] = not reasons
    fit["reason"] = (
        None
        if not reasons
        else "No lattice accepted: " + "; ".join(reasons) + ". The best candidate is "
        "under candidate and drawn on the figure, for you to judge."
    )
    _extend(fit, full.shape)
    if fit["bars"] is not None:
        _judge(fit)
    return fit


def _periods_in_view(step, other, h, w) -> float:
    """How many lattice periods along *step* the h x w image spans: its extent
    across the bars that cross *step* over their spacing."""
    normal = np.array([-other[1], other[0]])
    normal /= np.hypot(*normal)
    spacing = abs(step @ normal)
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]]) @ normal
    return float((corners.max() - corners.min()) / spacing)


# --- the squares ---------------------------------------------------------------


def _place_squares(fit, flat: np.ndarray) -> None:
    """The bars (`_bars`) and every square whose centre is in view: its
    lattice index, centre (working pixels) refined on its neighbourhood, its
    shift off the one lattice, how well its bar edges match the mean cell's,
    and whether its film is all in view.

    Squares are matched on edges, which show whichever way the bars and the
    film (intact or broken) differ. Each round folds the image with every
    square moved back by its shift, so the mean cell sharpens where the
    squares are off one lattice (a mosaic's seams) and the next round's fit
    is finer."""
    from scipy.ndimage import gaussian_gradient_magnitude

    basis = fit["basis"]
    h, w = flat.shape
    shortest = min(np.hypot(*basis[:, 0]), np.hypot(*basis[:, 1]))
    edges = gaussian_gradient_magnitude(flat, max(1.5, shortest / 40))
    inverse = np.linalg.inv(basis)
    ys, xs = np.mgrid[0:h, 0:w]
    uv0 = inverse @ np.vstack([xs.ravel(), ys.ravel()]).astype(float)
    uv = uv0
    squares: List[Dict[str, Any]] = []
    for _ in range(REFINE_ROUNDS):
        bars = _bars(_fold(flat, uv))
        if min(bars["width"]) <= 0:
            raise _NoLattice("no bars could be read off its mean cell")
        edge_cell = _fold(edges, uv)
        phase = np.array(bars["phase"])
        first = not squares
        if first:
            squares = _lattice_squares(basis, phase, h, w)
        for sq in squares:
            sq["lattice"] = basis @ (phase + 0.5 + np.array(sq["index"]))
            if not first and (sq["match"] or 0) < MATCH / 2:
                continue  # no bars to fit the first time: none the next
            delta, sq["match"] = _refine(
                edges, edge_cell, basis, sq["lattice"], sq["shift"]
            )
            sq["shift"] = sq["shift"] + delta
            if np.hypot(*sq["shift"]) > 0.3 * shortest:
                sq["match"] = None  # wandered off: its own fit is not trusted
        _settle_shifts(squares)
        uv = uv0 - inverse @ _shift_field(squares, phase, uv0)
    fit["bars"] = bars
    for sq in squares:
        sq["centre"] = sq["lattice"] + sq["shift"]
        sq["matched"] = sq["match"] is not None and sq["match"] >= MATCH
        sq["px"] = _full_px(fit, sq["centre"])
        sq["extrapolated"] = False
    fit["squares"] = squares


def _full_px(fit, q) -> np.ndarray:
    """Working pixels *q* of the fitted region as full-resolution pixels of
    the image."""
    return fit["origin"] + (np.asarray(q, dtype=float) + 0.5) * fit["s"] - 0.5


def _extend(fit, shape) -> None:
    """The squares outside the fitted region, by extending its lattice
    (``extrapolated``), each square's film corners in full-resolution
    pixels, whether they are all in the image, and every square's id."""
    if fit["bars"] is None:
        return
    s, basis = fit["s"], fit["basis"]
    height, width = shape
    phase = np.array(fit["bars"]["phase"])
    fitted = {sq["index"] for sq in fit["squares"]}
    inverse = np.linalg.inv(basis)
    corners = np.array([[0, width, 0, width], [0, 0, height, height]], dtype=float)
    uv = inverse @ ((corners - fit["origin"][:, None] + 0.5) / s - 0.5)
    lo = np.floor(uv.min(axis=1) - phase).astype(int) - 1
    hi = np.ceil(uv.max(axis=1) - phase).astype(int) + 1
    for i in range(lo[0], hi[0]):
        for j in range(lo[1], hi[1]):
            if (i, j) in fitted:
                continue
            centre = basis @ (phase + 0.5 + np.array([i, j]))
            px = _full_px(fit, centre)
            if 0 <= px[0] < width and 0 <= px[1] < height:
                fit["squares"].append(
                    {
                        "index": (i, j),
                        "centre": centre,
                        "shift": np.zeros(2),
                        "match": None,
                        "matched": False,
                        "px": px,
                        "extrapolated": True,
                    }
                )
    wu, wv = fit["bars"]["width"]
    half = np.array([(1 - wu) / 2, (1 - wv) / 2])
    for sq in fit["squares"]:
        sq["corners_px"] = [
            _full_px(fit, sq["centre"] + basis @ (half * np.array(c)))
            for c in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]
        sq["in_view"] = all(
            0 <= x <= width and 0 <= y <= height for x, y in sq["corners_px"]
        )
    _label(fit["squares"])


def _lattice_squares(basis, phase, h, w) -> List[Dict[str, Any]]:
    """The squares of the lattice whose centres (u = phase + 0.5 + i) fall
    in the h x w image, unshifted."""
    corners = np.linalg.inv(basis) @ np.array([[0, w, 0, w], [0, 0, h, h]], dtype=float)
    lo = np.floor(corners.min(axis=1) - phase).astype(int) - 1
    hi = np.ceil(corners.max(axis=1) - phase).astype(int) + 1
    squares = []
    for i in range(lo[0], hi[0]):
        for j in range(lo[1], hi[1]):
            centre = basis @ (phase + 0.5 + np.array([i, j]))
            if 0 <= centre[0] < w and 0 <= centre[1] < h:
                squares.append({"index": (i, j), "shift": np.zeros(2), "match": None})
    return squares


def _shift_field(squares, phase, uv0) -> np.ndarray:
    """Each pixel's square's shift (2 x pixels), by the cell of the one
    lattice it is in; the median shift where no square was placed."""
    index = np.array([sq["index"] for sq in squares])
    lo = index.min(axis=0)
    shape = index.max(axis=0) - lo + 1
    table = np.empty((shape[0], shape[1], 2))
    table[:] = np.median([sq["shift"] for sq in squares], axis=0)
    for sq in squares:
        table[sq["index"][0] - lo[0], sq["index"][1] - lo[1]] = sq["shift"]
    cells = np.floor(uv0 - phase[:, None]).astype(int) - lo[:, None]
    i = np.clip(cells[0], 0, shape[0] - 1)
    j = np.clip(cells[1], 0, shape[1] - 1)
    return table[i, j].T


def _cell_model(
    cell: np.ndarray, basis: np.ndarray, xs: np.ndarray, ys: np.ndarray
) -> np.ndarray:
    """The mean cell at working pixels (xs, ys), bilinear and periodic."""
    from scipy.ndimage import map_coordinates

    bins = cell.shape[0]
    uv = np.linalg.inv(basis) @ np.vstack([xs.ravel(), ys.ravel()])
    coords = (uv % 1.0) * bins - 0.5
    tiled = np.pad(cell, 1, mode="wrap")
    return map_coordinates(tiled, coords + 1, order=1, mode="nearest").reshape(xs.shape)


def _refine(flat, cell, basis, lattice, shift) -> Tuple[np.ndarray, Optional[float]]:
    """How much further than *shift* the square at *lattice* (working
    pixels) sits, laying the mean *cell* on it and its neighbourhood, and
    the correlation there; zero and None when too little of it is in view."""
    from skimage.feature import match_template

    centre = lattice + shift
    h, w = flat.shape
    reach = 0.6 * np.abs(basis).sum(axis=1)  # half the box about one cell
    margin = 0.25 * min(np.hypot(*basis[:, 0]), np.hypot(*basis[:, 1]))
    x0, x1 = int(centre[0] - reach[0]), int(centre[0] + reach[0]) + 1
    y0, y1 = int(centre[1] - reach[1]), int(centre[1] + reach[1]) + 1
    cx0, cx1, cy0, cy1 = max(x0, 0), min(x1, w), max(y0, 0), min(y1, h)
    if (cx1 - cx0) * (cy1 - cy0) < 0.5 * (x1 - x0) * (y1 - y0) or min(
        cx1 - cx0, cy1 - cy0
    ) < 8:
        return np.zeros(2), None
    window = flat[cy0:cy1, cx0:cx1]
    if window.std() <= 0:
        return np.zeros(2), 0.0
    m = int(np.ceil(margin))
    ys, xs = np.mgrid[cy0 - m : cy1 + m, cx0 - m : cx1 + m]
    model = _cell_model(cell, basis, xs - shift[0], ys - shift[1])
    # match_template slides the window over the model: a peak at (m, m) is no shift.
    score = match_template(model, window)
    peak = np.unravel_index(int(np.argmax(score)), score.shape)
    y, x = peak
    dy = (
        y
        - m
        + (_vertex(score[y - 1 : y + 2, x]) if 0 < y < score.shape[0] - 1 else 0.0)
    )
    dx = (
        x
        - m
        + (_vertex(score[y, x - 1 : x + 2]) if 0 < x < score.shape[1] - 1 else 0.0)
    )
    # The model moved by +d matches: the square sits at centre - d.
    return -np.array([dx, dy], dtype=float), float(score[y, x])


def _settle_shifts(squares) -> None:
    """A square that doesn't match the mean cell well takes its matched
    neighbours' median shift, or none."""
    by_index = {sq["index"]: sq for sq in squares}
    for sq in squares:
        if sq["match"] is not None and sq["match"] >= MATCH:
            continue
        i, j = sq["index"]
        near = [
            by_index[(i + a, j + b)]["shift"]
            for a in (-1, 0, 1)
            for b in (-1, 0, 1)
            if (a or b)
            and (i + a, j + b) in by_index
            and (by_index[(i + a, j + b)]["match"] or 0) >= MATCH
        ]
        sq["shift"] = np.median(near, axis=0) if near else np.zeros(2)


def _label(squares) -> None:
    """Each square's id: r<row>c<column>, counted from 1 at the top-left
    square in view."""
    if not squares:
        return
    i0 = min(sq["index"][0] for sq in squares)
    j0 = min(sq["index"][1] for sq in squares)
    for sq in squares:
        i, j = sq["index"]
        sq["id"] = f"r{j - j0 + 1}c{i - i0 + 1}"


def _judge(fit) -> None:
    """Warnings, and the fit's score from 0 to 1."""
    q, warnings, s = fit["quality"], fit["warnings"], fit["s"]
    squares = [sq for sq in fit["squares"] if not sq["extrapolated"]]
    matched = [sq for sq in squares if sq["matched"]]
    if min(q["periods_in_view"]) < 3:
        warnings.append(
            f"Only {min(q['periods_in_view']):.1f} periods fitted along one "
            "direction (warned under 3): the pitch and angle rest on few bars."
        )
    if min(q["explained"]) < 2 * MIN_EXPLAINED:
        warnings.append(
            f"The bars explain {min(q['explained']):.1%} of the fitted region along "
            f"one direction (warned under {2 * MIN_EXPLAINED:.0%}): faint bars, a "
            "weak fit."
        )
    if squares and len(matched) < 0.5 * len(squares):
        warnings.append(
            f"{len(matched)} of {len(squares)} fitted squares have match {MATCH} or "
            "more (warned under half): much of the region may be off the grid, "
            "covered or faint."
        )
    wu, wv = fit["bars"]["width"]
    spacing = [1 / np.hypot(*row) for row in np.linalg.inv(fit["basis"])]
    bar_px = min(wu * spacing[0], wv * spacing[1]) * s
    # How far squares sit from where the rest put the one lattice, and how
    # far neighbours jump from each other: a seam jumps, distortion drifts.
    own = np.array([sq["shift"] for sq in matched]).reshape(-1, 2)
    off = np.hypot(*(own - np.median(own, axis=0)).T) * s if len(own) else own
    by_index = {sq["index"]: sq["shift"] for sq in matched}
    jumps = [
        np.hypot(*(shift - by_index[(i + di, j + dj)])) * s
        for (i, j), shift in by_index.items()
        for di, dj in ((1, 0), (0, 1))
        if (i + di, j + dj) in by_index
    ]
    q["square_shift_px_p90"] = (
        round(float(np.percentile(off, 90)), 1) if len(off) else 0.0
    )
    q["neighbour_jump_px_p95"] = (
        round(float(np.percentile(jumps, 95)), 1) if jumps else 0.0
    )
    limit = max(0.25 * bar_px, 3.0 * s)
    cut = f"warned over {limit:.0f} px: a quarter of the narrower bar, or 3 working px"
    if q["neighbour_jump_px_p95"] > limit:
        warnings.append(
            f"Neighbouring squares jump up to {q['neighbour_jump_px_p95']:.0f} px "
            f"against one another ({cut}): the seams of a stitched mosaic? Each "
            "square's centre and bar distances use its own fit; check a site near "
            "a seam."
        )
    elif q["square_shift_px_p90"] > limit:
        warnings.append(
            f"Squares sit up to {q['square_shift_px_p90']:.0f} px off one lattice "
            f"({cut}): distortion or an uneven grid? Each square's centre and bar "
            "distances use its own fit."
        )
    # Full at three times the explained share accepted, and 4 periods.
    score = min(1.0, max(min(q["explained"]), 0) / (3 * MIN_EXPLAINED))
    score *= min(1.0, min(q["periods_in_view"]) / 4)
    q["score"] = round(float(score), 2)
    q["cutoffs"] = {
        "explained_accepted": MIN_EXPLAINED,
        "match_counted": MATCH,
        "shift_warned_px": round(float(limit), 1),
        "score_full_at": {"explained": 3 * MIN_EXPLAINED, "periods": 4},
    }


# --- answers -------------------------------------------------------------------


def _lattice_answer(fit, pixel: Optional[float]) -> Dict[str, Any]:
    basis, s, bars = fit["basis"], fit["s"], fit["bars"]
    inverse = np.linalg.inv(basis)
    out = {}
    for k, name in enumerate(("a", "b")):
        step = basis[:, k] * s
        spacing = s / np.hypot(*inverse[k])  # between the bars crossing this step
        bar = bars["width"][k] * spacing if bars is not None else None
        along = basis[:, 1 - k]  # the bars crossing this step run along the other
        out[name] = {
            "step_px": [round(float(step[0]), 1), round(float(step[1]), 1)],
            "pitch_px": round(float(np.hypot(*step)), 1),
            "pitch_m": _m(np.hypot(*step), pixel),
            "bar_width_px": None if bar is None else round(float(bar), 1),
            "bar_width_m": None if bar is None else _m(bar, pixel),
            "bar_angle_deg": round(
                float(np.degrees(np.arctan2(along[1], along[0]))), 1
            ),
        }
    a, b = basis[:, 0], basis[:, 1]
    out["angle_deg"] = round(float(np.degrees(np.arctan2(a[1], a[0]))), 2)
    out["angle_between_deg"] = round(
        float(np.degrees(np.arccos(a @ b / (np.hypot(*a) * np.hypot(*b))))), 1
    )
    out["bars"] = None if bars is None else ("darker" if bars["darker"] else "brighter")
    return out


def _m(px: float, pixel: Optional[float]) -> Optional[float]:
    return float(f"{px * pixel:.4g}") if pixel else None


def _square_answer(sq) -> Dict[str, Any]:
    return {
        "id": sq["id"],
        "px": [round(float(sq["px"][0]), 1), round(float(sq["px"][1]), 1)],
        "in_view": sq["in_view"],
        "match": None if sq["match"] is None else round(sq["match"], 2),
        "extrapolated": sq["extrapolated"],
    }


def _point_answer(fit, p: List[float], pixel: Optional[float]) -> Dict[str, Any]:
    """The square *p* (full-resolution pixels) is in, its distance to the
    nearest bar edge (negative on a bar) and its offset from the square's
    centre, by the lattice as that square's own fit places it; extrapolated
    when *p* is outside the fitted region."""
    s, basis = fit["s"], fit["basis"]
    if fit["bars"] is None:
        return {"px": p, "square": None, "bar_distance_px": None}
    inverse = np.linalg.inv(basis)
    cu, cv = fit["bars"]["phase"]
    wu, wv = fit["bars"]["width"]
    q = (np.array(p) - fit["origin"] + 0.5) / s - 0.5
    region = fit["box"]
    outside = region is not None and not (
        region[0] <= p[0] <= region[0] + region[2]
        and region[1] <= p[1] <= region[1] + region[3]
    )
    # The nearest square by the one lattice, then its own shift.
    uv = inverse @ q - np.array([cu + 0.5, cv + 0.5])
    index = tuple(int(k) for k in np.rint(uv))
    square = next((sq for sq in fit["squares"] if sq["index"] == index), None)
    shift = square["shift"] if square is not None else np.zeros(2)
    uv = inverse @ (q - shift) - np.array([cu + 0.5, cv + 0.5])
    index = tuple(int(k) for k in np.rint(uv))
    square = next((sq for sq in fit["squares"] if sq["index"] == index), None)
    frac = uv - np.rint(uv)  # -0.5 to 0.5 from the square's centre
    spacing = 1 / np.hypot(*inverse[0]), 1 / np.hypot(*inverse[1])
    # Perpendicular distance from each pair of bar edges, working pixels.
    edge = [
        (0.5 - abs(frac[0]) - wu / 2) * spacing[0],
        (0.5 - abs(frac[1]) - wv / 2) * spacing[1],
    ]
    distance = min(edge) * s
    centre = basis @ (np.rint(uv) + np.array([cu + 0.5, cv + 0.5])) + shift
    offset = (q - centre) * s
    on_bar = distance < 0
    # The signed distance is the fact; on_bar and square are read off it.
    return {
        "px": p,
        "bar_distance_px": round(float(distance), 1),
        "bar_distance_m": _m(distance, pixel),
        "offset_px": [round(float(offset[0]), 1), round(float(offset[1]), 1)],
        "offset_m": [_m(offset[0], pixel), _m(offset[1], pixel)] if pixel else None,
        "nearest_square": square["id"] if square is not None else None,
        "match": None
        if square is None or square["match"] is None
        else round(square["match"], 2),
        "on_bar": bool(on_bar),
        "square": square["id"] if square is not None and not on_bar else None,
        "extrapolated": bool(outside),
    }


# --- the figure ----------------------------------------------------------------

FIGURE_SIDE = 1600


def grid_figure(
    image: Planes, fit: Dict[str, Any], answer: Dict[str, Any]
) -> np.ndarray:
    """The image with each square's film outlined as fitted (solid where its
    bars show, dashed where they don't, dotted where extended past the
    fitted box), the listed squares labelled and the points marked with
    their distance to the nearest bar edge: to check the fit by eye. A
    candidate that was not accepted is drawn all in the warning colour,
    dashed, and the title says so."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    _, data = image.layer(None)
    full = np.asarray(data).max(axis=0) if data.ndim == 3 else np.asarray(data)
    height, width = full.shape
    step = max(1, math.ceil(max(height, width) / FIGURE_SIDE))
    shown = full[::step, ::step].astype(np.float32)
    lo, hi = np.percentile(shown, (0.5, 99.5))
    dpi, header = 100, 24  # the title's band, in figure pixels
    figure = Figure(
        figsize=(shown.shape[1] / dpi, (shown.shape[0] + header) / dpi),
        dpi=dpi,
        facecolor=PANEL_COLOR,
    )
    FigureCanvasAgg(figure)
    axes = figure.add_axes((0, 0, 1, shown.shape[0] / (shown.shape[0] + header)))
    axes.imshow(shown, cmap="gray", vmin=lo, vmax=hi, extent=(0, width, height, 0))
    axes.set_xlim(0, width)
    axes.set_ylim(height, 0)
    axes.axis("off")
    if fit["box"] is not None:
        x, y, w, h = fit["box"]
        axes.add_patch(
            Rectangle((x, y), w, h, fill=False, ec=ACCENT_COLOR, lw=1.2, ls="-.")
        )
    accepted = fit["accepted"]
    measured = answer if accepted else answer.get("candidate")
    if measured is None:
        title = f"No lattice, no candidate: {fit['reason']}"
    else:
        if fit["bars"] is None:
            # A candidate with no bars read: its lattice points only.
            basis = fit["basis"]
            nodes = [
                _full_px(fit, basis @ np.array([i, j]))
                for i in range(-60, 61)
                for j in range(-60, 61)
            ]
            nodes = [n for n in nodes if 0 <= n[0] < width and 0 <= n[1] < height]
            if nodes:
                axes.plot(*np.array(nodes).T, "+", ms=6, color=SEMANTIC_WARNING_COLOR)
        groups: Dict[str, list] = {"clear": [], "faint": [], "extended": []}
        for sq in fit["squares"]:
            kind = (
                "extended"
                if sq["extrapolated"]
                else "clear"
                if sq["matched"]
                else "faint"
            )
            groups[kind].append(list(sq["corners_px"]) + [sq["corners_px"][0]])
        styles = {
            "clear": (SAVED_POSITION_COLOUR, 1.2, "-"),
            "faint": (DEFECT_ORANGE_COLOR, 0.9, "--"),
            "extended": (TEXT_MUTED_COLOR, 0.9, ":"),
        }
        for kind, lines in groups.items():
            colour, width_, style = styles[kind]
            if not accepted:  # not accepted: one warning colour, dashed
                colour, style = (
                    SEMANTIC_WARNING_COLOR,
                    "--" if kind != "extended" else ":",
                )
            axes.add_collection(
                LineCollection(
                    lines, colors=colour, linewidths=width_, linestyles=style
                )
            )
        for sq in measured["squares"]:
            axes.text(*sq["px"], sq["id"], ha="center", va="center", **_LABEL)
        for p in measured["points"]:
            if p.get("bar_distance_px") is None:
                continue
            colour = (
                DEFECT_ORANGE_COLOR if p.get("on_bar") else SELECTED_POSITION_COLOUR
            )
            axes.plot(*p["px"], "o", ms=9, mfc="none", mec=colour, mew=1.6)
            axes.annotate(
                _distance_label(p),
                p["px"],
                xytext=(7, 7),
                textcoords="offset points",
                **dict(_LABEL, color=colour),
            )
        steps = answer["lattice"] if accepted else measured
        a, b = steps["a"], steps["b"]
        title = f"pitch {a['pitch_px']:.0f} / {b['pitch_px']:.0f} px"
        if a["bar_width_px"] is not None:
            title += f", bars {a['bar_width_px']:.0f} / {b['bar_width_px']:.0f} px"
        if "score" in measured["quality"]:
            title += (
                f", score {measured['quality']['score']:.2f}; "
                f"{measured['squares_matched']} of {measured['squares_fitted']} "
                f"fitted squares have match {MATCH}+" + (" (solid)" if accepted else "")
            )
        if not accepted:
            title = f"NOT ACCEPTED, best candidate drawn: {title}"
    figure.text(
        0.005,
        1 - 0.5 * header / (shown.shape[0] + header),
        title,
        fontsize=9,
        color=TEXT_STRONG_COLOR,
        va="center",
    )
    return rgb_of(figure)


_LABEL = {
    "fontsize": 8,
    "color": TEXT_STRONG_COLOR,
    "bbox": {"fc": PANEL_COLOR, "ec": "none", "alpha": 0.75, "pad": 1.5},
}


def _distance_label(point: Dict[str, Any]) -> str:
    """A point's distance to the nearest bar edge, in um when the scale is
    known."""
    if point.get("bar_distance_m") is not None:
        return f"{point['bar_distance_m'] * 1e6:.0f} um"
    if point.get("bar_distance_px") is not None:
        return f"{point['bar_distance_px']:.0f} px"
    return ""
