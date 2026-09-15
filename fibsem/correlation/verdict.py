"""The verdict on a correlation: leave-one-out, jackknife, rejection (FIB-956).

Not to be confused with :mod:`fibsem.correlation.fit_diagnostics`, which is
the per-point local fitter's diagnostic (the panel behind the confirm dialog).

After a run the user needs one answer -- good, check, or poor -- with a reason
they can act on. Everything here is computed from the placed pairs and the
seeded solver, with no model of the errors:

* **Leave-one-out error** per pair: refit from the other pairs and measure
  where this pair's FM point lands in the FIB image against the user's pick.
  The in-fit residual is optimistic because the pair pulled the fit; this is
  the pair's honest error. Blind to an error every pair shares (a wrong flip
  or pixel size), which is what the mirror and scale checks are for.
* **Jackknife on the target**: the POI projected through each leave-one-out
  fit; the spread says how much the target depends on any single pair, in the
  units the user cares about.
* **Mirror ratio**: the RMS of the fit seeded from the mirrored prior over the
  chosen fit's. Near 1 means the pairs' depths cannot say which way is deeper
  into the sample, and the depth correction could go the wrong way (FIB-880).
* **Depth span**, **scale against the pixel-size ratio**, and the POI's
  distance outside the fiducials' hull.

Calibrated on eight saved METEOR runs (the good population: RMS 0.17-0.39 um,
leave-one-out median 0.17-0.42 um, mirror ratio 1.45-4.5) and two Arctis
runs known to be bad (RMS 2-5 um, mirror ratio 1.10-1.21). Thresholds sit
outside the good population so the verdict does not fire on healthy runs;
a verdict that cries wolf is one that gets ignored.

Rejection is automatic and reversible; suggestions are never applied. The
pairs are the only independent evidence and depth is their weakest axis, so
the model may say where a pair *would* fit the others and the user confirms
by looking. Pure functions, no Qt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from fibsem.correlation.geometry import NominalTransform

__all__ = [
    "FitDiagnostics",
    "PairDiagnostic",
    "Reason",
    "Verdict",
    "diagnose",
    "reject_outliers",
    "verdict",
]

# ── thresholds (calibrated 2026-09-14, see module docstring) ──────────────
MIRROR_RATIO_AMBIGUOUS = 1.3  # below: the depth direction is not determined
LOO_BAD_UM = 2.0  # a pair this far from the others is wrong
LOO_CHECK_UM = 1.0  # worth a look
LOO_CHECK_RATIO = 3.0  # ... or this many times the median
MIN_PAIRS = 4
HULL_CHECK_UM = 5.0  # target this far outside the fiducials
JACKKNIFE_CHECK_UM = 0.5
SCALE_CHECK = 0.03  # fitted scale vs the pixel-size ratio
DEPTH_SPAN_COPLANAR_UM = 1.0
MIN_PLACED = 4  # fewer placed by the user than this is a check
Z_SCAN_SLICES = 8  # how far the suggested-z search looks each way


@dataclass(frozen=True)
class PairDiagnostic:
    index: int
    residual_um: float  # in the full fit
    loo_error_um: float  # refit without this pair


@dataclass(frozen=True)
class FitDiagnostics:
    rms_um: float
    pairs: List[PairDiagnostic]
    mirror_ratio: float
    depth_span_um: float
    scale_ratio: float  # fitted scale / (fm pixel / fib pixel)
    n_pairs: int
    n_accepted: int = 0
    poi_jackknife_um: Optional[float] = None
    poi_hull_distance_um: Optional[float] = None  # 0 inside the fiducials
    worst: Optional[int] = None  # index of the pair with the largest LOO error
    suggested_z: Optional[float] = None  # slice at which the worst pair fits best
    rotation: Optional[np.ndarray] = field(default=None, compare=False)

    @property
    def loo_median_um(self) -> float:
        return (
            float(np.median([p.loo_error_um for p in self.pairs]))
            if self.pairs
            else 0.0
        )

    @property
    def loo_max_um(self) -> float:
        return max((p.loo_error_um for p in self.pairs), default=0.0)

    @property
    def agreement_um(self) -> float:
        """The headline: how well the fiducials agree (RMS or the target's spread)."""
        return max(self.rms_um, self.poi_jackknife_um or 0.0)

    def to_dict(self) -> dict:
        return {
            "rms_um": self.rms_um,
            "pairs": [
                {
                    "index": p.index,
                    "residual_um": p.residual_um,
                    "loo_error_um": p.loo_error_um,
                }
                for p in self.pairs
            ],
            "mirror_ratio": self.mirror_ratio,
            "depth_span_um": self.depth_span_um,
            "scale_ratio": self.scale_ratio,
            "n_pairs": self.n_pairs,
            "n_accepted": self.n_accepted,
            "poi_jackknife_um": self.poi_jackknife_um,
            "poi_hull_distance_um": self.poi_hull_distance_um,
            "worst": self.worst,
            "suggested_z": self.suggested_z,
        }


@dataclass(frozen=True)
class Reason:
    text: str
    pair: Optional[int] = None  # index into the pairs, so the UI can select it


@dataclass(frozen=True)
class Verdict:
    tier: str  # "good" | "check" | "bad"
    headline: str
    reasons: List[Reason]

    @property
    def text(self) -> str:
        return " ".join([self.headline] + [r.text for r in self.reasons])


# ── the solver, in one place ──────────────────────────────────────────────


def _fit(fm_iso: np.ndarray, fib_xy: np.ndarray, prior: NominalTransform):
    """``(R, s, t, rms_px)``: the seeded fit and its centroid translation."""
    from fibsem.correlation.correlation_v2 import _fit_from_seed

    fib_xyz = np.column_stack([fib_xy, np.zeros(len(fib_xy))])
    R, s, rms = _fit_from_seed(fm_iso, fib_xyz, prior.eulers_deg(), prior.scale)
    t = fib_xy.mean(axis=0) - (s * R[:2, :] @ fm_iso.mean(axis=0))
    return R, s, t, rms


def _project(R, s, t, fm_iso: np.ndarray) -> np.ndarray:
    return (s * R[:2, :] @ np.atleast_2d(fm_iso).T).T + t


# ── diagnostics ───────────────────────────────────────────────────────────


def diagnose(
    fib_xy: Sequence[Sequence[float]],
    fm_xyz: Sequence[Sequence[float]],
    prior: NominalTransform,
    *,
    fib_pixel_size: float,
    fm_pixel_size: float,
    fm_pixel_size_z: float,
    poi: Optional[Sequence[float]] = None,
    accepted: Optional[Sequence[bool]] = None,
) -> FitDiagnostics:
    """Everything the verdict needs, from the pairs and the prior.

    ``fib_xy`` in FIB pixels, ``fm_xyz`` in FM pixels with z in *slices*;
    ``poi`` likewise. ``accepted`` marks pairs whose FM position was accepted
    from the projection unmoved (counted, not weighted). n + 2 seeded solves.
    """
    fib = np.asarray(fib_xy, dtype=float).reshape(-1, 2)
    fm_raw = np.asarray(fm_xyz, dtype=float).reshape(-1, 3)
    n = len(fib)
    if n != len(fm_raw) or n < 3:
        raise ValueError(f"need at least three pairs, got {n} FIB and {len(fm_raw)} FM")
    zan = fm_pixel_size_z / fm_pixel_size
    fm_iso = fm_raw * [1.0, 1.0, zan]
    um = fib_pixel_size * 1e6

    R, s, t, rms_px = _fit(fm_iso, fib, prior)
    residuals = np.linalg.norm(_project(R, s, t, fm_iso) - fib, axis=1) * um
    _, _, _, rms_mirror = _fit(fm_iso, fib, prior.mirrored())
    mirror_ratio = float(rms_mirror / rms_px) if rms_px > 0 else float("inf")

    poi_iso = (
        np.asarray(poi, dtype=float) * [1.0, 1.0, zan] if poi is not None else None
    )
    poi_full = _project(R, s, t, poi_iso)[0] if poi_iso is not None else None

    loo = np.zeros(n)
    poi_jk: List[np.ndarray] = []
    for i in range(n):
        keep = np.arange(n) != i
        Ri, si, ti, _ = _fit(fm_iso[keep], fib[keep], prior)
        loo[i] = np.linalg.norm(_project(Ri, si, ti, fm_iso[i])[0] - fib[i]) * um
        if poi_iso is not None:
            poi_jk.append(_project(Ri, si, ti, poi_iso)[0])

    worst = int(np.argmax(loo)) if n else None
    suggested_z = None
    if worst is not None and n >= 4 and _flagged(loo, worst):
        suggested_z = _best_z_for(worst, fib, fm_raw, zan, prior, um)

    jackknife = hull = None
    if poi_full is not None:
        jackknife = float(max(np.linalg.norm(np.array(poi_jk) - poi_full, axis=1))) * um
        hull = _hull_distance(fib, poi_full) * um

    expected_scale = fm_pixel_size / fib_pixel_size
    return FitDiagnostics(
        rms_um=float(rms_px * um),
        pairs=[PairDiagnostic(i, float(residuals[i]), float(loo[i])) for i in range(n)],
        mirror_ratio=mirror_ratio,
        depth_span_um=float(np.ptp(fm_raw[:, 2]) * fm_pixel_size_z * 1e6),
        scale_ratio=float(s / expected_scale) if expected_scale else float("nan"),
        n_pairs=n,
        n_accepted=int(sum(bool(a) for a in accepted)) if accepted is not None else 0,
        poi_jackknife_um=jackknife,
        poi_hull_distance_um=hull,
        worst=worst,
        suggested_z=suggested_z,
        rotation=R,
    )


def _flagged(loo: np.ndarray, i: int) -> bool:
    """Whether pair ``i``'s leave-one-out error is worth the user's attention:
    over the absolute threshold, or well above the median of all pairs (the
    median including this pair, so a small set is not flagged by one modest
    outlier: the good population's worst pairs reach 0.85 um on six pairs)."""
    return bool(
        loo[i] > LOO_CHECK_UM
        or (loo[i] > LOO_CHECK_RATIO * np.median(loo) and loo[i] > 0.5)
    )


def _best_z_for(i, fib, fm_raw, zan, prior, um) -> Optional[float]:
    """The slice at which pair ``i`` best agrees with the others, or None if
    no slice within the scan does better than the pick's own."""
    keep = np.arange(len(fib)) != i
    fm_iso = fm_raw * [1.0, 1.0, zan]
    Ri, si, ti, _ = _fit(fm_iso[keep], fib[keep], prior)
    z0 = fm_raw[i, 2]
    best_z, best_err = z0, np.inf
    for dz in range(-Z_SCAN_SLICES, Z_SCAN_SLICES + 1):
        q = fm_iso[i].copy()
        q[2] = (z0 + dz) * zan
        err = np.linalg.norm(_project(Ri, si, ti, q)[0] - fib[i]) * um
        if err < best_err - 1e-9:
            best_z, best_err = z0 + dz, err
    return float(best_z) if best_z != z0 else None


def _hull_distance(fib: np.ndarray, point: np.ndarray) -> float:
    """Pixels from ``point`` to the nearest fiducial when it lies outside their
    convex hull; 0 inside. Falls back to the nearest fiducial for degenerate
    (collinear) sets."""
    try:
        from scipy.spatial import Delaunay

        if Delaunay(fib).find_simplex(point[None])[0] >= 0:
            return 0.0
    except Exception as exc:  # collinear or too few points
        logging.debug(f"hull test fell back to nearest fiducial: {exc}")
    return float(np.min(np.linalg.norm(fib - point, axis=1)))


# ── rejection ─────────────────────────────────────────────────────────────


def reject_outliers(
    fib_xy,
    fm_xyz,
    prior: NominalTransform,
    *,
    fib_pixel_size: float,
    fm_pixel_size: float,
    fm_pixel_size_z: float,
    max_um: float = LOO_BAD_UM,
    keep_at_least: int = MIN_PAIRS,
) -> List[int]:
    """Indices to leave out: the worst pair while it is plainly wrong.

    Drop the pair with the largest leave-one-out error while it exceeds
    ``max_um``, refit, repeat; never below ``keep_at_least`` pairs. Only the
    Poor-tier threshold rejects automatically: a pair that is merely worth a
    look (the Check tier) is flagged for the user, not removed, so the verdict
    never says "have a look" about a pair it has already taken out.
    Reversible by the caller: the pairs stay on screen with status
    ``rejected``.
    """
    fib = np.asarray(fib_xy, dtype=float).reshape(-1, 2)
    fm = np.asarray(fm_xyz, dtype=float).reshape(-1, 3)
    alive = list(range(len(fib)))
    rejected: List[int] = []
    while len(alive) > keep_at_least:
        d = diagnose(
            fib[alive],
            fm[alive],
            prior,
            fib_pixel_size=fib_pixel_size,
            fm_pixel_size=fm_pixel_size,
            fm_pixel_size_z=fm_pixel_size_z,
        )
        errors = np.array([p.loo_error_um for p in d.pairs])
        worst = int(np.argmax(errors))
        if errors[worst] > max_um:
            rejected.append(alive.pop(worst))
        else:
            break
    return sorted(rejected)


# ── the verdict ───────────────────────────────────────────────────────────


def _fm(i: int) -> str:
    return f"FM {i + 1}"


def verdict(d: FitDiagnostics, rejected: Sequence[int] = ()) -> Verdict:
    """Good / check / poor with at most two reasons, in the sample's terms.

    No fit vocabulary in the text: it says *fiducials*, what is wrong in
    terms of the sample (deeper, off, from the nearest), and what to do.
    ``rejected`` are pair indices already left out, so the line can say so.
    """
    bad: List[Reason] = []
    check: List[Reason] = []

    if d.n_pairs < MIN_PAIRS:
        bad.append(Reason("Not enough fiducials; place more."))
    if d.mirror_ratio < MIRROR_RATIO_AMBIGUOUS:
        if d.depth_span_um < DEPTH_SPAN_COPLANAR_UM:
            bad.append(
                Reason(
                    "The fit is ambiguous: the fiducials are all at about the same "
                    "depth, so they cannot say which way is deeper into the sample."
                )
            )
        else:
            bad.append(
                Reason(
                    "The fit is ambiguous: the fiducials' depths do not agree with "
                    "each other, so they cannot say which way is deeper into the "
                    "sample. Check the z of each fiducial."
                )
            )
    if d.worst is not None:
        worst = d.pairs[d.worst]
        if worst.loo_error_um > LOO_BAD_UM:
            removed = " and was removed" if d.worst in rejected else ""
            bad.append(
                Reason(
                    f"{_fm(d.worst)} is {worst.loo_error_um:.0f} µm off{removed}.",
                    pair=d.worst,
                )
            )
        elif _flagged(np.array([p.loo_error_um for p in d.pairs]), d.worst):
            z_hint = ""
            if d.suggested_z is not None:
                z_hint = f" The burn may be at slice {d.suggested_z:.0f}."
            check.append(
                Reason(
                    f"{_fm(d.worst)} does not agree with the other fiducials "
                    f"({worst.loo_error_um:.1f} µm off).{z_hint} Have a look, or remove it.",
                    pair=d.worst,
                )
            )
    if d.poi_hull_distance_um is not None and d.poi_hull_distance_um > HULL_CHECK_UM:
        check.append(
            Reason(
                f"The target is {d.poi_hull_distance_um:.0f} µm from the nearest "
                "fiducial, so the fit is less reliable there. Add a fiducial near "
                "the target if you can."
            )
        )
    if d.poi_jackknife_um is not None and d.poi_jackknife_um > JACKKNIFE_CHECK_UM:
        check.append(
            Reason(
                f"The target moves {d.poi_jackknife_um:.1f} µm depending on which "
                "fiducial is left out."
            )
        )
    placed = d.n_pairs - d.n_accepted
    if d.n_accepted and placed < MIN_PLACED:
        check.append(
            Reason(
                f"Only {placed} fiducial{'s were' if placed != 1 else ' was'} placed by "
                "you; the rest are predictions you accepted."
            )
        )
    if np.isfinite(d.scale_ratio) and abs(d.scale_ratio - 1.0) > SCALE_CHECK:
        check.append(
            Reason(
                "The images do not scale as their pixel sizes say "
                f"({abs(d.scale_ratio - 1.0) * 100:.0f} % off); check the FM pixel size."
            )
        )
    if (
        d.depth_span_um < DEPTH_SPAN_COPLANAR_UM
        and d.mirror_ratio >= MIRROR_RATIO_AMBIGUOUS
    ):
        check.append(
            Reason(
                "The fiducials are all at about the same depth, so depth relies on "
                "the previous run rather than on them."
            )
        )

    if bad:
        headline = "Poor fit. Do not continue."
        reasons = bad[:2]
        if len(reasons) < 2 and check:
            reasons.append(check[0])
        return Verdict("bad", headline, reasons)
    if check:
        first = check[0]
        if first.pair is not None:
            headline = f"Check {_fm(first.pair)}."
        elif "target" in first.text:
            headline = "Check the target."
        else:
            headline = "Check the fit."
        return Verdict("check", headline, check[:2])
    return Verdict(
        "good",
        "Good fit.",
        [
            Reason(
                f"The fiducials agree to within {d.agreement_um:.1f} µm and "
                "nothing stands out."
            )
        ],
    )
