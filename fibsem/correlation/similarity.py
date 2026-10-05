"""A 2D similarity from point pairs: the closed-form fit behind "three-point" alignment.

Two images of one flat sample differ by a turn, a uniform scale and an offset once the
known geometry has been taken out -- which is what an aligned overview needs (FIB-1030),
and what three clicked pairs are enough to give. Umeyama's solution is exact and
closed-form: no seed, no iterations, no optimiser to wander, and it takes any number of
pairs from two up.

Numpy only. No Qt, no fibsem imports, so it runs on every CI job and can be tested
against hand-built cases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SimilarityFit:
    """`dst ≈ scale · R(rotation) · src + translation`, and how well it held."""

    scale: float
    rotation: float  # degrees, counter-clockwise in the frame's own handedness
    translation: Tuple[float, float]
    residuals: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rms: float = 0.0

    def apply(self, points) -> np.ndarray:
        """*points* (N, 2) through the fitted map."""
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        theta = math.radians(self.rotation)
        c, s = math.cos(theta), math.sin(theta)
        rotation = np.array([[c, -s], [s, c]])
        return self.scale * points @ rotation.T + np.asarray(self.translation)

    def matrix(self) -> np.ndarray:
        """The map as a 3x3 homogeneous matrix."""
        theta = math.radians(self.rotation)
        c, s = math.cos(theta), math.sin(theta)
        return np.array(
            [
                [self.scale * c, -self.scale * s, self.translation[0]],
                [self.scale * s, self.scale * c, self.translation[1]],
                [0.0, 0.0, 1.0],
            ]
        )


def fit_similarity(
    src, dst, fix_scale: bool = False, scale: Optional[float] = None
) -> SimilarityFit:
    """The similarity taking *src* onto *dst*, least squares (Umeyama, 1991).

    *src* and *dst* are (N, 2) with N >= 2, paired by index. `fix_scale` holds the
    scale at *scale* (1.0 if not given) and fits only the turn and the offset -- what
    three noisy clicks want when the pixel sizes are trusted. A reflection is never
    fitted: the geometry supplies mirrors, and a fit that could flip an image would hide
    a wrong camera transform behind a good-looking residual.

    Raises ValueError with fewer than two pairs, or when the pairs do not determine a
    map (all source points coincident).
    """
    src = np.asarray(src, dtype=float).reshape(-1, 2)
    dst = np.asarray(dst, dtype=float).reshape(-1, 2)
    if src.shape != dst.shape:
        raise ValueError(f"{len(src)} source points against {len(dst)} destination")
    n = len(src)
    if n < 2:
        raise ValueError(f"a similarity needs at least two pairs, got {n}")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centred = src - src_mean
    dst_centred = dst - dst_mean
    src_var = float((src_centred**2).sum() / n)
    if src_var <= 0.0:
        raise ValueError("the source points coincide; no map is determined")

    covariance = dst_centred.T @ src_centred / n
    u, d, vt = np.linalg.svd(covariance)
    # Proper rotation only: fold a reflection back rather than accept it.
    sign = np.eye(2)
    if np.linalg.det(u @ vt) < 0:
        sign[1, 1] = -1.0
    rotation = u @ sign @ vt
    if fix_scale:
        s = float(scale) if scale is not None else 1.0
    else:
        s = float(np.trace(np.diag(d) @ sign) / src_var)
    translation = dst_mean - s * rotation @ src_mean
    angle = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))

    fit = SimilarityFit(
        scale=s,
        rotation=angle,
        translation=(float(translation[0]), float(translation[1])),
    )
    mapped = fit.apply(src)
    residuals = np.linalg.norm(mapped - dst, axis=1)
    rms = float(math.sqrt(float((residuals**2).mean()))) if n else 0.0
    return SimilarityFit(
        scale=s,
        rotation=angle,
        translation=fit.translation,
        residuals=residuals,
        rms=rms,
    )


def mirrored_fit(
    src, dst, fix_scale: bool = False, scale: Optional[float] = None
) -> SimilarityFit:
    """The best similarity taking *src*, mirrored left to right, onto *dst*.

    Any mirror will do: a reflection about one line is one about any other and a
    turn, and the fit finds the turn. So this answers whether the pairs are the
    other way round, not about which axis.
    """
    src = np.asarray(src, dtype=float).reshape(-1, 2) * np.array([-1.0, 1.0])
    return fit_similarity(src, dst, fix_scale=fix_scale, scale=scale)


# The plain fit is poor -- its RMS this share of how far the targets spread -- and the
# mirrored one at most this share of it. Both, or three points nearly in a line (which
# fit either way about equally) would say "mirrored" on noise.
MIRROR_POOR_FIT = 0.1
MIRROR_BETTER_BY = 0.5


def looks_mirrored(
    src, dst, fix_scale: bool = False, scale: Optional[float] = None
) -> Optional[Tuple[float, float]]:
    """(RMS, mirrored RMS) when the pairs fit clearly better mirrored, else None.

    For an image whose file cannot say how its camera saw the sample: no turn lines
    up a mirror image, and what the user sees is only a fit that will not come good.
    Three pairs are enough -- a triangle's handedness is what a mirror reverses. Two
    say nothing, by the arithmetic rather than a rule: two points mirrored are two
    points turned, so both fits come out the same.
    """
    src = np.asarray(src, dtype=float).reshape(-1, 2)
    dst = np.asarray(dst, dtype=float).reshape(-1, 2)
    if len(src) < 2 or src.shape != dst.shape:
        return None
    spread = float(np.sqrt(((dst - dst.mean(axis=0)) ** 2).sum(axis=1).mean()))
    if spread <= 0.0:
        return None
    try:
        plain = fit_similarity(src, dst, fix_scale=fix_scale, scale=scale)
        mirrored = mirrored_fit(src, dst, fix_scale=fix_scale, scale=scale)
    except ValueError:
        return None
    if (
        plain.rms > MIRROR_POOR_FIT * spread
        and mirrored.rms < MIRROR_BETTER_BY * plain.rms
    ):
        return plain.rms, mirrored.rms
    return None
