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
