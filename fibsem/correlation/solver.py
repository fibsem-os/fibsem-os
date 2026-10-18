"""The rigid FM->FIB fit: a 3D rotation, a scale and a translation from fiducial pairs.

FM fiducials are 3D (x, y and depth, in isotropic FM pixels); their FIB picks are 2D.
The model is ``fib_xy = s * R[:2] @ fm_xyz + t``: a proper rotation ``R``, a scalar
scale ``s`` and an in-plane translation ``t``. Only the first two rows of ``R`` are
seen, so a fit and its mirror (depth axis reversed) are told apart by the fiducials'
depths alone; callers that care check both branches.

Each local solve is a Levenberg-Marquardt-style least squares over four numbers (a
rotation vector about the start, and the log of the scale); the translation is solved
in closed form by fitting about the centroids. Starts are either a seed (the geometry's
prediction, a prior run, or one branch of an earlier fit) or a fixed set covering every
rotation. Nothing is random, so the same picks always give the same answer.

Numpy and scipy only. No Qt.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# A start: a proper 3x3 rotation and a scale (FIB px per FM px).
Start = Tuple[np.ndarray, float]

# The fewest pairs that determine a fit: the centroids take the translation, leaving
# two offset vectors (four equations) for the four unknowns of rotation and scale.
# The floor for a run the user is shown is higher (config.MIN_FIDUCIAL_PAIRS).
MIN_PAIRS = 3

# Small fixed turns tried about a seed as well as the seed itself, so a seed a few
# tens of degrees out still reaches its basin. Degrees, about each axis both ways.
_SEED_NUDGE_DEG = 15.0


@dataclass(frozen=True)
class RigidFit:
    """``fib_xy ≈ scale * rotation[:2] @ fm_xyz + translation[:2]``, and how well it held."""

    rotation: np.ndarray  # (3, 3), proper
    scale: float
    translation: np.ndarray  # (3,), z is 0: the fit says nothing about FIB depth
    residuals: np.ndarray  # (N, 2), projected minus picked, FIB px

    @property
    def rms(self) -> float:
        """Root of the mean squared 2D distance between projections and picks, px."""
        n = len(self.residuals)
        return float(np.sqrt(np.square(self.residuals).sum() / n)) if n else 0.0

    def project(self, fm_xyz) -> np.ndarray:
        """(N, 3) FM points through the fit: (N, 3), the third column the depth."""
        pts = np.asarray(fm_xyz, dtype=float).reshape(-1, 3)
        return self.scale * pts @ self.rotation.T + self.translation

    def translation_about(self, center) -> np.ndarray:
        """The translation that goes with rotating about *center* instead of the origin."""
        c = np.asarray(center, dtype=float).reshape(3)
        return self.project(c)[0] - self.scale * c

    def eulers_deg(self) -> np.ndarray:
        """The rotation as x-convention Euler angles, degrees (see :func:`eulers_deg`)."""
        return eulers_deg(self.rotation)


# ── Euler angles ──────────────────────────────────────────────────────────


def eulers_deg(rotation) -> np.ndarray:
    """x-convention Euler angles ``(phi, theta, psi)`` in degrees, theta in [0, 180].

    The convention saved runs record: ``R = Rz(psi) @ Rx(theta) @ Rz(phi)``.
    """
    return Rotation.from_matrix(np.asarray(rotation, dtype=float)).as_euler(
        "zxz", degrees=True
    )


def rotation_from_eulers_deg(angles) -> np.ndarray:
    """The 3x3 rotation for x-convention Euler angles in degrees (inverse of the above)."""
    return Rotation.from_euler(
        "zxz", np.asarray(angles, dtype=float), degrees=True
    ).as_matrix()


# ── fitting ───────────────────────────────────────────────────────────────


def _pairs(fm_xyz, fib_xy) -> Tuple[np.ndarray, np.ndarray]:
    fm = np.asarray(fm_xyz, dtype=float).reshape(-1, 3)
    fib = np.asarray(fib_xy, dtype=float)
    fib = fib.reshape(len(fib), -1)[:, :2]
    if len(fm) != len(fib):
        raise ValueError(f"{len(fm)} FM points but {len(fib)} FIB points")
    if len(fm) < MIN_PAIRS:
        raise ValueError(
            f"A rigid fit needs at least {MIN_PAIRS} point pairs, got {len(fm)}"
        )
    return fm, fib


def _spread_scale(x: np.ndarray, y: np.ndarray) -> float:
    """A scale from the spread of the points alone: positive whatever the rotation."""
    sx = float(np.square(x).sum())
    sy = float(np.square(y).sum())
    return float(np.sqrt(sy / sx)) if sx > 0 and sy > 0 else 1.0


def _scale_for(rotation: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """The best scale for a fixed rotation, or the spread's if that one is not positive."""
    p = x @ rotation[:2].T
    denom = float(np.square(p).sum())
    s = float((p * y).sum()) / denom if denom > 0 else 0.0
    return s if np.isfinite(s) and s > 0 else _spread_scale(x, y)


def _local_fit(x: np.ndarray, y: np.ndarray, rotation: np.ndarray, scale: float):
    """One least-squares solve from a start, on centred points: ``(R, s, cost)``."""
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError(f"A seed rotation must be 3x3, got shape {rotation.shape}")
    scale = float(scale) if scale and scale > 0 else _scale_for(rotation, x, y)

    def unpack(p):
        r = Rotation.from_rotvec(p[:3]).as_matrix() @ rotation
        return r, scale * np.exp(p[3])

    def residuals(p):
        r, s = unpack(p)
        return (s * x @ r[:2].T - y).ravel()

    res = least_squares(residuals, np.zeros(4), method="lm", xtol=1e-12, ftol=1e-12)
    r, s = unpack(res.x)
    return r, float(s), float(np.square(res.fun).sum())


def fit_rigid(fm_xyz, fib_xy, starts: Iterable[Start]) -> RigidFit:
    """The best local solve over *starts*. A start's scale <= 0 means "work it out".

    Ties go to the earlier start, so put the one you trust first.
    """
    fm, fib = _pairs(fm_xyz, fib_xy)
    fm_c = fm.mean(axis=0)
    fib_c = fib.mean(axis=0)
    x = fm - fm_c
    y = fib - fib_c

    best = None
    for rotation, scale in starts:
        candidate = _local_fit(x, y, rotation, scale)
        if best is None or candidate[2] < best[2] - 1e-12:
            best = candidate
    if best is None:
        raise ValueError("No starts given")

    r, s, _ = best
    t = np.zeros(3)
    t[:2] = fib_c - s * r[:2] @ fm_c
    projected = s * fm @ r.T + t
    return RigidFit(
        rotation=r, scale=s, translation=t, residuals=projected[:, :2] - fib
    )


def seed_starts(rotation, scale: float) -> List[Start]:
    """The seed, then the seed turned a little about each axis both ways."""
    rotation = np.asarray(rotation, dtype=float)
    starts: List[Start] = [(rotation, float(scale))]
    for axis, sign in itertools.product(range(3), (1.0, -1.0)):
        rotvec = np.zeros(3)
        rotvec[axis] = sign * np.radians(_SEED_NUDGE_DEG)
        starts.append(
            (Rotation.from_rotvec(rotvec).as_matrix() @ rotation, float(scale))
        )
    return starts


def fit_from_seed(fm_xyz, fib_xy, rotation, scale: float) -> RigidFit:
    """The fit started at a known rotation and scale. The scale is still fitted."""
    return fit_rigid(fm_xyz, fib_xy, seed_starts(rotation, scale))


def _icosahedral_rotations() -> List[np.ndarray]:
    """The 60 rotations of the icosahedron: an even, fixed cover of every rotation.

    Built from the 120 unit icosians (quaternions), one of each ``±q`` pair.
    """
    phi = (1 + np.sqrt(5)) / 2
    quats = set()
    for i in range(4):
        for sign in (1.0, -1.0):
            q = [0.0] * 4
            q[i] = sign
            quats.add(tuple(q))
    for signs in itertools.product((0.5, -0.5), repeat=4):
        quats.add(signs)
    base = (0.0, 0.5, phi / 2, 1 / (2 * phi))
    even = [p for p in itertools.permutations(range(4)) if _is_even(p)]
    for perm in even:
        for signs in itertools.product((1.0, -1.0), repeat=3):
            values = [
                base[0],
                signs[0] * base[1],
                signs[1] * base[2],
                signs[2] * base[3],
            ]
            quats.add(tuple(values[perm[k]] for k in range(4)))
    rotations = {}
    for q in sorted(quats):
        q = np.asarray(q)
        lead = q[np.flatnonzero(np.abs(q) > 1e-9)[0]]
        key = tuple(np.round(q * np.sign(lead), 9))
        rotations.setdefault(key, Rotation.from_quat(q).as_matrix())
    return list(rotations.values())


def _is_even(perm: Sequence[int]) -> bool:
    inversions = sum(
        1
        for i in range(len(perm))
        for j in range(i + 1, len(perm))
        if perm[i] > perm[j]
    )
    return inversions % 2 == 0


_COVER = _icosahedral_rotations()


def global_starts(fm_xyz, fib_xy) -> List[Start]:
    """Starts for a fit with nothing known: the in-plane similarity, then the cover.

    The similarity of the xy picks gives the in-plane turn and scale exactly when the
    fiducials share a depth and the beams look straight down; the 60 fixed rotations
    of the cover reach every other basin.
    """
    from fibsem.correlation.similarity import fit_similarity

    fm, fib = _pairs(fm_xyz, fib_xy)
    starts: List[Start] = []
    try:
        sim = fit_similarity(fm[:, :2], fib)
        r = Rotation.from_euler("z", sim.rotation, degrees=True).as_matrix()
        starts.append((r, float(sim.scale)))
    except ValueError:
        pass
    starts += [(r, 0.0) for r in _COVER]
    return starts


def fit_unseeded(fm_xyz, fib_xy) -> RigidFit:
    """The best fit over :func:`global_starts`: for when there is no seed."""
    return fit_rigid(fm_xyz, fib_xy, global_starts(fm_xyz, fib_xy))
