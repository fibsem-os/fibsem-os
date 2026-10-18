"""The pyto fits the solver replaced, kept only to check the solver against.

These reproduce what ``correlation_v2`` did before ``fibsem.correlation.solver``:
pyto's ``Rigid3D.find_32`` with its default random restarts for an unseeded fit, and
the exact seed plus jittered restarts for a seeded one, both under a fixed RNG. They
go, with ``fibsem/correlation/pyto``, once the solver has matched them on saved runs.
"""

from __future__ import annotations

import contextlib
import os
from typing import Tuple

import numpy as np

from fibsem.correlation.pyto.rigid_3d import Rigid3D

_RNG = 20260909
_UNSEEDED = dict(
    randome=True,
    einit="gl2",
    einit_dist=0.1,
    scale=None,
    randoms=True,
    sinit="gl2",
    ninit=10,
)


def _find(fm: np.ndarray, fib: np.ndarray, **params) -> Tuple[np.ndarray, float, float]:
    state = np.random.get_state()
    np.random.seed(_RNG)
    try:
        with (
            open(os.devnull, "w") as fnull,
            contextlib.redirect_stdout(fnull),
            contextlib.redirect_stderr(fnull),
        ):
            t = Rigid3D.find_32(x=fm.T, y=fib[:, :2].T, **params)
    finally:
        np.random.set_state(state)
    return np.asarray(t.q, dtype=float), float(t.s_scalar), float(t.rmsError)


def unseeded(fm, fib) -> Tuple[np.ndarray, float, float]:
    """``(R, s, rms)`` from pyto's default restarts."""
    return _find(np.asarray(fm, float), np.asarray(fib, float), **_UNSEEDED)


def from_seed(fm, fib, rotation, scale) -> Tuple[np.ndarray, float, float]:
    """``(R, s, rms)``: best of the exact seed and two radii of jittered restarts."""
    fm = np.asarray(fm, float)
    fib = np.asarray(fib, float)
    eulers = Rigid3D.extract_euler(np.asarray(rotation, float), mode="x", ret="one")
    einit = Rigid3D.euler_to_ck(angles=eulers, mode="x")
    runs = [dict(randome=False, einit_dist=0.0, ninit=1)]
    runs += [dict(randome=True, einit_dist=j, ninit=8) for j in (0.05, 0.2)]
    best = None
    for run in runs:
        fit = _find(
            fm, fib, scale=None, einit=einit, randoms=False, sinit=float(scale), **run
        )
        if best is None or fit[2] < best[2]:
            best = fit
    return best
