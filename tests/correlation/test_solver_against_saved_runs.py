"""The solver against correlation runs saved on disk.

Skipped unless ``FIBSEM_CORRELATION_RUNS`` names a folder. Every run file under it
(``correlation.json`` or either legacy name) with a fitted result is refitted with the
fiducials that result was computed from, unseeded and seeded from the saved rotation.
The solver must fit at least as well as the saved result each time. The same check
runs in CI on synthetic runs written to a temporary folder::

    FIBSEM_CORRELATION_RUNS=/path/to/runs pytest tests/correlation/test_solver_against_saved_runs.py -s
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
import pytest

from fibsem.correlation import solver
from fibsem.correlation.correlation_v2 import (
    _coords_to_array,
    run_correlation_from_data,
)
from fibsem.correlation.geometry import rotation_angle_deg
from fibsem.correlation.history import _RUN_FILES, save_run
from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
    CorrelationState,
    PointType,
    PointXYZ,
    load_correlation_file,
)

from .test_nominal_transform import ENTRIES, _images

_ROOT = os.environ.get("FIBSEM_CORRELATION_RUNS")


def _run_files(root) -> List[Path]:
    if not root:
        return []
    found = []
    for path in sorted(Path(root).rglob("*.json")):
        if path.name in _RUN_FILES:
            found.append(path)
    return found


_FILES = _run_files(_ROOT)


def _fiducials(path: Path):
    """``(fm, fib, result)`` as the saved result was fitted, or a skip."""
    try:
        state = load_correlation_file(str(path))
    except Exception as exc:  # an unreadable file is not the solver's problem
        pytest.skip(f"cannot read {path}: {exc}")
    result = state.result
    if result is None or not result.rotation_matrix:
        pytest.skip("no fitted result saved")
    data = result.input_data or state.input_data
    fm = _coords_to_array(data.fm_coordinates).astype(float)
    fib = _coords_to_array(data.fib_coordinates).astype(float)
    if len(fm) < 3 or len(fm) != len(fib):
        pytest.skip(f"{len(fm)} FM against {len(fib)} FIB fiducials")
    fm[:, 2] *= result.fm_z_scale or 1.0
    return fm, fib, result


def _check(path: Path) -> None:
    fm, fib, result = _fiducials(path)

    new = solver.fit_unseeded(fm, fib)
    saved_r = np.asarray(result.rotation_matrix, dtype=float)
    seeded = solver.fit_from_seed(fm, fib, saved_r, result.scale)
    print(
        f"\n{path}: n={len(fm)} saved {result.rms_error:.3f} | unseeded "
        f"{new.rms:.3f} | from saved {seeded.rms:.3f} "
        f"({rotation_angle_deg(seeded.rotation, saved_r):.2f} deg from saved)"
    )
    assert new.rms <= result.rms_error + 1e-3
    assert seeded.rms <= result.rms_error + 1e-3


@pytest.mark.skipif(
    not _ROOT, reason="set FIBSEM_CORRELATION_RUNS to a folder of saved runs"
)
@pytest.mark.parametrize(
    "path", _FILES or [None], ids=[str(p) for p in _FILES] or ["no-runs"]
)
def test_solver_fits_saved_runs_at_least_as_well_as_they_were(path):
    if path is None:
        pytest.skip(f"no run files under {_ROOT}")
    _check(path)


def _synthetic_run(folder: Path, seed: int) -> None:
    """A run as the tab saves one, fitted from noisy picks of a known transform."""
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(seed)
    rotation = Rotation.random(random_state=seed).as_matrix()
    fib_image, fm_image = _images(ENTRIES[0])
    md = fm_image.metadata
    fm = rng.uniform(50, 450, (7, 3))
    fm[:, 2] = rng.uniform(5, 40, 7)  # slices; the fit scales them to xy pixels
    fm_iso = fm * [1.0, 1.0, md.pixel_size_z / md.pixel_size_x]
    fib = 1.7 * fm_iso @ rotation[:2].T + [700.0, 500.0] + rng.normal(0, 2.0, (7, 2))
    data = CorrelationInputData(
        fib_image=fib_image,
        fm_image=fm_image,
        fib_coordinates=[
            Coordinate(PointXYZ(x, y, 0.0), PointType.FIB) for x, y in fib
        ],
        fm_coordinates=[Coordinate(PointXYZ(*p), PointType.FM) for p in fm],
    )
    result = run_correlation_from_data(data)
    save_run(str(folder), CorrelationState(input_data=data, result=result))


def test_the_check_runs_on_synthetic_saved_runs(tmp_path):
    for seed in range(3):
        _synthetic_run(tmp_path / "lamella" / "Correlation" / f"run-{seed}", seed)
    files = _run_files(tmp_path)
    assert len(files) == 3
    for path in files:
        _check(path)
