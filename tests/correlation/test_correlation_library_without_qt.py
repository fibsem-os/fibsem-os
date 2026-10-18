"""The correlation steps the tab used to own run without Qt: point fits, the
seed and the fit, and writing a run folder."""

import datetime
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import fibsem.imaging.localise as localise
from fibsem.correlation.fit import Seed, fit_transform, seed_for
from fibsem.correlation.history import (
    CORRELATION_FILENAME,
    LamellaCorrelation,
    correlation_dir,
    new_run_dir,
    save_run,
)
from fibsem.correlation.point_fit import FitStatus, refine_point
from fibsem.correlation.structures import (
    Coordinate,
    CorrelationResult,
    CorrelationState,
    PointType,
    PointXYZ,
)


def _coord(x, y, z, kind=PointType.FIB):
    return Coordinate(PointXYZ(x, y, z), kind)


def test_the_library_imports_no_qt():
    code = (
        "import sys, fibsem.correlation.fit, fibsem.correlation.point_fit, "
        "fibsem.correlation.history, fibsem.correlation.refractive_index, "
        "fibsem.imaging.localise, fibsem.fm.interpolation; "
        "assert 'PyQt5' not in sys.modules"
    )
    root = str(Path(__file__).resolve().parents[2])  # this checkout's fibsem
    env = {**os.environ, "PYTHONPATH": root}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_refine_point_proposes_without_moving_the_point(monkeypatch):
    monkeypatch.setattr(
        localise, "hole_fitting_FIB", lambda img, x, y: (x + 3, y + 4, "d")
    )
    coord = _coord(20.0, 20.0, 0.0)
    image = SimpleNamespace(filtered_data=np.zeros((64, 64)))

    result = refine_point(coord, "fib", "Hole", fib_image=image)

    assert (coord.point.x, coord.point.y) == (20.0, 20.0)
    assert result.status is FitStatus.OK
    assert (result.fitted.x, result.fitted.y) == (23.0, 24.0)
    assert result.channel is None and result.diagnostic == "d"


def test_refine_point_uses_the_named_fm_channel_and_cutout(monkeypatch):
    seen = {}

    def gaussian(img, x, y, z, cutout):
        seen.update(shape=img.shape, cutout=cutout)
        return x, y, z + 1, None

    monkeypatch.setattr(localise, "target_fitting_fluorescence", gaussian)
    fm = SimpleNamespace(data=np.zeros((2, 5, 32, 32)))

    result = refine_point(
        _coord(10.0, 10.0, 2.0, PointType.FM),
        "fm",
        "Gaussian",
        fm_image=fm,
        channel=1,
        channel_name="green",
    )

    assert seen == {"shape": (5, 32, 32), "cutout": 5}
    assert result.channel == 1 and result.channel_name == "green"
    assert result.fitted.z == 3.0


def test_refine_point_returns_none_when_nothing_applies():
    coord = _coord(1.0, 1.0, 0.0)
    assert refine_point(coord, "fib", "Hole") is None
    fm = SimpleNamespace(data=np.zeros((1, 2, 8, 8)))
    assert refine_point(coord, "fm", "None", fm_image=fm, channel=0) is None


def test_a_failed_point_fit_is_a_result_not_an_exception(monkeypatch):
    def boom(img, x, y):
        raise RuntimeError("Optimal parameters not found: maxfev = 800")

    monkeypatch.setattr(localise, "hole_fitting_FIB", boom)
    image = SimpleNamespace(filtered_data=np.zeros((8, 8)))

    result = refine_point(_coord(1.0, 1.0, 0.0), "fib", "Hole", fib_image=image)

    assert result.status is FitStatus.ERROR
    assert "didn't converge" in result.message
    assert "maxfev" in result.detail


def test_no_images_no_seed():
    assert seed_for(None, None) == Seed()


def test_an_unseeded_fit_records_no_offset_or_verdict(monkeypatch):
    import fibsem.correlation.fit as fit_module

    result = CorrelationResult()
    monkeypatch.setattr(
        fit_module, "run_correlation_from_data", lambda d, nominal: result
    )

    fitted = fit_transform(SimpleNamespace(fib_coordinates=[]))

    assert fitted is result
    assert fitted.diagnostics is None and fitted.placement_offset is None


def test_a_saved_run_is_in_the_lamella_history(tmp_path):
    lamella = str(tmp_path / "01-lamella")
    run_dir = new_run_dir(lamella, now=datetime.datetime(2026, 10, 16, 12, 0, 0))

    assert run_dir.startswith(correlation_dir(lamella))
    assert not (tmp_path / "01-lamella").exists()  # named, not created

    path = save_run(run_dir, CorrelationState(result=CorrelationResult(rms_error=1.5)))

    assert path.endswith(CORRELATION_FILENAME)
    history = LamellaCorrelation.discover(correlation_dir(lamella))
    assert [r.path for r in history.runs] == [run_dir]
    assert history.latest().state.result.rms_error == 1.5
