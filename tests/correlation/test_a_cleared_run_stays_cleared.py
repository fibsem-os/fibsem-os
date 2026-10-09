"""A run in flight doesn't bring back a result the user has cleared (FIB-1242).

``_run`` hands a snapshot to a worker thread and leaves the widget usable. When
the user then loads another FIB image and agrees to clear the points, the run
is still going, and its result used to be adopted when it arrived: drawn on
the new image, filled into the Results tab and auto-saved beside zero inputs.
Clearing or loading a result now bumps a generation number, and a run started
under an older one delivers nothing.

Real worker thread, real fit, offscreen.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from datetime import datetime

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QMessageBox

from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
    CorrelationState,
    PointType,
    PointXYZ,
)
from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)
from fibsem.structures import FibsemImage


@pytest.fixture(autouse=True)
def _no_lut_download(monkeypatch):
    import fibsem.ui.correlation.widgets.refractive_index_widget as riw

    monkeypatch.setattr(riw, "_ensure_lut", lambda *a, **k: None, raising=False)


def _fm_image():
    meta = FluorescenceImageMetadata(
        acquisition_date=datetime(2026, 7, 24).isoformat(),
        pixel_size_x=100e-9,
        pixel_size_y=100e-9,
        pixel_size_z=300e-9,
        resolution=(200, 200),
        channels=[
            FluorescenceChannelMetadata(
                name="GFP",
                excitation_wavelength=488.0,
                emission_wavelength=520.0,
                power=0.3,
                exposure_time=0.05,
                gain=1.5,
                offset=50.0,
            )
        ],
    )
    return FluorescenceImage(data=np.zeros((1, 20, 200, 200), np.uint16), metadata=meta)


def _pairs():
    """FM points and their FIB images under a rotation and scale, with click noise."""
    fm = np.array(
        [[40, 50, 4], [150, 40, 9], [160, 150, 14], [50, 160, 6], [100, 100, 11]],
        float,
    )
    a = np.deg2rad(20)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    fib = (rot @ fm[:, :2].T).T * 1.5 + [60.0, 30.0]
    fib += np.random.default_rng(0).normal(0, 0.5, fib.shape)
    return (
        [Coordinate(PointXYZ(x, y, 0.0), PointType.FIB) for x, y in fib],
        [Coordinate(PointXYZ(x, y, z), PointType.FM) for x, y, z in fm],
    )


@pytest.fixture
def widget(qapp, tmp_path):
    from fibsem.ui.correlation.widgets.correlation_tab_widget import (
        CorrelationTabWidget,
    )

    w = CorrelationTabWidget()
    w.set_project_dir(str(tmp_path / "run"))
    w.set_fib_image(FibsemImage.generate_blank_image(resolution=(400, 300)))
    w.set_fm_image(_fm_image())
    fib, fm = _pairs()
    w.set_data(
        CorrelationInputData(
            fib_coordinates=fib,
            fm_coordinates=fm,
            poi_coordinates=[Coordinate(PointXYZ(90, 90, 8), PointType.POI)],
        )
    )
    assert w._can_run()
    yield w
    if w._worker is not None:
        w._worker.wait()
    w.close()


def _deliver(widget, qapp):
    """Let the worker finish and its queued result reach the GUI thread."""
    widget._worker.wait()
    qapp.processEvents()
    qapp.processEvents()


def _overlay_groups(widget):
    return widget._fib_canvas.picking.results._groups


def test_a_run_finishing_after_the_points_are_cleared_is_dropped(
    widget, qapp, tmp_path, monkeypatch
):
    other = tmp_path / "other_ib.tif"
    FibsemImage.generate_blank_image(resolution=(400, 300)).save(str(other))
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Ok)

    widget._run()
    widget._images_tab._load_fib(str(other))  # another image; clear the points
    assert widget._current_positions() == []
    _deliver(widget, qapp)

    assert widget._result is None
    assert not widget._btn_continue.isEnabled()
    assert widget._results_tab._lbl_agree.text() == "—"
    assert _overlay_groups(widget) == []
    saved = tmp_path / "run" / "correlation.json"
    assert not saved.exists() or CorrelationState.load(str(saved)).result is None


def test_an_undisturbed_run_still_lands(widget, qapp, tmp_path):
    widget._run()
    _deliver(widget, qapp)

    assert widget._result is not None, widget._lbl_status.text()
    assert widget._btn_continue.isEnabled()
    assert _overlay_groups(widget) != []
    saved = CorrelationState.load(str(tmp_path / "run" / "correlation.json"))
    assert saved.result is not None


def test_a_run_finishing_after_a_file_is_loaded_keeps_the_loaded_result(
    widget, qapp, tmp_path
):
    widget._run()
    _deliver(widget, qapp)
    saved = tmp_path / "saved.json"
    CorrelationState(input_data=widget.data, result=widget._result).save(str(saved))
    loaded_rms = widget._result.rms_error

    widget.data.fm_coordinates[0].point.x += 5.0  # a different answer
    widget._run()
    widget.load_correlation(str(saved))  # the user opens the saved run
    _deliver(widget, qapp)

    assert widget._result.rms_error == pytest.approx(loaded_rms)
    assert widget._btn_continue.isEnabled()
