"""A failed correlation run says why on the status line.

``_on_run_error`` wrote "Error: ..." and then refreshed the run button, whose
readiness text ("Ready to run with 5 pairs.") replaced it at once: the user saw
the run stop and never saw the reason. The error now stays until the next data
change or run.

Real widget, offscreen.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from datetime import datetime

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
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


def _data():
    """FM points and their FIB images under a rotation and scale, with click noise."""
    fm = np.array(
        [[40, 50, 4], [150, 40, 9], [160, 150, 14], [50, 160, 6], [100, 100, 11]],
        float,
    )
    a = np.deg2rad(20)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    fib = (rot @ fm[:, :2].T).T * 1.5 + [60.0, 30.0]
    fib += np.random.default_rng(0).normal(0, 0.5, fib.shape)
    return CorrelationInputData(
        fib_coordinates=[
            Coordinate(PointXYZ(x, y, 0.0), PointType.FIB) for x, y in fib
        ],
        fm_coordinates=[Coordinate(PointXYZ(x, y, z), PointType.FM) for x, y, z in fm],
        poi_coordinates=[Coordinate(PointXYZ(90, 90, 8), PointType.POI)],
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
    w.set_data(_data())
    assert w._can_run()
    yield w
    if w._worker is not None:
        w._worker.wait()
    w.close()


def test_the_error_stays_on_the_status_line(widget):
    widget._on_run_error("boom")

    assert widget._lbl_status.text() == "Error: boom"
    # the inputs are still runnable, so the user can try again
    assert widget._btn_run.isEnabled()


def test_a_run_that_fails_on_its_worker_says_why(widget, qapp, monkeypatch):
    import fibsem.ui.correlation.widgets.correlation_tab_widget as ctw

    def fail(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(ctw, "run_correlation_from_data", fail)

    widget._run()
    widget._worker.wait()
    qapp.processEvents()
    qapp.processEvents()

    assert widget._lbl_status.text() == "Error: boom"


def test_the_next_data_change_replaces_the_error(widget):
    widget._on_run_error("boom")

    widget._on_point_edited()  # the user moved a point

    assert "Error" not in widget._lbl_status.text()


def test_the_next_run_replaces_the_error(widget, qapp):
    widget._on_run_error("boom")

    widget._run()
    assert widget._lbl_status.text() == "Running…"
    widget._worker.wait()
    qapp.processEvents()
    qapp.processEvents()

    assert "Error" not in widget._lbl_status.text()
