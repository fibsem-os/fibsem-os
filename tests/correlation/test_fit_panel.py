"""The fit panel: a point's fit with its inputs beside it (FIB-1218, FIB-283).

Changing an input fits again at once; Accept applies the fit on screen, Keep
click applies nothing. Real widgets and real fits on synthetic images; offscreen.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QDialog

from fibsem.correlation.structures import Coordinate, PointType, PointXYZ
from fibsem.ui.correlation.widgets.fit_confirmation_dialog import (
    FitStatus,
    PointFitResult,
)
from fibsem.ui.correlation.widgets.fit_panel import FitInputs, FitPanelDialog


@pytest.fixture(autouse=True)
def _no_lut_download(monkeypatch):
    import fibsem.ui.correlation.widgets.refractive_index_widget as riw

    monkeypatch.setattr(riw, "_ensure_lut", lambda *a, **k: None, raising=False)


def _coord(x=40.0, y=40.0, z=0.0, pt=PointType.FIB):
    return Coordinate(PointXYZ(x, y, z), pt)


def _result(fitted=(43.0, 44.0), status=FitStatus.OK, message=None):
    initial = PointXYZ(40.0, 40.0, 0.0)
    return PointFitResult(
        coordinate=_coord(),
        method="Hole",
        channel=None,
        initial=initial,
        fitted=PointXYZ(*fitted, 0.0) if fitted else None,
        status=status,
        message=message,
    )


def _burn_image(cx=52.0, cy=40.0):
    """A dark burn 12 px right of the click at (40, 40)."""
    yy, xx = np.indices((80, 80), dtype=float)
    img = 200.0 - 120.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 3.0**2))
    return (img + np.random.default_rng(1).normal(0, 4, img.shape)).astype(np.float32)


# --- the panel ----------------------------------------------------------------


def test_changing_the_window_fits_again_and_accept_takes_the_new_fit(qapp):
    calls = []

    def refit(inputs):
        calls.append(inputs)
        return _result(fitted=(40.0 + inputs.window, 40.0))

    panel = FitPanelDialog(_result(), FitInputs(window=15), refit)
    panel._window_spin.setValue(8)

    assert [c.window for c in calls] == [8]
    assert panel.result.fitted.x == 48.0
    assert panel.inputs.window == 8


def test_a_failed_fit_shows_why_and_cannot_be_accepted(qapp):
    def refit(inputs):
        if inputs.window < 10:
            return _result(None, FitStatus.ERROR, "No clear burn here.")
        return _result()

    panel = FitPanelDialog(_result(), FitInputs(window=15), refit)
    assert panel._accept_btn.isEnabled()

    panel._window_spin.setValue(6)
    assert not panel._accept_btn.isEnabled()
    assert panel._reason.text() == "No clear burn here."
    assert panel._no_figure.isVisibleTo(panel)  # no figure, the space kept
    assert panel._canvas is None

    panel._window_spin.setValue(12)  # back to a fit
    assert panel._accept_btn.isEnabled()
    assert panel._reason.text() == ""


def test_only_the_inputs_a_fit_has_get_controls(qapp):
    fib = FitPanelDialog(_result(), FitInputs(window=15), lambda i: _result())
    assert fib._channel_combo is None and fib._looks_combo is None
    assert fib._fit_xy_check is None

    fid = FitPanelDialog(
        _result(),
        FitInputs(window=2, channel=1, looks="bright"),
        lambda i: _result(),
        channel_names=["reflection", "GFP"],
    )
    assert fid._channel_combo.currentText() == "GFP"
    assert fid._looks_combo.value() == "bright"
    assert fid._fit_xy_check is None

    target = FitPanelDialog(
        _result(),
        FitInputs(window=5, channel=0, fit_xy=False),
        lambda i: _result(),
        channel_names=["reflection", "GFP"],
    )
    assert target._looks_combo is None
    assert target._fit_xy_check.isChecked() is False


def test_each_fm_input_is_passed_to_the_fit(qapp):
    calls = []

    def refit(inputs):
        calls.append(inputs)
        return _result()

    panel = FitPanelDialog(
        _result(),
        FitInputs(window=5, channel=0, looks="dark"),
        refit,
        channel_names=["reflection", "GFP"],
    )
    panel._channel_combo.setCurrentIndex(1)
    panel._looks_combo.set_value("bright")
    assert (calls[-1].channel, calls[-1].looks) == (1, "bright")


# --- in the correlation widget -------------------------------------------------


def _widget(qapp):
    from fibsem.ui.correlation.widgets.correlation_tab_widget import (
        CorrelationTabWidget,
    )

    return CorrelationTabWidget()


def test_the_default_inputs_come_from_the_method_panel(qapp):
    from fibsem.correlation.config import CorrelationConfig, FitSettings

    w = _widget(qapp)
    w.set_correlation_config(
        CorrelationConfig(
            fit=FitSettings(
                fm_fiducial_method="Hole",
                fm_fiducial_polarity="bright",
                reflection_cutout=3,
                fluorescence_cutout=6,
            )
        )
    )
    fib = w._fit_inputs_for(_coord(pt=PointType.FIB))
    assert (fib.window, fib.channel, fib.looks, fib.fit_xy) == (15, None, None, None)
    fid = w._fit_inputs_for(_coord(pt=PointType.FM))
    assert (fid.window, fid.looks, fid.fit_xy) == (3, "bright", None)
    target = w._fit_inputs_for(_coord(pt=PointType.POI))
    # no x, y control until the target's xy fit can find a bright target
    assert (target.window, target.looks, target.fit_xy) == (6, None, None)


def test_the_window_reaches_the_fib_fit(qapp):
    """A burn 12 px from the click: found with a ±15 px window, not with ±6."""
    w = _widget(qapp)
    w._fib_image = SimpleNamespace(filtered_data=_burn_image())
    coord = _coord()
    wide = w._run_point_fit(coord, FitInputs(window=15))
    narrow = w._run_point_fit(coord, FitInputs(window=6))
    assert wide.status is FitStatus.OK
    assert wide.fitted.x == pytest.approx(52.0, abs=0.5)
    assert narrow.status is FitStatus.ERROR


def test_accept_applies_the_fit_the_panel_ended_on(qapp, monkeypatch):
    """Open the panel on a fit, change the window, accept: the point moves to
    the fit after the change, not the one the panel opened on."""
    import fibsem.ui.correlation.widgets.correlation_tab_widget as ctw

    w = _widget(qapp)
    w._fib_image = SimpleNamespace(filtered_data=_burn_image())
    coord = _coord()
    w._coords_tab.fib_list.add_coordinate(coord)

    def change_then_accept(panel):
        assert panel.result.status is FitStatus.OK  # ±15 found the burn
        panel._window_spin.setValue(6)  # nothing burn-shaped within ±6
        assert panel.result.status is FitStatus.ERROR
        panel._window_spin.setValue(14)
        return QDialog.Accepted

    monkeypatch.setattr(ctw.FitPanelDialog, "exec_", change_then_accept)
    w._on_refit_requested(coord)
    assert (coord.point.x, coord.point.y) == pytest.approx((52.0, 40.0), abs=0.5)
    assert coord.fitted is True


def test_keep_click_leaves_the_point(qapp, monkeypatch):
    import fibsem.ui.correlation.widgets.correlation_tab_widget as ctw

    w = _widget(qapp)
    w._fib_image = SimpleNamespace(filtered_data=_burn_image())
    coord = _coord()
    w._coords_tab.fib_list.add_coordinate(coord)
    monkeypatch.setattr(ctw.FitPanelDialog, "exec_", lambda self: QDialog.Rejected)
    w._on_refit_requested(coord)
    assert (coord.point.x, coord.point.y) == (40.0, 40.0)
    assert coord.fitted is False


def test_a_failed_refit_clears_the_last_fit_from_the_panel(qapp):
    """The previous fit's figure and "after" row must not stay on screen next
    to a failure: they would read as the answer."""
    from fibsem.correlation.fit_diagnostics import FitDiagnostic

    ok = _result()
    ok.diagnostic = FitDiagnostic(
        title="t", roi_xy=np.zeros((10, 10)), input_xy=(5, 5), fitted_xy=(6, 6)
    )
    panel = FitPanelDialog(
        ok,
        FitInputs(window=15),
        lambda i: _result(None, FitStatus.ERROR, "No clear burn here."),
    )
    panel.show()
    old = panel._canvas
    assert old is not None and old.isVisible()
    panel._window_spin.setValue(6)
    assert not old.isVisible()
    shown = [panel._details.itemAt(i).widget() for i in range(panel._details.count())]
    assert len(shown) == 1  # "before" only: no "after", no delta
