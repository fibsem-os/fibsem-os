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
from fibsem.ui.correlation.widgets.fit_panel import FitInputs, FitPanelDialog
from fibsem.ui.correlation.widgets.fit_result import (
    FitStatus,
    PointFitResult,
)


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
    assert (target.window, target.looks, target.fit_xy) == (6, None, False)


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


# --- one point after another (FIB-1218 6b) ---------------------------------------


def _three_burns():
    """Three dark burns, each 3 px right of where its point is placed."""
    yy, xx = np.indices((80, 120), dtype=float)
    img = np.full((80, 120), 200.0)
    for cx in (23.0, 53.0, 83.0):
        img -= 120.0 * np.exp(-((xx - cx) ** 2 + (yy - 40.0) ** 2) / (2 * 3.0**2))
    return (img + np.random.default_rng(1).normal(0, 4, img.shape)).astype(np.float32)


def _run_of_points(qapp, monkeypatch, answers):
    """Three FIB points; the panel answers each in turn from ``answers``."""
    import fibsem.ui.correlation.widgets.correlation_tab_widget as ctw

    w = _widget(qapp)
    w._fib_image = SimpleNamespace(filtered_data=_three_burns())
    coords = [_coord(x, 40.0) for x in (20.0, 50.0, 80.0)]
    for c in coords:
        w._coords_tab.fib_list.add_coordinate(c)
    seen = []

    def answer(panel):
        seen.append(
            (panel.result.coordinate, panel._accept_btn.text(), panel._keep_btn.text())
        )
        return answers[len(seen) - 1]

    monkeypatch.setattr(ctw.FitPanelDialog, "exec_", answer)
    return w, coords, seen


def test_accept_moves_on_to_the_next_point_until_the_last(qapp, monkeypatch):
    w, coords, seen = _run_of_points(qapp, monkeypatch, [QDialog.Accepted] * 3)
    w._on_refit_requested(coords[0])
    assert [s[0] for s in seen] == coords
    assert [c.point.x for c in coords] == pytest.approx([23.0, 53.0, 83.0], abs=0.5)
    assert [s[1] for s in seen] == ["Accept, next ›"] * 2 + ["Accept"]


def test_the_run_starts_at_the_point_asked_for(qapp, monkeypatch):
    w, coords, seen = _run_of_points(qapp, monkeypatch, [QDialog.Accepted] * 2)
    w._on_refit_requested(coords[1])
    assert [s[0] for s in seen] == coords[1:]
    assert coords[0].point.x == 20.0  # before the start: untouched


def test_keep_click_moves_on_without_moving_the_point(qapp, monkeypatch):
    from fibsem.ui.correlation.widgets.fit_panel import KEPT

    w, coords, seen = _run_of_points(
        qapp, monkeypatch, [KEPT, QDialog.Accepted, QDialog.Accepted]
    )
    w._on_refit_requested(coords[0])
    assert coords[0].point.x == 20.0 and coords[0].fitted is False
    assert coords[1].point.x == pytest.approx(53.0, abs=0.5)
    assert seen[0][2] == "Keep click, next ›"


def test_closing_the_panel_stops_the_run(qapp, monkeypatch):
    w, coords, seen = _run_of_points(qapp, monkeypatch, [QDialog.Rejected])
    w._on_refit_requested(coords[0])
    assert len(seen) == 1
    assert [c.point.x for c in coords] == [20.0, 50.0, 80.0]


# --- z: planes searched, browsing planes, z by hand (FIB-1218 6c) -----------------


def _hole_stack(cx=30.0, cy=30.0, cz=10, nz=21, size=60):
    """A dark reflection hole at (cx, cy), deepest at plane cz."""
    yy, xx = np.indices((size, size), dtype=float)
    blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 3.0**2))
    vol = np.full((nz, size, size), 200.0, dtype=np.float32)
    for zi in range(nz):
        vol[zi] -= 160.0 * np.exp(-((zi - cz) ** 2) / (2 * 2.0**2)) * blob
    return vol


def test_planes_searched_reach_the_reflection_fit():
    from fibsem.correlation.util import hole_fitting_reflection

    vol = _hole_stack(cz=10)
    *_, d = hole_fitting_reflection(vol, 30, 30, 10, 2)
    assert (d.z_axis[0], d.z_axis[-1]) == (0, 14)  # today's 10 below, 4 above
    *_, d = hole_fitting_reflection(vol, 30, 30, 10, 2, planes=(3, 5))
    assert (d.z_axis[0], d.z_axis[-1]) == (7, 15)


def _fm_panel(result, stack, refit=None):
    return FitPanelDialog(
        result,
        FitInputs(window=2, channel=0, looks="dark", planes=(10, 4)),
        refit or (lambda i: result),
        channel_names=["reflection"],
        stack=lambda ch: stack,
    )


def _fm_result(z=10.0, fitted=True, status=FitStatus.OK):
    initial = PointXYZ(30.0, 30.0, 8.0)
    return PointFitResult(
        coordinate=_coord(30.0, 30.0, 8.0, PointType.FM),
        method="Hole",
        channel=0,
        initial=initial,
        fitted=PointXYZ(30.4, 29.8, z) if fitted else None,
        status=status,
        message=None if fitted else "No clear burn here.",
    )


def test_the_planes_slider_follows_the_fit_and_sets_z_by_hand(qapp):
    panel = _fm_panel(_fm_result(z=10.2), _hole_stack())
    assert panel._plane_slider.value() == 10  # the plane the fit landed on
    assert panel._plane_slider.maximum() == 20
    panel._plane_slider.setValue(13)
    assert panel._use_plane_btn.text() == "Use plane 13 as z"
    panel._use_plane_btn.click()
    r = panel.result
    assert (r.fitted.x, r.fitted.y, r.fitted.z) == (30.4, 29.8, 13.0)  # xy of the fit
    assert r.method.endswith("z by hand")
    assert panel._status.text() == "z set by hand"
    assert panel._accept_btn.isEnabled()


def test_z_by_hand_after_a_failed_fit_keeps_the_click_xy(qapp):
    failed = _fm_result(fitted=False, status=FitStatus.ERROR)
    panel = _fm_panel(failed, _hole_stack())
    assert not panel._accept_btn.isEnabled()
    assert panel._plane_slider.value() == 8  # the clicked plane
    panel._plane_slider.setValue(10)
    panel._use_plane_btn.click()
    assert (panel.result.fitted.x, panel.result.fitted.y) == (30.0, 30.0)
    assert panel.result.fitted.z == 10.0
    assert panel._accept_btn.isEnabled()


def test_a_refit_replaces_z_by_hand(qapp):
    panel = _fm_panel(
        _fm_result(z=10.2), _hole_stack(), refit=lambda i: _fm_result(z=11.0)
    )
    panel._plane_slider.setValue(15)
    panel._use_plane_btn.click()
    panel._window_spin.setValue(3)  # an input change fits again
    assert panel.result.fitted.z == 11.0
    assert not panel.result.method.endswith("z by hand")
    assert panel._plane_slider.value() == 11


def test_fib_fits_have_no_plane_controls(qapp):
    panel = FitPanelDialog(_result(), FitInputs(window=15), lambda i: _result())
    assert panel._plane_slider is None and panel._planes_spins is None


def test_accepting_z_by_hand_moves_the_fm_point(qapp, monkeypatch):
    """End to end in the widget: open the panel on an FM fiducial, take plane 12
    as z, accept."""
    import fibsem.ui.correlation.widgets.correlation_tab_widget as ctw
    from fibsem.correlation.config import CorrelationConfig, FitSettings

    w = _widget(qapp)
    w.set_correlation_config(
        CorrelationConfig(fit=FitSettings(fm_fiducial_method="Hole"))
    )
    from datetime import datetime

    from fibsem.fm.structures import (
        FluorescenceChannelMetadata,
        FluorescenceImage,
        FluorescenceImageMetadata,
    )

    stack = _hole_stack(cz=10)
    meta = FluorescenceImageMetadata(
        acquisition_date=datetime(2026, 7, 24).isoformat(),
        pixel_size_x=100e-9,
        pixel_size_y=100e-9,
        pixel_size_z=300e-9,
        resolution=(60, 60),
        channels=[
            FluorescenceChannelMetadata(
                name="reflection",
                excitation_wavelength=488.0,
                emission_wavelength=None,
                power=0.3,
                exposure_time=0.05,
                gain=1.5,
                offset=50.0,
            )
        ],
    )
    w.set_fm_image(FluorescenceImage(data=stack[None], metadata=meta))
    coord = _coord(30.0, 30.0, 10.0, PointType.FM)
    w._coords_tab.fm_list.add_coordinate(coord)

    def take_plane_12(panel):
        assert panel.result.status is not FitStatus.ERROR
        panel._plane_slider.setValue(12)
        panel._use_plane_btn.click()
        return QDialog.Accepted

    monkeypatch.setattr(ctw.FitPanelDialog, "exec_", take_plane_12)
    w._on_refit_requested(coord)
    assert coord.point.z == 12.0
    assert (coord.point.x, coord.point.y) == pytest.approx((30.0, 30.0), abs=0.5)


# --- Save as default (FIB-1218 6d) -----------------------------------------------


def test_save_as_default_for_fib_fiducials_starts_the_next_fit_there(qapp, monkeypatch):
    """Save ±10 on the first point: the experiment's FIB window is 10, and the
    next point's panel opens on 10."""
    w, coords, seen = _run_of_points(qapp, monkeypatch, [QDialog.Accepted] * 3)
    windows = []
    import fibsem.ui.correlation.widgets.correlation_tab_widget as ctw

    def save_on_first(panel):
        windows.append(panel._window_spin.value())
        if len(windows) == 1:
            panel._window_spin.setValue(10)
            assert panel._save_btn.text() == "Save as default for FIB fiducials"
            panel._save_btn.click()
            assert not panel._save_btn.isEnabled()
        return QDialog.Accepted

    monkeypatch.setattr(ctw.FitPanelDialog, "exec_", save_on_first)
    w._on_refit_requested(coords[0])
    assert windows == [15, 10, 10]
    assert w.correlation_config.fit.fib_window == 10


def test_save_as_default_for_fm_fiducials_sets_the_method_panel(qapp):
    from fibsem.correlation.config import CorrelationConfig, FitSettings

    w = _widget(qapp)
    w.set_correlation_config(
        CorrelationConfig(fit=FitSettings(fm_fiducial_method="Hole"))
    )
    for name in ("reflection", "GFP"):
        w._coords_tab._fm_fid_ch_combo.addItem(name)
        w._coords_tab._fm_poi_ch_combo.addItem(name)
    w._save_fit_defaults(
        _coord(pt=PointType.FM),
        FitInputs(window=3, channel=1, looks="bright", planes=(6, 6)),
    )
    fit = w.correlation_config.fit
    assert (fit.reflection_cutout, fit.reflection_planes) == (3, (6, 6))
    assert (fit.fm_fiducial_channel, fit.fm_fiducial_polarity) == ("GFP", "bright")
    assert w._fit_inputs_for(_coord(pt=PointType.FM)) == FitInputs(
        window=3, window_label="Depth box ±", channel=1, looks="bright", planes=(6, 6)
    )

    w._save_fit_defaults(_coord(pt=PointType.POI), FitInputs(window=8, channel=0))
    fit = w.correlation_config.fit
    assert (fit.fluorescence_cutout, fit.fm_poi_channel) == (8, "reflection")
    assert fit.reflection_cutout == 3  # the fiducials' settings untouched


def test_changing_an_input_after_saving_offers_to_save_again(qapp):
    saved = []
    panel = FitPanelDialog(
        _result(),
        FitInputs(window=15),
        lambda i: _result(),
        save_default=saved.append,
        default_label="FIB fiducials",
    )
    panel._save_btn.click()
    assert saved == [FitInputs(window=15)]
    panel._window_spin.setValue(12)
    assert panel._save_btn.isEnabled()
    assert panel._save_btn.text() == "Save as default for FIB fiducials"


# --- what the Confirm fit dialog's tests covered, now the panel's -----------------


def test_a_wide_diagnostic_keeps_its_aspect(qapp):
    """A z + XY diagnostic renders wide (9 x 4.5), like the FM fits: shown wide,
    not squished square, and capped."""
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

    from fibsem.correlation.fit_diagnostics import FitDiagnostic

    r = _fm_result()
    r.diagnostic = FitDiagnostic(
        title="t",
        roi_xy=np.zeros((10, 10)),
        input_xy=(5.0, 5.0),
        z_axis=np.arange(5),
        z_signal=np.arange(5.0),
        z_fit=np.arange(5.0),
        z_input=2.0,
        z_fitted=2.5,
    )
    panel = FitPanelDialog(r, FitInputs(window=2), lambda i: r)
    canvas = panel.findChildren(FigureCanvasQTAgg)[0]
    assert canvas.minimumWidth() > canvas.minimumHeight()
    assert canvas.minimumWidth() <= 900


def test_the_channel_row_shows_for_fm_fits_only(qapp):
    from PyQt5.QtWidgets import QLabel

    def labels(panel):
        return {lbl.text() for lbl in panel.findChildren(QLabel)}

    fm = _fm_result()
    fm.channel_name = "reflection"
    fm_panel = FitPanelDialog(fm, FitInputs(window=2), lambda i: fm)
    assert {"channel", "reflection"} <= labels(fm_panel)
    fib_panel = FitPanelDialog(_result(), FitInputs(window=15), lambda i: _result())
    assert "channel" not in labels(fib_panel)


def test_a_failure_shows_a_short_chip_and_a_wrapped_reason(qapp):
    """The chip is a short title; the long message is on its own wrapped label,
    so the panel wraps it instead of stretching."""
    reason = (
        "The fit didn't converge — the feature may be too faint, or not centred "
        "in the search region."
    )
    failed = _result(None, FitStatus.ERROR, reason)
    panel = FitPanelDialog(failed, FitInputs(window=15), lambda i: failed)
    assert panel._status.text() == "Fit failed"
    assert panel._reason.text() == reason and panel._reason.wordWrap()


def test_enter_accepts_and_no_other_button_takes_it(qapp):
    from PyQt5.QtWidgets import QPushButton

    panel = FitPanelDialog(
        _result(),
        FitInputs(window=15),
        lambda i: _result(),
        save_default=lambda i: None,
        position=(0, 3),
    )
    buttons = panel.findChildren(QPushButton)
    assert [b for b in buttons if b.isDefault()] == [panel._accept_btn]
    assert not any(b.autoDefault() for b in buttons if b is not panel._accept_btn)
