"""Regression test: the main-tab widgets construct viewer-less (Phase 7.3/7.4).

After the cutover the AutoLamella main tab and standalone FibsemUI run without a
napari viewer — the quad-view controller is the display. The shared control
widgets (and their sub-widgets, e.g. ObjectiveControlWidget) must therefore build
with ``viewer=None``. This builds them against a real Demo microscope + a
controller, exactly as the hosts do, and would have caught the ObjectiveControlWidget
``hasattr(parent, "viewer")`` crash (the attribute exists but is None).

Run directly (headless):
    QT_QPA_PLATFORM=offscreen python fibsem/ui/widgets/tests/test_viewer_less_widgets.py
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtWidgets

# Create the QApplication before the napari/vispy-heavy fibsem imports below, or
# importing + using those widgets can abort ("QWidget before QApplication").
_QAPP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

from fibsem import utils
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.FibsemMovementWidget import FibsemMovementWidget
from fibsem.ui.widgets.fluorescence_control_widget import FMControlWidget
from fibsem.ui.widgets.canvas.quad_view import MicroscopeViewController

# FM-enabled Demo config so FMControlWidget (and ObjectiveControlWidget) build.
_CONFIG = "fibsem/config/sim-arctis-configuration.yaml"


def _app() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _host(microscope):
    """A parent that mimics a viewer-less host: a controller, viewer=None."""
    parent = QtWidgets.QWidget()
    parent.view_controller = MicroscopeViewController(parent=parent)
    parent.viewer = None
    parent.microscope = microscope
    return parent


def test_image_and_movement_build_viewer_less():
    _app()
    microscope, settings = utils.setup_session(config_path=_CONFIG)
    parent = _host(microscope)
    iw = FibsemImageSettingsWidget(
        microscope=microscope, image_settings=settings.image, parent=parent
    )
    assert iw.viewer is None
    assert iw._view_controller() is parent.view_controller
    parent.image_widget = iw
    mw = FibsemMovementWidget(microscope=microscope, parent=parent)
    assert mw.viewer is None
    # The SEM canvas is intentionally left empty ("No image") on connect — no blank
    # placeholder is seeded; a real frame only arrives via sem/fib_acquisition_signal
    # -> _on_acquire. The controller still tracks state for the canvas, sans image.
    state = parent.view_controller._states[parent.view_controller.sem_canvas]
    assert state.image is None


def test_movement_widget_disconnects_canvas_on_teardown():
    """Regression: FibsemMovementWidget connects to the app-lifetime quad-view canvases.
    It is torn down via removeTab + deleteLater (which fires neither closeEvent nor close),
    so unless _teardown_connections disconnects those slots first, a later canvas
    double-click fires on the deleted widget and PyQt calls qFatal -> the process aborts
    (SIGABRT). This drives the real connect + teardown path; the emit after teardown must
    be inert. Without the fix, that emit would SIGABRT this whole test process."""
    _app()
    microscope, settings = utils.setup_session(config_path=_CONFIG)
    parent = _host(microscope)
    parent.image_widget = FibsemImageSettingsWidget(
        microscope=microscope, image_settings=settings.image, parent=parent
    )
    ctrl = parent.view_controller
    mw = FibsemMovementWidget(microscope=microscope, parent=parent)

    # the SEM + FIB canvas double-click slots were registered and tracked for teardown
    assert len(mw._canvas_dbl_click_conns) == 2

    # teardown mirrors the host order: _teardown_connections BEFORE deleteLater
    mw._teardown_connections()
    assert mw._canvas_dbl_click_conns == []
    mw.deleteLater()
    _QAPP.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)

    # a double-click on the surviving canvas must not reach the deleted widget
    ctrl.sem_canvas.canvas_double_clicked.emit(1.0, 2.0, None)
    ctrl.fib_canvas.canvas_double_clicked.emit(1.0, 2.0, None)
    # idempotent — safe to call again
    mw._teardown_connections()


def test_auto_functions_are_mutually_exclusive():
    """Regression: the F9/F11 hotkeys call run_autocontrast/run_autofocus directly, so the
    disabled buttons don't guard them. Each must refuse to start while another auto-function
    is running (they share the beam + _auto_function_error); otherwise F11-then-F9 races two
    workers and leaves the autofocus sweep uncancellable."""
    _app()
    microscope, settings = utils.setup_session(config_path=_CONFIG)
    parent = _host(microscope)
    iw = FibsemImageSettingsWidget(
        microscope=microscope, image_settings=settings.image, parent=parent
    )
    # never launch real hardware workers in this test; _toggle_interactions cascades into
    # sibling widgets (parent.movement_widget) that this bare host doesn't have — stub it,
    # it's UI enable/disable, not part of the mutual-exclusion logic under test.
    iw._autocontrast_worker = MagicMock()
    iw._autofocus_worker = MagicMock()
    iw._toggle_interactions = MagicMock()

    # from idle, autocontrast starts and marks the widget busy
    iw.run_autocontrast()
    assert iw._auto_function_running is True
    iw._autocontrast_worker.assert_called_once()

    # busy with autocontrast -> F11 (run_autofocus) is refused, no sweep starts
    iw.run_autofocus()
    assert iw._autofocus_running is False
    iw._autofocus_worker.assert_not_called()

    # busy with autofocus -> F9 (run_autocontrast) is refused
    iw._autofocus_running = True  # (still _auto_function_running)
    iw._autocontrast_worker.reset_mock()
    iw.run_autocontrast()
    iw._autocontrast_worker.assert_not_called()


def test_fm_widget_and_objective_build_viewer_less():
    _app()
    microscope, _ = utils.setup_session(config_path=_CONFIG)
    assert microscope.fm is not None, "sim-arctis config should provide an fm"
    # FMControlWidget resolves the controller via parent_widget -> parent_widget.
    main_ui = QtWidgets.QWidget()
    main_ui.view_controller = MicroscopeViewController(parent=main_ui)
    parent = QtWidgets.QWidget()
    parent.parent_widget = main_ui
    parent.viewer = None
    parent.microscope = microscope
    # builds ObjectiveControlWidget (the sub-widget that registered a napari
    # wheel callback unconditionally) — must not raise viewer-less.
    fm = FMControlWidget(microscope=microscope, viewer=None, parent=parent)
    assert fm.viewer is None
    assert fm.objectiveControlWidget is not None


def _run_all():
    tests = [
        test_image_and_movement_build_viewer_less,
        test_movement_widget_disconnects_canvas_on_teardown,
        test_auto_functions_are_mutually_exclusive,
        test_fm_widget_and_objective_build_viewer_less,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    _run_all()
