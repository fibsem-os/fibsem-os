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

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets

# Create the QApplication before the napari/vispy-heavy fibsem imports below, or
# importing + using those widgets can abort ("QWidget before QApplication").
_QAPP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

from fibsem import utils
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.FibsemMovementWidget import FibsemMovementWidget
from fibsem.ui.widgets.fluorescence_control_widget import FMControlWidget
from fibsem.ui.widgets.quad_view import MicroscopeViewController

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
    # the controller received the seeded SEM image (viewer-less display works)
    state = parent.view_controller._states[parent.view_controller.sem_canvas]
    assert state.image is not None


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
        test_fm_widget_and_objective_build_viewer_less,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\nALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    _run_all()
