"""Regression: the retrofitted live ``FibsemSpotBurnWidget`` builds on the shared editor.

After the spot-burn consolidation the live widget embeds ``SpotBurnCoordinatesWidget`` and
drives it through a ``SpotBurnSettings`` payload (coordinates on the canvas overlay,
current/exposure on the form). This builds it against a real Demo microscope + a main-tab
controller (faked parent chain) and exercises the settings-based adapters + the overlay.
It does not run an actual burn.

Run directly (headless):
    QT_QPA_PLATFORM=offscreen python fibsem/ui/widgets/tests/test_spot_burn_live_widget.py
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal

# QApplication before the napari/vispy-heavy fibsem imports below.
_QAPP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

from fibsem import utils
from fibsem.imaging.spot import SpotBurnSettings
from fibsem.structures import BeamType, FibsemImage, Point
from fibsem.ui.FibsemSpotBurnWidget import FibsemSpotBurnWidget
from fibsem.ui.widgets.quad_view import MicroscopeViewController

_CONFIG = "fibsem/config/sim-arctis-configuration.yaml"


class _ImageWidget(QtWidgets.QWidget):
    viewer_update_signal = pyqtSignal()
    ib_image = None


def _host():
    """Parent chain: AutoLamellaUI -> AutoLamellaMainUI (view_controller) + image_widget."""
    microscope, _ = utils.setup_session(config_path=_CONFIG)
    main_ui = QtWidgets.QWidget()
    main_ui.view_controller = MicroscopeViewController(parent=main_ui)
    iw = _ImageWidget()
    iw.ib_image = FibsemImage.generate_blank_image(hfw=100e-6, random=False)
    parent_ui = QtWidgets.QWidget()
    parent_ui.microscope = microscope
    parent_ui.parent_widget = main_ui
    parent_ui.image_widget = iw
    return parent_ui, main_ui.view_controller, iw.ib_image


def test_live_spot_burn_widget_retrofit():
    parent_ui, ctrl, img = _host()
    h, w = img.data.shape[:2]

    sbw = FibsemSpotBurnWidget(parent=parent_ui)
    assert sbw.coord_editor.controller is ctrl
    assert not sbw.pushButton_run_spot_burn.isEnabled()

    # activate, then push workflow params (same order as the UI update handler)
    sbw.set_active()
    _QAPP.processEvents()
    current = sbw.comboBox_beam_current.itemData(0)
    sbw.update_parameters(
        {"milling_current": current, "exposure_time": 5,
         "coordinates": [Point(0.3, 0.4), Point(0.6, 0.7)]}
    )
    _QAPP.processEvents()

    pts = ctrl.overlay_points(BeamType.ION, "spot_burn")
    assert len(pts) == 2
    assert abs(pts[0][0] - 0.3 * w) < 1e-6
    assert sbw.doubleSpinBox_exposure_time.value() == 5
    assert sbw.pushButton_run_spot_burn.isEnabled()
    assert "2 points" in sbw.label_information.text()
    assert len(sbw.get_coordinates()) == 2

    # a canvas add flows overlay -> editor -> widget refresh
    ctrl.overlay_edited.emit(
        BeamType.ION, "spot_burn",
        [(0.3 * w, 0.4 * h), (0.6 * w, 0.7 * h), (0.5 * w, 0.5 * h)],
    )
    _QAPP.processEvents()
    assert len(sbw.get_coordinates()) == 3

    # the run payload is a SpotBurnSettings (coords from editor, current/exposure from form)
    s = sbw._current_settings()
    assert isinstance(s, SpotBurnSettings)
    assert len(s.coordinates) == 3
    assert s.milling_current == current and s.exposure_time == 5

    # clear -> empty overlay + disabled run
    sbw.clear_points_layer()
    _QAPP.processEvents()
    assert sbw.get_coordinates() == []
    assert ctrl.overlay_points(BeamType.ION, "spot_burn") == []
    assert not sbw.pushButton_run_spot_burn.isEnabled()

    # deactivate -> overlay removed
    sbw.set_inactive()
    _QAPP.processEvents()
    assert ctrl.overlay_points(BeamType.ION, "spot_burn") == []


if __name__ == "__main__":
    test_live_spot_burn_widget_retrofit()
    print("PASS")
