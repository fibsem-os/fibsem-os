"""SEM to FM in the Qt app: the stage control's FM button, the Calibration tab's
entry, and the calibration dialog, on the offset simulator."""

import os
import sys

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from PyQt5.QtCore import QEventLoop, QTimer

import fibsem.config as cfg
from fibsem import utils
from fibsem.structures.settings import FibsemStagePosition
from fibsem.structures.system import DeviceCalibration, InsertableDeviceState
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.FibsemMovementWidget import FibsemMovementWidget
from fibsem.ui.widgets.microscope_configuration_window import (
    MicroscopeConfigurationWindow,
)
from fibsem.ui.widgets.sem_to_fm_dialog import (
    STEP_BEAMS,
    STEP_ROTATION,
    SemToFmDialog,
)
from fibsem.ui.widgets.stage_control_widget import FM_NEEDS_CALIBRATION, travel_route

sys.path.insert(0, os.path.dirname(__file__))  # not str.rsplit: Windows paths
from test_viewer_less_widgets import _CanvasHost  # noqa: E402

IFLM_CONFIG = os.path.join(cfg.SHIPPED_CONFIG_PATH, "sim-iflm-configuration.yaml")
NOMINAL_X = 48.8e-3


def _pump(ms: int = 200) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec_()


def _until(condition, tries: int = 50) -> None:
    for _ in range(tries):
        if condition():
            return
        _pump(100)


def _offset_session(tmp_path, calibrated: bool):
    settings = utils.load_yaml(IFLM_CONFIG)
    if not calibrated:
        del settings["calibration"]["devices"]
    path = tmp_path / "configuration.yaml"
    utils.save_yaml(path, settings)
    microscope, image = utils.setup_session(config_path=str(path))
    return microscope, image, str(path)


def _control(host, microscope, image):
    host.image_widget = FibsemImageSettingsWidget(
        microscope=microscope,
        image_settings=image.image,
        parent=host,
        view_controller=lambda: host.view_controller,
        toggle_movement=lambda enable: host.movement_widget._toggle_interactions(
            enable, caller="ui"
        ),
    )
    host.movement_widget = FibsemMovementWidget(
        microscope=microscope,
        image_widget=host.image_widget,
        view_controller=lambda: host.view_controller,
        milling_widget=lambda: host.milling_widget,
        parent=host,
    )
    return host.movement_widget.control_widget


@pytest.fixture
def stage_control(qapp, tmp_path, monkeypatch):
    """The stage control on a calibrated offset simulator. Images are not retaken
    after a move, and every travel confirmation is answered OK and recorded."""
    microscope, image, _ = _offset_session(tmp_path, calibrated=True)
    host = _CanvasHost()
    control = _control(host, microscope, image)
    monkeypatch.setattr(control, "update_ui_after_movement", lambda *a, **k: None)
    asked = []

    def confirm(steps):
        asked.append(steps)
        return True

    monkeypatch.setattr(control, "_confirm_travel", confirm)
    yield microscope, control, asked
    host.deleteLater()


def _press(control, button) -> None:
    button.click()
    _until(lambda: control.pushButton_move.isEnabled())


# ── the FM button ────────────────────────────────────────────────────────


def test_the_fm_button_is_off_until_sem_to_fm_is_calibrated(qapp, tmp_path):
    microscope, image, _ = _offset_session(tmp_path, calibrated=False)
    host = _CanvasHost()
    control = _control(host, microscope, image)

    assert not control.pushButton_move_to_fm.isHidden()
    assert control.pushButton_move_to_fm.isEnabled() is False
    assert control.pushButton_move_to_fm.toolTip() == FM_NEEDS_CALIBRATION

    microscope.system.stage.device_calibration = DeviceCalibration(
        fibsem=FibsemStagePosition(x=0.0, y=0.0, z=0.0),
        fm=FibsemStagePosition(x=NOMINAL_X, y=0.0, z=0.0),
    )
    control.update_ui()
    _pump()
    assert control.pushButton_move_to_fm.isEnabled() is True
    host.deleteLater()


def test_a_compustage_has_no_fm_button(qapp):
    microscope, image = utils.setup_session(
        config_path=os.path.join(
            cfg.SHIPPED_CONFIG_PATH, "sim-arctis-configuration.yaml"
        )
    )
    host = _CanvasHost()
    control = _control(host, microscope, image)
    assert control.pushButton_move_to_fm.isHidden()
    host.deleteLater()


def test_the_route_is_listed_for_a_travel_and_not_for_a_turn(stage_control):
    microscope, _, _ = stage_control
    microscope.move_to_orientation("SEM")

    assert travel_route(microscope, "FM", "FIB") == [
        "Retract the objective",
        "Turn to the FIB orientation at the beams",
        "Travel 48.8 mm to the FM",
        "Insert the objective",
    ]
    assert travel_route(microscope, "FIBSEM", "FIB") == []


def test_the_fm_button_travels_and_sem_brings_the_stage_back(stage_control):
    microscope, control, asked = stage_control
    movement = microscope.stage_movement
    microscope.move_to_orientation("FIB")

    _press(control, control.pushButton_move_to_fm)
    assert movement.current_device() == "FM"
    assert microscope.fm.objective.state is InsertableDeviceState.INSERTED
    assert asked[-1][-2:] == ["Travel 48.8 mm to the FM", "Insert the objective"]

    _press(control, control.pushButton_move_to_sem_orientation)
    assert movement.current_device() == "FIBSEM"
    assert movement.stage_orientation() == "SEM"
    assert microscope.fm.objective.state is InsertableDeviceState.RETRACTED
    assert asked[-1] == [
        "Retract the objective",
        "Travel 48.8 mm back to the beams",
        "Turn to the SEM orientation",
    ]


def test_a_turn_at_the_beams_is_not_asked_about(stage_control):
    microscope, control, asked = stage_control
    microscope.move_to_orientation("FIB")

    _press(control, control.pushButton_move_to_sem_orientation)

    assert microscope.stage_movement.stage_orientation() == "SEM"
    assert asked == []


def test_a_cancelled_travel_does_not_move(stage_control, monkeypatch):
    microscope, control, _ = stage_control
    microscope.move_to_orientation("FIB")
    before = microscope.get_stage_position()
    monkeypatch.setattr(control, "_confirm_travel", lambda steps: False)

    control.pushButton_move_to_fm.click()
    _pump()

    assert microscope.get_stage_position().x == pytest.approx(before.x)
    assert control.pushButton_move_to_fm.isEnabled()


# ── the Calibration tab ──────────────────────────────────────────────────


def _calibration_tab_button(window):
    from PyQt5.QtWidgets import QPushButton

    return window.tabs.widget(2).findChild(QPushButton, "calibrate_sem_to_fm")


def test_the_calibration_tab_offers_sem_to_fm_on_an_offset_mount(qapp, tmp_path):
    microscope, _, _ = _offset_session(tmp_path, calibrated=False)
    window = MicroscopeConfigurationWindow(microscope)

    assert window.tabs.tabText(2) == "Calibration"
    assert _calibration_tab_button(window) is not None
    window.close_without_asking()

    arctis, _ = utils.setup_session(
        config_path=os.path.join(
            cfg.SHIPPED_CONFIG_PATH, "sim-arctis-configuration.yaml"
        )
    )
    window = MicroscopeConfigurationWindow(arctis)
    assert _calibration_tab_button(window) is None
    window.close_without_asking()


# ── the dialog ───────────────────────────────────────────────────────────


def test_the_dialog_captures_a_site_and_restores_on_cancel(qapp, tmp_path, monkeypatch):
    microscope, _, path = _offset_session(tmp_path, calibrated=False)
    microscope.move_to_orientation("FIB")
    before = microscope.system.stage.rotation_centre_correction
    dialog = SemToFmDialog(microscope, configuration_path=path)
    monkeypatch.setattr(dialog.calibration, "measure_coincidence", lambda name: {})
    assert dialog._step == STEP_ROTATION

    dialog.button_next.click()  # Skip
    assert dialog._step == STEP_BEAMS
    assert dialog.button_next.isEnabled() is False

    dialog.action_buttons["Capture beams"].click()
    _until(lambda: dialog.button_next.isEnabled())
    assert dialog.button_next.isEnabled()
    assert len(dialog.calibration.sites) == 1

    microscope.system.stage.rotation_centre_correction = (1e-6, 1e-6)
    dialog.reject()
    assert microscope.system.stage.rotation_centre_correction == before
