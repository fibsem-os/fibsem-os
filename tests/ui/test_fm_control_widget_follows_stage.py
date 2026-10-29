"""The FM tab's controls follow the stage on an offset FM.

On an offset mount the controls are off at the beams and on at the FM. They used to be
worked out only when the tab was built, when the FM started or stopped acquiring, and
from the tab's own buttons, so a travel to the FM made anywhere else (the Movement tab,
the SEM to FM calibration) left every control off at the FM, where the microscope said
it could image. Found on the odemis simulator, deployment B, 2026-10-26.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import QEventLoop, QTimer  # noqa: E402

import fibsem.config as cfg  # noqa: E402
from fibsem import utils  # noqa: E402
from fibsem.ui.widgets.fluorescence_control_widget import FMControlWidget  # noqa: E402
from tests.fixtures.fm_host import FMHost  # noqa: E402

IFLM_CONFIG = os.path.join(cfg.SHIPPED_CONFIG_PATH, "sim-iflm-configuration.yaml")


def _pump(ms: int = 100) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec_()


def _until(condition, tries: int = 50) -> bool:
    for _ in range(tries):
        if condition():
            return True
        _pump()
    return condition()


@pytest.fixture()
def widget(qapp):
    microscope, _ = utils.setup_session(config_path=IFLM_CONFIG)
    host = FMHost()
    w = FMControlWidget(
        microscope=microscope,
        parent=host,
        view_controller=lambda: host.view_controller,
    )
    w._test_host = host  # a collected parent would destroy the widget
    yield w
    w._teardown_connections()


def test_a_travel_made_elsewhere_turns_the_controls_on_and_back_off(widget):
    microscope = widget.microscope
    insert = widget.objectiveControlWidget.pushButton_insert_objective
    assert microscope.stage_movement.current_device() != "FM"
    assert not insert.isEnabled(), "on at the beams, where the objective is far away"

    microscope.stage_movement.move_to_device("FM")
    assert _until(insert.isEnabled), "still off at the FM after the travel"
    assert widget.pushButton_toggle_acquisition.isEnabled()

    microscope.stage_movement.move_to_device("FIBSEM")
    assert _until(lambda: not insert.isEnabled()), "still on back at the beams"


def test_teardown_stops_following_the_stage(widget):
    position = widget.microscope.stage.position
    widget._teardown_connections()
    assert not any("_on_stage_moved" in repr(s) for s in position.changed._slots)
