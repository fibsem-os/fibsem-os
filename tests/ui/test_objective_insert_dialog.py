"""The FM tab's Insert and Retract go through the stage movement service, the call
the App Server's route makes, with the tilt picked in the dialog."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
import time

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QDialog  # noqa: E402

import fibsem.config as cfg  # noqa: E402
from fibsem import utils  # noqa: E402
from fibsem.structures import InsertableDeviceState  # noqa: E402
from fibsem.ui.fm.widgets import objective_control_widget as module  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv)


class _Picked:
    """The dialog, answered: -180° picked."""

    def __init__(self, tilt_deg, parent=None):
        self.selected_tilt = -180.0

    def exec_(self):
        return QDialog.Accepted


@pytest.fixture
def widget(monkeypatch):
    microscope, _ = utils.setup_session(
        config_path=os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")
    )
    microscope.fm.objective.retract()
    calls = []
    service = microscope.stage_movement
    monkeypatch.setattr(
        service,
        "insert_objective",
        lambda tilt=None: calls.append(("insert", tilt)),
    )
    monkeypatch.setattr(
        service, "retract_objective", lambda: calls.append(("retract", None))
    )
    monkeypatch.setattr(module, "_InsertObjectiveDialog", _Picked)
    monkeypatch.setattr(module, "message_box_ui", lambda *a, **k: True)
    w = module.ObjectiveControlWidget(microscope.fm, microscope=microscope)
    w.calls = calls
    yield w
    w.cleanup()
    w.close()


def _drain(widget, timeout=5.0):
    deadline = time.monotonic() + timeout
    while widget._action_worker is not None and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)


def test_insert_asks_the_service_for_the_picked_tilt(widget):
    widget.insert_objective()
    _drain(widget)

    assert widget.calls == [("insert", pytest.approx(np.radians(-180.0)))]


def test_retract_goes_through_the_service(widget):
    widget.fm.objective.insert()
    assert widget.fm.objective.state is InsertableDeviceState.INSERTED

    widget.retract_objective()
    _drain(widget)

    assert widget.calls == [("retract", None)]
