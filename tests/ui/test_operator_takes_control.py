"""The operator's clicks take control of the microscope (FIB-1488): refused,
naming who is in control, while the agent, a script or a run acts; lent during
a question; beside the operator's own mill."""

from __future__ import annotations

import os
import threading

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")

from PyQt5.QtCore import QEventLoop, QTimer
from PyQt5.QtWidgets import QApplication

from fibsem import utils
from fibsem.acting import AGENT, OPERATOR, WORKFLOW
from fibsem.control import Control, MicroscopeBusy
from fibsem.structures import BeamType
from fibsem.ui import control as ui_control
from fibsem.ui.control import control_of, operator_acts
from fibsem.ui.qt.threading import FunctionWorker
from fibsem.ui.widgets.beam_settings_widget import FibsemBeamSettingsWidget

_app = QApplication.instance() or QApplication([])


def _run(worker: FunctionWorker, **start) -> bool:
    loop = QEventLoop()
    worker.finished.connect(loop.quit)
    QTimer.singleShot(5000, loop.quit)
    started = worker.start(**start)
    if started:
        loop.exec_()
    _app.processEvents()
    return started


@pytest.fixture
def toasts(monkeypatch):
    said = []
    monkeypatch.setattr(ui_control, "refused", lambda busy: said.append(str(busy)))
    return said


def test_a_worker_holds_control_as_the_operator_while_it_runs():
    control = Control()
    seen = []
    worker = FunctionWorker(lambda: seen.append(control.current.by))
    # A slot that acts finds control already given up.
    worker.returned.connect(lambda _: seen.append(control.current))
    assert _run(worker, control=control, action="move stage")
    assert seen == [OPERATOR, None]


def test_a_refused_worker_starts_nothing_and_reports_it(toasts):
    control = Control()
    ran, errors, finished = [], [], []
    worker = FunctionWorker(lambda: ran.append(1))
    worker.errored.connect(errors.append)
    worker.finished.connect(lambda: finished.append(1))
    with control.take(AGENT, "move the stage"):
        assert not worker.start(control=control, action="acquire images")
    assert ran == [] and finished == [1]
    assert isinstance(errors[0], MicroscopeBusy)
    assert errors[0].in_control.by == AGENT
    # The widget's error slot says it, so no toast as well.
    assert toasts == []


def test_a_refusal_nobody_listens_for_is_a_toast(toasts):
    control = Control()
    with control.take(WORKFLOW, "run"):
        assert not FunctionWorker(lambda: None).start(control=control, action="x")
    assert toasts == ["The microscope is busy: the workflow is in control (run)."]


def test_a_worker_without_control_runs_as_before():
    control = Control()
    ran = []
    with control.take(AGENT, "move"):
        assert _run(FunctionWorker(lambda: ran.append(1)))
    assert ran == [1]


def test_a_click_borrows_the_control_a_run_lends_during_a_question():
    control = Control()
    seen = []
    with control.take(WORKFLOW, "run"), control.lend():
        worker = FunctionWorker(lambda: seen.append(control.current.by))
        assert _run(worker, control=control, action="acquire images")
    assert seen == [OPERATOR]


def test_the_operators_clicks_never_refuse_each_other_and_the_agent_waits():
    control = Control()
    milling, release = threading.Event(), threading.Event()

    def mill():
        milling.set()
        release.wait(5)

    worker = FunctionWorker(mill)
    assert worker.start(control=control, action="mill")
    assert milling.wait(5)
    written = []
    # An image or a setting during their own mill goes ahead.
    assert operator_acts(control, "set current", written.append, 1)
    assert written == [1]
    with pytest.raises(MicroscopeBusy):
        with control.take(AGENT, "move the stage"):
            pass
    release.set()
    worker.join(5)
    _app.processEvents()
    assert control.current is None


def test_a_one_shot_write_is_refused_and_not_made(toasts):
    control = Control()
    written = []
    with control.take(AGENT, "acquire"):
        assert not operator_acts(control, "set current", written.append, 1)
    assert written == [] and len(toasts) == 1
    assert operator_acts(control, "set current", written.append, 2)
    assert written == [2]


def test_a_device_finds_its_microscopes_control():
    microscope, _ = utils.setup_session(manufacturer="Demo", setup_logging=False)
    assert control_of(microscope) is microscope.control
    assert control_of(microscope.beams[BeamType.ELECTRON]) is microscope.control
    assert control_of(None) is None


def test_a_refused_beam_setting_shows_the_beams_value_again(toasts):
    microscope, _ = utils.setup_session(manufacturer="Demo", setup_logging=False)
    widget = FibsemBeamSettingsWidget(microscope=microscope, beam_type=BeamType.ION)
    widget.populate_beam_combos()
    widget.update_from_settings(microscope.get_beam_settings(BeamType.ION))
    before = microscope.get_field_of_view(BeamType.ION)
    with microscope.control.take(AGENT, "acquire"):
        widget.hfw_spinbox.setValue(widget.hfw_spinbox.value() * 2)
    assert microscope.get_field_of_view(BeamType.ION) == pytest.approx(before)
    assert widget.get_settings().hfw == pytest.approx(before)
    assert toasts and "the agent is in control" in toasts[0]


def test_a_refused_beam_on_off_or_blank_changes_nothing(toasts):
    from fibsem.ui.widgets.dual_beam_widget import FibsemDualBeamWidget

    microscope, _ = utils.setup_session(manufacturer="Demo", setup_logging=False)
    widget = FibsemDualBeamWidget(microscope)
    bt = widget.beam_type
    on, blanked = microscope.is_on(bt), microscope.is_blanked(bt)
    with microscope.control.take(AGENT, "acquire"):
        widget._on_beam_on_clicked()
        widget._on_beam_blanked_clicked()
    assert (microscope.is_on(bt), microscope.is_blanked(bt)) == (on, blanked)
    assert len(toasts) == 2 and "the agent is in control" in toasts[0]
    widget._on_beam_blanked_clicked()
    assert microscope.is_blanked(bt) is not blanked


def _arctis():
    from pathlib import Path

    from fibsem import config as fcfg

    path = Path(fcfg.SHIPPED_CONFIG_PATH) / "sim-arctis-configuration.yaml"
    microscope, _ = utils.setup_session(
        config_path=str(path), manufacturer="Demo", setup_logging=False
    )
    return microscope


def test_a_refused_objective_move_leaves_the_objective(toasts):
    from PyQt5.QtWidgets import QWidget

    from fibsem.ui.fm.widgets.objective_control_widget import ObjectiveControlWidget

    microscope = _arctis()
    host = QWidget()
    host.microscope = microscope
    widget = ObjectiveControlWidget(microscope.fm, parent=host)
    objective = microscope.fm.objective
    objective.focus_position = objective.position + 50e-6
    before = objective.position
    with microscope.control.take(AGENT, "acquire"):
        widget.on_objective_position_changed(before * 1e6 + 20)
        widget.move_to_focus_position()
    assert objective.position == pytest.approx(before)
    assert len(toasts) == 2 and "the agent is in control" in toasts[0]
    widget.close()


def test_a_refused_camera_setting_shows_the_cameras_value_again(toasts):
    from fibsem.ui.fm.widgets.camera_widget import CameraWidget

    microscope = _arctis()
    widget = CameraWidget(microscope.fm)
    gain = microscope.fm.camera.gain
    with microscope.control.take(AGENT, "acquire"):
        widget.spinBox_gain.setValue(gain * 100 / 2)
    assert microscope.fm.camera.gain == pytest.approx(gain)
    assert widget.gain == pytest.approx(gain * 100)
    assert toasts and "the agent is in control" in toasts[0]
