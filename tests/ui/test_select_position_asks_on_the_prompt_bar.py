"""Select Position, supervised, asks everything it needs on the Microscope tab.

The tilt to the milling angle (Tilt or Skip) and the move to the milling
position (Continue) are ``state`` questions on the prompt bar. The point of
interest is the marker on the FIB view and the alignment area the rectangle
on it, each with Continue. Every answer is a decision on the record, the task
carries on, and nothing waits on it once it ends.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time

import pytest

pytest.importorskip("PyQt5")

from psygnal.containers import EventedDict
from PyQt5.QtCore import QCoreApplication, QEvent

from fibsem.acting import ActorKind
from fibsem.applications.autolamella.proposals import POINT_OF_INTEREST
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaWorkflowConfig,
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.applications.autolamella.workflows.tasks.rough import MillRoughTaskConfig
from fibsem.applications.autolamella.workflows.tasks.select_position import (
    SelectMillingPositionTask,
    SelectMillingPositionTaskConfig,
)
from fibsem.structures import BeamType, FibsemRectangle, FibsemStagePosition
from fibsem.workflows.attention import Attention
from fibsem.workflows.proposals import (
    ALIGNMENT_AREA,
    STATE,
    TASK_RESULT,
    Decision,
    DecisionOutcome,
)
from fibsem.workflows.status import HoldKind, TaskStatus

SETUP = "Setup Lamella Position"
ROUGH = "Rough Milling"
MILLING_ANGLE = 15.0


@pytest.fixture
def window(qapp, tmp_path):
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module

    win = module.AutoLamellaSingleWindowUI()
    win.autolamella_ui.system_widget.connect_to_microscope()
    ui = win.autolamella_ui
    experiment = Experiment(path=tmp_path, name="setup-exp")
    os.makedirs(experiment.path, exist_ok=True)
    experiment.task_protocol = AutoLamellaTaskProtocol(
        workflow_config=AutoLamellaWorkflowConfig(
            tasks=[
                AutoLamellaTaskDescription(
                    name=SETUP, required=True, attention=Attention.supervised
                ),
                AutoLamellaTaskDescription(name=ROUGH, required=True, requires=[SETUP]),
            ]
        )
    )
    experiment.add_new_lamella(
        ui.microscope.get_microscope_state(),
        EventedDict(
            {
                SETUP: SelectMillingPositionTaskConfig(
                    task_name=SETUP,
                    milling_angle=MILLING_ANGLE,
                    auto_milling_alignment=False,
                    use_autofocus=False,
                    select_poi=True,
                ),
                ROUGH: MillRoughTaskConfig(task_name=ROUGH),
            }
        ),
    )
    lamella = experiment.positions[0]
    lamella.path.mkdir(parents=True, exist_ok=True)
    lamella.milling_pose = ui.microscope.get_microscope_state()
    ui.experiment = experiment
    win._on_experiment_update()
    win.tab_widget.setCurrentIndex(0)
    win._started = []

    yield win
    for manager, thread in win._started:
        manager.stop()
        _pump(qapp, lambda: not thread.is_alive())
    microscope = ui.microscope
    ui.disconnect_from_microscope()
    if microscope is not None:
        microscope.disconnect()
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        win.close()
    finally:
        qapp.quit = original_quit
    win.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    qapp.processEvents()


def _pump(qapp, predicate, timeout_s=30.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _start(window):
    """The real task, its real run, on a worker thread."""
    ui = window.autolamella_ui
    experiment = ui.experiment
    manager = TaskManager(
        microscope=ui.microscope,
        experiment=experiment,
        parent_ui=ui,
        responder=ui.ui_responder,
        answerers=ui.session.answerers,
    )
    lamella = experiment.positions[0]
    task = SelectMillingPositionTask(
        microscope=ui.microscope,
        config=lamella.task_config[SETUP],
        lamella=lamella,
        parent_ui=ui,
        task_manager=manager,
    )
    seen = {}

    def _target():
        try:
            task.run()
        except Exception as exc:  # noqa: BLE001 - run re-raises for the manager
            seen["error"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    window._started.append((manager, thread))
    return task, thread, seen


def _prompt_says(ui, words: str) -> bool:
    # Text and enabled state, not isVisible(): the window is never shown
    # offscreen, so nothing in it is "visible".
    return (
        words in ui.label_instructions.text()
        and not ui.label_instructions.isHidden()
        and ui.pushButton_yes.isEnabled()
    )


def test_the_tilt_and_the_position_are_confirmed_on_the_prompt_bar(window, qapp):
    ui = window.autolamella_ui
    lamella = ui.experiment.positions[0]
    microscope = ui.microscope
    assert microscope.get_current_milling_angle() == pytest.approx(38.0)
    task, thread, seen = _start(window)

    # 1. the tilt
    assert _pump(qapp, lambda: _prompt_says(ui, "Tilt to the milling angle")), (
        ui.label_instructions.text(),
        ui.hold,
        seen,
    )
    assert (ui.pushButton_yes.text(), ui.pushButton_no.text()) == ("Tilt", "Skip")
    assert ui.pushButton_no.isEnabled(), "Skip is how not to tilt"
    assert ui.hold is not None and ui.hold.kind is HoldKind.question
    tilt = lamella.proposal(SETUP, STATE)
    assert tilt is not None and tilt.pending and tilt.asking
    assert microscope.get_current_milling_angle() == pytest.approx(38.0), (
        "not tilted until confirmed"
    )

    ui.pushButton_yes.click()

    # 2. the position, on the image taken at the milling angle
    assert _pump(qapp, lambda: _prompt_says(ui, "Double click the image")), (
        ui.label_instructions.text(),
        seen,
    )
    assert microscope.get_current_milling_angle() == pytest.approx(MILLING_ANGLE)
    assert tilt.current.outcome is DecisionOutcome.Confirmed
    assert tilt.current.actor.kind is ActorKind.operator and tilt.current.via == (
        "workflow"
    )
    position = lamella.proposal(SETUP, STATE)
    assert position is not tilt and position.asking
    assert position.provenance["reference_image"].endswith("_post_tilt_ib.tif")
    assert ui.pushButton_yes.text() == "Continue" and ui.pushButton_no.isHidden()

    ui.pushButton_yes.click()

    # 3. the point of interest, as the marker on the FIB view
    assert _pump(qapp, lambda: _prompt_says(ui, "Move the marker")), (
        ui.label_instructions.text(),
        seen,
    )
    pose = position.current.values["stage_position"]
    assert isinstance(pose, FibsemStagePosition), "filled in by the task"
    point = lamella.proposal(SETUP, POINT_OF_INTEREST)
    assert point.asking and ui.hold.kind is HoldKind.question
    controller = window.view_controller
    marker = controller.overlay_points(BeamType.ION, "poi")
    assert marker, "the marker is up"
    col, row = marker[0]
    controller.set_points(BeamType.ION, "poi", [(col + 40, row)])

    ui.pushButton_yes.click()

    # 4. the alignment area, as the rectangle on the FIB view
    assert _pump(qapp, lambda: _prompt_says(ui, "alignment area")), (
        ui.label_instructions.text(),
        seen,
    )
    assert point.current.outcome is DecisionOutcome.Confirmed
    moved = point.current.values["poi"]
    assert moved.x > 0 and lamella.poi == moved, "the dragged point, written"
    assert not controller.overlay_points(BeamType.ION, "poi"), "marker down"
    area = lamella.proposal(SETUP, ALIGNMENT_AREA)
    assert area.asking
    assert area.provenance["reference_image"].endswith("_post_tilt_ib.tif")
    assert isinstance(area.values["alignment_area"], FibsemRectangle)

    ui.pushButton_yes.click()

    assert _pump(qapp, lambda: not thread.is_alive()), "the task went on"
    assert "error" not in seen, seen
    assert area.current.outcome is DecisionOutcome.Confirmed
    assert area.current.values["alignment_area"] == area.values["alignment_area"]
    assert lamella.alignment_area == area.values["alignment_area"]
    assert [p.kind for p in lamella.proposals_from(SETUP)] == [
        STATE,
        STATE,
        POINT_OF_INTEREST,
        ALIGNMENT_AREA,
        TASK_RESULT,
    ]
    assert lamella.task_state.status is TaskStatus.Completed, "nothing waits"
    assert task.task_manager._defer_reason(lamella, ROUGH) is None


def test_skip_does_not_tilt(window, qapp):
    """Skip rejects the tilt: the task carries on at the angle it is at, and
    asks for the position there."""
    ui = window.autolamella_ui
    experiment = ui.experiment
    lamella = experiment.positions[0]
    microscope = ui.microscope
    task, thread, seen = _start(window)

    assert _pump(qapp, lambda: _prompt_says(ui, "Tilt to the milling angle")), (
        ui.label_instructions.text(),
        seen,
    )
    tilt = lamella.proposal(SETUP, STATE)
    ui.pushButton_no.click()

    assert _pump(qapp, lambda: _prompt_says(ui, "Double click the image")), (
        ui.label_instructions.text(),
        seen,
    )
    assert "error" not in seen, seen
    assert tilt.current.outcome is DecisionOutcome.Rejected
    assert tilt.current.reason == "Skip pressed"
    assert microscope.get_current_milling_angle() == pytest.approx(38.0), "not tilted"
    position = lamella.proposal(SETUP, STATE)
    assert position is not tilt and position.asking

    task.task_manager.stop()
    assert _pump(qapp, lambda: not thread.is_alive())
    assert isinstance(seen.get("error"), InterruptedError)
