"""A state question is answered on the prompt bar of the Microscope tab.

The prompt bar is the renderer for the ``state`` kind: the operator is at
the instrument, and confirming a position needs no image. Continue there is
a decision through ``Experiment.decide``, the one write path; the Review tab
lists the same question and can answer it too, and either way the prompt
comes down. The task then reads the instrument for the position as confirmed.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time

import pytest

pytest.importorskip("PyQt5")

from psygnal.containers import EventedDict
from PyQt5.QtCore import QCoreApplication, QEvent

from fibsem.applications.autolamella.proposals import (
    STATE,
    AuthorKind,
    DecisionOutcome,
)
from fibsem.applications.autolamella.structures import (
    Attention,
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaTaskStatus,
    AutoLamellaWorkflowConfig,
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.applications.autolamella.workflows.tasks.reference_image import (
    AcquireReferenceImageConfig,
    AcquireReferenceImageTask,
)
from fibsem.applications.autolamella.workflows.tasks.status import HoldKind
from fibsem.structures import FibsemStagePosition

REF = "Acquire Reference Image"


@pytest.fixture
def window(qapp, tmp_path):
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module

    win = module.AutoLamellaSingleWindowUI()
    win.autolamella_ui.system_widget.connect_to_microscope()
    ui = win.autolamella_ui
    experiment = Experiment(path=tmp_path, name="state-exp")
    os.makedirs(experiment.path, exist_ok=True)
    experiment.task_protocol = AutoLamellaTaskProtocol(
        workflow_config=AutoLamellaWorkflowConfig(
            tasks=[
                AutoLamellaTaskDescription(
                    name=REF, required=True, attention=Attention.supervised
                )
            ]
        )
    )
    experiment.add_new_lamella(
        ui.microscope.get_microscope_state(),
        EventedDict({REF: AcquireReferenceImageConfig(task_name=REF)}),
    )
    lamella = experiment.positions[0]
    lamella.path.mkdir(parents=True, exist_ok=True)
    lamella.milling_pose = ui.microscope.get_microscope_state()
    ui.experiment = experiment
    win._on_experiment_update()
    win._preferences.features.proposer_reviewer_workflow_enabled = True
    win._apply_review_visibility()
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


def _pump(qapp, predicate, timeout_s=15.0) -> bool:
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
    manager = TaskManager(microscope=ui.microscope, experiment=experiment, parent_ui=ui)
    manager.review_enabled = True
    lamella = experiment.positions[0]
    task = AcquireReferenceImageTask(
        microscope=ui.microscope,
        config=lamella.task_config[REF],
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


def _prompt_is_up(ui) -> bool:
    # Text and enabled state, not isVisible(): the window is never shown
    # offscreen, so nothing in it is "visible".
    return (
        "Press Continue" in ui.label_instructions.text()
        and not ui.label_instructions.isHidden()
        and ui.pushButton_yes.isEnabled()
    )


def _prompt_is_down(ui) -> bool:
    # The label doubles as the task's status line, so it is not empty once
    # the task carries on; what matters is that the question is gone from it.
    return (
        "Press Continue" not in ui.label_instructions.text()
        and ui._state_question is None
    )


def test_the_question_is_asked_on_the_prompt_bar_and_continue_decides_it(window, qapp):
    ui = window.autolamella_ui
    lamella = ui.experiment.positions[0]
    task, thread, seen = _start(window)

    assert _pump(qapp, lambda: _prompt_is_up(ui)), (
        "the prompt bar asks",
        ui.label_instructions.text(),
        ui.hold,
        {k: [p.kind for p in v] for k, v in lamella.proposals.items()},
        seen,
    )
    assert ui.pushButton_yes.text() == "Continue"
    assert ui.pushButton_no.isHidden(), "one verb"
    assert window.tab_widget.currentIndex() == 0, "not sent to the Review tab"
    assert ui.hold is not None and ui.hold.kind is HoldKind.decision
    question = lamella.proposal(REF, STATE)
    assert question is not None and question.pending and question.asking
    assert any("asking now" in s for s in window.review_tab.row_summaries()), (
        "listed in the Review tab too"
    )

    ui.pushButton_yes.click()

    assert _pump(qapp, lambda: not thread.is_alive()), "the task went on"
    assert "error" not in seen
    decision = question.current
    assert decision.outcome is DecisionOutcome.Confirmed
    assert decision.author.kind is AuthorKind.human and decision.via == "workflow"
    pose = decision.values["stage_position"]
    assert isinstance(pose, FibsemStagePosition), "filled in by the task"
    here = ui.microscope.get_stage_position()
    assert (pose.x, pose.y, pose.z) == pytest.approx((here.x, here.y, here.z))
    assert _prompt_is_down(ui), "the prompt came down"
    assert ui.hold is None
    assert lamella.task_state.status is AutoLamellaTaskStatus.AwaitingDecision, (
        "its own result still waits for a look, as any supervised task's does"
    )


def test_answered_from_the_review_tab_the_prompt_comes_down_too(window, qapp):
    ui = window.autolamella_ui
    lamella = ui.experiment.positions[0]
    task, thread, seen = _start(window)
    assert _pump(qapp, lambda: _prompt_is_up(ui))
    tab = window.review_tab
    assert tab.select(lamella.id, REF)

    tab.confirm_current()

    assert _pump(qapp, lambda: not thread.is_alive())
    assert "error" not in seen
    question = lamella.proposal(REF, STATE)
    assert question.current.via == "review"
    assert isinstance(
        question.current.values.get("stage_position"), FibsemStagePosition
    )
    assert _prompt_is_down(ui)


def test_stop_takes_the_prompt_down_and_withdraws_the_question(window, qapp):
    ui = window.autolamella_ui
    lamella = ui.experiment.positions[0]
    task, thread, seen = _start(window)
    assert _pump(qapp, lambda: _prompt_is_up(ui))

    task.task_manager.stop()

    assert _pump(qapp, lambda: not thread.is_alive())
    assert isinstance(seen.get("error"), InterruptedError)
    assert lamella.proposal(REF, STATE).withdrawn
    assert _pump(qapp, lambda: _prompt_is_down(ui))
    assert ui.hold is None
