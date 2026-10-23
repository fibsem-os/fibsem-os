"""A state question is answered on the prompt bar of the Microscope tab.

The prompt bar is the renderer for the ``state`` kind: the operator is at
the instrument, and confirming a position needs no image. Continue there is
a decision through ``Experiment.decide``, the one write path; an agent can
answer it too, and either way the prompt comes down. The task then reads the instrument for the position as confirmed.
"""

import os

from tests.fixtures.main_window import close_window

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time

import pytest

pytest.importorskip("PyQt5")

from psygnal.containers import EventedDict
from PyQt5.QtCore import QCoreApplication, QEvent

from fibsem.acting import ActorKind
from fibsem.applications.autolamella.structures import (
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.structures import FibsemStagePosition
from fibsem.workflows.attention import Attention
from fibsem.workflows.proposals import STATE, Decision, DecisionOutcome
from fibsem.workflows.site_tasks.reference_image import (
    AcquireReferenceImageConfig,
    AcquireReferenceImageTask,
)
from fibsem.workflows.status import HoldKind, TaskStatus
from tests.protocols import planned, task_protocol

REF = "Acquire Reference Image"


@pytest.fixture
def window(qapp, tmp_path):
    from fibsem.applications.autolamella.ui import window as module

    win = module.AutoLamellaWindow()
    win.system_widget.connect_to_microscope()
    ui = win
    experiment = Experiment(path=tmp_path, name="state-exp")
    os.makedirs(experiment.path, exist_ok=True)
    experiment.task_protocol = task_protocol(
        planned(name=REF, required=True, attention=Attention.supervised)
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
    win.tab_widget.setCurrentIndex(0)
    win._started = []

    yield win
    for manager, thread in win._started:
        manager.stop()
        _pump(qapp, lambda: not thread.is_alive())
    ui.session.disconnect()
    close_window(win)
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
    ui = window
    experiment = ui.experiment
    manager = TaskManager(
        microscope=ui.microscope,
        experiment=experiment,
        responder=ui.ui_responder,
        deciders=ui.session.deciders,
    )
    # The window hears it as it hears a run the runner starts.
    ui.session.runner.relay(manager)
    lamella = experiment.positions[0]
    task = AcquireReferenceImageTask(
        microscope=ui.microscope,
        config=lamella.task_config[REF],
        item=lamella,
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
        "Press Continue" in ui.panel.prompt_bar.message.text()
        and not ui.panel.prompt_bar.message.isHidden()
        and ui.panel.prompt_bar.yes.isEnabled()
    )


def _prompt_is_down(ui) -> bool:
    # The label doubles as the task's status line, so it is not empty once
    # the task carries on; what matters is that the question is gone from it.
    return (
        "Press Continue" not in ui.panel.prompt_bar.message.text()
        and ui.proposals._state_proposal is None
    )


def test_the_question_is_asked_on_the_prompt_bar_and_continue_decides_it(window, qapp):
    ui = window
    lamella = ui.experiment.positions[0]
    task, thread, seen = _start(window)

    assert _pump(qapp, lambda: _prompt_is_up(ui)), (
        "the prompt bar asks",
        ui.panel.prompt_bar.message.text(),
        ui.run_relay.hold,
        [(str(p.asked_by), p.kind) for p in lamella.proposals],
        seen,
    )
    assert ui.panel.prompt_bar.yes.text() == "Continue"
    assert ui.panel.prompt_bar.no.isHidden(), "one verb"
    assert window.tab_widget.currentIndex() == 0, "on the Microscope tab"
    assert ui.run_relay.hold is not None and ui.run_relay.hold.kind is HoldKind.proposal
    question = lamella.proposal(REF, STATE)
    assert question is not None and question.pending and question.asking

    ui.panel.prompt_bar.yes.click()

    assert _pump(qapp, lambda: not thread.is_alive()), "the task went on"
    assert "error" not in seen
    decision = question.current
    assert decision.outcome is DecisionOutcome.Confirmed
    assert decision.actor.kind is ActorKind.operator and decision.via == "workflow"
    pose = decision.values["stage_position"]
    assert isinstance(pose, FibsemStagePosition), "filled in by the task"
    here = ui.microscope.get_stage_position()
    assert (pose.x, pose.y, pose.z) == pytest.approx((here.x, here.y, here.z))
    assert _prompt_is_down(ui), "the prompt came down"
    assert ui.run_relay.hold is None
    assert lamella.task_state.status is TaskStatus.Completed, "nothing waits"


def test_answered_by_an_agent_the_prompt_comes_down_too(window, qapp):
    ui = window
    lamella = ui.experiment.positions[0]
    task, thread, seen = _start(window)
    assert _pump(qapp, lambda: _prompt_is_up(ui))
    asked = lamella.proposal(REF, STATE)

    result = ui.experiment.decide(
        lamella.id,
        REF,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            actor="agent:x",
            via="agent",
            proposal_id=asked.id,
        ),
    )
    assert result.applied, result.reason

    assert _pump(qapp, lambda: not thread.is_alive())
    assert "error" not in seen
    question = lamella.proposal(REF, STATE)
    assert question.current.via == "agent"
    assert isinstance(
        question.current.values.get("stage_position"), FibsemStagePosition
    )
    assert _prompt_is_down(ui)


def test_stop_takes_the_prompt_down_and_withdraws_the_question(window, qapp):
    ui = window
    lamella = ui.experiment.positions[0]
    task, thread, seen = _start(window)
    assert _pump(qapp, lambda: _prompt_is_up(ui))

    task.task_manager.stop()

    assert _pump(qapp, lambda: not thread.is_alive())
    assert isinstance(seen.get("error"), InterruptedError)
    assert lamella.proposal(REF, STATE).withdrawn
    assert _pump(qapp, lambda: _prompt_is_down(ui))
    assert ui.run_relay.hold is None
