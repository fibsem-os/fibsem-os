"""A supervised task that asks with ``ask`` holds the run until the decision.

The hold is the task manager's, not the responder's: the task records the
question, the manager raises the hold, the main window fronts the Review tab,
and the decision -- from the tab, from an agent -- releases the task with the
decided values. Stop withdraws the question. No prompt, no adapter.
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
    DETECTION,
    AuthorKind,
    Decision,
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
from fibsem.applications.autolamella.workflows.tasks.rough import (
    MillRoughTask,
    MillRoughTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.status import HoldKind
from fibsem.structures import Point

ROUGH = "Rough Milling"
PROPOSED = {"features": [{"name": "LamellaCentre", "px": Point(10, 20)}]}


class _AsksForADetection(MillRoughTask):
    questions = (DETECTION,)


@pytest.fixture
def window(qapp, tmp_path):
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module

    win = module.AutoLamellaSingleWindowUI()
    win.autolamella_ui.system_widget.connect_to_microscope()
    ui = win.autolamella_ui
    experiment = Experiment(path=tmp_path, name="ask-exp")
    os.makedirs(experiment.path, exist_ok=True)
    experiment.task_protocol = AutoLamellaTaskProtocol(
        workflow_config=AutoLamellaWorkflowConfig(
            tasks=[
                AutoLamellaTaskDescription(
                    name=ROUGH, required=True, attention=Attention.supervised
                )
            ]
        )
    )
    experiment.add_new_lamella(
        ui.microscope.get_microscope_state(),
        EventedDict({ROUGH: MillRoughTaskConfig(task_name=ROUGH)}),
    )
    experiment.positions[0].path.mkdir(parents=True, exist_ok=True)
    ui.experiment = experiment
    win._on_experiment_update()
    win._preferences.features.proposer_reviewer_workflow_enabled = True
    win._apply_review_visibility()
    win.tab_widget.setCurrentIndex(0)
    win._started = []  # (manager, thread) pairs, stopped before teardown

    yield win
    # A task still waiting on its question must be released before the
    # window goes: its hold and its notification both land on widgets that
    # are about to be deleted.
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


def _pump(qapp, predicate, timeout_s=10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _start(window):
    """Run a real task on a worker thread whose body asks and keeps the answer."""
    ui = window.autolamella_ui
    experiment = ui.experiment
    manager = TaskManager(microscope=ui.microscope, experiment=experiment, parent_ui=ui)
    manager.review_enabled = True
    lamella = experiment.positions[0]
    task = _AsksForADetection(
        microscope=ui.microscope,
        config=lamella.task_config[ROUGH],
        lamella=lamella,
        parent_ui=ui,
        task_manager=manager,
    )
    seen = {}

    def _run():
        seen["decision"] = task.ask(DETECTION, PROPOSED, image="ml-1_ib.tif")

    task._run = _run  # type: ignore[method-assign]

    def _target():
        try:
            task.run()
        except Exception as exc:  # noqa: BLE001 - run re-raises for the manager
            seen["error"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    window._started.append((manager, thread))
    return task, thread, seen


def test_the_run_holds_on_the_question_until_it_is_decided(window, qapp):
    ui = window.autolamella_ui
    lamella = ui.experiment.positions[0]
    task, thread, seen = _start(window)

    assert _pump(qapp, lambda: ui.hold is not None), "the run is held"
    assert ui.hold.kind is HoldKind.decision
    assert ui.hold.items == (f"{lamella.name}/{ROUGH}",)
    # The hold is set from the worker thread; the notification that fronts
    # the tab is queued to this one, so it may land a moment later.
    assert _pump(
        qapp, lambda: window.tab_widget.currentWidget() is window.review_tab
    ), "fronted"
    assert any("Holding" in s for s in window.review_tab.row_summaries()), (
        window.review_tab.row_summaries()
    )
    assert _pump(
        qapp,
        lambda: "waiting for your decision" in ui.label_workflow_information.text(),
    ), ui.label_workflow_information.text()
    proposal = lamella.proposal(ROUGH)
    assert proposal.kind == DETECTION and proposal.pending and proposal.asking
    assert thread.is_alive(), "stopped on its next line"

    moved = {"features": [{"name": "LamellaCentre", "px": Point(12, 26)}]}
    result = ui.experiment.decide(
        lamella.id,
        ROUGH,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values=moved,
            proposal_id=proposal.id,
        ),
    )
    assert result.applied, result.reason
    assert _pump(qapp, lambda: not thread.is_alive()), "released"

    decision = seen["decision"]
    assert decision.outcome is DecisionOutcome.Confirmed
    assert decision.values["features"][0]["px"] == Point(12, 26), "the decided value"
    assert decision.author.kind is AuthorKind.human
    assert ui.hold is None
    assert "error" not in seen
    # The question is decided; the task's own result is not: supervised with
    # the preference on and no inline confirmation of the result, it waits for
    # one in the Review tab, as any supervised task's does.
    question, result_proposal = lamella.proposals[ROUGH]
    assert question is proposal and result_proposal.kind == "task_result"
    assert not question.pending and result_proposal.pending
    assert lamella.task_state.status is AutoLamellaTaskStatus.AwaitingDecision


def test_stop_withdraws_the_question_and_cancels_the_task(window, qapp):
    ui = window.autolamella_ui
    lamella = ui.experiment.positions[0]
    task, thread, seen = _start(window)
    assert _pump(qapp, lambda: ui.hold is not None)
    proposal = lamella.proposal(ROUGH)

    task.task_manager.stop()

    assert _pump(qapp, lambda: not thread.is_alive())
    assert "decision" not in seen
    assert isinstance(seen.get("error"), InterruptedError)
    assert proposal.withdrawn
    assert ui.hold is None
    assert lamella.task_state.status is AutoLamellaTaskStatus.Cancelled


def test_every_declared_question_has_a_renderer_to_answer_it_in():
    """A task says what it asks; the Review tab has to be able to show it."""
    from fibsem.applications.autolamella.ui.review_tab_widget import REVIEW_RENDERERS
    from fibsem.applications.autolamella.workflows.tasks import get_tasks

    for name, cls in get_tasks().items():
        for kind in cls.questions:
            assert kind in REVIEW_RENDERERS, (
                f"{name} asks {kind!r}, which nothing renders"
            )
