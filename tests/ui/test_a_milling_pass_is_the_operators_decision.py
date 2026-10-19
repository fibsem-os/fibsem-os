"""A supervised milling task's answer at the mill (Run, Accept, or Skip) is the
operator's decision on that pass, recorded as theirs. The task proposes
nothing after it ends.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt5")

from fibsem.acting import ActorKind
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.applications.autolamella.workflows.tasks.rough import (
    MillRoughTask,
    MillRoughTaskConfig,
)
from fibsem.milling import FibsemMillingStage
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.workflows.proposals import ACTION_SKIP, MILLING, DecisionOutcome
from tests.fixtures.main_window import main_window

ROUGH = "Rough Milling"
MSG = "Run the rough mill."
FIRST = f"{MSG} Press Run Milling to mill, or Skip to go on without milling."


@pytest.fixture
def ui(qapp, monkeypatch, tmp_path):
    """A real main window (Demo) holding an experiment whose Rough Milling is
    supervised, with the beam time of the mill itself stubbed at the task."""
    from fibsem.workflows import task as task_module

    monkeypatch.setattr(task_module, "run_milling_task", _mill)
    with main_window(
        qapp,
        tmp_path,
        "continue-exp",
        {ROUGH: MillRoughTaskConfig(task_name=ROUGH)},
        supervised=[ROUGH],
    ) as window:
        yield window


def _mill(*args, **kwargs):
    time.sleep(0.05)
    return SimpleNamespace(error=None)


def _task(ui) -> MillRoughTask:
    experiment = ui.experiment
    manager = TaskManager(
        microscope=ui.microscope,
        experiment=experiment,
        responder=ui.ui_responder,
        deciders=ui.session.deciders,
    )
    lamella = experiment.positions[0]
    task = MillRoughTask(
        microscope=ui.microscope,
        config=lamella.task_config[ROUGH],
        lamella=lamella,
        task_manager=manager,
    )
    config = FibsemMillingTaskConfig(name="rough", stages=[FibsemMillingStage()])

    def _run():
        task.update_milling_config_ui(config, msg=MSG)

    task._run = _run  # type: ignore[method-assign]
    return task


def _run_on_worker_thread(task, ui, qapp):
    outcome = {}

    def target():
        try:
            task.run()
        except Exception as exc:  # noqa: BLE001 - the test inspects it
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        qapp.processEvents()
        if (
            ui.panel.prompt_bar.message.text() == FIRST
            and ui.panel.prompt_bar.no.isEnabled()
        ):
            return thread, outcome
        time.sleep(0.01)
    raise AssertionError("the mill prompt never appeared")


def _finish(thread, qapp):
    deadline = time.monotonic() + 10
    while thread.is_alive() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    thread.join(timeout=1.0)
    assert not thread.is_alive()


def test_skip_is_the_operators_decision_on_the_pass(ui, qapp):
    task = _task(ui)
    thread, outcome = _run_on_worker_thread(task, ui, qapp)

    ui.panel.prompt_bar.no.click()  # Skip
    _finish(thread, qapp)

    assert "error" not in outcome
    (proposal,) = ui.experiment.positions[0].proposals_from(ROUGH)
    assert proposal.kind == MILLING and not proposal.pending
    decision = proposal.current
    assert decision.outcome is DecisionOutcome.Confirmed
    assert decision.action == ACTION_SKIP
    assert decision.actor.kind is ActorKind.operator, "theirs, not the task's"
    assert decision.via == "workflow"
    assert not proposal.unreviewed, "a person decided it"
    assert ui.experiment.positions[0].latest_run_completed(ROUGH)


def test_the_mill_the_task_asks_for_writes_under_the_task_s_folder(
    ui, qapp, monkeypatch
):
    """The supervised mill is handed the task's output folder (FIB-1253)."""
    from fibsem.workflows import task as task_module

    folders = []

    def run(*args, **kwargs):
        folders.append(kwargs.get("output_dir"))
        return _mill()

    monkeypatch.setattr(task_module, "run_milling_task", run)
    task = _task(ui)
    thread, outcome = _run_on_worker_thread(task, ui, qapp)

    ui.panel.prompt_bar.yes.click()  # Run Milling
    deadline = time.monotonic() + 10
    while not (folders and ui.panel.prompt_bar.no.isEnabled()):
        assert time.monotonic() < deadline, "the mill never ran"
        qapp.processEvents()
        time.sleep(0.01)
    ui.panel.prompt_bar.no.click()  # Accept
    _finish(thread, qapp)

    assert "error" not in outcome
    assert folders == [task.output_dir]
