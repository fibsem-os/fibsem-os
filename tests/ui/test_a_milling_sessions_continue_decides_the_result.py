"""A supervised milling task's closing answer (Accept, or Skip) is the decision
on its result.

The operator watched the mill and answered: that is their judgement on
what the task produced, and it is recorded as theirs. Before this the record
said the producer agreed with itself (``auto:<task>``), and with the review
preference on the task would then have waited a second time in the Review tab
for a look it had already had.

Under Automated there is no session and no question, so nothing changes there.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt5")

from psygnal.containers import EventedDict

from fibsem.applications.autolamella.structures import (
    Attention,
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaWorkflowConfig,
    Experiment,
)
from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.applications.autolamella.workflows.tasks.rough import (
    MillRoughTask,
    MillRoughTaskConfig,
)
from fibsem.milling import FibsemMillingStage
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.workflows.proposals import TASK_RESULT, AuthorKind, DecisionOutcome

ROUGH = "Rough Milling"
MSG = "Run the rough mill."
FIRST = f"{MSG} Press Run Milling to mill, or Skip to go on without milling."


@pytest.fixture
def ui(qapp, monkeypatch, tmp_path):
    """A real AutoLamellaUI (Demo) holding an experiment whose Rough Milling is
    supervised, with the beam time of the mill itself stubbed at the task."""
    from fibsem.applications.autolamella.workflows.tasks import base

    monkeypatch.setattr(base, "run_milling_task", _mill)
    widget = AutoLamellaUI(parent_ui=None)
    widget.system_widget.connect_to_microscope()
    experiment = Experiment(path=tmp_path, name="continue-exp")
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
        widget.microscope.get_microscope_state(),
        EventedDict({ROUGH: MillRoughTaskConfig(task_name=ROUGH)}),
    )
    experiment.positions[0].path.mkdir(parents=True, exist_ok=True)
    widget.experiment = experiment
    yield widget
    if widget.microscope is not None:
        widget.microscope.disconnect()
    widget.close()


def _mill(*args, **kwargs):
    time.sleep(0.05)
    return SimpleNamespace(error=None)


def _task(ui, review_enabled: bool) -> MillRoughTask:
    experiment = ui.experiment
    manager = TaskManager(microscope=ui.microscope, experiment=experiment, parent_ui=ui)
    manager.review_enabled = review_enabled
    lamella = experiment.positions[0]
    task = MillRoughTask(
        microscope=ui.microscope,
        config=lamella.task_config[ROUGH],
        lamella=lamella,
        parent_ui=ui,
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
        if ui.label_instructions.text() == FIRST and ui.pushButton_no.isEnabled():
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


def test_continue_after_the_mill_is_the_operators_decision_on_the_result(ui, qapp):
    task = _task(ui, review_enabled=True)
    thread, outcome = _run_on_worker_thread(task, ui, qapp)

    ui.pushButton_no.click()  # Skip
    _finish(thread, qapp)

    assert "error" not in outcome
    proposal = ui.experiment.positions[0].proposal(ROUGH)
    assert proposal.kind == TASK_RESULT and not proposal.pending
    decision = proposal.current
    assert decision.outcome is DecisionOutcome.Confirmed
    assert decision.author.kind is AuthorKind.human, "theirs, not the producer's"
    assert decision.via == "workflow"
    assert not proposal.to_check, "a person already looked"
    assert not ui.experiment.positions[0].is_awaiting_decision(ROUGH), (
        "the task does not wait a second time for a look it already had"
    )


def test_with_the_preference_off_the_record_is_still_theirs(ui, qapp):
    """The preference hides the Review surface, not the record."""
    task = _task(ui, review_enabled=False)
    thread, outcome = _run_on_worker_thread(task, ui, qapp)

    ui.pushButton_no.click()
    _finish(thread, qapp)

    assert "error" not in outcome
    decision = ui.experiment.positions[0].proposal(ROUGH).current
    assert decision.author.kind is AuthorKind.human


def test_the_mill_the_task_asks_for_writes_under_the_task_s_folder(
    ui, qapp, monkeypatch
):
    """The supervised mill is handed the task's output folder (FIB-1253)."""
    from fibsem.applications.autolamella.workflows.tasks import base

    folders = []

    def run(*args, **kwargs):
        folders.append(kwargs.get("output_dir"))
        return _mill()

    monkeypatch.setattr(base, "run_milling_task", run)
    task = _task(ui, review_enabled=False)
    thread, outcome = _run_on_worker_thread(task, ui, qapp)

    ui.pushButton_yes.click()  # Run Milling
    deadline = time.monotonic() + 10
    while not (folders and ui.pushButton_no.isEnabled()):
        assert time.monotonic() < deadline, "the mill never ran"
        qapp.processEvents()
        time.sleep(0.01)
    ui.pushButton_no.click()  # Accept
    _finish(thread, qapp)

    assert "error" not in outcome
    assert folders == [task.output_dir]
