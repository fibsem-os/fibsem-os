"""A task's milling question leaves a record of what it milled (FIB-1257).

Each run records how it ended (``FibsemMillingTask``). A task that milled
nothing records that too, with why: Continue pressed without running, or
milling turned off for the task. Before, the record was silent either way, and
a task that skipped its mill looked like one that never got there.

The mill is the real one, on the Demo microscope, run by the task on the
workflow thread as the app runs it.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")

import threading
import time

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

ROUGH = "Rough Milling"
MSG = "Run the rough mill."
FIRST = f"{MSG} Press Run Milling to mill, or Skip to go on without milling."


@pytest.fixture
def ui(qapp, tmp_path):
    """A real AutoLamellaUI (Demo) holding an experiment whose Rough Milling is
    supervised."""
    widget = AutoLamellaUI(parent_ui=None)
    widget.system_widget.connect_to_microscope()
    experiment = Experiment(path=tmp_path, name="milling-record-exp")
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


def _config() -> FibsemMillingTaskConfig:
    config = FibsemMillingTaskConfig(name="rough", stages=[FibsemMillingStage()])
    config.alignment.enabled = False  # no drift correction: the mill is the subject
    config.acquisition.acquire_final_image = False
    return config


def _task(ui, parent_ui, milling_enabled=True):
    experiment = ui.experiment
    lamella = experiment.positions[0]
    task = MillRoughTask(
        microscope=ui.microscope,
        config=lamella.task_config[ROUGH],
        lamella=lamella,
        parent_ui=parent_ui,
        task_manager=TaskManager(
            microscope=ui.microscope, experiment=experiment, parent_ui=parent_ui
        ),
    )

    def _run():
        task.update_milling_config_ui(
            _config(), msg=MSG, milling_enabled=milling_enabled
        )

    task._run = _run  # type: ignore[method-assign]
    return task


def _milling_events(microscope):
    events = []
    microscope.session.record_signal.connect(
        lambda kind, payload: events.append(payload) if kind == "milling" else None
    )
    return events


def _run_on_worker_thread(task, ui, qapp):
    outcome = {}

    def target():
        try:
            task.run()
        except Exception as exc:  # noqa: BLE001 - the test inspects it
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    _wait(
        qapp,
        lambda: ui.label_instructions.text() == FIRST and ui.pushButton_no.isEnabled(),
    )
    return thread, outcome


def _wait(qapp, ready, timeout_s=20.0):
    deadline = time.monotonic() + timeout_s
    while not ready():
        assert time.monotonic() < deadline, "timed out"
        qapp.processEvents()
        time.sleep(0.01)


def _finish(thread, qapp):
    _wait(qapp, lambda: not thread.is_alive())
    thread.join(timeout=1.0)


def test_a_run_then_accept_records_the_run_and_no_skip(ui, qapp):
    events = _milling_events(ui.microscope)
    task = _task(ui, parent_ui=ui)
    thread, outcome = _run_on_worker_thread(task, ui, qapp)

    ui.pushButton_yes.click()  # Run Milling
    _wait(qapp, lambda: events and ui.pushButton_no.isEnabled())
    ui.pushButton_no.click()  # Accept
    _finish(thread, qapp)

    assert "error" not in outcome
    assert [run["status"] for run in events] == ["completed"]
    assert events[0]["stages"][0]["status"] == "completed"


def test_skip_records_a_skip(ui, qapp):
    events = _milling_events(ui.microscope)
    task = _task(ui, parent_ui=ui)
    thread, outcome = _run_on_worker_thread(task, ui, qapp)

    ui.pushButton_no.click()  # Skip
    _finish(thread, qapp)

    assert "error" not in outcome
    (skipped,) = events
    assert skipped["status"] == "skipped"
    assert skipped["reason"] == "skipped at the milling question"
    assert [s["status"] for s in skipped["stages"]] == ["skipped"]


def test_milling_turned_off_records_a_skip_without_a_window(ui):
    events = _milling_events(ui.microscope)

    _task(ui, parent_ui=None, milling_enabled=False).run()

    (skipped,) = events
    assert skipped["status"] == "skipped"
    assert skipped["reason"] == "milling is turned off for this task"
