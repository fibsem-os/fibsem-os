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

from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.applications.autolamella.workflows.tasks.rough import (
    MillRoughTask,
    MillRoughTaskConfig,
)
from fibsem.milling import FibsemMillingStage
from fibsem.milling.tasks import FibsemMillingTaskConfig
from tests.answering import answering
from tests.fixtures.main_window import main_window

ROUGH = "Rough Milling"
MSG = "Run the rough mill."
FIRST = f"{MSG} Press Run Milling to mill, or Skip to go on without milling."


@pytest.fixture
def ui(qapp, tmp_path):
    """A real main window (Demo) holding an experiment whose Rough Milling is
    supervised."""
    with main_window(
        qapp,
        tmp_path,
        "milling-record-exp",
        {ROUGH: MillRoughTaskConfig(task_name=ROUGH)},
        supervised=[ROUGH],
    ) as window:
        yield window


def _config() -> FibsemMillingTaskConfig:
    config = FibsemMillingTaskConfig(name="rough", stages=[FibsemMillingStage()])
    config.alignment.enabled = False  # no drift correction: the mill is the subject
    config.acquisition.acquire_final_image = False
    return config


def _task(ui, window, milling_enabled=True):
    experiment = ui.experiment
    lamella = experiment.positions[0]
    task = MillRoughTask(
        microscope=ui.microscope,
        config=lamella.task_config[ROUGH],
        lamella=lamella,
        task_manager=TaskManager(
            microscope=ui.microscope,
            experiment=experiment,
            responder=window.ui_responder if window is not None else None,
            deciders=window.session.deciders if window is not None else None,
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
        lambda: (
            ui.panel.prompt_bar.message.text() == FIRST
            and ui.panel.prompt_bar.no.isEnabled()
        ),
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
    task = _task(ui, window=ui)
    thread, outcome = _run_on_worker_thread(task, ui, qapp)

    ui.panel.prompt_bar.yes.click()  # Run Milling
    _wait(qapp, lambda: events and ui.panel.prompt_bar.no.isEnabled())
    ui.panel.prompt_bar.no.click()  # Accept
    _finish(thread, qapp)

    assert "error" not in outcome
    assert [run["status"] for run in events] == ["completed"]
    assert events[0]["stages"][0]["status"] == "completed"


def test_skip_records_a_skip(ui, qapp):
    events = _milling_events(ui.microscope)
    task = _task(ui, window=ui)
    thread, outcome = _run_on_worker_thread(task, ui, qapp)

    ui.panel.prompt_bar.no.click()  # Skip
    _finish(thread, qapp)

    assert "error" not in outcome
    (skipped,) = events
    assert skipped["status"] == "skipped"
    assert skipped["reason"] == "skipped at the milling question"
    assert [s["status"] for s in skipped["stages"]] == ["skipped"]


def test_milling_turned_off_records_a_skip_without_a_window(ui):
    events = _milling_events(ui.microscope)

    # Supervised with no window: the task's question waits, so it is answered.
    with answering(ui.experiment):
        _task(ui, window=None, milling_enabled=False).run()

    (skipped,) = events
    assert skipped["status"] == "skipped"
    assert skipped["reason"] == "milling is turned off for this task"
