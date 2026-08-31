"""The milling question over the Responder seam: RunMillingTask.

The old mill loop straddled threads four ways: ``ask_user(mill=...)`` parked on
the polled flag; the workflow thread emitted ``start_milling_signal`` over a
BlockingQueuedConnection; it sleep-polled ``widget.is_milling`` across the seam;
and it read ``get_config()`` back at the end. Now the whole loop — prompt, run
on Run Milling, wait for the widget's ``finished_milling_signal``, re-prompt,
read back and clear on Continue — lives in ``QtResponder`` on the GUI thread,
and the workflow blocks on one future whose answer is the config as used.

``run_milling_task`` is stubbed at the milling widget's import site: the run
still goes through the widget's real thread, buttons and finished signal — only
the beam time is gone.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI
from fibsem.applications.autolamella.workflows.interaction import (
    RunMillingTask,
    ask,
)
from fibsem.milling import FibsemMillingStage
from fibsem.milling.tasks import FibsemMillingTaskConfig

MSG = "Check the patterns, then run."


def _config(name: str) -> FibsemMillingTaskConfig:
    # One real stage: the widget's worker refuses a config with none enabled.
    return FibsemMillingTaskConfig(name=name, stages=[FibsemMillingStage()])


@pytest.fixture
def ui(qapp, monkeypatch):
    """A real AutoLamellaUI, connected (Demo), with the mill run itself stubbed."""
    from fibsem.ui.widgets import milling_widget as mw

    runs = []

    def fake_run_milling_task(microscope, config, parent_ui=None, **kwargs):
        runs.append(config)
        time.sleep(0.05)  # long enough for is_milling to be observable

    monkeypatch.setattr(mw, "run_milling_task", fake_run_milling_task)
    widget = AutoLamellaUI(parent_ui=None)
    widget.system_widget.connect_to_microscope()
    widget._mill_runs = runs  # for the tests to inspect
    yield widget
    if widget.microscope is not None:
        widget.microscope.disconnect()
    widget.close()


def _ask_on_worker_thread(ui, qapp, request, wait_for_prompt=True):
    outcome = {}

    def target():
        try:
            outcome["config"] = ask(
                ui.ui_responder, request, abort=ui._workflow_stop_event.is_set
            )
        except Exception as exc:  # noqa: BLE001 - the test inspects it
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    if wait_for_prompt:
        _wait_for_prompt(ui, qapp, request.message)
    return thread, outcome


def _wait_for_prompt(ui, qapp, msg, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if ui.label_instructions.text() == msg and ui.pushButton_yes.isEnabled():
            return
        time.sleep(0.01)
    raise AssertionError(f"the prompt {msg!r} never appeared")


def _finish(thread, qapp, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while thread.is_alive() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    thread.join(timeout=1.0)
    assert not thread.is_alive(), "the asker never returned"


def test_run_then_continue_runs_once_and_answers_the_editor_config(ui, qapp):
    request = RunMillingTask(config=_config("rough-mill"), message=MSG)
    thread, outcome = _ask_on_worker_thread(ui, qapp, request)

    assert ui.pushButton_yes.text() == "Run Milling"
    assert ui.pushButton_no.text() == "Continue"
    assert ui.WAITING_FOR_USER_INTERACTION is True

    ui.pushButton_yes.click()  # run
    # The prompt is down while the mill runs, and comes back when it finishes.
    _wait_for_prompt(ui, qapp, MSG)
    assert len(ui._mill_runs) == 1

    ui.pushButton_no.click()  # continue
    _finish(thread, qapp)

    assert "error" not in outcome
    assert isinstance(outcome["config"], FibsemMillingTaskConfig)
    assert outcome["config"].name == "rough-mill"
    assert ui.WAITING_FOR_USER_INTERACTION is False
    # The old handshakes are gone: nothing touched the polled flags.
    assert ui.WAITING_FOR_UI_UPDATE is False


def test_continue_without_running_answers_without_a_mill(ui, qapp):
    request = RunMillingTask(config=_config("skip"), message=MSG)
    thread, outcome = _ask_on_worker_thread(ui, qapp, request)

    ui.pushButton_no.click()
    _finish(thread, qapp)

    assert "error" not in outcome
    assert outcome["config"].name == "skip"
    assert ui._mill_runs == []


def test_unsupervised_runs_once_without_a_prompt(ui, qapp):
    request = RunMillingTask(config=_config("auto"), confirm=False, message=MSG)
    thread, outcome = _ask_on_worker_thread(ui, qapp, request, wait_for_prompt=False)

    _finish(thread, qapp)

    assert "error" not in outcome
    assert outcome["config"].name == "auto"
    assert len(ui._mill_runs) == 1
    # No prompt was ever shown for it.
    assert ui.WAITING_FOR_USER_INTERACTION is False


def test_disabled_milling_only_confirms_the_patterns(ui, qapp):
    request = RunMillingTask(config=_config("preview"), enabled=False, message=MSG)
    thread, outcome = _ask_on_worker_thread(ui, qapp, request)

    # Confirm-only: one button, and the positive click must not start a mill.
    assert ui.pushButton_yes.text() == "Continue"
    ui.pushButton_yes.click()
    _finish(thread, qapp)

    assert "error" not in outcome
    assert outcome["config"].name == "preview"
    assert ui._mill_runs == []


def test_the_answer_is_the_editors_config_not_the_requests(ui, qapp):
    request = RunMillingTask(config=_config("sent"), message=MSG)
    thread, outcome = _ask_on_worker_thread(ui, qapp, request)

    # The operator edits while the prompt is up; the answer must be the editor's.
    ui.milling_task_config_widget.update_from_settings(_config("edited-by-operator"))
    ui.pushButton_no.click()
    _finish(thread, qapp)

    assert outcome["config"].name == "edited-by-operator"


def test_a_stop_interrupts_the_prompt(ui, qapp):
    request = RunMillingTask(config=_config("stop"), message=MSG)
    thread, outcome = _ask_on_worker_thread(ui, qapp, request)

    ui._workflow_stop_event.set()
    try:
        _finish(thread, qapp)
    finally:
        ui._workflow_stop_event.clear()

    assert isinstance(outcome.get("error"), InterruptedError)


def test_a_stop_during_the_mill_does_not_resurrect_the_prompt(ui, qapp):
    # The bug from bench testing: Stop mid-mill aborted the waiter, but the
    # responder still held the question — so when the cancelled mill finished,
    # the prompt reappeared over a cleared editor, for nobody. Two defences now:
    # an abandoned ask cancels its future (a finished mill drops a cancelled
    # question), and the workflow-finished sweep abandons whatever the race
    # still managed to re-park before the cancel landed.
    request = RunMillingTask(config=_config("stopped-mid-mill"), message=MSG)
    thread, outcome = _ask_on_worker_thread(ui, qapp, request)

    ui.pushButton_yes.click()  # run
    ui._workflow_stop_event.set()
    try:
        _finish(thread, qapp)
    finally:
        ui._workflow_stop_event.clear()
    assert isinstance(outcome.get("error"), InterruptedError)

    # Let the (still running) stubbed mill finish and its signal land — this is
    # the window where the resurrect happened.
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)

    # What production does next: the workflow worker exits and fires the
    # finished signal, whose handler abandons anything the run left behind.
    ui._workflow_finished_signal.emit(True)
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)

    assert ui.label_instructions.text() != MSG, "the aborted prompt came back"
    assert ui.WAITING_FOR_USER_INTERACTION is False


def test_a_click_after_a_stop_starts_nothing(ui, qapp):
    # Stop while the prompt stands, then a click lands anyway (the buttons are
    # still up until the next status arrives). It must only take the stale
    # prompt down — not start a mill for a question nobody is waiting on.
    request = RunMillingTask(config=_config("stale-click"), message=MSG)
    thread, outcome = _ask_on_worker_thread(ui, qapp, request)

    ui._workflow_stop_event.set()
    try:
        _finish(thread, qapp)
    finally:
        ui._workflow_stop_event.clear()
    assert isinstance(outcome.get("error"), InterruptedError)

    ui.pushButton_yes.click()  # "Run Milling", but the asker is gone
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)

    assert ui._mill_runs == []
    assert ui.label_instructions.text() == ""
