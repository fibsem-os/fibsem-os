"""The spot burn question and its loop: the task burns, the question decides.

The loop is the task's, on the workflow thread, as the mill's is (FIB-1236):
it asks :class:`RunSpotBurn`, burns on ``run``, and asks again with the pass
counted and the burn's error. The responder only shows the settings in the
spot burn widget and answers with a :class:`SpotBurnDecision`: the action and
the settings (points, current, exposure) as the widget then held them.

``run_spot_burn`` and the image after it are stubbed at the task's import
site: the loop, the prompt, the widget and the answers are real, only the beam
time is gone.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt5")

from psygnal.containers import EventedDict

from fibsem.acting import TASK, current_actor
from fibsem.applications.autolamella.structures import (
    Attention,
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaWorkflowConfig,
    Experiment,
)
from fibsem.applications.autolamella.workflows.interaction import (
    ACTION_EDIT,
    ACTION_RUN,
    SpotBurnEdit,
)
from fibsem.applications.autolamella.workflows.tasks import base, spot_burn
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.applications.autolamella.workflows.tasks.spot_burn import (
    SpotBurnFiducialTask,
    SpotBurnFiducialTaskConfig,
)
from fibsem.structures import Point
from fibsem.workflows.status import HoldKind

SPOT = "Spot Burn Fiducial"
FIRST_HINT = "Press Run Spot Burn to burn, or Skip to go on without burning."
AGAIN_HINT = "Press Burn Again to burn again, or Accept to go on."


@pytest.fixture(scope="module")
def window(qapp, tmp_path_factory):
    """A real main window (Demo): entering spot burn mode arms the point
    overlay on the quad view's controller, which lives there."""
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module

    burns, failures, supervision = [], [], {"supervised": True}

    def fake_run_spot_burn(microscope, settings, beam_type, stop_event=None):
        burns.append(
            SimpleNamespace(
                settings=settings,
                actor=current_actor(),
                thread=threading.current_thread(),
            )
        )
        time.sleep(0.05)
        if failures:
            raise failures.pop(0)

    patches = {
        (spot_burn, "run_spot_burn"): fake_run_spot_burn,
        (spot_burn, "show_after_burn"): lambda microscope: None,
        (base, "get_task_supervision"): lambda *_: supervision["supervised"],
    }
    originals = {key: getattr(*key) for key in patches}
    for (module_, name), value in patches.items():
        setattr(module_, name, value)

    window = module.AutoLamellaSingleWindowUI()
    ui = window.autolamella_ui
    ui.system_widget.connect_to_microscope()
    # Read unconditionally by the main window's workflow update; first set on
    # the Run click.
    window._set_border_state("idle")
    path = tmp_path_factory.mktemp("spot-burn")
    experiment = Experiment(path=path, name="spot-burn-question-exp")
    os.makedirs(experiment.path, exist_ok=True)
    experiment.task_protocol = AutoLamellaTaskProtocol(
        workflow_config=AutoLamellaWorkflowConfig(
            tasks=[
                AutoLamellaTaskDescription(
                    name=SPOT, required=True, attention=Attention.supervised
                )
            ]
        )
    )
    experiment.add_new_lamella(
        ui.microscope.get_microscope_state(),
        EventedDict({SPOT: SpotBurnFiducialTaskConfig(task_name=SPOT)}),
    )
    experiment.positions[0].path.mkdir(parents=True, exist_ok=True)
    ui.experiment = experiment
    ui._burns = burns
    ui._burn_failures = failures
    ui._supervision = supervision
    yield window
    for (module_, name), value in originals.items():
        setattr(module_, name, value)
    if ui.microscope is not None:
        ui.microscope.disconnect()
    # closeEvent ends in app.quit(); on the shared test QApplication that latches
    # and breaks every later QEventLoop.exec_(): close with quit stubbed out.
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        window.close()
    finally:
        qapp.quit = original_quit


@pytest.fixture
def ui(window):
    ui = window.autolamella_ui
    ui._burns.clear()
    ui._burn_failures.clear()
    ui._supervision["supervised"] = True
    yield ui
    ui._workflow_stop_event.clear()


def _task(ui, points=((0.5, 0.5),)):
    lamella = ui.experiment.positions[0]
    config = lamella.task_config[SPOT]
    config.coordinates = [Point(x=x, y=y) for x, y in points]
    config.milling_current = 100e-12
    config.exposure_time = 7.0
    manager = TaskManager(
        microscope=ui.microscope, experiment=ui.experiment, parent_ui=ui
    )
    task = SpotBurnFiducialTask(
        microscope=ui.microscope,
        config=config,
        lamella=lamella,
        parent_ui=ui,
        task_manager=manager,
    )
    task._manager = manager
    return task


def _start(ui, qapp, points=((0.5, 0.5),), wait_for=FIRST_HINT):
    """Run the task's burn-and-ask loop on a worker thread, as the workflow
    does; wait for a prompt ending ``wait_for`` (None: don't)."""
    task = _task(ui, points)
    outcome = {}
    task._run = task.update_spot_burn_parameters_ui  # type: ignore[method-assign]

    def target():
        try:
            task.run()  # as the workflow runs it: for the task (FIB-1062)
        except Exception as exc:  # noqa: BLE001 - the test inspects it
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    if wait_for is not None:
        _wait_for_prompt(ui, qapp, wait_for)
    return task, thread, outcome


def _wait_for_prompt(ui, qapp, hint, timeout_s=10.0):
    _wait(
        qapp,
        lambda: (
            ui.label_instructions.text().endswith(hint)
            and ui.pushButton_yes.isEnabled()
        ),
        timeout_s,
        f"the prompt {hint!r} never appeared",
    )


def _wait(qapp, ready, timeout_s=10.0, why="timed out"):
    deadline = time.monotonic() + timeout_s
    while not ready():
        assert time.monotonic() < deadline, why
        qapp.processEvents()
        time.sleep(0.01)


def _finish(thread, qapp, timeout_s=10.0):
    _wait(qapp, lambda: not thread.is_alive(), timeout_s, "the asker never returned")
    thread.join(timeout=1.0)


def _question_id(ui):
    _request, question_id = ui.ui_responder.pending_question_and_id()
    assert question_id is not None
    return question_id


def _answer(ui, qapp, value, clicked_yes=True, asks_again=True):
    """Answer the pending question with ``value``, as an agent does; with
    ``asks_again``, wait for the next question."""
    question_id = _question_id(ui)
    answered = ui.ui_responder.submit_answer(clicked_yes, question_id, value=value)
    assert answered.result(timeout=5) is True
    if asks_again:
        _wait(
            qapp,
            lambda: (
                ui.ui_responder.pending_question_and_id()[1] not in (None, question_id)
            ),
        )
    return answered


def _points(settings):
    return [(round(p.x, 6), round(p.y, 6)) for p in settings.coordinates]


def test_run_then_accept_burns_once_on_the_workflow_thread(ui, qapp):
    task, thread, outcome = _start(ui, qapp)

    assert ui.pushButton_yes.text() == "Run Spot Burn"
    assert ui.pushButton_no.text() == "Skip"
    assert ui.spot_burn_widget._workflow_mode is True
    assert ui.hold is not None and ui.hold.kind is HoldKind.question

    ui.pushButton_yes.click()  # run
    _wait_for_prompt(ui, qapp, AGAIN_HINT)
    assert ui.pushButton_yes.text() == "Burn Again"
    assert ui.pushButton_no.text() == "Accept"
    assert ui.ui_responder.pending_question().passes == 1

    ui.pushButton_no.click()  # accept
    _finish(thread, qapp)

    assert "error" not in outcome
    (burn,) = ui._burns
    assert burn.actor == TASK
    assert burn.thread is thread
    assert _points(burn.settings) == [(0.5, 0.5)]
    assert burn.settings.exposure_time == 7.0
    # The question cleared the widget on its way out.
    assert ui.spot_burn_widget._workflow_mode is False
    assert ui.spot_burn_widget.get_coordinates() == []
    assert ui.hold is None


def test_burn_again_runs_a_second_pass(ui, qapp):
    task, thread, outcome = _start(ui, qapp)

    ui.pushButton_yes.click()
    _wait_for_prompt(ui, qapp, AGAIN_HINT)
    ui.pushButton_yes.click()  # burn again
    _wait(
        qapp,
        lambda: (
            ui.ui_responder.pending_question() is not None
            and ui.ui_responder.pending_question().passes == 2
        ),
    )
    ui.pushButton_no.click()
    _finish(thread, qapp)

    assert len(ui._burns) == 2


def test_skip_goes_on_without_a_burn(ui, qapp):
    task, thread, outcome = _start(ui, qapp)

    ui.pushButton_no.click()  # skip
    _finish(thread, qapp)

    assert "error" not in outcome
    assert ui._burns == []


def test_a_failed_burn_asks_again_with_the_error(ui, qapp):
    ui._burn_failures.append(RuntimeError("beam blanker"))
    task, thread, outcome = _start(ui, qapp)

    ui.pushButton_yes.click()
    _wait_for_prompt(ui, qapp, AGAIN_HINT)
    assert ui.label_instructions.text().startswith("Spot burn failed: beam blanker")
    assert ui.ui_responder.pending_question().last_error == "beam blanker"

    ui.pushButton_no.click()
    _finish(thread, qapp)
    assert "error" not in outcome


def test_a_run_with_no_points_burns_nothing_and_asks_again(ui, qapp):
    task, thread, outcome = _start(ui, qapp, points=())

    ui.pushButton_yes.click()  # run, with nothing placed
    _wait(
        qapp,
        lambda: (
            ui.ui_responder.pending_question() is not None
            and ui.ui_responder.pending_question().last_error is not None
        ),
    )
    request = ui.ui_responder.pending_question()
    assert request.passes == 0
    assert "No points to burn" in request.last_error
    assert ui._burns == []

    ui.pushButton_no.click()
    _finish(thread, qapp)
    assert "error" not in outcome


def test_unsupervised_burns_once_without_a_prompt(ui, qapp):
    ui._supervision["supervised"] = False
    task, thread, outcome = _start(ui, qapp, wait_for=None)
    _finish(thread, qapp)

    assert "error" not in outcome
    assert len(ui._burns) == 1


def test_unsupervised_with_no_points_skips(ui, qapp):
    ui._supervision["supervised"] = False
    task, thread, outcome = _start(ui, qapp, points=(), wait_for=None)
    _finish(thread, qapp)

    assert "error" not in outcome
    assert ui._burns == []


def test_unsupervised_a_failed_burn_fails_the_task(ui, qapp):
    ui._supervision["supervised"] = False
    ui._burn_failures.append(RuntimeError("beam blanker"))
    task, thread, outcome = _start(ui, qapp, wait_for=None)
    _finish(thread, qapp)

    assert isinstance(outcome.get("error"), RuntimeError)


def test_an_agents_edit_lands_in_the_widget_and_asks_again(ui, qapp):
    """``edit`` makes the changes and asks again; nothing burns. The points,
    current and exposure are in the widget for the operator to see, and in
    what the next run burns."""
    task, thread, outcome = _start(ui, qapp)

    edit = SpotBurnEdit(
        action=ACTION_EDIT,
        points=[Point(x=0.25, y=0.75), Point(x=0.6, y=0.1)],
        milling_current=3e-10,
        exposure_time=2.5,
    )
    answered = _answer(ui, qapp, edit)
    assert answered.adjusted is True
    assert ui._burns == []
    shown = ui.spot_burn_widget.get_settings()
    assert _points(shown) == [(0.25, 0.75), (0.6, 0.1)]
    assert shown.milling_current == pytest.approx(3e-10)
    assert shown.exposure_time == pytest.approx(2.5)

    _answer(ui, qapp, SpotBurnEdit(action=ACTION_RUN))
    _wait_for_prompt(ui, qapp, AGAIN_HINT)
    (burn,) = ui._burns
    assert _points(burn.settings) == [(0.25, 0.75), (0.6, 0.1)]
    assert burn.settings.exposure_time == pytest.approx(2.5)

    ui.pushButton_no.click()
    _finish(thread, qapp)
    assert "error" not in outcome
    # What was burned is what the protocol keeps.
    assert task.config.exposure_time == pytest.approx(2.5)
    assert len(task.config.coordinates) == 2


def test_accept_keeps_what_was_burned_not_a_later_edit(ui, qapp):
    task, thread, outcome = _start(ui, qapp)

    ui.pushButton_yes.click()
    _wait_for_prompt(ui, qapp, AGAIN_HINT)
    _answer(ui, qapp, SpotBurnEdit(action=ACTION_EDIT, exposure_time=30.0))
    ui.pushButton_no.click()  # accept
    _finish(thread, qapp)

    assert task.config.exposure_time == pytest.approx(7.0)


def test_the_live_settings_are_what_a_peek_reads(ui, qapp):
    task, thread, outcome = _start(ui, qapp)

    question_id, live = ui.ui_responder.peek_live_answer().result(timeout=5)
    assert question_id == _question_id(ui)
    assert _points(live) == [(0.5, 0.5)]
    assert live.exposure_time == 7.0

    # Echoing the points that stand is not adjusting them.
    echo = SpotBurnEdit(action="skip", points=live.coordinates)
    answered = _answer(ui, qapp, echo, clicked_yes=False, asks_again=False)
    assert answered.adjusted is False
    _finish(thread, qapp)
    assert ui._burns == []


def test_no_spot_burn_widget_goes_on_without_a_burn(ui, qapp):
    real_widget = ui.spot_burn_widget
    ui.spot_burn_widget = None
    try:
        task, thread, outcome = _start(ui, qapp, wait_for=None)
        _finish(thread, qapp)
    finally:
        ui.spot_burn_widget = real_widget

    assert "error" not in outcome
    assert ui._burns == []


def test_a_stop_interrupts_the_prompt(ui, qapp):
    task, thread, outcome = _start(ui, qapp)

    task._manager.stop()
    ui._workflow_stop_event.set()
    _finish(thread, qapp)

    assert isinstance(outcome.get("error"), InterruptedError)
    assert ui._burns == []
