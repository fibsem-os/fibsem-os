"""The spot burn question and its loop: the task burns, the question decides.

The loop is the task's, on the workflow thread, as the mill's is (FIB-1236):
each pass is a ``spot_burn`` question on the record, it burns on ``run``, and
asks again with the pass counted and the burn's error. The window shows the
settings in the spot burn widget with the pass on the prompt bar, and a button
decides it with the action and the settings (points, current, exposure) as the
widget then held them.

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

from fibsem.acting import TASK, Actor, current_actor
from fibsem.applications.autolamella.structures import (
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.structures import Point
from fibsem.workflows import site_task
from fibsem.workflows.attention import Attention
from fibsem.workflows.proposals import (
    ACTION_EDIT,
    ACTION_RUN,
    ACTION_SKIP,
    SPOT_BURN,
    Decision,
    DecisionOutcome,
)
from fibsem.workflows.site_tasks import spot_burn
from fibsem.workflows.site_tasks.spot_burn import (
    SpotBurnFiducialTask,
    SpotBurnFiducialTaskConfig,
)
from fibsem.workflows.status import HoldKind
from tests.protocols import planned, task_protocol
from tests.run_state import Stopped

SPOT = "Spot Burn Fiducial"
FIRST_HINT = "Press Run Spot Burn to burn, or Skip to go on without burning."
AGAIN_HINT = "Press Burn Again to burn again, or Accept to go on."


@pytest.fixture(scope="module")
def window(qapp, tmp_path_factory):
    """A real main window (Demo): entering spot burn mode arms the point
    overlay on the quad view's controller, which lives there."""
    from fibsem.applications.autolamella.ui import window as module

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
        (site_task, "get_task_supervision"): lambda *_: supervision["supervised"],
    }
    originals = {key: getattr(*key) for key in patches}
    for (module_, name), value in patches.items():
        setattr(module_, name, value)

    window = module.AutoLamellaWindow()
    ui = window
    ui.system_widget.connect_to_microscope()
    # Read unconditionally by the main window's workflow update; the window
    # sets it when a run starts.
    window.run_border.set_state("idle")
    path = tmp_path_factory.mktemp("spot-burn")
    experiment = Experiment(path=path, name="spot-burn-question-exp")
    os.makedirs(experiment.path, exist_ok=True)
    experiment.task_protocol = task_protocol(
        planned(name=SPOT, required=True, attention=Attention.supervised)
    )
    experiment.add_new_lamella(
        ui.microscope.get_microscope_state(),
        EventedDict({SPOT: SpotBurnFiducialTaskConfig(task_name=SPOT)}),
    )
    experiment.positions[0].path.mkdir(parents=True, exist_ok=True)
    ui.experiment = experiment
    window._on_experiment_update()
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
    ui = window
    ui._burns.clear()
    ui._burn_failures.clear()
    ui._supervision["supervised"] = True
    yield ui
    ui.session.runner.manager = None


def _task(ui, points=((0.5, 0.5),)):
    lamella = ui.experiment.positions[0]
    config = lamella.task_config[SPOT]
    config.coordinates = [Point(x=x, y=y) for x, y in points]
    config.milling_current = 100e-12
    config.exposure_time = 7.0
    manager = TaskManager(
        microscope=ui.microscope,
        experiment=ui.experiment,
        responder=ui.ui_responder,
        deciders=ui.session.deciders,
    )
    # The window hears it as it hears a run the runner starts.
    ui.session.runner.relay(manager)
    task = SpotBurnFiducialTask(
        microscope=ui.microscope,
        config=config,
        item=lamella,
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
            ui.panel.prompt_bar.message.text().endswith(hint)
            and ui.panel.prompt_bar.yes.isEnabled()
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


def _pass(ui):
    """The pass up on the prompt bar: the spot burn question being asked."""
    assert ui.proposals._loop_proposal is not None, "no pass is up"
    proposal = ui.proposals._loop_proposal[2]
    assert proposal.kind == SPOT_BURN and proposal.asking
    return proposal


def _current_pass(ui):
    """The spot burn question being asked, wherever it is shown; None when
    none is."""
    lamella = ui.experiment.positions[0]
    asked = [
        p for p in lamella.proposals_from(SPOT) if p.kind == SPOT_BURN and p.asking
    ]
    return asked[-1] if asked else None


def _decide(ui, qapp, action, settings=None, asks_again=True):
    """Decide the pass up as an agent does; with ``asks_again``, wait for the
    next one."""
    asked = _current_pass(ui)
    assert asked is not None
    result = ui.experiment.decide(
        ui.experiment.positions[0].id,
        SPOT,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            actor=Actor.agent("remote"),
            values={} if settings is None else {"spot_burn": settings},
            via="server",
            proposal_id=asked.id,
            action=action,
        ),
    )
    assert result.applied, result.reason
    if asks_again:
        _wait(qapp, lambda: (_current_pass(ui) or asked).id != asked.id)


def _points(settings):
    return [(round(p.x, 6), round(p.y, 6)) for p in settings.coordinates]


def test_run_then_accept_burns_once_on_the_workflow_thread(ui, qapp):
    task, thread, outcome = _start(ui, qapp)

    assert ui.panel.prompt_bar.yes.text() == "Run Spot Burn"
    assert ui.panel.prompt_bar.no.text() == "Skip"
    assert ui.panel.spot_burn_widget._workflow_mode is True
    assert ui.run_relay.hold is not None and ui.run_relay.hold.kind is HoldKind.proposal

    ui.panel.prompt_bar.yes.click()  # run
    _wait_for_prompt(ui, qapp, AGAIN_HINT)
    assert ui.panel.prompt_bar.yes.text() == "Burn Again"
    assert ui.panel.prompt_bar.no.text() == "Accept"
    assert _pass(ui).provenance["passes"] == 1

    ui.panel.prompt_bar.no.click()  # accept
    _finish(thread, qapp)

    assert "error" not in outcome
    (burn,) = ui._burns
    assert burn.actor == TASK
    assert burn.thread is thread
    assert _points(burn.settings) == [(0.5, 0.5)]
    assert burn.settings.exposure_time == 7.0
    # The question cleared the widget on its way out.
    assert ui.panel.spot_burn_widget._workflow_mode is False
    assert ui.panel.spot_burn_widget.get_coordinates() == []
    assert ui.run_relay.hold is None


def test_burn_again_runs_a_second_pass(ui, qapp):
    task, thread, outcome = _start(ui, qapp)

    ui.panel.prompt_bar.yes.click()
    _wait_for_prompt(ui, qapp, AGAIN_HINT)
    ui.panel.prompt_bar.yes.click()  # burn again
    _wait(
        qapp,
        lambda: (
            _current_pass(ui) is not None
            and _current_pass(ui).provenance["passes"] == 2
            and ui.panel.prompt_bar.no.isEnabled()
        ),
    )
    ui.panel.prompt_bar.no.click()
    _finish(thread, qapp)

    assert len(ui._burns) == 2


def test_skip_goes_on_without_a_burn(ui, qapp):
    task, thread, outcome = _start(ui, qapp)

    ui.panel.prompt_bar.no.click()  # skip
    _finish(thread, qapp)

    assert "error" not in outcome
    assert ui._burns == []


def test_a_failed_burn_asks_again_with_the_error(ui, qapp):
    ui._burn_failures.append(RuntimeError("beam blanker"))
    task, thread, outcome = _start(ui, qapp)

    ui.panel.prompt_bar.yes.click()
    _wait_for_prompt(ui, qapp, AGAIN_HINT)
    assert ui.panel.prompt_bar.message.text().startswith(
        "Spot burn failed: beam blanker"
    )
    assert _pass(ui).provenance["last_error"] == "beam blanker"

    ui.panel.prompt_bar.no.click()
    _finish(thread, qapp)
    assert "error" not in outcome


def test_a_run_with_no_points_burns_nothing_and_asks_again(ui, qapp):
    task, thread, outcome = _start(ui, qapp, points=())

    ui.panel.prompt_bar.yes.click()  # run, with nothing placed
    _wait(
        qapp,
        lambda: (
            _current_pass(ui) is not None
            and _current_pass(ui).provenance["last_error"] is not None
            and ui.panel.prompt_bar.no.isEnabled()
        ),
    )
    asked = _current_pass(ui)
    assert asked.provenance["passes"] == 0
    assert "No points to burn" in asked.provenance["last_error"]
    assert ui._burns == []

    ui.panel.prompt_bar.no.click()
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
    """``edit`` takes the changes and asks again; nothing burns. The points,
    current and exposure are in the widget for the operator to see, and in
    what the next run burns."""
    from fibsem.imaging.spot import SpotBurnSettings

    task, thread, outcome = _start(ui, qapp)

    edit = SpotBurnSettings(
        coordinates=[Point(x=0.25, y=0.75), Point(x=0.6, y=0.1)],
        milling_current=3e-10,
        exposure_time=2.5,
    )
    _decide(ui, qapp, ACTION_EDIT, edit)
    _wait_for_prompt(ui, qapp, FIRST_HINT)
    assert ui._burns == []
    shown = ui.panel.spot_burn_widget.get_settings()
    assert _points(shown) == [(0.25, 0.75), (0.6, 0.1)]
    assert shown.milling_current == pytest.approx(3e-10)
    assert shown.exposure_time == pytest.approx(2.5)

    _decide(ui, qapp, ACTION_RUN)
    _wait_for_prompt(ui, qapp, AGAIN_HINT)
    (burn,) = ui._burns
    assert _points(burn.settings) == [(0.25, 0.75), (0.6, 0.1)]
    assert burn.settings.exposure_time == pytest.approx(2.5)

    ui.panel.prompt_bar.no.click()
    _finish(thread, qapp)
    assert "error" not in outcome
    # What was burned is what the protocol keeps.
    assert task.config.exposure_time == pytest.approx(2.5)
    assert len(task.config.coordinates) == 2


def test_a_point_outside_the_frame_is_refused(ui, qapp):
    from fibsem.imaging.spot import SpotBurnSettings

    task, thread, outcome = _start(ui, qapp)
    asked = _pass(ui)
    result = ui.experiment.decide(
        ui.experiment.positions[0].id,
        SPOT,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            actor=Actor.agent("remote"),
            values={
                "spot_burn": SpotBurnSettings(
                    coordinates=[Point(x=1.5, y=0.5)],
                    milling_current=1e-10,
                    exposure_time=2.0,
                )
            },
            via="server",
            proposal_id=asked.id,
            action=ACTION_RUN,
        ),
    )
    assert not result.applied and "outside the frame" in result.reason
    _decide(ui, qapp, ACTION_SKIP, asks_again=False)
    _finish(thread, qapp)
    assert ui._burns == []


def test_accept_keeps_what_was_burned_not_a_later_edit(ui, qapp):
    from dataclasses import replace

    task, thread, outcome = _start(ui, qapp)

    ui.panel.prompt_bar.yes.click()
    _wait_for_prompt(ui, qapp, AGAIN_HINT)
    later = replace(_pass(ui).values["spot_burn"], exposure_time=30.0)
    _decide(ui, qapp, ACTION_EDIT, later)
    _wait_for_prompt(ui, qapp, AGAIN_HINT)
    ui.panel.prompt_bar.no.click()  # accept
    _finish(thread, qapp)

    assert task.config.exposure_time == pytest.approx(7.0)


def test_no_spot_burn_widget_still_asks_on_the_prompt_bar(window, ui, qapp):
    real_widget = ui.panel.spot_burn_widget
    ui.panel.spot_burn_widget = None
    try:
        task, thread, outcome = _start(ui, qapp, wait_for=None)
        _wait(
            qapp,
            lambda: (
                _current_pass(ui) is not None
                and ui.proposals._loop_proposal is not None
            ),
            why="the pass was not put on the prompt bar",
        )
        assert ui.panel.prompt_bar.no.text() == "Skip"
        ui.panel.prompt_bar.no.click()
        _finish(thread, qapp)
    finally:
        ui.panel.spot_burn_widget = real_widget

    assert "error" not in outcome
    assert ui._burns == []


def test_a_stop_interrupts_the_prompt(ui, qapp):
    task, thread, outcome = _start(ui, qapp)

    task._manager.stop()
    ui.session.runner.manager = Stopped()
    _finish(thread, qapp)

    assert isinstance(outcome.get("error"), InterruptedError)
    assert ui._burns == []
