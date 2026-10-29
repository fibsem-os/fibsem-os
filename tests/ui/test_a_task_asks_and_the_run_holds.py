"""A supervised task that asks with ``ask`` holds the run until the decision.

The hold is the task manager's, not the responder's: the task records the
question, the manager raises the hold, the main window puts the question up
on its widget, and the decision -- from Continue, from an agent -- releases
the task with the decided values. Stop withdraws the question. No prompt, no adapter.
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
from fibsem.applications.autolamella.workflows.tasks.rough import (
    MillRoughTask,
    MillRoughTaskConfig,
)
from fibsem.structures.settings import Point
from fibsem.workflows.attention import Attention
from fibsem.workflows.proposals import (
    DETECTION,
    POINT_OF_INTEREST,
    Decision,
    DecisionOutcome,
)
from fibsem.workflows.status import HoldKind, TaskStatus
from tests.protocols import planned, task_protocol

ROUGH = "Rough Milling"
PROPOSED = {"features": [{"name": "LamellaCentre", "px": Point(10, 20)}]}


class _AsksForADetection(MillRoughTask):
    asks = (DETECTION, POINT_OF_INTEREST)


@pytest.fixture
def window(qapp, tmp_path):
    from fibsem.applications.autolamella.ui import window as module

    win = module.AutoLamellaWindow()
    win.system_widget.connect_to_microscope()
    ui = win
    experiment = Experiment(path=tmp_path, name="ask-exp")
    os.makedirs(experiment.path, exist_ok=True)
    experiment.task_protocol = task_protocol(
        planned(name=ROUGH, required=True, attention=Attention.supervised)
    )
    experiment.add_new_lamella(
        ui.microscope.get_microscope_state(),
        EventedDict({ROUGH: MillRoughTaskConfig(task_name=ROUGH)}),
    )
    experiment.positions[0].path.mkdir(parents=True, exist_ok=True)
    ui.experiment = experiment
    win._on_experiment_update()
    win.tab_widget.setCurrentIndex(0)
    win._started = []  # (manager, thread) pairs, stopped before teardown

    yield win
    # A task still waiting on its question must be released before the
    # window goes: its hold and its notification both land on widgets that
    # are about to be deleted.
    for manager, thread in win._started:
        manager.stop()
        _pump(qapp, lambda: not thread.is_alive())
    ui.session.disconnect()
    close_window(win)
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


def _task(window):
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
    task = _AsksForADetection(
        microscope=ui.microscope,
        config=lamella.task_config[ROUGH],
        lamella=lamella,
        task_manager=manager,
    )
    return task, manager


def _start(window, kind=DETECTION, values=PROPOSED):
    """Run a real task on a worker thread whose body asks and keeps the answer."""
    task, manager = _task(window)
    seen = {}

    def _run():
        seen["decision"] = task.ask(kind, values, image="ml-1_ib.tif")

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
    ui = window
    lamella = ui.experiment.positions[0]
    task, thread, seen = _start(window)

    assert _pump(qapp, lambda: ui.run_relay.hold is not None), "the run is held"
    assert ui.run_relay.hold.kind is HoldKind.proposal
    assert ui.run_relay.hold.items == (f"{lamella.name}/{ROUGH}",)
    # The hold is set from the worker thread; the notification that puts the
    # question up is queued to this one, so it may land a moment later.
    assert _pump(qapp, lambda: ui.proposals._widget_proposal is not None), "put up"
    assert ui.panel.prompt_bar.yes.text() == "Continue"
    if ui.panel.det_widget is not None:
        assert ui.proposals.proposal_host() is ui.panel.det_widget, (
            "on the Detection tab"
        )
    assert _pump(
        qapp,
        lambda: (
            "waiting for you to answer" in ui.panel.label_workflow_information.text()
        ),
    ), ui.panel.label_workflow_information.text()
    proposal = lamella.proposal(ROUGH)
    assert proposal.kind == DETECTION and proposal.pending and proposal.asking
    assert thread.is_alive(), "stopped on its next line"

    moved = {"features": [{"name": "LamellaCentre", "px": Point(12, 26)}]}
    result = ui.experiment.decide(
        lamella.id,
        ROUGH,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            actor="operator:op",
            values=moved,
            proposal_id=proposal.id,
        ),
    )
    assert result.applied, result.reason
    assert _pump(qapp, lambda: not thread.is_alive()), "released"

    decision = seen["decision"]
    assert decision.outcome is DecisionOutcome.Confirmed
    assert decision.values["features"][0]["px"] == Point(12, 26), "the decided value"
    assert decision.actor.kind is ActorKind.operator
    assert ui.run_relay.hold is None
    assert "error" not in seen
    assert _pump(qapp, lambda: ui.proposals._widget_proposal is None), (
        "the prompt came down"
    )
    # The proposal is decided, and the task proposes nothing when it ends.
    (question,) = lamella.proposals_from(ROUGH)
    assert question is proposal
    assert not question.pending
    assert lamella.task_state.status is TaskStatus.Completed


def test_a_window_that_starts_listening_mid_hold_puts_the_question_up(window, qapp):
    """A run can be asking before the window listens to its experiment (a run
    started without it). The question already standing is put up at once."""
    ui = window
    lamella = ui.experiment.positions[0]
    task, thread, seen = _start(window)
    assert _pump(qapp, lambda: ui.proposals._widget_proposal is not None)
    experiment = ui.experiment
    window.attention.listen(None)
    ui.proposals.clear_widget_proposal(lamella.id, ROUGH)
    assert ui.proposals._widget_proposal is None

    window.attention.listen(experiment)

    assert ui.proposals._widget_proposal is not None, (
        "the standing question is up again"
    )
    assert ui.panel.prompt_bar.yes.text() == "Continue"


def test_continue_on_a_question_the_widget_never_got_confirms_it_as_proposed(
    window, qapp
):
    """The task asked without loading its detection into the window (a window
    that started listening mid-question has none): Continue confirms what was
    proposed instead of reading an empty Detection tab."""
    ui = window
    lamella = ui.experiment.positions[0]
    task, thread, seen = _start(window)
    assert _pump(qapp, lambda: ui.proposals._widget_proposal is not None)

    ui.panel.prompt_bar.yes.click()

    assert _pump(qapp, lambda: not thread.is_alive()), "released"
    assert "error" not in seen
    assert seen["decision"].outcome is DecisionOutcome.Confirmed
    assert seen["decision"].values == {}, "as proposed"
    assert lamella.proposal(ROUGH, DETECTION).current.actor.kind is ActorKind.operator


def test_a_point_asked_with_no_frame_to_show_it_on_is_confirmed_as_proposed(
    window, qapp
):
    """No FIB frame on the view and no reference image on disk: the marker
    cannot go up, and Continue confirms the point as proposed."""
    ui = window
    ui.panel.image_widget.ib_image = None
    task, thread, seen = _start(window, POINT_OF_INTEREST, {"poi": Point(1e-6, 2e-6)})
    assert _pump(qapp, lambda: ui.proposals._widget_proposal is not None), "put up"
    assert not ui.proposals.poi_marker.shown

    ui.panel.prompt_bar.yes.click()

    assert _pump(qapp, lambda: not thread.is_alive()), "released"
    assert "error" not in seen
    assert seen["decision"].outcome is DecisionOutcome.Confirmed
    assert seen["decision"].values == {}, "as proposed"


def _swap_view_frame(ui, hfw):
    """Put another FIB frame on the view, as a new image would, and say so."""
    from fibsem.structures.image import FibsemImage
    from fibsem.structures.settings import BeamType

    frame = FibsemImage.generate_blank_image(resolution=(1536, 1024), hfw=hfw)
    frame.metadata.image_settings.beam_type = BeamType.ION
    ui.panel.image_widget.ib_image = frame
    ui.panel.image_widget.viewer_update_signal.emit()
    return frame


def test_the_poi_marker_follows_the_view_to_another_frame(window, qapp):
    """FIB-1410: the view takes a frame at another field of view while the
    point waits. The marker moves to the same point on it, and Continue
    decides the point the operator saw, not a pixel read on the wrong frame."""
    from fibsem import conversions
    from fibsem.structures.settings import BeamType

    ui = window
    _swap_view_frame(ui, 150e-6)
    initial = Point(x=5e-6, y=-3e-6)
    task, thread, seen = _start(window, POINT_OF_INTEREST, {"poi": initial})
    assert _pump(qapp, lambda: ui.proposals.poi_marker.shown), "the marker is up"

    new = _swap_view_frame(ui, 80e-6)
    (col, row), *_ = window.view_controller.overlay_points(BeamType.ION, "poi")
    px = conversions.microscope_image_to_image_coordinates(
        initial, new.data.shape, new.metadata.pixel_size.x
    )
    assert (col, row) == pytest.approx((px.x, px.y), abs=1)

    ui.panel.prompt_bar.yes.click()

    assert _pump(qapp, lambda: not thread.is_alive()), "released"
    assert "error" not in seen
    decided = seen["decision"].values.get("poi", initial)
    assert decided.x == pytest.approx(initial.x, abs=new.metadata.pixel_size.x)
    assert decided.y == pytest.approx(initial.y, abs=new.metadata.pixel_size.x)


def test_stop_withdraws_the_question_and_cancels_the_task(window, qapp):
    ui = window
    lamella = ui.experiment.positions[0]
    task, thread, seen = _start(window)
    assert _pump(qapp, lambda: ui.run_relay.hold is not None)
    proposal = lamella.proposal(ROUGH)

    task.task_manager.stop()

    assert _pump(qapp, lambda: not thread.is_alive())
    assert "decision" not in seen
    assert isinstance(seen.get("error"), InterruptedError)
    assert proposal.withdrawn
    assert ui.run_relay.hold is None
    assert lamella.task_state.status is TaskStatus.Cancelled


def test_every_declared_question_has_a_widget_to_answer_it_on():
    """A task says what it asks; the window has to be able to put it up: on
    the widget its value is edited on, or the prompt bar for a state."""
    from fibsem.applications.autolamella.workflows.tasks import get_tasks
    from fibsem.ui.run.attention import WIDGET_KINDS
    from fibsem.workflows.proposals import LOOP_KINDS, STATE

    shown = set(WIDGET_KINDS) | set(LOOP_KINDS) | {STATE}
    for name, cls in get_tasks().items():
        for kind in cls.asks:
            assert kind in shown, f"{name} asks {kind!r}, which nothing puts up"


def test_the_run_lends_control_while_the_question_stands(window, qapp):
    """FIB-1270, supervised: the run is in control, and while its question
    stands whoever answers may act on the microscope first -- a server command
    here. After the decision, control is the run's again."""
    from fastapi.testclient import TestClient

    from fibsem.server import AuthConfig, build_server

    ui = window
    lamella = ui.experiment.positions[0]
    control = ui.microscope.control
    client = TestClient(
        build_server(ui.microscope, auth=AuthConfig.generate(level="drive", token="t")),
        raise_server_exceptions=False,
    )

    def _move():
        # Off this thread, which keeps pumping, as the GUI thread does.
        out = {}

        def _post():
            out["response"] = client.post(
                "/microscope/services/stage_movement/commands/move_relative",
                headers={"Authorization": "Bearer t"},
                json={
                    "kwargs": {
                        "delta": {
                            "name": None,
                            "x": 1e-6,
                            "y": 0.0,
                            "z": 0.0,
                            "r": 0.0,
                            "t": 0.0,
                            "coordinate_system": "RAW",
                        }
                    }
                },
            )

        threading.Thread(target=_post, daemon=True).start()
        assert _pump(qapp, lambda: "response" in out)
        return out["response"]

    # The run's thread takes control at its entry point, as the app's worker
    # does, and keeps it until told to finish.
    task, manager = _task(window)
    seen = {}
    finish = threading.Event()

    def _run_in_control():
        with control.take("workflow", "a lamella run"):
            try:
                seen["decision"] = task.ask(DETECTION, PROPOSED, image="ml-1_ib.tif")
            except Exception as exc:  # noqa: BLE001
                seen["error"] = exc
            seen["asked"] = True
            finish.wait(10)

    thread = threading.Thread(target=_run_in_control, daemon=True)
    thread.start()
    window._started.append((manager, thread))
    try:
        assert _pump(qapp, lambda: ui.run_relay.hold is not None), "the run is held"

        moved = _move()
        assert moved.status_code == 200, moved.text
        assert control.current.by == "workflow"

        proposal = lamella.proposal(ROUGH)
        result = ui.experiment.decide(
            lamella.id,
            ROUGH,
            Decision(
                outcome=DecisionOutcome.Confirmed,
                actor="operator:op",
                values=PROPOSED,
                proposal_id=proposal.id,
            ),
        )
        assert result.applied, result.reason
        assert _pump(qapp, lambda: "asked" in seen), "released"
        assert "error" not in seen

        refused = _move()
        assert refused.status_code == 409
        assert refused.json()["detail"]["in_control"]["by"] == "workflow"
    finally:
        finish.set()
        thread.join(timeout=10)
    assert control.current is None
