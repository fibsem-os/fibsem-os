"""The milling question and its loop: the task mills, the question decides.

The loop is the task's, on the workflow thread (FIB-1236): each pass is a
``milling`` question on the record, it mills on ``run``, and asks again with
the pass counted and the run's error. The window shows the pass on the prompt
bar over the editor, and a button decides it with the action and the config as
the editor then held it. So whoever decides, the operator at the editor or an
agent through ``decide``, changes what is milled the same way, by deciding
with it.

``run_milling_task`` is stubbed at the task's import site: the loop, the
prompt, the editor and the answers are real, only the beam time is gone.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt5")

from fibsem.acting import TASK, Actor, current_actor
from fibsem.applications.autolamella.workflows.tasks import base
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.applications.autolamella.workflows.tasks.rough import (
    MillRoughTask,
    MillRoughTaskConfig,
)
from fibsem.milling import FibsemMillingStage
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.workflows.proposals import (
    ACTION_ACCEPT,
    ACTION_EDIT,
    ACTION_RUN,
    MILLING,
    Decision,
    DecisionOutcome,
    patch_values,
)
from fibsem.workflows.status import HoldKind
from tests.fixtures.main_window import main_window
from tests.run_state import Stopped

ROUGH = "Rough Milling"
MSG = "Mill the lamella."
FIRST = f"{MSG} Press Run Milling to mill, or Skip to go on without milling."
AGAIN = f"{MSG} Press Mill Again to mill again, or Accept to go on."


def _config(name: str = "rough") -> FibsemMillingTaskConfig:
    return FibsemMillingTaskConfig(
        name=name,
        stages=[FibsemMillingStage(name="Rough 1"), FibsemMillingStage(name="Rough 2")],
    )


@pytest.fixture
def ui(qapp, monkeypatch, tmp_path):
    """A real AutoLamellaUI (Demo) with a supervised Rough Milling item, and
    the mill itself stubbed where the task calls it."""
    mills, failures, supervision = [], [], {"supervised": True}

    def fake_run_milling_task(microscope, config, stop_event=None, **kwargs):
        mills.append(
            SimpleNamespace(
                config=config,
                actor=current_actor(),
                thread=threading.current_thread(),
                output_dir=kwargs.get("output_dir"),
            )
        )
        time.sleep(0.05)
        for hook in list(on_mill):
            hook()
        return SimpleNamespace(error=failures.pop(0) if failures else None)

    on_mill = []
    monkeypatch.setattr(base, "run_milling_task", fake_run_milling_task)
    monkeypatch.setattr(
        base, "get_task_supervision", lambda *_: supervision["supervised"]
    )
    with main_window(
        qapp,
        tmp_path,
        "milling-question-exp",
        {ROUGH: MillRoughTaskConfig(task_name=ROUGH)},
        supervised=[ROUGH],
    ) as window:
        widget = window.autolamella_ui
        widget._mills = mills  # for the tests to inspect
        widget._mill_failures = failures  # errors the next runs end with, in order
        widget._supervision = supervision
        widget._on_mill = on_mill  # called at the end of each mill
        yield widget


def _task(ui):
    experiment = ui.experiment
    lamella = experiment.positions[0]
    manager = TaskManager(
        microscope=ui.microscope,
        experiment=experiment,
        parent_ui=ui,
        responder=ui.ui_responder,
        answerers=ui.session.answerers,
    )
    task = MillRoughTask(
        microscope=ui.microscope,
        config=lamella.task_config[ROUGH],
        lamella=lamella,
        parent_ui=ui,
        task_manager=manager,
    )
    task._manager = manager
    return task


def _start(ui, qapp, config=None, milling_enabled=True, wait_for=FIRST):
    """Run the task's mill-and-ask loop on a worker thread, as the workflow
    does; wait for ``wait_for`` on the prompt (None: don't)."""
    task = _task(ui)
    outcome = {}

    def _run():
        outcome["config"] = task.update_milling_config_ui(
            config or _config(), msg=MSG, milling_enabled=milling_enabled
        )

    task._run = _run  # type: ignore[method-assign]

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


def _wait_for_prompt(ui, qapp, text, timeout_s=10.0):
    _wait(
        qapp,
        lambda: ui.label_instructions.text() == text and ui.pushButton_yes.isEnabled(),
        timeout_s,
        f"the prompt {text!r} never appeared",
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
    """The pass up on the prompt bar: the milling question being asked."""
    assert ui._loop_question is not None, "no pass is up"
    proposal = ui._loop_question[2]
    assert proposal.kind == MILLING and proposal.asking
    return proposal


def _stop(ui, task):
    task._manager.stop()
    ui.session.runner.manager = Stopped()


def _settle(qapp, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)


def test_run_then_accept_mills_once_on_the_workflow_thread(ui, qapp):
    task, thread, outcome = _start(ui, qapp)

    assert ui.pushButton_yes.text() == "Run Milling"
    assert ui.pushButton_no.text() == "Skip"
    assert _pass(ui).provenance["passes"] == 0
    assert ui.hold is not None and ui.hold.kind is HoldKind.question

    ui.pushButton_yes.click()  # run
    _wait_for_prompt(ui, qapp, AGAIN)
    assert len(ui._mills) == 1
    assert ui._mills[0].thread is thread, "the task mills, not the GUI"
    assert _pass(ui).provenance["passes"] == 1
    assert ui.pushButton_yes.text() == "Mill Again"
    assert ui.pushButton_no.text() == "Accept"

    ui.pushButton_no.click()  # accept
    _finish(thread, qapp)

    assert "error" not in outcome
    assert outcome["config"].name == "rough"
    assert len(ui._mills) == 1
    assert ui.hold is None


def test_mill_again_runs_a_second_pass(ui, qapp):
    task, thread, outcome = _start(ui, qapp)

    ui.pushButton_yes.click()
    _wait_for_prompt(ui, qapp, AGAIN)
    ui.pushButton_yes.click()  # Mill Again
    _wait(qapp, lambda: len(ui._mills) == 2)
    _wait_for_prompt(ui, qapp, AGAIN)
    assert _pass(ui).provenance["passes"] == 2
    ui.pushButton_no.click()
    _finish(thread, qapp)

    assert "error" not in outcome


def test_a_failed_mill_asks_again_with_the_error(ui, qapp):
    """Supervised: the mill failed, so the prompt says so and asks again
    (FIB-1112). Mill Again retries; Accept is the operator's call."""
    ui._mill_failures.append(RuntimeError("could not load type"))
    task, thread, outcome = _start(ui, qapp)

    ui.pushButton_yes.click()  # run, and it fails
    _wait_for_prompt(ui, qapp, f"Milling failed: could not load type\n\n{AGAIN}")
    assert thread.is_alive(), "a failed mill does not end the question"
    assert _pass(ui).provenance["last_error"] == "could not load type"

    ui.pushButton_yes.click()  # retry, and it succeeds
    _wait_for_prompt(ui, qapp, AGAIN)
    assert len(ui._mills) == 2
    assert _pass(ui).provenance["passes"] == 2

    ui.pushButton_no.click()
    _finish(thread, qapp)

    assert "error" not in outcome


def test_accept_after_a_failed_mill_goes_on(ui, qapp):
    ui._mill_failures.append(RuntimeError("could not load type"))
    task, thread, outcome = _start(ui, qapp)

    ui.pushButton_yes.click()
    _wait_for_prompt(ui, qapp, f"Milling failed: could not load type\n\n{AGAIN}")
    ui.pushButton_no.click()
    _finish(thread, qapp)

    assert "error" not in outcome


def test_unsupervised_a_failed_mill_fails_the_task(ui, qapp):
    """Nobody to ask: the error reaches the task, which fails, as a headless
    run does."""
    error = RuntimeError("could not load type")
    ui._mill_failures.append(error)
    ui._supervision["supervised"] = False
    task, thread, outcome = _start(ui, qapp, wait_for=None)

    _finish(thread, qapp)

    assert outcome.get("error") is error
    assert len(ui._mills) == 1
    assert ui.hold is None


def test_skip_goes_on_without_a_mill(ui, qapp):
    task, thread, outcome = _start(ui, qapp)

    ui.pushButton_no.click()  # skip
    _finish(thread, qapp)

    assert "error" not in outcome
    assert ui._mills == []


def test_unsupervised_mills_once_without_a_prompt(ui, qapp):
    ui._supervision["supervised"] = False
    task, thread, outcome = _start(ui, qapp, wait_for=None)

    _finish(thread, qapp)

    assert "error" not in outcome
    assert len(ui._mills) == 1
    assert ui.hold is None


def test_disabled_milling_only_confirms_the_patterns(ui, qapp):
    task, thread, outcome = _start(
        ui,
        qapp,
        milling_enabled=False,
        wait_for=f"{MSG} Press Continue when the config is right.",
    )

    assert ui.pushButton_yes.text() == "Continue"
    ui.pushButton_yes.click()
    _finish(thread, qapp)

    assert "error" not in outcome
    assert ui._mills == []


def test_what_mills_is_what_the_editor_held(ui, qapp):
    """The operator's edits reach the mill, because the answer carries them."""
    task, thread, outcome = _start(ui, qapp)

    ui.milling_task_config_widget.update_from_settings(_config("edited-by-operator"))
    ui.pushButton_yes.click()
    _wait_for_prompt(ui, qapp, AGAIN)
    ui.pushButton_no.click()
    _finish(thread, qapp)

    assert ui._mills[0].config.name == "edited-by-operator"
    assert outcome["config"].name == "edited-by-operator"


def test_accept_returns_what_was_milled_not_a_later_edit(ui, qapp):
    """An edit after the last pass never ran, so the protocol must not
    claim it (FIB-580)."""
    task, thread, outcome = _start(ui, qapp)

    ui.pushButton_yes.click()
    _wait_for_prompt(ui, qapp, AGAIN)
    ui.milling_task_config_widget.update_from_settings(_config("never-milled"))
    ui.pushButton_no.click()
    _finish(thread, qapp)

    assert outcome["config"].name == "rough"


def test_an_agents_edit_lands_in_the_editor_and_asks_again(ui, qapp):
    """``edit`` takes the change and asks again; nothing mills. The edit is
    in the editor for the operator to see, and in the config the next run
    mills."""
    task, thread, outcome = _start(ui, qapp)
    lamella = ui.experiment.positions[0]
    asked = _pass(ui)

    edited = patch_values(
        MILLING, dict(asked.values), {"stages.Rough 2.pattern.depth": 2.5e-6}
    )
    result = ui.experiment.decide(
        lamella.id,
        ROUGH,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            actor=Actor.agent("remote"),
            values=edited,
            via="server",
            proposal_id=asked.id,
            action=ACTION_EDIT,
        ),
    )
    assert result.applied, result.reason
    _wait(qapp, lambda: ui._loop_question and ui._loop_question[2].id != asked.id)
    assert ui._mills == []
    shown = ui.milling_task_config_widget.get_config()
    assert shown.stages[1].pattern.depth == pytest.approx(2.5e-6)

    again = _pass(ui)
    result = ui.experiment.decide(
        lamella.id,
        ROUGH,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            actor=Actor.agent("remote"),
            via="server",
            proposal_id=again.id,
            action=ACTION_RUN,
        ),
    )
    assert result.applied, result.reason
    _wait_for_prompt(ui, qapp, AGAIN)
    assert ui._mills[0].config.stages[1].pattern.depth == pytest.approx(2.5e-6)

    ui.pushButton_no.click()
    _finish(thread, qapp)
    assert "error" not in outcome


def test_every_pass_is_on_the_record_with_its_action(ui, qapp):
    """One proposal per pass, sharing the run: Run, then Accept, each with
    who decided it."""
    task, thread, outcome = _start(ui, qapp)
    ui.pushButton_yes.click()
    _wait_for_prompt(ui, qapp, AGAIN)
    ui.pushButton_no.click()
    _finish(thread, qapp)

    passes = [
        p for p in ui.experiment.positions[0].proposals_from(ROUGH) if p.kind == MILLING
    ]
    assert [p.provenance["passes"] for p in passes] == [0, 1]
    assert [p.current.action for p in passes] == [ACTION_RUN, ACTION_ACCEPT]
    assert all(p.current.actor.kind.name == "operator" for p in passes)
    assert len({p.task_id for p in passes}) == 1, "one run"


def test_unsupervised_passes_are_recorded_unreviewed(ui, qapp):
    """Nobody asked: the kind's silence, run once then accept, on the record
    and nothing left waiting."""
    ui._supervision["supervised"] = False
    task, thread, outcome = _start(ui, qapp, wait_for=None)
    _finish(thread, qapp)

    passes = [
        p for p in ui.experiment.positions[0].proposals_from(ROUGH) if p.kind == MILLING
    ]
    assert [p.current.outcome for p in passes] == [DecisionOutcome.Unreviewed] * 2
    assert [p.current.action for p in passes] == [ACTION_RUN, ACTION_ACCEPT]
    assert ui.experiment.pending_proposals() == []


def test_a_stop_interrupts_the_prompt(ui, qapp):
    task, thread, outcome = _start(ui, qapp)

    _stop(ui, task)
    _finish(thread, qapp)

    assert isinstance(outcome.get("error"), InterruptedError)


def test_a_stop_during_the_mill_ends_the_task_without_asking_again(ui, qapp):
    task, thread, outcome = _start(ui, qapp)
    ui._on_mill.append(lambda: _stop(ui, task))  # Stop pressed mid-mill

    ui.pushButton_yes.click()
    _finish(thread, qapp)
    _settle(qapp, 0.3)

    assert isinstance(outcome.get("error"), InterruptedError)
    assert len(ui._mills) == 1
    assert ui.label_instructions.text() != AGAIN, "the aborted prompt came back"
    assert ui.hold is None


def test_a_click_after_a_stop_starts_nothing(ui, qapp):
    task, thread, outcome = _start(ui, qapp)

    _stop(ui, task)
    _finish(thread, qapp)
    assert isinstance(outcome.get("error"), InterruptedError)

    ui.pushButton_yes.click()  # "Run Milling", but the asker is gone
    _settle(qapp, 0.3)

    assert ui._mills == []


def test_flipping_to_auto_mid_mill_goes_on_without_asking_again(ui, qapp):
    # Supervision is read at each decision: turned off during a mill, the
    # mill's end goes on rather than asking (bench-found).
    task, thread, outcome = _start(ui, qapp)
    ui._on_mill.append(lambda: ui._supervision.update(supervised=False))

    ui.pushButton_yes.click()
    _finish(thread, qapp)

    assert "error" not in outcome
    assert len(ui._mills) == 1


def test_flipping_to_supervised_mid_mill_asks_at_its_end(ui, qapp):
    # The other direction: "dropping in on a milling run".
    ui._supervision["supervised"] = False
    ui._on_mill.append(lambda: ui._supervision.update(supervised=True))
    task, thread, outcome = _start(ui, qapp, wait_for=AGAIN)

    assert len(ui._mills) == 1
    ui.pushButton_no.click()  # accept
    _finish(thread, qapp)

    assert "error" not in outcome
    assert len(ui._mills) == 1


# ── who milled, and where it wrote (FIB-1062, FIB-1253) ──────────────────────


def test_every_mill_is_the_task_s_and_writes_under_its_folder(ui, qapp):
    ui._mill_failures.append(RuntimeError("could not load type"))
    task, thread, outcome = _start(ui, qapp)

    ui.pushButton_yes.click()  # run, and it fails
    _wait_for_prompt(ui, qapp, f"Milling failed: could not load type\n\n{AGAIN}")
    ui.pushButton_yes.click()  # the operator's retry
    _wait_for_prompt(ui, qapp, AGAIN)
    ui.pushButton_no.click()
    _finish(thread, qapp)

    assert [mill.actor for mill in ui._mills] == [TASK, TASK]
    assert [mill.output_dir for mill in ui._mills] == [task.output_dir] * 2


def test_a_mill_run_by_hand_is_not_the_task_s(ui, qapp, monkeypatch):
    """The editor's own Run, outside any question: unmarked, the operator's,
    with no task folder."""
    from fibsem.services.milling import Milling

    runs = []

    def by_hand(service, config, path=None):
        runs.append((current_actor(), path))

    monkeypatch.setattr(Milling, "run_task", by_hand)
    widget = ui.ui_responder._milling_widget().milling_widget
    widget.run_milling(_config("by-hand"))
    _wait(qapp, lambda: runs and not widget.is_milling)
    _settle(qapp, 0.1)  # its finished signal, before the window goes

    assert runs == [(None, None)]


def test_the_question_is_the_task_s_and_the_answer_the_operator_s(ui, qapp, tmp_path):
    from fibsem.acting import OPERATOR
    from fibsem.workflows.event_recording import (
        EVENTS_FILENAME,
        EventRecorder,
        read_events,
    )

    recorder = EventRecorder(
        ui.microscope,
        responder=ui.ui_responder,
        experiment_path=tmp_path,
        experiment=ui.experiment,
        default_actor=OPERATOR,
    )
    try:
        task, thread, _ = _start(ui, qapp)
        ui.pushButton_no.click()  # skip
        _finish(thread, qapp)
    finally:
        recorder.close()
    records = [
        r
        for r in read_events(tmp_path / EVENTS_FILENAME)
        if r["kind"] in ("proposal_asked", "proposal_decided")
        and r["payload"]["kind"] == MILLING
    ]
    actors = {r["kind"]: r["actor"] for r in records}
    assert actors == {"proposal_asked": "task", "proposal_decided": "operator"}
    decided = [r for r in records if r["kind"] == "proposal_decided"]
    assert decided[0]["payload"]["action"] == "skip"
