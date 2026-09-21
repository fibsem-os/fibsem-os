"""A detection asked mid-task is recorded, held and answered in the Review tab.

``QtResponder`` owns the wait for every question: the nonce, the hold, the
abort. For a question that carries a value it also puts it on the item's record
as a proposal (FIB-1025), and then the answer is a decision -- from the Review
tab, or from anything else that goes through ``Experiment.decide`` -- rather
than the prompt's button. These drive the real main window and the real
responder, ask from a worker thread the way a task does, and decide on the main
thread the way the tab and the agent server both do.

Whenever the question cannot go on the record it is asked the way it always
was, so the second half is about that: nothing here may leave a question
unasked.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from psygnal.containers import EventedDict
from PyQt5.QtWidgets import QWidget

from fibsem import conversions
from fibsem.applications.autolamella.proposals import (
    DETECTION,
    Decision,
    DecisionOutcome,
)
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    AutoLamellaTaskState,
    AutoLamellaTaskStatus,
    Experiment,
)
from fibsem.applications.autolamella.workflows.interaction import (
    ConfirmDetection,
    ask,
)
from fibsem.applications.autolamella.workflows.tasks.status import HoldKind
from fibsem.detection.detection import DetectedFeatures, LamellaCentre
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemImageMetadata,
    ImageSettings,
    MicroscopeState,
    Point,
)

TASK = "Mill Undercut"
RUN = "run-1"
OLD_PROMPT = "Confirm Feature Detection. Press Continue to proceed."


class _DetWidget(QWidget):
    """The three methods QtResponder drives on the Detection tab, on a real
    widget in the real tab bar. The real one needs the ``ml`` extra."""

    def __init__(self):
        super().__init__()
        self.det = None

    def set_detected_features(self, det):
        self.det = det

    def _get_detected_features(self):
        return self.det

    def confirm_button_clicked(self):
        pass


@pytest.fixture
def window(qapp, tmp_path):
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module

    win = module.AutoLamellaSingleWindowUI()
    win.autolamella_ui.system_widget.connect_to_microscope()
    ui = win.autolamella_ui
    det_widget = _DetWidget()
    ui.det_widget = det_widget
    ui.tabWidget.setTabVisible(ui.tabWidget.addTab(det_widget, "Detection"), False)

    experiment = Experiment(path=tmp_path, name="record-exp")
    os.makedirs(experiment.path, exist_ok=True)
    experiment.task_protocol = AutoLamellaTaskProtocol()
    experiment.add_new_lamella(MicroscopeState(), EventedDict())
    lamella = experiment.positions[0]
    lamella.task_history.append(
        AutoLamellaTaskState(name=TASK, status=AutoLamellaTaskStatus.InProgress)
    )
    lamella.task_state.name = TASK
    lamella.task_state.task_id = RUN
    lamella.task_state.status = AutoLamellaTaskStatus.InProgress
    ui.experiment = experiment
    win._on_experiment_update()

    # Interactive review on, through the path the preference takes.
    win._preferences.features.proposer_reviewer_workflow_enabled = True
    win._apply_review_visibility()
    win.tab_widget.setCurrentIndex(0)

    yield win
    if ui.microscope is not None:
        ui.microscope.disconnect()
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        win.close()
    finally:
        qapp.quit = original_quit


@pytest.fixture(autouse=True)
def training_data(monkeypatch):
    """What a confirmed detection writes for training, caught rather than
    written: the real one saves an image, a mask and a CSV row under the user's
    own ML data directory."""
    from fibsem.detection import utils as det_utils

    written = []
    monkeypatch.setattr(
        det_utils,
        "save_ml_feature_data",
        lambda det, initial_features=None: written.append((det, initial_features)),
    )
    return written


def _detection(folder=None, x: float = 3.0, y: float = 4.0) -> DetectedFeatures:
    """A detection on an 8x8 image. ``folder`` saves the image there first, as
    ``take_image_and_detect_features`` always does; without it the picture is
    only in memory."""
    feature = LamellaCentre()
    feature.px = Point(x, y)
    data = np.zeros((8, 8), dtype=np.uint8)
    image = FibsemImage(
        data=data,
        metadata=FibsemImageMetadata(
            image_settings=ImageSettings(beam_type=BeamType.ION, hfw=8e-9),
            pixel_size=Point(1e-9, 1e-9),
            microscope_state=MicroscopeState(),
        ),
    )
    if folder is not None:
        image.save(os.path.join(str(folder), f"ml-{time.monotonic_ns()}_ib"))
    return DetectedFeatures(
        features=[feature],
        image=data,
        mask=np.zeros((8, 8), dtype=np.uint8),
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        pixelsize=1e-9,
        fibsem_image=image,
    )


def _pump(qapp, predicate, timeout_s=10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _ask(window, qapp, request, abort=None):
    """Ask from a worker thread, as a task does, and wait until it is up."""
    responder = window.autolamella_ui.ui_responder
    outcome = {}

    def target():
        try:
            outcome["answer"] = ask(responder, request, abort=abort)
        except BaseException as exc:  # noqa: BLE001 - the test inspects it
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    assert _pump(qapp, lambda: responder.pending_question() is request), (
        "the question was never asked"
    )
    return thread, outcome


def _request(window, **kwargs) -> ConfirmDetection:
    lamella = window.autolamella_ui.experiment.positions[0]
    fields = {"item_id": lamella.id, "task_name": TASK}
    fields.update(kwargs)
    detection = fields.pop("detection", None) or _detection(lamella.path)
    return ConfirmDetection(detection=detection, **fields)


def _decide(window, outcome=DecisionOutcome.Confirmed, px=None, reason="", **kwargs):
    experiment = window.autolamella_ui.experiment
    lamella = experiment.positions[0]
    values = {}
    if outcome is DecisionOutcome.Confirmed:
        values = {"features": [{"name": "LamellaCentre", "px": px or Point(3, 4)}]}
    return experiment.decide(
        kwargs.get("item_id", lamella.id),
        kwargs.get("task_name", TASK),
        Decision(
            outcome=outcome,
            author=kwargs.get("author", "human:op"),
            values=values,
            reason=reason,
            task_id=RUN,
        ),
    )


def _finish(qapp, thread):
    assert _pump(qapp, lambda: not thread.is_alive()), "the asking task never resumed"


# ---------------------------------------------------------------------------
# Asked: on the record, held, and in front of the operator
# ---------------------------------------------------------------------------


def test_the_question_goes_on_the_items_record(window, qapp):
    lamella = window.autolamella_ui.experiment.positions[0]

    thread, _ = _ask(window, qapp, _request(window))

    proposal = lamella.proposals[TASK]
    assert proposal.kind == DETECTION
    assert proposal.asking and proposal.pending
    assert proposal.task_id == RUN

    _decide(window)
    _finish(qapp, thread)


def test_the_run_is_held_on_it_and_the_hold_says_where(window, qapp):
    """The part a second asker never had: Attention Required lights, and what
    releases it is a decision in the Review tab."""
    ui = window.autolamella_ui
    lamella = ui.experiment.positions[0]

    thread, _ = _ask(window, qapp, _request(window))

    assert ui.hold is not None and ui.hold.kind is HoldKind.question
    assert "Review tab" in ui.hold.releases and lamella.name in ui.hold.releases
    assert ui.hold.items == (f"{lamella.name}/{TASK}",)

    _decide(window)
    _finish(qapp, thread)
    assert ui.hold is None


def test_it_is_put_in_front_of_the_operator_in_the_review_tab(window, qapp):
    lamella = window.autolamella_ui.experiment.positions[0]

    thread, _ = _ask(window, qapp, _request(window))

    assert window.tab_widget.currentWidget() is window.review_tab
    assert window.review_tab._current_key() == (lamella.id, TASK)
    assert any("asking now" in row for row in window.review_tab.row_summaries())

    _decide(window)
    _finish(qapp, thread)


def test_the_detection_tab_is_not_raised_for_it(window, qapp):
    ui = window.autolamella_ui

    thread, _ = _ask(window, qapp, _request(window))

    assert ui.det_widget.det is None
    assert not ui.tabWidget.isTabVisible(ui.tabWidget.indexOf(ui.det_widget))

    _decide(window)
    _finish(qapp, thread)


def test_attention_required_goes_back_to_it(window, qapp):
    thread, _ = _ask(window, qapp, _request(window))
    assert window.autolamella_ui.ui_responder.question_host() is window.review_tab
    window.tab_widget.setCurrentIndex(0)

    window._on_user_attention_clicked()

    assert window.tab_widget.currentWidget() is window.review_tab

    _decide(window)
    _finish(qapp, thread)
    assert window.autolamella_ui.ui_responder.question_host() is None


# ---------------------------------------------------------------------------
# Answered: a decision releases the run
# ---------------------------------------------------------------------------


def test_a_confirmed_decision_releases_the_task_with_the_decided_points(window, qapp):
    request = _request(window)
    thread, outcome = _ask(window, qapp, request)

    result = _decide(window, px=Point(6, 1))

    assert result.applied, result.reason
    _finish(qapp, thread)
    feature = outcome["answer"].features[0]
    assert feature.px == Point(6, 1)
    assert feature.feature_m == conversions.image_to_microscope_image_coordinates(
        Point(6, 1), request.detection.image.data, request.detection.pixelsize
    ), "and in metres, which is what the task acts on"
    assert window.autolamella_ui.ui_responder.pending_question() is None


def test_a_confirmed_detection_still_writes_its_training_data(
    window, qapp, training_data
):
    """The Detection tab writes it on the Continue click: the image, the mask
    and how far each feature was moved, and the log records the reports read.
    Moving the question onto the record must not quietly stop that."""
    request = _request(window)
    thread, outcome = _ask(window, qapp, request)

    _decide(window, px=Point(6, 1))
    _finish(qapp, thread)

    assert len(training_data) == 1
    written, initial = training_data[0]
    assert written is outcome["answer"], "the corrected set"
    assert written.features[0].px == Point(6, 1)
    assert initial[0].px == Point(3, 4), "measured against what the model said"


def test_a_rejected_decision_raises_on_the_asking_task(window, qapp, training_data):
    thread, outcome = _ask(window, qapp, _request(window))

    _decide(window, outcome=DecisionOutcome.Rejected, reason="no lamella here")

    _finish(qapp, thread)
    assert isinstance(outcome["error"], RuntimeError)
    assert "no lamella here" in str(outcome["error"])
    assert training_data == [], "nothing was confirmed, so nothing to learn from"


def test_a_decision_on_something_else_does_not_release_it(window, qapp):
    thread, outcome = _ask(window, qapp, _request(window))

    result = _decide(window, task_name="Some Other Task")

    assert not result.applied
    assert thread.is_alive() and "answer" not in outcome
    assert window.autolamella_ui.ui_responder.pending_question() is not None

    _decide(window)
    _finish(qapp, thread)


def test_the_timeline_hears_who_answered(window, qapp):
    heard = []
    window.autolamella_ui.ui_responder.add_question_observer(
        lambda kind, payload: heard.append((kind, payload))
    )
    thread, _ = _ask(window, qapp, _request(window))
    nonce = heard[0][1]["nonce"]

    _decide(window, author="agent:claude")
    _finish(qapp, thread)

    assert [kind for kind, _ in heard] == ["prompt_raised", "prompt_answered"]
    assert heard[1][1] == {
        "type": "ConfirmDetection",
        "response": True,
        "answered_by": "agent",
        "nonce": nonce,
    }


def test_a_task_can_ask_again_after_being_answered(window, qapp):
    """An undercut asks four or more times in one run. Each is its own question
    and the listener for the last one must not answer the next."""
    thread, first = _ask(window, qapp, _request(window))
    _decide(window, px=Point(1, 1))
    _finish(qapp, thread)

    thread, second = _ask(window, qapp, _request(window))
    assert "answer" not in second, "the first decision did not leak into it"
    _decide(window, px=Point(2, 2))
    _finish(qapp, thread)

    assert first["answer"].features[0].px == Point(1, 1)
    assert second["answer"].features[0].px == Point(2, 2)


# ---------------------------------------------------------------------------
# The prompt's button and the agent's prompt answer are not answers
# ---------------------------------------------------------------------------


def test_the_prompt_button_goes_to_the_question_and_answers_nothing(window, qapp):
    ui = window.autolamella_ui
    thread, outcome = _ask(window, qapp, _request(window))
    assert _pump(qapp, lambda: ui.pushButton_yes.text() == "Go to Review")
    window.tab_widget.setCurrentIndex(0)

    ui.pushButton_yes.click()
    qapp.processEvents()

    assert window.tab_widget.currentWidget() is window.review_tab
    assert thread.is_alive() and "answer" not in outcome
    assert ui.ui_responder.pending_question() is not None
    assert ui.experiment.positions[0].proposals[TASK].pending

    _decide(window)
    _finish(qapp, thread)


def test_an_agent_answering_the_prompt_is_told_to_decide_instead(window, qapp):
    """A click would report "applied" and decide nothing. There is one way to
    answer a recorded question, for an agent as for a person."""
    ui = window.autolamella_ui
    lamella = ui.experiment.positions[0]
    thread, outcome = _ask(window, qapp, _request(window))
    _request_up, nonce = ui.ui_responder.pending_question_and_nonce()
    assert ui.ui_responder.recorded_question() == (lamella.id, TASK)

    answered = ui.ui_responder.submit_answer(True, nonce=nonce)
    assert _pump(qapp, answered.done)

    with pytest.raises(ValueError, match="decide"):
        answered.result()
    assert thread.is_alive() and "answer" not in outcome

    _decide(window, author="agent:claude")
    _finish(qapp, thread)
    assert ui.ui_responder.recorded_question() is None


def test_the_agent_server_says_how_it_is_answered(window, qapp):
    """The pending prompt names the decision to make, and answering the prompt
    anyway is refused without a click -- not reported as applied."""
    from fibsem.applications.autolamella.server.context import AgentContext

    ui = window.autolamella_ui
    lamella = ui.experiment.positions[0]
    context = AgentContext(ui)
    thread, outcome = _ask(window, qapp, _request(window))

    pending = context.pending_prompt()["pending"]
    assert pending["type"] == "ConfirmDetection"
    assert pending["answer_via"] == "decide"
    assert pending["decide"] == {"item_id": lamella.id, "task_name": TASK}

    refused = context.answer_prompt(True, nonce=pending["nonce"])
    assert refused == {
        "available": True,
        "applied": False,
        "stale": False,
        "answer_via": "decide",
        "item_id": lamella.id,
        "task_name": TASK,
    }
    assert thread.is_alive() and "answer" not in outcome

    _decide(window, author="agent:claude")
    _finish(qapp, thread)
    assert "answer_via" not in (context.pending_prompt()["pending"] or {})


# ---------------------------------------------------------------------------
# Not answered: the question is taken back
# ---------------------------------------------------------------------------


def test_stopping_the_run_takes_the_question_back(window, qapp):
    """The asker cancels its future when it is aborted; that is the moment the
    question stops existing, so it is withdrawn there and the prompt, the hold
    and the inbox row go with it."""
    ui = window.autolamella_ui
    lamella = ui.experiment.positions[0]
    stop = threading.Event()
    thread, outcome = _ask(window, qapp, _request(window), abort=stop.is_set)

    stop.set()
    _finish(qapp, thread)

    assert isinstance(outcome["error"], InterruptedError)
    proposal = lamella.proposals[TASK]
    assert _pump(qapp, lambda: proposal.withdrawn)
    assert not proposal.asking
    assert _pump(qapp, lambda: ui.hold is None)
    assert ui.ui_responder.question_host() is None
    assert not any("asking now" in row for row in window.review_tab.row_summaries())


def test_a_decision_after_the_run_stopped_is_refused(window, qapp):
    """It would be an answer nobody is waiting for, written against a run that
    is over."""
    stop = threading.Event()
    thread, _ = _ask(window, qapp, _request(window), abort=stop.is_set)
    stop.set()
    _finish(qapp, thread)
    lamella = window.autolamella_ui.experiment.positions[0]
    assert _pump(qapp, lambda: lamella.proposals[TASK].withdrawn)

    result = _decide(window)

    assert not result.applied and result.error_type == "stale_review"


def test_a_run_that_ends_with_the_question_up_takes_it_back(window, qapp):
    ui = window.autolamella_ui
    lamella = ui.experiment.positions[0]
    thread, outcome = _ask(window, qapp, _request(window))

    ui.ui_responder.abandon()
    _finish(qapp, thread)

    assert _pump(qapp, lambda: lamella.proposals[TASK].withdrawn)
    assert ui.hold is None


# ---------------------------------------------------------------------------
# When it cannot go on the record, it is asked the way it always was
# ---------------------------------------------------------------------------


def _asked_the_old_way(window, qapp, thread, outcome) -> None:
    ui = window.autolamella_ui
    lamella = ui.experiment.positions[0]
    assert _pump(qapp, lambda: ui.label_instructions.text() == OLD_PROMPT)
    assert ui.det_widget.det is not None, "on the Detection tab"
    assert lamella.proposals.get(TASK) is None, "and not on the record"
    assert ui.ui_responder.recorded_question() is None
    ui.pushButton_yes.click()
    _finish(qapp, thread)
    assert "answer" in outcome, outcome


def test_with_interactive_review_off_it_is_a_prompt(window, qapp):
    window._preferences.features.proposer_reviewer_workflow_enabled = False
    window._apply_review_visibility()

    thread, outcome = _ask(window, qapp, _request(window))

    _asked_the_old_way(window, qapp, thread, outcome)


def test_a_question_that_does_not_say_who_is_asking_is_a_prompt(window, qapp):
    thread, outcome = _ask(window, qapp, _request(window, item_id="", task_name=""))

    _asked_the_old_way(window, qapp, thread, outcome)


def test_a_question_about_an_item_that_is_gone_is_a_prompt(window, qapp):
    thread, outcome = _ask(window, qapp, _request(window, item_id="no-such-item"))

    _asked_the_old_way(window, qapp, thread, outcome)


def test_a_question_on_a_picture_that_was_never_saved_is_a_prompt(window, qapp):
    """It could not be answered in the Review tab -- there is nothing to show --
    and the Detection tab still has the image in memory."""
    thread, outcome = _ask(window, qapp, _request(window, detection=_detection()))

    _asked_the_old_way(window, qapp, thread, outcome)
