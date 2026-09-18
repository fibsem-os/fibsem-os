"""Asking through the record: the same Responder protocol, a different answer.

``ReviewResponder`` records a question as a proposal and completes the asking
task's future when a decision lands on it. The task cannot tell the difference
-- it gets the same answer type it would have got from a prompt -- but the
question is on the lamella while it is open, and anything that goes through
``Experiment.decide`` answers it.

A parallel track: nothing in the workflow asks through this yet, so these tests
are the only caller. They run the real ``ask`` from the interaction seam on a
worker thread, the way a task does, and answer on the main thread, the way the
Review tab and the agent server both do.

No Qt and no microscope: the point of recording a question is that answering it
needs neither.
"""

import os
import threading
import time

import numpy as np
import pytest
from psygnal.containers import EventedDict

from fibsem.applications.autolamella.proposals import (
    DETECTION,
    Decision,
    DecisionOutcome,
    Proposal,
)
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    AutoLamellaTaskStatus,
    Experiment,
)
from fibsem.applications.autolamella.workflows.interaction import (
    Confirm,
    ConfirmDetection,
    ask,
)
from fibsem.applications.autolamella.workflows.review_responder import ReviewResponder
from fibsem.detection.detection import DetectedFeatures, LamellaCentre
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemImageMetadata,
    ImageSettings,
    MicroscopeState,
    Point,
)

TASK = "Mill Rough"


@pytest.fixture
def experiment(tmp_path):
    exp = Experiment(path=tmp_path, name="ask-exp")
    os.makedirs(exp.path, exist_ok=True)
    exp.task_protocol = AutoLamellaTaskProtocol()
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    lamella = exp.positions[0]
    lamella.task_state.name = TASK
    lamella.task_state.task_id = "run-1"
    lamella.task_state.status = AutoLamellaTaskStatus.InProgress
    return exp


def _detection(x: float = 10.0, y: float = 20.0) -> DetectedFeatures:
    feature = LamellaCentre()
    feature.px = Point(x, y)
    data = np.zeros((8, 8), dtype=np.uint8)
    return DetectedFeatures(
        features=[feature],
        image=data,
        mask=np.zeros((8, 8), dtype=np.uint8),
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        pixelsize=1e-9,
        # The image the question is about. Carried in memory, which is the
        # point: whether the task ever wrote one is its own business.
        fibsem_image=FibsemImage(
            data=data,
            metadata=FibsemImageMetadata(
                image_settings=ImageSettings(beam_type=BeamType.ION, hfw=8e-9),
                pixel_size=Point(1e-9, 1e-9),
                microscope_state=MicroscopeState(),
            ),
        ),
    )


def _ask_on_worker_thread(experiment, detection, abort=None, previous=None):
    """Ask the way a task does: on the workflow thread, blocking on the future,
    while the main thread goes on to answer.

    ``previous`` is the proposal already on the record, for a second question
    on the same task: a re-ask replaces it, and without this the wait below
    would return the old one the instant it looked.
    """
    lamella = experiment.positions[0]
    responder = ReviewResponder(experiment, lamella.id, TASK)
    outcome = {}

    def target():
        try:
            outcome["answer"] = ask(
                responder, ConfirmDetection(detection=detection), abort=abort
            )
        except BaseException as exc:  # noqa: BLE001 - the test inspects it
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        recorded = lamella.proposals.get(TASK)
        if recorded is not None and recorded is not previous:
            return thread, outcome, recorded
        time.sleep(0.005)
    raise AssertionError("the question was never recorded")


def _finish(thread, timeout_s=5.0):
    thread.join(timeout=timeout_s)
    assert not thread.is_alive(), "the asking task never resumed"


# ---------------------------------------------------------------------------
# The question, on the record
# ---------------------------------------------------------------------------


def test_asking_records_the_question_as_a_proposal(experiment):
    thread, outcome, proposal = _ask_on_worker_thread(experiment, _detection())

    assert proposal.kind == DETECTION
    assert proposal.values["features"] == [
        {"name": "LamellaCentre", "px": Point(10, 20)}
    ]
    assert proposal.pending, "nobody has answered it"

    experiment.withdraw_proposal(experiment.positions[0].id, TASK, "test over")
    _finish(thread)


def test_the_question_names_the_run_that_is_actually_in_progress(experiment):
    """What a decision has to name, and what lets it land while the task runs."""
    thread, outcome, proposal = _ask_on_worker_thread(experiment, _detection())

    assert proposal.task_id == "run-1"
    assert proposal.asking, "the run is parked on it"

    experiment.withdraw_proposal(experiment.positions[0].id, TASK, "test over")
    _finish(thread)


def test_a_request_with_no_value_to_record_is_refused_not_recorded(experiment):
    """A yes/no prompt carries nothing to put in the record, so it says so
    rather than recording an empty question nobody can answer usefully."""
    lamella = experiment.positions[0]
    responder = ReviewResponder(experiment, lamella.id, TASK)

    with pytest.raises(NotImplementedError):
        ask(responder, Confirm(message="Continue?"))

    assert lamella.proposals.get(TASK) is None


# ---------------------------------------------------------------------------
# The answer
# ---------------------------------------------------------------------------


def test_a_confirmed_decision_releases_the_task_with_the_decided_points(experiment):
    lamella = experiment.positions[0]
    thread, outcome, proposal = _ask_on_worker_thread(experiment, _detection())

    result = experiment.decide(
        lamella.id,
        TASK,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"features": [{"name": "LamellaCentre", "px": Point(12, 26)}]},
            task_id="run-1",
        ),
    )

    assert result.applied, result.reason
    _finish(thread)
    answer = outcome["answer"]
    assert answer.features[0].px == Point(12, 26), "the task got the corrected point"


def test_confirming_unchanged_answers_with_what_the_model_said(experiment):
    """The commonest answer: the model was right, and the task carries on with
    exactly what it proposed."""
    lamella = experiment.positions[0]
    thread, outcome, proposal = _ask_on_worker_thread(experiment, _detection())

    experiment.decide(
        lamella.id,
        TASK,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values=dict(proposal.values),
            task_id="run-1",
        ),
    )

    _finish(thread)
    assert outcome["answer"].features[0].px == Point(10, 20)


def test_the_answer_does_not_mutate_the_question(experiment):
    """The proposal is what the delta is measured against, so the correction
    must land on a copy."""
    sent = _detection()
    lamella = experiment.positions[0]
    thread, outcome, _ = _ask_on_worker_thread(experiment, sent)

    experiment.decide(
        lamella.id,
        TASK,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"features": [{"name": "LamellaCentre", "px": Point(99, 99)}]},
            task_id="run-1",
        ),
    )

    _finish(thread)
    assert sent.features[0].px == Point(10, 20), "the request is untouched"


def test_the_delta_is_on_the_record_afterwards(experiment):
    """What the whole thing is for: the model said one thing, a person said
    another, and the difference is on the lamella."""
    lamella = experiment.positions[0]
    thread, _, proposal = _ask_on_worker_thread(experiment, _detection())

    experiment.decide(
        lamella.id,
        TASK,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"features": [{"name": "LamellaCentre", "px": Point(12, 26)}]},
            task_id="run-1",
        ),
    )
    _finish(thread)

    assert proposal.values["features"][0]["px"] == Point(10, 20), "what was proposed"
    assert proposal.current.values["features"][0]["px"] == Point(12, 26), (
        "what was said"
    )
    assert proposal.current.author.name == "op"


def test_a_rejected_decision_raises_on_the_asking_task(experiment):
    lamella = experiment.positions[0]
    thread, outcome, _ = _ask_on_worker_thread(experiment, _detection())

    experiment.decide(
        lamella.id,
        TASK,
        Decision(
            outcome=DecisionOutcome.Rejected,
            author="human:op",
            reason="no lamella here",
            task_id="run-1",
        ),
    )

    _finish(thread)
    assert isinstance(outcome["error"], RuntimeError)
    assert "no lamella here" in str(outcome["error"])


# ---------------------------------------------------------------------------
# Endings that are not answers
# ---------------------------------------------------------------------------


def test_withdrawing_the_question_unwinds_the_asking_task(experiment):
    lamella = experiment.positions[0]
    thread, outcome, _ = _ask_on_worker_thread(experiment, _detection())

    experiment.withdraw_proposal(lamella.id, TASK, "the run was stopped")

    _finish(thread)
    assert "answer" not in outcome, "nothing was answered"


def test_an_abort_closes_the_question_and_stops_listening(experiment):
    """Stop mid-question: the task unwinds, and a decision that arrives after
    it must not try to release a run that has gone."""
    lamella = experiment.positions[0]
    stop = threading.Event()
    thread, outcome, proposal = _ask_on_worker_thread(
        experiment, _detection(), abort=stop.is_set
    )

    stop.set()
    _finish(thread)

    assert isinstance(outcome["error"], InterruptedError)
    assert not proposal.asking, "the question is closed"
    assert proposal.pending, "and it was never answered"


def test_a_decision_arriving_after_an_abort_is_harmless(experiment):
    lamella = experiment.positions[0]
    stop = threading.Event()
    thread, outcome, _ = _ask_on_worker_thread(
        experiment, _detection(), abort=stop.is_set
    )
    stop.set()
    _finish(thread)
    lamella.task_state.status = AutoLamellaTaskStatus.Completed

    result = experiment.decide(
        lamella.id,
        TASK,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"features": [{"name": "LamellaCentre", "px": Point(1, 1)}]},
            task_id="run-1",
        ),
    )

    assert result.applied, "the record still takes it; nobody is waiting, that is all"


# ---------------------------------------------------------------------------
# The record survives the round trip
# ---------------------------------------------------------------------------


def test_a_recorded_question_round_trips_through_the_file(experiment):
    thread, _, proposal = _ask_on_worker_thread(experiment, _detection())

    back = Proposal.from_dict(proposal.to_dict())

    assert back.kind == DETECTION
    assert back.values["features"] == [{"name": "LamellaCentre", "px": Point(10, 20)}]
    assert not back.asking, "a question does not survive a reload"

    experiment.withdraw_proposal(experiment.positions[0].id, TASK, "test over")
    _finish(thread)


def test_the_delta_is_per_feature_not_one_number_for_the_set(experiment):
    """Which feature the model got wrong is the part worth keeping. Found in
    the app: the record said ``{'features': None}`` because a delta over a
    named set had nowhere to go."""
    lamella = experiment.positions[0]
    thread, _, proposal = _ask_on_worker_thread(experiment, _detection())
    proposal.values["features"].append({"name": "ImageCentre", "px": Point(50, 60)})

    result = experiment.decide(
        lamella.id,
        TASK,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={
                "features": [
                    {"name": "LamellaCentre", "px": Point(12, 26)},
                    {"name": "ImageCentre", "px": Point(50, 60)},
                ]
            },
            task_id="run-1",
        ),
    )
    _finish(thread)

    assert result.applied, result.reason
    moved = result.delta["features"]
    assert moved["LamellaCentre"] == Point(2, 6), "moved right and down"
    assert moved["ImageCentre"] == Point(0, 0), "the model was right about this one"


def test_the_question_image_goes_in_its_own_directory(experiment):
    """Under the item, but not among the task's outputs: a question's picture
    is not a result, it is what somebody was shown when they were asked."""
    lamella = experiment.positions[0]
    thread, _, proposal = _ask_on_worker_thread(experiment, _detection())

    stored = proposal.provenance["reference_image"]

    assert stored.startswith("questions/"), stored
    assert "/" in stored and "\\" not in stored, "portable on any platform"
    assert os.path.exists(os.path.join(str(lamella.path), stored))

    experiment.withdraw_proposal(lamella.id, TASK, "test over")
    _finish(thread)


def test_a_re_ask_writes_its_own_picture(experiment):
    """The run is in the name, so a second question does not overwrite the one
    a superseded proposal still points at."""
    lamella = experiment.positions[0]
    thread, _, first = _ask_on_worker_thread(experiment, _detection())
    experiment.withdraw_proposal(lamella.id, TASK, "asking again")
    _finish(thread)
    lamella.task_state.task_id = "run-2"

    thread, _, second = _ask_on_worker_thread(experiment, _detection(), previous=first)

    assert first.provenance["reference_image"] != second.provenance["reference_image"]
    assert os.path.exists(
        os.path.join(str(lamella.path), first.provenance["reference_image"])
    ), "the first question's picture survives"

    experiment.withdraw_proposal(lamella.id, TASK, "test over")
    _finish(thread)


def test_recording_a_question_says_so(experiment):
    """The inbox otherwise only re-derives when a task *finishes*, and an
    in-run question is raised halfway through one -- so without this it would
    not appear until something else happened to refresh the tab. Found in the
    app: the question was on the record and invisible until a filter was
    touched."""
    lamella = experiment.positions[0]
    heard = []
    experiment.asked.connect(lambda item_id, task: heard.append((item_id, task)))

    thread, _, _ = _ask_on_worker_thread(experiment, _detection())

    assert heard == [(lamella.id, TASK)]

    experiment.withdraw_proposal(lamella.id, TASK, "test over")
    _finish(thread)


def test_an_answered_question_is_kept_when_the_next_one_is_asked(experiment):
    """Same rule as a task's own proposal: a decided one is superseded, so its
    answer and its delta stay on the record."""
    lamella = experiment.positions[0]
    thread, _, first = _ask_on_worker_thread(experiment, _detection())
    experiment.decide(
        lamella.id,
        TASK,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"features": [{"name": "LamellaCentre", "px": Point(12, 26)}]},
            task_id="run-1",
        ),
    )
    _finish(thread)
    lamella.task_state.task_id = "run-2"

    thread, _, second = _ask_on_worker_thread(experiment, _detection(), previous=first)

    assert second.superseded == [first], "the answered one is still there"

    experiment.withdraw_proposal(lamella.id, TASK, "test over")
    _finish(thread)


def test_an_unanswered_question_is_replaced_rather_than_kept(experiment):
    """Nothing was answered, so there is nothing worth keeping under it."""
    lamella = experiment.positions[0]
    thread, _, first = _ask_on_worker_thread(experiment, _detection())
    experiment.withdraw_proposal(lamella.id, TASK, "asking again")
    _finish(thread)
    lamella.task_state.task_id = "run-2"

    thread, _, second = _ask_on_worker_thread(experiment, _detection(), previous=first)

    assert second.superseded == [first], "withdrawn is decided, so it is kept"

    experiment.withdraw_proposal(lamella.id, TASK, "test over")
    _finish(thread)


# ---------------------------------------------------------------------------
# More than one question in a run
# ---------------------------------------------------------------------------


def _answer_it(experiment, lamella, px):
    return experiment.decide(
        lamella.id,
        TASK,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"features": [{"name": "LamellaCentre", "px": px}]},
            task_id=lamella.task_state.task_id,
        ),
    )


def test_a_task_can_ask_twice_in_one_run(experiment):
    """Two detections back to back: ordinary, and each is its own question.
    The task blocks on the first answer before asking again, so there is never
    more than one open."""
    lamella = experiment.positions[0]
    thread, first_outcome, first = _ask_on_worker_thread(experiment, _detection())
    assert _answer_it(experiment, lamella, Point(11, 21)).applied
    _finish(thread)

    thread, second_outcome, second = _ask_on_worker_thread(
        experiment, _detection(30, 40), previous=first
    )
    assert _answer_it(experiment, lamella, Point(31, 41)).applied
    _finish(thread)

    assert first_outcome["answer"].features[0].px == Point(11, 21)
    assert second_outcome["answer"].features[0].px == Point(31, 41)
    assert second.superseded == [first], "the first is kept under the second"


def test_two_questions_in_one_run_keep_their_own_pictures(experiment):
    """They share a run id, so the run alone does not make the file name
    unique -- the first question's image would be written over and its record
    would point at the second's."""
    lamella = experiment.positions[0]
    thread, _, first = _ask_on_worker_thread(experiment, _detection())
    _answer_it(experiment, lamella, Point(11, 21))
    _finish(thread)
    thread, _, second = _ask_on_worker_thread(
        experiment, _detection(30, 40), previous=first
    )

    one = first.provenance["reference_image"]
    two = second.provenance["reference_image"]
    assert one != two, (one, two)
    assert os.path.exists(os.path.join(str(lamella.path), one))
    assert os.path.exists(os.path.join(str(lamella.path), two))

    experiment.withdraw_proposal(lamella.id, TASK, "test over")
    _finish(thread)


def test_only_the_open_question_is_in_the_inbox(experiment):
    """One row per item and task, and it is the question being asked now: the
    answered one is history, not something still to do."""
    lamella = experiment.positions[0]
    thread, _, first = _ask_on_worker_thread(experiment, _detection())
    _answer_it(experiment, lamella, Point(11, 21))
    _finish(thread)
    thread, _, second = _ask_on_worker_thread(
        experiment, _detection(30, 40), previous=first
    )

    pending = [p for _, _, p in experiment.pending_proposals()]

    assert pending == [second]
    assert second.asking and not first.asking

    experiment.withdraw_proposal(lamella.id, TASK, "test over")
    _finish(thread)


def test_the_record_says_which_model_proposed_it(experiment):
    """``ConfirmDetection`` is the question's type, not its author. A
    correction is only comparable against the checkpoint that produced it, so
    the model and the checkpoint are both on the record."""
    lamella = experiment.positions[0]
    detection = _detection()
    detection.checkpoint = "/models/autolamella-mega-20240107.pt"

    thread, _, proposal = _ask_on_worker_thread(experiment, detection)

    assert proposal.provenance["proposer"] == "autolamella-mega-20240107"
    assert proposal.provenance["checkpoint"] == "/models/autolamella-mega-20240107.pt"

    experiment.withdraw_proposal(lamella.id, TASK, "test over")
    _finish(thread)


def test_a_question_with_no_checkpoint_still_names_a_proposer(experiment):
    lamella = experiment.positions[0]

    thread, _, proposal = _ask_on_worker_thread(experiment, _detection())

    assert proposal.provenance["proposer"] == "segmentation model"

    experiment.withdraw_proposal(lamella.id, TASK, "test over")
    _finish(thread)
