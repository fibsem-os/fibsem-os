"""How a question crosses into the record and back.

``proposal_for`` turns a request into the proposal that records it, and
``answer_from`` turns a decision on that proposal back into the answer the
asking task expects -- the same type it would have got from a prompt. Between
the two sits ``Experiment.ask_proposal`` and ``Experiment.decide``, so these
tests run the real record path: build, record, decide, read the answer back.

Nothing here asks or waits. Whoever owns the wait does that, and releasing a
parked task on a decision is tested where that lives. What is pinned here is
what gets written down, and that the answer read back off it is the one a task
can act on.

No Qt and no microscope: the point of recording a question is that answering it
needs neither.
"""

import os
import time

import numpy as np
import pytest
from psygnal.containers import EventedDict

from fibsem import conversions
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
    ReviewDetection,
)
from fibsem.applications.autolamella.workflows.question_adapters import (
    answer_from,
    answered,
    proposal_for,
)
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


def _detection(x: float = 10.0, y: float = 20.0, save_to=None) -> DetectedFeatures:
    """A detection on an 8x8 image. ``save_to`` writes the image there first,
    the way ``take_image_and_detect_features`` always does, so it knows its own
    file; without it the picture exists only in memory."""
    feature = LamellaCentre()
    feature.px = Point(x, y)
    data = np.zeros((8, 8), dtype=np.uint8)
    detection = DetectedFeatures(
        features=[feature],
        image=data,
        mask=np.zeros((8, 8), dtype=np.uint8),
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        pixelsize=1e-9,
        fibsem_image=FibsemImage(
            data=data,
            metadata=FibsemImageMetadata(
                image_settings=ImageSettings(beam_type=BeamType.ION, hfw=8e-9),
                pixel_size=Point(1e-9, 1e-9),
                microscope_state=MicroscopeState(),
            ),
        ),
    )
    if save_to is not None:
        detection.fibsem_image.save(os.path.join(str(save_to), "ml-test_ib"))
    return detection


def _record(experiment, detection):
    """Record a detection question the way its asker will: build the proposal
    from the request, then put it on the item as the one the run is parked on.
    Returns the request too, because the answer is read back against it."""
    lamella = experiment.positions[0]
    if detection.fibsem_image.filepath is None:
        # The real asker always saves what it detects on, into the item's
        # folder; a test that says nothing about the picture gets the same.
        stem = f"ml-{len(os.listdir(str(lamella.path)))}_ib"
        detection.fibsem_image.save(os.path.join(str(lamella.path), stem))
    request = ReviewDetection(detection=detection, item_id=lamella.id, task_name=TASK)
    proposal = proposal_for(request, experiment, lamella)
    assert proposal is not None
    assert experiment.ask_proposal(lamella.id, TASK, proposal)
    return request, proposal


def _confirm(experiment, px, names=("LamellaCentre",)):
    lamella = experiment.positions[0]
    points = px if isinstance(px, (list, tuple)) else [px]
    return experiment.decide(
        lamella.id,
        TASK,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"features": [{"name": n, "px": p} for n, p in zip(names, points)]},
            task_id=lamella.task_state.task_id,
        ),
    )


def _pump(predicate, timeout_s=5.0) -> bool:
    """Wait for something the GUI thread has to deliver.

    ``asked`` is *posted* to the main thread when it is raised off it, so with
    a Qt application around nothing arrives until somebody spins the loop.
    Raised on the main thread, as here, the first look succeeds.
    """
    try:
        from PyQt5.QtCore import QCoreApplication

        app = QCoreApplication.instance()
    except ImportError:
        app = None
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        if app is not None:
            app.processEvents()
        time.sleep(0.005)
    return predicate()


# ---------------------------------------------------------------------------
# The question, on the record
# ---------------------------------------------------------------------------


def test_a_question_is_recorded_as_a_proposal(experiment):
    _, proposal = _record(experiment, _detection())

    assert experiment.positions[0].proposals[TASK] is proposal
    assert proposal.kind == DETECTION
    assert proposal.values["features"] == [
        {"name": "LamellaCentre", "px": Point(10, 20)}
    ]
    assert proposal.pending, "nobody has answered it"


def test_the_question_names_the_run_that_is_actually_in_progress(experiment):
    """What a decision has to name, and what lets it land while the task runs."""
    _, proposal = _record(experiment, _detection())

    assert proposal.task_id == "run-1"
    assert proposal.asking, "the run is parked on it"


def test_a_request_with_no_value_has_nothing_to_record(experiment):
    """A yes/no prompt carries nothing to put in the record, so it stays on the
    prompt path rather than becoming an empty question nobody can answer
    usefully -- and no decision can be read back as its answer."""
    lamella = experiment.positions[0]
    request = Confirm(message="Continue?")

    assert proposal_for(request, experiment, lamella) is None
    with pytest.raises(NotImplementedError):
        answer_from(request, {})
    assert lamella.proposals.get(TASK) is None


def test_the_detection_question_as_it_always_was_is_not_recorded(experiment):
    """``ConfirmDetection`` is asked by ``update_detection_ui``, here and in
    code outside this repository, and answered on the Detection tab. It stays
    exactly that: only the second request type has anything to record."""
    lamella = experiment.positions[0]
    detection = _detection()
    detection.fibsem_image.save(os.path.join(str(lamella.path), "ml-plain_ib"))
    request = ConfirmDetection(detection=detection)

    assert proposal_for(request, experiment, lamella) is None
    with pytest.raises(NotImplementedError):
        answer_from(request, {})


# ---------------------------------------------------------------------------
# The answer
# ---------------------------------------------------------------------------


def test_a_confirmed_decision_answers_with_the_decided_points(experiment):
    request, proposal = _record(experiment, _detection())

    result = _confirm(experiment, Point(12, 26))

    assert result.applied, result.reason
    answer = answer_from(request, proposal.current.values)
    assert answer.features[0].px == Point(12, 26), "the task gets the corrected point"


def test_the_answer_moves_in_metres_too(experiment):
    """``feature_m`` is what a task acts on -- every stage move reads it, none
    reads ``px`` -- so a corrected pixel with the model's metres still beside
    it would be recorded as a correction and then milled where the model said."""
    detection = _detection()
    detection.features[0].feature_m = conversions.image_to_microscope_image_coordinates(
        detection.features[0].px, detection.image.data, detection.pixelsize
    )
    request, proposal = _record(experiment, detection)

    _confirm(experiment, Point(2, 6))

    answer = answer_from(request, proposal.current.values)
    expected = conversions.image_to_microscope_image_coordinates(
        Point(2, 6), detection.image.data, detection.pixelsize
    )
    assert answer.features[0].feature_m == expected
    assert answer.features[0].feature_m != detection.features[0].feature_m, (
        "the fixture has to move the point far enough to change the metres"
    )


def test_confirming_unchanged_answers_with_what_the_model_said(experiment):
    """The commonest answer: the model was right, and the task carries on with
    exactly what it proposed."""
    lamella = experiment.positions[0]
    request, proposal = _record(experiment, _detection())

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

    answer = answer_from(request, proposal.current.values)
    assert answer.features[0].px == Point(10, 20)


def test_the_answer_does_not_mutate_the_question(experiment):
    """The proposal is what the delta is measured against, so the correction
    must land on a copy."""
    sent = _detection()
    request, proposal = _record(experiment, sent)

    _confirm(experiment, Point(99, 99))
    answer_from(request, proposal.current.values)

    assert sent.features[0].px == Point(10, 20), "the request is untouched"


def test_the_delta_is_on_the_record_afterwards(experiment):
    """What the whole thing is for: the model said one thing, a person said
    another, and the difference is on the lamella."""
    _, proposal = _record(experiment, _detection())

    _confirm(experiment, Point(12, 26))

    assert proposal.values["features"][0]["px"] == Point(10, 20), "what was proposed"
    assert proposal.current.values["features"][0]["px"] == Point(12, 26), (
        "what was said"
    )
    assert proposal.current.author.name == "op"


def test_the_delta_is_per_feature_not_one_number_for_the_set(experiment):
    """Which feature the model got wrong is the part worth keeping. Found in
    the app: the record said ``{'features': None}`` because a delta over a
    named set had nowhere to go."""
    _, proposal = _record(experiment, _detection())
    proposal.values["features"].append({"name": "ImageCentre", "px": Point(50, 60)})

    result = _confirm(
        experiment,
        [Point(12, 26), Point(50, 60)],
        names=("LamellaCentre", "ImageCentre"),
    )

    assert result.applied, result.reason
    moved = result.delta["features"]
    assert moved["LamellaCentre"] == Point(2, 6), "moved right and down"
    assert moved["ImageCentre"] == Point(0, 0), "the model was right about this one"


def test_answering_closes_the_question(experiment):
    """Answered, nothing is waiting on it, so it stops being the exception that
    lets a decision land on a running task."""
    _, proposal = _record(experiment, _detection())

    _confirm(experiment, Point(12, 26))

    assert not proposal.asking


# ---------------------------------------------------------------------------
# The record survives the round trip
# ---------------------------------------------------------------------------


def test_a_recorded_question_round_trips_through_the_file(experiment):
    _, proposal = _record(experiment, _detection())

    back = Proposal.from_dict(proposal.to_dict())

    assert back.kind == DETECTION
    assert back.values["features"] == [{"name": "LamellaCentre", "px": Point(10, 20)}]
    assert not back.asking, "a question does not survive a reload"


def test_the_record_names_the_file_the_task_saved(experiment):
    """The path actually written -- beam suffix, extension and all -- rather
    than one rebuilt from a filename stem, which is how a reader ends up
    hunting for a file. Nothing new is written: there is one picture of this
    acquisition and the record points at it."""
    lamella = experiment.positions[0]
    before = sorted(os.listdir(str(lamella.path)))

    _, proposal = _record(experiment, _detection(save_to=lamella.path))

    stored = proposal.provenance["reference_image"]
    assert stored == "ml-test_ib.tif"
    assert os.path.isfile(os.path.join(str(lamella.path), stored))
    assert sorted(os.listdir(str(lamella.path))) == sorted(before + [stored])


def test_a_file_in_a_folder_under_the_item_is_named_portably(experiment):
    """Relative to the item's folder, forward slashes whatever the platform:
    the record has to survive a moved experiment and a different machine."""
    lamella = experiment.positions[0]
    sub = os.path.join(str(lamella.path), "detections")
    os.makedirs(sub)

    _, proposal = _record(experiment, _detection(save_to=sub))

    assert proposal.provenance["reference_image"] == "detections/ml-test_ib.tif"


def test_a_file_outside_the_item_is_named_in_full(experiment, tmp_path_factory):
    """Not under the folder, so there is no relative name worth keeping; a
    reader's join onto the item's folder leaves a full path alone."""
    lamella = experiment.positions[0]
    elsewhere = tmp_path_factory.mktemp("elsewhere")

    _, proposal = _record(experiment, _detection(save_to=elsewhere))

    stored = proposal.provenance["reference_image"]
    assert os.path.isabs(stored)
    assert os.path.isfile(os.path.join(str(lamella.path), stored))


def test_a_question_on_a_picture_that_was_never_saved_is_not_recorded(experiment):
    """The answer is given on the image, so a review without one is a review
    nobody can see. It stays on the prompt path, which shows the picture it has
    in memory -- and nothing is written or recorded on its behalf."""
    lamella = experiment.positions[0]
    before = sorted(os.listdir(str(lamella.path)))
    request = ReviewDetection(
        detection=_detection(), item_id=lamella.id, task_name=TASK
    )

    assert proposal_for(request, experiment, lamella) is None
    assert lamella.proposals.get(TASK) is None
    assert sorted(os.listdir(str(lamella.path))) == before


def test_recording_a_question_says_so(experiment):
    """The inbox otherwise only re-derives when a task *finishes*, and an
    in-run question is raised halfway through one -- so without this it would
    not appear until something else happened to refresh the tab. Found in the
    app: the question was on the record and invisible until a filter was
    touched."""
    lamella = experiment.positions[0]
    heard = []
    experiment.asked.connect(lambda item_id, task: heard.append((item_id, task)))

    _record(experiment, _detection())

    assert _pump(lambda: heard == [(lamella.id, TASK)]), heard


# ---------------------------------------------------------------------------
# What is kept under the next question
# ---------------------------------------------------------------------------


def test_an_answered_question_is_kept_when_the_next_one_is_asked(experiment):
    """Same rule as a task's own proposal: a decided one is superseded, so its
    answer and its delta stay on the record."""
    lamella = experiment.positions[0]
    _, first = _record(experiment, _detection())
    _confirm(experiment, Point(12, 26))
    lamella.task_state.task_id = "run-2"

    _, second = _record(experiment, _detection())

    assert second.superseded == [first], "the answered one is still there"


def test_a_withdrawn_question_is_kept_when_the_next_one_is_asked(experiment):
    """Withdrawn is decided: a question raised and abandoned is worth knowing,
    because it usually means nobody was there."""
    lamella = experiment.positions[0]
    _, first = _record(experiment, _detection())
    experiment.withdraw_proposal(lamella.id, TASK, "asking again")
    lamella.task_state.task_id = "run-2"

    _, second = _record(experiment, _detection())

    assert second.superseded == [first]


def test_a_question_left_open_is_replaced_rather_than_kept(experiment):
    """Nothing was said about it -- not even that it was taken back -- so there
    is nothing worth keeping under the next one."""
    _, first = _record(experiment, _detection())

    _, second = _record(experiment, _detection(30, 40))

    assert second.superseded == []
    assert experiment.positions[0].proposals[TASK] is second


# ---------------------------------------------------------------------------
# More than one question in a run
# ---------------------------------------------------------------------------


def test_a_task_can_ask_twice_in_one_run(experiment):
    """Two detections back to back: ordinary, and each is its own question.
    The task blocks on the first answer before asking again, so there is never
    more than one open."""
    first_request, first = _record(experiment, _detection())
    assert _confirm(experiment, Point(11, 21)).applied

    second_request, second = _record(experiment, _detection(30, 40))
    assert _confirm(experiment, Point(31, 41)).applied

    first_answer = answer_from(first_request, first.current.values)
    second_answer = answer_from(second_request, second.current.values)
    assert first_answer.features[0].px == Point(11, 21)
    assert second_answer.features[0].px == Point(31, 41)
    assert second.superseded == [first], "the first is kept under the second"


def test_only_the_open_question_is_in_the_inbox(experiment):
    """One row per item and task, and it is the question being asked now: the
    answered one is history, not something still to do."""
    _, first = _record(experiment, _detection())
    _confirm(experiment, Point(11, 21))
    _, second = _record(experiment, _detection(30, 40))

    pending = [p for _, _, p in experiment.pending_proposals()]

    assert pending == [second]
    assert second.asking and not first.asking


# ---------------------------------------------------------------------------
# Who proposed it
# ---------------------------------------------------------------------------


def test_the_record_says_which_model_proposed_it(experiment):
    """``ConfirmDetection`` is the question's type, not its author. A
    correction is only comparable against the checkpoint that produced it, so
    the model and the checkpoint are both on the record."""
    detection = _detection()
    detection.checkpoint = "/models/autolamella-mega-20240107.pt"

    _, proposal = _record(experiment, detection)

    assert proposal.provenance["proposer"] == "autolamella-mega-20240107"
    assert proposal.provenance["checkpoint"] == "/models/autolamella-mega-20240107.pt"


def test_a_question_with_no_checkpoint_still_names_a_proposer(experiment):
    _, proposal = _record(experiment, _detection())

    assert proposal.provenance["proposer"] == "segmentation model"


# ---------------------------------------------------------------------------
# What else an answer entails
# ---------------------------------------------------------------------------


def test_a_confirmed_detection_writes_its_training_data(experiment, monkeypatch):
    """What the Detection tab writes on its Continue click, so a question moved
    onto the record does not quietly stop writing it: the corrected set,
    measured against what the model said."""
    from fibsem.detection import utils as det_utils

    written = []
    monkeypatch.setattr(
        det_utils,
        "save_ml_feature_data",
        lambda det, initial_features=None: written.append((det, initial_features)),
    )
    request, proposal = _record(experiment, _detection())
    _confirm(experiment, Point(2, 6))
    answer = answer_from(request, proposal.current.values)

    answered(request, answer)

    assert len(written) == 1
    assert written[0][0] is answer
    assert written[0][1][0].px == Point(10, 20)
    assert answer.mask.dtype == np.uint8, "PIL cannot write anything else"


def test_a_failed_export_does_not_fail_the_answer(experiment, monkeypatch, caplog):
    """By then the decision is on the record and the task is about to carry
    on; a CSV that could not be written is a log line, not a failed run."""
    from fibsem.detection import utils as det_utils

    def boom(det, initial_features=None):
        raise OSError("disk full")

    monkeypatch.setattr(det_utils, "save_ml_feature_data", boom)
    request, proposal = _record(experiment, _detection())
    _confirm(experiment, Point(2, 6))

    answered(request, answer_from(request, proposal.current.values))

    assert "what follows an answer failed" in caplog.text


def test_a_request_with_no_adapter_has_nothing_to_follow(experiment):
    answered(Confirm(message="Continue?"), True)
