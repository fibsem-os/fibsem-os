"""Correcting a detection in the Review tab.

The renderer for the ``detection`` kind: one draggable marker per feature, on
the image the model ran on, pre-placed where it put them. Confirm answers with
wherever they have been left -- moved or not -- because the task is parked on
the whole set and an unmoved marker is the answer that says the model was
right.

A feature's point is already in image pixels, unlike a point of interest, so
these also pin that nothing converts it on the way in or out.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from psygnal.containers import EventedDict

pytest.importorskip("PyQt5")

import fibsem.applications.autolamella.ui.review_tab_widget as R
from fibsem.applications.autolamella.proposals import (
    DETECTION,
    Decision,
    DecisionOutcome,
    Proposal,
)
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    AutoLamellaTaskState,
    AutoLamellaTaskStatus,
    Experiment,
)
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemImageMetadata,
    FibsemStagePosition,
    ImageSettings,
    MicroscopeState,
    Point,
)

TASK = "Mill Rough"
RUN = "run-1"
PIXELSIZE = 25e-9


def _fib_image() -> FibsemImage:
    metadata = FibsemImageMetadata(
        image_settings=ImageSettings(beam_type=BeamType.ION, hfw=512 * PIXELSIZE),
        pixel_size=Point(PIXELSIZE, PIXELSIZE),
        microscope_state=MicroscopeState(stage_position=FibsemStagePosition()),
    )
    return FibsemImage(data=np.zeros((512, 512), dtype=np.uint8), metadata=metadata)


@pytest.fixture
def experiment(tmp_path) -> Experiment:
    exp = Experiment(path=tmp_path, name="detection-exp")
    exp.task_protocol = AutoLamellaTaskProtocol()
    os.makedirs(exp.path, exist_ok=True)
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    lamella = exp.positions[0]
    lamella.path.mkdir(parents=True, exist_ok=True)
    ref = os.path.join(str(lamella.path), "ref_detection_ib")
    _fib_image().save(ref)
    # The task's record: waiting on the decision, as a question's is once the
    # task is no longer parked on it; the live task_state below is InProgress.
    lamella.task_history.append(
        AutoLamellaTaskState(name=TASK, status=AutoLamellaTaskStatus.AwaitingDecision)
    )
    lamella.task_state.name = TASK
    lamella.task_state.task_id = RUN
    lamella.task_state.status = AutoLamellaTaskStatus.InProgress
    lamella.proposals[TASK] = [
        Proposal(
            kind=DETECTION,
            values={
                "features": [
                    {"name": "LamellaCentre", "px": Point(256.0, 256.0)},
                    {"name": "ImageCentre", "px": Point(100.0, 120.0)},
                ]
            },
            provenance={
                "task_id": RUN,
                "proposer": "ConfirmDetection",
                "reference_image": ref + ".tif",
                "in_run": True,
            },
        )
    ]
    return exp


@pytest.fixture
def tab(qapp, experiment) -> R.ReviewTabWidget:
    widget = R.ReviewTabWidget()
    widget.set_experiment(experiment)
    return widget


def _renderer(tab):
    renderer = tab._renderer_for(DETECTION)
    experiment = tab._experiment
    lamella = experiment.positions[0]
    renderer.set_proposal(experiment, lamella, TASK, lamella.proposal(TASK))
    return renderer


def test_the_kind_has_a_renderer_of_its_own(tab):
    assert isinstance(_renderer(tab), R.DetectionReviewRenderer)


def test_every_feature_is_a_point_on_one_overlay(tab):
    """One overlay holding them all, which is what makes every marker
    selectable and draggable -- a canvas arms one overlay at a time, so a
    marker per overlay would leave only the last one editable."""
    renderer = _renderer(tab)

    points = renderer._controller.overlay_points(BeamType.ION, renderer.OVERLAY)

    assert points == [(256.0, 256.0), (100.0, 120.0)], "in image pixels, unconverted"


def test_each_feature_keeps_the_colour_it_carries(tab):
    """A feature knows its own colour, so a lamella centre looks the same here
    as it does in the detection widget -- and two features on one image are
    told apart by it rather than all being magenta."""
    renderer = _renderer(tab)

    centre = renderer._colour("LamellaCentre")
    other = renderer._colour("ImageCentre")

    assert centre and other and centre != other, (centre, other)


def test_a_feature_this_build_does_not_know_still_gets_a_colour(tab):
    """A name from a newer model is not an error; it is a feature this build
    has not heard of."""
    assert _renderer(tab)._colour("SomethingNewer") == "magenta"


def test_confirming_untouched_answers_with_what_the_model_said(tab):
    """The commonest answer, and the one that records the model as right."""
    renderer = _renderer(tab)

    values = renderer.current_values()

    assert values["features"] == [
        {"name": "LamellaCentre", "px": Point(256.0, 256.0)},
        {"name": "ImageCentre", "px": Point(100.0, 120.0)},
    ]


def test_moving_a_marker_is_what_confirm_carries(tab):
    renderer = _renderer(tab)
    from fibsem.ui.widgets.canvas.canvas_state import PointsSpec

    renderer._controller.set_overlay(
        BeamType.ION,
        PointsSpec(
            id=renderer.OVERLAY,
            points=[(300.0, 280.0), (100.0, 120.0)],
            marker="+",
        ),
    )

    values = renderer.current_values()

    assert values["features"][0] == {"name": "LamellaCentre", "px": Point(300.0, 280.0)}
    assert values["features"][1]["px"] == Point(100.0, 120.0), "the other is untouched"


def test_every_feature_is_answered_even_when_only_one_moved(tab):
    """The task is parked on the whole set, so a partial answer would leave it
    with a feature it never got a point for."""
    renderer = _renderer(tab)

    names = [f["name"] for f in renderer.current_values()["features"]]

    assert names == ["LamellaCentre", "ImageCentre"]


def test_once_decided_the_correction_is_drawn_beside_the_proposal(tab, experiment):
    """The delta is shown, not only recorded: what was proposed stays in
    orange under what was decided."""
    renderer = _renderer(tab)
    decision = Decision(
        outcome=DecisionOutcome.Confirmed,
        author="human:op",
        values={"features": [{"name": "LamellaCentre", "px": Point(300.0, 280.0)}]},
        task_id=RUN,
    )

    renderer.set_read_only(decision)

    assert renderer._controller.overlay_points(BeamType.ION, "proposed") == [
        (256.0, 256.0),
        (100.0, 120.0),
    ], "what the model said, still drawn"
    assert renderer._controller.overlay_points(BeamType.ION, renderer.OVERLAY) == [
        (300.0, 280.0)
    ], "what was decided"


def test_the_line_says_what_was_found_and_where(tab):
    renderer = _renderer(tab)

    fact = renderer._fact()

    assert "LamellaCentre, ImageCentre" in fact
    assert "ref_detection_ib.tif" in fact


def test_a_question_with_no_image_on_disk_still_reads(tab, experiment):
    """A detection is asked the moment the image comes back, and whether it
    was saved depends on the task's settings -- so the panel has to hold up
    without one."""
    lamella = experiment.positions[0]
    lamella.proposal(TASK).provenance["reference_image"] = ""
    renderer = tab._renderer_for(DETECTION)

    renderer.set_proposal(experiment, lamella, TASK, lamella.proposal(TASK))

    # isHidden rather than isVisible: the renderer has no shown parent here,
    # which makes every child report itself invisible whatever it was set to.
    assert not renderer.no_image.isHidden()
    assert renderer.current_values()["features"], "the record still answers"


# ---------------------------------------------------------------------------
# Holding the workflow
# ---------------------------------------------------------------------------


def test_a_question_the_run_is_parked_on_gets_its_own_group(tab, experiment):
    """Everything under Waiting can be left for later. This one cannot: the
    task is stopped until it is answered."""
    experiment.positions[0].proposal(TASK).asking = True

    tab.refresh()
    headers = [h.label.text() for h in tab._headers]

    assert any(h.startswith("Holding the workflow · 1") for h in headers), headers
    assert not any(h.startswith("Waiting") for h in headers), "it is not just waiting"


def test_it_goes_back_under_waiting_once_nothing_is_parked_on_it(tab, experiment):
    experiment.positions[0].proposal(TASK).asking = False

    tab.refresh()
    headers = [h.label.text() for h in tab._headers]

    assert any(h.startswith("Waiting · 1") for h in headers), headers
    assert not any(h.startswith("Holding") for h in headers)


def test_the_badge_counts_it_like_any_other_pending_proposal(tab, experiment):
    """Being held is about urgency, not about whether it is pending -- the
    count of what is undecided must not change when it moves group."""
    experiment.positions[0].proposal(TASK).asking = True
    tab.refresh()
    held = tab._pending

    experiment.positions[0].proposal(TASK).asking = False
    tab.refresh()

    assert held == tab._pending == 1


def test_the_line_says_the_task_is_parked_rather_than_nothing_is_held(tab, experiment):
    """ "nothing is held" is true of the requires edges and false of the run:
    the task itself is stopped, waiting to be told."""
    lamella = experiment.positions[0]
    lamella.proposal(TASK).asking = True
    renderer = _renderer(tab)

    line = renderer.line.text()

    assert "waiting on this" in line
    assert "nothing is held" not in line


def test_dragged_points_come_back_as_plain_floats(tab, experiment):
    """The canvas hands back numpy scalars and the record is YAML, which
    refuses an np.float64 outright -- so a decision made by dragging would be
    accepted and then fail to save. Found by answering one in the app."""
    import yaml

    renderer = _renderer(tab)
    from fibsem.ui.widgets.canvas.canvas_state import PointsSpec

    renderer._controller.set_overlay(
        BeamType.ION,
        PointsSpec(
            id=renderer.OVERLAY,
            points=[
                (np.float64(300.5), np.float64(280.25)),
                (np.float64(101.0), np.float64(121.0)),
            ],
            marker="+",
        ),
    )

    values = renderer.current_values()

    for feature in values["features"]:
        assert type(feature["px"].x) is float, type(feature["px"].x)
        assert type(feature["px"].y) is float
    # the shape the experiment is actually written in
    yaml.safe_dump([f["px"].to_dict() for f in values["features"]])


def test_an_answer_made_by_dragging_can_be_saved(tab, experiment):
    """End to end over the thing that broke: decide with dragged points, then
    write the experiment."""
    from fibsem.applications.autolamella.proposals import Decision, DecisionOutcome
    from fibsem.ui.widgets.canvas.canvas_state import PointsSpec

    renderer = _renderer(tab)
    renderer._controller.set_overlay(
        BeamType.ION,
        PointsSpec(
            id=renderer.OVERLAY,
            points=[
                (np.float64(300.5), np.float64(280.25)),
                (np.float64(9.0), np.float64(8.0)),
            ],
            marker="+",
        ),
    )
    lamella = experiment.positions[0]
    lamella.task_state.status = AutoLamellaTaskStatus.Completed

    result = experiment.decide(
        lamella.id,
        TASK,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values=renderer.current_values(),
            task_id=RUN,
        ),
    )

    assert result.applied, result.reason
    experiment.save()
