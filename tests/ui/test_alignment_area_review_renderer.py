"""Checking an alignment area in the Review tab (FIB-1053).

The renderer for the ``alignment_area`` kind: the proposed rectangle on the
last FIB image, in the same alignment overlay the Microscope tab edits with.
Confirm answers with the rectangle wherever it has been dragged to; once
decided it is shown read-only.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from psygnal.containers import EventedDict

pytest.importorskip("PyQt5")

import fibsem.applications.autolamella.ui.review_tab_widget as R
from fibsem.applications.autolamella.proposals import (
    ALIGNMENT_AREA,
    Decision,
    DecisionOutcome,
    Proposal,
)
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    AutoLamellaTaskStatus,
    Experiment,
)
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemImageMetadata,
    FibsemRectangle,
    FibsemStagePosition,
    ImageSettings,
    MicroscopeState,
    Point,
)

TASK = "Setup Lamella Position"
RUN = "run-1"
PROPOSED = FibsemRectangle(left=0.2, top=0.3, width=0.4, height=0.3)


def _fib_image() -> FibsemImage:
    metadata = FibsemImageMetadata(
        image_settings=ImageSettings(beam_type=BeamType.ION, hfw=512 * 25e-9),
        pixel_size=Point(25e-9, 25e-9),
        microscope_state=MicroscopeState(stage_position=FibsemStagePosition()),
    )
    return FibsemImage(data=np.zeros((512, 512), dtype=np.uint8), metadata=metadata)


@pytest.fixture
def experiment(tmp_path) -> Experiment:
    exp = Experiment(path=tmp_path, name="area-exp")
    exp.task_protocol = AutoLamellaTaskProtocol()
    os.makedirs(exp.path, exist_ok=True)
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    lamella = exp.positions[0]
    lamella.path.mkdir(parents=True, exist_ok=True)
    _fib_image().save(os.path.join(str(lamella.path), "ref_post_tilt_ib"))
    lamella.task_state.name = TASK
    lamella.task_state.task_id = RUN
    lamella.task_state.status = AutoLamellaTaskStatus.InProgress
    question = Proposal(
        kind=ALIGNMENT_AREA,
        values={"alignment_area": PROPOSED},
        provenance={
            "task_id": RUN,
            "proposer": TASK,
            "reference_image": "ref_post_tilt_ib.tif",
            "message": "Check the alignment area.",
        },
    )
    assert exp.ask_proposal(lamella.id, TASK, question)
    return exp


@pytest.fixture
def tab(qapp, experiment) -> R.ReviewTabWidget:
    widget = R.ReviewTabWidget()
    widget.set_experiment(experiment)
    return widget


def _renderer(tab):
    renderer = tab._renderer_for(ALIGNMENT_AREA)
    experiment = tab._experiment
    lamella = experiment.positions[0]
    renderer.set_proposal(experiment, lamella, TASK, lamella.proposal(TASK))
    return renderer


def test_the_kind_has_a_renderer_of_its_own(tab):
    assert isinstance(_renderer(tab), R.AlignmentAreaReviewRenderer)


def test_the_proposed_rectangle_is_on_the_ion_canvas_to_edit(tab):
    renderer = _renderer(tab)

    assert renderer._controller.alignment_area(BeamType.ION) == PROPOSED
    assert renderer.current_values() == {"alignment_area": PROPOSED}


def test_dragging_it_is_what_confirm_carries(tab):
    renderer = _renderer(tab)
    moved = FibsemRectangle(left=0.25, top=0.35, width=0.4, height=0.3)

    renderer._controller.set_alignment_edit(BeamType.ION, moved, editing=True)

    assert renderer.current_values() == {"alignment_area": moved}


def test_the_line_says_what_was_asked_and_where(tab):
    renderer = _renderer(tab)

    fact = renderer._fact()

    assert fact.startswith(f"{TASK} asked at ")
    assert "Left: 0.20, Top: 0.30, Width: 0.40, Height: 0.30" in fact
    assert fact.endswith("on ref_post_tilt_ib.tif.")


def test_a_decision_lands_and_is_shown_read_only(tab, experiment):
    renderer = _renderer(tab)
    lamella = experiment.positions[0]
    proposal = lamella.proposal(TASK)
    moved = FibsemRectangle(left=0.25, top=0.35, width=0.4, height=0.3)

    result = experiment.decide(
        lamella.id,
        TASK,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"alignment_area": moved},
            proposal_id=proposal.id,
        ),
    )
    assert result.applied, result.reason
    renderer.set_read_only(proposal.current)

    assert proposal.current.values["alignment_area"] == moved
    assert renderer._controller.alignment_area(BeamType.ION) == moved
    assert renderer.btn_confirm.isHidden() or not renderer.btn_confirm.isEnabled()
