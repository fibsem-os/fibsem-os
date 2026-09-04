"""The Review tab: the inbox is derived from the experiment, the renderer is
chosen by proposal kind, and the two verbs go through Experiment.decide.

No microscope: a proposal, its reference image on disk, and an experiment
are all that the tab needs -- which is the point of it being usable days
later on a machine with no instrument.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from psygnal.containers import EventedDict

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from fibsem.applications.autolamella.proposals import (  # noqa: E402
    MILLING_SETUP,
    Decision,
    DecisionOutcome,
    Proposal,
)
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaWorkflowConfig,
    Experiment,
    Verdict,
)
from fibsem.applications.autolamella.ui import review_tab_widget as R  # noqa: E402
from fibsem.applications.autolamella.ui.workflow_config_widget import (  # noqa: E402
    WorkflowTaskRowWidget,
)
from fibsem.applications.autolamella.workflows.tasks.rough import (  # noqa: E402
    MillRoughTaskConfig,
)
from fibsem.structures import (  # noqa: E402
    BeamType,
    FibsemImage,
    FibsemImageMetadata,
    FibsemStagePosition,
    ImageSettings,
    MicroscopeState,
    Point,
)

SETUP = "Setup Lamella Position"
FIDUCIAL = "Mill Fiducial"
ROUGH = "Rough Milling"
PIXELSIZE = 1e-7  # 100 nm/px on a 512 x 512 frame


def _fib_image() -> FibsemImage:
    metadata = FibsemImageMetadata(
        image_settings=ImageSettings(beam_type=BeamType.ION, hfw=512 * PIXELSIZE),
        pixel_size=Point(PIXELSIZE, PIXELSIZE),
        microscope_state=MicroscopeState(stage_position=FibsemStagePosition()),
    )
    return FibsemImage(data=np.zeros((512, 512), dtype=np.uint8), metadata=metadata)


@pytest.fixture
def experiment(tmp_path) -> Experiment:
    exp = Experiment(path=tmp_path, name="review-exp")
    exp.task_protocol = AutoLamellaTaskProtocol(
        workflow_config=AutoLamellaWorkflowConfig(
            tasks=[
                AutoLamellaTaskDescription(
                    name=SETUP, supervise=False, required=True, review=True
                ),
                AutoLamellaTaskDescription(
                    name=FIDUCIAL, supervise=False, required=True, requires=[SETUP]
                ),
                AutoLamellaTaskDescription(
                    name=ROUGH, supervise=False, required=True, requires=[FIDUCIAL]
                ),
            ]
        )
    )
    os.makedirs(exp.path, exist_ok=True)
    exp.add_new_lamella(
        MicroscopeState(stage_position=FibsemStagePosition()),
        EventedDict({ROUGH: MillRoughTaskConfig(task_name=ROUGH)}),
    )
    lamella = exp.positions[0]
    lamella.path.mkdir(parents=True, exist_ok=True)
    ref = os.path.join(str(lamella.path), "ref_setup_ib")
    _fib_image().save(ref)
    lamella.proposals[SETUP] = Proposal(
        kind=MILLING_SETUP,
        values={"poi": Point(0.0, 0.0)},
        provenance={"proposer": "centre-of-image", "reference_image": ref + ".tif"},
    )
    return exp


@pytest.fixture
def tab(qapp, experiment) -> R.ReviewTabWidget:
    widget = R.ReviewTabWidget()
    widget.set_experiment(experiment)
    return widget


def test_the_inbox_is_derived_from_the_experiment(tab, experiment):
    assert tab.pending_count == 1
    texts = [tab.list.item(i).text() for i in range(tab.list.count())]
    assert texts[0].startswith("MILLING POSITIONS")
    assert experiment.positions[0].name in texts[1] and SETUP in texts[1]
    renderer = tab.stack.currentWidget()
    assert isinstance(renderer, R.MillingSetupReviewRenderer)
    assert renderer.task_chip.text() == SETUP
    assert "Waiting on this: Mill Fiducial, Rough Milling" in renderer.waiting.text()
    assert renderer._image is not None, "the reference image from provenance"


def test_confirm_submits_the_marker_and_the_delta_is_computed(tab, experiment, qapp):
    lamella = experiment.positions[0]
    rough_before = (
        lamella.task_config[ROUGH].milling["mill_rough"].stages[0].pattern.point
    )
    heard = []
    tab.decided.connect(lambda item_id, task: heard.append((item_id, task)))
    renderer = tab.stack.currentWidget()
    # The reviewer drags the marker 20 px right, 10 px up of centre.
    renderer._controller.set_points(BeamType.ION, "poi", [(256 + 20, 256 - 10)])

    tab.confirm_current()
    qapp.processEvents()

    proposal = lamella.proposals[SETUP]
    assert not proposal.pending
    assert proposal.current.outcome is DecisionOutcome.Confirmed
    assert proposal.current.author.startswith("human:")
    assert lamella.poi.x == pytest.approx(20 * PIXELSIZE)
    assert lamella.poi.y == pytest.approx(10 * PIXELSIZE)
    assert proposal.delta()["poi"].x == pytest.approx(20 * PIXELSIZE)
    assert proposal.values["poi"] == Point(0.0, 0.0), "the proposal is untouched"
    rough_after = (
        lamella.task_config[ROUGH].milling["mill_rough"].stages[0].pattern.point
    )
    assert rough_after.x == pytest.approx(rough_before.x + 20 * PIXELSIZE)
    assert heard == [(lamella.id, SETUP)]
    assert tab.pending_count == 0
    assert tab.stack.currentWidget() is tab.empty
    assert (Path(experiment.path) / "experiment.yaml").exists(), "saved"


def test_reject_needs_a_reason_and_retires_the_lamella(tab, experiment, monkeypatch):
    lamella = experiment.positions[0]
    from PyQt5.QtWidgets import QInputDialog

    monkeypatch.setattr(
        QInputDialog, "getText", staticmethod(lambda *a, **k: ("", True))
    )
    tab.reject_current()
    assert lamella.proposals[SETUP].pending, "an empty reason is not a reject"
    assert not lamella.is_failure

    monkeypatch.setattr(
        QInputDialog, "getText", staticmethod(lambda *a, **k: ("no usable site", True))
    )
    tab.reject_current()
    assert not lamella.proposals[SETUP].pending
    assert lamella.is_failure
    assert lamella.quality.verdict is Verdict.FAILED
    assert lamella.quality.reason == "no usable site"
    assert lamella.quality.author.startswith("human:")
    assert tab.pending_count == 0


def test_a_decision_made_elsewhere_refreshes_the_inbox(tab, experiment):
    """The agent server decides through the same function; the tab follows."""
    lamella = experiment.positions[0]
    assert tab.pending_count == 1
    experiment.decide(
        lamella.id,
        SETUP,
        Decision(outcome=DecisionOutcome.Confirmed, author="agent:test", values={}),
    )
    assert tab.pending_count == 0


def test_an_unregistered_kind_still_gets_the_two_verbs(tab, experiment, qapp):
    lamella = experiment.positions[0]
    lamella.proposals["other"] = Proposal(kind="site_pick_v9", values={})
    tab.refresh()
    assert tab.pending_count == 2
    tab._select_entry(1)
    renderer = tab.stack.currentWidget()
    assert isinstance(renderer, R._UnknownKindRenderer)
    assert "no review renderer" in renderer.label.text()
    tab.confirm_current()
    assert lamella.proposals["other"].current.outcome is DecisionOutcome.Confirmed


def test_the_row_toggle_sets_review_and_follows_the_flag(qapp, monkeypatch):
    from fibsem.applications.autolamella.ui import workflow_config_widget as W

    monkeypatch.setattr(W, "_review_available", lambda: True)
    task = AutoLamellaTaskDescription(name=SETUP, supervise=True, required=True)
    row = WorkflowTaskRowWidget(task)
    assert row.btn_review.isVisibleTo(row)
    changed = []
    row.review_changed.connect(changed.append)
    row.btn_review.click()
    assert task.review is True and changed == [task]
    assert "Review" in row.btn_review.toolTip()
    row.btn_review.click()
    assert task.review is False

    monkeypatch.setattr(W, "_review_available", lambda: False)
    hidden = WorkflowTaskRowWidget(task)
    assert not hidden.btn_review.isVisibleTo(hidden)
