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
    TASK_RESULT,
    Decision,
    DecisionOutcome,
    Proposal,
)
from fibsem.applications.autolamella.structures import (  # noqa: E402
    Attention,
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaTaskState,
    AutoLamellaTaskStatus,
    AutoLamellaWorkflowConfig,
    Experiment,
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
                    name=SETUP, required=True, attention=Attention.review
                ),
                AutoLamellaTaskDescription(
                    name=FIDUCIAL, required=True, requires=[SETUP]
                ),
                AutoLamellaTaskDescription(
                    name=ROUGH, required=True, requires=[FIDUCIAL]
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
    lamella.task_history.append(
        AutoLamellaTaskState(name=SETUP, status=AutoLamellaTaskStatus.AwaitingDecision)
    )
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
    texts = tab.row_summaries()
    assert texts[0] == "Waiting · 1"
    assert experiment.positions[0].name in texts[1] and SETUP in texts[1]
    assert tab.list.itemWidget(tab.list.item(1)) is not None, "a real row widget"
    renderer = tab.stack.currentWidget()
    assert isinstance(renderer, R.MillingSetupReviewRenderer)
    assert renderer.task_chip.text() == SETUP
    assert renderer.line.text() == "Waiting for your decision · 2 tasks held"
    assert "Mill Fiducial, Rough Milling" in renderer.line.toolTip()
    assert "centre-of-image" in renderer.line.toolTip(), "the record is a hover away"
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


def test_reject_needs_a_reason_and_fails_the_task(tab, experiment, monkeypatch):
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
    assert lamella.task_history[-1].status is AutoLamellaTaskStatus.Failed
    assert "no usable site" in lamella.task_history[-1].status_message
    assert not lamella.is_failure, "a failed task is not a defective lamella"
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


def test_show_decided_lists_past_decisions_read_only(tab, experiment, qapp):
    lamella = experiment.positions[0]
    renderer = tab.stack.currentWidget()
    renderer._controller.set_points(BeamType.ION, "poi", [(256 + 20, 256)])
    tab.confirm_current()
    assert tab.pending_count == 0 and tab.row_summaries() == ["Nothing waiting"]

    tab.show_decided.setChecked(True)
    assert tab.pending_count == 0, "decided rows are not pending"
    texts = tab.row_summaries()
    assert texts[0] == "Decided · 1" and "confirmed" in texts[1]
    tab._select_entry(0)
    shown = tab.stack.currentWidget()
    assert isinstance(shown, R.MillingSetupReviewRenderer)
    assert not shown.btn_confirm.isEnabled() and not shown.btn_reject.isEnabled()
    assert shown.line.text().startswith("✓  Confirmed by you at"), "reads as you"
    assert "moved 2.0 µm" in shown.line.text(), "the delta is shown (20 px at 100 nm)"
    assert "+2.00" in shown.line.toolTip(), "the exact delta is a hover away"
    assert shown.position.text() == "decided 1 of 1 · read-only"
    assert "Unblocked:" in shown.line.toolTip()
    assert shown._controller.overlay_points(BeamType.ION, "confirmed"), (
        "the confirmed marker is drawn beside the proposed one"
    )
    before = lamella.proposals[SETUP].decisions[:]
    tab.confirm_current()
    assert lamella.proposals[SETUP].decisions == before, "read-only means read-only"

    tab.show_decided.setChecked(False)
    assert tab.row_summaries() == ["Nothing waiting"]
    assert tab.show_decided.isVisible() or tab.show_decided.parent() is not None, (
        "the toggle lives on the first header, even when nothing is listed"
    )


def _auto_confirm(proposal: Proposal, proposer: str = "centre-of-image") -> None:
    proposal.decisions.append(
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author=f"auto:{proposer}",
            values=dict(proposal.values),
        )
    )


def test_a_proposal_the_producer_applied_is_to_check_not_waiting(tab, experiment):
    """Advise mode: the producer confirmed its own proposal, the run went on,
    and a look is owed. It is neither pending (nothing waits on it) nor
    decided (nobody looked), so it has its own group and its own verb."""
    lamella = experiment.positions[0]
    _auto_confirm(lamella.proposals[SETUP])
    tab.refresh()
    assert tab.pending_count == 0, "nothing is stalled on this"
    assert tab.check_count == 1
    texts = tab.row_summaries()
    assert texts[0] == "To check · 1" and "to check" in texts[1]
    renderer = tab.stack.currentWidget()
    assert isinstance(renderer, R.MillingSetupReviewRenderer)
    assert renderer.btn_confirm.text() == "Acknowledge"
    assert renderer.btn_confirm.isEnabled() and renderer.btn_reject.isEnabled()
    assert renderer.position.text() == "to check 1 of 1"
    assert renderer.line.text().startswith("Applied automatically at")
    assert renderer.line.text().endswith("not checked yet")
    assert "went ahead" in renderer.line.toolTip()
    assert "auto ·" not in renderer.line.text()
    assert "centre-of-image" not in renderer.line.text(), "the record is a hover away"
    assert renderer._controller.overlay_points(BeamType.ION, "confirmed"), (
        "what was applied is on screen"
    )
    tab.show_decided.setChecked(True)
    assert "Decided" not in " ".join(tab.row_summaries()), "not decided either"


def test_acknowledging_records_a_look_and_writes_nothing(tab, experiment, qapp):
    lamella = experiment.positions[0]
    proposal = lamella.proposals[SETUP]
    _auto_confirm(proposal)
    tab.refresh()
    poi_before = lamella.poi
    rough_before = (
        lamella.task_config[ROUGH].milling["mill_rough"].stages[0].pattern.point
    )
    counts = []
    tab.counts_changed.connect(lambda w, c: counts.append((w, c)))

    tab.confirm_current()
    qapp.processEvents()

    assert not proposal.to_check
    ack = proposal.current
    assert ack.outcome is DecisionOutcome.Confirmed
    assert ack.author.startswith("human:") and ack.values == {}
    assert proposal.applied.author == "auto:centre-of-image", "the applied one stays"
    assert lamella.poi == poi_before
    assert (
        lamella.task_config[ROUGH].milling["mill_rough"].stages[0].pattern.point
        == rough_before
    )
    assert tab.check_count == 0 and counts[-1] == (0, 0)
    assert tab.stack.currentWidget() is tab.empty
    line = R.describe_decision(proposal, experiment)
    assert line.startswith("Applied by auto · centre-of-image at ")
    assert "checked by you at" in line
    tab.show_decided.setChecked(True)
    assert "Decided · 1" in tab.row_summaries()[0], "now it is decided"
    assert "checked by you" in tab.list.item(1).toolTip(), (
        "the row says a look happened"
    )


def test_after_an_acknowledgement_the_next_row_is_selected(tab, experiment, qapp):
    """A run of acknowledgements is a run of Returns."""
    lamella = experiment.positions[0]
    _auto_confirm(lamella.proposals[SETUP])
    lamella.proposals[FIDUCIAL] = Proposal(
        kind=MILLING_SETUP,
        values={"poi": Point(0.0, 0.0)},
        provenance={"proposer": "centre-of-image"},
    )
    _auto_confirm(lamella.proposals[FIDUCIAL])
    tab.refresh()
    assert tab.check_count == 2
    tab._select_entry(0)
    tab.confirm_current()
    qapp.processEvents()
    assert tab.check_count == 1
    assert tab._current_index() == 0, "the row that took its place"
    assert tab.stack.currentWidget().task_chip.text() == FIDUCIAL
    tab.confirm_current()
    assert tab.check_count == 0


def test_mark_all_as_checked_records_a_look_on_every_to_check_row(
    tab, experiment, qapp
):
    """A log that piled up while the tab was hidden clears in one click; what
    is waiting for a real decision is untouched."""
    lamella = experiment.positions[0]
    _auto_confirm(lamella.proposals[SETUP])
    lamella.proposals[FIDUCIAL] = Proposal(
        kind=MILLING_SETUP, values={"poi": Point(0.0, 0.0)}, provenance={}
    )
    _auto_confirm(lamella.proposals[FIDUCIAL])
    lamella.proposals[ROUGH] = Proposal(
        kind=MILLING_SETUP, values={"poi": Point(0.0, 0.0)}
    )
    tab.refresh()
    assert tab.check_count == 2 and tab.pending_count == 1
    first = tab.list.itemWidget(tab.list.item(0))
    assert isinstance(first, R._GroupHeaderRow) and first.button is None
    assert tab.show_decided.parent() is first, "the toggle sits on the first header"
    check_header = tab.list.itemWidget(tab.list.item(2))
    assert isinstance(check_header, R._GroupHeaderRow)
    assert check_header.button.text() == "Mark all as checked"

    check_header.button.click()
    qapp.processEvents()

    assert tab.check_count == 0 and tab.pending_count == 1, "waiting is untouched"
    for name in (SETUP, FIDUCIAL):
        d = lamella.proposals[name].current
        assert d.author.startswith("human:") and d.values == {} and d.via == "review"
    assert lamella.proposals[ROUGH].pending


def test_rejecting_a_checked_proposal_leaves_the_finished_task_alone(
    tab, experiment, monkeypatch
):
    """The producer confirmed its own result and the task completed; a later
    reject is a note on the record, not a change to the outcome."""
    lamella = experiment.positions[0]
    lamella.set_task_status(SETUP, AutoLamellaTaskStatus.Completed)
    _auto_confirm(lamella.proposals[SETUP])
    tab.refresh()
    from PyQt5.QtWidgets import QInputDialog

    monkeypatch.setattr(
        QInputDialog, "getText", staticmethod(lambda *a, **k: ("milled wrong", True))
    )
    tab.reject_current()
    assert lamella.task_history[-1].status is AutoLamellaTaskStatus.Completed
    assert not lamella.is_failure
    assert not lamella.proposals[SETUP].to_check


def test_go_to_lamella_hands_the_item_over(tab, experiment):
    heard = []
    tab.open_item_requested.connect(heard.append)
    tab.stack.currentWidget().btn_open.click()
    assert heard == [experiment.positions[0]]


def test_a_task_result_renders_both_images_and_confirms_with_no_values(
    tab, experiment, qapp
):
    """The generic kind: what a task did, for someone to look at. Nothing to
    drag, nothing written; the verbs are the same two."""
    lamella = experiment.positions[0]
    del lamella.proposals[SETUP]
    eb = os.path.join(str(lamella.path), "ref_rough_eb")
    _fib_image().save(eb)
    lamella.proposals[ROUGH] = Proposal(
        kind=TASK_RESULT,
        values={},
        provenance={
            "proposer": "task",
            "task_name": ROUGH,
            "status": "Completed",
            "started_at": 1000.0,
            "ended_at": 1130.0,
            "reference_image": "ref_setup_ib.tif",
            "reference_image_eb": "ref_rough_eb.tif",
            "failure": "",
        },
    )
    tab.refresh()
    assert tab.pending_count == 1
    renderer = tab.stack.currentWidget()
    assert isinstance(renderer, R.TaskResultReviewRenderer)
    assert renderer._image is not None and renderer._electron is not None
    assert renderer._controller.widget._sem_panel.isVisibleTo(renderer)
    assert renderer.line.text() == "Waiting for your decision · nothing is held"
    assert "Rough Milling completed in 2.2 min" in renderer.line.toolTip()
    assert renderer.btn_confirm.text() == "Confirm · looks right"
    assert not renderer._controller.overlay_points(BeamType.ION, "poi")

    tab.confirm_current()
    qapp.processEvents()
    proposal = lamella.proposals[ROUGH]
    assert proposal.current.outcome is DecisionOutcome.Confirmed
    assert proposal.current.values == {}
    assert R.describe_decision(proposal, experiment).startswith("Confirmed by you")

    # a failed run reads as such, and an auto-recorded one is to check
    lamella.proposals[FIDUCIAL] = Proposal(
        kind=TASK_RESULT,
        values={},
        provenance={"task_name": FIDUCIAL, "status": "Failed", "failure": "drift"},
    )
    _auto_confirm(lamella.proposals[FIDUCIAL], proposer="task")
    tab.refresh()
    assert tab.check_count == 1
    renderer = tab.stack.currentWidget()
    assert renderer.btn_confirm.text() == "Acknowledge"
    assert renderer.line.text().startswith("Recorded automatically at")
    assert "Mill Fiducial failed: drift" in renderer.line.toolTip()
    assert "No reference images" in renderer.line.toolTip()


def test_author_labels_and_row_details():
    exp = Experiment(path=Path("/tmp/claude-501/x"), name="e", metadata={"user": "Pat"})
    assert R.author_label("human:Pat", exp) == "you"
    assert R.author_label("human:Sam", exp) == "Sam"
    assert R.author_label("agent:claude", exp) == "agent · claude"
    assert R.author_label("auto:centre-of-image", exp) == "auto · centre-of-image"
    p = Proposal(kind=MILLING_SETUP, values={"poi": Point(0, 0)})
    p.decisions.append(
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:Pat",
            values={"poi": Point(0, 0)},
        )
    )
    assert R.delta_label(p) == "as proposed"
    p.decisions.append(
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:Pat",
            values={"poi": Point(3e-6, 4e-6)},
        )
    )
    assert R.delta_label(p) == "moved 5.0 µm"
    assert R.describe_decision(p, exp).startswith("Confirmed by you at ")


def test_the_row_chip_offers_review_only_with_the_flag(qapp, monkeypatch):
    from fibsem.applications.autolamella.ui import workflow_config_widget as W

    monkeypatch.setattr(W, "_agent_supervision_available", lambda: False)
    monkeypatch.setattr(W, "_review_available", lambda: True)
    task = AutoLamellaTaskDescription(
        name=SETUP, attention=Attention.supervised, required=True
    )
    row = WorkflowTaskRowWidget(task)
    assert row.btn_attention.text() == "Supervised"
    changed = []
    row.attention_changed.connect(changed.append)
    row.btn_attention.click()
    assert task.attention is Attention.review and changed == [task]
    assert row.btn_attention.text() == "Review"
    row.btn_attention.click()
    assert task.attention is Attention.automated
    assert row.btn_attention.text() == "Automated"

    monkeypatch.setattr(W, "_review_available", lambda: False)
    task.attention = Attention.review
    off = WorkflowTaskRowWidget(task)
    assert off.btn_attention.text() == "Automated", "runs as what it will run as"
    off.btn_attention.click()
    assert task.attention is Attention.supervised, "Review is not offered"
