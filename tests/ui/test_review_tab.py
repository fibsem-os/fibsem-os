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
    POINT_OF_INTEREST,
    TASK_RESULT,
    AuthorKind,
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
RUN = "run-1"  # the run every proposal here is from
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
                    name=SETUP, required=True, attention=Attention.review_later
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
    lamella.proposals[SETUP] = [
        Proposal(
            kind=POINT_OF_INTEREST,
            values={"poi": Point(0.0, 0.0)},
            provenance={
                "task_id": RUN,
                "proposer": "centre-of-image",
                "reference_image": ref + ".tif",
            },
        )
    ]
    return exp


@pytest.fixture
def warnings(monkeypatch) -> list:
    """A refused decision warns in a modal box, which would block offscreen:
    recorded here instead."""
    shown: list = []
    monkeypatch.setattr(
        R.QMessageBox,
        "warning",
        staticmethod(lambda _parent, _title, text: shown.append(text)),
    )
    return shown


@pytest.fixture
def tab(qapp, experiment, warnings) -> R.ReviewTabWidget:
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
    assert isinstance(renderer, R.PointOfInterestReviewRenderer)
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

    proposal = lamella.proposal(SETUP)
    assert not proposal.pending
    assert proposal.current.outcome is DecisionOutcome.Confirmed
    assert proposal.current.author.kind is AuthorKind.human
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
    assert lamella.proposal(SETUP).pending, "an empty reason is not a reject"
    assert not lamella.is_failure

    monkeypatch.setattr(
        QInputDialog, "getText", staticmethod(lambda *a, **k: ("no usable site", True))
    )
    tab.reject_current()
    assert not lamella.proposal(SETUP).pending
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
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="agent:test",
            values={"poi": Point(0.0, 0.0)},
            task_id=RUN,
        ),
    )
    assert tab.pending_count == 0


def test_an_unregistered_kind_still_gets_the_two_verbs(tab, experiment, qapp):
    lamella = experiment.positions[0]
    lamella.proposals["other"] = [
        Proposal(kind="site_pick_v9", values={}, provenance={"task_id": RUN})
    ]
    tab.refresh()
    assert tab.pending_count == 2
    tab._select_entry(1)
    renderer = tab.stack.currentWidget()
    assert isinstance(renderer, R._UnknownKindRenderer)
    assert "no review renderer" in renderer.label.text()
    tab.confirm_current()
    assert lamella.proposal("other").current.outcome is DecisionOutcome.Confirmed


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
    assert isinstance(shown, R.PointOfInterestReviewRenderer)
    assert not shown.btn_confirm.isEnabled() and not shown.btn_reject.isEnabled()
    assert shown.line.text().startswith("✓  Confirmed by you at"), "reads as you"
    assert "moved 2.0 µm" in shown.line.text(), "the delta is shown (20 px at 100 nm)"
    assert "+2.00" in shown.line.toolTip(), "the exact delta is a hover away"
    assert shown.position.text() == "decided 1 of 1 · read-only"
    assert "Unblocked:" in shown.line.toolTip()
    assert shown._controller.overlay_points(BeamType.ION, "confirmed"), (
        "the confirmed marker is drawn beside the proposed one"
    )
    before = lamella.proposal(SETUP).decisions[:]
    tab.confirm_current()
    assert lamella.proposal(SETUP).decisions == before, "read-only means read-only"

    tab.show_decided.setChecked(False)
    assert tab.row_summaries() == ["Nothing waiting"]
    assert tab.show_decided.parent() is not None, (
        "the chip lives on the filter row, so an empty list still offers it"
    )


def _auto_confirm(proposal: Proposal, proposer: str = "centre-of-image") -> None:
    proposal.decisions.append(
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author=f"auto:{proposer}",
            values=dict(proposal.values),
            task_id=proposal.task_id,
        )
    )


def test_a_proposal_the_producer_applied_is_to_check_not_waiting(tab, experiment):
    """Advise mode: the producer confirmed its own proposal, the run went on,
    and a look is owed. It is neither pending (nothing waits on it) nor
    decided (nobody looked), so it has its own group and its own verb."""
    lamella = experiment.positions[0]
    _auto_confirm(lamella.proposal(SETUP))
    tab.refresh()
    assert tab.pending_count == 0, "nothing is stalled on this"
    assert tab.check_count == 1
    texts = tab.row_summaries()
    assert texts[0] == "To check · 1" and "to check" in texts[1]
    renderer = tab.stack.currentWidget()
    assert isinstance(renderer, R.PointOfInterestReviewRenderer)
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
    proposal = lamella.proposal(SETUP)
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
    assert ack.author.kind is AuthorKind.human and ack.values == {}
    assert str(proposal.applied.author) == "auto:centre-of-image", (
        "the applied one stays"
    )
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
    _auto_confirm(lamella.proposal(SETUP))
    lamella.proposals[FIDUCIAL] = [
        Proposal(
            kind=POINT_OF_INTEREST,
            values={"poi": Point(0.0, 0.0)},
            provenance={"task_id": RUN, "proposer": "centre-of-image"},
        )
    ]
    _auto_confirm(lamella.proposal(FIDUCIAL))
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
    _auto_confirm(lamella.proposal(SETUP))
    lamella.proposals[FIDUCIAL] = [
        Proposal(
            kind=POINT_OF_INTEREST,
            values={"poi": Point(0.0, 0.0)},
            provenance={"task_id": RUN},
        )
    ]
    _auto_confirm(lamella.proposal(FIDUCIAL))
    lamella.proposals[ROUGH] = [
        Proposal(
            kind=POINT_OF_INTEREST,
            values={"poi": Point(0.0, 0.0)},
            provenance={"task_id": RUN},
        )
    ]
    tab.refresh()
    assert tab.check_count == 2 and tab.pending_count == 1
    first = tab.list.itemWidget(tab.list.item(0))
    assert isinstance(first, R._GroupHeaderRow) and first.button is None
    check_header = tab.list.itemWidget(tab.list.item(2))
    assert isinstance(check_header, R._GroupHeaderRow)
    assert check_header.button.text() == "Mark all as checked"

    check_header.button.click()
    qapp.processEvents()

    assert tab.check_count == 0 and tab.pending_count == 1, "waiting is untouched"
    for name in (SETUP, FIDUCIAL):
        d = lamella.proposal(name).current
        assert (
            d.author.kind is AuthorKind.human and d.values == {} and d.via == "review"
        )
    assert lamella.proposal(ROUGH).pending


def test_rejecting_a_checked_proposal_leaves_the_finished_task_alone(
    tab, experiment, monkeypatch
):
    """The producer confirmed its own result and the task completed; a later
    reject is a note on the record, not a change to the outcome."""
    lamella = experiment.positions[0]
    lamella.set_task_status(SETUP, AutoLamellaTaskStatus.Completed)
    _auto_confirm(lamella.proposal(SETUP))
    tab.refresh()
    from PyQt5.QtWidgets import QInputDialog

    monkeypatch.setattr(
        QInputDialog, "getText", staticmethod(lambda *a, **k: ("milled wrong", True))
    )
    tab.reject_current()
    assert lamella.task_history[-1].status is AutoLamellaTaskStatus.Completed
    assert not lamella.is_failure
    assert not lamella.proposal(SETUP).to_check


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
    lamella.proposals[ROUGH] = [
        Proposal(
            kind=TASK_RESULT,
            values={},
            provenance={
                "task_id": RUN,
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
    ]
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
    proposal = lamella.proposal(ROUGH)
    assert proposal.current.outcome is DecisionOutcome.Confirmed
    assert proposal.current.values == {}
    assert R.describe_decision(proposal, experiment).startswith("Confirmed by you")

    # a failed run reads as such, and an auto-recorded one is to check
    lamella.proposals[FIDUCIAL] = [
        Proposal(
            kind=TASK_RESULT,
            values={},
            provenance={
                "task_id": RUN,
                "task_name": FIDUCIAL,
                "status": "Failed",
                "failure": "drift",
            },
        )
    ]
    _auto_confirm(lamella.proposal(FIDUCIAL), proposer="task")
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
    p = Proposal(kind=POINT_OF_INTEREST, values={"poi": Point(0, 0)})
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
    assert task.attention is Attention.review_later and changed == [task]
    assert row.btn_attention.text() == "Review later"
    row.btn_attention.click()
    assert task.attention is Attention.automated
    assert row.btn_attention.text() == "Automated"

    monkeypatch.setattr(W, "_review_available", lambda: False)
    task.attention = Attention.review_later
    off = WorkflowTaskRowWidget(task)
    assert off.btn_attention.text() == "Automated", "runs as what it will run as"
    off.btn_attention.click()
    assert task.attention is Attention.supervised, "Review is not offered"


def test_a_decision_on_a_run_replaced_while_shown_is_refused(
    tab, experiment, warnings, qapp
):
    """The tab shows run-1; the task re-runs underneath it. Confirm names the
    run it showed, so it is refused with a warning and nothing is written."""
    lamella = experiment.positions[0]
    lamella.proposals[SETUP] = [
        Proposal(
            kind=POINT_OF_INTEREST,
            values={"poi": Point(7e-6, 0.0)},
            provenance={"task_id": "run-2"},
        )
    ]  # not refreshed: the tab still shows run-1

    tab.confirm_current()
    qapp.processEvents()

    assert warnings and "re-run since you looked" in warnings[-1]
    assert lamella.proposal(SETUP).pending
    assert lamella.poi == Point(0.0, 0.0)


def test_mark_all_as_checked_acknowledges_only_the_runs_it_listed(
    tab, experiment, qapp
):
    lamella = experiment.positions[0]
    _auto_confirm(lamella.proposal(SETUP))
    tab.refresh()
    assert tab.check_count == 1
    rerun = Proposal(
        kind=POINT_OF_INTEREST,
        values={"poi": Point(7e-6, 0.0)},
        provenance={"task_id": "run-2"},
    )
    _auto_confirm(rerun)
    lamella.proposals[SETUP] = [rerun]  # re-ran after the list was drawn

    tab.acknowledge_all()
    qapp.processEvents()

    assert rerun.to_check, "the run nobody was shown is still to check"
    assert all(d.author.kind is not AuthorKind.human for d in rerun.decisions)


# ---------------------------------------------------------------------------
# Grids (FIB-1002)
# ---------------------------------------------------------------------------


def _sem_image() -> FibsemImage:
    metadata = FibsemImageMetadata(
        image_settings=ImageSettings(beam_type=BeamType.ELECTRON, hfw=512 * PIXELSIZE),
        pixel_size=Point(PIXELSIZE, PIXELSIZE),
        microscope_state=MicroscopeState(stage_position=FibsemStagePosition()),
    )
    return FibsemImage(data=np.zeros((512, 512), dtype=np.uint8), metadata=metadata)


def _grid_waiting(experiment):
    """Grid-01's SEM overview under review, its stitched image on disk, and a
    FIB overview that requires it."""
    from fibsem.applications.autolamella.structures import GridRecord
    from fibsem.applications.autolamella.workflows.tasks.grid import (
        BeamOverviewGridTaskConfig,
    )

    protocol = experiment.grid_protocol
    protocol.add(
        BeamOverviewGridTaskConfig(
            task_name="SEM Overview", attention=Attention.review_later
        )
    )
    protocol.add(
        BeamOverviewGridTaskConfig(
            task_name="FIB Overview", orientation="FIB", requires=["SEM Overview"]
        )
    )
    grid = experiment.add_grid(GridRecord(name="Grid-01"))
    directory = experiment.grid_path(grid) / "SEM Overview"
    directory.mkdir(parents=True)
    _sem_image().save(str(directory / "overview"))
    grid.task_history.append(
        AutoLamellaTaskState(
            name="SEM Overview", status=AutoLamellaTaskStatus.AwaitingDecision
        )
    )
    grid.proposals["SEM Overview"] = [
        Proposal(
            kind=TASK_RESULT,
            provenance={"task_id": RUN, "reference_image": "SEM Overview/overview.tif"},
        )
    ]
    return grid


def test_a_grid_overview_is_shown_on_the_sem_canvas_with_go_to_grid(
    tab, experiment, qapp
):
    grid = _grid_waiting(experiment)
    tab.refresh()
    (index,) = [i for i, e in enumerate(tab._entries) if e[0] is grid]
    tab._select_entry(index)
    renderer = tab.stack.currentWidget()
    view = renderer._controller.widget

    assert renderer.btn_open.text() == "Go to grid"
    assert not view._sem_panel.isHidden(), "the SEM image is on the SEM canvas"
    assert view._fib_panel.isHidden(), "and not labelled FIB"
    assert renderer.line.text() == "Waiting for your decision · 1 task held"
    assert "FIB Overview" in renderer.line.toolTip()

    heard = []
    tab.open_item_requested.connect(heard.append)
    renderer.btn_open.click()
    assert heard == [grid]


@pytest.fixture
def microscope():
    """A connected instrument, for the one review that needs its geometry to
    turn a marked position into the poses a lamella is made from."""
    import fibsem.config as cfg
    from fibsem import utils

    microscope, _ = utils.setup_session(
        manufacturer="Demo",
        config_path=os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml"),
    )
    yield microscope
    microscope.disconnect()


def _positions_waiting(experiment, positions=(), name="Grid-02"):
    """A grid's overview under review for where the lamellae go, with its
    stitched image on disk."""
    from fibsem.applications.autolamella.proposals import OVERVIEW_POSITIONS
    from fibsem.applications.autolamella.structures import GridRecord

    grid = experiment.add_grid(GridRecord(name=name))
    directory = experiment.grid_path(grid) / "SEM Overview"
    directory.mkdir(parents=True, exist_ok=True)
    _sem_image().save(str(directory / "overview"))
    grid.task_history.append(
        AutoLamellaTaskState(
            name="SEM Overview", status=AutoLamellaTaskStatus.AwaitingDecision
        )
    )
    grid.proposals["SEM Overview"] = [
        Proposal(
            kind=OVERVIEW_POSITIONS,
            values={"positions": list(positions)},
            provenance={
                "task_id": f"{RUN}-{name}",
                "reference_image": "SEM Overview/overview.tif",
            },
        )
    ]
    return grid


def _positions_renderer(tab, grid):
    tab.refresh()
    (index,) = [i for i, e in enumerate(tab._entries) if e[0] is grid]
    tab._select_entry(index)
    return tab.stack.currentWidget()


class TestOverviewPositions:
    """The review that creates the lamellae: what it draws, what it will not
    let you edit, and what a confirm carries."""

    def test_the_grid_lamellae_are_drawn_locked_and_the_placed_ones_are_not(
        self, tab, experiment, qapp
    ):
        grid = _positions_waiting(experiment)
        lamella = experiment.positions[0]
        lamella.grid_id = grid.id
        renderer = _positions_renderer(tab, grid)

        canvas = renderer.canvas
        assert [p.name for p in canvas._positions] == [lamella.name], (
            "what the grid already has, drawn for context"
        )
        assert canvas._movable is False, "and not editable from a review"
        assert canvas.draft_positions == [], "nothing placed yet"

    def test_placing_needs_a_microscope_and_says_so(self, tab, experiment, qapp):
        """Reading the review needs nothing; placing a position needs the
        instrument's geometry, so without one nothing is placed and the line
        says why."""
        grid = _positions_waiting(experiment)
        renderer = _positions_renderer(tab, grid)
        assert renderer._microscope is None

        renderer._on_add_requested(FibsemStagePosition(x=1e-4, y=0, z=0, r=0, t=0))

        assert renderer.current_values() == {"positions": []}
        assert "connect a microscope" in renderer.line.text().lower()

    def test_a_placed_position_becomes_a_draft_and_the_confirm_counts_it(
        self, tab, experiment, qapp, microscope
    ):
        grid = _positions_waiting(experiment)
        renderer = _positions_renderer(tab, grid)
        tab.set_microscope(microscope)

        renderer._on_add_requested(FibsemStagePosition(x=1e-4, y=0, z=0, r=0, t=0))
        renderer._on_add_requested(FibsemStagePosition(x=-1e-4, y=0, z=0, r=0, t=0))

        assert len(renderer.canvas.draft_positions) == 2, "drawn as drafts"
        assert renderer.btn_confirm.text() == "Confirm · add 2 lamellae"
        values = renderer.current_values()["positions"]
        assert len(values) == 2
        assert all(v.milling is not None for v in values), (
            "the poses a lamella is made from, read here where there is an instrument"
        )

    def test_a_draft_can_be_taken_back_off(self, tab, experiment, qapp, microscope):
        grid = _positions_waiting(experiment)
        renderer = _positions_renderer(tab, grid)
        tab.set_microscope(microscope)
        renderer._on_add_requested(FibsemStagePosition(x=1e-4, y=0, z=0, r=0, t=0))
        renderer._on_add_requested(FibsemStagePosition(x=-1e-4, y=0, z=0, r=0, t=0))

        renderer._on_remove_requested(0)

        assert len(renderer.current_values()["positions"]) == 1
        assert renderer.btn_confirm.text() == "Confirm · add 1 lamella"

    def test_confirming_none_is_still_an_answer(self, tab, experiment, qapp):
        grid = _positions_waiting(experiment)
        renderer = _positions_renderer(tab, grid)
        assert renderer.btn_confirm.text() == "Confirm · add none"
        assert renderer.current_values() == {"positions": []}

    def test_a_to_check_row_places_nothing_and_offers_nothing(
        self, tab, experiment, qapp, microscope
    ):
        """Found in the app: on an automated overview the right-click still
        offered "Add Position Here" and the click did nothing. A decided
        proposal is looked at, not changed, so the canvas is told to offer
        neither -- an action that silently does nothing reads as broken."""
        from fibsem.applications.autolamella.proposals import Decision, DecisionOutcome

        grid = _positions_waiting(experiment)
        renderer = _positions_renderer(tab, grid)
        tab.set_microscope(microscope)
        renderer.set_to_check(
            Decision(
                outcome=DecisionOutcome.Confirmed,
                author="auto:place-by-hand",
                values={"positions": []},
            )
        )

        assert renderer.canvas._placing is False, "the menu offers nothing"
        renderer._on_add_requested(FibsemStagePosition(x=1e-4, y=0, z=0, r=0, t=0))
        assert renderer.current_values() == {"positions": []}

        # and rather than a dead end, it says where placing is done
        assert renderer.line.text() == (
            "Decided automatically · no lamellae were placed · "
            "add them on the Grids tab"
        )
        assert "Grids tab" in renderer.line.toolTip()
        assert renderer.btn_open.text() == "Go to grid", "the way there"

    def test_marks_survive_a_look_at_another_grid(
        self, tab, experiment, qapp, microscope
    ):
        """Found in the app: one renderer serves every row of this kind, so
        selecting another grid used to throw away what had been placed. The
        placements belong to the run, and come back with it."""
        first = _positions_waiting(experiment)
        second = _positions_waiting(experiment, name="Grid-03")
        renderer = _positions_renderer(tab, first)
        tab.set_microscope(microscope)
        renderer._on_add_requested(FibsemStagePosition(x=1e-4, y=0, z=0, r=0, t=0))
        renderer._on_add_requested(FibsemStagePosition(x=-1e-4, y=0, z=0, r=0, t=0))
        assert len(renderer.current_values()["positions"]) == 2

        _positions_renderer(tab, second)
        assert renderer.current_values() == {"positions": []}, "the other grid's"

        _positions_renderer(tab, first)
        assert len(renderer.current_values()["positions"]) == 2, "still there"

    def test_a_re_run_does_not_inherit_the_marks(
        self, tab, experiment, qapp, microscope
    ):
        """Kept per run, so a new overview -- a different image, possibly a
        different stage position -- opens clean rather than with marks made
        on the one it replaced."""
        grid = _positions_waiting(experiment)
        renderer = _positions_renderer(tab, grid)
        tab.set_microscope(microscope)
        renderer._on_add_requested(FibsemStagePosition(x=1e-4, y=0, z=0, r=0, t=0))
        assert len(renderer.current_values()["positions"]) == 1

        # the task runs again: a new proposal, naming a new run
        grid.proposal("SEM Overview").provenance["task_id"] = "another-run"
        _positions_renderer(tab, grid)

        assert renderer.current_values() == {"positions": []}

    def test_a_decided_review_places_nothing_more(
        self, tab, experiment, qapp, microscope
    ):
        """Read-only once decided: the lamellae exist, and changing them is a
        re-run of the overview rather than an edit here."""
        grid = _positions_waiting(experiment)
        renderer = _positions_renderer(tab, grid)
        tab.set_microscope(microscope)
        renderer.set_read_only(
            Decision(outcome=DecisionOutcome.Confirmed, author="human:op", values={})
        )

        renderer._on_add_requested(FibsemStagePosition(x=1e-4, y=0, z=0, r=0, t=0))

        assert renderer.current_values() == {"positions": []}


def test_a_lamella_row_still_shows_fib_and_go_to_lamella(tab, experiment):
    renderer = tab.stack.currentWidget()
    view = renderer._controller.widget
    assert renderer.btn_open.text() == "Go to lamella"
    assert not view._fib_panel.isHidden()


def test_waiting_on_and_gated_read_the_grid_protocol_for_a_grid(experiment):
    """A grid's held tasks and its review mode come from the grid protocol,
    never from the lamella workflow, whose task names mean something else."""
    grid = _grid_waiting(experiment)
    assert R.waiting_on(experiment, "SEM Overview", grid) == ["FIB Overview"]
    assert R.is_gated(experiment, "SEM Overview", grid) is True
    assert R.is_gated(experiment, "FIB Overview", grid) is False
    lamella = experiment.positions[0]
    assert R.waiting_on(experiment, SETUP, lamella) == [FIDUCIAL, ROUGH]
    assert R.waiting_on(experiment, SETUP, grid) == []
    assert R.is_gated(experiment, SETUP, lamella) is True


# ---------------------------------------------------------------------------
# Fluorescence results and unreadable images (FIB-1004)
# ---------------------------------------------------------------------------


def _fluorescence_overview(path: Path):
    """A genuine two-channel, three-plane fluorescence result, saved as the
    grid task saves its mosaic: an OME-TIFF."""
    from fibsem.fm.structures import (
        FluorescenceChannelMetadata,
        FluorescenceImage,
        FluorescenceImageMetadata,
    )

    channels = [
        FluorescenceChannelMetadata(
            name=name,
            color=color,
            excitation_wavelength=ex,
            emission_wavelength=em,
            power=1.0,
            exposure_time=0.1,
            gain=1.0,
            offset=0.0,
        )
        for name, color, ex, em in (
            ("GFP", "#00FF00", 488, 509),
            ("RFP", "#FF0000", 561, 584),
        )
    ]
    data = np.zeros((2, 3, 16, 16), dtype=np.uint16)
    data[0, 1, 4:8, 4:8] = 4000  # a bright square in one plane of GFP
    data[1, 2, 10:14, 10:14] = 3000
    image = FluorescenceImage(
        data=data,
        metadata=FluorescenceImageMetadata(
            acquisition_date="2026-09-17T00:00:00",
            pixel_size_x=1e-6,
            pixel_size_y=1e-6,
            resolution=(16, 16),
            channels=channels,
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(path))
    return path


def _fm_grid_proposal(experiment, filename="overview.ome.tiff", write=True):
    from fibsem.applications.autolamella.structures import GridRecord

    grid = experiment.add_grid(GridRecord(name="Grid-FM"))
    relative = f"FM Overview/{filename}"
    target = experiment.grid_path(grid) / relative
    if write:
        _fluorescence_overview(target)
    grid.task_history.append(
        AutoLamellaTaskState(
            name="FM Overview", status=AutoLamellaTaskStatus.AwaitingDecision
        )
    )
    grid.proposals["FM Overview"] = [
        Proposal(
            kind=TASK_RESULT,
            provenance={"task_id": RUN, "reference_image": relative},
        )
    ]
    return grid, target


def _select(tab, item):
    tab.refresh()
    (index,) = [i for i, e in enumerate(tab._entries) if e[0] is item]
    tab._select_entry(index)
    return tab.stack.currentWidget()


def test_a_fluorescence_overview_loads_as_a_fluorescence_image(experiment):
    from fibsem.fm.structures import FluorescenceImage

    grid, _ = _fm_grid_proposal(experiment)
    image = R._load_reference_image(experiment, grid, grid.proposal("FM Overview"))
    assert isinstance(image, FluorescenceImage)
    assert image.data.shape[-2:] == (16, 16)


def test_a_fluorescence_overview_is_shown_on_the_fm_page(tab, experiment):
    grid, _ = _fm_grid_proposal(experiment)
    renderer = _select(tab, grid)
    view = renderer._controller.widget
    assert view._stack.currentIndex() == 1, "the fluorescence page"
    assert not view.isHidden() and renderer.no_image.isHidden()
    assert renderer._image is None, "never a beam image, nothing drawn on it"
    assert "not found" not in renderer.line.toolTip()


def test_the_agent_preview_of_a_fluorescence_overview_is_its_composite(experiment):
    from fibsem.applications.autolamella.server.prompts import _preview_payload

    grid, _ = _fm_grid_proposal(experiment)
    image = R._load_reference_image(experiment, grid, grid.proposal("FM Overview"))
    composite = R.review_preview(image)
    assert composite.shape == (16, 16, 3)
    assert composite.reshape(-1, 3).max() > 0, "both channels' signal, projected"
    payload = _preview_payload(composite)
    assert payload is not None and payload["image_b64_jpeg"]


def test_an_unreadable_image_clears_the_last_one_and_says_so(tab, experiment, qapp):
    """From one result with a readable image to another, of the same kind,
    whose file is not an image: nothing of the first is left under the
    second's verbs."""
    first = _grid_waiting(experiment)  # an SEM overview on disk
    renderer = _select(tab, first)
    sem = renderer._controller.get_canvas(BeamType.ELECTRON)
    assert renderer._image is not None
    assert renderer._controller._states[sem].image is not None

    grid, target = _fm_grid_proposal(experiment, "overview.tif", write=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"not a tiff")
    again = _select(tab, grid)

    assert again is renderer, "the same renderer, so the same canvases"
    assert renderer._image is None and renderer._fluorescence is None
    assert renderer._controller._states[sem].image is None, "the SEM image is gone"
    assert renderer._controller.widget.isHidden()
    assert not renderer.no_image.isHidden()
    assert "could not be read: overview.tif" in renderer.no_image.text()

    _select(tab, first)
    assert renderer.no_image.isHidden() and not renderer._controller.widget.isHidden()


def test_a_result_with_no_image_recorded_says_so(tab, experiment):
    from fibsem.applications.autolamella.structures import GridRecord

    grid = experiment.add_grid(GridRecord(name="Grid-None"))
    grid.task_history.append(
        AutoLamellaTaskState(
            name="SEM Overview", status=AutoLamellaTaskStatus.AwaitingDecision
        )
    )
    grid.proposals["SEM Overview"] = [
        Proposal(kind=TASK_RESULT, provenance={"task_id": RUN, "reference_image": ""})
    ]
    renderer = _select(tab, grid)
    assert renderer.no_image.text() == "No image was recorded for this result."
