"""Proposal records, and Experiment.decide as the one way a decision lands.

No Qt here: decide() runs as a plain call when there is no application, which
is what a script or a headless review gets. The main-thread marshalling is
covered in tests/ui/test_decide_main_thread.py.
"""

import os
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from psygnal.containers import EventedDict

from fibsem.applications.autolamella.proposals import (
    POINT_OF_INTEREST,
    PROPOSAL_KINDS,
    TASK_RESULT,
    Alternative,
    Author,
    AuthorKind,
    Decision,
    DecisionOutcome,
    Proposal,
    ProposalKind,
    auto_author,
    compute_delta,
    register_proposal_kind,
)
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    AutoLamellaTaskState,
    AutoLamellaTaskStatus,
    Experiment,
    GridRecord,
    Verdict,
)
from fibsem.applications.autolamella.workflows.tasks.rough import MillRoughTaskConfig
from fibsem.applications.autolamella.workflows.tasks.select_position import (
    SelectMillingPositionTaskConfig,
)
from fibsem.structures import FibsemStagePosition, MicroscopeState, Point

SETUP = "Setup Lamella Position"
# the run a proposal is from, and that a decision names
RUN = "run-1"
ROUGH = "Rough Milling"


def _experiment(tmp_path: Path) -> Experiment:
    exp = Experiment(path=tmp_path, name="test-exp")
    exp.task_protocol = AutoLamellaTaskProtocol()
    os.makedirs(exp.path, exist_ok=True)
    exp.add_new_lamella(
        MicroscopeState(stage_position=FibsemStagePosition()),
        EventedDict(
            {
                SETUP: SelectMillingPositionTaskConfig(task_name=SETUP),
                ROUGH: MillRoughTaskConfig(task_name=ROUGH),
            }
        ),
    )
    return exp


def _proposal(poi=Point(1e-6, 2e-6)) -> Proposal:
    return Proposal(
        kind=POINT_OF_INTEREST,
        values={"poi": poi},
        confidence=None,
        alternatives=[
            Alternative(values={"poi": Point(5e-6, 0)}, score=0.3, reason="near bar")
        ],
        provenance={
            "proposer": "centre",
            "reference_image": "ref_x.tif",
            "task_id": RUN,
        },
    )


# ── records ──────────────────────────────────────────────────────────────────


def test_proposal_round_trips_through_yaml_with_points_intact():
    p = _proposal()
    p.decisions.append(
        Decision(
            task_id=RUN,
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"poi": Point(1.5e-6, 2e-6)},
        )
    )
    again = Proposal.from_dict(yaml.safe_load(yaml.safe_dump(p.to_dict())))
    assert again.kind == POINT_OF_INTEREST
    assert again.values["poi"] == Point(1e-6, 2e-6)
    assert again.alternatives[0].values["poi"] == Point(5e-6, 0)
    assert again.alternatives[0].reason == "near bar"
    assert again.provenance == p.provenance
    assert again.current.outcome is DecisionOutcome.Confirmed
    assert again.current.values["poi"] == Point(1.5e-6, 2e-6)
    assert not again.pending


def test_a_decision_records_where_it_was_made():
    d = Decision(
        task_id=RUN, outcome=DecisionOutcome.Confirmed, author="human:a", via="workflow"
    )
    assert Decision.from_dict(d.to_dict()).via == "workflow"
    assert Decision.from_dict({"outcome": "Confirmed", "author": "human:a"}).via == ""


def test_a_tasks_proposals_are_a_list_oldest_first_and_the_last_is_current():
    from fibsem.applications.autolamella.proposals import (
        current_proposal,
        proposals_from_dict,
        proposals_to_dict,
        record,
    )

    first = _proposal(Point(1e-6, 0))
    first.decisions.append(
        Decision(
            task_id=RUN,
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"poi": Point(2e-6, 0)},
        )
    )
    proposals = []
    record(proposals, first)
    second = record(proposals, _proposal(Point(0, 0)))
    second.decisions.append(
        Decision(
            task_id=RUN,
            outcome=DecisionOutcome.Rejected,
            author="human:op",
            reason="no",
        )
    )
    third = record(proposals, _proposal(Point(0, 1e-6)))
    assert current_proposal(proposals) is third and third.pending
    assert [p.values["poi"] for p in proposals] == [
        Point(1e-6, 0),
        Point(0, 0),
        Point(0, 1e-6),
    ], "oldest first, flat"
    assert proposals[0].current.values["poi"] == Point(2e-6, 0)
    again = proposals_from_dict(
        yaml.safe_load(yaml.safe_dump(proposals_to_dict({"t": proposals})))
    )["t"]
    assert len(again) == 3
    assert again[0].delta()["poi"].x == pytest.approx(1e-6)
    assert [p.id for p in again] == [p.id for p in proposals]


def test_a_runs_questions_of_one_kind_all_stand_and_a_rerun_replaces_them():
    """Setup confirms the tilt and then the position: two ``state`` questions
    from one run, and both are current -- the earlier is a different
    question, not a replaced one. A re-run's question of the same kind does
    replace them. A proposal with no run stamped on it goes by the old
    last-of-kind rule."""
    from fibsem.applications.autolamella.proposals import STATE, current_proposals

    def question(run: str, stamped: bool = True) -> Proposal:
        return Proposal(
            kind=STATE,
            values={"stage_position": FibsemStagePosition()},
            provenance={"task_id": run} if stamped else {},
        )

    tilt, position, point = question("run-1"), question("run-1"), _proposal()
    point.provenance["task_id"] = "run-1"
    assert current_proposals([tilt, position, point]) == [tilt, position, point]

    again = question("run-2")
    assert current_proposals([tilt, position, point, again]) == [point, again], (
        "the re-run's question replaces both of the first run's"
    )

    old, newer = question("", stamped=False), question("", stamped=False)
    assert current_proposals([old, newer]) == [newer], "unstamped: last of kind"

    withdrawn = question("run-3")
    withdrawn.decisions.append(
        Decision(task_id="run-3", outcome=DecisionOutcome.Withdrawn, author="human:op")
    )
    asked_again = question("run-3")
    assert current_proposals([withdrawn, asked_again]) == [asked_again], (
        "withdrawn and asked again in the same run: replaced"
    )


def test_recording_replaces_only_an_unanswered_proposal_of_the_same_kind():
    """A re-run's first question lands after the point the last run left
    open. That point is another kind: it stays for ``expire_open`` to close,
    rather than being dropped as if it were the question asked again."""
    from fibsem.applications.autolamella.proposals import STATE, record

    open_point = _proposal()
    proposals = [open_point]
    question = record(
        proposals,
        Proposal(kind=STATE, values={"stage_position": FibsemStagePosition()}),
    )
    assert proposals == [open_point, question], "the open point is kept"

    fresh_point = record(proposals, _proposal(Point(0, 0)))
    assert proposals == [question, fresh_point], "the unanswered point is replaced"


def test_recording_over_an_unanswered_proposal_replaces_it():
    """Nobody answered it, so there is nothing to keep; two open proposals
    for one task would be two questions where one was asked."""
    from fibsem.applications.autolamella.proposals import record

    proposals = []
    record(proposals, _proposal(Point(1e-6, 0)))
    latest = record(proposals, _proposal(Point(0, 0)))
    assert proposals == [latest]


def test_delta_is_computed_from_proposed_and_confirmed_never_declared():
    p = _proposal(Point(1e-6, 2e-6))
    assert p.delta() == {}, "no decision, no delta"
    p.decisions.append(
        Decision(
            task_id=RUN,
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"poi": Point(1e-6, 2e-6)},
        )
    )
    assert p.delta()["poi"] == Point(0.0, 0.0), "confirmed unchanged: zero delta"
    p.decisions.append(
        Decision(
            task_id=RUN,
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"poi": Point(3e-6, 2e-6)},
        )
    )
    assert p.delta()["poi"].x == pytest.approx(2e-6), "latest decision is current"
    assert p.delta()["poi"].y == pytest.approx(0.0)
    assert p.values["poi"] == Point(1e-6, 2e-6), "the proposal itself is untouched"
    assert compute_delta(2.0, 3.5) == 1.5
    assert compute_delta("a", "b") is None


def test_an_author_is_a_kind_and_a_name_and_travels_as_kind_colon_name():
    a = Author.parse("agent:claude")
    assert a == Author(AuthorKind.agent, "claude") and str(a) == "agent:claude"
    assert a.label == "agent · claude"
    assert Author.parse("human:op").label == "op"
    assert Author.parse("human:").label == "someone"
    assert Author.parse("auto:").label == "auto · unknown"
    assert Author.parse("Pat") == Author(AuthorKind.human, "Pat"), (
        "no known prefix: a person whose name is the whole string"
    )
    assert Author.parse(a) is a
    d = Decision(
        task_id=RUN, outcome=DecisionOutcome.Confirmed, author="auto:current-poi"
    )
    assert d.author == auto_author("current-poi"), "a string in is parsed"
    assert Decision.from_dict(d.to_dict()).author == d.author
    assert d.to_dict()["author"] == "auto:current-poi", "the file form is unchanged"


def test_to_check_clears_only_when_a_person_looked():
    p = Proposal(
        kind=POINT_OF_INTEREST,
        values={"poi": Point(0.0, 0.0)},
        provenance={"task_id": RUN},
    )
    assert not p.to_check, "nothing decided yet: it is pending, not to check"
    p.decisions.append(
        Decision(
            task_id=RUN,
            outcome=DecisionOutcome.Confirmed,
            author=auto_author("current-poi"),
        )
    )
    assert p.to_check
    p.decisions.append(
        Decision(task_id=RUN, outcome=DecisionOutcome.Confirmed, author="agent:x")
    )
    assert p.to_check, "an agent looked; a person has not"
    p.decisions.append(
        Decision(task_id=RUN, outcome=DecisionOutcome.Confirmed, author="human:op")
    )
    assert not p.to_check


def test_kinds_declare_their_values_in_code():
    assert PROPOSAL_KINDS[POINT_OF_INTEREST].values == ("poi",)
    register_proposal_kind(ProposalKind(name="site_pick", values=("sites",)))
    assert PROPOSAL_KINDS["site_pick"].values == ("sites",)


def test_items_persist_their_proposals(tmp_path):
    exp = _experiment(tmp_path)
    lamella = exp.positions[0]
    lamella.proposals[SETUP] = [_proposal()]
    grid = exp.add_grid(GridRecord(name="Grid-01"))
    grid.proposals["overview"] = [
        Proposal(kind="site_pick", values={"n": 3}, provenance={"task_id": RUN})
    ]
    exp.save()

    again = Experiment.load(Path(exp.path) / "experiment.yaml")
    assert again.positions[0].proposal(SETUP).values["poi"] == Point(1e-6, 2e-6)
    assert again.positions[0].proposal(SETUP).pending
    assert again.grids[0].proposal("overview").values == {"n": 3}
    # Pending with nothing waiting on them: open, listed to check, not waiting.
    assert again.pending_proposals() == []
    assert [(item.name, name) for item, name, _p in again.proposals_to_check()] == [
        (again.positions[0].name, SETUP),
        ("Grid-01", "overview"),
    ]


def test_old_experiments_load_with_no_proposals(tmp_path):
    exp = _experiment(tmp_path)
    data = exp.to_dict()
    for p in data["positions"]:
        del p["proposals"]
    again = Experiment.from_dict(data)
    assert again.positions[0].proposals == {}


# ── decide ───────────────────────────────────────────────────────────────────


def test_confirm_writes_the_value_through_and_syncs_patterns(tmp_path):
    exp = _experiment(tmp_path)
    lamella = exp.positions[0]
    lamella.proposals[SETUP] = [_proposal(Point(0.0, 0.0))]
    rough_point_before = (
        lamella.task_config[ROUGH].milling["mill_rough"].stages[0].pattern.point
    )
    heard = []
    exp.decided.connect(lambda item_id, task: heard.append((item_id, task)))

    result = exp.decide(
        lamella.id,
        SETUP,
        Decision(
            task_id=RUN,
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"poi": Point(2e-6, -1e-6)},
        ),
    )

    assert result.applied is True
    assert result.delta["poi"] == Point(2e-6, -1e-6)
    assert lamella.poi == Point(2e-6, -1e-6)
    assert ROUGH in result.synced_tasks
    rough_point_after = (
        lamella.task_config[ROUGH].milling["mill_rough"].stages[0].pattern.point
    )
    assert rough_point_after.x == pytest.approx(rough_point_before.x + 2e-6)
    assert rough_point_after.y == pytest.approx(rough_point_before.y - 1e-6)
    assert lamella.proposal(SETUP).values["poi"] == Point(0.0, 0.0), (
        "confirming must not overwrite the proposal"
    )
    assert not lamella.proposal(SETUP).pending
    assert heard == [(lamella.id, SETUP)]
    assert not lamella.is_failure


def test_a_decision_finishes_a_task_that_was_awaiting_one(tmp_path):
    """The task ran and stopped short of finished. Confirm completes it, on the
    history entry and on the live task_state when that is the same run."""
    exp = _experiment(tmp_path)
    lamella = exp.positions[0]
    lamella.task_state = AutoLamellaTaskState(
        name=SETUP, status=AutoLamellaTaskStatus.AwaitingDecision
    )
    lamella.task_history.append(deepcopy(lamella.task_state))
    lamella.proposals[SETUP] = [_proposal()]
    assert lamella.is_awaiting_decision(SETUP) and not lamella.has_completed_task(SETUP)

    exp.decide(
        lamella.id,
        SETUP,
        Decision(
            task_id=RUN,
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"poi": _proposal().values["poi"]},
        ),
    )

    assert lamella.has_completed_task(SETUP)
    assert lamella.task_state.status is AutoLamellaTaskStatus.Completed
    assert not lamella.is_awaiting_decision(SETUP)


def test_reject_fails_the_waiting_task_and_leaves_the_lamella_alone(tmp_path):
    exp = _experiment(tmp_path)
    lamella = exp.positions[0]
    lamella.task_history.append(
        AutoLamellaTaskState(name=SETUP, status=AutoLamellaTaskStatus.AwaitingDecision)
    )
    lamella.proposals[SETUP] = [_proposal()]

    result = exp.decide(
        lamella.id,
        SETUP,
        Decision(
            task_id=RUN,
            outcome=DecisionOutcome.Rejected,
            author="human:op",
            reason="no usable site",
        ),
    )

    assert result.applied is True
    entry = lamella.task_history[-1]
    assert entry.status is AutoLamellaTaskStatus.Failed
    assert entry.status_message == "Rejected by op: no usable site"
    assert not lamella.is_failure, "a failed task is not a defective lamella"
    assert lamella.quality.verdict is Verdict.UNASSESSED
    assert lamella.poi == Point(0.0, 0.0), "nothing was written through"


def test_a_decision_on_a_finished_task_changes_only_the_record(tmp_path):
    """A result someone checks (or rejects) after the task completed on its own
    stays Completed: the decision is about the record, the outcome stands."""
    exp = _experiment(tmp_path)
    lamella = exp.positions[0]
    lamella.task_history.append(
        AutoLamellaTaskState(name=SETUP, status=AutoLamellaTaskStatus.Completed)
    )
    lamella.proposals[SETUP] = [_proposal()]

    exp.decide(
        lamella.id,
        SETUP,
        Decision(
            task_id=RUN,
            outcome=DecisionOutcome.Rejected,
            author="human:op",
            reason="meh",
        ),
    )

    assert lamella.task_history[-1].status is AutoLamellaTaskStatus.Completed
    assert not lamella.proposal(SETUP).pending


def test_reject_on_a_grid_proposal_creates_nothing_and_retires_nothing(tmp_path):
    exp = _experiment(tmp_path)
    register_proposal_kind(ProposalKind(name="site_pick", values=("sites",)))
    grid = exp.add_grid(GridRecord(name="Grid-01"))
    grid.proposals["overview"] = [
        Proposal(kind="site_pick", values={"sites": []}, provenance={"task_id": RUN})
    ]

    result = exp.decide(
        grid.id,
        "overview",
        Decision(
            task_id=RUN,
            outcome=DecisionOutcome.Rejected,
            author="human:op",
            reason="empty",
        ),
    )

    assert result.applied is True
    assert grid.quality.verdict is Verdict.UNASSESSED
    assert not grid.proposal("overview").pending
    assert len(exp.positions) == 1


def _grid_awaiting(exp: Experiment, task_name: str = "Overview") -> GridRecord:
    """A grid whose overview task ran, stopped short of finished, and proposed
    its result: the live task_state and its frozen history entry are the same
    run, as a grid task leaves them."""
    grid = exp.add_grid(GridRecord(name="Grid-01"))
    state = grid.task_state
    state.name = task_name
    state.task_id = "run-1"
    state.status = AutoLamellaTaskStatus.AwaitingDecision
    grid.task_history.append(deepcopy(state))
    grid.proposals[task_name] = [
        Proposal(kind=TASK_RESULT, provenance={"task_id": state.task_id})
    ]
    return grid


def test_a_decision_finishes_a_grid_task_that_was_awaiting_one(tmp_path):
    exp = _experiment(tmp_path)
    grid = _grid_awaiting(exp)
    assert grid.is_awaiting_decision("Overview")
    assert not grid.has_completed_task("Overview")

    result = exp.decide(
        grid.id,
        "Overview",
        Decision(
            task_id=grid.proposal("Overview").task_id,
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={},
        ),
    )

    assert result.applied is True
    assert grid.has_completed_task("Overview")
    assert grid.task_state.status is AutoLamellaTaskStatus.Completed
    assert not grid.is_awaiting_decision("Overview")
    assert not grid.proposal("Overview").pending


def test_reject_fails_the_waiting_grid_task_and_leaves_its_quality_alone(tmp_path):
    exp = _experiment(tmp_path)
    grid = _grid_awaiting(exp)

    result = exp.decide(
        grid.id,
        "Overview",
        Decision(
            task_id=grid.proposal("Overview").task_id,
            outcome=DecisionOutcome.Rejected,
            author="human:op",
            reason="all ice",
        ),
    )

    assert result.applied is True
    entry = grid.task_history[-1]
    assert entry.status is AutoLamellaTaskStatus.Failed
    assert entry.status_message == "Rejected by op: all ice"
    assert grid.task_state.status is AutoLamellaTaskStatus.Failed
    assert grid.quality.verdict is Verdict.UNASSESSED, "a person sets the verdict"


def test_confirming_values_on_a_grid_is_refused_before_anything_is_written(tmp_path):
    """No value is written through to a grid; the refusal leaves the task
    waiting and the proposal pending rather than half-applied."""
    exp = _experiment(tmp_path)
    grid = _grid_awaiting(exp)

    result = exp.decide(
        grid.id,
        "Overview",
        Decision(
            task_id=grid.proposal("Overview").task_id,
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"poi": Point(1e-6, 0)},
        ),
    )

    assert result.applied is False
    assert "does not carry ['poi']" in result.reason
    assert grid.proposal("Overview").pending
    assert grid.is_awaiting_decision("Overview")


def test_a_decision_on_a_finished_grid_task_changes_only_the_record(tmp_path):
    exp = _experiment(tmp_path)
    grid = _grid_awaiting(exp)
    grid.set_task_status("Overview", AutoLamellaTaskStatus.Completed)

    exp.decide(
        grid.id,
        "Overview",
        Decision(
            task_id=grid.proposal("Overview").task_id,
            outcome=DecisionOutcome.Rejected,
            author="human:op",
            reason="meh",
        ),
    )

    assert grid.task_history[-1].status is AutoLamellaTaskStatus.Completed
    assert not grid.proposal("Overview").pending


def test_a_grid_status_change_leaves_the_live_state_of_a_later_run_alone(tmp_path):
    """set_task_status moves the frozen entry always, and the live task_state
    only when it is the same run: a later task has since reused the object."""
    exp = _experiment(tmp_path)
    grid = _grid_awaiting(exp)
    grid.task_state.name = "Screening"
    grid.task_state.task_id = "run-2"
    grid.task_state.status = AutoLamellaTaskStatus.InProgress

    grid.set_task_status("Overview", AutoLamellaTaskStatus.Completed)

    assert grid.task_history[-1].status is AutoLamellaTaskStatus.Completed
    assert grid.task_state.status is AutoLamellaTaskStatus.InProgress


def test_a_repeated_grid_task_is_decided_on_its_latest_run_only(tmp_path):
    """The same task run twice leaves two history entries. Whether it awaits a
    decision, and what a decision changes, is the latest run's; the earlier
    run's recorded outcome stands."""
    exp = _experiment(tmp_path)
    grid = _grid_awaiting(exp)
    first = grid.task_history[-1]
    first.status = AutoLamellaTaskStatus.Failed
    first.status_message = "first run failed"
    grid.task_state.task_id = "run-2"
    grid.task_state.status = AutoLamellaTaskStatus.AwaitingDecision
    grid.task_history.append(deepcopy(grid.task_state))
    grid.proposals["Overview"] = [
        Proposal(kind=TASK_RESULT, provenance={"task_id": "run-2"})
    ]
    assert grid.is_awaiting_decision("Overview")

    exp.decide(
        grid.id,
        "Overview",
        Decision(
            task_id=grid.proposal("Overview").task_id,
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={},
        ),
    )

    assert grid.task_history[-1].status is AutoLamellaTaskStatus.Completed
    assert grid.task_state.status is AutoLamellaTaskStatus.Completed
    assert first.status is AutoLamellaTaskStatus.Failed
    assert first.status_message == "first run failed"


def test_an_earlier_run_awaiting_a_decision_does_not_make_a_later_one_wait(
    tmp_path,
):
    exp = _experiment(tmp_path)
    grid = _grid_awaiting(exp)
    grid.task_state.task_id = "run-2"
    grid.task_state.status = AutoLamellaTaskStatus.Completed
    grid.task_history.append(deepcopy(grid.task_state))

    assert not grid.is_awaiting_decision("Overview")
    exp.decide(
        grid.id,
        "Overview",
        Decision(
            task_id=grid.proposal("Overview").task_id,
            outcome=DecisionOutcome.Rejected,
            author="human:op",
            reason="no",
        ),
    )
    assert grid.task_history[-1].status is AutoLamellaTaskStatus.Completed
    assert grid.task_history[0].status is AutoLamellaTaskStatus.AwaitingDecision


def test_reject_needs_a_reason(tmp_path):
    exp = _experiment(tmp_path)
    lamella = exp.positions[0]
    lamella.proposals[SETUP] = [_proposal()]
    result = exp.decide(
        lamella.id,
        SETUP,
        Decision(task_id=RUN, outcome=DecisionOutcome.Rejected, author="human:op"),
    )
    assert result.applied is False
    assert lamella.proposal(SETUP).pending
    assert not lamella.is_failure


def test_confirming_a_value_nothing_consumes_is_refused_before_anything_is_written(
    tmp_path,
):
    exp = _experiment(tmp_path)
    lamella = exp.positions[0]
    lamella.proposals[SETUP] = [_proposal()]
    result = exp.decide(
        lamella.id,
        SETUP,
        Decision(
            task_id=RUN,
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"n": 3},
        ),
    )
    assert result.applied is False and "does not carry ['n']" in result.reason
    assert lamella.proposal(SETUP).pending, "no half-applied decision was left"


# ── a refused decision changes nothing (FIB-1003) ────────────────────────────


def _snapshot(lamella):
    """Everything a confirmed poi can touch, and the record it lands on."""
    return (
        deepcopy(lamella.poi),
        [
            deepcopy(stage.pattern.point)
            for config in lamella.task_config.values()
            for milling in (config.milling or {}).values()
            for stage in milling.stages
        ],
        [
            deepcopy(d)
            for ps in lamella.proposals.values()
            for p in ps
            for d in p.decisions
        ],
        [(t.name, t.status, t.status_message) for t in lamella.task_history],
    )


def _awaiting(exp, kind_proposal, task_name=SETUP):
    lamella = exp.positions[0]
    lamella.task_history.append(
        AutoLamellaTaskState(
            name=task_name, status=AutoLamellaTaskStatus.AwaitingDecision
        )
    )
    lamella.proposals[task_name] = [kind_proposal]
    return lamella


@pytest.mark.parametrize(
    "value",
    [
        {"x": 1},  # a dict that is not a point
        "centre",
        Point("a", 0.0),  # not numeric
        Point(float("nan"), 0.0),
        Point(True, 0.0),
    ],
    ids=["dict", "string", "text-axis", "nan", "bool-axis"],
)
def test_a_wrongly_typed_value_is_refused_and_nothing_moves(tmp_path, value):
    exp = _experiment(tmp_path)
    lamella = _awaiting(exp, _proposal())
    before = _snapshot(lamella)
    heard = []
    exp.decided.connect(lambda *a: heard.append(a))

    result = exp.decide(
        lamella.id,
        SETUP,
        Decision(
            task_id=RUN,
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"poi": value},
        ),
    )

    assert result.applied is False and "poi must" in result.reason
    assert _snapshot(lamella) == before
    assert lamella.proposal(SETUP).pending
    assert lamella.is_awaiting_decision(SETUP)
    assert heard == []


def test_a_value_the_proposals_kind_does_not_carry_is_refused(tmp_path):
    """A task_result proposal carries no values: a poi confirmed on it is
    refused even though a writer for poi exists."""
    exp = _experiment(tmp_path)
    lamella = _awaiting(
        exp, Proposal(kind=TASK_RESULT, provenance={"task_id": RUN}), task_name=ROUGH
    )
    before = _snapshot(lamella)

    result = exp.decide(
        lamella.id,
        ROUGH,
        Decision(
            task_id=RUN,
            outcome=DecisionOutcome.Confirmed,
            author="agent:model",
            values={"poi": Point(5e-6, 5e-6)},
        ),
    )

    assert result.applied is False
    assert "task_result proposal does not carry ['poi']" in result.reason
    assert _snapshot(lamella) == before
    assert lamella.proposal(ROUGH).pending


def test_an_item_that_cannot_take_the_value_is_refused(tmp_path):
    exp = _experiment(tmp_path)
    grid = exp.add_grid(GridRecord(name="Grid-01"))
    grid.proposals["overview"] = [_proposal()]

    result = exp.decide(
        grid.id,
        "overview",
        Decision(
            task_id=RUN,
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"poi": Point(1e-6, 0.0)},
        ),
    )

    assert result.applied is False and "has no poi" in result.reason
    assert grid.proposal("overview").pending
    assert not hasattr(grid, "poi")


def test_a_planning_error_is_refused_before_the_decision_is_appended(
    tmp_path, monkeypatch
):
    """Whatever goes wrong while working out the writes, nothing is written."""
    exp = _experiment(tmp_path)
    lamella = _awaiting(exp, _proposal())
    before = _snapshot(lamella)

    def broken_plan(point):
        raise RuntimeError("pattern has no point")

    monkeypatch.setattr(lamella, "poi_sync_plan", broken_plan)
    result = exp.decide(
        lamella.id,
        SETUP,
        Decision(
            task_id=RUN,
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"poi": Point(1e-6, 0.0)},
        ),
    )

    assert result.applied is False and "pattern has no point" in result.reason
    assert _snapshot(lamella) == before
    assert lamella.is_awaiting_decision(SETUP)


def _raise(*_args):
    raise RuntimeError("a subscriber failed")


def test_a_subscriber_that_raises_on_the_poi_write_undoes_the_decision(tmp_path):
    """Lamella is evented: assigning poi runs its subscribers. One that raises
    fails the decision after planning; nothing may be left half-applied."""
    exp = _experiment(tmp_path)
    lamella = _awaiting(exp, _proposal())
    before = _snapshot(lamella)
    heard = []
    exp.decided.connect(lambda *a: heard.append(a))
    lamella.events.poi.connect(_raise)

    result = exp.decide(
        lamella.id,
        SETUP,
        Decision(
            task_id=RUN,
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"poi": Point(2e-6, -1e-6)},
        ),
    )

    assert result.applied is False and "a subscriber failed" in result.reason
    assert _snapshot(lamella) == before, "poi, patterns, record and task put back"
    assert lamella.proposal(SETUP).pending
    assert lamella.is_awaiting_decision(SETUP)
    assert heard == []


def test_a_failure_moving_the_task_status_undoes_the_values_too(tmp_path):
    """The values land, then the status move raises: both are put back."""
    exp = _experiment(tmp_path)
    lamella = _awaiting(exp, _proposal())
    before = _snapshot(lamella)
    lamella.task_history[-1].events.status.connect(_raise)

    result = exp.decide(
        lamella.id,
        SETUP,
        Decision(
            task_id=RUN,
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"poi": Point(2e-6, -1e-6)},
        ),
    )

    assert result.applied is False
    assert _snapshot(lamella) == before
    assert lamella.proposal(SETUP).pending


def test_a_decided_subscriber_that_raises_does_not_unmake_the_decision(tmp_path):
    """Once committed the decision stands; a failing listener is logged."""
    exp = _experiment(tmp_path)
    lamella = _awaiting(exp, _proposal())
    exp.decided.connect(_raise)

    result = exp.decide(
        lamella.id,
        SETUP,
        Decision(
            task_id=RUN,
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"poi": Point(2e-6, -1e-6)},
        ),
    )

    assert result.applied is True
    assert lamella.poi == Point(2e-6, -1e-6)
    assert not lamella.proposal(SETUP).pending
    assert lamella.has_completed_task(SETUP)


def test_sync_tasks_to_poi_moves_the_patterns_its_plan_names(tmp_path):
    """The GUI path and the decision path share one plan."""
    exp = _experiment(tmp_path)
    lamella = exp.positions[0]
    target = Point(3e-6, 1e-6)
    names, moves = lamella.poi_sync_plan(target)
    assert names == [ROUGH] and moves
    before = _snapshot(lamella)

    assert lamella.sync_tasks_to_poi(target) == names
    assert [pattern.point for pattern, _ in moves] == [moved for _, moved in moves]
    assert _snapshot(lamella) != before


def test_decide_refuses_a_missing_item_or_proposal(tmp_path):
    exp = _experiment(tmp_path)
    lamella = exp.positions[0]
    confirm = Decision(
        task_id=RUN, outcome=DecisionOutcome.Confirmed, author="human:op", values={}
    )
    assert exp.decide("no-such-id", SETUP, confirm).applied is False
    assert exp.decide(lamella.id, SETUP, confirm).applied is False


def _running(lamella, task_name, task_id="run-2"):
    """The item has a run in progress: which task, and which run of it."""
    lamella.task_state.name = task_name
    lamella.task_state.task_id = task_id
    lamella.task_state.status = AutoLamellaTaskStatus.InProgress


def test_looking_at_an_earlier_result_is_not_refused_by_a_later_run(tmp_path):
    """FIB-1008: a grid's tasks run back to back, so something is nearly always
    running on the item whose earlier result is being looked at. An
    acknowledgement writes nothing, so it lands."""
    exp = _experiment(tmp_path)
    lamella = exp.positions[0]
    lamella.task_history.append(
        AutoLamellaTaskState(name=SETUP, status=AutoLamellaTaskStatus.Completed)
    )
    lamella.proposals[SETUP] = [_proposal()]
    lamella.proposal(SETUP).decisions.append(
        Decision(
            task_id=RUN,
            outcome=DecisionOutcome.Confirmed,
            author=auto_author("centre"),
            values={"poi": Point(1e-6, 2e-6)},
        )
    )
    _running(lamella, ROUGH)
    before = _snapshot(lamella)

    result = exp.decide(
        lamella.id,
        SETUP,
        Decision(task_id=RUN, outcome=DecisionOutcome.Confirmed, author="human:op"),
    )

    assert result.applied is True and result.running is False
    assert not lamella.proposal(SETUP).to_check, "the look is recorded"
    assert _snapshot(lamella)[:2] == before[:2], "poi and patterns untouched"


def test_rejecting_a_task_that_is_not_the_one_running_lands(tmp_path):
    exp = _experiment(tmp_path)
    lamella = _awaiting(exp, _proposal())
    _running(lamella, ROUGH)

    result = exp.decide(
        lamella.id,
        SETUP,
        Decision(
            task_id=RUN,
            outcome=DecisionOutcome.Rejected,
            author="human:op",
            reason="no site",
        ),
    )

    assert result.applied is True
    assert lamella.task_history[-1].status is AutoLamellaTaskStatus.Failed
    assert lamella.task_state.status is AutoLamellaTaskStatus.InProgress, (
        "the running task is left alone"
    )


def test_deciding_the_run_that_is_in_progress_is_still_refused(tmp_path):
    """The original case: the answer is Stop, not a decision."""
    exp = _experiment(tmp_path)
    lamella = _awaiting(exp, _proposal())
    _running(lamella, SETUP, task_id=RUN)  # the run the proposal is from

    result = exp.decide(
        lamella.id,
        SETUP,
        Decision(
            task_id=RUN,
            outcome=DecisionOutcome.Rejected,
            author="human:op",
            reason="stop",
        ),
    )

    assert result.applied is False and result.running is True
    assert f"is running {SETUP}" in result.reason
    assert lamella.proposal(SETUP).pending


def test_a_rerun_of_the_same_task_does_not_block_looking_at_the_earlier_run(
    tmp_path,
):
    """It is the run that matters, not the task name: a second run of the task
    is a different run, and the result being looked at is the first one's."""
    exp = _experiment(tmp_path)
    lamella = _awaiting(exp, _proposal())
    _running(lamella, SETUP, task_id="run-2")
    before = _snapshot(lamella)

    result = exp.decide(
        lamella.id,
        SETUP,
        Decision(
            task_id=RUN,
            outcome=DecisionOutcome.Rejected,
            author="human:op",
            reason="the first attempt missed",
        ),
    )

    assert result.applied is True and result.running is False
    assert _snapshot(lamella)[:2] == before[:2], "poi and patterns untouched"
    assert lamella.task_state.task_id == "run-2", "the running run is left alone"
    assert lamella.task_state.status is AutoLamellaTaskStatus.InProgress


def test_decide_refuses_a_value_written_under_any_running_task(tmp_path):
    """Values reach the item a running task is reading: that is a stop."""
    exp = _experiment(tmp_path)
    lamella = exp.positions[0]
    lamella.proposals[SETUP] = [_proposal()]
    _running(lamella, ROUGH)

    result = exp.decide(
        lamella.id,
        SETUP,
        Decision(
            task_id=RUN,
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"poi": Point(1e-6, 0)},
        ),
    )
    assert result.applied is False
    assert result.running is True
    assert lamella.proposal(SETUP).pending
    assert lamella.poi == Point(0.0, 0.0)


def test_decisions_append_and_the_latest_is_current(tmp_path):
    """A second decision is appended beside the first, never over it. On a
    decided proposal a confirm is a look: it carries no values, so the applied
    decision stays the first."""
    exp = _experiment(tmp_path)
    lamella = exp.positions[0]
    lamella.proposals[SETUP] = [_proposal(Point(0.0, 0.0))]
    first = Decision(
        task_id=RUN,
        outcome=DecisionOutcome.Confirmed,
        author="human:a",
        values={"poi": Point(1e-6, 0)},
    )
    second = Decision(task_id=RUN, outcome=DecisionOutcome.Confirmed, author="human:b")
    assert exp.decide(lamella.id, SETUP, first).applied
    assert exp.decide(lamella.id, SETUP, second).applied
    proposal = lamella.proposal(SETUP)
    assert [str(d.author) for d in proposal.decisions] == ["human:a", "human:b"]
    assert proposal.current is second
    assert proposal.applied is first
    assert lamella.poi == Point(1e-6, 0)


# ── a decision names the run it saw; a look cannot edit (FIB-1003) ───────────


def test_a_decision_that_names_no_run_is_refused(tmp_path):
    exp = _experiment(tmp_path)
    lamella = _awaiting(exp, _proposal())
    before = _snapshot(lamella)

    result = exp.decide(
        lamella.id,
        SETUP,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"poi": Point(1e-6, 0)},
        ),
    )

    assert result.applied is False and result.error_type == "missing_field"
    assert _snapshot(lamella) == before


def test_a_decision_on_a_run_the_task_has_since_replaced_is_refused(tmp_path):
    """The reviewer looked at run-1; the task re-ran and run-2 is pending now.
    The decision on run-1 does not land on run-2."""
    exp = _experiment(tmp_path)
    lamella = _awaiting(exp, _proposal())
    seen = lamella.proposal(SETUP)
    lamella.proposals[SETUP] = [
        Proposal(
            kind=POINT_OF_INTEREST,
            values={"poi": Point(7e-6, 0)},
            provenance={"task_id": "run-2"},
        )
    ]
    before = _snapshot(lamella)

    result = exp.decide(
        lamella.id,
        SETUP,
        Decision(
            task_id=seen.task_id,
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values=dict(seen.values),
        ),
    )

    assert result.applied is False and result.error_type == "stale_review"
    assert "re-run since you looked" in result.reason
    assert _snapshot(lamella) == before
    assert lamella.proposal(SETUP).pending


def test_a_proposal_recorded_before_runs_were_named_cannot_be_decided(tmp_path):
    exp = _experiment(tmp_path)
    lamella = _awaiting(exp, Proposal(kind=TASK_RESULT), task_name=ROUGH)

    result = exp.decide(
        lamella.id,
        ROUGH,
        Decision(task_id=RUN, outcome=DecisionOutcome.Rejected, author="a", reason="x"),
    )

    assert result.applied is False and result.error_type == "stale_review"
    assert "re-run it to decide it" in result.reason
    assert lamella.proposal(ROUGH).pending


def test_an_acknowledgement_with_values_is_refused_and_nothing_moves(tmp_path):
    """A producer-applied proposal is to check. Confirming it records a look;
    with values it would be an edit, and is refused."""
    exp = _experiment(tmp_path)
    lamella = exp.positions[0]
    lamella.task_history.append(
        AutoLamellaTaskState(name=SETUP, status=AutoLamellaTaskStatus.Completed)
    )
    lamella.proposals[SETUP] = [_proposal(Point(1e-6, 0))]
    assert exp.decide(
        lamella.id,
        SETUP,
        Decision(
            task_id=RUN,
            outcome=DecisionOutcome.Confirmed,
            author=auto_author("current-poi"),
            values={"poi": Point(1e-6, 0)},
        ),
    ).applied
    assert lamella.proposal(SETUP).to_check
    before = _snapshot(lamella)

    result = exp.decide(
        lamella.id,
        SETUP,
        Decision(
            task_id=RUN,
            outcome=DecisionOutcome.Confirmed,
            author="agent:model",
            values={"poi": Point(9e-6, 9e-6)},
        ),
    )

    assert result.applied is False and result.error_type == "invalid_value"
    assert "already decided" in result.reason
    assert _snapshot(lamella) == before
    assert lamella.proposal(SETUP).to_check, "the refused look is not a look"


def test_an_empty_confirm_on_a_pending_proposal_with_values_is_refused(tmp_path):
    exp = _experiment(tmp_path)
    lamella = _awaiting(exp, _proposal())
    before = _snapshot(lamella)

    result = exp.decide(
        lamella.id,
        SETUP,
        Decision(task_id=RUN, outcome=DecisionOutcome.Confirmed, author="human:op"),
    )

    assert result.applied is False and result.error_type == "invalid_value"
    assert "needs its values ['poi']" in result.reason
    assert _snapshot(lamella) == before
    assert lamella.is_awaiting_decision(SETUP)


def test_a_producer_applied_proposal_is_to_check_until_someone_looks(tmp_path):
    """Advise mode: every decision is the producer's own, so a look is owed.
    An acknowledgement carries no values (writes nothing) and clears it; the
    applied decision is still the one whose values were written."""
    exp = _experiment(tmp_path)
    lamella = exp.positions[0]
    lamella.proposals[SETUP] = [_proposal(Point(0.0, 0.0))]
    proposal = lamella.proposal(SETUP)
    assert not proposal.to_check and proposal.applied is None, "pending, not applied"
    auto = Decision(
        task_id=RUN,
        outcome=DecisionOutcome.Confirmed,
        author="auto:centre-of-image",
        values={"poi": Point(2e-6, 0)},
    )
    assert exp.decide(lamella.id, SETUP, auto).applied
    assert proposal.to_check and proposal.applied is auto
    assert lamella.poi == Point(2e-6, 0)
    assert exp.pending_proposals() == []
    assert [t for _i, t, _p in exp.proposals_to_check()] == [SETUP]

    ack = Decision(
        task_id=RUN, outcome=DecisionOutcome.Confirmed, author="human:a", values={}
    )
    result = exp.decide(lamella.id, SETUP, ack)
    assert result.applied and result.synced_tasks == [] and result.delta == {}
    assert not proposal.to_check and proposal.current is ack
    assert proposal.applied is auto, "the acknowledgement did not apply anything"
    assert lamella.poi == Point(2e-6, 0)
    assert exp.proposals_to_check() == []


def test_author_names_the_declared_operator(tmp_path):
    exp = Experiment(path=tmp_path, name="e", metadata={"user": "Operator Name"})
    assert str(exp.author()) == "human:Operator Name"
    anonymous = Experiment(path=tmp_path, name="f")
    assert anonymous.author().kind is AuthorKind.human


def test_decide_and_save_share_the_write_lock(tmp_path):
    """Neither can observe the other half-done: a save that starts while a
    decision is being applied waits for it, and vice versa."""
    import threading

    from fibsem.applications.autolamella import structures as S

    exp = _experiment(tmp_path)
    lamella = exp.positions[0]
    lamella.proposals[SETUP] = [_proposal(Point(0.0, 0.0))]
    order = []
    holding = threading.Event()
    release = threading.Event()

    def hold_the_lock():
        with S.EXPERIMENT_WRITE_LOCK:
            order.append("locked")
            holding.set()
            release.wait(5)
            order.append("unlocked")

    t = threading.Thread(target=hold_the_lock)
    t.start()
    holding.wait(5)
    saver = threading.Thread(target=lambda: (exp.save(), order.append("saved")))
    saver.start()
    # _decide, the unmarshalled inner: with a QApplication in the process (the
    # UI suites share it) decide() would park on a main thread this test
    # holds. The lock is the subject here, not the marshalling.
    decider = threading.Thread(
        target=lambda: (
            exp._decide(
                lamella.id,
                SETUP,
                Decision(
                    task_id=RUN,
                    outcome=DecisionOutcome.Confirmed,
                    author="human:op",
                    values={"poi": Point(1e-6, 0)},
                ),
            ),
            order.append("decided"),
        )
    )
    decider.start()
    import time

    time.sleep(0.2)
    assert order == ["locked"], "save and decide must both wait on the lock"
    release.set()
    for th in (t, saver, decider):
        th.join(5)
    assert order[0:2] == ["locked", "unlocked"]
    assert set(order[2:]) == {"saved", "decided"}
