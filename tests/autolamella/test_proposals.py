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
        provenance={"proposer": "centre", "reference_image": "ref_x.tif"},
    )


# ── records ──────────────────────────────────────────────────────────────────


def test_proposal_round_trips_through_yaml_with_points_intact():
    p = _proposal()
    p.decisions.append(
        Decision(
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
    d = Decision(outcome=DecisionOutcome.Confirmed, author="human:a", via="workflow")
    assert Decision.from_dict(d.to_dict()).via == "workflow"
    assert Decision.from_dict({"outcome": "Confirmed", "author": "human:a"}).via == ""


def test_a_superseded_proposal_stays_on_the_record_flat_and_oldest_first():
    from fibsem.applications.autolamella.proposals import supersede

    first = _proposal(Point(1e-6, 0))
    first.decisions.append(
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"poi": Point(2e-6, 0)},
        )
    )
    second = supersede(first, _proposal(Point(0, 0)))
    third = supersede(second, _proposal(Point(0, 1e-6)))
    assert third.pending
    assert [p.values["poi"] for p in third.superseded] == [
        Point(1e-6, 0),
        Point(0, 0),
    ], "oldest first, flat"
    assert third.superseded[0].current.values["poi"] == Point(2e-6, 0)
    assert second.superseded == [], "moved, not nested"
    again = Proposal.from_dict(yaml.safe_load(yaml.safe_dump(third.to_dict())))
    assert len(again.superseded) == 2
    assert again.superseded[0].delta()["poi"].x == pytest.approx(1e-6)


def test_delta_is_computed_from_proposed_and_confirmed_never_declared():
    p = _proposal(Point(1e-6, 2e-6))
    assert p.delta() == {}, "no decision, no delta"
    p.decisions.append(
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"poi": Point(1e-6, 2e-6)},
        )
    )
    assert p.delta()["poi"] == Point(0.0, 0.0), "confirmed unchanged: zero delta"
    p.decisions.append(
        Decision(
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
    d = Decision(outcome=DecisionOutcome.Confirmed, author="auto:current-poi")
    assert d.author == auto_author("current-poi"), "a string in is parsed"
    assert Decision.from_dict(d.to_dict()).author == d.author
    assert d.to_dict()["author"] == "auto:current-poi", "the file form is unchanged"


def test_to_check_clears_only_when_a_person_looked():
    p = Proposal(kind=POINT_OF_INTEREST, values={"poi": Point(0.0, 0.0)})
    assert not p.to_check, "nothing decided yet: it is pending, not to check"
    p.decisions.append(
        Decision(outcome=DecisionOutcome.Confirmed, author=auto_author("current-poi"))
    )
    assert p.to_check
    p.decisions.append(Decision(outcome=DecisionOutcome.Confirmed, author="agent:x"))
    assert p.to_check, "an agent looked; a person has not"
    p.decisions.append(Decision(outcome=DecisionOutcome.Confirmed, author="human:op"))
    assert not p.to_check


def test_kinds_declare_their_values_in_code():
    assert PROPOSAL_KINDS[POINT_OF_INTEREST].values == ("poi",)
    register_proposal_kind(ProposalKind(name="site_pick", values=("sites",)))
    assert PROPOSAL_KINDS["site_pick"].values == ("sites",)


def test_items_persist_their_proposals(tmp_path):
    exp = _experiment(tmp_path)
    lamella = exp.positions[0]
    lamella.proposals[SETUP] = _proposal()
    grid = exp.add_grid(GridRecord(name="Grid-01"))
    grid.proposals["overview"] = Proposal(kind="site_pick", values={"n": 3})
    exp.save()

    again = Experiment.load(Path(exp.path) / "experiment.yaml")
    assert again.positions[0].proposals[SETUP].values["poi"] == Point(1e-6, 2e-6)
    assert again.positions[0].proposals[SETUP].pending
    assert again.grids[0].proposals["overview"].values == {"n": 3}
    assert [(item.name, name) for item, name, _p in again.pending_proposals()] == [
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
    lamella.proposals[SETUP] = _proposal(Point(0.0, 0.0))
    rough_point_before = (
        lamella.task_config[ROUGH].milling["mill_rough"].stages[0].pattern.point
    )
    heard = []
    exp.decided.connect(lambda item_id, task: heard.append((item_id, task)))

    result = exp.decide(
        lamella.id,
        SETUP,
        Decision(
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
    assert lamella.proposals[SETUP].values["poi"] == Point(0.0, 0.0), (
        "confirming must not overwrite the proposal"
    )
    assert not lamella.proposals[SETUP].pending
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
    lamella.proposals[SETUP] = _proposal()
    assert lamella.is_awaiting_decision(SETUP) and not lamella.has_completed_task(SETUP)

    exp.decide(
        lamella.id,
        SETUP,
        Decision(outcome=DecisionOutcome.Confirmed, author="human:op", values={}),
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
    lamella.proposals[SETUP] = _proposal()

    result = exp.decide(
        lamella.id,
        SETUP,
        Decision(
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
    lamella.proposals[SETUP] = _proposal()

    exp.decide(
        lamella.id,
        SETUP,
        Decision(outcome=DecisionOutcome.Rejected, author="human:op", reason="meh"),
    )

    assert lamella.task_history[-1].status is AutoLamellaTaskStatus.Completed
    assert not lamella.proposals[SETUP].pending


def test_reject_on_a_grid_proposal_creates_nothing_and_retires_nothing(tmp_path):
    exp = _experiment(tmp_path)
    register_proposal_kind(ProposalKind(name="site_pick", values=("sites",)))
    grid = exp.add_grid(GridRecord(name="Grid-01"))
    grid.proposals["overview"] = Proposal(kind="site_pick", values={"sites": []})

    result = exp.decide(
        grid.id,
        "overview",
        Decision(outcome=DecisionOutcome.Rejected, author="human:op", reason="empty"),
    )

    assert result.applied is True
    assert grid.quality.verdict is Verdict.UNASSESSED
    assert not grid.proposals["overview"].pending
    assert len(exp.positions) == 1


def test_reject_needs_a_reason(tmp_path):
    exp = _experiment(tmp_path)
    lamella = exp.positions[0]
    lamella.proposals[SETUP] = _proposal()
    result = exp.decide(
        lamella.id, SETUP, Decision(outcome=DecisionOutcome.Rejected, author="human:op")
    )
    assert result.applied is False
    assert lamella.proposals[SETUP].pending
    assert not lamella.is_failure


def test_confirming_a_value_nothing_consumes_is_refused_before_anything_is_written(
    tmp_path,
):
    exp = _experiment(tmp_path)
    lamella = exp.positions[0]
    lamella.proposals[SETUP] = _proposal()
    result = exp.decide(
        lamella.id,
        SETUP,
        Decision(outcome=DecisionOutcome.Confirmed, author="human:op", values={"n": 3}),
    )
    assert result.applied is False and "does not carry ['n']" in result.reason
    assert lamella.proposals[SETUP].pending, "no half-applied decision was left"


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
        [deepcopy(d) for p in lamella.proposals.values() for d in p.decisions],
        [(t.name, t.status, t.status_message) for t in lamella.task_history],
    )


def _awaiting(exp, kind_proposal, task_name=SETUP):
    lamella = exp.positions[0]
    lamella.task_history.append(
        AutoLamellaTaskState(
            name=task_name, status=AutoLamellaTaskStatus.AwaitingDecision
        )
    )
    lamella.proposals[task_name] = kind_proposal
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
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"poi": value},
        ),
    )

    assert result.applied is False and "poi must" in result.reason
    assert _snapshot(lamella) == before
    assert lamella.proposals[SETUP].pending
    assert lamella.is_awaiting_decision(SETUP)
    assert heard == []


def test_a_value_the_proposals_kind_does_not_carry_is_refused(tmp_path):
    """A task_result proposal carries no values: a poi confirmed on it is
    refused even though a writer for poi exists."""
    exp = _experiment(tmp_path)
    lamella = _awaiting(exp, Proposal(kind=TASK_RESULT), task_name=ROUGH)
    before = _snapshot(lamella)

    result = exp.decide(
        lamella.id,
        ROUGH,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="agent:model",
            values={"poi": Point(5e-6, 5e-6)},
        ),
    )

    assert result.applied is False
    assert "task_result proposal does not carry ['poi']" in result.reason
    assert _snapshot(lamella) == before
    assert lamella.proposals[ROUGH].pending


def test_an_item_that_cannot_take_the_value_is_refused(tmp_path):
    exp = _experiment(tmp_path)
    grid = exp.add_grid(GridRecord(name="Grid-01"))
    grid.proposals["overview"] = _proposal()

    result = exp.decide(
        grid.id,
        "overview",
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"poi": Point(1e-6, 0.0)},
        ),
    )

    assert result.applied is False and "has no poi" in result.reason
    assert grid.proposals["overview"].pending
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
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"poi": Point(1e-6, 0.0)},
        ),
    )

    assert result.applied is False and "pattern has no point" in result.reason
    assert _snapshot(lamella) == before
    assert lamella.is_awaiting_decision(SETUP)


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
    confirm = Decision(outcome=DecisionOutcome.Confirmed, author="human:op", values={})
    assert exp.decide("no-such-id", SETUP, confirm).applied is False
    assert exp.decide(lamella.id, SETUP, confirm).applied is False


def test_decide_refuses_while_a_task_is_running_on_the_item(tmp_path):
    """A decision under a running consumer is a stop, not a decision."""
    exp = _experiment(tmp_path)
    lamella = exp.positions[0]
    lamella.proposals[SETUP] = _proposal()
    lamella.task_state.name = ROUGH
    lamella.task_state.status = AutoLamellaTaskStatus.InProgress

    result = exp.decide(
        lamella.id,
        SETUP,
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="human:op",
            values={"poi": Point(1e-6, 0)},
        ),
    )
    assert result.applied is False
    assert result.running is True
    assert lamella.proposals[SETUP].pending
    assert lamella.poi == Point(0.0, 0.0)


def test_decisions_append_and_the_latest_is_current(tmp_path):
    exp = _experiment(tmp_path)
    lamella = exp.positions[0]
    lamella.proposals[SETUP] = _proposal(Point(0.0, 0.0))
    first = Decision(
        outcome=DecisionOutcome.Confirmed,
        author="human:a",
        values={"poi": Point(1e-6, 0)},
    )
    second = Decision(
        outcome=DecisionOutcome.Confirmed,
        author="human:b",
        values={"poi": Point(3e-6, 0)},
    )
    exp.decide(lamella.id, SETUP, first)
    exp.decide(lamella.id, SETUP, second)
    proposal = lamella.proposals[SETUP]
    assert [str(d.author) for d in proposal.decisions] == ["human:a", "human:b"]
    assert proposal.current is second
    assert lamella.poi == Point(3e-6, 0)
    assert proposal.delta()["poi"] == Point(3e-6, 0.0)


def test_a_producer_applied_proposal_is_to_check_until_someone_looks(tmp_path):
    """Advise mode: every decision is the producer's own, so a look is owed.
    An acknowledgement carries no values (writes nothing) and clears it; the
    applied decision is still the one whose values were written."""
    exp = _experiment(tmp_path)
    lamella = exp.positions[0]
    lamella.proposals[SETUP] = _proposal(Point(0.0, 0.0))
    proposal = lamella.proposals[SETUP]
    assert not proposal.to_check and proposal.applied is None, "pending, not applied"
    auto = Decision(
        outcome=DecisionOutcome.Confirmed,
        author="auto:centre-of-image",
        values={"poi": Point(2e-6, 0)},
    )
    assert exp.decide(lamella.id, SETUP, auto).applied
    assert proposal.to_check and proposal.applied is auto
    assert lamella.poi == Point(2e-6, 0)
    assert exp.pending_proposals() == []
    assert [t for _i, t, _p in exp.proposals_to_check()] == [SETUP]

    ack = Decision(outcome=DecisionOutcome.Confirmed, author="human:a", values={})
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
    lamella.proposals[SETUP] = _proposal(Point(0.0, 0.0))
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
