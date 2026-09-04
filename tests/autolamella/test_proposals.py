"""Proposal records, and Experiment.decide as the one way a decision lands.

No Qt here: decide() runs as a plain call when there is no application, which
is what a script or a headless review gets. The main-thread marshalling is
covered in tests/ui/test_decide_main_thread.py.
"""

import os
from pathlib import Path

import pytest
import yaml
from psygnal.containers import EventedDict

from fibsem.applications.autolamella.proposals import (
    MILLING_SETUP,
    PROPOSAL_KINDS,
    Alternative,
    Decision,
    DecisionOutcome,
    Proposal,
    ProposalKind,
    compute_delta,
    register_proposal_kind,
)
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
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
        kind=MILLING_SETUP,
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
    assert again.kind == MILLING_SETUP
    assert again.values["poi"] == Point(1e-6, 2e-6)
    assert again.alternatives[0].values["poi"] == Point(5e-6, 0)
    assert again.alternatives[0].reason == "near bar"
    assert again.provenance == p.provenance
    assert again.current.outcome is DecisionOutcome.Confirmed
    assert again.current.values["poi"] == Point(1.5e-6, 2e-6)
    assert not again.pending


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


def test_kinds_declare_gating_in_code():
    assert PROPOSAL_KINDS[MILLING_SETUP].gating is True
    assert Proposal(kind=MILLING_SETUP).gating is True
    register_proposal_kind(
        ProposalKind(name="site_pick", gating=False, values=("sites",))
    )
    assert Proposal(kind="site_pick").gating is False
    assert Proposal(kind="never-registered").gating is True, (
        "unknown kinds are treated as gating"
    )


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


def test_reject_on_a_gating_kind_retires_the_item_with_the_reviewer_as_author(tmp_path):
    exp = _experiment(tmp_path)
    lamella = exp.positions[0]
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
    assert lamella.is_failure
    assert lamella.quality.verdict is Verdict.FAILED
    assert lamella.quality.author == "human:op"
    assert lamella.quality.reason == "no usable site"
    assert lamella.quality.at_task == SETUP
    assert lamella.quality.decision_id == (lamella.id, SETUP)
    assert lamella.poi == Point(0.0, 0.0), "nothing was written through"


def test_reject_on_a_generative_kind_creates_nothing_and_retires_nothing(tmp_path):
    exp = _experiment(tmp_path)
    register_proposal_kind(
        ProposalKind(name="site_pick", gating=False, values=("sites",))
    )
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
    assert result.applied is False and "No consumer" in result.reason
    assert lamella.proposals[SETUP].pending, "no half-applied decision was left"


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
    assert [d.author for d in proposal.decisions] == ["human:a", "human:b"]
    assert proposal.current is second
    assert lamella.poi == Point(3e-6, 0)
    assert proposal.delta()["poi"] == Point(3e-6, 0.0)


def test_author_names_the_declared_operator(tmp_path):
    exp = Experiment(path=tmp_path, name="e", metadata={"user": "Operator Name"})
    assert exp.author() == "human:Operator Name"
    anonymous = Experiment(path=tmp_path, name="f")
    assert anonymous.author().startswith("human:")


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
