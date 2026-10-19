"""A loop's passes as proposals: a mill's and a spot burn's.

Each pass is one ``milling`` or ``spot_burn`` proposal whose decision carries
an action (run, edit, skip, accept). No Qt here; the prompt bar and the Review
tab are covered in tests/ui/test_run_milling_question.py and
tests/ui/test_run_spot_burn_question.py.
"""

import os
from pathlib import Path

import pytest
from psygnal.containers import EventedDict

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.ui.loop_question import loop_prompt
from fibsem.applications.autolamella.workflows.tasks.rough import MillRoughTaskConfig
from fibsem.imaging.spot import SpotBurnSettings
from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.structures import FibsemStagePosition, MicroscopeState, Point
from fibsem.workflows.proposals import (
    ACTION_ACCEPT,
    ACTION_EDIT,
    ACTION_RUN,
    ACTION_SKIP,
    MILLING,
    PROPOSAL_KINDS,
    SPOT_BURN,
    STATE,
    TASK_RESULT,
    Decision,
    DecisionOutcome,
    Proposal,
    ValueRefused,
    patch_values,
    prepare_values,
)
from fibsem.workflows.status import TaskState, TaskStatus
from tests.proposal_records import set_proposals

ROUGH = "Rough Milling"
RUN = "run-1"
FIRST = [ACTION_RUN, ACTION_EDIT, ACTION_SKIP]
AFTER = [ACTION_RUN, ACTION_EDIT, ACTION_ACCEPT]


def _experiment(tmp_path: Path) -> Experiment:
    exp = Experiment(path=tmp_path, name="loop-exp")
    exp.task_protocol = AutoLamellaTaskProtocol()
    os.makedirs(exp.path, exist_ok=True)
    exp.add_new_lamella(
        MicroscopeState(stage_position=FibsemStagePosition()),
        EventedDict({ROUGH: MillRoughTaskConfig(task_name=ROUGH)}),
    )
    return exp


def _config() -> FibsemMillingTaskConfig:
    return FibsemMillingTaskConfig.from_stages(
        [FibsemMillingStage(name="Trench")], name="Rough Milling"
    )


def _pass(actions=FIRST, passes=0, **provenance) -> Proposal:
    return Proposal(
        kind=MILLING,
        values={"milling_config": _config()},
        provenance={
            "task_id": RUN,
            "actions": list(actions),
            "passes": passes,
            **provenance,
        },
    )


def _decision(outcome=DecisionOutcome.Confirmed, action="", **values) -> Decision:
    return Decision(
        task_id=RUN, outcome=outcome, actor="operator:op", values=values, action=action
    )


def test_a_pass_is_decided_as_one_of_the_actions_it_allows():
    kind = PROPOSAL_KINDS[MILLING]
    assert kind.allowed_actions(_pass()) == tuple(FIRST)
    assert kind.allowed_actions(_pass(AFTER, passes=1)) == tuple(AFTER)
    # one the kind does not know is dropped
    assert kind.allowed_actions(_pass([ACTION_ACCEPT, "dance"])) == (ACTION_ACCEPT,)
    # silence runs the first pass, then goes on
    assert kind.silent_action(_pass()) == ACTION_RUN
    assert kind.silent_action(_pass(AFTER, passes=1)) == ACTION_ACCEPT
    assert kind.silent_action(_pass([ACTION_ACCEPT])) == ACTION_ACCEPT
    # a yes/no kind has none
    state = Proposal(kind=STATE, provenance={"task_id": RUN})
    assert PROPOSAL_KINDS[STATE].allowed_actions(state) == ()
    assert PROPOSAL_KINDS[STATE].silent_action(state) == ""


def test_a_decision_s_action_is_saved_only_when_it_has_one():
    decided = _decision(action=ACTION_RUN)
    assert Decision.from_dict(decided.to_dict()).action == ACTION_RUN
    plain = _decision().to_dict()
    assert "action" not in plain
    assert Decision.from_dict(plain).action == ""


def test_decide_refuses_a_pass_without_an_allowed_action(tmp_path):
    exp = _experiment(tmp_path)
    lamella = exp.positions[0]
    asked = _pass()
    asked.asking = True  # the task waits on it
    set_proposals(lamella, ROUGH, [asked])

    missing = exp.decide(lamella.id, ROUGH, _decision())
    assert missing.applied is False and "decided as one of" in missing.reason
    # accept is for after a pass
    early = exp.decide(lamella.id, ROUGH, _decision(action=ACTION_ACCEPT))
    assert early.applied is False and f"not {ACTION_ACCEPT!r}" in early.reason
    assert lamella.proposal(ROUGH).pending, "nothing written"

    assert exp.decide(lamella.id, ROUGH, _decision(action=ACTION_SKIP)).applied
    assert lamella.proposal(ROUGH).current.action == ACTION_SKIP


def test_rejecting_a_pass_needs_no_action(tmp_path):
    exp = _experiment(tmp_path)
    lamella = exp.positions[0]
    set_proposals(lamella, ROUGH, [_pass()])
    rejected = Decision(
        task_id=RUN,
        outcome=DecisionOutcome.Rejected,
        actor="operator:op",
        reason="wrong lamella",
    )
    assert exp.decide(lamella.id, ROUGH, rejected).applied


def test_decide_refuses_an_action_on_a_yes_no_question(tmp_path):
    exp = _experiment(tmp_path)
    lamella = exp.positions[0]
    set_proposals(lamella, ROUGH, [Proposal(kind=STATE, provenance={"task_id": RUN})])
    result = exp.decide(lamella.id, ROUGH, _decision(action=ACTION_RUN))
    assert result.applied is False and "action" in result.reason


def test_a_pass_nobody_was_asked_is_recorded_with_what_silence_does(tmp_path):
    """Recorded unasked, the pass is decided Unreviewed with the action silence
    takes, and nothing is left waiting on it."""
    exp = _experiment(tmp_path)
    lamella = exp.positions[0]
    exp.record_unasked(lamella.id, ROUGH, _pass(), reason="nobody was asked")
    (recorded,) = lamella.proposals
    assert recorded.current.outcome is DecisionOutcome.Unreviewed
    assert recorded.current.action == ACTION_RUN, "silence runs it"
    assert exp.pending_proposals() == []


def test_a_milling_patch_sets_fields_on_a_copy():
    values = {"milling_config": _config()}
    patched = patch_values(MILLING, values, {"stages.Trench.pattern.depth": 1.5e-6})
    assert patched["milling_config"].stages[0].pattern.depth == pytest.approx(1.5e-6)
    assert values["milling_config"].stages[0].pattern.depth != pytest.approx(1.5e-6)
    with pytest.raises(ValueRefused):
        patch_values(MILLING, values, {"stages.Nowhere.pattern.depth": 1e-6})
    with pytest.raises(ValueRefused, match="not edited by field"):
        patch_values(STATE, {}, {"x": 1})


def test_a_spot_burn_is_refused_off_the_frame_or_out_of_range(tmp_path):
    exp = _experiment(tmp_path)
    lamella = exp.positions[0]

    def prepare(**settings):
        values = {"spot_burn": SpotBurnSettings(**settings)}
        return prepare_values(exp, lamella, SPOT_BURN, values)

    prepare(coordinates=[Point(0.5, 0.5)], milling_current=60e-12, exposure_time=5)
    for bad in (
        dict(coordinates=[Point(1.5, 0.5)], milling_current=60e-12, exposure_time=5),
        dict(coordinates=[Point(0.5, 0.5)], milling_current=0, exposure_time=5),
        dict(coordinates=[Point(0.5, 0.5)], milling_current=60e-12, exposure_time=0),
    ):
        with pytest.raises(ValueRefused):
            prepare(**bad)
    with pytest.raises(ValueRefused, match="spot-burn settings"):
        prepare_values(exp, lamella, SPOT_BURN, {"spot_burn": {"coordinates": []}})


def test_the_prompt_bar_names_what_each_button_does():
    first = loop_prompt(_pass(message="Check the trench."))
    assert (first.positive, first.yes) == ("Run Milling", ACTION_RUN)
    assert (first.negative, first.no) == ("Skip", ACTION_SKIP)
    assert first.message.startswith("Check the trench.")

    again = loop_prompt(_pass(AFTER, passes=1, last_error="stage moved"))
    assert (again.positive, again.negative) == ("Mill Again", "Accept")
    assert again.no == ACTION_ACCEPT
    assert again.message.startswith("Milling failed: stage moved")

    off = loop_prompt(_pass([ACTION_ACCEPT], passes=1))
    assert (off.positive, off.yes, off.negative) == ("Continue", ACTION_ACCEPT, None)

    burn = Proposal(
        kind=SPOT_BURN,
        values={"spot_burn": SpotBurnSettings()},
        provenance={"task_id": RUN, "actions": AFTER, "passes": 2},
    )
    assert loop_prompt(burn).positive == "Burn Again"
