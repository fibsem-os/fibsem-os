"""The watcher's lines for an instrument kind's own events.

The watcher (`fibsem.ai.agent.watch`) says what any instrument's agent hears:
proposals, requests, messages, the level, workflows and tasks. What only one
kind of instrument does, it leaves to that kind's lines here: the FIB-SEM's
milling (`fibsem`). A kind with none (a cryo-FM, or a playbook with no
instrument) hears none of them.

Each kind's ``LINES`` maps an event kind to a function from the event to its
wake-up, ``(kind, text)`` or ``(kind, text, facts)``, or None for no line.

Which lines wake the agent is the playbook's (``wakes``, by the names in
`GROUPS`); a line that doesn't goes in the agent's next turn. `ALWAYS` wake it
whatever the playbook says. Within that, the agent's ``wake_on`` policy may
hold routine lines for a digest: `NEEDS` says which are routine.
"""

from typing import FrozenSet, Iterable, Optional

from fibsem.ai.agent import instruments
from fibsem.ai.agent.wake_lines import fibsem
from fibsem.ai.agent.wake_lines.base import Line, Lines, item_of

__all__ = [
    "ALWAYS",
    "GROUPS",
    "HOLD",
    "HOLD_IF_OK",
    "Line",
    "Lines",
    "NEEDS",
    "WAKE",
    "WAKE_MODES",
    "for_kind",
    "item_of",
    "needs",
    "waking",
]

# Each kind's own lines, by its name; a kind not here has none.
KINDS = {"fibsem": fibsem.LINES}

# What a playbook's ``wakes`` names: groups of the watcher's line kinds
# (`fibsem.ai.agent.watch`, and each kind's own).
GROUPS = {
    # What waits on the agent: a run stopped on a proposal, a suggestion or a
    # request decided.
    "decisions": frozenset({"proposal", "proposal_released", "suggestion", "request"}),
    # How a run goes: workflows, tasks and the long commands the agent started.
    "run": frozenset(
        {
            "workflow_started",
            "workflow_ended",
            "workflow_failed",
            "quiet",
            "task_started",
            "task_completed",
            "task_failed",
            "task_cancelled",
            "task_skipped",
            "command_ended",
            "command_failed",
        }
    ),
    # The instrument's checks: autofocus, alignment and the FIB-SEM's mills.
    "checks": frozenset(
        {
            "autofocus",
            "alignment",
            fibsem.MILL_STAGE_STARTED,
            fibsem.MILL_ENDED,
            fibsem.MILL_FAILED,
            fibsem.MILL_READING,
            fibsem.STOP_RULE,
        }
    ),
    # The operator's notes on an item and edits to the plan.
    "notes": frozenset({"note", "plan_edited"}),
}
# Wake the agent whatever its playbook says: the operator reaching it, the
# watch's own state, and the wake-ups and deadlines the agent set itself.
ALWAYS = frozenset(
    {
        "message",
        "level",
        "level_lowered",
        "operator_stop",
        "gap",
        "restarted",
        "agent_off",
        "agent_back",
        "server_unavailable",
        "server_gone",
        "no_events",
        "unreachable",
        "alarm",
        "deadline",
    }
)


# What each line needs under the agent's ``wake_on`` policy (`wake_policy`):
# WAKE wakes it at once under every policy; HOLD is routine, and under
# ``digest`` waits for the next wake, in its digest; HOLD_IF_OK is routine only
# when its line says it went as expected (`Wake.facts` ``ok``), else WAKE. A
# kind not here wakes it. This is the one place the split is made.
WAKE, HOLD, HOLD_IF_OK = "wake", "hold", "hold if ok"
NEEDS = {
    # The operator reaching the agent, and the level it works at.
    "message": WAKE,
    # A /btw side question: answered beside the agent's turns and never
    # queued (`Harness._put`); a plain watch reader takes it as a message.
    "aside": WAKE,
    "level": WAKE,
    "level_lowered": WAKE,
    "operator_stop": WAKE,
    "plan_edited": WAKE,
    # The watch's own state: lost events, a restarted app, the level at Off.
    "gap": WAKE,
    "restarted": WAKE,
    "agent_off": WAKE,
    "agent_back": WAKE,
    "server_unavailable": WAKE,
    "server_gone": WAKE,
    "no_events": WAKE,
    "unreachable": WAKE,
    # The agent's own timer (and a turn cut at its step limit), a deadline,
    # and a running workflow that has gone quiet.
    "alarm": WAKE,
    "deadline": WAKE,
    "quiet": WAKE,
    # What waits on the agent, or answers it: a run stopped on a proposal, one
    # released by someone else, its suggestion or request decided or done.
    "proposal": WAKE,
    "proposal_released": WAKE,
    "suggestion": WAKE,
    "request": WAKE,
    # Failures, and runs that did not go as planned. A skipped task is said
    # with why (a missing prerequisite), and is worth a look at once.
    "command_failed": WAKE,
    "workflow_failed": WAKE,
    "task_failed": WAKE,
    "task_cancelled": WAKE,
    "task_skipped": WAKE,
    fibsem.MILL_FAILED: WAKE,
    fibsem.MILL_READING: WAKE,
    fibsem.STOP_RULE: WAKE,
    # A run starting and ending: rare, and the moments to look at the plan
    # and to sum up.
    "workflow_started": WAKE,
    "workflow_ended": WAKE,
    # Routine: the play-by-play of a run going as planned.
    "task_started": HOLD,
    "task_completed": HOLD,
    fibsem.MILL_STAGE_STARTED: HOLD,
    # The operator's note on a grid: not to the agent.
    "note": HOLD,
    # Routine when it went well: a check that passed, a mill whose every
    # stage finished, a long command someone else started that ended done.
    # The agent's own command ending is what it waits for, so it wakes it
    # (unless it already has the end: `wake_policy.WakeOn.saw_run_end`).
    "autofocus": HOLD_IF_OK,
    "alignment": HOLD_IF_OK,
    fibsem.MILL_ENDED: HOLD_IF_OK,
    "command_ended": HOLD_IF_OK,
}
# What a wake_on policy is: every line wakes the agent, or routine lines wait
# for a digest.
WAKE_MODES = ("all", "digest")


def needs(kind: Optional[str]) -> str:
    """What a line of *kind* needs (`NEEDS`); WAKE for one it doesn't name."""
    return NEEDS.get(kind or "", WAKE)


def waking(groups: Optional[Iterable[str]]) -> Optional[FrozenSet[str]]:
    """The line kinds a playbook's ``wakes`` lets wake the agent, besides
    `ALWAYS`; None, every line, for a playbook that names none."""
    if groups is None:
        return None
    return frozenset().union(*(GROUPS[name] for name in groups))


def for_kind(kind: Optional[str]) -> Lines:
    """The lines instrument *kind* adds to the watcher's (the default kind's
    for None); none for a kind without its own."""
    return dict(KINDS.get(kind or instruments.DEFAULT, {}))
