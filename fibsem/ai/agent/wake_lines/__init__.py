"""The watcher's lines for an instrument kind's own events.

The watcher (`fibsem.ai.agent.watch`) says what any instrument's agent hears:
proposals, requests, messages, the level, workflows and tasks. What only one
kind of instrument does, it leaves to that kind's lines here: the FIB-SEM's
milling (`fibsem`). A kind with none (a cryo-FM, or a playbook with no
instrument) hears none of them.

Each kind's ``LINES`` maps an event kind to a function from the event to its
wake-up, ``(kind, text)``, or None for no line.

Which lines wake the agent is the playbook's (``wakes``, by the names in
`GROUPS`); a line that doesn't goes in the agent's next turn. `ALWAYS` wake it
whatever the playbook says.
"""

from typing import FrozenSet, Iterable, Optional

from fibsem.ai.agent import instruments
from fibsem.ai.agent.wake_lines import fibsem
from fibsem.ai.agent.wake_lines.base import Line, Lines, item_of

__all__ = ["ALWAYS", "GROUPS", "Line", "Lines", "for_kind", "item_of", "waking"]

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
