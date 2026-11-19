"""Typed payloads for the event kinds an agent or operator is told about.

A record's payload is a plain dict on the wire and in events.jsonl. For the
kinds below, this module names its fields once: a reader takes
``read_payload(record)`` and gets attributes, not ``data.get("...")`` strings
that drift from what the writer wrote.

Reading is forgiving: a field missing from an older file is None, and one this
module does not know is left out. Writing is checked in the tests: a payload
of a *closed* kind with a field not named here fails them, so a new field is
added here first.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Type, TypeVar

from fibsem.workflows.events import kinds as k

P = TypeVar("P", bound="Payload")


@dataclass(frozen=True)
class Payload:
    # A closed payload has only the fields named here; an open one carries
    # others too (a run's labels, a note's extras), which readers ignore.
    closed: ClassVar[bool] = True

    @classmethod
    def read(cls: Type[P], data: Optional[Mapping[str, Any]]) -> P:
        """The payload of *data*, its unknown fields left out."""
        data = data or {}
        return cls(**{f.name: data.get(f.name) for f in fields(cls)})

    @classmethod
    def names(cls) -> frozenset:
        return frozenset(f.name for f in fields(cls))


@dataclass(frozen=True)
class TaskEvent(Payload):
    """A run's task, item or workflow lifecycle (`fibsem.hooks.HookContext`)."""

    task_name: Optional[str] = None
    task_type: Optional[str] = None
    item_name: Optional[str] = None
    item_id: Optional[str] = None
    task_id: Optional[str] = None
    experiment_id: Optional[str] = None
    experiment_name: Optional[str] = None
    tasks_remaining: Optional[int] = None
    tasks_total: Optional[int] = None
    task_state: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    skip_reason: Optional[str] = None
    missing_prereqs: Optional[List[str]] = None
    task_counts: Optional[Dict[str, int]] = None
    iteration: Optional[int] = None
    timestamp: Optional[float] = None


@dataclass(frozen=True)
class ProposalHold(Payload):
    """The run holds on a proposal, or lets go of it."""

    item_name: Optional[str] = None
    task_name: Optional[str] = None
    # Who may decide it, on proposal_waiting.
    deciders: Optional[List[str]] = None


@dataclass(frozen=True)
class SuggestionDeclined(Payload):
    """The operator declined the agent's suggestion."""

    proposal_id: Optional[str] = None
    suggestion_id: Optional[str] = None
    kind: Optional[str] = None
    item: Optional[str] = None
    task: Optional[str] = None
    proposed: Optional[Any] = None
    suggested_by: Optional[str] = None
    reason: Optional[str] = None


@dataclass(frozen=True)
class ProposalDecided(Payload):
    """A proposal was decided: a run's question, a suggestion, or the agent's
    request for actions (``kind`` "actions")."""

    closed: ClassVar[bool] = False

    proposal_id: Optional[str] = None
    kind: Optional[str] = None
    item: Optional[str] = None
    item_id: Optional[str] = None
    task: Optional[str] = None
    decision: Optional[str] = None
    outcome: Optional[str] = None
    decided: Optional[Any] = None
    proposed: Optional[Any] = None
    reason: Optional[str] = None
    actor: Optional[str] = None
    via: Optional[str] = None
    suggestion_id: Optional[str] = None
    suggested_by: Optional[str] = None


@dataclass(frozen=True)
class CommandRun(Payload):
    """A long service command (a mill, a spot burn) started or ended. Open:
    the run's labels ride along."""

    closed: ClassVar[bool] = False

    run_id: Optional[str] = None
    service: Optional[str] = None
    command: Optional[str] = None
    state: Optional[str] = None
    error: Optional[str] = None
    # A caller waited for it, so it has the end already.
    waited: Optional[bool] = None


@dataclass(frozen=True)
class LevelSet(Payload):
    """The operator set the agent's level."""

    level: Optional[str] = None
    previous: Optional[str] = None


@dataclass(frozen=True)
class PlanEdited(Payload):
    """The operator edited sections of the agent's plan."""

    sections: Optional[List[str]] = None
    session_id: Optional[str] = None


@dataclass(frozen=True)
class MillingStageStarted(Payload):
    milling_run_id: Optional[str] = None
    milling_task_name: Optional[str] = None
    stage: Optional[Dict[str, Any]] = None
    stage_index: Optional[int] = None
    total_stages: Optional[int] = None


@dataclass(frozen=True)
class Milling(Payload):
    """A milling run ended: how, and its stages."""

    milling_run_id: Optional[str] = None
    milling_task_name: Optional[str] = None
    status: Optional[str] = None
    # Why it ended so, when it was stopped or refused.
    reason: Optional[str] = None
    error: Optional[str] = None
    path: Optional[str] = None
    stages: Optional[List[Any]] = None
    started_at: Optional[Any] = None


_LIFECYCLE = (
    k.TASK_STARTED,
    k.TASK_COMPLETED,
    k.TASK_FAILED,
    k.TASK_CANCELLED,
    k.TASK_SKIPPED,
    k.ITEM_COMPLETED,
    k.WORKFLOW_STARTED,
    k.WORKFLOW_COMPLETED,
    k.WORKFLOW_CANCELLED,
    k.WORKFLOW_STALLED,
    k.WORKFLOW_FAILED,
    k.EXPERIMENT_COMPLETED,
)

# The typed kinds, and their payloads.
PAYLOADS: Dict[str, Type[Payload]] = {
    **{kind: TaskEvent for kind in _LIFECYCLE},
    k.PROPOSAL_HELD: ProposalHold,
    k.PROPOSAL_RELEASED: ProposalHold,
    k.PROPOSAL_WAITING: ProposalHold,
    k.SUGGESTION_DECLINED: SuggestionDeclined,
    k.PROPOSAL_DECIDED: ProposalDecided,
    k.COMMAND_STARTED: CommandRun,
    k.COMMAND_ENDED: CommandRun,
    k.AGENT_LEVEL_SET: LevelSet,
    k.AGENT_PLAN_EDITED: PlanEdited,
    k.MILLING_STAGE_STARTED: MillingStageStarted,
    k.MILLING: Milling,
}


def payload_type(kind: str) -> Optional[Type[Payload]]:
    return PAYLOADS.get(kind)


def read_payload(record: Mapping[str, Any]) -> Optional[Payload]:
    """The typed payload of *record*, or None for a kind not typed here."""
    cls = PAYLOADS.get(record.get("kind", ""))
    if cls is None:
        return None
    return cls.read(record.get("payload"))


def unknown_fields(kind: str, payload: Any) -> List[str]:
    """The fields of *payload* that a closed kind's payload does not name."""
    cls = PAYLOADS.get(kind)
    if cls is None or not cls.closed or not isinstance(payload, Mapping):
        return []
    return sorted(set(payload) - cls.names())
