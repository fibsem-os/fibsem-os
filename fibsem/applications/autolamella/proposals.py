"""What a task proposed, and what a reviewer decided about it.

A producing task completes and leaves a ``Proposal`` on its item, keyed by the
task's name. Nothing downstream reads the proposal directly: the consumer of
the decision is gated until somebody appends a ``Decision``, and confirming
writes the decided values through to the item (``lamella.poi`` keeps meaning
*the confirmed point of interest*). The proposed values are never overwritten
-- the delta between what was proposed and what was confirmed is computed from
the two, which is what makes a review produce a corrected label instead of a
self-reported one.

"Awaiting review" is not a status anywhere. The producer is ``Completed``, the
consumer has not started, and the work queue is rebuilt every run; what is
true is that the proposal has no decision, and everything else is derived from
that.

The records here are plain data. The one write path is
``Experiment.decide``, which owns the thread and the lock.
"""

from __future__ import annotations

import contextlib
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

from fibsem.structures import Point

__all__ = [
    "Alternative",
    "Author",
    "AuthorKind",
    "Decision",
    "Proposer",
    "TaskResultProposer",
    "DecisionOutcome",
    "DecisionResult",
    "PROPOSAL_KINDS",
    "POINT_OF_INTEREST",
    "Proposal",
    "ProposalKind",
    "compute_delta",
    "has_value_writer",
    "human_author",
    "known_value_names",
    "agent_author",
    "register_proposal_kind",
    "PreparedWrite",
    "prepare_values",
    "supersede",
    "ValueRefused",
]

# A point on an image: the point of interest the tasks that follow it sync to.
POINT_OF_INTEREST = "point_of_interest"
# What a task did, for someone to look at: no values, the final reference
# images in provenance. Recorded by the base task class for any task whose
# review is on and that did not propose a kind of its own.
TASK_RESULT = "task_result"


class DecisionOutcome(Enum):
    Confirmed = auto()
    Rejected = auto()


class AuthorKind(str, Enum):
    """What kind of thing decided. ``automated`` is the producer confirming
    its own record so the run continues; a person's look at it is a later
    decision. ``agent`` is a connected agent deciding over the server: not
    automatic (a decider acted) and not a person (someone may still want to
    check), which is why it has a kind of its own."""

    automated = "auto"
    human = "human"
    agent = "agent"


@dataclass(frozen=True)
class Author:
    """Who made a decision: the kind of thing, and its name -- the operator,
    the agent's model, the producer that confirmed its own record. Stored and
    sent as ``kind:name`` (``human:op``, ``auto:current-poi``), which is also
    what ``str()`` gives, so the wire and the file are unchanged."""

    kind: AuthorKind
    name: str = ""

    def __str__(self) -> str:
        return f"{self.kind.value}:{self.name}"

    @classmethod
    def parse(cls, text: Union[str, "Author"]) -> "Author":
        """The ``kind:name`` form back into an Author. Text with no known
        prefix is a human whose name is the whole string: every record ever
        written carried a prefix, so this is for hand-typed input."""
        if isinstance(text, Author):
            return text
        kind, sep, name = str(text or "").partition(":")
        if sep and kind in {k.value for k in AuthorKind}:
            return cls(AuthorKind(kind), name)
        return cls(AuthorKind.human, str(text or ""))

    @property
    def label(self) -> str:
        """How it reads: the name for a person, "agent · model", "auto ·
        proposer"; "someone" / "unknown" when the name is blank."""
        if self.kind is AuthorKind.human:
            return self.name or "someone"
        return f"{self.kind.value} · {self.name or 'unknown'}"


def human_author(name: str) -> Author:
    return Author(AuthorKind.human, name)


def agent_author(model: str) -> Author:
    return Author(AuthorKind.agent, model)


def auto_author(proposer: str) -> Author:
    """The author of a decision nobody made: the producer confirms its own
    proposal so the run continues, and the record says so."""
    return Author(AuthorKind.automated, proposer)


# ---------------------------------------------------------------------------
# Kinds: declared in code by the producing task, never configured
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProposalKind:
    """A kind names the set of values a proposal may carry. A value exists
    because a later task consumes it; a kind with no values is a result for
    someone to look at. What a decision does to the run is not a property of
    the kind: a task that waits on a decision is AwaitingDecision, and the
    decision finishes it (Completed or Failed) like any other task."""

    name: str
    values: Tuple[str, ...]  # the value names a proposal of this kind may carry


PROPOSAL_KINDS: Dict[str, ProposalKind] = {}


def register_proposal_kind(kind: ProposalKind) -> ProposalKind:
    PROPOSAL_KINDS[kind.name] = kind
    return kind


register_proposal_kind(ProposalKind(name=POINT_OF_INTEREST, values=("poi",)))
register_proposal_kind(ProposalKind(name=TASK_RESULT, values=()))


# ---------------------------------------------------------------------------
# Values: how each named value is stored, and how confirming writes it through
# ---------------------------------------------------------------------------
#
# A value exists in a proposal because a later task consumes it. ``poi`` is
# read by the milling tasks that sync their patterns to it. The name is the
# contract, so codecs and writers are keyed by it rather than by kind.


def _point_to_dict(p: Any) -> Any:
    return p.to_dict() if isinstance(p, Point) else p


def _point_from_dict(d: Any) -> Any:
    return Point.from_dict(d) if isinstance(d, dict) and "x" in d else d


_VALUE_CODECS: Dict[str, Tuple[Callable[[Any], Any], Callable[[Any], Any]]] = {
    "poi": (_point_to_dict, _point_from_dict),
}


def _encode_values(values: Dict[str, Any]) -> Dict[str, Any]:
    return {
        name: _VALUE_CODECS.get(name, (lambda v: v, lambda v: v))[0](value)
        for name, value in values.items()
    }


def _decode_values(values: Dict[str, Any]) -> Dict[str, Any]:
    return {
        name: _VALUE_CODECS.get(name, (lambda v: v, lambda v: v))[1](value)
        for name, value in (values or {}).items()
    }


class ValueRefused(ValueError):
    """A confirmed value that cannot be written: refused before anything is."""


def _quietly(obj: Any):
    """Assignments on ``obj`` without its events: an undo restores what nobody
    was told had changed, and a subscriber must not be able to stop it."""
    events = getattr(obj, "events", None)
    blocked = getattr(events, "blocked", None)
    return blocked() if callable(blocked) else contextlib.nullcontext()


@dataclass
class PreparedWrite:
    """A planned write: ``apply`` does it (assignments only, returning the
    tasks whose patterns moved); ``undo`` puts back what was there when it was
    planned. Assignments are not failure-free -- an evented item runs its
    subscribers on each one -- so a caller applies, and undoes on any error,
    before it commits the decision."""

    apply: Callable[[], List[str]]
    undo: Callable[[], None]


def _prepare_poi(item: Any, value: Any) -> PreparedWrite:
    """The GUI's move path, planned in full before any of it happens: the new
    point, then every pattern that follows it. Same domain plan, same order --
    a write that bypassed the sync left the rough and polishing patterns
    detached from the new point."""
    if not isinstance(value, Point):
        raise ValueRefused(f"poi must be a Point, not {type(value).__name__}.")
    for axis in (value.x, value.y):
        if isinstance(axis, bool) or not isinstance(axis, (int, float)):
            raise ValueRefused(f"poi must have numeric x and y, not {value!r}.")
        if not math.isfinite(axis):
            raise ValueRefused(f"poi must be finite, not {value!r}.")
    plan = getattr(item, "poi_sync_plan", None)
    if plan is None:
        raise ValueRefused(f"{getattr(item, 'name', 'this item')} has no poi.")
    synced, moves = plan(value)
    old_poi = item.poi
    old_points = [(pattern, pattern.point) for pattern, _ in moves]

    def apply() -> List[str]:
        item.poi = value
        for pattern, moved in moves:
            pattern.point = moved
        if synced:
            logging.info(f"Synced tasks to POI: {synced}")
        return list(synced)

    def undo() -> None:
        with _quietly(item):
            item.poi = old_poi
        for pattern, point in reversed(old_points):
            with _quietly(pattern):
                pattern.point = point

    return PreparedWrite(apply=apply, undo=undo)


# name -> prepare(item, value): checks the value and plans every effect without
# touching the item, returning how to apply it and how to undo it.
_VALUE_WRITERS: Dict[str, Callable[[Any, Any], PreparedWrite]] = {
    "poi": _prepare_poi,
}


def has_value_writer(name: str) -> bool:
    return name in _VALUE_WRITERS


def known_value_names() -> List[str]:
    return sorted(_VALUE_WRITERS)


def prepare_values(item: Any, kind: str, values: Dict[str, Any]) -> PreparedWrite:
    """Check every confirmed value and plan every write, touching nothing.

    Refused (``ValueRefused``) when a name is not one the proposal's kind
    carries, when nothing consumes it, when a value has the wrong type, or when
    the item cannot take it. Otherwise returns one write that applies them all
    (returning the tasks whose patterns moved) and one undo that puts every
    one of them back."""
    carried = PROPOSAL_KINDS[kind].values if kind in PROPOSAL_KINDS else ()
    foreign = [n for n in values if n not in carried]
    if foreign:
        raise ValueRefused(
            f"A {kind} proposal does not carry {foreign}; it carries {list(carried)}."
        )
    unknown = [n for n in values if not has_value_writer(n)]
    if unknown:
        raise ValueRefused(
            f"No consumer writes {unknown}; known values: {known_value_names()}."
        )
    steps = [_VALUE_WRITERS[name](item, value) for name, value in values.items()]

    def apply() -> List[str]:
        synced: List[str] = []
        for step in steps:
            synced.extend(step.apply() or [])
        return synced

    def undo() -> None:
        for step in reversed(steps):
            step.undo()

    return PreparedWrite(apply=apply, undo=undo)


def compute_delta(proposed: Any, confirmed: Any) -> Any:
    """confirmed - proposed, for the value types a delta means something on.
    None where it does not."""
    if isinstance(proposed, Point) and isinstance(confirmed, Point):
        return Point(x=confirmed.x - proposed.x, y=confirmed.y - proposed.y)
    if isinstance(proposed, (int, float)) and isinstance(confirmed, (int, float)):
        return confirmed - proposed
    return None


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class Alternative:
    """A candidate the proposer considered and did not pick, with why. The part
    of a proposal a general reviewer can actually check: rejections are easier
    to verify than acceptances."""

    values: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "values": _encode_values(self.values),
            "score": self.score,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Alternative":
        return cls(
            values=_decode_values(data.get("values", {})),
            score=data.get("score"),
            reason=data.get("reason", ""),
        )


@dataclass
class Decision:
    """One reviewer's answer. Appended, never edited: a reversal is a second
    decision beside the first, with its own author and time."""

    outcome: DecisionOutcome
    author: Author
    values: Dict[str, Any] = field(default_factory=dict)  # confirmed values
    reason: str = ""  # required on Rejected
    timestamp: float = field(default_factory=lambda: datetime.timestamp(datetime.now()))
    # Where it was decided: "workflow" (inline, in the task's own question, or
    # the producer confirming its own proposal), "review" (the Review tab) or
    # "server" (an agent over the API). The author says who; this says where,
    # so an export can tell an inline answer from a tab decision.
    via: str = ""
    # Which run of the producing task this decides: the task_id of the proposal
    # the decider was shown. Experiment.decide refuses a decision that names no
    # run, or a run the proposal is no longer from (the task re-ran since).
    task_id: str = ""

    def __post_init__(self) -> None:
        # a string from the file, the wire or a test is accepted and parsed
        self.author = Author.parse(self.author)

    def to_dict(self) -> dict:
        return {
            "outcome": self.outcome.name,
            "author": str(self.author),
            "values": _encode_values(self.values),
            "reason": self.reason,
            "timestamp": self.timestamp,
            "via": self.via,
            "task_id": self.task_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Decision":
        return cls(
            outcome=DecisionOutcome[data["outcome"]],
            author=Author.parse(data.get("author", "")),
            values=_decode_values(data.get("values", {})),
            reason=data.get("reason", ""),
            timestamp=data.get("timestamp", 0.0),
            via=data.get("via", ""),
            task_id=data.get("task_id", ""),
        )


@dataclass
class Proposal:
    """What a task proposed for its item, and every decision made about it.

    ``values`` are the proposer's answer, keyed by value name, and are never
    overwritten. ``provenance`` says what they were computed from -- at least
    the proposer's name and the reference image the values sit on, since a
    delta only means something against the same image.
    """

    kind: str
    values: Dict[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None
    alternatives: List[Alternative] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    decisions: List[Decision] = field(default_factory=list)
    created_at: float = field(
        default_factory=lambda: datetime.timestamp(datetime.now())
    )
    # Earlier proposals for the same item and task, oldest first, each with
    # its decisions. A deliberate re-run of the producing task supersedes a
    # decided proposal rather than keeping or overwriting it: the operator
    # asked for a new answer on a new image, and the old answer -- and its
    # delta -- stays on the record. The old value is not carried over as the
    # new default; a stale default is the rubber stamp the delta detects.
    superseded: List["Proposal"] = field(default_factory=list)

    @property
    def pending(self) -> bool:
        return not self.decisions

    @property
    def task_id(self) -> str:
        """The run that made this proposal: the producing task's task_id, stamped
        on the provenance when it is proposed. What a decision names. Empty for
        a proposal recorded before runs were named, which cannot be decided."""
        return str(self.provenance.get("task_id") or "")

    @property
    def current(self) -> Optional[Decision]:
        """The latest decision; the log is the truth, this is the answer."""
        return self.decisions[-1] if self.decisions else None

    @property
    def to_check(self) -> bool:
        """Applied by its own producer, or decided by an agent, and not looked
        at by a person since. A person's acknowledgement -- or their reject --
        is a later decision, which clears it."""
        return bool(self.decisions) and all(
            d.author.kind is not AuthorKind.human for d in self.decisions
        )

    @property
    def applied(self) -> Optional[Decision]:
        """The decision whose values were written through: the latest
        Confirmed one that carried values. An acknowledgement carries none."""
        for d in reversed(self.decisions):
            if d.outcome is DecisionOutcome.Confirmed and d.values:
                return d
        return None

    def delta(self, decision: Optional[Decision] = None) -> Dict[str, Any]:
        """confirmed - proposed per value, for the given (default: current)
        decision. Empty when there is no confirmed decision."""
        decision = decision or self.current
        if decision is None or decision.outcome is not DecisionOutcome.Confirmed:
            return {}
        return {
            name: compute_delta(self.values.get(name), value)
            for name, value in decision.values.items()
        }

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "values": _encode_values(self.values),
            "confidence": self.confidence,
            "alternatives": [a.to_dict() for a in self.alternatives],
            "provenance": dict(self.provenance),
            "decisions": [d.to_dict() for d in self.decisions],
            "created_at": self.created_at,
            "superseded": [p.to_dict() for p in self.superseded],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Proposal":
        return cls(
            kind=data["kind"],
            values=_decode_values(data.get("values", {})),
            confidence=data.get("confidence"),
            alternatives=[
                Alternative.from_dict(a) for a in data.get("alternatives", [])
            ],
            provenance=dict(data.get("provenance", {})),
            decisions=[Decision.from_dict(d) for d in data.get("decisions", [])],
            created_at=data.get("created_at", 0.0),
            superseded=[Proposal.from_dict(p) for p in data.get("superseded", [])],
        )


# ---------------------------------------------------------------------------
# Proposers: the part of proposing that differs per kind
# ---------------------------------------------------------------------------
#
# Every proposal is a task result -- what ran, when, how it ended, the final
# images -- plus, for kinds that have them, values a decision can edit and a
# later task consumes. The task base records the result part for every kind
# (AutoLamellaTask.propose); a proposer supplies only the values. TASK_RESULT
# is the kind with none.


class Proposer(Protocol):
    """Names a kind and computes its values for a finished task, as a
    Proposal of that kind: values, confidence, alternatives and whatever
    provenance the proposer has to add (a model name, say). The task base
    stamps the result part -- which task, when, how it ended, the final
    images -- onto that provenance for every kind. ``name`` is what the
    record says proposed the values and what the producer's own confirmation
    is signed as; empty means the task itself. A task holds one
    (``AutoLamellaTask.proposer``); swapping it -- a segmentation model for
    the current point, say -- changes nothing downstream, which keys by kind.
    None declines: nothing to propose, no record."""

    kind: str
    name: str
    version: int

    def propose(self, task: Any) -> Optional[Proposal]: ...


class TaskResultProposer:
    """The default: no values, the record is the result. Signed as the task."""

    kind = TASK_RESULT
    name = ""
    version = 1

    def propose(self, task: Any) -> Optional[Proposal]:
        return Proposal(kind=self.kind)


def supersede(old: Optional[Proposal], new: Proposal) -> Proposal:
    """``new`` replaces ``old`` for the same item and task, keeping ``old`` and
    everything before it on the record, oldest first, flat (no nesting)."""
    if old is not None:
        history = list(old.superseded)
        old.superseded = []
        new.superseded = history + [old]
    return new


def proposals_to_dict(proposals: Dict[str, Proposal]) -> Dict[str, dict]:
    return {name: p.to_dict() for name, p in proposals.items()}


def proposals_from_dict(data: Optional[Dict[str, dict]]) -> Dict[str, Proposal]:
    return {name: Proposal.from_dict(p) for name, p in (data or {}).items()}


@dataclass
class DecisionResult:
    """What ``Experiment.decide`` did. ``applied`` False carries the reason;
    ``running`` True means the consumer had already started and the answer is
    to stop it, not to decide."""

    applied: bool
    reason: str = ""
    running: bool = False
    # Why a refusal, for a client to act on: "missing_field" (the decision
    # names no run), "stale_review" (the task re-ran since it was shown),
    # "invalid_value" (the values cannot be confirmed as given). Empty when
    # applied, and for the other refusals (running, nothing pending).
    error_type: str = ""
    delta: Dict[str, Any] = field(default_factory=dict)
    synced_tasks: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "applied": self.applied,
            "reason": self.reason,
            "running": self.running,
            "error_type": self.error_type,
            "delta": _encode_values(self.delta),
            "synced_tasks": list(self.synced_tasks),
        }
