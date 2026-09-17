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

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from fibsem.structures import Point

__all__ = [
    "Alternative",
    "Author",
    "AuthorKind",
    "Decision",
    "DecisionOutcome",
    "DecisionResult",
    "PROPOSAL_KINDS",
    "MILLING_SETUP",
    "Proposal",
    "ProposalKind",
    "compute_delta",
    "has_value_writer",
    "human_author",
    "known_value_names",
    "agent_author",
    "register_proposal_kind",
    "supersede",
    "write_value",
]

# The milling position: the point of interest the milling tasks follow.
MILLING_SETUP = "milling_setup"
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


register_proposal_kind(ProposalKind(name=MILLING_SETUP, values=("poi", "fiducial")))
register_proposal_kind(ProposalKind(name=TASK_RESULT, values=()))


# ---------------------------------------------------------------------------
# Values: how each named value is stored, and how confirming writes it through
# ---------------------------------------------------------------------------
#
# A value exists in a proposal because a later task consumes it. ``poi`` is
# read by the milling tasks that sync their patterns to it; ``fiducial`` will
# be read by the fiducial task. The name is the contract, so codecs and
# writers are keyed by it rather than by kind.


def _point_to_dict(p: Any) -> Any:
    return p.to_dict() if isinstance(p, Point) else p


def _point_from_dict(d: Any) -> Any:
    return Point.from_dict(d) if isinstance(d, dict) and "x" in d else d


_VALUE_CODECS: Dict[str, Tuple[Callable[[Any], Any], Callable[[Any], Any]]] = {
    "poi": (_point_to_dict, _point_from_dict),
    "fiducial": (_point_to_dict, _point_from_dict),
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


def _write_poi(item: Any, value: Point) -> List[str]:
    """The GUI's move path: set the point, then sync the patterns that follow
    it. Same domain call, same order -- a write that bypassed the sync left the
    rough and polishing patterns detached from the new point."""
    item.poi = value
    synced = item.sync_tasks_to_poi()
    if synced:
        logging.info(f"Synced tasks to POI: {synced}")
    return list(synced)


_VALUE_WRITERS: Dict[str, Callable[[Any, Any], Any]] = {
    "poi": _write_poi,
}


def has_value_writer(name: str) -> bool:
    return name in _VALUE_WRITERS


def known_value_names() -> List[str]:
    return sorted(_VALUE_WRITERS)


def write_value(item: Any, name: str, value: Any) -> Any:
    """Write one confirmed value through to its item. Unknown names are an
    error at the write, not silently dropped: a proposal carrying a value
    nothing consumes is a producer bug."""
    try:
        writer = _VALUE_WRITERS[name]
    except KeyError:
        raise KeyError(
            f"No writer for proposal value {name!r}; known: {sorted(_VALUE_WRITERS)}"
        ) from None
    return writer(item, value)


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
    delta: Dict[str, Any] = field(default_factory=dict)
    synced_tasks: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "applied": self.applied,
            "reason": self.reason,
            "running": self.running,
            "delta": _encode_values(self.delta),
            "synced_tasks": list(self.synced_tasks),
        }
