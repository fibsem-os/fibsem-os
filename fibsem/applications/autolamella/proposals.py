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
from typing import Any, Callable, Dict, List, Optional, Tuple

from fibsem.structures import Point

__all__ = [
    "Alternative",
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
    "write_value",
]

# The one kind v1 has. Its consumer mills, so a reject retires the item.
MILLING_SETUP = "milling_setup"


class DecisionOutcome(Enum):
    Confirmed = auto()
    Rejected = auto()


def human_author(name: str) -> str:
    return f"human:{name}" if name else "human:"


def agent_author(model: str) -> str:
    return f"agent:{model}" if model else "agent:"


# ---------------------------------------------------------------------------
# Kinds: declared in code by the producing task, never configured
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProposalKind:
    """What a proposal of this kind means to the run.

    ``gating``: a consumer is queued and waiting on the decision, so a reject
    means *nothing further here* and retires the item. Not gating (generative):
    the review creates the items later work operates on, so a reject creates
    nothing and the run moves on. The same deadline policy is safe on one and
    dangerous on the other, which is why this is a property of the kind and
    not a setting.
    """

    name: str
    gating: bool
    values: Tuple[str, ...]  # the value names a proposal of this kind may carry


PROPOSAL_KINDS: Dict[str, ProposalKind] = {}


def register_proposal_kind(kind: ProposalKind) -> ProposalKind:
    PROPOSAL_KINDS[kind.name] = kind
    return kind


register_proposal_kind(
    ProposalKind(name=MILLING_SETUP, gating=True, values=("poi", "fiducial"))
)


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
    author: str  # "human:<name>" | "agent:<model>"
    values: Dict[str, Any] = field(default_factory=dict)  # confirmed values
    reason: str = ""  # required on Rejected
    timestamp: float = field(default_factory=lambda: datetime.timestamp(datetime.now()))

    def to_dict(self) -> dict:
        return {
            "outcome": self.outcome.name,
            "author": self.author,
            "values": _encode_values(self.values),
            "reason": self.reason,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Decision":
        return cls(
            outcome=DecisionOutcome[data["outcome"]],
            author=data.get("author", ""),
            values=_decode_values(data.get("values", {})),
            reason=data.get("reason", ""),
            timestamp=data.get("timestamp", 0.0),
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

    @property
    def pending(self) -> bool:
        return not self.decisions

    @property
    def current(self) -> Optional[Decision]:
        """The latest decision; the log is the truth, this is the answer."""
        return self.decisions[-1] if self.decisions else None

    @property
    def gating(self) -> bool:
        kind = PROPOSAL_KINDS.get(self.kind)
        return kind.gating if kind is not None else True

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
        )


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
