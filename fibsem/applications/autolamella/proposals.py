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
import hashlib
import logging
import math
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

from fibsem.applications.autolamella.poses import LamellaPoses
from fibsem.structures import MicroscopeState, Point

__all__ = [
    "Alternative",
    "Author",
    "AuthorKind",
    "Decision",
    "Proposer",
    "TaskResultProposer",
    "DecisionOutcome",
    "DecisionResult",
    "DETECTION",
    "PROPOSAL_KINDS",
    "OVERVIEW_POSITIONS",
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
    "record",
    "current_proposal",
    "ValueRefused",
]

# A point on an image: the point of interest the tasks that follow it sync to.
POINT_OF_INTEREST = "point_of_interest"
# Where the lamellae go on a grid: the positions a decision creates them at.
# The only kind whose confirmed values make items rather than edit the one the
# proposal is on.
OVERVIEW_POSITIONS = "overview_positions"
# Where a model put the features it was asked to find, for someone to correct:
# the one kind so far that is asked *during* a task rather than after it
# (FIB-1025). Its value is the feature set, keyed by name, so a delta is per
# feature and not one number for the lot.
DETECTION = "detection"
# What a task did, for someone to look at: no values, the final reference
# images in provenance. Recorded by the base task class for any task whose
# review is on and that did not propose a kind of its own.
TASK_RESULT = "task_result"


class DecisionOutcome(Enum):
    """What was decided. ``Confirmed`` and ``Rejected`` are answers; a decider
    looked and said something. ``Withdrawn`` is not: the question was taken
    back because the thing that asked it is gone, so there is no answer to
    read and nothing to compare a proposal against."""

    Confirmed = auto()
    Rejected = auto()
    Withdrawn = auto()


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
register_proposal_kind(ProposalKind(name=OVERVIEW_POSITIONS, values=("positions",)))
register_proposal_kind(ProposalKind(name=DETECTION, values=("features",)))
register_proposal_kind(ProposalKind(name=TASK_RESULT, values=()))


# ---------------------------------------------------------------------------
# Values: how each named value is stored, and how confirming writes it through
# ---------------------------------------------------------------------------
#
# A value exists in a proposal because a later task consumes it. ``poi`` is
# read by the milling tasks that sync their patterns to it. The name is the
# contract, so codecs and writers are keyed by it rather than by kind.


# A position a decision creates a lamella at is carried as the pose pair the
# instrument's geometry produced for it (``build_lamella_poses``), which is
# what ``Experiment.add_new_lamella`` takes.
#
# Not a point on an image and not a bare stage position: either would have
# made a decision need a microscope to turn into poses, and there is no
# instrument where a decision is made -- the Review tab days later, or an
# agent over the server. The geometry is read where the position is placed
# instead, in the renderer or in a proposer running at the beam.


def _positions_to_dict(value: Any) -> Any:
    if not isinstance(value, (list, tuple)):
        return value
    return [
        {
            "milling": p.milling.to_dict(),
            "fluorescence": (
                p.fluorescence.to_dict() if p.fluorescence is not None else None
            ),
        }
        if isinstance(p, LamellaPoses)
        else p
        for p in value
    ]


def _positions_from_dict(value: Any) -> Any:
    if not isinstance(value, (list, tuple)):
        return value
    out: List[Any] = []
    for p in value:
        if not isinstance(p, dict) or "milling" not in p:
            out.append(p)
            continue
        fm = p.get("fluorescence")
        out.append(
            LamellaPoses(
                milling=MicroscopeState.from_dict(p["milling"]),
                fluorescence=MicroscopeState.from_dict(fm) if fm else None,
            )
        )
    return out


def _features_to_dict(value: Any) -> Any:
    """A detected feature set as ``[{name, px}]``.

    Only the name and the pixel it sits on: the mask, the rgb and the image
    are megabytes and already on disk beside the task's outputs, and a delta
    is computed from the points alone.
    """
    if not isinstance(value, (list, tuple)):
        return value
    return [
        {"name": f["name"], "px": _point_to_dict(f["px"])} if isinstance(f, dict) else f
        for f in value
    ]


def _features_from_dict(value: Any) -> Any:
    if not isinstance(value, (list, tuple)):
        return value
    return [
        {"name": f.get("name", ""), "px": _point_from_dict(f.get("px"))}
        if isinstance(f, dict)
        else f
        for f in value
    ]


def _point_to_dict(p: Any) -> Any:
    return p.to_dict() if isinstance(p, Point) else p


def _point_from_dict(d: Any) -> Any:
    return Point.from_dict(d) if isinstance(d, dict) and "x" in d else d


_VALUE_CODECS: Dict[str, Tuple[Callable[[Any], Any], Callable[[Any], Any]]] = {
    "poi": (_point_to_dict, _point_from_dict),
    "positions": (_positions_to_dict, _positions_from_dict),
    "features": (_features_to_dict, _features_from_dict),
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

    @classmethod
    def nothing(cls) -> "PreparedWrite":
        """A value that is checked and written nowhere: its consumer is not the
        item. An in-run answer is the first -- the task parked on it applies it
        (FIB-1025) -- and an answer given at the instrument, already applied by
        the hardware when it is recorded, is the same shape."""
        return cls(apply=lambda: [], undo=lambda: None)


def _prepare_poi(experiment: Any, item: Any, value: Any) -> PreparedWrite:
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


def _prepare_positions(experiment: Any, item: Any, value: Any) -> PreparedWrite:
    """The only write that makes items: one lamella per position, on the grid
    the proposal is on.

    Planned in full before any of it happens, like every other write here, and
    undone together: a half-created set is worse than none, and the decision is
    only appended once the whole set is on the experiment. The poses come from
    the value rather than the instrument (see the codec above), so nothing here
    needs a microscope.
    """
    if not isinstance(value, (list, tuple)):
        raise ValueRefused(f"positions must be a list, not {type(value).__name__}.")
    placed: List[LamellaPoses] = []
    for entry in value:
        if not isinstance(entry, LamellaPoses):
            raise ValueRefused(
                f"every position must be a LamellaPoses, not {type(entry).__name__}."
            )
        if entry.milling is None:
            raise ValueRefused("every position needs a milling pose.")
        placed.append(entry)
    add = getattr(experiment, "add_new_lamella", None)
    if add is None:
        raise ValueRefused("positions can only be written to an experiment.")
    grid_id = getattr(item, "id", None)
    if grid_id is None:
        raise ValueRefused(
            f"{getattr(item, 'name', 'this item')} has no id to stamp on a lamella."
        )
    protocol = getattr(experiment, "task_protocol", None)
    if placed and getattr(protocol, "lamella_defaults", None) is None:
        # Checked here, not discovered half way through creating them: a
        # lamella is built from the protocol's defaults, and an experiment
        # with no protocol loaded cannot make one.
        raise ValueRefused(
            "No protocol is loaded, so a lamella cannot be created. Load one "
            "for this experiment and decide again."
        )
    made: List[Any] = []

    def apply() -> List[str]:
        for entry in placed:
            # add_new_lamella returns nothing and appends, so the lamella it
            # made is the one on the end. Taken here rather than assumed later:
            # undo has to remove exactly what this write added.
            add(
                microscope_state=entry.milling,
                task_config=deepcopy(getattr(protocol, "task_config", None)) or {},
                fluorescence_pose=entry.fluorescence,
                grid_id=grid_id,
            )
            made.append(experiment.positions[-1])
        if made:
            logging.info(
                f"Created {len(made)} lamella(e) on {getattr(item, 'name', '?')}: "
                + ", ".join(getattr(m, "name", "?") or "?" for m in made)
            )
        return []

    def undo() -> None:
        # Removing what this write made, newest first. Nothing else can have
        # taken them: the decision has not been appended and the experiment
        # lock is still held.
        for lamella in reversed(made):
            try:
                experiment.positions.remove(lamella)
            except ValueError:
                logging.warning(
                    f"Could not undo the creation of {getattr(lamella, 'name', '?')}"
                )
        made.clear()

    return PreparedWrite(apply=apply, undo=undo)


def _prepare_features(experiment: Any, item: Any, value: Any) -> PreparedWrite:
    """Checked, and written nowhere.

    Every other writer here edits the item, because its proposal is decided
    after the task that made it has ended and there is nobody left to act on
    the answer. A detection is asked *during* a task (FIB-1025): the consumer
    is the task itself, parked on the answer, which applies it with the
    instrument in the state it asked in. Writing it to the item here would be
    a second, later application of the same correction.

    So the value is still checked -- a malformed answer must be refused before
    the run is released on it -- and then left for its waiter.
    """
    if not isinstance(value, (list, tuple)):
        raise ValueRefused(f"features must be a list, not {type(value).__name__}.")
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueRefused(
                f"every feature must be a name and a point, not {type(entry).__name__}."
            )
        if not entry.get("name"):
            raise ValueRefused("every feature needs a name to be matched back by.")
        if not isinstance(entry.get("px"), Point):
            raise ValueRefused(f"{entry['name']} needs a point in image pixels.")
    return PreparedWrite.nothing()


# name -> prepare(experiment, item, value): checks the value and plans every
# effect without touching anything, returning how to apply it and how to undo
# it. The experiment is there for the one write that makes items rather than
# editing the one the proposal is on.
_VALUE_WRITERS: Dict[str, Callable[[Any, Any, Any], PreparedWrite]] = {
    "poi": _prepare_poi,
    "positions": _prepare_positions,
    "features": _prepare_features,
}


def has_value_writer(name: str) -> bool:
    return name in _VALUE_WRITERS


def known_value_names() -> List[str]:
    return sorted(_VALUE_WRITERS)


def prepare_values(
    experiment: Any, item: Any, kind: str, values: Dict[str, Any]
) -> PreparedWrite:
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
    steps = [
        _VALUE_WRITERS[name](experiment, item, value) for name, value in values.items()
    ]

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
    if isinstance(proposed, (list, tuple)) and isinstance(confirmed, (list, tuple)):
        # A named set -- a detection's features -- so the delta is per name and
        # not one number for the lot: which feature the model got wrong is the
        # part worth keeping. Matched by name rather than by position, because
        # a decider answers the set and need not answer it in order.
        was = {
            f["name"]: f.get("px")
            for f in proposed
            if isinstance(f, dict) and f.get("name")
        }
        now = {
            f["name"]: f.get("px")
            for f in confirmed
            if isinstance(f, dict) and f.get("name")
        }
        if not was or not now:
            return None
        moved = {
            name: compute_delta(was[name], point)
            for name, point in now.items()
            if name in was
        }
        return moved or None
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
    # Which proposal this decides: the ``Proposal.id`` the decider was shown.
    # The run is not enough to say so -- a task may ask several questions in
    # one run, and they share a task_id -- so a decision that names its
    # proposal is checked against that, and the run is kept as what it is: a
    # fact about where the proposal came from. Empty on records and from
    # callers that predate it, which are checked by the run as before.
    proposal_id: str = ""

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
            "proposal_id": self.proposal_id,
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
            proposal_id=data.get("proposal_id", ""),
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
    # Whether the task that made this is parked on it right now, waiting to be
    # told the answer -- an in-run question rather than a result left for
    # later (FIB-1025). It is the one thing that lets a decision land on a
    # running task, so it is deliberately **not persisted**: a question only
    # exists while something is waiting on it, and a flag that survived a
    # reload would claim a waiter that is gone.
    asking: bool = field(default=False, compare=False, repr=False)
    # This proposal's own name, minted when it is made. What a decision names
    # (``Decision.proposal_id``), and what anything outside the record -- an
    # event, a responder waiting on an answer -- points at: ``(item, task)``
    # stops being unique the moment a task asks twice. Opaque: nothing reads
    # anything out of it.
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

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
    def withdrawn(self) -> bool:
        """Closed without an answer: whatever asked this is gone -- the task
        failed, the run stopped, the operator aborted -- so the question was
        taken back rather than left open forever.

        Not ``pending`` (there is a decision on the record) and not an answer
        either, which is why it is its own property: everything that reads a
        decision as what somebody said has to skip these.
        """
        d = self.current
        return d is not None and d.outcome is DecisionOutcome.Withdrawn

    @property
    def to_check(self) -> bool:
        """Applied by its own producer, or decided by an agent, and not looked
        at by a person since. A person's acknowledgement -- or their reject --
        is a later decision, which clears it.

        A withdrawn proposal is never to check: nobody should be asked to
        acknowledge a question that was taken back before it was answered.
        """
        return (
            bool(self.decisions)
            and not self.withdrawn
            and all(d.author.kind is not AuthorKind.human for d in self.decisions)
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
            "id": self.id,
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
            id=data.get("id") or _id_for_a_record_without_one(data),
        )


def _id_for_a_record_without_one(data: dict) -> str:
    """An id for a proposal saved before proposals had one.

    Derived from the record rather than minted, so it is the same every time
    the file is read: two readers of one experiment -- the app and a monitor,
    this session and the next -- must agree on what a proposal is called
    without either of them having to save first.
    """
    provenance = data.get("provenance", {}) or {}
    seed = "|".join(
        str(part)
        for part in (
            data.get("kind", ""),
            provenance.get("task_id", ""),
            repr(data.get("created_at", 0.0)),
        )
    )
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Proposers: the part of proposing that differs per kind
# ---------------------------------------------------------------------------
#
# Every proposal is a task result -- what ran, when, how it ended, the final
# images -- plus, for kinds that have them, values a decision can edit and a
# later task consumes. The result part is recorded for every kind, on a
# lamella or a grid (workflows/tasks/proposing.py); a proposer supplies only
# the values. TASK_RESULT is the kind with none.


class Proposer(Protocol):
    """Names a kind and computes its values for a finished task, as a
    Proposal of that kind: values, confidence, alternatives and whatever
    provenance the proposer has to add (a model name, say). ``proposing.propose``
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


def record(proposals: List[Proposal], proposal: Proposal) -> Proposal:
    """Append ``proposal`` to an item's list for one task, and return it.

    The list is the record of everything that task proposed on the item,
    oldest first: a deliberate re-run leaves the old answer -- and its delta --
    on the record and puts the new one after it; a question the task asked
    mid-run sits before the run's own result. The old value is never carried
    over as the new default; a stale default is the rubber stamp the delta
    detects. A trailing proposal nobody answered is dropped first: it was
    never decided, so there is nothing to keep, and two open proposals for one
    task would be two questions where only one was ever asked.
    """
    if proposals and proposals[-1].pending:
        proposals.pop()
    proposals.append(proposal)
    return proposal


def current_proposal(
    proposals: Optional[List[Proposal]], kind: Optional[str] = None
) -> Optional[Proposal]:
    """The proposal a decision, the gate and the inbox act on: the last one
    for the task, or the last of ``kind``. Everything before it is what a
    later proposal replaced, and is on the record for its decisions and its
    delta only."""
    if not proposals:
        return None
    if kind is None:
        return proposals[-1]
    for p in reversed(proposals):
        if p.kind == kind:
            return p
    return None


def proposals_to_dict(proposals: Dict[str, List[Proposal]]) -> Dict[str, list]:
    return {name: [p.to_dict() for p in ps] for name, ps in proposals.items()}


def proposals_from_dict(
    data: Optional[Dict[str, list]],
) -> Dict[str, List[Proposal]]:
    return {
        name: [Proposal.from_dict(p) for p in ps] for name, ps in (data or {}).items()
    }


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
