"""What a task proposed, and what was decided about it.

A task asks a ``Proposal`` on its item while it runs, saying who asked it
(``asked_by``: the task, today). A ``Decision`` on it says who decided what,
and confirming writes the decided values through to the item. The proposed
values are never overwritten: the delta between what was proposed and what was
confirmed is computed from the two, which is what makes a decision produce a
corrected label instead of a self-reported one.

The records, the kinds and the value registries are here, for any
application's workflow. An application registers its own kinds
(``register_proposal_kind``) and, for each value they carry, how it is stored
(``register_value_codec``) and how confirming writes it to an item
(``register_value_writer``).

What a loop's pass says on the prompt bar: a mill's, a spot burn's.

A pass is a ``milling`` or ``spot_burn`` proposal (one per pass, see
``fibsem.workflows.proposals``) whose allowed actions depend on where the loop
is: before any pass it is run or skip, after one run again or accept, with
running off only accept. The prompt bar has two buttons, so each pass names
what they do here, and the words come from the kind.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import math
import threading
import uuid
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from fibsem.acting import Actor, ActorKind
from fibsem.structures import (
    FibsemRectangle,
    FibsemStagePosition,
    MicroscopeState,
    Point,
)
from fibsem.util.timestamps import now, to_datetime, to_iso

# One process, one open experiment: a single lock serialises the writers that
# can run on different threads at once -- the workflow thread saving after a
# task, and a decision from the GUI thread or the App Server. Held
# briefly by both, so neither serialises a half-applied write of the other.
EXPERIMENT_WRITE_LOCK = threading.RLock()

# Where a model put the features it was asked to find, for someone to correct: asked
# while the task runs. Its value is the feature set, keyed by name, so a delta is per
# feature and not one number for the lot.
DETECTION = "detection"
# The instrument's state as a task arrived at it, for the operator to confirm
# before the task goes on ("Acquire reference image. Press Continue when
# ready."). The proposal is the stage position then; the decision is a
# confirmation, and the task fills in the position as it stands afterwards,
# so a move the operator made first is the delta. Asked mid-task, so its
# value is written nowhere: the task is what uses it.
STATE = "state"
# The reduced area the alignment reference is taken in: a rectangle in
# fractions of the FIB frame, dragged on the last FIB image. Asked mid-task,
# right before that reference is acquired, so it is written nowhere: the
# task acquires with the area it is handed.
ALIGNMENT_AREA = "alignment_area"
# One pass of a loop the task runs and asks about again: a mill, a spot burn.
# The value is what the pass would run (the milling config, the spot-burn
# settings) and the decision says what to do with it (``actions``): run it and
# ask again, take an edit and ask again, or go on. One proposal per pass, so
# every pass and who decided it is on the record. Asked mid-task, so the value
# is written nowhere: the task runs what it is handed.
MILLING = "milling"
SPOT_BURN = "spot_burn"
# A point on an image: the point of interest the tasks that follow it sync to.
# Confirming writes it to the item, which says what follows it
# (``poi_sync_plan``): a lamella's milling patterns, a volume position's slice.
POINT_OF_INTEREST = "point_of_interest"
# What an agent at Assist asks to do: its tool calls, run in order once the
# operator approves. About the experiment, not an item; the value
# is the calls with the operator's lines for each, written nowhere.
ACTIONS = "actions"

# What a pass may be decided as. Each has one meaning whenever it is asked:
# ``run`` runs the value as decided and asks again; ``edit`` takes
# the value as decided and asks again, nothing runs; ``skip`` goes on without
# running it (before any pass); ``accept`` goes on with what was run (after a
# pass, or when running is off).
ACTION_RUN = "run"
ACTION_EDIT = "edit"
ACTION_SKIP = "skip"
ACTION_ACCEPT = "accept"
LOOP_ACTIONS = (ACTION_RUN, ACTION_EDIT, ACTION_SKIP, ACTION_ACCEPT)


class DecisionOutcome(Enum):
    """What was decided. ``Confirmed`` and ``Rejected`` are answers; a decider
    looked and said something. The other two are not, and only the record
    writes them. ``Withdrawn``: the question was taken back because the thing
    that asked it is gone, so there is no answer to read and nothing to
    compare a proposal against. ``Unreviewed``: nobody was asked (the task is
    automated, or the proposal is off in its settings), and the task used its
    own value. Never agreement:
    nothing that measures a proposer's or an agent's record may count it as
    one."""

    Confirmed = auto()
    Rejected = auto()
    Withdrawn = auto()
    Unreviewed = auto()


# ---------------------------------------------------------------------------
# Kinds: declared in code by the producing task, never configured
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProposalKind:
    """A kind names the set of values a proposal may carry. A value exists
    because a later task consumes it; a kind with no values is a result for
    someone to look at. What a reject of a question means is the kind's to
    say."""

    name: str
    values: Tuple[str, ...]  # the value names a proposal of this kind may carry
    # How the kind is named to a person, in a sentence.
    label: str = ""
    # What a decision on this kind says, beyond Confirmed: empty for a yes/no
    # kind. A kind with actions is confirmed with one of them, and a proposal
    # may narrow them for its own pass (``provenance["actions"]``: no skip
    # once something has run).
    actions: Tuple[str, ...] = ()
    # What silence means, for a kind with actions: the first of these the
    # proposal allows is what an unasked pass is recorded as deciding. A
    # loop's is ("accept", "run"): run the first pass once, then go on.
    silence: Tuple[str, ...] = ()

    def allowed_actions(self, proposal: "Proposal") -> Tuple[str, ...]:
        """What ``proposal`` may be decided as: its own narrowing of this
        kind's actions, else all of them."""
        narrowed = proposal.provenance.get("actions")
        if narrowed is None:
            return self.actions
        return tuple(a for a in narrowed if a in self.actions)

    def silent_action(self, proposal: "Proposal") -> str:
        """What an unasked ``proposal`` is recorded as deciding: empty for a
        yes/no kind."""
        allowed = self.allowed_actions(proposal)
        return next((a for a in self.silence if a in allowed), "")


PROPOSAL_KINDS: Dict[str, ProposalKind] = {}


def kind_label(kind: str) -> str:
    """The kind's name for a person: "Point of interest", "Position". A kind
    this build has not registered reads as its name with the underscores
    out, so a record from a newer build still says something."""
    registered = PROPOSAL_KINDS.get(kind)
    if registered is not None and registered.label:
        return registered.label
    return kind.replace("_", " ").capitalize()


def proposal_label(proposal: "Proposal") -> str:
    """What this proposal asks, for a person: its own ``label`` (a tilt asked
    as a position reads "Tilt"), else its kind's."""
    return str(proposal.provenance.get("label") or kind_label(proposal.kind))


def register_proposal_kind(kind: ProposalKind) -> ProposalKind:
    PROPOSAL_KINDS[kind.name] = kind
    return kind


register_proposal_kind(
    ProposalKind(name=DETECTION, values=("features",), label="Detection")
)
register_proposal_kind(
    ProposalKind(
        name=STATE,
        values=("stage_position",),
        label="Position",
    )
)
register_proposal_kind(
    ProposalKind(
        name=ALIGNMENT_AREA, values=("alignment_area",), label="Alignment area"
    )
)
register_proposal_kind(
    ProposalKind(
        name=MILLING,
        values=("milling_config",),
        label="Milling",
        actions=LOOP_ACTIONS,
        silence=(ACTION_ACCEPT, ACTION_RUN),
    )
)
register_proposal_kind(
    ProposalKind(
        name=SPOT_BURN,
        values=("spot_burn",),
        label="Spot burn",
        actions=LOOP_ACTIONS,
        silence=(ACTION_ACCEPT, ACTION_RUN),
    )
)
register_proposal_kind(
    ProposalKind(name=POINT_OF_INTEREST, values=("poi",), label="Point of interest")
)
register_proposal_kind(
    ProposalKind(name=ACTIONS, values=("actions",), label="Agent request")
)


# ---------------------------------------------------------------------------
# Values: how each named value is stored, and how confirming writes it through
# ---------------------------------------------------------------------------
#
# A value exists in a proposal because a later task consumes it. ``poi`` is
# read by the milling tasks that sync their patterns to it. The name is the
# contract, so codecs and writers are keyed by it rather than by kind.


_VALUE_CODECS: Dict[str, Tuple[Callable[[Any], Any], Callable[[Any], Any]]] = {}


def register_value_codec(
    name: str, to_dict: Callable[[Any], Any], from_dict: Callable[[Any], Any]
) -> None:
    """How the value called ``name`` is stored: a value with no codec is stored
    as it is."""
    _VALUE_CODECS[name] = (to_dict, from_dict)


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


def _point_to_dict(p: Any) -> Any:
    return p.to_dict() if isinstance(p, Point) else p


def _point_from_dict(d: Any) -> Any:
    return Point.from_dict(d) if isinstance(d, dict) and "x" in d else d


def features_to_dict(value: Any) -> Any:
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


register_value_codec("features", features_to_dict, _features_from_dict)
register_value_codec(
    "stage_position",
    lambda p: p.to_dict() if isinstance(p, FibsemStagePosition) else p,
    lambda d: FibsemStagePosition.from_dict(d) if isinstance(d, dict) else d,
)
register_value_codec(
    "alignment_area",
    lambda r: r.to_dict() if isinstance(r, FibsemRectangle) else r,
    lambda d: FibsemRectangle.from_dict(d) if isinstance(d, dict) else d,
)


@dataclass(frozen=True)
class ItemPoses:
    """Both poses for an item placed on an overview, whichever side it was
    marked from: where it is worked on with the beams, and where it is looked
    at under fluorescence.

    Carried as poses rather than a point on an image or a bare stage
    position: either would need a microscope to turn into poses wherever the
    item is made. The geometry is read where the position is placed instead.

    `fluorescence` is None when the microscope has no fluorescence detector --
    there is nothing to derive it from, and a pose invented for an instrument
    that does not exist would be worse than its absence.
    """

    beam: MicroscopeState
    fluorescence: Optional[MicroscopeState] = None


def _milling_config_from_dict(value: Any) -> Any:
    from fibsem.milling.tasks import FibsemMillingTaskConfig

    return (
        FibsemMillingTaskConfig.from_dict(value) if isinstance(value, dict) else value
    )


def _spot_burn_from_dict(value: Any) -> Any:
    from fibsem.imaging.spot import SpotBurnSettings

    return SpotBurnSettings.from_dict(value) if isinstance(value, dict) else value


def _to_dict(value: Any) -> Any:
    return value.to_dict() if hasattr(value, "to_dict") else value


register_value_codec("milling_config", _to_dict, _milling_config_from_dict)
register_value_codec("spot_burn", _to_dict, _spot_burn_from_dict)
register_value_codec("poi", _point_to_dict, _point_from_dict)


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
        -- and an answer given at the instrument, already applied by
        the hardware when it is recorded, is the same shape."""
        return cls(apply=lambda: [], undo=lambda: None)


_VALUE_WRITERS: Dict[str, Callable[[Any, Any, Any], PreparedWrite]] = {}


def register_value_writer(
    name: str, prepare: Callable[[Any, Any, Any], PreparedWrite]
) -> None:
    """How confirming the value called ``name`` writes it through: ``prepare``
    takes the experiment, the item and the value, checks it, and plans the
    write (``PreparedWrite``) without touching anything."""
    _VALUE_WRITERS[name] = prepare


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


# The values the in-run questions carry: checked, and written nowhere.


def _prepare_features(experiment: Any, item: Any, value: Any) -> PreparedWrite:
    """Checked, and written nowhere.

    A value decided after its task has ended (a point of interest, say) is written to
    the item, because there is nobody left to act on the answer. A detection is asked
    *during* a task: the consumer is the task itself, parked on the answer, which
    applies it with the instrument in the state it asked in. Writing it to the item here
    would be a second, later application of the same correction.

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


def _prepare_stage_position(experiment: Any, item: Any, value: Any) -> PreparedWrite:
    """Checked, and written nowhere, for the reason ``_prepare_features`` gives:
    a state is confirmed mid-task, and the instrument already holds it."""
    if not isinstance(value, FibsemStagePosition):
        raise ValueRefused(
            f"stage_position must be a stage position, not {type(value).__name__}."
        )
    return PreparedWrite.nothing()


def _prepare_alignment_area(experiment: Any, item: Any, value: Any) -> PreparedWrite:
    """Checked, and written nowhere, for the reason ``_prepare_features``
    gives: the task acquires its alignment reference with the area it is
    handed. A rectangle that is not a valid reduced area is refused here,
    before the run is released on it."""
    if not isinstance(value, FibsemRectangle):
        raise ValueRefused(
            f"alignment_area must be a rectangle, not {type(value).__name__}."
        )
    if not value.is_valid_reduced_area:
        raise ValueRefused(
            f"alignment_area {value.pretty_string} is not inside the frame."
        )
    return PreparedWrite.nothing()


register_value_writer("features", _prepare_features)
register_value_writer("stage_position", _prepare_stage_position)
register_value_writer("alignment_area", _prepare_alignment_area)


def _prepare_milling_config(experiment: Any, item: Any, value: Any) -> PreparedWrite:
    """Checked, and written nowhere: the task mills the config it is handed,
    and what it milled goes on the task's own config when the loop ends."""
    from fibsem.milling.tasks import FibsemMillingTaskConfig

    if not isinstance(value, FibsemMillingTaskConfig):
        raise ValueRefused(
            f"milling_config must be a milling config, not {type(value).__name__}."
        )
    return PreparedWrite.nothing()


def _prepare_spot_burn(experiment: Any, item: Any, value: Any) -> PreparedWrite:
    """Checked, and written nowhere, as ``milling_config`` is: every point in
    the frame (fractions, 0 to 1), a positive current, an exposure in range."""
    import math

    from fibsem.imaging.spot import SpotBurnSettings
    from fibsem.services.spot_burn import in_frame

    if not isinstance(value, SpotBurnSettings):
        raise ValueRefused(
            f"spot_burn must be spot-burn settings, not {type(value).__name__}."
        )
    for pt in value.coordinates:
        if not isinstance(pt, Point) or not in_frame(pt):
            raise ValueRefused(
                f"point {pt} is outside the frame: x and y are fractions, 0 to 1."
            )
    for key in ("milling_current", "exposure_time"):
        number = getattr(value, key)
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            or number <= 0
        ):
            raise ValueRefused(f"{key} is a positive number.")
    limits = SpotBurnSettings.params["exposure_time"].limits
    if not limits.min <= value.exposure_time <= limits.max:
        raise ValueRefused(
            f"exposure_time {value.exposure_time:g} s is outside "
            f"{limits.min:g} to {limits.max:g} s."
        )
    return PreparedWrite.nothing()


register_value_writer("milling_config", _prepare_milling_config)
register_value_writer("spot_burn", _prepare_spot_burn)


def _prepare_poi(experiment: Any, item: Any, value: Any) -> PreparedWrite:
    """The new point, then every pattern that follows it (the item's
    ``poi_sync_plan``), planned in full before any of it happens: a write that
    bypassed the sync left those patterns detached from the new point."""
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
        item.poi = deepcopy(value)  # the record keeps its own point
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


register_value_writer("poi", _prepare_poi)


# How a value is edited by naming fields rather than sending it whole: a
# decider changing one depth of a milling config sends ``{"stages.Polish
# 1.pattern.depth": 1e-6}``, not the config. ``patch(value, fields)`` returns
# the value with the fields set, a new one, or raises ``ValueRefused``.
_VALUE_PATCHERS: Dict[str, Callable[[Any, Dict[str, Any]], Any]] = {}


def register_value_patcher(
    name: str, patch: Callable[[Any, Dict[str, Any]], Any]
) -> None:
    _VALUE_PATCHERS[name] = patch


def patch_values(
    kind: str, values: Dict[str, Any], fields: Dict[str, Any]
) -> Dict[str, Any]:
    """``values`` with ``fields`` set on the one value of ``kind`` that takes
    a patch. Refused (``ValueRefused``) when the kind has no such value."""
    carried = PROPOSAL_KINDS[kind].values if kind in PROPOSAL_KINDS else ()
    patchable = [n for n in carried if n in _VALUE_PATCHERS]
    if len(patchable) != 1:
        raise ValueRefused(f"A {kind} proposal's values are not edited by field.")
    name = patchable[0]
    if name not in values:
        raise ValueRefused(f"No {name} to edit.")
    return {**values, name: _VALUE_PATCHERS[name](values[name], fields)}


def _patch_milling_config(config: Any, fields: Dict[str, Any]) -> Any:
    from fibsem.milling.tasks import (
        InvalidMillingEdit,
        patch_milling_config,
    )

    try:
        return patch_milling_config(config, fields)
    except InvalidMillingEdit as exc:
        raise ValueRefused(str(exc)) from None


register_value_patcher("milling_config", _patch_milling_config)


def compute_delta(proposed: Any, confirmed: Any) -> Any:
    """confirmed - proposed, for the value types a delta means something on.
    None where it does not."""
    if isinstance(proposed, Point) and isinstance(confirmed, Point):
        return Point(x=confirmed.x - proposed.x, y=confirmed.y - proposed.y)
    if isinstance(proposed, FibsemStagePosition) and isinstance(
        confirmed, FibsemStagePosition
    ):
        axes = (proposed.x, proposed.y, proposed.z, proposed.r, proposed.t)
        axes += (confirmed.x, confirmed.y, confirmed.z, confirmed.r, confirmed.t)
        if any(a is None for a in axes):
            return None
        return confirmed - proposed
    if isinstance(proposed, (int, float)) and isinstance(confirmed, (int, float)):
        return confirmed - proposed
    if isinstance(proposed, FibsemRectangle) and isinstance(confirmed, FibsemRectangle):
        # per edge, in fractions of the frame: how far it was dragged and by
        # how much it was resized, not one number for both
        return FibsemRectangle(
            left=confirmed.left - proposed.left,
            top=confirmed.top - proposed.top,
            width=confirmed.width - proposed.width,
            height=confirmed.height - proposed.height,
        )
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
class Decision:
    """One decider's answer. Appended, never edited: a reversal is a second
    decision beside the first, with its own actor and time."""

    outcome: DecisionOutcome
    actor: Actor
    values: Dict[str, Any] = field(default_factory=dict)  # confirmed values
    reason: str = ""  # required on Rejected
    # Aware, written as ISO 8601 with its offset; older records hold a
    # POSIX float, which reads as this machine's zone. None when the record has none.
    timestamp: Optional[datetime] = field(default_factory=now)
    # Where it was decided: "workflow" (in the window, or by the task itself),
    # "server" (an agent over the API), or "review" in records written before
    # the Review tab was removed. The actor says who; this says where.
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
    # What was decided, for a kind with actions (a loop's pass: run, edit,
    # skip, accept). Empty for a yes/no kind, and not saved then.
    action: str = ""
    # The suggestion this decision approved; empty when it approved none.
    suggestion_id: str = ""
    # The evidence the decider cited (`fibsem.workflows.evidence`), by id.
    evidence: str = ""

    def __post_init__(self) -> None:
        # a string from the file, the wire or a test is accepted and parsed
        self.actor = Actor.parse(self.actor)

    def to_dict(self) -> dict:
        doc = {
            "outcome": self.outcome.name,
            "actor": str(self.actor),
            "values": _encode_values(self.values),
            "reason": self.reason,
            "timestamp": to_iso(self.timestamp),
            "via": self.via,
            "task_id": self.task_id,
            "proposal_id": self.proposal_id,
        }
        if self.action:
            doc["action"] = self.action
        if self.suggestion_id:
            doc["suggestion_id"] = self.suggestion_id
        if self.evidence:
            doc["evidence"] = self.evidence
        return doc

    @classmethod
    def from_dict(cls, data: dict) -> "Decision":
        return cls(
            outcome=DecisionOutcome[data["outcome"]],
            actor=Actor.parse(data.get("actor", data.get("author", ""))),
            values=_decode_values(data.get("values", {})),
            reason=data.get("reason", ""),
            timestamp=to_datetime(data.get("timestamp")),
            via=data.get("via", ""),
            task_id=data.get("task_id", ""),
            proposal_id=data.get("proposal_id", ""),
            action=data.get("action", ""),
            suggestion_id=data.get("suggestion_id", ""),
            evidence=data.get("evidence", ""),
        )


@dataclass
class Suggestion:
    """What someone other than the decider (an agent, at Assist) puts forward
    on a pending proposal: the decision they would make, and why. Nothing is
    decided by it; the decider approves it, declines it, or decides otherwise.

    ``patch`` is kept as sent when the suggester edited fields of the proposed
    value (a milling config's depth); ``values`` when it sent whole values.
    ``evidence`` is the id of the evidence it cites; ``image``, the beam image
    (inside the experiment) its value is drawn on.
    """

    actor: Actor
    reason: str
    outcome: DecisionOutcome = DecisionOutcome.Confirmed
    values: Dict[str, Any] = field(default_factory=dict)
    patch: Dict[str, Any] = field(default_factory=dict)
    action: str = ""
    image: str = ""
    evidence: str = ""
    timestamp: Optional[datetime] = field(default_factory=now)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    declined: bool = False
    declined_reason: str = ""

    def __post_init__(self) -> None:
        self.actor = Actor.parse(self.actor)

    def values_for(self, proposal: "Proposal") -> Dict[str, Any]:
        """The values a decision approving this carries: ``values``, or the
        proposed ones with ``patch`` set on them."""
        if self.patch:
            return patch_values(
                proposal.kind, dict(self.values or proposal.values), self.patch
            )
        return dict(self.values)

    def to_dict(self) -> dict:
        doc = {
            "id": self.id,
            "actor": str(self.actor),
            "outcome": self.outcome.name,
            "reason": self.reason,
            "timestamp": to_iso(self.timestamp),
        }
        if self.values:
            doc["values"] = _encode_values(self.values)
        if self.patch:
            doc["patch"] = dict(self.patch)
        for name in ("action", "image", "evidence", "declined_reason"):
            if getattr(self, name):
                doc[name] = getattr(self, name)
        if self.declined:
            doc["declined"] = True
        return doc

    @classmethod
    def from_dict(cls, data: dict) -> "Suggestion":
        return cls(
            actor=Actor.parse(data.get("actor", "")),
            reason=data.get("reason", ""),
            outcome=DecisionOutcome[data.get("outcome", "Confirmed")],
            values=_decode_values(data.get("values", {})),
            patch=dict(data.get("patch", {})),
            action=data.get("action", ""),
            image=data.get("image", ""),
            evidence=data.get("evidence", ""),
            timestamp=to_datetime(data.get("timestamp")),
            id=data.get("id") or uuid.uuid4().hex,
            declined=bool(data.get("declined", False)),
            declined_reason=data.get("declined_reason", ""),
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
    provenance: Dict[str, Any] = field(default_factory=dict)
    decisions: List[Decision] = field(default_factory=list)
    # As Decision.timestamp. The id of a proposal saved before proposals
    # had one is derived from the value as stored, before this reads it.
    created_at: Optional[datetime] = field(default_factory=now)
    # Whether the task that made this is parked on it right now, waiting to be
    # told the answer -- an in-run question rather than a result left for
    # later. It is the one thing that lets a decision land on a
    # running task, so it is deliberately **not persisted**: a question only
    # exists while something is waiting on it, and a flag that survived a
    # reload would claim a waiter that is gone.
    asking: bool = field(default=False, compare=False, repr=False)
    # Whether the asker fills in the values of an "as it stands" confirmation
    # once it lands (``BaseTask.ask`` with ``decided``). Not persisted: the
    # record holds the decision once, with its values.
    filling_in: bool = field(default=False, compare=False, repr=False)
    # This proposal's own name, minted when it is made. What a decision names
    # (``Decision.proposal_id``), and what anything outside the record -- an
    # event, a responder waiting on an answer -- points at: ``(item, task)``
    # stops being unique the moment a task asks twice. Opaque: nothing reads
    # anything out of it.
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    # Who asked: the task that made it, an agent or a model. None for one that
    # nothing asked yet; the item's record stamps it when it is recorded.
    asked_by: Optional[Actor] = None
    # The latest suggestion on it, while pending; a new one replaces it.
    suggestion: Optional[Suggestion] = None

    def asked_by_task(self, task_name: str) -> bool:
        """Whether *task_name* asked this: what a task's view of an item reads."""
        return self.asked_by == Actor.task(task_name)

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
    def applied(self) -> Optional[Decision]:
        """The decision whose values were written through: the latest
        Confirmed one that carried values."""
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
            "provenance": dict(self.provenance),
            "decisions": [d.to_dict() for d in self.decisions],
            "created_at": to_iso(self.created_at),
            "id": self.id,
            "asked_by": str(self.asked_by) if self.asked_by is not None else None,
            **(
                {"suggestion": self.suggestion.to_dict()}
                if self.suggestion is not None
                else {}
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Proposal":
        return cls(
            kind=data["kind"],
            values=_decode_values(data.get("values", {})),
            provenance=dict(data.get("provenance", {})),
            decisions=[Decision.from_dict(d) for d in data.get("decisions", [])],
            created_at=to_datetime(data.get("created_at")),
            id=data.get("id") or _id_for_a_record_without_one(data),
            asked_by=Actor.parse(data["asked_by"]) if data.get("asked_by") else None,
            suggestion=Suggestion.from_dict(data["suggestion"])
            if data.get("suggestion")
            else None,
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


def record(proposals: List[Proposal], proposal: Proposal) -> Proposal:
    """Append ``proposal`` to an item's list, and return it.

    The list is the record of everything proposed on the item, oldest first;
    a re-run leaves the old proposal and its decision on the record and puts
    the new one after it. The previous proposal of the same kind and asker is
    dropped first if nobody decided it: there is nothing to keep, and two
    pending proposals of one kind would be two asks where only one was made.
    """
    for i in range(len(proposals) - 1, -1, -1):
        if (
            proposals[i].kind == proposal.kind
            and proposals[i].asked_by == proposal.asked_by
        ):
            if proposals[i].pending:
                del proposals[i]
            break
    proposals.append(proposal)
    return proposal


def current_proposal(
    proposals: Optional[List[Proposal]], kind: Optional[str] = None
) -> Optional[Proposal]:
    """The last proposal for the task, or the last of ``kind``."""
    if not proposals:
        return None
    if kind is None:
        return proposals[-1]
    for p in reversed(proposals):
        if p.kind == kind:
            return p
    return None


def current_proposals(proposals: Optional[List[Proposal]]) -> List[Proposal]:
    """The proposals a decision, the gate and the inbox act on, in the order
    they were made: the last run's proposals of each kind. A task's run leaves
    its result after the questions it asked, and a question is not replaced
    by a result -- they are different kinds -- so both are current; a run that
    asks the same kind twice (Setup confirms the tilt, then the position)
    asked two questions, and both stand. What a re-run replaces is the earlier
    run's proposals of the kinds it makes again: on the record for their
    decisions and their deltas only. So is a question the run withdrew and
    asked again. A proposal with no run stamped on it (a record from before
    runs were) is replaced by any later one of its kind.
    """
    if not proposals:
        return []
    latest: Dict[str, Proposal] = {}
    for p in reversed(proposals):
        latest.setdefault(p.kind, p)
    return [
        p
        for p in proposals
        if p is latest[p.kind]
        or (p.task_id and p.task_id == latest[p.kind].task_id and not p.withdrawn)
    ]


def proposals_to_dict(proposals: List[Proposal]) -> List[dict]:
    return [p.to_dict() for p in proposals]


def proposals_from_dict(
    data: Union[List[dict], Dict[str, list], None],
) -> List[Proposal]:
    """An item's proposals as saved: a list, each saying who asked it. A file
    saved before then keyed them by the task that asked, which is read as
    that task's asking."""
    if isinstance(data, dict):
        out = []
        for name, ps in data.items():
            for p in ps:
                proposal = Proposal.from_dict(p)
                if proposal.asked_by is None:
                    proposal.asked_by = Actor.task(name)
                out.append(proposal)
        return out
    return [Proposal.from_dict(p) for p in data or []]


@dataclass
class DecisionResult:
    """What ``Experiment.decide`` did. ``applied`` False carries the reason;
    ``running`` True means the consumer had already started and the answer is
    to stop it, not to decide."""

    applied: bool
    reason: str = ""
    running: bool = False
    # Why a refusal, for a client to act on: "missing_field" (the decision
    # names no run), "stale_proposal" (the task re-ran since it was shown),
    # "invalid_value" (the values cannot be confirmed as given). Empty when
    # applied, and for the other refusals (running, nothing pending).
    error_type: str = ""
    synced_tasks: List[str] = field(default_factory=list)
    # Applied, but the experiment could not be saved after it: the decision is
    # in memory and not yet on disk. Empty when it was saved.
    save_error: str = ""

    def to_dict(self) -> dict:
        """``applied``, and of the rest only what is set."""
        doc = {"applied": self.applied}
        for name in ("reason", "running", "error_type", "synced_tasks", "save_error"):
            value = getattr(self, name)
            if value:
                doc[name] = list(value) if name == "synced_tasks" else value
        return doc


@dataclass(frozen=True)
class _Words:
    run: str
    again: str
    noun: str
    without: str
    failed: str
    off: str = "Press Continue when it is right."


_WORDS = {
    MILLING: _Words(
        run="Run Milling",
        again="Mill Again",
        noun="mill",
        without="without milling",
        failed="Milling failed",
        off="Press Continue when the config is right.",
    ),
    SPOT_BURN: _Words(
        run="Run Spot Burn",
        again="Burn Again",
        noun="burn",
        without="without burning",
        failed="Spot burn failed",
    ),
}

LOOP_KINDS = tuple(_WORDS)


@dataclass(frozen=True)
class LoopPrompt:
    """The prompt bar for one pass: its text, its buttons, and the action each
    button decides (None: no such button)."""

    message: str
    positive: str
    negative: Optional[str]
    yes: str
    no: Optional[str]


def loop_prompt(proposal: Proposal) -> LoopPrompt:
    """The prompt bar for ``proposal``, a pass of a loop."""
    words = _WORDS[proposal.kind]
    allowed = PROPOSAL_KINDS[proposal.kind].allowed_actions(proposal)
    passes = int(proposal.provenance.get("passes") or 0)
    message = str(proposal.provenance.get("message") or "")
    if ACTION_RUN not in allowed:
        prompt = LoopPrompt(
            f"{message} {words.off}", "Continue", None, ACTION_ACCEPT, None
        )
    elif passes == 0:
        prompt = LoopPrompt(
            f"{message} Press {words.run} to {words.noun}, or Skip to go on "
            f"{words.without}.",
            words.run,
            "Skip",
            ACTION_RUN,
            ACTION_SKIP,
        )
    else:
        prompt = LoopPrompt(
            f"{message} Press {words.again} to {words.noun} again, or Accept to go on.",
            words.again,
            "Accept",
            ACTION_RUN,
            ACTION_ACCEPT,
        )
    error = proposal.provenance.get("last_error")
    if error:
        prompt = replace(prompt, message=f"{words.failed}: {error}\n\n{prompt.message}")
    return prompt
