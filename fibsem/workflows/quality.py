"""A person's verdict on an item: good, rework, failed, or not yet looked at.

Set by a person, or a reviewer acting for one, and never by a task. One type for
every kind of item.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, Tuple

from psygnal import evented

from fibsem.codec import NamedEnum, Timestamp, decode, encode
from fibsem.codec.upgrades import records as upgrade
from fibsem.util.timestamps import now, to_datetime


class Verdict(NamedEnum, Enum):
    """A judgement about a lamella or a grid, made by a person (or a reviewer
    acting for one) and never by a task.

    ``UNASSESSED`` and ``GOOD`` are different answers: nobody has looked, versus
    somebody looked and it was fine.
    """

    UNASSESSED = auto()
    GOOD = auto()
    REWORK = auto()
    FAILED = auto()

    # Aliases for the names the old enums used. ``Verdict["NONE"]`` resolves,
    # ``Verdict.NONE is Verdict.UNASSESSED``, and iteration skips them.
    NONE = UNASSESSED
    FAILURE = FAILED
    POOR = FAILED


@evented
@dataclass
class QualityRecord:
    """The current human verdict on an item, attributed.

    One type for lamellae and grids, replacing ``DefectState`` (lamella) and the
    bare ``GridQuality`` enum (grid), which were the same idea under two names
    with neither recording who set it. Holds the *current* verdict only; the
    full trail of what was proposed and decided lives on the item's proposals,
    and ``decision_id`` points at the decision that set this verdict when a
    review did.
    """

    verdict: Verdict = field(default=Verdict.UNASSESSED)
    actor: str = ""  # "operator:<name>" | "agent:<model>"; "" when unrecorded
    reason: str = ""
    at_task: str = ""  # the task the item was judged at
    # When the verdict was set: aware, written as ISO 8601 with its offset;
    # older files hold a POSIX float, which reads as this machine's zone.
    updated_at: Optional[Timestamp] = None
    decision_id: Optional[Tuple[str, str]] = None  # (item_id, task_name)

    def __post_init__(self) -> None:
        # a caller that still passes a POSIX float gets it read as one
        self.updated_at = to_datetime(self.updated_at)

    # -- names the old DefectState API used; kept so callers migrate one at a time
    @property
    def state(self) -> Verdict:
        return self.verdict

    @state.setter
    def state(self, value: Verdict) -> None:
        self.verdict = value

    @property
    def description(self) -> str:
        return self.reason

    @description.setter
    def description(self, value: str) -> None:
        self.reason = value

    @property
    def last_completed_task(self) -> str:
        return self.at_task

    @last_completed_task.setter
    def last_completed_task(self, value: str) -> None:
        self.at_task = value

    def to_dict(self) -> dict:
        return encode(self)

    @classmethod
    def from_dict(cls, data: Any) -> "QualityRecord":
        """Read a verdict saved by any version (`fibsem.codec.upgrades.records`)."""
        return decode(cls, upgrade.quality(data))

    def clear(self):
        self.verdict = Verdict.UNASSESSED
        self.actor = ""
        self.reason = ""
        self.at_task = ""
        self.updated_at = None
        self.decision_id = None

    def set_defect(
        self,
        description: str = "",
        state: Verdict = Verdict.FAILED,
        actor: str = "",
    ):
        self.verdict = state
        self.reason = description
        self.actor = str(actor)  # an Actor or its kind:name form
        self.updated_at = now()


_quality_record_init = QualityRecord.__init__


def _quality_record_init_with_old_names(
    self,
    *args,
    state: Optional[Verdict] = None,
    description: Optional[str] = None,
    last_completed_task: Optional[str] = None,
    **kwargs,
):
    """``DefectState(state=..., description=..., last_completed_task=...)`` keeps
    constructing. The dataclass ``__init__`` only knows the new field names, so
    the old keywords are mapped here; a caller passing both forms gets the new
    one."""
    if state is not None:
        kwargs.setdefault("verdict", state)
    if description is not None:
        kwargs.setdefault("reason", description)
    if last_completed_task is not None:
        kwargs.setdefault("at_task", last_completed_task)
    _quality_record_init(self, *args, **kwargs)


QualityRecord.__init__ = _quality_record_init_with_old_names  # type: ignore[method-assign]
