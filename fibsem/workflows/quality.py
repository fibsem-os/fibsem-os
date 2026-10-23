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


@evented
@dataclass
class QualityRecord:
    """The current human verdict on an item, attributed.

    One type for lamellae and grids. Holds the *current* verdict only; the
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
