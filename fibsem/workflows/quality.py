"""A person's verdict on an item: good, rework, failed, or not yet looked at.

Set by a person, or a reviewer acting for one, and never by a task. One type for
every kind of item.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, Tuple, Union

from psygnal import evented

from fibsem.acting import Actor
from fibsem.util.timestamps import now, to_datetime, to_iso


class Verdict(Enum):
    """A judgement about a lamella or a grid, made by a person (or a reviewer
    acting for one) and never by a task.

    ``UNASSESSED`` and ``GOOD`` are different answers: nobody has looked, versus
    somebody looked and it was fine.
    """

    UNASSESSED = auto()
    GOOD = auto()
    REWORK = auto()
    FAILED = auto()


def _actor_text(text: str) -> str:
    """A stored actor in today's ``kind:name`` form; blank stays blank."""
    return str(Actor.parse(text)) if text else ""


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
    # When the verdict was set: aware, written as ISO 8601 with its offset
    # (FIB-1197); older files hold a POSIX float, which reads as this machine's zone.
    updated_at: Optional[datetime] = None
    decision_id: Optional[Tuple[str, str]] = None  # (item_id, task_name)

    def __post_init__(self) -> None:
        # a caller that still passes a POSIX float gets it read as one
        self.updated_at = to_datetime(self.updated_at)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.name,
            "actor": self.actor,
            "reason": self.reason,
            "at_task": self.at_task,
            "updated_at": to_iso(self.updated_at),
            "decision_id": list(self.decision_id) if self.decision_id else None,
        }

    @classmethod
    def from_dict(cls, data: Union[dict, str, None]) -> "QualityRecord":
        """Read the current shape, the older ``state`` / ``description`` /
        ``last_completed_task`` shape, the older still ``has_defect`` /
        ``requires_rework`` bools, and the bare name string ``GridRecord`` used to
        write. Unknown names read as ``UNASSESSED``."""
        if not data:
            return cls()
        if isinstance(data, str):
            return cls(verdict=_verdict_by_name(data))
        if "has_defect" in data:
            if data.get("has_defect"):
                verdict = (
                    Verdict.REWORK if data.get("requires_rework") else Verdict.FAILED
                )
            else:
                verdict = Verdict.UNASSESSED
            return cls(
                verdict=verdict,
                reason=data.get("description", ""),
                updated_at=to_datetime(data.get("updated_at")),
            )
        decision_id = data.get("decision_id")
        return cls(
            verdict=_verdict_by_name(data.get("verdict", data.get("state", ""))),
            actor=_actor_text(data.get("actor", data.get("author", ""))),
            reason=data.get("reason", data.get("description", "")),
            at_task=data.get("at_task", data.get("last_completed_task", "")),
            updated_at=to_datetime(data.get("updated_at")),
            decision_id=tuple(decision_id) if decision_id else None,
        )

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


# The names older files wrote for a verdict.
_OLD_VERDICT_NAMES = {
    "NONE": Verdict.UNASSESSED,
    "FAILURE": Verdict.FAILED,
    "POOR": Verdict.FAILED,
}


def _verdict_by_name(name: str) -> Verdict:
    try:
        return Verdict[name]
    except KeyError:
        return _OLD_VERDICT_NAMES.get(name, Verdict.UNASSESSED)
