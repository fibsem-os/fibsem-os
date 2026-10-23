"""What the progress reports of a long-running job share: a status that knows
whether, and how, the job ended.

Milling, fluorescence acquisition, tiled acquisition and spot burn each report
with their own record and their own status enum (each consumer reads its own
fields). Each status enum subclasses `ProgressStatus` and names its ending
members once with `ends`; `is_terminal` and `outcome` are then answered here,
never restated as a membership tuple at a consumer.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional

__all__ = ["DONE", "CANCELLED", "FAILED", "ProgressStatus", "ends"]

DONE = "done"
CANCELLED = "cancelled"
FAILED = "failed"

_OUTCOMES: Dict["ProgressStatus", str] = {}


class ProgressStatus(str, Enum):
    """A job's status. A `str` mixin, so a member compares equal to its value
    and reads as itself in a log line."""

    @property
    def outcome(self) -> Optional[str]:
        """How the job ended (`DONE`, `CANCELLED` or `FAILED`), or None while
        it runs."""
        return _OUTCOMES.get(self)

    @property
    def is_terminal(self) -> bool:
        """Whether this ends the job, however it ended."""
        return self in _OUTCOMES


def ends(
    done: ProgressStatus,
    cancelled: Optional[ProgressStatus] = None,
    failed: Optional[ProgressStatus] = None,
) -> None:
    """Name the members of a status enum that end its job."""
    for status, outcome in ((done, DONE), (cancelled, CANCELLED), (failed, FAILED)):
        if status is not None:
            _OUTCOMES[status] = outcome
