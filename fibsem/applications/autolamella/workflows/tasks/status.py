"""The typed payload carried on ``workflow_update_signal``'s ``status`` key.

Only the *nested* record, deliberately. The envelope around it -- ``{"msg": ...,
"status": ...}`` -- is untouched, because most of that signal is not a signal at all:
ten of its thirteen emits are blocking calls built from a shared mutable bool and a
polling loop, and untangling those is a threading redesign (FIB-826). This record is
independent of that work in both directions and can land before or after it.

Unlike the other progress signals, there was no vocabulary to design here. ``status``
has always been a real ``AutoLamellaTaskStatus`` on the wire; what was missing was a
declared shape around it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus

if TYPE_CHECKING:
    from fibsem.applications.autolamella.workflows.tasks.queue import WorkItem


@dataclass(frozen=True, eq=False)
class WorkflowStatusUpdate:
    """One report on a task's lifecycle, from ``TaskManager._emit_status``.

    ``eq=False`` is load-bearing, for a different reason than ``TiledProgress``'s.
    Three fields hold lists, and ``WorkItem`` is a *mutable* dataclass, so it is
    unhashable. ``frozen=True`` with the default ``eq=True`` generates a ``__hash__``
    over every field, which raises ``TypeError: unhashable type: 'list'`` the moment
    anything puts a report in a set or uses it as a key. There is no numpy here --
    if you came looking for the tiled contract's reason, this is not it.
    """

    task_name: str = "Unknown Task"
    item_name: str = "Unknown Lamella"
    status: AutoLamellaTaskStatus = AutoLamellaTaskStatus.NotStarted
    timestamp: Optional[float] = None
    error_message: Optional[str] = None
    task_duration: Optional[float] = None
    skip_reason: Optional[str] = None
    # Position in the *live* queue, and ALREADY 1-based: the producer writes
    # `position + 1`. Deliberately unlike `milling_progress_signal`'s 0-based
    # `current_stage`, which seven consumers each increment. There is nothing to
    # correct here, so do not port a `display_*` property over -- it would
    # double-count.
    queue_position: Optional[int] = None
    queue_total: int = 0
    # A snapshot of copies, not the live queue: `Queue.items` returns
    # `[copy.copy(i) for i in self._items]` under a lock. Safe to read from the GUI
    # thread, and no further defensive copying is wanted.
    queue_items: List["WorkItem"] = field(default_factory=list)
    # The plan this run was launched with. Informational only -- the live queue may
    # since have diverged from it. No in-repo production consumer reads either of
    # these; they are kept for the same reason `lamella_name` is, and should be
    # dropped together with it.
    task_names: List[str] = field(default_factory=list)
    lamella_names: List[str] = field(default_factory=list)

    @property
    def lamella_name(self) -> str:
        """Deprecated alias for :attr:`item_name`.

        A property rather than a field so there is one source of truth and the two
        cannot drift. Drop it alongside the ``HookContext`` shims after v0.6.
        """
        return self.item_name

    @classmethod
    def from_payload(
        cls, payload: Union["WorkflowStatusUpdate", Dict[str, Any], None]
    ) -> "WorkflowStatusUpdate":
        """Accept either the typed report or the dict the signal still carries.

        Total by construction, and that is the point rather than politeness. This
        runs inside a queued Qt slot, where PyQt5 turns any escaping exception into
        ``qFatal`` -- the process aborts mid-run and the abort reaches no logfile
        (FIB-329). This signal has already killed the app that way once, over a
        missing ``msg`` key. So every field has a default and nothing here indexes.

        Deleted once the producer flips; until then the dict is the live path.
        """
        if isinstance(payload, cls):
            return payload
        if not payload:
            return cls()

        status = payload.get("status")
        if not isinstance(status, AutoLamellaTaskStatus):
            # Not a lie worth telling loudly: `NotStarted` fires no consumer branch,
            # so an unrecognised value renders as "nothing in particular happened"
            # rather than crashing or picking a wrong outcome.
            status = AutoLamellaTaskStatus.NotStarted

        return cls(
            task_name=payload.get("task_name") or "Unknown Task",
            item_name=payload.get("item_name") or "Unknown Lamella",
            status=status,
            timestamp=payload.get("timestamp"),
            error_message=payload.get("error_message"),
            task_duration=payload.get("task_duration"),
            skip_reason=payload.get("skip_reason"),
            queue_position=payload.get("queue_position"),
            queue_total=payload.get("queue_total") or 0,
            queue_items=list(payload.get("queue_items") or []),
            task_names=list(payload.get("task_names") or []),
            lamella_names=list(payload.get("lamella_names") or []),
        )
