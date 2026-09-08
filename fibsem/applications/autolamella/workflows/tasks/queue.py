"""Thread-safe task queue for autolamella workflow execution.

The queue is ordered ``[history..., active?, pending...]``: items are consumed
in order, so everything ahead of the active item has already run. All mutation
is confined to the pending items — history and the active item are immutable.

Every mutation is anchored to a :attr:`WorkItem.id` rather than to a position.
The UI thread is always working from a stale snapshot (the worker may consume
an item at any moment), so a position means something different by the time it
arrives. Anchoring to identity makes that harmless: a move whose anchor has
since been consumed is refused rather than silently misapplied.

Reordering permutes the pending items *within their existing slots*, so no
operation except :meth:`TaskQueue.remove` can change what is in the queue — and
``remove`` marks rather than deletes, so the run summary still records it.
"""

import copy
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional, Tuple
from uuid import uuid4

from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus


@dataclass
class WorkItem:
    """A single (item, task) work unit.

    The item is whatever the run is over: a lamella on the lamella workflow, a
    grid on the grid workflow. ``item_name`` is the vocabulary the hooks, the
    status payload and the timeline already speak.
    """

    item_name: str
    task_name: str
    status: AutoLamellaTaskStatus = AutoLamellaTaskStatus.NotStarted
    id: str = field(default_factory=lambda: str(uuid4()))

    @property
    def lamella_name(self) -> str:
        """Deprecated alias for :attr:`item_name`; drop with the v0.6 shims."""
        return self.item_name

    @property
    def is_pending(self) -> bool:
        """Whether this item is still waiting to run — the only mutable state."""
        return self.status is AutoLamellaTaskStatus.NotStarted


class QueueOp(Enum):
    """Outcome of a queue mutation."""

    OK = auto()
    NOT_FOUND = auto()  # no item with that id
    NOT_PENDING = auto()  # item is active, already run, or removed
    BAD_ANCHOR = auto()  # the target item is missing or not pending
    NO_OP = auto()  # already in the requested position


@dataclass(frozen=True)
class QueueResult:
    """Result of a mutation, with the queue version it produced.

    The distinct :class:`QueueOp` values exist so the UI can explain a refusal
    ("already running") instead of failing silently.
    """

    op: QueueOp
    version: int

    @property
    def ok(self) -> bool:
        return self.op is QueueOp.OK


class TaskQueue:
    """Thread-safe mutable queue of work items.

    The worker thread calls :meth:`next` / :meth:`mark_done` to consume items.
    The UI thread calls :meth:`add`, :meth:`remove` and the ``move_*`` methods
    to mutate the pending items.
    """

    def __init__(self):
        self._items: List[WorkItem] = []
        # Re-entrant: next() holds it while it asks the skip predicate, and the
        # predicate may ask the queue (has_pending_pair) on the same thread.
        self._lock = threading.RLock()
        self._active: Optional[WorkItem] = None
        self._version = 0
        # Original matrix dimensions for status dict compat
        self._task_names: List[str] = []
        self._item_names: List[str] = []

    # --- Build ---

    def build_from_matrix(
        self, task_names: List[str], item_names: List[str], item_outer: bool = False
    ) -> List[WorkItem]:
        """Populate queue from a task x item matrix.

        Task-outer by default (every item's Trench, then every item's Undercut),
        which is the lamella workflow's order. ``item_outer`` runs every task on
        one item before moving to the next -- the grid workflow's order, where
        an item costs a grid exchange to reach and is exchanged once.
        """
        with self._lock:
            self._task_names = list(task_names)
            self._item_names = list(item_names)
            if item_outer:
                self._items = [
                    WorkItem(item_name=name, task_name=tn)
                    for name in item_names
                    for tn in task_names
                ]
            else:
                self._items = [
                    WorkItem(item_name=name, task_name=tn)
                    for tn in task_names
                    for name in item_names
                ]
            self._active = None
            self._version += 1
            return [copy.copy(i) for i in self._items]

    def build_from_pairs(
        self,
        pairs: List[Tuple[str, str]],
        task_names: Optional[List[str]] = None,
        item_names: Optional[List[str]] = None,
    ) -> List[WorkItem]:
        """Populate the queue from an explicit ``(item_name, task_name)`` plan.

        For a run whose steps are not a plain matrix -- the grid workflow puts a
        load step ahead of each grid's tasks. ``task_names`` and ``item_names``
        are the launch plan the status payload reports; by default they are read
        off the pairs in first-seen order.
        """
        with self._lock:
            self._task_names = (
                list(task_names)
                if task_names is not None
                else list(dict.fromkeys(t for _, t in pairs))
            )
            self._item_names = (
                list(item_names)
                if item_names is not None
                else list(dict.fromkeys(n for n, _ in pairs))
            )
            self._items = [WorkItem(item_name=n, task_name=t) for n, t in pairs]
            self._active = None
            self._version += 1
            return [copy.copy(i) for i in self._items]

    # --- Internal helpers (caller must hold self._lock) ---

    def _find(self, item_id: str) -> Optional[WorkItem]:
        for item in self._items:
            if item.id == item_id:
                return item
        return None

    def _pending(self) -> List[WorkItem]:
        return [i for i in self._items if i.is_pending]

    def _pending_start(self) -> int:
        """Index of the first pending item; ``len(items)`` if there are none.

        Insert positions clamp to this so nothing can be placed ahead of the
        active item or into history.
        """
        for i, item in enumerate(self._items):
            if item.is_pending:
                return i
        return len(self._items)

    def _reposition(self, new_order: List[WorkItem]) -> QueueOp:
        """Write a permutation of the pending items back into the pending slots.

        Non-pending items (history, active, removed) keep their exact positions,
        so this can neither move nor lose them. ``new_order`` must be a
        permutation of the current pending items.
        """
        slots = [i for i, item in enumerate(self._items) if item.is_pending]
        if [self._items[s] for s in slots] == new_order:
            return QueueOp.NO_OP
        for slot, item in zip(slots, new_order):
            self._items[slot] = item
        self._version += 1
        return QueueOp.OK

    def _move(self, item_id: str, to_index: int) -> QueueResult:
        """Move a pending item to ``to_index`` within the pending sequence."""
        item = self._find(item_id)
        if item is None:
            return QueueResult(QueueOp.NOT_FOUND, self._version)
        if not item.is_pending:
            return QueueResult(QueueOp.NOT_PENDING, self._version)

        order = self._pending()
        order.remove(item)
        order.insert(max(0, min(to_index, len(order))), item)
        return QueueResult(self._reposition(order), self._version)

    def _anchor_index(self, target_id: str) -> Optional[int]:
        """Position of ``target_id`` within the pending sequence, if pending."""
        target = self._find(target_id)
        if target is None or not target.is_pending:
            return None
        return self._pending().index(target)

    # --- Mutation (all thread-safe) ---

    def add(
        self,
        item_name: str,
        task_name: str,
        *,
        before: Optional[str] = None,
        after: Optional[str] = None,
        front: bool = False,
    ) -> Optional[WorkItem]:
        """Add a work item, anchored to an existing item or to the queue ends.

        Position is one of: ``before``/``after`` an existing *pending* item,
        ``front=True`` (run next), or — by default — the end of the queue.

        Returns the new item, or ``None`` if an anchor was given but is missing
        or no longer pending. Adding a duplicate of an already-completed pair is
        allowed: that is how a task is re-run.
        """
        item = WorkItem(item_name=item_name, task_name=task_name)
        with self._lock:
            if before is not None or after is not None:
                anchor_id = before if before is not None else after
                anchor = self._find(anchor_id)  # type: ignore[arg-type]
                if anchor is None or not anchor.is_pending:
                    logging.warning(
                        f"Cannot add {task_name} for {item_name}: anchor "
                        f"{anchor_id} is missing or no longer pending."
                    )
                    return None
                pos = self._items.index(anchor)
                self._items.insert(pos if before is not None else pos + 1, item)
            elif front:
                self._items.insert(self._pending_start(), item)
            else:
                self._items.append(item)
            self._version += 1
        return item

    def remove(self, item_id: str) -> QueueResult:
        """Mark a pending item as Removed so it will not run.

        The item is kept in the queue rather than deleted: it stays in the run
        summary (see ``TaskManager.build_run_summary_dataframe``), the timeline
        can show it struck through, and nothing is ever unlinked.
        """
        with self._lock:
            item = self._find(item_id)
            if item is None:
                return QueueResult(QueueOp.NOT_FOUND, self._version)
            if not item.is_pending:
                return QueueResult(QueueOp.NOT_PENDING, self._version)
            item.status = AutoLamellaTaskStatus.Removed
            self._version += 1
            return QueueResult(QueueOp.OK, self._version)

    def move_before(self, item_id: str, target_id: str) -> QueueResult:
        """Move a pending item to just before another pending item."""
        with self._lock:
            index = self._anchor_index(target_id)
            if index is None:
                return QueueResult(QueueOp.BAD_ANCHOR, self._version)
            current = self._anchor_index(item_id)
            # removing the item first shifts a later anchor down by one
            if current is not None and current < index:
                index -= 1
            return self._move(item_id, index)

    def move_after(self, item_id: str, target_id: str) -> QueueResult:
        """Move a pending item to just after another pending item."""
        with self._lock:
            index = self._anchor_index(target_id)
            if index is None:
                return QueueResult(QueueOp.BAD_ANCHOR, self._version)
            current = self._anchor_index(item_id)
            if current is not None and current < index:
                index -= 1
            return self._move(item_id, index + 1)

    def move_to_front(self, item_id: str) -> QueueResult:
        """Move a pending item to the front of the queue — i.e. run it next."""
        with self._lock:
            return self._move(item_id, 0)

    def move_to_back(self, item_id: str) -> QueueResult:
        """Move a pending item to the end of the queue."""
        with self._lock:
            return self._move(item_id, len(self._items))

    def nudge(self, item_id: str, delta: int) -> QueueResult:
        """Move a pending item ``delta`` places within the pending sequence.

        Use -1 / +1 for move-up / move-down. Clamps at both ends and reports
        ``NO_OP`` there rather than wrapping. Removed items are skipped over,
        so a nudge always moves past one *runnable* neighbour.
        """
        with self._lock:
            current = self._anchor_index(item_id)
            if current is None:
                item = self._find(item_id)
                op = QueueOp.NOT_FOUND if item is None else QueueOp.NOT_PENDING
                return QueueResult(op, self._version)
            return self._move(item_id, current + delta)

    def reorder_pending(self, item_ids: List[str]) -> QueueResult:
        """Pull the named pending items to the front, in the order given.

        Prefix semantics, never replacement: any pending item not named keeps
        its relative order and follows. A caller passing a filtered or stale
        subset therefore reorders what it knows about and leaves the rest
        alone, instead of destroying it.
        """
        with self._lock:
            by_id = {i.id: i for i in self._pending()}
            named = [by_id[i] for i in dict.fromkeys(item_ids) if i in by_id]
            rest = [i for i in self._pending() if i not in named]
            return QueueResult(self._reposition(named + rest), self._version)

    # --- Iteration (called by worker thread) ---

    def next(
        self, skip: Optional[Callable[[WorkItem], bool]] = None
    ) -> Optional[WorkItem]:
        """Return the next pending item, mark it InProgress, or None if none.

        ``skip`` passes over a pending item *for this call only*: it stays
        pending and is offered again on the next call. That is how work that
        cannot run yet -- its prerequisite is still queued, or its proposal is
        awaiting a decision -- is deferred rather than retired. Skipping keeps
        the queue's order; nothing is reordered around a deferred item.

        None means nothing is runnable now, which is not the same as nothing
        is left: ask ``is_empty`` to tell "done" from "everything left is
        deferred".
        """
        with self._lock:
            for item in self._items:
                if item.is_pending and not (skip is not None and skip(item)):
                    item.status = AutoLamellaTaskStatus.InProgress
                    self._active = item
                    self._version += 1
                    return item
            self._active = None
            return None

    def has_pending_pair(self, item_name: str, task_name: str) -> bool:
        """Whether a (item, task) pair is still waiting to run."""
        with self._lock:
            return any(
                i.is_pending and i.item_name == item_name and i.task_name == task_name
                for i in self._items
            )

    def mark_done(self, item: WorkItem, status: AutoLamellaTaskStatus) -> None:
        """Mark item as completed/failed/skipped.

        Looked up by id, so it also works with a copy taken from ``items``.
        """
        with self._lock:
            target = self._find(item.id)
            if target is None:
                return
            target.status = status
            if self._active is not None and self._active.id == target.id:
                self._active = None
            self._version += 1

    # --- Query ---

    @property
    def version(self) -> int:
        """Bumped on every change. Lets a consumer detect a stale snapshot."""
        with self._lock:
            return self._version

    @property
    def items(self) -> List[WorkItem]:
        """Return a copy of every item — safe to read from another thread."""
        with self._lock:
            return [copy.copy(i) for i in self._items]

    @property
    def counts(self) -> Tuple[int, int]:
        """(pending, total), without copying every item just to count them.

        `items` and `pending` both build a new list, which is the wrong tool when the
        caller only wants a length -- and hooks ask on every event.
        """
        with self._lock:
            return sum(1 for i in self._items if i.is_pending), len(self._items)

    @property
    def pending(self) -> List[WorkItem]:
        """Return a copy of the items still waiting to run."""
        with self._lock:
            return [copy.copy(i) for i in self._items if i.is_pending]

    @property
    def active(self) -> Optional[WorkItem]:
        with self._lock:
            return copy.copy(self._active) if self._active is not None else None

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return not any(i.is_pending for i in self._items)

    @property
    def task_names(self) -> List[str]:
        """Original task names list (for status dict compat)."""
        return list(self._task_names)

    @property
    def item_names(self) -> List[str]:
        """Original item names list (for status dict compat)."""
        return list(self._item_names)

    @property
    def lamella_names(self) -> List[str]:
        """Deprecated alias for :attr:`item_names`; drop with the v0.6 shims."""
        return self.item_names
