"""Thread-safe task queue for autolamella workflow execution."""

import copy
import threading
from dataclasses import dataclass, field
from typing import List, Optional
from uuid import uuid4

from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus


@dataclass
class WorkItem:
    """A single (lamella, task) work unit."""
    lamella_name: str
    task_name: str
    status: AutoLamellaTaskStatus = AutoLamellaTaskStatus.NotStarted
    id: str = field(default_factory=lambda: str(uuid4()))


class TaskQueue:
    """Thread-safe mutable queue of work items.

    Worker thread calls next() to consume items.
    UI thread calls add/remove/reorder to mutate the pending portion.
    """

    def __init__(self):
        self._items: List[WorkItem] = []
        self._lock = threading.Lock()
        self._active: Optional[WorkItem] = None
        # Original matrix dimensions for status dict compat
        self._task_names: List[str] = []
        self._lamella_names: List[str] = []

    # --- Build ---

    def build_from_matrix(self, task_names: List[str],
                          lamella_names: List[str]) -> List[WorkItem]:
        """Populate queue from task x lamella matrix (task-outer, lamella-inner)."""
        with self._lock:
            self._task_names = list(task_names)
            self._lamella_names = list(lamella_names)
            self._items = [
                WorkItem(lamella_name=ln, task_name=tn)
                for tn in task_names
                for ln in lamella_names
            ]
            self._active = None
            return list(self._items)

    # --- Mutation (all thread-safe) ---

    def add(self, lamella_name: str, task_name: str,
            index: Optional[int] = None) -> WorkItem:
        """Add a work item at position (default: end of pending items)."""
        item = WorkItem(lamella_name=lamella_name, task_name=task_name)
        with self._lock:
            pending = [i for i in self._items
                       if i.status == AutoLamellaTaskStatus.NotStarted]
            if index is None or index >= len(pending):
                self._items.append(item)
            else:
                target = pending[index]
                pos = self._items.index(target)
                self._items.insert(pos, item)
        return item

    def remove(self, item_id: str) -> bool:
        """Remove a pending item by ID. Returns False if item is active/completed."""
        with self._lock:
            for i, item in enumerate(self._items):
                if item.id == item_id:
                    if item is self._active:
                        return False
                    if item.status != AutoLamellaTaskStatus.NotStarted:
                        return False
                    self._items.pop(i)
                    return True
        return False

    def reorder(self, item_ids: List[str]) -> None:
        """Reorder pending items to match the given ID order.
        Non-pending items stay in place."""
        with self._lock:
            id_to_item = {i.id: i for i in self._items
                         if i.status == AutoLamellaTaskStatus.NotStarted}
            non_pending = [i for i in self._items
                          if i.status != AutoLamellaTaskStatus.NotStarted]
            reordered_pending = [id_to_item[iid] for iid in item_ids
                                if iid in id_to_item]
            self._items = non_pending + reordered_pending

    # --- Iteration (called by worker thread) ---

    def next(self) -> Optional[WorkItem]:
        """Return the next pending item, mark it InProgress, or None if queue is empty."""
        with self._lock:
            for item in self._items:
                if item.status == AutoLamellaTaskStatus.NotStarted:
                    item.status = AutoLamellaTaskStatus.InProgress
                    self._active = item
                    return item
            self._active = None
            return None

    def mark_done(self, item: WorkItem, status: AutoLamellaTaskStatus) -> None:
        """Mark item as completed/failed/skipped."""
        with self._lock:
            item.status = status
            if self._active is item:
                self._active = None

    # --- Query ---

    @property
    def items(self) -> List[WorkItem]:
        """Return a deep copy — safe to read from another thread."""
        with self._lock:
            return [copy.copy(i) for i in self._items]

    @property
    def pending(self) -> List[WorkItem]:
        with self._lock:
            return [i for i in self._items
                    if i.status == AutoLamellaTaskStatus.NotStarted]

    @property
    def active(self) -> Optional[WorkItem]:
        with self._lock:
            return self._active

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return not any(i.status == AutoLamellaTaskStatus.NotStarted
                          for i in self._items)

    @property
    def task_names(self) -> List[str]:
        """Original task names list (for status dict compat)."""
        return list(self._task_names)

    @property
    def lamella_names(self) -> List[str]:
        """Original lamella names list (for status dict compat)."""
        return list(self._lamella_names)
