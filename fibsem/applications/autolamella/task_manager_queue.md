# TaskManager Refactor Plan

## Context

`run_tasks()` in `tasks.py:1268` is a 150-line function with nested loops, skip logic, status emission, error handling, and stop-event checking. Wrapping it in a class enables:
1. Cleaner state management (no passing `parent_ui` everywhere for flags)
2. A path toward a dynamic queue system where (lamella, task) pairs can be added/removed/reordered at runtime

## Phase 1: Immediate Refactor — `TaskManager` class (DONE)

### Files modified
- `fibsem/applications/autolamella/workflows/tasks/tasks.py` — added `TaskManager`, kept `run_tasks` as thin wrapper
- `fibsem/applications/autolamella/ui/AutoLamellaUI.py` — updated `_run_tasks_worker` to use `TaskManager`

### `TaskManager` class design

```python
class TaskManager:
    """Manages execution of autolamella tasks across lamellas."""

    def __init__(self,
                 microscope: FibsemMicroscope,
                 experiment: 'Experiment',
                 parent_ui: Optional['AutoLamellaUI'] = None):
        self.microscope = microscope
        self.experiment = experiment
        self.parent_ui = parent_ui
        self._stop_event = threading.Event()

    # --- Public API ---
    def run(self, task_names: List[str],
            required_lamella: Optional[List[str]] = None) -> None:
        """Run tasks (current behavior, direct replacement for run_tasks())."""

    def stop(self) -> None:
        """Signal the manager to stop after current task completes."""
        self._stop_event.set()

    @property
    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    # --- Internal helpers ---
    def _should_skip(self, lamella, task_name, required_lamella) -> Optional[str]:
        """Return skip reason string, or None if task should run."""
        # Consolidates the 3 skip checks (not required, is_failure, missing prereqs)

    def _emit_status(self, task_name, task_names, lamella,
                     required_lamella, status, error_message=None,
                     task_duration=None) -> None:
        """Emit workflow_update_signal with standard status dict."""
        # Deduplicates the 3 near-identical signal emission blocks

    def _run_single_task(self, task_name, lamella) -> Optional[Exception]:
        """Execute one task for one lamella. Returns exception or None."""
        # Wraps run_task() + experiment.save() + error handling
```

### Key decisions
- **`_workflow_stop_event` stays on `parent_ui`** — `_check_for_abort()` in `workflows/ui.py:26` reads `parent_ui._workflow_stop_event.is_set()` and is called from ~10 places throughout `ui.py` and `AutoLamellaTask._check_for_abort()`. Changing the event location would require touching all of those call sites. Instead:
  - `TaskManager.stop()` sets `parent_ui._workflow_stop_event` (if parent_ui exists) **and** its own `self._stop_event`
  - `TaskManager.run()` checks `self._stop_event` (works headless too)
  - All existing `_check_for_abort(parent_ui)` calls continue working unchanged
  - `AutoLamellaUI.stop_task_workflow` calls `self._task_manager.stop()` instead of directly setting the event
- **`run_tasks()` function preserved** as a thin wrapper for backward compat / headless usage:
  ```python
  def run_tasks(microscope, experiment, task_names, required_lamella=None, parent_ui=None):
      manager = TaskManager(microscope, experiment, parent_ui)
      manager.run(task_names, required_lamella)
  ```
- **`_emit_status` deduplicates** the 3 copy-pasted signal emission blocks (InProgress, Completed/Failed, Skipped)
- **`_should_skip` consolidates** the skip logic into one method returning a reason string or None

### `_check_for_abort` strategy
**Phase 1 (now)**: Keep `_check_for_abort(parent_ui)` reading `parent_ui._workflow_stop_event` unchanged. `TaskManager.stop()` sets both events. Zero changes to `ui.py` or `AutoLamellaTask`.

**Phase 2 (with queue system)**: Migrate to passing `TaskManager` to tasks. `AutoLamellaTask.__init__` gets `task_manager: Optional[TaskManager]` param, and `_check_for_abort` checks `self.task_manager.is_stopped`. The `ui.py` helper functions (`ask_user`, `run_milling`, etc.) access the manager via `parent_ui._task_manager`. This gives tasks access to queue state for future features (progress reporting, dynamic re-prioritization).

### Changes to `AutoLamellaUI`
- `_run_tasks_worker` creates a `TaskManager` and calls `manager.run()`
- Store `self._task_manager = manager` so `stop_task_workflow` can call `self._task_manager.stop()`
- `_workflow_stop_event` remains on the UI — `TaskManager.stop()` sets both its own event and the UI's event

---

## Phase 2: Queue System — Detailed Implementation Plan

### Overview

Replace the fixed `task_names × required_lamella` nested loop in `TaskManager.run()` with a mutable `TaskQueue` of `WorkItem`s. The queue is built upfront from the same task×lamella matrix, but can be mutated (add/remove/reorder) by the UI thread while the worker thread executes items.

### Key constraint: status dict compatibility

The current `_emit_status` sends fields that UI consumers depend on:
- `task_names`, `lamella_names` — fixed lists (the original matrix dimensions)
- `current_task_index`, `current_lamella_index` — indices into those lists
- `outer_idx = task_idx * n_lamellas + lam_idx` — used by `WorkflowProgressWidget`

**Strategy**: Keep emitting these fields for backward compat. The queue stores `task_names` and `lamella_names` as its "plan" lists. `_emit_status` derives indices from the current `WorkItem`. Dynamically added items that don't fit the original matrix get appended to the timeline widget.

---

### New file: `fibsem/applications/autolamella/workflows/tasks/queue.py`

```python
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
        """Populate queue from task × lamella matrix (task-outer, lamella-inner)."""
        with self._lock:
            self._task_names = list(task_names)
            self._lamella_names = list(lamella_names)
            self._items = [
                WorkItem(lamella_name=ln, task_name=tn)
                for tn in task_names
                for ln in lamella_names
            ]
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
                # Insert before the index-th pending item
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
                        return False  # Cannot remove active item
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
            # Remove all pending from list
            non_pending = [i for i in self._items
                          if i.status != AutoLamellaTaskStatus.NotStarted]
            # Rebuild: non-pending in original order, then pending in new order
            reordered_pending = [id_to_item[iid] for iid in item_ids
                                if iid in id_to_item]
            # Find insertion point: after last non-pending item
            self._items = non_pending + reordered_pending

    # --- Iteration (called by worker thread) ---

    def next(self) -> Optional[WorkItem]:
        """Return the next pending item, or None if queue is empty."""
        with self._lock:
            for item in self._items:
                if item.status == AutoLamellaTaskStatus.NotStarted:
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
        with self._lock:
            return list(self._items)

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
```

---

### Changes to `manager.py`

**Replace the nested loop in `run()` with queue-based iteration:**

```python
from fibsem.applications.autolamella.workflows.tasks.queue import TaskQueue, WorkItem

class TaskManager:
    def __init__(self, ...):
        ...
        self.queue = TaskQueue()

    def run(self, task_names: List[str],
            required_lamella: Optional[List[str]] = None) -> None:
        if required_lamella is None:
            required_lamella = [p.name for p in self.experiment.positions]

        self.queue.build_from_matrix(task_names, required_lamella)
        self._run_queue()

    def _run_queue(self) -> None:
        """Process queue until empty or stopped."""
        while not self.is_stopped:
            item = self.queue.next()
            if item is None:
                break

            lamella = self.experiment.get_lamella_by_name(item.lamella_name)
            if lamella is None:
                self.queue.mark_done(item, AutoLamellaTaskStatus.Skipped)
                continue

            skip_reason = self._should_skip(lamella, item.task_name,
                                            self.queue.lamella_names)
            if skip_reason == "not_required" or skip_reason == "failure":
                self.queue.mark_done(item, AutoLamellaTaskStatus.Skipped)
                continue
            if skip_reason == "missing_prereqs":
                self._emit_status_for_item(item, lamella,
                                           AutoLamellaTaskStatus.Skipped,
                                           msg=f"Skipping {lamella.name}: missing prereqs.")
                self.queue.mark_done(item, AutoLamellaTaskStatus.Skipped)
                continue

            # InProgress
            self._emit_status_for_item(item, lamella,
                                       AutoLamellaTaskStatus.InProgress,
                                       msg=f"Starting task {item.task_name} for {lamella.name}.")

            # Execute
            err = self._run_single_task(item.task_name, lamella)
            final_status = lamella.task_state.status
            self.queue.mark_done(item, final_status)

            # Completed/Failed
            msg = (f"Completed task {item.task_name} for {lamella.name}."
                   if err is None else
                   f"Error in task {item.task_name} for {lamella.name}.")
            self._emit_status_for_item(item, lamella, final_status,
                                       error_message=lamella.task_state.status_message,
                                       task_duration=lamella.task_state.duration,
                                       msg=msg)

        update_status_ui(self.parent_ui, "", workflow_info="All tasks completed.")
        print(self.experiment.task_history_dataframe())

    def _emit_status_for_item(self, item: WorkItem, lamella, status, **kwargs):
        """Emit status using queue context for index computation."""
        task_names = self.queue.task_names
        lamella_names = self.queue.lamella_names
        # Derive indices for backward compat
        task_idx = task_names.index(item.task_name) if item.task_name in task_names else 0
        lam_idx = lamella_names.index(item.lamella_name) if item.lamella_name in lamella_names else 0

        self._emit_status(
            task_name=item.task_name,
            task_names=task_names,
            lamella=lamella,
            required_lamella=lamella_names,
            status=status,
            **kwargs,
        )
```

**`_emit_status` stays unchanged** — it already computes indices from `task_names.index(task_name)` and `required_lamella.index(lamella.name)`.

**`_should_skip` stays unchanged** — it just needs `lamella`, `task_name`, `required_lamella`.

**`_run_single_task` stays unchanged** — it just needs `task_name`, `lamella`.

---

### Changes to `__init__.py`

Export new types:
```python
from fibsem.applications.autolamella.workflows.tasks.queue import TaskQueue, WorkItem
```

---

### Changes to `AutoLamellaUI._run_tasks_worker`

No changes needed — it already calls `self._task_manager.run(task_names, lamella_names)`.
The queue is internal to `TaskManager.run()`.

For future dynamic queue mutation from the UI:
```python
# UI thread can call (thread-safe):
self._task_manager.queue.add("Lamella_003", "mill_rough")
self._task_manager.queue.remove(item_id)
self._task_manager.queue.reorder([id1, id2, id3])
```

---

### Changes to `WorkflowProgressWidget`

**No immediate changes needed.** The status dict emitted by `_emit_status` has the same structure. `set_workflow()` is called once at start with the matrix dimensions, and `update_from_status()` uses the same index computation.

**Future enhancement**: When items are dynamically added, emit a `queue_changed` signal that triggers `set_workflow()` re-initialization with the updated item list. This is deferred — the first iteration just replaces the loop with queue iteration.

---

### Migration strategy (incremental)

**Step 1**: Create `queue.py` with `WorkItem` and `TaskQueue`.

**Step 2**: Refactor `TaskManager.run()` to use `self.queue.build_from_matrix()` + `_run_queue()`. Keep `_emit_status` unchanged. The external behavior is identical — same status dicts, same ordering.

**Step 3**: Expose `self.queue` on `TaskManager` for UI access. UI can read `queue.pending`, `queue.active`, `queue.items` for display.

**Step 4** (future): Add queue mutation from UI — add/remove/reorder buttons. Emit `queue_changed` signal for timeline refresh.

---

### Verification

- `python -c "from fibsem.applications.autolamella.workflows.tasks.queue import TaskQueue, WorkItem; print('OK')"`
- `python -m pytest tests/ -q` — all existing tests should pass unchanged
- Manual: run autolamella UI with Demo microscope, verify workflow executes identically (same status bar text, same timeline, same toasts)
