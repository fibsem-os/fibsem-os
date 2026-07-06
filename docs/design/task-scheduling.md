# Task Scheduling for AutoLamella Workflows

## What Exists Today

Tasks can be scheduled to start at a specific date/time rather than running
immediately.

- **Data model** — `AutoLamellaTaskDescription.scheduled_at: Optional[datetime]`
  (`fibsem/applications/autolamella/structures.py`), serialized as ISO strings in
  `to_dict`/`from_dict`. `AutoLamellaWorkflowConfig.get_scheduled_at(task_name)`
  looks it up by name.
- **Editor UI** — the task editor dialog (`WorkflowTaskEditorWidget`,
  `fibsem/ui/widgets/workflow_task_editor_widget.py`) has a *Properties* panel with a
  "Schedule start time" checkbox + a `QDateTimeEdit` (12-hour AM/PM, up/down step
  arrows, no calendar popup). The picker is bounded to **now → now + 2 days**
  (`_MAX_SCHEDULE_DAYS_AHEAD`); a live hint shows how far ahead the time is
  ("Starts 8h 30m from now …") and warns when a step is blocked by a bound.
- **Run-time wait** — `TaskManager._wait_until_scheduled(...)`
  (`fibsem/applications/autolamella/workflows/tasks/manager.py`) blocks before the
  task's `InProgress` status is emitted, in an interruptible loop
  (`while not self.is_stopped`). It emits a ~15s countdown to the **bottom status bar**
  via `update_status_ui(..., status_bar=msg)` → `workflow_update_signal` →
  `AutoLamellaMainUI._on_workflow_update`.
- **Cancel during wait** — the Stop button is shown as soon as the workflow starts
  (`set_workflow_running()` in `_on_run_workflow_clicked`), so a scheduled-but-not-yet-
  started run is cancellable. Stop sets the manager's `_stop_event`, which the wait loop
  checks.

## Deferred: "Resume Workflow Now" button

A status-bar button to skip the scheduled wait and start the task immediately, **without**
cancelling the run and **without** changing the saved `scheduled_at`. Discussed and
intentionally deferred — the need is real but infrequent, and the workaround (Stop → edit
task → uncheck schedule → Apply → Run) exists. Add it when an operator actually wants it.

### Why a new event (not reuse Stop)

`_stop_event` cancels the run (the outer `while not self.is_stopped` loop exits, no task
runs). "Resume Now" must end the wait *and then run the task*, so it needs its own
event — on skip, `is_stopped` stays False and `_run_queue` falls through to run the task.

### Implementation sketch (~25 lines, no new threads)

1. **`manager.py` — `TaskManager`**
   - Add `self._skip_wait_event = threading.Event()` in `__init__`; add `skip_wait()` →
     `self._skip_wait_event.set()` (mirrors `stop()`).
   - In `_wait_until_scheduled`: clear the event on entry; change the loop guard to
     `while not self.is_stopped and not self._skip_wait_event.is_set():`. The existing
     `if self.is_stopped: break` in `_run_queue` is unchanged — on skip it falls through
     and runs the task. Clearing on entry scopes the skip to the *current* wait, so a
     later scheduled task still waits (see decision 1).
2. **`workflows/ui.py`** — add a `waiting: bool = False` flag to `update_status_ui`,
   included in the emitted dict; the manager passes `waiting=True` on its countdown emits.
3. **`AutoLamellaUI.py`** — add `skip_scheduled_wait()` mirroring `stop_task_workflow`:
   guard on `is_workflow_running` / `_task_manager is not None`, then
   `self._task_manager.skip_wait()`.
4. **`AutoLamellaMainUI.py`**
   - In `_create_status_bar`: create `self.resume_now_btn = QPushButton("Resume Workflow Now")`,
     hidden by default, `addPermanentWidget` next to `stop_workflow_btn`, connect to a
     handler that calls `self.autolamella_ui.skip_scheduled_wait()`.
   - In `_on_workflow_update`: `self.resume_now_btn.setVisible(bool(info.get("waiting")))`;
     hide it once a real task `status` dict arrives (task started).
   - Hide it in `hide_workflow_running()` and `_on_workflow_finished()`. Do NOT show it in
     `set_workflow_running()` (that fires at run start, before any wait).

   Alternative to the `waiting` flag: a dedicated `scheduled_wait_signal(bool)` on
   `AutoLamellaUI` — cleaner separation, slightly more plumbing. The flag-on-existing-signal
   matches how the countdown message is already routed.

### Decisions to settle before building

1. **Scope** — skip only the current wait (recommended; clear the event each wait) vs. skip
   all remaining scheduled waits this run (don't clear).
2. **Confirmation** — Stop has a Yes/No dialog; Resume Now is non-destructive, so likely one
   click, no dialog.
3. **Persistence** — must NOT clear `task.scheduled_at`; the schedule stays in the protocol
   for next time.
4. **Styling/label** — `SECONDARY_BUTTON_STYLESHEET` vs `PRIMARY`; "Resume Workflow Now" vs
   "Start Now".

### Testing

- **Unit (manager, headless)** — start a wait far in the future on a worker thread, call
  `skip_wait()` from the main thread, assert the wait returns promptly and `is_stopped` is
  False (task proceeds). Assert a fresh wait re-arms (current-wait scope).
- **Manual UI** — schedule a task ~10 min out, Run → "Resume Workflow Now" appears beside
  Stop with the countdown → click it → task starts immediately, button disappears, run
  continues. Confirm Stop still cancels during the wait and `scheduled_at` is unchanged.
