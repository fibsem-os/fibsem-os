"""What a HookContext carries, and why it has to stand alone.

A webhook consumer is out of process and out of band: it cannot ask a follow-up
question, so the payload has to make sense by itself. See FIB-489.
"""

import logging
from dataclasses import dataclass, field
from typing import List

from fibsem.hooks import (
    FunctionHook,
    HookContext,
    HookEvent,
    HookManager,
    NotificationHook,
    render_template,
)


@dataclass
class _MutableState:
    """Stands in for AutoLamellaTaskState: one object, reused for every run."""

    status: str = "InProgress"
    outputs: List[str] = field(default_factory=list)


class _Uncopyable:
    def __deepcopy__(self, memo):
        raise RuntimeError("cannot copy this")


# ---------------------------------------------------------------------------
# the task_state snapshot
# ---------------------------------------------------------------------------


def test_the_context_snapshots_task_state():
    """The producer reuses one state object per run, so a hook that defers work would
    otherwise read whatever the *next* task has since written into it."""
    live = _MutableState(status="InProgress")

    context = HookContext(event=HookEvent.TASK_STARTED, task_state=live)
    live.status = "Completed"  # the next task moves on
    live.outputs.append("later.tif")

    assert context.task_state.status == "InProgress"
    assert context.task_state.outputs == []


def test_the_snapshot_is_deep_not_a_shallow_copy():
    live = _MutableState()
    context = HookContext(event=HookEvent.TASK_STARTED, task_state=live)

    live.outputs.append("later.tif")

    assert context.task_state.outputs is not live.outputs


def test_a_state_that_cannot_be_copied_does_not_break_the_producer(caplog):
    """This runs in the producer's call stack, before HookManager.fire's try/except
    exists to contain anything -- so it must degrade, not raise."""
    with caplog.at_level(logging.ERROR):
        context = HookContext(event=HookEvent.TASK_FAILED, task_state=_Uncopyable())

    assert isinstance(context.task_state, _Uncopyable)  # the live reference
    assert any("snapshot task_state" in r.message for r in caplog.records)


def test_no_task_state_is_fine():
    assert HookContext(event=HookEvent.WORKFLOW_STARTED).task_state is None


# ---------------------------------------------------------------------------
# identity and progress
# ---------------------------------------------------------------------------


def _ctx(**kwargs) -> HookContext:
    defaults = dict(
        event=HookEvent.TASK_FAILED,
        task_name="MillRough",
        item_name="lamella-3",
        item_id="lam-uuid",
        task_id="task-uuid",
        experiment_id="exp-uuid",
        experiment_name="my-experiment",
        tasks_remaining=5,
        tasks_total=20,
        error="stage timeout",
    )
    defaults.update(kwargs)
    return HookContext(**defaults)


def test_the_motivating_message_is_now_renderable():
    """The failure toast this issue was opened for: a failure does not stop the run,
    so a bare FAILED reads as a dead workflow."""
    rendered = render_template(
        "Task {task_name} FAILED for {item_name}: {error} "
        "— continuing, {tasks_remaining} tasks remaining",
        _ctx(),
    )

    assert rendered == (
        "Task MillRough FAILED for lamella-3: stage timeout "
        "— continuing, 5 tasks remaining"
    )


def test_identity_placeholders_render():
    rendered = render_template(
        "{experiment_name}/{item_name} {item_id} {task_id} {experiment_id}", _ctx()
    )

    assert rendered == "my-experiment/lamella-3 lam-uuid task-uuid exp-uuid"


def test_an_absent_count_renders_blank_not_none():
    """Workflow events from a producer with no queue have no counts; a template must
    not print the word None at a user."""
    rendered = render_template("[{tasks_remaining}]", _ctx(tasks_remaining=None))

    assert rendered == "[]"


def test_the_payload_reaches_a_function_hook():
    received: List[HookContext] = []
    manager = HookManager()
    manager.register(
        FunctionHook(name="f", events=[HookEvent.TASK_FAILED], callback=received.append)
    )

    manager.fire(_ctx())

    got = received[0]
    assert (got.item_id, got.task_id) == ("lam-uuid", "task-uuid")
    assert (got.experiment_id, got.experiment_name) == ("exp-uuid", "my-experiment")
    assert (got.tasks_remaining, got.tasks_total) == (5, 20)


def test_errors_carry_the_message_only():
    """Decided: no traceback. The payload can leave the machine via a webhook, and a
    traceback is unbounded -- the logfile keeps the full one."""
    received = []
    hook = NotificationHook(
        name="n",
        events=[HookEvent.TASK_FAILED],
        message_template="{error}",
        _notify=lambda msg, typ: received.append(msg),
    )

    hook.run(_ctx(error=str(RuntimeError("stage timeout"))))

    assert received == ["stage timeout"]
    assert "Traceback" not in received[0]
