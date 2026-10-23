"""What a lamella run, or an addition to the running queue, would do and cost,
with nothing started: the Workflow tab's confirmations and the App Server's
plan route both read it, so their numbers agree."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from fibsem.applications.autolamella.workflows.estimate import (
    estimate_lamella_run,
    estimate_queue_addition,
    run_step_seconds,
)
from fibsem.workflows.estimate import AdditionEstimate, RunEstimate
from fibsem.workflows.runner import QueueEdit, RunRequest, item_task_pairs

__all__ = ["LamellaRunPlan", "plan_lamella_run"]


@dataclass(frozen=True)
class LamellaRunPlan:
    """The ``(lamella, task)`` pairs in queue order and what they cost, or why
    the request would be refused. ``already_queued`` names the pairs an
    addition would queue a second time."""

    pairs: Tuple[Tuple[str, str], ...] = ()
    request: Optional[RunRequest] = None
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    add: bool = False
    run_next: bool = False
    already_queued: List[str] = field(default_factory=list)
    estimate: Optional[RunEstimate] = None
    addition: Optional[AdditionEstimate] = None

    @property
    def valid(self) -> bool:
        return not self.reason


def plan_lamella_run(
    runner: Any,
    experiment: Any,
    request: RunRequest,
    run_next: bool = False,
    add: bool = False,
) -> LamellaRunPlan:
    """*request* as a fresh run (its :class:`RunEstimate`) or, with *add*, as
    work joining the running queue (its :class:`AdditionEstimate`), checked
    as the start or the addition will be."""
    if not add:
        checked = runner.check(experiment, request)
        if not checked.started:
            return LamellaRunPlan(reason=checked.reason, details=checked.details)
        resolved = checked.request or request
        pairs = tuple(item_task_pairs(resolved))
        return LamellaRunPlan(
            pairs=pairs,
            request=resolved,
            estimate=estimate_lamella_run(experiment, pairs),
        )
    plan = runner.plan_addition(request, run_next)
    if isinstance(plan, QueueEdit):
        return LamellaRunPlan(
            reason=plan.reason, details=plan.details, add=True, run_next=run_next
        )
    queue = runner.manager.queue
    pending = {(i.item_name, i.task_name) for i in queue.pending}
    stage = getattr(runner.manager.microscope, "sample_handling", None)
    return LamellaRunPlan(
        pairs=plan.pairs,
        request=plan.request,
        add=True,
        run_next=plan.run_next,
        already_queued=[f"{t} for {n}" for n, t in plan.pairs if (n, t) in pending],
        addition=estimate_queue_addition(
            experiment,
            queue,
            list(plan.pairs),
            plan.run_next,
            run_step_seconds(experiment, stage),
        ),
    )
