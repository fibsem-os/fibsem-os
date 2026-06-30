"""Load-aware scheduling for multi-grid lamella execution.

The schedule is a **first-class plan**: :func:`build_plan` bakes the grid-greedy
order at build time (forward-simulating the loaded-set with :func:`select_next`)
and returns the ordered work items plus an event stream. The *same* function
produces the preview and the order the runner executes, so they cannot drift.
The runner walks that order and skips in place — it never re-orders — which stays
optimal because grid-greedy keeps each grid's items contiguous.

Design doc: ``docs/design/multi-grid-lamella-execution.md``.
"""

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import Experiment, Lamella
    from fibsem.microscope import FibsemMicroscope

T = TypeVar("T")

GRID_GREEDY = "grid_greedy"
TASK_GREEDY = "task_greedy"

__all__ = [
    "GRID_GREEDY",
    "TASK_GREEDY",
    "select_next",
    "should_skip",
    "ScheduleEvent",
    "PlanItem",
    "Plan",
    "build_plan",
    "plan_for_run",
]


def select_next(
    items: Sequence[T],
    loaded_ids: Iterable[str],
    grid_of: Callable[[T], Optional[str]],
) -> Optional[T]:
    """Pick the next item to run, preferring ones on an already-loaded grid.

    This is the per-step rule behind grid-greedy ordering: an already-loaded grid
    is fully drained before any item that would force an exchange is chosen.

    Args:
        items: pending items in canonical order (task-outer, lamella-inner).
        loaded_ids: ids of the grids currently (virtually) in the working slot(s).
        grid_of: maps an item to its grid id; ``None`` means the item is
            reachable on whatever is loaded (unlinked / single-grid lamellae).

    Returns:
        The chosen item, or ``None`` if ``items`` is empty.
    """
    if not items:
        return None
    loaded = set(loaded_ids)
    for it in items:
        gid = grid_of(it)
        if gid is None or gid in loaded:
            return it  # runnable now, zero exchange cost
    return items[0]  # nothing reachable: the first canonical item forces an exchange


def should_skip(
    experiment: "Experiment",
    lamella: "Lamella",
    task_name: str,
    required_lamella: Sequence[str],
    *,
    is_completed: Optional[Callable[["Lamella", str], bool]] = None,
) -> Optional[str]:
    """Return a skip reason for a ``(task, lamella)`` item, or ``None`` to run it.

    The single skip predicate shared by the planner and the runner so the plan
    agrees with execution. ``is_completed`` lets the planner consult a virtual
    completion set (work it has already placed) in addition to real history;
    the runner uses the default (real ``task_history``).
    """
    if is_completed is None:
        is_completed = lambda lam, t: lam.has_completed_task(t)  # noqa: E731

    if required_lamella and lamella.name not in required_lamella:
        return "not_required"
    if lamella.is_failure:
        return "failure"

    proto = getattr(experiment, "task_protocol", None)
    workflow_config = getattr(proto, "workflow_config", None) if proto else None
    if workflow_config is not None:
        try:
            requirements = workflow_config.requirements(task_name)
        except Exception:  # noqa: BLE001 - unknown task / partial protocol
            requirements = None
        if requirements and not all(is_completed(lamella, req) for req in requirements):
            return "missing_prereqs"
    return None


@dataclass
class ScheduleEvent:
    """One step in a plan's event stream (for preview + cost estimation).

    ``kind`` is ``load`` (first placement into a free slot), ``exchange`` (a
    placement that evicts a loaded grid), ``realign`` (fires after every real
    placement), or ``work`` (a single ``(task, lamella)``).
    """

    kind: str
    grid: Optional[str]  # grid name (human-readable), or None for unlinked lamellae
    lamella: Optional[str] = None
    task: Optional[str] = None


@dataclass
class PlanItem:
    """A single ordered unit of work — the baked schedule the queue executes."""

    lamella: str
    task: str


@dataclass
class Plan:
    """The materialised schedule: the baked order + a preview event stream."""

    items: List[PlanItem] = field(default_factory=list)
    events: List[ScheduleEvent] = field(default_factory=list)
    # (lamella, task, reason) for work dropped at plan time — surfaced in the
    # preview/confirmation so e.g. a forgotten prerequisite is caught up front.
    skipped: List[Tuple[str, str, str]] = field(default_factory=list)
    n_exchanges: int = 0
    n_realigns: int = 0
    n_work: int = 0
    n_skipped: int = 0
    grid_order: List[Optional[str]] = field(default_factory=list)
    items_per_grid: Dict[Optional[str], int] = field(default_factory=dict)

    @property
    def pairs(self) -> List[tuple]:
        """The ordered ``(item_name, task_name)`` pairs for ``build_from_plan``."""
        return [(it.lamella, it.task) for it in self.items]


def build_plan(
    experiment: "Experiment",
    task_names: Sequence[str],
    lamella_names: Optional[Sequence[str]] = None,
    *,
    loaded_ids: Iterable[str] = (),
    capacity: int = 1,
    policy: str = GRID_GREEDY,
) -> Plan:
    """Build the schedule for a set of lamellae + tasks — the single ordering authority.

    Walks the ``(task × lamella)`` matrix in canonical task-outer order and
    reorders it for ``policy`` by forward-simulating the loaded-set. Skips are
    applied exactly as the runner applies them (the shared :func:`should_skip`,
    against history + work already placed), so the plan predicts execution on the
    happy path. With no microscope: pure function of its arguments.

    Args:
        experiment: provides lamella<->grid links and the task protocol.
        task_names: tasks to run, in workflow order.
        lamella_names: lamellae to run on; defaults to all positions.
        loaded_ids: grid ids that start loaded (seed from the holder).
        capacity: working slots (1 = autoloader, >1 = static multi-slot holder).
        policy: ``GRID_GREEDY`` (minimise exchanges) or ``TASK_GREEDY``.

    Returns:
        A :class:`Plan` (ordered items + event stream + tallies).
    """
    if lamella_names is None:
        lamella_names = [lam.name for lam in experiment.positions]

    lam_by_name = {lam.name: lam for lam in experiment.positions}

    def grid_id_of(name: str) -> Optional[str]:
        lam = lam_by_name.get(name)
        if lam is None:
            return None
        grid = experiment.get_grid_for_lamella(lam)
        return grid._id if grid is not None else None

    def grid_name_of_id(gid: Optional[str]) -> Optional[str]:
        grid = experiment.get_grid_by_id(gid)
        return grid.name if grid is not None else None

    # work accrued within this virtual run, so prereqs resolve as the runner sees them
    accrued = set()

    def is_completed(lam: "Lamella", task: str) -> bool:
        return lam.has_completed_task(task) or (lam.name, task) in accrued

    # canonical task-outer pending list. Skips are decided by `should_skip` in the
    # loop below (failure / missing prereqs) — mirroring the runner exactly. We do
    # NOT drop already-completed tasks here: re-running a completed task is allowed
    # (the original behaviour). Only names that don't resolve to a lamella are
    # dropped up front, since the loop indexes them directly.
    pending: List[PlanItem] = []
    skipped: List[Tuple[str, str, str]] = []
    for task in task_names:
        for name in lamella_names:
            if name not in lam_by_name:
                skipped.append((name, task, "no_lamella"))
                continue
            pending.append(PlanItem(lamella=name, task=task))

    loaded: List[str] = list(loaded_ids)
    plan = Plan()

    while pending:
        if policy == TASK_GREEDY:
            it = pending[0]
        else:
            it = select_next(pending, loaded, lambda x: grid_id_of(x.lamella))
        pending.remove(it)

        lam = lam_by_name[it.lamella]
        reason = should_skip(experiment, lam, it.task, lamella_names,
                             is_completed=is_completed)
        if reason is not None:
            skipped.append((it.lamella, it.task, reason))
            continue

        gid = grid_id_of(it.lamella)
        gname = grid_name_of_id(gid)
        if gid is not None and gid not in loaded:
            if len(loaded) >= capacity:
                loaded.pop(0)  # evict the oldest working slot
                plan.events.append(ScheduleEvent(kind="exchange", grid=gname))
            else:
                plan.events.append(ScheduleEvent(kind="load", grid=gname))
            plan.events.append(ScheduleEvent(kind="realign", grid=gname))
            loaded.append(gid)
            if gname not in plan.grid_order:
                plan.grid_order.append(gname)

        plan.items.append(it)
        plan.events.append(
            ScheduleEvent(kind="work", grid=gname, lamella=it.lamella, task=it.task))
        plan.items_per_grid[gname] = plan.items_per_grid.get(gname, 0) + 1
        accrued.add((it.lamella, it.task))

    plan.skipped = skipped
    plan.n_exchanges = sum(1 for e in plan.events if e.kind == "exchange")
    plan.n_realigns = sum(1 for e in plan.events if e.kind == "realign")
    plan.n_work = len(plan.items)
    plan.n_skipped = len(skipped)
    return plan


def plan_for_run(
    experiment: "Experiment",
    microscope: "FibsemMicroscope",
    task_names: Sequence[str],
    lamella_names: Optional[Sequence[str]] = None,
    *,
    policy: str = GRID_GREEDY,
) -> Plan:
    """Build the run plan from the microscope's current holder state.

    The shared entry point so the UI preview and the runner produce the *same*
    plan (seeded loaded-set + slot capacity read off the holder). Pure aside from
    reading holder occupancy.
    """
    loaded_ids = {g._id for g in experiment.get_loaded_grids(microscope)}
    holder = getattr(getattr(microscope, "_stage", None), "holder", None)
    # autoloaders currently expose a single working slot, so len(slots) == 1; a
    # static holder may have several (grids placed manually). A loader clears the
    # holder on each exchange, so it never co-loads — revisit if a multi-slot
    # autoloader ever appears.
    capacity = max(len(holder.slots), 1) if getattr(holder, "slots", None) else 1
    return build_plan(experiment, task_names, lamella_names,
                      loaded_ids=loaded_ids, capacity=capacity, policy=policy)
