"""Tests for the plan-object multi-grid scheduler (Phase 1).

Covers the selection primitive (`select_next`), the shared skip predicate
(`should_skip`), the planner (`build_plan`), `TaskQueue.build_from_plan`, and a
plan-equals-execution check through `TaskManager`.
"""

import os
import tempfile
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskState,
    AutoLamellaTaskStatus,
    Experiment,
    GridRecord,
    Lamella,
)
from fibsem.applications.autolamella.workflows.tasks.queue import TaskQueue
from fibsem.applications.autolamella.workflows.tasks.scheduling import (
    GRID_GREEDY,
    TASK_GREEDY,
    build_plan,
    select_next,
    should_skip,
)

TASKS = ["mill_trench", "rough_mill", "polish"]


def _experiment(spec):
    """In-memory experiment: {grid_name: n_lamellae}. Returns (exp, {name: rec})."""
    exp = Experiment.create(path=tempfile.mkdtemp(), name="sched-test")
    recs = {}
    for gname, n in spec.items():
        rec = GridRecord(name=gname)
        exp.add_grid(rec)
        recs[gname] = rec
        for i in range(n):
            exp.positions.append(
                Lamella(petname=f"{gname}{i + 1}", path=f"{exp.path}/{gname}{i + 1}",
                        number=i + 1, grid_id=rec._id))
    return exp, recs


def _complete(lamella, task_name):
    st = AutoLamellaTaskState()
    st.name = task_name
    lamella.task_history.append(st)


def _requires(mapping):
    """A stub task_protocol whose workflow_config.requirements(t) returns mapping[t]."""
    return SimpleNamespace(
        workflow_config=SimpleNamespace(requirements=lambda t: mapping.get(t, [])))


# --- select_next ---

def test_select_next_prefers_loaded_grid():
    items = ["a", "b", "c"]
    grid_of = {"a": "G1", "b": "G2", "c": "G2"}.get
    assert select_next(items, {"G2"}, grid_of) == "b"


def test_select_next_falls_back_to_first_when_none_loaded():
    items = ["a", "b"]
    grid_of = {"a": "G1", "b": "G2"}.get
    assert select_next(items, set(), grid_of) == "a"  # forces an exchange to G1


def test_select_next_unlinked_item_reachable_anywhere():
    grid_of = {"x": None, "a": "G1"}.get
    assert select_next(["x", "a"], set(), grid_of) == "x"


def test_select_next_empty():
    assert select_next([], {"G1"}, lambda x: None) is None


# --- should_skip ---

def test_should_skip_not_required():
    exp = SimpleNamespace(task_protocol=None)
    lam = SimpleNamespace(name="L9", is_failure=False, has_completed_task=lambda t: False)
    assert should_skip(exp, lam, "t", ["L1", "L2"]) == "not_required"


def test_should_skip_missing_prereq():
    exp = SimpleNamespace(task_protocol=_requires({"polish": ["rough_mill"]}))
    lam = SimpleNamespace(name="L1", is_failure=False, has_completed_task=lambda t: False)
    assert should_skip(exp, lam, "polish", ["L1"]) == "missing_prereqs"


def test_should_skip_prereq_satisfied_via_is_completed():
    exp = SimpleNamespace(task_protocol=_requires({"polish": ["rough_mill"]}))
    lam = SimpleNamespace(name="L1", is_failure=False, has_completed_task=lambda t: False)
    done = lambda l, t: t == "rough_mill"  # the planner's virtual-accrual view
    assert should_skip(exp, lam, "polish", ["L1"], is_completed=done) is None


def test_should_skip_tolerates_no_protocol():
    exp = SimpleNamespace(task_protocol=None)
    lam = SimpleNamespace(name="L1", is_failure=False, has_completed_task=lambda t: False)
    assert should_skip(exp, lam, "anything", ["L1"]) is None


# --- build_plan: grid-greedy ---

def test_grid_greedy_minimises_exchanges():
    exp, _ = _experiment({"A": 3, "B": 2, "C": 3})
    p = build_plan(exp, TASKS, policy=GRID_GREEDY)
    assert p.n_work == 24
    assert p.n_exchanges == 2          # grids - 1, the single-pass minimum
    assert p.n_realigns == 3           # one per grid loaded
    assert p.grid_order == ["A", "B", "C"]
    assert p.items_per_grid == {"A": 9, "B": 6, "C": 9}


def test_plan_items_match_work_events():
    exp, _ = _experiment({"A": 2, "B": 2})
    p = build_plan(exp, TASKS, policy=GRID_GREEDY)
    from_events = [(e.lamella, e.task) for e in p.events if e.kind == "work"]
    from_items = [(it.lamella, it.task) for it in p.items]
    assert from_items == from_events    # the two views of the order agree


def test_grid_greedy_drains_grid_before_swapping():
    exp, _ = _experiment({"A": 2, "B": 2})
    p = build_plan(exp, TASKS, policy=GRID_GREEDY)
    grids = [e.grid for e in p.events if e.kind == "work"]
    assert grids == ["A"] * 6 + ["B"] * 6


# --- build_plan: task-greedy ---

def test_task_greedy_reloads_per_phase():
    exp, _ = _experiment({"A": 3, "B": 2, "C": 3})
    p = build_plan(exp, TASKS, policy=TASK_GREEDY)
    assert p.n_work == 24
    assert p.n_exchanges == 8           # 3 grids x 3 phases = 9 loads, 1 initial + 8
    assert p.n_realigns == 9


# --- capacity / seeded / skip ---

def test_capacity_two_avoids_exchanges():
    exp, _ = _experiment({"A": 2, "B": 2})
    p = build_plan(exp, TASKS, capacity=2, policy=GRID_GREEDY)
    assert p.n_exchanges == 0           # both grids co-loaded → no swap
    assert p.n_realigns == 2


def test_seeded_loaded_grid_runs_without_load_event():
    exp, recs = _experiment({"A": 2, "B": 2})
    p = build_plan(exp, TASKS, loaded_ids=[recs["A"]._id], policy=GRID_GREEDY)
    assert p.events[0].kind == "work" and p.events[0].grid == "A"
    assert p.grid_order == ["B"]        # only B needed loading
    assert p.n_realigns == 1


def test_completed_tasks_are_rerunnable_not_skipped():
    # the original behaviour: a previously-completed task can be run again
    exp, recs = _experiment({"A": 1, "B": 1})
    lam_a = exp.get_lamellae_for_grid(recs["A"])[0]
    for t in TASKS:
        _complete(lam_a, t)
    p = build_plan(exp, TASKS, policy=GRID_GREEDY)
    assert p.n_skipped == 0
    assert p.n_work == 6                # A's three (re-run) + B's three
    assert "A" in p.grid_order


def test_failed_lamella_is_skipped():
    from fibsem.applications.autolamella.structures import DefectType
    exp, recs = _experiment({"A": 2})
    failed = exp.get_lamellae_for_grid(recs["A"])[0]
    failed.defect.state = DefectType.FAILURE
    p = build_plan(exp, TASKS, policy=GRID_GREEDY)
    assert p.n_work == 3               # only the non-failed lamella's 3 tasks
    assert p.n_skipped == 3            # the failed lamella's 3 tasks
    assert all(r == "failure" and lam == failed.name for lam, _, r in p.skipped)


def test_unlinked_lamella_never_triggers_exchange():
    exp, _ = _experiment({"A": 1})
    exp.positions.append(
        Lamella(petname="loose", path=f"{exp.path}/loose", number=99, grid_id=None))
    p = build_plan(exp, ["mill_trench"], policy=GRID_GREEDY)
    assert p.n_work == 2
    assert p.grid_order == ["A"]
    assert p.n_realigns == 1


# --- build_plan honours prerequisites (the fidelity the plan must match) ---

def test_prereq_dependent_included_when_prereq_selected():
    exp, _ = _experiment({"A": 2})
    exp.task_protocol = _requires({"polish": ["rough_mill"]})
    p = build_plan(exp, ["rough_mill", "polish"], policy=GRID_GREEDY)
    # rough_mill is placed first (task-outer) so polish's prereq is satisfied
    assert p.n_work == 4 and p.n_skipped == 0


def test_prereq_dependent_dropped_when_prereq_not_selected():
    exp, _ = _experiment({"A": 2})
    exp.task_protocol = _requires({"polish": ["rough_mill"]})
    p = build_plan(exp, ["polish"], policy=GRID_GREEDY)   # prereq not in the run
    assert p.n_work == 0 and p.n_skipped == 2             # both polish items dropped
    # the reason is recorded so the UI/preview can explain it
    assert sorted(p.skipped) == [("A1", "polish", "missing_prereqs"),
                                 ("A2", "polish", "missing_prereqs")]


# --- TaskQueue.build_from_plan ---

def test_queue_build_from_plan_walks_in_order():
    q = TaskQueue()
    pairs = [("A1", "t1"), ("A2", "t1"), ("B1", "t1")]
    q.build_from_plan(pairs)
    walked = []
    while (it := q.next()) is not None:
        walked.append((it.item_name, it.task_name))
        q.mark_done(it, AutoLamellaTaskStatus.Completed)
    assert walked == pairs
    assert q.item_names == ["A1", "A2", "B1"]   # unique, first-appearance order


# --- plan_for_run seeds loaded-set from the holder ---

def test_plan_for_run_seeds_loaded_from_holder():
    from fibsem import utils
    from fibsem.microscopes._stage import SampleGrid
    from fibsem.applications.autolamella.workflows.tasks.scheduling import plan_for_run

    microscope, _ = utils.setup_session(manufacturer="Demo")
    exp, _ = _experiment({"A": 1, "B": 1})
    # establish a clean holder, then place grid A into a working slot
    for slot in microscope._stage.holder.slots.values():
        slot.loaded_grid = None
    next(iter(microscope._stage.holder.slots.values())).loaded_grid = SampleGrid(name="A")

    p = plan_for_run(exp, microscope, TASKS)
    assert p.events[0].kind == "work" and p.events[0].grid == "A"  # A already loaded
    assert p.grid_order == ["B"]                                   # only B needed loading


# --- plan == execution through the manager ---

def test_manager_executes_in_plan_order():
    from fibsem import utils
    from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager

    microscope, _ = utils.setup_session(manufacturer="Demo")
    exp = Experiment.create(path=tempfile.mkdtemp(), name="exec")
    for i in range(3):                       # unlinked lamellae → no grid loading
        exp.positions.append(
            Lamella(petname=f"L{i + 1}", path=f"{exp.path}/L{i + 1}", number=i + 1))

    tm = TaskManager(microscope, exp)
    executed = []

    def fake_run(task_name, lamella):
        executed.append((lamella.name, task_name))
        lamella.task_state.status = AutoLamellaTaskStatus.Completed
        return None

    tm._run_single_task = fake_run
    tm.run(["t1", "t2"])

    planned = [(it.lamella, it.task) for it in tm.plan.items]
    assert executed == planned               # runner follows the baked plan exactly
