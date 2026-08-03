"""Tests for the autolamella TaskQueue.

The load-bearing property is count invariance: no operation except remove()
may change what is in the queue, and remove() marks rather than deletes. The
previous replacement-based reorder() silently destroyed pending items, so that
property is asserted directly rather than only implied by the ordering tests.
"""

import random
import threading

import pytest

from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus as Status
from fibsem.applications.autolamella.workflows.tasks.queue import (
    QueueOp,
    TaskQueue,
    WorkItem,
)


@pytest.fixture
def queue() -> TaskQueue:
    """A 2 task x 2 lamella queue: 4 pending items, nothing consumed."""
    q = TaskQueue()
    q.build_from_matrix(["Trench", "Undercut"], ["L1", "L2"])
    return q


def pairs(q: TaskQueue):
    return [(i.lamella_name, i.task_name) for i in q.items]


def statuses(q: TaskQueue):
    return [i.status for i in q.items]


def pending_pairs(q: TaskQueue):
    return [(i.lamella_name, i.task_name) for i in q.pending]


# ── Build / consume ───────────────────────────────────────────────────────────

def test_build_from_matrix_is_task_outer_lamella_inner(queue):
    assert pairs(queue) == [
        ("L1", "Trench"), ("L2", "Trench"),
        ("L1", "Undercut"), ("L2", "Undercut"),
    ]
    assert all(i.status is Status.NotStarted for i in queue.items)


def test_next_marks_in_progress_and_sets_active(queue):
    item = queue.next()
    assert (item.lamella_name, item.task_name) == ("L1", "Trench")
    assert item.status is Status.InProgress
    assert queue.active.id == item.id


def test_mark_done_clears_active(queue):
    item = queue.next()
    queue.mark_done(item, Status.Completed)
    assert queue.active is None
    assert statuses(queue)[0] is Status.Completed


def test_mark_done_accepts_a_snapshot_copy(queue):
    """items returns copies; marking one by identity must still land."""
    queue.next()
    snapshot = queue.items[0]
    assert snapshot is not queue._items[0]
    queue.mark_done(snapshot, Status.Failed)
    assert statuses(queue)[0] is Status.Failed


def test_is_empty_reflects_only_pending(queue):
    assert not queue.is_empty
    while (item := queue.next()) is not None:
        queue.mark_done(item, Status.Completed)
    assert queue.is_empty


def test_next_skips_removed_items(queue):
    first = queue.pending[0]
    queue.remove(first.id)
    item = queue.next()
    assert (item.lamella_name, item.task_name) == ("L2", "Trench")


# ── add ───────────────────────────────────────────────────────────────────────

def test_add_appends_by_default(queue):
    queue.add("L3", "Polishing")
    assert pairs(queue)[-1] == ("L3", "Polishing")


def test_add_front_lands_after_the_active_item(queue):
    active = queue.next()
    queue.add("L3", "Polishing", front=True)
    result = pairs(queue)
    # history/active untouched at position 0, new item first among pending
    assert result[0] == (active.lamella_name, active.task_name)
    assert result[1] == ("L3", "Polishing")


def test_add_before_and_after_an_anchor(queue):
    anchor = queue.pending[1]  # ("L2", "Trench")
    queue.add("Lbefore", "T", before=anchor.id)
    queue.add("Lafter", "T", after=anchor.id)
    result = pairs(queue)
    i = result.index(("L2", "Trench"))
    assert result[i - 1] == ("Lbefore", "T")
    assert result[i + 1] == ("Lafter", "T")


def test_add_with_a_consumed_anchor_is_refused(queue):
    item = queue.next()
    queue.mark_done(item, Status.Completed)
    before = len(queue.items)
    assert queue.add("L3", "Polishing", before=item.id) is None
    assert len(queue.items) == before


def test_add_duplicate_of_completed_pair_is_rerun(queue):
    """Re-run is expressed as adding a duplicate of a finished pair."""
    item = queue.next()
    queue.mark_done(item, Status.Completed)
    queue.add(item.lamella_name, item.task_name, front=True)
    nxt = queue.next()
    assert (nxt.lamella_name, nxt.task_name) == (item.lamella_name, item.task_name)
    assert nxt.id != item.id


# ── remove ────────────────────────────────────────────────────────────────────

def test_remove_marks_rather_than_deletes(queue):
    target = queue.pending[2]
    before = len(queue.items)
    assert queue.remove(target.id).ok
    assert len(queue.items) == before
    assert statuses(queue)[2] is Status.Removed


def test_remove_refuses_the_active_item(queue):
    active = queue.next()
    result = queue.remove(active.id)
    assert result.op is QueueOp.NOT_PENDING
    assert statuses(queue)[0] is Status.InProgress


def test_remove_refuses_completed_and_already_removed(queue):
    item = queue.next()
    queue.mark_done(item, Status.Completed)
    assert queue.remove(item.id).op is QueueOp.NOT_PENDING

    target = queue.pending[0]
    assert queue.remove(target.id).ok
    assert queue.remove(target.id).op is QueueOp.NOT_PENDING


def test_remove_unknown_id(queue):
    assert queue.remove("nope").op is QueueOp.NOT_FOUND


def test_removed_item_keeps_its_position(queue):
    """A struck-through row should stay where the user saw it."""
    target = queue.pending[1]
    queue.remove(target.id)
    assert pairs(queue)[1] == (target.lamella_name, target.task_name)


def test_counts_excludes_removed_from_pending_but_not_from_total(queue):
    """counts() feeds the hook payload's tasks_remaining / tasks_total.

    tasks_remaining must drop when work is removed — it is what the failure toast
    tells someone at the instrument is still coming. tasks_total must not:
    test_hook_run_context_is_a_snapshot_not_a_live_view asserts it is a fixed
    denominator across the run, and a total that shrank mid-run would contradict
    that. Same reason the timeline header counts Removed toward done — otherwise
    the counter can never reach its total.
    """
    queue.remove(queue.pending[0].id)
    assert queue.counts == (3, 4)


# ── moves ─────────────────────────────────────────────────────────────────────

def test_move_to_front_runs_it_next(queue):
    target = queue.pending[3]
    assert queue.move_to_front(target.id).ok
    assert queue.next().id == target.id


def test_move_to_front_stays_behind_the_active_item(queue):
    active = queue.next()
    target = queue.pending[-1]
    queue.move_to_front(target.id)
    assert pairs(queue)[0] == (active.lamella_name, active.task_name)
    assert pairs(queue)[1] == (target.lamella_name, target.task_name)


def test_move_to_back(queue):
    target = queue.pending[0]
    assert queue.move_to_back(target.id).ok
    assert pairs(queue)[-1] == (target.lamella_name, target.task_name)


def test_move_before_and_after(queue):
    p = queue.pending
    assert queue.move_before(p[3].id, p[0].id).ok
    assert pending_pairs(queue)[0] == (p[3].lamella_name, p[3].task_name)

    p = queue.pending
    assert queue.move_after(p[0].id, p[2].id).ok
    assert pending_pairs(queue)[2] == (p[0].lamella_name, p[0].task_name)


def test_move_before_forward_lands_immediately_before_target(queue):
    """Moving an earlier item forward must not overshoot by one."""
    p = queue.pending
    assert queue.move_before(p[0].id, p[3].id).ok
    result = pending_pairs(queue)
    i = result.index((p[3].lamella_name, p[3].task_name))
    assert result[i - 1] == (p[0].lamella_name, p[0].task_name)


def test_move_with_a_consumed_anchor_is_refused(queue):
    item = queue.next()
    target = queue.pending[0]
    before = pairs(queue)
    assert queue.move_before(target.id, item.id).op is QueueOp.BAD_ANCHOR
    assert pairs(queue) == before


def test_move_refuses_non_pending_and_unknown(queue):
    active = queue.next()
    anchor = queue.pending[0]
    assert queue.move_before(active.id, anchor.id).op is QueueOp.NOT_PENDING
    assert queue.move_to_front("nope").op is QueueOp.NOT_FOUND


# ── nudge ─────────────────────────────────────────────────────────────────────

def test_nudge_moves_one_place(queue):
    p = queue.pending
    assert queue.nudge(p[2].id, -1).ok
    assert pending_pairs(queue)[1] == (p[2].lamella_name, p[2].task_name)
    assert queue.nudge(p[2].id, +1).ok
    assert pending_pairs(queue)[2] == (p[2].lamella_name, p[2].task_name)


def test_nudge_clamps_at_both_ends(queue):
    p = queue.pending
    assert queue.nudge(p[0].id, -1).op is QueueOp.NO_OP
    assert queue.nudge(p[-1].id, +1).op is QueueOp.NO_OP
    assert pending_pairs(queue) == [(i.lamella_name, i.task_name) for i in p]


def test_nudge_skips_over_removed_items(queue):
    p = queue.pending
    queue.remove(p[1].id)
    assert queue.nudge(p[0].id, +1).ok
    # p[0] moved past the removed item to sit after the next runnable one
    assert pending_pairs(queue)[0] == (p[2].lamella_name, p[2].task_name)
    assert pending_pairs(queue)[1] == (p[0].lamella_name, p[0].task_name)


def test_nudge_refuses_the_active_item(queue):
    active = queue.next()
    assert queue.nudge(active.id, +1).op is QueueOp.NOT_PENDING


# ── reorder_pending ───────────────────────────────────────────────────────────

def test_reorder_pending_with_partial_list_keeps_everything(queue):
    """Regression: the old replacement-based reorder() destroyed unnamed items."""
    p = queue.pending
    assert queue.reorder_pending([p[3].id]).ok
    assert len(queue.items) == 4
    result = pending_pairs(queue)
    assert result[0] == (p[3].lamella_name, p[3].task_name)
    # the unnamed items follow, in their original relative order
    assert result[1:] == [(i.lamella_name, i.task_name) for i in (p[0], p[1], p[2])]


def test_reorder_pending_full_list(queue):
    p = list(reversed(queue.pending))
    assert queue.reorder_pending([i.id for i in p]).ok
    assert pending_pairs(queue) == [(i.lamella_name, i.task_name) for i in p]


def test_reorder_pending_ignores_unknown_and_duplicate_ids(queue):
    p = queue.pending
    assert queue.reorder_pending([p[1].id, "nope", p[1].id]).ok
    assert len(queue.items) == 4
    assert pending_pairs(queue)[0] == (p[1].lamella_name, p[1].task_name)


def test_reorder_pending_leaves_history_and_active_pinned(queue):
    done = queue.next()
    queue.mark_done(done, Status.Completed)
    active = queue.next()
    queue.reorder_pending([i.id for i in reversed(queue.pending)])
    assert pairs(queue)[0] == (done.lamella_name, done.task_name)
    assert pairs(queue)[1] == (active.lamella_name, active.task_name)
    assert statuses(queue)[:2] == [Status.Completed, Status.InProgress]


# ── invariants ────────────────────────────────────────────────────────────────

def test_no_move_ever_changes_the_item_count(queue):
    """The single rule that would have caught the original data-loss bug."""
    rng = random.Random(1234)
    ids = [i.id for i in queue.items]
    for _ in range(200):
        op = rng.choice(["front", "back", "before", "after", "nudge", "reorder"])
        a, b = rng.choice(ids), rng.choice(ids)
        if op == "front":
            queue.move_to_front(a)
        elif op == "back":
            queue.move_to_back(a)
        elif op == "before":
            queue.move_before(a, b)
        elif op == "after":
            queue.move_after(a, b)
        elif op == "nudge":
            queue.nudge(a, rng.choice([-2, -1, 1, 2]))
        else:
            sample = rng.sample(ids, rng.randint(0, len(ids)))
            queue.reorder_pending(sample)
        assert len(queue.items) == 4
        assert sorted(i.id for i in queue.items) == sorted(ids)


def test_history_is_unreachable_from_every_mutation(queue):
    done = queue.next()
    queue.mark_done(done, Status.Completed)
    active = queue.next()
    target = queue.pending[-1]

    for call in (
        lambda: queue.move_to_front(target.id),
        lambda: queue.move_before(target.id, done.id),
        lambda: queue.move_after(target.id, active.id),
        lambda: queue.nudge(target.id, -5),
        lambda: queue.reorder_pending([target.id]),
    ):
        call()
        assert pairs(queue)[0] == (done.lamella_name, done.task_name)
        assert pairs(queue)[1] == (active.lamella_name, active.task_name)


# ── version ───────────────────────────────────────────────────────────────────

def test_version_increments_on_real_mutations(queue):
    target = queue.pending[-1]
    before = queue.version
    assert queue.move_to_front(target.id).ok
    assert queue.version > before


def test_version_unchanged_on_no_op_and_refusal(queue):
    p = queue.pending
    before = queue.version
    assert queue.nudge(p[0].id, -1).op is QueueOp.NO_OP
    assert queue.move_to_front("nope").op is QueueOp.NOT_FOUND
    assert queue.version == before


# ── snapshots / threading ─────────────────────────────────────────────────────

def test_items_and_pending_return_copies(queue):
    snapshot = queue.items
    snapshot[0].status = Status.Failed
    assert statuses(queue)[0] is Status.NotStarted
    assert queue.pending[0] is not queue._items[0]


def test_concurrent_mutation_during_consumption_loses_nothing():
    q = TaskQueue()
    q.build_from_matrix([f"T{t}" for t in range(5)], [f"L{n}" for n in range(20)])
    total = len(q.items)
    stop = threading.Event()
    seen = []

    def consume():
        while not stop.is_set():
            item = q.next()
            if item is None:
                return
            seen.append(item.id)
            q.mark_done(item, Status.Completed)

    def mutate():
        rng = random.Random(7)
        while not stop.is_set():
            snapshot = q.items          # deliberately stale by the time it's used
            if not snapshot:
                continue
            a, b = rng.choice(snapshot), rng.choice(snapshot)
            q.move_before(a.id, b.id)
            q.nudge(a.id, rng.choice([-1, 1]))
            q.reorder_pending([i.id for i in rng.sample(snapshot, 3)])

    consumer = threading.Thread(target=consume)
    mutator = threading.Thread(target=mutate, daemon=True)
    consumer.start()
    mutator.start()
    consumer.join(timeout=30)
    stop.set()
    mutator.join(timeout=5)

    assert not consumer.is_alive()
    assert len(q.items) == total            # nothing lost or duplicated
    assert len(set(seen)) == len(seen)      # nothing run twice
    assert len(seen) == total               # everything ran exactly once
    assert q.is_empty


def test_mutating_a_consumed_item_is_refused_not_misapplied(queue):
    """The stale-snapshot case: the UI acts on a row the worker just took."""
    snapshot = queue.items
    consumed = queue.next()
    stale = next(i for i in snapshot if i.id == consumed.id)
    assert queue.move_to_front(stale.id).op is QueueOp.NOT_PENDING
    assert queue.remove(stale.id).op is QueueOp.NOT_PENDING
