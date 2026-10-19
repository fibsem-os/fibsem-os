"""Items that repeat their tasks run in iterations, one iteration per item at a time.

A volume position repeats its tasks once per slice. The manager queues only an
item's current iteration, and asks the app for the next once it drains, so a hold
on one position's iteration parks that position while the others go on.
"""

from types import SimpleNamespace
from typing import List, Optional, Tuple

from fibsem.workflows.manager import BaseTaskManager
from fibsem.workflows.queue import WorkItem
from fibsem.workflows.status import TaskStatus

TASKS = ["Mill", "Image"]


class _Rounds(BaseTaskManager):
    def __init__(self, iterations: int, held: Optional[Tuple[str, int]] = None):
        super().__init__(microscope=None, experiment=SimpleNamespace(id="e", name="e"))
        self.iterations = iterations
        self.held = held
        self.asked: List[Tuple[str, Optional[int]]] = []

    def _next_iteration(self, item_name, finished):
        self.asked.append((item_name, finished))
        following = 0 if finished is None else finished + 1
        return (following, TASKS) if following < self.iterations else None

    def _item_defer_reason(self, item: WorkItem) -> Optional[str]:
        if self.held == (item.item_name, item.iteration):
            return "awaiting_decision"
        return None


def _drain(manager: BaseTaskManager) -> List[Tuple[str, str, Optional[int]]]:
    ran = []
    while True:
        item = manager._next_item()
        if item is None:
            return ran
        # one iteration per item in the queue, the one running included
        for name in {i.item_name for i in manager.queue.items}:
            waiting = {
                i.iteration
                for i in manager.queue.items
                if i.item_name == name
                and (i.is_pending or i.status is TaskStatus.InProgress)
            }
            assert len(waiting) <= 1, (name, waiting)
        ran.append((item.item_name, item.task_name, item.iteration))
        manager.queue.mark_done(item, TaskStatus.Completed)


def test_each_item_runs_its_rounds_in_turn():
    manager = _Rounds(iterations=3)
    for name in ("p1", "p2"):
        for task in TASKS:
            manager.queue.add(name, task, iteration=0)

    ran = _drain(manager)

    assert ran == [
        (name, task, r) for r in range(3) for name in ("p1", "p2") for task in TASKS
    ]
    # asked once per drained iteration, and once to hear there are no more
    assert manager.asked.count(("p1", 2)) == 1


def test_a_held_round_parks_its_item_while_the_others_go_on():
    manager = _Rounds(iterations=3, held=("p1", 1))
    for name in ("p1", "p2"):
        for task in TASKS:
            manager.queue.add(name, task, iteration=0)

    ran = _drain(manager)

    assert [r for name, _, r in ran if name == "p1"] == [0, 0]
    assert [r for name, _, r in ran if name == "p2"] == [0, 0, 1, 1, 2, 2]
    # p1's held iteration is still queued, and no later one behind it
    assert {(i.item_name, i.iteration) for i in manager.queue.pending} == {("p1", 1)}


def test_items_without_rounds_run_as_queued():
    manager = BaseTaskManager(
        microscope=None, experiment=SimpleNamespace(id="e", name="e")
    )
    manager.queue.build_from_pairs([("l1", "Trench"), ("l2", "Trench")])

    assert _drain(manager) == [("l1", "Trench", None), ("l2", "Trench", None)]


def test_a_status_report_names_its_queue_entry_and_iteration():
    reports = []
    manager = _Rounds(iterations=1)
    manager.status.connect(lambda event: reports.append(event.report))
    entry = manager.queue.add("p1", "Mill", iteration=0)

    manager._emit_report(entry, "p1", TaskStatus.InProgress)

    assert reports[-1].work_item_id == entry.id
    assert reports[-1].iteration == 0


def test_a_run_knows_its_iteration():
    from fibsem.workflows.task import BaseTask

    class _Task(BaseTask):
        supervised = False

        def _run(self):
            pass

        def pre_task(self):
            pass

        def post_task(self):
            pass

    manager = _Rounds(iterations=2)
    manager.queue.add("p1", "Mill", iteration=1)
    manager._next_item()
    config = SimpleNamespace(task_name="Mill", task_type="MILL", display_name="Mill")
    item = SimpleNamespace(name="p1", id="p1", task_state=SimpleNamespace())

    task = _Task(None, config, item, task_manager=manager)
    task._start_task_state()

    assert task.iteration == 1
    assert item.task_state.iteration == 1
    assert manager.hook_run_context()["iteration"] == 1


def test_an_image_says_which_round_took_it():
    from fibsem.structures import FibsemExperimentRef

    ref = FibsemExperimentRef(id="e", name="e")
    ref.set_workflow_metadata(item_id="p1", task_name="Mill", iteration=4)

    assert FibsemExperimentRef.from_dict(ref.to_dict()).iteration == 4
    ref.clear_workflow_metadata()
    assert ref.iteration is None
