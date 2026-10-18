"""Items that repeat their tasks run in rounds, one round per item at a time.

A volume position repeats its tasks once per slice. The manager queues only an
item's current round, and asks the app for the next once it drains, so a hold
on one position's round parks that position while the others go on.
"""

from types import SimpleNamespace
from typing import List, Optional, Tuple

from fibsem.workflows.manager import BaseTaskManager
from fibsem.workflows.queue import WorkItem
from fibsem.workflows.status import TaskStatus

TASKS = ["Mill", "Image"]


class _Rounds(BaseTaskManager):
    def __init__(self, rounds: int, held: Optional[Tuple[str, int]] = None):
        super().__init__(microscope=None, experiment=SimpleNamespace(id="e", name="e"))
        self.rounds = rounds
        self.held = held
        self.asked: List[Tuple[str, Optional[int]]] = []

    def _next_round(self, item_name, finished):
        self.asked.append((item_name, finished))
        following = 0 if finished is None else finished + 1
        return (following, TASKS) if following < self.rounds else None

    def _item_defer_reason(self, item: WorkItem) -> Optional[str]:
        if self.held == (item.item_name, item.round):
            return "awaiting_decision"
        return None


def _drain(manager: BaseTaskManager) -> List[Tuple[str, str, Optional[int]]]:
    ran = []
    while True:
        item = manager._next_item()
        if item is None:
            return ran
        # one round per item in the queue, the one running included
        for name in {i.item_name for i in manager.queue.items}:
            waiting = {
                i.round
                for i in manager.queue.items
                if i.item_name == name
                and (i.is_pending or i.status is TaskStatus.InProgress)
            }
            assert len(waiting) <= 1, (name, waiting)
        ran.append((item.item_name, item.task_name, item.round))
        manager.queue.mark_done(item, TaskStatus.Completed)


def test_each_item_runs_its_rounds_in_turn():
    manager = _Rounds(rounds=3)
    for name in ("p1", "p2"):
        for task in TASKS:
            manager.queue.add(name, task, round=0)

    ran = _drain(manager)

    assert ran == [
        (name, task, r) for r in range(3) for name in ("p1", "p2") for task in TASKS
    ]
    # asked once per drained round, and once to hear there are no more
    assert manager.asked.count(("p1", 2)) == 1


def test_a_held_round_parks_its_item_while_the_others_go_on():
    manager = _Rounds(rounds=3, held=("p1", 1))
    for name in ("p1", "p2"):
        for task in TASKS:
            manager.queue.add(name, task, round=0)

    ran = _drain(manager)

    assert [r for name, _, r in ran if name == "p1"] == [0, 0]
    assert [r for name, _, r in ran if name == "p2"] == [0, 0, 1, 1, 2, 2]
    # p1's held round is still queued, and no later one behind it
    assert {(i.item_name, i.round) for i in manager.queue.pending} == {("p1", 1)}


def test_items_without_rounds_run_as_queued():
    manager = BaseTaskManager(
        microscope=None, experiment=SimpleNamespace(id="e", name="e")
    )
    manager.queue.build_from_pairs([("l1", "Trench"), ("l2", "Trench")])

    assert _drain(manager) == [("l1", "Trench", None), ("l2", "Trench", None)]


def test_a_status_report_names_its_queue_entry_and_round():
    reports = []
    window = SimpleNamespace(
        workflow_status_signal=SimpleNamespace(
            emit=lambda event: reports.append(event.report)
        ),
        queue_changed_signal=SimpleNamespace(emit=lambda _payload: None),
    )
    manager = _Rounds(rounds=1)
    manager.parent_ui = window
    entry = manager.queue.add("p1", "Mill", round=0)

    manager._emit_report(entry, "p1", TaskStatus.InProgress)

    assert reports[-1].work_item_id == entry.id
    assert reports[-1].round == 0


def test_a_run_knows_its_round():
    from fibsem.workflows.task import BaseTask

    class _Task(BaseTask):
        review = False

        def _run(self):
            pass

        def pre_task(self):
            pass

        def post_task(self):
            pass

    manager = _Rounds(rounds=2)
    manager.queue.add("p1", "Mill", round=1)
    manager._next_item()
    config = SimpleNamespace(task_name="Mill", task_type="MILL", display_name="Mill")
    item = SimpleNamespace(name="p1", id="p1", task_state=SimpleNamespace())

    task = _Task(None, config, item, task_manager=manager)
    task._start_task_state()

    assert task.round == 1
    assert item.task_state.round == 1
    assert manager.hook_run_context()["round"] == 1


def test_an_image_says_which_round_took_it():
    from fibsem.structures import FibsemExperimentRef

    ref = FibsemExperimentRef(id="e", name="e")
    ref.set_workflow_metadata(item_id="p1", task_name="Mill", round=4)

    assert FibsemExperimentRef.from_dict(ref.to_dict()).round == 4
    ref.clear_workflow_metadata()
    assert ref.round is None
