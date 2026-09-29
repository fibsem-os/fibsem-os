"""A GUI worker runs with the context it was started from (FIB-1062).

``fibsem.acting`` marks who is acting as a context variable, and a new thread
starts without one. ``FunctionWorker`` -- every ``thread_worker`` site, the
milling and spot-burn widgets among them -- copies the starting thread's
context into its body, so work started for a task is the task's in the
experiment's record.
"""

import pytest

pytest.importorskip("PyQt5")

from fibsem.acting import TASK, acting, current_actor  # noqa: E402
from fibsem.ui.qt.threading import FunctionWorker, thread_worker  # noqa: E402


def _run(worker):
    worker.start()
    worker.join(5)


def test_a_worker_started_for_a_task_works_for_it():
    seen = []
    with acting(TASK):
        _run(FunctionWorker(lambda: seen.append(current_actor())))
    assert seen == [TASK]


def test_a_worker_started_from_the_gui_is_unmarked():
    seen = []
    _run(FunctionWorker(lambda: seen.append(current_actor())))
    assert seen == [None]


def test_a_thread_worker_site_is_marked_by_whoever_starts_it():
    """Created unmarked, started marked: the mark is taken when it starts."""
    seen = []

    @thread_worker
    def body():
        seen.append(current_actor())

    worker = body()
    with acting(TASK):
        _run(worker)
    assert seen == [TASK]
