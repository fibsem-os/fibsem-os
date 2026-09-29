"""Who is acting on the microscope, per thread of control (FIB-1062)."""

import threading

from fibsem.acting import AGENT, TASK, acting, current_actor


def test_a_mark_holds_inside_its_block_and_is_restored_after():
    assert current_actor() is None
    with acting(TASK):
        assert current_actor() == TASK
        with acting(AGENT):
            assert current_actor() == AGENT
        assert current_actor() == TASK
    assert current_actor() is None


def test_a_mark_is_restored_when_the_block_raises():
    try:
        with acting(TASK):
            raise RuntimeError("the task failed")
    except RuntimeError:
        pass
    assert current_actor() is None


def test_a_mark_does_not_reach_another_thread():
    # A task's mark must not make a UI worker running beside it the task's.
    seen = []
    with acting(TASK):
        worker = threading.Thread(target=lambda: seen.append(current_actor()))
        worker.start()
        worker.join()
    assert seen == [None]


def test_as_a_decorator_it_marks_each_call():
    @acting(TASK)
    def run():
        return current_actor()

    assert run() == TASK and run() == TASK
    assert current_actor() is None
