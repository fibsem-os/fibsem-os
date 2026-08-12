"""The thumbnail is on disk before anything is told the task finished.

`post_task` set `task_state.status = Completed` and wrote the thumbnail afterwards. The
lamella card subscribes to `task_state.events.status`, and its handler is
`@ensure_main_thread` — so the status assignment queued a read onto the GUI thread while
this thread carried on to write the file. The read landed inside the write:

    OSError: image file is truncated
      lamella_card_widget.py:344 in refresh -> lamella.get_thumbnail()

Structural rather than unlucky: it happens on the same two adjacent statements at the end
of every task.

A quieter consequence of the same ordering: the refresh fired *before* the write, so the
card read the **previous** thumbnail. The picture from the task that had just finished was
never the one displayed.

The write is guarded, because ordering it first would otherwise let a thumbnail failure
stop a finished task being marked Completed — a much worse failure than a missing picture.

The atomic write in `save_thumbnail` is the other half and is kept: it holds the property
independent of anyone's call ordering, including future callers.
"""

import ast
import inspect
import textwrap

import pytest


def _post_task_body():
    from fibsem.applications.autolamella.workflows.tasks.base import AutoLamellaTask

    return ast.parse(textwrap.dedent(inspect.getsource(AutoLamellaTask.post_task)))


def _statement_index(tree, predicate) -> int:
    """Index of the first top-level statement in `post_task` matching *predicate*."""
    body = tree.body[0].body
    for i, node in enumerate(body):
        if predicate(ast.dump(node)):
            return i
    return -1


class TestTheOrder:
    def test_the_thumbnail_is_written_before_the_status_is_set(self):
        tree = _post_task_body()

        thumbnail = _statement_index(tree, lambda d: "save_thumbnail" in d)
        status = _statement_index(
            tree, lambda d: "AutoLamellaTaskStatus" in d and "Completed" in d
        )

        assert thumbnail != -1, "post_task no longer saves a thumbnail"
        assert status != -1, "post_task no longer marks the task Completed"
        assert thumbnail < status, (
            "the thumbnail is written after the status is set, so the card's refresh "
            "(queued to the GUI thread by that assignment) races the write and reads "
            "the previous thumbnail even when it wins"
        )

    def test_the_write_is_guarded(self):
        """Ordering it first must not let a thumbnail failure block task completion."""
        tree = _post_task_body()
        body = tree.body[0].body

        guarded = any(
            isinstance(node, ast.Try) and "save_thumbnail" in ast.dump(node)
            for node in ast.walk(tree)
        )
        assert guarded, (
            "the thumbnail write is not wrapped — a failure would now stop a finished "
            "task being marked Completed, which is worse than a missing picture"
        )
        assert body, "post_task is empty"


@pytest.fixture
def task(tmp_path):
    """A real task, run through `post_task`. Only `_run` is abstract."""
    from fibsem import utils
    from fibsem.applications.autolamella.structures import (
        AutoLamellaTaskState,
        Lamella,
    )
    # A real, registered config: `task_type` is a ClassVar the abstract base leaves
    # unset, so the base class cannot be instantiated on its own.
    from fibsem.applications.autolamella.workflows.tasks import MillTrenchTaskConfig
    from fibsem.applications.autolamella.workflows.tasks.base import AutoLamellaTask

    class _Task(AutoLamellaTask):
        def _run(self):  # pragma: no cover - never called here
            raise NotImplementedError

    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    lamella = Lamella(path=str(tmp_path / "lam"), number=1, petname="test")
    lamella.task_state = AutoLamellaTaskState(name="TestTask")
    return _Task(
        microscope=microscope,
        config=MillTrenchTaskConfig(task_name="TestTask"),
        lamella=lamella,
    )


class TestItActuallyBehaves:
    def test_the_thumbnail_is_on_disk_before_the_status_changes(self, task):
        """What the card sees. The status change is what triggers its read."""
        import numpy as np

        from fibsem.structures import FibsemImage

        task._last_fib_image = FibsemImage(data=np.full((32, 32), 9, np.uint8))
        seen = []
        task.lamella.task_state.events.status.connect(
            lambda *_: seen.append(int(task.lamella.get_thumbnail()[0, 0, 0]))
        )

        task.post_task()

        assert seen == [9], (
            f"the subscriber saw {seen} — the thumbnail from the task that just "
            f"finished should already be the one on disk when the status is announced"
        )

    def test_a_failing_thumbnail_does_not_stop_the_task_completing(self, task, monkeypatch):
        from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus

        def exploding(self, image):
            raise RuntimeError("disk went away")

        monkeypatch.setattr(type(task.lamella), "save_thumbnail", exploding)
        task._last_fib_image = object()  # never reaches PIL

        task.post_task()

        assert task.lamella.task_state.status is AutoLamellaTaskStatus.Completed, (
            "a thumbnail failure stopped a finished task being marked Completed — the "
            "picture is worth far less than an accurate task state"
        )
