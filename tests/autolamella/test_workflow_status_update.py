"""The typed record carried on `workflow_update_signal`'s `status` key (FIB-827).

Nothing emits one yet -- `TaskManager._emit_status` still builds a dict, and both
consumers decode through `from_payload`, which accepts either. What is worth pinning
before the producer flips is the shape, and in particular the two properties that are
cheap to get wrong and expensive to notice: a record that cannot be hashed, and a
decode that can raise inside a Qt slot.
"""

import dataclasses
import pathlib

import pytest

from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus
from fibsem.applications.autolamella.workflows.tasks.queue import WorkItem
from fibsem.applications.autolamella.workflows.tasks.status import WorkflowStatusUpdate


class TestTheDecodeIsTotal:
    """`from_payload` runs inside a queued Qt slot.

    PyQt5 turns any exception escaping such a slot into `qFatal`: the process aborts
    mid-run and the abort reaches no logfile (FIB-329). This signal has already killed
    the app that way once, over a missing `msg` key. So these are not politeness tests
    -- each one is a payload that must not be able to take the application down.
    """

    @pytest.mark.parametrize(
        "payload",
        [None, {}, {"status": {}}, {"task_name": None}, {"queue_items": None}],
    )
    def test_a_partial_payload_decodes_rather_than_raising(self, payload):
        report = WorkflowStatusUpdate.from_payload(payload)
        assert isinstance(report, WorkflowStatusUpdate)

    def test_an_unrecognised_status_becomes_notstarted(self):
        """A neutral value, chosen because it fires no consumer branch. Rendering
        "nothing in particular happened" beats crashing or picking a wrong outcome."""
        report = WorkflowStatusUpdate.from_payload({"status": "info"})

        assert report.status is AutoLamellaTaskStatus.NotStarted

    def test_a_typed_report_passes_straight_through(self):
        original = WorkflowStatusUpdate(task_name="Trench")

        assert WorkflowStatusUpdate.from_payload(original) is original


class TestTheShape:
    def test_a_report_carrying_queue_items_can_be_hashed(self):
        """`eq=False` is load-bearing, and for a different reason than `TiledProgress`.

        Three fields hold lists and `WorkItem` is a mutable dataclass, so it is
        unhashable. Left as `eq=True`, `frozen=True` would generate a `__hash__` over
        every field that raises `TypeError` here. There is no numpy involved -- if the
        tiled contract's reason is what you came looking for, this is not it.
        """
        report = WorkflowStatusUpdate(
            queue_items=[WorkItem(lamella_name="L1", task_name="Trench")]
        )

        assert {report, report} == {report}

    def test_a_work_item_really_is_unhashable(self):
        """The hazard lives in `WorkItem`, not in this test's imagination. If this ever
        stops raising, `eq=False` has lost its reason and the comment is wrong."""
        with pytest.raises(TypeError):
            hash(WorkItem(lamella_name="L1", task_name="Trench"))

    def test_a_report_cannot_be_written_to(self):
        report = WorkflowStatusUpdate()

        with pytest.raises(dataclasses.FrozenInstanceError):
            report.task_name = "Undercut"

    def test_lamella_name_is_derived_not_stored(self):
        """A property, so the deprecated alias cannot drift from `item_name` and
        deletes cleanly with the HookContext shims after v0.6."""
        report = WorkflowStatusUpdate(item_name="L99")

        assert report.lamella_name == "L99"
        assert "lamella_name" not in {f.name for f in dataclasses.fields(report)}


class TestTheQueueSnapshot:
    """`None` and `[]` mean different things, and the timeline acts on the difference.

    The dict carried that distinction for free -- an absent key read as `None`, an empty
    queue read as `[]`. A record defaulting the field to `[]` would collapse them, and
    the visible symptom is stale rows left on screen when a queue empties.
    """

    def test_an_absent_snapshot_is_none_not_empty(self):
        assert WorkflowStatusUpdate.from_payload({}).queue_items is None

    def test_an_empty_snapshot_stays_empty(self):
        report = WorkflowStatusUpdate.from_payload({"queue_items": []})

        assert report.queue_items == []
        assert report.queue_items is not None

    def test_the_timeline_leaves_rows_alone_without_a_snapshot(self):
        """ "No snapshot" means "this payload says nothing about the queue".

        Read as source text rather than imported. This file lives in
        `tests/autolamella/`, which runs in CI *without* the `[ui]` extra, and
        `workflow_timeline_widget` imports PyQt5 at module level -- importing it here
        passes locally and fails on every CI build.
        """
        import fibsem

        source = (
            pathlib.Path(fibsem.__file__).parent
            / "applications"
            / "autolamella"
            / "ui"
            / "workflow_timeline_widget.py"
        ).read_text()
        body = source[source.index("def update_from_status") :]
        body = body[: body.index("\n    def ", 1)]

        assert "queue_items is None" in body, (
            "update_from_status must distinguish an absent snapshot from an empty one; "
            "a falsy check collapses them and leaves stale rows when a queue empties"
        )


class TestQueuePosition:
    def test_it_is_already_one_based_and_is_not_adjusted(self):
        """The producer writes `position + 1`. Deliberately unlike milling's 0-based
        `current_stage`, which seven consumers each increment -- porting a `display_*`
        helper here would double-count."""
        report = WorkflowStatusUpdate.from_payload(
            {"queue_position": 1, "queue_total": 4}
        )

        assert (report.queue_position, report.queue_total) == (1, 4)

    def test_an_absent_position_stays_absent(self):
        """`None` means "not in the queue", which is not the same as position 0."""
        assert WorkflowStatusUpdate.from_payload({}).queue_position is None
