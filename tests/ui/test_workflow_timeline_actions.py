"""Headless tests for the timeline's queue-actions menu (FIB-476).

Every row gets the same menu; what varies is which entries are enabled. A row
with no menu would teach nothing about why it has no menu, so the rule is shown
rather than inferred.

Enablement is computed when the menu is built, not cached on the row — the
worker may have consumed the row since the last paint. That still leaves a race
between the menu opening and the click, which the owner handles via QueueResult.

Uses the shared offscreen ``qapp`` fixture from tests/conftest.py.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus as Status
from fibsem.applications.autolamella.workflows.tasks.queue import QueueOp, TaskQueue
from fibsem.applications.autolamella.ui.workflow_timeline_widget import WorkflowProgressWidget


@pytest.fixture
def queue() -> TaskQueue:
    q = TaskQueue()
    q.build_from_matrix(["Trench", "Undercut"], ["L1", "L2"])
    return q


@pytest.fixture
def widget(qapp, queue) -> WorkflowProgressWidget:
    """A timeline mid-run: item 0 finished, item 1 running, 2 and 3 pending."""
    w = WorkflowProgressWidget()
    done = queue.next()
    queue.mark_done(done, Status.Completed)
    queue.next()  # marks item 1 InProgress
    w.set_workflow(queue.items)
    w.set_actions_enabled(True)
    return w


def actions(widget: WorkflowProgressWidget, index: int) -> dict:
    """{action_id: enabled} for the menu on row *index*."""
    menu = widget.build_row_menu(index)
    return {a.data(): a.isEnabled() for a in menu.actions() if a.data()}


# ── What each row state offers ────────────────────────────────────────────────

def test_a_pending_row_offers_every_move(widget):
    assert actions(widget, 2) == {
        "move_up": False,       # already the first pending item
        "move_down": True,
        "run_next": False,      # already runs next
        "remove": True,
    }


def test_the_last_pending_row_cannot_move_down(widget):
    assert actions(widget, 3) == {
        "move_up": True,
        "move_down": False,
        "run_next": True,
        "remove": True,
    }


def test_the_running_row_offers_only_stop_task(widget):
    """It cannot be moved or removed — but it can be abandoned (FIB-499). The
    greyed entries are still what explain the rest of the rule."""
    assert actions(widget, 1) == {
        "move_up": False, "move_down": False,
        "run_next": False, "remove": False,
        "stop_task": True,
    }


def test_the_running_row_says_why_the_rest_are_greyed(widget):
    """The tooltip explains a disabled entry, so it does not land on Stop Task."""
    menu = widget.build_row_menu(1)
    tips = {a.data(): a.toolTip() for a in menu.actions() if a.data()}

    assert tips["remove"] == "This task is already running"
    # Qt defaults an action's tooltip to its own text, so the check is that Stop
    # Task did not inherit the explanation meant for the disabled entries.
    assert tips["stop_task"] != tips["remove"]


def test_only_the_running_row_can_be_stopped(widget):
    """Nothing else is under way, so there is nothing to interrupt."""
    for index in (0, 2, 3):
        assert "stop_task" not in actions(widget, index)


def test_a_finished_row_offers_only_run_again(widget):
    assert actions(widget, 0) == {
        "move_up": False, "move_down": False,
        "run_next": False, "remove": False,
        "run_again": True,
    }


def test_run_again_is_absent_while_a_row_could_still_run(widget):
    """It would be meaningless on a queued row and wrong on the running one."""
    for index in (1, 2, 3):
        assert "run_again" not in actions(widget, index)


def test_a_skipped_row_can_be_run_again(widget, queue):
    """A skip is often a prerequisite that has since been satisfied."""
    item = queue.pending[0]
    queue.mark_done(item, Status.Skipped)
    widget.refresh_queue(queue.items)
    assert actions(widget, 2)["run_again"] is True


def test_no_menu_for_a_row_that_is_gone(widget):
    assert widget.build_row_menu(99) is None


# ── The signal the owner acts on ──────────────────────────────────────────────

def test_triggering_an_action_reports_it_against_the_row_id(widget, queue):
    seen = []
    widget.queue_action_requested.connect(lambda a, i: seen.append((a, i)))

    menu = widget.build_row_menu(3)
    next(a for a in menu.actions() if a.data() == "run_next").trigger()

    assert seen == [("run_next", widget._items[3].id)]


def test_the_id_survives_the_row_moving_under_the_menu(widget, queue):
    """The menu is keyed on identity, so a reorder between opening and clicking
    cannot redirect the action onto whatever slid into that position."""
    target = widget._items[3]
    menu = widget.build_row_menu(3)

    queue.move_to_front(target.id)
    widget.refresh_queue(queue.items)

    seen = []
    widget.queue_action_requested.connect(lambda a, i: seen.append((a, i)))
    next(a for a in menu.actions() if a.data() == "remove").trigger()

    assert seen == [("remove", target.id)]


# ── Actions off ───────────────────────────────────────────────────────────────

def test_the_button_appears_on_hover_and_leaves_again(widget):
    """Right-click alone is undiscoverable; a permanent button on every row of a
    dense timeline is noise. Hover is the compromise."""
    from PyQt5.QtCore import QEvent

    row = widget._outer._rows[2]
    assert not row._btn_actions.isVisibleTo(row)

    row.enterEvent(QEvent(QEvent.Enter))
    assert row._btn_actions.isVisibleTo(row)

    row.leaveEvent(QEvent(QEvent.Leave))
    assert not row._btn_actions.isVisibleTo(row)


def test_hovering_a_row_shows_nothing_when_no_run_is_live(widget):
    from PyQt5.QtCore import QEvent

    widget.set_actions_enabled(False)
    row = widget._outer._rows[2]
    row.enterEvent(QEvent(QEvent.Enter))
    assert not row._btn_actions.isVisibleTo(row)


def test_rows_hide_the_button_when_no_run_is_live(widget):
    widget.set_actions_enabled(False)
    assert all(not r._btn_actions.isVisible() for r in widget._outer._rows)
    assert all(not r._actions_enabled for r in widget._outer._rows)


def test_rows_added_later_inherit_the_current_setting(widget, queue):
    queue.add("L9", "Polish")
    widget.refresh_queue(queue.items)
    assert widget._outer._rows[-1]._actions_enabled is True

    widget.set_actions_enabled(False)
    queue.add("L8", "Polish")
    widget.refresh_queue(queue.items)
    assert widget._outer._rows[-1]._actions_enabled is False


# ── The queue calls the menu maps onto ────────────────────────────────────────

def test_each_action_does_what_the_menu_says(queue):
    """Guards the mapping in AutoLamellaMainUI._on_queue_action."""
    queue.next()  # item 0 running; 1, 2, 3 pending
    last = queue.pending[-1]

    assert queue.nudge(last.id, -1).op is QueueOp.OK
    assert queue.pending[-2].id == last.id

    assert queue.move_to_front(last.id).op is QueueOp.OK
    assert queue.pending[0].id == last.id

    assert queue.nudge(last.id, -1).op is QueueOp.NO_OP  # already first
    assert queue.remove(last.id).op is QueueOp.OK
    assert last.id not in [i.id for i in queue.pending]


def test_acting_on_the_running_row_is_refused_not_misapplied(queue):
    """What the owner turns into "... is already running"."""
    active = queue.next()
    for result in (queue.nudge(active.id, -1), queue.move_to_front(active.id),
                   queue.remove(active.id)):
        assert result.op is QueueOp.NOT_PENDING


def test_run_again_queues_a_fresh_item_without_touching_the_original(queue):
    done = queue.next()
    queue.mark_done(done, Status.Failed)

    added = queue.add(done.lamella_name, done.task_name, front=True)

    assert added.id != done.id
    assert queue.pending[0].id == added.id
    assert queue.items[0].status is Status.Failed  # the original attempt stands


# ── What the user is told ─────────────────────────────────────────────────────

def describe(action: str, op: QueueOp) -> str:
    from fibsem.applications.autolamella.workflows.tasks.queue import QueueResult
    from fibsem.applications.autolamella.ui.AutoLamellaMainUI import (
        AutoLamellaSingleWindowUI,
    )
    return AutoLamellaSingleWindowUI._queue_action_message(
        action, "Trench for L1", QueueResult(op, 1)
    )


def test_every_successful_action_is_confirmed():
    """There is no modal on remove, so the status line is the only acknowledgement
    the user gets that a queued task is gone."""
    assert describe("remove", QueueOp.OK) == "Removed Trench for L1 from the queue."
    assert describe("run_next", QueueOp.OK) == "Trench for L1 will run next."
    assert describe("move_up", QueueOp.OK) == "Moved Trench for L1 up."
    assert describe("move_down", QueueOp.OK) == "Moved Trench for L1 down."


def test_a_row_that_started_running_says_so_rather_than_nothing():
    assert describe("remove", QueueOp.NOT_PENDING) == "Trench for L1 is already running."


def test_hitting_the_end_of_the_queue_says_nothing():
    """Already first or last — the user can see that; a message would be noise."""
    assert describe("move_up", QueueOp.NO_OP) == ""


def test_a_vanished_row_is_reported():
    assert describe("move_up", QueueOp.NOT_FOUND) == "Trench for L1 is no longer in the queue."


def test_the_dark_theme_greys_disabled_menu_items():
    """Load-bearing for the whole menu design, and easy to lose in a restyle.

    NAPARI_STYLE sets a colour on QMenu, which also paints disabled items unless
    an explicit :disabled rule overrides it — so without this a greyed action is
    pixel-identical to an available one, and the menu teaches nothing.
    """
    from fibsem.ui.stylesheets import NAPARI_STYLE
    assert "QMenu::item:disabled" in NAPARI_STYLE
