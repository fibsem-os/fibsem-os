"""The estimate column and the header's remaining/finish line (FIB-666).

The arithmetic itself is tested in tests/autolamella/test_workflow_estimate.py, without a
widget. What is checked here is what the timeline *says*, and when it changes its mind:
the column predicts while it can and reports once it cannot, and the header quotes a clock
only while a clock can honestly be quoted.

`NAPARI_STYLE` is applied to the QApplication deliberately. The app sets it, and its bare
`QWidget { background-color: ... }` rule reaches every descendant -- a render without it
cannot show a label painting a block over the row's selection tint, which is precisely the
class of bug that got through twice on the pre-flight dialog.

Uses the shared offscreen ``qapp`` fixture from tests/conftest.py.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from datetime import datetime, timedelta

from PyQt5.QtWidgets import QLabel

from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus as Status
from fibsem.applications.autolamella.ui.workflow_timeline_widget import (
    WorkflowProgressWidget,
)
from fibsem.applications.autolamella.workflows.tasks.queue import TaskQueue
from fibsem.ui.stylesheets import NAPARI_STYLE

START = 1000.0  # the fake clock every status payload is stamped with


@pytest.fixture
def app(qapp):
    qapp.setStyleSheet(NAPARI_STYLE)
    return qapp


@pytest.fixture
def queue() -> TaskQueue:
    q = TaskQueue()
    q.build_from_matrix(["Trench", "Undercut"], ["L1", "L2"])
    return q


@pytest.fixture
def widget(app, queue) -> WorkflowProgressWidget:
    """A live workflow: four items, every one of them estimated at 100 s."""
    w = WorkflowProgressWidget()
    w.set_actions_enabled(True)  # "a workflow is live"
    w.set_estimates({(i.lamella_name, i.task_name): 100.0 for i in queue.items})
    w.set_workflow(queue.items)
    return w


def status_for(queue, item, status, **extra) -> dict:
    payload = {
        "task_name": item.task_name,
        "item_name": item.lamella_name,
        "status": status,
        "queue_items": queue.items,
        "timestamp": START,
    }
    payload.update(extra)
    return payload


def start_next(widget, queue):
    item = queue.next()
    widget.update_from_status(status_for(queue, item, Status.InProgress))
    return item


def tick(widget, monkeypatch, seconds: float) -> None:
    """Advance the widget's clock to `seconds` after the active task started."""
    monkeypatch.setattr(
        "fibsem.applications.autolamella.ui.workflow_timeline_widget.time.time",
        lambda: START + seconds,
    )
    widget._update_elapsed()


def column(widget) -> list:
    return [s.trailing for s in widget._outer._steps]


# ── pending rows ──────────────────────────────────────────────────────────────


def test_every_pending_row_shows_its_estimate(widget):
    assert column(widget) == ["1m 40s"] * 4


def test_a_row_with_no_estimate_shows_nothing_rather_than_a_guess(app, queue):
    w = WorkflowProgressWidget()
    w.set_actions_enabled(True)
    w.set_estimates({("L1", "Trench"): 100.0})
    w.set_workflow(queue.items)
    assert column(w) == ["1m 40s", "", "", ""]


def test_estimates_arriving_after_the_rows_still_reach_them(app, queue):
    """The owner pushes them separately from the queue snapshot, so the order the two
    arrive in must not decide whether the column is ever filled."""
    w = WorkflowProgressWidget()
    w.set_actions_enabled(True)
    w.set_workflow(queue.items)
    assert column(w) == [""] * 4
    w.set_estimates({(i.lamella_name, i.task_name): 100.0 for i in queue.items})
    assert column(w) == ["1m 40s"] * 4


def test_an_item_added_mid_workflow_gets_its_estimate(widget, queue):
    added = queue.add("L9", "Polish")
    estimates = {(i.lamella_name, i.task_name): 100.0 for i in queue.items}
    estimates[("L9", "Polish")] = 250.0  # dict | dict is 3.9+; CI still runs 3.8
    widget.set_estimates(estimates)
    widget.refresh_queue(queue.items)
    assert added is not None
    assert column(widget)[-1] == "4m 10s"


# ── the running row ───────────────────────────────────────────────────────────


def test_the_running_row_counts_down(widget, queue, monkeypatch):
    start_next(widget, queue)
    tick(widget, monkeypatch, 40.0)
    assert column(widget)[0] == "1m 00s left"


def test_the_running_row_stops_predicting_once_the_estimate_is_spent(
    widget, queue, monkeypatch
):
    """It reports elapsed instead -- the same kind of number the row will show when it
    finishes. Never a negative, never a stalled zero, and never a new word for what is
    ordinary variance."""
    start_next(widget, queue)
    tick(widget, monkeypatch, 460.0)
    assert column(widget)[0] == "7m 40s"
    assert "left" not in column(widget)[0]


def test_the_elapsed_moves_out_of_the_subtitle_when_the_column_takes_it_over(
    widget, queue, monkeypatch
):
    """Both at once would be the same number twice."""
    start_next(widget, queue)
    tick(widget, monkeypatch, 40.0)
    assert widget._outer._steps[0].subtitle == "Trench (40s)"
    tick(widget, monkeypatch, 460.0)
    assert widget._outer._steps[0].subtitle == "Trench"


def test_a_running_row_with_no_estimate_just_shows_elapsed_in_the_subtitle(
    app, queue, monkeypatch
):
    w = WorkflowProgressWidget()
    w.set_actions_enabled(True)
    w.set_workflow(queue.items)
    start_next(w, queue)
    tick(w, monkeypatch, 40.0)
    assert column(w)[0] == ""
    assert w._outer._steps[0].subtitle == "Trench (40s)"


# ── supervised waits ──────────────────────────────────────────────────────────


def test_a_wait_for_a_human_pauses_the_countdown_rather_than_spending_it(
    widget, queue, monkeypatch
):
    """40 s of work then a 300 s pause leaves the same 60 s of work to do. Left running,
    the wait would spend the whole estimate with all of the task still ahead of it."""
    start_next(widget, queue)
    tick(widget, monkeypatch, 40.0)
    monkeypatch.setattr(
        "fibsem.applications.autolamella.ui.workflow_timeline_widget.time.time",
        lambda: START + 40.0,
    )
    widget.set_waiting_for_user(True)
    tick(widget, monkeypatch, 340.0)
    assert column(widget)[0] == "waiting"

    monkeypatch.setattr(
        "fibsem.applications.autolamella.ui.workflow_timeline_widget.time.time",
        lambda: START + 340.0,
    )
    widget.set_waiting_for_user(False)
    tick(widget, monkeypatch, 340.0)
    assert column(widget)[0] == "1m 00s left"


def test_the_elapsed_keeps_running_through_a_wait(widget, queue, monkeypatch):
    """Only the countdown pauses. `AutoLamellaTaskState.duration` is end minus start and
    counts the wait, so an elapsed that skipped it would snap on completion and disagree
    with the experiment record."""
    start_next(widget, queue)
    monkeypatch.setattr(
        "fibsem.applications.autolamella.ui.workflow_timeline_widget.time.time",
        lambda: START,
    )
    widget.set_waiting_for_user(True)
    tick(widget, monkeypatch, 300.0)
    assert widget._outer._steps[0].subtitle == "Trench (5m 00s)", "elapsed froze"
    widget.set_waiting_for_user(False)
    tick(widget, monkeypatch, 300.0)
    assert widget._outer._steps[0].subtitle == "Trench (5m 00s)"


def test_the_pause_belongs_to_one_task_not_the_workflow(widget, queue, monkeypatch):
    """A pause on the first task must not go on crediting the second."""
    first = start_next(widget, queue)
    widget.set_waiting_for_user(True)
    queue.mark_done(first, Status.Completed)
    widget.update_from_status(
        status_for(queue, first, Status.Completed, task_duration=50.0)
    )
    start_next(widget, queue)
    assert widget._paused_total == 0.0
    assert widget._waiting_for_user is False


# ── finished rows ─────────────────────────────────────────────────────────────


def test_a_finished_row_shows_what_it_actually_took(widget, queue):
    item = start_next(widget, queue)
    queue.mark_done(item, Status.Completed)
    widget.update_from_status(
        status_for(queue, item, Status.Completed, task_duration=112.0)
    )
    assert column(widget)[0] == "1m 52s"
    assert "1m 52s" not in widget._outer._steps[0].subtitle


def test_a_finished_row_is_not_overwritten_by_its_own_estimate(widget, queue):
    """`_refresh_trailing` runs on every sync; a row that has run has superseded the
    estimate and must keep the actual."""
    item = start_next(widget, queue)
    queue.mark_done(item, Status.Completed)
    widget.update_from_status(
        status_for(queue, item, Status.Completed, task_duration=112.0)
    )
    widget.refresh_queue(queue.items)
    assert column(widget)[0] == "1m 52s"


def test_the_two_clocks_in_the_panel_agree_on_their_format(widget, queue):
    """The completion stamp and the header's finish time sit inches apart, so `06:59PM`
    beside `7:13PM` would read as two different conventions."""
    item = start_next(widget, queue)
    queue.mark_done(item, Status.Completed)
    widget.update_from_status(
        status_for(
            queue, item, Status.Completed, task_duration=112.0, completed_at=START
        )
    )
    assert not widget._outer._steps[0].subtitle.split("· ")[-1].startswith("0")


# ── the header line ───────────────────────────────────────────────────────────


def test_the_header_quotes_what_is_left_and_when_it_lands(widget):
    text = widget._summary.text()
    # isHidden, not isVisible: nothing in this test is shown, and isVisible() is False
    # for every descendant of an unshown parent -- so it would answer the same whether
    # the line had been put up or not, and the negative tests below would pass on their
    # own. isHidden() reports the widget's own state.
    assert not widget._summary.isHidden()
    assert text.startswith("6m 40s left · finishes ")


def test_the_header_is_silent_when_no_workflow_is_live(app, queue):
    """The timeline stays on screen afterwards; a remaining time beside a queue that is
    not going to run would be a lie."""
    w = WorkflowProgressWidget()
    w.set_estimates({(i.lamella_name, i.task_name): 100.0 for i in queue.items})
    w.set_workflow(queue.items)
    assert w._summary.isHidden()


def test_the_header_drops_the_clock_while_waiting_for_a_human(
    widget, queue, monkeypatch
):
    """The wait is unbounded, so any finish time quoted through it is a guess dressed as
    a fact. The work still to do is knowable either way."""
    start_next(widget, queue)
    tick(widget, monkeypatch, 10.0)
    widget.set_waiting_for_user(True)
    text = widget._summary.text()
    assert "finishes" not in text
    assert text.endswith("of work left · waiting for you")


def test_a_scheduled_hold_is_inside_the_time_the_header_quotes(app, queue):
    """ "When do I come back" includes four hours of dead time in the middle."""
    w = WorkflowProgressWidget()
    w.set_actions_enabled(True)
    w.set_estimates({(i.lamella_name, i.task_name): 100.0 for i in queue.items})
    w.set_schedule({"Undercut": datetime.now() + timedelta(hours=4)})
    w.set_workflow(queue.items)
    assert w._summary.text().startswith("4h 0")


def test_the_header_goes_quiet_once_there_is_nothing_left_to_do(widget, queue):
    for _ in range(4):
        item = start_next(widget, queue)
        queue.mark_done(item, Status.Completed)
        widget.update_from_status(
            status_for(queue, item, Status.Completed, task_duration=100.0)
        )
    assert widget._summary.isHidden()


# ── the layout this column has to survive ─────────────────────────────────────


def test_a_long_name_elides_instead_of_pushing_the_column_off_screen(app):
    """The rows sit in a QScrollArea that is widget-resizable with horizontal scrolling
    off, so the contents can never be narrower than the widest row and anything past the
    viewport is unreachable. Before the labels elided, one long petname took the estimate
    off EVERY row: measured at 280px, the first row's estimate ended 59px off screen.
    """
    q = TaskQueue()
    q.build_from_matrix(["Trench"], ["L1", "a-very-long-petname-here-indeed"])
    w = WorkflowProgressWidget()
    w.set_actions_enabled(True)
    w.set_estimates({(i.lamella_name, i.task_name): 100.0 for i in q.items})
    w.set_workflow(q.items)
    w.resize(280, 200)
    w.show()
    app.processEvents()

    viewport = w._outer._scroll.viewport()
    assert w._outer._contents.minimumSizeHint().width() <= viewport.width()
    for row in w._outer._rows:
        edge = row._trailing.mapTo(viewport, row._trailing.rect().topRight()).x()
        assert edge <= viewport.width(), "the estimate column is off screen"
    long_row = w._outer._rows[1]
    assert QLabel.text(long_row._label) != long_row._label.text(), "not elided"
    assert long_row._label.toolTip() == "a-very-long-petname-here-indeed"
    w.close()


def test_the_column_does_not_move_when_the_hover_button_appears(widget, app):
    """A hidden widget is empty to the layout, so revealing the actions button used to
    shove the column 28px left and back again on the way out."""
    widget.resize(360, 200)
    widget.show()
    app.processEvents()
    row = widget._outer._rows[0]
    before = row._trailing.mapTo(widget, row._trailing.rect().topLeft()).x()
    row._btn_actions.setVisible(True)
    app.processEvents()
    after = row._trailing.mapTo(widget, row._trailing.rect().topLeft()).x()
    assert before == after
    widget.close()


# ── through the real thing ────────────────────────────────────────────────────
# Everything above feeds the widget by hand. This drives a real TaskManager over a real
# Experiment whose lamellae carry real task configs, and takes the estimates from
# `estimated_duration` exactly as AutoLamellaMainUI._push_timeline_estimates does — so
# the number on the row is one the shipped path actually produces, not one the test
# chose. A stand-in config returning 100.0 would prove none of that.

from PyQt5.QtCore import QObject, pyqtSignal  # noqa: E402

from fibsem.applications.autolamella.structures import Experiment, Lamella  # noqa: E402
from fibsem.applications.autolamella.workflows.tasks.fiducial import (  # noqa: E402
    MillFiducialTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.manager import (
    TaskManager,  # noqa: E402
)
from fibsem.applications.autolamella.workflows.tasks.select_position import (  # noqa: E402
    SelectMillingPositionTaskConfig,
)

TASKS = {
    "Setup Lamella Position": SelectMillingPositionTaskConfig,
    "Mill Fiducial": MillFiducialTaskConfig,
}


class _ParentUI(QObject):
    """The two signals TaskManager emits on, as real Qt signals."""

    workflow_status_signal = pyqtSignal(object)
    queue_changed_signal = pyqtSignal(dict)


class _NoMicroscope:
    """Enough microscope for TaskManager's constructor; nothing is executed here."""

    fm = None


@pytest.fixture
def live(app, tmp_path):
    """A real manager, a real experiment, and the widget wired the way the app wires it."""
    experiment = Experiment(path=str(tmp_path), name="estimate-test")
    for i, petname in enumerate(("01-lamella", "02-lamella"), start=1):
        lamella = Lamella(path=experiment.path, number=i, petname=petname)
        for task_name, config_cls in TASKS.items():
            lamella.task_config[task_name] = config_cls()
        experiment.positions.append(lamella)

    widget = WorkflowProgressWidget()
    parent_ui = _ParentUI()
    parent_ui.workflow_status_signal.connect(
        lambda event: widget.update_from_status(event.report)
    )
    parent_ui.queue_changed_signal.connect(
        lambda info: widget.refresh_queue(info["queue_items"])
    )
    manager = TaskManager(
        microscope=_NoMicroscope(), experiment=experiment, parent_ui=parent_ui
    )
    names = [p.name for p in experiment.positions]
    manager.queue.build_from_matrix(list(TASKS), names)

    # What AutoLamellaMainUI does at workflow start.
    widget.set_actions_enabled(True)
    widget.set_estimates(
        {
            (lamella.name, task_name): lamella.task_config[task_name].estimated_duration
            for lamella in experiment.positions
            for task_name in TASKS
        }
    )
    widget.set_workflow(manager.queue.items)
    return manager, widget, experiment


def test_a_real_config_reaches_the_column_through_the_managers_own_payload(live):
    """The queue is task-outer, lamella-inner, so item 1 is the same task on the other
    lamella — and its column must read what that lamella's own config actually costs."""
    manager, widget, experiment = live
    sibling = experiment.positions[1].task_config["Setup Lamella Position"]

    item = manager.queue.next()
    lamella = manager.experiment.get_lamella_by_name(item.lamella_name)
    manager._emit_status(item=item, lamella=lamella, status=Status.InProgress)

    assert column(widget)[1] == _fmt(sibling.estimated_duration)
    # Sanity, not a pin: a real task takes real time, and re-measuring a cost primitive
    # should not break this test.
    assert 5.0 < sibling.estimated_duration < 3600.0


def _fmt(seconds: float) -> str:
    from fibsem.ui.widgets.preflight import format_duration

    return format_duration(seconds)


def test_the_header_totals_every_item_the_real_queue_holds(live):
    """Four items over two tasks and two lamellae, each priced from its own config."""
    manager, widget, experiment = live
    total = sum(
        lamella.task_config[task_name].estimated_duration
        for lamella in experiment.positions
        for task_name in TASKS
    )
    assert widget._summary.text().startswith(f"{_fmt(total)} left · finishes ")
