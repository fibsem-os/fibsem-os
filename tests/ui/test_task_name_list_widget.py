"""Status chips on the task list's rows.

The Lamella tab used to say "Task 'X' has been completed." in a cyan sentence under
the image pickers. Completion is a property of a row, so it is drawn on the row: a
chip at the right, in the colour the rest of the window uses for that state, and
nothing at all for a task that has not started.
"""

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import Qt  # noqa: E402
from PyQt5.QtWidgets import QApplication, QLabel  # noqa: E402

from fibsem.ui.widgets.custom_widgets import TaskNameListWidget  # noqa: E402


@pytest.fixture
def qapp():
    yield QApplication.instance() or QApplication([])


def _chips(widget: TaskNameListWidget):
    """Row index -> chip text, for rows that carry one."""
    out = {}
    for i in range(widget._list.count()):
        row = widget._list.itemWidget(widget._list.item(i))
        if row is not None:
            out[i] = row.findChild(QLabel).text()
    return out


def test_only_named_rows_get_a_chip(qapp):
    widget = TaskNameListWidget()
    widget.set_tasks(["Setup", "Mill Fiducial", "Rough Milling", "Polishing"])

    widget.set_task_states(
        {"Setup": ("Completed", "#4caf50"), "Rough Milling": ("In Progress", "#50a6ff")}
    )

    assert _chips(widget) == {0: "Completed", 2: "In Progress"}


def test_chips_survive_a_repopulate_and_clear_when_asked(qapp):
    widget = TaskNameListWidget()
    widget.set_tasks(["Setup", "Polishing"])
    widget.set_task_states({"Setup": ("Completed", "#4caf50")})

    widget.set_tasks(["Setup", "Polishing", "Extra"])
    assert _chips(widget) == {0: "Completed"}

    widget.set_task_states({})
    assert _chips(widget) == {}


def test_the_chip_does_not_change_the_row_text_or_selection(qapp):
    """The chip rides in an item widget; the row's text and click handling stay the
    list's own, so `selected_task` and every existing caller keep working."""
    widget = TaskNameListWidget()
    widget.set_tasks(["Setup", "Polishing"])
    widget.set_task_states({"Polishing": ("Completed", "#4caf50")})

    widget.select("Polishing")

    assert widget.selected_task == "Polishing"
    assert widget._list.item(1).text() == "Polishing"
    row = widget._list.itemWidget(widget._list.item(1))
    assert row.testAttribute(Qt.WA_TransparentForMouseEvents)
