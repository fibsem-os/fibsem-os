"""Clearing a selection has to clear every part of it (FIB-577, FIB-578).

Two bugs from the same shape: a widget that holds selection state in more places than
the checkboxes the user can see, and a clear path that only reached some of them.

* `LamellaListWidget` / `WorkflowConfigWidget` -- the header "Select All" box, and (in
  the workflow one) a `_checked` cache that reorders rebuild rows from. Starting a run
  and adding to the queue both clear the selection, and both used to call the header's
  own slot, which updates neither.
* `LamellaCardContainer` -- `_selected_id`, which survived the clear half of the
  clear-then-re-add rebuild the host performs on every experiment change.

The tests drive the public API the hosts actually call, and additionally the header
signal, because that is the path where the old code was already correct and a fix must
not regress it.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    Lamella,
)
from fibsem.applications.autolamella.ui.lamella_card_widget import LamellaCardContainer
from fibsem.applications.autolamella.ui.lamella_list_widget import LamellaListWidget
from fibsem.applications.autolamella.ui.workflow_config_widget import (
    WorkflowConfigWidget,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def lamellae(tmp_path):
    return [
        Lamella(path=str(tmp_path), number=i, petname=f"lamella-{i}") for i in range(3)
    ]


@pytest.fixture()
def tasks():
    return [
        AutoLamellaTaskDescription(name=f"task-{i}", supervise=False, required=True)
        for i in range(3)
    ]


def _lamella_list(lamellae, checked: bool) -> LamellaListWidget:
    widget = LamellaListWidget()
    for lamella in lamellae:
        widget.add_lamella(lamella, checked=checked)
    return widget


def _workflow_list(tasks, checked: bool) -> WorkflowConfigWidget:
    widget = WorkflowConfigWidget()
    for task in tasks:
        widget.add_task(task, checked=checked)
    return widget


# ── FIB-577: the header follows the rows ─────────────────────────────────────


def test_clearing_the_lamella_selection_unticks_the_header(qapp, lamellae):
    """The reported bug: rows cleared, header still reading "Select All"."""
    widget = _lamella_list(lamellae, checked=True)
    assert widget._header.checkbox_all.checkState() == Qt.Checked

    widget.set_all_selected(False)

    assert widget.get_selected() == []
    assert widget._header.checkbox_all.checkState() == Qt.Unchecked


def test_clearing_the_task_selection_unticks_the_header(qapp, tasks):
    widget = _workflow_list(tasks, checked=True)
    assert widget._header.checkbox_all.checkState() == Qt.Checked

    widget.set_all_selected(False)

    assert widget.get_selected() == []
    assert widget._header.checkbox_all.checkState() == Qt.Unchecked


def test_reorder_after_a_clear_does_not_resurrect_the_selection(qapp, tasks):
    """`_on_reordered` rebuilds rows from `_checked`, so the cache has to be cleared too.

    Otherwise a drag-and-drop after the run started re-ticks everything, and the next
    Run Workflow queues tasks the user watched being cleared.
    """
    widget = _workflow_list(tasks, checked=True)

    widget.set_all_selected(False)
    assert set(widget._checked.values()) == {False}

    widget._on_reordered(list(reversed(tasks)))

    assert widget.get_selected() == []
    assert widget._header.checkbox_all.checkState() == Qt.Unchecked


def test_selecting_all_ticks_every_row(qapp, lamellae):
    """The other direction, for a list that starts empty-handed."""
    widget = _lamella_list(lamellae, checked=False)

    widget.set_all_selected(True)

    assert len(widget.get_selected()) == len(lamellae)
    assert widget._header.checkbox_all.checkState() == Qt.Checked


@pytest.mark.parametrize("checked", [True, False])
def test_the_header_checkbox_still_drives_the_rows(qapp, lamellae, checked):
    """Regression guard: the header signal now lands on the public method."""
    widget = _lamella_list(lamellae, checked=not checked)
    seen = []
    widget.selection_changed.connect(seen.append)

    widget._header.checkbox_all.setChecked(checked)

    assert len(widget.get_selected()) == (len(lamellae) if checked else 0)
    assert seen, "the header path must still emit selection_changed"


# ── FIB-578: the card container forgets what was selected ────────────────────


def test_first_click_after_a_rebuild_selects_rather_than_deselects(qapp, lamellae):
    """The host clears and re-adds the same lamellae, whose ids are stable.

    A `_selected_id` surviving the clear made the container disagree with the card it
    had just built, and the click that should select took the toggle-off branch.
    """
    container = LamellaCardContainer(columns=1)
    for lamella in lamellae:
        container.add_lamella(lamella)
    container._on_card_clicked(lamellae[0])
    assert container._selected_id == lamellae[0].id

    container.clear()
    assert container._selected_id is None
    for lamella in lamellae:
        container.add_lamella(lamella)

    seen = []
    container.lamella_selected.connect(seen.append)
    container._on_card_clicked(lamellae[0])

    assert seen == [lamellae[0]]
    assert container._selected_id == lamellae[0].id


def test_clicking_the_selected_card_still_deselects(qapp, lamellae):
    """Toggle-off is deliberate and stays -- the bug was when it fired, not that it does."""
    container = LamellaCardContainer(columns=1)
    for lamella in lamellae:
        container.add_lamella(lamella)
    container._on_card_clicked(lamellae[0])

    seen = []
    container.lamella_selected.connect(seen.append)
    container._on_card_clicked(lamellae[0])

    assert seen == [None]
    assert container._selected_id is None


def test_removing_the_selected_lamella_clears_the_selection(qapp, lamellae):
    container = LamellaCardContainer(columns=1)
    for lamella in lamellae:
        container.add_lamella(lamella)
    container._on_card_clicked(lamellae[0])

    container.remove_lamella(lamellae[0])

    assert container._selected_id is None


def test_removing_an_unselected_lamella_leaves_the_selection_alone(qapp, lamellae):
    container = LamellaCardContainer(columns=1)
    for lamella in lamellae:
        container.add_lamella(lamella)
    container._on_card_clicked(lamellae[0])

    container.remove_lamella(lamellae[1])

    assert container._selected_id == lamellae[0].id
