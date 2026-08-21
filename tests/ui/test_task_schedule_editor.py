"""The task editor's scheduling controls (FIB-612).

The runtime side of scheduling is covered in
`tests/autolamella/test_scheduled_tasks.py`; this is the other end, where a
`scheduled_at` is set and — the part that had no coverage and is easy to get
wrong — cleared again.

Clearing matters more than setting. While scheduling sat behind a preference,
Apply only wrote `scheduled_at` when the preference was on, so a schedule saved
in a protocol could outlive any control that could see it. Now that Apply always
writes, unticking the box has to be the way back out.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from datetime import datetime, timedelta

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import QDateTime
from PyQt5.QtWidgets import QApplication

from fibsem.applications.autolamella.structures import AutoLamellaTaskDescription
from fibsem.applications.autolamella.ui.workflow_task_editor_widget import (
    WorkflowTaskEditorWidget,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _task(scheduled_at=None) -> AutoLamellaTaskDescription:
    return AutoLamellaTaskDescription(
        name="MillRough", supervise=True, required=True, scheduled_at=scheduled_at
    )


def _applied(editor: WorkflowTaskEditorWidget) -> AutoLamellaTaskDescription:
    """Click Apply and return the task it emitted."""
    emitted = []
    editor.apply_clicked.connect(emitted.append)
    editor._on_apply()
    assert emitted, "Apply emitted nothing"
    return emitted[0]


# ── setting a schedule ────────────────────────────────────────────────────────


def test_ticking_the_box_writes_the_chosen_time(qapp):
    editor = WorkflowTaskEditorWidget(_task())
    when = datetime.now().replace(second=0, microsecond=0) + timedelta(hours=3)

    editor._schedule_cb.setChecked(True)
    editor._dt_edit.setDateTime(
        QDateTime(when.year, when.month, when.day, when.hour, when.minute)
    )

    assert _applied(editor).scheduled_at == when
    editor.deleteLater()


def test_the_seconds_are_dropped(qapp):
    """The control offers minutes, so a stored time carrying seconds could never be
    reproduced by the editor -- it would come back a different value than it went in."""
    editor = WorkflowTaskEditorWidget(_task())
    editor._schedule_cb.setChecked(True)

    applied = _applied(editor)

    assert applied.scheduled_at.second == 0
    assert applied.scheduled_at.microsecond == 0
    editor.deleteLater()


# ── clearing one ──────────────────────────────────────────────────────────────


def test_unticking_the_box_clears_the_schedule(qapp):
    """The way back out. Without this a task scheduled once stays scheduled, and the
    only way to unschedule it is to hand-edit the protocol file."""
    editor = WorkflowTaskEditorWidget(_task(datetime.now() + timedelta(days=1)))

    editor._schedule_cb.setChecked(False)

    assert _applied(editor).scheduled_at is None
    editor.deleteLater()


def test_an_unscheduled_task_stays_unscheduled(qapp):
    """Apply on a task nobody scheduled must not invent a time from the date edit,
    which seeds itself to 'now' whether or not the box is ticked."""
    editor = WorkflowTaskEditorWidget(_task())

    assert _applied(editor).scheduled_at is None
    editor.deleteLater()


# ── what the controls show when a task is loaded ──────────────────────────────


def test_loading_a_scheduled_task_shows_its_time(qapp):
    when = (datetime.now() + timedelta(days=2)).replace(second=0, microsecond=0)
    editor = WorkflowTaskEditorWidget(_task())

    editor.load_task(_task(when))

    assert editor._schedule_cb.isChecked()
    assert editor._dt_edit.isEnabled()
    assert editor._dt_edit.dateTime().toPyDateTime() == when
    editor.deleteLater()


def test_loading_an_unscheduled_task_leaves_the_date_edit_dead(qapp):
    """Reloading is where stale state shows up: the editor is reused across tasks."""
    editor = WorkflowTaskEditorWidget(_task(datetime.now() + timedelta(days=1)))

    editor.load_task(_task())

    assert not editor._schedule_cb.isChecked()
    assert not editor._dt_edit.isEnabled()
    editor.deleteLater()


def test_the_schedule_controls_are_on_screen(qapp):
    """They were hidden behind a preference. A control that exists but is never shown
    is the same as no control at all, and that is not something a test of `_on_apply`
    would notice."""
    editor = WorkflowTaskEditorWidget(_task())
    editor.show()

    assert not editor._schedule_container.isHidden()
    editor.deleteLater()
