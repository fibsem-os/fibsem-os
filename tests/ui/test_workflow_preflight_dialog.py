"""What the pre-flight dialog says before a workflow starts.

Its job is the two figures a user actually decides on -- how long, and when it finishes
-- plus the two things that stop those figures being the whole story: a step that waits
for a human, and a step that holds until a wall-clock time.

Built here from plain WorkflowEstimate objects rather than through the app: the
arithmetic is covered in tests/autolamella/test_workflow_estimate.py, and what is
checked here is what reaches the screen.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_workflow_preflight_dialog.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

# CI installs `.[test]`, not `.[ui]`, so PyQt5 is absent there.
pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QLabel, QToolButton  # noqa: E402

from fibsem.applications.autolamella.ui.workflow_preflight_dialog import (  # noqa: E402
    WorkflowPreflightDialog,
)
from fibsem.applications.autolamella.workflows.workflow_estimate import (  # noqa: E402
    TaskEstimate,
    WorkflowEstimate,
)
from fibsem.ui.stylesheets import NAPARI_STYLE  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv)
# The app stylesheet, because without it these tests are run under conditions the user
# never sees. Its bare `QWidget { background-color: ... }` rule reaches every descendant
# and paints the surface colour behind any label that has not declared itself
# transparent -- a lighter block around every word, invisible offscreen without it.
_app.setStyleSheet(NAPARI_STYLE)

NOW = datetime(2026, 8, 18, 14, 0, 0)


def _estimate(tasks=None, **kwargs) -> WorkflowEstimate:
    tasks = tasks if tasks is not None else [
        TaskEstimate(name="Mill Fiducial", lamella_count=2, seconds=198.0, supervised=False)
    ]
    defaults = dict(
        tasks=tasks,
        lamella_names=["01-fancy-mite", "02-civil-cub"],
        work_seconds=sum(t.seconds for t in tasks),
        hold_seconds=0.0,
        started_at=NOW,
        expected_finish=NOW + timedelta(seconds=sum(t.seconds for t in tasks)),
    )
    defaults.update(kwargs)
    return WorkflowEstimate(**defaults)


@pytest.fixture
def dialog():
    built = []

    def build(estimate=None):
        d = WorkflowPreflightDialog(estimate if estimate is not None else _estimate())
        built.append(d)
        return d

    yield build
    for d in built:
        d.close()


def _texts(dialog) -> str:
    """Every label in the dialog, joined -- what the user can read."""
    return "\n".join(label.text() for label in dialog.findChildren(QLabel))


# ── the two figures ──────────────────────────────────────────────────────────

def test_it_shows_the_duration_and_the_finish_clock(dialog):
    """Two, not one: the duration says whether to start, the clock says when to come
    back, and the second is the one people act on."""
    d = dialog(_estimate())
    text = _texts(d)
    assert "3m 18s" in text
    assert "2:03PM" in text


def test_the_duration_is_the_work_not_the_wall_clock(dialog):
    """A held workflow finishes late without taking longer to run."""
    at = NOW + timedelta(hours=4)
    tasks = [TaskEstimate("Polishing", 2, 600.0, supervised=False, scheduled_at=at)]
    d = dialog(_estimate(tasks=tasks, hold_seconds=4 * 3600,
                         expected_finish=at + timedelta(seconds=600)))
    text = _texts(d)
    assert "10m 00s" in text, "the duration is the work"
    assert "6:10PM" in text, "the finish is after the hold"


# ── the breakdown ────────────────────────────────────────────────────────────

def test_every_task_gets_a_row_with_its_count_and_duration(dialog):
    tasks = [
        TaskEstimate("Mill Fiducial", 2, 198.0, supervised=False),
        TaskEstimate("Rough Milling", 3, 1224.0, supervised=False),
    ]
    text = _texts(dialog(_estimate(tasks=tasks)))
    assert "Mill Fiducial" in text and "×2" in text and "3m 18s" in text
    assert "Rough Milling" in text and "×3" in text and "20m 24s" in text


def test_the_step_count_is_the_queue_not_the_task_count(dialog):
    """One item per (task, lamella), the same shape build_from_matrix produces."""
    tasks = [
        TaskEstimate("Mill Fiducial", 2, 198.0, supervised=False),
        TaskEstimate("Rough Milling", 2, 1224.0, supervised=False),
    ]
    assert "4 steps" in _texts(dialog(_estimate(tasks=tasks)))


# ── the exceptions ───────────────────────────────────────────────────────────

def test_a_supervised_step_is_flagged_and_its_work_still_counted(dialog):
    """Only the pause is unbounded. Dropping the task would understate the total by
    the work it does, so the row carries a chip instead of a blank."""
    tasks = [TaskEstimate("Polishing", 2, 785.0, supervised=True)]
    text = _texts(dialog(_estimate(tasks=tasks)))
    assert "Supervised" in text
    assert "13m 05s" in text, "the work is still quoted"
    assert "waits for you" in text


def test_a_scheduled_step_says_how_long_the_workflow_holds(dialog):
    """The hold is invisible in the duration, and someone reading only that figure
    would be wrong by its length."""
    at = NOW + timedelta(hours=6)
    tasks = [
        TaskEstimate("Polishing", 2, 785.0, supervised=False, scheduled_at=at,
                     hold_seconds=6 * 3600)
    ]
    text = _texts(dialog(_estimate(tasks=tasks, hold_seconds=6 * 3600)))
    assert "Scheduled 8:00PM" in text
    assert "6h 00m" in text


def test_each_scheduled_step_is_quoted_its_own_hold_not_the_total(dialog):
    """Two scheduled tasks share one workflow hold; quoting the whole of it against
    each would say the same wrong number twice. Summarised once instead."""
    tasks = [
        TaskEstimate("Mill Fiducial", 2, 198.0, supervised=False,
                     scheduled_at=NOW + timedelta(hours=2), hold_seconds=2 * 3600),
        TaskEstimate("Polishing", 2, 785.0, supervised=False,
                     scheduled_at=NOW + timedelta(hours=6), hold_seconds=4 * 3600),
    ]
    text = _texts(dialog(_estimate(tasks=tasks, hold_seconds=6 * 3600)))
    assert "2 steps are scheduled" in text
    assert "6h 00m" in text, "the summary quotes the total"
    assert text.count("The workflow holds") == 1, "one note, not one per task"


def test_several_supervised_steps_are_counted_not_listed(dialog):
    """Every row already carries its own chip; naming them again grows a sentence
    without adding anything -- and reads as 'A, B and C waits for you'."""
    tasks = [
        TaskEstimate("Setup Lamella Position", 2, 59.0, supervised=True),
        TaskEstimate("Mill Fiducial", 2, 198.0, supervised=True),
        TaskEstimate("Polishing", 2, 785.0, supervised=True),
    ]
    text = _texts(dialog(_estimate(tasks=tasks)))
    assert "3 steps wait for you" in text
    assert "waits for you" not in text, "singular phrasing is for the single case"


def test_an_unsupervised_workflow_says_nothing_about_waiting(dialog):
    text = _texts(dialog(_estimate()))
    assert "waits for you" not in text
    assert "supervised" not in text.lower()


# ── the lamella disclosure ───────────────────────────────────────────────────

def test_the_lamella_names_are_hidden_until_asked_for(dialog):
    """A per-task breakdown replaces the names with ×2; they come back once, here,
    rather than repeated under every task."""
    d = dialog(_estimate())
    assert d._lamella_names_label.isVisible() is False

    d.show()
    d.findChild(QToolButton).setChecked(True)
    assert d._lamella_names_label.isVisible() is True
    assert "01-fancy-mite" in d._lamella_names_label.text()


def test_an_overnight_finish_says_tomorrow(dialog):
    """The case this dialog exists for. A bare clock time for a workflow finishing the
    next morning is wrong by a day, and a date stamp reads as a filename where a person
    would just say "tomorrow"."""
    tasks = [TaskEstimate("Rough Milling", 2, 72000.0, supervised=False)]
    text = _texts(dialog(_estimate(tasks=tasks, expected_finish=NOW + timedelta(hours=16, minutes=15))))
    assert "Tomorrow, 6:15AM" in text


def test_a_finish_today_is_just_the_clock(dialog):
    text = _texts(dialog(_estimate()))
    assert "Tomorrow" not in text
    assert "2026-08-18" not in text, "no date needed for a workflow finishing today"


def test_a_finish_further_out_keeps_its_date(dialog):
    """"In 3 days" is arithmetic the reader has to do."""
    tasks = [TaskEstimate("Rough Milling", 2, 259200.0, supervised=False)]
    text = _texts(dialog(_estimate(tasks=tasks, expected_finish=NOW + timedelta(days=3))))
    assert "2026-08-21" in text


def test_a_hundred_lamellae_do_not_push_the_dialog_apart(dialog):
    """Regression: the expanded names wrapped to a dozen lines and overlapped the
    metrics, the note and every task row. The list truncates and says how many it did
    not show -- a selection of a hundred is confirmed by spot-checking a few and
    trusting the count."""
    from fibsem.applications.autolamella.ui.workflow_preflight_dialog import (
        _NAMES_SHOWN,
    )

    def _expanded_names(count):
        d = dialog(_estimate(lamella_names=[f"{i:02d}-scale-lamella" for i in range(count)]))
        d.show()
        d.findChild(QToolButton).setChecked(True)
        return d._lamella_names_label.text()

    hundred = _expanded_names(100)
    thousand = _expanded_names(1000)

    assert "+ 88 more" in hundred, "the count is what a truncated list has to give back"
    assert "+ 988 more" in thousand

    # The mechanism that bounds the dialog, asserted rather than its pixel height:
    # ten times the lamellae must render the same number of names.
    assert hundred.count("scale-lamella") == _NAMES_SHOWN
    assert thousand.count("scale-lamella") == _NAMES_SHOWN


def test_a_short_lamella_list_is_shown_whole(dialog):
    """The common case: a handful of lamellae, every one of them visible."""
    d = dialog(_estimate())
    d.show()
    d.findChild(QToolButton).setChecked(True)
    text = d._lamella_names_label.text()
    assert "01-fancy-mite" in text and "02-civil-cub" in text
    assert "more" not in text


def test_the_breakdown_is_per_task_not_per_lamella(dialog):
    """What keeps the dialog a fixed size at any scale: five tasks over a hundred
    lamellae is still five rows."""
    tasks = [
        TaskEstimate("Mill Fiducial", 100, 19800.0, supervised=False),
        TaskEstimate("Rough Milling", 100, 61200.0, supervised=False),
    ]
    text = _texts(dialog(_estimate(tasks=tasks, lamella_names=[f"l{i}" for i in range(100)])))
    assert "×100" in text
    assert "200 steps" in text


# ── nothing to do ────────────────────────────────────────────────────────────

def test_run_is_refused_when_no_task_is_selected(dialog):
    """The queue would build empty and the workflow end immediately; say so here
    rather than after the dialog is dismissed."""
    d = dialog(_estimate(tasks=[], work_seconds=0.0, expected_finish=NOW))
    assert d.button_run.isEnabled() is False
    assert "No tasks are selected." in d.button_run.toolTip()


def test_run_is_offered_when_there_is_work(dialog):
    assert dialog(_estimate()).button_run.isEnabled() is True


# ── the chips keep their own background ──────────────────────────────────────

def test_the_panels_do_not_restyle_the_chips_inside_them(dialog):
    """Regression, and invisible without running the app: a bare `QFrame { ... }`
    selector matches every QFrame *inside* the widget it is set on, and
    preflight.chip() is a QFrame -- so the task block was repainting each chip with
    the panel's fill and border on top of the chip's own.

    Only visible on a real style; offscreen rendering hid it. Pinned by the selector
    rather than by pixels for that reason.
    """
    from PyQt5.QtWidgets import QFrame

    tasks = [TaskEstimate("Polishing", 2, 785.0, supervised=True)]
    d = dialog(_estimate(tasks=tasks))

    panels = [
        f for f in d.findChildren(QFrame)
        if f.objectName() in {"preflightTaskBlock", "preflightMetric"}
    ]
    assert len(panels) == 3, "one task block and two metric cards"
    for panel in panels:
        assert f"QFrame#{panel.objectName()}" in panel.styleSheet()
        assert "QFrame {" not in panel.styleSheet(), (
            "a bare selector would cascade into the chips"
        )


def test_the_lamella_button_is_not_a_raised_control(dialog):
    """With no background rule a QToolButton paints the platform default, which reads
    as a button beside the flat chips it sits next to."""
    d = dialog(_estimate())
    assert "background: transparent" in d.findChild(QToolButton).styleSheet()


def test_everything_on_a_panel_declares_itself_transparent(dialog):
    """Regression, and only visible with the app stylesheet loaded: NAPARI_STYLE sets
    `QWidget { background-color: SURFACE_COLOR }` for every widget, so a label that
    only sets `color` paints the surface colour onto the darker panel behind it.

    The panels are scoped by object name and therefore no longer shadow that rule for
    their children, which makes each child's own declaration the only thing standing
    between the design and a lighter block around every word.
    """
    from PyQt5.QtWidgets import QFrame

    tasks = [TaskEstimate("Polishing", 2, 785.0, supervised=True)]
    d = dialog(_estimate(tasks=tasks))

    def sits_directly_on(panel, label) -> bool:
        """Whether the panel is the nearest frame behind this label.

        A chip is a QFrame with its own bare-selector rule, which shadows the app
        stylesheet for the label inside it -- so those are already covered and are not
        the panel's problem.
        """
        widget = label.parentWidget()
        while widget is not panel and widget is not None:
            if isinstance(widget, QFrame):
                return False
            widget = widget.parentWidget()
        return widget is panel

    panels = [
        f for f in d.findChildren(QFrame)
        if f.objectName() in {"preflightTaskBlock", "preflightMetric"}
    ]
    assert panels, "guard: the panels are findable"
    checked = 0
    for panel in panels:
        for label in panel.findChildren(QLabel):
            if not sits_directly_on(panel, label):
                continue
            checked += 1
            assert "background: transparent" in label.styleSheet(), (
                f"{label.text()!r} would paint the surface colour onto the panel"
            )
    assert checked >= 4, "guard: the metric captions and values were actually reached"
