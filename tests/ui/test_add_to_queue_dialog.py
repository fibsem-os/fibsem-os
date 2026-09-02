"""What the Add to Queue dialog says about time (FIB-731).

The arithmetic is tested in tests/autolamella/test_workflow_estimate.py without a widget.
What is checked here is the dialog's own job: showing two figures that are allowed to
disagree, saying why when they do, and staying usable when there is no estimate at all.

`NAPARI_STYLE` is applied to the QApplication deliberately -- the app sets it, and its
bare `QWidget { background-color: ... }` rule reaches every descendant, so a render
without it cannot show a label painting a block onto the panel behind it.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_add_to_queue_dialog.py
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from datetime import datetime, timedelta

from PyQt5.QtWidgets import QDialog, QLabel

from fibsem.applications.autolamella.ui.AutoLamellaMainUI import (
    confirm_add_to_queue_dialog,
)
from fibsem.applications.autolamella.workflows.tasks.queue import WorkItem
from fibsem.applications.autolamella.workflows.workflow_estimate import (
    estimate_addition,
)
from fibsem.ui.stylesheets import NAPARI_STYLE

NOW = datetime(2026, 8, 19, 14, 0, 0)
LAMELLA = ["01-lamella", "02-lamella"]
TASKS = ["Rough Milling", "Polishing"]
PAIRS = [(ln, tn) for tn in TASKS for ln in LAMELLA]


@pytest.fixture
def app(qapp):
    qapp.setStyleSheet(NAPARI_STYLE)
    return qapp


@pytest.fixture
def show(app, monkeypatch):
    """Open the dialog without blocking, and hand it back for inspection."""
    built = []

    def fake_exec(self):
        built.append(self)
        return QDialog.Rejected

    monkeypatch.setattr(QDialog, "exec_", fake_exec)

    def open_it(estimate=None, already=(), run_next=False):
        confirm_add_to_queue_dialog(LAMELLA, TASKS, run_next, list(already),
                                    estimate=estimate)
        return built[-1]

    yield open_it
    for dlg in built:
        dlg.close()


def texts(dialog) -> list:
    return [lab.text() for lab in dialog.findChildren(QLabel)]


def joined(dialog) -> str:
    return " | ".join(texts(dialog))


def queue(*spec) -> list:
    return [WorkItem(item_name=ln, task_name=tn) for ln, tn in spec]


def estimate(items, seconds=600.0, run_next=False, schedule=None):
    return estimate_addition(items, PAIRS, lambda i: seconds, run_next=run_next,
                             schedule=schedule, now=NOW)


# ── the two figures ───────────────────────────────────────────────────────────

def test_it_shows_what_the_addition_costs_and_where_it_lands(show):
    dialog = show(estimate(queue(("03-lamella", "Rough Milling"))))
    body = joined(dialog)
    assert "Adds" in body
    assert "40m 00s" in body       # 4 added tasks x 600 s
    assert "Expected finish" in body
    assert "was " in body          # the finish it moved from


def test_the_finish_it_quotes_is_the_one_after_the_addition(show):
    est = estimate(queue(("03-lamella", "Rough Milling")))
    dialog = show(est)
    from fibsem.ui.widgets.preflight import format_clock
    after = format_clock(est.finish_after, est.finish_before)
    before = format_clock(est.finish_before, est.finish_before)
    assert after in joined(dialog)
    assert f"was {before}" in joined(dialog)
    assert after != before


def test_the_task_count_sits_under_the_duration(show):
    assert "4 task(s)" in joined(show(estimate(queue(("03-lamella", "Rough Milling")))))


# ── when the two figures disagree ─────────────────────────────────────────────

def test_work_absorbed_by_a_scheduled_wait_says_why_it_costs_nothing(show):
    """An hour of work that moves the finish by nothing is the right answer and reads
    as a bug, so the reason goes on screen beside it."""
    at = NOW + timedelta(hours=6)
    est = estimate_addition(queue(("03-lamella", "Polishing")), [("04-lamella", "Setup")],
                            lambda i: 600.0, run_next=True,
                            schedule={"Polishing": at}, now=NOW)
    assert est.work_seconds > 0 and est.delay_seconds == 0
    assert "costs no extra time overall" in joined(show(est))


def test_partly_absorbed_work_says_how_much_actually_lands_late(show):
    at = NOW + timedelta(hours=6)
    est = estimate_addition(
        queue(("03-lamella", "Polishing")), PAIRS, lambda i: 600.0,
        run_next=True, schedule={"Polishing": at}, now=NOW,
    )
    assert 0 < est.delay_seconds < est.work_seconds
    assert "fits inside it" in joined(show(est))


def test_an_ordinary_addition_explains_nothing(show):
    """The note exists for the surprising case. On a plain queue the two figures agree
    and a sentence saying so would be noise."""
    body = joined(show(estimate(queue(("03-lamella", "Rough Milling")))))
    assert "fits inside" not in body
    assert "costs no extra time" not in body


# ── degrading ─────────────────────────────────────────────────────────────────

def test_with_no_estimate_at_all_the_figures_are_dropped_not_faked(show):
    """A protocol whose configs cannot be priced still has to be able to queue work."""
    body = joined(show(None))
    assert "Adds" not in body
    assert "Expected finish" not in body
    assert "Rough Milling, Polishing" in body      # the dialog still does its job


def test_an_unpriced_estimate_is_treated_as_no_estimate(show):
    est = estimate_addition(queue(("03-lamella", "Rough Milling")), PAIRS,
                            lambda i: None, now=NOW)
    assert est.is_priced is False
    assert "Expected finish" not in joined(show(est))


def test_a_partly_priced_addition_admits_how_much_it_covers(show):
    """Otherwise it reads as a complete estimate that happens to be quick."""
    seconds_for = lambda i: 600.0 if i.lamella_name == "01-lamella" else None
    est = estimate_addition(queue(("03-lamella", "Rough Milling")), PAIRS,
                            seconds_for, now=NOW)
    assert "2 estimated" in joined(show(est))


# ── what the dialog said before, and still must ───────────────────────────────

def test_the_already_queued_warning_survives(show):
    """The one thing this dialog says that nothing else does: a task queued twice mills
    twice. It is stated rather than acted on, so it has to stay stated."""
    dialog = show(estimate(queue(("03-lamella", "Rough Milling"))),
                  already=["Polishing for 01-lamella", "Polishing for 02-lamella"])
    body = joined(dialog)
    assert "2 already queued" in body
    assert "run twice" in body
    assert "Polishing for 01-lamella" in body


def test_the_duplicate_notice_carries_no_accent_colour(show):
    """A consequence to notice, not an alarm. The bold count is enough to catch and
    Cancel is right there; three lines of body text beat three orange ones out-shouting
    the two figures that lead the dialog."""
    from fibsem.ui.tokens import DEFECT_ORANGE_COLOR
    dialog = show(None, already=["Polishing for 01-lamella"])
    warning = next(lab for lab in dialog.findChildren(QLabel)
                   if "already queued" in lab.text())
    from fibsem.ui.widgets.preflight import TEXT, TEXT_MUTED
    assert DEFECT_ORANGE_COLOR not in warning.text()
    assert DEFECT_ORANGE_COLOR not in warning.styleSheet()
    # Body, not the secondary shade: this is content, not a detail-block label.
    assert TEXT in warning.styleSheet()
    assert TEXT_MUTED not in warning.styleSheet()
    assert "<b>" in warning.text(), "the count still needs to be findable"


def test_a_long_already_queued_list_is_truncated_rather_than_unbounded(show):
    already = [f"Polishing for {i:02d}-lamella" for i in range(20)]
    body = joined(show(None, already=already))
    assert "20 already queued" in body
    assert "…" in body


def test_the_position_is_named_both_ways(show):
    assert "At the end of the queue" in joined(show(None, run_next=False))
    assert "Run next" in joined(show(None, run_next=True))


def test_the_count_in_the_button_matches_what_will_be_queued(show):
    from PyQt5.QtWidgets import QPushButton
    dialog = show(None)
    labels = [b.text() for b in dialog.findChildren(QPushButton)]
    assert "Add 4 task(s)" in labels
    assert "Cancel" in labels


def test_cancel_is_the_default_button(show):
    """Adding work to a running queue is not the safe outcome of a stray Return."""
    from PyQt5.QtWidgets import QPushButton
    dialog = show(None)
    default = [b.text() for b in dialog.findChildren(QPushButton) if b.isDefault()]
    assert default == ["Cancel"]


def test_all_three_dialogs_quote_time_the_same_way(show):
    """This one, the pre-flight dialog and the timeline all describe the same workflow;
    they cannot use different formatters and stay believable."""
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as main_ui
    from fibsem.ui.widgets import preflight
    assert main_ui.preflight.format_duration is preflight.format_duration
    assert main_ui.preflight.format_clock is preflight.format_clock


# ── the note must not invent a wait ───────────────────────────────────────────

def test_a_queue_with_nothing_scheduled_never_mentions_a_scheduled_wait(show):
    """Regression. `delay_seconds` comes back through a datetime and `work_seconds` is a
    difference of two float sums, so for the same quantity they disagreed by 9e-13 --
    enough for a `>=` test to conclude the work had been partly absorbed and explain a
    wait that did not exist. Reported from the running app on a plain queue.
    """
    lamella = [f"{i:02d}-lamella" for i in range(1, 7)]
    pairs = [(ln, tn) for tn in TASKS for ln in lamella]
    est = estimate_addition(queue(("07-lamella", "Polishing")), pairs,
                            lambda i: 505.3, now=NOW)   # nothing scheduled anywhere
    assert est.work_seconds != est.delay_seconds, "the float gap this guards is gone"
    assert est.hold_seconds == 0.0
    assert "scheduled wait" not in joined(show(est))


def test_a_difference_too_small_to_render_is_not_explained(show):
    """Both figures go through `format_duration`, so a gap that rounds away leaves two
    identical numbers with a paragraph between them saying they differ."""
    at = NOW + timedelta(hours=6)
    est = estimate_addition(queue(("03-lamella", "Polishing")), [("04-lamella", "Setup")],
                            lambda i: 600.0, schedule={"Polishing": at}, now=NOW)
    object.__setattr__(est, "delay_seconds", est.work_seconds - 0.2)
    assert est.hold_seconds > 0
    assert "fits inside it" not in joined(show(est))


def test_a_real_partial_absorption_is_still_explained(show):
    """The guards must not silence the case the note exists for."""
    at = NOW + timedelta(hours=6)
    est = estimate_addition(queue(("03-lamella", "Polishing")), PAIRS, lambda i: 600.0,
                            run_next=True, schedule={"Polishing": at}, now=NOW)
    assert est.hold_seconds > 0
    assert 0 < est.delay_seconds < est.work_seconds
    assert "fits inside it" in joined(show(est))
