"""Adding work to a running queue from the timeline header (FIB-477).

The lamella x task picker already exists as LamellaWorkflowWidget, stays live
during a run and is cleared when one is launched, so the header button commits
that selection rather than opening a second picker.

Everything here is a real Experiment, real Lamellas, a real task protocol and a
real TaskQueue. The one stand-in is a microscope.
"""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaWorkflowConfig,
    Experiment,
    Lamella,
)
from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus as Status
from fibsem.applications.autolamella.ui.workflow_timeline_widget import (
    WorkflowProgressWidget,
)
from fibsem.applications.autolamella.workflows.tasks.queue import TaskQueue
from fibsem.applications.autolamella.workflows.tasks.reference_image import (
    AcquireReferenceImageConfig,
)

TASKS = ["Trench", "Undercut", "Polishing"]


@pytest.fixture
def experiment(tmp_path: Path) -> Experiment:
    exp = Experiment(path=tmp_path, name="test-exp")
    exp.task_protocol = AutoLamellaTaskProtocol(
        workflow_config=AutoLamellaWorkflowConfig(
            tasks=[
                AutoLamellaTaskDescription(name=n, supervise=False, required=False)
                for n in TASKS
            ]
        ),
        task_config={n: AcquireReferenceImageConfig(task_name=n) for n in TASKS},
    )
    for i, name in enumerate(("L1", "L2"), start=1):
        lamella = Lamella(path=Path(exp.path) / name, number=i, petname=name)
        for n in TASKS:
            lamella.task_config[n] = AcquireReferenceImageConfig(task_name=n)
        exp.positions.append(lamella)
    return exp


@pytest.fixture
def queue(experiment) -> TaskQueue:
    q = TaskQueue()
    q.build_from_matrix(["Trench"], [p.name for p in experiment.positions])
    q.next()  # L1 Trench running; L2 Trench pending
    return q


def add_batch(queue: TaskQueue, lamella_names, task_names, run_next: bool):
    """The ordering logic from AutoLamellaMainUI._on_add_to_queue."""
    added, anchor = [], None
    for task_name in task_names:
        for lamella_name in lamella_names:
            if not run_next:
                item = queue.add(lamella_name, task_name)
            elif anchor is None:
                item = queue.add(lamella_name, task_name, front=True)
            else:
                item = queue.add(lamella_name, task_name, after=anchor)
            anchor = item.id
            added.append(item)
    return added


def pairs(queue: TaskQueue):
    return [(i.lamella_name, i.task_name) for i in queue.items]


# ── Ordering ──────────────────────────────────────────────────────────────────


def test_added_at_the_end_keeps_task_outer_lamella_inner(queue):
    """Matching build_from_matrix, so added work interleaves like the original."""
    add_batch(queue, ["L1", "L2"], ["Undercut", "Polishing"], run_next=False)

    assert pairs(queue)[2:] == [
        ("L1", "Undercut"),
        ("L2", "Undercut"),
        ("L1", "Polishing"),
        ("L2", "Polishing"),
    ]


def test_run_next_keeps_the_batch_in_order(queue):
    """Regression: front=True on every add puts each new item ahead of the last,
    so the batch comes out backwards. Anchor each after the previous instead."""
    add_batch(queue, ["L1", "L2"], ["Undercut", "Polishing"], run_next=True)

    assert [(i.lamella_name, i.task_name) for i in queue.pending][:4] == [
        ("L1", "Undercut"),
        ("L2", "Undercut"),
        ("L1", "Polishing"),
        ("L2", "Polishing"),
    ]


def test_run_next_lands_after_the_running_item_not_before_it(queue):
    add_batch(queue, ["L1"], ["Polishing"], run_next=True)

    assert pairs(queue)[0] == ("L1", "Trench")  # still running, still first
    assert pairs(queue)[1] == ("L1", "Polishing")


def test_adding_never_disturbs_the_running_item(queue):
    running = queue.active
    add_batch(queue, ["L1", "L2"], ["Undercut"], run_next=True)

    assert queue.active.id == running.id
    assert queue.items[0].status is Status.InProgress


# ── Duplicates are added, not dropped ─────────────────────────────────────────


def test_a_pair_already_queued_is_added_again(queue):
    """The operator asked for it. _should_skip has no already-completed check —
    which is what makes Run again work — so the duplicate really does run again,
    and the confirm dialog says so rather than silently dropping it."""
    before = len(queue.items)
    add_batch(queue, ["L2"], ["Trench"], run_next=False)

    assert len(queue.items) == before + 1
    assert pairs(queue).count(("L2", "Trench")) == 2


def test_what_the_dialog_reports_as_already_queued(queue):
    """The set the confirm dialog warns about: pending pairs only."""
    pending = {(i.lamella_name, i.task_name) for i in queue.pending}
    lamella_names, task_names = ["L1", "L2"], ["Trench"]
    already = [
        f"{t} for {ln}"
        for t in task_names
        for ln in lamella_names
        if (ln, t) in pending
    ]

    # L1 Trench is running, not pending, so it is not flagged; L2 Trench is
    assert already == ["Trench for L2"]


# ── Task config backfill ──────────────────────────────────────────────────────


def test_a_lamella_missing_a_task_config_gets_it_from_the_base_protocol(experiment):
    """run_task raises if the lamella has no config for the task, and lamellae get
    a deepcopy of the protocol at creation — so one made before a task joined the
    protocol has no config for it. apply_lamella_config had no callers before this."""
    lamella = experiment.get_lamella_by_name("L2")
    del lamella.task_config["Polishing"]
    assert "Polishing" not in lamella.task_config

    updated = experiment.apply_lamella_config(["L2"], ["Polishing"])

    assert updated == 1
    assert "Polishing" in lamella.task_config


def test_backfill_leaves_other_lamellae_alone(experiment):
    l1 = experiment.get_lamella_by_name("L1")
    l1.task_config["Polishing"].task_name = "tuned-by-hand"

    experiment.apply_lamella_config(["L2"], ["Polishing"])

    assert l1.task_config["Polishing"].task_name == "tuned-by-hand"


def test_backfill_is_a_noop_when_the_protocol_also_lacks_the_task(experiment):
    """Nothing to copy from — must not raise, and must not invent a config."""
    lamella = experiment.get_lamella_by_name("L2")
    del lamella.task_config["Polishing"]
    del experiment.task_protocol.task_config["Polishing"]

    experiment.apply_lamella_config(["L2"], ["Polishing"])

    assert "Polishing" not in lamella.task_config


def test_only_lamellae_actually_missing_the_config_are_backfilled(experiment):
    """What the handler computes before calling apply_lamella_config."""
    experiment.get_lamella_by_name("L2").task_config.pop("Undercut")
    lamellae = experiment.positions
    task_names = ["Undercut"]

    missing = [
        lam.name
        for lam in lamellae
        if any(t not in lam.task_config for t in task_names)
    ]

    assert missing == ["L2"]


# ── The header button ─────────────────────────────────────────────────────────


@pytest.fixture
def widget(qapp, queue) -> WorkflowProgressWidget:
    w = WorkflowProgressWidget()
    w.set_workflow(queue.items)
    return w


def test_the_add_button_shares_the_queue_editing_gate(widget):
    """Same question as the row menus: is there a live queue to edit."""
    assert not widget._btn_add.isVisibleTo(widget)

    widget.set_actions_enabled(True)
    assert widget._btn_add.isVisibleTo(widget)

    widget.set_actions_enabled(False)
    assert not widget._btn_add.isVisibleTo(widget)


def test_the_button_starts_disabled_with_nothing_selected(widget):
    """Shown but greyed rather than absent, so it says what it needs."""
    widget.set_actions_enabled(True)

    assert widget._btn_add.isVisibleTo(widget)
    assert not widget._btn_add.isEnabled()
    assert "Select a lamella and a task" in widget._btn_add.toolTip()


def test_a_selection_enables_the_button(widget):
    widget.set_actions_enabled(True)
    widget.set_add_enabled(True, "Add to queue: 2 lamella, 1 task")
    assert widget._btn_add.isEnabled()

    widget.set_add_enabled(False, "Select a task to add to the queue")
    assert not widget._btn_add.isEnabled()
    assert widget._btn_add.toolTip() == "Select a task to add to the queue"


def entries(widget) -> list:
    """The menu's selectable actions, ignoring the heading and separator."""
    return [
        a
        for a in widget._btn_add.menu().actions()
        if not a.isSeparator() and a.isEnabled()
    ]


def test_the_menu_is_headed_by_what_it_does(widget):
    """A disabled action rather than addSection(), whose text the styled QMenu
    does not draw — the heading would have silently vanished."""
    actions = widget._btn_add.menu().actions()
    assert actions[0].text() == "Add to queue:"
    assert not actions[0].isEnabled()


def test_the_menu_offers_both_destinations(widget):
    """The whole button opens the menu, so neither option hides behind an arrow."""
    from PyQt5.QtWidgets import QToolButton

    assert widget._btn_add.text() == "Add to Queue"
    assert widget._btn_add.popupMode() == QToolButton.InstantPopup
    assert [a.text() for a in entries(widget)] == [
        "Add to End of Queue",
        "Add to Run Next",
    ]


def test_each_menu_entry_asks_for_its_own_destination(widget):
    widget.set_actions_enabled(True)
    widget.set_add_enabled(True)
    seen = []
    widget.add_to_queue_requested.connect(seen.append)

    end, run_next = entries(widget)
    end.trigger()
    run_next.trigger()

    assert seen == [False, True]


def test_a_disabled_button_reaches_no_menu(widget):
    """Nothing selected: the button cannot be opened, so neither entry is live."""
    widget.set_actions_enabled(True)
    assert not widget._btn_add.isEnabled()
