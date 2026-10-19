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
from fibsem.applications.autolamella.ui.workflow_timeline_widget import (
    WorkflowProgressWidget,
)
from fibsem.applications.autolamella.workflows.tasks.reference_image import (
    AcquireReferenceImageConfig,
)
from fibsem.workflows.queue import TaskQueue
from fibsem.workflows.status import TaskStatus as Status

TASKS = ["Trench", "Undercut", "Polishing"]


@pytest.fixture
def experiment(tmp_path: Path) -> Experiment:
    exp = Experiment(path=tmp_path, name="test-exp")
    exp.task_protocol = AutoLamellaTaskProtocol(
        workflow_config=AutoLamellaWorkflowConfig(
            tasks=[AutoLamellaTaskDescription(name=n, required=False) for n in TASKS]
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


def runner_over(queue: TaskQueue, experiment, recorded=None):
    """A session's runner whose run is ``queue``'s, over ``experiment``;
    ``recorded`` collects the queue edits it records."""
    from types import SimpleNamespace

    from fibsem.workflows.runner import ITEMS
    from tests.run_state import stub_session

    manager = SimpleNamespace(
        queue=queue,
        notify_queue_changed=lambda: None,
        experiment=experiment,
        microscope=None,
        RUN_KIND=ITEMS,
        RUN_DESCRIPTION="a lamella run",
    )
    session = stub_session(manager)
    if recorded is not None:
        buffer = SimpleNamespace(
            append=lambda kind, payload, actor=None: recorded.append(
                (kind, payload, actor)
            )
        )
        session.app_server = SimpleNamespace(event_buffer=buffer)
    return session.runner


def add_batch(queue: TaskQueue, experiment, lamella_names, task_names, run_next):
    """Add to queue, as the button does: through the session's runner."""
    from fibsem.workflows.runner import ITEMS, RunRequest

    edit = runner_over(queue, experiment).add(
        RunRequest(
            kind=ITEMS, task_names=tuple(task_names), names=tuple(lamella_names)
        ),
        run_next,
    )
    assert edit.applied, edit.reason
    return list(edit.items)


def pairs(queue: TaskQueue):
    return [(i.item_name, i.task_name) for i in queue.items]


# ── Ordering ──────────────────────────────────────────────────────────────────


def test_added_at_the_end_keeps_task_outer_lamella_inner(queue, experiment):
    """Matching build_from_matrix, so added work interleaves like the original."""
    add_batch(queue, experiment, ["L1", "L2"], ["Undercut", "Polishing"], False)

    assert pairs(queue)[2:] == [
        ("L1", "Undercut"),
        ("L2", "Undercut"),
        ("L1", "Polishing"),
        ("L2", "Polishing"),
    ]


def test_run_next_keeps_the_batch_in_order(queue, experiment):
    """Regression: front=True on every add puts each new item ahead of the last,
    so the batch comes out backwards. Anchor each after the previous instead."""
    add_batch(queue, experiment, ["L1", "L2"], ["Undercut", "Polishing"], True)

    assert [(i.item_name, i.task_name) for i in queue.pending][:4] == [
        ("L1", "Undercut"),
        ("L2", "Undercut"),
        ("L1", "Polishing"),
        ("L2", "Polishing"),
    ]


def test_run_next_lands_after_the_running_item_not_before_it(queue, experiment):
    add_batch(queue, experiment, ["L1"], ["Polishing"], True)

    assert pairs(queue)[0] == ("L1", "Trench")  # still running, still first
    assert pairs(queue)[1] == ("L1", "Polishing")


def test_adding_never_disturbs_the_running_item(queue, experiment):
    running = queue.active
    add_batch(queue, experiment, ["L1", "L2"], ["Undercut"], True)

    assert queue.active.id == running.id
    assert queue.items[0].status is Status.InProgress


# ── Duplicates are added, not dropped ─────────────────────────────────────────


def test_a_pair_already_queued_is_added_again(queue, experiment):
    """The operator asked for it. _should_skip has no already-completed check —
    which is what makes Run again work — so the duplicate really does run again,
    and the confirm dialog says so rather than silently dropping it."""
    before = len(queue.items)
    add_batch(queue, experiment, ["L2"], ["Trench"], False)

    assert len(queue.items) == before + 1
    assert pairs(queue).count(("L2", "Trench")) == 2


def test_what_the_dialog_reports_as_already_queued(queue):
    """The set the confirm dialog warns about: pending pairs only."""
    pending = {(i.item_name, i.task_name) for i in queue.pending}
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
    """What the lamella run rules ready before the work is queued."""
    from fibsem.applications.autolamella.workflows.runs import (
        give_lamellae_their_configs,
    )
    from fibsem.workflows.runner import ITEMS, RunRequest

    l1 = experiment.get_lamella_by_name("L1")
    l1.task_config["Undercut"].task_name = "tuned-by-hand"
    experiment.get_lamella_by_name("L2").task_config.pop("Undercut")

    give_lamellae_their_configs(
        experiment,
        RunRequest(kind=ITEMS, task_names=("Undercut",), names=("L1", "L2")),
    )

    assert "Undercut" in experiment.get_lamella_by_name("L2").task_config
    assert l1.task_config["Undercut"].task_name == "tuned-by-hand"


def test_added_work_gets_its_configs_before_it_is_queued(queue, experiment):
    experiment.get_lamella_by_name("L2").task_config.pop("Polishing")

    add_batch(queue, experiment, ["L2"], ["Polishing"], False)

    assert "Polishing" in experiment.get_lamella_by_name("L2").task_config


def test_the_agent_s_requeue_gets_its_config_too(queue, experiment):
    """The App Server's requeue goes through the same rules as the button."""
    experiment.get_lamella_by_name("L2").task_config.pop("Polishing")

    edit = runner_over(queue, experiment).requeue("L2", "Polishing")

    assert edit.applied, edit.reason
    assert "Polishing" in experiment.get_lamella_by_name("L2").task_config


def test_an_operator_s_queue_edit_is_recorded(queue, experiment):
    """Before, only the agent's edits were: the operator's left no record."""
    from fibsem.acting import OPERATOR
    from fibsem.workflows.runner import ITEMS, RunRequest

    recorded = []
    runner = runner_over(queue, experiment, recorded)
    (added,) = runner.add(
        RunRequest(kind=ITEMS, task_names=("Undercut",), names=("L2",)),
        actor=OPERATOR,
    ).items
    runner.edit("L2", "Undercut", "remove", actor=OPERATOR)

    assert [
        (kind, p["action"], p["queue_item_id"], actor) for kind, p, actor in recorded
    ] == [
        ("queue_changed", "add", added.id, OPERATOR),
        ("queue_changed", "remove", added.id, OPERATOR),
    ]


def test_work_the_rules_refuse_is_not_queued(queue, experiment):
    from fibsem.workflows.runner import ITEMS, RunRequest

    before = len(queue.items)
    edit = runner_over(queue, experiment).add(
        RunRequest(kind=ITEMS, task_names=("No Such Task",), names=("L1",))
    )

    assert not edit.applied and edit.error_type == "refused"
    assert edit.details["task_names"] == TASKS
    assert len(queue.items) == before


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
