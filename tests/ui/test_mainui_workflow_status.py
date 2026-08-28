"""What the main window does with a fire-and-forget status payload, pinned pre-move.

`AutoLamellaSingleWindowUI._on_workflow_update` is the main-window consumer of
`workflow_update_signal`: transient status-bar text, and the task manager's
lifecycle reports (timeline feed, the "Workflow: ..." message, run/stop buttons).
Status is about to move to its own signal, so this pins the observable behaviour
the move must preserve.

This is the first test to construct the real main window offscreen. The only
thing that ever prevented it is `add_minimap_tab`, which builds a `napari.Viewer`
whose vispy canvas calls `glGetIntegerv` with no GL context and takes the process
down with SIGSEGV — so the fixture stubs that one method out. Nothing here goes
near the minimap; every other tab is real. When the minimap tab is deleted
(#585/#586) the stub can go.

Latent init gap, found by these tests: `_border_state` is first assigned on the
Run-workflow click, never in `__init__`, and `_on_workflow_update` reads it
unconditionally — so a status payload arriving without a prior Run click is an
AttributeError inside a queued slot, which PyQt5 turns into a process abort
(FIB-329). Production is protected by flow order only; the fixture supplies the
same precondition explicitly.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus
from fibsem.applications.autolamella.workflows.tasks.status import (
    WorkflowStatusUpdate,
)


@pytest.fixture(scope="module")
def main_ui(qapp):
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module

    original = module.AutoLamellaSingleWindowUI.add_minimap_tab
    module.AutoLamellaSingleWindowUI.add_minimap_tab = lambda self: None
    try:
        window = module.AutoLamellaSingleWindowUI()
    finally:
        module.AutoLamellaSingleWindowUI.add_minimap_tab = original
    # First connect builds the protocol editor's full UI (its lock path runs on
    # every lifecycle report); Demo, so no hardware.
    window.autolamella_ui.system_widget.connect_to_microscope()
    # What _on_run_workflow_clicked does before any status can arrive: the
    # border state exists only from the Run click onward, and the handler under
    # test reads it unconditionally. A bare window never receives status in
    # production -- see the latent-init note in the module docstring.
    window._set_border_state("idle")
    yield window
    if window.autolamella_ui.microscope is not None:
        window.autolamella_ui.microscope.disconnect()
    window.close()


def test_a_transient_status_bar_message_is_shown(main_ui):
    main_ui._on_workflow_update({"status_bar": "Scheduled start in 5 s"})

    assert main_ui.status_bar.currentMessage() == "Scheduled start in 5 s"


def test_a_payload_without_status_bar_text_leaves_the_bar_alone(main_ui):
    main_ui.status_bar.showMessage("previous message")

    main_ui._on_workflow_update({"msg": "not for the status bar"})

    assert main_ui.status_bar.currentMessage() == "previous message"


def test_a_lifecycle_report_shows_the_run_on_the_status_bar(main_ui):
    # queue_items=None on purpose: "no snapshot" leaves the timeline rows alone,
    # so this exercises the status-bar half without building a queue. The
    # timeline half is characterised in test_workflow_timeline_sync.
    report = WorkflowStatusUpdate(
        task_name="Rough Milling",
        item_name="lamella-01",
        status=AutoLamellaTaskStatus.InProgress,
        queue_position=2,
        queue_total=5,
        queue_items=None,
    )

    main_ui._on_workflow_update({"msg": "", "status": report})

    assert (
        main_ui.status_bar.currentMessage()
        == "Workflow: Rough Milling | lamella-01 | 2/5"
    )


def test_a_lifecycle_report_flips_the_run_button_to_stop(main_ui):
    main_ui.hide_workflow_running()
    assert not main_ui.stop_workflow_btn.isVisibleTo(main_ui)

    report = WorkflowStatusUpdate(
        task_name="Rough Milling",
        item_name="lamella-01",
        status=AutoLamellaTaskStatus.InProgress,
    )
    main_ui._on_workflow_update({"msg": "", "status": report})

    assert main_ui.stop_workflow_btn.isVisibleTo(main_ui)
    assert not main_ui.run_workflow_btn.isVisibleTo(main_ui)
