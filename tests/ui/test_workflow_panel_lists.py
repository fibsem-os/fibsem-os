"""The workflow panel's lists share one header, one row size and one type scale
(`list_chrome`), so they read as one widget rather than three.

They had a header each: a bold "Select All" over "Status" on the lamellae, a
larger "Select All" with a + on the tasks. Rows were 34 and 40 px, names 11 and
15 px, and a long task name was cut mid-word at the panel's width.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskDescription,
    Lamella,
)
from fibsem.applications.autolamella.ui import list_chrome  # noqa: E402
from fibsem.applications.autolamella.ui.lamella_list_widget import (  # noqa: E402
    LamellaListWidget,
)
from fibsem.applications.autolamella.ui.workflow_config_widget import (  # noqa: E402
    WorkflowConfigWidget,
)


@pytest.fixture
def lists(qapp, tmp_path):
    lamellae = LamellaListWidget()
    for n, name in enumerate(["01-tough-goose", "02-pro-moose"], start=1):
        lamellae.add_lamella(
            Lamella(petname=name, path=str(tmp_path / name), number=n), checked=True
        )
    tasks = WorkflowConfigWidget()
    for name, requires in [
        ("Setup Lamella Position", []),
        ("Acquire Fluorescence Image", []),
        ("Rough Milling", ["Mill Fiducial"]),
    ]:
        tasks.add_task(AutoLamellaTaskDescription(name=name, requires=requires))
    yield lamellae, tasks
    lamellae.deleteLater()
    tasks.deleteLater()


def test_both_lists_have_the_shared_header(lists):
    lamellae, tasks = lists
    for widget, title, count in ((lamellae, "Lamella", "2"), (tasks, "Tasks", "3")):
        assert isinstance(widget._header, list_chrome.ListHeader)
        assert widget._header.title_label.text() == title
        assert widget._header.count_label.text() == count


def test_both_lists_have_the_same_rows(lists):
    lamellae, tasks = lists
    for widget in lists:
        item = widget._list.item(0)
        assert item.sizeHint().height() == list_chrome.ROW_HEIGHT
    name_px = f"font-size: {list_chrome.NAME_PX}px"
    assert name_px in lamellae._row(0).name_label.styleSheet()
    assert name_px in tasks._row(0).name_label.styleSheet()


def test_a_task_without_a_dependency_gives_its_name_the_room(lists):
    _, tasks = lists
    no_dependency, with_dependency = tasks._row(1), tasks._row(2)
    assert no_dependency.requires_label.isHidden()
    assert not with_dependency.requires_label.isHidden()
    assert with_dependency.requires_label.text() == "after Mill Fiducial"


def test_the_count_follows_the_list(lists):
    lamellae, tasks = lists
    tasks.clear()
    assert tasks._header.count_label.text() == ""
    lamellae.clear()
    assert lamellae._header.count_label.text() == ""


def test_the_chip_is_the_tinted_one(lists):
    _, tasks = lists
    chip = tasks._row(0).btn_attention
    assert chip.height() == list_chrome.CHIP_HEIGHT
    assert "border: none" in chip.styleSheet() and "rgba(" in chip.styleSheet()


# --- the panel: settings behind ⚙, no footer (the status line says it) -------------


@pytest.fixture
def panel(qapp):
    from fibsem.applications.autolamella.ui.lamella_workflow_widget import (
        LamellaWorkflowWidget,
    )

    panel = LamellaWorkflowWidget()
    yield panel
    panel._settings_popup.hide()
    panel.deleteLater()


def test_the_workflow_settings_are_behind_the_gear(panel):
    """Name, description and Turn beams off were a quarter of the panel on every
    view; they open from the ⚙ on the task list's header."""
    assert panel.info.window() is panel._settings_popup, "not in the panel's layout"
    panel.show()
    panel.btn_settings.click()
    assert panel._settings_popup.isVisible()
    assert panel.info.name_edit.isVisible()


def test_the_gear_says_when_something_in_it_is_set(panel):
    from fibsem.applications.autolamella.structures import AutoLamellaWorkflowOptions

    assert panel.btn_settings.toolTip() == "Workflow settings"
    panel.set_options(AutoLamellaWorkflowOptions(turn_beams_off=True))
    assert panel.btn_settings.toolTip() == "Workflow settings (some set)"


def test_the_hints_are_the_question_marks_tooltip(panel):
    from PyQt5.QtWidgets import QLabel

    from fibsem.applications.autolamella.ui.lamella_workflow_widget import (
        TASK_LIST_HINTS,
    )

    assert panel.btn_help.toolTip() == TASK_LIST_HINTS
    texts = [label.text() for label in panel.findChildren(QLabel)]
    assert not any("Drag to reorder" in t for t in texts), "no footer line"
    assert not any("to run the workflow" in t for t in texts)


@pytest.fixture
def no_quit(qapp):
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        yield
    finally:
        qapp.quit = original_quit


def test_the_status_line_says_what_the_selection_allows(no_quit, qapp, tmp_path):
    """The footer's "Select a lamella and a task to run the workflow" is the status
    line's last instruction now, and it follows the ticks."""
    from fibsem.applications.autolamella.structures import (
        AutoLamellaTaskProtocol,
        Experiment,
    )
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module

    window = module.AutoLamellaSingleWindowUI()
    ui = window.autolamella_ui
    ui.system_widget.connect_to_microscope()
    experiment = Experiment(path=str(tmp_path), name="panel")
    experiment.positions.append(
        Lamella(petname="01-test", path=str(tmp_path / "01-test"), number=1)
    )
    ui.experiment = experiment
    experiment.task_protocol = AutoLamellaTaskProtocol()
    window._update_instructions()
    workflow = window.lamella_workflow_widget
    workflow.add_lamella(experiment.positions[0], checked=False)
    workflow.add_task(AutoLamellaTaskDescription(name="Polishing"), checked=True)
    window._on_workflow_selection_changed()
    assert window.status_bar.text == "Select a lamella to run the workflow."
    workflow.lamella_list.set_all_selected(True)
    window._on_workflow_selection_changed()
    assert window.status_bar.text == "Ready to run: 1 lamella, 1 task."
    ui.microscope.disconnect()
    window.close()
    window.deleteLater()
    qapp.processEvents()
