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
