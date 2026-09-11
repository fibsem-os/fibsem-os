"""Rebuilding the run-selection lists keeps the operator's ticks (FIB-966, FIB-967).

Both lists are the run selection, and both are rebuilt on routine events: the lamella
list on every insert or removal in the experiment, the task list on every protocol
edit. Rebuilding them unticked emptied the selection and silently disabled Run. The
grid workflow widget already preserved its ticks across its own rebuild; these two now
do the same.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_rebuild_keeps_ticks.py
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaWorkflowConfig,
    Lamella,
)
from fibsem.applications.autolamella.ui.lamella_list_widget import LamellaListWidget
from fibsem.applications.autolamella.ui.workflow_config_widget import (
    WorkflowConfigWidget,
)

_app = QApplication.instance() or QApplication(sys.argv)


def _lamellae(n: int):
    return [
        Lamella(path="", number=i + 1, petname=f"lamella-{i + 1}") for i in range(n)
    ]


def _tasks(names):
    return [
        AutoLamellaTaskDescription(name=n, supervise=False, required=False)
        for n in names
    ]


def _tick(widget, rows):
    for i in rows:
        widget._row(i).checkbox.setChecked(True)


# ── lamella list ──────────────────────────────────────────────────────────


def test_set_lamellae_keeps_the_ticks_when_a_lamella_is_added():
    widget = LamellaListWidget()
    lamellae = _lamellae(3)
    widget.set_lamellae(lamellae)
    _tick(widget, [0, 2])

    widget.set_lamellae(lamellae + _lamellae(1))

    assert widget.get_selected() == [lamellae[0], lamellae[2]]
    widget.close()


def test_set_lamellae_drops_the_tick_of_a_removed_lamella_only():
    widget = LamellaListWidget()
    lamellae = _lamellae(3)
    widget.set_lamellae(lamellae)
    _tick(widget, [0, 1])

    widget.set_lamellae([lamellae[0], lamellae[2]])

    assert widget.get_selected() == [lamellae[0]]
    widget.close()


def test_set_lamellae_matches_by_id_not_by_name():
    widget = LamellaListWidget()
    old = _lamellae(1)
    widget.set_lamellae(old)
    _tick(widget, [0])
    twin = Lamella(path="", number=1, petname=old[0].petname)  # same name, new id

    widget.set_lamellae([twin])

    assert widget.get_selected() == []
    widget.close()


def test_set_lamellae_from_empty_starts_unticked():
    widget = LamellaListWidget()
    widget.set_lamellae(_lamellae(2))
    assert widget.get_selected() == []
    widget.close()


# ── task list ─────────────────────────────────────────────────────────────


def test_set_config_keeps_the_ticks_across_a_field_edit():
    widget = WorkflowConfigWidget()
    widget.set_config(AutoLamellaWorkflowConfig(tasks=_tasks(["a", "b", "c"])))
    _tick(widget, [0, 2])

    # A protocol edit hands over fresh task objects with the same names.
    widget.set_config(AutoLamellaWorkflowConfig(tasks=_tasks(["a", "b", "c"])))

    assert [t.name for t in widget.get_selected()] == ["a", "c"]
    widget.close()


def test_set_config_drops_the_tick_of_a_removed_task_only():
    widget = WorkflowConfigWidget()
    widget.set_config(AutoLamellaWorkflowConfig(tasks=_tasks(["a", "b", "c"])))
    _tick(widget, [0, 1])

    widget.set_config(AutoLamellaWorkflowConfig(tasks=_tasks(["a", "c"])))

    assert [t.name for t in widget.get_selected()] == ["a"]
    widget.close()


def test_set_config_from_empty_starts_unticked():
    widget = WorkflowConfigWidget()
    widget.set_config(AutoLamellaWorkflowConfig(tasks=_tasks(["a", "b"])))
    assert widget.get_selected() == []
    widget.close()
