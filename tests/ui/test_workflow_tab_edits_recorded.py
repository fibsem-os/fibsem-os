"""The Workflow tab's plan edits go on the record as the App Server's do: one
``workflow_changed`` per change, with the server's change dicts."""

import os
from dataclasses import replace
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.ui import window as window_module  # noqa: E402
from fibsem.applications.autolamella.ui.workflow import (  # noqa: E402
    tab as workflow_module,
)
from tests.protocols import planned, task_protocol  # noqa: E402

Tab = workflow_module.WorkflowTab


def _tab():
    protocol = task_protocol(planned(name="Trench"), planned(name="Undercut"))
    changes = []
    tab = SimpleNamespace(
        _session=SimpleNamespace(
            experiment=SimpleNamespace(task_protocol=protocol),
            runner=SimpleNamespace(run_refusal=lambda what: ""),
        ),
        _on_config_changed=lambda config, *made: changes.extend(made),
        lamella_workflow_widget=SimpleNamespace(
            get_tasks=lambda: list(protocol.workflow_config.tasks),
            workflow=SimpleNamespace(refresh_all=lambda: None),
        ),
    )
    tab._put_back_if_running = lambda what: False
    tab._save_workflow_config = lambda change: Tab._save_workflow_config(tab, change)
    return tab, protocol, changes


def test_a_reorder_is_the_server_s_change():
    tab, protocol, changes = _tab()
    trench, undercut = protocol.workflow_config.tasks
    Tab._on_workflow_task_order_changed(tab, [undercut, trench])
    assert changes == [{"field": "order", "order": ["Undercut", "Trench"]}]


def test_a_removal_and_an_addition_are_the_server_s_changes():
    tab, protocol, changes = _tab()
    undercut = protocol.workflow_config.tasks[1]
    Tab._on_task_removed(tab, undercut)
    Tab._on_task_added(tab, undercut)
    assert changes == [
        {"field": "tasks", "removed": "Undercut"},
        {"field": "tasks", "added": "Undercut"},
    ]


def test_edit_task_records_only_what_changed():
    tab, protocol, changes = _tab()
    undercut = protocol.workflow_config.get("Undercut")
    Tab._on_task_edited(tab, replace(undercut), [])
    assert changes == []
    Tab._on_task_edited(tab, replace(undercut, required=False), ["Trench"])
    assert changes == [
        {
            "field": "requirements",
            "task_name": "Undercut",
            "requires": ["Trench"],
            "required": False,
        }
    ]


def test_the_window_saves_and_records_each_change(tmp_path):
    saves, recorded = [], []
    experiment = SimpleNamespace(
        task_protocol=SimpleNamespace(workflow_config=None),
        save=lambda **kwargs: saves.append(kwargs),
    )
    window = SimpleNamespace(
        experiment=experiment,
        session=SimpleNamespace(record_event=lambda *e: recorded.append(e)),
        panel=SimpleNamespace(refresh=lambda: None),
    )
    config = object()
    change = {"field": "turn_beams_off", "value": True}
    window_module.AutoLamellaWindow._on_workflow_config_changed(window, config, change)
    assert experiment.task_protocol.workflow_config is config
    assert saves == [{"save_protocol": True}]
    assert recorded == [("workflow_changed", change)]
