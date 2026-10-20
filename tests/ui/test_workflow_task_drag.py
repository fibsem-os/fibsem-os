"""A drag on the Workflow tab reorders through `AutoLamellaWorkflowConfig.reorder`,
the App Server's rule: a task dragged above one it requires is put back (FIB-1446)."""

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.ui.workflow import (
    tab as module,  # noqa: E402  # noqa: E402
)
from tests.protocols import planned, task_protocol


def _window(qapp, monkeypatch):
    protocol = task_protocol(
        planned(name="Trench"),
        planned(name="Undercut", requires=["Trench"]),
    )
    saved, shown, toasts = [], [], []
    monkeypatch.setattr(
        module.notification_service, "show_toast", lambda m, *a: toasts.append(m)
    )
    window = SimpleNamespace(
        _session=SimpleNamespace(experiment=SimpleNamespace(task_protocol=protocol)),
        _on_config_changed=saved.append,
        lamella_workflow_widget=SimpleNamespace(set_workflow_config=shown.append),
    )
    return window, protocol.workflow_config, saved, shown, toasts


def test_a_drag_above_a_requirement_is_put_back(qapp, monkeypatch):
    window, config, saved, shown, toasts = _window(qapp, monkeypatch)
    undercut, trench = config.tasks[1], config.tasks[0]

    module.WorkflowTab._on_workflow_task_order_changed(window, [undercut, trench])
    qapp.processEvents()

    assert config.workflow == ["Trench", "Undercut"]
    assert saved == [] and shown == [config] and toasts


def test_a_valid_drag_is_saved(qapp, monkeypatch):
    window, config, saved, shown, toasts = _window(qapp, monkeypatch)
    window._session.experiment.task_protocol.task_config["Undercut"].requires = []
    undercut, trench = config.tasks[1], config.tasks[0]

    module.WorkflowTab._on_workflow_task_order_changed(window, [undercut, trench])

    assert config.workflow == ["Undercut", "Trench"]
    assert saved == [config] and toasts == []
