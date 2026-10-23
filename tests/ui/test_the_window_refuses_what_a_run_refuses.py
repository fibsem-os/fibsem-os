"""While a workflow runs, the window refuses the edits the App Server refuses
then, with the same words: adding, removing, reordering or editing tasks, and
changing the experiment or protocol (`WorkflowRunner.run_refusal`)."""

import os
from datetime import datetime, timedelta
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.ui import window as window_module  # noqa: E402
from fibsem.applications.autolamella.ui.protocol import (  # noqa: E402
    protocol_tab as protocol_module,
)
from fibsem.applications.autolamella.ui.workflow import (
    tab as workflow_module,  # noqa: E402
)
from fibsem.ui import run_refusal  # noqa: E402
from fibsem.workflows.runner import WorkflowRunner  # noqa: E402
from fibsem.workflows.task_config import TaskPlan  # noqa: E402
from tests.protocols import planned, task_protocol  # noqa: E402


def _runner(running: bool):
    runner = SimpleNamespace(is_running=running)
    runner.run_refusal = lambda what: WorkflowRunner.run_refusal(runner, what)
    return runner


@pytest.fixture
def toasts(monkeypatch):
    said = []
    monkeypatch.setattr(
        run_refusal.notification_service, "show_toast", lambda m, *a: said.append(m)
    )
    return said


def test_the_words_are_the_app_server_s():
    assert _runner(False).run_refusal("add tasks") == ""
    assert (
        _runner(True).run_refusal("add tasks")
        == "A workflow is running; add tasks when it has stopped."
    )


@pytest.mark.parametrize(
    "method", ["create_experiment", "load_experiment", "load_protocol"]
)
def test_the_experiment_and_protocol_wait_for_the_run(method, toasts, monkeypatch):
    opened = []
    for dialog in (
        "create_experiment_dialog",
        "load_experiment_dialog",
        "load_task_protocol_dialog",
    ):
        monkeypatch.setattr(
            window_module, dialog, lambda *a, **k: opened.append(a) or None
        )
    window = SimpleNamespace(
        session=SimpleNamespace(runner=_runner(True)),
        microscope=object(),
        experiment=object(),
    )
    getattr(window_module.AutoLamellaWindow, method)(window)
    assert opened == []
    assert toasts == ["A workflow is running; change experiments when it has stopped."]


@pytest.mark.parametrize(
    "method, what",
    [
        ("_on_add_task_clicked", "add tasks"),
        ("_on_remove_task_clicked", "remove tasks"),
    ],
)
def test_the_protocol_tab_s_tasks_wait_for_the_run(method, what, toasts):
    editor = SimpleNamespace(session=SimpleNamespace(runner=_runner(True)))
    # Refused before it reads anything else: no dialog, no selection.
    getattr(protocol_module.AutoLamellaProtocolTaskConfigEditor, method)(editor)
    assert toasts == [f"A workflow is running; {what} when it has stopped."]


def _workflow_tab(running: bool):
    protocol = task_protocol(planned(name="Trench"), planned(name="Undercut"))
    saved, shown = [], []
    tab = SimpleNamespace(
        _session=SimpleNamespace(
            experiment=SimpleNamespace(task_protocol=protocol),
            runner=_runner(running),
        ),
        _on_config_changed=saved.append,
        _save_workflow_config=lambda *a: saved.append("saved"),
        lamella_workflow_widget=SimpleNamespace(set_workflow_config=shown.append),
    )
    tab._put_back_if_running = lambda what: (
        workflow_module.WorkflowTab._put_back_if_running(tab, what)
    )
    return tab, protocol.workflow_config, saved, shown


@pytest.mark.parametrize(
    "method, what",
    [
        ("_on_task_added", "add tasks"),
        ("_on_task_removed", "remove tasks"),
        ("_on_workflow_task_order_changed", "reorder tasks"),
    ],
)
def test_the_workflow_tab_s_tasks_are_put_back_during_a_run(method, what, toasts, qapp):
    tab, config, saved, shown = _workflow_tab(running=True)
    trench, undercut = config.tasks
    getattr(workflow_module.WorkflowTab, method)(tab, [undercut, trench])
    qapp.processEvents()
    assert config.workflow == ["Trench", "Undercut"]
    assert saved == [] and shown == [config]
    assert toasts == [f"A workflow is running; {what} when it has stopped."]


def test_a_removal_with_no_run_is_saved(toasts):
    tab, config, saved, shown = _workflow_tab(running=False)
    workflow_module.WorkflowTab._on_task_removed(tab, config.tasks[1])
    assert saved == ["saved"] and shown == [] and toasts == []


def test_edit_task_s_requirements_wait_for_the_run_and_its_schedule_does_not(
    monkeypatch,
):
    said = []
    monkeypatch.setattr(
        workflow_module.notification_service,
        "show_toast",
        lambda m, *a: said.append(m),
    )
    tab, config, saved, _ = _workflow_tab(running=True)
    tab._on_config_changed = lambda config, *changes: saved.extend(changes)
    tab.lamella_workflow_widget.workflow = SimpleNamespace(refresh_all=lambda: None)
    at = datetime.now() + timedelta(hours=1)
    edited = TaskPlan(name="Undercut", required=False, scheduled_at=at)
    workflow_module.WorkflowTab._on_task_edited(tab, edited, ["Trench"])
    protocol = tab._session.experiment.task_protocol
    assert protocol.requirements("Undercut") == [] and config.get("Undercut").required
    assert config.get("Undercut").scheduled_at == at
    assert said == ["A workflow is running; edit tasks when it has stopped."]
    assert saved == [
        {
            "field": "scheduled_at",
            "task_name": "Undercut",
            "scheduled_at": at.isoformat(),
        }
    ]


def test_the_lamella_template_waits_for_the_run(toasts):
    editor = SimpleNamespace(session=SimpleNamespace(runner=_runner(True)))
    # Refused before the dialog opens.
    protocol_module.AutoLamellaProtocolTaskConfigEditor._on_lamella_defaults_clicked(
        editor
    )
    assert toasts == [
        "A workflow is running; edit the lamella defaults when it has stopped."
    ]


def test_reading_the_stage_into_a_lamella_waits_for_the_run(toasts):
    from fibsem.applications.autolamella.ui.panel.experiment import ExperimentTab

    tab = SimpleNamespace(
        session=SimpleNamespace(runner=_runner(True)),
        microscope=object(),
        protocol=object(),
        experiment=SimpleNamespace(positions=[object()]),
    )
    # Refused before it reads the stage or asks.
    ExperimentTab.update_lamella_position_ui(tab)
    assert ExperimentTab.add_new_lamella(tab, stage_position=None) is None
    assert toasts == [
        "A workflow is running; set a pose from the stage when it has stopped.",
        "A workflow is running; add an item where the stage is when it has stopped.",
    ]
