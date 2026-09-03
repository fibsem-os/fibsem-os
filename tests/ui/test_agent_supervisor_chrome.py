"""The agent-supervision chrome: purple only when there is actually an agent.

The supervisor designation is display-and-watchdog semantics, and the display
half is gated hard: with no agent server running, a task designated
``supervisor: agent`` looks exactly like plain supervised — none of the agent
chrome exists for a user who never enabled the feature.

Same offscreen main-window harness as test_mainui_workflow_status (minimap
stubbed; everything else real)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from psygnal.containers import EventedDict

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.structures import MicroscopeState


@pytest.fixture(scope="module")
def main_ui(qapp):
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module

    window = module.AutoLamellaSingleWindowUI()
    window.autolamella_ui.system_widget.connect_to_microscope()
    yield window
    if window.autolamella_ui.microscope is not None:
        window.autolamella_ui.microscope.disconnect()
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        window.close()
    finally:
        qapp.quit = original_quit


class _RunningHost:
    running = True


@pytest.fixture
def agent_supervised_task(main_ui, tmp_path):
    """A protocol whose Rough Milling is designated supervisor: agent."""
    ui = main_ui.autolamella_ui
    experiment = Experiment(path=tmp_path / "exp", name="chrome-exp")
    experiment.task_protocol = AutoLamellaTaskProtocol()
    experiment.task_protocol.workflow_config.tasks.append(
        AutoLamellaTaskDescription(
            name="Rough Milling",
            supervise=True,
            required=True,
            supervisor="agent",
        )
    )
    experiment.add_new_lamella(MicroscopeState(), EventedDict())
    previous = (ui.experiment, ui._agent_server_host, main_ui._current_task_name)
    ui.experiment = experiment
    main_ui._current_task_name = "Rough Milling"
    yield experiment
    ui.experiment, ui._agent_server_host, main_ui._current_task_name = previous


def test_designation_is_invisible_without_a_running_server(
    main_ui, agent_supervised_task
):
    main_ui.autolamella_ui._agent_server_host = None
    assert main_ui._update_supervised_status() is True
    assert main_ui.supervised_status_btn.text() == "Supervised"
    assert main_ui._running_border_state("Rough Milling") == "supervised"


def test_designation_shows_agent_chrome_with_a_running_server(
    main_ui, agent_supervised_task
):
    main_ui.autolamella_ui._agent_server_host = _RunningHost()
    assert main_ui._update_supervised_status() is True
    assert main_ui.supervised_status_btn.text() == "Agent"
    assert main_ui._running_border_state("Rough Milling") == "agent"


def test_an_unsupervised_task_is_automated_regardless(main_ui, agent_supervised_task):
    main_ui.autolamella_ui._agent_server_host = _RunningHost()
    task = agent_supervised_task.task_protocol.workflow_config.tasks[-1]
    task.supervise = False
    assert main_ui._update_supervised_status() is False
    assert main_ui.supervised_status_btn.text() == "Automated"
    assert main_ui._running_border_state("Rough Milling") == "automated"
