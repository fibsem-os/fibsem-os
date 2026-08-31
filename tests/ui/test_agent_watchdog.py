"""The agent watchdog: a question addressed to the agent stays purple and
quiet while the timer counts; expiry hands it to the operator with the
ordinary waiting chrome. Lives in the app so it survives the agent's own
loop dying.

Same offscreen main-window harness as the supervisor-chrome tests."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time

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

    original = module.AutoLamellaSingleWindowUI.add_minimap_tab
    module.AutoLamellaSingleWindowUI.add_minimap_tab = lambda self: None
    try:
        window = module.AutoLamellaSingleWindowUI()
    finally:
        module.AutoLamellaSingleWindowUI.add_minimap_tab = original
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
def agent_question_standing(main_ui, tmp_path):
    """An agent-designated task with a question standing (waiting flag up)."""
    ui = main_ui.autolamella_ui
    experiment = Experiment(path=tmp_path / "exp", name="watchdog-exp")
    experiment.task_protocol = AutoLamellaTaskProtocol()
    experiment.task_protocol.workflow_config.tasks.append(
        AutoLamellaTaskDescription(
            name="Mill Fiducial", supervise=True, required=True, supervisor="agent"
        )
    )
    experiment.add_new_lamella(MicroscopeState(), EventedDict())
    previous = (
        ui.experiment,
        ui._agent_server_host,
        main_ui._current_task_name,
        ui.WAITING_FOR_USER_INTERACTION,
    )
    ui.experiment = experiment
    ui._agent_server_host = _RunningHost()
    main_ui._current_task_name = "Mill Fiducial"
    ui.WAITING_FOR_USER_INTERACTION = True
    yield experiment
    main_ui._agent_watchdog.stop()
    main_ui._agent_watchdog_expired = False
    (
        ui.experiment,
        ui._agent_server_host,
        main_ui._current_task_name,
        ui.WAITING_FOR_USER_INTERACTION,
    ) = previous
    main_ui._refresh_workflow_indicators()


def test_an_agent_question_holds_purple_and_quiet(main_ui, agent_question_standing):
    main_ui._on_question_event("prompt_raised", {})
    assert main_ui._agent_watchdog.isActive()
    assert main_ui._border_state == "agent"  # not "waiting"
    assert main_ui.user_attention_btn.isHidden()


def test_an_answer_disarms_the_watchdog(main_ui, agent_question_standing):
    main_ui._on_question_event("prompt_raised", {})
    assert main_ui._agent_watchdog.isActive()
    main_ui.autolamella_ui.WAITING_FOR_USER_INTERACTION = False
    main_ui._on_question_event(
        "prompt_answered", {"answered_by": "agent", "response": True}
    )
    assert not main_ui._agent_watchdog.isActive()
    assert main_ui._agent_watchdog_expired is False


def test_expiry_hands_the_question_to_the_operator(
    main_ui, agent_question_standing, qapp, monkeypatch
):
    # The deadline comes from Preferences -> Agent; shorten it at the seam.
    monkeypatch.setattr(main_ui, "_watchdog_ms", lambda: 50)
    main_ui._on_question_event("prompt_raised", {})
    assert main_ui._border_state == "agent"

    deadline = time.monotonic() + 5
    while not main_ui._agent_watchdog_expired and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)

    assert main_ui._agent_watchdog_expired is True
    assert main_ui._border_state == "waiting"  # the ordinary chrome took over
    assert not main_ui.user_attention_btn.isHidden()


def test_a_human_question_escalates_immediately(main_ui, agent_question_standing):
    # Same standing question, but designated back to the human: no hold.
    task = agent_question_standing.task_protocol.workflow_config.tasks[-1]
    task.supervisor = "human"
    main_ui._on_question_event("prompt_raised", {})
    assert not main_ui._agent_watchdog.isActive()
    assert main_ui._border_state == "waiting"
    assert not main_ui.user_attention_btn.isHidden()


def test_expiry_with_nothing_standing_is_a_noop(main_ui, agent_question_standing):
    main_ui.autolamella_ui.WAITING_FOR_USER_INTERACTION = False
    main_ui._on_agent_watchdog_expired()
    assert main_ui._agent_watchdog_expired is False
