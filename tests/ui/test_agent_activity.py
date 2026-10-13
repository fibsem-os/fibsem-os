"""The window says what the agent is doing on the microscope (FIB-1375): the status
bar's line while the agent is in control, and a toast for an action that runs on."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402

pytest.importorskip("PyQt5")

from fibsem.acting import AGENT, OPERATOR  # noqa: E402
from fibsem.applications.autolamella.ui import agent_activity  # noqa: E402
from fibsem.applications.autolamella.ui.agent_activity import (  # noqa: E402
    AgentActivity,
)
from fibsem.control import WORKFLOW, Control  # noqa: E402
from fibsem.ui.widgets.status_bar import FibsemStatusBar  # noqa: E402


def _wait(ms: int) -> None:
    from PyQt5.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec_()


@pytest.fixture
def toasts(monkeypatch):
    shown = []
    monkeypatch.setattr(
        agent_activity.notification_service,
        "show_toast",
        lambda message, kind="info": shown.append(message),
    )
    return shown


@pytest.fixture
def window(qapp):
    bar = FibsemStatusBar()
    activity = AgentActivity(bar)
    # Only `control` is read from the microscope; the Control is the real one.
    microscope = SimpleNamespace(control=Control())
    activity.set_microscope(microscope)
    yield bar, activity, microscope.control
    activity.set_microscope(None)
    bar.deleteLater()


def test_the_line_follows_the_agent_in_control(window, toasts):
    bar, activity, control = window
    with control.take(AGENT, "focus sweep"):
        assert activity.action == "focus sweep"
        assert bar.text.startswith("Agent focus sweep")
    assert activity.action is None
    assert bar.showing != "agent"
    assert toasts == []  # over at once: no toast


def test_a_run_or_the_operator_in_control_is_not_the_agent(window):
    bar, activity, control = window
    for by in (WORKFLOW, OPERATOR):
        with control.take(by, "a lamellae run"):
            assert activity.action is None
            assert bar.showing != "agent"


def test_the_agent_borrowing_a_runs_control_is_shown_and_given_back(window):
    _bar, activity, control = window
    with control.take(WORKFLOW, "a lamellae run"):
        with control.lend():
            with control.take(AGENT, "move stage relative"):
                assert activity.action == "move stage relative"
        assert activity.action is None


def test_a_take_on_a_server_thread_reaches_the_line(window):
    _bar, activity, control = window
    taken, release = threading.Event(), threading.Event()

    def agent():
        with control.take(AGENT, "focus sweep"):
            taken.set()
            release.wait(5)

    thread = threading.Thread(target=agent)
    thread.start()
    taken.wait(5)
    _wait(50)
    assert activity.action == "focus sweep"
    release.set()
    thread.join(5)
    _wait(50)
    assert activity.action is None


def test_an_action_that_runs_on_is_announced_and_its_end_too(
    window, toasts, monkeypatch
):
    monkeypatch.setattr(agent_activity, "ANNOUNCE_MS", 10)
    _bar, _activity, control = window
    with control.take(AGENT, "focus sweep"):
        _wait(60)
        assert toasts == ["Agent: focus sweep"]
    assert len(toasts) == 2
    assert toasts[1].startswith("Agent: focus sweep ended after ")
