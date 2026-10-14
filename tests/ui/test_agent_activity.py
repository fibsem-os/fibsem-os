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


class _Views:
    """Records the overlays put on the views (the quad view's reducer API)."""

    def __init__(self):
        self.overlays = {}

    def set_overlay(self, beam, spec):
        self.overlays[(beam, spec.id)] = spec

    def remove_overlay(self, beam, overlay_id):
        self.overlays.pop((beam, overlay_id), None)


@pytest.fixture
def demo(qapp, monkeypatch):
    from fibsem import config as cfg
    from fibsem import utils

    monkeypatch.setenv("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(
        config_path=cfg.DEFAULT_CONFIGURATION_PATH,
        manufacturer="Demo",
        setup_logging=False,
    )
    views = _Views()
    bar = FibsemStatusBar()
    activity = AgentActivity(bar, view_controller=lambda: views)
    activity.set_microscope(microscope)
    yield microscope, activity, views
    activity.set_microscope(None)
    bar.deleteLater()


def _rectangle():
    from fibsem.structures import FibsemRectangleSettings

    return FibsemRectangleSettings(
        width=10e-6, height=5e-6, depth=1e-6, centre_x=0, centre_y=0
    )


def _recipe():
    from fibsem.structures import FibsemMillingSettings

    return FibsemMillingSettings(
        milling_current=7.6e-9, milling_voltage=30e3, hfw=80e-6
    )


def test_the_agents_patterns_are_drawn_on_the_fib_view_until_the_run_ends(demo):
    from fibsem.structures import BeamType

    microscope, activity, views = demo
    with microscope.control.take(AGENT, "prepare"):
        microscope.milling.prepare(_recipe(), [_rectangle(), _rectangle()])
    spec = views.overlays[(BeamType.ION, agent_activity.OVERLAY_ID)]
    (stage,) = spec.stages
    assert stage.name == "Agent"
    assert len(stage.define_patterns()) == 2
    with microscope.control.take(AGENT, "run"):
        microscope.milling.run()
    assert views.overlays == {}
    assert not activity.showing_patterns


def test_a_runs_patterns_are_the_milling_panels_not_the_agents(demo):
    microscope, activity, views = demo
    with microscope.control.take(WORKFLOW, "a lamellae run"):
        microscope.milling.prepare(_recipe(), [_rectangle()])
    assert views.overlays == {}
    assert not activity.showing_patterns
