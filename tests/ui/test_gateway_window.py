"""The gateway's window: what it says, its buttons, and that closing it asks
first."""

import threading
import types

import pytest

pytest.importorskip("qtpy.QtWidgets")

from qtpy.QtWidgets import QApplication, QLabel, QMessageBox  # noqa: E402

from fibsem.gateway.gateway import (  # noqa: E402
    OFF,
    RUNNING,
    STARTING,
    Gateway,
    GatewayState,
)
from fibsem.gateway.server import GatewayAuth  # noqa: E402
from fibsem.gateway.ui import window as module  # noqa: E402
from fibsem.gateway.ui.window import GatewayWindow, mode_colour  # noqa: E402
from fibsem.theme.tokens import PRIMARY_COLOR, SEMANTIC_WARNING_COLOR  # noqa: E402


class Agent:
    def __init__(self):
        self.run = None
        self.ended = []

    def state(self):
        return types.SimpleNamespace(running=self.run)

    def end(self, who, via):
        self.ended.append((who, via))


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def _answer(monkeypatch, answer, asked=None):
    def question(parent, title, text):
        if asked is not None:
            asked.append(text)
        return answer

    monkeypatch.setattr(QMessageBox, "question", question)


@pytest.fixture
def made(qapp, tmp_path, monkeypatch):
    gateway = Gateway(home=tmp_path, discovery=lambda: None)
    auth = GatewayAuth.from_disk(tmp_path / "token.json", tmp_path / "paired.json")
    agent = Agent()
    stopping = threading.Event()
    quits = []
    clock = Clock()
    monkeypatch.setattr(qapp, "quit", lambda: quits.append(1))
    # acts run at once, on this thread
    monkeypatch.setattr(
        module.threading,
        "Thread",
        lambda target, name, daemon: (
            types.SimpleNamespace(start=lambda: None)
            if name == "gateway-window"
            else types.SimpleNamespace(start=target)
        ),
    )
    window = GatewayWindow(gateway, auth, 8010, agent, stopping, None, clock)
    yield types.SimpleNamespace(
        window=window,
        gateway=gateway,
        agent=agent,
        stopping=stopping,
        quits=quits,
        clock=clock,
        auth=auth,
    )
    window._timer.stop()
    window._quitting = True
    window.close()


def test_it_says_what_runs(made):
    window, gateway = made.window, made.gateway
    assert "Off" in window.app.line.text()
    assert window.app.button.text() == "Start"
    assert "Connects once the app runs" in window.microscope.line.text()
    assert "No agent" in window.agent.line.text()
    assert window.agent.button.isHidden()
    gateway._state.mode, gateway._state.window = RUNNING, True
    gateway._state.app, gateway._state.experiment = "AutoLamella", "/data/grid-03"
    window.refresh()
    window._show_read(
        {"state": "running", "level": "assist", "goal": "Mill 01 to 04"},
        {"connected": True, "configuration": {"name": "arctis"}},
    )
    assert "AutoLamella, with its window" in window.app.line.text()
    assert window.app.detail.text() == "grid-03"
    assert window.app.button.text() == "Stop"
    assert "Connected · arctis" in window.microscope.line.text()
    assert "Running at Assist" in window.agent.line.text()
    assert window.agent.detail.text() == "Mill 01 to 04"
    assert not window.agent.button.isHidden()


def test_an_agent_from_another_computer_that_went_quiet_says_so(made):
    window = made.window
    quiet = "Not heard from for 3 min. If it runs on another computer, its SSH tunnel may be down."
    window._show_read(None, None, {"name": "agent-laptop", "quiet": quiet})
    assert "agent-laptop connected" in window.agent.line.text()
    assert window.agent.detail.text() == quiet
    assert window.agent.button.isHidden()


def test_start_starts_the_only_app_and_stop_asks_first(made, monkeypatch):
    window, gateway = made.window, made.gateway
    started, stopped = [], []
    monkeypatch.setattr(
        gateway,
        "state",
        lambda: types.SimpleNamespace(
            apps=[types.SimpleNamespace(key="autolamella", name="AutoLamella")]
        ),
    )
    monkeypatch.setattr(gateway, "start_app", lambda who, via, app: started.append(app))
    monkeypatch.setattr(
        gateway, "stop", lambda who, via, force, window: stopped.append(force)
    )
    window.app.button.click()
    assert started == ["autolamella"]
    gateway._state.mode, gateway._state.window = RUNNING, True
    window.refresh()
    _answer(monkeypatch, QMessageBox.StandardButton.No)
    window.app.button.click()
    assert stopped == []
    _answer(monkeypatch, QMessageBox.StandardButton.Yes)
    window.app.button.click()
    assert stopped == [False]


def test_end_ends_the_agent_after_asking(made, monkeypatch):
    _answer(monkeypatch, QMessageBox.StandardButton.Yes)
    made.window._end_agent()
    assert made.agent.ended == [("operator", "window")]


def test_the_pairing_code_shows_and_goes_after_five_minutes(made):
    window = made.window
    assert window.pair_button.text() == "Pair"
    window.pair_button.click()
    code = window.code_box.line.text()
    assert window.pair_button.text() == "Hide" and len(code) == 4
    assert made.auth.pairing.state()["code"] == code
    made.clock.now = module.CODE_TTL_S + 1
    window.refresh()
    assert window.pair_button.text() == "Pair" and window.code_box.detail.isHidden()
    window.pair_button.click()
    window.pair_button.click()
    assert window.pair_button.text() == "Pair", "Hide hides the code"


def test_closing_asks_and_no_keeps_the_gateway(made, monkeypatch):
    window = made.window
    asked = []
    _answer(monkeypatch, QMessageBox.StandardButton.No, asked)
    window.show()
    window.stop_button.click()
    assert window.isVisible() and made.quits == []
    assert "Stop the gateway?" in asked[0]
    _answer(monkeypatch, QMessageBox.StandardButton.Yes)
    window.close()
    assert not window.isVisible() and len(made.quits) == 1


def test_a_stop_from_elsewhere_closes_it_without_asking(made, monkeypatch):
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a: pytest.fail("asked on a stop")
    )
    made.window.show()
    made.stopping.set()
    made.window.refresh()
    assert not made.window.isVisible() and len(made.quits) == 1


def test_the_question_says_what_happens_to_the_app():
    def state(**kw):
        return GatewayState(version="0", **kw)

    assert "keeps running" in module.stop_question(state(mode=RUNNING, window=True))
    assert "stops too" in module.stop_question(state(mode=RUNNING, window=False))
    assert "page stops" in module.stop_question(state(mode=OFF))
    running = state(mode=RUNNING, window=False)
    assert "takes it back" in module.stop_question(running, restart=True)
    assert mode_colour(state(mode=RUNNING, window=True)) == PRIMARY_COLOR
    starting = state(mode=OFF, switch=STARTING, switching=True)
    assert mode_colour(starting) == SEMANTIC_WARNING_COLOR


def test_restart_asks_then_restarts(made, monkeypatch):
    window = made.window
    assert window.restart_button.isHidden(), "no restart without the gateway's hook"
    restarts = []
    window._restart = lambda: restarts.append(1)
    _answer(monkeypatch, QMessageBox.StandardButton.No)
    window.restart_gateway()
    assert restarts == []
    asked = []
    _answer(monkeypatch, QMessageBox.StandardButton.Yes, asked)
    window.restart_gateway()
    assert restarts == [1] and "starts again" in asked[0]


def test_a_long_goal_stays_on_one_line_with_the_rest_in_the_tooltip(made):
    window = made.window
    window.resize(420, 400)
    window.show()
    goal = "Testing session, do some stuff, demo microscope. " * 6
    window._show_read({"state": "running", "level": "drive", "goal": goal}, None)
    QApplication.processEvents()
    detail = window.agent.detail
    assert detail.text() == goal and detail.toolTip() == goal
    drawn = QLabel.text(detail)
    assert drawn != goal and drawn.endswith("…")
    assert detail.height() <= detail.fontMetrics().lineSpacing() + 4
