"""The gateway's window: what it says, and that closing it asks first."""

import threading

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QMessageBox  # noqa: E402

from fibsem.gateway.gateway import OFF, RUNNING, STARTING, Gateway  # noqa: E402
from fibsem.gateway.server import GatewayAuth  # noqa: E402
from fibsem.gateway.ui import window as module  # noqa: E402
from fibsem.gateway.ui.window import GatewayWindow, mode_colour  # noqa: E402
from fibsem.theme.tokens import PRIMARY_COLOR, SEMANTIC_WARNING_COLOR  # noqa: E402


class Agent:
    def __init__(self):
        self.state = None

    def running(self):
        return self.state


@pytest.fixture
def made(qapp, tmp_path, monkeypatch):
    gateway = Gateway(home=tmp_path, discovery=lambda: None)
    auth = GatewayAuth.from_disk(tmp_path / "token.json", tmp_path / "paired.json")
    agent = Agent()
    stopping = threading.Event()
    quits = []
    monkeypatch.setattr(module, "quit_gateway", lambda g: quits.append(g))
    monkeypatch.setattr(qapp, "quit", lambda: None)
    window = GatewayWindow(gateway, auth, 8010, agent, stopping)
    yield window, gateway, agent, stopping, quits
    window._timer.stop()
    window._quitting = True
    window.close()


def test_it_says_what_runs(made):
    window, gateway, agent, _, _ = made
    assert window.app_label.text() == "The app is off."
    assert window.agent_label.text() == "No agent."
    assert window.experiment_label.isHidden()
    gateway._state.mode, gateway._state.window = RUNNING, True
    gateway._state.app, gateway._state.experiment = "AutoLamella", "/data/grid-03"
    agent.state = "running"
    window.refresh()
    assert window.app_label.text() == "AutoLamella is running with its window."
    assert window.experiment_label.text() == "grid-03"
    assert window.agent_label.text() == "The agent is running."
    gateway._state.window = False
    agent.state = "waiting"
    window.refresh()
    assert window.app_label.text() == "AutoLamella is running without a window."
    assert window.agent_label.text() == "The agent is waiting for the app."


def test_closing_asks_and_no_keeps_the_gateway(made, monkeypatch):
    window, _, _, _, quits = made
    asked = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda parent, title, text: asked.append(text) or QMessageBox.No,
    )
    window.show()
    window.stop_button.click()
    assert window.isVisible() and quits == []
    assert "Stop the gateway?" in asked[0]
    monkeypatch.setattr(
        QMessageBox, "question", lambda parent, title, text: QMessageBox.Yes
    )
    window.close()
    assert not window.isVisible() and len(quits) == 1


def test_a_stop_from_elsewhere_closes_it_without_asking(made, monkeypatch):
    window, _, _, stopping, quits = made
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a: pytest.fail("asked on a stop")
    )
    window.show()
    stopping.set()
    window.refresh()
    assert not window.isVisible() and len(quits) == 1


def test_the_question_says_what_happens_to_the_app(made):
    from fibsem.gateway.gateway import GatewayState

    assert "keeps running" in module.stop_question(
        GatewayState(version="0", mode=RUNNING, window=True)
    )
    assert "stops too" in module.stop_question(
        GatewayState(version="0", mode=RUNNING, window=False)
    )
    assert "page stops" in module.stop_question(GatewayState(version="0", mode=OFF))


def test_the_badge_follows_the_app():
    from fibsem.gateway.gateway import GatewayState

    assert (
        mode_colour(GatewayState(version="0", mode=RUNNING, window=True))
        == PRIMARY_COLOR
    )
    starting = GatewayState(version="0", mode=OFF, switch=STARTING, switching=True)
    assert mode_colour(starting) == SEMANTIC_WARNING_COLOR
