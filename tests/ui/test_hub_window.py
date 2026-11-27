"""The Hub's window: what it says of the instruments and the agents on this
PC, its pairing code, and that closing it asks first."""

import threading

import pytest

pytest.importorskip("qtpy.QtWidgets")

from qtpy.QtWidgets import QMessageBox  # noqa: E402

from fibsem.hub.hub import HubState, InstrumentState  # noqa: E402
from fibsem.hub.server import hub_auth  # noqa: E402
from fibsem.hub.ui.window import (  # noqa: E402
    HubWindow,
    agents_line,
    instruments_line,
)
from fibsem.theme.tokens import (  # noqa: E402
    GRAY_SECONDARY_COLOR,
    OK_COLOR,
    WARN_COLOR,
)


class Hub:
    def __init__(self):
        self.rows = []

    def state(self):
        return HubState(instruments=list(self.rows))


class Agents:
    def __init__(self):
        self.found = {}

    def state(self):
        return dict(self.found)


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


@pytest.fixture
def made(qapp, tmp_path, monkeypatch):
    hub, agents, clock = Hub(), Agents(), Clock()
    auth = hub_auth(tmp_path / "token.json", tmp_path / "paired.json")
    stopping, forward = threading.Event(), threading.Event()
    quits = []
    monkeypatch.setattr(qapp, "quit", lambda: quits.append(1))
    window = HubWindow(
        hub, auth, 8020, agents, stopping, clock, (threading.Event(), forward)
    )
    yield window, hub, agents, clock, stopping, forward, quits
    window._quitting = True
    window.close()


def test_the_instruments_row_says_who_needs_someone():
    assert instruments_line(HubState())[2] == GRAY_SECONDARY_COLOR
    rows = [
        InstrumentState("Scope 1", paired=True, reachable=True),
        InstrumentState("Scope 2", paired=True, reachable=False),
        InstrumentState("Scope 3", paired=False, reachable=True),
    ]
    text, detail, colour = instruments_line(HubState(instruments=rows))
    assert text == "3 instruments · 1 reachable"
    assert detail == "Not reachable: Scope 2 · Not paired: Scope 3"
    assert colour == WARN_COLOR
    text, detail, colour = instruments_line(HubState(instruments=rows[:1]))
    assert (text, detail, colour) == ("1 instrument · 1 reachable", "", OK_COLOR)


def test_the_agents_row_counts_those_running_here():
    assert agents_line({})[0] == "No agent"
    assert agents_line({"Scope 1": {"state": "ended"}})[0] == "No agent"
    found = {"Scope 1": {"state": "running"}, "Scope 2": {"state": "away"}}
    text, detail, colour = agents_line(found)
    assert text == "2 agents on this PC"
    assert detail == "Waiting for its instrument: Scope 2"
    assert colour == WARN_COLOR
    assert agents_line({"Scope 1": {"state": "running"}}) == (
        "1 agent on this PC",
        "Scope 1",
        OK_COLOR,
    )


def test_the_window_shows_what_the_hub_last_read(made):
    window, hub, agents, *_ = made
    assert "No instruments yet" in window.instruments.line.text()
    hub.rows = [InstrumentState("Scope 1", paired=True, reachable=True)]
    agents.found = {"Scope 1": {"state": "running"}}
    window.refresh()
    assert "1 instrument · 1 reachable" in window.instruments.line.text()
    assert "1 agent on this PC" in window.agents.line.text()


def test_pair_shows_a_code_that_signs_a_browser_in_then_hides_it(made):
    window, _, _, clock, *_ = made
    window.pair_button.click()
    code = window.code_box.line.text()
    assert code.isdigit() and window.pair_button.text() == "Hide"
    clock.now = 10_000.0
    window.refresh()
    assert window.pair_button.text() == "Pair"
    assert window.code_box.line.text() != code


def test_closing_asks_and_says_what_happens_to_the_agents(made, monkeypatch):
    window, _, agents, _, _, _, quits = made
    agents.found = {"Scope 1": {"state": "running"}}
    asked = []

    def question(parent, title, text):
        asked.append(text)
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "question", question)
    window.show()
    window.close()
    assert not quits and "The agent on this PC ends" in asked[0]
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a: QMessageBox.StandardButton.Yes
    )
    window.close()
    assert quits == [1]


def test_the_stop_route_closes_it_and_the_shortcut_brings_it_forward(made):
    window, _, _, _, stopping, forward, quits = made
    raised = []
    window.bring_to_front = lambda: raised.append(1)
    forward.set()
    window.refresh()
    assert raised == [1] and not forward.is_set()
    stopping.set()
    window.refresh()
    assert quits == [1]
