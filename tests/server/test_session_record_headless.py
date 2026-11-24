"""On the Demo microscope, served with no window: the conversation is kept
with the agent's session and comes back after the app restarts, and the
agent's run-state notes stay with the experiment and read back."""

import os
import shutil

import httpx
import pytest

import fibsem.config as fibsem_cfg
from fibsem.applications.autolamella import config as cfg
from fibsem.applications.autolamella import launch
from fibsem.server.agent import files as agent_files
from fibsem.server.operator_token import load_operator_token
from fibsem.workflows.events.recording import EVENTS_FILENAME, read_events
from tests.fixtures.headless import (
    AGENT_TOKEN,
    confine_host,
    run_in_background,
    serving,
)


@pytest.fixture
def places(tmp_path, monkeypatch):
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setattr(fibsem_cfg, "CONFIG_PATH", str(config_dir))
    monkeypatch.setattr(
        fibsem_cfg, "USER_PREFERENCES_PATH", str(config_dir / "user-preferences.yaml")
    )
    protocol = tmp_path / "protocol.yaml"
    shutil.copy(cfg.TASK_PROTOCOL_PATH, protocol)
    monkeypatch.setattr(cfg, "TASK_PROTOCOL_PATH", str(protocol))
    return tmp_path


@pytest.fixture
def discovery(monkeypatch, places):
    return confine_host(monkeypatch, places)


def _serve(discovery, argv, talk):
    stop, worker, result = run_in_background(launch.main, argv)
    try:
        found = serving(discovery)
        token = load_operator_token(discovery.parent / "operator-token.json")
        agent = {"Authorization": f"Bearer {AGENT_TOKEN}"}
        operator = {"Authorization": f"Bearer {token}"}
        with httpx.Client(base_url=found["url"], timeout=30) as client:
            return talk(client, agent, operator)
    finally:
        stop.set()
        worker.join(60)
        assert result == [0]


def _texts(client, operator):
    answer = client.get("/app/agent/recent?rows=50", headers=operator)
    assert answer.is_success, answer.text
    return [row["text"] for row in answer.json()["rows"]]


def test_the_chat_and_notes_survive_a_restart(discovery, places):
    folder = places / "grid-1"

    def first(client, agent, operator):
        reported = client.post(
            "/app/agent/session", json={"session_id": "s1"}, headers=agent
        )
        assert reported.is_success, reported.text
        sent = client.post(
            "/app/agent/messages", json={"text": "Start with grid 1"}, headers=operator
        )
        assert sent.is_success, sent.text
        answered = client.post(
            "/app/experiment/messages",
            json={"text": "Starting grid 1.", "reply_to": sent.json()["id"]},
            headers=agent,
        )
        assert answered.is_success, answered.text
        noted = client.post(
            "/app/experiment/notes",
            json={"text": "[run_state] grid 1 of 2"},
            headers=agent,
        )
        assert noted.is_success, noted.text
        texts = _texts(client, operator)
        assert "Start with grid 1" in texts and "Starting grid 1." in texts

    _serve(discovery, [str(folder), "--create"], first)

    session = agent_files.session_folder("s1") / EVENTS_FILENAME
    in_session = [r["payload"].get("text") for r in read_events(session)]
    in_experiment = [
        r["payload"].get("text") for r in read_events(folder / EVENTS_FILENAME)
    ]
    assert "Start with grid 1" in in_session and "Starting grid 1." in in_session
    assert "[run_state] grid 1 of 2" in in_experiment
    assert "Start with grid 1" not in in_experiment

    def again(client, agent, operator):
        # Before the agent reports again: the session that worked on it.
        texts = _texts(client, operator)
        assert "Start with grid 1" in texts and "Starting grid 1." in texts
        notes = client.get(
            "/app/experiment/notes", params={"tags": ["run_state"]}, headers=agent
        )
        assert notes.is_success, notes.text
        assert [n["text"] for n in notes.json()["notes"]] == ["[run_state] grid 1 of 2"]

    _serve(discovery, [str(folder)], again)


def test_with_no_experiment_open_the_session_keeps_it(discovery, places):
    def talk(client, agent, operator):
        client.post("/app/agent/session", json={"session_id": "s2"}, headers=agent)
        sent = client.post(
            "/app/agent/messages", json={"text": "Nothing open yet"}, headers=operator
        )
        assert sent.is_success, sent.text
        noted = client.post(
            "/app/experiment/notes", json={"text": "[intake] two grids"}, headers=agent
        )
        assert noted.is_success, noted.text
        assert "Nothing open yet" in _texts(client, operator)
        notes = client.get(
            "/app/experiment/notes", params={"tags": ["intake"]}, headers=agent
        )
        assert [n["text"] for n in notes.json()["notes"]] == ["[intake] two grids"]

    _serve(discovery, [], talk)

    session = agent_files.session_folder("s2") / EVENTS_FILENAME
    texts = [r["payload"].get("text") for r in read_events(session)]
    assert "Nothing open yet" in texts and "[intake] two grids" in texts
