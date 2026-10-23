"""``fibsem-autolamella-server`` opens an AutoLamella experiment on the Demo
microscope and serves it, with no window, until it is stopped. The agent
starts at Watch and the operator raises it from the chat page's routes."""

import os
import shutil

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

import fibsem.config as fibsem_cfg  # noqa: E402
from fibsem.applications.autolamella import config as cfg  # noqa: E402
from fibsem.applications.autolamella import launch  # noqa: E402
from fibsem.applications.autolamella.structures import Experiment  # noqa: E402
from fibsem.server.operator_token import load_operator_token  # noqa: E402
from tests.fixtures.headless import (  # noqa: E402
    allowed_agent,
    confine_host,
    run_in_background,
    serving,
)


@pytest.fixture
def places(tmp_path, monkeypatch):
    """Isolated preferences and protocol, the Demo microscope at no delay."""
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


def test_a_new_experiment_is_served_and_the_operator_hands_over(
    discovery, places, capsys
):
    folder = places / "grid-1"
    stop, worker, result = run_in_background(launch.main, [str(folder), "--create"])
    try:
        found = serving(discovery)
        operator_token = load_operator_token(discovery.parent / "operator-token.json")
        with httpx.Client(base_url=found["url"]) as client:
            agent_token = allowed_agent(found, operator_token)
            agent = {"Authorization": f"Bearer {agent_token}"}
            operator = {"Authorization": f"Bearer {operator_token}"}
            capabilities = client.get("/capabilities", headers=agent).json()
            assert capabilities["app_name"] == "AutoLamella"
            assert client.get("/ui/agent", headers=operator).status_code == 200
            # Watch first: the agent reads but can't add a position
            site = {"name": "site-a"}
            refused = client.post("/app/experiment/items", json=site, headers=agent)
            assert refused.status_code == 403
            drive = {"level": "drive"}
            raised = client.put("/app/agent/level", json=drive, headers=operator)
            assert raised.is_success, raised.text
            added = client.post("/app/experiment/items", json=site, headers=agent)
            assert added.is_success, added.text
    finally:
        stop.set()
        worker.join(60)

    assert result == [0]
    experiment = Experiment.load(folder / "experiment.yaml")
    assert [p.name for p in experiment.positions] == ["site-a"]
    assert (folder / "protocol.yaml").is_file()
    assert fibsem_cfg.get_last_experiment_file() == str(folder / "experiment.yaml")
    out = capsys.readouterr().out
    # The plain chat address: the token stays out of the terminal.
    assert "/ui/agent (Pair this browser)" in out and "#token=" not in out


def test_with_no_folder_none_is_open_and_last_opens_the_most_recent(
    discovery, places, capsys
):
    assert launch.main([str(places / "grid-1"), "--create"], _stopped()) == 0
    assert launch.main([str(places / "grid-2"), "--create"], _stopped()) == 0
    capsys.readouterr()
    assert launch.main([], _stopped()) == 0
    assert "is serving no experiment at" in capsys.readouterr().out
    assert launch.main(["--last"], _stopped()) == 0
    assert "is serving grid-2 at" in capsys.readouterr().out


def test_the_agent_creates_the_experiment_in_the_named_folder(discovery, places):
    folder = places / "experiments"
    folder.mkdir()
    stop, worker, result = run_in_background(
        launch.main, ["--experiments", str(folder)]
    )
    try:
        found = serving(discovery)
        operator_token = load_operator_token(discovery.parent / "operator-token.json")
        with httpx.Client(base_url=found["url"]) as client:
            agent_token = allowed_agent(found, operator_token)
            agent = {"Authorization": f"Bearer {agent_token}"}
            operator = {"Authorization": f"Bearer {operator_token}"}
            # no experiment open yet
            assert client.get("/app/experiment", headers=agent).status_code == 409
            drive = {"level": "drive"}
            raised = client.put("/app/agent/level", json=drive, headers=operator)
            assert raised.is_success, raised.text
            made = client.post(
                "/app/experiment/create", json={"name": "day-1/grid-a"}, headers=agent
            )
            assert made.is_success, made.text
            listed = client.get("/app/experiment/all", headers=agent).json()
            assert [e["name"] for e in listed["items"]] == ["grid-a"]
    finally:
        stop.set()
        worker.join(60)
    assert result == [0]
    assert (folder / "day-1" / "grid-a" / "experiment.yaml").is_file()


def test_what_is_refused_before_connecting(places, capsys):
    assert launch.main(["--last"]) == 1
    assert "no recent experiment" in capsys.readouterr().out
    assert launch.main([str(places / "x"), "--last"]) == 1
    assert "--last" in capsys.readouterr().out
    assert launch.main(["--experiments", str(places / "missing")]) == 1
    assert "is not a folder" in capsys.readouterr().out
    assert launch.main([str(places / "none")]) == 1
    assert "--create" in capsys.readouterr().out
    assert launch.main([str(places), "--create"]) == 1
    assert "exists" in capsys.readouterr().out
    assert launch.main([str(places / "x"), "--protocol", "p.yaml"]) == 1
    assert "--protocol is for --create" in capsys.readouterr().out
    bad = places / "bad.yaml"
    bad.write_text("not: [a protocol")
    assert launch.main([str(places / "y"), "--create", "--protocol", str(bad)]) == 1
    assert "could not load the protocol" in capsys.readouterr().out


def _stopped():
    import threading

    stop = threading.Event()
    stop.set()
    return stop


def test_a_window_flag_without_the_window_is_refused(places):
    with pytest.raises(SystemExit):
        launch.main(["--quickstart"])
