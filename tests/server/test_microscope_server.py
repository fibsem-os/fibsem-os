"""``fibsem-microscope-server`` serves the Demo microscope and pairing, with no
app and no agent, and another computer's remote microscope uses it."""

import os

import httpx
import pytest

import fibsem.config as fibsem_cfg
from fibsem.client.remote_microscope import RemoteMicroscope
from fibsem.server import microscope_server
from fibsem.server.operator_token import load_operator_token
from tests.fixtures.headless import confine_host, run_in_background, serving


@pytest.fixture
def discovery(tmp_path, monkeypatch):
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setattr(fibsem_cfg, "CONFIG_PATH", str(config_dir))
    monkeypatch.setattr(
        fibsem_cfg, "USER_PREFERENCES_PATH", str(config_dir / "user-preferences.yaml")
    )
    # The agent would be on here, so its being off is this command's doing.
    monkeypatch.setenv("FIBSEM_UNRELEASED", "1")
    return confine_host(monkeypatch, tmp_path)


def test_it_serves_the_microscope_with_no_app_or_agent(discovery, capsys):
    stop, worker, result = run_in_background(
        microscope_server.main, ["--manufacturer", "Demo", "--pair"]
    )
    try:
        found = serving(discovery)
        token = load_operator_token(discovery.parent / "operator-token.json")
        operator = {"Authorization": f"Bearer {token}"}
        with httpx.Client(base_url=found["url"], headers=operator) as client:
            capabilities = client.get("/capabilities").json()
            assert capabilities["routers"] == {
                "microscope": True,
                "services": True,
                "app": False,
            }
            assert capabilities["agent"] is False
            assert capabilities["level"] == "off"
            assert client.get("/app/agent").status_code == 404
            assert client.get("/microscope").is_success
        remote = RemoteMicroscope.connect(found["url"], token)
        try:
            assert remote.get_stage_position() is not None
            assert set(remote.services) >= {"imaging", "milling"}
        finally:
            remote.disconnect()
    finally:
        stop.set()
        worker.join(60)
    assert result == [0]
    out = capsys.readouterr().out
    assert "The microscope server is serving the" in out
    assert "Pairing code:" in out and "fibsem-cli pair" in out
    # No agent: nothing about the chat or setting one up.
    assert "/ui/agent" not in out and "fibsem-mcp" not in out
