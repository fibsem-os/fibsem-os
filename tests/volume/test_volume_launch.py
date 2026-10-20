"""``fibsem-volume`` opens a volume experiment on a microscope and serves it,
with no window, until it is stopped."""

import json
import socket
import threading
import time

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fibsem.applications.volume import launch  # noqa: E402
from fibsem.applications.volume.experiment import VolumeExperiment  # noqa: E402
from fibsem.server import hosting  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def discovery(monkeypatch, tmp_path):
    """The host confined to a test port and a discovery file under tmp."""
    real_host, port = hosting.AppServerHost, _free_port()
    path = tmp_path / "app-server.json"

    def confined(session, app_context, **kwargs):
        return real_host(
            session,
            app_context,
            port=port,
            discovery_path=path,
            serve_microscope=kwargs.get("serve_microscope", False),
        )

    monkeypatch.setattr(hosting, "AppServerHost", confined)
    return path


def _serving(discovery, timeout: float = 60) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if discovery.is_file():
            return json.loads(discovery.read_text())
        time.sleep(0.1)
    raise AssertionError("fibsem-volume never served")


def test_a_new_experiment_is_served_until_stopped(discovery, tmp_path, capsys):
    folder, stop, result = tmp_path / "vol", threading.Event(), []
    worker = threading.Thread(
        target=lambda: result.append(launch.main([str(folder), "--create"], stop))
    )
    worker.start()
    try:
        found = _serving(discovery)
        capabilities = httpx.get(
            f"{found['url']}/capabilities",
            headers={"Authorization": f"Bearer {found['token']}"},
        ).json()
        assert capabilities["app_name"] == "Volume"
        assert capabilities["parts"] == ["items"]
    finally:
        stop.set()
        worker.join(60)

    assert result == [0]
    assert VolumeExperiment.load(folder).name == "vol"
    assert "/chat#token=" in capsys.readouterr().out


def test_an_experiment_must_exist_unless_created(tmp_path, capsys):
    assert launch.main([str(tmp_path / "none")]) == 1
    assert "--create" in capsys.readouterr().out
    assert launch.main([str(tmp_path), "--create"]) == 1
