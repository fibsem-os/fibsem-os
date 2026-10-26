"""``fibsem-volume`` opens a volume experiment on a microscope and serves it,
with no window, until it is stopped."""

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fibsem.applications.volume import launch  # noqa: E402
from fibsem.applications.volume.experiment import VolumeExperiment  # noqa: E402
from fibsem.server.operator_token import load_operator_token  # noqa: E402
from tests.fixtures.headless import (  # noqa: E402
    allowed_agent,
    confine_host,
    run_in_background,
    serving,
)


@pytest.fixture
def discovery(monkeypatch, tmp_path):
    return confine_host(monkeypatch, tmp_path)


def test_a_new_experiment_is_served_until_stopped(discovery, tmp_path, capsys):
    folder = tmp_path / "vol"
    stop, worker, result = run_in_background(launch.main, [str(folder), "--create"])
    try:
        found = serving(discovery)
        operator_token = load_operator_token(discovery.parent / "operator-token.json")
        capabilities = httpx.get(
            f"{found['url']}/capabilities",
            headers={"Authorization": f"Bearer {allowed_agent(found, operator_token)}"},
        ).json()
        assert capabilities["app_name"] == "Volume"
        assert capabilities["parts"] == ["items", "poses"]
    finally:
        stop.set()
        worker.join(60)

    assert result == [0]
    assert VolumeExperiment.load(folder).name == "vol"
    out = capsys.readouterr().out
    # The plain chat address: the token stays out of the terminal.
    assert "/ui/agent (Pair this browser)" in out and "#token=" not in out


def test_an_experiment_must_exist_unless_created(tmp_path, capsys):
    assert launch.main([str(tmp_path / "none")]) == 1
    assert "--create" in capsys.readouterr().out
    assert launch.main([str(tmp_path), "--create"]) == 1


def test_with_the_window_no_experiment_is_needed(monkeypatch):
    pytest.importorskip("PyQt5")
    import fibsem.applications.volume.ui.window as window

    opened = []
    monkeypatch.setattr(
        window, "main", lambda argv, splash=None, session=None: opened.append(session)
    )

    window.host([])

    (session,) = opened
    assert session.microscope is None and session.experiment is None
