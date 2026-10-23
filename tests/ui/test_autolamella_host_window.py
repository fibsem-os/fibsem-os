"""``fibsem-autolamella-ui`` is the AutoLamella host with the window: the
window opens on the host's session."""

import os

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("fastapi")

import fibsem.applications.autolamella.ui.launch as ui_launch  # noqa: E402
import fibsem.config as fibsem_cfg  # noqa: E402


@pytest.fixture
def places(tmp_path, monkeypatch):
    """Isolated preferences, the Demo microscope at no delay."""
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    monkeypatch.setattr(fibsem_cfg, "CONFIG_PATH", str(tmp_path))
    monkeypatch.setattr(
        fibsem_cfg, "USER_PREFERENCES_PATH", str(tmp_path / "user-preferences.yaml")
    )
    return tmp_path


def test_with_the_window_the_host_opens_it_on_its_session(places, monkeypatch):
    """``fibsem-autolamella-ui`` is the host with the window: it opens on the
    host's session, with the window's own flags, and nothing connects
    until a flag names a microscope."""
    opened = []

    def run_ui(argv, splash=None, session=None):
        opened.append((argv, splash, session))
        return 0

    monkeypatch.setattr(ui_launch, "run_ui", run_ui)

    assert ui_launch.host(["--quickstart"], splash="splash") == 0
    ((argv, splash, session),) = opened
    assert argv == ["--quickstart"] and splash == "splash"
    assert session.microscope is None

    assert ui_launch.host(["--manufacturer", "Demo"]) == 0
    session = opened[-1][2]
    try:
        assert session.microscope is not None and session.recorder is not None
    finally:
        session.disconnect()
