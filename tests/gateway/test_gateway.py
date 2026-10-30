"""The gateway's app: starting and stopping its process without a window, a
window a desktop launch claims, the refusals, and the history, with stand-in
apps (tests/gateway/sleeper.py) in real child processes."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from fibsem.gateway.gateway import OFF, RUNNING, Gateway, GatewayRefused
from fibsem.tools.apps import App

ROOT = Path(__file__).resolve().parents[2]


def _app(serve):
    return App(
        name="Stand-in",
        app_id="test.StandIn",
        command="stand-in",
        target="tests.gateway.sleeper:serve",
        icon=ROOT / "fibsem" / "ui" / "icons" / "fibsemos.svg",
        serve=f"tests.gateway.sleeper:{serve}" if serve else None,
    )


@pytest.fixture(autouse=True)
def _importable(monkeypatch):
    # The child imports the stand-ins from this checkout.
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")])
    )


class Fakes:
    def __init__(self):
        self.found = None
        self.running = False


@pytest.fixture
def fakes():
    return Fakes()


def _gateway(tmp_path, fakes, serve="serve", stop_wait_s=20.0, connection=None):
    registry = {"stand-in": _app(serve), "window-only": _app(None)}
    return Gateway(
        connection=connection,
        home=tmp_path,
        app_registry=lambda: registry,
        last=lambda: "/experiments/grid3",
        discovery=lambda: fakes.found,
        running=lambda url, token: fakes.running,
        stop_wait_s=stop_wait_s,
    )


def _wait_for(check, timeout=20.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if check():
            return True
        time.sleep(0.05)
    return False


def _log(state):
    return Path(state.last_exit.log).read_text()


def _headless(gateway, who="operator", via="page", **kwargs):
    return gateway.start_app(who, via, app="stand-in", window=False, **kwargs)


def test_without_a_window_it_runs_the_apps_serve_and_stops_it_cleanly(tmp_path, fakes):
    gateway = _gateway(tmp_path, fakes)
    state = _headless(gateway, experiment="exp-1")
    assert (state.mode, state.window) == (RUNNING, False)
    assert (state.app, state.experiment) == ("stand-in", "exp-1")
    assert state.who == "operator" and state.pid
    state = gateway.stop("mac-mini", "hub")
    assert state.mode == OFF
    assert state.last_exit.code == 0 and not state.last_exit.killed
    assert "serving exp-1 --no-connect" in _log(state), "the operator connects"
    switches = gateway.history().switches
    assert [(s.to_mode, s.who, s.result, s.window) for s in switches] == [
        (OFF, "mac-mini", "stopped", False),
        (RUNNING, "operator", "started", False),
    ]


def test_the_gateways_connection_flags_connect_as_it_starts(tmp_path, fakes):
    gateway = _gateway(tmp_path, fakes, connection=["--manufacturer", "Demo"])
    assert gateway.state().connection == ["--manufacturer", "Demo"]
    _headless(gateway, last=True)
    assert "serving --last --manufacturer Demo\n" in _log(
        gateway.stop("operator", "page")
    )


def test_last_runs_the_most_recent_experiment(tmp_path, fakes):
    gateway = _gateway(tmp_path, fakes)
    assert _headless(gateway, last=True).experiment == "last"
    assert "serving --last" in _log(gateway.stop("operator", "page"))


def test_the_state_names_the_last_experiment(tmp_path, fakes):
    assert _gateway(tmp_path, fakes).state().last_experiment == "/experiments/grid3"


def test_an_app_that_wont_stop_is_killed(tmp_path, fakes):
    gateway = _gateway(tmp_path, fakes, serve="stubborn", stop_wait_s=1.0)
    _headless(gateway, last=True)
    assert gateway.stop("operator", "page").last_exit.killed


def test_a_running_workflow_refuses_a_stop_unless_forced(tmp_path, fakes):
    gateway = _gateway(tmp_path, fakes)
    state = _headless(gateway, last=True)
    fakes.found = {"pid": state.pid, "url": "http://127.0.0.1:8001"}
    fakes.running = True
    with pytest.raises(GatewayRefused) as refused:
        gateway.stop("mac-mini", "hub")
    assert refused.value.error_type == "run_in_progress"
    assert gateway.state().mode == RUNNING
    assert gateway.history().switches[0].result.startswith("refused")
    assert gateway.stop("operator", "page", force=True).mode == OFF


def test_a_host_it_didnt_start_blocks_a_start(tmp_path, fakes):
    gateway = _gateway(tmp_path, fakes)
    fakes.found = {"pid": 424242, "url": "http://127.0.0.1:8001"}
    assert gateway.state().holder.pid == 424242
    with pytest.raises(GatewayRefused) as refused:
        _headless(gateway, last=True)
    assert refused.value.error_type == "held"
    assert gateway.state().mode == OFF


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"app": "nope"}, "bad_app"),
        ({"app": "window-only", "window": False}, "no_headless"),
    ],
)
def test_bad_starts_are_refused(tmp_path, fakes, kwargs, error):
    with pytest.raises(GatewayRefused) as refused:
        _gateway(tmp_path, fakes).start_app("operator", "page", **kwargs)
    assert refused.value.error_type == error


def test_an_app_that_ends_on_its_own_goes_back_to_off(tmp_path, fakes):
    gateway = _gateway(tmp_path, fakes, serve="quits")
    _headless(gateway, last=True)
    assert _wait_for(lambda: (gateway.poll(), gateway.state().mode)[1] == OFF)
    assert gateway.state().last_exit.code == 3
    assert gateway.history().switches[0].result == "exited (3)"


def test_a_claimed_window_is_never_closed_remotely(tmp_path, fakes):
    gateway = _gateway(tmp_path, fakes)
    window = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        state = gateway.claim(window.pid, "stand-in", "operator", "desktop launch")
        assert (state.mode, state.window, state.pid) == (RUNNING, True, window.pid)
        fakes.found = {"pid": window.pid, "url": "http://127.0.0.1:8001"}
        assert gateway.state().holder is None  # its own App Server
        for act in (
            lambda: gateway.stop("operator", "page", force=True),
            lambda: _headless(gateway, "mac-mini", "hub", last=True),
        ):
            with pytest.raises(GatewayRefused) as refused:
                act()
            assert refused.value.error_type == "window_open"
        with pytest.raises(GatewayRefused):
            gateway.claim(os.getpid(), "stand-in", "operator", "desktop launch")
    finally:
        window.kill()
        window.wait()
    fakes.found = None
    assert _wait_for(lambda: (gateway.poll(), gateway.state().mode)[1] == OFF)
    assert gateway.history().switches[0].result == "closed"


@pytest.mark.skipif(
    sys.platform == "win32", reason="a shell script stands in for python"
)
def test_with_the_window_it_opens_the_apps_desktop_launch(tmp_path, fakes):
    seen = tmp_path / "seen"
    fake_python = tmp_path / "python"
    fake_python.write_text(f'#!/bin/sh\necho "$@ $FIBSEM_GATEWAY" > {seen}\nsleep 60\n')
    fake_python.chmod(0o755)
    gateway = _gateway(tmp_path, fakes, connection=["--manufacturer", "Demo"])
    gateway._python = str(fake_python)
    state = gateway.start_app("operator", "tray", app="stand-in", last=True)
    try:
        assert (state.mode, state.window) == (RUNNING, True)
        # The shell creates the file before it writes the line.
        assert _wait_for(lambda: seen.exists() and seen.read_text().strip())
        assert seen.read_text().split() == [
            "-m",
            "fibsem.tools.desktop_launch",
            "stand-in",
            "--last",
            "--manufacturer",
            "Demo",
            "--host",
            "1",
        ]
    finally:
        os.kill(state.pid, 9)
        _wait_for(lambda: (gateway.poll(), gateway.state().mode)[1] == OFF)
