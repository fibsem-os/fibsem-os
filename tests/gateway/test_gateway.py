"""The gateway's modes: starting and stopping a headless app's process, a
window a desktop launch claims, the refusals, and the history, with stand-in
apps (tests/gateway/sleeper.py) in real child processes."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from fibsem.gateway.gateway import (
    HEADLESS,
    OFF,
    WINDOW,
    Configuration,
    Gateway,
    GatewayRefused,
)
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


def _gateway(tmp_path, fakes, serve="serve", stop_wait_s=20.0):
    registry = {"stand-in": _app(serve), "window-only": _app(None)}
    return Gateway(
        connection=["--manufacturer", "Demo"],
        home=tmp_path,
        app_registry=lambda: registry,
        saved_configurations=lambda: [
            Configuration("arctis", "/configs/arctis.yaml", default=True),
            Configuration("demo", "/configs/demo.yaml"),
        ],
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


def test_headless_starts_the_apps_serve_and_stops_it_cleanly(tmp_path, fakes):
    gateway = _gateway(tmp_path, fakes)
    state = gateway.switch(
        HEADLESS, "operator", "page", app="stand-in", experiment="exp-1"
    )
    assert (state.mode, state.app, state.experiment) == (HEADLESS, "stand-in", "exp-1")
    assert state.who == "operator" and state.pid
    state = gateway.stop("mac-mini", "hub")
    assert state.mode == OFF
    assert state.last_exit.code == 0 and not state.last_exit.killed
    assert "serving exp-1 --config /configs/arctis.yaml --manufacturer Demo" in _log(
        state
    )
    switches = gateway.history().switches
    assert [(s.to_mode, s.who, s.result) for s in switches] == [
        (OFF, "mac-mini", "stopped"),
        (HEADLESS, "operator", "started"),
    ]


def test_last_runs_the_most_recent_experiment(tmp_path, fakes):
    gateway = _gateway(tmp_path, fakes)
    assert (
        gateway.switch(
            HEADLESS, "operator", "page", app="stand-in", last=True
        ).experiment
        == "last"
    )
    assert "serving --last" in _log(gateway.stop("operator", "page"))


def test_headless_connects_with_the_configuration_picked(tmp_path, fakes):
    gateway = _gateway(tmp_path, fakes)
    state = gateway.switch(
        HEADLESS, "operator", "page", app="stand-in", last=True, configuration="demo"
    )
    assert state.configuration == "demo"
    assert "--config /configs/demo.yaml" in _log(gateway.stop("operator", "page"))
    with pytest.raises(GatewayRefused) as refused:
        gateway.switch(HEADLESS, "operator", "page", app="stand-in", configuration="x")
    assert refused.value.error_type == "bad_configuration"


def test_an_app_that_wont_stop_is_killed(tmp_path, fakes):
    gateway = _gateway(tmp_path, fakes, serve="stubborn", stop_wait_s=1.0)
    gateway.switch(HEADLESS, "operator", "page", app="stand-in", last=True)
    assert gateway.stop("operator", "page").last_exit.killed


def test_a_running_workflow_refuses_a_stop_unless_forced(tmp_path, fakes):
    gateway = _gateway(tmp_path, fakes)
    state = gateway.switch(HEADLESS, "operator", "page", app="stand-in", last=True)
    fakes.found = {"pid": state.pid, "url": "http://127.0.0.1:8001"}
    fakes.running = True
    with pytest.raises(GatewayRefused) as refused:
        gateway.stop("mac-mini", "hub")
    assert refused.value.error_type == "run_in_progress"
    assert gateway.state().mode == HEADLESS
    assert gateway.history().switches[0].result.startswith("refused")
    assert gateway.stop("operator", "page", force=True).mode == OFF


def test_an_app_server_it_didnt_start_blocks_a_switch(tmp_path, fakes):
    gateway = _gateway(tmp_path, fakes)
    fakes.found = {"pid": 424242, "url": "http://127.0.0.1:8001"}
    assert gateway.state().holder.pid == 424242
    with pytest.raises(GatewayRefused) as refused:
        gateway.switch(HEADLESS, "operator", "page", app="stand-in", last=True)
    assert refused.value.error_type == "held"
    assert gateway.state().mode == OFF


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"mode": "sideways", "app": "stand-in"}, "bad_mode"),
        ({"mode": HEADLESS, "app": "nope", "last": True}, "bad_app"),
        ({"mode": HEADLESS, "app": "window-only", "last": True}, "no_headless"),
    ],
)
def test_bad_switches_are_refused(tmp_path, fakes, kwargs, error):
    with pytest.raises(GatewayRefused) as refused:
        _gateway(tmp_path, fakes).switch(who="operator", via="page", **kwargs)
    assert refused.value.error_type == error


def test_an_app_that_ends_on_its_own_goes_back_to_off(tmp_path, fakes):
    gateway = _gateway(tmp_path, fakes, serve="quits")
    gateway.switch(HEADLESS, "operator", "page", app="stand-in", last=True)
    assert _wait_for(lambda: (gateway.poll(), gateway.state().mode)[1] == OFF)
    assert gateway.state().last_exit.code == 3
    assert gateway.history().switches[0].result == "exited (3)"


def test_a_claimed_window_is_never_closed_remotely(tmp_path, fakes):
    gateway = _gateway(tmp_path, fakes)
    window = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        state = gateway.claim(window.pid, "stand-in", "operator", "desktop launch")
        assert (state.mode, state.pid) == (WINDOW, window.pid)
        fakes.found = {"pid": window.pid, "url": "http://127.0.0.1:8001"}
        assert gateway.state().holder is None  # its own App Server
        for act in (
            lambda: gateway.stop("operator", "page", force=True),
            lambda: gateway.switch(
                HEADLESS, "mac-mini", "hub", app="stand-in", last=True
            ),
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
def test_window_mode_opens_the_apps_desktop_launch(tmp_path, fakes):
    seen = tmp_path / "seen"
    fake_python = tmp_path / "python"
    fake_python.write_text(f'#!/bin/sh\necho "$@ $FIBSEM_GATEWAY" > {seen}\nsleep 60\n')
    fake_python.chmod(0o755)
    gateway = _gateway(tmp_path, fakes)
    gateway._python = str(fake_python)
    state = gateway.switch(WINDOW, "operator", "tray", app="stand-in")
    try:
        assert state.mode == WINDOW
        assert _wait_for(seen.exists)
        assert seen.read_text().split() == [
            "-m",
            "fibsem.tools.desktop_launch",
            "stand-in",
            "window",
        ]
    finally:
        os.kill(state.pid, 9)
        _wait_for(lambda: (gateway.poll(), gateway.state().mode)[1] == OFF)
