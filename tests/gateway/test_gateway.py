"""The gateway's app: starting and stopping its process without a window, a
window a desktop launch claims, the refusals, and the history, with stand-in
apps (tests/gateway/sleeper.py) in real child processes."""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from fibsem.gateway.gateway import (
    OFF,
    RUNNING,
    Gateway,
    GatewayRefused,
    RecentExperiment,
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


CONFIGURATIONS = [
    {"name": "arctis", "path": "/cfg/arctis.yaml", "default": True},
    {"name": "hydra", "path": "/cfg/hydra.yaml", "default": False},
]


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
        recent=lambda: [RecentExperiment("grid3", "/experiments/grid3", "AutoLamella")],
        new_folder=lambda: "/experiments",
        discovery=lambda: fakes.found,
        running=lambda url, token: fakes.running,
        configurations=lambda: CONFIGURATIONS,
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


def _experiment(tmp_path, name="exp-1"):
    folder = tmp_path / name
    folder.mkdir()
    (folder / "experiment.yaml").write_text(f"name: {name}\n")
    return str(folder)


def test_without_a_window_it_runs_the_apps_serve_and_stops_it_cleanly(tmp_path, fakes):
    gateway = _gateway(tmp_path, fakes)
    exp = _experiment(tmp_path)
    state = _headless(gateway, experiment=exp)
    assert (state.mode, state.window) == (RUNNING, False)
    assert (state.app, state.experiment) == ("stand-in", exp)
    assert state.who == "operator" and state.pid
    state = gateway.stop("mac-mini", "hub")
    assert state.mode == OFF
    assert state.last_exit.code == 0 and not state.last_exit.killed
    assert f"serving {exp} --no-connect" in _log(state), "the operator connects"
    switches = gateway.history().switches
    assert [(s.to_mode, s.who, s.result, s.window) for s in switches] == [
        (OFF, "mac-mini", "stopped", False),
        (RUNNING, "operator", "started", False),
    ]


def test_a_new_experiment_is_made_with_create(tmp_path, fakes):
    gateway = _gateway(tmp_path, fakes)
    new = str(tmp_path / "grid-4")
    _headless(gateway, experiment=new, create=True)
    assert f"serving {new} --create --no-connect" in _log(
        gateway.stop("operator", "page")
    )


@pytest.mark.parametrize(
    "folder, create, error",
    [
        ("exp-1", True, "exists"),
        ("missing/grid-4", True, "no_parent"),
        ("loose", False, "no_experiment"),
        (None, True, "no_folder"),
    ],
)
def test_an_experiment_the_app_could_not_open_is_refused(
    tmp_path, fakes, folder, create, error
):
    _experiment(tmp_path)
    (tmp_path / "loose").mkdir()
    gateway = _gateway(tmp_path, fakes)
    with pytest.raises(GatewayRefused) as refused:
        _headless(
            gateway,
            experiment=str(tmp_path / folder) if folder else None,
            create=create,
        )
    assert refused.value.error_type == error
    assert gateway.state().mode == OFF


def test_the_gateways_connection_flags_connect_as_it_starts(tmp_path, fakes):
    gateway = _gateway(tmp_path, fakes, connection=["--manufacturer", "Demo"])
    assert gateway.state().connection == ["--manufacturer", "Demo"]
    _headless(gateway, last=True)
    assert "serving --last --manufacturer Demo\n" in _log(
        gateway.stop("operator", "page")
    )


def test_the_state_lists_recent_experiments_and_where_new_ones_go(tmp_path, fakes):
    state = _gateway(tmp_path, fakes).state()
    assert [(r.name, r.folder) for r in state.recent_experiments] == [
        ("grid3", "/experiments/grid3")
    ]
    assert state.experiments_folder == "/experiments"


def test_the_state_gives_the_running_apps_address(tmp_path, fakes):
    gateway = _gateway(tmp_path, fakes)
    assert gateway.state().app_url is None
    fakes.found = {"url": "http://127.0.0.1:8123", "pid": 4321}
    assert gateway.state().app_url == "http://127.0.0.1:8123"


def test_the_state_lists_the_saved_configurations(tmp_path, fakes):
    listed = _gateway(tmp_path, fakes).state().configurations
    assert [(c.name, c.default) for c in listed] == [("arctis", True), ("hydra", False)]


@pytest.mark.parametrize(
    "configuration, flags",
    [(None, "--config /cfg/arctis.yaml"), ("hydra", "--config /cfg/hydra.yaml")],
)
def test_a_start_can_connect_a_saved_configuration(
    tmp_path, fakes, configuration, flags
):
    gateway = _gateway(tmp_path, fakes, connection=["--manufacturer", "Demo"])
    _headless(gateway, last=True, connect=True, configuration=configuration)
    assert f"serving --last {flags}\n" in _log(gateway.stop("operator", "page"))


def test_a_start_can_leave_the_microscope_unconnected(tmp_path, fakes):
    gateway = _gateway(tmp_path, fakes, connection=["--manufacturer", "Demo"])
    _headless(gateway, last=True, connect=False)
    assert "serving --last --no-connect\n" in _log(gateway.stop("operator", "page"))


def test_an_unknown_configuration_refuses_the_start(tmp_path, fakes):
    gateway = _gateway(tmp_path, fakes)
    with pytest.raises(GatewayRefused) as refused:
        _headless(gateway, connect=True, configuration="krios")
    assert refused.value.error_type == "unknown_configuration"
    assert gateway.state().mode == OFF


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


def test_a_claimed_window_is_closed_only_by_the_operator(tmp_path, fakes):
    gateway = _gateway(tmp_path, fakes)
    window = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        state = gateway.claim(window.pid, "stand-in", "operator", "desktop launch")
        assert (state.mode, state.window, state.pid) == (RUNNING, True, window.pid)
        fakes.found = {"pid": window.pid, "url": "http://127.0.0.1:8001"}
        assert gateway.state().holder is None  # its own App Server
        for act in (
            lambda: gateway.stop("mac-mini", "hub", force=True),
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


def test_the_state_names_newer_code(tmp_path, fakes):
    from fibsem.update_check import Newer

    gateway = Gateway(home=tmp_path, newer=lambda: Newer(behind=3))
    assert gateway.state().newer is None, "nothing known before a check"
    gateway.check_newer()
    assert gateway.state().newer == Newer(behind=3)


def test_a_failed_update_check_shows_nothing(tmp_path):
    def offline():
        raise OSError("offline")

    gateway = Gateway(home=tmp_path, newer=offline)
    gateway.check_newer()
    assert gateway.state().newer is None


WINDOW = """
import os, sys, time
ask = sys.argv[1] + "/gateway-close-%d" % os.getpid()
while not os.path.exists(ask):
    time.sleep(0.05)
"""


def test_the_operator_closes_a_window_as_its_close_button_does(tmp_path, fakes):
    gateway = _gateway(tmp_path, fakes)
    window = subprocess.Popen([sys.executable, "-c", WINDOW, str(tmp_path)])
    # Reaped as it ends, as a window the gateway didn't start is.
    reaper = threading.Thread(target=window.wait)
    reaper.start()
    try:
        gateway.claim(window.pid, "stand-in", "operator", "desktop launch")
        state = gateway.stop("operator", "page", window=True)
    finally:
        window.kill()
        reaper.join()
    assert state.mode == OFF
    assert window.returncode == 0, "it closed itself"
    assert not list(tmp_path.glob("gateway-close-*"))
    assert gateway.history().switches[0].result == "stopped"


def test_a_window_that_doesnt_close_is_left_to_answer_at_the_pc(tmp_path, fakes):
    gateway = _gateway(tmp_path, fakes, stop_wait_s=0.5)
    window = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        gateway.claim(window.pid, "stand-in", "operator", "desktop launch")
        with pytest.raises(GatewayRefused) as refused:
            gateway.stop("operator", "page", window=True)
        assert refused.value.error_type == "window_didnt_close"
        assert window.poll() is None, "never killed"
        assert gateway.state().mode == RUNNING
        assert not list(tmp_path.glob("gateway-close-*"))
    finally:
        window.kill()
        window.wait()
