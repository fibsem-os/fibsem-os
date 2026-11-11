"""The discovery file's liveness check, on POSIX and on Windows."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from fibsem.server import discovery


def test_this_process_is_alive():
    assert discovery._pid_alive(os.getpid())


def test_a_process_that_has_ended_is_not():
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    assert not discovery._pid_alive(child.pid)


@pytest.mark.parametrize("pid", [None, "123", 0, -1, 2**32])
def test_a_pid_that_cannot_be_a_process_is_not_alive(pid):
    assert not discovery._pid_alive(pid)


def test_windows_asks_for_the_exit_code_and_never_signals(monkeypatch):
    # On Windows os.kill(pid, 0) sends a Ctrl-C to the process group.
    def kill(pid, sig):
        raise AssertionError("os.kill on Windows")

    monkeypatch.setattr(discovery, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(discovery.os, "kill", kill)
    asked = []
    monkeypatch.setattr(
        discovery, "_windows_pid_alive", lambda pid: asked.append(pid) or True
    )
    assert discovery._pid_alive(1234)
    assert asked == [1234]


def test_a_client_beside_a_second_host_reads_that_hosts_file(tmp_path):
    # An eval run's agent is told its own server's file, not this PC's.
    found = tmp_path / "app-server.json"
    code = (
        "from fibsem.server import discovery as d; "
        "print(d.DISCOVERY_FILE == d.LEGACY_DISCOVERY_FILE, d.DISCOVERY_FILE)"
    )
    env = {**os.environ, discovery.DISCOVERY_FILE_ENV: str(found)}
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    ).stdout.split()
    assert out == ["True", str(found)]
