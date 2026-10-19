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
