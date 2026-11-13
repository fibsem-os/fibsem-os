"""Other processes: whether one runs, ending one, and the stop file."""

import os
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from fibsem.util import process


def test_this_process_is_alive():
    assert process.pid_alive(os.getpid())


def test_a_process_that_has_ended_is_not():
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    assert not process.pid_alive(child.pid)


@pytest.mark.parametrize("pid", [None, "123", 0, -1, 2**32])
def test_a_pid_that_cannot_be_a_process_is_not_alive(pid):
    assert not process.pid_alive(pid)


def test_windows_asks_for_the_exit_code_and_never_signals(monkeypatch):
    # On Windows os.kill(pid, 0) sends a Ctrl-C to the process group.
    def kill(pid, sig):
        raise AssertionError("os.kill on Windows")

    monkeypatch.setattr(process, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(process.os, "kill", kill)
    asked = []
    monkeypatch.setattr(
        process, "_windows_pid_alive", lambda pid: asked.append(pid) or True
    )
    assert process.pid_alive(1234)
    assert asked == [1234]


def test_a_process_that_ignores_terminate_is_killed():
    code = (
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print(1, flush=True); time.sleep(60)"
    )
    child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE)
    with child.stdout:
        child.stdout.readline()
    killed = process.end(child, 0.5)
    assert child.poll() is not None
    assert killed or sys.platform == "win32"


def test_a_killed_pid_is_gone():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    process.kill_pid(child.pid)
    child.wait(10)
    process.kill_pid(child.pid)  # gone: nothing happens


def test_the_stop_file_calls_stop_once_it_appears(tmp_path):
    stopped = threading.Event()
    stop_file = tmp_path / "stop"
    process.watch_stop_file(stop_file, stopped.set, poll_s=0.01)
    assert not stopped.wait(0.1)
    stop_file.touch()
    assert stopped.wait(5)
