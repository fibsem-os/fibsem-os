"""Other processes on this PC: whether one still runs, ending one, and the
stop file a process with no console is asked to stop by. A file works on every
platform, where a signal to a process with no console doesn't on Windows."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

STOP_POLL_S = 0.5


def pid_alive(pid) -> bool:
    """Whether process *pid* is running; False for anything not a pid."""
    if not isinstance(pid, int) or not 0 < pid < 2**32:
        return False
    if sys.platform == "win32":
        return _windows_pid_alive(pid)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _windows_pid_alive(pid: int) -> bool:
    """Whether a Windows process is still running, from its exit code.

    Never ``os.kill(pid, 0)`` here: on Windows signal 0 is ``CTRL_C_EVENT``,
    so it sends a Ctrl-C to the console process group ``pid`` instead of
    asking whether it exists.
    """
    import ctypes
    from ctypes import wintypes

    process_query_limited_information = 0x1000
    still_active = 259
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    )
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def kill_pid(pid: int) -> None:
    """End process *pid* at once (on Windows, TerminateProcess); nothing if
    it has gone."""
    try:
        os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
    except OSError:
        pass


def end(proc: subprocess.Popen, wait_s: float) -> bool:
    """Ask *proc* to end, and kill it after *wait_s*. Whether it was killed."""
    if proc.poll() is not None:
        return False
    proc.terminate()
    try:
        proc.wait(wait_s)
        return False
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return True


def watch_stop_file(
    stop_file: Path,
    stop: Callable[[], None],
    poll_s: float = STOP_POLL_S,
    name: str = "stop-file",
) -> threading.Thread:
    """Call *stop* once *stop_file* exists, from a daemon thread."""

    def run() -> None:
        while not stop_file.exists():
            time.sleep(poll_s)
        stop()

    thread = threading.Thread(target=run, name=name, daemon=True)
    thread.start()
    return thread
