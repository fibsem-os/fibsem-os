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
from typing import Callable, Dict, Optional, Tuple

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


# Two readings of one process's start differ by less than this; a pid that
# now belongs to another process started later differs by more.
STARTED_SLACK_S = 2.0


PS_KEPT_S = 10.0
_ps_seen: Dict[int, Tuple[float, float]] = {}


def process_started(pid) -> Optional[float]:
    """When process *pid* started, in epoch seconds; None when it isn't
    running or this platform can't say."""
    if not pid_alive(pid):
        return None
    try:
        if sys.platform == "win32":
            return _windows_started(pid)
        if sys.platform.startswith("linux"):
            return _linux_started(pid)
        # ps is a process of its own: read again at most every PS_KEPT_S.
        kept = _ps_seen.get(pid)
        if kept is not None and time.monotonic() - kept[0] < PS_KEPT_S:
            return kept[1]
        started = _ps_started(pid)
        _ps_seen[pid] = (time.monotonic(), started)
        return started
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def same_process(pid, started=None) -> bool:
    """Whether *pid* is still the process that started at *started* (as
    `process_started` gave it when its file was written): alive, and not a
    reused pid. With no *started*, or none to compare, whether it is alive."""
    if not pid_alive(pid):
        return False
    if not isinstance(started, (int, float)) or isinstance(started, bool):
        return True
    now = process_started(pid)
    return now is None or abs(now - started) < STARTED_SLACK_S


def _linux_started(pid: int) -> float:
    with open(f"/proc/{pid}/stat") as f:
        # the command may hold spaces and brackets: count after its last ")"
        fields = f.read().rsplit(")", 1)[1].split()
    ticks = int(fields[19])
    with open("/proc/stat") as f:
        boot = next(int(line.split()[1]) for line in f if line.startswith("btime"))
    return boot + ticks / os.sysconf("SC_CLK_TCK")


def _ps_started(pid: int) -> float:
    said = subprocess.run(
        ["ps", "-o", "lstart=", "-p", str(pid)],
        capture_output=True,
        text=True,
        timeout=5,
        env={**os.environ, "LC_ALL": "C"},
    ).stdout.strip()
    return time.mktime(time.strptime(said, "%a %b %d %H:%M:%S %Y"))


def _windows_started(pid: int) -> Optional[float]:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    handle = kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    try:
        times = [wintypes.FILETIME() for _ in range(4)]
        if not kernel32.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
            return None
        created = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
        # 100 ns steps since 1601
        return created / 1e7 - 11644473600.0
    finally:
        kernel32.CloseHandle(handle)
