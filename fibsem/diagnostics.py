"""What a program records when it fails where nothing was catching.

:func:`install` is called once by every entry point. From then on an error
nobody caught, on any thread, goes to the log and to a scrubbed record under
the log folder, and a hard crash of the interpreter leaves its stack in a file
beside them. It also starts the program's own log file (`fibsem.util.app_log`). Nothing here reaches the network; the records are for Report an
Issue to offer on the next launch.
"""

from __future__ import annotations

import faulthandler
import json
import logging
import os
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import fibsem.config as cfg
from fibsem.tools.bug_report import scrub_text
from fibsem.util import app_log
from fibsem.util.timestamps import now_iso
from fibsem.versioning import get_version_string

logger = logging.getLogger(__name__)

# A slot that fails on a timer fails many times a second: past this many records
# a process only logs.
MAX_RECORDS_PER_PROCESS = 20
# Older records are deleted when a program starts.
MAX_RECORDS_KEPT = 50
# The same error is told to ``on_error`` again only after this long.
TELL_AGAIN_AFTER_S = 60.0

_FATAL_MARKERS = ("Fatal Python error", "Windows fatal exception")

_lock = threading.Lock()
_app: Optional[str] = None
_on_error: Optional[Callable[["CrashRecord"], None]] = None
_written = 0
_told: Dict[Tuple[str, str], float] = {}
_fault_file = None


@dataclass
class CrashRecord:
    """One error nobody caught, as it is saved."""

    app: str
    version: str
    at: str
    pid: int
    """The process that failed; 0 when only its faulthandler file says so."""
    thread: str
    error: str
    """The exception's type, or "fatal" for a hard crash of the interpreter."""
    traceback: str
    """Scrubbed as a bug report is."""
    seen: bool = False
    """Whether the operator has been offered it."""
    path: Optional[str] = None

    @property
    def summary(self) -> str:
        """The last line of the traceback, which names the error."""
        lines = [line for line in self.traceback.strip().splitlines() if line.strip()]
        return lines[-1].strip() if lines else self.error


def crash_folder() -> Path:
    return Path(cfg.LOG_PATH) / "crashes"


def install(app: str, on_error: Optional[Callable[[CrashRecord], None]] = None) -> None:
    """Record uncaught errors for the program called *app*. Safe to call again:
    the hooks go in once, and a later ``on_error`` replaces the earlier one.

    ``on_error`` hears each new kind of error, on the thread it happened on.
    """
    global _app, _on_error
    with _lock:
        first = _app is None
        _app = app
        if on_error is not None:
            _on_error = on_error
    if not first:
        return
    app_log.start(app)
    try:
        _keep_fatal_error_of_last_run(app)
        _prune()
        _enable_faulthandler(app)
    except Exception:
        logger.warning("Could not set up crash records.", exc_info=True)
    _chain_excepthook()
    _chain_threading_excepthook()


def _chain_excepthook() -> None:
    previous = sys.excepthook

    def hook(exc_type, exc, tb):
        if not issubclass(exc_type, KeyboardInterrupt):
            _record(exc_type, exc, tb, threading.current_thread().name)
        # The log has it now; the default hook would only print it again.
        if previous is not sys.__excepthook__ or issubclass(
            exc_type, KeyboardInterrupt
        ):
            previous(exc_type, exc, tb)

    sys.excepthook = hook


def _chain_threading_excepthook() -> None:
    previous = threading.excepthook

    def hook(args):
        if args.exc_type is SystemExit:
            return
        name = args.thread.name if args.thread is not None else "unknown thread"
        _record(args.exc_type, args.exc_value, args.exc_traceback, name)
        if previous is not threading.__excepthook__:
            previous(args)

    threading.excepthook = hook


def _record(exc_type, exc, tb, thread: str) -> None:
    """Log the error and save it. Never raises: this runs inside the hooks."""
    try:
        logger.error(
            "Uncaught %s on %s", exc_type.__name__, thread, exc_info=(exc_type, exc, tb)
        )
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        record = _save(exc_type.__name__, text, thread)
        if record is not None:
            _tell(record, _where(tb))
    except Exception:
        logger.warning("Could not record an uncaught error.", exc_info=True)


def _where(tb) -> str:
    """The innermost frame, to tell one error from the next of its type."""
    frames = traceback.extract_tb(tb)
    return f"{frames[-1].filename}:{frames[-1].lineno}" if frames else ""


def _save(
    error: str, text: str, thread: str, pid: Optional[int] = None
) -> Optional[CrashRecord]:
    global _written
    with _lock:
        if _written >= MAX_RECORDS_PER_PROCESS:
            return None
        _written += 1
        number = _written
    record = CrashRecord(
        app=_app or "fibsem",
        version=get_version_string(),
        at=now_iso(),
        pid=os.getpid() if pid is None else pid,
        thread=thread,
        error=error,
        traceback=scrub_text(text),
    )
    folder = crash_folder()
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    path = folder / f"{stamp}-{record.pid}-{number:02d}.json"
    record.path = str(path)
    _write(record)
    return record


def _write(record: CrashRecord) -> None:
    data = asdict(record)
    data.pop("path")
    Path(record.path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _tell(record: CrashRecord, where: str) -> None:
    key = (record.error, where)
    now = time.monotonic()
    with _lock:
        last = _told.get(key)
        if last is not None and now - last < TELL_AGAIN_AFTER_S:
            return
        _told[key] = now
        on_error = _on_error
    if on_error is not None:
        on_error(record)


# -- hard crashes -------------------------------------------------------------


def _fault_path(app: str) -> Path:
    name = "".join(c if c.isalnum() else "-" for c in app.lower())
    return crash_folder() / f"faulthandler-{name}.log"


def _enable_faulthandler(app: str) -> None:
    """Dump every thread's stack to a file on a fatal signal. A file, because a
    program started from a shortcut has no console to print to."""
    global _fault_file
    path = _fault_path(app)
    path.parent.mkdir(parents=True, exist_ok=True)
    _fault_file = open(path, "w", encoding="utf-8")  # noqa: SIM115 - held open
    faulthandler.enable(file=_fault_file, all_threads=True)


def _keep_fatal_error_of_last_run(app: str) -> None:
    """Turn what the last run's faulthandler wrote into a record, before the
    file is emptied for this run."""
    path = _fault_path(app)
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    if not any(marker in text for marker in _FATAL_MARKERS):
        return
    # pid 0: the run that crashed is over, whichever process it was.
    _save("fatal", text[-20000:], "unknown thread", pid=0)


# -- reading them back --------------------------------------------------------


def records() -> List[CrashRecord]:
    """Every saved record, oldest first. A file that does not read is skipped."""
    folder = crash_folder()
    if not folder.is_dir():
        return []
    found = []
    for path in sorted(folder.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            found.append(CrashRecord(**data, path=str(path)))
        except Exception:
            logger.debug("Skipping unreadable crash record %s", path)
    return found


def unseen_from_earlier_runs() -> List[CrashRecord]:
    """Records the operator has not been offered, left by a program that is no
    longer this one."""
    return [r for r in records() if not r.seen and r.pid != os.getpid()]


def mark_seen(seen: List[CrashRecord]) -> None:
    for record in seen:
        record.seen = True
        try:
            _write(record)
        except OSError:
            logger.debug("Could not mark %s as seen", record.path)


def _prune() -> None:
    folder = crash_folder()
    if not folder.is_dir():
        return
    for path in sorted(folder.glob("*.json"))[:-MAX_RECORDS_KEPT]:
        try:
            path.unlink()
        except OSError:
            pass
