"""The program's own log file, written from start-up on.

An experiment's ``logfile.log`` starts when an experiment is opened, so what
happens before that -- connecting, the load dialog, a failure at start-up -- was
written nowhere. :func:`start` adds one file per program under the log folder,
and it stays for the life of the process, alongside the experiment's.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import fibsem.config as cfg

FOLDER = "app"
# Checked when a program starts: a file past this is kept as ``<name>.1.log``.
MAX_BYTES = 5_000_000
FORMAT = (
    "%(asctime)s — %(process)d — %(name)s — %(levelname)s — "
    "%(funcName)s:%(lineno)d — %(message)s"
)

_handler: Optional[logging.Handler] = None


def folder() -> Path:
    return Path(cfg.LOG_PATH) / FOLDER


def path_for(app: str) -> Path:
    name = "".join(c if c.isalnum() else "-" for c in app.lower())
    return folder() / f"{name}.log"


def handler() -> Optional[logging.Handler]:
    """The program's log handler, once started. ``configure_logging`` keeps it
    when it replaces the root logger's handlers."""
    return _handler


def start(app: str) -> Optional[Path]:
    """Log this process to the file of the program called *app*. Safe to call
    again. Returns the file, or None when it could not be opened."""
    global _handler
    if _handler is not None:
        return Path(_handler.baseFilename)
    path = path_for(app)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _roll_over(path)
        new = logging.FileHandler(path, encoding="utf-8", delay=True)
    except OSError:
        logging.getLogger(__name__).warning(
            "Could not open the program log %s", path, exc_info=True
        )
        return None
    new.setLevel(logging.INFO)
    new.setFormatter(logging.Formatter(FORMAT))
    root = logging.getLogger()
    if root.level == logging.NOTSET or root.level > logging.INFO:
        # Handlers already there keep hearing what they heard before.
        for other in root.handlers:
            if other.level == logging.NOTSET:
                other.setLevel(root.level)
        root.setLevel(logging.INFO)
    root.addHandler(new)
    _handler = new
    return path


def restart() -> None:
    """Put the program's log back on the root logger after something replaced
    its handlers. The file opens again at the next line written."""
    if _handler is not None and _handler not in logging.getLogger().handlers:
        logging.getLogger().addHandler(_handler)


def _roll_over(path: Path) -> None:
    """Keep one older file. Done at start-up and not while writing: a second
    copy of the program has the file open, and Windows will not rename it then,
    in which case this run goes on in the same file."""
    try:
        if path.stat().st_size < MAX_BYTES:
            return
        os.replace(path, path.with_name(f"{path.stem}.1.log"))
    except OSError:
        pass
