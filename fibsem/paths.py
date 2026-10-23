"""Where fibsemOS keeps what it writes, apart from what ships in the package.

Everything an install writes lives under one state root: ``FIBSEM_HOME`` when it is
set, else ``~/.fibsem``. The package folder holds only shipped files and is never
written. The files that let this account's processes find each other (the App
Server's discovery file, pairing tokens, the agent's files) stay in ``~/.fibsem``
whatever ``FIBSEM_HOME`` says, so a process started without it still finds them.
"""

import os
from pathlib import Path

STATE_ROOT_ENV = "FIBSEM_HOME"

PACKAGE_DIR = Path(__file__).resolve().parent


def user_dir() -> Path:
    """``~/.fibsem``: discovery, pairing and agent files, for this account."""
    return Path.home() / ".fibsem"


def state_root() -> Path:
    """The folder everything else this install writes goes under."""
    override = os.environ.get(STATE_ROOT_ENV)
    return Path(override).expanduser() if override else user_dir()
