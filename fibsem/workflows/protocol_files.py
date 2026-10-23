"""An app's protocol files, for every app: the files in its protocol folder by
name, the recents, and making a protocol the open experiment's."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fibsem.config import record_recent_experiment
from fibsem.workflows.experiment import EXPERIMENT_FILENAME
from fibsem.workflows.session import PROTOCOL

__all__ = [
    "adopt_protocol",
    "list_protocols",
    "protocol_by_name",
    "protocol_file_for",
    "record_recent",
]


def list_protocols(folder: str, default: Optional[str] = None) -> List[Dict[str, Any]]:
    """The protocol files an agent may choose: those in *folder*, plus
    *default* when it lives elsewhere. By file name, sorted."""
    paths = {}
    if os.path.isdir(folder):
        for name in os.listdir(folder):
            if name.endswith((".yaml", ".yml")):
                paths[name] = os.path.join(folder, name)
    if default:
        paths.setdefault(os.path.basename(default), default)
    return [
        {"name": name, "path": path, "default": path == default}
        for name, path in sorted(paths.items())
    ]


def protocol_by_name(folder: str, default: Optional[str], name: str) -> Optional[str]:
    """The path of the listed protocol file called *name*, or None."""
    listed = list_protocols(folder, default)
    return next((p["path"] for p in listed if p["name"] == name), None)


def protocol_file_for(folder: str, name: str) -> str:
    """The path in *folder* a protocol saved as *name* goes to: a plain file
    name, ``.yaml`` added when it has no extension. Raises ValueError for a
    name that is a path."""
    name = name.strip()
    if not name or os.path.basename(name) != name or name in (".", ".."):
        raise ValueError(f"A protocol file name, not a path: {name!r}.")
    if not name.endswith((".yaml", ".yml")):
        name += ".yaml"
    return os.path.join(folder, name)


def record_recent(experiment) -> None:
    """Make *experiment* the last one used, at the top of the recents."""
    record_recent_experiment(os.path.join(experiment.path, EXPERIMENT_FILENAME))


def adopt_protocol(session, protocol) -> None:
    """Make *protocol* the open experiment's and save it beside it: the Load
    Protocol dialog's OK. Where the experiment is edited; whoever draws the
    protocol hears it on ``session.changed``."""
    session.experiment.task_protocol = protocol
    session.experiment.save_protocol()
    session.changed.emit(PROTOCOL)
