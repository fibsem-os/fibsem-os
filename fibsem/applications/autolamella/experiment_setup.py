"""Creating an experiment and choosing its protocol, without the dialogs.

What the Create Experiment and Load Protocol dialogs do once their fields are
filled in, so the App Server runs the same steps for an agent. The dialogs keep
their own browsing, validation messages and confirmations; the agent is kept to
known places: new experiments go in the default experiments folder, and a
protocol is one of the files in the protocol folder.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fibsem.applications.autolamella import config as cfg
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.config import (
    add_recent_experiment,
    load_user_preferences,
    save_user_preferences,
)


def default_experiment_directory() -> str:
    """Where a new experiment goes: the preference when it names a folder that
    exists, else the AutoLamella log folder (the Create dialog's default)."""
    preferred = load_user_preferences().experiment.default_experiment_directory
    if preferred and os.path.isdir(preferred):
        return str(preferred)
    if preferred:
        logging.warning(
            f"Preference default_experiment_directory '{preferred}' does not exist; using default."
        )
    return str(cfg.LOG_PATH)


def default_protocol_path() -> str:
    """The protocol a new experiment starts with: the preference when the file
    exists, else the shipped task protocol."""
    preferred = load_user_preferences().experiment.default_protocol_path
    if preferred and os.path.isfile(preferred):
        return str(preferred)
    if preferred:
        logging.warning(
            f"Preference default_protocol_path '{preferred}' does not exist; using default."
        )
    return str(cfg.TASK_PROTOCOL_PATH)


def protocol_directory() -> str:
    """The folder the protocol dialogs open at."""
    return os.path.dirname(str(cfg.TASK_PROTOCOL_PATH))


def list_protocols() -> List[Dict[str, Any]]:
    """The protocol files an agent may choose: those in the protocol folder,
    plus the default when it lives elsewhere. By file name, sorted."""
    folder = protocol_directory()
    paths = {}
    if os.path.isdir(folder):
        for name in os.listdir(folder):
            if name.endswith((".yaml", ".yml")):
                paths[name] = os.path.join(folder, name)
    default = default_protocol_path()
    paths.setdefault(os.path.basename(default), default)
    return [
        {"name": name, "path": path, "default": path == default}
        for name, path in sorted(paths.items())
    ]


def protocol_by_name(name: str) -> Optional[str]:
    """The path of the listed protocol file called *name*, or None."""
    return next((p["path"] for p in list_protocols() if p["name"] == name), None)


def record_recent(experiment: Experiment) -> None:
    """Make *experiment* the last one used, at the top of the recents."""
    prefs = load_user_preferences()
    prefs.experiment.last_experiment_path = str(experiment.path)
    add_recent_experiment(prefs, os.path.join(experiment.path, "experiment.yaml"))
    save_user_preferences(prefs)


def load_for_adoption(path) -> Tuple[Optional[Experiment], Optional[str]]:
    """Load the experiment at *path* to open it, or None and why not: the bar
    the Load dialog and Quickload set. One with no protocol has nothing to run,
    and opening it would only look loaded."""
    try:
        experiment = Experiment.load(Path(path))
    except Exception as exc:  # noqa: BLE001 - a broken file is a data fact
        return None, f"could not load {path}: {exc}"
    if experiment.task_protocol is None:
        return None, f"{experiment.name} has no protocol"
    return experiment, None


def create_experiment(
    directory: str,
    name: str,
    protocol: AutoLamellaTaskProtocol,
    metadata: Optional[Dict[str, Any]] = None,
) -> Experiment:
    """Create an experiment in *directory* with *protocol*, save the protocol
    beside it and record it as the most recent: the Create dialog's OK."""
    experiment = Experiment.create(path=Path(directory), name=name, metadata=metadata)
    experiment.task_protocol = protocol
    protocol.save(os.path.join(experiment.path, "protocol.yaml"))
    record_recent(experiment)
    return experiment
