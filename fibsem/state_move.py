"""Bring an install's files out of the package folder into the state root, once.

Releases before the state root wrote settings, logs, experiments and protocols
inside the installed package. On the first start with no ``<root>/config``, this
copies the settings (the originals stay, so an older release still finds them)
and moves the logs, experiments, protocols and scripts, then repoints the
preferences and the list of configurations at the new places. Every file it
copies or moves is logged and listed in ``<root>/MOVED-FROM.txt``.

It imports nothing from ``fibsem.config``, which calls it before reading anything.
"""

import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# What the package ships in config/ and the AutoLamella protocol folder. Anything
# else there was written by the install.
SHIPPED_CONFIGURATIONS = frozenset(
    {
        "jeol-configuration.yaml",
        "microscope-configuration.yaml",
        "odemis-configuration.yaml",
        "sim-arctis-configuration.yaml",
        "sim-iflm-configuration.yaml",
        "tescan-configuration.yaml",
        "tfs-aquilos2-configuration.yaml",
        "tfs-arctis-configuration.yaml",
        "tfs-hydra-configuration.yaml",
    }
)
SHIPPED_FM_FILES = frozenset({"fm-configuration-default.yaml"})
SHIPPED_PROTOCOLS = frozenset({"task-protocol.yaml", "task-protocol-waffle.yaml"})

PREFERENCES_FILE = "user-preferences.yaml"
CONFIGURATIONS_FILE = "user-configurations.yaml"
RECORD_FILE = "MOVED-FROM.txt"
YAML_SUFFIXES = (".yaml", ".yml")

# What the last run did, for the first log file opened after it
# (`fibsem.utils.configure_logging`): it runs at import, before any log file is set up.
_pending: List[str] = []


@dataclass
class Moves:
    """What one run copied and moved, old path to new."""

    copied: Dict[str, str] = field(default_factory=dict)
    moved: Dict[str, str] = field(default_factory=dict)
    left: List[str] = field(default_factory=list)

    def new_path(self, old: str) -> Optional[str]:
        """Where *old*, or the file or folder it is inside, went; None if nowhere."""
        old = _norm(old)
        for table in (self.moved, self.copied):
            for source, target in table.items():
                if old == _norm(source):
                    return target
                if _inside(old, source):
                    return os.path.join(target, os.path.relpath(old, source))
        return None

    def lines(self) -> List[str]:
        out = [f"copied {s} -> {t}" for s, t in self.copied.items()]
        out += [f"moved {s} -> {t}" for s, t in self.moved.items()]
        out += [f"left {s}" for s in self.left]
        return out


def _norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def _inside(path: str, folder) -> bool:
    return _norm(path).startswith(_norm(folder) + os.sep)


def move_package_state(package_dir: Path, root: Path, scripts_from: Path) -> Moves:
    """Copy and move what an older release wrote inside *package_dir* to *root*.

    Does nothing when ``<root>/config`` exists: that root is already in use."""
    moves = Moves()
    config_to = root / "config"
    if config_to.exists():
        return moves
    config_from = package_dir / "config"
    autolamella = package_dir / "applications" / "autolamella"

    _copy_settings(config_from, config_to, moves)
    _move_children(package_dir / "log", root / "log", moves)
    _move_experiments(autolamella / "log", root / "experiments", moves)
    _move_protocols(autolamella / "protocol", root / "autolamella" / "protocols", moves)
    _move_children(scripts_from, root / "autolamella" / "scripts", moves)
    _repoint_configurations(config_from, config_to, moves)
    _repoint_preferences(config_to / PREFERENCES_FILE, package_dir, moves)

    if moves.copied or moves.moved:
        _record(root, package_dir, moves)
    return moves


def _copy(source: Path, target: Path, moves: Moves) -> None:
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
        moves.copied[str(source)] = str(target)
    except OSError as e:
        logging.warning(f"Could not copy {source} to {target}: {e}")
        moves.left.append(str(source))


def _move(source: Path, target: Path, moves: Moves) -> None:
    if target.exists():
        logging.warning(f"Not moving {source}: {target} already exists.")
        moves.left.append(str(source))
        return
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        moves.moved[str(source)] = str(target)
    except OSError as e:
        logging.warning(f"Could not move {source} to {target}: {e}")
        moves.left.append(str(source))


def _copy_settings(config_from: Path, config_to: Path, moves: Moves) -> None:
    """Every file the package does not ship, the session state and the FM folder."""
    if not config_from.is_dir():
        return
    for entry in sorted(config_from.iterdir()):
        if entry.is_file() and entry.suffix in YAML_SUFFIXES:
            if entry.name not in SHIPPED_CONFIGURATIONS:
                _copy(entry, config_to / entry.name, moves)
        elif entry.is_dir() and entry.name == "session":
            _copy(entry, config_to / entry.name, moves)
        elif entry.is_dir() and entry.name == "fm":
            for item in sorted(entry.iterdir()):
                if item.is_file() and item.name not in SHIPPED_FM_FILES:
                    _copy(item, config_to / "fm" / item.name, moves)


def _move_children(source: Path, target: Path, moves: Moves) -> None:
    if not source.is_dir():
        return
    for entry in sorted(source.iterdir()):
        if entry.name.startswith("."):
            continue
        _move(entry, target / entry.name, moves)


def _move_experiments(source: Path, target: Path, moves: Moves) -> None:
    """The folders holding an experiment.yaml; anything else stays."""
    if not source.is_dir():
        return
    for entry in sorted(source.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_dir() and (entry / "experiment.yaml").is_file():
            _move(entry, target / entry.name, moves)
        else:
            moves.left.append(str(entry))


def _move_protocols(source: Path, target: Path, moves: Moves) -> None:
    """The protocol files at the top of the folder that the package does not ship."""
    if not source.is_dir():
        return
    for entry in sorted(source.iterdir()):
        if (
            entry.is_file()
            and entry.suffix in YAML_SUFFIXES
            and entry.name not in SHIPPED_PROTOCOLS
        ):
            _move(entry, target / entry.name, moves)


def _read(path: Path) -> Optional[dict]:
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        logging.warning(f"Could not read {path}: {e}")
        return None
    return data if isinstance(data, dict) else None


def _write(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


def _repoint_configurations(config_from: Path, config_to: Path, moves: Moves) -> None:
    """Point the list of configurations at the copies. A configuration it names
    inside the package that was not copied (a shipped one, edited where it was) is
    copied now, so a later save goes to the copy rather than into the package."""
    path = config_to / CONFIGURATIONS_FILE
    data = _read(path) if path.exists() else None
    if not data or not isinstance(data.get("configurations"), dict):
        return
    changed = False
    for entry in data["configurations"].values():
        old = entry.get("path") if isinstance(entry, dict) else None
        if not old or not _inside(old, config_from):
            continue
        new = moves.new_path(old)
        if new is None and os.path.isfile(old):
            target = config_to / os.path.basename(old)
            _copy(Path(old), target, moves)
            new = moves.new_path(old)
        if new is not None:
            entry["path"] = new
            changed = True
    if changed:
        _write(path, data)


def _repoint_preferences(path: Path, package_dir: Path, moves: Moves) -> None:
    """The experiment folder, default protocol and recents, where they moved."""
    data = _read(path) if path.exists() else None
    experiment = data.get("experiment") if data else None
    if not isinstance(experiment, dict):
        return

    def where(value):
        if not value or not _inside(value, package_dir):
            return value
        return moves.new_path(value) or value

    before = dict(experiment)
    folder = experiment.get("default_experiment_directory")
    if folder and _inside(folder, package_dir):
        # The new default, <root>/experiments, takes over.
        experiment["default_experiment_directory"] = ""
    for key in ("default_protocol_path", "last_experiment_path"):
        experiment[key] = where(experiment.get(key))
    recents = experiment.get("recent_experiments")
    if isinstance(recents, list):
        experiment["recent_experiments"] = [where(p) for p in recents]
    if experiment != before:
        _write(path, data)


def _record(root: Path, package_dir: Path, moves: Moves) -> None:
    summary = (
        f"Moved fibsemOS settings from {package_dir} to {root}: "
        f"{len(moves.copied)} copied (originals left in place), "
        f"{len(moves.moved)} moved, {len(moves.left)} left where they were. "
        f"Listed in {root / RECORD_FILE}."
    )
    logging.warning(summary)
    _pending[:] = [summary] + moves.lines()
    try:
        root.mkdir(parents=True, exist_ok=True)
        with open(root / RECORD_FILE, "w") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')}\n{summary}\n\n")
            f.write("\n".join(moves.lines()) + "\n")
    except OSError as e:
        logging.warning(f"Could not write {root / RECORD_FILE}: {e}")


def log_pending() -> None:
    """Write what the move did into the log just configured, once."""
    for line in _pending:
        logging.warning(line)
    _pending.clear()
