"""Creating and opening a volume experiment: what the window's New and Load and
the App Server's create and load both do."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from fibsem.applications.volume.experiment import VolumeExperiment
from fibsem.applications.volume.protocol import VolumeProtocol
from fibsem.workflows.protocol_files import record_recent

__all__ = ["create_experiment", "load_for_adoption"]


def create_experiment(
    directory,
    name: str,
    protocol: Optional[VolumeProtocol] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> VolumeExperiment:
    """A new experiment *name* in *directory* with *protocol* (the default for
    None), saved and recorded as the most recent."""
    experiment = VolumeExperiment.create(
        Path(directory), name, metadata=metadata, task_protocol=protocol
    )
    record_recent(experiment)
    return experiment


def load_for_adoption(path) -> Tuple[Optional[VolumeExperiment], Optional[str]]:
    """The experiment at *path* (its folder or experiment.yaml) to open, or
    None and why not."""
    try:
        return VolumeExperiment.load(Path(path)), None
    except Exception as exc:  # noqa: BLE001 - a broken file is a data fact
        return None, f"could not load {path}: {exc}"
