"""Which volume tasks exist, and how a saved configuration becomes one."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

from fibsem.applications.volume.tasks.acquisition import (
    VolumeSEMAcquisitionTask,
    VolumeSEMAcquisitionTaskConfig,
)
from fibsem.applications.volume.tasks.base import VolumeTask
from fibsem.applications.volume.tasks.milling import (
    VolumeFIBMillingTask,
    VolumeFIBMillingTaskConfig,
)
from fibsem.plugins.loader import PluginRegistry, subclass_of
from fibsem.workflows.task_config import TaskConfig, load_task_configs

if TYPE_CHECKING:
    from fibsem.applications.volume.experiment import VolumeExperiment
    from fibsem.applications.volume.position import VolumePosition
    from fibsem.microscope import FibsemMicroscope

__all__ = [
    "VOLUME_TASK_ENTRY_POINT_GROUP",
    "VOLUME_TASK_PLUGINS",
    "VolumeFIBMillingTask",
    "VolumeFIBMillingTaskConfig",
    "VolumeSEMAcquisitionTask",
    "VolumeSEMAcquisitionTaskConfig",
    "VolumeTask",
    "get_volume_tasks",
    "load_volume_task_configs",
    "run_volume_task",
]

BUILTIN_VOLUME_TASKS: Dict[str, Type[VolumeTask]] = {
    cls.config_cls.task_type: cls
    for cls in (VolumeFIBMillingTask, VolumeSEMAcquisitionTask)
}

VOLUME_TASK_ENTRY_POINT_GROUP = "fibsem.volume.tasks"

# Built-ins and the fibsem.volume.tasks entry points. To add a plugin volume
# task, add to your package's pyproject.toml:
#
#     [project.entry-points.'fibsem.volume.tasks']
#     my_task = "my_package.tasks:MyVolumeTask"
VOLUME_TASK_PLUGINS: PluginRegistry[Type[VolumeTask]] = PluginRegistry(
    group=VOLUME_TASK_ENTRY_POINT_GROUP,
    kind="volume task",
    resolve=subclass_of(VolumeTask, lambda cls: cls.config_cls.task_type),
    builtins=BUILTIN_VOLUME_TASKS,
)


def get_volume_tasks() -> Dict[str, Type[VolumeTask]]:
    """Every volume task, by task_type: built-ins, then runtime registrations,
    then plugins."""
    return VOLUME_TASK_PLUGINS.all()


def load_volume_task_configs(
    ddict: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, TaskConfig]:
    """name -> config, skipping (with a warning) any type this build does not know."""
    return load_task_configs(VOLUME_TASK_PLUGINS, ddict, kind="volume task")


def run_volume_task(
    microscope: "FibsemMicroscope",
    task_name: str,
    experiment: "VolumeExperiment",
    position: "VolumePosition",
    iteration: Optional[int] = None,
    task_manager: Optional[Any] = None,
) -> VolumeTask:
    """Run the named task from the experiment's protocol on one position, for
    slice ``iteration`` (the manager's, when it runs one).

    The saved config is deep-copied per run, so a task that mutates its config
    does not write back into the protocol.
    """
    saved = experiment.task_protocol.task_config.get(task_name)
    if saved is None:
        raise KeyError(f"Task '{task_name}' is not in the volume protocol.")
    task_cls = VOLUME_TASK_PLUGINS.get(saved.task_type)
    if task_cls is None:
        raise KeyError(f"Volume task type '{saved.task_type}' is not registered.")
    task = task_cls(
        microscope=microscope,
        config=deepcopy(saved),
        position=position,
        experiment=experiment,
        task_manager=task_manager,
    )
    if iteration is not None:
        task.iteration = iteration
    task.run()
    return task
