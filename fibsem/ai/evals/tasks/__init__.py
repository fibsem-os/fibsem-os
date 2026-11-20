"""The eval tasks, one folder each, with everything for the task: each
``task*.py`` in a folder defines one task as ``TASK`` (a folder may hold a
static and a live one), ``make*.py`` its cases from the simulator and
``scripted.py`` its baseline. ``get(name)`` and ``names()`` load them on
first use, so a task module can use the evals' shared helpers; a new task
is a new folder, found without being listed anywhere."""

import importlib
import pkgutil
from typing import Dict, List, Optional

from fibsem.ai.evals.tasks.base import EvalTask, FromRun, Run, Sim

__all__ = ["EvalTask", "FromRun", "Run", "Sim", "all_tasks", "get", "names"]

_TASKS: Dict[str, EvalTask] = {}


def all_tasks() -> Dict[str, EvalTask]:
    if not _TASKS:
        for folder in pkgutil.iter_modules(__path__):
            if not folder.ispkg:
                continue
            package = importlib.import_module(f"{__name__}.{folder.name}")
            for info in pkgutil.iter_modules(package.__path__):
                if info.name.startswith("task"):
                    module = importlib.import_module(f"{package.__name__}.{info.name}")
                    _TASKS[module.TASK.name] = module.TASK
    return _TASKS


def get(name: Optional[str]) -> Optional[EvalTask]:
    return all_tasks().get(name) if name else None


def names() -> List[str]:
    return sorted(all_tasks())
