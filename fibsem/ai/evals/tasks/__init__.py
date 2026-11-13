"""The eval tasks, one module each. ``get(name)`` and ``names()`` load them
on first use, so a task module can use the evals' shared helpers."""

import importlib
import pkgutil
from typing import Dict, List, Optional

from fibsem.ai.evals.tasks.base import EvalTask, Run, Sim

__all__ = ["EvalTask", "Run", "Sim", "all_tasks", "get", "names"]

_TASKS: Dict[str, EvalTask] = {}


def all_tasks() -> Dict[str, EvalTask]:
    if not _TASKS:
        for info in pkgutil.iter_modules(__path__):
            if info.name == "base":
                continue
            task = importlib.import_module(f"{__name__}.{info.name}").TASK
            _TASKS[task.name] = task
    return _TASKS


def get(name: Optional[str]) -> Optional[EvalTask]:
    return all_tasks().get(name) if name else None


def names() -> List[str]:
    return sorted(all_tasks())
