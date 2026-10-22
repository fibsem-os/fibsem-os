"""Which application wrote an experiment.yaml, and in which format.

Every experiment writes a top-level ``app`` and ``format_version``. A file with
no ``app`` was written before the tag existed, when AutoLamella was the only
application. Imports nothing heavier than the standard library, so the recents
list and directory listings can read the tag without loading anything.
"""

from __future__ import annotations

from typing import Any, Mapping

# The application every file written before the tag was.
DEFAULT_APP = "autolamella"
# The experiment.yaml format this build writes. 2: an item's proposals are one
# list, each saying who asked it, not keyed by the task that asked. 3: task
# configs are flat, and a protocol says id, task_config and workflow_config.
# 4: a task's attention and requires are on its plan entry, not its config.
FORMAT_VERSION = 4


class WrongApplication(ValueError):
    """An experiment.yaml written by another application than the one opening it."""

    def __init__(self, path: Any, app: str, expected: str) -> None:
        self.path = path
        self.app = app
        self.expected = expected
        super().__init__(
            f"{path} is a {app} experiment, not a {expected} one. Open it in {app}."
        )


def saved_app(ddict: Mapping[str, Any]) -> str:
    """The application that wrote ``ddict``, a loaded experiment.yaml."""
    return ddict.get("app") or DEFAULT_APP


def check_app(ddict: Mapping[str, Any], expected: str, path: Any = "") -> None:
    """Raise :class:`WrongApplication` unless ``expected`` wrote ``ddict``.

    Reading another application's file would drop every key it does not know,
    and the next save would overwrite the file without them.
    """
    app = saved_app(ddict)
    if app != expected:
        raise WrongApplication(path, app, expected)
