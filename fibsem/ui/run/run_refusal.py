"""The window's side of a refusal: said on a toast, with the App Server's words
for an edit no run may see change under it (`WorkflowRunner.run_refusal`)."""

from typing import Any

from fibsem.ui import notification_service


def refused(why: str) -> bool:
    """Whether there is a refusal, *why*; says it on a toast when there is."""
    if why:
        notification_service.show_toast(why, "warning")
    return bool(why)


def refused_during_run(runner: Any, what: str) -> bool:
    """Whether *what* must wait for the run; says why when it must."""
    return refused(runner.run_refusal(what))
