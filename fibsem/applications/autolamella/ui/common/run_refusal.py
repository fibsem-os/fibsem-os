"""The window's side of `WorkflowRunner.run_refusal`: an edit no run may see
change under it is refused while one runs, with the App Server's words."""

from typing import Any

from fibsem.ui import notification_service


def refused_during_run(runner: Any, what: str) -> bool:
    """Whether *what* must wait for the run; says why when it must."""
    why = runner.run_refusal(what)
    if why:
        notification_service.show_toast(why, "warning")
    return bool(why)
