"""The operator's clicks take control of the microscope.

A worker takes it with ``FunctionWorker.start(control=...)``; a one-shot write on
the GUI thread takes it with :func:`operator_acts`. Either way a refusal names
who is in control, and nothing is done. The operator's own clicks never refuse
each other. Stops never take it (``STOP_COMMANDS``).
"""

from typing import Any, Callable, Optional

from fibsem.acting import OPERATOR
from fibsem.control import Control, MicroscopeBusy
from fibsem.ui import notification_service

__all__ = ["control_of", "operator_acts", "refused"]


def control_of(device: Any) -> Optional[Control]:
    """The control of the microscope a device or service belongs to, or None."""
    owner = device
    while owner is not None:
        control = getattr(owner, "control", None)
        if isinstance(control, Control):
            return control
        owner = getattr(owner, "parent", None)
    return None


def refused(busy: MicroscopeBusy) -> None:
    """Tell the operator their click was refused, and who is in control."""
    notification_service.show_toast(str(busy), "warning")


def operator_acts(
    control: Optional[Control],
    action: str,
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> bool:
    """Call ``fn`` holding control as the operator. False, with a toast, when
    someone else is in control; the caller shows the value again."""
    if control is None:
        fn(*args, **kwargs)
        return True
    try:
        with control.take(OPERATOR, action, join=True):
            fn(*args, **kwargs)
    except MicroscopeBusy as busy:
        refused(busy)
        return False
    return True
