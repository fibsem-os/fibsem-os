"""A server's error answer as the local exception it names, so a remote
call raises what the local call would: the remote driver's, and the client's
(`fibsem.client`)."""

from typing import Any, Callable, Dict, Optional

from fibsem.acting import ActorKind
from fibsem.cancellation import NothingToStop, OperationCancelledError
from fibsem.control import InControl, MicroscopeBusy
from fibsem.devices.core import (
    DeviceFault,
    ParameterReadOnly,
    ParameterUnavailable,
    ResourceBusy,
)
from fibsem.devices.manipulator import ManipulatorLimitError
from fibsem.devices.sample_loader import GridExchangeError
from fibsem.devices.stage import StageLimitError

_ERRORS: Dict[str, Callable[[str], Exception]] = {
    "ParameterUnavailable": ParameterUnavailable,
    "ParameterReadOnly": ParameterReadOnly,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "NotImplementedError": NotImplementedError,
    "StageLimitError": StageLimitError,
    "ManipulatorLimitError": ManipulatorLimitError,
    "OperationCancelledError": OperationCancelledError,
    "NothingToStop": NothingToStop,
    "GridExchangeError": GridExchangeError,
}


def error_from(
    detail: Dict[str, Any], status: Optional[int] = None
) -> Optional[Exception]:
    """The local exception a server's error *detail* names, or None when it has
    no local meaning. *status* is the answer's HTTP status, when there was one.

    The detail has error_type and message; an older server sends error and
    detail. A long command's failed run carries the same detail.
    """
    kind = detail.get("error_type", detail.get("error"))
    message = detail.get("message", detail.get("detail"))
    if kind == "DeviceFault" or status == 502:
        # the hardware refused on the far side: a driver's own subclass arrives
        # as the fault it is
        return DeviceFault(message, device=detail.get("device"))
    if kind == "ResourceBusy":
        return ResourceBusy(detail.get("resource"), detail.get("holder"), message)
    if kind == "microscope_busy":
        held = detail.get("in_control") or {}
        by = ActorKind(held.get("by", ActorKind.operator.value))
        return MicroscopeBusy(InControl(by, held.get("action", ""), since=0.0))
    if kind in _ERRORS:
        return _ERRORS[kind](message)
    return None
