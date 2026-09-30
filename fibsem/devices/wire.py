"""How parameter values cross the network between a device server and a remote device.

JSON has no tuple, enum or dataclass. Both sides encode with ``to_wire`` and decode with
``from_wire`` against the parameter's declared type, so a remote read returns what the
local read returns: a tuple stays a tuple, a ``Point`` stays a ``Point``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

# How the server sends an array (a camera frame) and the remote driver recognises one.
NPY_MEDIA_TYPE = "application/x-npy"


def to_wire(value: Any) -> Any:
    """A JSON-ready form: a structure with ``to_dict`` sends that, an enum its value,
    a tuple a list."""
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict") and not isinstance(value, type):
        return value.to_dict()
    if isinstance(value, tuple):
        return [to_wire(item) for item in value]
    return value


def from_wire(type_: type, value: Any) -> Any:
    """The inverse of ``to_wire`` for a parameter declared as ``type_``."""
    if type_ is tuple and isinstance(value, list):
        return tuple(value)
    if isinstance(type_, type) and issubclass(type_, Enum) and value is not None:
        return value if isinstance(value, type_) else type_(value)
    if isinstance(value, dict) and hasattr(type_, "from_dict"):
        return type_.from_dict(value)
    return value
