"""How parameter values cross the network between a device server and a remote device.

JSON has no tuple, enum or dataclass. Both sides encode with ``to_wire`` and decode with
``from_wire`` against the parameter's declared type, so a remote read returns what the
local read returns: a tuple stays a tuple, a ``Point`` stays a ``Point``.
"""

from __future__ import annotations

import logging
import typing
from enum import Enum
from typing import Any, Callable, Dict, NamedTuple, Union

# How the server sends an array (a camera frame) and the remote driver recognises one.
NPY_MEDIA_TYPE = "application/x-npy"

# The JSON header a `Frame`'s metadata travels in, beside its array.
FRAME_METADATA_HEADER = "X-Frame-Metadata"


class Frame(NamedTuple):
    """A camera frame and what it was taken with, read next to the hardware.

    A command returns one so that building the image needs no reads after it: on a
    remote FM, each of those reads is a request. ``metadata`` is plain JSON-ready data;
    `fibsem.devices.fm.FM.acquire_frame` lists its keys.
    """

    data: Any  # np.ndarray; Any so this module doesn't import numpy
    metadata: Dict[str, Any]


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


def _declared_type(hint: Any) -> Any:
    """``Optional[X]`` is X; any other hint is itself."""
    if typing.get_origin(hint) is Union:
        args = [arg for arg in typing.get_args(hint) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return hint


def decode_kwargs(func: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """A command's arguments as its signature declares them: a ``Point`` argument
    sent as a dict arrives as a ``Point``. Arguments without a usable hint pass as
    they came."""
    try:
        hints = typing.get_type_hints(func)
    except Exception as error:  # a hint only importable for type checking
        logging.debug(f"{func.__qualname__}: no type hints to decode with ({error})")
        return dict(kwargs)
    return {
        name: from_wire(_declared_type(hints[name]), value) if name in hints else value
        for name, value in kwargs.items()
    }
