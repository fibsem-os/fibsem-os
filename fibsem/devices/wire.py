"""How parameter values cross the network between a device server and a remote device.

JSON has no tuple, enum or dataclass. Both sides encode with ``to_wire`` and decode with
``from_wire`` against the parameter's declared type, so a remote read returns what the
local read returns: a tuple stays a tuple, a ``Point`` stays a ``Point``.
"""

from __future__ import annotations

import collections.abc
import logging
import types
import typing
from enum import Enum
from typing import Any, Callable, Dict, List, NamedTuple, Union

# How the server sends an array (a camera frame) and the remote driver recognises one.
NPY_MEDIA_TYPE = "application/x-npy"

# The JSON header a `Frame`'s metadata travels in, beside its array.
FRAME_METADATA_HEADER = "X-Frame-Metadata"


# How the server sends several frames (a z-stack) in one answer: an ``np.savez``
# archive of the arrays, with their metadata as one JSON entry inside it rather than
# a header, which a long stack would outgrow.
FRAMES_MEDIA_TYPE = "application/x-npz-frames"


class Frame(NamedTuple):
    """A camera frame and what it was taken with, read next to the hardware.

    A command returns one so that building the image needs no reads after it: on a
    remote FM, each of those reads is a request. ``metadata`` is plain JSON-ready data;
    `fibsem.devices.fm.FM.acquire_frame` lists its keys.
    """

    data: Any  # np.ndarray; Any so this module doesn't import numpy
    metadata: Dict[str, Any]


def frames_to_bytes(frames: List[Frame]) -> bytes:
    """Frames as one ``np.savez`` archive: ``frame_<i>`` each array, ``metadata`` the
    JSON list of their metadata, in order."""
    import io
    import json

    import numpy as np

    buffer = io.BytesIO()
    arrays = {f"frame_{i}": np.asarray(frame.data) for i, frame in enumerate(frames)}
    metadata = json.dumps([frame.metadata for frame in frames])
    np.savez(buffer, metadata=np.array(metadata), **arrays)
    return buffer.getvalue()


def frames_from_bytes(content: bytes) -> List[Frame]:
    """The inverse of ``frames_to_bytes``."""
    import io
    import json

    import numpy as np

    with np.load(io.BytesIO(content), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        return [Frame(archive[f"frame_{i}"], md) for i, md in enumerate(metadata)]


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


# The origins of the sequence hints ``from_wire`` decodes element by element.
_SEQUENCES = (list, tuple, collections.abc.Sequence)


def from_wire(type_: type, value: Any) -> Any:
    """The inverse of ``to_wire`` for a parameter declared as ``type_``.

    A ``Sequence[X]``, ``List[X]`` or ``Tuple[X, ...]`` decodes each element as X, so
    a list of pattern dicts arrives as patterns.

    A number or bool may arrive as its text, ``"4.0"``: an agent's MCP client sends a
    value whose schema names no type as a string. It is read as the declared type,
    and a whole float (``4.0``) as an int where the type is int. Text that doesn't
    parse passes as it came, for the parameter's own check to refuse.
    """
    if type_ in (int, float, bool) and not isinstance(value, bool):
        if isinstance(value, str):
            return _scalar_from_text(type_, value)
        if type_ is int and isinstance(value, float) and value.is_integer():
            return int(value)
    if type_ is tuple and isinstance(value, list):
        return tuple(value)
    origin = typing.get_origin(type_)
    if origin in _SEQUENCES and isinstance(value, list):
        args = typing.get_args(type_)
        if origin is tuple and args and args[-1] is not Ellipsis:  # Tuple[X, Y]
            return tuple(from_wire(arg, item) for arg, item in zip(args, value))
        items = [from_wire(args[0], item) if args else item for item in value]
        return tuple(items) if origin is tuple else items
    if isinstance(type_, type) and issubclass(type_, Enum) and value is not None:
        return value if isinstance(value, type_) else _enum_member(type_, value)
    if isinstance(value, dict) and hasattr(type_, "from_dict"):
        return type_.from_dict(value)
    return value


def _scalar_from_text(type_: type, text: str) -> Any:
    """``text`` read as an int, float or bool, or ``text`` itself if it isn't one."""
    word = text.strip()
    if type_ is bool:
        return {"true": True, "false": False}.get(word.lower(), text)
    try:
        number = float(word)
    except ValueError:
        return text
    if type_ is float:
        return number
    return int(number) if number.is_integer() else text


def _enum_member(type_: type, value: Any) -> Any:
    """The member with this value, or with this name: an agent reading
    ``<BeamType.ELECTRON: 1>`` in a signature sends ``"ELECTRON"``."""
    try:
        return type_(value)
    except ValueError:
        if isinstance(value, str) and value in type_.__members__:
            return type_.__members__[value]
        raise


class ArgumentError(ValueError):
    """A command's argument that does not decode as its declared type."""

    def __init__(self, parameter: str, error: Exception):
        super().__init__(f"{parameter}: {error}")
        self.parameter = parameter


def _declared_type(hint: Any) -> Any:
    """``Optional[X]`` is X; any other hint is itself."""
    if typing.get_origin(hint) is Union:
        args = [arg for arg in typing.get_args(hint) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return hint


def type_hints(func: Callable[..., Any]) -> Dict[str, Any]:
    """``func``'s parameter hints, each resolved on its own: one that names something
    imported only for type checking is left out, not the rest with it."""
    try:
        return typing.get_type_hints(func)
    except Exception:
        pass
    target = getattr(func, "__func__", func)  # a bound method's function
    globalns = getattr(target, "__globals__", {})
    hints: Dict[str, Any] = {}
    for name, annotation in getattr(target, "__annotations__", {}).items():
        holder = types.SimpleNamespace(__annotations__={name: annotation})
        try:
            hints[name] = typing.get_type_hints(holder, globalns)[name]
        except Exception as error:  # a hint only importable for type checking
            logging.debug(f"{func.__qualname__}: no hint for {name!r} ({error})")
    return hints


def decode_kwargs(func: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """A command's arguments as its signature declares them: a ``Point`` argument
    sent as a dict arrives as a ``Point``. Arguments without a usable hint pass as
    they came.

    Raises:
        ArgumentError: an argument does not decode; it names which.
    """
    hints = type_hints(func)
    decoded = {}
    for name, value in kwargs.items():
        if name not in hints:
            decoded[name] = value
            continue
        try:
            decoded[name] = from_wire(_declared_type(hints[name]), value)
        except Exception as error:
            raise ArgumentError(name, error) from error
    return decoded
