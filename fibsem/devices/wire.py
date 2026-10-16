"""How parameter values cross the network between a device server and a remote device.

JSON has no tuple, enum or dataclass. Both sides encode with ``to_wire`` and decode with
``from_wire`` against the parameter's declared type, so a remote read returns what the
local read returns: a tuple stays a tuple, a ``Point`` stays a ``Point``.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import difflib
import inspect
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
    ``<BeamType.ELECTRON: 1>`` in a signature sends ``"ELECTRON"``, or its value as
    text, ``"1"``."""
    try:
        return type_(value)
    except ValueError:
        if isinstance(value, str) and value in type_.__members__:
            return type_.__members__[value]
        number = _scalar_from_text(int, value) if isinstance(value, str) else value
        if isinstance(number, int):
            try:
                return type_(number)
            except ValueError:
                pass
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
    # A bound method's function, and under a wrapper (a long command's) the
    # function it wraps, whose module the hints name things in.
    target = inspect.unwrap(getattr(func, "__func__", func))
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

    A structure sent as a dict must name only keys it has, at any depth: its
    ``from_dict`` would drop any other without a word and use the default in its
    place (`refuse_unknown_keys`).

    Raises:
        ArgumentError: an argument does not decode, or names a key its structure
            doesn't have; it names which (``settings.hfw``).
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
        refuse_unknown_keys(name, value, decoded[name])
    return decoded


def _is_structure(value: Any) -> bool:
    """A decoded structure: a dataclass, or anything else that writes ``to_dict``."""
    if isinstance(value, type):
        return False
    return dataclasses.is_dataclass(value) or hasattr(value, "to_dict")


def _known_keys(structure: Any) -> List[str]:
    """The keys ``structure`` reads from a dict: its dataclass fields, the keys its
    ``to_dict`` writes (so what it sends comes back in), and any it names in
    ``FROM_DICT_KEYS`` (a pattern's ``type``, which picks its class)."""
    keys = set(getattr(type(structure), "FROM_DICT_KEYS", ()))
    if dataclasses.is_dataclass(structure):
        keys.update(f.name for f in dataclasses.fields(structure))
    try:
        written = structure.to_dict()
    except Exception:  # a structure that can't write itself is checked by fields
        written = None
    if isinstance(written, dict):
        keys.update(written)
    return sorted(keys)


def _home_of(key: str, structure: Any, path: str, depth: int = 3) -> Any:
    """Where ``key`` belongs when it was sent one or more levels too high: the path
    of the nearest structure inside ``structure`` that has it, else None."""
    if depth == 0:
        return None
    children = []
    for name in _known_keys(structure):
        child = getattr(structure, name, None)
        if _is_structure(child):
            if key in _known_keys(child):
                return f"{path}.{name}.{key}"
            children.append((name, child))
    for name, child in children:
        home = _home_of(key, child, f"{path}.{name}", depth - 1)
        if home is not None:
            return home
    return None


def refuse_unknown_keys(path: str, sent: Any, decoded: Any) -> None:
    """Raise `ArgumentError` for the first key in ``sent``, at any depth, that the
    structure it decoded into doesn't have. The error names the key's path
    (``settings.hfw``) and, when a structure inside has that key, where it belongs
    (``settings.image_settings.hfw``), else the closest key it does have.

    Checked against what was decoded, not the declared type, so a list of patterns
    is checked against each pattern's own class.
    """
    if isinstance(sent, list) and isinstance(decoded, (list, tuple)):
        for index, (item, structure) in enumerate(zip(sent, decoded)):
            refuse_unknown_keys(f"{path}[{index}]", item, structure)
        return
    if not isinstance(sent, dict) or not _is_structure(decoded):
        return
    known = _known_keys(decoded)
    if not known:
        return
    for key, value in sent.items():
        if key not in known:
            where = _home_of(key, decoded, path)
            if where is None:
                close = difflib.get_close_matches(str(key), known, n=1)
                where = f"{path}.{close[0]}" if close else None
            message = f"{type(decoded).__name__} has no '{key}'"
            message += "." if where is None else f"; did you mean {where}?"
            message += f" It takes: {', '.join(known)}."
            raise ArgumentError(f"{path}.{key}", UnknownKeyError(message))
        refuse_unknown_keys(f"{path}.{key}", value, getattr(decoded, key, None))


class UnknownKeyError(ValueError):
    """A key in a structure's dict that the structure doesn't have."""
