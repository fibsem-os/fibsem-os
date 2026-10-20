"""How values cross the network between a server and its clients.

Values are written with ``to_wire`` and read with ``fibsem.codec.decode`` against the
declared type, so a remote read returns what the local read returns: a tuple stays a
tuple, a ``Point`` a ``Point``. A command's arguments are read strictly
(``decode_kwargs``): a key its structure doesn't have is refused, not dropped.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import difflib
import importlib
import inspect
import logging
import types
import typing
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
from pydantic import PydanticSchemaGenerationError, ValidationError

from fibsem.codec import Tagged, decode, encode

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

    data: Any  # np.ndarray
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
    """``value`` as JSON-ready data, as ``fibsem.codec.encode`` writes it: an enum by
    name or value as it declares, a tuple as a list.

    One of several classes saved under a name (a pattern's shape) is written as
    its base writes it, with the name, so it reads back where the base is declared.
    A result assembled with its images (an autofocus result) has no codec form; it
    writes its own ``to_dict``, which leaves the images out.
    """
    try:
        return encode(value, _tagged_base(type(value)))
    except PydanticSchemaGenerationError:
        if hasattr(value, "to_dict"):
            return value.to_dict()
        raise


def _tagged_base(cls: type) -> Optional[type]:
    """The class that saves ``cls`` under its name (`Tagged`), or None."""
    return next((base for base in cls.__mro__ if Tagged in base.__bases__), None)


class ArgumentError(ValueError):
    """A command's argument that does not decode as its declared type."""

    def __init__(self, parameter: str, error: Exception):
        super().__init__(f"{parameter}: {error}")
        self.parameter = parameter


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


def decode_argument(name: str, hint: Any, value: Any) -> Any:
    """``value`` read as ``hint``, strictly: a key its structure doesn't have is
    refused at any depth, since a default in its place would hide the mistake.

    Raises:
        ArgumentError: it does not decode; it names where (``settings.hfw``) and,
            for an unknown key, where that key belongs or the closest one there is.
    """
    try:
        return decode(hint, value, strict=True)
    except ValidationError as error:
        raise _argument_error(name, hint, value, error) from None
    except PydanticSchemaGenerationError:
        return value  # a type with no codec form (an in-process stop event)


def decode_kwargs(func: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """A command's arguments as its signature declares them (`decode_argument`): a
    ``Point`` argument sent as a dict arrives as a ``Point``. Arguments without a
    usable hint pass as they came."""
    hints = type_hints(func)
    return {
        name: decode_argument(name, hints[name], value) if name in hints else value
        for name, value in kwargs.items()
    }


def _argument_error(
    name: str, hint: Any, sent: Any, error: ValidationError
) -> ArgumentError:
    first = error.errors()[0]
    loc = tuple(first["loc"])
    if first["type"] in ("unexpected_keyword_argument", "extra_forbidden"):
        try:
            decoded = decode(hint, sent)
        except ValidationError:
            decoded = None
        path, holder = _walk(name, decoded, loc[:-1])
        key = str(loc[-1])
        path = f"{path}.{key}"
        known = _known_keys(holder)
        where = _home_of(key, holder, path.rsplit(".", 1)[0]) if known else None
        if where is None and known:
            close = difflib.get_close_matches(key, known, n=1)
            where = f"{path.rsplit('.', 1)[0]}.{close[0]}" if close else None
        message = f"{type(holder).__name__} has no '{key}'"
        message += "." if where is None else f"; did you mean {where}?"
        if known:
            message += f" It takes: {', '.join(known)}."
        return ArgumentError(path, ValueError(message))
    try:
        decoded = decode(hint, sent)
    except ValidationError:
        decoded = None
    path, _ = _walk(name, decoded, loc)
    message = first["msg"].removeprefix("Value error, ")
    if first["type"] != "value_error":
        message += f", not {first['input']!r}"
    return ArgumentError(path, ValueError(message))


def _walk(name: str, decoded: Any, loc: Tuple[Any, ...]) -> Tuple[str, Any]:
    """The path ``loc`` names, as it was sent (``stage.monitors[0].config``), and
    the structure at its end. A structure saved with its fields under a key
    (``{name, config}``) adds that key to the path."""
    path, node = name, decoded
    for key in loc:
        if isinstance(key, int):
            path += f"[{key}]"
            node = node[key] if isinstance(node, (list, tuple)) else None
            continue
        if isinstance(node, Tagged) and node.SAVED_UNDER is not None:
            path += f".{node.SAVED_UNDER}"
        path += f".{key}"
        node = getattr(node, str(key), None)
    if isinstance(node, Tagged) and node.SAVED_UNDER is not None:
        path += f".{node.SAVED_UNDER}"
    return path, node


def _known_keys(structure: Any) -> List[str]:
    """The keys ``structure`` reads: its dataclass's fields."""
    if not dataclasses.is_dataclass(structure) or isinstance(structure, type):
        return []
    return sorted(f.name for f in dataclasses.fields(structure) if f.init)


def _home_of(key: str, structure: Any, path: str, depth: int = 3) -> Optional[str]:
    """Where ``key`` belongs when it was sent one or more levels too high: the path
    of the nearest structure inside ``structure`` that has it, else None."""
    if depth == 0:
        return None
    children = []
    for name in _known_keys(structure):
        child = getattr(structure, name, None)
        if _known_keys(child):
            if key in _known_keys(child):
                return f"{path}.{name}.{key}"
            children.append((name, child))
    for name, child in children:
        home = _home_of(key, child, f"{path}.{name}", depth - 1)
        if home is not None:
            return home
    return None


# The generic forms `type_path` writes and `type_from_path` reads.
_GENERICS = {
    "List": typing.List,
    "Tuple": typing.Tuple,
    "Dict": typing.Dict,
    "Optional": typing.Optional,
    "Union": typing.Union,
    "Sequence": typing.Sequence,
}
_PLAIN = {"int": int, "float": float, "str": str, "bool": bool, "None": type(None)}
_PLAIN.update({"Any": Any, "dict": dict, "list": list, "tuple": tuple})
_PLAIN["numpy.ndarray"] = np.ndarray  # a command's array, sent as npy bytes


def type_path(hint: Any) -> str:
    """A type as text a client can read back (`type_from_path`): fibsem classes by
    their full path, ``fibsem.structures.FibsemStagePosition``, inside the
    generics they're declared with, ``List[fibsem.structures.GridSlot]``."""
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin is Union:
        rest = [arg for arg in args if arg is not type(None)]
        if len(rest) == 1 and len(args) == 2:
            return f"Optional[{type_path(rest[0])}]"
        return f"Union[{', '.join(type_path(arg) for arg in args)}]"
    if origin is not None:
        name = {list: "List", tuple: "Tuple", dict: "Dict"}.get(origin)
        if name is None and origin is collections.abc.Sequence:
            name = "Sequence"
        if name is None:
            raise ValueError(f"no path for {hint!r}")
        parts = ["..." if arg is Ellipsis else type_path(arg) for arg in args]
        return f"{name}[{', '.join(parts)}]"
    for name, plain in _PLAIN.items():
        if hint is plain:
            return name
    if isinstance(hint, type) and hint.__module__.startswith("fibsem."):
        return f"{hint.__module__}.{hint.__qualname__}"
    raise ValueError(f"no path for {hint!r}")


def type_from_path(path: str) -> Any:
    """The type `type_path` wrote. Only fibsem's own classes are imported.

    Raises:
        ValueError: ``path`` names anything else.
    """
    path = path.strip()
    if "[" in path:
        name, inner = path.split("[", 1)
        if name not in _GENERICS or not inner.endswith("]"):
            raise ValueError(f"not a type path: {path!r}")
        args = tuple(
            Ellipsis if part == "..." else type_from_path(part)
            for part in _split_args(inner[:-1])
        )
        return _GENERICS[name][args if len(args) > 1 else args[0]]
    if path in _PLAIN:
        return _PLAIN[path]
    module, _, qualname = path.rpartition(".")
    if not module.startswith("fibsem.") or not qualname.isidentifier():
        raise ValueError(f"not a fibsem type: {path!r}")
    return getattr(importlib.import_module(module), qualname)


def _split_args(text: str) -> List[str]:
    """``text``'s top-level comma-separated parts, generics kept whole."""
    parts, depth, start = [], 0, 0
    for index, char in enumerate(text):
        depth += {"[": 1, "]": -1}.get(char, 0)
        if char == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return [part.strip() for part in parts]
