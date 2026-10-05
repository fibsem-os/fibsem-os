"""How parameter values cross the network between a device server and a remote device.

JSON has no tuple, enum or dataclass. Both sides encode with ``to_wire`` and decode with
``from_wire`` against the parameter's declared type, so a remote read returns what the
local read returns: a tuple stays a tuple, a ``Point`` stays a ``Point``.
"""

from __future__ import annotations

import logging
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
