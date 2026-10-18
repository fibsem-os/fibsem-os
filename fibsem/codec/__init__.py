"""Structures to plain data and back: the one codec for files and for the wire.

``encode`` writes a structure as JSON-ready data (dicts, lists, strings, numbers) and
``decode`` reads it back as the type asked for, from field names and type hints alone.
The engine is pydantic, run over the stdlib dataclasses as they are: nothing here
asks a class to change its base or its fields.

Two readers, one difference. A file may carry keys this version doesn't know (one
written by a newer build, or a key retired since), so ``decode`` ignores them. On the
wire a key the structure doesn't have is a mistake by the caller, which a default in
its place would hide, so ``decode(..., strict=True)`` refuses it at any depth.

A missing key gets the field's default. Numbers sent as text (``"4.0"``, from an
agent's MCP client) read as numbers, and a whole float as an int.

What the generic codec can't know, a type or a mixin here says:

- `NamedEnum`: an enum written by name, and read by name or by value.
- `Tagged`: a base class whose subclasses are saved with a ``type`` key naming which.
- `Array`: a numpy array, written as nested lists.
- `Timestamp`: a time, read from any form ``fibsem.util.timestamps`` reads and
  written as ISO 8601 with its offset.

Files in older shapes are upgraded before they reach ``decode``, by the reader for
that kind of file; nothing in a class reads an old shape.
"""

from __future__ import annotations

import functools
from datetime import datetime
from typing import Annotated, Any, Type, TypeVar

import numpy as np
from pydantic import BeforeValidator, PlainSerializer, PlainValidator, TypeAdapter
from pydantic_core import core_schema

from fibsem.util.timestamps import to_datetime, to_iso

T = TypeVar("T")

# The validation context key a nested decode reads to stay strict.
_STRICT = "strict"


@functools.cache
def _adapter(type_: Any) -> TypeAdapter:
    return TypeAdapter(type_)


def _plain(value: Any) -> Any:
    """A value pydantic has no serializer for, as plain data: a numpy scalar where
    a field declares a bool or a float, an array where it declares a list."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot encode a {type(value).__name__}")


def encode(value: Any, type_: Any = None) -> Any:
    """``value`` as JSON-ready data. ``type_`` is what to write it as, when that
    isn't its own class (a pattern held where its base class is declared)."""
    return _adapter(type(value) if type_ is None else type_).dump_python(
        value, mode="json", fallback=_plain, warnings=False
    )


def decode(type_: Type[T], data: Any, *, strict: bool = False) -> T:
    """``data`` read as ``type_``.

    Unknown keys are ignored, as a file needs; ``strict`` refuses them at any depth,
    as the wire does.

    Raises:
        pydantic.ValidationError: a ``ValueError`` naming each field that didn't
            read, by its path (``image_settings.hfw``).
    """
    return _adapter(type_).validate_python(
        data,
        extra="forbid" if strict else None,
        context={_STRICT: strict},
    )


def _strict(info: Any) -> bool:
    context = getattr(info, "context", None)
    return bool(context and context.get(_STRICT))


class NamedEnum:
    """Mix into an enum to write it by name (``ELECTRON``) and read it by name or by
    value (``ELECTRON``, ``"electron"``, ``1``, ``"1"``)::

        class BeamType(NamedEnum, Enum):
            ELECTRON = 1
            ION = 2

    A name is matched exactly, then in any case. A value whose member is ``None``
    reads from null.
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        return core_schema.no_info_plain_validator_function(
            cls._codec_member,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda member: member.name, when_used="json"
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(cls, schema: Any, handler: Any) -> Any:
        return {"enum": [member.name for member in cls], "title": cls.__name__}

    @classmethod
    def _codec_member(cls, value: Any) -> Any:
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError:
            pass
        if isinstance(value, str):
            members = cls.__members__
            if value in members:
                return members[value]
            folded = {name.casefold(): member for name, member in members.items()}
            if value.casefold() in folded:
                return folded[value.casefold()]
            try:
                return cls(int(value))
            except ValueError:
                pass
        raise ValueError(
            f"{value!r} is not a {cls.__name__}; one of {', '.join(cls.__members__)}"
        )


class Tagged:
    """Mix into a base class whose subclasses are saved with a ``type`` key that
    names which one, so a field declared as the base reads back as the subclass::

        class FibsemPatternSettings(Tagged, ABC): ...

    The class that lists ``Tagged`` among its bases is the one that dispatches.
    A subclass declared by name reads and writes as a plain dataclass, without the
    key. ``type_name`` and ``subclass_named`` are the class name and a search of the
    subclasses; a base with a registry (plugins) overrides them.
    """

    TYPE_KEY = "type"

    @classmethod
    def type_name(cls) -> str:
        return cls.__name__

    @classmethod
    def subclass_named(cls, name: str) -> type:
        """The subclass whose ``type_name`` is ``name``.

        Raises:
            ValueError: no subclass has that name.
        """
        pending = list(cls.__subclasses__())
        while pending:
            sub = pending.pop()
            if sub.type_name() == name:
                return sub
            pending.extend(sub.__subclasses__())
        raise ValueError(f"no {cls.__name__} named {name!r}")

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        if source is not cls or Tagged not in cls.__bases__:
            return handler(source)

        def read(value: Any, info: Any) -> Any:
            if isinstance(value, cls):
                return value
            if not isinstance(value, dict) or cls.TYPE_KEY not in value:
                raise ValueError(f"a {cls.__name__} needs a {cls.TYPE_KEY!r} key")
            sub = cls.subclass_named(value[cls.TYPE_KEY])
            rest = {k: v for k, v in value.items() if k != cls.TYPE_KEY}
            return decode(sub, rest, strict=_strict(info))

        def write(value: Any) -> Any:
            return {cls.TYPE_KEY: type(value).type_name(), **encode(value)}

        return core_schema.with_info_plain_validator_function(
            read,
            serialization=core_schema.plain_serializer_function_ser_schema(
                write, when_used="json"
            ),
        )


def _array(value: Any) -> np.ndarray:
    return value if isinstance(value, np.ndarray) else np.asarray(value)


# A numpy array, written as nested lists. Its dtype is what numpy makes of the lists.
Array = Annotated[
    np.ndarray,
    PlainValidator(_array),
    PlainSerializer(lambda value: np.asarray(value).tolist(), when_used="json"),
]

# A time, read from a POSIX float, an ISO string with or without its offset, or the
# month/day/year form older files carry (`to_datetime`), and written as ISO 8601 with
# its offset. A naive time stays naive (CONTRIBUTING.md, "Times").
Timestamp = Annotated[
    datetime,
    BeforeValidator(to_datetime),
    PlainSerializer(to_iso, when_used="json"),
]
