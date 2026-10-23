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
- `NotSaved`: a field that belongs to the running object, never written.
- `Timestamp`: a time, read from any form ``fibsem.util.timestamps`` reads and
  written as ISO 8601 with its offset.
- `LowerBound`, `UpperBound`: a limit's end, unbounded written as null (JSON has
  no infinity) and null read back as unbounded.
- `saved`: a field saved under another key, read as something other than its
  default when its key is absent, or required to be present.

Each of these has a JSON schema, so ``TypeAdapter(cls).json_schema()`` describes
what ``encode`` writes (a route's response in ``/openapi.json``).

Files in older shapes are upgraded before they reach ``decode``, by the reader for
that kind of file; nothing in a class reads an old shape.
"""

from __future__ import annotations

import copy
import functools
import math
from datetime import datetime
from typing import (
    Annotated,
    Any,
    Callable,
    ClassVar,
    Mapping,
    Optional,
    Sequence,
    Type,
    TypeVar,
)

import numpy as np
from pydantic import (
    BeforeValidator,
    Field,
    PlainSerializer,
    PlainValidator,
    TypeAdapter,
)
from pydantic_core import core_schema

from fibsem.util.timestamps import to_aware, to_datetime, to_iso

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
    if type_ is None and isinstance(value, (np.generic, np.ndarray)):
        return _plain(value)
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
    subclasses; a base with a registry (plugins) overrides them, or ``read_tagged``
    to decide what a name it has no class for reads as.

    ``TYPE_KEY`` is the key that names the class. With ``SAVED_UNDER`` set, the
    fields are saved under that key beside it (``{name, config}``) rather than
    beside the name.
    """

    TYPE_KEY: ClassVar[str] = "type"
    SAVED_UNDER: ClassVar[Optional[str]] = None

    @classmethod
    def type_name(cls) -> str:
        return cls.__name__

    def tagged_name(self) -> str:
        """The name this value is saved under: its class's, unless the value says."""
        return type(self).type_name()

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
    def tagged_variants(cls) -> Mapping[str, type]:
        """Each name a value may be saved under, with the class it reads as: what
        the JSON schema lists. A base with a registry overrides it, as it does
        ``subclass_named``."""
        found, pending = {}, list(cls.__subclasses__())
        while pending:
            sub = pending.pop(0)
            pending.extend(sub.__subclasses__())
            if not getattr(sub, "__abstractmethods__", None):
                found[sub.type_name()] = sub
        return found

    @classmethod
    def read_tagged(cls, name: Optional[str], data: Any, strict: bool) -> Any:
        """The value saved as ``data`` under ``name`` (None when the key is absent)."""
        if name is None:
            raise ValueError(f"a {cls.__name__} needs a {cls.TYPE_KEY!r} key")
        return decode(cls.subclass_named(name), data, strict=strict)

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        if source is not cls or Tagged not in cls.__bases__:
            return handler(source)

        def read(value: Any, info: Any) -> Any:
            if isinstance(value, cls):
                return value
            if not isinstance(value, dict):
                raise ValueError(f"a {cls.__name__} is saved as a dict")
            strict = _strict(info)
            rest = {k: v for k, v in value.items() if k != cls.TYPE_KEY}
            if cls.SAVED_UNDER is not None:
                if strict and set(rest) - {cls.SAVED_UNDER}:
                    extra = ", ".join(sorted(set(rest) - {cls.SAVED_UNDER}))
                    raise ValueError(
                        f"a {cls.__name__} takes {cls.TYPE_KEY!r} and "
                        f"{cls.SAVED_UNDER!r}, not {extra}"
                    )
                rest = rest.get(cls.SAVED_UNDER) or {}
            return cls.read_tagged(value.get(cls.TYPE_KEY), rest, strict)

        def write(value: Any) -> Any:
            fields = encode(value)
            if cls.SAVED_UNDER is not None:
                return {cls.TYPE_KEY: value.tagged_name(), cls.SAVED_UNDER: fields}
            return {cls.TYPE_KEY: value.tagged_name(), **fields}

        return core_schema.with_info_plain_validator_function(
            read,
            serialization=core_schema.plain_serializer_function_ser_schema(
                write, when_used="json"
            ),
            ref=f"{cls.__module__}.{cls.__qualname__}:tagged",
        )

    @classmethod
    def __get_pydantic_json_schema__(cls, schema: Any, handler: Any) -> Any:
        """One variant per name in ``tagged_variants``, each with its ``type`` key."""
        if Tagged not in cls.__bases__:
            return handler(schema)
        variants = []
        for type_name, sub in cls.tagged_variants().items():
            body = handler(_adapter(sub).core_schema)
            name = {"const": type_name}
            if cls.SAVED_UNDER is not None:
                body = {
                    "type": "object",
                    "properties": {cls.TYPE_KEY: name, cls.SAVED_UNDER: body},
                    "required": [cls.TYPE_KEY],
                    "title": type_name,
                }
            else:
                body = handler.resolve_ref_schema(body)
                body = {
                    **body,
                    "properties": {cls.TYPE_KEY: name, **body.get("properties", {})},
                    "required": [cls.TYPE_KEY, *body.get("required", [])],
                }
            variants.append(body)
        return {"oneOf": variants, "title": cls.__name__}


def saved(
    *,
    keys: Optional[Mapping[str, str]] = None,
    reads: Optional[Mapping[str, str]] = None,
    absent: Optional[Mapping[str, Any]] = None,
    required: Sequence[str] = (),
) -> Callable[[type], type]:
    """Decorate a dataclass whose saved shape differs from its fields::

        @saved(keys={"image_settings": "image"})
        @dataclass
        class FibsemImageMetadata: ...

    ``keys``: a field written and read under another key, by field name.
    ``reads``: another key a field is also read from when its own is absent, as a
    configuration spells it; it is written under the field's name.
    ``absent``: what a field reads as when its key is absent, where that differs
    from the default a new instance gets (a missing date is unknown, not now).
    ``required``: fields whose key must be present, though a new instance has a
    default (a point read with no ``y`` is a mistake, not the origin).

    The rules hold wherever the class is decoded, on its own or inside another,
    and its JSON schema names the saved keys and, as read, the required ones.
    Old file shapes are not for this: those belong to the reader for the file.
    """
    keys = dict(keys or {})
    reads = dict(reads or {})
    absent = dict(absent or {})
    required = tuple(required)

    def before(data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        for name, key in keys.items():
            if key in data:
                data[name] = data.pop(key)
        for name, key in reads.items():
            if name not in data and key in data:
                data[name] = data.pop(key)
        for name, value in absent.items():
            if name not in data:
                data[name] = copy.deepcopy(value)
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"missing {', '.join(missing)}")
        return data

    def write(value: Any, handler: Any) -> Any:
        data = handler(value)
        for name, key in keys.items():
            if name in data:
                data[key] = data.pop(name)
        return data

    def json_schema(schema: Any, handler: Any) -> Any:
        found = handler(schema)
        written = handler.resolve_ref_schema(found)
        if "properties" in written:
            written["properties"] = {
                keys.get(name, name): prop
                for name, prop in written["properties"].items()
            }
        needed = [keys.get(name, name) for name in written.get("required", [])]
        if handler.mode == "validation":
            needed += [keys.get(n, n) for n in required if keys.get(n, n) not in needed]
        if needed:
            written["required"] = needed
        return found

    def apply(cls: type) -> type:
        previous = getattr(cls, "__get_pydantic_core_schema__", None)

        def schema(owner: type, source: Any, handler: Any) -> Any:
            inner = previous(source, handler) if previous else handler(source)
            outer = core_schema.no_info_before_validator_function(
                before, inner, metadata={"pydantic_js_functions": [json_schema]}
            )
            if keys:
                outer["serialization"] = (
                    core_schema.wrap_serializer_function_ser_schema(write, schema=inner)
                )
            return outer

        cls.__get_pydantic_core_schema__ = classmethod(schema)
        return cls

    return apply


def _array(value: Any) -> np.ndarray:
    return value if isinstance(value, np.ndarray) else np.asarray(value)


# A numpy array, written as nested lists. Its dtype is what numpy makes of the lists.
Array = Annotated[
    np.ndarray,
    PlainValidator(_array, json_schema_input_type=list),
    PlainSerializer(
        lambda value: np.asarray(value).tolist(), return_type=list, when_used="json"
    ),
]

# A time, read from a POSIX float, an ISO string with or without its offset, or the
# month/day/year form older files carry (`to_datetime`), and written as ISO 8601 with
# its offset. A naive time stays naive (CONTRIBUTING.md, "Times").
Timestamp = Annotated[
    datetime,
    BeforeValidator(to_datetime),
    PlainSerializer(to_iso, when_used="json"),
]

# A time that is an instant: as `Timestamp`, but a value with no zone reads as None
# (`to_aware`), since its instant is unknown.
AwareTimestamp = Annotated[
    datetime,
    BeforeValidator(to_aware),
    PlainSerializer(to_iso, when_used="json"),
]


def _finite_or_none(value: float) -> Optional[float]:
    return None if math.isinf(value) else value


# A limit's lower and upper end: an infinite one (a Tescan stage axis) is written
# as null, and null reads back as unbounded.
LowerBound = Annotated[
    float,
    BeforeValidator(
        lambda value: -math.inf if value is None else value,
        json_schema_input_type=Optional[float],
    ),
    PlainSerializer(_finite_or_none, return_type=Optional[float], when_used="json"),
]
UpperBound = Annotated[
    float,
    BeforeValidator(
        lambda value: math.inf if value is None else value,
        json_schema_input_type=Optional[float],
    ),
    PlainSerializer(_finite_or_none, return_type=Optional[float], when_used="json"),
]


class _NotRead:
    """Reads any value as None (on the wire, refuses it), without looking at the
    field's type, which may be one the codec can't read (an image)."""

    def __get_pydantic_core_schema__(self, source: Any, handler: Any) -> Any:
        def read(value: Any, info: Any) -> None:
            if _strict(info):
                raise ValueError("is not sent; it belongs to the running object")
            return None

        return core_schema.with_info_plain_validator_function(read)

    def __get_pydantic_json_schema__(self, schema: Any, handler: Any) -> Any:
        return {}


# A field that belongs to the running object (an image, a path for one run, what
# a pattern computed), not to its saved form: never written, and read as None (on
# the wire, refused). Only for a field whose default is None.
NotSaved = Field(exclude=True)
NotSaved.metadata.append(_NotRead())
