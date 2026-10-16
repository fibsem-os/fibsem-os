"""JSON schemas from what a value declares: its type and its ``Parameter``.

One description serves the forms and the agent. A dataclass field's ``Parameter``
(``fibsem.structures.field_specs``) and a device parameter carry the same SI unit,
SI limits, choices, doc and ``Display``; this module writes them as JSON Schema, so a
client with no copy of the classes (an agent) knows what a command's arguments are.

A dataclass becomes a ``$defs`` entry, referenced by name, with one property per
field. A schema leaves nothing out that the server would refuse, but is not a
contract it checks: values are decoded by their declared types
(``fibsem.devices.wire.decode_kwargs``), and the device's own checks refuse the rest.
Units and displays are carried as ``unit`` and ``display`` beside the standard
keywords, so a client can show and convert the value as the GUI does.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import typing
from enum import Enum
from typing import Any, Callable, Dict, Optional, Union

from fibsem.structures import Parameter, RangeLimit, display_to_dict, field_specs

Schema = Dict[str, Any]

_SCALARS = {bool: "boolean", int: "integer", float: "number", str: "string"}


def value_schema(hint: Any, defs: Dict[str, Schema]) -> Schema:
    """A type as JSON Schema: a scalar, an enum (by value), a list or tuple, a mapping,
    ``Optional`` (or null), or a dataclass (a ``$ref`` into ``defs``, which it fills).
    Anything else is ``{}``: any value."""
    if hint is None or hint is type(None):
        return {"type": "null"}
    if hint in _SCALARS:
        return {"type": _SCALARS[hint]}
    if isinstance(hint, type) and issubclass(hint, Enum):
        return {
            "enum": [_plain(member.value) for member in hint],
            "title": hint.__name__,
        }
    if isinstance(hint, type) and dataclasses.is_dataclass(hint):
        return dataclass_schema(hint, defs)
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin is Union:
        options = [value_schema(arg, defs) for arg in args]
        if {} in options:  # one option takes any value, so the union does
            return {}
        return options[0] if len(options) == 1 else {"anyOf": options}
    if origin is typing.Literal:
        return {"enum": [_plain(arg) for arg in args]}
    if origin in (list, tuple, set, frozenset) or _is_sequence(origin):
        if origin is tuple and args and args[-1] is not Ellipsis:
            return {
                "type": "array",
                "prefixItems": [value_schema(arg, defs) for arg in args],
                "minItems": len(args),
                "maxItems": len(args),
            }
        schema: Schema = {"type": "array"}
        if args:
            schema["items"] = value_schema(args[0], defs)
        return schema
    if origin is dict or _is_mapping(origin):
        schema = {"type": "object"}
        if len(args) == 2:
            schema["additionalProperties"] = value_schema(args[1], defs)
        return schema
    if isinstance(hint, type) and issubclass(hint, tuple) and hasattr(hint, "_fields"):
        hints = typing.get_type_hints(hint)
        return {
            "type": "array",
            "title": hint.__name__,
            "prefixItems": [value_schema(hints.get(f), defs) for f in hint._fields],
        }
    return {}


def parameter_schema(parameter: Parameter, defs: Dict[str, Schema]) -> Schema:
    """A ``Parameter`` as JSON Schema: its type's, with its doc, unit, SI limits and
    choices, and how to show it. A composite's limits are per field (a stage
    position), so they go on its properties."""
    schema = dict(value_schema(parameter.type, defs))
    if parameter.doc:
        schema["description"] = parameter.doc
    if parameter.unit:
        schema["unit"] = parameter.unit
    limits = parameter.limits
    if isinstance(limits, RangeLimit):
        if math.isfinite(limits.min):
            schema["minimum"] = limits.min
        if math.isfinite(limits.max):
            schema["maximum"] = limits.max
    if parameter.choices is not None:
        schema["enum"] = [_plain(choice) for choice in parameter.choices]
    if parameter.not_used:
        schema["not_used"] = parameter.not_used
    if parameter.device_parameter:
        # its live limits and choices are that device parameter's (list_devices)
        schema["device_parameter"] = parameter.device_parameter
    display = display_to_dict(parameter.display)
    if display:
        schema["display"] = display
    return schema


def dataclass_schema(cls: type, defs: Dict[str, Schema]) -> Schema:
    """A ``$ref`` to ``cls`` in ``defs``, adding it (and the dataclasses its fields
    hold) the first time: an object with one property per field, each from the
    field's ``Parameter``. A field with no default is required."""
    ref = {"$ref": f"#/$defs/{cls.__name__}"}
    if cls.__name__ in defs:
        return ref
    defs[cls.__name__] = {}  # claimed first, so a field of its own type refers back
    specs = field_specs(cls)
    properties: Dict[str, Schema] = {}
    required = []
    for f in dataclasses.fields(cls):
        if not f.init:
            continue
        properties[f.name] = parameter_schema(specs[f.name], defs)
        if (
            f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING
        ):
            required.append(f.name)
    schema: Schema = {"type": "object", "title": cls.__name__, "properties": properties}
    doc = inspect.getdoc(cls)
    if doc and not doc.startswith(f"{cls.__name__}("):  # not the generated one
        schema["description"] = doc
    if required:
        schema["required"] = required
    defs[cls.__name__] = schema
    return ref


def arguments_schema(func: Callable[..., Any]) -> Schema:
    """A command's (or query's) arguments as one JSON Schema object, with the
    dataclasses they hold under ``$defs``. ``self`` and ``*args``/``**kwargs`` are
    left out; an argument with no default is required."""
    from fibsem.devices.wire import type_hints

    hints = type_hints(func)
    defs: Dict[str, Schema] = {}
    properties: Dict[str, Schema] = {}
    required = []
    for name, p in inspect.signature(func).parameters.items():
        if name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        schema = value_schema(hints.get(name), defs) if name in hints else {}
        if p.default is p.empty:
            required.append(name)
        else:
            default = _default(p.default)
            if default is not _NO_DEFAULT:
                schema = {**schema, "default": default}
        properties[name] = schema
    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    if defs:
        schema["$defs"] = defs
    return schema


_NO_DEFAULT = object()


def _default(value: Any) -> Any:
    """A default as JSON, or ``_NO_DEFAULT`` for one that isn't plain data."""
    plain = _plain(value)
    if plain is None or isinstance(plain, (bool, int, float, str)):
        if isinstance(plain, float) and not math.isfinite(plain):
            return _NO_DEFAULT
        return plain
    return _NO_DEFAULT


def _plain(value: Any) -> Any:
    from fibsem.devices.wire import to_wire

    return to_wire(value)


def _is_sequence(origin: Optional[type]) -> bool:
    import collections.abc as abc

    return origin in (abc.Sequence, abc.MutableSequence, abc.Set, abc.Iterable)


def _is_mapping(origin: Optional[type]) -> bool:
    import collections.abc as abc

    return origin in (abc.Mapping, abc.MutableMapping)
