"""JSON schemas from what a value declares: its type and its ``Parameter``.

One description serves the forms and the agent. A dataclass field's ``Parameter``
(``fibsem.structures.params.field_specs``, a class's ``.params``) and a device parameter carry the same SI unit,
SI limits, choices, doc and ``Display``; this module writes them as JSON Schema, so a
client with no copy of the classes (an agent) knows what a command's arguments are.

The schema is the codec's (``fibsem.codec``): what ``decode`` reads, with saved keys,
tagged classes and enums by name as the codec has them, and fields it never reads
left out. A dataclass used in more than one place is a ``$defs`` entry, referenced
by name. Each field adds
its ``Parameter``: units and displays are carried as ``unit`` and ``display`` beside
the standard keywords, so a client can show and convert the value as the GUI does.
A schema is not a contract the server checks: values are decoded by their declared
types (``fibsem.devices.wire.decode_kwargs``), and the device's own checks refuse
the rest.
"""

from __future__ import annotations

import inspect
import math
from typing import Any, Callable, Dict

from pydantic import PydanticSchemaGenerationError, TypeAdapter
from pydantic.json_schema import GenerateJsonSchema

from fibsem.structures.params import Parameter, RangeLimit, display_to_dict, field_specs

Schema = Dict[str, Any]

_REF = "#/$defs/{model}"


class _Described(GenerateJsonSchema):
    """The codec's schema, with each dataclass field's ``Parameter`` on its property
    and the class's own docstring as its description."""

    def dataclass_schema(self, schema: Any) -> Schema:
        written = super().dataclass_schema(schema)
        cls = schema["cls"]
        written.pop("description", None)
        doc = inspect.getdoc(cls)
        if doc and not doc.startswith(f"{cls.__name__}("):  # not the generated one
            written["description"] = doc
        specs = field_specs(cls)
        for name, prop in written.get("properties", {}).items():
            if name in specs:
                _describe(prop, specs[name])
        return written

    def field_title_should_be_set(self, schema: Any) -> bool:
        return False

    def field_is_present(self, field: Any) -> bool:
        # a field never written belongs to the running object: nothing to send
        return not field.get("serialization_exclude") and super().field_is_present(
            field
        )


def value_schema(hint: Any, defs: Dict[str, Schema]) -> Schema:
    """A type as JSON Schema, as ``fibsem.codec.decode`` reads it. A dataclass used
    in more than one place is a ``$ref`` into ``defs``, which it fills. A type with no codec
    form (an in-process stop event) is ``{}``: any value."""
    if hint is None:
        hint = type(None)
    try:
        schema = TypeAdapter(hint).json_schema(
            schema_generator=_Described, ref_template=_REF
        )
    except PydanticSchemaGenerationError:
        return {}
    defs.update(schema.pop("$defs", {}))
    return schema


def parameter_schema(parameter: Parameter, defs: Dict[str, Schema]) -> Schema:
    """A ``Parameter`` as JSON Schema: its type's, with its doc, unit, SI limits and
    choices, and how to show it. A composite's limits are per field (a stage
    position), so they go on its properties."""
    schema = value_schema(parameter.type, defs) if parameter.type is not None else {}
    _describe(schema, parameter)
    return schema


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


def _describe(schema: Schema, parameter: Parameter) -> None:
    """Add ``parameter``'s doc, unit, SI limits, choices and display to ``schema``."""
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
