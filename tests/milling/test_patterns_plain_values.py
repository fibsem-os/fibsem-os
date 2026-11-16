"""Every milling pattern's shapes hold plain Python and write as plain YAML, at its
defaults and when its numbers come in as numpy."""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Iterator, Type

import numpy as np
import pytest
import yaml

import fibsem.structures.image
import fibsem.structures.settings
import fibsem.structures.system
from fibsem.milling.patterning import get_patterns
from fibsem.milling.patterning.base import BasePattern
from fibsem.structures.settings import Point

PATTERNS = list(get_patterns().values())


def _numpy_scalars(value: Any, path: str = "") -> Iterator[str]:
    """Where in *value* a numpy scalar sits. An array (a bitmap) is data, not one."""
    if isinstance(value, np.generic):
        yield path or "<value>"
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _numpy_scalars(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            yield from _numpy_scalars(item, f"{path}[{i}]")
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from _numpy_scalars(
                getattr(value, field.name), f"{path}.{field.name}"
            )


def _with_numpy(pattern: BasePattern) -> BasePattern:
    """*pattern* with every float field and its point as numpy."""
    for field in dataclasses.fields(pattern):
        value = getattr(pattern, field.name)
        if isinstance(value, float):
            setattr(pattern, field.name, np.float64(value))
        elif isinstance(value, int) and not isinstance(value, bool):
            setattr(pattern, field.name, np.int64(value))
    pattern.point = Point(np.float64(1e-6), np.float64(-2e-6))
    return pattern


def _check(pattern: BasePattern) -> None:
    shapes = pattern.define()
    assert shapes, f"{pattern.name} defined no shapes"
    for i, shape in enumerate(shapes):
        data = shape.to_dict()
        assert list(_numpy_scalars(data)) == [], f"{pattern.name} shape {i}"
        yaml.safe_dump(data)
    yaml.safe_dump(pattern.to_dict())


@pytest.mark.parametrize("cls", PATTERNS, ids=lambda c: c.name)
def test_default_shapes_hold_plain_python(cls: Type[BasePattern]) -> None:
    pattern = cls()
    for i, shape in enumerate(pattern.define()):
        assert list(_numpy_scalars(shape)) == [], f"{cls.name} shape {i}"
    _check(cls())


@pytest.mark.parametrize("cls", PATTERNS, ids=lambda c: c.name)
def test_numpy_inputs_write_as_plain_yaml(cls: Type[BasePattern]) -> None:
    _check(_with_numpy(cls()))


def _structures() -> list:
    found = {}
    for module in (
        fibsem.structures.settings,
        fibsem.structures.system,
        fibsem.structures.image,
    ):
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if (
                dataclasses.is_dataclass(cls)
                and "to_dict" in dir(cls)
                and not inspect.isabstract(cls)
            ):
                found[cls.__qualname__] = cls
    return [found[name] for name in sorted(found)]


@pytest.mark.parametrize("cls", _structures(), ids=lambda c: c.__qualname__)
def test_structure_to_dict_writes_as_plain_yaml(cls: type) -> None:
    try:
        obj = cls()
    except TypeError:
        pytest.skip("needs arguments to construct")
    for field in dataclasses.fields(obj):
        value = getattr(obj, field.name)
        if isinstance(value, float):
            setattr(obj, field.name, np.float64(value))
        elif isinstance(value, int) and not isinstance(value, bool):
            setattr(obj, field.name, np.int64(value))
    data = obj.to_dict()
    assert list(_numpy_scalars(data)) == []
    yaml.safe_dump(data)
