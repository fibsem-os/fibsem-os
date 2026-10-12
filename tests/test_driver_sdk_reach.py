"""A driver's devices and services reach the SDK through its connection, not the microscope.

Each driver package builds one connection object at connect (``drivers/<x>/connection.py``)
and hands it to every device and service it builds. Inside their methods, nothing reads
``self.parent._<private>`` or ``self.parent.connection`` (or the same through a local
``microscope = self.parent``): that reach is what kept the SDK code on the microscope
class. The builders at module level are where the connection is handed over, so they
are not checked.

Drivers join the list as their SDK code moves (FIB-1316). ``ALLOWED`` names what is left
on the microscope for a later step, not the SDK: the shared image bookkeeping, which the
base acquire implementation takes over, and services writing a beam through the
microscope rather than their role.
"""

import ast
import importlib
import inspect

import pytest

DRIVERS = ["tescan"]

ALLOWED = {"_last_imaging_settings", "_set_additional_metadata", "_write_beam"}


def _reaches(tree: ast.AST):
    """``self.parent.<name>`` or ``microscope.<name>`` inside a class's methods, where
    ``<name>`` is private or ``connection``."""
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        for node in ast.walk(cls):
            if not isinstance(node, ast.Attribute):
                continue
            target = node.value
            through_parent = (
                isinstance(target, ast.Attribute)
                and target.attr == "parent"
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            )
            through_local = isinstance(target, ast.Name) and target.id == "microscope"
            if not (through_parent or through_local):
                continue
            name = node.attr
            if (name.startswith("_") or name == "connection") and name not in ALLOWED:
                yield f"{cls.name}:{node.lineno} {name}"


@pytest.mark.parametrize("driver", DRIVERS)
@pytest.mark.parametrize("part", ["devices", "services"])
def test_devices_and_services_do_not_reach_the_sdk_through_the_microscope(driver, part):
    module = importlib.import_module(f"fibsem.drivers.{driver}.{part}")
    found = list(_reaches(ast.parse(inspect.getsource(module))))
    assert found == [], (
        "Use the connection the device or service was handed:\n  " + "\n  ".join(found)
    )
