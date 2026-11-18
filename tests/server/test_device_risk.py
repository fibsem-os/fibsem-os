"""Risk labels: each parameter a driver writes, and each command, says what it
risks, next to its code; /microscope/devices and /microscope/services publish
them. No gate reads them yet.
"""

import ast
import importlib
import inspect
import pkgutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import fibsem
import fibsem.devices
import fibsem.services
from fibsem.ai.agent import blocklist
from fibsem.devices.core import (
    Device,
    DeviceParameter,
    _command_risk,
    _Controllable,
    command,
    query,
)
from fibsem.devices.risk import Risk
from fibsem.server import AuthConfig, build_server

TOKEN = "test-token"


def _base_classes():
    """The device and service classes fibsem declares, which every driver's
    subclass shares."""
    found = set()
    for package in (fibsem.devices, fibsem.services):
        for info in pkgutil.iter_modules(package.__path__):
            module = importlib.import_module(f"{package.__name__}.{info.name}")
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if cls.__module__ == module.__name__ and issubclass(cls, _Controllable):
                    found.add(cls)
    return sorted(found, key=lambda cls: cls.__qualname__)


def _written() -> set:
    """Every parameter some class in fibsem writes (``write_<name>``), read from
    the source so a vendor driver counts without its SDK installed."""
    names = set()
    for path in Path(fibsem.__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name.startswith(
                        "write_"
                    ):
                        names.add(item.name[len("write_") :])
    return names


def _commands(cls):
    return [
        name
        for name, attr in inspect.getmembers(cls, inspect.isfunction)
        if hasattr(attr, "__device_command__")
    ]


def test_every_command_is_labelled():
    unlabelled = [
        f"{cls.__name__}.{name}"
        for cls in _base_classes()
        for name in _commands(cls)
        if _command_risk(cls.__mro__, name) is None
    ]
    assert unlabelled == []


def test_every_parameter_a_driver_writes_is_labelled():
    written = _written()
    unlabelled = [
        f"{cls.__name__}.{name}"
        for cls in _base_classes()
        for name, spec in cls.declared_parameters().items()
        if name in written and spec.risk is None
    ]
    assert unlabelled == []


def test_what_the_blocklist_blocks_is_labelled_never():
    """The blocklist and the labels agree, so the gate can read either."""
    written = _written()
    disagree = []
    for cls, blocks in blocklist._BLOCKED:
        writes, commands = blocks.get("write", ()), blocks.get("command", ())
        params = cls.declared_parameters()
        if writes == blocklist.ALL:
            writes = [name for name in params if name in written]
        if commands == blocklist.ALL:
            commands = _commands(cls)
        disagree += [
            f"{cls.__name__}.{name}"
            for name in writes
            if params[name].risk is not Risk.NEVER
        ]
        disagree += [
            f"{cls.__name__}.{name}()"
            for name in commands
            if _command_risk(cls.__mro__, name) is not Risk.NEVER
        ]
    assert disagree == []


class _Lamp(Device):
    brightness = DeviceParameter(float, risk="optics")

    @command(risk="moves_hardware")
    def swing(self) -> None:
        pass

    @query
    def where(self) -> str:
        return "here"


def test_a_backend_keeps_its_class_labels():
    class VendorLamp(_Lamp):
        brightness = DeviceParameter(float)

        @command
        def swing(self) -> None:
            pass

    lamp = VendorLamp("lamp")
    assert VendorLamp.brightness.risk is Risk.OPTICS
    assert lamp.commands["swing"].risk is Risk.MOVES_HARDWARE
    assert lamp.queries["where"].risk is Risk.READ_ONLY


def test_a_backend_cant_change_a_label():
    with pytest.raises(TypeError, match="can't change a parameter's risk"):

        class _WeakerParameter(_Lamp):
            brightness = DeviceParameter(float, risk="local")

    with pytest.raises(TypeError, match="can't change a command's risk"):

        class _WeakerCommand(_Lamp):
            @command(risk="read_only")
            def swing(self) -> None:
                pass


def test_a_label_that_isnt_one_is_refused_where_it_is_declared():
    with pytest.raises(ValueError, match="not a risk label"):
        DeviceParameter(float, risk="harmless")
    with pytest.raises(ValueError, match="not a risk label"):
        command(risk="harmless")


@pytest.fixture(scope="module")
def client(microscope):
    app = build_server(microscope, auth=AuthConfig(token=TOKEN))
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers.update({"Authorization": f"Bearer {TOKEN}"})
        yield client


def test_devices_and_services_publish_their_labels(client):
    devices = client.get("/microscope/devices").json()
    ion = devices["ion"]
    assert ion["parameters"]["current"]["risk"] == "optics"
    assert ion["parameters"]["on"]["risk"] == "never"
    assert ion["commands"]["acquire"]["risk"] == "acquires"
    assert ion["commands"]["turn_on"]["risk"] == "high_impact"
    assert devices["stage"]["commands"]["move_relative"]["risk"] == "moves_hardware"
    assert devices["stage"]["queries"]["pose_table"]["risk"] == "read_only"

    services = client.get("/microscope/services").json()
    assert services["milling"]["commands"]["run_stage"]["risk"] == "alters_sample"
    assert services["milling"]["commands"]["stop"]["risk"] == "stops"
    assert services["milling"]["queries"]["estimate_time"]["risk"] == "read_only"
