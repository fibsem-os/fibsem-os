"""Every structure the codec reads has a JSON schema, and what it writes matches it:
the App Server's /openapi.json is built from these."""

import dataclasses
import importlib
import warnings

import jsonschema
import pytest
import yaml
from pydantic import TypeAdapter

from fibsem.codec import encode

MODULES = [
    "fibsem.structures.settings",
    "fibsem.structures.image",
    "fibsem.structures.params",
    "fibsem.structures.system",
    "fibsem.milling.base",
    "fibsem.workflows.item",
    "fibsem.workflows.site_task",
    "fibsem.workflows.task_config",
    "fibsem.applications.autolamella.structures",
    "fibsem.applications.volume.experiment",
    "fibsem.applications.volume.position",
    "fibsem.applications.volume.protocol",
]

# Field descriptions that hold functions: never encoded.
NOT_STRUCTURES = {"Display", "Parameter"}


def _structures():
    importlib.import_module("fibsem.applications.autolamella.workflows.tasks")
    for name in MODULES:
        module = importlib.import_module(name)
        for cls in vars(module).values():
            if (
                isinstance(cls, type)
                and dataclasses.is_dataclass(cls)
                and cls.__module__ == name
                and cls.__name__ not in NOT_STRUCTURES
            ):
                try:
                    adapter = TypeAdapter(cls)
                except Exception:
                    continue  # holds a type the codec can't read (an image)
                if adapter.pydantic_complete:
                    yield pytest.param(adapter, id=f"{name}.{cls.__name__}")


@pytest.mark.parametrize("mode", ["validation", "serialization"])
@pytest.mark.parametrize("adapter", list(_structures()))
def test_a_structure_has_a_json_schema(adapter, mode):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert adapter.json_schema(mode=mode)


@pytest.mark.parametrize("mode", ["validation", "serialization"])
def test_the_default_protocols_task_configs_match_their_schema(mode):
    from fibsem.applications.autolamella.experiment_setup import default_protocol_path
    from fibsem.applications.autolamella.structures import (
        AutoLamellaTaskConfig,
        AutoLamellaTaskProtocol,
    )

    with open(default_protocol_path()) as f:
        protocol = AutoLamellaTaskProtocol.from_dict(yaml.safe_load(f))
    schema = TypeAdapter(AutoLamellaTaskConfig).json_schema(mode=mode)
    for config in protocol.task_config.values():
        jsonschema.validate(encode(config, AutoLamellaTaskConfig), schema)


def test_a_lamella_matches_its_schema(tmp_path):
    from fibsem.applications.autolamella.structures import Lamella

    lamella = Lamella(petname="01-a", path=str(tmp_path / "01-a"), number=1)
    schema = TypeAdapter(Lamella).json_schema(mode="serialization")
    jsonschema.validate(encode(lamella), schema)
