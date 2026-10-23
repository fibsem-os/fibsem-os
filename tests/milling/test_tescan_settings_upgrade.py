"""A recipe saved before the driver block keeps its Tescan settings.

Tescan's preset, spot size, rate, dwell time and spacing sat at a recipe's top
level. Every reader of a protocol, an experiment and an old pre-task protocol puts
them in ``driver.Tescan``.
"""

import copy
from pathlib import Path

import yaml

from fibsem.applications.autolamella.structures import AutoLamellaTaskProtocol
from fibsem.codec.upgrades import milling, protocol

PROTOCOLS = (
    Path(__file__).parents[2] / "fibsem" / "applications" / "autolamella" / "protocol"
)

FLAT = {
    "preset": "30 keV; 20 nA",
    "spot_size": 2.0e-6,
    "rate": 3.0e-10,
    "dwell_time": 2.0e-6,
    "spacing": 0.5,
}


def _recipe(**extra):
    return {"milling_current": 2e-9, "hfw": 80e-6, **FLAT, **extra}


def test_the_flat_keys_go_to_the_tescan_block():
    data = _recipe()
    upgraded = milling.settings(data)
    assert upgraded["driver"] == {"Tescan": FLAT}
    assert not set(FLAT) & set(upgraded)
    assert upgraded["milling_current"] == 2e-9
    assert data == _recipe(), "the step copies what it is given"


def test_a_key_already_in_the_block_wins():
    data = _recipe(driver={"Tescan": {"preset": "30 keV; 1 nA"}, "Other": {"x": 1}})
    driver = milling.settings(data)["driver"]
    assert driver["Tescan"] == {**FLAT, "preset": "30 keV; 1 nA"}
    assert driver["Other"] == {"x": 1}


def test_a_current_recipe_is_unchanged():
    data = {"milling_current": 2e-9, "driver": {"Tescan": {"rate": 1e-9}}}
    assert milling.settings(data) == data
    assert "driver" not in milling.settings({"milling_current": 2e-9})


def _shipped() -> dict:
    """The shipped task protocol, as this version saves it."""
    path = PROTOCOLS / "task-protocol.yaml"
    return AutoLamellaTaskProtocol.from_dict(yaml.safe_load(path.read_text())).to_dict()


def _flattened(config: dict) -> dict:
    """A shipped task protocol as it was saved before the driver block."""
    old = copy.deepcopy(config)
    for task in old["task_config"].values():
        for milling_task in (task.get("milling") or {}).values():
            for stage in milling_task["stages"]:
                recipe = stage["milling"]
                recipe.update((recipe.pop("driver", None) or {}).pop("Tescan", {}))
                recipe.update(FLAT)
    return old


def test_a_protocol_opens_with_its_tescan_settings():
    loaded = AutoLamellaTaskProtocol.from_dict(_flattened(_shipped()))
    stages = [
        stage
        for task in loaded.task_config.values()
        for milling_task in task.milling.values()
        for stage in milling_task.stages
    ]
    assert stages
    for stage in stages:
        tescan = stage.milling.driver["Tescan"]
        assert {key: getattr(tescan, key) for key in FLAT} == FLAT


def test_an_experiment_s_lamella_configs_are_upgraded():
    configs = _flattened(_shipped())["task_config"]
    upgraded = protocol.experiment({"positions": [{"task_config": configs}]})
    for task in upgraded["positions"][0]["task_config"].values():
        for milling_task in (task.get("milling") or {}).values():
            for stage in milling_task["stages"]:
                assert stage["milling"]["driver"]["Tescan"] == FLAT


def test_an_old_pre_task_protocol_keeps_its_preset():
    old = yaml.safe_load((PROTOCOLS / "legacy" / "protocol-waffle.yaml").read_text())
    loaded = AutoLamellaTaskProtocol.from_dict(old)
    presets = {
        stage.milling.driver["Tescan"].preset
        for task in loaded.task_config.values()
        for milling_task in task.milling.values()
        for stage in milling_task.stages
    }
    assert "30 keV; 20 nA" in presets
