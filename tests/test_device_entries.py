"""Building devices from the backend's defaults and `hardware.devices` entries.

`hardware.devices` is an overlay: an entry switches a device the backend has off, gives
it another driver, or adds one. `resolve_device_entries` says what to build and with
which driver; `build_device_entries` builds it, failing connect only for a `required`
device it cannot build.
"""

import logging
import os

import pytest
import yaml

import fibsem.config as cfg
from fibsem import manufacturers, utils
from fibsem.devices.entries import (
    REMOTE_DRIVER,
    DeviceBuildError,
    ResolvedEntry,
    build_device_entries,
    resolve_device_entries,
)
from fibsem.drivers import registry
from fibsem.structures import BeamType, DeviceEntry

DEFAULTS = (
    DeviceEntry(name="stage", type="stage"),
    DeviceEntry(name="manipulator", type="manipulator"),
    DeviceEntry(name="chamber", type="chamber"),
)


def _configured(*entries):
    return {entry["name"]: DeviceEntry.from_dict(entry) for entry in entries}


def _resolve(*entries, manufacturer="Demo"):
    return resolve_device_entries(DEFAULTS, _configured(*entries), manufacturer)


# -- resolving entries ------------------------------------------------------------


def test_without_entries_the_backend_builds_its_defaults_with_its_own_driver():
    resolved = _resolve()

    assert [item.name for item in resolved] == ["stage", "manipulator", "chamber"]
    assert {item.driver for item in resolved} == {manufacturers.DEMO}


def test_a_disabled_entry_is_not_built():
    resolved = _resolve({"name": "manipulator", "enabled": False})

    assert [item.name for item in resolved] == ["stage", "chamber"]


def test_an_entry_with_a_new_name_is_added_after_the_defaults():
    resolved = _resolve(
        {"name": "chamber_b", "type": "chamber"}, {"name": "stage", "x": 1}
    )

    assert [item.name for item in resolved] == [
        "stage",
        "manipulator",
        "chamber",
        "chamber_b",
    ]
    assert resolved[0].entry.options == {"x": 1}


def test_a_disabled_entry_the_backend_does_not_have_is_not_built():
    resolved = _resolve({"name": "chamber_b", "type": "chamber", "enabled": False})

    assert "chamber_b" not in [item.name for item in resolved]


def test_the_driver_defaults_to_the_manufacturer_in_its_canonical_spelling():
    resolved = _resolve(manufacturer="thermo")

    assert {item.driver for item in resolved} == {manufacturers.THERMOFISHER}


def test_an_entry_names_its_own_driver():
    resolved = _resolve(
        {"name": "chamber", "driver": "Remote", "address": "10.0.0.2"},
        {"name": "manipulator", "driver": "tescan"},
    )
    drivers = {item.name: item.driver for item in resolved}

    assert drivers == {
        "stage": manufacturers.DEMO,
        "manipulator": manufacturers.TESCAN,
        "chamber": REMOTE_DRIVER,
    }
    assert resolved[2].entry.options == {"address": "10.0.0.2"}


def test_without_a_driver_or_a_manufacturer_resolving_fails():
    with pytest.raises(ValueError, match="no driver"):
        _resolve(manufacturer=None)


# -- building them ---------------------------------------------------------------


def _item(name, type_, required=None, driver="Test"):
    return ResolvedEntry(DeviceEntry(name=name, type=type_, required=required), driver)


class _Builder:
    def __init__(self, build):
        self.build = build

    def load(self):
        return self.build


@pytest.fixture
def builders(monkeypatch):
    """The "Test" driver's builders by type, as the registry would give them; any
    other driver is one nothing is registered as."""
    table = {}

    def device_builder(driver, type_):
        if driver != "Test":
            raise NotImplementedError(f"Manufacturer {driver} not supported.")
        build = table.get(type_)
        return None if build is None else _Builder(build)

    monkeypatch.setattr(registry, "device_builder", device_builder)
    return table


def test_each_device_is_built_by_its_drivers_builder_in_order(builders):
    calls = []

    def build(entry, context):
        calls.append((entry.name, list(context.built), context.microscope))
        context.shared["seen"] = context.shared.get("seen", 0) + 1
        return entry.name.upper()

    builders.update(stage=build, chamber=build)
    shared = {}
    built = build_device_entries(
        [_item("stage", "stage"), _item("chamber", "chamber")], "scope", shared=shared
    )

    assert built == {"stage": "STAGE", "chamber": "CHAMBER"}
    # A later builder sees the devices built before it, and the same scratch space.
    assert calls == [("stage", [], "scope"), ("chamber", ["stage"], "scope")]
    assert shared == {"seen": 2}


def test_a_device_without_a_builder_is_skipped_with_a_warning(builders, caplog):
    builders["chamber"] = lambda entry, context: "chamber"
    with caplog.at_level(logging.WARNING):
        built = build_device_entries(
            [_item("laser", "laser"), _item("chamber", "chamber", driver="remote")],
            None,
        )

    assert built == {}
    assert "'laser' was not built: driver 'Test' has no builder" in caplog.text
    assert "'chamber' was not built: driver 'remote' has no builder" in caplog.text


def test_a_required_device_without_a_builder_fails_connect(builders):
    with pytest.raises(DeviceBuildError, match="'laser' was not built"):
        build_device_entries([_item("laser", "laser", required=True)], None)


def test_a_builder_that_fails_skips_its_device_unless_it_is_required(builders, caplog):
    def broken(entry, context):
        raise RuntimeError("no answer")

    builders["chamber"] = broken
    with caplog.at_level(logging.WARNING):
        built = build_device_entries([_item("chamber", "chamber")], None)
    assert built == {}
    assert "building it failed: no answer" in caplog.text

    with pytest.raises(DeviceBuildError, match="no answer"):
        build_device_entries([_item("chamber", "chamber", required=True)], None)


# -- on the Demo -------------------------------------------------------------------


def _demo_with(tmp_path, *entries):
    with open(cfg.DEFAULT_CONFIGURATION_PATH) as f:
        config = yaml.safe_load(f)
    config["info"]["manufacturer"] = "Demo"
    config["hardware"]["devices"].extend(entries)
    path = tmp_path / "configuration.yaml"
    path.write_text(yaml.safe_dump(config))
    microscope, _ = utils.setup_session(setup_logging=False, config_path=str(path))
    return microscope


def _arctis_with(tmp_path, *entries):
    """The shipped Arctis simulator, with *entries* replacing its own of one name."""
    with open(os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")) as f:
        config = yaml.safe_load(f)
    names = {entry["name"] for entry in entries}
    devices = [e for e in config["hardware"]["devices"] if e["name"] not in names]
    config["hardware"]["devices"] = devices + list(entries)
    path = tmp_path / "configuration.yaml"
    path.write_text(yaml.safe_dump(config))
    microscope, _ = utils.setup_session(setup_logging=False, config_path=str(path))
    return microscope


def test_the_arctis_simulator_builds_its_sample_loader_from_its_entry(tmp_path):
    microscope = _arctis_with(tmp_path)

    device = microscope.devices["sample_loader"]
    assert device.capacity.get_value() == 12
    assert [
        s.number for s in device.magazine.get_value().slots if s.state.value != "empty"
    ] == [1, 2, 3]
    assert microscope._stage.loader.device is device


def test_a_compustage_with_its_sample_loader_switched_off_has_none(tmp_path):
    microscope = _arctis_with(tmp_path, {"name": "sample_loader", "enabled": False})

    assert "sample_loader" not in microscope.devices
    assert microscope._stage.loader is None  # grids are exchanged by hand
    assert microscope._stage.holder.slots  # the working slot is still there


def test_the_demo_builds_every_device_it_has(tmp_path):
    microscope = _demo_with(tmp_path)

    assert {"electron", "ion", "stage", "chamber", "manipulator"} <= set(
        microscope.devices
    )


def test_the_demo_leaves_out_a_disabled_manipulator(tmp_path):
    microscope = _demo_with(tmp_path, {"name": "manipulator", "enabled": False})

    assert "manipulator" not in microscope.devices
    assert microscope.manipulator_device is None


def test_the_demo_builds_an_added_device_under_its_own_name(tmp_path):
    microscope = _demo_with(tmp_path, {"name": "chamber_b", "type": "chamber"})

    assert microscope.devices["chamber_b"].name == "chamber_b"
    assert microscope.chamber_device is microscope.devices["chamber"]


def test_the_demo_fails_connect_for_a_required_device_it_cannot_build(tmp_path):
    with pytest.raises(DeviceBuildError, match="'laser'"):
        _demo_with(tmp_path, {"name": "laser", "type": "laser", "required": True})


def test_the_demo_goes_on_without_a_device_it_cannot_build(tmp_path):
    microscope = _demo_with(tmp_path, {"name": "laser", "type": "laser"})

    assert "laser" not in microscope.devices
    assert set(microscope.beams) == {BeamType.ELECTRON, BeamType.ION}
