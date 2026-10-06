"""Building devices from the backend's defaults and `hardware.devices` entries.

`hardware.devices` is an overlay: an entry switches a device the backend has off, gives
it another driver, or adds one. `resolve_device_entries` says what to build and with
which driver; `build_device_entries` builds it, failing connect only for a `required`
device it cannot build.
"""

import logging

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
from fibsem.microscopes import registry
from fibsem.structures import BeamType, DeviceEntry

DEFAULTS = (
    DeviceEntry(name="stage", type="stage"),
    DeviceEntry(name="manipulator", type="manipulator"),
    DeviceEntry(name="gis", type="gis"),
)


def _configured(*entries):
    return {entry["name"]: DeviceEntry.from_dict(entry) for entry in entries}


def _resolve(*entries, manufacturer="Demo"):
    return resolve_device_entries(DEFAULTS, _configured(*entries), manufacturer)


# -- resolving entries ------------------------------------------------------------


def test_without_entries_the_backend_builds_its_defaults_with_its_own_driver():
    resolved = _resolve()

    assert [item.name for item in resolved] == ["stage", "manipulator", "gis"]
    assert {item.driver for item in resolved} == {manufacturers.DEMO}


def test_a_disabled_entry_is_not_built():
    resolved = _resolve({"name": "manipulator", "enabled": False})

    assert [item.name for item in resolved] == ["stage", "gis"]


def test_an_entry_with_a_new_name_is_added_after_the_defaults():
    resolved = _resolve({"name": "gis_pt", "type": "gis"}, {"name": "stage", "x": 1})

    assert [item.name for item in resolved] == ["stage", "manipulator", "gis", "gis_pt"]
    assert resolved[0].entry.options == {"x": 1}


def test_a_disabled_entry_the_backend_does_not_have_is_not_built():
    resolved = _resolve({"name": "gis_pt", "type": "gis", "enabled": False})

    assert "gis_pt" not in [item.name for item in resolved]


def test_the_driver_defaults_to_the_manufacturer_in_its_canonical_spelling():
    resolved = _resolve(manufacturer="thermo")

    assert {item.driver for item in resolved} == {manufacturers.THERMOFISHER}


def test_an_entry_names_its_own_driver():
    resolved = _resolve(
        {"name": "gis", "driver": "Remote", "address": "10.0.0.2"},
        {"name": "manipulator", "driver": "tescan"},
    )
    drivers = {item.name: item.driver for item in resolved}

    assert drivers == {
        "stage": manufacturers.DEMO,
        "manipulator": manufacturers.TESCAN,
        "gis": REMOTE_DRIVER,
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

    builders.update(stage=build, gis=build)
    shared = {}
    built = build_device_entries(
        [_item("stage", "stage"), _item("gis", "gis")], "scope", shared=shared
    )

    assert built == {"stage": "STAGE", "gis": "GIS"}
    # A later builder sees the devices built before it, and the same scratch space.
    assert calls == [("stage", [], "scope"), ("gis", ["stage"], "scope")]
    assert shared == {"seen": 2}


def test_a_device_without_a_builder_is_skipped_with_a_warning(builders, caplog):
    builders["gis"] = lambda entry, context: "gis"
    with caplog.at_level(logging.WARNING):
        built = build_device_entries(
            [_item("laser", "laser"), _item("gis", "gis", driver="remote")], None
        )

    assert built == {}
    assert "'laser' was not built: driver 'Test' has no builder" in caplog.text
    assert "'gis' was not built: driver 'remote' has no builder" in caplog.text


def test_a_required_device_without_a_builder_fails_connect(builders):
    with pytest.raises(DeviceBuildError, match="'laser' was not built"):
        build_device_entries([_item("laser", "laser", required=True)], None)


def test_a_builder_that_fails_skips_its_device_unless_it_is_required(builders, caplog):
    def broken(entry, context):
        raise RuntimeError("no answer")

    builders["gis"] = broken
    with caplog.at_level(logging.WARNING):
        built = build_device_entries([_item("gis", "gis")], None)
    assert built == {}
    assert "building it failed: no answer" in caplog.text

    with pytest.raises(DeviceBuildError, match="no answer"):
        build_device_entries([_item("gis", "gis", required=True)], None)


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


def test_the_demo_builds_every_device_it_has(tmp_path):
    microscope = _demo_with(tmp_path)

    assert {"electron", "ion", "stage", "chamber", "manipulator", "gis"} <= set(
        microscope.devices
    )
    assert microscope.gis_device is microscope.devices["gis"]


def test_the_demo_leaves_out_a_disabled_manipulator_and_gis(tmp_path):
    microscope = _demo_with(
        tmp_path,
        {"name": "manipulator", "enabled": False},
        {"name": "gis", "enabled": False},
    )

    assert "manipulator" not in microscope.devices
    assert microscope.manipulator_device is None
    assert microscope.gis_device is None
    assert dict(microscope.gis_devices) == {}


def test_the_demo_builds_a_second_gis_under_its_own_name(tmp_path):
    microscope = _demo_with(tmp_path, {"name": "gis_pt", "type": "gis"})

    assert set(microscope.gis_devices) == {"gis", "gis_pt"}
    assert microscope.devices["gis_pt"].name == "gis_pt"
    assert microscope.gis_device is microscope.devices["gis"]


def test_the_demo_fails_connect_for_a_required_device_it_cannot_build(tmp_path):
    with pytest.raises(DeviceBuildError, match="'laser'"):
        _demo_with(tmp_path, {"name": "laser", "type": "laser", "required": True})


def test_the_demo_goes_on_without_a_device_it_cannot_build(tmp_path):
    microscope = _demo_with(tmp_path, {"name": "laser", "type": "laser"})

    assert "laser" not in microscope.devices
    assert set(microscope.beams) == {BeamType.ELECTRON, BeamType.ION}
