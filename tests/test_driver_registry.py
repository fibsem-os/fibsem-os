"""The driver registry: which class connects to each manufacturer, on which port, and
the configuration values a new configuration for it starts from.

``setup_session`` used to pick the class and port in an ``if manufacturer == ...``
chain. These pin that the registry gives every built-in manufacturer the class and
port the chain did, and that ``setup_session`` connects through it.
"""

import pytest

from fibsem import config as cfg
from fibsem import manufacturers, utils
from fibsem.microscopes import registry
from fibsem.microscopes.registry import DriverEntry, get_driver, register_driver

# What the old chain in setup_session did, per manufacturer.
BUILT_IN = {
    manufacturers.THERMOFISHER: (
        "fibsem.microscopes.autoscript:ThermoMicroscope",
        7520,
    ),
    manufacturers.TESCAN: ("fibsem.microscopes.tescan:TescanMicroscope", 8300),
    manufacturers.ODEMIS: (
        "fibsem.microscopes.odemis_microscope:OdemisThermoMicroscope",
        None,
    ),
    manufacturers.DEMO: ("fibsem.microscopes.device_demo:DemoMicroscope", 7520),
}


# The column tilts config.DEFAULT_CONFIGURATION_VALUES listed before the registry.
COLUMN_TILTS = {
    manufacturers.THERMOFISHER: {"ion-column-tilt": 52, "electron-column-tilt": 0},
    manufacturers.TESCAN: {"ion-column-tilt": 55, "electron-column-tilt": 0},
    manufacturers.DEMO: {"ion-column-tilt": 52, "electron-column-tilt": 0},
}


@pytest.fixture
def restore_registry():
    """Put back the registry, and the milling-time model setup_session installs."""
    from fibsem.milling import base

    saved = dict(registry._DRIVERS)
    estimator = base._milling_time_estimator
    yield
    registry._DRIVERS.clear()
    registry._DRIVERS.update(saved)
    base.set_milling_time_estimator(estimator)


@pytest.mark.parametrize("manufacturer", list(BUILT_IN))
def test_built_in_drivers_keep_their_class_and_port(manufacturer):
    entry = get_driver(manufacturer)
    assert (entry.microscope_class, entry.port) == BUILT_IN[manufacturer]


def test_the_default_configuration_values_come_from_the_registry():
    assert cfg.DEFAULT_CONFIGURATION_VALUES == COLUMN_TILTS
    assert registry.default_configuration_values() == COLUMN_TILTS


def test_the_available_manufacturers_are_the_drivers_with_defaults():
    """Odemis has a driver but no defaults, as it had no column tilts before."""
    assert cfg.AVAILABLE_MANUFACTURERS == list(COLUMN_TILTS)


def test_a_registered_driver_brings_its_defaults(restore_registry):
    register_driver(
        DriverEntry("Zeiss", f"{__name__}:_Recorder", 1, config={"ion-column-tilt": 54})
    )
    assert registry.default_configuration_values()["Zeiss"] == {"ion-column-tilt": 54}


def test_only_the_built_in_drivers_are_registered():
    assert registry.registered_manufacturers() == list(BUILT_IN)


def test_listing_the_drivers_imports_no_driver():
    """Importing a driver module may import its vendor SDK."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import fibsem.microscopes.registry as r\n"
        "r.registered_manufacturers()\n"
        "loaded = [m for m in ('fibsem.microscopes.autoscript', "
        "'fibsem.microscopes.tescan', 'fibsem.microscopes.odemis_microscope', "
        "'fibsem.microscopes.device_demo') if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("spelling", ["Thermo", "thermo fisher", "ThermoFisher"])
def test_any_known_spelling_finds_the_driver(spelling):
    assert get_driver(spelling).manufacturer == manufacturers.THERMOFISHER


def test_the_demo_driver_loads_its_class():
    from fibsem.microscopes.device_demo import DemoMicroscope

    assert get_driver(manufacturers.DEMO).load() is DemoMicroscope


def test_an_unknown_manufacturer_is_refused_as_before():
    with pytest.raises(NotImplementedError, match="Manufacturer Zeiss not supported."):
        utils.setup_session(manufacturer="Zeiss", setup_logging=False)


def test_setup_session_connects_through_the_registry():
    from fibsem.microscopes.device_demo import DemoMicroscope

    microscope, _ = utils.setup_session(manufacturer="Demo", setup_logging=False)
    assert type(microscope) is DemoMicroscope


class _Recorder:
    """Stands in for a driver class, recording how setup_session connects it."""

    connected = []

    def __init__(self, system):
        self.system = system

    def connect_to_microscope(self, ip_address, port):
        _Recorder.connected.append((ip_address, port))

    @staticmethod
    def estimate_stage_milling_time(*args, **kwargs):
        return 0.0


@pytest.mark.parametrize("port,expected", [(1234, [("10.0.0.1", 1234)]), (None, [])])
def test_setup_session_connects_on_the_registered_port(
    restore_registry, port, expected
):
    _Recorder.connected = []
    register_driver(DriverEntry(manufacturers.DEMO, f"{__name__}:_Recorder", port))
    microscope, _ = utils.setup_session(
        manufacturer="Demo", ip_address="10.0.0.1", setup_logging=False
    )
    assert isinstance(microscope, _Recorder)
    assert _Recorder.connected == expected


def test_registering_under_an_alias_replaces_the_canonical_entry(restore_registry):
    register_driver(DriverEntry("tescan", f"{__name__}:_Recorder", 1))
    assert get_driver(manufacturers.TESCAN).microscope_class == f"{__name__}:_Recorder"
    assert registry.registered_manufacturers() == list(BUILT_IN)


def test_the_configurations_port_overrides_the_registered_one(
    restore_registry, tmp_path
):
    import yaml

    config = utils.load_yaml(cfg.DEFAULT_CONFIGURATION_PATH)
    config["info"]["port"] = 4321
    path = tmp_path / "configuration.yaml"
    path.write_text(yaml.safe_dump(config))

    _Recorder.connected = []
    register_driver(DriverEntry(manufacturers.DEMO, f"{__name__}:_Recorder", 1234))
    microscope, _ = utils.setup_session(
        config_path=path,
        manufacturer="Demo",
        ip_address="10.0.0.1",
        setup_logging=False,
    )
    assert _Recorder.connected == [("10.0.0.1", 4321)]
    assert microscope.system.info.port == 4321


def test_connect_microscope_builds_and_connects_the_registered_driver(
    restore_registry,
):
    _Recorder.connected = []
    register_driver(DriverEntry(manufacturers.DEMO, f"{__name__}:_Recorder", 1234))
    system = utils.load_microscope_configuration(None, None).system
    system.info.manufacturer = manufacturers.DEMO
    system.info.ip_address = "10.0.0.1"

    microscope = registry.connect_microscope(system)

    assert isinstance(microscope, _Recorder)
    assert microscope.system is system
    assert _Recorder.connected == [("10.0.0.1", 1234)]
