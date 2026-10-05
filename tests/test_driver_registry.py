"""The driver registry: which class connects to each manufacturer, on which port, and
the configuration values a new configuration for it starts from.

``setup_session`` used to pick the class and port in an ``if manufacturer == ...``
chain. These pin that each built-in driver's ``DRIVER`` record gives the class and
port the chain did, and that ``setup_session`` connects through the registry.
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
    from fibsem.plugins import loader

    saved = dict(registry.DRIVER_PLUGINS.registered)
    plugins = loader._CACHE.get(registry.DRIVER_ENTRY_POINT_GROUP)
    estimator = base._milling_time_estimator
    yield
    registry.DRIVER_PLUGINS.registered.clear()
    registry.DRIVER_PLUGINS.registered.update(saved)
    loader._CACHE.pop(registry.DRIVER_ENTRY_POINT_GROUP, None)
    if plugins is not None:
        loader._CACHE[registry.DRIVER_ENTRY_POINT_GROUP] = plugins
    base.set_milling_time_estimator(estimator)


@pytest.mark.parametrize("manufacturer", list(BUILT_IN))
def test_built_in_drivers_keep_their_class_and_port(manufacturer):
    entry = get_driver(manufacturer)
    assert entry.manufacturer == manufacturer
    assert (entry.microscope_class, entry.config.get("port")) == BUILT_IN[manufacturer]


@pytest.mark.parametrize("manufacturer", list(BUILT_IN))
def test_each_built_in_record_is_its_driver_modules_own(manufacturer):
    """The record lives beside the class, and the module imports without its SDK
    (none of the vendor SDKs are installed here, odemis included)."""
    import importlib

    module = importlib.import_module(BUILT_IN[manufacturer][0].partition(":")[0])
    assert get_driver(manufacturer) is module.DRIVER


def _tilts(values):
    return {
        manufacturer: {k: v for k, v in config.items() if k.endswith("column-tilt")}
        for manufacturer, config in values.items()
    }


def test_the_default_configuration_values_come_from_the_registry():
    """The tilts are what the constant listed; each driver's port rides along. A
    plugin driver's values, if one is installed, come after the built-ins'."""
    for values in (
        cfg.DEFAULT_CONFIGURATION_VALUES,
        registry.default_configuration_values(),
    ):
        assert list(values)[: len(COLUMN_TILTS)] == list(COLUMN_TILTS)
        assert _tilts({m: values[m] for m in COLUMN_TILTS}) == COLUMN_TILTS
    assert cfg.DEFAULT_CONFIGURATION_VALUES[manufacturers.TESCAN]["port"] == 8300


def test_the_configuration_constants_import_no_driver():
    """config.py is imported by the driver modules, so it can't import them back."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import fibsem.config\n"
        "loaded = [m for m in ('fibsem.microscopes.autoscript', "
        "'fibsem.microscopes.tescan', 'fibsem.microscopes.odemis_microscope', "
        "'fibsem.microscopes.device_demo') if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_the_available_manufacturers_are_the_drivers_with_defaults():
    """Odemis has a driver but no defaults, as it had no column tilts before."""
    available = cfg.AVAILABLE_MANUFACTURERS
    assert available[: len(COLUMN_TILTS)] == list(COLUMN_TILTS)
    assert manufacturers.ODEMIS not in available


def test_a_registered_driver_brings_its_defaults(restore_registry):
    register_driver(
        DriverEntry("Zeiss", f"{__name__}:_Recorder", {"ion-column-tilt": 54})
    )
    assert registry.default_configuration_values()["Zeiss"] == {"ion-column-tilt": 54}


def test_the_built_in_drivers_are_registered_first():
    """In order, ahead of any plugin driver (CI installs one, tests/fixtures/plugin)."""
    registered = registry.registered_manufacturers()
    assert registered[: len(BUILT_IN)] == list(BUILT_IN)


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


RECORDER = "Recorder Microscopes"


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
    register_driver(
        DriverEntry(
            RECORDER,
            f"{__name__}:_Recorder",
            config={} if port is None else {"port": port},
        )
    )
    microscope, _ = utils.setup_session(
        manufacturer=RECORDER, ip_address="10.0.0.1", setup_logging=False
    )
    assert isinstance(microscope, _Recorder)
    assert _Recorder.connected == expected


@pytest.mark.parametrize("spelling", ["tescan", manufacturers.TESCAN])
def test_a_runtime_driver_cannot_take_a_built_in_manufacturer(
    restore_registry, spelling
):
    """The order patterns, strategies and tasks have: built-ins come first."""
    register_driver(DriverEntry(spelling, f"{__name__}:_Recorder", {"port": 1}))
    assert (
        get_driver(manufacturers.TESCAN).microscope_class
        == (BUILT_IN[manufacturers.TESCAN][0])
    )
    assert registry.registered_manufacturers()[: len(BUILT_IN)] == list(BUILT_IN)


def test_the_configurations_port_overrides_the_registered_one(
    restore_registry, tmp_path
):
    import yaml

    config = utils.load_yaml(cfg.DEFAULT_CONFIGURATION_PATH)
    config["info"]["port"] = 4321
    path = tmp_path / "configuration.yaml"
    path.write_text(yaml.safe_dump(config))

    _Recorder.connected = []
    register_driver(DriverEntry(RECORDER, f"{__name__}:_Recorder", {"port": 1234}))
    microscope, _ = utils.setup_session(
        config_path=path,
        manufacturer=RECORDER,
        ip_address="10.0.0.1",
        setup_logging=False,
    )
    assert _Recorder.connected == [("10.0.0.1", 4321)]
    assert microscope.system.info.port == 4321


def test_connect_microscope_builds_and_connects_the_registered_driver(
    restore_registry,
):
    _Recorder.connected = []
    register_driver(DriverEntry(RECORDER, f"{__name__}:_Recorder", {"port": 1234}))
    system = utils.load_microscope_configuration(None, None).system
    system.info.manufacturer = RECORDER
    system.info.ip_address = "10.0.0.1"

    microscope = registry.connect_microscope(system)

    assert isinstance(microscope, _Recorder)
    assert microscope.system is system
    assert _Recorder.connected == [("10.0.0.1", 1234)]


# ---------------------------------------------------------------------------
# The fibsem.drivers entry point group
# ---------------------------------------------------------------------------


class _EntryPoint:
    """Stands in for an installed entry point; ``load`` returns *target*."""

    dist = None

    def __init__(self, name, target):
        self.name = name
        self.value = f"some_plugin:{name}"
        self._target = target

    def load(self):
        if isinstance(self._target, Exception):
            raise self._target
        return self._target


def _install(monkeypatch, *entry_points):
    """Make *entry_points* the installed fibsem.drivers group, not yet read."""
    from fibsem.plugins import loader

    monkeypatch.setattr(loader, "_entry_points", lambda group: iter(entry_points))
    loader._CACHE.pop(registry.DRIVER_ENTRY_POINT_GROUP, None)


def _read(monkeypatch, *entry_points):
    """Read *entry_points* as if they were the installed fibsem.drivers group."""
    _install(monkeypatch, *entry_points)
    records = registry.get_driver_plugin_records()
    return {record.entry_point: record for record in records}


def test_a_plugin_driver_registers_the_record_it_returns(restore_registry, monkeypatch):
    entry = DriverEntry(
        "JEOL", f"{__name__}:_Recorder", {"port": 5000, "ion-column-tilt": 53}
    )
    records = _read(monkeypatch, _EntryPoint("jeol", lambda: entry))

    assert records["jeol"].loaded
    assert records["jeol"].name == "JEOL" and records["jeol"].obj is entry
    assert get_driver("JEOL") is entry
    assert registry.default_configuration_values()["JEOL"]["ion-column-tilt"] == 53


def test_get_driver_reads_the_entry_points(restore_registry, monkeypatch):
    entry = DriverEntry("JEOL", f"{__name__}:_Recorder")
    _install(monkeypatch, _EntryPoint("jeol", lambda: entry))
    assert get_driver("JEOL") is entry


def test_a_plugin_cannot_take_a_built_in_manufacturer(restore_registry, monkeypatch):
    """It loads, as a pattern shadowed by a built-in does, and is not used."""
    records = _read(
        monkeypatch,
        _EntryPoint("thermo", lambda: DriverEntry("Thermo", f"{__name__}:_Recorder")),
    )

    assert records["thermo"].loaded
    assert records["thermo"].name == manufacturers.THERMOFISHER
    assert (
        get_driver(manufacturers.THERMOFISHER).microscope_class
        == (BUILT_IN[manufacturers.THERMOFISHER][0])
    )


def test_a_runtime_driver_comes_before_a_plugin(restore_registry, monkeypatch):
    registered = DriverEntry("JEOL", f"{__name__}:_Recorder", {"port": 1})
    plugin = DriverEntry("JEOL", f"{__name__}:_Recorder", {"port": 2})
    _install(monkeypatch, _EntryPoint("jeol", lambda: plugin))
    register_driver(registered)

    assert get_driver("JEOL") is registered
    assert registry.registered_manufacturers().count("JEOL") == 1


def test_a_later_plugin_takes_the_manufacturer(restore_registry, monkeypatch):
    first = DriverEntry("JEOL", f"{__name__}:_Recorder", {"port": 1})
    second = DriverEntry("JEOL", f"{__name__}:_Recorder", {"port": 2})
    records = _read(
        monkeypatch,
        _EntryPoint("first", lambda: first),
        _EntryPoint("second", lambda: second),
    )

    assert get_driver("JEOL") is second
    assert records["first"].loaded and records["second"].loaded


def test_a_plugin_that_looks_up_a_driver_as_it_loads_does_not_recurse(
    restore_registry, monkeypatch
):
    entry = DriverEntry("JEOL", f"{__name__}:_Recorder")

    def driver():
        with pytest.raises(NotImplementedError):
            get_driver("JEOL")
        return entry

    records = _read(monkeypatch, _EntryPoint("jeol", driver))
    assert records["jeol"].loaded
    assert get_driver("JEOL") is entry


@pytest.mark.parametrize(
    "target,error",
    [
        (ImportError("No module named 'jeol_sdk'"), "ImportError: No module named"),
        (lambda: "fibsem_jeol:JeolMicroscope", "returned str, not a DriverEntry"),
        (DriverEntry("JEOL", "x:Y"), "TypeError"),  # a record, not a function
    ],
)
def test_a_broken_plugin_is_recorded_and_registers_nothing(
    restore_registry, monkeypatch, target, error
):
    records = _read(monkeypatch, _EntryPoint("broken", target))

    assert error in records["broken"].error
    assert "JEOL" not in registry.registered_manufacturers()
    with pytest.raises(NotImplementedError):
        get_driver("JEOL")
