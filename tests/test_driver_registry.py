"""The driver registry: which class connects to each manufacturer, on which port, and
the configuration values a new configuration for it starts from.

``setup_session`` used to pick the class and port in an ``if manufacturer == ...``
chain. These pin that each built-in driver's ``DRIVER`` record gives the class and
port the chain did, and that ``setup_session`` connects through the registry.
"""

import pytest

from fibsem import config as cfg
from fibsem import utils
from fibsem.drivers import registry
from fibsem.drivers.registry import DriverEntry, get_driver, register_driver
from fibsem.structures.settings import BeamType
from fibsem.structures.system import DeviceEntry

# What the old chain in setup_session did, per manufacturer.
BUILT_IN = {
    "ThermoFisher": (
        "fibsem.drivers.autoscript.microscope:ThermoMicroscope",
        7520,
    ),
    "Tescan": ("fibsem.drivers.tescan.microscope:TescanMicroscope", 8300),
    "Odemis": (
        "fibsem.drivers.odemis.microscope:OdemisThermoMicroscope",
        None,
    ),
    "Demo": ("fibsem.drivers.demo.microscope:DemoMicroscope", 7520),
}


# The column tilts each built-in driver's setup gives a new configuration.
COLUMN_TILTS = {
    "ThermoFisher": {"ion": 52, "electron": 0},
    "Tescan": {"ion": 55, "electron": 0},
    "Demo": {"ion": 52, "electron": 0},
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
    assert (entry.microscope_class, entry.port) == BUILT_IN[manufacturer]


@pytest.mark.parametrize("manufacturer", list(BUILT_IN))
def test_each_built_in_record_is_its_driver_modules_own(manufacturer):
    """The record is the ``__init__`` of the class's driver package, and the class's
    module imports without its SDK (none of the vendor SDKs are installed here, odemis
    included)."""
    import importlib

    module_name = BUILT_IN[manufacturer][0].partition(":")[0]
    importlib.import_module(module_name)
    package = importlib.import_module(module_name.rpartition(".")[0])
    assert get_driver(manufacturer) is package.DRIVER


def test_each_built_in_setup_gives_its_column_tilts():
    """In the configuration file's layout, as ``hardware.devices`` entries."""
    setups = dict(registry.driver_setups())
    for manufacturer, tilts in COLUMN_TILTS.items():
        devices = {d["name"]: d["column_tilt"] for d in setups[manufacturer].devices}
        assert devices == tilts, manufacturer


def test_the_configuration_constants_import_no_driver():
    """config.py is imported by the driver modules, so it can't import them back."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import fibsem.config\n"
        "loaded = [m for m in ('fibsem.drivers.autoscript.microscope', "
        "'fibsem.drivers.tescan.microscope', 'fibsem.drivers.odemis.microscope', "
        "'fibsem.drivers.demo.microscope') if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


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
        "import fibsem.drivers.registry as r\n"
        "r.registered_manufacturers()\n"
        "loaded = [m for m in ('fibsem.drivers.autoscript.microscope', "
        "'fibsem.drivers.tescan.microscope', 'fibsem.drivers.odemis.microscope', "
        "'fibsem.drivers.demo.microscope') if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("spelling", ["Thermo", "thermo fisher", "ThermoFisher"])
def test_any_known_spelling_finds_the_driver(spelling):
    assert get_driver(spelling).manufacturer == "ThermoFisher"


def test_the_demo_driver_loads_its_class():
    from fibsem.drivers.demo.microscope import DemoMicroscope

    assert get_driver("Demo").load() is DemoMicroscope


def test_an_unknown_manufacturer_is_refused_as_before():
    with pytest.raises(NotImplementedError, match="Manufacturer Zeiss not supported."):
        utils.setup_session(manufacturer="Zeiss", setup_logging=False)


def test_setup_session_connects_through_the_registry():
    from fibsem.drivers.demo.microscope import DemoMicroscope

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


@pytest.mark.parametrize("port,expected", [(1234, [("10.0.0.1", 1234)]), (None, [])])
def test_setup_session_connects_on_the_registered_port(
    restore_registry, port, expected
):
    _Recorder.connected = []
    register_driver(
        DriverEntry(
            RECORDER,
            f"{__name__}:_Recorder",
            port=port,
        )
    )
    microscope, _ = utils.setup_session(
        manufacturer=RECORDER, ip_address="10.0.0.1", setup_logging=False
    )
    assert isinstance(microscope, _Recorder)
    assert _Recorder.connected == expected


@pytest.mark.parametrize("spelling", ["tescan", "Tescan"])
def test_a_runtime_driver_cannot_take_a_built_in_manufacturer(
    restore_registry, spelling
):
    """The order patterns, strategies and tasks have: built-ins come first."""
    register_driver(DriverEntry(spelling, f"{__name__}:_Recorder", port=1))
    assert get_driver("Tescan").microscope_class == (BUILT_IN["Tescan"][0])
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
    register_driver(DriverEntry(RECORDER, f"{__name__}:_Recorder", port=1234))
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
    register_driver(DriverEntry(RECORDER, f"{__name__}:_Recorder", port=1234))
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
    records = registry.DRIVER_PLUGINS.plugin_records()
    return {record.entry_point: record for record in records}


def test_a_plugin_driver_registers_the_record_it_returns(restore_registry, monkeypatch):
    entry = DriverEntry("Hitachi", f"{__name__}:_Recorder", port=5000)
    records = _read(monkeypatch, _EntryPoint("hitachi", lambda: entry))

    assert records["hitachi"].loaded
    assert records["hitachi"].name == "Hitachi" and records["hitachi"].obj is entry
    assert get_driver("Hitachi") is entry


def test_get_driver_reads_the_entry_points(restore_registry, monkeypatch):
    entry = DriverEntry("Hitachi", f"{__name__}:_Recorder")
    _install(monkeypatch, _EntryPoint("hitachi", lambda: entry))
    assert get_driver("Hitachi") is entry


def test_a_plugin_cannot_take_a_built_in_manufacturer(restore_registry, monkeypatch):
    """It loads, as a pattern shadowed by a built-in does, and is not used."""
    records = _read(
        monkeypatch,
        _EntryPoint("thermo", lambda: DriverEntry("Thermo", f"{__name__}:_Recorder")),
    )

    assert records["thermo"].loaded
    assert records["thermo"].name == "ThermoFisher"
    assert get_driver("ThermoFisher").microscope_class == (BUILT_IN["ThermoFisher"][0])


def test_a_runtime_driver_comes_before_a_plugin(restore_registry, monkeypatch):
    registered = DriverEntry("Hitachi", f"{__name__}:_Recorder", port=1)
    plugin = DriverEntry("Hitachi", f"{__name__}:_Recorder", port=2)
    _install(monkeypatch, _EntryPoint("hitachi", lambda: plugin))
    register_driver(registered)

    assert get_driver("Hitachi") is registered
    assert registry.registered_manufacturers().count("Hitachi") == 1


def test_a_later_plugin_takes_the_manufacturer(restore_registry, monkeypatch):
    first = DriverEntry("Hitachi", f"{__name__}:_Recorder", port=1)
    second = DriverEntry("Hitachi", f"{__name__}:_Recorder", port=2)
    records = _read(
        monkeypatch,
        _EntryPoint("first", lambda: first),
        _EntryPoint("second", lambda: second),
    )

    assert get_driver("Hitachi") is second
    assert records["first"].loaded and records["second"].loaded


def test_a_plugin_that_looks_up_a_driver_as_it_loads_does_not_recurse(
    restore_registry, monkeypatch
):
    entry = DriverEntry("Hitachi", f"{__name__}:_Recorder")

    def driver():
        with pytest.raises(NotImplementedError):
            get_driver("Hitachi")
        return entry

    records = _read(monkeypatch, _EntryPoint("hitachi", driver))
    assert records["hitachi"].loaded
    assert get_driver("Hitachi") is entry


@pytest.mark.parametrize(
    "target,error",
    [
        (ImportError("No module named 'hitachi_sdk'"), "ImportError: No module named"),
        (lambda: "fibsem_hitachi:HitachiMicroscope", "returned str, not a DriverEntry"),
        (DriverEntry("Hitachi", "x:Y"), "TypeError"),  # a record, not a function
    ],
)
def test_a_broken_plugin_is_recorded_and_registers_nothing(
    restore_registry, monkeypatch, target, error
):
    records = _read(monkeypatch, _EntryPoint("broken", target))

    assert error in records["broken"].error
    assert "Hitachi" not in registry.registered_manufacturers()
    with pytest.raises(NotImplementedError):
        get_driver("Hitachi")


def test_an_unset_port_is_saved_as_null():
    """Unset, and saved as null, is the driver's registered port."""
    from fibsem.structures.system import SystemInfo

    assert SystemInfo.from_dict({}).to_dict()["port"] is None
    assert SystemInfo.from_dict({"port": 4321}).to_dict()["port"] == 4321

    config = utils.load_yaml(cfg.DEFAULT_CONFIGURATION_PATH)
    config["info"]["port"] = 4321
    assert "info.port" not in utils.unrecognised_configuration_keys(config)


# ---------------------------------------------------------------------------
# Device builders
# ---------------------------------------------------------------------------

DEMO_DEVICE_TYPES = ("beam", "stage", "chamber", "manipulator")


def test_a_driver_offers_a_builder_by_device_type():
    for spelling in ("Demo", "demo"):
        builder = registry.device_builder(spelling, "stage")
        assert builder.build == "fibsem.drivers.demo.devices:build_demo_stage"
    assert registry.device_builder("Demo", "laser") is None
    with pytest.raises(NotImplementedError, match="Manufacturer acme not supported."):
        registry.device_builder("acme", "stage")


def test_a_registered_drivers_builders_are_found(restore_registry):
    builder = registry.DeviceBuilder(f"{__name__}:_build_laser")
    register_driver(
        DriverEntry("Acme", f"{__name__}:_Recorder", devices={"laser": builder})
    )

    assert registry.device_builder("Acme", "laser") is builder
    context = registry.BuildContext(microscope=None)
    laser = builder.load()(
        DeviceEntry.from_dict({"name": "laser", "type": "laser"}), context
    )
    assert laser == ("laser", context)


def _build_laser(entry, context):
    return entry.name, context


def test_the_demo_builders_build_each_device_from_its_entry():
    from fibsem.drivers.demo import devices as demo

    microscope, _ = utils.setup_session(manufacturer="Demo", setup_logging=False)
    context = registry.BuildContext(microscope=microscope)
    built = {}
    for name, device_type in [("electron", "beam"), ("ion", "beam")] + [
        (t, t) for t in DEMO_DEVICE_TYPES[1:]
    ]:
        entry = DeviceEntry.from_dict({"name": name, "type": device_type})
        build = registry.device_builder("Demo", device_type).load()
        built[name] = build(entry, context)

    assert isinstance(built["electron"], demo.DemoBeam)
    assert built["electron"].beam_type is BeamType.ELECTRON
    assert built["ion"].beam_type is BeamType.ION
    assert isinstance(built["stage"], demo.DemoStage)
    assert isinstance(built["chamber"], demo.DemoChamber)
    assert isinstance(built["manipulator"], demo.DemoManipulator)
    # One connect's devices start from the same parts, as bind_demo_* do.
    resources, start = context.shared["Demo"]
    assert built["electron"].resources is built["ion"].resources is resources
    # A device is named after its entry, so a second manipulator is told apart.
    manipulator = registry.device_builder("Demo", "manipulator").load()
    entry = DeviceEntry.from_dict({"name": "manipulator_2", "type": "manipulator"})
    assert manipulator(entry, context).name == "manipulator_2"
    with pytest.raises(ValueError, match="'electron' or 'ion'"):
        beam = registry.device_builder("Demo", "beam").load()
        beam(DeviceEntry(name="third", type="beam"), context)


# ---------------------------------------------------------------------------
# Drivers that only build devices
# ---------------------------------------------------------------------------

DEVICE_ONLY = DriverEntry(
    "oxford",
    devices={"manipulator": registry.DeviceBuilder(f"{__name__}:_build_laser")},
)


def test_a_plugin_can_build_devices_and_no_microscope(restore_registry, monkeypatch):
    _install(monkeypatch, _EntryPoint("oxford", lambda: DEVICE_ONLY))

    for spelling in ("oxford", "Oxford", " OXFORD "):
        builder = registry.device_builder(spelling, "manipulator")
        assert builder is DEVICE_ONLY.devices["manipulator"]
    assert not DEVICE_ONLY.builds_microscope


def test_a_device_only_driver_is_not_a_manufacturer(restore_registry, monkeypatch):
    _install(monkeypatch, _EntryPoint("oxford", lambda: DEVICE_ONLY))

    assert "oxford" not in registry.registered_manufacturers()
    system = utils.load_microscope_configuration(None, None).system
    system.info.manufacturer = "oxford"
    with pytest.raises(NotImplementedError, match="builds devices, not a microscope"):
        registry.connect_microscope(system)


def test_a_listing_shows_what_a_device_only_driver_builds(
    restore_registry, monkeypatch
):
    _install(monkeypatch, _EntryPoint("oxford", lambda: DEVICE_ONLY))
    assert registry.DRIVER_PLUGINS.describe(get_driver("oxford")) == (
        "devices: manipulator"
    )


def test_a_device_only_driver_builds_an_added_entry(restore_registry):
    """On the Demo, an entry naming the driver is built by it, next to the Demo's
    own devices."""
    from fibsem.devices.entries import build_device_entries, resolve_device_entries

    register_driver(DEVICE_ONLY)
    resolved = resolve_device_entries(
        [DeviceEntry(name="stage", type="stage")],
        {"needle": DeviceEntry(name="needle", type="manipulator", driver="Oxford")},
        "Demo",
    )
    built = build_device_entries(resolved[1:], microscope=None)
    assert built["needle"][0] == "needle"


# ---------------------------------------------------------------------------
# Names and their spellings (FIB-300): the registry's normalize_name, which
# configuration files, image headers and the command line all go through.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, name",
    [
        # ThermoFisher: the spellings configs, the driver and session setup use
        ("Thermo", "ThermoFisher"),
        ("ThermoFisher", "ThermoFisher"),
        ("thermo fisher", "ThermoFisher"),
        ("Thermo Fisher Scientific", "ThermoFisher"),
        ("THERMOFISHER", "ThermoFisher"),
        # Tescan: the config spelling and the live image-header spelling
        ("Tescan", "Tescan"),
        ("TESCAN", "Tescan"),
        ("  tescan  ", "Tescan"),
        ("DEMO", "Demo"),
        ("odemis", "Odemis"),
        ("Remote", "remote"),
    ],
)
def test_every_spelling_normalises_to_the_drivers_name(raw, name):
    assert registry.normalize_name(raw) == name


@pytest.mark.parametrize("raw", ["Hitachi", "", "  ", "Unknown", None, 42])
def test_anything_unknown_passes_through_unchanged(raw):
    """Lenient on purpose: reading an old or foreign file never fails here."""
    assert registry.normalize_name(raw) == raw


def test_each_built_in_names_itself():
    for name in BUILT_IN:
        assert registry.normalize_name(name) == name
        assert get_driver(name).manufacturer == name


def test_a_plugin_brings_its_own_spellings(restore_registry, monkeypatch):
    entry = DriverEntry("Hitachi", f"{__name__}:_Recorder", aliases=("hitachi hht",))
    _install(monkeypatch, _EntryPoint("hitachi", lambda: entry))

    assert registry.normalize_name("Hitachi HHT") == "Hitachi"
    assert registry.normalize_name("hitachi") == "Hitachi"
    assert get_driver("hitachi hht") is entry


def test_a_plugin_cannot_take_a_built_in_spelling(restore_registry, monkeypatch):
    """A built-in's alias wins over a plugin claiming the same spelling, in the usual
    built-in, registered, plugin order."""
    entry = DriverEntry("Thermo2", f"{__name__}:_Recorder", aliases=("thermo",))
    _install(monkeypatch, _EntryPoint("thermo2", lambda: entry))

    assert registry.normalize_name("thermo") == "ThermoFisher"
    assert registry.normalize_name("Thermo2") == "Thermo2"


def test_each_microscope_class_names_its_own_driver():
    """A driver's class names its builds and answers ``manufacturer`` with
    ``driver_name``, so it must be its package's ``DRIVER.manufacturer``."""
    for name in BUILT_IN:
        driver = get_driver(name)
        cls = driver.load()
        assert cls.driver_name == driver.manufacturer
        assert cls.manufacturer.fget(object.__new__(cls)) == name


def test_a_class_without_a_driver_name_does_not_claim_one():
    """The base used to answer ThermoFisher for every class that did not say."""
    from fibsem.microscope import FibsemMicroscope

    class Nameless(FibsemMicroscope):
        pass

    Nameless.__abstractmethods__ = frozenset()
    with pytest.raises(NotImplementedError, match="Nameless sets no driver_name"):
        FibsemMicroscope.manufacturer.fget(object.__new__(Nameless))


@pytest.mark.parametrize(
    "stored, expected",
    [
        ("TESCAN", "Tescan"),  # live image header
        ("Thermo", "ThermoFisher"),  # config file spelling
        ("Demo", "Demo"),
        ("Hitachi", "Hitachi"),  # unknown: kept verbatim, never dropped
    ],
)
def test_system_info_normalises_the_manufacturer_on_read(stored, expected):
    from fibsem.structures.system import SystemInfo

    info = SystemInfo.from_dict({"manufacturer": stored})
    assert info.manufacturer == expected
    assert SystemInfo.from_dict(info.to_dict()).manufacturer == expected


def test_system_info_without_a_manufacturer_reads_unknown():
    from fibsem.structures.system import SystemInfo

    assert SystemInfo.from_dict({}).manufacturer == "Unknown"


def test_setup_session_accepts_any_demo_spelling(tmp_path):
    microscope, settings = utils.setup_session(
        session_path=str(tmp_path), manufacturer="DEMO", setup_logging=False
    )
    assert type(microscope).__name__ == "DemoMicroscope"
    assert microscope.manufacturer == "Demo"
    assert settings.system.info.manufacturer == "Demo"
    microscope.disconnect()


def test_the_setup_wizard_writes_registry_names():
    """The guided setup writes config_value into configurations as
    ``info.manufacturer``, so another spelling would not find the driver."""
    from fibsem.guided_setup import manufacturers

    for m in manufacturers():
        assert registry.normalize_name(m.config_value) == m.config_value


def test_the_headless_servers_offer_a_plugin_driver(restore_registry, monkeypatch):
    from fibsem.server import headless

    _install(
        monkeypatch,
        _EntryPoint("hitachi", lambda: DriverEntry("Hitachi", f"{__name__}:_Recorder")),
    )
    args = headless.connection_parser().parse_args(["--manufacturer", "hitachi"])
    assert args.manufacturer == "Hitachi"
