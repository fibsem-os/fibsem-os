"""Which driver connects to which manufacturer's microscope (FIB-1123).

Each driver package describes itself with a ``DRIVER`` record in its ``__init__``: the
canonical manufacturer name it answers to and the other spellings it accepts, its
``FibsemMicroscope`` class, and its configuration (the port it connects on, for one).
:func:`normalize_name` turns any spelling into that name.
``utils.setup_session`` connects through ``connect_microscope`` instead of an
``if manufacturer == ...`` chain.

The built-in drivers (``fibsem.drivers.<driver>``) are listed here, by where their
record lives, and a record is imported only when that driver is asked for. The record
is alone in the package's ``__init__``, so reading it imports none of the driver's
modules; those must not need the vendor SDK either (each one tries the SDK and carries
on without it), but they are slow, so listing the drivers never imports one.

A driver outside fibsem registers through the ``fibsem.drivers`` entry point group,
naming a function that returns its ``DriverEntry``::

    [project.entry-points."fibsem.drivers"]
    jeol = "fibsem_jeol.registry:driver"

A driver that only builds devices (a manipulator on its own controller, configured as a
``hardware.devices`` entry next to a vendor's microscope) registers the same way, with
no ``microscope_class``; it is never offered as a manufacturer.

Built-ins in code, plugins in packages, as in fibsem's other plugin groups. A record
rather than the class, so the format survives the microscope classes becoming device
builders (FIB-1160): the entry is what changes, not the contract. The entry points
are read once, by ``fibsem.plugins.loader``, the first time a driver is looked up or
listed. When two drivers claim one manufacturer, the same order decides as for
patterns, strategies and tasks: built-in, then registered at runtime, then plugin.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
)

from fibsem.plugins.loader import PluginRegistry, PluginRejected

if TYPE_CHECKING:
    from fibsem.devices.core import Device
    from fibsem.drivers.setup import DriverSetup, SetupDevice
    from fibsem.microscope import FibsemMicroscope
    from fibsem.plugins.loader import PluginRecord
    from fibsem.structures import DeviceEntry, SystemSettings


@dataclass(frozen=True)
class DriverEntry:
    """One driver, as the connect code needs to know it."""

    manufacturer: str
    """The canonical manufacturer name, as ``SystemInfo.manufacturer`` carries it.
    For a driver that only builds devices, the name a device entry's ``driver:``
    gives it."""

    microscope_class: Optional[str] = None
    """The driver's ``FibsemMicroscope`` subclass, as ``"module:Class"``. ``None`` for
    a driver that only builds devices (a manipulator on its own controller, say): it
    is named on a ``hardware.devices`` entry, next to a vendor's microscope, and is
    never a manufacturer."""

    port: Optional[int] = None
    """The port ``connect_microscope`` connects on (a configuration's ``info.port``
    overrides it); without either, the driver is not connected by address at all
    (Odemis reaches its instrument through its own back end, so it is constructed and
    nothing more). The values a new configuration starts with, such as the column
    tilts, are the setup's (``DriverSetup.devices``)."""

    devices: Mapping[str, "DeviceBuilder"] = field(default_factory=dict)
    """How this driver builds a device, by the device entry's ``type`` (``beam``,
    ``stage``, ``chamber``, ...): what a ``hardware.devices`` entry naming this driver
    is built with. See :func:`device_builder`."""

    remote_devices: Mapping[str, str] = field(default_factory=dict)
    """The class a remote copy of this driver's device is, by entry ``type``, as
    ``"module:Class"``, imported when a ``driver: remote`` entry reaches such a
    device on a device server. The class mixes ``RemoteDevice`` into what the
    device declares, so its own parameters and commands cross the wire too::

        remote_devices={"scan_generator": "my_plugin.remote:RemoteMyScanner"}

    A type with none, or a driver not installed on the coordinator, gets the type's
    interface class from the remote driver (``RemoteScanner``), which has only the
    interface's parameters. See ``fibsem.drivers.remote.devices.remote_class``."""

    aliases: Tuple[str, ...] = ()
    """Other spellings of the name ("thermo", "thermo fisher scientific"), as
    configuration files, image headers and the command line carry them.
    :func:`normalize_name` maps each to :attr:`manufacturer`, ignoring case; the name
    itself matches in any case without being listed."""

    setup: Optional[str] = None
    """What the first-run guided setup offers for this driver, as
    ``"module:attribute"`` naming a ``fibsem.drivers.setup.DriverSetup``, imported
    when the wizard lists the drivers (:func:`driver_setups`). ``None`` for a driver
    the wizard does not offer.

    A driver that only builds devices names a ``SetupDevice``, or a tuple of them,
    instead: the devices the wizard offers (:func:`device_setups`), whose keys are
    also what building such an entry checks (:func:`device_setup`)."""

    milling_settings: Optional[str] = None
    """The driver's own milling settings class, as ``"module:Class"`` naming a
    ``fibsem.structures.DriverMillingSettings``: the settings no other driver means
    the same by (Tescan's preset and rate). A milling recipe keeps them in its
    ``driver`` block under this driver's name, so the module must import nothing
    heavier than ``fibsem.structures``: every PC reads the block. ``None`` for a
    driver that mills with the shared fields only."""

    @property
    def builds_microscope(self) -> bool:
        """Whether the driver connects a microscope, not only devices."""
        return self.microscope_class is not None

    def load(self) -> Type["FibsemMicroscope"]:
        """Import and return the driver's class."""
        if self.microscope_class is None:
            raise NotImplementedError(
                f"Driver {self.manufacturer} builds devices, not a microscope: name it "
                "as the driver of a hardware.devices entry, not as the manufacturer."
            )
        return _import(self.microscope_class)


@dataclass(frozen=True)
class DeviceBuilder:
    """How a driver builds one type of device from its configuration entry."""

    build: str
    """The build function, as ``"module:function"``, imported when first used. It is
    called as ``build(entry, context)`` with the ``DeviceEntry`` and a
    :class:`BuildContext`, and returns the connected device, or raises."""

    def load(self) -> "BuildFn":
        """Import and return the build function."""
        return _import(self.build)


@dataclass
class BuildContext:
    """What a build function gets besides its entry, for one connect."""

    microscope: "FibsemMicroscope"
    """The microscope being connected. A vendor driver's connection lives on it."""

    built: Mapping[str, "Device"] = field(default_factory=dict)
    """The devices built so far in this connect, by entry name."""

    shared: Dict[str, Any] = field(default_factory=dict)
    """Scratch space for this connect's builders, so the entries of one driver can
    share something, such as one connection per remote address. Key it by driver."""

    cleanups: List[Callable[[], None]] = field(default_factory=list)
    """What `on_disconnect` was given, in order."""

    def on_disconnect(self, fn: Callable[[], None]) -> None:
        """Call *fn* when the microscope disconnects, after every device's own
        ``disconnect``. For what the entries of one driver share, such as one
        connection per address in `shared`: no one device owns it to close it.
        Called in reverse order of registration; one that raises is logged and the
        rest still run."""
        self.cleanups.append(fn)


BuildFn = Callable[["DeviceEntry", BuildContext], "Device"]
"""A device builder's function: ``build(entry, context) -> Device``."""


DRIVER_ENTRY_POINT_GROUP = "fibsem.drivers"


def _import(target: str) -> Any:
    """The object a ``"module:attribute"`` string names."""
    module_name, _, attribute = target.partition(":")
    return getattr(importlib.import_module(module_name), attribute)


# The built-in drivers, by where each one's DRIVER record lives.
_BUILT_IN: Dict[str, str] = {
    "ThermoFisher": "fibsem.drivers.autoscript:DRIVER",
    "Tescan": "fibsem.drivers.tescan:DRIVER",
    "Odemis": "fibsem.drivers.odemis:DRIVER",
    "Demo": "fibsem.drivers.demo:DRIVER",
}

# The built-in drivers that build devices and are no microscope, by name, with where
# their builders by type live. ``remote`` is a device on its own PC, reached through
# its device server; it is named only on a device entry, never as a manufacturer.
REMOTE_DRIVER = "remote"
_DEVICE_DRIVERS: Dict[str, str] = {
    REMOTE_DRIVER: "fibsem.drivers.remote.devices:DEVICE_BUILDERS",
}


class _BuiltInDrivers(Mapping[str, DriverEntry]):
    """The built-in records by manufacturer, each imported when it is looked up."""

    def __getitem__(self, manufacturer: str) -> DriverEntry:
        return _import(_BUILT_IN[manufacturer])

    def __contains__(self, manufacturer: object) -> bool:
        return manufacturer in _BUILT_IN

    def __iter__(self) -> Iterator[str]:
        return iter(_BUILT_IN)

    def __len__(self) -> int:
        return len(_BUILT_IN)


class _DriverRegistry(PluginRegistry[DriverEntry]):
    def describe_builtin(self, name: str) -> str:
        # Where the record lives, so a listing imports no driver.
        return _BUILT_IN[name]


def _resolve_driver(function: Any) -> Tuple[str, DriverEntry]:
    """A ``fibsem.drivers`` entry point names a function returning the record."""
    entry = function()
    if not isinstance(entry, DriverEntry):
        raise PluginRejected(f"returned {type(entry).__name__}, not a DriverEntry")
    entry = _canonical(entry)
    return entry.manufacturer, entry


DRIVER_PLUGINS: PluginRegistry[DriverEntry] = _DriverRegistry(
    group=DRIVER_ENTRY_POINT_GROUP,
    kind="driver",
    resolve=_resolve_driver,
    builtins=_BuiltInDrivers(),
    describe=lambda entry: entry.microscope_class or _describe_devices(entry),
)


def _describe_devices(entry: DriverEntry) -> str:
    """What a listing shows for a driver that only builds devices."""
    return "devices: " + (", ".join(entry.devices) or "none")


def register_driver(entry: DriverEntry) -> None:
    """Make *entry* the driver for its manufacturer, unless a built-in has it.

    Registering a manufacturer again replaces its registered driver. The name is
    normalised (:func:`normalize_name`), so an entry registered as "jeol" replaces
    one registered as "JEOL".
    """
    entry = _canonical(entry)
    DRIVER_PLUGINS.register(entry.manufacturer, entry)


def _canonical(entry: DriverEntry) -> DriverEntry:
    manufacturer = normalize_name(entry.manufacturer)
    if manufacturer != entry.manufacturer:
        entry = replace(entry, manufacturer=manufacturer)
    return entry


def normalize_name(raw: Optional[str]) -> Optional[str]:
    """The registry name of the driver *raw* names, in any spelling it accepts.

    Matches a driver's name in any case, or one of its ``aliases``, ignoring
    surrounding spaces, looking at the built-ins, then those registered at runtime,
    then the plugins. ``remote`` stays ``remote``. Anything else (an unknown name,
    an empty string, ``None``, a value that is not a string) passes through
    unchanged, so reading an old file can never fail on its manufacturer.
    """
    if not isinstance(raw, str):
        return raw
    wanted = raw.strip().casefold()
    if wanted == REMOTE_DRIVER:
        return REMOTE_DRIVER
    if not wanted:
        return raw
    for name in _BUILT_IN:
        if _matches(name, DRIVER_PLUGINS.builtins[name], wanted):
            return name
    for drivers in (DRIVER_PLUGINS.registered, DRIVER_PLUGINS.plugins()):
        for name, entry in drivers.items():
            if _matches(name, entry, wanted):
                return name
    return raw


def _matches(name: str, entry: DriverEntry, wanted: str) -> bool:
    spellings = (name, *entry.aliases)
    return any(wanted == spelling.strip().casefold() for spelling in spellings)


def get_driver(manufacturer: Optional[str]) -> DriverEntry:
    """The driver for *manufacturer*, in any spelling it accepts.

    Raises ``NotImplementedError`` for a manufacturer no driver answers to, with the
    message ``setup_session`` has always given.
    """
    name = normalize_name(manufacturer)
    entry = DRIVER_PLUGINS.get(name) if isinstance(name, str) else None
    if entry is None:
        raise NotImplementedError(f"Manufacturer {manufacturer} not supported.")
    return entry


def device_builder(driver: Optional[str], type: str) -> Optional[DeviceBuilder]:
    """How *driver* builds a device of *type*, or ``None`` if it builds none.

    *driver* is a device entry's ``driver:``, in any spelling a manufacturer has, or
    ``remote``; pass the manufacturer for an entry that names none. Raises
    ``NotImplementedError`` for a driver nothing is registered as, as
    :func:`get_driver` does.
    """
    name = normalize_name(driver)
    if name in _DEVICE_DRIVERS:
        return _import(_DEVICE_DRIVERS[name]).get(type)
    return get_driver(name).devices.get(type)


def milling_settings_classes() -> Dict[str, type]:
    """Each driver's own milling settings class, by the driver's name: the
    built-ins, then those registered at runtime, then the plugins'. Imports each
    driver's record and its settings module, none of its other modules."""
    found: Dict[str, type] = {}
    for name in _BUILT_IN:
        entry = DRIVER_PLUGINS.builtins[name]
        if entry.milling_settings is not None:
            found[name] = _import(entry.milling_settings)
    for name, entry in _all_drivers():
        if name not in found and entry.milling_settings is not None:
            try:
                found[name] = _import(entry.milling_settings)
            except Exception as e:
                logging.warning(f"Driver {name}'s milling settings: {e}")
    return found


def milling_settings_name(settings_class: type) -> Optional[str]:
    """The name of the driver whose own milling settings class is
    *settings_class*: the block a recipe keeps them under. None for a class no
    driver names."""
    for name, cls in milling_settings_classes().items():
        if cls is settings_class:
            return name
    return None


def connect_microscope(system: "SystemSettings") -> "FibsemMicroscope":
    """The microscope ``system.info.manufacturer`` names, built and connected.

    Connects to ``system.info.ip_address`` on ``system.info.port`` when the
    configuration names one, else on the driver's ``port``. With neither, the
    microscope is built and not connected.
    """
    driver = get_driver(system.info.manufacturer)
    microscope = driver.load()(system)
    port = system.info.port
    if port is None:
        port = driver.port
    if port is not None:
        microscope.connect_to_microscope(ip_address=system.info.ip_address, port=port)
    return microscope


def registered_manufacturers() -> List[str]:
    """The manufacturers a driver is registered for: the built-ins, then those
    registered at runtime, then the plugins'. A driver that only builds devices is no
    manufacturer, so it is left out. Imports no built-in driver."""
    names = list(_BUILT_IN)
    others = [*DRIVER_PLUGINS.registered.items(), *DRIVER_PLUGINS.plugins().items()]
    for manufacturer, entry in others:
        if manufacturer not in names and entry.builds_microscope:
            names.append(manufacturer)
    return names


def driver_setups() -> List[Tuple[str, "DriverSetup"]]:
    """Each driver the guided setup offers, as (manufacturer, its ``DriverSetup``),
    in ``registered_manufacturers`` order. Imports every built-in driver's record, and
    each setup's module."""
    entries = [get_driver(m) for m in registered_manufacturers()]
    return [
        (entry.manufacturer, _import(entry.setup))
        for entry in entries
        if entry.setup is not None
    ]


def device_setups() -> List[Tuple[str, "SetupDevice"]]:
    """Each device a device-only driver offers in the guided setup, as (driver, its
    ``SetupDevice``), in registry order. Imports each such driver's setup module; one
    that fails to import is logged and left out, so a broken plugin can't take the
    wizard down."""
    found = []
    for name, entry in _all_drivers():
        if entry.builds_microscope or entry.setup is None:
            continue
        try:
            found += [(name, device) for device in _setup_devices(entry)]
        except Exception as e:
            logging.warning(f"Driver {name}'s setup could not be loaded: {e}")
    return found


def device_setup(driver: Optional[str], type: str) -> Optional["SetupDevice"]:
    """The ``SetupDevice`` *driver* declares for entries of *type*, or ``None``.
    Never raises for a driver that is not installed."""
    try:
        entry = get_driver(driver)
    except NotImplementedError:
        return None
    if entry.builds_microscope or entry.setup is None:
        return None
    return next((d for d in _setup_devices(entry) if d.type == type), None)


def _setup_devices(entry: DriverEntry) -> Tuple["SetupDevice", ...]:
    offered = _import(entry.setup)
    return tuple(offered) if isinstance(offered, (tuple, list)) else (offered,)


def _all_drivers() -> List[Tuple[str, DriverEntry]]:
    """Every registered driver, as (name, record): those registered at runtime, then
    the plugins'. The built-ins all build microscopes, so none is imported."""
    seen: Dict[str, DriverEntry] = {}
    for drivers in (DRIVER_PLUGINS.registered, DRIVER_PLUGINS.plugins()):
        for name, entry in drivers.items():
            seen.setdefault(name, entry)
    return list(seen.items())


