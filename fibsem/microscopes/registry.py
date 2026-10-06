"""Which driver connects to which manufacturer's microscope (FIB-1123).

Each driver module describes itself with a module-level ``DRIVER`` record: the
canonical manufacturer it answers to (as spelled in ``fibsem.manufacturers``), its
``FibsemMicroscope`` class, and its configuration (the port it connects on, for one).
``utils.setup_session`` connects through ``connect_microscope`` instead of an
``if manufacturer == ...`` chain.

The built-in drivers are listed here, by where their record lives, and a record is
imported only when that driver is asked for. Importing a driver module must not need
its vendor SDK (each one tries the SDK and carries on without it), but it is still
slow, so listing the drivers never imports one.

A driver outside fibsem registers through the ``fibsem.drivers`` entry point group,
naming a function that returns its ``DriverEntry``::

    [project.entry-points."fibsem.drivers"]
    jeol = "fibsem_jeol.registry:driver"

Built-ins in code, plugins in packages, as in fibsem's other plugin groups. A record
rather than the class, so the format survives the microscope classes becoming device
builders (FIB-1160): the entry is what changes, not the contract. The entry points
are read once, by ``fibsem.plugins.loader``, the first time a driver is looked up or
listed. When two drivers claim one manufacturer, the same order decides as for
patterns, strategies and tasks: built-in, then registered at runtime, then plugin.
"""

from __future__ import annotations

import importlib
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

from fibsem import manufacturers
from fibsem.plugins.loader import PluginRegistry, PluginRejected

if TYPE_CHECKING:
    from fibsem.devices.core import Device
    from fibsem.microscope import FibsemMicroscope
    from fibsem.plugins.loader import PluginRecord
    from fibsem.structures import DeviceEntry, SystemSettings


@dataclass(frozen=True)
class DriverEntry:
    """One driver, as the connect code needs to know it."""

    manufacturer: str
    """The canonical manufacturer name, as ``SystemInfo.manufacturer`` carries it."""

    microscope_class: str
    """The driver's ``FibsemMicroscope`` subclass, as ``"module:Class"``."""

    config: Mapping[str, Any] = field(default_factory=dict)
    """The driver's own facts, and the default configuration values for its
    instruments. ``port`` is the port ``connect_microscope`` connects on (a
    configuration's ``info.port`` overrides it); without either, the driver is not
    connected by address at all (Odemis reaches its instrument through its own back
    end, so it is constructed and nothing more). Column tilts (``ion-column-tilt``,
    ``electron-column-tilt``) are what ``config.DEFAULT_CONFIGURATION_VALUES`` is
    built from."""

    devices: Mapping[str, "DeviceBuilder"] = field(default_factory=dict)
    """How this driver builds a device, by the device entry's ``type`` (``beam``,
    ``stage``, ``gis``, ...): what a ``hardware.devices`` entry naming this driver is
    built with. See :func:`device_builder`."""

    def load(self) -> Type["FibsemMicroscope"]:
        """Import and return the driver's class."""
        return _import(self.microscope_class)


@dataclass(frozen=True)
class DeviceBuilder:
    """How a driver builds one type of device from its configuration entry."""

    build: str
    """The build function, as ``"module:function"``, imported when first used. It is
    called as ``build(entry, context)`` with the ``DeviceEntry`` and a
    :class:`BuildContext`, and returns the connected device, or raises."""

    implements: Tuple[str, ...] = ()
    """The interfaces the device implements, by class name (``"Scanner"``), for a
    role to be bound to it (FIB-1167). Nothing reads it yet."""

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


BuildFn = Callable[["DeviceEntry", BuildContext], "Device"]
"""A device builder's function: ``build(entry, context) -> Device``."""


DRIVER_ENTRY_POINT_GROUP = "fibsem.drivers"


def _import(target: str) -> Any:
    """The object a ``"module:attribute"`` string names."""
    module_name, _, attribute = target.partition(":")
    return getattr(importlib.import_module(module_name), attribute)


# The built-in drivers, by where each one's DRIVER record lives.
_BUILT_IN: Dict[str, str] = {
    manufacturers.THERMOFISHER: "fibsem.microscopes.autoscript:DRIVER",
    manufacturers.TESCAN: "fibsem.microscopes.tescan:DRIVER",
    manufacturers.ODEMIS: "fibsem.microscopes.odemis_microscope:DRIVER",
    manufacturers.DEMO: "fibsem.microscopes.device_demo:DRIVER",
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
    describe=lambda entry: entry.microscope_class,
)


def register_driver(entry: DriverEntry) -> None:
    """Make *entry* the driver for its manufacturer, unless a built-in has it.

    Registering a manufacturer again replaces its registered driver. The name is
    normalised, so an entry registered as "jeol" and one as "JEOL" are one driver
    if ``fibsem.manufacturers`` knows them as one.
    """
    entry = _canonical(entry)
    DRIVER_PLUGINS.register(entry.manufacturer, entry)


def _canonical(entry: DriverEntry) -> DriverEntry:
    manufacturer = manufacturers.normalize_manufacturer(entry.manufacturer)
    if manufacturer != entry.manufacturer:
        entry = replace(entry, manufacturer=manufacturer)
    return entry


def get_driver(manufacturer: Optional[str]) -> DriverEntry:
    """The driver for *manufacturer*, in any known spelling.

    Raises ``NotImplementedError`` for a manufacturer no driver answers to, with the
    message ``setup_session`` has always given.
    """
    entry = DRIVER_PLUGINS.get(manufacturers.normalize_manufacturer(manufacturer))
    if entry is None:
        raise NotImplementedError(f"Manufacturer {manufacturer} not supported.")
    return entry


def device_builder(driver: Optional[str], type: str) -> Optional[DeviceBuilder]:
    """How *driver* builds a device of *type*, or ``None`` if it builds none.

    *driver* is a device entry's ``driver:``, in any spelling a manufacturer has;
    pass the manufacturer for an entry that names none. Raises ``NotImplementedError``
    for a driver nothing is registered as, as :func:`get_driver` does.
    """
    return get_driver(driver).devices.get(type)


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
        port = driver.config.get("port")
    if port is not None:
        microscope.connect_to_microscope(ip_address=system.info.ip_address, port=port)
    return microscope


def registered_manufacturers() -> List[str]:
    """The manufacturers a driver is registered for: the built-ins, then those
    registered at runtime, then the plugins'. Imports no built-in driver."""
    names = list(_BUILT_IN)
    for manufacturer in [*DRIVER_PLUGINS.registered, *DRIVER_PLUGINS.plugins()]:
        if manufacturer not in names:
            names.append(manufacturer)
    return names


def default_configuration_values() -> Dict[str, Dict[str, Any]]:
    """Each manufacturer's driver config, for the drivers that have any, in
    ``registered_manufacturers`` order. Imports every built-in driver's module."""
    entries = [get_driver(m) for m in registered_manufacturers()]
    return {entry.manufacturer: dict(entry.config) for entry in entries if entry.config}


def get_driver_plugin_records() -> Tuple["PluginRecord", ...]:
    """Every ``fibsem.drivers`` entry point and what became of it, read once.

    Includes the plugins that failed. One whose manufacturer a built-in or a
    runtime registration also claims loads, and is not the driver used.
    """
    return DRIVER_PLUGINS.plugin_records()
