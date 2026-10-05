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
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Type

from fibsem import manufacturers

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope
    from fibsem.structures import SystemSettings


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

    def load(self) -> Type["FibsemMicroscope"]:
        """Import and return the driver's class."""
        return _import(self.microscope_class)


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

# Drivers registered at runtime. One for a built-in manufacturer replaces it.
_DRIVERS: Dict[str, DriverEntry] = {}


def register_driver(entry: DriverEntry) -> None:
    """Make *entry* the driver for its manufacturer.

    Registering a manufacturer again replaces its driver, built-in or not. The name
    is normalised, so an entry registered as "Thermo" is the ThermoFisher driver.
    """
    manufacturer = manufacturers.normalize_manufacturer(entry.manufacturer)
    if manufacturer != entry.manufacturer:
        entry = replace(entry, manufacturer=manufacturer)
    _DRIVERS[manufacturer] = entry


def get_driver(manufacturer: Optional[str]) -> DriverEntry:
    """The driver for *manufacturer*, in any known spelling.

    Raises ``NotImplementedError`` for a manufacturer no driver answers to, with the
    message ``setup_session`` has always given.
    """
    canonical = manufacturers.normalize_manufacturer(manufacturer)
    entry = _DRIVERS.get(canonical)
    if entry is None and canonical in _BUILT_IN:
        entry = _import(_BUILT_IN[canonical])
    if entry is None:
        raise NotImplementedError(f"Manufacturer {manufacturer} not supported.")
    return entry


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
    """The manufacturers a driver is registered for: the built-ins, then the rest
    in registration order. Imports no driver."""
    return list(_BUILT_IN) + [m for m in _DRIVERS if m not in _BUILT_IN]


def default_configuration_values() -> Dict[str, Dict[str, Any]]:
    """Each manufacturer's driver config, for the drivers that have any: the
    built-ins, then the rest in registration order.

    Reads every built-in driver's record, so it imports their modules.
    """
    entries = [get_driver(m) for m in registered_manufacturers()]
    return {entry.manufacturer: dict(entry.config) for entry in entries if entry.config}
