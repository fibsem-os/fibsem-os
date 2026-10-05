"""Which driver connects to which manufacturer's microscope (FIB-1123).

One entry per driver: the canonical manufacturer it answers to (as spelled in
``fibsem.manufacturers``), its ``FibsemMicroscope`` class, and the port it connects
on. ``utils.setup_session`` reads it instead of an ``if manufacturer == ...`` chain,
so adding a driver is one ``register_driver`` call rather than an edit to the
connect code.

The class is named as ``"module:Class"`` and imported only when that driver is
asked for. Importing a driver module may import its vendor SDK, which is slow and
absent on most computers, so listing the drivers must never import one.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Dict, List, Optional, Type

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

    port: Optional[int] = None
    """The port ``connect_to_microscope`` is called with. ``None`` means the driver
    is not connected by address at all: Odemis reaches its instrument through its
    own back end, so it is constructed and nothing more."""

    def load(self) -> Type["FibsemMicroscope"]:
        """Import and return the driver's class."""
        module_name, _, class_name = self.microscope_class.partition(":")
        return getattr(importlib.import_module(module_name), class_name)


_DRIVERS: Dict[str, DriverEntry] = {}


def register_driver(entry: DriverEntry) -> None:
    """Make *entry* the driver for its manufacturer.

    Registering a manufacturer again replaces its driver. The name is normalised, so
    an entry registered as "Thermo" is the ThermoFisher driver.
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
    entry = _DRIVERS.get(manufacturers.normalize_manufacturer(manufacturer))
    if entry is None:
        raise NotImplementedError(f"Manufacturer {manufacturer} not supported.")
    return entry


def connect_microscope(system: "SystemSettings") -> "FibsemMicroscope":
    """The microscope ``system.info.manufacturer`` names, built and connected.

    Connects to ``system.info.ip_address`` on the driver's port. A driver with no
    port is built and not connected.
    """
    driver = get_driver(system.info.manufacturer)
    microscope = driver.load()(system)
    if driver.port is not None:
        microscope.connect_to_microscope(
            ip_address=system.info.ip_address, port=driver.port
        )
    return microscope


def registered_manufacturers() -> List[str]:
    """The manufacturers a driver is registered for, in registration order."""
    return list(_DRIVERS)


# The built-in drivers. The ports are the ones setup_session has always used.
register_driver(
    DriverEntry(
        manufacturers.THERMOFISHER,
        "fibsem.microscopes.autoscript:ThermoMicroscope",
        port=7520,
    )
)
register_driver(
    DriverEntry(
        manufacturers.TESCAN, "fibsem.microscopes.tescan:TescanMicroscope", port=8300
    )
)
register_driver(
    DriverEntry(
        manufacturers.ODEMIS,
        "fibsem.microscopes.odemis_microscope:OdemisThermoMicroscope",
    )
)
register_driver(
    DriverEntry(
        manufacturers.DEMO, "fibsem.microscopes.device_demo:DemoMicroscope", port=7520
    )
)
