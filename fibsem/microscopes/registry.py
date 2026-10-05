"""Which driver connects to which manufacturer's microscope (FIB-1123).

One entry per driver: the canonical manufacturer it answers to (as spelled in
``fibsem.manufacturers``), its ``FibsemMicroscope`` class, the port it connects on,
and the configuration values a new configuration for it starts from.
``utils.setup_session`` reads it instead of an ``if manufacturer == ...`` chain, so
adding a driver is one ``register_driver`` call rather than an edit to the connect
code.

The class is named as ``"module:Class"`` and imported only when that driver is
asked for. Importing a driver module may import its vendor SDK, which is slow and
absent on most computers, so listing the drivers must never import one.

A driver outside fibsem registers through the ``fibsem.drivers`` entry point group,
naming a function that returns its ``DriverEntry``::

    [project.entry-points."fibsem.drivers"]
    jeol = "fibsem_jeol.registry:driver"

A record rather than the class, so the format survives the microscope classes
becoming device builders (FIB-1160): the entry is what changes, not the contract.
The entry points are read once, on the first ``get_driver`` (so on connect), or by
``load_driver_plugins``. A built-in driver keeps its manufacturer: a plugin that
claims one is recorded and not registered.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple, Type

from fibsem import manufacturers

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope


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
    own back end, so it is constructed and nothing more. A configuration's
    ``info.port`` overrides it."""

    config: Mapping[str, Any] = field(default_factory=dict)
    """Default configuration values for this manufacturer's instruments, such as
    ``ion-column-tilt``. ``config.DEFAULT_CONFIGURATION_VALUES`` is built from these.
    Empty for a driver no configuration is generated for (Odemis)."""

    def load(self) -> Type["FibsemMicroscope"]:
        """Import and return the driver's class."""
        module_name, _, class_name = self.microscope_class.partition(":")
        return getattr(importlib.import_module(module_name), class_name)


@dataclass(frozen=True)
class DriverPlugin:
    """One ``fibsem.drivers`` entry point, and what became of it.

    Kept whether or not it registered, so a plugin that failed or was refused can
    be reported rather than only logged.
    """

    entry_point: str
    """The entry point's own name, as written in the plugin's pyproject."""

    value: str
    """The declared target, e.g. ``"fibsem_jeol.registry:driver"``."""

    distribution: Optional[str] = None
    version: Optional[str] = None

    entry: Optional[DriverEntry] = None
    """The record it returned, or ``None`` if it returned none."""

    error: Optional[str] = None
    """Why it is not the registered driver, or ``None`` if it is."""

    @property
    def registered(self) -> bool:
        return self.error is None


DRIVER_ENTRY_POINT_GROUP = "fibsem.drivers"

_DRIVERS: Dict[str, DriverEntry] = {}

# None until the entry points have been read.
_PLUGINS: Optional[Tuple[DriverPlugin, ...]] = None


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
    message ``setup_session`` has always given. Reads the ``fibsem.drivers`` entry
    points first, the first time it is called.
    """
    load_driver_plugins()
    entry = _DRIVERS.get(manufacturers.normalize_manufacturer(manufacturer))
    if entry is None:
        raise NotImplementedError(f"Manufacturer {manufacturer} not supported.")
    return entry


def registered_manufacturers() -> List[str]:
    """The manufacturers a driver is registered for, in registration order.

    Plugin drivers are included once the entry points have been read; this does not
    read them, so it is safe to call at import.
    """
    return list(_DRIVERS)


def default_configuration_values() -> Dict[str, Dict[str, Any]]:
    """Each manufacturer's default configuration values, for the drivers that have
    any, in registration order.

    Like ``registered_manufacturers``, this does not read the entry points.
    """
    return {
        manufacturer: dict(entry.config)
        for manufacturer, entry in _DRIVERS.items()
        if entry.config
    }


def load_driver_plugins() -> Tuple[DriverPlugin, ...]:
    """Read the ``fibsem.drivers`` entry points, once, registering what they return.

    Never raises: a plugin that fails to load is recorded and logged, because a
    third-party package must not stop fibsem from connecting to anything else. A
    plugin claiming a built-in driver's manufacturer is refused; when two plugins
    claim the same one, the later one is the driver, as in the other plugin groups.
    """
    global _PLUGINS
    if _PLUGINS is not None:
        return _PLUGINS
    # Set before loading, so a plugin that asks for a driver while it loads does not
    # start a second read.
    _PLUGINS = ()

    from fibsem.plugins.loader import _distribution_of, _entry_points

    records: List[DriverPlugin] = []
    claimed: Dict[str, int] = {}  # manufacturer -> index of the plugin holding it
    for entry_point in _entry_points(DRIVER_ENTRY_POINT_GROUP):
        distribution, version = _distribution_of(entry_point)
        record = DriverPlugin(
            entry_point=entry_point.name,
            value=entry_point.value,
            distribution=distribution,
            version=version,
        )
        try:
            entry = entry_point.load()()
        except Exception as exc:
            logging.error(
                "Could not load the driver plugin '%s'",
                entry_point.value,
                exc_info=True,
            )
            records.append(replace(record, error=f"{type(exc).__name__}: {exc}"))
            continue

        if not isinstance(entry, DriverEntry):
            reason = f"returned {type(entry).__name__}, not a DriverEntry"
            logging.warning("Invalid driver plugin '%s': %s", entry_point.value, reason)
            records.append(replace(record, error=reason))
            continue

        manufacturer = manufacturers.normalize_manufacturer(entry.manufacturer)
        if manufacturer in _BUILT_IN:
            reason = f"{manufacturer} has a built-in driver - the built-in is used"
            logging.warning("Driver plugin '%s' refused: %s", entry_point.value, reason)
            records.append(replace(record, entry=entry, error=reason))
            continue

        if manufacturer in claimed:
            earlier = claimed[manufacturer]
            records[earlier] = replace(
                records[earlier],
                error=f"{manufacturer} was taken by '{entry_point.value}'",
            )
        register_driver(entry)
        claimed[manufacturer] = len(records)
        records.append(replace(record, entry=entry))
        logging.info("Loaded the %s driver from '%s'", manufacturer, entry_point.value)

    _PLUGINS = tuple(records)
    return _PLUGINS


# The built-in drivers. The ports are the ones setup_session has always used; the
# column tilts [degrees] are the ones config.DEFAULT_CONFIGURATION_VALUES listed.
register_driver(
    DriverEntry(
        manufacturers.THERMOFISHER,
        "fibsem.microscopes.autoscript:ThermoMicroscope",
        port=7520,
        config={"ion-column-tilt": 52, "electron-column-tilt": 0},
    )
)
register_driver(
    DriverEntry(
        manufacturers.TESCAN,
        "fibsem.microscopes.tescan:TescanMicroscope",
        port=8300,
        config={"ion-column-tilt": 55, "electron-column-tilt": 0},
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
        manufacturers.DEMO,
        "fibsem.microscopes.device_demo:DemoMicroscope",
        port=7520,
        config={"ion-column-tilt": 52, "electron-column-tilt": 0},
    )
)

_BUILT_IN = frozenset(_DRIVERS)
