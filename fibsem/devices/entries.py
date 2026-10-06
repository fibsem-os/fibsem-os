"""Which devices a microscope builds, from its backend's defaults and `hardware.devices`.

A backend builds the devices it always has, and the configuration's list is an overlay
on them (`DeviceEntry`): an entry switches one off, gives it another driver or keys, or
adds one the backend cannot find for itself. `resolve_device_entries` turns the two into
the devices to build, in order, each with the driver that builds it, and
`build_device_entries` builds each with its driver's builder for its type.

Which builders a driver has is the driver's own record (`DriverEntry.devices`,
`fibsem.microscopes.registry.device_builder`). A builder takes the entry and a
`BuildContext`: the microscope being connected, the devices built before it, and
scratch space its driver's builders share for the connect.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Collection,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
)

from fibsem import manufacturers
from fibsem.structures import DeviceEntry, read_device_entries

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope
    from fibsem.microscopes.registry import DeviceBuilder
    from fibsem.structures import SystemSettings

# The driver for a device on its own PC, reached through the device server.
REMOTE_DRIVER = "remote"


@dataclass(frozen=True)
class ResolvedEntry:
    """One device to build: its entry, and the driver that builds it."""

    entry: DeviceEntry
    driver: str

    @property
    def name(self) -> str:
        return self.entry.name

    @property
    def type(self) -> str:
        return self.entry.type

    @property
    def required(self) -> bool:
        """Whether connecting fails when this device cannot be built."""
        return bool(self.entry.required)


def configured_device_entries(system: "SystemSettings") -> Dict[str, DeviceEntry]:
    """Every device the configuration names, by name, as the file would state it."""
    return read_device_entries(system.to_dict())


def _driver_name(driver: Optional[str], manufacturer: Optional[str]) -> str:
    """The registry name of an entry's driver: its own, else the manufacturer's."""
    name = driver if driver else manufacturer
    if not name:
        raise ValueError("A device has no driver and info.manufacturer is not set.")
    if name.strip().lower() == REMOTE_DRIVER:
        return REMOTE_DRIVER
    return manufacturers.normalize_manufacturer(name)


def resolve_device_entries(
    defaults: Sequence[DeviceEntry],
    configured: Mapping[str, DeviceEntry],
    manufacturer: Optional[str],
) -> List[ResolvedEntry]:
    """The devices to build, in order, with the driver for each.

    *defaults* are the devices the backend builds by itself, in the order it builds
    them. A configured entry with the same name changes that device: its `enabled`,
    `driver`, `required` and keys win over the default's. A configured entry with a
    new name is a device the backend adds, after its own, in file order. An entry that
    is `enabled: false` is left out, so its driver never touches it; absent `enabled`
    is the default's, and a configured device the backend does not build by itself is
    built unless it says otherwise.
    """
    merged: Dict[str, DeviceEntry] = {}
    for default in defaults:
        entry = configured.get(default.name)
        if entry is not None:
            entry = DeviceEntry.from_dict({**default.to_dict(), **entry.to_dict()})
        merged[default.name] = entry or default
    for name, entry in configured.items():
        merged.setdefault(name, entry)

    resolved = []
    for entry in merged.values():
        if entry.enabled is False:
            logging.info(f"Device '{entry.name}' is disabled in the configuration.")
            continue
        resolved.append(ResolvedEntry(entry, _driver_name(entry.driver, manufacturer)))
    return resolved


def resolve_system_devices(
    system: "SystemSettings",
    defaults: Sequence[DeviceEntry],
    types: Optional[Collection[str]] = None,
    exclude_types: Collection[str] = (),
    driver: Optional[str] = None,
) -> List[ResolvedEntry]:
    """`resolve_device_entries` for a connected system's configuration.

    An entry that names no driver is built by *driver*, the backend's own, else by
    `info.manufacturer`'s.

    A backend that builds its devices in steps (the beams, then the stage, ...)
    resolves each step's *types* on its own; `exclude_types` leaves out the types
    built elsewhere, for the step that builds whatever else the file adds.
    """
    configured = {
        name: entry
        for name, entry in configured_device_entries(system).items()
        if (types is None or entry.type in types) and entry.type not in exclude_types
    }
    return resolve_device_entries(
        defaults, configured, driver or system.info.manufacturer
    )


class DeviceBuildError(RuntimeError):
    """A required device could not be built."""


def build_device_entries(
    resolved: Sequence[ResolvedEntry],
    microscope: "FibsemMicroscope",
    shared: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build each device with its driver's builder for its type, in order.

    *shared* starts the builders' scratch space (`BuildContext.shared`), for a backend
    that hands its driver's builders something of its own. A device whose driver has
    no builder for its type, or whose builder fails, is not built: connecting fails for
    a `required` one (`DeviceBuildError`), and logs a warning and goes on without it
    for any other. Returns the built devices by name, in the order they were built.
    """
    from fibsem.microscopes.registry import BuildContext

    built: Dict[str, Any] = {}
    context = BuildContext(
        microscope=microscope, built=built, shared=shared if shared is not None else {}
    )
    for item in resolved:
        builder = _builder(item)
        if builder is None:
            reason = f"driver '{item.driver}' has no builder for a '{item.type}' device"
            _not_built(item, reason)
            continue
        try:
            device = builder.load()(item.entry, context)
        except Exception as e:
            _not_built(item, f"building it failed: {e}", e)
            continue
        if device is not None:
            built[item.name] = device
    return built


def _builder(item: ResolvedEntry) -> Optional["DeviceBuilder"]:
    """The builder *item*'s driver has for its type; None for a driver nothing is
    registered as, too."""
    from fibsem.microscopes.registry import device_builder

    try:
        return device_builder(item.driver, item.type)
    except NotImplementedError:
        return None


def _not_built(
    item: ResolvedEntry, reason: str, error: Optional[Exception] = None
) -> None:
    message = f"Device '{item.name}' was not built: {reason}."
    if item.required:
        raise DeviceBuildError(message) from error
    logging.warning(message)
