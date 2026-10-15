"""Which devices a microscope builds, from its backend's defaults and `hardware.devices`.

A backend builds the devices it always has, and the configuration's list is an overlay
on them (`DeviceEntry`): an entry switches one off, gives it another driver or keys, or
adds one the backend cannot find for itself. `resolve_device_entries` turns the two into
the devices to build, in order, each with the driver that builds it, and
`build_device_entries` builds each with its driver's builder for its type.

Which builders a driver has is the driver's own record (`DriverEntry.devices`,
`fibsem.drivers.registry.device_builder`). A builder takes the entry and a
`BuildContext`: the microscope being connected, the devices built before it, and
scratch space its driver's builders share for the connect.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
)

from fibsem.devices.core import Device
from fibsem.drivers.registry import REMOTE_DRIVER, normalize_name
from fibsem.structures import DeviceEntry, read_device_entries

if TYPE_CHECKING:
    from fibsem.drivers.registry import DeviceBuilder
    from fibsem.microscope import FibsemMicroscope
    from fibsem.structures import SystemSettings


@dataclass(frozen=True)
class ResolvedEntry:
    """One device to build: its entry, and the driver that builds it.

    `listed` is whether the configuration names it, rather than it being only one of
    the devices its backend builds by itself.
    """

    entry: DeviceEntry
    driver: str
    listed: bool = False

    @property
    def name(self) -> str:
        return self.entry.name

    @property
    def type(self) -> str:
        return self.entry.type

    @property
    def required(self) -> bool:
        """Whether connecting fails when this device cannot be built: the entry's
        `required`, else whether the configuration names it. A device the
        configuration names is one the site says it has; one the backend only
        assumes is left out instead."""
        if self.entry.required is not None:
            return bool(self.entry.required)
        return self.listed


def configured_device_entries(system: "SystemSettings") -> Dict[str, DeviceEntry]:
    """Every device the configuration names, by name, as the file would state it."""
    return read_device_entries(system.to_dict())


def _driver_name(driver: Optional[str], manufacturer: Optional[str]) -> str:
    """The registry name of an entry's driver: its own, else the manufacturer's."""
    name = driver if driver else manufacturer
    if not name:
        raise ValueError("A device has no driver and info.manufacturer is not set.")
    return normalize_name(name)


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
    built unless it says otherwise. A configured device is `required` unless its entry
    says `required: false`; one only in *defaults* is not, unless it says so.
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
        resolved.append(
            ResolvedEntry(
                entry,
                _driver_name(entry.driver, manufacturer),
                listed=entry.name in configured,
            )
        )
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
    microscope: Optional["FibsemMicroscope"],
    shared: Optional[Dict[str, Any]] = None,
    cleanups: Optional[List[Callable[[], None]]] = None,
) -> Dict[str, Any]:
    """Build each device with its driver's builder for its type, in order.

    *shared* starts the builders' scratch space (`BuildContext.shared`), for a backend
    that hands its driver's builders something of its own. A device whose driver has
    no builder for its type or can't be imported, or whose builder fails, is not built: connecting fails for
    a `required` one (`DeviceBuildError`); any other is logged as a warning and marked
    not fitted on *microscope*, and connecting goes on without it. Returns the built devices by name, in the order they were built.
    """
    from fibsem.drivers.registry import BuildContext

    built: Dict[str, Any] = {}
    context = BuildContext(
        microscope=microscope, built=built, shared=shared if shared is not None else {}
    )
    try:
        for item in resolved:
            try:
                builder = _builder(item)
            except ImportError as e:
                # such as the remote driver without the server extra
                reason = f"driver '{item.driver}' could not be loaded: {e}"
                _not_built(item, microscope, reason, e)
                continue
            except NotImplementedError:
                # nothing is registered as the driver: a plugin not installed in this
                # environment, or a typo
                reason = (
                    f"no driver '{item.driver}' is installed (installed: "
                    f"{', '.join(_installed_drivers())}); check that its plugin is "
                    "installed in this environment"
                )
                _not_built(item, microscope, reason)
                continue
            if builder is None:
                reason = (
                    f"driver '{item.driver}' has no builder for a '{item.type}' device"
                )
                _not_built(item, microscope, reason)
                continue
            try:
                entry = _checked_entry(item)
            except Exception as e:
                _not_built(item, microscope, f"its keys are wrong: {e}", e)
                continue
            try:
                device = builder.load()(entry, context)
            except Exception as e:
                _not_built(item, microscope, f"building it failed: {e}", e)
                continue
            if device is not None:
                if isinstance(device, Device):
                    if device.driver is None:
                        device.driver = item.driver
                    if device.entry_type is None:
                        device.entry_type = item.type
                built[item.name] = device
    finally:
        # registered even when a required device fails, so what was opened closes
        if cleanups is None:
            cleanups = getattr(microscope, "_disconnect_cleanups", None)
        if context.cleanups and cleanups is not None:
            cleanups.extend(context.cleanups)
    return built


def disconnect_devices(
    devices: Mapping[str, Any], cleanups: List[Callable[[], None]]
) -> None:
    """Disconnect *devices* in reverse build order, then run *cleanups* (what their
    builders registered with ``BuildContext.on_disconnect``) last first, and empty
    the list so none runs twice. One that raises is logged and the rest still run."""
    for name, device in reversed(list(devices.items())):
        close = getattr(device, "disconnect", None)
        if close is None:
            continue
        try:
            close()
        except Exception as e:
            logging.warning(f"Device '{name}' did not disconnect cleanly: {e}")
    pending = list(cleanups)
    cleanups.clear()
    for cleanup in reversed(pending):
        try:
            cleanup()
        except Exception as e:
            logging.warning(f"A device connection did not close cleanly: {e}")


def _builder(item: ResolvedEntry) -> Optional["DeviceBuilder"]:
    """The builder *item*'s driver has for its type, or None. Raises
    ``NotImplementedError`` for a driver nothing is registered as."""
    from fibsem.drivers.registry import device_builder

    return device_builder(item.driver, item.type)


def _checked_entry(item: ResolvedEntry) -> DeviceEntry:
    """*item*'s entry, its keys checked against the ``SetupDevice`` its driver
    declares for its type and the defaults filled in; unchanged where it declares
    none. Raises ``ValueError`` for an unknown key or one of the wrong type."""
    from fibsem.drivers.registry import device_setup

    schema = device_setup(item.driver, item.type)
    if schema is None:
        return item.entry
    return replace(item.entry, options=schema.options(item.entry.options, item.driver))


def _installed_drivers() -> List[str]:
    """Every driver a device entry can name, without importing any of them."""
    from fibsem.drivers.registry import DRIVER_PLUGINS

    names = [*DRIVER_PLUGINS.builtins, *DRIVER_PLUGINS.registered]
    names += [n for n in DRIVER_PLUGINS.plugins() if n not in names]
    return [*names, REMOTE_DRIVER]


def _not_built(
    item: ResolvedEntry,
    microscope: Optional["FibsemMicroscope"],
    reason: str,
    error: Optional[Exception] = None,
) -> None:
    message = f"Device '{item.name}' was not built: {reason}."
    if item.required:
        raise DeviceBuildError(message) from error
    logging.warning(message)
    mark = getattr(microscope, "_mark_device_not_built", None)
    if mark is not None:
        mark(item.name, reason)


class RoleBindingError(ValueError):
    """A `roles:` binding in `hardware.devices` can't be made."""


def bind_device_roles(
    resolved: Sequence[ResolvedEntry],
    devices: Mapping[str, Any],
    configured: Collection[str] = (),
) -> None:
    """Fill each role an entry binds (``roles: {scanner: scan_generator}``) with the
    device of that name, once every device is built.

    A bound device is only referenced: the parent never builds, disables or tears it
    down. A binding to a device that wasn't built (``enabled: false``, or a device
    that isn't required and failed) is skipped with a warning, and the parent keeps
    its own way. *configured* names the file's entries, switched-off ones included,
    so that a binding to a name nothing has is told apart from those.

    A role the parent doesn't have, a device that doesn't implement the role's
    interface, a name no entry has, or bindings that form a cycle is a
    `RoleBindingError`.
    """
    known = {item.name for item in resolved} | set(configured)
    bindings = {item.name: dict(item.entry.roles or {}) for item in resolved}
    _check_no_cycle(bindings)
    for item in resolved:
        for role, target in bindings[item.name].items():
            where = f"hardware.devices: {item.name}.roles.{role}"
            if target not in known:
                raise RoleBindingError(f"{where}: no device is named '{target}'")
            parent = devices.get(item.name)
            if parent is None:
                continue
            device = devices.get(target)
            if device is None:
                logging.warning(
                    f"{where}: '{target}' was not built; role left unbound."
                )
                continue
            try:
                parent.fill_roles(**{role: device})
            except TypeError as e:
                raise RoleBindingError(f"{where}: {e}") from None
            logging.info(f"{item.name}.{role} is bound to '{target}'.")


def _check_no_cycle(bindings: Mapping[str, Mapping[str, str]]) -> None:
    """Raise if following the bindings from any device leads back to it."""
    done: set = set()

    def visit(name: str, path: List[str]) -> None:
        if name in path:
            cycle = " -> ".join([*path[path.index(name) :], name])
            raise RoleBindingError(f"hardware.devices: roles form a cycle: {cycle}")
        if name in done:
            return
        for target in bindings.get(name, {}).values():
            visit(target, [*path, name])
        done.add(name)

    for name in bindings:
        visit(name, [])
