"""What every instrument the App Server serves has, whatever kind it is.

`Instrument` is the generic part of `fibsem.microscope.FibsemMicroscope`: who is in
control, the devices and services it built, the session they report to, and letting
go of them. The server and the gateway type against it, so they serve any kind:
a FIB/SEM microscope (`FibsemMicroscope`, kind ``fibsem``) or devices built from a
Device Server file with no microscope (`DeviceInstrument`).
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Dict, List, Mapping, Optional

from fibsem.control import Control
from fibsem.devices.core import Session
from fibsem.devices.entries import disconnect_devices

__all__ = ["DeviceInstrument", "Instrument"]


class _PerInstance:
    """A class-level default made once per instance, on first use.

    For state every backend needs whether or not its ``__init__`` calls the base
    class's: each instrument gets its own, made by ``factory(instance)``. Assigning the
    attribute on an instance replaces it there, as with a plain attribute.
    """

    def __init__(self, factory: Callable[[Any], Any]):
        self._factory = factory
        self._name = ""
        self._guard = threading.Lock()

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    def __get__(self, instance: Any, owner: Optional[type] = None) -> Any:
        if instance is None:
            return self
        try:
            return instance.__dict__[self._name]
        except KeyError:
            pass
        with self._guard:  # two threads asking first must get the same one
            if self._name not in instance.__dict__:
                instance.__dict__[self._name] = self._factory(instance)
            return instance.__dict__[self._name]


class Instrument(ABC):
    """An instrument: its devices and services, who is in control, and its session."""

    #: What kind of instrument this is, as a playbook's ``instrument:`` names it.
    kind: ClassVar[str]

    #: The experiment this instrument runs in; its devices and services report to it.
    session = _PerInstance(lambda _: Session())

    #: Who is in control: one action at a time (`fibsem.control`).
    control = _PerInstance(lambda _: Control())

    # Every device built, by name, in build order.
    _devices = _PerInstance(lambda _: {})

    # What the builders asked to have closed at disconnect (`BuildContext.on_disconnect`).
    _disconnect_cleanups = _PerInstance(lambda _: [])

    @property
    def devices(self) -> Mapping[str, Any]:
        """Every device this instrument built, by name. Read-only."""
        return MappingProxyType(self._devices)

    @property
    def services(self) -> Mapping[str, Any]:
        """Every service this instrument built, by name. Read-only. None here."""
        return MappingProxyType({})

    def record_event(self, kind: str, payload: Dict[str, Any]) -> None:
        """Put a fact on the experiment's record (`Session.record`). Never raises."""
        self.session.record(kind, payload)

    def disconnect(self) -> None:
        """Let go of the instrument: each device, last built first, then what the
        builders share, then `_disconnect`. A device or cleanup that raises is
        logged and the rest still run; an error in `_disconnect` is raised."""
        disconnect_devices(self._devices, self._disconnect_cleanups)
        self._disconnect()

    @abstractmethod
    def _disconnect(self) -> None:
        """Close the instrument's own client. `disconnect` calls it after the devices."""

    @abstractmethod
    def get_state(self) -> Any:
        """The instrument's state as a whole, to restore with `set_state`."""

    @abstractmethod
    def set_state(self, state: Any) -> None:
        """Restore a state `get_state` read."""


class DeviceInstrument(Instrument):
    """Devices built from a Device Server file (its ``hardware.devices``), with no
    microscope: each by its own driver, as the Device Server builds them. It has no
    state as a whole: `get_state` and `set_state` raise ``NotImplementedError``.
    """

    kind: ClassVar[str] = "devices"

    def __init__(
        self,
        devices: Mapping[str, Any],
        cleanups: Optional[List[Callable[[], None]]] = None,
    ) -> None:
        self._devices = dict(devices)
        self._disconnect_cleanups = list(cleanups or [])

    @classmethod
    def from_config(cls, source: Any) -> "DeviceInstrument":
        """Build the devices a Device Server file lists (a path, or its settings
        already read), with every FM's light off."""
        from fibsem.server.devices import configured_devices, switch_off_fms

        devices, cleanups = configured_devices(source)
        switch_off_fms(devices.values())
        return cls(devices, cleanups)

    def _disconnect(self) -> None:
        pass

    def get_state(self) -> Any:
        raise NotImplementedError(
            "Devices built on their own have no state as a whole."
        )

    def set_state(self, state: Any) -> None:
        raise NotImplementedError(
            "Devices built on their own have no state as a whole."
        )
