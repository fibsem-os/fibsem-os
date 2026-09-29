"""The GasInjector device: the gas injection system (GIS), or a multichem.

Four read-only parameters describe it: ``gas``, and whether it is ``inserted``,
``heated`` and ``opened``. Each change is a command, because each moves hardware
or waits on it: ``insert``, ``retract``, ``heater_on``, ``heater_off``, ``open``
and ``close``. A deposition is a sequence of those with a wait in the middle, so
it is not a command here; ``cryo_deposition_v2`` runs it.

A backend implements the parameters with ``read_<name>`` methods and the commands
with one hook each (``_insert``, ``_retract``, ...). The base class claims the
``gis`` resource and reads the changed state back, which updates its cache and
emits its change signal.
"""

from __future__ import annotations

from typing import Any, Optional

from fibsem.devices.core import Device, Parameter, command

GIS_RESOURCE = "gis"


class GasInjector(Device):
    gas = Parameter(str, doc="The gas in use.")
    inserted = Parameter(bool, doc="The needle is inserted, not retracted.")
    heated = Parameter(bool, doc="The heater is on.")
    opened = Parameter(bool, doc="The valve is open.")

    def __init__(self, parent: Any = None, **kwargs: Any):
        super().__init__(name="gis", parent=parent, **kwargs)

    @command
    def insert(self, position: Optional[str] = None) -> bool:
        """Insert, at a named position on a multichem. Returns whether inserted."""
        with self.resources.claim(GIS_RESOURCE):
            self._insert(position)
            return self.inserted.get_value()

    @command
    def retract(self) -> bool:
        """Retract. Returns whether still inserted."""
        with self.resources.claim(GIS_RESOURCE):
            self._retract()
            return self.inserted.get_value()

    @command
    def heater_on(self, gas: Optional[str] = None) -> bool:
        """Turn the heater on, for a gas on a multichem. Returns whether heated."""
        with self.resources.claim(GIS_RESOURCE):
            self._heater_on(gas)
            return self.heated.get_value()

    @command
    def heater_off(self) -> bool:
        """Turn the heater off. Returns whether still heated."""
        with self.resources.claim(GIS_RESOURCE):
            self._heater_off()
            return self.heated.get_value()

    @command
    def open(self) -> bool:
        """Open the valve. Returns whether open."""
        with self.resources.claim(GIS_RESOURCE):
            self._open()
            return self.opened.get_value()

    @command
    def close(self) -> bool:
        """Close the valve. Returns whether still open."""
        with self.resources.claim(GIS_RESOURCE):
            self._close()
            return self.opened.get_value()

    # -- what a backend implements -------------------------------------------------

    def _insert(self, position: Optional[str]) -> None:
        raise NotImplementedError

    def _retract(self) -> None:
        raise NotImplementedError

    def _heater_on(self, gas: Optional[str]) -> None:
        raise NotImplementedError

    def _heater_off(self) -> None:
        raise NotImplementedError

    def _open(self) -> None:
        raise NotImplementedError

    def _close(self) -> None:
        raise NotImplementedError
