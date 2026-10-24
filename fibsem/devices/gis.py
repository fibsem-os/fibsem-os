"""The GasInjector device: one gas injection system (GIS) port, or a multichem.

Five read-only parameters describe it: ``gas``, where the needle is (``state``),
whether it is ``heated`` and ``opened``, and the heater's ``temperature``. Each
change is a command, because each moves hardware or waits on it: ``insert``,
``retract``, ``heater_on``, ``heater_off``, ``open``, ``close`` and ``purge``. A
deposition is a sequence of those with a wait in the middle, so it is an operation
on top of the device rather than a command of it.

``purge`` clears the line before a deposition: it opens the valve with the needle
retracted, waits, and closes it, so the first gas to come out goes into the chamber
rather than onto the sample. It refuses while the needle is in. A multichem purges
natively, and its driver implements ``_purge``; every other driver gets the open,
wait, close sequence here.

A backend implements the parameters with ``read_<name>`` methods and the commands
with one hook each (``_insert``, ``_retract``, ...). The base class claims the
``gis`` resource and reads the changed state back, which updates its cache and
emits its change signal. A driver whose instrument can say whether this GIS is
fitted answers ``is_installed``.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from fibsem.devices.core import Device, DeviceParameter, command
from fibsem.structures.system import InsertableDeviceState

GIS_RESOURCE = "gis"


class GasInjector(Device):
    gas = DeviceParameter(str, doc="The gas in use.")
    state = DeviceParameter(InsertableDeviceState, doc="Where the needle is.")
    heated = DeviceParameter(bool, doc="The heater is on.")
    opened = DeviceParameter(bool, doc="The valve is open.")
    temperature = DeviceParameter(float, unit="K", doc="The heater's temperature.")

    def __init__(self, parent: Any = None, **kwargs: Any):
        super().__init__(name="gis", parent=parent, **kwargs)

    @command
    def insert(self, position: Optional[str] = None) -> InsertableDeviceState:
        """Insert, at a named position on a multichem. Returns the state afterwards."""
        with self.resources.claim(GIS_RESOURCE):
            self._insert(position)
            return self.state.get_value()

    @command
    def retract(self) -> InsertableDeviceState:
        """Retract. Returns the state afterwards."""
        with self.resources.claim(GIS_RESOURCE):
            self._retract()
            return self.state.get_value()

    @command
    def heater_on(self, gas: Optional[str] = None) -> bool:
        """Turn the heater on, for a gas on a multichem. Returns whether heated."""
        with self.resources.claim(GIS_RESOURCE):
            self._heater_on(gas)
            self._read_temperature()
            return self.heated.get_value()

    @command
    def heater_off(self) -> bool:
        """Turn the heater off. Returns whether still heated."""
        with self.resources.claim(GIS_RESOURCE):
            self._heater_off()
            self._read_temperature()
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

    @command
    def purge(self, seconds: float) -> bool:
        """Open the valve with the needle retracted for *seconds*, then close it.
        Refuses while the needle is inserted. Returns whether still open."""
        seconds = float(seconds)
        if seconds < 0:
            raise ValueError(f"{self.name}: a purge cannot last {seconds} s.")
        with self.resources.claim(GIS_RESOURCE):
            if self.state.get_value() is not InsertableDeviceState.RETRACTED:
                raise RuntimeError(
                    f"{self.name}: retract the needle before purging; a purge with "
                    "the needle in puts the gas on the sample."
                )
            self._purge(seconds)
            return self.opened.get_value()

    def _read_temperature(self) -> None:
        if "temperature" in self._bound:
            self.temperature.get_value()

    def facts(self) -> dict:
        return {"is_installed": self._installed_or_none()}

    def _installed_or_none(self) -> Optional[bool]:
        try:
            return self.is_installed()
        except Exception:  # a missing GIS and a sick connection look the same
            return None

    # -- what a backend implements -------------------------------------------------

    def is_installed(self) -> Optional[bool]:
        """Whether the instrument reports this GIS fitted, or None if it cannot say."""
        return None

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

    def _purge(self, seconds: float) -> None:
        """Open, wait, close. The valve is closed again even when the wait fails."""
        self._open()
        try:
            self.opened.get_value()
            self._wait(seconds)
        finally:
            self._close()

    def _wait(self, seconds: float) -> None:
        """Wait while the valve is open. A simulated driver pretends."""
        time.sleep(seconds)
