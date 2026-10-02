"""The Chamber device: the vacuum.

Two parameters describe it, both read-only: ``state`` and ``pressure``. Pumping and
venting take minutes and can fail, so they are commands, ``pump`` and ``vent``, not
a settable state.

A backend implements the parameters with ``read_state``/``read_pressure`` and the
commands with two hooks, ``_pump`` and ``_vent``. The base class claims the
``chamber`` resource and reads the state and pressure back, which updates their
caches and emits their change signals.
"""

from __future__ import annotations

from typing import Any, Dict

from fibsem.devices.core import Device, Parameter, command
from fibsem.structures import ChamberState

CHAMBER_RESOURCE = "chamber"

# Today's chamber keys, for a backend that routes them to its chamber device. A get
# key routes to a parameter; a set key that is a verb routes to a command.
CHAMBER_ROUTES: Dict[str, str] = {
    "chamber_state": "state",
    "chamber_pressure": "pressure",
}
CHAMBER_COMMAND_ROUTES: Dict[str, str] = {
    "pump_chamber": "pump",
    "vent_chamber": "vent",
}


class Chamber(Device):
    state = Parameter(ChamberState, doc="The vacuum state.")
    pressure = Parameter(float, unit="Pa", doc="Chamber pressure.")

    def __init__(self, parent: Any = None, **kwargs: Any):
        super().__init__(name="chamber", parent=parent, **kwargs)

    @command
    def pump(self) -> ChamberState:
        """Pump the chamber. Returns the state afterwards."""
        with self.resources.claim(CHAMBER_RESOURCE):
            self._pump()
            return self._read_back()

    @command
    def vent(self) -> ChamberState:
        """Vent the chamber. Returns the state afterwards."""
        with self.resources.claim(CHAMBER_RESOURCE):
            self._vent()
            return self._read_back()

    def _read_back(self) -> ChamberState:
        self.pressure.get_value()
        return self.state.get_value()

    # -- what a backend implements -------------------------------------------------

    def _pump(self) -> None:
        raise NotImplementedError

    def _vent(self) -> None:
        raise NotImplementedError
