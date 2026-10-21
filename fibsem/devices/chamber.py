"""The Chamber device: the vacuum.

Two parameters describe it, both read-only: ``state`` and ``pressure``. Pumping and
venting take minutes and can fail, so they are commands, ``pump`` and ``vent``, not
a settable state.

A cryo chamber may also report, all read-only: the touch alarm, the nitrogen flow and
heat exchanger that cool it, the stage and shield temperatures, and its cryo
``compartment`` (the chamber side of an autoloader: state, dewar level, temperatures).
A backend binds those its hardware has; the rest stay absent.

A backend implements the parameters with ``read_state``/``read_pressure`` and the
commands with two hooks, ``_pump`` and ``_vent``. The base class claims the
``chamber`` resource and reads the state and pressure back, which updates their
caches and emits their change signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from fibsem.codec import decode, encode
from fibsem.devices.core import Device, DeviceParameter, command
from fibsem.structures import ChamberState

CHAMBER_RESOURCE = "chamber"


@dataclass(frozen=True)
class CryoCompartment:
    """A cryo compartment, as the hardware reports it: its temperature state
    (``"Cryo"``, ``"Room"``, or empty when it does not say), its dewar level from 0 to
    1 (None when it does not say), and each temperature sensor's reading in kelvin."""

    state: str = ""
    dewar_level: Optional[float] = None
    temperatures: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return encode(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CryoCompartment":
        return decode(cls, data)


class Chamber(Device):
    state = DeviceParameter(ChamberState, doc="The vacuum state.")
    pressure = DeviceParameter(float, unit="Pa", doc="Chamber pressure.")
    touch_alarm = DeviceParameter(
        str,
        doc='What touched ("Manipulator", "GIS", ...), or empty when the touch '
        "alarm is not triggered.",
    )
    n2_flow = DeviceParameter(float, unit="kg/s", doc="Nitrogen mass flow.")
    n2_flow_on = DeviceParameter(bool, doc="Whether the nitrogen flow is on.")
    heat_exchanger_on = DeviceParameter(bool, doc="Whether the heat exchanger is on.")
    stage_temperature = DeviceParameter(float, unit="K", doc="Stage temperature.")
    shield_temperature = DeviceParameter(
        float, unit="K", doc="Cryo shield temperature."
    )
    compartment = DeviceParameter(
        CryoCompartment,
        doc="The chamber's cryo compartment: state, dewar level, temperatures.",
    )

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
