"""The ranges and choices the FM widgets offer, as the connected FM reports them.

A channel's exposure, power and gain, and the camera's binning, are bound to the FM's
devices (``ChannelSettings`` declares ``device_parameter="camera.exposure_time"``), so
``bound_spec`` resolves them the way every bound field is: what the FM's device
reports, else what the device class declares when there is no FM or it can't say.

Power and gain are percentages of the hardware's maximum. Where the driver knows what
that maximum is in the hardware's own units (watts, camera gain), the box's tooltip
shows the value in those units too.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Mapping, Optional, Tuple

from PyQt5.QtCore import QEvent, QObject
from PyQt5.QtWidgets import QAbstractSpinBox, QToolTip

from fibsem.fm.structures import ChannelSettings
from fibsem.structures.params import Parameter
from fibsem.ui.widgets.custom_widgets import ValueSpinBox
from fibsem.ui.widgets.form_builder import bound_spec, configure_spinbox

NativeScale = Tuple[float, Optional[str]]

# The channel numbers with a box of their own, each bound to one of the FM's devices.
CHANNEL_NUMBERS = ("exposure_time", "power", "gain")

_BINNING = Parameter(device_parameter="camera.binning")


def _in_steps(
    limits: Tuple[float, float], scale: float, decimals: int
) -> Tuple[float, float]:
    """``limits`` times ``scale``, narrowed to values a box showing ``decimals``
    places can hold, so its minimum never displays as a value the FM refuses."""
    step = 10.0**-decimals
    low, high = (float(v) * scale / step for v in limits)
    low, high = math.ceil(round(low, 6)) * step, math.floor(round(high, 6)) * step
    return (round(low, decimals), round(high, decimals))


def _read(fm: Any, what: str, read) -> Optional[Any]:
    if fm is None:
        return None
    try:
        return read(fm)
    except Exception as e:  # a widget must open even when the FM can't answer
        logging.warning(f"Could not read the FM's {what}, using the defaults: {e}")
        return None


def _devices(fm: Any) -> Mapping[str, Any]:
    """The FM's devices by name; none without an FM, or when it can't say."""
    return _read(fm, "devices", lambda fm: getattr(fm, "devices", None)) or {}


def channel_range(fm: Any, name: str) -> Tuple[float, float]:
    """The range a channel's ``name`` box offers, in the units it shows."""
    spec = bound_spec(ChannelSettings.params[name], _devices(fm))
    limits = spec.limits
    return _in_steps((limits.min, limits.max), spec.scale, spec.display.decimals)


def channel_ranges(fm: Any) -> Dict[str, Tuple[float, float]]:
    """``channel_range`` for each of the channel's numbers."""
    return {name: channel_range(fm, name) for name in CHANNEL_NUMBERS}


def channel_spinbox(name: str, shown_range: Tuple[float, float]) -> ValueSpinBox:
    """A box for one of a channel's numbers, shown as ``ChannelSettings`` declares
    it (its device's unit, step and decimals) over ``shown_range``, what the FM
    reports."""
    spinbox = ValueSpinBox()
    configure_spinbox(spinbox, ChannelSettings.params[name])
    spinbox.setRange(*shown_range)
    return spinbox


def shown_channel_value(channel: ChannelSettings, name: str) -> float:
    """A channel's number in the units its box shows; an unset one shows as 0."""
    value = getattr(channel, name)
    return (value or 0.0) * ChannelSettings.params[name].scale


def stored_channel_value(spinbox: QAbstractSpinBox, name: str) -> float:
    """What a channel box shows, in the units the channel stores."""
    return spinbox.value() / ChannelSettings.params[name].scale


def available_binnings(fm: Any) -> Tuple[int, ...]:
    """The binnings the camera takes."""
    return tuple(int(b) for b in bound_spec(_BINNING, _devices(fm)).choices)


def power_scale(fm: Any) -> Optional[NativeScale]:
    """Full power in the light's own units, as (value, unit), when the driver knows."""
    return _read(
        fm,
        "power units",
        lambda fm: getattr(fm.light_source, "power_native_scale", None),
    )


def gain_scale(fm: Any) -> Optional[NativeScale]:
    """A gain of 100% in the camera's own units, as (value, unit), when known."""
    return _read(
        fm, "gain units", lambda fm: getattr(fm.camera, "gain_native_scale", None)
    )


def in_native_units(percent: float, scale: NativeScale) -> str:
    """``percent`` of ``scale`` in the hardware's units: "0.12 W", or "6 of 16" for a
    camera gain with no unit."""
    full, unit = scale
    value = percent / 100.0 * full
    if unit:
        return f"{value:.3g} {unit}"
    return f"{value:.3g} of {full:.3g}"


class _NativeUnitsTooltip(QObject):
    """Shows a percentage box's value in hardware units in its tooltip, worked out
    when the tooltip opens so it is right however the value was set."""

    def __init__(self, spin: QAbstractSpinBox, scale: NativeScale) -> None:
        super().__init__(spin)
        self._scale = scale
        self._base = spin.toolTip()

    def text(self, spin: QAbstractSpinBox) -> str:
        native = in_native_units(spin.value(), self._scale)
        return f"{self._base}: {native}" if self._base else native

    def eventFilter(self, spin: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.ToolTip:
            QToolTip.showText(event.globalPos(), self.text(spin), spin)
            return True
        return False


def show_native_units(spin: QAbstractSpinBox, scale: Optional[NativeScale]) -> None:
    """Give a percentage box a tooltip in hardware units; without a scale it keeps
    its tooltip as it is."""
    if scale is None:
        return
    tooltip = _NativeUnitsTooltip(spin, scale)
    spin.installEventFilter(tooltip)
    spin._native_units_tooltip = tooltip  # what a test reads, and kept alive
