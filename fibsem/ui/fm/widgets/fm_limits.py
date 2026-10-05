"""The ranges and choices the FM widgets offer, as the connected FM reports them.

Every FM answers ``camera.available_binnings``, ``camera.exposure_time_limits`` and
``light_source.power_limits``; on an FM over devices those come from the device
parameters' metadata. Each function falls back to the widgets' old fixed range when
there is no FM or it can't say, so a widget still opens.

Power and gain are percentages of the hardware's maximum. Where the driver knows what
that maximum is in the hardware's own units (watts, camera gain), the box's tooltip
shows the value in those units too.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional, Tuple

from PyQt5.QtCore import QEvent, QObject
from PyQt5.QtWidgets import QAbstractSpinBox, QToolTip

NativeScale = Tuple[float, Optional[str]]

EXPOSURE_RANGE_MS: Tuple[float, float] = (1.0, 10000.0)
POWER_RANGE_PERCENT: Tuple[float, float] = (0.0, 100.0)
BINNINGS: Tuple[int, ...] = (1, 2, 4, 8)


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


def exposure_range_ms(fm: Any, decimals: int = 1) -> Tuple[float, float]:
    """The exposure times the camera takes, in ms."""
    limits = _read(fm, "exposure limits", lambda fm: fm.camera.exposure_time_limits)
    if limits is None:
        return EXPOSURE_RANGE_MS
    return _in_steps(limits, 1e3, decimals)


def power_range_percent(fm: Any, decimals: int = 1) -> Tuple[float, float]:
    """The light source's power range, in percent of its maximum."""
    limits = _read(fm, "power limits", lambda fm: fm.light_source.power_limits)
    if limits is None:
        return POWER_RANGE_PERCENT
    return _in_steps(limits, 1e2, decimals)


def available_binnings(fm: Any) -> Tuple[int, ...]:
    """The binnings the camera takes."""
    found = _read(fm, "binnings", lambda fm: fm.camera.available_binnings)
    return tuple(int(b) for b in found) if found else BINNINGS


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
