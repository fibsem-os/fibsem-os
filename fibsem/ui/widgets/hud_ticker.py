"""Minimal HUD ticker overlay.

A single-line semi-transparent strip pinned to the bottom of a parent widget,
showing key stage and acquisition metadata from a FibsemImage.

Usage::

    ticker = HUDTicker(parent=image_view)
    ticker.set_image(fibsem_image)

The ticker resizes itself to always match the parent's width.
"""

import math
from typing import Optional

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath

from fibsem.structures import FibsemImage
from fibsem.ui.widgets.drag_distance import _fmt_distance

_BG = QColor(10, 10, 12, 190)
_TEXT = QColor(210, 210, 210, 230)
_DIVIDER = QColor(80, 80, 90, 180)
_LABEL = QColor(120, 170, 255, 200)   # blue-ish key labels
_HEIGHT = 26
_PADDING = 12
_DIVIDER_W = 1


def _deg(radians: Optional[float]) -> str:
    if radians is None:
        return "—"
    return f"{math.degrees(radians):.1f}°"

def _um(metres: Optional[float]) -> str:
    if metres is None:
        return "—"
    return _fmt_distance(metres)

def _kv(volts: Optional[float]) -> str:
    if volts is None:
        return "—"
    return f"{volts / 1e3:.2f} kV"

def _na(amps: Optional[float]) -> str:
    if amps is None:
        return "—"
    abs_a = abs(amps)
    if abs_a < 1e-9:
        return f"{amps * 1e12:.0f} pA"
    return f"{amps * 1e9:.2f} nA"


class HUDTicker(QtWidgets.QWidget):
    """Single-line HUD strip overlaid at the bottom of the parent widget."""

    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setFixedHeight(_HEIGHT)
        self._segments: list[tuple[str, str]] = []  # [(label, value), ...]
        self._refit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_image(self, image: Optional[FibsemImage]) -> None:
        """Populate the ticker from a FibsemImage's metadata."""
        self._segments = []
        if image is None:
            self.update()
            return

        meta = image.metadata
        state = meta.microscope_state

        # --- Stage ---
        if state and state.stage_position:
            sp = state.stage_position
            self._segments += [
                ("X", _um(sp.x)),
                ("Y", _um(sp.y)),
                ("Z", _um(sp.z)),
                ("T", _deg(sp.t)),
            ]
        else:
            self._segments += [("X", "—"), ("Y", "—"), ("Z", "—"), ("T", "—")]

        self._segments.append(None)  # divider

        # --- Acquisition ---
        s = meta.image_settings
        self._segments += [
            ("HFW",  _um(s.hfw) if s else "—"),
            ("px",   _fmt_distance(meta.pixel_size.x) + "/px"),
            ("res",  f"{s.resolution[0]}×{s.resolution[1]}" if s else "—"),
        ]

        # Timestamp
        if state and state.timestamp:
            import datetime
            ts = datetime.datetime.fromtimestamp(state.timestamp).strftime("%H:%M:%S")
            self._segments.append(None)
            self._segments.append(("t", ts))

        self.update()

    # ------------------------------------------------------------------
    # Resize with parent
    # ------------------------------------------------------------------

    def _refit(self) -> None:
        if self.parent():
            pw = self.parent().width()
            self.setGeometry(0, self.parent().height() - _HEIGHT, pw, _HEIGHT)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()

        # Background pill (rounded top corners only)
        path = QPainterPath()
        path.addRoundedRect(0, 0, w, h, 4, 4)
        painter.fillPath(path, _BG)

        font = QFont("Consolas", 9)
        painter.setFont(font)
        fm = painter.fontMetrics()

        x = _PADDING

        for seg in self._segments:
            if seg is None:
                # Divider
                painter.setPen(_DIVIDER)
                mid = h // 2
                painter.drawLine(x, mid - 7, x, mid + 7)
                x += _DIVIDER_W + _PADDING
                continue

            label, value = seg

            # Label
            painter.setPen(_LABEL)
            lw = fm.horizontalAdvance(label + " ")
            painter.drawText(x, 0, lw, h, Qt.AlignVCenter | Qt.AlignLeft, label)
            x += lw

            # Value
            painter.setPen(_TEXT)
            vw = fm.horizontalAdvance(value + "  ")
            painter.drawText(x, 0, vw, h, Qt.AlignVCenter | Qt.AlignLeft, value)
            x += vw

            if x > w - _PADDING:
                break

        painter.end()


class HUDTickerMixin:
    """Mixin that adds a HUD ticker to the bottom of any QWidget."""

    def setup_hud_ticker(self) -> None:
        self._hud_ticker = HUDTicker(parent=self)
        self._hud_ticker.show()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "_hud_ticker"):
            self._hud_ticker._refit()
