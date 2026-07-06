"""Reusable contrast / gamma popover for image canvases.

A floating ``QFrame`` with Min / Max / Gamma sliders + Reset, matching the look
of the coincidence viewer's histogram panel. The host owns its image; this
widget only holds the contrast/gamma state, emits :attr:`changed` when it moves,
and provides :meth:`normalize` + :meth:`apply` to turn a frame into display data.

Typical use::

    self.ctrl = ContrastGammaControl(self)          # floats over `self`
    self.ctrl.changed.connect(self.redraw)
    norm = ContrastGammaControl.normalize(image.data)   # cache once
    # in redraw:
    if self.ctrl.is_default():
        data, clim = image.data, None
    else:
        data, clim = self.ctrl.apply(norm), (0.0, 1.0)
    # toggled by a checkable button:
    button.toggled.connect(lambda on: self.ctrl.set_open(on, button))
"""

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)
from superqt import QDoubleSlider

from fibsem.autofunctions.gamma import apply_gamma

_PANEL_STYLE = (
    "QFrame { background: rgba(30,33,36,230); border: 1px solid #555;"
    " border-radius: 4px; }"
    "QLabel { color: #d1d2d4; font-size: 10px; background: transparent; border: none; }"
    "QPushButton { background: rgba(60,63,70,200); border: 1px solid #666;"
    " border-radius: 3px; color: #d1d2d4; font-size: 10px; padding: 2px 8px; }"
    "QPushButton:hover { background: rgba(80,83,90,220); }"
)


class ContrastGammaControl(QFrame):
    """Floating contrast/gamma popover. Emits ``changed`` on any adjustment."""

    changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(_PANEL_STYLE)
        self.setFixedWidth(220)
        self.setVisible(False)
        self._min, self._max, self._gamma = 0.0, 1.0, 1.0
        self._anchor = None

        form = QFormLayout(self)
        form.setContentsMargins(8, 6, 8, 6)
        form.setSpacing(3)

        def row(lo, hi, default, step, cb):
            sld = QDoubleSlider(Qt.Horizontal)
            sld.setRange(lo, hi)
            sld.setSingleStep(step)
            sld.setValue(default)
            lbl = QLabel(f"{default:.2f}")
            lbl.setFixedWidth(34)
            cont = QWidget()
            h = QHBoxLayout(cont)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(4)
            h.addWidget(sld)
            h.addWidget(lbl)
            sld.valueChanged.connect(lambda v: (lbl.setText(f"{v:.2f}"), cb(v)))
            return sld, cont

        self.sld_min, row_min = row(0.0, 1.0, 0.0, 0.01, self._on_min)
        self.sld_max, row_max = row(0.0, 1.0, 1.0, 0.01, self._on_max)
        self.sld_gamma, row_gamma = row(0.1, 3.0, 1.0, 0.05, self._on_gamma)
        form.addRow("Min", row_min)
        form.addRow("Max", row_max)
        form.addRow("Gamma", row_gamma)
        btn_reset = QPushButton("Reset")
        btn_reset.clicked.connect(self.reset)
        form.addRow("", btn_reset)

    # --- state --------------------------------------------------------------

    @property
    def contrast_min(self) -> float:
        return self._min

    @property
    def contrast_max(self) -> float:
        return self._max

    @property
    def gamma(self) -> float:
        return self._gamma

    def is_default(self) -> bool:
        """True when no adjustment is applied (so the host can show the raw image)."""
        return self._min == 0.0 and self._max == 1.0 and self._gamma == 1.0

    def reset(self) -> None:
        self._min, self._max, self._gamma = 0.0, 1.0, 1.0
        self.sld_min.setValue(0.0)
        self.sld_max.setValue(1.0)
        self.sld_gamma.setValue(1.0)
        self.changed.emit()

    # --- processing ---------------------------------------------------------

    @staticmethod
    def normalize(frame: np.ndarray) -> np.ndarray:
        """Scale a raw frame to ``[0, 1]`` (cache this; pass it to :meth:`apply`)."""
        f = frame.astype(np.float32)
        lo, hi = float(f.min()), float(f.max())
        return (f - lo) / (hi - lo) if hi > lo else np.zeros_like(f)

    def apply(self, norm: np.ndarray) -> np.ndarray:
        """Clip + rescale + gamma a normalized ``[0, 1]`` frame for display."""
        lo, hi = self._min, self._max
        out = np.clip(norm, lo, hi)
        if hi > lo:
            out = (out - lo) / (hi - lo)
        if self._gamma != 1.0:
            out = apply_gamma(out, self._gamma)
        return out

    # --- popover visibility -------------------------------------------------

    def set_open(self, open_: bool, anchor=None) -> None:
        """Show/hide the popover; when opening, anchor it under ``anchor``."""
        if anchor is not None:
            self._anchor = anchor
        self.setVisible(open_)
        if open_:
            self.reposition()
            self.raise_()

    def reposition(self) -> None:
        """Re-anchor under the anchor button at the host's top-right."""
        parent = self.parentWidget()
        if parent is None:
            return
        self.adjustSize()
        x = parent.width() - self.width() - 4
        if self._anchor is not None:
            y = self._anchor.y() + self._anchor.height() + 4
        else:
            y = 4
        self.move(max(4, x), y)

    # --- handlers -----------------------------------------------------------

    def _on_min(self, val: float) -> None:
        if val >= self._max:
            val = max(0.0, self._max - 0.01)
            self.sld_min.setValue(val)
        self._min = val
        self.changed.emit()

    def _on_max(self, val: float) -> None:
        if val <= self._min:
            val = min(1.0, self._min + 0.01)
            self.sld_max.setValue(val)
        self._max = val
        self.changed.emit()

    def _on_gamma(self, val: float) -> None:
        self._gamma = val
        self.changed.emit()
