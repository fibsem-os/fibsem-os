"""Turning canvas annotations on and off.

A real-space canvas accumulates things drawn *over* the data: where the stage can
travel, where the holder's grids are, where saved positions sit, a gridbar lattice. Each
is useful and none is useful all the time, and a canvas that draws all of them always is
one you end up reading around rather than reading.

This is the surface for saying which. Deliberately small and deliberately not a canvas
feature: what may be toggled is whatever its owner happens to draw, and the canvas does
not know -- overlays are supplied to it, so the list belongs to whoever supplied them.

Shared rather than written per tab. Three consumers want the same control (the FIB/SEM
overview, the fluorescence overview, and the napari minimap, which has the pattern in
hand-rolled form), and every new overlay after this one -- the current field of view
(FIB-524), a scale label (FIB-523) -- wants somewhere to hang rather than another
always-on annotation. Only the FIB/SEM overview uses it today; the other two are a
straight substitution when someone gets to them (FIB-572).

Not layer controls. Colour and contrast for the image *data* are resolved upstream of
the canvas by whatever composites it, and are FIB-415. These are annotations on top.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QCheckBox, QVBoxLayout, QWidget

# (key, label, shown by default)
OverlayEntry = Tuple[str, str, bool]


class CanvasOverlayControls(QWidget):
    """Checkboxes for a set of named overlays, one row each.

    Emits :attr:`toggled` with the key and its new state. Owners keep their own drawing
    logic and ask :meth:`is_visible` -- this holds the answer, not the overlay, so an
    owner that redraws from scratch (as the overview does on every stage move) does not
    have to reconstruct visibility from whatever artists happen to exist.
    """

    toggled = pyqtSignal(str, bool)

    def __init__(self, entries: Iterable[OverlayEntry], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._boxes: Dict[str, QCheckBox] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        for key, label, shown in entries:
            box = QCheckBox(label)
            box.setChecked(shown)
            # `key=key` binds per iteration; a bare closure would capture the loop
            # variable and report the last key for every box.
            box.toggled.connect(lambda checked, key=key: self.toggled.emit(key, checked))
            layout.addWidget(box)
            self._boxes[key] = box

    def is_visible(self, key: str) -> bool:
        """Whether *key* is shown. Unknown keys are shown, not hidden.

        An overlay whose control was never added is one nobody chose to hide, and the
        alternative -- defaulting to hidden -- makes a forgotten entry look like a
        drawing bug rather than a missing checkbox.
        """
        box = self._boxes.get(key)
        return True if box is None else box.isChecked()

    def set_visible(self, key: str, visible: bool) -> None:
        """Set one overlay's state, emitting :attr:`toggled` if it changed."""
        box = self._boxes.get(key)
        if box is not None:
            box.setChecked(bool(visible))

    def set_enabled(self, key: str, enabled: bool) -> None:
        """Grey out one control without changing what it says.

        For an overlay that has nothing to draw right now -- no holder configured, no
        saved positions. Distinct from unchecking it, which is the user's answer and
        must survive the thing becoming available again.
        """
        box = self._boxes.get(key)
        if box is not None:
            box.setEnabled(bool(enabled))

    def keys(self) -> Tuple[str, ...]:
        return tuple(self._boxes)
