"""A drawing of the chamber, from the stage position the app last heard about.

The quad view's fourth cell. A chamber camera answers "where is the stage and what
is pointing at it", but on a Thermo system reading one takes the imaging channel the
beams and the fluorescence microscope share (`ThermoMicroscope.acquire_chamber_image`),
so it cannot sit on screen refreshing. This draws the same answer from numbers the app
already has: the side view the setup wizard draws (`StageDiagram`), at the stage's
tilt, with the ion column at the system's own angle.

Nothing here reads the microscope. `MicroscopeViewController.update_info` hands it the
position that every stage update already carries, so it is exactly as current as the
STAGE line on the canvases; hovering over it says when that was.

It is a picture, not a readout: the numbers are on the info bar beside it. And it is a
schematic. It knows the columns' angles and the shuttle's pre-tilt; it does not
know where the pole piece, the detectors or the chamber walls are, and its tooltip
says so.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional, Sequence

from PyQt5.QtCore import QRect, QRectF, QSize, Qt
from PyQt5.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QLabel, QStackedWidget, QToolButton, QVBoxLayout, QWidget

from fibsem.structures import FibsemStagePosition
from fibsem.ui.tokens import (
    BORDER_COLOR,
    CANVAS_BG,
    FIB_BEAM_MUTED_COLOUR,
    NEUTRAL_300,
    NEUTRAL_650,
    PRIMARY_ACCENT,
    SEM_BEAM_MUTED_COLOUR,
)
from fibsem.ui.widgets.canvas.stage_map import ZOOM_NAMES, StageMap
from fibsem.ui.widgets.guided_setup_dialog import StageDiagram

_EMPTY_STYLE = "color: #777; font-size: 12px;"
_ZOOM_BUTTON_STYLE = (
    "QToolButton { background: #2b2d31; color: #ddd; border: 1px solid #444;"
    " border-radius: 4px; font-size: 13px; }"
    "QToolButton:hover { border-color: #666; }"
)
# The inset's share of the cell's width, and its height for that width.
_INSET_WIDTH = 0.3
_INSET_ASPECT = 0.72
_INSET_MARGIN = 8
# The smallest the side view is drawn at before it is shrunk into the inset: about
# what its stage and columns need to be laid out without crowding.
_SIDE_RENDER_MIN = QSize(300, 220)
SIDE, MAP = "side", "map"


def is_half_turn(r_radians: float, rotation_reference_degrees: float) -> bool:
    """Whether the stage is turned more than a quarter turn from the reference.

    That is what mirrors the side view, so it is read from the rotation itself rather
    than from the orientation's name: a compustage has no rotation axis and reaches
    FIB by tilting, and a stage between orientations still has a rotation to draw.

    A side view can only be honest at the reference or a half turn from it; anywhere
    else the sample tilts out of the plane of the columns. The stage is at one or the
    other in practice, so in between it snaps to the nearer rather than projecting.
    """
    delta = math.degrees(r_radians) - rotation_reference_degrees
    delta = (delta + 180.0) % 360.0 - 180.0
    return abs(delta) > 90.0


class _Inset(QWidget):
    """The smaller of the two views, drawn over the corner of the larger. A click
    swaps them."""

    def __init__(self, scene: "_Scene") -> None:
        super().__init__(scene)
        self._scene = scene
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Swap the views")

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        try:
            rect = QRectF(self.rect())
            if self._scene.main == SIDE:
                # At the zoom last chosen on the large map: shrinking the map to the
                # corner should not throw away which part of it the user asked for.
                stage_map = self._scene.map
                stage_map.paint_map(painter, rect, stage_map.zoom, detailed=False)
            else:
                # The side view has fixed-size parts and no scale to zoom, so it is
                # drawn at a working size and shrunk, rather than drawn small. Not at
                # the cell's size: its parts sit in the middle of a big panel, and
                # shrinking all of it leaves them a speck.
                diagram = self._scene.diagram
                size = QSize(
                    max(_SIDE_RENDER_MIN.width(), 2 * self.width()),
                    max(_SIDE_RENDER_MIN.height(), 2 * self.height()),
                )
                hidden_size = diagram.size()
                diagram.resize(size)
                pixmap = QPixmap(size)
                diagram.render(pixmap)
                diagram.resize(hidden_size)
                painter.fillRect(rect, QColor(CANVAS_BG))
                scaled = pixmap.scaled(
                    self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                painter.drawPixmap(
                    (self.width() - scaled.width()) // 2,
                    (self.height() - scaled.height()) // 2,
                    scaled,
                )
            painter.setPen(QPen(QColor(BORDER_COLOR), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        finally:
            painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.button() == Qt.LeftButton:
            self._scene.swap()
        event.accept()


class _Scene(QWidget):
    """The side view and the stage map: one fills the cell, the other is an inset in
    its top-left corner, where the side view leaves room at every orientation."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.diagram = StageDiagram(pre_tilt=0.0, parent=self)
        self.diagram.setMinimumHeight(0)
        self.diagram.set_readout_visible(False)
        ChamberView._mute(self.diagram)
        self.map = StageMap(self)
        self.main = SIDE

        self.inset = _Inset(self)
        self.zoom_out_button = self._zoom_button("−", self.map.zoom_out)
        self.zoom_in_button = self._zoom_button("+", self.map.zoom_in)
        self.map.zoom_changed.connect(self._refresh_zoom_tooltips)
        self._refresh_zoom_tooltips()
        self._layout()

    def _zoom_button(self, text: str, slot) -> QToolButton:
        button = QToolButton(self)
        button.setText(text)
        button.setFixedSize(26, 26)
        button.setStyleSheet(_ZOOM_BUTTON_STYLE)
        button.clicked.connect(slot)
        return button

    def _refresh_zoom_tooltips(self, *_args) -> None:
        zoom = self.map.zoom
        self.zoom_out_button.setToolTip(
            f"Zoom out to {ZOOM_NAMES[zoom - 1].lower()}" if zoom > 0 else "Zoomed out"
        )
        self.zoom_in_button.setToolTip(
            f"Zoom in to {ZOOM_NAMES[zoom + 1].lower()}"
            if zoom + 1 < len(ZOOM_NAMES)
            else "Zoomed in"
        )
        self.inset.update()

    def swap(self) -> None:
        self.main = MAP if self.main == SIDE else SIDE
        self._layout()

    def refresh(self) -> None:
        self.diagram.update()
        self.map.update()
        self.inset.update()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._layout()

    def _layout(self) -> None:
        full = self.rect()
        # Both keep the full size whether shown or not: the hidden one is what the
        # inset draws, and the side view is rendered at this size to be shrunk.
        self.diagram.setGeometry(full)
        self.map.setGeometry(full)
        self.diagram.setVisible(self.main == SIDE)
        self.map.setVisible(self.main == MAP)

        width = max(int(full.width() * _INSET_WIDTH), 60)
        self.inset.setGeometry(
            QRect(_INSET_MARGIN, _INSET_MARGIN, width, int(width * _INSET_ASPECT))
        )
        self.inset.raise_()

        on_map = self.main == MAP
        for index, button in enumerate((self.zoom_in_button, self.zoom_out_button)):
            button.setVisible(on_map)
            button.move(
                full.width() - _INSET_MARGIN - (index + 1) * (button.width() + 4) + 4,
                full.height() - _INSET_MARGIN - button.height(),
            )
            button.raise_()
        self.inset.update()


class ChamberView(QWidget):
    """The chamber, drawn: a side view of the stage and the columns, and a map of the
    stage from above. Follows the stage position."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.scene = _Scene()
        self.diagram = self.scene.diagram
        self.map = self.scene.map

        self._empty = QLabel("No stage position yet", alignment=Qt.AlignCenter)
        self._empty.setStyleSheet(_EMPTY_STYLE)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._empty)
        self._stack.addWidget(self.scene)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._stack)

    @staticmethod
    def _mute(diagram: StageDiagram) -> None:
        """Quieter than the wizard's palette: this sits beside the image canvases.

        On the canvas background with no frame of its own, beams softened, and the
        shuttle in the quad view's selection blue rather than the wizard's saturated
        gradient, which was the loudest thing on the tab.
        """
        diagram.BACKGROUND_COLOUR = CANVAS_BG
        diagram.FRAME_COLOUR = None
        diagram.SEM_COLOUR = SEM_BEAM_MUTED_COLOUR
        diagram.FIB_COLOUR = FIB_BEAM_MUTED_COLOUR
        diagram.PLATE_COLOUR = NEUTRAL_650
        diagram.SHUTTLE_COLOURS = (PRIMARY_ACCENT, PRIMARY_ACCENT)
        diagram.GRID_COLOUR = NEUTRAL_300
        diagram.WEDGE_COLOUR = NEUTRAL_300

    @property
    def has_position(self) -> bool:
        return self._stack.currentWidget() is self.scene

    def set_microscope(self, microscope) -> None:
        """The configuration the map is drawn from: limits, slots, orientations."""
        self.map.set_microscope(microscope)

    def set_positions(self, positions: Sequence[FibsemStagePosition]) -> None:
        """The lamellae the map marks at grid zoom."""
        self.map.set_positions(positions)
        self.scene.inset.update()

    def set_stage(
        self,
        stage_position: FibsemStagePosition,
        orientation: str,
        pre_tilt: float,
        column_tilt: float,
        rotation_reference: float,
    ) -> None:
        """Draw the stage at *stage_position*; angles in degrees, the position's in radians.

        A position with no tilt leaves the drawing as it was: there is nothing to draw
        it at, and a guess would be worse than the last real one.
        """
        if stage_position is None or stage_position.t is None:
            return
        mirrored = stage_position.r is not None and is_half_turn(
            stage_position.r, rotation_reference
        )
        self.diagram.set_pre_tilt(pre_tilt)
        self.diagram.set_column_tilt(column_tilt)
        self.diagram.set_orientation(
            name="" if orientation in (None, "NONE") else orientation,
            stage_tilt=math.degrees(stage_position.t),
            mirrored=mirrored,
        )
        self.map.set_stage(stage_position)
        self._stack.setCurrentWidget(self.scene)
        self.scene.setToolTip(
            f"Schematic, not a camera. Stage as of {datetime.now():%H:%M:%S}."
        )
        self.scene.refresh()
