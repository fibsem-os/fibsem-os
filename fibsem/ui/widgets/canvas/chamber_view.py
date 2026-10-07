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
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QStackedWidget, QVBoxLayout, QWidget

from fibsem.structures import FibsemStagePosition
from fibsem.ui.tokens import (
    CANVAS_BG,
    FIB_BEAM_MUTED_COLOUR,
    NEUTRAL_300,
    NEUTRAL_650,
    PRIMARY_ACCENT,
    SEM_BEAM_MUTED_COLOUR,
)
from fibsem.ui.widgets.guided_setup_dialog import StageDiagram

_EMPTY_STYLE = "color: #777; font-size: 12px;"


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


class ChamberView(QWidget):
    """Side view of the stage and the columns, following the stage position."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.diagram = StageDiagram(pre_tilt=0.0)
        self.diagram.setMinimumHeight(0)
        self.diagram.set_readout_visible(False)
        self._mute(self.diagram)

        self._empty = QLabel("No stage position yet", alignment=Qt.AlignCenter)
        self._empty.setStyleSheet(_EMPTY_STYLE)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._empty)
        self._stack.addWidget(self.diagram)

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
        return self._stack.currentWidget() is self.diagram

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
        self._stack.setCurrentWidget(self.diagram)
        self.diagram.setToolTip(
            f"Schematic, not a camera. Stage as of {datetime.now():%H:%M:%S}."
        )
