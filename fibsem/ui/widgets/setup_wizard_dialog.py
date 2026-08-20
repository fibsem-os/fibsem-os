"""The first-run setup wizard: from a fresh install to a configuration that connects.

Five steps for everybody; the Arctis is shown the third rather than asked it:

1. **Microscope** -- first, because the answer decides which of the rest are still
   questions.
2. **Computer and connection** -- never skipped. The address is a property of the site
   that no shipped file can imply, and step 3 reads the stage, so the connection has to
   be *proven* here rather than merely typed.
3. **Stage configuration** -- read-only for the Arctis, whose compustage has no
   reference rotation to measure and no pre-tilted shuttle; the step is shown and
   explained rather than skipped, so the diagram can say why there is nothing to enter.
   Every other model prefills from its shipped configuration and still asks: those
   values belong to the shuttle fitted and to how a sample is loaded at that site, not
   to the model.
4. **Folders**.
5. **Review and save**.

The answers live in :mod:`fibsem.setup_wizard`, which is Qt-free and does the writing.
This module is the collection of them and nothing else.

Connections are never opened alongside an existing one. If the application is already
connected, the wizard borrows that microscope to read the stage and leaves it alone
otherwise; if it is not, the wizard opens its own and closes it on the way out. Two
AutoScript clients from one process is not a thing to find out about on a first run.
"""

from __future__ import annotations

import logging
import math
import os
from typing import List, Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal

from fibsem import config as cfg
from fibsem import setup_wizard as wizard
from fibsem.setup_wizard import (
    AVAILABLE_MANUFACTURERS,
    COMPUTER_LOCATIONS,
    MICROSCOPE_MODELS,
    MODEL_ICON,
    SetupChoices,
)
from fibsem.ui import stylesheets
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.qt.threading import FunctionWorker
from fibsem.ui.tokens import (
    BORDER_COLOR,
    NEUTRAL_500,
    PANEL_COLOR,
    PRIMARY_COLOR,
    TEXT_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
)
from fibsem.ui.utils import install_wheel_blocker_recursive
from fibsem.ui.widgets.custom_widgets import QDirectoryLineEdit, QFileLineEdit

logger = logging.getLogger(__name__)

OK_GREEN = "#4ec26a"
FAIL_RED = "#e05252"
# The fluorescence objective's barrel, which on a compustage looks *up* at the sample
# from underneath -- at the FM orientation the stage has flipped the grid over. Black,
# like the real thing, and outlined so it reads against a panel nearly as dark.
FM_BARREL_COLOUR = "#0d0f12"

# Painted on top of a panel, which already has the background and the border. Without
# this every child label repaints the panel's rounded rectangle as a square one, since
# the application stylesheet sets a background on bare QWidget.
ON_PANEL = "background: transparent; border: none;"

# The shuttle's base, below the ramp, in the stage diagram. The body's height follows
# the ramp rather than being fixed: a wedge is only as deep as its slope needs, so at
# zero pre-tilt this collapses to a thin plate carrying the grid, which is what a
# holder without a wedge in it actually looks like.
BASE_THICKNESS = 8

# The pre-tilt wedge is annotated on top of the shuttle body, so it needs to read
# against the accent colour rather than in it.
WEDGE_COLOUR = "#dfe5ec"

STEP_MICROSCOPE = 0
STEP_CONNECTION = 1
STEP_STAGE = 2
STEP_FOLDERS = 3
STEP_REVIEW = 4

STEP_TITLES = [
    "Microscope",
    "Computer and connection",
    "Stage configuration",
    "Folders",
    "Review and save",
]

STEP_SUBTITLES = [
    "Picks the configuration to start from. Every value stays editable afterwards.",
    "Where this runs, and how it reaches the instrument. The stage step reads from the "
    "microscope, so this comes first.",
    "The two values that describe how the sample sits on the stage.",
    "Two different places: where this configuration is saved, and where experiments "
    "are created.",
    "These answers are written as a new microscope configuration.",
]


# ---------------------------------------------------------------------------
# Small painted pieces
# ---------------------------------------------------------------------------


def _label(
    text: str,
    size: int = 11,
    color: str = TEXT_COLOR,
    bold: bool = False,
    wrap: bool = False,
) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    label.setWordWrap(wrap)
    label.setStyleSheet(
        f"{ON_PANEL} color: {color}; font-size: {size}px;"
        f"{' font-weight: bold;' if bold else ''}"
    )
    return label


def _panel(name: str = "wizardPanel") -> QtWidgets.QFrame:
    frame = QtWidgets.QFrame()
    frame.setObjectName(name)
    frame.setStyleSheet(
        f"QFrame#{name} {{ background: {PANEL_COLOR};"
        f" border: 1px solid {BORDER_COLOR}; border-radius: 6px; }}"
    )
    return frame


class ChoiceCard(QtWidgets.QFrame):
    """One of a set of mutually exclusive answers, drawn as a card.

    Focusable and operable from the keyboard. A card that can only be clicked looks
    the same in a screenshot and is unusable to anyone driving the dialog by tab and
    space, which on an instrument PC with a trackball is not a rare way to work.
    """

    clicked = pyqtSignal()

    def __init__(self, title: str, summary: str, icon: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("choiceCard")
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self._icon_name = icon
        self._selected = False

        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(12, 10, 12, 10)
        row.setSpacing(10)

        self._icon = QtWidgets.QLabel()
        self._icon.setStyleSheet(ON_PANEL)
        self._icon.setFixedSize(22, 22)
        row.addWidget(self._icon, 0, Qt.AlignVCenter)

        column = QtWidgets.QVBoxLayout()
        column.setSpacing(2)
        self._title = _label(title, 12, TEXT_COLOR, bold=True)
        self._summary = _label(summary, 10, TEXT_MUTED_COLOR, wrap=True)
        column.addWidget(self._title)
        column.addWidget(self._summary)
        row.addLayout(column, 1)

        self._tick = QtWidgets.QLabel()
        self._tick.setStyleSheet(ON_PANEL)
        self._tick.setFixedSize(18, 18)
        row.addWidget(self._tick, 0, Qt.AlignVCenter)

        self.set_selected(False)

    def is_selected(self) -> bool:
        return self._selected

    def set_selected(self, selected: bool) -> None:
        self._selected = bool(selected)
        border = PRIMARY_COLOR if selected else BORDER_COLOR
        fill = "#12293d" if selected else PANEL_COLOR
        self.setStyleSheet(
            f"QFrame#choiceCard {{ background: {fill};"
            f" border: {2 if selected else 1}px solid {border}; border-radius: 6px; }}"
            f"QFrame#choiceCard:focus {{ border: 2px solid {PRIMARY_COLOR}; }}"
        )
        self._title.setStyleSheet(
            f"{ON_PANEL} color: {TEXT_STRONG_COLOR if selected else TEXT_COLOR};"
            " font-size: 12px; font-weight: bold;"
        )
        if self._icon_name:
            colour = PRIMARY_COLOR if selected else NEUTRAL_500
            self._icon.setPixmap(fibsem_icon(self._icon_name, color=colour).pixmap(22, 22))
        if selected:
            self._tick.setPixmap(
                fibsem_icon("mdi:check-circle", color=PRIMARY_COLOR).pixmap(18, 18)
            )
        else:
            self._tick.clear()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.setFocus(Qt.MouseFocusReason)
        self.clicked.emit()
        super().mousePressEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.key() in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
            self.clicked.emit()
            return
        super().keyPressEvent(event)


class StatusCard(QtWidgets.QFrame):
    """An icon, a headline and a sentence. What happened, and what it means."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("statusCard")
        self.setStyleSheet(
            f"QFrame#statusCard {{ background: {PANEL_COLOR};"
            f" border: 1px solid {BORDER_COLOR}; border-radius: 6px; }}"
        )
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(12, 10, 12, 10)
        row.setSpacing(10)

        self._icon = QtWidgets.QLabel()
        self._icon.setStyleSheet(ON_PANEL)
        self._icon.setFixedSize(20, 20)
        row.addWidget(self._icon, 0, Qt.AlignTop)

        column = QtWidgets.QVBoxLayout()
        column.setSpacing(2)
        self._title = _label("", 11, TEXT_COLOR, bold=True)
        self._detail = _label("", 10, TEXT_MUTED_COLOR, wrap=True)
        column.addWidget(self._title)
        column.addWidget(self._detail)
        row.addLayout(column, 1)

    def show_status(self, icon: str, colour: str, title: str, detail: str) -> None:
        self._icon.setPixmap(fibsem_icon(icon, color=colour).pixmap(20, 20))
        self._title.setStyleSheet(
            f"{ON_PANEL} color: {colour}; font-size: 11px; font-weight: bold;"
        )
        self._title.setText(title)
        self._detail.setText(detail)
        self.setVisible(True)


class StageDiagram(QtWidgets.QWidget):
    """Side view of stage, shuttle and both beams, with the pre-tilt as a wedge.

    Drawn from the numbers on the page rather than shipped as an image. A picture of
    35 degrees sitting beside a spin box reading 27 is worse than no picture: it is
    the one thing on the step a user can check their real stage against, so it has to
    be checking the value they actually entered.

    **This is the SEM orientation**, which also covers MILLING. ``microscope.py``
    defines the three: SEM is ``rotation_reference`` with the stage at
    ``shuttle_pre_tilt``; MILLING shares that rotation and differs only in tilt; FIB is
    ``rotation_180`` -- a half turn, which reverses the pre-tilt's sense relative to the
    beams. So SEM and MILLING are the same side view and FIB cannot be, which is why
    the readout says so rather than letting the picture imply otherwise.

    The SEM orientation is drawn because it is the checkable one: set the stage to your
    pre-tilt and the sample comes flat, square to the electron beam. Someone can hold
    that against the real instrument and see immediately whether the number they typed
    is right, which is the whole reason the picture is here.

    The stage tilt is whatever was read from the instrument; until something is read it
    shows the pre-tilt, which is the SEM orientation itself.
    """

    SEM_COLOUR = "#4ec26a"
    FIB_COLOUR = "#e8a020"
    # Degrees from vertical, and drawn to the right of the SEM column.
    FIB_ANGLE = 52.0
    # The offset FM objective's own axis. It coincides with the ion column today, but it
    # is a separate mount and not obliged to: kept as its own constant so the day it
    # differs is one edit here rather than a hunt through code that assumed they were
    # the same number.
    FM_ANGLE = 52.0

    def effective_stage_tilt(self) -> float:
        """The stage tilt to draw: what was read, or the SEM orientation."""
        return self._pre_tilt if self._stage_tilt is None else self._stage_tilt

    def ramp_tilt(self) -> float:
        """The shuttle's wedge, as this view sees it.

        Negated by a half turn, because rotating the stage 180 degrees about its
        vertical axis mirrors the whole profile left to right. That single sign is why
        `column_tilt - shuttle_pre_tilt` is the tilt that squares the sample to the ion
        beam: mirrored, the wedge adds to the stage tilt instead of cancelling it.
        """
        return -self._pre_tilt if self._mirrored else self._pre_tilt

    def surface_tilt(self) -> float:
        """Degrees the sample surface is turned clockwise on screen.

        Unmirrored, the shuttle's pre-tilt works *against* the stage tilt, so the two
        cancel at the SEM orientation. Everything the diagram draws rotates by this or
        by the stage tilt alone, so the sign lives in one place.
        """
        return self.effective_stage_tilt() - self.ramp_tilt()

    def sample_normal(self) -> tuple:
        """The direction the sample faces, in screen coordinates (y down)."""
        angle = math.radians(self.surface_tilt())
        return (math.sin(angle), -math.cos(angle))

    def milling_angle(self) -> float:
        """The angle between the ion beam and the sample surface.

        The quantity `convert_stage_tilt_to_milling_angle` computes as
        `90 - column_tilt + stage_tilt - pretilt`. Taken from the drawn geometry rather
        than that formula because the formula assumes the unmirrored frame: at the FIB
        orientation the stage has turned 180 degrees, which flips the pre-tilt's sign,
        and the formula would report 20 degrees for a beam that is actually square on.

        Absolute, because a compustage mills the *back* of the grid -- there the beam is
        normal to the surface from behind, which is still a 90 degree approach.
        """
        normal = self.sample_normal()
        fib = self.beam_direction(self.FIB_ANGLE)
        dot = normal[0] * fib[0] + normal[1] * fib[1]
        between = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
        return abs(90.0 - between)

    @classmethod
    def beam_direction(cls, degrees_from_vertical: float) -> tuple:
        """From the sample toward a beam's source, in screen coordinates."""
        angle = math.radians(degrees_from_vertical)
        return (math.sin(angle), -math.cos(angle))

    def __init__(self, pre_tilt: float = 35.0, stage_tilt=None, mirrored: bool = False,
                 show_fm: bool = False, orientation: str = "", parent=None):
        super().__init__(parent)
        self._pre_tilt = pre_tilt
        # None means nothing has been read yet, which is different from a stage that
        # was read at zero.
        self._stage_tilt = stage_tilt
        # A half turn about the vertical stage axis is a left-right mirror in this
        # view, which is what makes the FIB orientation drawable here at all -- see
        # `ramp_tilt`. False everywhere the wizard uses this; the reference material
        # is what needs the other orientations.
        self._mirrored = mirrored
        self._show_fm = show_fm
        self._objective_inserted = True
        # Set for an offset FM mount, where the stage translates in x to a separate
        # station. None means the beams and the sample are at the same point.
        self._station_offset_mm = None
        self._orientation = orientation
        self.setMinimumHeight(212)

    def set_orientation(self, name: str = "", stage_tilt=None, mirrored: bool = False,
                        show_fm: bool = False, objective_inserted: bool = True) -> None:
        """Show a named orientation, rather than only the SEM one."""
        self._orientation = name
        self._stage_tilt = stage_tilt
        self._mirrored = mirrored
        self._show_fm = show_fm
        self._objective_inserted = objective_inserted
        self.update()

    def set_pre_tilt(self, degrees: float) -> None:
        self._pre_tilt = float(degrees)
        self.update()

    def set_stage_tilt(self, degrees: float) -> None:
        self._stage_tilt = float(degrees)
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt naming
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        width, height = self.width(), self.height()
        painter.fillRect(self.rect(), QtGui.QColor(PANEL_COLOR))
        painter.setPen(QtGui.QPen(QtGui.QColor(BORDER_COLOR), 1))
        painter.drawRoundedRect(0, 0, width - 1, height - 1, 6, 6)

        cx, cy = width * 0.56, height * 0.66
        total = self.surface_tilt()

        # Read-out first, in the top-left corner. At the bottom it sat where the stage
        # plate and the shuttle swing as the angles change, and a caption crossed by
        # the thing it describes is worse than one that is merely early.
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(TEXT_MUTED_COLOR))
        lines = [
            f"shuttle pre-tilt   {self._pre_tilt:g}°",
            f"stage tilt   {self.effective_stage_tilt():g}°",
            "sample square to the electron beam"
            if abs(total) < 0.05
            else f"{abs(total):g}° from the SEM orientation",
        ]
        # No FIB at an offset station, so no angle to it worth quoting.
        if self._station_offset_mm is None:
            lines.append(f"milling angle   {self.milling_angle():.0f}°   (FIB to sample)")
        if self._orientation:
            lines.insert(0, f"{self._orientation} orientation"
                         + ("   ·   rotated 180°" if self._mirrored else ""))
        else:
            lines.append("FIB orientation is a half turn away")
        for index, line in enumerate(lines):
            painter.drawText(
                QtCore.QRectF(12, 10 + index * 14, width - 24, 14),
                Qt.AlignLeft | Qt.AlignVCenter,
                line,
            )

        # Beams first, so the shuttle drawn over them occludes whatever falls below the
        # sample surface. A beam that carries on through the holder reads as passing
        # into it rather than landing on it.
        if self._station_offset_mm is None:
            self._draw_beam(painter, cx, cy, 0, self.SEM_COLOUR, "SEM")
            self._draw_beam(painter, cx, cy, self.FIB_ANGLE, self.FIB_COLOUR,
                            f"FIB  {self.FIB_ANGLE:g}°")
            if self._show_fm:
                self._draw_objective(painter, cx, cy, 0.0, from_below=True,
                                     inserted=self._objective_inserted)
        else:
            # The columns are not here. Drawing them at the sample would put three
            # instruments at one point when the stage has driven 50 mm away from two
            # of them -- the whole thing this mount does.
            self._draw_objective(painter, cx, cy, self.FM_ANGLE, from_below=False,
                                 inserted=self._objective_inserted)
            self._draw_station_offset(painter, cx, cy, width, height)

        # Everything below is drawn in the stage's frame, with the sample at the origin.
        # The shuttle sits *flat* on the plate and carries its pre-tilt in its own
        # shape -- a low front, a ramp, a raised shelf behind, which is the profile a
        # pre-tilted shuttle actually has. Rotating a plain bar by pre-tilt + stage tilt
        # arrives at the same sample angle while drawing a holder nobody would
        # recognise.
        painter.save()
        painter.translate(cx, cy)
        painter.rotate(self.effective_stage_tilt())

        # The ramp, rising to the right by the pre-tilt, with the sample seated on it.
        ramp = math.radians(self.ramp_tilt())
        half = 26.0
        low = QtCore.QPointF(-half * math.cos(ramp), half * math.sin(ramp))
        high = QtCore.QPointF(half * math.cos(ramp), -half * math.sin(ramp))
        front_x, back_x = low.x() - 15, high.x() + 15
        plate_y = BASE_THICKNESS + half * abs(math.sin(ramp))

        painter.setPen(QtGui.QPen(QtGui.QColor(NEUTRAL_500), 2))
        painter.drawLine(-95, int(plate_y), 95, int(plate_y))
        painter.setPen(QtGui.QPen(QtGui.QColor(NEUTRAL_500), 1, Qt.DashLine))
        painter.drawLine(-95, int(plate_y) + 8, 95, int(plate_y) + 8)

        body = QtGui.QPolygonF([
            QtCore.QPointF(front_x, plate_y),
            QtCore.QPointF(front_x, low.y()),
            low,
            high,
            QtCore.QPointF(back_x, high.y()),
            QtCore.QPointF(back_x, plate_y),
        ])
        gradient = QtGui.QLinearGradient(front_x, 0, back_x, 0)
        gradient.setColorAt(0, QtGui.QColor(PRIMARY_COLOR))
        gradient.setColorAt(1, QtGui.QColor("#0a5c9e"))
        painter.setPen(Qt.NoPen)
        painter.setBrush(gradient)
        painter.drawPolygon(body)

        # The grid, lying on the ramp rather than on top of the block.
        painter.save()
        painter.rotate(-self.ramp_tilt())
        painter.setBrush(QtGui.QColor("#d9dde3"))
        painter.drawRoundedRect(QtCore.QRectF(-13, -4, 26, 4), 1, 1)
        painter.restore()

        # The wedge, annotated where it physically is: at the foot of the ramp, between
        # the shuttle's own base and the face the sample sits on. Measuring it out in
        # open space was readable but described an angle nothing in the picture had.
        label_at = None
        if self._pre_tilt:
            radius = 34
            # Light, not the accent: this is drawn over the shuttle body, which is
            # the accent colour, so blue on blue was unreadable. Reads as a
            # dimension line over the part, which is what it is.
            painter.setPen(QtGui.QPen(QtGui.QColor(WEDGE_COLOUR), 1, Qt.DotLine))
            painter.setBrush(Qt.NoBrush)
            # A half turn swaps which end of the ramp is its foot, so the wedge has to
            # move with it; drawn from the wrong end it floats off the holder entirely.
            span = self.ramp_tilt()
            foot = low if span >= 0 else high
            start = 0.0 if span >= 0 else 180.0
            reach = radius + 12 if span >= 0 else -(radius + 12)
            painter.drawLine(foot, QtCore.QPointF(foot.x() + reach, foot.y()))
            painter.drawArc(
                QtCore.QRectF(foot.x() - radius, foot.y() - radius, 2 * radius, 2 * radius),
                int(start * 16),
                int(span * 16),
            )
            # Mapped out of the rotated frame, so the text itself stays upright -- a
            # label that tilts with the stage is the one thing here nobody can read.
            middle = math.radians(start + span / 2)
            label_at = painter.transform().map(
                QtCore.QPointF(
                    foot.x() + math.cos(middle) * (radius + 14),
                    foot.y() - math.sin(middle) * (radius + 14),
                )
            )
        painter.restore()

        if label_at is not None:
            font = painter.font()
            font.setPointSize(8)
            painter.setFont(font)
            painter.setPen(QtGui.QColor(WEDGE_COLOUR))
            painter.drawText(
                QtCore.QRectF(label_at.x() - 22, label_at.y() - 8, 44, 16),
                Qt.AlignCenter,
                f"{self._pre_tilt:g}°",
            )
        painter.end()

    def _draw_station_offset(self, painter, cx, cy, width, height) -> None:
        """How far the stage drove to reach this station, along the viewing axis.

        x points into the screen here, so the travel is straight away from the reader
        and has no length in this view at all. An arrow across the picture would be the
        one thing a side view cannot honestly draw -- it would read as sliding sideways.
        So it is marked the way any drawing marks a vector going away: a circled cross,
        the tail of an arrow seen from behind.
        """
        radius = 9.0
        x = cx + 96.0
        y = min(height - 30.0, cy + 58.0)
        painter.setPen(QtGui.QPen(QtGui.QColor(TEXT_MUTED_COLOR), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QtCore.QPointF(x, y), radius, radius)
        span = radius * 0.62
        painter.drawLine(QtCore.QPointF(x - span, y - span), QtCore.QPointF(x + span, y + span))
        painter.drawLine(QtCore.QPointF(x - span, y + span), QtCore.QPointF(x + span, y - span))

        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.drawText(
            QtCore.QRectF(x - 150, y + radius + 4, 300, 26),
            Qt.AlignHCenter | Qt.AlignTop,
            f"≈{self._station_offset_mm:g} mm in x, into the page,\nfrom the SEM / FIB station",
        )

    def _draw_objective(self, painter, cx, cy, angle_degrees: float, from_below: bool,
                        inserted: bool) -> None:
        """The fluorescence objective, as a barrel rather than a beam.

        It is a physical lens that travels along z, not a column firing from far away,
        and its bulk is the part that matters: `FluorescenceObjective.position_changed`
        notes that a guard suppresses stage z and r while the objective is inserted,
        because moving then "moves the stage with the objective in the chamber". So the
        gap between the front element and the sample is the quantity worth drawing.
        """
        gap = 18.0 if inserted else 46.0
        length = 30.0
        edge = QtGui.QColor(NEUTRAL_500)
        glass = QtGui.QColor("#dfe5ec")
        if not inserted:
            edge.setAlpha(120)
            glass.setAlpha(120)

        painter.save()
        painter.translate(cx, cy)
        # Rotated onto the objective's own axis, which is not always the vertical: on an
        # offset mount it comes in along the ion column's angle, so the sample -- which
        # is normal to the FIB there -- faces it square on. `from_below` is the extra
        # half turn a compustage needs, having flipped the grid over to meet it.
        painter.rotate(angle_degrees + (180.0 if from_below else 0.0))
        painter.setPen(QtGui.QPen(edge, 1))
        painter.setBrush(QtGui.QColor(FM_BARREL_COLOUR))
        painter.drawPolygon(QtGui.QPolygonF([
            QtCore.QPointF(-7, -gap), QtCore.QPointF(7, -gap),
            QtCore.QPointF(15, -gap - length), QtCore.QPointF(-15, -gap - length),
        ]))
        # The front element, so which end faces the sample is never in doubt.
        painter.setPen(Qt.NoPen)
        painter.setBrush(glass)
        painter.drawRect(QtCore.QRectF(-7, -gap - 3, 14, 3))
        # The working distance, which is what a collision closes.
        painter.setPen(QtGui.QPen(edge, 1, Qt.DotLine))
        painter.drawLine(QtCore.QPointF(0, 0), QtCore.QPointF(0, -gap))
        label_at = painter.transform().map(QtCore.QPointF(0, -gap - length - 20))
        painter.restore()

        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(TEXT_MUTED_COLOR))
        label = "FM objective" + ("" if inserted else "  (retracted)")
        painter.drawText(
            QtCore.QRectF(label_at.x() - 60, label_at.y() - 7, 120, 14),
            Qt.AlignCenter, label,
        )

    @staticmethod
    def _draw_beam(painter, cx, cy, angle_degrees, colour, name,
                   from_below: bool = False) -> None:
        painter.setPen(QtGui.QPen(QtGui.QColor(colour), 2))
        angle = math.radians(angle_degrees)
        # Below means the objective looks up at the sample, which is where the FM sits
        # once a compustage has turned the grid over: the same ray through the pivot,
        # pointing the other way.
        away = -1.0 if from_below else 1.0
        room = (painter.device().height() - cy) if from_below else cy
        reach = min(118, max(24.0, room - 24))
        x2 = cx + away * math.sin(angle) * reach
        y2 = cy - away * math.cos(angle) * reach
        painter.drawLine(QtCore.QPointF(x2, y2), QtCore.QPointF(cx, cy))
        # No arrow head. The shuttle is painted over these, so a head at the sample
        # would be buried in it -- the two lines converging and stopping at the surface
        # already say where the beams land.
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(colour))
        painter.drawText(
            QtCore.QRectF(x2 - 34, y2 + (2 if from_below else -16), 68, 14),
            Qt.AlignCenter, name
        )


class _RailRow(QtWidgets.QWidget):
    """One step in the left-hand rail."""

    def __init__(self, index: int, title: str, parent=None) -> None:
        super().__init__(parent)
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 4, 0, 4)
        row.setSpacing(9)

        self._marker = QtWidgets.QLabel(str(index + 1))
        self._marker.setAlignment(Qt.AlignCenter)
        self._marker.setFixedSize(20, 20)
        row.addWidget(self._marker, 0, Qt.AlignTop)

        column = QtWidgets.QVBoxLayout()
        column.setSpacing(0)
        self._title = _label(title, 11, TEXT_MUTED_COLOR)
        self._title.setWordWrap(True)
        self._note = _label("", 9, TEXT_MUTED_COLOR)
        self._note.setVisible(False)
        column.addWidget(self._title)
        column.addWidget(self._note)
        row.addLayout(column, 1)

    def set_state(self, state: str, note: str = "") -> None:
        """``current``, ``done``, ``skipped`` or ``pending``."""
        colours = {
            "current": (PRIMARY_COLOR, TEXT_STRONG_COLOR),
            "done": (OK_GREEN, TEXT_COLOR),
            "skipped": (BORDER_COLOR, TEXT_MUTED_COLOR),
            "pending": (BORDER_COLOR, TEXT_MUTED_COLOR),
        }
        marker_colour, title_colour = colours.get(state, colours["pending"])
        filled = state in ("current", "done")
        self._marker.setStyleSheet(
            f"border: 1px solid {marker_colour}; border-radius: 10px;"
            f" background: {marker_colour if filled else 'transparent'};"
            f" color: {'#ffffff' if filled else TEXT_MUTED_COLOR}; font-size: 10px;"
        )
        self._title.setStyleSheet(
            f"{ON_PANEL} color: {title_colour}; font-size: 11px;"
            f"{' font-weight: bold;' if state == 'current' else ''}"
        )
        self._note.setText(note)
        self._note.setVisible(bool(note))


# ---------------------------------------------------------------------------
# The wizard
# ---------------------------------------------------------------------------


class SetupWizardDialog(QtWidgets.QDialog):
    """Collects the answers in :class:`~fibsem.setup_wizard.SetupChoices` and saves them.

    ``configuration_saved`` carries the *registered* name, which is not always the name
    that was typed -- ``register_configuration`` suffixes a collision rather than
    refusing -- so the connection tab has something it can actually select.
    """

    configuration_saved = pyqtSignal(str)

    def __init__(self, parent=None, microscope=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Microscope Setup")
        # The default is sized so the stage step -- the tallest, because of the diagram
        # -- fits without scrolling. The minimum is lower so the dialog still opens on a
        # 768-pixel laptop screen; below the default the step bodies scroll rather than
        # clip.
        self.setMinimumSize(760, 560)
        self.resize(860, 700)

        self.choices = SetupChoices()
        # The application's connection, if it has one. Borrowed, never closed here.
        self._external_microscope = microscope
        # Ours, if we opened one. Closed on the way out.
        self._microscope = None
        self._worker: Optional[FunctionWorker] = None
        self._index = STEP_MICROSCOPE
        self._stage_read: Optional[str] = None
        # Set before the widget is torn down, so a connection that lands afterwards
        # knows not to touch it. See _on_connected.
        self._closed = False

        self._build_ui()
        self._show_step(STEP_MICROSCOPE)

    # -- layout ------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._build_rail())

        divider = QtWidgets.QFrame()
        divider.setFrameShape(QtWidgets.QFrame.VLine)
        divider.setStyleSheet(f"color: {BORDER_COLOR};")
        outer.addWidget(divider)

        right = QtWidgets.QVBoxLayout()
        right.setContentsMargins(20, 18, 20, 16)
        right.setSpacing(14)

        self._title_label = _label("", 16, TEXT_STRONG_COLOR, bold=True)
        self._subtitle_label = _label("", 11, TEXT_MUTED_COLOR, wrap=True)
        right.addWidget(self._title_label)
        right.addWidget(self._subtitle_label)

        self._stack = QtWidgets.QStackedWidget()
        for build in (
            self._build_microscope_step,
            self._build_connection_step,
            self._build_stage_step,
            self._build_folders_step,
            self._build_review_step,
        ):
            # Scrolled, with the bar left to appear on its own. A fixed-height body
            # that silently clips is the failure mode here -- the button someone needs
            # is the one below the fold.
            area = QtWidgets.QScrollArea()
            area.setWidgetResizable(True)
            area.setFrameShape(QtWidgets.QFrame.NoFrame)
            area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            area.setStyleSheet("background: transparent;")
            area.setWidget(build())
            self._stack.addWidget(area)
        right.addWidget(self._stack, 1)

        right.addLayout(self._build_footer())
        outer.addLayout(right, 1)

        # Every step body is inside a scroll area, and a spin box that changes value
        # when the page scrolls past it is worse here than anywhere: the rotation
        # reference is a number nobody re-checks after typing it once.
        install_wheel_blocker_recursive(self)

        # Once the stage controls exist. The first _select_model runs while step 1 is
        # still being built, so its prefill had nothing to write into.
        if self.choices.stage_is_editable:
            self._prefill_stage_from_model()

    def _build_rail(self) -> QtWidgets.QWidget:
        rail = QtWidgets.QWidget()
        rail.setFixedWidth(196)
        rail.setStyleSheet(f"background: {PANEL_COLOR};")
        column = QtWidgets.QVBoxLayout(rail)
        column.setContentsMargins(18, 20, 12, 18)
        column.setSpacing(2)

        column.addWidget(_label("Microscope setup", 12, TEXT_STRONG_COLOR, bold=True))
        column.addSpacing(12)

        self._rail_rows: List[_RailRow] = []
        for index, title in enumerate(STEP_TITLES):
            row = _RailRow(index, title)
            self._rail_rows.append(row)
            column.addWidget(row)
        column.addStretch()
        return rail

    def _build_footer(self) -> QtWidgets.QHBoxLayout:
        footer = QtWidgets.QHBoxLayout()
        footer.setSpacing(8)

        self.button_cancel = QtWidgets.QPushButton("Cancel")
        self.button_cancel.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.button_cancel.clicked.connect(self.reject)
        footer.addWidget(self.button_cancel)
        footer.addStretch()

        self.button_back = QtWidgets.QPushButton("Back")
        self.button_back.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.button_back.clicked.connect(self._on_back)
        footer.addWidget(self.button_back)

        self.button_next = QtWidgets.QPushButton("Next")
        self.button_next.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.button_next.setDefault(True)
        self.button_next.clicked.connect(self._on_next)
        footer.addWidget(self.button_next)
        return footer

    # -- step 1: microscope -------------------------------------------------

    def _build_microscope_step(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(8)

        self._model_cards = {}
        for model in MICROSCOPE_MODELS:
            card = ChoiceCard(model.label, model.summary, MODEL_ICON)
            card.clicked.connect(lambda key=model.key: self._select_model(key))
            self._model_cards[model.key] = card
            column.addWidget(card)

        # Only asked when the model does not answer it. Shown rather than hidden when
        # inapplicable would be a second layout to get wrong; the row simply appears.
        self._manufacturer_row = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(self._manufacturer_row)
        row.setContentsMargins(0, 4, 0, 0)
        row.setSpacing(8)
        caption = _label("Manufacturer", 11, TEXT_COLOR)
        caption.setFixedWidth(120)
        self._manufacturer_combo = QtWidgets.QComboBox()
        self._manufacturer_combo.addItems(AVAILABLE_MANUFACTURERS)
        self._manufacturer_combo.setCurrentText(cfg.DEFAULT_MANUFACTURER)
        self._manufacturer_combo.setFixedWidth(180)
        row.addWidget(caption)
        row.addWidget(self._manufacturer_combo)
        row.addStretch()
        column.addWidget(self._manufacturer_row)

        column.addWidget(
            _label(
                "Each option starts from a configuration file that ships with the "
                "application, so everything this wizard does not ask about is already "
                "the right answer for that model.",
                10,
                TEXT_MUTED_COLOR,
                wrap=True,
            )
        )
        column.addStretch()
        self._select_model(self.choices.model_key)
        return page

    def _select_model(self, key: str) -> None:
        self.choices.model_key = key
        for card_key, card in self._model_cards.items():
            card.set_selected(card_key == key)
        self._manufacturer_row.setVisible(not wizard.get_model(key).knows_manufacturer)
        if self.choices.stage_is_readonly:
            # Answers to a question this model no longer asks. Left in place they would
            # be written anyway -- and for a compustage, whose shipped pair is rotation
            # reference 0 with rotation_180 0, a value typed for some other model is
            # exactly the wrong thing to keep.
            self.choices.rotation_reference = None
            self.choices.shuttle_pre_tilt = None
            self._stage_read = None
        self._prefill_stage_from_model()
        if hasattr(self, "_stage_note"):
            self._stage_note.setText("")
        self._update_stage_controls()
        self._update_rail()

    def _prefill_stage_from_model(self) -> None:
        """Seed the stage controls from the chosen model's shipped configuration.

        A starting point, not an answer -- the pre-tilt belongs to whichever shuttle is
        fitted and the reference rotation to how this site loads a sample, so the step
        is still asked. Confirming a number is easier than recalling one.
        """
        # Guarded because selecting a model happens while step 1 is being built, which
        # is before step 3 exists.
        if not hasattr(self, "_rotation_spin"):
            return
        values = wizard.shipped_stage_values(self.choices.model)
        self._rotation_spin.setValue(values["rotation_reference"])
        self._pre_tilt_spin.setValue(values["shuttle_pre_tilt"])

    # -- step 2: computer and connection ------------------------------------

    def _build_connection_step(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(8)

        self._location_cards = {}
        for location in COMPUTER_LOCATIONS:
            card = ChoiceCard(location.label, location.summary, location.icon)
            card.clicked.connect(lambda key=location.key: self._select_location(key))
            self._location_cards[location.key] = card
            column.addWidget(card)

        column.addSpacing(6)
        address_row = QtWidgets.QHBoxLayout()
        address_row.setSpacing(8)
        caption = _label("Microscope address", 11, TEXT_COLOR)
        caption.setFixedWidth(140)
        self._address_edit = QtWidgets.QLineEdit()
        self._address_edit.setFixedWidth(200)
        self.button_test = QtWidgets.QPushButton("Test Connection")
        self.button_test.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.button_test.clicked.connect(self._on_test_connection)
        address_row.addWidget(caption)
        address_row.addWidget(self._address_edit)
        address_row.addWidget(self.button_test)
        address_row.addStretch()
        column.addLayout(address_row)

        self._address_hint = _label("", 10, TEXT_MUTED_COLOR, wrap=True)
        self._address_hint.setContentsMargins(148, 0, 0, 0)
        column.addWidget(self._address_hint)

        self._connection_status = StatusCard()
        self._connection_status.setVisible(False)
        column.addWidget(self._connection_status)
        column.addStretch()

        self._select_location(self.choices.location_key)
        return page

    def _select_location(self, key: str) -> None:
        self.choices.location_key = key
        for card_key, card in self._location_cards.items():
            card.set_selected(card_key == key)

        offline = key == wizard.LOCATION_OFFLINE
        self._address_edit.setText(wizard.default_address(key))
        self._address_edit.setEnabled(not offline)
        self._address_edit.setPlaceholderText("not needed offline" if offline else "")
        self._address_hint.setText(
            ""
            if offline
            else "Prefilled from the choice above. On the microscope PC itself this "
            "would be localhost."
        )
        self.button_test.setEnabled(not offline and self._external_microscope is None)

        if offline:
            self._connection_status.show_status(
                "mdi:cloud-off-outline",
                TEXT_MUTED_COLOR,
                "Simulator",
                "No instrument to reach. Stage settings come from the configuration "
                "you picked, and everything runs against the simulator.",
            )
        elif self._external_microscope is not None:
            # Defensively read: this paints a status card from a slot, where an
            # AttributeError is a process abort rather than a missing sentence.
            system = getattr(self._external_microscope, "system", None)
            info = getattr(system, "info", None)
            named = (
                f" to {info.manufacturer}-{info.model} at {info.ip_address}"
                if info is not None
                else ""
            )
            detail = (
                f"Using the connection the application already has{named}. "
                "A second connection is not opened from here."
            )
            self._connection_status.show_status(
                "mdi:check-circle", OK_GREEN, "Already connected", detail
            )
        else:
            self._connection_status.setVisible(False)
        self._update_rail()

    def _on_test_connection(self) -> None:
        """Open a connection with the answers so far, off the GUI thread."""
        if self._worker is not None and self._worker.is_alive():
            return
        self._disconnect_own()

        self.choices.address = self._address_edit.text().strip()
        self.button_test.setEnabled(False)
        self._connection_status.show_status(
            "mdi:progress-clock",
            PRIMARY_COLOR,
            "Connecting…",
            f"Reaching {self.choices.address or 'the instrument'}.",
        )

        self._worker = FunctionWorker(self._connect, self._current_manufacturer())
        self._worker.returned.connect(self._on_connected)
        self._worker.errored.connect(self._on_connect_failed)
        self._worker.start()

    def _connect(self, manufacturer: Optional[str]):
        from fibsem import utils

        # setup_logging=False: this runs inside a session that already configured its
        # logging, and setup_session would otherwise reconfigure the root handlers
        # underneath the running application.
        microscope, _ = utils.setup_session(
            config_path=self.choices.model.path,
            ip_address=self.choices.address,
            manufacturer=manufacturer,
            setup_logging=False,
        )
        return microscope

    def _on_connected(self, microscope) -> None:
        if self._closed:
            # Closed while the connection was still opening. Touching a widget here
            # would raise inside a slot, which PyQt5 turns into a process abort
            # (FIB-329) -- and the client would leak with nothing left to close it.
            try:
                microscope.disconnect()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"Could not close a late wizard connection: {e}")
            return
        self._microscope = microscope
        info = microscope.system.info
        self._connection_status.show_status(
            "mdi:check-circle",
            OK_GREEN,
            "Connected",
            f"{info.manufacturer} {info.model} at {info.ip_address}. "
            "The stage settings can now be read from the instrument.",
        )
        self.button_test.setEnabled(True)
        self._update_stage_controls()
        self._update_rail()

    def _on_connect_failed(self, error) -> None:
        if self._closed:
            return
        self._connection_status.show_status(
            "mdi:alert-circle",
            FAIL_RED,
            "Could not connect",
            f"No response from {self.choices.address or 'the instrument'}: {error}. "
            "You can carry on and finish setup, but the stage settings will have to "
            "be entered by hand.",
        )
        self.button_test.setEnabled(True)
        self._update_stage_controls()

    # -- step 3: stage ------------------------------------------------------

    def _build_stage_step(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(12)

        read = _panel()
        read_layout = QtWidgets.QVBoxLayout(read)
        read_layout.setContentsMargins(14, 12, 14, 12)
        read_layout.setSpacing(8)
        read_layout.addWidget(
            _label("Reference rotation", 12, TEXT_STRONG_COLOR, bold=True)
        )
        # Held, because the page explains itself differently when it is not asking.
        self._rotation_blurb = _label("", 10, TEXT_MUTED_COLOR, wrap=True)
        read_layout.addWidget(self._rotation_blurb)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        self.button_read_stage = QtWidgets.QPushButton("Read Current Stage Position")
        self.button_read_stage.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.button_read_stage.clicked.connect(self._on_read_stage)
        self._rotation_spin = QtWidgets.QDoubleSpinBox()
        self._rotation_spin.setRange(-360.0, 360.0)
        self._rotation_spin.setDecimals(2)
        self._rotation_spin.setSuffix("°")
        self._rotation_spin.setFixedWidth(110)
        row.addWidget(self.button_read_stage)
        row.addWidget(self._rotation_spin)
        row.addStretch()
        read_layout.addLayout(row)
        self._stage_note = _label("", 10, TEXT_MUTED_COLOR, wrap=True)
        read_layout.addWidget(self._stage_note)
        column.addWidget(read)

        tilt = _panel()
        tilt_layout = QtWidgets.QVBoxLayout(tilt)
        tilt_layout.setContentsMargins(14, 12, 14, 12)
        tilt_layout.setSpacing(8)
        tilt_layout.addWidget(_label("Shuttle pre-tilt", 12, TEXT_STRONG_COLOR, bold=True))
        self._pre_tilt_blurb = _label("", 10, TEXT_MUTED_COLOR, wrap=True)
        tilt_layout.addWidget(self._pre_tilt_blurb)
        self._pre_tilt_spin = QtWidgets.QDoubleSpinBox()
        self._pre_tilt_spin.setRange(-90.0, 90.0)
        self._pre_tilt_spin.setDecimals(1)
        self._pre_tilt_spin.setSuffix("°")
        self._pre_tilt_spin.setValue(35.0)
        self._pre_tilt_spin.setFixedWidth(110)
        self._pre_tilt_spin.valueChanged.connect(self._on_pre_tilt_changed)
        tilt_row = QtWidgets.QHBoxLayout()
        tilt_row.addWidget(self._pre_tilt_spin)
        tilt_row.addStretch()
        tilt_layout.addLayout(tilt_row)
        column.addWidget(tilt)

        # Its own panel rather than inside one: it paints its background and border.
        self._stage_diagram = StageDiagram(self._pre_tilt_spin.value())
        column.addWidget(self._stage_diagram)
        column.addStretch()
        return page

    def _on_pre_tilt_changed(self, value: float) -> None:
        self._stage_diagram.set_pre_tilt(value)

    def _on_read_stage(self) -> None:
        microscope = self._active_microscope()
        if microscope is None:
            return
        try:
            position = microscope.get_stage_position()
        except Exception as e:
            logger.warning(f"Could not read the stage position: {e}")
            self._stage_note.setText(f"Could not read the stage position: {e}")
            return
        rotation = float(np.degrees(position.r))
        tilt = float(np.degrees(position.t))
        self._rotation_spin.setValue(rotation)
        # The diagram was drawing a flat stage until now, which is right while nothing
        # has been read and wrong the moment something has.
        self._stage_diagram.set_stage_tilt(tilt)
        self._stage_read = f"read from the stage at tilt {tilt:.2f}°"
        self._stage_note.setText(
            f"Read {rotation:.2f}° rotation, with the stage tilted {tilt:.2f}°."
        )

    def _update_stage_controls(self) -> None:
        if not hasattr(self, "button_read_stage"):
            return

        # The compustage answers this step, so it is shown and explained rather than
        # asked. Disabled rather than hidden: an empty page says nothing, while a
        # filled-in one that cannot be edited says the answer is already known.
        editable = self.choices.stage_is_editable
        self._rotation_spin.setEnabled(editable)
        self._pre_tilt_spin.setEnabled(editable)
        if not editable:
            self.button_read_stage.setEnabled(False)
            self._rotation_blurb.setText(
                f"A {self.choices.model.label} has a compustage: it reaches the other "
                "side of the sample by tilting rather than rotating, so there is no "
                "reference rotation to measure. It comes from the configuration."
            )
            self._pre_tilt_blurb.setText(
                "Nothing to enter: this instrument has no pre-tilted shuttle, which is "
                "why the holder below is flat and the sample already sits square to "
                "the electron beam."
            )
            self._stage_note.setText("")
            return

        self._rotation_blurb.setText(
            "Put the stage in the SEM orientation — sample flat, square to the electron "
            "beam — then read it. That rotation becomes the reference every orientation "
            "is measured from; MILLING shares it, and FIB is a half turn from it. The "
            "wizard reads the stage; it never moves it. The value below starts from the "
            "configuration you chose — confirm it rather than assume it, since it "
            "depends on how a sample is loaded here."
        )
        self._pre_tilt_blurb.setText(
            "The fixed tilt built into the shuttle or holder. Entered rather than read: "
            "the stage cannot tell the difference between its own tilt and the "
            "shuttle's. Prefilled from the configuration you chose, but it belongs to "
            "whichever shuttle is actually fitted. It is also the stage tilt of the SEM "
            "orientation, which is what the diagram below shows."
        )

        connected = self._active_microscope() is not None
        self.button_read_stage.setEnabled(connected)
        if not connected and not self._stage_note.text():
            self._stage_note.setText(
                "Not connected, so the stage cannot be read. Enter the rotation by "
                "hand, or go back and test the connection."
            )

    # -- step 4: folders ----------------------------------------------------

    def _build_folders_step(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(16)

        preferences = cfg.load_user_preferences()

        # First, because it is the one directory this wizard actually writes to. It
        # used to be unasked and unnamed, which left "Experiments" as the only folder
        # question on the page and invited it being answered for both.
        self._configuration_dir = QDirectoryLineEdit()
        self._configuration_dir.setText(cfg.CONFIG_PATH)
        self._configuration_dir.textChanged.connect(self._on_configuration_dir_changed)
        column.addLayout(
            self._directory_row(
                "Configuration",
                self._configuration_dir,
                "Where this configuration file is saved. The default is inside the "
                "installed package, so an upgrade replaces it — point it elsewhere to "
                "keep it.",
            )
        )

        self._experiment_dir = QDirectoryLineEdit()
        self._experiment_dir.setText(
            preferences.experiment.default_experiment_directory
            or os.path.expanduser("~")
        )
        column.addLayout(
            self._directory_row(
                "Experiments",
                self._experiment_dir,
                "Somewhere else entirely: where a new experiment is created unless you "
                "say otherwise. Experiments hold data, and there will be many of them.",
            )
        )

        self._protocol_file = QFileLineEdit()
        self._protocol_file.setText(preferences.experiment.default_protocol_path)
        column.addLayout(
            self._directory_row(
                "Default protocol",
                self._protocol_file,
                "Optional. The protocol a new experiment starts from.",
            )
        )
        column.addStretch()
        return page

    def _on_configuration_dir_changed(self, text: str) -> None:
        """Kept live so the review screen's path is never stale."""
        self.choices.configuration_directory = text.strip()

    @staticmethod
    def _directory_row(caption: str, editor: QtWidgets.QWidget, hint: str):
        outer = QtWidgets.QVBoxLayout()
        outer.setSpacing(3)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        label = _label(caption, 11, TEXT_COLOR)
        label.setFixedWidth(140)
        row.addWidget(label)
        row.addWidget(editor, 1)
        outer.addLayout(row)
        hint_label = _label(hint, 10, TEXT_MUTED_COLOR, wrap=True)
        hint_label.setContentsMargins(148, 0, 0, 0)
        outer.addWidget(hint_label)
        return outer

    # -- step 5: review -----------------------------------------------------

    def _build_review_step(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(14)

        name_row = QtWidgets.QVBoxLayout()
        name_row.setSpacing(3)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        caption = _label("Configuration name", 11, TEXT_COLOR)
        caption.setFixedWidth(140)
        self._name_edit = QtWidgets.QLineEdit()
        self._name_edit.setPlaceholderText("e.g. Arctis Bay 2")
        self._name_edit.textChanged.connect(self._update_review)
        row.addWidget(caption)
        row.addWidget(self._name_edit, 1)
        name_row.addLayout(row)
        self._filename_hint = _label("", 10, TEXT_MUTED_COLOR, wrap=True)
        self._filename_hint.setContentsMargins(148, 0, 0, 0)
        name_row.addWidget(self._filename_hint)
        column.addLayout(name_row)

        self._default_check = QtWidgets.QCheckBox("Set as the default configuration")
        self._default_check.setStyleSheet(
            f"background: transparent; color: {TEXT_COLOR}; font-size: 11px;"
        )
        self._default_check.setChecked(True)
        self._default_check.setToolTip(
            "Selected automatically the next time the application starts."
        )
        column.addWidget(self._default_check)

        self._summary_panel = _panel()
        self._summary_layout = QtWidgets.QVBoxLayout(self._summary_panel)
        self._summary_layout.setContentsMargins(14, 12, 14, 12)
        self._summary_layout.setSpacing(6)
        column.addWidget(self._summary_panel)

        column.addWidget(
            _label(
                "Not set up here: imaging and milling defaults, and sample holder slot "
                "positions. Both use the values shipped with the configuration and stay "
                "editable afterwards.",
                10,
                TEXT_MUTED_COLOR,
                wrap=True,
            )
        )
        column.addStretch()
        return page

    def _update_review(self) -> None:
        """Rebuild the summary from what would actually be written."""
        self.choices.name = self._name_edit.text().strip()
        # The whole path, since the folder is now a question rather than a constant.
        # A wizard whose entire output is one file should say where that file goes.
        self._filename_hint.setText(
            "Saved as "
            + os.path.join(
                self.choices.resolved_configuration_directory,
                f"{wizard.configuration_slug(self.choices.name)}.yaml",
            )
            + ", and listed on the connection tab under this name."
        )

        while self._summary_layout.count():
            item = self._summary_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        try:
            config = wizard.build_configuration(self.choices)
        except Exception as e:  # pragma: no cover - unreadable shipped file
            logger.warning(f"Could not build the configuration summary: {e}")
            self._summary_layout.addWidget(
                _label(f"Could not read the base configuration: {e}", 11, FAIL_RED, wrap=True)
            )
            return

        info = config.get("info", {})
        stage = config.get("stage", {})
        # "from the shipped configuration" is said rather than left implicit: a value
        # nobody typed still ends up in the file, and it should be visible that it was
        # chosen for them rather than by them.
        from_file = " (from the shipped configuration)" if self.choices.stage_is_readonly else ""
        rows = [
            ("Microscope", self.choices.model.label),
            ("Computer", self.choices.location.label),
            ("Address", info.get("ip_address") or "not used offline"),
            ("Manufacturer", info.get("manufacturer", "")),
            (
                "Reference rotation",
                f"{float(stage.get('rotation_reference', 0)):.2f}°"
                + (f" ({self._stage_read})" if self._stage_read and self.choices.stage_is_editable else from_file),
            ),
            (
                "Shuttle pre-tilt",
                f"{float(stage.get('shuttle_pre_tilt', 0)):.1f}°{from_file}",
            ),
            ("Configuration folder", self.choices.resolved_configuration_directory),
            ("Experiments folder", self.choices.experiment_directory or "not set"),
        ]
        if self.choices.stage_is_editable:
            # Only shown when the wizard derived it. It is the rotation the other side
            # of the sample is reached at, and someone who typed 250° should see that
            # this became 70° rather than discover it from a stage that turned the
            # wrong way.
            rows.insert(
                5,
                (
                    "Rotated 180°",
                    f"{float(stage.get('rotation_180', 0)):.2f}° (derived)",
                ),
            )
        if self.choices.protocol_path:
            rows.append(("Default protocol", self.choices.protocol_path))

        for caption, value in rows:
            row = QtWidgets.QHBoxLayout()
            label = _label(caption, 10, TEXT_MUTED_COLOR)
            label.setFixedWidth(150)
            row.addWidget(label)
            row.addWidget(_label(value, 11, TEXT_COLOR, wrap=True), 1)
            container = QtWidgets.QWidget()
            container.setStyleSheet("background: transparent;")
            container.setLayout(row)
            row.setContentsMargins(0, 0, 0, 0)
            self._summary_layout.addWidget(container)

    # -- navigation ---------------------------------------------------------

    def _is_skipped(self, index: int) -> bool:
        """Nothing is skipped any more.

        The compustage's stage step is shown read-only instead, so the diagram can say
        why there is nothing to enter. Kept as a single predicate rather than deleted,
        because it is the one place a future skip would belong.
        """
        return False

    def _update_rail(self) -> None:
        for index, row in enumerate(self._rail_rows):
            # The note travels with the step whatever state it is in, because it says
            # something about the step rather than about where you are in the wizard.
            note = (
                "set by the configuration"
                if index == STEP_STAGE and self.choices.stage_is_readonly
                else ""
            )
            if self._is_skipped(index):
                row.set_state("skipped", note)
            elif index == self._index:
                row.set_state("current", note)
            elif index < self._index:
                row.set_state("done", note)
            else:
                row.set_state("pending", note)

    def _show_step(self, index: int) -> None:
        self._index = index
        self._stack.setCurrentIndex(index)
        self._title_label.setText(STEP_TITLES[index])
        self._subtitle_label.setText(STEP_SUBTITLES[index])
        self.button_back.setEnabled(index > STEP_MICROSCOPE)
        is_last = index == STEP_REVIEW
        self.button_next.setText("Save && Finish" if is_last else "Next")
        if index == STEP_STAGE:
            self._update_stage_controls()
        if is_last:
            self._update_review()
        self._update_rail()

    def _next_index(self, start: int, step: int) -> int:
        index = start + step
        while 0 <= index < len(STEP_TITLES) and self._is_skipped(index):
            index += step
        return index

    def _on_back(self) -> None:
        index = self._next_index(self._index, -1)
        if index >= STEP_MICROSCOPE:
            self._show_step(index)

    def _on_next(self) -> None:
        self._read_current_step()
        if self._index == STEP_REVIEW:
            self._save()
            return
        index = self._next_index(self._index, +1)
        if index < len(STEP_TITLES):
            self._show_step(index)

    def _read_current_step(self) -> None:
        """Pull the current step's controls into ``choices``.

        On leaving rather than on every keystroke: the review step is built from
        ``choices``, so it only has to be right at the moment it is read.
        """
        if self._index == STEP_MICROSCOPE:
            self.choices.manufacturer = (
                self._manufacturer_combo.currentText()
                if self.choices.needs_manufacturer
                else None
            )
        elif self._index == STEP_CONNECTION:
            self.choices.address = self._address_edit.text().strip()
            if self.choices.is_offline:
                self.choices.address = ""
        elif self._index == STEP_STAGE and self.choices.stage_is_editable:
            # Only when the step asked. Reading the disabled controls back would turn
            # "the wizard did not ask" into "the user answered 0", and the derivation
            # that then runs hands the compustage a 180-degree rotation it does not
            # have -- the shipped pair is 0 *and* 0.
            self.choices.rotation_reference = self._rotation_spin.value()
            self.choices.shuttle_pre_tilt = self._pre_tilt_spin.value()
        elif self._index == STEP_FOLDERS:
            self.choices.configuration_directory = self._configuration_dir.text().strip()
            self.choices.experiment_directory = self._experiment_dir.text().strip()
            self.choices.protocol_path = self._protocol_file.text().strip()
        elif self._index == STEP_REVIEW:
            self.choices.name = self._name_edit.text().strip()
            self.choices.set_as_default = self._default_check.isChecked()

    def _current_manufacturer(self) -> Optional[str]:
        if self.choices.is_offline:
            return "Demo"
        if self.choices.needs_manufacturer:
            return self._manufacturer_combo.currentText()
        return None

    # -- finishing ----------------------------------------------------------

    def _save(self) -> None:
        if not self.choices.name:
            QtWidgets.QMessageBox.warning(
                self,
                "Name the configuration",
                "Give the configuration a name. It is what the connection tab lists, "
                "and what the file is named after.",
            )
            self._name_edit.setFocus()
            return

        try:
            result = wizard.apply_setup(self.choices)
        except Exception as e:
            logger.error(f"Setup wizard could not save the configuration: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "Could not save the configuration",
                f"{e}\n\nNothing was changed. Go back and adjust the answers, or "
                f"cancel and configure the microscope by hand.",
            )
            return

        message = (
            f"Saved {result.configuration_name} to {result.path}."
            + ("\n\nIt is now the default configuration." if result.is_default else "")
        )
        if result.warnings:
            message += "\n\n" + "\n".join(result.warnings)
        QtWidgets.QMessageBox.information(self, "Setup complete", message)

        self.configuration_saved.emit(result.configuration_name)
        self.accept()

    # -- lifecycle ----------------------------------------------------------

    def _active_microscope(self):
        """Whichever connection is available for reading the stage."""
        return self._microscope or self._external_microscope

    def _disconnect_own(self) -> None:
        """Close the connection this dialog opened, and only that one."""
        if self._microscope is None:
            return
        try:
            self._microscope.disconnect()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Could not close the wizard's connection: {e}")
        self._microscope = None

    def done(self, result: int) -> None:  # noqa: D102 - Qt override
        # Both Accept and Reject land here, which is the point: a wizard that leaves a
        # client open on finish is the same leak as one that leaves it open on cancel.
        self._closed = True
        self._disconnect_own()
        super().done(result)


def open_setup_wizard(parent=None, microscope=None) -> Optional[str]:
    """Run the wizard modally; return the saved configuration name, or None."""
    dialog = SetupWizardDialog(parent=parent, microscope=microscope)
    saved: List[str] = []
    dialog.configuration_saved.connect(saved.append)
    dialog.exec_()
    return saved[0] if saved else None


def main():  # pragma: no cover - manual harness
    """Standalone harness, in the dark palette the dialog is painted for."""
    import sys

    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(stylesheets.NAPARI_STYLE)
    print(open_setup_wizard())
    sys.exit(0)


if __name__ == "__main__":  # pragma: no cover
    main()
