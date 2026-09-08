"""What the instrument was doing, rendered.

A ``MicroscopeState`` is stage position, both beams, both detectors and a timestamp,
and it is what every lamella pose is made of (``Lamella.poses`` is literally
``Dict[str, MicroscopeState]``) and what every image carries in its metadata. Until
now nothing displayed one. Three partial renderers had grown instead: the HUD ticker
read a few fields out of image metadata, the lamella pose list showed
``stage_position.pretty`` and dropped the rest, and the selected-lamella panel reached
past all of it for the objective position alone. So the beam and detector settings
recorded at a pose were written to disk and shown to nobody.

**It renders; it does not read.** No microscope reference, no device calls, no polling.
``set_state`` is given a record and draws it. The refresh button emits a signal and
stops there, the same contract ``stage_position_widget`` uses, because reading the
instrument is a device call and the host is the only thing that knows what one costs.

**Two states, one widget.** A saved pose and the live instrument are the same record
from different sources, so they are the same widget with a different label. Passing a
reference through ``set_reference`` turns on the comparison, which is the mode with
something to say: *how far is the instrument from where this lamella was milled*, asked
before the operator presses Move To.

Display only. ``Move To`` and ``Restore`` belong to the host: the moment this widget
can drive a stage it needs a threading story, and that is a different widget.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import List, Optional, Tuple

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.structures import BeamSettings, FibsemDetectorSettings, MicroscopeState
from fibsem.ui.tokens import (
    CURRENT_POSITION_COLOUR,
    SAVED_POSITION_COLOUR,
    TEXT_COLOR,
    TEXT_MUTED_COLOR,
    WARN_COLOR,
)
from fibsem.ui.widgets.custom_widgets import TitledPanel
from fibsem.utils import (
    NOT_AVAILABLE,
    format_angle,
    format_current,
    format_distance,
    format_resolution_as_str,
    format_value,
    format_voltage,
)

# The two colours the comparison is drawn in, and they are not chosen here.
# `tokens.py` documents CURRENT_POSITION_COLOUR as "where the stage is now" and
# SAVED_POSITION_COLOUR as "a marked position", and the FIB/SEM and fluorescence
# overviews already draw stage markers in them. An operator who has used either
# canvas has been taught that yellow is live and cyan is saved; inventing a second
# convention for the same distinction in a panel beside them would be worse than
# either convention alone.
_SAVED_COLOUR = SAVED_POSITION_COLOUR
_LIVE_COLOUR = CURRENT_POSITION_COLOUR

# Grid columns. Named because three of the four are referred to from two places, and
# an off-by-one here puts a value in the label column rather than raising.
_KEY_COLUMN = 0
_SAVED_COLUMN = 1
_GAP_COLUMN = 2
_LIVE_COLUMN = 3
_KEY_COLUMN_MIN_W = 92  # fits "Working dist." at the default UI font
_GAP_COLUMN_W = 14

# Rows are compared as *rendered strings*, not as floats. Two records holding
# 2.0e-11 and 2.0000001e-11 both read "20 pA", and marking that row as a difference
# would be reporting a change the panel cannot show. Comparing what is drawn asks the
# same question the reader is asking: do these two columns say different things?


class MicroscopeStateWidget(QWidget):
    """One ``MicroscopeState``, optionally against another.

    Parameters
    ----------
    show_refresh:
        Whether to offer a refresh button. A saved pose has nothing to refresh, so the
        embedding list turns it off; a live readout turns it on and owns the signal.
    show_rotation:
        Whether the stage has a rotation axis. Pass
        ``microscope.is_available("stage_rotation")``, which reads the instrument's own
        axes since FIB-834.

        This is not cosmetic. A compustage has no rotation axis at all -- AutoScript's
        ``CompustagePosition`` carries ``x``, ``y``, ``z``, ``a`` and nothing else --
        so ``stage_position_from_autoscript`` writes ``r=0.0`` as a **literal**,
        because ``FibsemStagePosition`` needs a number and ``get_stage_orientation``
        raises on ``None``. The record therefore cannot distinguish "the stage is at
        rotation zero" from "this stage does not rotate", and drawing ``0.0°`` asserts
        the first. The widget has no microscope to ask, and the host does.
    """

    #: Refresh was pressed. Reading the instrument is a device call and this widget
    #: does not make those -- the host decides what one costs.
    refresh_requested = pyqtSignal()

    def __init__(
        self,
        show_refresh: bool = False,
        show_rotation: bool = True,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._state: Optional[MicroscopeState] = None
        self._reference: Optional[MicroscopeState] = None
        self._show_refresh = show_refresh
        self._show_rotation = show_rotation
        self._setup_ui()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        self.label_title = QLabel("—")
        self.label_title.setStyleSheet("font-weight: bold;")
        # Chips, not a sentence. Which record is on screen is the first thing a
        # reader needs and the thing they will glance back at, and a chip carrying
        # the colour its column is drawn in answers it without being read.
        self.chip_saved = _chip("saved", _SAVED_COLOUR)
        self.chip_live = _chip("live", _LIVE_COLOUR)
        header.addWidget(self.label_title)
        header.addWidget(self.chip_saved)
        header.addWidget(self.chip_live)
        header.addStretch()

        self.button_refresh = QPushButton("Refresh")
        self.button_refresh.setVisible(self._show_refresh)
        self.button_refresh.clicked.connect(self.refresh_requested)
        header.addWidget(self.button_refresh)
        outer.addLayout(header)

        # Stage is not collapsible. A pose in a list has to say where it is without
        # being opened, and "where" is the whole of what a pose list is for.
        self.grid_stage = _ValueGrid()
        self.panel_stage = TitledPanel(
            "Stage", content=self.grid_stage, collapsible=False
        )
        outer.addWidget(self.panel_stage)

        # The beams collapse, and start collapsed: a pose row that expands to forty
        # lines is not a row. Their headline values go on the panel title instead, so
        # a shut section still reports that the FIB is at 30 kV.
        self.grid_electron = _ValueGrid()
        self.panel_electron = TitledPanel("Electron", content=self.grid_electron)
        self.panel_electron.collapse()
        outer.addWidget(self.panel_electron)

        self.grid_ion = _ValueGrid()
        self.panel_ion = TitledPanel("Ion", content=self.grid_ion)
        self.panel_ion.collapse()
        outer.addWidget(self.panel_ion)

        # There is no fluorescence section. `MicroscopeState` carries only
        # `objective_position` for it -- no channel, no filter -- and
        # `get_microscope_state` does not even populate that (AutoLamellaUI patches it
        # in at the call site afterwards). A section that could never fill is worse
        # than no section: it reads as an instrument fault rather than a gap in the
        # record. It arrives when the record can answer for it.

        self.label_delta = QLabel("")
        self.label_delta.setStyleSheet(f"color: {WARN_COLOR};")
        self.label_delta.setVisible(False)
        outer.addWidget(self.label_delta)

        self.label_timestamp = QLabel("")
        self.label_timestamp.setStyleSheet(
            f"color: {TEXT_MUTED_COLOR}; font-size: 10pt;"
        )
        outer.addWidget(self.label_timestamp)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_state(self, state: Optional[MicroscopeState], title: str = "") -> None:
        """Render ``state``.

        ``title`` is passed in rather than derived. Naming a pose is the host's job --
        the pose list has the key it was stored under, and classifying a position as
        an orientation needs a microscope, which this widget does not have.
        """
        self._state = state
        self.label_title.setText(title or "Microscope state")
        self._refresh()

    def set_reference(self, reference: Optional[MicroscopeState]) -> None:
        """Compare against ``reference``, or pass ``None`` to stop comparing."""
        self._reference = reference
        self._refresh()

    def clear(self) -> None:
        self._state = None
        self._reference = None
        self._refresh()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        state = self._state
        comparing = state is not None and self._reference is not None
        # Both chips when comparing; otherwise the one that describes this record --
        # a live readout has a refresh button, a saved pose does not.
        self.chip_saved.setVisible(
            state is not None and (comparing or not self._show_refresh)
        )
        self.chip_live.setVisible(
            state is not None and (comparing or self._show_refresh)
        )

        if state is None:
            for grid in (self.grid_stage, self.grid_electron, self.grid_ion):
                grid.clear()
            self.panel_electron.set_title("Electron")
            self.panel_ion.set_title("Ion")
            self.label_delta.setVisible(False)
            self.label_timestamp.setText("")
            return

        reference = self._reference
        self.grid_stage.set_rows(
            _stage_rows(state, self._show_rotation),
            _stage_rows(reference, self._show_rotation) if comparing else None,
        )
        self.grid_electron.set_rows(
            _beam_rows(state.electron_beam, state.electron_detector),
            _beam_rows(reference.electron_beam, reference.electron_detector)
            if comparing
            else None,
        )
        self.grid_ion.set_rows(
            _beam_rows(state.ion_beam, state.ion_detector),
            _beam_rows(reference.ion_beam, reference.ion_detector)
            if comparing
            else None,
        )

        self.panel_electron.set_title(
            f"Electron   {_beam_headline(state.electron_beam)}"
        )
        self.panel_ion.set_title(f"Ion   {_beam_headline(state.ion_beam)}")

        separation = _separation(state, reference) if comparing else None
        self.label_delta.setVisible(separation is not None)
        if separation is not None:
            self.label_delta.setText(
                f"Stage is {format_distance(separation)} from the saved pose"
            )

        self.label_timestamp.setText(_timestamp(state.timestamp))


# ---------------------------------------------------------------------------
# Row construction -- pure, so the formatting is testable without a widget
# ---------------------------------------------------------------------------


def _stage_rows(
    state: Optional[MicroscopeState], show_rotation: bool = True
) -> List[Tuple[str, str]]:
    """The stage axes, omitting R on a stage that has none.

    Omitted rather than shown as ``NOT_AVAILABLE``: an em-dash means "the instrument
    did not report this", and a compustage's missing rotation is not a gap in a
    reading -- the axis does not exist. A row that will always be empty invites the
    question of why.
    """
    axes = ["X", "Y", "Z"] + (["R"] if show_rotation else []) + ["T"]
    if state is None or state.stage_position is None:
        return [(axis, NOT_AVAILABLE) for axis in axes]
    position = state.stage_position
    values = {
        "X": format_distance(position.x),
        "Y": format_distance(position.y),
        "Z": format_distance(position.z),
        "R": format_angle(position.r),
        "T": format_angle(position.t),
    }
    return [(axis, values[axis]) for axis in axes]


def _beam_rows(
    beam: Optional[BeamSettings], detector: Optional[FibsemDetectorSettings]
) -> List[Tuple[str, str]]:
    resolution = beam.resolution if beam else None
    return [
        ("Voltage", format_voltage(beam.voltage if beam else None)),
        ("Current", format_current(beam.beam_current if beam else None)),
        ("HFW", format_distance(beam.hfw if beam else None)),
        ("Working dist.", format_distance(beam.working_distance if beam else None)),
        (
            "Resolution",
            format_resolution_as_str(resolution) if resolution else NOT_AVAILABLE,
        ),
        (
            "Dwell",
            format_value(beam.dwell_time, "s")
            if beam and beam.dwell_time
            else NOT_AVAILABLE,
        ),
        ("Scan rotation", format_angle(beam.scan_rotation if beam else None)),
        ("Detector", _detector(detector)),
    ]


def _detector(detector: Optional[FibsemDetectorSettings]) -> str:
    """``ETD / SE``.

    Abbreviated because the mode names are long -- "SecondaryElectrons" is wider than
    the value column and would wrap every row it appeared on -- and because type and
    mode are read together or not at all.
    """
    if detector is None:
        return NOT_AVAILABLE
    mode = {
        "SecondaryElectrons": "SE",
        "BackscatteredElectrons": "BSE",
        "SecondaryIons": "SI",
    }.get(detector.mode, detector.mode)
    return f"{detector.type} / {mode}"


def _beam_headline(beam: Optional[BeamSettings]) -> str:
    """What a collapsed section still says, so shutting one loses less."""
    if beam is None:
        return NOT_AVAILABLE
    return f"{format_voltage(beam.voltage)} · {format_current(beam.beam_current)}"


def _separation(state: MicroscopeState, reference: MicroscopeState) -> Optional[float]:
    """Straight-line distance between two stage positions, in metres.

    One number an operator can act on. It does collapse three axes into one, which is
    why the rows above it stay -- the scalar says *whether* to look, the rows say
    where.
    """
    here, there = state.stage_position, reference.stage_position
    if here is None or there is None:
        return None
    squares = 0.0
    for axis in ("x", "y", "z"):
        a, b = getattr(here, axis), getattr(there, axis)
        if a is None or b is None:
            return None
        squares += (a - b) ** 2
    return math.sqrt(squares)


def _timestamp(value: Optional[float]) -> str:
    if not value:
        return ""
    return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


class _ValueGrid(QWidget):
    """Label/value rows, or label/saved/live rows when comparing.

    Rebuilt on every update rather than diffed. These are at most eight rows of text
    and the alternative is a cache that can disagree with the record it draws.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(2, 2, 2, 2)
        self._layout.setHorizontalSpacing(10)
        self._layout.setVerticalSpacing(1)
        # The label column gets a floor. Without one the three-column comparison
        # squeezed it until "Working dist." and "Scan rotation" were clipped to
        # "Working dist" and "Scan rotatio" -- which the tests could not see, because
        # a QLabel reports the text it was given whether or not it has room to draw it.
        self._layout.setColumnMinimumWidth(_KEY_COLUMN, _KEY_COLUMN_MIN_W)
        self._layout.setColumnStretch(_SAVED_COLUMN, 1)
        # A blank column between the two value columns. When every row matches they
        # are all muted, and without a gap they read as one run of numbers rather than
        # two things being compared.
        self._layout.setColumnMinimumWidth(_GAP_COLUMN, _GAP_COLUMN_W)
        self._layout.setColumnStretch(_LIVE_COLUMN, 1)

    def clear(self) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

    def set_rows(
        self,
        rows: List[Tuple[str, str]],
        reference: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        self.clear()
        comparing = reference is not None
        if comparing:
            self._add_header()

        by_key = dict(reference) if comparing else {}
        offset = 1 if comparing else 0

        for index, (key, value) in enumerate(rows):
            row = index + offset
            self._layout.addWidget(_key_label(key), row, _KEY_COLUMN)
            if not comparing:
                self._layout.addWidget(
                    _value_label(value, TEXT_COLOR), row, _SAVED_COLUMN
                )
                continue

            # The reference is the *saved* side and `rows` is the live one, so the
            # saved column is drawn from `by_key`. Compared on the rendered strings:
            # two records holding 2.0e-11 and 2.0000001e-11 both read "20 pA", and a
            # row showing the same thing twice is not a difference.
            was = by_key.get(key, NOT_AVAILABLE)
            differs = was != value
            self._layout.addWidget(
                _value_label(was, _SAVED_COLOUR if differs else TEXT_MUTED_COLOR),
                row,
                _SAVED_COLUMN,
            )
            self._layout.addWidget(
                _value_label(value, _LIVE_COLOUR if differs else TEXT_MUTED_COLOR),
                row,
                _LIVE_COLUMN,
            )

    def _add_header(self) -> None:
        for column, text in ((_SAVED_COLUMN, "Saved"), (_LIVE_COLUMN, "Live")):
            label = QLabel(text)
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            label.setStyleSheet(f"color: {TEXT_MUTED_COLOR}; font-size: 9pt;")
            self._layout.addWidget(label, 0, column)


def _chip(text: str, colour: str) -> QLabel:
    """A small pill in the colour its column is drawn in."""
    label = QLabel(text)
    label.setStyleSheet(
        f"color: {colour}; border: 1px solid {colour}; border-radius: 7px;"
        f"padding: 0px 6px; font-size: 9pt;"
    )
    label.setVisible(False)
    return label


def _key_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(f"color: {TEXT_MUTED_COLOR};")
    return label


def _value_label(text: str, colour: str) -> QLabel:
    label = QLabel(text)
    label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    # Monospaced so the digits line up down the column; a stage position read as a
    # ragged list of numbers is harder to compare than one that is aligned.
    label.setStyleSheet(f"color: {colour}; font-family: monospace;")
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return label
