"""Guided calibration of a sample holder's slot positions.

Today the holder widget copies whatever the stage position is into a slot when a
button is pressed, and the stage module overwrites every slot's rotation and tilt
with the SEM orientation at startup. A position captured at any other pose keeps
its x/y/z but is re-expressed with SEM r/t -- silently wrong -- and nothing tells
the operator which orientation to be in.

This dialog makes the contract explicit. It moves the stage to the calibration
orientation itself, walks the operator slot by slot ("drive to the centre of the
grid in this slot, then Capture"), reads x/y/z from the stage and stamps r/t from
the orientation, offers "Move to slot" to confirm the capture, sanity-checks
spacing and stage limits, and only then saves. It refuses to capture at any other
orientation rather than taking a number it would have to reinterpret.

Non-modal on purpose: the operator drives the stage from the Microscope tab while
this sits beside it. The live position comes from the microscope's own
``stage_position_changed`` signal; nothing here polls the hardware.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from datetime import datetime
from typing import Dict, List, Optional

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal

from fibsem import config as cfg
from fibsem.microscopes._stage import GRID_RADIUS, SampleHolder, SlotCalibration
from fibsem.structures import FibsemStagePosition
from fibsem.ui import stylesheets
from fibsem.ui.qt.threading import FunctionWorker
from fibsem.ui.tokens import (
    BORDER_COLOR,
    ERROR_COLOR,
    OK_COLOR,
    PANEL_COLOR,
    SURFACE_COLOR,
    TEXT_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
    WARN_COLOR,
)
from fibsem.ui.utils import install_wheel_blocker_recursive
from fibsem.ui.widgets.custom_widgets import ValueSpinBox
from fibsem.ui.widgets.guided_setup_dialog import StageDiagram

# Slot positions are stored at this orientation; every other pose is derived from
# them by the stable-move projection, which is what the startup stamping in
# `_create_sample_stage` has always assumed. Now it is written down, and enforced.
CALIBRATION_ORIENTATION = "SEM"

# Two slots whose captured centres are closer than this cannot both be right.
MIN_SLOT_SEPARATION = 2 * GRID_RADIUS

STEP_HOLDER = 0
STEP_ORIENTATION = 1
# slot steps follow, one per slot, then the review step


def fibsem_version() -> str:
    try:
        from fibsem import __version__

        return str(__version__)
    except Exception:  # noqa: BLE001 - provenance, not a dependency
        return "unknown"


def _label(
    text: str,
    size: int = 12,
    color: str = TEXT_COLOR,
    bold: bool = False,
    wrap: bool = False,
) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    label.setWordWrap(wrap)
    label.setStyleSheet(
        f"background: transparent; color: {color}; font-size: {size}px;"
        f"{' font-weight: bold;' if bold else ''}"
    )
    return label


class _RailRow(QtWidgets.QLabel):
    """One step in the left-hand rail: pending, current, or done."""

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(title, parent)
        self.setContentsMargins(10, 6, 10, 6)
        self.set_state("pending")

    def set_state(self, state: str) -> None:
        colour = {
            "done": OK_COLOR,
            "current": TEXT_STRONG_COLOR,
            "pending": TEXT_MUTED_COLOR,
        }[state]
        background = "#2d3f5c" if state == "current" else "transparent"
        self.setStyleSheet(
            f"background: {background}; color: {colour}; border-radius: 3px;"
            f" font-size: 12px;{' font-weight: bold;' if state == 'current' else ''}"
        )


class HolderCalibrationDialog(QtWidgets.QDialog):
    """Walk the operator through capturing every slot centre at one orientation.

    ``holder_saved`` carries the holder after the configuration file is written.
    The holder object is the live one; its slots are only modified on Save, so
    Cancel leaves it exactly as it was.
    """

    holder_saved = pyqtSignal(object)  # SampleHolder
    # The microscope's position signal is a psygnal, emitted on whichever thread
    # moved the stage. This Qt signal is the hop onto the GUI thread.
    _stage_moved = pyqtSignal(object)  # FibsemStagePosition

    def __init__(
        self,
        microscope,
        holder: SampleHolder,
        parent=None,
        save_path: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Calibrate sample holder")
        self.setModal(False)
        self.setMinimumSize(640, 480)
        self.resize(720, 540)
        self.setStyleSheet(f"background: {SURFACE_COLOR}; color: {TEXT_COLOR};")

        self._microscope = microscope
        self._holder = holder
        self._save_path = save_path or cfg.SAMPLE_HOLDER_CONFIGURATION_PATH
        # The working set: captured this session, applied to the holder on Save.
        self._captured: Dict[str, FibsemStagePosition] = {}
        self._capacity = max(1, int(holder.capacity))
        self._holder_name = holder.name
        self._index = STEP_HOLDER
        self._worker: Optional[FunctionWorker] = None
        self._closed = False

        self._build_ui()
        self._show_step(STEP_HOLDER)

        self._stage_moved.connect(self._on_stage_moved)
        try:
            microscope.stage_position_changed.connect(self._stage_moved.emit)
        except Exception as e:  # noqa: BLE001 - a microscope without the signal
            logging.debug(f"No stage position signal to subscribe to: {e}")

    # -- what the steps are ---------------------------------------------------

    @property
    def slot_names(self) -> List[str]:
        return [f"Slot-{i + 1:02d}" for i in range(self._capacity)]

    @property
    def step_titles(self) -> List[str]:
        n = self._capacity
        return (
            ["Holder", "Orientation"]
            + [f"Slot {i + 1} of {n}" for i in range(n)]
            + ["Check & save"]
        )

    @property
    def review_step(self) -> int:
        return 2 + self._capacity

    def _slot_for_step(self, index: int) -> Optional[str]:
        if STEP_ORIENTATION < index < self.review_step:
            return self.slot_names[index - 2]
        return None

    @property
    def current_slot(self) -> Optional[str]:
        return self._slot_for_step(self._index)

    # -- layout ----------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._rail = QtWidgets.QWidget()
        self._rail.setFixedWidth(180)
        self._rail.setStyleSheet(f"background: {PANEL_COLOR};")
        self._rail_layout = QtWidgets.QVBoxLayout(self._rail)
        self._rail_layout.setContentsMargins(10, 16, 10, 12)
        self._rail_layout.setSpacing(2)
        self._rail_rows: List[_RailRow] = []
        self._rail_note = _label(
            f"Positions are stored at the {CALIBRATION_ORIENTATION} orientation. "
            "Other orientations are derived from them, so capturing anywhere else "
            "is refused.",
            11,
            TEXT_MUTED_COLOR,
            wrap=True,
        )
        self._rebuild_rail()
        outer.addWidget(self._rail)

        divider = QtWidgets.QFrame()
        divider.setFrameShape(QtWidgets.QFrame.VLine)
        divider.setStyleSheet(f"color: {BORDER_COLOR};")
        outer.addWidget(divider)

        right = QtWidgets.QVBoxLayout()
        right.setContentsMargins(18, 16, 18, 14)
        right.setSpacing(12)
        self._title = _label("", 16, TEXT_STRONG_COLOR, bold=True)
        self._subtitle = _label("", 12, TEXT_COLOR, wrap=True)
        right.addWidget(self._title)
        right.addWidget(self._subtitle)

        self._stack = QtWidgets.QStackedWidget()
        self._stack.addWidget(self._build_holder_page())
        self._stack.addWidget(self._build_orientation_page())
        self._stack.addWidget(self._build_slot_page())
        self._stack.addWidget(self._build_review_page())
        right.addWidget(self._stack, 1)

        right.addLayout(self._build_footer())
        outer.addLayout(right, 1)

        install_wheel_blocker_recursive(self)
        for button in self.findChildren(QtWidgets.QPushButton):
            button.setAutoDefault(False)
            button.setDefault(False)

    def _rebuild_rail(self) -> None:
        while self._rail_layout.count():
            item = self._rail_layout.takeAt(0)
            if item.widget() is not None and item.widget() is not self._rail_note:
                item.widget().deleteLater()
        self._rail_rows = []
        for title in self.step_titles:
            row = _RailRow(title)
            self._rail_rows.append(row)
            self._rail_layout.addWidget(row)
        self._rail_layout.addStretch()
        self._rail_layout.addWidget(self._rail_note)

    def _build_holder_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        form.setSpacing(8)
        self.name_edit = QtWidgets.QLineEdit(self._holder_name)
        self.capacity_spin = ValueSpinBox(
            minimum=1.0,
            maximum=12.0,
            step=1.0,
            decimals=0,
            tooltip="Number of grid slots on this holder",
        )
        self.capacity_spin.setValue(float(self._capacity))
        self.capacity_spin.valueChanged.connect(self._on_capacity_changed)
        stage = getattr(self._microscope.system, "stage", None)
        pre_tilt = getattr(stage, "shuttle_pre_tilt", 0.0)
        rotation = getattr(stage, "rotation_reference", 0.0)
        form.addRow("Name", self.name_edit)
        form.addRow("Slots", self.capacity_spin)
        form.addRow("Pre-tilt", _label(f"{pre_tilt:.1f}°   system configuration"))
        form.addRow("Ref. rotation", _label(f"{rotation:.1f}°   system configuration"))
        return page

    def _build_orientation_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setSpacing(10)
        # The same side view the setup wizard draws, here read-only: it follows the
        # stage, so the operator can see which orientation they are at before the
        # dialog refuses a capture in words.
        stage_settings = getattr(self._microscope.system, "stage", None)
        self.stage_diagram = StageDiagram(
            pre_tilt=float(getattr(stage_settings, "shuttle_pre_tilt", 0.0))
        )
        layout.addWidget(self.stage_diagram)
        self.button_orientation = QtWidgets.QPushButton(
            f"Move to {CALIBRATION_ORIENTATION} orientation"
        )
        self.button_orientation.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.button_orientation.clicked.connect(self._on_move_to_orientation)
        self.orientation_status = _label("", 12, TEXT_MUTED_COLOR, wrap=True)
        layout.addWidget(self.button_orientation, 0, Qt.AlignLeft)
        layout.addWidget(self.orientation_status)
        layout.addStretch()
        return page

    def _build_slot_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setSpacing(10)

        box = QtWidgets.QFrame()
        box.setStyleSheet(
            f"background: {PANEL_COLOR}; border: 1px solid {BORDER_COLOR};"
            " border-radius: 4px;"
        )
        grid = QtWidgets.QGridLayout(box)
        grid.setContentsMargins(12, 10, 12, 10)
        grid.setHorizontalSpacing(16)
        grid.addWidget(_label("Stage now", 11, TEXT_MUTED_COLOR), 0, 0)
        self.live_position = _label("—", 12, TEXT_STRONG_COLOR)
        grid.addWidget(self.live_position, 0, 1)
        grid.addWidget(_label("Captured", 11, TEXT_MUTED_COLOR), 1, 0)
        self.captured_position = _label("not yet", 12, TEXT_MUTED_COLOR)
        grid.addWidget(self.captured_position, 1, 1)
        grid.setColumnStretch(1, 1)
        layout.addWidget(box)

        self.capture_status = _label("", 12, TEXT_MUTED_COLOR, wrap=True)
        layout.addWidget(self.capture_status)

        buttons = QtWidgets.QHBoxLayout()
        self.button_capture = QtWidgets.QPushButton("Capture")
        self.button_capture.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.button_capture.clicked.connect(self._on_capture)
        self.button_move_to_slot = QtWidgets.QPushButton("Move to slot")
        self.button_move_to_slot.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.button_move_to_slot.setToolTip(
            "Drive back to the captured position to confirm it"
        )
        self.button_move_to_slot.clicked.connect(self._on_move_to_slot)
        buttons.addWidget(self.button_capture)
        buttons.addWidget(self.button_move_to_slot)
        buttons.addStretch()
        layout.addLayout(buttons)
        layout.addStretch()
        return page

    def _build_review_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setSpacing(8)
        self.review_table = QtWidgets.QTableWidget(0, 3)
        self.review_table.setHorizontalHeaderLabels(["Slot", "Position", "Note"])
        self.review_table.horizontalHeader().setStretchLastSection(True)
        self.review_table.verticalHeader().setVisible(False)
        self.review_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.review_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.review_table.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(self.review_table, 1)
        self.review_summary = _label("", 12, TEXT_COLOR, wrap=True)
        layout.addWidget(self.review_summary)
        return page

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
        self.button_skip = QtWidgets.QPushButton("Skip slot")
        self.button_skip.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.button_skip.setToolTip("Keep this slot's existing position")
        self.button_skip.clicked.connect(self._on_skip)
        footer.addWidget(self.button_skip)
        self.button_next = QtWidgets.QPushButton("Next")
        self.button_next.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.button_next.clicked.connect(self._on_next)
        footer.addWidget(self.button_next)
        return footer

    # -- navigation ------------------------------------------------------------

    def _show_step(self, index: int) -> None:
        self._index = index
        for i, row in enumerate(self._rail_rows):
            row.set_state(
                "done" if i < index else "current" if i == index else "pending"
            )

        slot = self._slot_for_step(index)
        if index == STEP_HOLDER:
            self._stack.setCurrentIndex(0)
            self._title.setText("Sample holder")
            self._subtitle.setText(
                "Name the holder and say how many grid slots it has. Pre-tilt and "
                "reference rotation come from the system configuration."
            )
        elif index == STEP_ORIENTATION:
            self._stack.setCurrentIndex(1)
            self._title.setText("Orientation")
            self._subtitle.setText(
                f"Slot positions are captured at the {CALIBRATION_ORIENTATION} "
                "orientation, flat and pre-tilted. The dialog moves the stage there; "
                "you do not need to."
            )
            self._refresh_orientation_status()
        elif slot is not None:
            self._stack.setCurrentIndex(2)
            number = index - 1
            self._title.setText(f"Slot {number}: drive to the centre of the grid")
            self._subtitle.setText(
                "Use the stage controls, or click on the image, to move until the "
                f"centre of the grid in {slot} is under the electron beam. Then press "
                "Capture. Move to slot drives back to what was captured, to check it."
            )
            self._refresh_slot_page()
        else:
            self._stack.setCurrentIndex(3)
            self._title.setText("Check and save")
            self._subtitle.setText(
                "Every slot, what will be saved for it, and anything that looks wrong."
            )
            self._refresh_review()

        last = index == self.review_step
        self.button_back.setEnabled(index > STEP_HOLDER)
        self.button_skip.setVisible(slot is not None)
        self.button_next.setText("Save" if last else "Next")

    def _on_back(self) -> None:
        if self._index > STEP_HOLDER:
            self._show_step(self._index - 1)

    def _on_next(self) -> None:
        if self._index == STEP_HOLDER:
            self._holder_name = self.name_edit.text().strip() or self._holder_name
        if self._index == STEP_ORIENTATION and not self._at_calibration_orientation():
            self._refresh_orientation_status()
            return
        if self._index == self.review_step:
            self._save()
            return
        self._show_step(self._index + 1)

    def _on_skip(self) -> None:
        slot = self.current_slot
        if slot is not None:
            self._captured.pop(slot, None)
        self._show_step(self._index + 1)

    def _on_capacity_changed(self) -> None:
        self._capacity = max(1, int(self.capacity_spin.value()))
        for name in list(self._captured):
            if name not in self.slot_names:
                del self._captured[name]
        self._rebuild_rail()
        self._show_step(self._index)

    # -- orientation -----------------------------------------------------------

    def _stage_orientation(self) -> Optional[str]:
        try:
            return self._microscope.get_stage_orientation()
        except Exception as e:  # noqa: BLE001 - no stage, or no r/t
            logging.debug(f"Could not read the stage orientation: {e}")
            return None

    def _at_calibration_orientation(self) -> bool:
        return self._stage_orientation() == CALIBRATION_ORIENTATION

    def _update_diagram(self, position: FibsemStagePosition) -> None:
        """Draw the stage as it is: its tilt, and mirrored at the FIB orientation."""
        if position is None or position.t is None:
            return
        try:
            orientation = self._microscope.get_stage_orientation(position)
        except Exception:  # noqa: BLE001 - r or t missing; nothing to name
            orientation = ""
        self.stage_diagram.set_orientation(
            name=orientation or "",
            stage_tilt=float(math.degrees(position.t)),
            mirrored=orientation == "FIB",
        )

    def _refresh_orientation_status(self) -> None:
        try:
            self._update_diagram(self._microscope.get_stage_position())
        except Exception as e:  # noqa: BLE001 - the words below still say where we are
            logging.debug(f"Could not read the stage position: {e}")
        orientation = self._stage_orientation()
        if orientation == CALIBRATION_ORIENTATION:
            self._set_status(
                self.orientation_status,
                f"The stage is at the {CALIBRATION_ORIENTATION} orientation.",
                OK_COLOR,
            )
        else:
            shown = orientation or "an unknown orientation"
            self._set_status(
                self.orientation_status,
                f"The stage is at {shown}. Press the button to move it; Next is "
                "refused until it is there.",
                WARN_COLOR,
            )

    def _on_move_to_orientation(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self.button_orientation.setEnabled(False)
        self._set_status(self.orientation_status, "Moving…", TEXT_MUTED_COLOR)
        self._worker = FunctionWorker(
            self._microscope.move_to_orientation, CALIBRATION_ORIENTATION
        )
        self._worker.returned.connect(lambda _: self._after_orientation_move(None))
        self._worker.errored.connect(self._after_orientation_move)
        self._worker.start()

    def _after_orientation_move(self, error) -> None:
        if self._closed:
            return
        self.button_orientation.setEnabled(True)
        if error is not None:
            self._set_status(
                self.orientation_status, f"Move failed: {error}", ERROR_COLOR
            )
            return
        self._refresh_orientation_status()

    # -- slots -----------------------------------------------------------------

    def _refresh_slot_page(self) -> None:
        slot = self.current_slot
        captured = self._captured.get(slot) if slot else None
        if captured is not None:
            self.captured_position.setText(captured.pretty)
            self.captured_position.setStyleSheet(
                f"background: transparent; color: {TEXT_STRONG_COLOR}; font-size: 12px;"
            )
        else:
            existing = self._holder.slots.get(slot) if slot else None
            if existing is not None and existing.position is not None:
                self.captured_position.setText(
                    f"not yet   (saved: {existing.position.pretty})"
                )
            else:
                self.captured_position.setText("not yet")
            self.captured_position.setStyleSheet(
                f"background: transparent; color: {TEXT_MUTED_COLOR}; font-size: 12px;"
            )
        self.button_move_to_slot.setEnabled(captured is not None)
        self._set_status(self.capture_status, "", TEXT_MUTED_COLOR)
        try:
            self._on_stage_moved(self._microscope.get_stage_position())
        except Exception as e:  # noqa: BLE001 - shown as a dash until it moves
            logging.debug(f"Could not read the stage position: {e}")

    def _on_stage_moved(self, position: FibsemStagePosition) -> None:
        if self._closed or position is None:
            return
        self.live_position.setText(position.pretty)
        self._update_diagram(position)

    def _on_capture(self) -> None:
        """Read x/y/z from the stage; r/t come from the calibration orientation."""
        slot = self.current_slot
        if slot is None:
            return
        orientation = self._stage_orientation()
        if orientation != CALIBRATION_ORIENTATION:
            self._set_status(
                self.capture_status,
                f"Refused: the stage is at {orientation or 'an unknown orientation'}, "
                f"not {CALIBRATION_ORIENTATION}. Go back to the Orientation step.",
                ERROR_COLOR,
            )
            return
        try:
            live = self._microscope.get_stage_position()
            reference = self._microscope.get_orientation(CALIBRATION_ORIENTATION)
        except Exception as e:  # noqa: BLE001 - surfaced, not swallowed
            self._set_status(self.capture_status, f"Capture failed: {e}", ERROR_COLOR)
            return
        captured = FibsemStagePosition(
            name=slot,
            x=live.x,
            y=live.y,
            z=live.z,
            r=reference.r,
            t=reference.t,
            coordinate_system=live.coordinate_system,
        )
        self._captured[slot] = captured
        self._set_status(
            self.capture_status,
            f"Captured {slot} at {captured.pretty}.",
            OK_COLOR,
        )
        self._refresh_slot_page()

    def _on_move_to_slot(self) -> None:
        slot = self.current_slot
        captured = self._captured.get(slot) if slot else None
        if captured is None or (self._worker is not None and self._worker.is_alive()):
            return
        self._set_status(
            self.capture_status, "Moving to the captured position…", TEXT_MUTED_COLOR
        )
        self._worker = FunctionWorker(
            self._microscope.safe_absolute_stage_movement, captured
        )
        self._worker.returned.connect(
            lambda _: self._set_status(
                self.capture_status,
                "At the captured position. If the grid centre is under the beam, "
                "move on; otherwise re-centre and Capture again.",
                OK_COLOR,
            )
        )
        self._worker.errored.connect(
            lambda e: self._set_status(
                self.capture_status, f"Move failed: {e}", ERROR_COLOR
            )
        )
        self._worker.start()

    # -- review and save -------------------------------------------------------

    def _position_for(self, slot: str) -> Optional[FibsemStagePosition]:
        if slot in self._captured:
            return self._captured[slot]
        existing = self._holder.slots.get(slot)
        return existing.position if existing is not None else None

    def review_notes(self) -> Dict[str, List[str]]:
        """What the review step says about each slot: warnings, and refusals."""
        notes: Dict[str, List[str]] = {name: [] for name in self.slot_names}
        limits = getattr(getattr(self._microscope, "_stage", None), "limits", {}) or {}
        positions = {name: self._position_for(name) for name in self.slot_names}
        for name, position in positions.items():
            if position is None:
                notes[name].append("no position")
                continue
            if name not in self._captured:
                notes[name].append("kept the saved position")
            for axis in ("x", "y"):
                limit = limits.get(axis)
                value = getattr(position, axis, None)
                if limit is not None and value is not None:
                    if value < limit.min or value > limit.max:
                        notes[name].append(f"outside the stage {axis} limit")
        names = list(positions)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                pa, pb = positions[a], positions[b]
                if pa is None or pb is None:
                    continue
                if None in (pa.x, pa.y, pb.x, pb.y):
                    continue
                if math.hypot(pa.x - pb.x, pa.y - pb.y) < MIN_SLOT_SEPARATION:
                    notes[a].append(f"less than a grid apart from {b}")
                    notes[b].append(f"less than a grid apart from {a}")
        return notes

    def can_save(self) -> bool:
        return not any(
            "outside" in note
            for notes in self.review_notes().values()
            for note in notes
        )

    def _refresh_review(self) -> None:
        notes = self.review_notes()
        names = self.slot_names
        self.review_table.setRowCount(len(names))
        for row, name in enumerate(names):
            position = self._position_for(name)
            self.review_table.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
            self.review_table.setItem(
                row,
                1,
                QtWidgets.QTableWidgetItem(position.pretty if position else "—"),
            )
            note = QtWidgets.QTableWidgetItem("; ".join(notes[name]))
            if any("outside" in n for n in notes[name]):
                note.setForeground(Qt.red)
            self.review_table.setItem(row, 2, note)
        self.review_table.resizeColumnsToContents()

        captured = len(self._captured)
        if not self.can_save():
            self._set_status(
                self.review_summary,
                "A slot is outside the stage limits. Go back and capture it again.",
                ERROR_COLOR,
            )
        elif any("apart" in n for ns in notes.values() for n in ns):
            self._set_status(
                self.review_summary,
                f"{captured} of {len(names)} slots captured. Two slots are closer "
                "than a grid; check them before saving.",
                WARN_COLOR,
            )
        else:
            self._set_status(
                self.review_summary,
                f"{captured} of {len(names)} slots captured this session. Save writes "
                f"{self._save_path}.",
                TEXT_COLOR,
            )
        self.button_next.setEnabled(self.can_save())

    def _save(self) -> None:
        if not self.can_save():
            self._refresh_review()
            return
        holder = self._holder
        holder.name = self._holder_name
        holder.capacity = self._capacity
        holder._ensure_slots()
        stage_settings = getattr(self._microscope.system, "stage", None)
        record = SlotCalibration(
            orientation=CALIBRATION_ORIENTATION,
            pre_tilt=float(getattr(stage_settings, "shuttle_pre_tilt", 0.0)),
            rotation_reference=float(
                getattr(stage_settings, "rotation_reference", 0.0)
            ),
            captured_at=datetime.now().isoformat(timespec="seconds"),
            fibsem_version=fibsem_version(),
        )
        for name, position in self._captured.items():
            holder.slots[name].position = position
            holder.slots[name].calibration = replace(record)
        try:
            holder.save(self._save_path)
        except Exception as e:  # noqa: BLE001 - surfaced on the review page
            self._set_status(self.review_summary, f"Save failed: {e}", ERROR_COLOR)
            return
        logging.info(
            f"Saved sample holder '{holder.name}' with {len(self._captured)} "
            f"recalibrated slot(s) to {self._save_path}."
        )
        self.holder_saved.emit(holder)
        self.accept()

    # -- plumbing --------------------------------------------------------------

    @staticmethod
    def _set_status(label: QtWidgets.QLabel, text: str, colour: str) -> None:
        label.setText(text)
        label.setStyleSheet(
            f"background: transparent; color: {colour}; font-size: 12px;"
        )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._closed = True
        try:
            self._microscope.stage_position_changed.disconnect(self._stage_moved.emit)
        except Exception:  # noqa: BLE001 - never subscribed, or already gone
            pass
        super().closeEvent(event)


__all__ = ["HolderCalibrationDialog", "CALIBRATION_ORIENTATION"]
