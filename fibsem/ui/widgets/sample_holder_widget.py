"""The sample holder as hardware: its slots, whether each is calibrated, and what sits in it.

Two things used to live here and now live elsewhere. Slot *positions* are set only
by the calibration wizard (``holder_calibration_dialog``), which moves the stage to
the right orientation first and writes a record saying so; the per-row capture
button that took whatever the stage said is gone. Holder *geometry* -- the slot
count -- is set in the wizard's first step, because changing it means recalibrating.

What is left is what an operator needs at a glance: is each slot calibrated, when,
against what, which grid is in it, and a way to drive to it. Pre-tilt and reference
rotation come from the system configuration and are shown as facts, not as inputs.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.microscopes._stage import GridSlot, SampleGrid, SampleHolder
from fibsem.ui import stylesheets
from fibsem.ui.tokens import OK_COLOR, SURFACE_COLOR, TEXT_MUTED_COLOR, WARN_COLOR
from fibsem.ui.widgets.custom_widgets import TitledPanel

_ROW_HEIGHT = 36
_SLOT_LABEL_WIDTH = 64
_NAME_FIELD_STYLE = (
    "QLineEdit { background: transparent; border: 1px solid transparent;"
    " border-radius: 3px; padding: 2px 6px; }"
    "QLineEdit:hover, QLineEdit:focus { border-color: #3d4251; background: #1e2027; }"
)


def _set_chip(label: QLabel, text: str, colour: str) -> None:
    """Restyle a pill label in place: text on a tint of its own colour."""
    rgb = QColor(colour)
    label.setText(text)
    label.setStyleSheet(
        f"background-color: rgba({rgb.red()}, {rgb.green()}, {rgb.blue()}, 0.15);"
        f" color: {colour}; padding: 2px 9px; border-radius: 10px; font-size: 11px;"
    )


def _captured_when(slot: GridSlot) -> str:
    """'SEM · 2 Sep 11:24' from the calibration record, or what there is of it."""
    record = slot.calibration
    if record is None:
        return ""
    when = record.captured_at
    if len(when) >= 16 and when[10] == "T":
        # 2026-09-02T11:24:09 -> 2 Sep 11:24, without a datetime round trip
        months = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
        try:
            month = months[int(when[5:7]) - 1]
            when = f"{int(when[8:10])} {month} {when[11:16]}"
        except (ValueError, IndexError):
            pass
    return f"{record.orientation} · {when}" if when else record.orientation


class _SlotRow(QWidget):
    """One slot: name, the grid in it (editable inline), calibration, and Move."""

    move_clicked = pyqtSignal(object)  # GridSlot
    grid_named = pyqtSignal(object, str)  # GridSlot, new name ("" clears it)

    def __init__(self, slot: GridSlot, has_microscope: bool, parent=None) -> None:
        super().__init__(parent)
        self.slot = slot
        self._has_microscope = has_microscope
        self.setAttribute(Qt.WA_TranslucentBackground)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 6, 2)
        layout.setSpacing(8)

        self.slot_label = QLabel()
        self.slot_label.setFixedWidth(_SLOT_LABEL_WIDTH)
        self.slot_label.setStyleSheet("font-weight: bold; background: transparent;")
        layout.addWidget(self.slot_label)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("empty")
        self.name_edit.setToolTip("The grid in this slot; leave blank for none")
        self.name_edit.setStyleSheet(_NAME_FIELD_STYLE)
        self.name_edit.setClearButtonEnabled(True)
        self.name_edit.editingFinished.connect(self._on_name_edited)
        layout.addWidget(self.name_edit, 1)

        self.calibration_chip = QLabel()
        layout.addWidget(self.calibration_chip)

        self.btn_move = QPushButton("Move")
        self.btn_move.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.btn_move.setFixedHeight(24)
        self.btn_move.clicked.connect(lambda: self.move_clicked.emit(self.slot))
        layout.addWidget(self.btn_move)

        self.refresh()

    def refresh(self) -> None:
        slot = self.slot
        self.slot_label.setText(slot.name)
        grid = slot.loaded_grid
        if self.name_edit.text() != (grid.name if grid else ""):
            self.name_edit.setText(grid.name if grid else "")

        if slot.is_calibrated:
            _set_chip(self.calibration_chip, _captured_when(slot), OK_COLOR)
            self.calibration_chip.setToolTip(
                f"Calibrated at the {slot.calibration.orientation} orientation, "
                f"pre-tilt {slot.calibration.pre_tilt:g}°, reference rotation "
                f"{slot.calibration.rotation_reference:g}°\n"
                f"{slot.position.pretty}"
            )
        else:
            _set_chip(self.calibration_chip, "not calibrated", WARN_COLOR)
            self.calibration_chip.setToolTip(
                "No trusted position: run Calibrate slot positions"
            )

        movable = self._has_microscope and slot.is_calibrated
        self.btn_move.setEnabled(movable)
        if movable:
            tip = "Drive the stage to this slot"
        elif self._has_microscope:
            tip = "Not calibrated: nothing to move to"
        else:
            tip = "No microscope connected"
        self.btn_move.setToolTip(tip)

    def _on_name_edited(self) -> None:
        name = self.name_edit.text().strip()
        current = self.slot.loaded_grid.name if self.slot.loaded_grid else ""
        if name != current:
            self.grid_named.emit(self.slot, name)


class SampleHolderWidget(QWidget):
    """The holder's slots: calibration state, the grid in each, and Move.

    ``holder_changed`` fires after the holder object was edited here (a rename, a
    grid named or cleared) and after the wizard saved a calibration; it is what the
    auto-save and any host listens to. ``set_holder`` swaps which holder is shown.
    """

    holder_changed = pyqtSignal(object)  # SampleHolder

    def __init__(self, microscope=None, parent=None):
        super().__init__(parent)
        self._microscope = microscope
        self._holder: Optional[SampleHolder] = None
        self._calibration_dialog = None
        self._rows: List[_SlotRow] = []
        self._setup_ui()
        self.holder_changed.connect(self._auto_save)
        self.setEnabled(False)

    # -- layout ----------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(6, 6, 6, 6)
        inner_layout.setSpacing(6)

        header = QHBoxLayout()
        header.setSpacing(8)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Holder name")
        self.name_edit.setToolTip("What this holder is called in the configuration")
        self.name_edit.setStyleSheet(
            _NAME_FIELD_STYLE + " QLineEdit { font-weight: bold; }"
        )
        self.name_edit.editingFinished.connect(self._on_name_edited)
        header.addWidget(self.name_edit, 1)
        self.status_chip = QLabel()
        header.addWidget(self.status_chip)
        self.btn_calibrate = QPushButton("Calibrate…")
        self.btn_calibrate.setToolTip(
            "Walk through each slot at the SEM orientation and capture its centre.\n"
            "Also where the slot count is set."
        )
        self.btn_calibrate.clicked.connect(self._on_calibrate)
        header.addWidget(self.btn_calibrate)
        inner_layout.addLayout(header)

        self.facts_label = QLabel()
        self.facts_label.setStyleSheet(
            f"color: {TEXT_MUTED_COLOR}; font-size: 11px; background: transparent;"
        )
        inner_layout.addWidget(self.facts_label)

        self._list = QListWidget()
        self._list.setStyleSheet(stylesheets.LIST_WIDGET_STYLESHEET)
        self._list.setSelectionMode(QListWidget.NoSelection)
        self._list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setMinimumHeight(2 * _ROW_HEIGHT)
        inner_layout.addWidget(self._list)

        self.hint_label = QLabel(
            "Slot positions come from the calibration wizard. Until a slot is "
            "calibrated nothing can move to it, and overviews draw no outline for it."
        )
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet(
            f"color: {TEXT_MUTED_COLOR}; font-size: 11px; background: transparent;"
        )
        inner_layout.addWidget(self.hint_label)

        self._panel = TitledPanel("Sample Holder", content=inner, collapsible=True)
        layout.addWidget(self._panel)

    # -- public API ------------------------------------------------------------

    def set_holder(self, holder: Optional[SampleHolder]) -> None:
        self._holder = holder
        self.setEnabled(holder is not None)
        self.refresh()

    @property
    def current_holder(self) -> Optional[SampleHolder]:
        return self._holder

    def refresh(self) -> None:
        holder = self._holder
        self._list.clear()
        self._rows = []
        if holder is None:
            self.name_edit.setText("")
            self.facts_label.setText("")
            _set_chip(self.status_chip, "no holder", TEXT_MUTED_COLOR)
            return

        if self.name_edit.text() != holder.name:
            self.name_edit.setText(holder.name)

        slots = sorted(holder.slots.values(), key=lambda s: s.index)
        calibrated = sum(1 for s in slots if s.is_calibrated)
        if slots and calibrated == len(slots):
            _set_chip(
                self.status_chip, f"{calibrated} of {len(slots)} calibrated", OK_COLOR
            )
        elif calibrated:
            _set_chip(
                self.status_chip,
                f"{calibrated} of {len(slots)} calibrated",
                WARN_COLOR,
            )
        else:
            _set_chip(self.status_chip, "not calibrated", WARN_COLOR)
        self.btn_calibrate.setStyleSheet(
            stylesheets.PRIMARY_BUTTON_STYLESHEET
            if calibrated < len(slots)
            else stylesheets.SECONDARY_BUTTON_STYLESHEET
        )
        self.btn_calibrate.setEnabled(self._microscope is not None)
        self.hint_label.setVisible(calibrated < len(slots))

        plural = "" if len(slots) == 1 else "s"
        self.facts_label.setText(
            f"{len(slots)} slot{plural} · pre-tilt {holder.pre_tilt:g}° · "
            f"reference rotation {holder.reference_rotation:g}°   (system configuration)"
        )

        has_microscope = self._microscope is not None
        for slot in slots:
            row = _SlotRow(slot, has_microscope=has_microscope)
            row.move_clicked.connect(self._on_move_slot)
            row.grid_named.connect(self._on_grid_named)
            item = QListWidgetItem(self._list)
            item.setSizeHint(row.sizeHint())
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self._list.addItem(item)
            self._list.setItemWidget(item, row)
            self._rows.append(row)
        self._list.setFixedHeight(max(2, len(slots)) * _ROW_HEIGHT + 4)

    def _row_widget(self, i: int) -> Optional[_SlotRow]:
        return self._rows[i] if 0 <= i < len(self._rows) else None

    # -- edits -----------------------------------------------------------------

    def _on_name_edited(self) -> None:
        if self._holder is None:
            return
        name = self.name_edit.text().strip()
        if name and name != self._holder.name:
            self._holder.name = name
            self.holder_changed.emit(self._holder)
        elif not name:
            self.name_edit.setText(self._holder.name)

    def _on_grid_named(self, slot: GridSlot, name: str) -> None:
        if self._holder is None:
            return
        if not name:
            slot.loaded_grid = None
        elif slot.loaded_grid is None:
            slot.loaded_grid = SampleGrid(name=name)
        else:
            slot.loaded_grid.name = name
        self.holder_changed.emit(self._holder)

    def _on_move_slot(self, slot: GridSlot) -> None:
        if self._microscope is None:
            return
        try:
            self._microscope._stage.move_to_slot(slot.name)
        except Exception as e:  # noqa: BLE001 - a refused or failed move is reported
            logging.warning(f"Failed to move to slot '{slot.name}': {e}")

    # -- calibration -----------------------------------------------------------

    def _on_calibrate(self) -> None:
        """Open the guided calibration beside the main window; refresh when it saves."""
        if self._microscope is None or self._holder is None:
            return
        from fibsem.ui.widgets.holder_calibration_dialog import (
            HolderCalibrationDialog,
        )

        dialog = HolderCalibrationDialog(self._microscope, self._holder, parent=self)
        dialog.holder_saved.connect(self._on_calibration_saved)
        self._calibration_dialog = dialog  # keep it alive; it is non-modal
        dialog.show()

    def _on_calibration_saved(self, holder: SampleHolder) -> None:
        self.set_holder(holder)
        # The dialog already wrote the file; this tells hosts the slots moved.
        self.holder_changed.emit(holder)

    def _auto_save(self, holder: SampleHolder) -> None:
        if holder is None:
            return
        from fibsem import config as cfg

        try:
            holder.save(cfg.SAMPLE_HOLDER_CONFIGURATION_PATH)
        except Exception as e:  # noqa: BLE001 - a failed save is reported, not fatal
            logging.warning(f"Auto-save of sample holder failed: {e}")


if __name__ == "__main__":
    import sys

    from PyQt5.QtWidgets import QApplication

    from fibsem import utils

    logging.basicConfig(level=logging.DEBUG)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    microscope, settings = utils.setup_session(config_path=None)
    holder = microscope._stage.holder

    widget = SampleHolderWidget(microscope=microscope)
    widget.setStyleSheet(f"background: {SURFACE_COLOR}; color: #d1d2d4;")
    widget.set_holder(holder)

    def on_holder_changed(h: SampleHolder) -> None:
        for slot in h.slots.values():
            g = slot.loaded_grid.name if slot.loaded_grid else "Empty"
            print(f"  {slot.name}: {g}  calibrated={slot.is_calibrated}")

    widget.holder_changed.connect(on_holder_changed)
    widget.resize(520, 320)
    widget.show()
    sys.exit(app.exec_())
