"""Microscope tab · Sample · coating: a sputter coat and a GIS deposition.

Each panel runs one command of the coating service (`fibsem.services.coating`),
the same command the agent's coating tools call, off the GUI thread and in control
of the microscope. A panel shows only when its device is fitted. The deposition
position is the instrument's and is shown, not edited. The devices' single commands
(insert, open, ...) stay in the configuration window's Devices tab.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt import ensure_main_thread

from fibsem.services.coating import CoatingProgress, CoatingStatus, position_text
from fibsem.structures import InsertableDeviceState
from fibsem.ui import stylesheets
from fibsem.ui.control import control_of
from fibsem.ui.qt.threading import FunctionWorker
from fibsem.ui.tokens import TEXT_MUTED_COLOR
from fibsem.ui.widgets.custom_widgets import TitledPanel, ValueComboBox, ValueSpinBox
from fibsem.ui.widgets.progress_widget import FibsemProgressWidget, ProgressUpdate

HIDE_PROGRESS_DELAY_MS = 2000  # how long the "Done" bar stays up before hiding
_MAX_SECONDS = 600


def progress_update(report: CoatingProgress) -> ProgressUpdate:
    """A coating report as the progress bar shows it."""
    if report.status is CoatingStatus.FAILED:
        return ProgressUpdate.failed(report.error or "Failed")
    if report.status is CoatingStatus.CANCELLED:
        return ProgressUpdate(finished=True, message="Cancelled")
    if report.status is CoatingStatus.FINISHED:
        return ProgressUpdate.done()
    if report.remaining_time is not None and report.total_time:
        return ProgressUpdate.countdown(
            report.remaining_time, report.total_time, message=report.step
        )
    return ProgressUpdate.indeterminate(report.step)


def _muted(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet(f"color: {TEXT_MUTED_COLOR}; background: transparent;")
    return label


def _seconds_box(value: float) -> ValueSpinBox:
    box = ValueSpinBox(suffix="s", minimum=1, maximum=_MAX_SECONDS, step=1, decimals=0)
    box.setValue(value)
    return box


class CoatingWidget(QWidget):
    """The Sputter Coater and GIS Deposition panels over ``microscope.coating``."""

    def __init__(self, microscope: Any, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.microscope = microscope
        self.coating = microscope.coating
        self.worker: Optional[FunctionWorker] = None
        self._running: Optional[str] = None  # "sputter" or "deposit" while one runs

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.sputter_panel: Optional[TitledPanel] = None
        self.gis_panel: Optional[TitledPanel] = None
        if self.coating.commands["sputter"].available:
            self.sputter_panel = self._build_sputter()
            layout.addWidget(self.sputter_panel)
        if self.coating.commands["deposit"].available:
            self.gis_panel = self._build_gis()
            layout.addWidget(self.gis_panel)

        self.coating.progress.changed.connect(self._on_progress)
        if self.gis_panel is not None:
            gis = self.coating.gis
            for name in ("state", "heated", "opened", "temperature"):
                if name in gis.parameters:
                    getattr(gis, name).changed.connect(self._on_gis_changed)
            self._show_gis_state()

    # -- the panels ----------------------------------------------------------------

    def _build_sputter(self) -> TitledPanel:
        coater = self.coating.sputter_coater
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(8, 8, 8, 8)
        form = QFormLayout()
        current = coater.current.get_value()
        choices = list(coater.current.metadata.choices or []) or [current]
        self.sputter_current = ValueComboBox(
            items=choices, value=current, unit="A", decimals=1
        )
        form.addRow("Current", self.sputter_current)
        self.sputter_seconds = _seconds_box(30)
        form.addRow("Time", self.sputter_seconds)
        self.sputter_grids: List[QCheckBox] = []
        if coater.coats_grids:
            row = QHBoxLayout()
            for number in (1, 2):
                box = QCheckBox(str(number))
                box.setChecked(True)
                self.sputter_grids.append(box)
                row.addWidget(box)
            row.addStretch(1)
            form.addRow("Grids", row)
        lay.addLayout(form)
        self.sputter_progress = FibsemProgressWidget(inner)
        self.sputter_progress.setVisible(False)
        lay.addWidget(self.sputter_progress)
        self.sputter_button = QPushButton("Sputter")
        self.sputter_button.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.sputter_button.clicked.connect(self.run_sputter)
        lay.addWidget(self.sputter_button)
        return TitledPanel("Sputter Coater", content=inner)

    def _build_gis(self) -> TitledPanel:
        gis = self.coating.gis
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(8, 8, 8, 8)
        form = QFormLayout()
        form.addRow("Gas", QLabel(gis.gas.get_value() or "-"))
        self.gis_state = QLabel()
        form.addRow("State", self.gis_state)
        self.deposit_seconds = _seconds_box(30)
        form.addRow("Time", self.deposit_seconds)
        position = None
        if "deposition_position" in self.coating.parameters:
            position = self.coating.deposition_position.get_value()
        self.deposit_position = QLabel(position_text(position))
        form.addRow("Position", self.deposit_position)
        lay.addLayout(form)
        if position is not None:
            lay.addWidget(
                _muted(
                    "Moves here, inserts, warms, opens for the time, closes and "
                    "retracts, then returns the stage to where it started."
                )
            )
        else:
            lay.addWidget(
                _muted("Inserts, warms, opens for the time, closes and retracts.")
            )
        self.deposit_progress = FibsemProgressWidget(inner)
        self.deposit_progress.setVisible(False)
        lay.addWidget(self.deposit_progress)
        self.deposit_button = QPushButton("Deposit")
        self.deposit_button.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.deposit_button.clicked.connect(self._on_deposit_clicked)
        lay.addWidget(self.deposit_button)
        return TitledPanel("GIS Deposition", content=inner)

    # -- running -------------------------------------------------------------------

    def run_sputter(self) -> None:
        grids = [i + 1 for i, box in enumerate(self.sputter_grids) if box.isChecked()]
        self._start(
            "sputter",
            self.coating.sputter,
            self.sputter_seconds.value(),
            current=self.sputter_current.value(),
            grids=grids or None,
        )

    def run_deposit(self) -> None:
        self._start("deposit", self.coating.deposit, self.deposit_seconds.value())

    def _on_deposit_clicked(self) -> None:
        if self._running == "deposit":
            logging.info("Stopping the deposition...")
            self.coating.stop()
        else:
            self.run_deposit()

    def _start(self, which: str, fn: Any, *args: Any, **kwargs: Any) -> None:
        self._running = which
        self._show_running()
        self.worker = FunctionWorker(fn, *args, **kwargs)
        self.worker.returned.connect(self._finished)
        self.worker.errored.connect(self._errored)
        action = "sputter coat" if which == "sputter" else "GIS deposition"
        self.worker.start(control=control_of(self.microscope), action=action)

    def _finished(self, _result: Any) -> None:
        bar = self._bar()
        self._running = None
        self._show_running()
        if bar is not None:
            QTimer.singleShot(HIDE_PROGRESS_DELAY_MS, bar.reset_if_finished)

    def _errored(self, error: Any) -> None:
        # the failed bar stays up, so a failure is still there to read
        logging.error(f"Coating failed: {error}")
        self._running = None
        self._show_running()

    def _bar(self) -> Optional[FibsemProgressWidget]:
        if self._running == "sputter":
            return self.sputter_progress
        if self._running == "deposit":
            return self.deposit_progress
        return None

    def _show_running(self) -> None:
        """The buttons as the run in progress leaves them: Deposit is Cancel while
        a deposition runs; a sputter run can't be stopped."""
        busy = self._running is not None
        if self.sputter_panel is not None:
            self.sputter_button.setEnabled(not busy)
        if self.gis_panel is not None:
            depositing = self._running == "deposit"
            self.deposit_button.setText("Cancel" if depositing else "Deposit")
            self.deposit_button.setStyleSheet(
                stylesheets.DANGER_BUTTON_STYLESHEET
                if depositing
                else stylesheets.PRIMARY_BUTTON_STYLESHEET
            )
            self.deposit_button.setEnabled(depositing or not busy)

    @ensure_main_thread
    def _on_progress(self, report: CoatingProgress) -> None:
        bar = self._bar()
        if bar is None or report is None:
            return
        bar.setVisible(True)
        bar.update_progress(progress_update(report))

    @ensure_main_thread
    def _on_gis_changed(self, *_: Any) -> None:
        self._show_gis_state()

    def _show_gis_state(self) -> None:
        gis = self.coating.gis
        try:
            state = gis.state.get_value()
            words = [
                "Inserted" if state is InsertableDeviceState.INSERTED else "Retracted",
                "Heated" if gis.heated.get_value() else "Heater off",
                "Open" if gis.opened.get_value() else "Closed",
            ]
        except Exception:  # noqa: BLE001 - a read that fails shows as unknown
            logging.debug("could not read the GIS state", exc_info=True)
            words = ["Unknown"]
        self.gis_state.setText(" · ".join(words))

    def _teardown_connections(self) -> None:
        """Drop the service's and devices' connections; the microscope outlives this
        widget. Idempotent."""
        try:
            self.coating.progress.changed.disconnect(self._on_progress)
        except Exception:
            pass
        if self.gis_panel is not None:
            gis = self.coating.gis
            for name in ("state", "heated", "opened", "temperature"):
                if name in gis.parameters:
                    try:
                        getattr(gis, name).changed.disconnect(self._on_gis_changed)
                    except Exception:
                        pass
