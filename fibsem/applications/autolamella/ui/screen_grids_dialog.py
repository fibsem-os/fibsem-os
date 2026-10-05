"""Screen all grids: read the magazine, scan it if it needs one, name the grids, run.

One dialog in front of the run (FIB-1138). It opens on what the autoloader
already knows. A magazine that has not been scanned since it was opened reads
every slot as unknown, so the dialog offers the scan; the scan takes about ten
minutes, and the operator names the grids by slot while it runs. Nothing is
written while the dialog is open: the names are held here, and **Screen** is the
only commit. It writes the names that differ to their slots, records the grids
under their final names, and hands the run exactly those grids. Cancel writes
nothing and starts nothing; a scan already under way still finishes on its own.

A slot whose description is already set keeps that name after the scan (it is
the name xT shows), and its row says what it replaced. On a fixed holder there
is nothing to scan, so the dialog opens at the review.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.ui.grid_workflow_widget import (
    beam_name,
    is_default_grid_name,
)
from fibsem.applications.autolamella.workflows.tasks.grid.manager import (
    LOAD_ENTRY_NAME,
    grid_has_run,
    plan_grid_run,
)
from fibsem.applications.autolamella.workflows.tasks.grid.screening import (
    GridNamingError,
    name_and_record_grids,
    naming_problems,
    screen_names,
)
from fibsem.microscopes._stage import GridInventoryEntry, GridSlotState
from fibsem.structures import BeamType
from fibsem.ui import stylesheets
from fibsem.ui.qt.threading import thread_worker
from fibsem.ui.tokens import (
    ACCENT_COLOR,
    DISABLED_TEXT_COLOR,
    ERROR_COLOR,
    TEXT_COLOR,
    TEXT_MUTED_COLOR,
)
from fibsem.ui.widgets.preflight import (
    BACKGROUND,
    ON_PANEL,
    TEXT_STRONG,
    detail_block,
    meta_label,
    metric,
    warning_label,
)
from fibsem.ui.widgets.sample_holder_widget import _NAME_FIELD_STYLE

# The stages the dialog moves through. Reading is the instant read on opening.
READING = "reading"
NOT_SCANNED = "not_scanned"
SCANNING = "scanning"
REVIEW = "review"

_STEPS_SHOWN = 12
_NOTE_STYLE = f"color: {TEXT_MUTED_COLOR}; font-size: 11px; {ON_PANEL}"
_WARN_NOTE_STYLE = f"color: {stylesheets.SEMANTIC_WARNING_COLOR}; font-size: 11px;"
_LINK_STYLE = f"color: {ACCENT_COLOR}; text-decoration: none;"

# A scan cancelled with the dialog keeps running on its own thread; the worker
# is held here until it ends, so it is not collected mid-scan, and a second
# Screen all grids can be refused while it runs.
_scans_running: List[object] = []


def scan_in_progress() -> bool:
    """Whether a scan started from this dialog is still running."""
    return bool(_scans_running)


def _link(text: str, href: str) -> str:
    return f'<a href="{href}" style="{_LINK_STYLE}">{text}</a>'


@dataclass
class _Row:
    """One slot's line: its field, its state, and the note under the name."""

    slot_name: str
    number: int
    field: QLineEdit
    state: QLabel
    note: QLabel
    # The name the slot's description held after the scan, when it was set:
    # the name xT shows. None for an unnamed (default) slot.
    xt_name: Optional[str] = None
    # What the operator had typed when the scan brought an xT name in its place.
    replaced: str = ""


class ScreenGridsDialog(QDialog):
    """Scan, name and confirm before Screen all grids runs.

    ``grid_names`` holds the grids to screen once the dialog is accepted, named
    and recorded. ``synchronous`` runs the read and the scan inline, for tests.
    """

    def __init__(
        self,
        stage,
        experiment,
        task_names: List[str],
        output_root: str,
        beams_off: Sequence[BeamType] = (),
        synchronous: bool = False,
        clock: Callable[[], float] = time.monotonic,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.stage = stage
        self.experiment = experiment
        self.task_names = list(task_names)
        self.grid_names: List[str] = []
        self._beams_off = list(beams_off)
        self._synchronous = synchronous
        self._clock = clock
        self._stage_name = READING
        self._inventory: List[GridInventoryEntry] = []
        self._scanned_once = False
        self._scan_started = 0.0
        self._worker = None
        self._closed = False
        self._rows: Dict[str, _Row] = {}

        self.setWindowTitle("Screen all grids")
        self.setMinimumWidth(640)
        self.setStyleSheet(f"background: {BACKGROUND};")
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        title = QLabel("Name the grids, then run the tasks on every present grid")
        title.setStyleSheet(
            f"font-size: 14px; font-weight: bold; color: {TEXT_STRONG}; {ON_PANEL}"
        )
        layout.addWidget(title)

        self.banner = warning_label("")
        self.banner.hide()
        layout.addWidget(self.banner)

        busy = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        self.progress.setStyleSheet(stylesheets.INDETERMINATE_PROGRESS_BAR_STYLESHEET)
        self.elapsed_label = QLabel("")
        self.elapsed_label.setStyleSheet(_NOTE_STYLE)
        busy.addWidget(self.progress, stretch=1)
        busy.addWidget(self.elapsed_label)
        self.busy_row = QWidget()
        self.busy_row.setLayout(busy)
        self.busy_row.hide()
        layout.addWidget(self.busy_row)

        self.slots_box = QWidget()
        self.slots_grid = QGridLayout(self.slots_box)
        self.slots_grid.setContentsMargins(0, 0, 0, 0)
        self.slots_grid.setHorizontalSpacing(10)
        self.slots_grid.setVerticalSpacing(2)
        layout.addWidget(self.slots_box)

        # The summary, the plan and the warnings: rebuilt on every change.
        self.summary_box = QWidget()
        self.summary_layout = QVBoxLayout(self.summary_box)
        self.summary_layout.setContentsMargins(0, 0, 0, 0)
        self.summary_layout.setSpacing(8)
        layout.addWidget(self.summary_box)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet(f"color: {ERROR_COLOR}; font-size: 11px;")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        layout.addWidget(self.error_label)

        buttons = QHBoxLayout()
        self.btn_scan = QPushButton("Scan magazine")
        self.btn_scan.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.btn_scan.clicked.connect(self._on_scan)
        buttons.addWidget(self.btn_scan)
        buttons.addStretch(1)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_screen = QPushButton("Screen")
        self.btn_screen.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.btn_screen.clicked.connect(self._on_screen)
        buttons.addWidget(self.btn_cancel)
        buttons.addWidget(self.btn_screen)
        layout.addLayout(buttons)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)

        self._run(self.stage.get_inventory, self._on_read, self._on_read_failed)
        self._apply()

    # -- what the dialog knows -------------------------------------------------

    @property
    def stage_name(self) -> str:
        return self._stage_name

    @property
    def has_loader(self) -> bool:
        return self.stage.loader is not None

    def names(self) -> Dict[str, str]:
        """The name typed for each slot, as held: nothing is written until Screen."""
        return {slot: row.field.text().strip() for slot, row in self._rows.items()}

    def _entry(self, slot_name: str) -> Optional[GridInventoryEntry]:
        return next((e for e in self._inventory if e.slot_name == slot_name), None)

    def _final_names(self) -> Dict[str, str]:
        return screen_names(self._inventory, self.names())

    # -- the read and the scan, off the GUI thread -------------------------------

    def _run(self, job, on_done, on_failed) -> None:
        if self._synchronous:
            try:
                job()
            except Exception as e:  # noqa: BLE001 - said in the dialog
                on_failed(e)
                return
            on_done()
            return
        worker = self._job_worker(job)
        worker.returned.connect(lambda _r: on_done())
        worker.errored.connect(on_failed)
        worker.finished.connect(lambda: self._forget(worker))
        self._worker = worker
        _scans_running.append(worker)
        worker.start()

    @thread_worker
    def _job_worker(self, job) -> None:
        job()

    @staticmethod
    def _forget(worker) -> None:
        if worker in _scans_running:
            _scans_running.remove(worker)

    def _on_read(self) -> None:
        if self._closed:
            return
        self._inventory = self.stage.grid_inventory()
        unknown = any(e.state is GridSlotState.UNKNOWN for e in self._inventory)
        self._stage_name = NOT_SCANNED if self.has_loader and unknown else REVIEW
        self._build_rows()
        if self._stage_name == REVIEW:
            self._fill_from_slots()
        self._apply()

    def _on_read_failed(self, error: Exception) -> None:
        if self._closed:
            return
        logging.warning(f"Could not read the magazine: {error}")
        self._inventory = self.stage.grid_inventory()
        self._stage_name = NOT_SCANNED if self.has_loader else REVIEW
        self._build_rows()
        self._show_error(f"Could not read the magazine: {error}")
        self._apply()

    def _on_scan(self) -> None:
        if not self.has_loader or self._stage_name == SCANNING:
            return
        self._stage_name = SCANNING
        self._scan_started = self._clock()
        self._show_error("")
        self._timer.start()
        self._apply()
        self._run(self.stage.run_inventory, self._on_scanned, self._on_scan_failed)

    def _on_scanned(self) -> None:
        self._timer.stop()
        if self._closed:
            return
        self._scanned_once = True
        self._inventory = self.stage.grid_inventory()
        self._stage_name = REVIEW
        self._take_xt_names()
        self._apply()

    def _on_scan_failed(self, error: Exception) -> None:
        self._timer.stop()
        if self._closed:
            return
        logging.warning(f"Magazine scan failed: {error}")
        self._inventory = self.stage.grid_inventory()
        unknown = any(e.state is GridSlotState.UNKNOWN for e in self._inventory)
        self._stage_name = NOT_SCANNED if unknown else REVIEW
        self._show_error(f"The scan did not finish: {error}")
        self._apply()

    def _tick(self) -> None:
        self._apply_busy()

    # -- the rows ----------------------------------------------------------------

    def _build_rows(self) -> None:
        """One row per slot, built once: a field keeps what is typed in it while
        the scan runs and the rows below it change."""
        if self._rows:
            return
        for i, entry in enumerate(self._inventory):
            number = QLabel(f"{entry.index + 1:02d}")
            number.setStyleSheet(_NOTE_STYLE)
            field = QLineEdit()
            field.setStyleSheet(_NAME_FIELD_STYLE)
            field.setFixedHeight(26)
            field.textEdited.connect(lambda _t, s=entry.slot_name: self._on_edited(s))
            state = QLabel("")
            state.setStyleSheet(_NOTE_STYLE)
            note = QLabel("")
            note.setStyleSheet(_NOTE_STYLE)
            note.setTextFormat(Qt.RichText)
            note.linkActivated.connect(
                lambda href, s=entry.slot_name: self._on_link(s, href)
            )
            self.slots_grid.addWidget(number, i, 0)
            self.slots_grid.addWidget(field, i, 1)
            self.slots_grid.addWidget(state, i, 2)
            self.slots_grid.addWidget(note, i, 3)
            self.slots_grid.setColumnStretch(1, 1)
            self.slots_grid.setColumnStretch(3, 1)
            self._rows[entry.slot_name] = _Row(
                entry.slot_name, entry.index + 1, field, state, note
            )

    def _fill_from_slots(self) -> None:
        """Already scanned: the names the slots hold are the starting point."""
        for entry in self._inventory:
            row = self._rows.get(entry.slot_name)
            if row is None or not entry.present or not entry.name:
                continue
            if not is_default_grid_name(entry.name):
                row.xt_name = entry.name
                row.field.setText(entry.name)

    def _take_xt_names(self) -> None:
        """After a scan, a slot whose description is set keeps that name. What
        the operator typed there is kept on the row, to take back with a click."""
        for entry in self._inventory:
            row = self._rows.get(entry.slot_name)
            if row is None:
                continue
            typed = row.field.text().strip()
            if entry.present and entry.name and not is_default_grid_name(entry.name):
                row.xt_name = entry.name
                if typed and typed != entry.name:
                    row.replaced = typed
                row.field.setText(entry.name)
            else:
                row.xt_name = None
                row.replaced = ""

    def _on_edited(self, _slot_name: str) -> None:
        self._show_error("")
        self._apply()

    def _on_link(self, slot_name: str, href: str) -> None:
        """A row's link: take a name back, the typed one or the xT one."""
        row = self._rows.get(slot_name)
        if row is None:
            return
        if href == "typed" and row.replaced:
            row.field.setText(row.replaced)
        elif href == "xt" and row.xt_name:
            row.field.setText(row.xt_name)
        self._apply()

    # -- drawing -----------------------------------------------------------------

    def _apply(self) -> None:
        self._apply_banner()
        self._apply_busy()
        for row in self._rows.values():
            self._apply_row(row)
        self._apply_summary()
        self._apply_buttons()

    def _apply_banner(self) -> None:
        text = ""
        if self._stage_name == READING:
            text = "Reading the magazine…"
        elif self._stage_name == NOT_SCANNED:
            text = (
                "The magazine has not been scanned since it was opened, so no slot "
                "is known. Scan it to find the grids. You can name them by slot "
                "while it scans."
            )
        elif self._stage_name == SCANNING:
            text = (
                "Scanning the magazine. Keep it shut until the scan ends. Names "
                "typed now are used when you press Screen."
            )
        self.banner.setText(text)
        self.banner.setVisible(bool(text))

    def _apply_busy(self) -> None:
        busy = self._stage_name in (READING, SCANNING)
        self.busy_row.setVisible(busy)
        if self._stage_name == SCANNING:
            seconds = int(self._clock() - self._scan_started)
            self.elapsed_label.setText(
                f"{seconds // 60}:{seconds % 60:02d} elapsed · a scan takes about "
                "10 minutes"
            )
        else:
            self.elapsed_label.setText("")

    def _apply_row(self, row: _Row) -> None:
        entry = self._entry(row.slot_name)
        text = row.field.text().strip()
        record = (
            self.experiment.get_grid_by_name(entry.name)
            if entry is not None and entry.name
            else None
        )
        has_run = record is not None and grid_has_run(record)
        row.field.setReadOnly(has_run)
        state, note = "", ""
        if entry is None or entry.state is GridSlotState.UNKNOWN:
            state = "scanning" if self._stage_name == SCANNING else "unknown"
            row.field.setPlaceholderText(f"Grid-{row.number:02d}")
        elif not entry.present:
            state = "empty"
            row.field.setPlaceholderText("")
            if text:
                note = "Empty slot: this name will not be used."
        else:
            state = "loaded" if entry.loaded else "present"
            row.field.setPlaceholderText(entry.name or "")
            if has_run:
                note = "Has run: its name is fixed."
                if text != entry.name:
                    row.field.setText(entry.name)
            elif row.xt_name and text and text != row.xt_name:
                note = (
                    f"Will replace the xT name '{row.xt_name}' · "
                    f"{_link(f'Use {row.xt_name!r}', 'xt')}"
                )
            elif row.xt_name and (not text or text == row.xt_name):
                note = "From xT"
                if row.replaced:
                    note += (
                        f" · replaced '{row.replaced}' · "
                        f"{_link(f'Use {row.replaced!r}', 'typed')}"
                    )
            elif not text and entry.name and is_default_grid_name(entry.name):
                note = "Default name"
        row.state.setText(state)
        row.note.setText(note)
        greyed = entry is not None and entry.state is GridSlotState.EMPTY
        color = DISABLED_TEXT_COLOR if greyed else TEXT_COLOR
        row.field.setStyleSheet(_NAME_FIELD_STYLE + f" QLineEdit {{ color: {color}; }}")
        row.note.setStyleSheet(
            _WARN_NOTE_STYLE if note.startswith("Will replace") else _NOTE_STYLE
        )

    def _apply_summary(self) -> None:
        while self.summary_layout.count():
            item = self.summary_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if self._stage_name != REVIEW:
            return
        final = self._final_names()
        names = list(final.values())
        loaded = {e.name for e in self._inventory if e.loaded}
        exchanges = (
            sum(1 for e in self._inventory if e.present and e.name not in loaded)
            if self.has_loader
            else 0
        )
        metrics = QHBoxLayout()
        metrics.addWidget(metric("Grids", str(len(names))))
        metrics.addWidget(metric("Tasks per grid", str(len(self.task_names))))
        metrics.addWidget(
            metric("Exchanges", str(exchanges), "" if exchanges else "all loaded")
        )
        box = QWidget()
        box.setLayout(metrics)
        self.summary_layout.addWidget(box)

        plan = plan_grid_run(self.task_names, names)
        lines = [
            f"{grid}  ·  {'load' if step == LOAD_ENTRY_NAME else step}"
            for grid, step in plan[:_STEPS_SHOWN]
        ]
        if len(plan) > _STEPS_SHOWN:
            lines.append(f"… and {len(plan) - _STEPS_SHOWN} more steps")
        rows = (
            [("Plan", lines[0])] + [("", line) for line in lines[1:]]
            if lines
            else [("Plan", "no grids found")]
        )
        self.summary_layout.addWidget(detail_block(rows))
        self.summary_layout.addWidget(
            meta_label(f"Output: {self.experiment.path}/grids/<grid>/")
        )

        if self._beams_off:
            beams = " and ".join(beam_name(b) for b in self._beams_off)
            plural = len(self._beams_off) > 1
            self.summary_layout.addWidget(
                warning_label(
                    f"The {beams} beam{'s are' if plural else ' is'} off. The run "
                    f"turns {'them' if plural else 'it'} on when it starts."
                )
            )
        first = [
            name
            for name in names
            if not (
                (record := self.experiment.get_grid_by_name(name)) is not None
                and grid_has_run(record)
            )
        ]
        if first:
            listed = ", ".join(
                f"{n} (default name)" if is_default_grid_name(n) else n for n in first
            )
            self.summary_layout.addWidget(
                warning_label(
                    f"Names are fixed once a grid has run. Running for the first "
                    f"time: {listed}. This is the last chance to change them."
                )
            )
        for problem in naming_problems(self.experiment, self._inventory, self.names()):
            label = QLabel(problem)
            label.setStyleSheet(f"color: {ERROR_COLOR}; font-size: 11px;")
            label.setWordWrap(True)
            self.summary_layout.addWidget(label)

    def _apply_buttons(self) -> None:
        stage = self._stage_name
        self.btn_scan.setVisible(self.has_loader)
        self.btn_scan.setText(
            "Scan again" if stage == REVIEW or self._scanned_once else "Scan magazine"
        )
        self.btn_scan.setEnabled(stage in (NOT_SCANNED, REVIEW))
        count = len(self._final_names()) if stage == REVIEW else 0
        self.btn_screen.setText(
            f"Screen {count} grid{'s' if count != 1 else ''}"
            if stage == REVIEW
            else "Screen"
        )
        ok = (
            stage == REVIEW
            and count > 0
            and not naming_problems(self.experiment, self._inventory, self.names())
        )
        self.btn_screen.setEnabled(ok)
        # The main button is the next step: the scan until there is something
        # to screen.
        primary, secondary = (
            (self.btn_scan, self.btn_screen)
            if stage == NOT_SCANNED
            else (self.btn_screen, self.btn_scan)
        )
        primary.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        primary.setDefault(True)
        secondary.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        secondary.setDefault(False)

    def _show_error(self, text: str) -> None:
        self.error_label.setText(text)
        self.error_label.setVisible(bool(text))

    # -- the commit ----------------------------------------------------------------

    def _on_screen(self) -> None:
        """The one commit: name the slots, record the grids, and accept."""
        if self._stage_name != REVIEW:
            return
        try:
            self.grid_names = name_and_record_grids(
                self.stage, self.experiment, self.names()
            )
        except GridNamingError as e:
            # Slots written before a refused one keep their names; read the
            # inventory again so the rows show what the hardware holds.
            self._inventory = self.stage.grid_inventory()
            self._show_error(str(e))
            self._apply()
            return
        self._closed = True
        self.accept()

    def reject(self) -> None:
        """Cancel writes nothing. A scan under way finishes on its own."""
        self._closed = True
        self._timer.stop()
        super().reject()
