"""The Scripts manager dialog.

Everything the Scripts menu cannot express: the resolved folder, per-script load
errors, which scripts write or run automatically, and the source path and content
hash of what would actually run. A menu has nowhere to put any of that, so a
failed script would simply be absent from it — the silent omission this exists to
avoid.

Application-agnostic: it drives a :class:`ScriptRunner`, which the host wires up.
See FIB-338.
"""

from datetime import datetime

from fibsem.constants import TIME_DISPLAY_AMPM_SHORT
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFontMetrics
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QInputDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fibsem.cancellation import OperationCancelledError
from fibsem.scripting import DiscoveredScript, ScriptResult
from fibsem.ui.stylesheets import (
    PRIMARY_BUTTON_STYLESHEET,
    STOP_WORKFLOW_BUTTON_STYLESHEET,
)
from fibsem.ui.utils import open_path_in_file_explorer
from fibsem.ui.widgets.script_runner import ScriptRunner

# Palette (matching fibsem.ui.stylesheets napari theme)
_BG = "#262930"
_PANEL = "#1e2027"
_ROW_ALT = "#2b2f38"
_BORDER = "#3d4251"
_TEXT = "#d6d6d6"
_TEXT_STRONG = "#f0f1f2"
_TEXT_MUTED = "#868e93"
_ACCENT = "#50a6ff"
_ERROR = "#d04040"

# Colour here says *what a script is*, not how frightened to be. Amber for
# Microscope and Writes meant the two most common rows were permanently lit as
# warnings, which is alarm fatigue in a list you browse rather than act on -- and
# it left red meaning two different things. The real gate is the confirmation
# dialog: modal, worded, defaulting to Cancel, and shown at the moment of action.
# So these are categorical, and _ERROR now means exactly one thing in this
# dialog: the file is broken.
_MICROSCOPE = "#4caf72"
_WRITES = "#4d9de0"

# One type scale for the whole dialog, in px to match the rest of the app's
# stylesheets. Ten literals were scattered across the file, so "make it smaller"
# meant finding all of them and keeping the steps between them consistent by hand.
_FS_TITLE = 14
_FS_NAME = 12
_FS_BODY = 11
_FS_SMALL = 10

_COLUMNS = ["Script", "Type", "Last run"]

# on_workflow_completed is parsed, but nothing fires it: the workflow hook runs on
# a worker thread, and these scripts touch evented experiment state that must only
# be mutated from the GUI thread. It gets no chip -- a row badge for a feature that
# does nothing is a distraction on every row that declares it, and it read as a
# promise besides. Kept in the tooltips, which is where it costs nothing: an author
# who declared the flag still has somewhere to find out it is inert.
AUTO_NOT_CONNECTED = "auto: declared, but scripts do not run automatically yet"

_TEMPLATE = '''"""Describe what this script does — this line becomes its tooltip."""

# writes = True   # uncomment if the script changes state that should be saved


def run(ctx):
    # ctx carries whatever the application provides.
    # Return a string, a Path or a DataFrame to show the result.
    return "done"
'''

_TABLE_STYLE = f"""
QTableWidget {{
    background-color: {_BG};
    alternate-background-color: {_ROW_ALT};
    border: 1px solid {_BORDER};
    border-radius: 6px;
    color: {_TEXT};
    gridline-color: transparent;
    outline: none;
}}
QHeaderView::section {{
    background-color: {_PANEL};
    color: {_TEXT_MUTED};
    border: none;
    padding: 6px 8px;
    font-size: {_FS_SMALL}px;
}}
QTableWidget::item {{ padding: 6px 8px; border-bottom: 1px solid #31353f; }}
QTableWidget::item:selected {{ background-color: #2d3947; color: {_TEXT_STRONG}; }}
"""

_SECONDARY_BUTTON_STYLE = f"""
QPushButton {{
    background-color: transparent;
    border: 1px solid {_BORDER};
    color: {_TEXT};
    border-radius: 6px;
    padding: 5px 12px;
    font-size: {_FS_BODY}px;
}}
QPushButton:hover {{ background-color: {_ROW_ALT}; }}
QPushButton:disabled {{ color: {_TEXT_MUTED}; }}
"""


class _ElidedLabel(QLabel):
    """A QLabel that elides rather than clipping mid-glyph.

    The Script column stretches, so the width is not known when the row is built
    and any one-off elide would be wrong at the next size. Re-eliding on resize
    tracks it. The size policy is Ignored so the full text cannot drive the
    column wider, which would defeat the point.
    """

    def __init__(self, text: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self._full_text = text
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        # At paint time the width is always the real one. Doing this on resize
        # instead misses a label that is laid out at its final size and never
        # resized, which then clips. Re-eliding from _full_text rather than from
        # the displayed text keeps it stable: setText schedules one more paint,
        # the second computes the same string, and it settles.
        metrics = QFontMetrics(self.font())
        elided = metrics.elidedText(self._full_text, Qt.ElideRight, self.width())
        if elided != self.text():
            self.setText(elided)  # QLabel draws it, so the stylesheet colour survives
        super().paintEvent(event)


def _script_type(script: DiscoveredScript) -> "tuple[str, Optional[str]]":
    """(label, chip colour) describing what a script is allowed to touch.

    A ``None`` colour means the type earns no chip. Only Data does: it is the
    default, so it appeared on nearly every row and distinguished nothing, while
    costing width in a column that has to fit three chips. The label is still
    returned, because the tooltip can say "Type: Data" for free.
    """
    if not script.is_runnable:
        return "Error", _ERROR
    if script.uses_microscope:
        return "Microscope", _MICROSCOPE
    return "Data", None


class ScriptManagerDialog(QDialog):
    """Lists every script in the folder — including the ones that failed to load."""

    def __init__(self, runner: ScriptRunner, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.runner = runner
        self.scripts: List[DiscoveredScript] = []
        # Populated as scripts are run, so the table gives immediate feedback.
        # Deliberately not persisted -- nothing records run history yet.
        self.last_run: Dict[str, str] = {}

        self.setWindowTitle("Scripts")
        self.setStyleSheet(f"QDialog {{ background-color: {_BG}; }}")
        self.resize(820, 640)
        self.setMinimumWidth(720)
        self._build()
        self.refresh()

    # --- construction ---

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(2)
        self.title_label = QLabel("User scripts")
        self.title_label.setStyleSheet(
            f"font-size: {_FS_TITLE}px; font-weight: 500; color: {_TEXT_STRONG};"
        )
        # counts and location are meta, not the heading -- the heading says what
        # this dialog is, the line under it says what is currently in it.
        self.meta_label = QLabel()
        self.meta_label.setStyleSheet(f"font-size: {_FS_BODY}px; color: {_TEXT_MUTED};")
        self.meta_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.meta_label.setMinimumWidth(160)
        titles.addWidget(self.title_label)
        titles.addWidget(self.meta_label)
        header.addLayout(titles, 1)  # take the free width so the path can elide into it

        self.open_folder_button = QPushButton("Open folder")
        self.open_folder_button.setStyleSheet(_SECONDARY_BUTTON_STYLE)
        self.open_folder_button.clicked.connect(self.runner.open_folder)
        self.rescan_button = QPushButton("Rescan folder")
        self.rescan_button.setStyleSheet(_SECONDARY_BUTTON_STYLE)
        # "Rescan", not "Reload": scripts are loaded fresh on every run anyway, so
        # this only picks up files added or deleted since the dialog opened.
        self.rescan_button.clicked.connect(self.refresh)
        self.new_script_button = QPushButton("New script…")
        self.new_script_button.setStyleSheet(_SECONDARY_BUTTON_STYLE)
        self.new_script_button.clicked.connect(self.new_script)
        self.change_folder_button = QPushButton("Change folder…")
        self.change_folder_button.setStyleSheet(_SECONDARY_BUTTON_STYLE)
        self.change_folder_button.clicked.connect(self.change_folder)
        header.addWidget(self.new_script_button)
        header.addWidget(self.change_folder_button)
        header.addWidget(self.open_folder_button)
        header.addWidget(self.rescan_button)
        layout.addLayout(header)

        self.table = self._build_table()
        layout.addWidget(self.table)

        detail_panel = QWidget()
        detail_panel.setStyleSheet(
            f"background-color: {_PANEL}; border: 1px solid {_BORDER}; border-radius: 6px;"
        )
        detail_layout = QHBoxLayout(detail_panel)
        detail_layout.setContentsMargins(11, 9, 11, 9)
        self.detail_label = QLabel()
        self.detail_label.setTextFormat(Qt.RichText)
        self.detail_label.setStyleSheet(
            f"border: none; font-size: {_FS_BODY}px; color: {_TEXT};"
        )
        # the consequence of running this sits opposite the facts about it, so it
        # reads as a warning rather than another line of metadata
        self.consequence_label = QLabel()
        self.consequence_label.setTextFormat(Qt.RichText)
        self.consequence_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.consequence_label.setStyleSheet(f"border: none; font-size: {_FS_BODY}px;")
        detail_layout.addWidget(self.detail_label, 1)
        detail_layout.addWidget(self.consequence_label, 0)
        layout.addWidget(detail_panel)

        footer = QHBoxLayout()
        self.hint_label = QLabel()
        self.hint_label.setStyleSheet(f"font-size: {_FS_BODY}px; color: {_TEXT_MUTED};")
        footer.addWidget(self.hint_label)
        footer.addStretch()
        close_button = QPushButton("Close")
        close_button.setStyleSheet(_SECONDARY_BUTTON_STYLE)
        close_button.clicked.connect(self.reject)
        self.run_button = QPushButton("Run script")
        self.run_button.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self.run_button.clicked.connect(self.run_selected)
        footer.addWidget(close_button)
        footer.addWidget(self.run_button)
        layout.addLayout(footer)

    def _build_table(self) -> QTableWidget:
        table = QTableWidget()
        table.setStyleSheet(_TABLE_STYLE)
        table.setColumnCount(len(_COLUMNS))
        table.setHorizontalHeaderLabels(_COLUMNS)
        table.verticalHeader().setVisible(False)
        # rows carry a stacked name + description, so they need room to breathe
        table.verticalHeader().setDefaultSectionSize(56)
        table.setShowGrid(False)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.itemSelectionChanged.connect(self._on_selection_changed)
        table.doubleClicked.connect(self.run_selected)

        header = table.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, header.Stretch)
        table.setColumnWidth(2, 130)
        return table

    @staticmethod
    def _chip(text: str, colour: str) -> QLabel:
        """A pill label: text on a tint of its own colour.

        No dot. Once every chip had one it separated nothing, and it cost real
        width in a 130px column that has to fit three chips on one row.
        """
        rgb = QColor(colour)
        tint = f"rgba({rgb.red()}, {rgb.green()}, {rgb.blue()}, 0.15)"
        chip = QLabel(text)
        chip.setStyleSheet(
            f"background-color: {tint}; color: {colour};"
            f"padding: 2px 9px; border-radius: 10px; font-size: {_FS_SMALL}px;"
        )
        # sizeHint does not account for the stylesheet padding, so the last character
        # clips. Measure the text and pin the width -- at the size it actually
        # renders at, or the default font's metrics pad every chip by the difference.
        font = chip.font()
        font.setPixelSize(_FS_SMALL)
        metrics = QFontMetrics(font)
        chip.setMinimumWidth(metrics.horizontalAdvance(text) + 26)
        return chip

    def _name_cell(self, script: DiscoveredScript) -> QWidget:
        """Two lines: the filename, and what it does (or why it will not load)."""
        widget = QWidget()
        widget.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 8, 10, 8)
        layout.setSpacing(2)

        summary = script.description if script.is_runnable else script.error
        name = _ElidedLabel(script.name)
        name.setStyleSheet(
            f"font-family: Menlo, monospace; font-size: {_FS_NAME}px;"
            f"background: transparent;"
            f"color: {_TEXT_STRONG if script.is_runnable else _TEXT_MUTED};"
        )
        detail = _ElidedLabel(summary)
        detail.setStyleSheet(
            f"font-size: {_FS_BODY}px; background: transparent;"
            f"color: {_TEXT if script.is_runnable else _ERROR};"
        )
        layout.addWidget(name)
        layout.addWidget(detail)
        # the summary is elided in the row, so the tooltip has to carry it in full
        widget.setToolTip(f"{summary}\n\n{script.path}\n{script.content_hash}")
        return widget

    def _type_cell(self, script: DiscoveredScript) -> QWidget:
        """Chips for anything out of the ordinary; the tooltip carries the rest.

        Only Microscope, Error and Writes get a chip. Data is the default and
        on_workflow_completed does nothing yet, so both would be badges that appear
        constantly and tell you no more than their absence would.
        """
        widget = QWidget()
        widget.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        label, colour = _script_type(script)
        if colour is not None:
            layout.addWidget(self._chip(label, colour))
        if script.writes:
            layout.addWidget(self._chip("Writes", _WRITES))
        layout.addStretch()
        notes = [f"Type: {label}"]
        if script.writes:
            notes.append("writes: saves the experiment when finished")
        if script.on_workflow_completed:
            notes.append(AUTO_NOT_CONNECTED)
        widget.setToolTip("\n".join(notes))
        return widget

    # --- data ---

    def refresh(self, keep_selection: bool = True) -> None:
        """Re-read the folder and rebuild the table.

        The selected script is restored by name, so running one does not bounce
        the selection back to the top of the list.
        """
        previous = self.selected_script().name if (keep_selection and self.selected_script()) else None

        self.scripts = self.runner.discover()
        failed = sum(1 for s in self.scripts if not s.is_runnable)
        runnable = len(self.scripts) - failed

        summary = f"{runnable} script{'s' if runnable != 1 else ''}"
        if failed:
            summary += f", {failed} failed to load"
        self._summary = summary
        self._update_meta()

        self.table.clearContents()
        self.table.setRowCount(0)  # destroys the previous rows' cell widgets
        self.table.setRowCount(len(self.scripts))
        for row, script in enumerate(self.scripts):
            # cell widgets, not items: a QTableWidgetItem cannot carry a pill
            # background, and the name cell needs two lines.
            self.table.setCellWidget(row, 0, self._name_cell(script))
            self.table.setCellWidget(row, 1, self._type_cell(script))

            last_item = QTableWidgetItem(self.last_run.get(script.name, "—"))
            last_item.setForeground(QColor(_TEXT_MUTED))
            self.table.setItem(row, 2, last_item)
            # an empty item under each widget keeps row selection working
            for col in (0, 1):
                self.table.setItem(row, col, QTableWidgetItem())

        self._fit_type_column()

        if self.scripts:
            names = [s.name for s in self.scripts]
            row = names.index(previous) if previous in names else 0
            self.table.selectRow(row)
        self._on_selection_changed()

    def _fit_type_column(self) -> None:
        """Widen the type column to the widest row of chips.

        ResizeToContents is no use here: it measures the item's size hint, and
        these cells hold widgets over empty items. A row carries at most one type
        chip plus a chip per flag, so any fixed width clips as soon as a script
        declares more than one.

        Floored at the header's own width: a folder of plain data scripts now has
        no chips at all, which collapsed this to 28px and clipped the word "Type".
        """
        widths = [
            self.table.cellWidget(row, 1).sizeHint().width()
            for row in range(self.table.rowCount())
            if self.table.cellWidget(row, 1) is not None
        ]
        header = self.table.horizontalHeader()
        floor = header.fontMetrics().horizontalAdvance(_COLUMNS[1]) + 24
        self.table.setColumnWidth(1, max(max(widths, default=0) + 12, floor))

    def _update_meta(self) -> None:
        """Counts plus the folder, elided to whatever width the dialog has.

        The path is often far longer than anything else in the dialog and would
        otherwise dictate its width. The full value stays in the tooltip.
        """
        directory = str(self.runner.scripts_directory())
        # elide against the label's real width -- with a floor, because showEvent
        # fires before layout and a near-zero width elides the path away entirely
        available = max(self.meta_label.width(), 240)
        elided = QFontMetrics(self.meta_label.font()).elidedText(
            directory, Qt.ElideMiddle, available
        )
        self.meta_label.setText(f"{self._summary}  ·  {elided}")
        self.meta_label.setToolTip(directory)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        if getattr(self, "_summary", None) is not None:
            self._update_meta()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().showEvent(event)
        self._update_meta()

    # --- selection + running ---

    def selected_script(self) -> Optional[DiscoveredScript]:
        row = self.table.currentRow()
        if 0 <= row < len(self.scripts):
            return self.scripts[row]
        return None

    def _on_selection_changed(self) -> None:
        script = self.selected_script()
        host_ready, reason = self.runner.availability()

        if script is None:
            empty = (
                "No scripts in this folder yet — use New script… to create one."
                if not self.scripts else "No script selected."
            )
            self.detail_label.setText(f'<span style="color:{_TEXT_MUTED};">{empty}</span>')
            self.consequence_label.setText("")
            self.run_button.setEnabled(False)
            self.hint_label.setText(reason)
            return

        self.detail_label.setText(
            f'<span style="color:{_TEXT_MUTED};">Source</span>&nbsp; {script.path.name}'
            f'<br><span style="color:{_TEXT_MUTED};">Hash</span>&nbsp; {script.content_hash}'
        )

        # short phrases, not sentences: this sits opposite the Source/Hash block and
        # a full sentence clips against it once the dialog narrows. The tooltip
        # carries the longer wording.
        if not script.is_runnable:
            consequence, colour, explain = "● Cannot load", _ERROR, script.error
        elif script.uses_microscope:
            # matches the Microscope chip: the words carry the weight here, and the
            # confirmation dialog is what actually stops you
            consequence, colour, explain = (
                "● Controls the microscope", _MICROSCOPE,
                "This script controls the hardware directly. Nothing checks what it "
                "does — no limits, no interlocks. It runs in the background, and Stop "
                "only works if the script itself checks for it.",
            )
        elif script.writes:
            consequence, colour, explain = (
                "● Modifies and saves the experiment", _WRITES,
                "This script changes the experiment and saves it when it finishes.",
            )
        else:
            consequence, colour, explain = (
                "● Read-only", _TEXT_MUTED, "This script does not change anything.",
            )
        if script.on_workflow_completed:
            # already a chip on the row -- repeating it on screen is what pushed
            # this line past the panel edge on a narrow dialog, so tooltip only
            explain += f" {AUTO_NOT_CONNECTED}."
        self.consequence_label.setToolTip(explain)
        # Same trap as the chips: a rich-text QLabel under-reports sizeHint
        # against its rendered width, so the tail clips. Pin it from the metrics
        # of the plain text before wrapping it in the colour span.
        metrics = QFontMetrics(self.consequence_label.font())
        self.consequence_label.setMinimumWidth(metrics.horizontalAdvance(consequence) + 10)
        consequence = f'<span style="color:{colour};">{consequence}</span>'
        self.consequence_label.setText(consequence)

        self.run_button.setEnabled(script.is_runnable and host_ready)
        self.hint_label.setText(reason)
        self._sync_run_button()

    def change_folder(self) -> None:
        """Point the dialog at a different folder for this session."""
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose a scripts folder", str(self.runner.scripts_directory())
        )
        if not chosen:
            return
        self.runner.set_directory(Path(chosen))
        self.refresh(keep_selection=False)

    def new_script(self) -> None:
        """Create a stub script in the current folder and reveal it."""
        name, accepted = QInputDialog.getText(self, "New script", "File name:", text="my_script")
        if not accepted or not name.strip():
            return

        directory = self.runner.scripts_directory()
        path = directory / (name.strip() if name.strip().endswith(".py") else f"{name.strip()}.py")
        if path.exists():
            self.runner.notify(f"{path.name} already exists.", "warning")
            return

        try:
            directory.mkdir(parents=True, exist_ok=True)
            path.write_text(_TEMPLATE)
        except OSError as e:
            self.runner.notify(f"Could not create {path.name}: {e}", "error")
            return

        self.refresh(keep_selection=False)
        names = [s.name for s in self.scripts]
        if path.stem in names:
            self.table.selectRow(names.index(path.stem))
        open_path_in_file_explorer(str(directory))

    def run_selected(self) -> None:
        # while a script is running this button is Stop -- see _sync_run_button
        if self.runner.is_running:
            self.runner.stop()
            return

        script = self.selected_script()
        if script is None or not script.is_runnable:
            return

        # the runner owns the wait cursor -- it has to be set after its
        # confirmation, not before
        self.runner.run(script, on_finished=lambda r, s=script: self._record_run(s, r))
        # a microscope script is still running at this point -- the stamp and the
        # refresh happen in _record_run, which the worker delivers on the GUI thread
        self._sync_run_button()

    def _record_run(self, script: DiscoveredScript, result: ScriptResult) -> None:
        """Stamp the outcome once the script has actually finished."""
        outcome = "cancelled" if isinstance(result.error, OperationCancelledError) else (
            "ok" if result.ok else "failed"
        )
        stamp = datetime.now().strftime(TIME_DISPLAY_AMPM_SHORT).lstrip('0').lower()
        self.last_run[script.name] = f"{stamp} {outcome}"
        self._sync_run_button()
        self.refresh()

    def _sync_run_button(self) -> None:
        """Run, or Stop while a script is running.

        One button rather than two: a Stop that is greyed out almost always is
        noise, and the state it reports is the same state Run is reporting.
        """
        running = self.runner.is_running
        self.run_button.setText("Stop script" if running else "Run script")
        self.run_button.setStyleSheet(
            STOP_WORKFLOW_BUTTON_STYLESHEET if running else PRIMARY_BUTTON_STYLESHEET
        )
        if running:
            self.run_button.setEnabled(True)  # Stop must never be the disabled one
        # everything that would start a second run, or move the ground under the
        # one already going
        for button in (self.new_script_button, self.change_folder_button,
                       self.rescan_button):
            button.setEnabled(not running)
        self.table.setEnabled(not running)
