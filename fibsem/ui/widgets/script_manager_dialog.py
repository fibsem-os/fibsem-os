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
from typing import Dict, List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fibsem.scripting import DiscoveredScript
from fibsem.ui.stylesheets import PRIMARY_BUTTON_STYLESHEET
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
_WARN = "#e0a030"
_ERROR = "#d04040"

_COLUMNS = ["Script", "Type", "Last run"]

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
    font-size: 11px;
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
    font-size: 12px;
}}
QPushButton:hover {{ background-color: {_ROW_ALT}; }}
QPushButton:disabled {{ color: {_TEXT_MUTED}; }}
"""


def _script_type(script: DiscoveredScript) -> "tuple[str, str]":
    """(label, colour) describing what a script is allowed to touch."""
    if not script.is_runnable:
        return "Error", _ERROR
    if script.uses_microscope:
        return "Microscope", _WARN
    return "Data", _TEXT_MUTED


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
        self.resize(760, 580)
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
        self.title_label = QLabel()
        self.title_label.setStyleSheet(
            f"font-size: 14px; font-weight: 500; color: {_TEXT_STRONG};"
        )
        self.path_label = QLabel()
        self.path_label.setStyleSheet(f"font-size: 12px; color: {_TEXT_MUTED};")
        titles.addWidget(self.title_label)
        titles.addWidget(self.path_label)
        header.addLayout(titles)
        header.addStretch()

        self.open_folder_button = QPushButton("Open folder")
        self.open_folder_button.setStyleSheet(_SECONDARY_BUTTON_STYLE)
        self.open_folder_button.clicked.connect(self.runner.open_folder)
        self.rescan_button = QPushButton("Rescan folder")
        self.rescan_button.setStyleSheet(_SECONDARY_BUTTON_STYLE)
        # "Rescan", not "Reload": scripts are loaded fresh on every run anyway, so
        # this only picks up files added or deleted since the dialog opened.
        self.rescan_button.clicked.connect(self.refresh)
        header.addWidget(self.open_folder_button)
        header.addWidget(self.rescan_button)
        layout.addLayout(header)

        self.table = self._build_table()
        layout.addWidget(self.table)

        self.detail_label = QLabel()
        self.detail_label.setTextFormat(Qt.RichText)
        self.detail_label.setWordWrap(True)
        self.detail_label.setStyleSheet(
            f"background-color: {_PANEL}; border: 1px solid {_BORDER};"
            f"border-radius: 6px; padding: 9px 11px; font-size: 12px; color: {_TEXT};"
        )
        layout.addWidget(self.detail_label)

        footer = QHBoxLayout()
        self.hint_label = QLabel()
        self.hint_label.setStyleSheet(f"font-size: 12px; color: {_TEXT_MUTED};")
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
        table.verticalHeader().setDefaultSectionSize(62)
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
        table.setColumnWidth(1, 210)
        table.setColumnWidth(2, 100)
        return table

    @staticmethod
    def _chip(text: str, colour: str, dot: bool = True) -> QLabel:
        """A pill label: coloured dot + text, on a tint of the same colour."""
        rgb = QColor(colour)
        tint = f"rgba({rgb.red()}, {rgb.green()}, {rgb.blue()}, 0.15)"
        marker = f'<span style="color:{colour};">&#9679;</span> ' if dot else ""
        chip = QLabel(f"{marker}{text}")
        chip.setStyleSheet(
            f"background-color: {tint}; color: {colour};"
            f"padding: 2px 9px; border-radius: 10px; font-size: 11px;"
        )
        return chip

    def _name_cell(self, script: DiscoveredScript) -> QWidget:
        """Two lines: the filename, and what it does (or why it will not load)."""
        widget = QWidget()
        widget.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 10, 10, 10)
        layout.setSpacing(3)

        name = QLabel(script.name)
        name.setStyleSheet(
            f"font-family: Menlo, monospace; font-size: 13px; background: transparent;"
            f"color: {_TEXT_STRONG if script.is_runnable else _TEXT_MUTED};"
        )
        detail = QLabel(script.description if script.is_runnable else script.error)
        detail.setStyleSheet(
            f"font-size: 12px; background: transparent;"
            f"color: {_TEXT if script.is_runnable else _ERROR};"
        )
        layout.addWidget(name)
        layout.addWidget(detail)
        widget.setToolTip(f"{script.path}\n{script.content_hash}")
        return widget

    def _type_cell(self, script: DiscoveredScript) -> QWidget:
        """The type chip, plus a chip per declared flag."""
        widget = QWidget()
        widget.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        label, colour = _script_type(script)
        layout.addWidget(self._chip(label, colour))
        if script.writes:
            layout.addWidget(self._chip("writes", _WARN, dot=False))
        if script.on_workflow_completed:
            layout.addWidget(self._chip("auto", _ACCENT, dot=False))
        layout.addStretch()
        notes = [f"Type: {label}"]
        if script.writes:
            notes.append("writes: saves the experiment when finished")
        if script.on_workflow_completed:
            notes.append("auto: also runs when a workflow finishes")
        widget.setToolTip("\n".join(notes))
        return widget

    # --- data ---

    def refresh(self) -> None:
        """Re-read the folder and rebuild the table."""
        self.scripts = self.runner.discover()
        failed = sum(1 for s in self.scripts if not s.is_runnable)
        runnable = len(self.scripts) - failed

        summary = f"{runnable} script{'s' if runnable != 1 else ''}"
        if failed:
            summary += f", {failed} failed to load"
        self.title_label.setText(summary)
        self.path_label.setText(str(self.runner.scripts_directory()))

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

        if self.scripts:
            self.table.selectRow(0)
        self._on_selection_changed()

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
            self.detail_label.setText(
                f'<span style="color:{_TEXT_MUTED};">No script selected.</span>'
            )
            self.run_button.setEnabled(False)
            self.hint_label.setText(reason)
            return

        parts = [
            f'<span style="color:{_TEXT_MUTED};">Source</span>&nbsp; {script.path.name}',
            f'<span style="color:{_TEXT_MUTED};">Hash</span>&nbsp; {script.content_hash}',
        ]
        detail = "<br>".join(parts)

        if not script.is_runnable:
            detail += f'<br><span style="color:{_ERROR};">● {script.error}</span>'
        elif script.uses_microscope:
            detail += (
                f'<br><span style="color:{_WARN};">● Needs the microscope — '
                f"not supported yet.</span>"
            )
        elif script.writes:
            detail += (
                f'<br><span style="color:{_WARN};">● Modifies the experiment and '
                f"saves it when finished.</span>"
            )
        if script.on_workflow_completed:
            detail += (
                f'<br><span style="color:{_ACCENT};">● Also runs automatically when '
                f"a workflow finishes.</span>"
            )

        self.detail_label.setText(detail)
        # uses_microscope is disabled rather than offered-then-refused: the runner
        # would reject it anyway (FIB-340), and a button that does nothing but
        # complain is worse than one that is visibly unavailable.
        self.run_button.setEnabled(
            script.is_runnable and host_ready and not script.uses_microscope
        )
        self.hint_label.setText(reason)

    def run_selected(self) -> None:
        script = self.selected_script()
        if script is None or not script.is_runnable:
            return
        result = self.runner.run(script)
        if result is not None:
            outcome = "ok" if result.ok else "failed"
            self.last_run[script.name] = f"{datetime.now().strftime('%H:%M')} {outcome}"
        self.refresh()
