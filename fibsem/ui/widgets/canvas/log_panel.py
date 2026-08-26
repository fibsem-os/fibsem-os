"""Live-tailing viewer for the process logfile, for the quad-view 4th cell.

Reads whatever ``fibsem.utils.configure_logging`` is currently writing to
(``<experiment>/logfile.log``) -- this is the real Python ``logging`` output
(every module, DEBUG and up), not the toast/notification bus. Tails the file
with a ``QFileSystemWatcher`` rather than attaching a logging.Handler, so it
is unaffected by ``logging.basicConfig(..., force=True)`` re-pointing the
root logger at a new experiment.
"""

from __future__ import annotations

import os
import re
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

from PyQt5.QtCore import QFileSystemWatcher, Qt, QTimer
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
)

from fibsem.ui.tokens import (
    ACCENT_COLOR,
    BORDER_COLOR,
    ERROR_COLOR,
    GRAY_CONSOLE_COLOR,
    NEUTRAL_650,
    PANEL_COLOR,
    ROW_ALT_COLOR,
    TEXT_COLOR,
    TEXT_MUTED_COLOR,
    WARN_COLOR,
)

# Matches fibsem.utils.configure_logging's format string exactly:
#   "%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s"
_LOG_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) — "
    r"(?P<name>.+?) — "
    r"(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL) — "
    r"(?P<func>.+?) — "
    r"(?P<msg>.*)$"
)

_SEVERITY = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
_LEVEL_COLOR = {
    "DEBUG": NEUTRAL_650,
    "INFO": ACCENT_COLOR,
    "WARNING": WARN_COLOR,
    "ERROR": ERROR_COLOR,
    "CRITICAL": ERROR_COLOR,
}
_FILTER_LEVELS = [("Debug", 10), ("Info", 20), ("Warning", 30), ("Error", 40)]

_MAX_ENTRIES = 2000  # bounded scrollback, so a long session can't grow this unbounded
_MAX_INITIAL_BYTES = (
    200_000  # how much of an existing (resumed) logfile to backfill on attach
)
_DEBOUNCE_MS = 200  # coalesce a burst of writes (e.g. a running workflow logging at
# DEBUG on every hardware call) into one read+render pass


@dataclass
class _LogEntry:
    timestamp: str
    level: str
    message: str

    @property
    def severity(self) -> int:
        return _SEVERITY.get(self.level, 0)


def _format_entry(entry: _LogEntry) -> str:
    time_only = (
        entry.timestamp[11:19] if len(entry.timestamp) >= 19 else entry.timestamp
    )
    return f"{time_only}  {entry.level:<8}{entry.message}"


class LogPanelWidget(QFrame):
    """Tails a logfile and renders it as a filterable, auto-scrolling list.

    Call :meth:`set_log_path` when the active experiment (and therefore the
    logfile ``fibsem.utils.configure_logging`` points at) changes; pass
    ``None`` to detach and clear.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background: {GRAY_CONSOLE_COLOR};")

        self._log_path: Optional[str] = None
        self._offset = 0
        self._entries: deque = deque(maxlen=_MAX_ENTRIES)
        self._last_item: Optional[QListWidgetItem] = None
        self._min_severity = 20  # "Info" default: hides DEBUG noise until asked for

        self._watcher = QFileSystemWatcher(self)
        self._watcher.fileChanged.connect(self._on_file_changed)
        # Debounced: QFileSystemWatcher does not guarantee one signal per write, and a
        # workflow logging at DEBUG on every hardware call can write far faster than the
        # GUI thread should be reacting to individual notifications.
        self._pending_timer = QTimer(self)
        self._pending_timer.setSingleShot(True)
        self._pending_timer.setInterval(_DEBOUNCE_MS)
        self._pending_timer.timeout.connect(self._read_new_lines)

        self._filter = QComboBox()
        self._filter.setStyleSheet(
            f"QComboBox {{ background: transparent; color: {TEXT_COLOR}; "
            f"border: 1px solid {BORDER_COLOR}; border-radius: 3px; padding: 1px 6px; "
            f"font-size: 11px; }} "
            f"QComboBox QAbstractItemView {{ background: {PANEL_COLOR}; color: {TEXT_COLOR}; "
            f"selection-background-color: {ROW_ALT_COLOR}; }}"
        )
        for label, severity in _FILTER_LEVELS:
            self._filter.addItem(label, severity)
        self._filter.setCurrentIndex(1)  # "Info"
        self._filter.currentIndexChanged.connect(self._on_filter_changed)

        top = QHBoxLayout()
        top.setContentsMargins(6, 3, 6, 3)
        top.addWidget(
            QLabel("Level:", styleSheet=f"color: {TEXT_MUTED_COLOR}; font-size: 11px;")
        )
        top.addWidget(self._filter)
        top.addStretch(1)

        self._list = QListWidget()
        self._list.setStyleSheet(
            f"QListWidget {{ background: {GRAY_CONSOLE_COLOR}; color: {TEXT_COLOR}; "
            f"border: none; }} QListWidget::item {{ padding: 0px 6px; }}"
        )
        font = QFont("Menlo")
        font.setStyleHint(QFont.Monospace)
        font.setPointSize(9)
        self._list.setFont(font)
        self._list.setSelectionMode(QListWidget.NoSelection)
        self._list.setVerticalScrollMode(QListWidget.ScrollPerPixel)

        self._empty_label = QLabel("No logfile open", alignment=Qt.AlignCenter)
        self._empty_label.setStyleSheet(f"color: {TEXT_MUTED_COLOR}; font-size: 11px;")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addLayout(top)
        lay.addWidget(self._list, stretch=1)
        lay.addWidget(self._empty_label)
        self._empty_label.setVisible(True)
        self._list.setVisible(False)

    # ── attach / detach ───────────────────────────────────────────────────
    def set_log_path(self, path: Optional[str]) -> None:
        """Point the tail at *path* (the current experiment's logfile), or
        ``None`` to detach and clear."""
        self._pending_timer.stop()
        if self._log_path and self._log_path in self._watcher.files():
            self._watcher.removePath(self._log_path)
        self._log_path = path
        self._offset = 0
        self._entries.clear()
        self._list.clear()
        self._last_item = None

        if not path:
            self._empty_label.setVisible(True)
            self._list.setVisible(False)
            return

        self._empty_label.setVisible(False)
        self._list.setVisible(True)

        if os.path.exists(path):
            self._backfill(path)
            self._watcher.addPath(path)

    def teardown(self) -> None:
        """Stop tailing. Safe to call repeatedly."""
        self._pending_timer.stop()
        if self._log_path and self._log_path in self._watcher.files():
            self._watcher.removePath(self._log_path)

    def showEvent(self, event) -> None:
        """A widget hosted in a hidden-by-default floating window can accumulate a
        long backlog while never laid out; re-assert the scroll position on every
        transition to visible rather than trusting scrollToBottom() calls made
        while unshown (see the quad-view screenshot bug this class started with)."""
        super().showEvent(event)
        self._list.scrollToBottom()

    # ── tailing ────────────────────────────────────────────────────────────
    def _backfill(self, path: str) -> None:
        """Load the tail end of an already-existing (e.g. resumed) logfile."""
        size = os.path.getsize(path)
        start = max(0, size - _MAX_INITIAL_BYTES)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(start)
                if start:
                    f.readline()  # drop a line we likely seeked into the middle of
                chunk = f.read()
                self._offset = f.tell()
        except OSError:
            return
        self._ingest_batch(chunk.splitlines())

    def _on_file_changed(self, path: str) -> None:
        if path != self._log_path:
            return
        # Coalesce a burst of near-simultaneous notifications into one read pass,
        # restarting the window on every new signal.
        self._pending_timer.start()
        # Some platforms drop a path from the watch list after it's modified
        # (e.g. editors that replace-on-write); re-arm defensively.
        if self._log_path and self._log_path not in self._watcher.files():
            if os.path.exists(self._log_path):
                self._watcher.addPath(self._log_path)

    def _read_new_lines(self) -> None:
        if not self._log_path or not os.path.exists(self._log_path):
            return
        try:
            with open(self._log_path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(self._offset)
                chunk = f.read()
                self._offset = f.tell()
        except OSError:
            return
        if not chunk:
            return
        self._ingest_batch(chunk.splitlines())

    def _ingest_batch(self, lines: List[str]) -> None:
        """Parse *lines* and render them as one batch: a single "was I scrolled to the
        bottom" check and a single scroll/prune pass at the end, rather than one per
        line. Doing this per-line is what made a burst of DEBUG-level writes (a
        workflow logging on every hardware call) hang the GUI thread -- each
        ``scrollToBottom()`` forces a full layout pass."""
        at_bottom = self._is_scrolled_to_bottom()
        added_any = False
        for line in lines:
            if self._ingest_line(line):
                added_any = True
        if not added_any:
            return
        while self._list.count() > _MAX_ENTRIES:
            self._list.takeItem(0)
        if at_bottom:
            self._list.scrollToBottom()

    def _ingest_line(self, line: str) -> bool:
        """Parse *line* and, if it passes the current filter, append (or fold into
        the previous entry) a row. Returns whether a visible row changed, so the
        caller knows whether a scroll/prune pass is warranted."""
        if not line.strip():
            return False
        m = _LOG_LINE_RE.match(line)
        if m:
            entry = _LogEntry(
                timestamp=m.group("ts"), level=m.group("level"), message=m.group("msg")
            )
            self._entries.append(entry)
            if entry.severity < self._min_severity:
                self._last_item = None
                return False
            item = QListWidgetItem(_format_entry(entry))
            item.setForeground(QColor(_LEVEL_COLOR.get(entry.level, TEXT_COLOR)))
            self._list.addItem(item)
            self._last_item = item
            return True
        elif self._entries:
            # Continuation line (e.g. a traceback from logging.exception) --
            # fold into the previous entry rather than showing as a bare row.
            self._entries[-1].message += "\n" + line
            if self._last_item is not None:
                self._last_item.setText(_format_entry(self._entries[-1]))
                return True
        return False

    def _is_scrolled_to_bottom(self) -> bool:
        bar = self._list.verticalScrollBar()
        return bar.value() >= bar.maximum() - 2

    def _on_filter_changed(self) -> None:
        self._min_severity = self._filter.currentData()
        self._rebuild_list()

    def _rebuild_list(self) -> None:
        self._list.clear()
        self._last_item = None
        for entry in self._entries:
            if entry.severity >= self._min_severity:
                item = QListWidgetItem(_format_entry(entry))
                item.setForeground(QColor(_LEVEL_COLOR.get(entry.level, TEXT_COLOR)))
                self._list.addItem(item)
                self._last_item = item
        self._list.scrollToBottom()
