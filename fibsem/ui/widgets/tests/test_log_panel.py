"""Standalone demo + scripted checks: LogPanelWidget tailing a real logfile.

Run:
    PYTHONPATH=<worktree> python fibsem/ui/widgets/tests/test_log_panel.py

Writes to a temp file using the exact format fibsem.utils.configure_logging
produces, backfills a LogPanelWidget from it, then shows a window with an
"Append line" button (writes + simulates the file-change tick deterministically,
rather than relying on inotify timing) so you can watch the level filter and
auto-scroll live.
"""
import logging
import os
import sys
import tempfile
import time

from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.ui.widgets.canvas.log_panel import LogPanelWidget

_FMT = "%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s"


def _write_seed_lines(path: str) -> None:
    logger = logging.getLogger("test_log_panel.seed")
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FMT))
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    logger.debug("polling stage position")
    logger.info("connected to microscope: Demo")
    logger.warning("autofocus below threshold, retrying")
    logger.error("failed to acquire image: timeout")
    try:
        raise ValueError("bad pixel size")
    except ValueError:
        logger.exception("alignment failed")
    logger.info("trench milling started")
    handler.close()


def _run_checks(path: str) -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    w = LogPanelWidget()
    w.set_log_path(path)

    assert len(w._entries) == 6, f"expected 6 parsed entries, got {len(w._entries)}"
    assert w._entries[0].level == "DEBUG"
    assert w._entries[3].level == "ERROR"
    assert "\n" in w._entries[4].message, "traceback lines should fold into the exception entry"
    # default filter is Info: the DEBUG row should be backfilled but not rendered
    assert w._list.count() == 5, f"expected 5 rows at Info level, got {w._list.count()}"

    w._filter.setCurrentIndex(0)  # Debug: show everything
    assert w._list.count() == 6

    w._filter.setCurrentIndex(3)  # Error: only ERROR/CRITICAL-severity rows
    assert w._list.count() == 2, f"expected 2 error-or-worse rows, got {w._list.count()}"

    w.set_log_path(None)
    assert w._list.count() == 0
    assert not w._empty_label.isHidden()

    _check_burst_does_not_hang()
    _check_scroll_survives_hidden_start(path)

    print("all checks passed")
    w.deleteLater()


def _check_scroll_survives_hidden_start(path: str) -> None:
    """Regression guard: the log widget now lives in a floating window hidden by
    default (a normal user shouldn't see raw DEBUG/INFO output unasked), so it can
    accumulate a long backlog before ever being laid out. Confirms the first
    show() still lands scrolled to the true latest entry, not wherever a
    scrollToBottom() call made while unshown happened to leave it."""
    w = LogPanelWidget()
    w.hide()
    w.set_log_path(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write("2026-08-25 12:00:00,000 — root — INFO — poll:1 — freshest line while hidden\n")
    w._read_new_lines()

    w.show()
    last = w._list.item(w._list.count() - 1)
    assert last is not None and "freshest line while hidden" in last.text(), (
        "scroll position did not land on the true last entry after the first show()"
    )
    w.deleteLater()


def _check_burst_does_not_hang() -> None:
    """Regression guard: a running workflow can log DEBUG on every hardware call,
    so a burst of thousands of lines can land between watcher ticks. Rendering
    them used to call scrollToBottom() (a full layout pass) per line -- this
    checks the batched path stays fast even in the worst case (Debug filter,
    nothing filtered out)."""
    tmp_dir = tempfile.mkdtemp(prefix="log_panel_burst_")
    path = os.path.join(tmp_dir, "logfile.log")
    open(path, "w").close()

    w = LogPanelWidget()
    w.set_log_path(path)
    w._filter.setCurrentIndex(0)  # Debug: worst case

    n = 5000
    with open(path, "a", encoding="utf-8") as f:
        for i in range(n):
            f.write(f"2026-08-25 11:00:00,000 — root — DEBUG — poll:1 — hardware poll {i}\n")

    start = time.perf_counter()
    w._read_new_lines()
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"burst of {n} lines took {elapsed:.2f}s -- likely back to per-line scrollToBottom()"
    last = w._list.item(w._list.count() - 1)
    assert last is not None and f"hardware poll {n - 1}" in last.text()
    w.deleteLater()


def main() -> None:
    tmp_dir = tempfile.mkdtemp(prefix="log_panel_demo_")
    path = os.path.join(tmp_dir, "logfile.log")
    _write_seed_lines(path)
    _run_checks(path)

    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        return  # headless CI/agent run: scripted checks above are the point

    app = QApplication(sys.argv)
    win = QWidget()
    win.resize(420, 380)
    win.setWindowTitle("LogPanelWidget demo")

    panel = LogPanelWidget()
    panel.set_log_path(path)

    counter = {"n": 0}

    def append_line():
        counter["n"] += 1
        levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        level = levels[counter["n"] % 4]
        line = (
            f'2026-08-25 12:00:{counter["n"]:02d},000 — demo — {level} — append_line:1 — '
            f'live line {counter["n"]}\n'
        )
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
        panel._on_file_changed(path)  # deterministic, doesn't depend on OS watch timing

    btn = QPushButton("Append line")
    btn.clicked.connect(append_line)

    info = QLabel("Backfilled from a seeded logfile. Use the level filter, or append new lines.")
    info.setWordWrap(True)

    bar = QHBoxLayout()
    bar.addWidget(btn)
    bar.addStretch(1)

    lay = QVBoxLayout(win)
    lay.addWidget(info)
    lay.addLayout(bar)
    lay.addWidget(panel, stretch=1)

    win.show()
    app.exec_()


if __name__ == "__main__":
    main()
