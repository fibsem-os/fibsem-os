"""Generic progress widget for fibsem UI.

``FibsemProgressWidget`` is a purely reactive ``QWidget`` that renders progress
updates from a plain dict payload.  The calling code emits updates at whatever
cadence it likes; the widget just renders what it receives.

Dict schema
-----------
All keys are optional.  Mode is auto-detected from the keys present.

Numeric mode — ``current`` + ``total`` present::

    {"current": int, "total": int, "message": str, "finished": bool}

Countdown mode — ``remaining_seconds`` present (with optional ``total_seconds``)::

    {"remaining_seconds": float, "total_seconds": float, "message": str, "finished": bool}

Combined mode — numeric + time (e.g. spot burn: N points, each with exposure time)::

    {
        "current": int, "total": int,
        "remaining_seconds": float, "total_seconds": float,
        "message": str, "finished": bool,
    }

Message-only / indeterminate — no numeric or time keys::

    {"message": str, "finished": bool}

The ``finished`` key, if ``True``, completes the bar and shows a "Done" label.
Call ``reset()`` to return the widget to its initial hidden state.
"""

from PyQt5.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

from fibsem.ui import stylesheets


class FibsemProgressWidget(QWidget):
    """Generic progress bar widget.

    Purely reactive — call :meth:`update_progress` with a dict payload
    whenever progress changes.  See module docstring for the dict schema.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self.reset()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._label = QLabel()
        self._label.setStyleSheet("color: #d6d6d6; font-size: 11px;")
        layout.addWidget(self._label)

        self._bar = QProgressBar()
        self._bar.setTextVisible(True)
        layout.addWidget(self._bar)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_progress(self, info: dict) -> None:
        """Update the widget from a progress dict.

        Parameters
        ----------
        info:
            Dict with any combination of the keys described in the module
            docstring.
        """
        finished = info.get("finished", False)
        message = info.get("message", "")

        current = info.get("current")
        total = info.get("total")
        remaining = info.get("remaining_seconds")
        total_secs = info.get("total_seconds")

        has_numeric = current is not None and total is not None
        has_time = remaining is not None

        # Show widget on first update
        self.setVisible(True)

        # Update optional label
        if message:
            self._label.setText(message)
            self._label.setVisible(True)
        else:
            self._label.setVisible(False)

        if finished:
            self._bar.setStyleSheet(stylesheets.MILLING_PROGRESS_BAR_STYLESHEET)
            self._bar.setMaximum(100)
            self._bar.setValue(100)
            self._bar.setFormat("Done")
            self._bar.setVisible(True)
            return

        if has_numeric and has_time:
            # Combined: show item count in format, time drives bar fill
            self._bar.setStyleSheet(stylesheets.MILLING_PROGRESS_BAR_STYLESHEET)
            elapsed = (total_secs - remaining) if total_secs is not None else 0.0
            if total_secs and total_secs > 0:
                self._bar.setMinimum(0)
                self._bar.setMaximum(int(total_secs * 10))
                self._bar.setValue(max(0, int(elapsed * 10)))
            else:
                self._bar.setMinimum(0)
                self._bar.setMaximum(0)  # indeterminate fallback
            self._bar.setFormat(f"{current}/{total} \u00b7 {int(remaining)}s remaining")

        elif has_numeric:
            # Numeric only: bar fills as current/total
            self._bar.setStyleSheet(stylesheets.MILLING_PROGRESS_BAR_STYLESHEET)
            self._bar.setMinimum(0)
            self._bar.setMaximum(int(total))
            self._bar.setValue(int(current))
            self._bar.setFormat(f"{current} / {total}")

        elif has_time:
            # Countdown: bar drains from full to empty
            self._bar.setStyleSheet(stylesheets.INDETERMINATE_PROGRESS_BAR_STYLESHEET)
            if total_secs and total_secs > 0:
                self._bar.setMinimum(0)
                self._bar.setMaximum(int(total_secs * 10))
                self._bar.setValue(max(0, int(remaining * 10)))
            else:
                # No total known — just show remaining as indeterminate
                self._bar.setMinimum(0)
                self._bar.setMaximum(0)
            self._bar.setFormat(f"{int(remaining)}s remaining")

        else:
            # Message-only / indeterminate busy bar
            self._bar.setStyleSheet(stylesheets.INDETERMINATE_PROGRESS_BAR_STYLESHEET)
            self._bar.setMinimum(0)
            self._bar.setMaximum(0)  # animated busy indicator
            self._bar.setFormat(message or "")

        self._bar.setVisible(True)

    def reset(self) -> None:
        """Return to initial hidden state."""
        self._label.setText("")
        self._label.setVisible(False)
        self._bar.setMinimum(0)
        self._bar.setMaximum(100)
        self._bar.setValue(0)
        self._bar.setFormat("")
        self._bar.setVisible(False)
        self.setVisible(False)
