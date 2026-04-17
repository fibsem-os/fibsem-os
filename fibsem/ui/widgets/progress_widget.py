"""Generic progress widget for fibsem UI.

``FibsemProgressWidget`` is a purely reactive ``QWidget`` that renders progress
updates from a :class:`ProgressUpdate` dataclass.  The calling code emits
updates at whatever cadence it likes; the widget just renders what it receives.

Progress modes
--------------
Mode is auto-detected from the fields set on :class:`ProgressUpdate`.

Numeric — ``current`` and ``total`` non-zero::

    ProgressUpdate.numeric(current=5, total=15, message="Acquiring tile")

Countdown — ``remaining_seconds`` non-zero::

    ProgressUpdate.countdown(remaining_seconds=30, total_seconds=60)

Combined — numeric items each with a time budget (e.g. spot burn)::

    ProgressUpdate.combined(current=2, total=5, remaining_seconds=8, total_seconds=10)

Indeterminate / message-only — no numeric or time values::

    ProgressUpdate.indeterminate(message="Moving stage...")

Done — completes the bar and shows "Done"::

    ProgressUpdate.done()

Call :meth:`FibsemProgressWidget.reset` to return to the initial hidden state.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt5.QtWidgets import QHBoxLayout, QProgressBar, QWidget

from fibsem.ui import stylesheets
from fibsem.ui.widgets.custom_widgets import _SpinnerLabel


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class ProgressUpdate:
    """Typed payload for :class:`FibsemProgressWidget`.

    Use the convenience constructors (:meth:`numeric`, :meth:`countdown`,
    :meth:`combined`, :meth:`indeterminate`, :meth:`done`) rather than
    constructing directly so intent is explicit at the call site.
    """

    current: int = 0
    total: int = 0
    remaining_seconds: float = 0.0
    total_seconds: float = 0.0
    message: str = ""
    finished: bool = False

    @classmethod
    def numeric(cls, current: int, total: int, message: str = "") -> "ProgressUpdate":
        """Progress by item count: ``current`` out of ``total``."""
        return cls(current=current, total=total, message=message)

    @classmethod
    def countdown(
        cls,
        remaining_seconds: float,
        total_seconds: float = 0.0,
        message: str = "",
    ) -> "ProgressUpdate":
        """Progress by time remaining."""
        return cls(
            remaining_seconds=remaining_seconds,
            total_seconds=total_seconds,
            message=message,
        )

    @classmethod
    def combined(
        cls,
        current: int,
        total: int,
        remaining_seconds: float,
        total_seconds: float = 0.0,
        message: str = "",
    ) -> "ProgressUpdate":
        """Item count *and* time remaining (e.g. N points, each with exposure time)."""
        return cls(
            current=current,
            total=total,
            remaining_seconds=remaining_seconds,
            total_seconds=total_seconds,
            message=message,
        )

    @classmethod
    def indeterminate(cls, message: str = "") -> "ProgressUpdate":
        """Busy / indeterminate state with an optional status message."""
        return cls(message=message)

    @classmethod
    def done(cls) -> "ProgressUpdate":
        """Signals completion — widget shows 100 % and "Done"."""
        return cls(finished=True)


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class FibsemProgressWidget(QWidget):
    """Generic progress bar widget.

    Purely reactive — call :meth:`update_progress` with a :class:`ProgressUpdate`
    whenever progress changes.  Call :meth:`reset` to return to hidden state.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self.reset()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._spinner = _SpinnerLabel(size=16, step_deg=30, interval_ms=40, color=stylesheets.WHITE_ICON_COLOR)
        layout.addWidget(self._spinner)

        self._bar = QProgressBar()
        self._bar.setTextVisible(True)
        self._bar.setStyleSheet(stylesheets.MILLING_PROGRESS_BAR_STYLESHEET)
        layout.addWidget(self._bar)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_progress(self, info: ProgressUpdate) -> None:
        """Update the widget from a :class:`ProgressUpdate`."""
        has_numeric = info.current > 0 or info.total > 0
        has_time = info.remaining_seconds > 0.0

        prefix = f"{info.message} — " if info.message else ""

        self.setVisible(True)

        if info.finished:
            self._spinner.stop()
            self._spinner.clear()  # blank icon, fixed size keeps space reserved
            self._bar.setMaximum(100)
            self._bar.setValue(100)
            self._bar.setFormat("Done")
            self._bar.setVisible(True)
            return

        if has_numeric and has_time:
            self._spinner.stop()
            self._spinner.clear()  # blank icon, fixed size keeps space reserved
            elapsed = (info.total_seconds - info.remaining_seconds) if info.total_seconds > 0 else 0.0
            if info.total_seconds > 0:
                self._bar.setMinimum(0)
                self._bar.setMaximum(int(info.total_seconds * 10))
                self._bar.setValue(max(0, int(elapsed * 10)))
            else:
                self._bar.setMinimum(0)
                self._bar.setMaximum(100)
                self._bar.setValue(100)
            self._bar.setFormat(
                f"{prefix}{info.current}/{info.total} \u00b7 {int(info.remaining_seconds)}s remaining"
            )

        elif has_numeric:
            self._spinner.stop()
            self._spinner.clear()  # blank icon, fixed size keeps space reserved
            self._bar.setMinimum(0)
            self._bar.setMaximum(int(info.total))
            self._bar.setValue(int(info.current))
            self._bar.setFormat(f"{prefix}{info.current} / {info.total}")

        elif has_time:
            self._spinner.stop()
            self._spinner.clear()  # blank icon, fixed size keeps space reserved
            if info.total_seconds > 0:
                self._bar.setMinimum(0)
                self._bar.setMaximum(int(info.total_seconds * 10))
                self._bar.setValue(max(0, int(info.remaining_seconds * 10)))
            else:
                self._bar.setMinimum(0)
                self._bar.setMaximum(100)
                self._bar.setValue(100)
            self._bar.setFormat(f"{prefix}{int(info.remaining_seconds)}s remaining")

        else:
            # Indeterminate: spinner + static full bar with message text
            self._bar.setMinimum(0)
            self._bar.setMaximum(100)
            self._bar.setValue(100)
            self._bar.setFormat(info.message or "")
            self._spinner.setVisible(True)
            self._spinner.start()

        self._bar.setVisible(True)

    def reset(self) -> None:
        """Return to initial hidden state."""
        self._spinner.stop()
        self._spinner.clear()
        self._bar.setMinimum(0)
        self._bar.setMaximum(100)
        self._bar.setValue(0)
        self._bar.setFormat("")
        self._bar.setVisible(False)
        self.setVisible(False)