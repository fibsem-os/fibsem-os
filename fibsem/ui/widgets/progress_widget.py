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

Failed — completes the bar, shows an error message, and turns the bar red::

    ProgressUpdate.failed("Spot burn failed")

Call :meth:`FibsemProgressWidget.reset` to return to the initial hidden state, or
:meth:`FibsemProgressWidget.reset_if_finished` from delayed hide-after-done timers
(it leaves the widget alone if a newer operation is already rendering progress).
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
    # NOTE: not `failed` -- the `failed()` classmethod below would shadow the field's
    # class attribute, and @dataclass would take the method object as the default.
    is_failed: bool = False

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

    @classmethod
    def failed(cls, message: str = "Failed") -> "ProgressUpdate":
        """Signals failure — completes the bar but shows ``message`` instead of "Done"."""
        return cls(finished=True, is_failed=True, message=message)


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

        self._spinner = _SpinnerLabel(
            size=16, step_deg=30, interval_ms=40, color=stylesheets.WHITE_ICON_COLOR
        )
        layout.addWidget(self._spinner)

        self._bar = QProgressBar()
        self._bar.setTextVisible(True)
        self._failed_style = False
        self._bar.setStyleSheet(stylesheets.PROGRESS_BAR_STYLESHEET)
        layout.addWidget(self._bar)

    def _set_failed_style(self, failed: bool) -> None:
        """Swap the chunk colour between the normal and error styles.

        Guarded on the current state: setStyleSheet forces a restyle, and
        update_progress runs on every frame of a countdown.
        """
        if failed == self._failed_style:
            return
        self._failed_style = failed
        self._bar.setStyleSheet(
            stylesheets.FAILED_PROGRESS_BAR_STYLESHEET
            if failed
            else stylesheets.PROGRESS_BAR_STYLESHEET
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_progress(self, info: ProgressUpdate) -> None:
        """Update the widget from a :class:`ProgressUpdate`."""
        has_numeric = info.current > 0 or info.total > 0
        has_time = info.remaining_seconds > 0.0

        prefix = f"{info.message} — " if info.message else ""

        self._finished = info.finished
        self._set_failed_style(info.is_failed)
        self.setVisible(True)

        if info.finished:
            self._spinner.stop()
            self._spinner.clear()  # blank icon, fixed size keeps space reserved
            self._bar.setMaximum(100)
            self._bar.setValue(100)
            self._bar.setFormat(info.message or "Done")
            self._bar.setVisible(True)
            return

        if has_numeric and has_time:
            self._spinner.stop()
            self._spinner.clear()  # blank icon, fixed size keeps space reserved
            elapsed = (
                (info.total_seconds - info.remaining_seconds)
                if info.total_seconds > 0
                else 0.0
            )
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

    def reset_if_finished(self) -> None:
        """Reset only if the most recent update was a finished/failed one.

        For delayed hide-after-done timers on a shared widget: if another
        operation has started rendering progress in the meantime, leave it alone.
        """
        if self._finished:
            self.reset()

    def reset(self) -> None:
        """Return to initial hidden state."""
        self._finished = False
        self._set_failed_style(False)
        self._spinner.stop()
        self._spinner.clear()
        self._bar.setMinimum(0)
        self._bar.setMaximum(100)
        self._bar.setValue(0)
        self._bar.setFormat("")
        self._bar.setVisible(False)
        self.setVisible(False)
