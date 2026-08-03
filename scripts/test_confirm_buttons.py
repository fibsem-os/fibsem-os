#!/usr/bin/env python
"""Standalone demo of four 'are you sure' start buttons, side by side.

All four guard the same action; they differ in what the user has to do, and in
whether the wait happens before or after the action starts:

``HoldButton``      the bar fills only while the mouse (or space/enter) is held
                    down; letting go early cancels and the bar drains back.
``ClickButton``     one click starts the fill and it runs unattended; a second
                    click cancels it.
``TwoStageButton``  one click arms it ('Confirm?'), a second click within the
                    arming window fires. No waiting — the bar just shows how
                    long the window stays open.
``UndoButton``      fires immediately, then offers a few seconds to undo. The
                    only one where the action is not delayed at all.

None of them are wired into the app — this is a harness for comparing the
patterns by feel before picking one.

Usage:
    python scripts/test_confirm_buttons.py
"""

import sys

from PyQt5.QtCore import QElapsedTimer, QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# How long the bar takes to fill before a fill-style button fires.
DURATION_MS = 3000
# How long a two-stage button stays armed, and how long an undo stays offered.
WINDOW_MS = 4000
# ~60 fps, so the bar animates smoothly rather than stepping.
TICK_MS = 16
# How long 'Started' stays on the button before it resets to idle.
CONFIRM_MS = 900
# Cancelling drains the bar rather than snapping to zero, so a slip is visible.
DRAIN_MS = 250

# Bar modes. FILL counts up to the action; COUNTDOWN counts a window down to
# zero; DRAIN is the short cosmetic unwind after a cancel.
IDLE, FILL, DRAIN, COUNTDOWN = "idle", "fill", "drain", "countdown"


class ProgressButtonBase(QPushButton):
    """Shared machinery for a button that shows a bar before (or after) it fires.

    Subclasses decide *when* the bar moves — see ``_begin_fill``,
    ``_start_countdown`` and ``_cancel`` — and what it says, via
    ``_bar_text`` / ``_bar_color``. Everything here is timing and painting.

    ``triggered`` is emitted once per completed guard. The ordinary ``clicked``
    signal is only used internally by the click-driven subclasses.
    """

    triggered = pyqtSignal()

    def __init__(self, text: str, duration_ms: int = DURATION_MS, parent=None):
        super().__init__(text, parent)
        self._label = text
        self._duration_ms = duration_ms

        self._progress = 0.0        # 0..1, what the bar draws
        self._mode = IDLE
        self._confirmed = False     # showing 'Started', input ignored

        self._elapsed = QElapsedTimer()
        self._start_progress = 0.0  # progress when this phase began
        self._phase_ms = duration_ms  # how long the current phase runs for

        self._timer = QTimer(self)
        self._timer.setInterval(TICK_MS)
        self._timer.timeout.connect(self._on_tick)

        self.setMinimumHeight(44)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        font = self.font()
        font.setBold(True)
        self.setFont(font)

    # -- public API ------------------------------------------------------

    @property
    def progress(self) -> float:
        return self._progress

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def active(self) -> bool:
        """True while the guard is live — filling, or inside its window."""
        return self._mode in (FILL, COUNTDOWN)

    def reset(self) -> None:
        """Return to idle, whatever state the button is in."""
        self._timer.stop()
        self._mode = IDLE
        self._confirmed = False
        self._progress = 0.0
        self.update()

    # -- phases -----------------------------------------------------------

    def _enter(self, mode: str, phase_ms: int) -> None:
        self._mode = mode
        self._start_progress = self._progress
        self._phase_ms = phase_ms
        self._elapsed.start()
        self._timer.start()
        self.update()

    def _begin_fill(self) -> None:
        # Starting again mid-drain resumes from where the bar is, so a
        # fumbled attempt does not feel like it threw away the progress.
        self._enter(FILL, self._duration_ms)

    def _start_countdown(self, window_ms: int) -> None:
        self._progress = 1.0
        self._enter(COUNTDOWN, window_ms)

    def _cancel(self) -> None:
        if not self.active:
            return
        if self._progress <= 0.0:
            self.reset()
            return
        self._enter(DRAIN, DRAIN_MS)

    def _on_tick(self) -> None:
        fraction = self._elapsed.elapsed() / self._phase_ms

        if self._mode == FILL:
            self._progress = min(1.0, self._start_progress + fraction)
            if self._progress >= 1.0:
                self._complete()
                return
        else:
            self._progress = max(0.0, self._start_progress - fraction)
            if self._progress <= 0.0:
                self._timer.stop()
                if self._mode == COUNTDOWN:
                    self._on_expired()
                    return
                self._mode = IDLE

        self.update()

    def _complete(self) -> None:
        """Fire, flash 'Started', then reset."""
        self._timer.stop()
        self._mode = IDLE
        self._confirmed = True
        self._progress = 1.0
        self.update()
        self.triggered.emit()
        QTimer.singleShot(CONFIRM_MS, self.reset)

    def _on_expired(self) -> None:
        """The countdown window closed. Subclasses override to react."""
        self.reset()

    # -- painting ---------------------------------------------------------

    def _remaining_seconds(self) -> float:
        if self._mode == COUNTDOWN:
            return self._phase_ms * self._progress / 1000.0
        return self._duration_ms * (1.0 - self._progress) / 1000.0

    def _bar_text(self) -> str:
        """Label shown while the bar is live. Overridden per subclass."""
        return f"{self._remaining_seconds():.1f}s"

    def _bar_color(self) -> QColor:
        return QColor("#0084ff")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = 6.0

        path = QPainterPath()
        path.addRoundedRect(rect, radius, radius)

        if self._confirmed:
            background, fill = QColor("#2e7d32"), QColor("#43a047")
            text = "Started"
        else:
            background, fill = QColor("#3d4048"), self._bar_color()
            text = self._bar_text() if self.active else self._label

        painter.fillPath(path, background)

        # The fill is clipped to the rounded outline, so the leading edge stays
        # square while the ends still follow the button's corners.
        if self._progress > 0.0:
            painter.save()
            painter.setClipPath(path)
            painter.fillRect(
                QRectF(rect.left(), rect.top(), rect.width() * self._progress, rect.height()),
                fill,
            )
            painter.restore()

        painter.setPen(QColor("#43a047") if self._confirmed else QColor("#5a5d66"))
        painter.drawPath(path)

        painter.setPen(QColor("#f0f0f0"))
        painter.drawText(self.rect(), Qt.AlignCenter, text)


class HoldButton(ProgressButtonBase):
    """Fires only after being held down continuously for the full duration.

    The strongest of the four: a stray click or a Space keypress on a focused
    button cannot reach the action, because it needs sustained intent.
    """

    def __init__(self, text: str = "Hold to Start", duration_ms: int = DURATION_MS, parent=None):
        super().__init__(text, duration_ms, parent)

    def _bar_text(self) -> str:
        return f"Hold… {self._remaining_seconds():.1f}s"

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and not self._confirmed:
            self._begin_fill()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._cancel()
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        # Dragging off the button while held cancels, matching how a normal
        # click can be aborted by sliding away before release.
        if self.active and not self.rect().contains(event.pos()):
            self._cancel()
        super().mouseMoveEvent(event)

    def keyPressEvent(self, event):
        # Space/Enter hold works too, so the button is usable from the keyboard.
        if event.key() in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
            if not event.isAutoRepeat() and not self._confirmed:
                self._begin_fill()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
            if not event.isAutoRepeat():
                self._cancel()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event):
        self._cancel()
        super().focusOutEvent(event)


class ClickButton(ProgressButtonBase):
    """One click arms the countdown; it runs unattended until a second click.

    The same delay as :class:`HoldButton` without asking the user to keep a
    finger down — so the escape hatch is a cancelling click rather than a
    release, and the label says so while it counts down.
    """

    def __init__(self, text: str = "Click to Start", duration_ms: int = DURATION_MS, parent=None):
        super().__init__(text, duration_ms, parent)
        self.clicked.connect(self._on_clicked)

    def _bar_text(self) -> str:
        return f"Cancel · {self._remaining_seconds():.1f}s"

    def _on_clicked(self) -> None:
        if self._confirmed:
            return
        self._cancel() if self.active else self._begin_fill()


class TwoStageButton(ProgressButtonBase):
    """Click to arm, click again to fire — no waiting in between.

    The bar drains rather than fills: it is showing how long the armed window
    stays open, not how long until something happens. Cheapest guard of the
    four, since a decisive user can fire it as fast as a double click.
    """

    def __init__(self, text: str = "Click Twice to Start", window_ms: int = WINDOW_MS, parent=None):
        super().__init__(text, window_ms, parent)
        self._window_ms = window_ms
        self.clicked.connect(self._on_clicked)

    def _bar_text(self) -> str:
        return f"Confirm? · {self._remaining_seconds():.1f}s"

    def _bar_color(self) -> QColor:
        return QColor("#c77800")  # amber: armed, not yet committed

    def _on_clicked(self) -> None:
        if self._confirmed:
            return
        if self._mode == COUNTDOWN:
            self._complete()
        else:
            self._start_countdown(self._window_ms)


class UndoButton(ProgressButtonBase):
    """Fires on the first click, then offers a window to take it back.

    The only one of the four that does not make the user wait: the action
    starts immediately and the bar is the undo window draining. Right when the
    operation can be torn down cleanly — and wrong when the first moment is
    already irreversible, since by then there is nothing to undo.
    """

    undone = pyqtSignal()

    def __init__(self, text: str = "Start (undoable)", window_ms: int = WINDOW_MS, parent=None):
        super().__init__(text, window_ms, parent)
        self._window_ms = window_ms
        self.clicked.connect(self._on_clicked)

    def _bar_text(self) -> str:
        return f"Undo · {self._remaining_seconds():.1f}s"

    def _bar_color(self) -> QColor:
        return QColor("#2e7d32")  # green: already running

    def _on_clicked(self) -> None:
        if self._mode == COUNTDOWN:
            self.reset()
            self.undone.emit()
            return
        self._start_countdown(self._window_ms)
        self.triggered.emit()  # fires first, asks later

    def _on_expired(self) -> None:
        # Nothing to confirm — it has been running the whole time.
        self.reset()


class DemoWindow(QWidget):
    """Minimal window showing all four guards plus a log of what fired."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Confirm Button Demo")
        self.setStyleSheet("background-color: #262930; color: #f0f0f0;")

        self._count = 0

        title = QLabel("Four ways to guard the same action")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("", 11))

        self.hold_button = HoldButton("Hold to Start")
        self.click_button = ClickButton("Click to Start")
        self.two_stage_button = TwoStageButton("Click Twice to Start")
        self.undo_button = UndoButton("Start (undoable)")

        self.status = QLabel("Waiting…")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setStyleSheet("color: #9aa0a6;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(6)
        layout.addWidget(title)
        layout.addSpacing(10)

        for button, hint, source in (
            (self.hold_button, "hold 3s — release to cancel", "hold"),
            (self.click_button, "click once — click again to cancel", "click"),
            (self.two_stage_button, "click, then confirm within 4s", "two-stage"),
            (self.undo_button, "starts now — 4s to undo", "undo"),
        ):
            button.triggered.connect(lambda s=source: self._on_triggered(s))
            layout.addWidget(button)
            layout.addWidget(self._hint(hint))
            layout.addSpacing(10)

        self.undo_button.undone.connect(self._on_undone)

        layout.addWidget(self.status)
        self.resize(340, 440)

    @staticmethod
    def _hint(text: str) -> QLabel:
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: #7c8189; font-size: 11px;")
        return label

    def _on_triggered(self, source: str) -> None:
        self._count += 1
        self.status.setText(f"Started via {source}! (x{self._count})")

    def _on_undone(self) -> None:
        self.status.setText("Undone — rolled back")


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = DemoWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
