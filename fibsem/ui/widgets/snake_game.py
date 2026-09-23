"""Snake, hidden in the quad view's spare cell for something to do while milling runs.

An easter egg: click the "No Data" cell and type ``snake``. Space starts and pauses,
the arrow keys or WASD steer, and Q puts the cell back. The game pauses itself when it
is hidden (another tab), but not when it loses the keyboard: a toast takes focus on
every acquisition, and pausing on that stopped the game each time an image arrived.

The rules are :class:`SnakeGame`, which has no Qt in it; :class:`SnakeGameWidget`
only draws them and steps them on a timer.

A bug in here must never take the app down. PyQt5 aborts the process when an exception
escapes a handler Qt called, and the app installs no excepthook to stop it, so every
handler with logic in it is :func:`_contained`: the error is logged and the game
switches itself off, which puts "No Data" back in the cell.
"""

from __future__ import annotations

import functools
import logging
import random
from collections import deque
from enum import Enum
from typing import Deque, List, Optional, Tuple

from PyQt5.QtCore import QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter
from PyQt5.QtWidgets import QWidget

from fibsem.ui.tokens import (
    ACCENT_COLOR,
    BORDER_COLOR,
    CANVAS_BG,
    ERROR_COLOR,
    NEUTRAL_900,
    OK_COLOR,
    PRIMARY_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
)

Cell = Tuple[int, int]  # (column, row)

UP: Cell = (0, -1)
DOWN: Cell = (0, 1)
LEFT: Cell = (-1, 0)
RIGHT: Cell = (1, 0)

_KEY_DIRECTIONS = {
    Qt.Key_Up: UP,
    Qt.Key_W: UP,
    Qt.Key_Down: DOWN,
    Qt.Key_S: DOWN,
    Qt.Key_Left: LEFT,
    Qt.Key_A: LEFT,
    Qt.Key_Right: RIGHT,
    Qt.Key_D: RIGHT,
}

_MAX_QUEUED_TURNS = 2  # two quick presses inside one tick both count
_START_INTERVAL_MS = 130
_FASTEST_INTERVAL_MS = 60
_SPEEDUP_MS_PER_FOOD = 4
_HUD_HEIGHT = 18
_MARGIN = 6


def _contained(handler):
    """Run a Qt-invoked handler so an exception switches the game off instead of
    aborting the app. Once off, the handler does nothing."""

    @functools.wraps(handler)
    def wrapper(self, *args, **kwargs):
        if self.broken:
            return None
        try:
            return handler(self, *args, **kwargs)
        except Exception:
            logging.exception("Snake failed and has switched itself off")
            self.broken = True
            self._timer.stop()
            # after this handler returns, so the host swaps the cell outside a paint
            QTimer.singleShot(0, self.quit_requested.emit)
            return None

    return wrapper


class SnakeGame:
    """The rules: a snake on a ``cols`` x ``rows`` board that dies on the walls and itself."""

    def __init__(
        self, cols: int = 20, rows: int = 14, rng: Optional[random.Random] = None
    ) -> None:
        self.cols = cols
        self.rows = rows
        self.best = 0
        self._rng = rng if rng is not None else random.Random()
        self.reset()

    def reset(self) -> None:
        """A three-cell snake in the middle of the board, heading right."""
        col, row = self.cols // 2, self.rows // 2
        self.snake: Deque[Cell] = deque([(col, row), (col - 1, row), (col - 2, row)])
        self.direction = RIGHT
        self._turns: Deque[Cell] = deque()
        self.score = 0
        self.alive = True
        self.food = self._place_food()

    @property
    def head(self) -> Cell:
        return self.snake[0]

    @property
    def won(self) -> bool:
        """The snake fills the board, so there is nowhere left to put food."""
        return self.food is None

    def turn(self, direction: Cell) -> None:
        """Queue a turn for a coming tick. A reversal onto the snake's own neck is ignored."""
        last = self._turns[-1] if self._turns else self.direction
        if direction in (last, (-last[0], -last[1])):
            return
        if len(self._turns) < _MAX_QUEUED_TURNS:
            self._turns.append(direction)

    def step(self) -> bool:
        """Advance one tick; return whether the game goes on."""
        if not self.alive or self.won:
            return False
        if self._turns:
            self.direction = self._turns.popleft()
        head = (self.head[0] + self.direction[0], self.head[1] + self.direction[1])
        eating = head == self.food
        # the tail leaves its cell this tick unless the snake is growing into it
        body = list(self.snake) if eating else list(self.snake)[:-1]
        on_board = 0 <= head[0] < self.cols and 0 <= head[1] < self.rows
        if not on_board or head in body:
            self.alive = False
            return False
        self.snake.appendleft(head)
        if eating:
            self.score += 1
            self.best = max(self.best, self.score)
            self.food = self._place_food()
        else:
            self.snake.pop()
        return not self.won

    def interval_ms(self) -> int:
        """Milliseconds per tick: a little faster for every food eaten."""
        return max(
            _FASTEST_INTERVAL_MS, _START_INTERVAL_MS - _SPEEDUP_MS_PER_FOOD * self.score
        )

    def _place_food(self) -> Optional[Cell]:
        occupied = set(self.snake)
        free: List[Cell] = [
            (col, row)
            for col in range(self.cols)
            for row in range(self.rows)
            if (col, row) not in occupied
        ]
        return self._rng.choice(free) if free else None


class SnakeState(Enum):
    IDLE = "idle"
    PLAYING = "playing"
    PAUSED = "paused"
    OVER = "over"


class SnakeGameWidget(QWidget):
    """Draws a :class:`SnakeGame` and steps it on a timer while it has the keyboard."""

    # Q was pressed, or the game failed: the host should put back what the cell showed.
    quit_requested = pyqtSignal()

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        cols: int = 20,
        rows: int = 14,
        rng: Optional[random.Random] = None,
    ) -> None:
        super().__init__(parent)
        self.game = SnakeGame(cols, rows, rng=rng)
        self.state = SnakeState.IDLE
        self.broken = False  # an exception switched the game off for good
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self.setFocusPolicy(Qt.ClickFocus)

    # ── state ─────────────────────────────────────────────────────────────
    def start(self) -> None:
        """Start a new game, or resume a paused one."""
        if self.state is not SnakeState.PAUSED:
            self.game.reset()
        self.state = SnakeState.PLAYING
        self._timer.start(self.game.interval_ms())
        self.update()

    def pause(self) -> None:
        if self.state is not SnakeState.PLAYING:
            return
        self.state = SnakeState.PAUSED
        self._timer.stop()
        self.update()

    @_contained
    def _tick(self) -> None:
        if self.game.step():
            self._timer.setInterval(self.game.interval_ms())
        else:
            self.state = SnakeState.OVER
            self._timer.stop()
        self.update()

    # ── input ─────────────────────────────────────────────────────────────
    @_contained
    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        key = event.key()
        if key == Qt.Key_Space:
            if self.state is SnakeState.PLAYING:
                self.pause()
            else:
                self.start()
        elif key == Qt.Key_Q:
            self.pause()
            self.quit_requested.emit()
        elif key in _KEY_DIRECTIONS:
            if self.state is SnakeState.PLAYING:
                self.game.turn(_KEY_DIRECTIONS[key])
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def focusInEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().focusInEvent(event)
        self.update()  # the prompt reads differently with and without the keyboard

    def focusOutEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().focusOutEvent(event)
        self.update()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self.pause()
        super().hideEvent(event)

    # ── drawing ───────────────────────────────────────────────────────────
    def _board_rect(self) -> Tuple[QRectF, float]:
        """The board, centred below the score line, and the side of one square cell."""
        width = self.width() - 2 * _MARGIN
        height = self.height() - _HUD_HEIGHT - _MARGIN
        cell = max(1.0, min(width / self.game.cols, height / self.game.rows))
        board_w, board_h = cell * self.game.cols, cell * self.game.rows
        left = _MARGIN + (width - board_w) / 2
        top = _HUD_HEIGHT + (height - board_h) / 2
        return QRectF(left, top, board_w, board_h), cell

    @_contained
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(CANVAS_BG))

        board, cell = self._board_rect()
        painter.setPen(QColor(BORDER_COLOR))
        painter.setBrush(QColor(NEUTRAL_900))
        painter.drawRect(board)

        def square(pos: Cell, colour: str) -> None:
            inset = max(1.0, cell * 0.1)
            painter.setBrush(QColor(colour))
            painter.drawRoundedRect(
                QRectF(
                    board.left() + pos[0] * cell + inset,
                    board.top() + pos[1] * cell + inset,
                    cell - 2 * inset,
                    cell - 2 * inset,
                ),
                cell * 0.25,
                cell * 0.25,
            )

        painter.setPen(Qt.NoPen)
        game = self.game
        if game.food is not None:
            square(game.food, OK_COLOR)
        for pos in list(game.snake)[1:]:
            square(pos, PRIMARY_COLOR)
        square(game.head, ACCENT_COLOR if game.alive else ERROR_COLOR)

        font = QFont(self.font())
        font.setPixelSize(11)
        painter.setFont(font)
        painter.setPen(QColor(TEXT_MUTED_COLOR))
        painter.drawText(
            QRectF(6, 0, self.width() - 12, _HUD_HEIGHT),
            Qt.AlignVCenter | Qt.AlignLeft,
            f"Score {game.score}  ·  Best {game.best}",
        )

        lines = self._message()
        if lines:
            painter.fillRect(board, QColor(0, 0, 0, 150))
            self._draw_message(painter, board, *lines)
        painter.end()

    def _message(self) -> Optional[Tuple[str, str]]:
        """The title and hint over the board, or None while playing."""
        start = "Space to start" if self.hasFocus() else "Click here, then Space"
        if self.state is SnakeState.IDLE:
            return "Snake", f"{start}  ·  arrows or WASD  ·  Q to quit"
        if self.state is SnakeState.PAUSED:
            return "Paused", (
                "Space to resume  ·  Q to quit"
                if self.hasFocus()
                else "Click here, then Space"
            )
        if self.state is SnakeState.OVER:
            title = "Board cleared" if self.game.won else "Game over"
            return f"{title}  ·  {self.game.score}", f"{start}  ·  Q to quit"
        return None

    def _draw_message(
        self, painter: QPainter, board: QRectF, title: str, hint: str
    ) -> None:
        mid = board.top() + board.height() * 0.3  # above the snake's starting row
        font = QFont(self.font())
        font.setPixelSize(16)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(TEXT_STRONG_COLOR))
        painter.drawText(
            QRectF(board.left(), mid - 24, board.width(), 22), Qt.AlignCenter, title
        )
        font.setPixelSize(11)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QColor(TEXT_MUTED_COLOR))
        painter.drawText(
            QRectF(board.left(), mid + 2, board.width(), 18), Qt.AlignCenter, hint
        )
