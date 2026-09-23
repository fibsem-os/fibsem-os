"""Snake in the quad view's spare cell: the rules, and the easter egg that opens it.

The easter egg is behind ``features.easter_eggs_enabled``, off by default. Every test
here gets its own preferences file, so none of them reads this machine's.

QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_snake_game.py -q
"""

import random

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import Qt  # noqa: E402
from PyQt5.QtTest import QTest  # noqa: E402

from fibsem.ui.widgets.snake_game import (  # noqa: E402
    LEFT,
    RIGHT,
    UP,
    SnakeGame,
    SnakeGameWidget,
    SnakeState,
)


@pytest.fixture(autouse=True)
def _preferences(tmp_path, monkeypatch):
    """A private, absent preferences file: every flag at its default, so easter eggs off."""
    import fibsem.config as cfg

    monkeypatch.setattr(
        cfg, "USER_PREFERENCES_PATH", str(tmp_path / "user-preferences.yaml")
    )


def _set_easter_eggs(enabled):
    import fibsem.config as cfg

    prefs = cfg.load_user_preferences()
    prefs.features.easter_eggs_enabled = enabled
    cfg.save_user_preferences(prefs)


@pytest.fixture
def easter_eggs():
    _set_easter_eggs(True)


def _active_quad(qapp):
    """A quad view in the active window: focus only moves inside the active window."""
    from fibsem.ui.widgets.canvas.quad_view import QuadViewWidget

    quad = QuadViewWidget()
    quad.show()
    qapp.setActiveWindow(quad)
    assert QTest.qWaitForWindowActive(quad)
    return quad


def _game(cols=10, rows=10):
    return SnakeGame(cols, rows, rng=random.Random(0))


def _place(game, snake, direction, food):
    """Put the board in a known position: *snake* is head first."""
    game.snake.clear()
    game.snake.extend(snake)
    game.direction = direction
    game.food = food


# ── rules ─────────────────────────────────────────────────────────────────


def test_a_step_moves_the_head_and_keeps_the_length():
    game = _game()
    _place(game, [(5, 5), (4, 5), (3, 5)], RIGHT, food=(0, 0))
    assert game.step()
    assert list(game.snake) == [(6, 5), (5, 5), (4, 5)]


def test_eating_grows_scores_and_moves_the_food_off_the_snake():
    game = _game()
    _place(game, [(5, 5), (4, 5), (3, 5)], RIGHT, food=(6, 5))
    assert game.step()
    assert list(game.snake) == [(6, 5), (5, 5), (4, 5), (3, 5)]
    assert (game.score, game.best) == (1, 1)
    assert game.food is not None and game.food not in game.snake


def test_the_wall_ends_the_game():
    game = _game()
    _place(game, [(9, 5), (8, 5), (7, 5)], RIGHT, food=(0, 0))
    assert not game.step()
    assert not game.alive
    assert not game.step()  # stays over


def test_running_into_itself_ends_the_game():
    game = _game()
    # a hook: turning up from (5, 5) lands on (5, 4), which is body
    _place(game, [(5, 5), (4, 5), (4, 4), (5, 4), (6, 4)], RIGHT, food=(0, 0))
    game.turn(UP)
    assert not game.step()
    assert not game.alive


def test_the_cell_the_tail_is_leaving_is_free():
    game = _game()
    # a 2x2 loop: the head moves onto the cell the tail vacates this tick
    _place(game, [(5, 5), (5, 4), (4, 4), (4, 5)], LEFT, food=(0, 0))
    assert game.step()
    assert game.head == (4, 5)


def test_reversing_onto_its_neck_is_ignored():
    game = _game()
    _place(game, [(5, 5), (4, 5), (3, 5)], RIGHT, food=(0, 0))
    game.turn(LEFT)
    assert game.step()
    assert game.head == (6, 5)


def test_two_quick_turns_in_one_tick_both_count():
    game = _game()
    _place(game, [(5, 5), (4, 5), (3, 5)], RIGHT, food=(0, 0))
    game.turn(UP)
    game.turn(LEFT)  # a U-turn in two presses, faster than one tick
    game.step()
    game.step()
    assert list(game.snake)[:2] == [(4, 4), (5, 4)]


def test_filling_the_board_wins():
    game = SnakeGame(cols=4, rows=1, rng=random.Random(0))
    assert game.food == (3, 0)  # the only free cell
    assert not game.step()
    assert game.alive and game.won
    assert game.score == 1


def test_it_speeds_up_to_a_floor():
    game = _game()
    start = game.interval_ms()
    game.score = 5
    assert game.interval_ms() < start
    game.score = 1000
    assert game.interval_ms() == 60


def test_reset_keeps_the_best_score():
    game = _game()
    _place(game, [(5, 5), (4, 5), (3, 5)], RIGHT, food=(6, 5))
    game.step()
    game.reset()
    assert (game.score, game.best, len(game.snake)) == (0, 1, 3)


# ── widget ────────────────────────────────────────────────────────────────


@pytest.fixture
def widget(qapp, destroy_widgets_after_test):
    w = SnakeGameWidget(rng=random.Random(0))
    w.resize(400, 300)
    w.show()
    return w


def test_space_starts_pauses_and_resumes(widget):
    QTest.keyClick(widget, Qt.Key_Space)
    assert widget.state is SnakeState.PLAYING
    assert widget._timer.isActive()
    QTest.keyClick(widget, Qt.Key_Space)
    assert widget.state is SnakeState.PAUSED
    assert not widget._timer.isActive()
    head = widget.game.head
    QTest.keyClick(widget, Qt.Key_Space)
    assert widget.state is SnakeState.PLAYING
    assert widget.game.head == head  # resumed, not restarted


def test_arrow_keys_and_wasd_steer(widget):
    QTest.keyClick(widget, Qt.Key_Space)
    col, row = widget.game.head
    QTest.keyClick(widget, Qt.Key_Down)
    widget._tick()
    assert widget.game.head == (col, row + 1)
    QTest.keyClick(widget, Qt.Key_D)
    widget._tick()
    assert widget.game.head == (col + 1, row + 1)
    assert widget.game.direction == RIGHT


def test_hiding_pauses(widget):
    QTest.keyClick(widget, Qt.Key_Space)
    widget.hide()
    assert widget.state is SnakeState.PAUSED
    assert not widget._timer.isActive()


def test_losing_the_keyboard_keeps_playing(qapp, destroy_widgets_after_test):
    # a toast takes focus on every acquisition; pausing on it stopped the game per image
    from PyQt5.QtWidgets import QLineEdit, QVBoxLayout, QWidget

    host = QWidget()
    game = SnakeGameWidget(rng=random.Random(0))
    other = QLineEdit()
    layout = QVBoxLayout(host)
    layout.addWidget(game)
    layout.addWidget(other)
    host.show()
    qapp.setActiveWindow(host)  # focus events only reach widgets in the active window
    assert QTest.qWaitForWindowActive(host)
    game.setFocus()
    assert game.hasFocus()
    QTest.keyClick(game, Qt.Key_Space)
    other.setFocus()
    qapp.processEvents()
    # focus really moved, so the check below means something
    assert qapp.focusWidget() is other
    assert game.state is SnakeState.PLAYING
    assert game._timer.isActive()


def test_dying_stops_the_timer_and_space_plays_again(widget):
    QTest.keyClick(widget, Qt.Key_Space)
    QTest.keyClick(widget, Qt.Key_Up)
    for _ in range(widget.game.rows):
        widget._tick()
    assert widget.state is SnakeState.OVER
    assert not widget._timer.isActive()
    QTest.keyClick(widget, Qt.Key_Space)
    assert widget.state is SnakeState.PLAYING
    assert widget.game.alive and len(widget.game.snake) == 3


def test_every_state_paints(widget):
    for state in SnakeState:
        widget.state = state
        assert not widget.grab().isNull()


# ── the easter egg ────────────────────────────────────────────────────────


def test_typing_snake_into_the_empty_cell_opens_the_game_and_q_closes_it(
    qapp, destroy_widgets_after_test, easter_eggs
):
    from fibsem.ui.widgets.canvas.quad_view import QuadViewWidget

    quad = QuadViewWidget()
    quad.show()
    assert quad.snake_game is None  # nothing built until asked for

    QTest.keyClicks(quad.placeholder, "no snakes here")  # "snake" is inside it
    assert quad.snake_game is not None
    assert quad._spare.currentWidget() is quad.snake_game

    QTest.keyClick(quad.snake_game, Qt.Key_Space)
    assert quad.snake_game.state is SnakeState.PLAYING
    QTest.keyClick(quad.snake_game, Qt.Key_Q)
    assert quad._spare.currentWidget() is quad.placeholder
    assert quad.snake_game.state is SnakeState.PAUSED


def test_other_typing_leaves_the_empty_cell_alone(
    qapp, destroy_widgets_after_test, easter_eggs
):
    from fibsem.ui.widgets.canvas.quad_view import QuadViewWidget

    quad = QuadViewWidget()
    quad.show()
    QTest.keyClicks(quad.placeholder, "snak e")
    assert quad.snake_game is None
    assert quad._spare.currentWidget() is quad.placeholder


# ── the flag ──────────────────────────────────────────────────────────────


def test_off_by_default_the_empty_cell_never_takes_the_keyboard(
    qapp, destroy_widgets_after_test
):
    quad = _active_quad(qapp)
    QTest.mouseClick(quad.placeholder, Qt.LeftButton)
    assert not quad.placeholder.hasFocus()
    QTest.keyClicks(quad.placeholder, "snake")  # even sent straight to it
    assert quad.snake_game is None
    assert quad._spare.currentWidget() is quad.placeholder


def test_turning_it_on_needs_no_restart(qapp, destroy_widgets_after_test):
    quad = _active_quad(qapp)  # built while the flag is off
    _set_easter_eggs(True)
    QTest.mouseClick(quad.placeholder, Qt.LeftButton)
    assert quad.placeholder.hasFocus()
    QTest.keyClicks(quad.placeholder, "snake")
    assert quad._spare.currentWidget() is quad.snake_game
    assert quad.snake_game.hasFocus()


def test_an_unreadable_preference_counts_as_off(
    qapp, destroy_widgets_after_test, monkeypatch, easter_eggs
):
    import fibsem.config as cfg

    def boom():
        raise RuntimeError("injected")

    monkeypatch.setattr(cfg, "load_user_preferences", boom)
    quad = _active_quad(qapp)
    QTest.mouseClick(quad.placeholder, Qt.LeftButton)
    assert not quad.placeholder.hasFocus()
    QTest.keyClicks(quad.placeholder, "snake")
    assert quad.snake_game is None


# ── a bug in the game must not take the app down ─────────────────────────
# PyQt5 aborts the process when an exception escapes a handler Qt called, and the app
# installs no excepthook, so a game bug mid-milling would end the run with it.


def test_a_bug_in_the_game_switches_it_off_instead_of_raising(
    qapp, destroy_widgets_after_test, easter_eggs
):
    from fibsem.ui.widgets.canvas.quad_view import QuadViewWidget

    quad = QuadViewWidget()
    quad.show()
    QTest.keyClicks(quad.placeholder, "snake")
    widget = quad.snake_game
    QTest.keyClick(widget, Qt.Key_Space)

    def boom():
        raise RuntimeError("injected")

    widget.game.step = boom
    widget._tick()  # raises without the guard
    assert widget.broken
    assert not widget._timer.isActive()
    qapp.processEvents()  # the swap back is deferred out of the failing handler
    assert quad._spare.currentWidget() is quad.placeholder

    QTest.keyClicks(quad.placeholder, "snake")  # and it stays off
    assert quad._spare.currentWidget() is quad.placeholder


def test_a_game_that_cannot_start_leaves_the_empty_cell(
    qapp, destroy_widgets_after_test, monkeypatch, easter_eggs
):
    from fibsem.ui.widgets import snake_game
    from fibsem.ui.widgets.canvas.quad_view import QuadViewWidget

    def boom(*args, **kwargs):
        raise RuntimeError("injected")

    monkeypatch.setattr(snake_game, "SnakeGameWidget", boom)
    quad = QuadViewWidget()
    quad.show()
    QTest.keyClicks(quad.placeholder, "snake")  # raises without the guard
    assert quad.snake_game is None
    assert quad._spare.currentWidget() is quad.placeholder
