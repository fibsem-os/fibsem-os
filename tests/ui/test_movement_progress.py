"""Where movement progress goes, now that it is no longer a stream of toasts.

One click-to-move emits four progress messages inside ~45 ms. As toasts they stack
into a wall of popups saying nothing the moving stage does not already show; in the
instructions label they read as one status for the duration of the move.

They are invisible today because ``display.toasts_enabled`` defaults to False. This
is what has to be true before that default can change.
"""

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from fibsem.ui.FibsemMovementWidget import (
    ACQUIRING_IMAGES,
    INSTRUCTIONS_TEXT,
    FibsemMovementWidget,
)

# The real sequence one click-to-move produces, in order, taken from the emit sites.
CLICK_TO_MOVE = [
    {"msg": "Moving the stage…"},
    {"msg": ACQUIRING_IMAGES},
    {"finished": True},
]


@pytest.fixture
def widget(qapp, monkeypatch):
    # The widget connects to canvases and a microscope it has no business reaching in
    # a test; only the progress handler and its label are under test here.
    monkeypatch.setattr(FibsemMovementWidget, "setup_connections", lambda self: None)
    w = FibsemMovementWidget.__new__(FibsemMovementWidget)
    from PyQt5 import QtWidgets

    QtWidgets.QWidget.__init__(w)
    w.label_movement_instructions = QtWidgets.QLabel(INSTRUCTIONS_TEXT)
    w._update_position_readout = lambda: None
    yield w
    w.deleteLater()


def test_progress_reaches_the_label(widget):
    widget.handle_movement_progress_update({"msg": "Moving stage..."})
    assert widget.label_movement_instructions.text() == "Moving stage..."


def test_progress_does_not_toast(widget, monkeypatch):
    """The whole point: four popups per click is worse than none."""
    from fibsem.ui import notification_service

    shown = []
    monkeypatch.setattr(
        notification_service,
        "show_toast",
        lambda *a, **k: shown.append(a),
    )
    for update in CLICK_TO_MOVE:
        widget.handle_movement_progress_update(update)
    assert shown == []


def test_the_instructions_come_back_when_the_move_finishes(widget):
    """Otherwise the last progress line sits there as though the stage were still
    moving, and the instructions are gone for the rest of the session."""
    for update in CLICK_TO_MOVE:
        widget.handle_movement_progress_update(update)
    assert widget.label_movement_instructions.text() == INSTRUCTIONS_TEXT


def test_no_progress_message_says_updating_ui(widget):
    """Developer phrasing was fine while nobody could see it.

    These were toasts on a preference that defaults to off, so the wording never
    reached a user. In the label it is on screen.
    """
    for update in CLICK_TO_MOVE:
        widget.handle_movement_progress_update(update)
        assert "updating UI" not in widget.label_movement_instructions.text()


def test_one_start_message_not_two(widget):
    """The dropped message was replaced within 38 ms by the next one.

    Nobody ever read it, and it was the one that duplicated in the log.
    """
    starts = [u["msg"] for u in CLICK_TO_MOVE if "msg" in u and "Moving" in u["msg"]]
    assert len(starts) == 1, starts


def test_a_vertical_move_says_so(widget):
    """A plain double-click moves laterally, Alt + double-click along the beam axis.

    The user chose between them, so the status should not call both the same thing.
    """
    import inspect

    from fibsem.ui import FibsemMovementWidget as module

    source = inspect.getsource(module)
    assert "Moving the stage vertically…" in source


def test_every_path_says_the_same_thing_while_acquiring(widget):
    """Three paths had drifted to "updating images", "taking new images", and nothing.

    Checked against the source rather than the constant alone, which would still pass
    if every path quietly stopped using it.
    """
    import inspect

    from fibsem.ui import FibsemMovementWidget as module

    source = inspect.getsource(module)
    assert source.count('{"msg": ACQUIRING_IMAGES}') == 3
    for stale in ("updating images", "taking new images", "updating UI"):
        assert stale not in source, stale
