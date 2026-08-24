"""Where movement progress goes, now that it is no longer a stream of toasts.

One click-to-move emits four progress messages inside ~45 ms. As toasts they stack
into a wall of popups saying nothing the moving stage does not already show. They go
to the quad-view info bar instead -- *not* to the instructions label on the Movement
tab, because five of the six paths that start a stage move start it from somewhere
else, and a message on a hidden tab is one nobody reads. The info bar sits beside the
canvas that was clicked and is visible from every tab.

They are invisible today because ``display.toasts_enabled`` defaults to False. This is
what has to be true before that default can change.
"""

from __future__ import annotations

import os
import sys
import threading

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from PyQt5 import QtWidgets
from PyQt5.QtCore import QEventLoop, QTimer

# The real construction seam: a host with a view controller, a real image widget and a
# Demo microscope. Building the widget through `__new__` and hand-injecting a label
# would let every test here pass with `movement_progress_signal` disconnected.
sys.path.insert(0, os.path.dirname(__file__))  # not str.rsplit: Windows paths
from test_viewer_less_widgets import (  # noqa: E402
    _CanvasHost,
    _image_widget,
    _movement_widget,
)

from fibsem.structures import BeamType, FibsemStagePosition, Point  # noqa: E402
from fibsem.ui.FibsemMovementWidget import (  # noqa: E402
    ACQUIRING_IMAGES,
    INSTRUCTIONS_TEXT,
)

TARGET = FibsemStagePosition(x=1e-6, y=1e-6, z=1e-6, r=0.0, t=0.0)


def _pump(ms: int = 200) -> None:
    """Let queued signals and the debounced info-bar render run."""
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec_()


def _run(condition, tries: int = 15) -> None:
    """Pump until *condition* holds, or give up and let the assertion report it."""
    for _ in range(tries):
        _pump()
        if condition():
            return


@pytest.fixture
def movement(qapp):
    host = _CanvasHost()
    _image_widget(host)
    widget = _movement_widget(host)
    yield widget
    host.deleteLater()


def _info(widget, beam: BeamType = BeamType.ELECTRON) -> dict:
    """The info-bar fields on one canvas, as {key: text}."""
    controller = widget._view_controller()
    canvas = (
        controller.sem_canvas if beam is BeamType.ELECTRON else controller.fib_canvas
    )
    return dict(controller._states[canvas].info)


def _status(widget, beam: BeamType = BeamType.ELECTRON):
    return _info(widget, beam).get("move")


def _record(widget) -> list:
    """Every status the info bar held, in order, as the move ran."""
    seen = []
    widget.movement_progress_signal.connect(lambda _d: seen.append(_status(widget)))
    return seen


# --- it reaches the info bar, on every canvas ---------------------------------


def test_progress_reaches_the_info_bar(movement):
    seen = _record(movement)
    movement._move_to_absolute_position(TARGET)
    _run(lambda: seen and seen[-1] is None)
    assert any(s and s.startswith("Moving to") for s in seen), seen


def test_every_canvas_carries_it(movement):
    """A stage move moves what all three canvases are looking at, so the status is not
    the electron canvas's business alone -- the operator may be watching any of them."""
    movement._set_move_status("Moving the stage…")
    controller = movement._view_controller()
    for canvas in (controller.sem_canvas, controller.fib_canvas, controller.fm_canvas):
        assert dict(controller._states[canvas].info).get("move") == "Moving the stage…"


def test_it_does_not_disturb_the_stage_readout(movement):
    """The info bar is shared. `STAGE:` is what this widget already writes there, and a
    status that replaced it would trade one reading for another."""
    before = _info(movement).get("stage")
    assert before, "no stage readout to begin with"
    movement._set_move_status("Moving the stage…")
    assert _info(movement).get("stage") == before


# --- the reason it is not on the Movement tab ---------------------------------


def test_the_status_shows_from_another_tab(movement):
    """The regression this replaced: click-to-move works whatever tab is showing --
    `_click_to_move_available` gates on the button being enabled and nothing else, and
    the canvas sits beside the tabs, not inside them. A status on the Movement tab is
    therefore usually written where nobody can see it."""
    tabs = QtWidgets.QTabWidget()
    tabs.addTab(QtWidgets.QLabel("Image"), "Image")
    tabs.addTab(movement, "Movement")
    tabs.addTab(QtWidgets.QLabel("Milling"), "Milling")
    tabs.show()
    tabs.setCurrentIndex(2)  # Milling: the Movement tab is now hidden
    _pump(150)

    assert not movement.label_movement_instructions.isVisible(), (
        "the Movement tab is showing; this test proves nothing"
    )
    seen = _record(movement)
    movement._move_to_absolute_position(TARGET)
    _run(lambda: seen and seen[-1] is None)
    assert any(s and s.startswith("Moving to") for s in seen), seen
    tabs.close()


def test_the_instructions_label_is_left_alone(movement):
    """It says what a double-click does, which does not change while the stage moves."""
    seen = []
    movement.movement_progress_signal.connect(
        lambda _d: seen.append(movement.label_movement_instructions.text())
    )
    movement._move_to_absolute_position(TARGET)
    _run(lambda: seen and _status(movement) is None)
    assert set(seen) == {INSTRUCTIONS_TEXT}, seen


# --- no toasts ----------------------------------------------------------------


def test_progress_does_not_toast(movement, monkeypatch):
    """The whole point: four popups per click is worse than none.

    Scoped to the movement messages. The image widget still raises one of its own for
    the acquisition that follows -- that is its message on its own path, and one popup
    per move was never the complaint.
    """
    from fibsem.ui import notification_service

    shown = []
    monkeypatch.setattr(
        notification_service,
        "show_toast",
        lambda *a, **k: shown.append(a[0] if a else ""),
    )
    progress = []
    movement.movement_progress_signal.connect(lambda d: progress.append(d.get("msg")))
    movement._move_to_absolute_position(TARGET)
    _run(lambda: _status(movement) is None)

    assert any(progress), "the move emitted no progress at all"
    assert [t for t in shown if t in progress] == [], shown


# --- when it clears -----------------------------------------------------------


def test_it_clears_when_the_images_land_not_before(movement):
    """`update_ui_after_movement` only *queues* the acquisition before the worker
    returns, so `finished` arrives with ~a second of acquiring still to come.
    `move_stage_finished` already declines to re-enable the buttons in that window; if
    the status cleared there it would read "nothing is happening" over the acquisition
    it just announced, and offer a double-click that is still disabled."""
    at_finished = []
    movement.movement_progress_signal.connect(
        lambda d: (
            d.get("finished")
            and at_finished.append(
                (_status(movement), movement.image_widget.is_acquiring)
            )
        )
    )
    movement._move_to_absolute_position(TARGET)
    _run(lambda: at_finished and not movement.image_widget.is_acquiring)

    assert at_finished, "the move never finished"
    status, acquiring = at_finished[-1]
    assert acquiring, "the images had already landed; this test proves nothing"
    assert status == ACQUIRING_IMAGES, status
    assert _status(movement) is None, "the status outlived the acquisition"


def test_it_clears_when_the_move_fails(movement, monkeypatch):
    """A raising move still ends: `FunctionWorker` emits `finished` from a `finally`.
    Without that the info bar would keep 'Moving to…' for the rest of the session."""
    monkeypatch.setattr(
        movement.microscope,
        "safe_absolute_stage_movement",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stage refused")),
    )
    movement._move_to_absolute_position(TARGET)
    _run(lambda: _status(movement) is None)
    assert _status(movement) is None


# --- what it says -------------------------------------------------------------


@pytest.mark.parametrize(
    "vertical, expected",
    [(False, "Moving the stage…"), (True, "Moving the stage vertically…")],
)
def test_a_vertical_move_says_so(movement, vertical, expected):
    """A plain double-click moves laterally, Alt + double-click along the beam axis.
    The user chose between them, so the status should not call both the same thing."""
    seen = _record(movement)
    movement._execute_stage_move(
        BeamType.ION, Point(x=1e-6, y=1e-6), vertical_move=vertical
    )
    _pump()
    assert seen[0] == expected, seen


def test_every_path_says_the_same_thing_while_acquiring(movement):
    """Three paths had drifted to "updating images", "taking new images", and nothing
    at all -- the same phase looked different depending on how the move was started.

    Driven through each path rather than counted in the source, which would still pass
    if a path stopped reaching the emit at all."""
    paths = {
        "absolute": lambda: movement._move_to_absolute_position(TARGET),
        "click": lambda: movement._execute_stage_move(
            BeamType.ION, Point(x=1e-6, y=1e-6), vertical_move=False
        ),
        "orientation": lambda: movement.move_to_orientation("SEM"),
    }
    for name, start in paths.items():
        seen = _record(movement)
        start()
        _run(lambda: _status(movement) is None)
        assert ACQUIRING_IMAGES in seen, f"{name} path: {seen}"


def test_no_progress_message_says_updating_ui(movement):
    """Developer phrasing was fine while nobody could see it. These were toasts on a
    preference that defaults to off, so the wording never reached a user."""
    seen = _record(movement)
    movement._move_to_absolute_position(TARGET)
    _run(lambda: seen and seen[-1] is None)
    assert not any(s and "updating UI" in s for s in seen), seen


# --- the threading question ---------------------------------------------------


def test_the_status_is_written_on_the_gui_thread(movement):
    """`movement_progress_signal` is a `pyqtSignal` on a widget with main-thread
    affinity, so emitting it from a worker queues delivery onto the GUI thread -- which
    is why the handler needs no `@ensure_main_thread` (`update_ui_after_movement` does,
    because a worker calls *that* directly rather than through a signal). Asserted
    rather than assumed: writing the info bar off-thread would be a Qt violation."""
    threads = []
    movement.movement_progress_signal.connect(
        lambda _d: threads.append(threading.current_thread().name)
    )
    movement._move_to_absolute_position(TARGET)
    _run(lambda: _status(movement) is None)
    assert threads, "the handler never ran"
    assert set(threads) == {"MainThread"}, threads
