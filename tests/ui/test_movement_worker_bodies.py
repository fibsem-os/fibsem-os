"""What the two stage-move workers do, in order, and where each part of it runs.

``absolute_movement_worker`` and ``move_to_orientation_worker`` each perform one
operation off the GUI thread and arrange three effects around it: a message naming the
target, the blocking move, a message saying the images are being retaken, and the
refresh itself. The workers used to do all four themselves, from the worker thread.
They now do only the move; the reporting sits on the GUI thread on either side, before
``start()`` and in the ``returned`` slot.

Two layers, deliberately:

* **The body**, called directly through ``__wrapped__``. These say the operation is all
  that is left in there -- the rule the split exists to establish, and the one a later
  edit is most likely to walk back by reaching for ``self`` from the thread again.
* **The path**, started for real through the public entry points. Everything except the
  move now happens on the GUI thread, so the *order* of the effects is observable end to
  end rather than scrambled by queued delivery. These assertions predate the split and
  are unchanged by it, which is what makes them the equivalence argument.
"""

from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from PyQt5.QtCore import QEventLoop, QTimer

# Same construction seam the rest of tests/ui uses: a host with a view controller, a
# real image widget and a Demo microscope. A hand-injected stand-in would let a worker
# that never touched the microscope pass.
sys.path.insert(0, os.path.dirname(__file__))  # not str.rsplit: Windows paths
from test_viewer_less_widgets import (  # noqa: E402
    _CanvasHost,
    _image_widget,
    _movement_widget,
)

from fibsem.structures import FibsemStagePosition  # noqa: E402
from fibsem.ui.FibsemMovementWidget import (  # noqa: E402
    ACQUIRING_IMAGES,
    FibsemMovementWidget,
)

TARGET = FibsemStagePosition(x=1e-6, y=1e-6, z=1e-6, r=0.0, t=0.0)


def _pump(ms: int = 200) -> None:
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


def _trace(movement, monkeypatch) -> list:
    """Every reported effect, in the order it happens.

    ``update_ui_after_movement`` is replaced on the instance, which bypasses its
    ``@ensure_main_thread``. Harmless now that the caller is already the GUI thread,
    and it keeps the recorder from being queued behind the assertions.
    """
    seen = []

    def _reported(ddict: dict) -> None:
        # Only the messages. `move_stage_finished` also emits {"finished": True}, which
        # carries no message and is pinned by the buttons-back test instead.
        if ddict.get("msg") is not None:
            seen.append(("msg", ddict["msg"]))

    movement.movement_progress_signal.connect(_reported)

    def _moved(stage_position, *args, **kwargs):
        seen.append(("move", stage_position))

    def _oriented(orientation, *args, **kwargs):
        seen.append(("orient", orientation))

    monkeypatch.setattr(
        movement.microscope, "safe_absolute_stage_movement", _moved, raising=False
    )
    monkeypatch.setattr(
        movement.microscope, "move_to_orientation", _oriented, raising=False
    )
    monkeypatch.setattr(
        movement,
        "update_ui_after_movement",
        lambda *a, **k: seen.append(("refresh", None)),
    )
    return seen


def _kinds(seen) -> list:
    return [kind for kind, _ in seen]


def _body(name):
    """The undecorated worker function -- ``@thread_worker`` would put it on a thread,
    where the raise is swallowed into ``errored`` and nothing is observable in place."""
    return getattr(FibsemMovementWidget, name).__wrapped__


def _explode(*args, **kwargs):
    raise RuntimeError("the stage did not get there")


# --- the body performs the operation and nothing else ------------------------


def test_the_absolute_worker_body_only_moves(movement, monkeypatch):
    """No message, no refresh -- one call to the microscope.

    This is the rule the split exists to establish. Anything the body adds here is a
    widget touched from a worker thread, which is the whole hazard.
    """
    seen = _trace(movement, monkeypatch)
    _body("absolute_movement_worker")(movement, stage_position=TARGET)

    assert _kinds(seen) == ["move"], seen
    assert seen[0][1] is TARGET, "the worker did not move to the position it was given"


def test_the_orientation_worker_body_only_moves(movement, monkeypatch):
    seen = _trace(movement, monkeypatch)
    _body("move_to_orientation_worker")(movement, "SEM")

    assert _kinds(seen) == ["orient"], seen
    assert seen[0][1] == "SEM"


def test_a_failed_body_raises_rather_than_reporting(movement, monkeypatch):
    """The exception leaves the body, so ``FunctionWorker`` turns it into ``errored``
    and the ``returned`` slot -- which is where the reporting now lives -- never runs."""
    seen = _trace(movement, monkeypatch)
    monkeypatch.setattr(movement.microscope, "safe_absolute_stage_movement", _explode)

    with pytest.raises(RuntimeError):
        _body("absolute_movement_worker")(movement, stage_position=TARGET)

    assert seen == [], seen


# --- the whole path, unchanged by the split ----------------------------------


def test_the_absolute_move_is_bracketed_by_its_two_messages(movement, monkeypatch):
    """Says where it is going, goes there, says it is retaking the images, retakes them.

    The order is the behaviour: the first message has to be out before the blocking
    call or nobody reads it while the stage moves, and ACQUIRING_IMAGES has to wait
    until the move is done or it claims the acquisition has started while the stage
    is still travelling.
    """
    seen = _trace(movement, monkeypatch)
    movement._move_to_absolute_position(TARGET)
    _run(lambda: _kinds(seen) == ["msg", "move", "msg", "refresh"])

    assert _kinds(seen) == ["msg", "move", "msg", "refresh"], seen
    assert seen[1][1] is TARGET
    assert seen[2][1] == ACQUIRING_IMAGES


def test_the_orientation_move_is_bracketed_by_its_two_messages(movement, monkeypatch):
    seen = _trace(movement, monkeypatch)
    movement.move_to_orientation("SEM")
    _run(lambda: _kinds(seen) == ["msg", "orient", "msg", "refresh"])

    assert _kinds(seen) == ["msg", "orient", "msg", "refresh"], seen
    assert seen[1][1] == "SEM"
    assert seen[2][1] == ACQUIRING_IMAGES


def test_the_absolute_move_names_its_target(movement, monkeypatch):
    """The operator is told which position, not just that something is moving."""
    seen = _trace(movement, monkeypatch)
    movement._move_to_absolute_position(TARGET)
    _run(lambda: seen)
    assert TARGET.pretty in seen[0][1], seen[0][1]


def test_the_orientation_move_names_its_orientation(movement, monkeypatch):
    seen = _trace(movement, monkeypatch)
    movement.move_to_orientation("MILLING")
    _run(lambda: seen)
    assert "MILLING orientation" in seen[0][1], seen[0][1]


def test_a_failed_move_reports_nothing_further(movement, monkeypatch):
    """Nothing after the move runs, so the widget never claims to be acquiring images
    for a move that did not happen."""
    seen = _trace(movement, monkeypatch)
    monkeypatch.setattr(movement.microscope, "safe_absolute_stage_movement", _explode)

    movement._move_to_absolute_position(TARGET)
    _run(lambda: movement.pushButton_move.isEnabled())

    assert _kinds(seen) == ["msg"], seen
    assert ACQUIRING_IMAGES not in [value for _, value in seen]


def test_a_failed_orientation_move_reports_nothing_further(movement, monkeypatch):
    seen = _trace(movement, monkeypatch)
    monkeypatch.setattr(movement.microscope, "move_to_orientation", _explode)

    movement.move_to_orientation("SEM")
    _run(lambda: movement.pushButton_move.isEnabled())

    assert _kinds(seen) == ["msg"], seen


def test_an_unknown_orientation_is_refused_before_any_thread_starts(movement):
    """The guard is on the entry point, not in the worker, so a bad name fails where
    the caller can see it instead of on a thread."""
    with pytest.raises(ValueError):
        movement.move_to_orientation("SIDEWAYS")


def test_a_failed_move_gives_the_buttons_back(movement, monkeypatch):
    """``move_stage_finished`` hangs off ``finished``, which fires on both outcomes.

    A move that raises must not leave the widget permanently disabled -- the operator
    would have to restart the app to try again.
    """
    monkeypatch.setattr(movement.microscope, "safe_absolute_stage_movement", _explode)
    assert movement.pushButton_move.isEnabled(), "precondition: buttons start enabled"

    movement._move_to_absolute_position(TARGET)
    _run(lambda: movement.pushButton_move.isEnabled())
    assert movement.pushButton_move.isEnabled(), (
        "a failed move left the movement buttons disabled"
    )


def test_the_refresh_runs_before_the_buttons_are_reconsidered(movement, monkeypatch):
    """``returned`` fires before ``finished``, and the widget depends on that.

    ``update_ui_after_movement`` queues the acquisition; ``move_stage_finished`` then
    asks whether one is running and declines to re-enable the buttons if so. Reverse
    the two and the buttons come back for the second it takes the images to land,
    offering a double-click that is still disabled. Hanging the refresh off
    ``finished`` instead of ``returned`` would do exactly that.
    """
    order = []
    monkeypatch.setattr(
        movement.microscope, "safe_absolute_stage_movement", lambda *a, **k: None
    )
    monkeypatch.setattr(
        movement, "update_ui_after_movement", lambda *a, **k: order.append("refresh")
    )
    original = movement.move_stage_finished
    monkeypatch.setattr(
        movement,
        "move_stage_finished",
        lambda *a, **k: (order.append("finished"), original())[1],
    )

    movement._move_to_absolute_position(TARGET)
    _run(lambda: "finished" in order)
    assert order == ["refresh", "finished"], order


def test_a_real_move_keeps_the_buttons_down_until_the_images_land(movement):
    """The same dance with nothing stubbed out -- a real Demo move and a real retake.

    Every other test here replaces ``update_ui_after_movement``, so none of them
    exercises the acquisition it starts. A move that quietly stopped retaking images
    would satisfy all of them: the refresh was called, and that is all they check.
    This one lets it run, so ``is_acquiring`` has to actually become true -- which is
    what ``move_stage_finished`` reads when it decides whether to give the buttons
    back, and the one fact in this file that a stub cannot supply honestly.

    The buttons must be down from the moment the move starts, still down when
    ``finished`` arrives mid-acquisition, and back only once the images have landed.
    """
    seen = []
    movement.movement_progress_signal.connect(
        lambda d: seen.append(
            (
                dict(d),
                movement.pushButton_move.isEnabled(),
                movement.image_widget.is_acquiring,
            )
        )
    )

    assert movement.pushButton_move.isEnabled(), "precondition: buttons start enabled"
    movement._move_to_absolute_position(TARGET)
    assert not movement.pushButton_move.isEnabled(), (
        "buttons stayed live during the move"
    )

    _run(
        lambda: (
            len(seen) >= 3
            and movement.pushButton_move.isEnabled()
            and not movement.image_widget.is_acquiring
        ),
        tries=60,
    )

    assert [d.get("msg") for d, _, _ in seen[:2]] == [
        seen[0][0].get("msg"),
        ACQUIRING_IMAGES,
    ], seen
    assert not any(enabled for _, enabled, _ in seen), (
        f"the buttons came back while the move was still reporting: {seen}"
    )
    assert seen[-1][0].get("finished") is True, seen[-1]
    assert seen[-1][2] is True, (
        "the acquisition had not started when the move finished -- move_stage_finished "
        "would have re-enabled the buttons over it"
    )
    assert movement.pushButton_move.isEnabled(), "the buttons never came back"
