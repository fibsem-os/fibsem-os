"""What the two stage-move workers actually do, in order, before FIB-828 rewrites them.

``absolute_movement_worker`` and ``move_to_orientation_worker`` each run one operation
off the GUI thread and interleave three cross-thread effects around it: a message
naming the target, the blocking move, a message saying the images are being retaken,
and the refresh itself. FIB-828 splits the operation out of the reporting, so these
tests exist to say what "the same behaviour" means once it has.

Two layers, deliberately:

* **The body**, called directly through ``__wrapped__`` on the GUI thread. Nothing is
  queued, so the *order* of the effects is observable -- which is the whole point of
  the messages. A worker that emitted both up front, or moved before it said where to,
  would satisfy a test that only checked both messages arrived.
* **The path**, started for real through the public entry points. These assert what
  must survive the extraction whatever shape the body ends up in, including the
  failure case, where the operation raises and the widget still has to come back.
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
    """Every cross-thread effect the worker performs, in the order it performs them.

    ``update_ui_after_movement`` is replaced on the instance, which bypasses its
    ``@ensure_main_thread`` -- the recorder runs wherever the caller is rather than
    being queued. That is what makes the ordering below readable, and it is why the
    *threading* of the refresh is pinned by the path tests instead of here.
    """
    seen = []
    movement.movement_progress_signal.connect(
        lambda d: seen.append(("msg", d.get("msg")))
    )

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


def _body(name):
    """The undecorated worker function -- ``@thread_worker`` would put it on a thread,
    where a raise is swallowed into ``errored`` and the ordering is unobservable."""
    return getattr(FibsemMovementWidget, name).__wrapped__


def _explode(*args, **kwargs):
    raise RuntimeError("the stage did not get there")


# --- absolute moves ----------------------------------------------------------


def test_the_absolute_move_is_bracketed_by_its_two_messages(movement, monkeypatch):
    """Says where it is going, goes there, says it is retaking the images, retakes them.

    The order is the behaviour: the first message has to be out before the blocking
    call or nobody reads it while the stage moves, and ACQUIRING_IMAGES has to wait
    until the move is done or it claims the acquisition has started while the stage
    is still travelling.
    """
    seen = _trace(movement, monkeypatch)
    _body("absolute_movement_worker")(movement, stage_position=TARGET)

    kinds = [kind for kind, _ in seen]
    assert kinds == ["msg", "move", "msg", "refresh"], seen
    assert seen[1][1] is TARGET, "the worker did not move to the position it was given"
    assert seen[2][1] == ACQUIRING_IMAGES


def test_the_absolute_move_names_its_target(movement, monkeypatch):
    """The operator is told which position, not just that something is moving."""
    seen = _trace(movement, monkeypatch)
    _body("absolute_movement_worker")(movement, stage_position=TARGET)
    assert TARGET.pretty in seen[0][1], seen[0][1]


def test_a_failed_absolute_move_reports_nothing_further(movement, monkeypatch):
    """The exception leaves the body -- ``FunctionWorker`` turns it into ``errored``.

    Nothing after the move runs, so the widget never claims to be acquiring images
    for a move that did not happen. Whatever FIB-828 does with the reporting, a
    failure must still stop the sequence here rather than continue past it.
    """
    seen = _trace(movement, monkeypatch)
    monkeypatch.setattr(movement.microscope, "safe_absolute_stage_movement", _explode)

    with pytest.raises(RuntimeError):
        _body("absolute_movement_worker")(movement, stage_position=TARGET)

    assert [kind for kind, _ in seen] == ["msg"], seen
    assert ACQUIRING_IMAGES not in [value for _, value in seen]


# --- orientation moves -------------------------------------------------------


def test_the_orientation_move_is_bracketed_by_its_two_messages(movement, monkeypatch):
    seen = _trace(movement, monkeypatch)
    _body("move_to_orientation_worker")(movement, "SEM")

    kinds = [kind for kind, _ in seen]
    assert kinds == ["msg", "orient", "msg", "refresh"], seen
    assert seen[1][1] == "SEM"
    assert seen[2][1] == ACQUIRING_IMAGES


def test_the_orientation_move_names_its_orientation(movement, monkeypatch):
    seen = _trace(movement, monkeypatch)
    _body("move_to_orientation_worker")(movement, "MILLING")
    assert "MILLING" in seen[0][1], seen[0][1]


def test_a_failed_orientation_move_reports_nothing_further(movement, monkeypatch):
    seen = _trace(movement, monkeypatch)
    monkeypatch.setattr(movement.microscope, "move_to_orientation", _explode)

    with pytest.raises(RuntimeError):
        _body("move_to_orientation_worker")(movement, "SEM")

    assert [kind for kind, _ in seen] == ["msg"], seen


def test_an_unknown_orientation_is_refused_before_any_thread_starts(movement):
    """The guard is on the entry point, not in the worker, so a bad name fails where
    the caller can see it instead of on a thread."""
    with pytest.raises(ValueError):
        movement.move_to_orientation("SIDEWAYS")


# --- the whole path, which must survive the extraction unchanged -------------


def test_the_absolute_move_path_refreshes_the_ui(movement, monkeypatch):
    """Started for real: the refresh has to reach the widget however the body is
    arranged, and it has to reach it on the GUI thread."""
    refreshed = []
    monkeypatch.setattr(
        movement.microscope, "safe_absolute_stage_movement", lambda *a, **k: None
    )
    monkeypatch.setattr(
        movement, "update_ui_after_movement", lambda *a, **k: refreshed.append(True)
    )

    movement._move_to_absolute_position(TARGET)
    _run(lambda: refreshed)
    assert refreshed, "the move finished without refreshing the UI"


def test_the_orientation_path_refreshes_the_ui(movement, monkeypatch):
    refreshed = []
    monkeypatch.setattr(
        movement.microscope, "move_to_orientation", lambda *a, **k: None
    )
    monkeypatch.setattr(
        movement, "update_ui_after_movement", lambda *a, **k: refreshed.append(True)
    )

    movement.move_to_orientation("SEM")
    _run(lambda: refreshed)
    assert refreshed, "the orientation move finished without refreshing the UI"


def test_a_failed_move_gives_the_buttons_back(movement, monkeypatch):
    """``move_stage_finished`` hangs off ``finished``, which fires on both outcomes.

    A move that raises must not leave the widget permanently disabled -- the operator
    would have to restart the app to try again. This is the failure path FIB-828 has
    the most room to break, because the extraction moves what runs after the operation.
    """
    monkeypatch.setattr(movement.microscope, "safe_absolute_stage_movement", _explode)
    assert movement.pushButton_move.isEnabled(), "precondition: buttons start enabled"

    movement._move_to_absolute_position(TARGET)
    _run(lambda: movement.pushButton_move.isEnabled())
    assert movement.pushButton_move.isEnabled(), (
        "a failed move left the movement buttons disabled"
    )
