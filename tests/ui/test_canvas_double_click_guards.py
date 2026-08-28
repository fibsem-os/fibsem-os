"""What a canvas double-click decides, and where each decision is made.

Five guards stand between a double-click and a stage move: the Shift modifier, milling
in progress, no image acquired yet, a click landing outside the image, and -- one level
down in ``_execute_stage_move`` -- a SEM-view vertical move on a backend that cannot do
one. Only the first of those is checked on the GUI thread today. The rest are evaluated
inside ``_canvas_double_click_worker``, on the worker thread, after the widget has
already disabled its buttons in anticipation of a move.

Nothing here is a threading bug; the guards give the right answers. What is untested is
the answers themselves, so this file pins them before they are moved.

Two of these tests pin behaviour that is expected to *change*: a refused click currently
disables and re-enables the buttons, and reports ``finished`` for a move that never
happened. They are marked where they appear. Recording them is the point -- it is what
makes the follow-up visible as a behaviour change rather than a silent one.
"""

from __future__ import annotations

import os
import sys
import threading

import numpy as np
import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from PyQt5.QtCore import QEventLoop, QTimer

# The real construction seam: a host with a view controller, a real image widget and a
# Demo microscope. The guards read `image_widget.eb_image` and `parent.milling_widget`,
# so a hand-built stand-in would be answering different questions.
sys.path.insert(0, os.path.dirname(__file__))  # not str.rsplit: Windows paths
from test_viewer_less_widgets import (  # noqa: E402
    _CanvasHost,
    _image,
    _image_widget,
    _movement_widget,
    _session,
)

from fibsem.structures import BeamType, Point  # noqa: E402
from fibsem.ui import notification_service  # noqa: E402
from fibsem.ui.FibsemMovementWidget import FibsemMovementWidget  # noqa: E402


def _pump(ms: int = 150) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec_()


def _run(condition, tries: int = 20) -> None:
    for _ in range(tries):
        _pump()
        if condition():
            return


@pytest.fixture
def movement(qapp):
    """A movement widget whose image widget already holds a SEM and a FIB image."""
    host = _CanvasHost()
    image_widget = _image_widget(host)
    image_widget._on_acquire(_image(BeamType.ELECTRON))
    image_widget._on_acquire(_image(BeamType.ION))
    widget = _movement_widget(host)
    yield widget
    host.deleteLater()


@pytest.fixture
def toasts(monkeypatch):
    events = []
    monkeypatch.setattr(
        notification_service,
        "show_toast",
        lambda msg, notification_type="info": events.append((msg, notification_type)),
    )
    return events


@pytest.fixture
def dispatched(movement, monkeypatch):
    """Every `_execute_stage_move` call, as (beam_type, point, vertical_move)."""
    calls = []
    monkeypatch.setattr(
        movement,
        "_execute_stage_move",
        lambda beam_type, point, vertical_move, coords=None: calls.append(
            (beam_type, point, vertical_move)
        ),
    )
    return calls


def _click(movement, beam=BeamType.ELECTRON, x=100.0, y=100.0, modifiers=frozenset()):
    """Run the worker body on this thread -- `@thread_worker` would hide the outcome."""
    FibsemMovementWidget._canvas_double_click_worker.__wrapped__(
        movement, beam, x, y, set(modifiers)
    )


def _centre(movement, beam=BeamType.ELECTRON):
    image = (
        movement.image_widget.eb_image
        if beam is BeamType.ELECTRON
        else movement.image_widget.ib_image
    )
    h, w = image.data.shape[:2]
    return w / 2.0, h / 2.0


# --- a click that survives every guard ---------------------------------------


def test_a_click_inside_the_image_dispatches_a_move(movement, dispatched, toasts):
    x, y = _centre(movement)
    _click(movement, BeamType.ELECTRON, x, y)

    assert len(dispatched) == 1, dispatched
    beam, point, vertical = dispatched[0]
    assert beam is BeamType.ELECTRON
    assert isinstance(point, Point)
    assert vertical is False, "a plain double-click is a lateral move"
    assert toasts == [], "a move that went ahead should say nothing"


def test_the_clicked_pixel_becomes_a_microscope_offset(movement, dispatched):
    """The centre of the image is the beam axis, so clicking it asks for no offset.

    Pins the conversion itself, not just that one happened -- a click that dispatched
    the raw pixel coordinates would drive the stage a very long way.
    """
    x, y = _centre(movement)
    _click(movement, BeamType.ELECTRON, x, y)

    _beam, point, _vertical = dispatched[0]
    assert np.isclose(point.x, 0.0, atol=1e-9), point
    assert np.isclose(point.y, 0.0, atol=1e-9), point


def test_a_click_off_centre_offsets_in_metres(movement, dispatched):
    """Off-centre by a known number of pixels is off-axis by pixels x pixelsize."""
    cx, cy = _centre(movement)
    pixelsize = movement.image_widget.eb_image.metadata.pixel_size.x
    _click(movement, BeamType.ELECTRON, cx + 100, cy)

    _beam, point, _vertical = dispatched[0]
    assert np.isclose(abs(point.x), 100 * pixelsize, rtol=1e-6), (point, pixelsize)


def test_alt_asks_for_a_vertical_move(movement, dispatched):
    x, y = _centre(movement)
    _click(movement, BeamType.ION, x, y, modifiers={"Alt"})

    _beam, _point, vertical = dispatched[0]
    assert vertical is True


def test_the_fib_canvas_dispatches_against_the_fib_image(movement, dispatched):
    """Each canvas carries its own image; a click on one must not be measured against
    the other's pixel size."""
    x, y = _centre(movement, BeamType.ION)
    _click(movement, BeamType.ION, x, y)

    beam, _point, _vertical = dispatched[0]
    assert beam is BeamType.ION


# --- the four guards inside the worker ---------------------------------------


def test_a_shift_click_is_not_a_move(movement, dispatched, toasts):
    """Shift is the canvas's own modifier, so it must not also drive the stage --
    and it is a deliberate gesture, so it is refused silently."""
    x, y = _centre(movement)
    _click(movement, BeamType.ELECTRON, x, y, modifiers={"Shift"})

    assert dispatched == []
    assert toasts == [], "a Shift-click is not an error worth a popup"


def test_a_click_during_milling_is_refused_out_loud(movement, dispatched, toasts):
    """The stage must not move under a running mill, and the operator is told why --
    unlike the silent guards, this one refuses something they meant to do.

    `is_milling` is a read-only property that asks whether a thread is alive, and it
    delegates twice (viewer -> run controls -> thread). Driving it with a real blocked
    thread exercises that whole chain; assigning a stand-in would skip the delegation
    that has already broken this guard once.
    """
    from fibsem.ui.widgets.milling_task_viewer_widget import MillingTaskViewerWidget

    viewer = MillingTaskViewerWidget(microscope=_session()[0])
    movement.parent.milling_widget = viewer

    release = threading.Event()
    thread = threading.Thread(target=release.wait, daemon=True)
    thread.start()
    viewer.milling_widget._milling_thread = thread
    try:
        assert viewer.is_milling, "precondition: the widget should report milling"
        x, y = _centre(movement)
        _click(movement, BeamType.ELECTRON, x, y)
    finally:
        release.set()
        thread.join(timeout=5)

    assert dispatched == []
    assert len(toasts) == 1, toasts
    assert "milling" in toasts[0][0].lower(), toasts


def test_the_no_image_guard_never_fires(qapp, toasts):
    """FINDING, pinned as it stands: the "No image available to move from" branch is
    unreachable in the app.

    `FibsemImageSettingsWidget.__init__` seeds `eb_image` / `ib_image` with
    `generate_blank_image(...)` as internal fallbacks, so neither is ever None once the
    widget exists, and the blank carries metadata. A double-click before any acquisition
    therefore converts against the *placeholder's* pixel size -- derived from the
    configured resolution and hfw rather than from anything acquired -- and dispatches a
    real stage move, while the canvas still reads "No image".

    Whether an operator can reach it depends on whether an empty canvas emits
    double-clicks at all, which is not settled here. The guard's intent is defeated
    either way: it cannot distinguish "never acquired" from "acquired".
    """
    host = _CanvasHost()
    image_widget = _image_widget(host)  # deliberately not acquired into
    movement = _movement_widget(host)
    calls = []
    movement._execute_stage_move = lambda *a, **k: calls.append(a)

    assert image_widget.eb_image is not None, "the placeholder is what makes it dead"
    assert image_widget.eb_image.metadata is not None

    _click(movement, BeamType.ELECTRON, 100.0, 100.0)

    assert len(calls) == 1, "the guard fired -- if this changed, the finding is fixed"
    assert toasts == [], toasts
    host.deleteLater()


def test_a_click_outside_the_image_is_ignored(movement, dispatched, toasts):
    """The canvas is larger than the image it draws, so clicks land off it routinely.
    Silent, because it is a miss rather than a refusal."""
    image = movement.image_widget.eb_image
    h, w = image.data.shape[:2]
    _click(movement, BeamType.ELECTRON, w + 50.0, h + 50.0)

    assert dispatched == []
    assert toasts == [], "clicking past the edge of the image is not an error"


def test_the_far_edge_is_outside_and_the_near_edge_is_inside(movement, dispatched):
    """The bounds are half-open: 0 is on the image, width is not."""
    image = movement.image_widget.eb_image
    h, w = image.data.shape[:2]

    _click(movement, BeamType.ELECTRON, 0.0, 0.0)
    assert len(dispatched) == 1, "the first pixel should be clickable"

    _click(movement, BeamType.ELECTRON, float(w), float(h))
    assert len(dispatched) == 1, "one past the last pixel should not be"


# --- what the widget does around a refusal (expected to change) --------------


def test_a_refused_click_still_toggles_the_buttons(movement, toasts):
    """CURRENT BEHAVIOUR, pinned so that changing it is visible.

    `_on_canvas_double_click` disables the buttons before starting the worker, and the
    worker is where the refusal is decided -- so a Shift-click spawns a thread, disables
    the buttons, decides to do nothing, and re-enables them. The operator sees a flicker
    for a gesture that was never going to move the stage.

    Moving the guards onto the GUI thread removes this; when it does, this test should
    be updated to assert the buttons never moved, not deleted.
    """
    assert movement.pushButton_move.isEnabled(), "precondition: buttons start enabled"

    seen = []
    original = movement._toggle_interactions
    # Each call round-trips (movement -> image widget -> movement, terminated by the
    # `caller` sentinel), so what is asserted is the transition, not the call count.
    movement._toggle_interactions = lambda enable, caller=None: (
        seen.append(enable),
        original(enable, caller),
    )[1]

    movement._on_canvas_double_click(BeamType.ELECTRON, 100.0, 100.0, {"Shift"})
    _run(lambda: movement.pushButton_move.isEnabled() and True in seen)

    assert False in seen, f"the buttons were never disabled: {seen}"
    assert seen[0] is False and seen[-1] is True, (
        f"expected the buttons disabled and then re-enabled around a refusal: {seen}"
    )


def test_a_refused_click_still_reports_finished(movement, toasts):
    """CURRENT BEHAVIOUR, pinned so that changing it is visible.

    `move_stage_finished` hangs off the worker's `finished`, which fires whether or not
    the body did anything -- so a refused click emits `{"finished": True}` and refreshes
    the stage readout for a move that never happened.
    """
    reports = []
    movement.movement_progress_signal.connect(lambda d: reports.append(dict(d)))

    movement._on_canvas_double_click(BeamType.ELECTRON, 100.0, 100.0, {"Shift"})
    _run(lambda: reports)

    assert reports == [{"finished": True}], reports
