"""The vertical-move capability gate (FIB-507 T4a, FIB-785).

An Alt-click vertical move dispatches to ``microscope.vertical_move(..., beam_type=...)``,
where ``beam_type`` names the view the offset was measured in. Not every backend can
correct from every view, so the widget asks ``supports_vertical_move`` first: a backend
that cannot used to fall through to ``stable_move`` silently, and the operator asked to
restore coincidence and got a sample-plane move instead. The widget must refuse with a
toast, and leave every other dispatch branch untouched.

The refusal and the dispatch live either side of a thread. ``_execute_stage_move``
decides, on the GUI thread, whether the move may go ahead at all; ``_stage_move_worker``
performs whichever of the two calls applies. So the refusal is asserted on the former
and the branches on the latter, called through ``__wrapped__``.

No Qt event loop or microscope required: the widget is created without __init__ and
only the attributes each function touches are provided. The recorders below borrow the
real ``supports_vertical_move`` rather than reimplementing it, so they cannot drift from
the contract the widget is being tested against.
"""

from unittest.mock import Mock

import pytest

try:
    from fibsem.ui import notification_service
    from fibsem.ui.widgets.stage_control_widget import StageControlWidget

    _MISSING_UI_DEPS = None
except ImportError as e:  # pragma: no cover - exercised only on UI-less CI
    _MISSING_UI_DEPS = str(e)

from fibsem.microscope import FibsemMicroscope
from fibsem.structures import BeamType, Point

pytestmark = pytest.mark.skipif(
    bool(_MISSING_UI_DEPS), reason=f"UI dependencies not installed: {_MISSING_UI_DEPS}"
)


class RecordingMicroscope:
    """Records move calls. Corrects coincidence from either view (Thermo/Tescan)."""

    vertical_move_views = (BeamType.ION, BeamType.ELECTRON)
    supports_vertical_move = FibsemMicroscope.supports_vertical_move

    def __init__(self):
        self.calls = []

    def stable_move(self, dx, dy, beam_type):
        self.calls.append(("stable_move", dx, dy, beam_type))

    def vertical_move(self, dy, dx=0.0, beam_type=BeamType.ION):
        self.calls.append(("vertical_move", dx, dy, beam_type))


class FibOnlyMicroscope(RecordingMicroscope):
    """A backend that can only correct coincidence from the FIB view."""

    vertical_move_views = (BeamType.ION,)


@pytest.fixture
def toasts(monkeypatch):
    events = []
    monkeypatch.setattr(
        notification_service,
        "show_toast",
        lambda msg, notification_type="info": events.append((msg, notification_type)),
    )
    return events


def make_widget(microscope):
    # sip forbids object.__new__ on QWidget subclasses; the class's own __new__
    # allocates just the Python wrapper (no QApplication / C++ widget needed)
    widget = StageControlWidget.__new__(StageControlWidget)
    widget.microscope = microscope
    widget._report_move = Mock()
    widget.update_ui_after_movement = Mock()
    return widget


def refuse_or_start(widget, beam_type, vertical):
    """The GUI-thread half: it either declines, or commits to starting a worker."""
    widget._execute_stage_move(
        beam_type=beam_type, point=Point(1e-6, 2e-6), vertical_move=vertical
    )


def dispatch(widget, beam_type, vertical):
    """The worker half, run in place -- `@thread_worker` would put it on a thread."""
    StageControlWidget._stage_move_worker.__wrapped__(
        widget, beam_type, Point(1e-6, 2e-6), vertical
    )


def test_an_unsupported_view_refuses_with_a_toast(toasts):
    m = FibOnlyMicroscope()
    refuse_or_start(make_widget(m), BeamType.ELECTRON, vertical=True)

    assert m.calls == []  # no silent stable_move fallback
    assert len(toasts) == 1
    msg, level = toasts[0]
    assert "not supported" in msg and "SEM view" in msg
    assert level == "warning"


def test_the_sem_view_is_passed_through_as_the_beam_type(toasts):
    m = RecordingMicroscope()
    dispatch(make_widget(m), BeamType.ELECTRON, vertical=True)

    # dx is still discarded on this path -- see FIB-773
    assert m.calls == [("vertical_move", 0, 2e-6, BeamType.ELECTRON)]
    assert toasts == []


def test_the_fib_view_still_carries_dx(toasts):
    m = RecordingMicroscope()
    dispatch(make_widget(m), BeamType.ION, vertical=True)

    assert m.calls == [("vertical_move", 1e-6, 2e-6, BeamType.ION)]
    assert toasts == []


def test_a_stable_move_is_unaffected(toasts):
    """The gate only ever applies to a vertical move, so a backend that declares
    no SEM view still gets a plain stable move from the SEM canvas."""
    m = FibOnlyMicroscope()
    dispatch(make_widget(m), BeamType.ELECTRON, vertical=False)

    assert [c[0] for c in m.calls] == ["stable_move"]
    assert toasts == []


def test_a_real_backend_satisfies_the_widget(toasts):
    """The wiring, against a real DemoMicroscope rather than a recorder: the widget
    asks the capability query and then calls vertical_move with the view. The
    simulator answers yes to both views, so neither Alt-click is refused."""
    import os

    import fibsem.config as fibsem_config
    from fibsem import utils

    scope, _ = utils.setup_session(
        manufacturer="Demo",
        config_path=os.path.join(
            os.path.dirname(fibsem_config.__file__),
            "config",
            "microscope-configuration.yaml",
        ),
    )
    widget = make_widget(scope)
    start = scope.get_stage_position()

    # the GUI-thread half asks the query, and a real simulator answers yes
    assert scope.supports_vertical_move(BeamType.ELECTRON)
    # the worker half then reaches the backend's SEM branch and moves the stage
    dispatch(widget, BeamType.ELECTRON, vertical=True)

    assert toasts == []
    assert not scope.get_stage_position().is_close2(start, tol=1e-9)
