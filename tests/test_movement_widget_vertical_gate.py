"""The SEM-view vertical-move gate (FIB-507 T4a).

``FibsemMovementWidget._execute_stage_move`` dispatches an Alt-click vertical move
to ``move_coincident_from_sem`` for the ELECTRON view -- a method only some
backends implement (ThermoFisher, Odemis). Backends without it (e.g. TESCAN) used
to fall through to ``stable_move`` silently: the operator asked to restore
coincidence and got a sample-plane move instead. The widget must refuse with a
toast, and leave every other dispatch branch untouched.

No Qt event loop or microscope required: the widget is created without __init__
and only the attributes ``_execute_stage_move`` touches are provided.
"""

from unittest.mock import Mock

import pytest

try:
    from fibsem.ui import notification_service
    from fibsem.ui.FibsemMovementWidget import FibsemMovementWidget

    _MISSING_UI_DEPS = None
except ImportError as e:  # pragma: no cover - exercised only on UI-less CI
    _MISSING_UI_DEPS = str(e)

from fibsem.structures import BeamType, Point

pytestmark = pytest.mark.skipif(
    bool(_MISSING_UI_DEPS), reason=f"UI dependencies not installed: {_MISSING_UI_DEPS}"
)


class RecordingMicroscope:
    """Records move calls; has no move_coincident_from_sem (the TESCAN case)."""

    def __init__(self):
        self.calls = []

    def stable_move(self, dx, dy, beam_type):
        self.calls.append(("stable_move", dx, dy, beam_type))

    def vertical_move(self, dx, dy):
        self.calls.append(("vertical_move", dx, dy))


class SemCoincidentMicroscope(RecordingMicroscope):
    """A backend that implements the SEM-view coincidence move (Thermo/Odemis)."""

    def move_coincident_from_sem(self, dx, dy):
        self.calls.append(("move_coincident_from_sem", dx, dy))


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
    widget = FibsemMovementWidget.__new__(FibsemMovementWidget)
    widget.microscope = microscope
    widget.movement_progress_signal = Mock()
    widget.update_ui_after_movement = Mock()
    return widget


def execute(widget, beam_type, vertical):
    widget._execute_stage_move(
        beam_type=beam_type, point=Point(1e-6, 2e-6), vertical_move=vertical
    )


def test_sem_vertical_without_backend_support_refuses_with_a_toast(toasts):
    m = RecordingMicroscope()
    execute(make_widget(m), BeamType.ELECTRON, vertical=True)

    assert m.calls == []  # no silent stable_move fallback
    assert len(toasts) == 1
    msg, level = toasts[0]
    assert "not supported" in msg and "FIB view" in msg
    assert level == "warning"


def test_sem_vertical_with_backend_support_moves_coincident(toasts):
    m = SemCoincidentMicroscope()
    execute(make_widget(m), BeamType.ELECTRON, vertical=True)

    assert [c[0] for c in m.calls] == ["move_coincident_from_sem"]
    assert toasts == []


def test_fib_vertical_is_unaffected(toasts):
    m = RecordingMicroscope()
    execute(make_widget(m), BeamType.ION, vertical=True)

    assert [c[0] for c in m.calls] == ["vertical_move"]
    assert toasts == []


def test_sem_stable_move_is_unaffected(toasts):
    m = RecordingMicroscope()
    execute(make_widget(m), BeamType.ELECTRON, vertical=False)

    assert [c[0] for c in m.calls] == ["stable_move"]
    assert toasts == []
