"""The wheel move runs off the GUI thread, and coalesces (FIB-625).

`_execute_wheel_move` is a `qdebounced` wrapper, and `qdebounced` fires on a QTimer in
the GUI thread -- so the blocking `move_absolute` inside it stopped every repaint for
its duration, and the live FM canvas froze while you focused. Measured on the simulator:
150 ms debounce + ~500 ms move + ~210 ms of the device reads `_notify_moved` does,
roughly 860 ms of dead interface per nudge. The frames were arriving the whole time --
`FluorescenceMicroscope.start_acquisition` streams on its own thread -- they just could
not be drawn.

The debounce coalesces a scroll *burst* into one move. What it cannot do is coalesce
across the move itself, because the move used to hold the thread the debounce runs on.
Now that it does not, a nudge landing mid-move has to replace the target rather than
queue a second trip, or a slow objective turns a flick of the wheel into a march.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
import threading
import time

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QWidget

from fibsem.constants import METRE_TO_MICRON
from fibsem.fm.microscope import FluorescenceMicroscope
from fibsem.ui.fm.widgets.objective_control_widget import ObjectiveControlWidget

_app = QApplication.instance() or QApplication(sys.argv)

_GUI_THREAD = threading.current_thread()


class _Host(QWidget):
    """The duck-typed parent `_on_canvas_scroll` consults before moving anything."""

    is_acquisition_active = False
    viewer = None
    microscope = None

    def _view_controller(self):
        return None


@pytest.fixture
def fm():
    return FluorescenceMicroscope()


@pytest.fixture
def widget(fm):
    w = ObjectiveControlWidget(fm)
    yield w
    w.cleanup()
    w.close()


def _drain(timeout=5.0):
    """Spin the event loop until the queued worker signals have been delivered."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)


def _settle(widget, timeout=5.0):
    """Spin until no move is in flight."""
    deadline = time.monotonic() + timeout
    while widget._wheel_worker is not None and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.005)
    _drain(0.05)


class TestTheMoveLeavesTheGuiThread:
    def test_the_hardware_call_happens_on_a_worker(self, fm, widget, monkeypatch):
        """The whole point. `move_absolute` must not run on the thread that repaints."""
        threads = []
        real = type(fm.objective).move_absolute

        def recording(self, position):
            threads.append(threading.current_thread())
            return real(self, position)

        monkeypatch.setattr(type(fm.objective), "move_absolute", recording)

        widget._wheel_target_um = fm.objective.position * METRE_TO_MICRON + 1.0
        widget._execute_wheel_move_impl()
        _settle(widget)

        assert threads, "no move ran"
        assert all(t is not _GUI_THREAD for t in threads)

    def test_the_call_returns_before_the_move_does(self, fm, widget, monkeypatch):
        """It has to return promptly or the repaint is still blocked -- just at a
        different call site."""
        started = threading.Event()
        release = threading.Event()
        real = type(fm.objective).move_absolute

        def blocking(self, position):
            started.set()
            release.wait(5.0)
            return real(self, position)

        monkeypatch.setattr(type(fm.objective), "move_absolute", blocking)

        widget._wheel_target_um = fm.objective.position * METRE_TO_MICRON + 1.0
        t0 = time.perf_counter()
        widget._execute_wheel_move_impl()
        elapsed = time.perf_counter() - t0

        try:
            assert started.wait(5.0), "the worker never started"
            assert elapsed < 0.5, f"the GUI thread was held for {elapsed * 1e3:.0f} ms"
        finally:
            release.set()
            _settle(widget)


class TestItCoalescesAcrossTheMove:
    def test_a_nudge_mid_move_replaces_the_target_rather_than_queueing(
        self, fm, widget, monkeypatch
    ):
        """Two nudges during one move must produce two moves total, not three: the
        first, then one more to wherever the wheel ended up."""
        started = threading.Event()
        release = threading.Event()
        targets = []
        real = type(fm.objective).move_absolute

        def blocking(self, position):
            targets.append(position)
            if len(targets) == 1:
                started.set()
                release.wait(5.0)
            return real(self, position)

        monkeypatch.setattr(type(fm.objective), "move_absolute", blocking)
        base = fm.objective.position * METRE_TO_MICRON

        widget._wheel_target_um = base + 1.0
        widget._execute_wheel_move_impl()
        assert started.wait(5.0)

        # two more notches while the first move is still going
        widget._wheel_target_um = base + 2.0
        widget._execute_wheel_move_impl()
        widget._wheel_target_um = base + 3.0
        widget._execute_wheel_move_impl()

        release.set()
        _settle(widget)

        assert len(targets) == 2, f"expected 2 moves, got {len(targets)}"
        assert targets[-1] * METRE_TO_MICRON == pytest.approx(base + 3.0)

    def test_a_target_unchanged_since_the_move_started_does_not_move_again(
        self, fm, widget, monkeypatch
    ):
        """The chase must not run on every completion, or one nudge loops forever."""
        targets = []
        real = type(fm.objective).move_absolute

        def recording(self, position):
            targets.append(position)
            return real(self, position)

        monkeypatch.setattr(type(fm.objective), "move_absolute", recording)

        widget._wheel_target_um = fm.objective.position * METRE_TO_MICRON + 1.0
        widget._execute_wheel_move_impl()
        _settle(widget)

        assert len(targets) == 1

    def test_the_readout_never_jumps_backwards_during_a_burst(
        self, fm, widget, monkeypatch
    ):
        """A move that lands mid-burst has already been scrolled past. Writing its
        position to the spinbox snapped the number backwards, and it sat there for the
        whole of the next move -- several hundred ms of showing the wrong position on
        the readout you are focusing by. Invisible before the move came off the GUI
        thread, because nothing repainted in between."""
        widget.parent_widget = _Host()
        seen = []
        spin = widget.doubleSpinBox_objective_position
        real_set = spin.setValue
        monkeypatch.setattr(
            spin, "setValue", lambda v: (seen.append(round(v, 1)), real_set(v))[1]
        )

        started = threading.Event()
        release = threading.Event()
        real = type(fm.objective).move_absolute

        def blocking(self, position):
            if not started.is_set():
                started.set()
                release.wait(5.0)
            return real(self, position)

        monkeypatch.setattr(type(fm.objective), "move_absolute", blocking)

        widget._on_canvas_scroll(0, 0, +1, {"Shift"})
        widget._execute_wheel_move_impl()
        assert started.wait(5.0)
        seen.clear()  # from here on: what the user watches
        for _ in range(4):  # keep scrolling while the first move is out
            widget._on_canvas_scroll(0, 0, +1, {"Shift"})
        release.set()
        _settle(widget)

        backwards = [b for a, b in zip(seen, seen[1:]) if b < a]
        assert not backwards, f"readout went backwards to {backwards} in {seen}"

    def test_the_spinbox_ends_where_the_wheel_left_it(self, fm, widget):
        base = fm.objective.position * METRE_TO_MICRON
        widget._wheel_target_um = base + 2.0
        widget._execute_wheel_move_impl()
        _settle(widget)

        assert widget.doubleSpinBox_objective_position.value() == pytest.approx(
            base + 2.0
        )


class TestItDoesNotRaceTheOtherObjectiveActions:
    """With the move off the GUI thread everything here stays clickable through it,
    where the frozen interface used to make a second command impossible. Two commands
    for one objective is the hazard, and retracting it out from under a move in
    progress is the one that matters."""

    @pytest.fixture
    def mid_move(self, fm, widget, monkeypatch):
        """A widget with a move in flight, held there for the length of the test."""
        started = threading.Event()
        release = threading.Event()
        real = type(fm.objective).move_absolute

        def blocking(self, position):
            started.set()
            release.wait(5.0)
            return real(self, position)

        monkeypatch.setattr(type(fm.objective), "move_absolute", blocking)
        widget._wheel_target_um = fm.objective.position * METRE_TO_MICRON + 1.0
        widget._execute_wheel_move_impl()
        assert started.wait(5.0)
        _app.processEvents()
        yield widget
        release.set()
        _settle(widget)

    @pytest.mark.parametrize(
        "action",
        [
            "insert_objective",
            "retract_objective",
            "move_to_focus_position",
            # Records rather than moves -- but the number it records is a fresh device
            # read, and mid-move that is wherever the objective is passing.
            "set_focus_position",
        ],
    )
    def test_each_one_stands_down_mid_move(self, fm, mid_move, action, monkeypatch):
        commanded = []
        for name in ("move_absolute", "insert", "retract"):
            monkeypatch.setattr(
                type(fm.objective),
                name,
                lambda self, *a, _n=name: commanded.append(_n),
            )
        # Any dialog reached would mean the guard let the action through; offscreen it
        # would also hang the run rather than fail it.
        monkeypatch.setattr(
            "fibsem.ui.fm.widgets.objective_control_widget.message_box_ui",
            lambda *a, **k: pytest.fail("the action ran and opened a dialog"),
        )

        getattr(mid_move, action)()

        assert commanded == []

    def test_it_does_not_touch_the_buttons(self, fm, mid_move):
        """The regression this replaces. The buttons belong to
        `FMControlWidget._update_acquisition_button_states`, which re-derives them from
        `may_start` -- disabling them here and re-enabling on completion overwrote its
        answer, so scrolling during a live stream left Insert and Retract clickable when
        the stream forbids them. A guard on the action needs no button at all."""
        assert mid_move.pushButton_retract_objective.isEnabled()
        assert mid_move.pushButton_insert_objective.isEnabled()

    def test_the_actions_work_again_once_the_move_lands(self, fm, widget, monkeypatch):
        widget._wheel_target_um = fm.objective.position * METRE_TO_MICRON + 1.0
        widget._execute_wheel_move_impl()
        _settle(widget)

        assert widget._objective_busy_reason() is None

    def test_a_failed_move_does_not_leave_the_guard_stuck_on(
        self, fm, widget, monkeypatch
    ):
        def boom(self, position):
            raise RuntimeError("objective unreachable")

        monkeypatch.setattr(type(fm.objective), "move_absolute", boom)

        widget._wheel_target_um = fm.objective.position * METRE_TO_MICRON + 1.0
        widget._execute_wheel_move_impl()
        _settle(widget)

        assert widget._objective_busy_reason() is None


class TestNothingDrivesTheObjectiveOverSomethingElse:
    """One question -- is anything driving the objective -- asked by every way in.

    Three parties can be: another acquisition (FIB-515, and only the microscope can
    answer it -- FIB-513), a focus adjustment (FIB-625), or a threaded insert/retract
    (FIB-628). Each of the five entry points used to ask a different question, or none.
    """

    @pytest.fixture
    def widget(self, fm):
        w = ObjectiveControlWidget(fm, parent=_Host())
        yield w
        w.cleanup()
        w.close()

    def test_an_insert_or_retract_blocks_the_wheel(self, fm, widget, monkeypatch):
        """The direction that was always open: insert and retract have always been
        threaded, so a scroll during a retraction went straight through -- and retract
        is the collision-relevant motion (FIB-628)."""
        moves = []
        monkeypatch.setattr(
            type(fm.objective), "move_absolute", lambda self, p: moves.append(p)
        )
        widget._action_worker = object()  # a retract on its way

        widget._on_canvas_scroll(0, 0, +1, {"Shift"})
        widget._execute_wheel_move_impl()
        _drain(0.1)

        assert moves == []
        assert widget._objective_busy_reason() == "an objective insert or retract"

    def test_something_starting_inside_the_debounce_window_stops_the_move(
        self, fm, widget, monkeypatch
    ):
        """The scroll handlers guard when the notch arrives, but the debounce puts
        150 ms between that and the move. A retract starting in between would otherwise
        find a move already on its way."""
        moves = []
        monkeypatch.setattr(
            type(fm.objective), "move_absolute", lambda self, p: moves.append(p)
        )
        widget._on_canvas_scroll(0, 0, +1, {"Shift"})  # accepted: nothing busy yet
        widget._action_worker = object()  # ... and now a retract starts

        widget._execute_wheel_move_impl()  # what the debounce timer fires
        _drain(0.1)

        assert moves == []

    def test_an_acquisition_blocks_the_wheel(self, fm, widget, monkeypatch):
        """`is_interactive` is the microscope's answer. Asking whether *this widget* was
        busy is what let another tab's tileset through (FIB-513)."""
        moves = []
        monkeypatch.setattr(
            type(fm.objective), "move_absolute", lambda self, p: moves.append(p)
        )
        fm.set_acquiring(True, reason="a tileset")
        try:
            widget._on_canvas_scroll(0, 0, +1, {"Shift"})
            widget._execute_wheel_move_impl()
            _drain(0.1)

            assert moves == []
            assert widget._objective_busy_reason() == "a tileset"
        finally:
            fm.set_acquiring(False)

    def test_a_live_stream_does_not_block_the_wheel(self, fm, widget):
        """Deliberately not busy. Focusing while watching the stream is what this panel
        is for, and is the whole point of FIB-625."""
        fm.start_acquisition(channel_settings=None)
        try:
            assert widget._objective_busy_reason() is None
        finally:
            fm.stop_acquisition()

    def test_the_spinbox_stands_down_too(self, fm, widget, monkeypatch):
        """The last unguarded way in (FIB-515)."""
        moves = []
        monkeypatch.setattr(
            type(fm.objective), "move_absolute", lambda self, p: moves.append(p)
        )
        widget._action_worker = object()

        widget.on_objective_position_changed(
            fm.objective.position * METRE_TO_MICRON + 1.0
        )

        assert moves == []

    def test_refusing_the_spinbox_does_not_reenter(self, fm, widget, monkeypatch):
        """Putting the spinbox back raises `valueChanged`, which is wired straight back
        to this handler. Qt does not warn; it simply never returns."""
        widget.doubleSpinBox_objective_position.valueChanged.connect(
            widget.on_objective_position_changed
        )
        widget._action_worker = object()
        calls = []
        real = widget.on_objective_position_changed
        monkeypatch.setattr(
            widget,
            "on_objective_position_changed",
            lambda p: (calls.append(p), real(p))[1],
        )
        monkeypatch.setattr(type(fm.objective), "move_absolute", lambda self, p: None)

        widget.on_objective_position_changed(
            fm.objective.position * METRE_TO_MICRON + 1.0
        )

        assert len(calls) <= 1


class TestTeardown:
    def test_cleanup_cuts_a_move_in_flight_loose(self, fm, widget, monkeypatch):
        """The slots touch widgets, and an emit into one Qt has destroyed on the C++
        side is a segfault rather than an exception (FIB-550). The move itself is left
        to finish -- waiting for it is the blocking this change removes."""
        started = threading.Event()
        release = threading.Event()
        real = type(fm.objective).move_absolute

        def blocking(self, position):
            started.set()
            release.wait(5.0)
            return real(self, position)

        monkeypatch.setattr(type(fm.objective), "move_absolute", blocking)

        landed = []
        widget._on_wheel_move_finished = lambda *a: landed.append(a)

        widget._wheel_target_um = fm.objective.position * METRE_TO_MICRON + 1.0
        widget._execute_wheel_move_impl()
        assert started.wait(5.0)

        widget.cleanup()
        release.set()
        _drain(1.0)

        assert widget._wheel_worker is None
        assert landed == [], "the worker called back into a torn-down widget"

    def test_cleanup_is_safe_with_no_move_in_flight(self, widget):
        widget.cleanup()
        widget.cleanup()  # idempotent, like the rest of the teardown
