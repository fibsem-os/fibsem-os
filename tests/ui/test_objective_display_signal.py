"""The objective displays follow the signal instead of the device (FIB-576).

FIB-534 added `ObjectiveLens.position_changed` and moved one consumer -- the overview
info bar -- onto it. Two were left, deliberately, so the signal could run on a
microscope before three widgets depended on it.

The two are not the same problem, and the issue's list conflates them:

* **The quad-view info bar was polling.** `MicroscopeViewController.update_info` read
  `fm.objective.position` whenever the caller passed none, and
  `FibsemMovementWidget._update_position_readout` calls it on every stage-position
  update without passing one. On a TFS system that read takes the shared imaging
  channel, so every stage move pointed the microscope at the FM and back to write a
  number into a label. That is FIB-517's mechanism in a text field.

* **The control widget was stale, not polling.** Its reads are guards, about-to-commit
  reads, or a user pressing "Refresh Position" -- none of them on a poll. What it
  lacked was any way to hear about moves it did not make, so an autofocus sweep or a
  z-stack left the spinbox showing the position from before it.

The reads that stay are pinned in `tests/fm/test_objective_signal.py`: guards read the
device every time, because a stale "Retracted" moves the stage with the objective in the
chamber.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from fibsem.constants import METRE_TO_MICRON
from fibsem.fm.microscope import FluorescenceMicroscope
from fibsem.ui.fm.widgets.objective_control_widget import ObjectiveControlWidget

_app = QApplication.instance() or QApplication(sys.argv)


# Stay inside the widget's 1 mm "large movement" threshold. Above it,
# `on_objective_position_changed` opens a modal confirmation -- which, offscreen, waits
# for a click that never comes and hangs the run rather than failing it. The simulated
# objective starts at -10 mm, so an absolute value like 42 um is a large move by
# accident and looks like an infinite loop when it stalls.
_SMALL_STEP_UM = 0.5


def _nearby_um(fm) -> float:
    """A target close enough to the current position to skip the confirmation."""
    return fm.objective.position * METRE_TO_MICRON + _SMALL_STEP_UM


@pytest.fixture
def fm():
    return FluorescenceMicroscope()


@pytest.fixture
def widget(fm):
    w = ObjectiveControlWidget(fm)
    yield w
    w.close()


class TestItFollowsMovesItDidNotMake:
    def test_an_external_move_reaches_the_spinbox(self, fm, widget):
        """The gap. An autofocus sweep or a z-stack moves the objective, and before this
        the widget went on showing whatever it last set itself."""
        fm.objective.move_absolute(150e-6)

        assert widget.doubleSpinBox_objective_position.value() == pytest.approx(
            150e-6 * METRE_TO_MICRON
        )

    def test_it_subscribes_exactly_once(self, fm, widget):
        assert len(fm.objective.position_changed) == 1


class TestTheLoopThatWouldNotWarn:
    """The piece where a mistake is an infinite loop rather than a stale label.

        spinBox.valueChanged -> move_absolute -> position_changed -> setValue
                             -> valueChanged -> ...

    Qt does not warn about that; it simply never returns. `setValue` therefore has to
    happen with the spinbox's signals blocked, which is what
    `update_objective_position_labels` already did for its own refreshes and is why the
    new slot routes through it rather than touching the spinbox directly.
    """

    def test_driving_the_spinbox_moves_the_objective_exactly_once(self, fm, widget):
        moves = []
        fm.objective.position_changed.connect(
            lambda position, state: moves.append(position)
        )
        target_um = _nearby_um(fm)

        widget.doubleSpinBox_objective_position.setValue(target_um)

        assert len(moves) == 1, (
            f"the spinbox produced {len(moves)} moves for one edit -- the signal is "
            f"feeding back into the control that raised it"
        )
        assert moves[0] == pytest.approx(target_um / METRE_TO_MICRON)

    def test_the_echo_does_not_move_the_objective_again(self, fm, widget):
        """The same thing from the other side: the position the widget is told about
        must not be re-issued as a move."""
        widget.doubleSpinBox_objective_position.setValue(_nearby_um(fm))
        moves = []
        fm.objective.position_changed.connect(
            lambda position, state: moves.append(position)
        )

        echoed = fm.objective.position + 5e-6
        widget._on_objective_moved(echoed, "Inserted")

        assert moves == [], "handling a move emitted another one"
        assert widget.doubleSpinBox_objective_position.value() == pytest.approx(
            echoed * METRE_TO_MICRON
        )


class TestTeardown:
    def test_closing_drops_the_subscription(self, fm):
        """This is the widget's only psygnal, and the reason it needed a `closeEvent` at
        all. The signal belongs to the microscope and outlives the widget; an emit into
        one Qt has already torn down is a segfault, not an exception (FIB-550)."""
        w = ObjectiveControlWidget(fm)
        assert len(fm.objective.position_changed) == 1

        w.close()

        assert len(fm.objective.position_changed) == 0, (
            "the subscription survived close(), so a later move reaches a widget whose "
            "C++ half is gone"
        )

    def test_a_move_after_closing_does_not_reach_the_widget(self, fm):
        w = ObjectiveControlWidget(fm)
        before = w.doubleSpinBox_objective_position.value()
        w.close()

        fm.objective.move_absolute(250e-6)

        assert w.doubleSpinBox_objective_position.value() == before


class TestTheInfoBarStoppedAskingTheDevice:
    """`update_info` runs on every stage-position update, and took the shared imaging
    channel each time to read a number that had not changed."""

    def test_it_does_not_read_the_objective_when_none_is_supplied(self, monkeypatch):
        from fibsem.ui.widgets.canvas.quad_view import MicroscopeViewController

        controller = MicroscopeViewController()
        microscope = _StubMicroscope()
        reads = []
        type(microscope.fm.objective).position = property(
            lambda self: (reads.append(1), 1e-6)[1]
        )

        controller.update_info(microscope, objective_position=200e-6)
        controller.update_info(microscope)  # a stage update, carrying no objective

        assert reads == [], (
            "the info bar read the objective off the device -- on a TFS system that "
            "takes the shared imaging channel, once per stage update (FIB-517)"
        )

    def test_it_keeps_the_last_position_it_was_told(self, monkeypatch):
        from fibsem.ui.widgets.canvas.quad_view import MicroscopeViewController

        controller = MicroscopeViewController()
        microscope = _StubMicroscope()

        controller.update_info(microscope, objective_position=200e-6)

        assert controller._objective_position == pytest.approx(200e-6)


class TestTheLabelStillExistsBeforeTheFirstMove:
    """Remembering instead of reading loses the starting value, and with it the whole
    field: nothing pushes a position at construction, so the readout the FM canvas used
    to have on the first stage update was simply gone until someone moved the
    objective."""

    def test_the_first_stage_update_seeds_the_readout(self):
        from fibsem.ui.widgets.canvas.quad_view import MicroscopeViewController

        controller = MicroscopeViewController()
        microscope = _StubMicroscope()
        type(microscope.fm.objective).position = property(lambda self: 200e-6)

        controller.update_info(microscope)  # a stage update, carrying no objective

        info = dict(controller._states[controller._widget.fm_canvas].info)
        assert info.get("objective") == "OBJECTIVE: 200.0 µm"

    def test_the_seeding_read_happens_once_and_only_once(self):
        """The point of the exercise. One read to have something to show is not the
        per-stage-update poll this replaced."""
        from fibsem.ui.widgets.canvas.quad_view import MicroscopeViewController

        controller = MicroscopeViewController()
        microscope = _StubMicroscope()
        reads = []
        type(microscope.fm.objective).position = property(
            lambda self: (reads.append(1), 200e-6)[1]
        )

        for _ in range(5):
            controller.update_info(microscope)

        assert reads == [1]

    def test_a_failing_read_is_not_retried_on_every_stage_update(self):
        """Otherwise the poll comes back on exactly the system that can least afford
        it -- one that cannot read its objective."""
        from fibsem.ui.widgets.canvas.quad_view import MicroscopeViewController

        controller = MicroscopeViewController()
        microscope = _StubMicroscope()
        reads = []

        def _boom(self):
            reads.append(1)
            raise RuntimeError("objective unreachable")

        type(microscope.fm.objective).position = property(_boom)

        for _ in range(5):
            controller.update_info(microscope)  # swallowed + logged, as before

        assert reads == [1]

    def test_being_told_a_position_first_means_no_read_at_all(self):
        from fibsem.ui.widgets.canvas.quad_view import MicroscopeViewController

        controller = MicroscopeViewController()
        microscope = _StubMicroscope()
        reads = []
        type(microscope.fm.objective).position = property(
            lambda self: (reads.append(1), 200e-6)[1]
        )

        controller.update_info(microscope, objective_position=150e-6)
        controller.update_info(microscope)

        assert reads == []
        assert controller._objective_position == pytest.approx(150e-6)


class _StubObjective:
    pass


class _StubFM:
    def __init__(self):
        # A fresh type per instance. The tests install `position` as a property on the
        # objective's *class*, so a shared one would carry a previous test's read
        # counter -- or its raising getter -- into the next.
        self.objective = type("_StubObjective", (_StubObjective,), {})()


class _StubMicroscope:
    """Only what `update_info` touches before reaching the objective branch."""

    def __init__(self):
        self.fm = _StubFM()
        self._stage_position = _StubStagePosition()
        self.current_grid = "grid-1"

    def get_stage_orientation(self, stage_position=None):
        return "MILLING"

    def get_current_milling_angle(self, stage_position=None):
        return 15.0


class _StubStagePosition:
    pretty_string = "X:0.00mm, Y:0.00mm"
