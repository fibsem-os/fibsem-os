"""The overview tab, from a stage that is not posed for fluorescence (FIB-436).

An overview acquired at the wrong orientation is not merely unhelpful: the canvas frame
is built from the origin's rotation and tilt, so the tiles would be placed against a
projection that does not describe them. The tab used to let you configure and start one
from anywhere.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox

from fibsem.ui.fm.widgets import fm_overview_widget as module
from fibsem.ui.fm.widgets.fm_overview_widget import FMOverviewWidget


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def widget(qapp):
    from fibsem.ui.fm.overview_app import build_microscope

    return FMOverviewWidget(build_microscope())


# Compustage tilts the whole orientation question: -180° is where the FM looks at the
# sample, 0° is where the electron beam does. Posed directly rather than through
# `move_to_microscope("FIBSEM")`, whose compustage path still goes via the deprecated
# `move_flat_to_beam` and so cannot be called from a suite that errors on warnings.
_TILT_DEG = {"FM": -180.0, "SEM": 0.0, "MILLING": 20.0}


def _pose(widget, orientation: str) -> None:
    """Put the stage at *orientation*, then tell the widget as a real move would."""
    import numpy as np

    from fibsem.structures import FibsemStagePosition

    widget.microscope.move_stage_absolute(
        FibsemStagePosition(
            x=0.0, y=0.0, z=0.0, r=0.0, t=np.deg2rad(_TILT_DEG[orientation])
        )
    )
    assert widget.microscope.get_stage_orientation() == orientation
    widget._on_stage_moved(widget.microscope.get_stage_position())


@pytest.fixture()
def accept_dialog(monkeypatch):
    """Confirm the run, and count how many times the dialog was reached.

    Patched in the refusal tests too: it is modal, so a guard that stopped working would
    hang the suite rather than fail it.
    """
    opened = []

    class _Dialog:
        def __init__(self, **kwargs):
            opened.append(kwargs)

        def exec_(self):
            return QDialog.Accepted

    monkeypatch.setattr(module, "FMOverviewConfirmationDialog", _Dialog)
    return opened


@pytest.fixture()
def no_worker(monkeypatch):
    """Stop anything that would start a thread, and record what it was handed."""
    started = []

    class _Worker:
        def __init__(self, func, *args, **kwargs):
            started.append((func, args))

        def start(self):
            pass

        def is_alive(self):
            return True

    monkeypatch.setattr(module, "FunctionWorker", _Worker)
    return started


class TestTheGate:
    def test_the_simulator_starts_posed_for_fluorescence(self, widget):
        """The premise of every other test here, and worth asserting rather than
        assuming: a gate that was open because nothing could close it would look
        exactly like a gate that works."""
        assert widget.microscope.get_stage_orientation() == "FM"
        assert widget.at_fm_orientation() is True

    def test_the_wrong_pose_closes_it(self, widget):
        _pose(widget, "SEM")

        assert widget.at_fm_orientation() is False

    def test_acquire_is_disabled_from_the_wrong_pose(self, widget, qapp):
        assert widget.button_acquire.isEnabled() is True

        _pose(widget, "SEM")
        qapp.processEvents()

        assert widget.button_acquire.isEnabled() is False

    def test_acquire_comes_back_on_the_way_home(self, widget, qapp):
        _pose(widget, "SEM")
        qapp.processEvents()
        assert widget.button_acquire.isEnabled() is False, "nothing to come back from"

        _pose(widget, "FM")
        qapp.processEvents()

        assert widget.button_acquire.isEnabled() is True

    def test_display_and_navigation_are_left_alone(self, widget):
        """Looking at an already-acquired overview from the wrong pose is harmless, and
        the settings stay editable so a run can be set up while the stage is moving."""
        _pose(widget, "SEM")

        assert widget.settings_widget.isEnabled() is True
        assert widget.channel_widget.isEnabled() is True

    def test_the_stage_can_still_be_driven_from_the_wrong_pose(self, widget):
        """What the deleted `has_valid_orientation` check claimed to stop.

        The whole scene is re-placed through whatever pose the stage is in, so a click
        still lands on the feature it points at.
        """
        _pose(widget, "SEM")

        assert widget._may_move() is True

    def test_marking_is_still_refused_from_the_wrong_pose(self, widget):
        """Stricter than moving, and unchanged: a marked position becomes a lamella's
        fluorescence pose, and one carrying SEM rotation and tilt is not one."""
        _pose(widget, "SEM")

        assert widget._position_menu(0.0, 0.0) is None


class TestTheRefusal:
    def test_acquire_refuses_from_the_wrong_pose(self, widget, accept_dialog, no_worker):
        """The button is not the guard: a host can call this, and the stage can move
        between the click and here."""
        _pose(widget, "SEM")

        widget.acquire()

        assert accept_dialog == [], "asked the user to confirm a run it then refused"

    def test_acquire_runs_once_the_stage_is_posed(self, widget, accept_dialog, no_worker):
        _pose(widget, "SEM")
        _pose(widget, "FM")

        widget.acquire()

        assert len(accept_dialog) == 1


class TestTheBanner:
    def test_it_is_hidden_when_the_pose_is_right(self, widget):
        assert widget.orientation_banner.isVisibleTo(widget) is False

    def test_it_appears_and_names_both_orientations(self, widget):
        _pose(widget, "SEM")

        assert widget.orientation_banner.isVisibleTo(widget) is True
        notice = widget.orientation_notice.text()
        assert "SEM" in notice, notice
        assert "FM" in notice, notice

    def test_the_button_names_where_it_goes(self, widget):
        """From `default_orientation`, which the control widget can change at runtime --
        a button naming one orientation while going to another is worse than none."""
        _pose(widget, "SEM")

        assert widget.button_move_to_fm.text() == "Move to FM"

        widget.fm.default_orientation = "SEM-ish"
        widget._refresh_orientation_banner()

        assert widget.button_move_to_fm.text() == "Move to SEM-ish"

    def test_it_goes_away_again(self, widget):
        _pose(widget, "SEM")
        assert widget.orientation_banner.isVisibleTo(widget) is True

        _pose(widget, "FM")

        assert widget.orientation_banner.isVisibleTo(widget) is False


class TestTheMoveAction:
    def test_it_asks_before_moving(self, widget, no_worker, monkeypatch):
        """A real stage move, so it gets the same confirmation as the others here."""
        _pose(widget, "SEM")
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)

        widget.move_to_fm_orientation()

        assert no_worker == [], "moved the stage without asking"

    def test_it_moves_when_confirmed(self, widget, no_worker, monkeypatch):
        _pose(widget, "SEM")
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)

        widget.move_to_fm_orientation()

        assert len(no_worker) == 1
        func, args = no_worker[0]
        assert func == widget._move_to_orientation_worker
        assert args == ("FM",)

    def test_the_worker_drives_the_stage(self, widget):
        _pose(widget, "SEM")

        widget._move_to_orientation_worker("FM")

        assert widget.microscope.get_stage_orientation() == "FM"

    def test_a_failed_move_is_logged_not_raised(self, widget, monkeypatch, caplog):
        """It runs on a worker thread, where an exception has nowhere to go."""
        def _boom(target):
            raise RuntimeError("stage refused")

        monkeypatch.setattr(widget.microscope, "move_to_microscope", _boom)

        widget._move_to_orientation_worker("FM")

        assert "stage refused" in caplog.text

    def test_it_refuses_during_a_run(self, widget, no_worker, monkeypatch):
        """The stage is walking the grid; taking it somewhere else mid-tileset is not a
        thing a confirmation dialog should be able to authorise."""
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
        widget._set_running(True)

        widget.move_to_fm_orientation()

        assert no_worker == []
