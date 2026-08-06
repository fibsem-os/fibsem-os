"""The overview tab, from a stage the FM cannot work at (FIB-436).

An overview acquired at the wrong orientation is not merely unhelpful: the canvas frame
is built from the origin's rotation and tilt, so the tiles would be placed against a
projection that does not describe them. The tab used to let you configure and start one
from anywhere, and the one orientation check it did have -- before a click-to-move --
asked `fm.has_valid_orientation()`, which the `ALLOW_UNKNOWN_ORIENTATIONS` escape hatch
answers yes to unconditionally.

Two different questions are asked, and which one goes where is most of what is tested
here: `acquisition_orientations` (where the objective can image the sample) gates
acquiring and driving the stage; `default_orientation` (the single pose a fluorescence
position is written down as) gates marking. Neither is `valid_orientations`, which is
the looser "FM control is allowed here" and still includes the beam poses.
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
# `move_to_microscope`, which only knows FIBSEM and FM -- MILLING and NONE have to be
# set here anyway, so all four come from one place.
#
# "NONE" is what `get_stage_orientation` returns for a pose it cannot name at all.
_TILT_DEG = {"FM": -180.0, "SEM": 0.0, "MILLING": 20.0, "NONE": -90.0}


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
        assert widget.at_acquisition_orientation() is True

    def test_the_wrong_pose_closes_it(self, widget):
        _pose(widget, "NONE")

        assert widget.at_acquisition_orientation() is False

    def test_acquire_is_disabled_from_the_wrong_pose(self, widget, qapp):
        assert widget.button_acquire.isEnabled() is True

        _pose(widget, "NONE")
        qapp.processEvents()

        assert widget.button_acquire.isEnabled() is False

    def test_acquire_comes_back_on_the_way_home(self, widget, qapp):
        _pose(widget, "NONE")
        qapp.processEvents()
        assert widget.button_acquire.isEnabled() is False, "nothing to come back from"

        _pose(widget, "FM")
        qapp.processEvents()

        assert widget.button_acquire.isEnabled() is True

    def test_display_and_navigation_are_left_alone(self, widget):
        """Looking at an already-acquired overview from the wrong pose is harmless, and
        the settings stay editable so a run can be set up while the stage is moving."""
        _pose(widget, "NONE")

        assert widget.settings_widget.isEnabled() is True
        assert widget.channel_widget.isEnabled() is True

    @pytest.mark.parametrize("orientation", ["SEM", "MILLING", "NONE"])
    def test_every_beam_pose_closes_it(self, widget, qapp, orientation):
        """MILLING especially: it is in `valid_orientations`, so a gate asking that
        question let a full tileset be started from it -- and on every mount we support
        the sample is flipped or translated away from the objective there, so the tiles
        would be of nothing."""
        _pose(widget, orientation)
        qapp.processEvents()

        assert widget.at_acquisition_orientation() is False
        assert widget.button_acquire.isEnabled() is False
        assert widget._may_move() is False

    def test_the_list_is_what_decides_not_a_hard_coded_pose(self, widget, qapp):
        """A system whose objective sees the sample from more than one pose widens
        `acquisition_orientations`; nothing here compares against `default_orientation`."""
        widget.fm.acquisition_orientations = ["FM", "SEM"]
        _pose(widget, "SEM")
        qapp.processEvents()

        assert widget.at_acquisition_orientation() is True
        assert widget.button_acquire.isEnabled() is True
        assert widget._may_move() is True

    def test_marking_stays_on_the_fluorescence_pose_alone(self, widget):
        """The distinction the gate rests on. Stricter than acquiring or moving: a
        marked position becomes a lamella's *fluorescence* pose, and one carrying SEM
        rotation and tilt is not one, however right it looked on screen -- so widening
        `acquisition_orientations` must not widen this."""
        widget.fm.acquisition_orientations = ["FM", "SEM"]
        _pose(widget, "SEM")

        assert widget.at_acquisition_orientation() is True
        assert widget.at_fluorescence_pose() is False
        assert widget._may_move() is True, "the looser check should still pass here"
        assert widget._position_menu(0.0, 0.0) is None


class TestOffsetMounts:
    """Where the question cannot be asked, it must not be answered "no" (FIB-93).

    `_update_orientations` gives a non-compustage system an FM orientation copied from
    its FIB one, so `get_stage_orientation` at the fluorescence position comes back as a
    beam orientation and never as "FM". A gate that refused on that would leave the tab
    permanently dead on every METEOR rather than guarding anything.
    """

    @pytest.fixture()
    def offset_widget(self, qapp):
        from fibsem import utils
        from fibsem.fm.microscope import FluorescenceMicroscope

        microscope, _ = utils.setup_session(manufacturer="Demo")
        microscope.stage_is_compustage = False
        microscope._update_orientations()
        if microscope.fm is None:
            microscope.fm = FluorescenceMicroscope(parent=microscope)
        return FMOverviewWidget(microscope)

    def test_the_fluorescence_pose_does_not_report_itself_as_fm(self, offset_widget):
        """The premise. If this ever starts returning "FM", the exemption below can go."""
        microscope = offset_widget.microscope
        microscope.move_stage_absolute(microscope.get_orientation("FM"))

        assert microscope.get_stage_orientation() != "FM"

    def test_acquisition_is_not_gated_there(self, offset_widget, qapp):
        microscope = offset_widget.microscope
        microscope.move_stage_absolute(microscope.get_orientation("FM"))
        offset_widget._on_stage_moved(microscope.get_stage_position())
        qapp.processEvents()

        assert offset_widget.at_acquisition_orientation() is True
        assert offset_widget.button_acquire.isEnabled() is True
        assert offset_widget._may_move() is True
        assert offset_widget.orientation_banner.isVisibleTo(offset_widget) is False


class TestTheRefusal:
    def test_acquire_refuses_from_the_wrong_pose(self, widget, accept_dialog, no_worker):
        """The button is not the guard: a host can call this, and the stage can move
        between the click and here."""
        _pose(widget, "NONE")

        widget.acquire()

        assert accept_dialog == [], "asked the user to confirm a run it then refused"

    def test_acquire_runs_once_the_stage_is_posed(self, widget, accept_dialog, no_worker):
        _pose(widget, "NONE")
        _pose(widget, "FM")

        widget.acquire()

        assert len(accept_dialog) == 1


class TestTheBanner:
    def test_it_is_hidden_when_the_pose_is_right(self, widget):
        assert widget.orientation_banner.isVisibleTo(widget) is False

    def test_it_names_where_the_stage_is_and_where_it_needs_to_be(self, widget):
        _pose(widget, "SEM")

        assert widget.orientation_banner.isVisibleTo(widget) is True
        assert widget.orientation_notice.text() == (
            "Stage is at the SEM orientation — an overview needs to be at the FM "
            "orientation."
        )

    def test_an_unrecognised_pose_is_not_called_the_none_orientation(self, widget):
        """`get_stage_orientation` returns "NONE" for a pose it cannot name, and "the
        NONE orientation" is not a thing to say to anyone."""
        _pose(widget, "NONE")

        assert widget.orientation_notice.text() == (
            "Stage is at a position that is not a recognised orientation — an overview "
            "needs to be at the FM orientation."
        )

    @pytest.mark.parametrize(
        "allowed,expected",
        [
            (["FM", "SEM"], "the FM or SEM orientation"),
            (["FM", "SEM", "MILLING"], "the FM, SEM or MILLING orientation"),
        ],
    )
    def test_it_names_every_orientation_that_would_do(self, widget, allowed, expected):
        """Not only the one the button goes to: on a system configured for more than one
        the stage may already be a shorter move from a different one."""
        widget.fm.acquisition_orientations = allowed
        _pose(widget, "NONE")

        assert widget.orientation_notice.text().endswith(f"needs to be at {expected}.")

    def test_it_stays_hidden_at_a_valid_but_non_fluorescence_pose(self, widget):
        widget.fm.acquisition_orientations = ["FM", "SEM"]
        _pose(widget, "SEM")

        assert widget.orientation_banner.isVisibleTo(widget) is False

    def test_the_button_names_where_it_goes(self, widget):
        """From `default_orientation`, which the control widget can change at runtime --
        a button naming one orientation while going to another is worse than none."""
        _pose(widget, "NONE")

        assert widget.button_move_to_fm.text() == "Move to FM"

        widget.fm.default_orientation = "SEM-ish"
        widget._refresh_orientation_banner()

        assert widget.button_move_to_fm.text() == "Move to SEM-ish"

    def test_it_goes_away_again(self, widget):
        _pose(widget, "NONE")
        assert widget.orientation_banner.isVisibleTo(widget) is True

        _pose(widget, "FM")

        assert widget.orientation_banner.isVisibleTo(widget) is False


class TestTheMoveAction:
    def test_it_asks_before_moving(self, widget, no_worker, monkeypatch):
        """A real stage move, so it gets the same confirmation as the others here."""
        _pose(widget, "NONE")
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)

        widget.move_to_fm_orientation()

        assert no_worker == [], "moved the stage without asking"

    def test_it_moves_when_confirmed(self, widget, no_worker, monkeypatch):
        _pose(widget, "NONE")
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)

        widget.move_to_fm_orientation()

        assert len(no_worker) == 1
        func, args = no_worker[0]
        assert func == widget._move_to_orientation_worker
        assert args == ("FM",)

    def test_the_worker_drives_the_stage(self, widget):
        _pose(widget, "NONE")

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
