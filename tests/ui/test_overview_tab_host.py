"""The Overview tab's host half: turning a point on the canvas into a lamella.

`FibsemOverviewWidget` emits requests and knows nothing about experiments;
`AutoLamellaOverviewTab` is what gives them meaning. The split is the point, so what is
tested here is the *meaning* — which pose a marker uses, what moves with a move, what a
defect does to the display — rather than the canvas, which has its own tests.

The whole window cannot be constructed here: it still builds the napari minimap, and a
`napari.Viewer` segfaults under the offscreen platform (measured, exit 139). So the tab
is exercised against a real microscope and a real experiment with a stub in place of the
window, and the window's own wiring is checked structurally in
`test_overview_tab_wiring.py`.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_overview_tab_host.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

# CI installs `.[test]`, not `.[ui]`, so PyQt5 is absent there.
pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.applications.autolamella.structures import (  # noqa: E402
    DefectType,
    Experiment,
)
from fibsem.applications.autolamella.ui.autolamella_overview_tab import (  # noqa: E402
    AutoLamellaOverviewTab,
)
from fibsem.structures import FibsemStagePosition  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv)


@pytest.fixture(scope="module")
def microscope():
    scope, _ = utils.setup_session(manufacturer="Demo")
    return scope


class _StubWindow:
    """Everything the tab reaches for on its host, and nothing else.

    Deliberately small: if this grows, the tab has started depending on the window
    rather than on an experiment, which is the coupling the split exists to prevent.
    """

    def __init__(self, microscope, experiment):
        self.microscope = microscope
        self.experiment = experiment
        self.added = []

    def add_new_lamella(self, stage_position=None, **kwargs):
        """A real `Lamella` with real poses, so the property setters behave as they do
        live -- `stage_position` is a property over the milling pose, not a field."""
        from fibsem.applications.autolamella.poses import (
            MILLING_ORIENTATION,
            build_lamella_poses,
        )
        from fibsem.applications.autolamella.structures import Lamella

        number = len(self.added) + 1
        name = f"Lamella-{number:02d}"
        if stage_position is None:
            beam = self.microscope.get_orientation(MILLING_ORIENTATION)
            stage_position = FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=beam.r, t=beam.t)
        poses = build_lamella_poses(self.microscope, stage_position)
        lamella = Lamella(
            petname=name, path=os.path.join(str(self.experiment.path), name), number=number
        )
        lamella.milling_pose = poses.milling
        lamella.fluorescence_pose = poses.fluorescence
        self.added.append(lamella)
        self.experiment.positions.append(lamella)
        return lamella


@pytest.fixture
def tab(microscope, tmp_path):
    # `Experiment` puts itself in a subdirectory of `path` and does not create it;
    # `save()` writes into it, so the handlers that persist would raise without this.
    experiment = Experiment(path=tmp_path, name="overview-tab-test")
    os.makedirs(str(experiment.path), exist_ok=True)
    window = _StubWindow(microscope, experiment)
    tab = AutoLamellaOverviewTab(window)
    tab.refresh_microscope()
    assert tab.is_available, "the tab did not build its widget"
    yield tab
    tab._drop_overview()


def _at(base, dx=0.0, dy=0.0):
    return FibsemStagePosition(
        x=base.x + dx, y=base.y + dy, z=base.z, r=base.r, t=base.t
    )


def _lamella(tab, microscope, dx=0.0, dy=0.0):
    """A lamella added the way the canvas adds one."""
    return tab.autolamella_ui.add_new_lamella(
        stage_position=_at(microscope.get_stage_position(), dx, dy)
    )


class TestItMarksTheBeamSidePose:
    def test_a_lamella_is_marked_at_its_milling_pose(self, tab, microscope):
        """The opposite choice from the fluorescence tab, and the one that matters:
        `lamella.stage_position` *is* the milling pose.

        Asserted on the **tilt**, not on x/y, and that is the whole point of this test.
        On a compustage the two poses share x, y, z and r and differ only in tilt
        (measured: -23 degrees against -180). So an assertion on position cannot tell
        them apart here at all -- it passes just as happily against the fluorescence
        pose, which is exactly what a mutation proved before this was rewritten.

        The tilt is not cosmetic: it is the base pose the projection works from, and on
        an offset mount the two poses do not share x/y either (FIB-93). Getting this
        wrong is invisible on this simulator and wrong on a real instrument.
        """
        lamella = _lamella(tab, microscope, dx=120e-6)
        tab.refresh_positions()

        marked = tab.overview._positions
        assert [p.name for p in marked] == [lamella.name]
        assert marked[0].x == pytest.approx(lamella.stage_position.x)
        assert marked[0].y == pytest.approx(lamella.stage_position.y)

        milling_tilt = lamella.milling_pose.stage_position.t
        fluorescence_tilt = lamella.fluorescence_pose.stage_position.t
        assert milling_tilt != pytest.approx(fluorescence_tilt), (
            "the two poses are indistinguishable on this microscope, so this test "
            "cannot check which one was used"
        )
        assert marked[0].t == pytest.approx(milling_tilt), (
            "the marker carries the fluorescence pose, not the milling one"
        )

    def test_a_lamella_with_no_position_is_not_marked(self, tab, microscope):
        """It cannot be placed, and placing it at the origin would look identical to a
        lamella that really is there."""
        lamella = _lamella(tab, microscope)
        lamella.stage_position = None
        tab.refresh_positions()
        assert tab.overview._positions == []
        # Still in the list, though: that is where you find out it exists.
        assert tab.lamella_list._list.count() == 1


class TestDefects:
    def test_a_defective_lamella_is_flagged_on_the_canvas(self, tab, microscope):
        """The tab this replaces coloured failed and rework lamellae differently on the
        overview, and that is a real signal: it is the one you should not just
        re-target."""
        ok = _lamella(tab, microscope, dx=50e-6)
        bad = _lamella(tab, microscope, dx=-50e-6)
        bad.defect.state = DefectType.FAILURE
        tab.refresh_positions()

        assert tab.overview._flagged == {bad.name}
        assert ok.name not in tab.overview._flagged

    def test_rework_is_flagged_as_well_as_failure(self, tab, microscope):
        lamella = _lamella(tab, microscope)
        lamella.defect.state = DefectType.REWORK
        tab.refresh_positions()
        assert tab.overview._flagged == {lamella.name}

    def test_setting_a_defect_is_saved_and_redrawn(self, tab, microscope):
        """The row writes `lamella.defect` and emits; nothing was listening, so the
        change lived in memory and was gone on reload (FIB-564). Redrawn as well as
        saved, because the marker's colour is one of the things it decides."""
        lamella = _lamella(tab, microscope)
        tab.refresh_positions()
        assert tab.overview._flagged == set()

        lamella.defect.state = DefectType.FAILURE
        tab._on_defect_changed(lamella)

        assert tab.overview._flagged == {lamella.name}
        assert os.path.exists(os.path.join(str(tab.experiment.path), "experiment.yaml"))


class TestMovingAPosition:
    def test_moving_a_lamella_takes_its_fluorescence_pose_with_it(
        self, tab, microscope
    ):
        """The two poses describe one piece of sample from two sides. Left behind, the
        fluorescence one goes on naming where this lamella *used to be* — and nothing
        about a stale pose looks wrong."""
        lamella = _lamella(tab, microscope)
        lamella.update_milling_angle(microscope)
        from fibsem.applications.autolamella.poses import sync_fluorescence_pose

        sync_fluorescence_pose(microscope, lamella)
        before = lamella.fluorescence_pose
        before_position = (
            None if before is None else (before.stage_position.x, before.stage_position.y)
        )

        target = _at(microscope.get_stage_position(), dx=250e-6, dy=-125e-6)
        tab._on_move_requested(lamella.name, target)

        assert lamella.stage_position.x == pytest.approx(target.x)
        assert lamella.stage_position.y == pytest.approx(target.y)
        if before_position is not None:
            after = lamella.fluorescence_pose.stage_position
            assert (after.x, after.y) != before_position, (
                "the fluorescence pose was left behind at the old location"
            )

    def test_moving_an_unknown_name_does_nothing(self, tab, microscope):
        _lamella(tab, microscope)
        tab._on_move_requested("not-a-lamella", _at(microscope.get_stage_position()))
        # No exception, and nothing moved.
        assert tab.experiment.positions[0].stage_position is not None


class TestSelectionSync:
    def test_clicking_a_marker_announces_the_lamella_not_the_name(
        self, tab, microscope
    ):
        """The canvas knows the name it drew; only the host can turn that into a
        lamella, which is what the rest of the window needs."""
        lamella = _lamella(tab, microscope)
        tab.refresh_positions()

        announced = []
        tab.lamella_selected.connect(announced.append)
        tab._on_marker_clicked(lamella.name)

        assert announced == [lamella]

    def test_a_selection_this_tab_raised_does_not_come_back_around(
        self, tab, microscope
    ):
        """The window answers a selection by syncing every list it knows about, this
        one included. Re-selecting the row under a click that is still happening moves
        the selection out from under the user."""
        lamella = _lamella(tab, microscope)
        tab.refresh_positions()

        announced = []
        tab.lamella_selected.connect(announced.append)
        # The window's reply, arriving while this tab is mid-announcement.
        tab.lamella_selected.connect(lambda lam: tab.set_selected(lam))
        tab._on_list_selection(lamella)

        assert len(announced) == 1, "the selection echoed"


class TestLifecycle:
    def test_the_widget_is_rebuilt_when_the_microscope_changes(
        self, tab, microscope, tmp_path
    ):
        """A reconnection hands out a new object, and the old widget would go on reading
        geometry from an instrument nobody is driving."""
        first = tab.overview
        tab.refresh_microscope()
        assert tab.overview is first, "rebuilt for the same microscope"

        other, _ = utils.setup_session(manufacturer="Demo")
        tab.autolamella_ui.microscope = other
        tab.refresh_microscope()
        assert tab.overview is not first, "kept a widget bound to the old microscope"
        assert tab.overview is not None

    def test_the_list_survives_the_widget_being_rebuilt(self, tab, microscope):
        """`add_settings_section` reparents the list into the overview's column, so Qt
        destroying the overview would take the list with it — and the next build would
        hand a dead C++ object back to `add_settings_section`."""
        listed = tab.lamella_list
        other, _ = utils.setup_session(manufacturer="Demo")
        tab.autolamella_ui.microscope = other
        tab.refresh_microscope()

        assert tab.lamella_list is listed
        listed.set_lamella([])  # must not raise on a deleted C++ object

    def test_dropping_the_widget_releases_its_subscriptions(self, tab, microscope):
        signals = (
            microscope.tiled_acquisition_signal,
            microscope.stage_position_changed,
        )
        before = [len(s) for s in signals]
        tab._drop_overview()
        after = [len(s) for s in signals]
        assert all(a < b for a, b in zip(after, before)), (
            f"a subscription survived the drop: {before} -> {after}"
        )
        assert not tab.is_available
