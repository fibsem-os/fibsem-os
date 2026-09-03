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
    """A simulated Arctis, because these tabs are half about fluorescence.

    A plain Demo session is not a compustage, and `DemoMicroscope` only builds a
    `SimulatedFluorescenceMicroscope` when `sim.is_compustage` is set. Without one,
    `microscope.fm` is None -- and then `build_lamella_poses` returns no fluorescence
    pose, which `Lamella.fluorescence_pose`'s setter rejects, and the fluorescence tab
    builds no widget at all. That took out 19 of the 27 tests here, invisibly: CI
    installs `.[test]` rather than `.[ui]`, so the whole file skips there (FIB-734).

    `sim-arctis-configuration.yaml` is the same one `test_beam_stage_projection.py`
    uses for its compustage half, so the two agree on what an Arctis is.
    """
    import fibsem.config as fibsem_config

    path = os.path.join(
        os.path.dirname(fibsem_config.__file__),
        "config",
        "sim-arctis-configuration.yaml",
    )
    scope, _ = utils.setup_session(manufacturer="Demo", config_path=path)
    assert scope.stage_is_compustage, "the config stopped being a compustage"
    assert scope.fm is not None, "no fluorescence microscope to test the FM tab against"
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
            stage_position = FibsemStagePosition(
                x=0.0, y=0.0, z=0.0, r=beam.r, t=beam.t
            )
        poses = build_lamella_poses(self.microscope, stage_position)
        lamella = Lamella(
            petname=name,
            path=os.path.join(str(self.experiment.path), name),
            number=number,
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
            None
            if before is None
            else (before.stage_position.x, before.stage_position.y)
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


class TestTheFlagOff:
    """What "off by default" has to mean.

    Hiding the tab is only half of it. The widget subscribes to the microscope for its
    lifetime, so one built and then hidden goes on redrawing overlays on every stage
    move and counting tiles through every acquisition, for a tab nobody can open --
    and goes on holding psygnal references that have to be torn down later. The
    fluorescence tab's builder documents exactly this as the reason it builds and
    destroys rather than leaving one behind; this is the beam side keeping the promise.
    """

    def test_a_disabled_tab_builds_nothing(self, microscope, tmp_path):
        """The flag is answered before construction, not after: a microscope is present
        and connected here, and the tab still has to decline to build."""
        experiment = Experiment(path=tmp_path, name="overview-flag-off")
        os.makedirs(str(experiment.path), exist_ok=True)
        signals = (
            microscope.tiled_acquisition_signal,
            microscope.stage_position_changed,
        )
        before = [len(s) for s in signals]

        tab = AutoLamellaOverviewTab(_StubWindow(microscope, experiment))
        tab.set_enabled(False)
        tab.refresh_microscope()

        assert tab.overview is None, "built the widget with the flag off"
        assert not tab.is_available
        assert [len(s) for s in signals] == before, (
            "subscribed to the microscope with the flag off"
        )

    def test_turning_it_off_drops_what_was_built(self, tab, microscope):
        signals = (
            microscope.tiled_acquisition_signal,
            microscope.stage_position_changed,
        )
        before = [len(s) for s in signals]
        tab.set_enabled(False)
        assert tab.overview is None
        assert not tab.is_available
        assert all(a < b for a, b in zip([len(s) for s in signals], before)), (
            "a subscription survived the flag being turned off"
        )

    def test_turning_it_back_on_rebuilds(self, tab):
        """The flag can be flipped in preferences while the window is up, so off is not
        a one-way door -- and the rebuild has to be the tab's own doing, because the
        window only ever tells it the new answer."""
        tab.set_enabled(False)
        tab.set_enabled(True)
        assert tab.is_available, "the tab did not come back"

    def test_repeating_the_same_answer_does_not_rebuild(self, tab):
        """`_apply_overview_canvas_visibility` runs on every preferences change and
        every connection. A widget rebuilt each time would throw away the canvas view,
        the loaded overviews and the settings column on any unrelated setting."""
        first = tab.overview
        tab.set_enabled(True)
        assert tab.overview is first, "rebuilt on an unchanged flag"


@pytest.fixture
def fm_tab(microscope, tmp_path):
    """The fluorescence tab against the same stub window, so the two can be compared.

    Added with the pose hook (FIB-709): once both tabs answer `_pose_of`, the question
    "does this tab mark its own pose" is one test shape, and asking it of only one of
    them is how the mistake it guards against would reach the other.
    """
    from fibsem.applications.autolamella.ui.autolamella_fluorescence_overview_tab import (
        AutoLamellaFluorescenceOverviewTab,
    )

    experiment = Experiment(path=tmp_path, name="fm-overview-tab-test")
    os.makedirs(str(experiment.path), exist_ok=True)
    window = _StubWindow(microscope, experiment)
    tab = AutoLamellaFluorescenceOverviewTab(window)
    tab.refresh_microscope()
    assert tab.is_available, "the fluorescence tab did not build its widget"
    yield tab
    tab._drop_overview()


class TestItMarksTheFluorescenceSidePose:
    """The mirror of `TestItMarksTheBeamSidePose`, and the reason `_pose_of` is a hook.

    Both tabs now inherit `refresh_positions`, so the only thing deciding what each one
    marks is one small method. A mistake there is invisible to any assertion on x/y --
    see the beam-side test for the measurement.
    """

    def test_a_lamella_is_marked_at_its_fluorescence_pose(self, fm_tab, microscope):
        lamella = _lamella(fm_tab, microscope, dx=90e-6)
        fm_tab.refresh_positions()

        marked = fm_tab.overview._positions
        assert [p.name for p in marked] == [lamella.name]

        milling_tilt = lamella.milling_pose.stage_position.t
        fluorescence_tilt = lamella.fluorescence_pose.stage_position.t
        assert milling_tilt != pytest.approx(fluorescence_tilt), (
            "the two poses are indistinguishable on this microscope, so this test "
            "cannot check which one was used"
        )
        assert marked[0].t == pytest.approx(fluorescence_tilt), (
            "the marker carries the milling pose, not the fluorescence one"
        )

    def test_a_lamella_with_no_fluorescence_pose_is_listed_but_not_marked(
        self, fm_tab, microscope
    ):
        """It cannot be placed on this canvas, and it must not vanish from the list --
        which is the only place that says it exists at all."""
        lamella = _lamella(fm_tab, microscope, dx=70e-6)
        # Dropped from the pose dict rather than assigned None -- the setter takes only
        # a `MicroscopeState`, and an experiment saved before fluorescence poses existed
        # simply has no such key. This is that lamella.
        lamella.poses.pop("FLUORESCENCE", None)
        fm_tab.refresh_positions()

        assert fm_tab.overview._positions == []
        assert [row.lamella.name for row in fm_tab.lamella_list._rows()] == [
            lamella.name
        ]

    def test_move_to_drives_to_the_fluorescence_pose(self, fm_tab, microscope):
        """Moving to the milling pose from here would swing the stage 180 degrees away
        from the view being centred."""
        lamella = _lamella(fm_tab, microscope, dx=40e-6)
        drove_to = []
        fm_tab.overview.move_to = drove_to.append

        fm_tab._on_move_to_requested(lamella)

        assert len(drove_to) == 1
        assert drove_to[0].t == pytest.approx(
            lamella.fluorescence_pose.stage_position.t
        )


class TestBothTabsAnswerIsAcquiring:
    """Whether a run is in progress has to be askable of *either* tab.

    The window has to know before it lets anything drive the stage: a click-to-move on
    one overview while the other is mid-tileset does not fail loudly, it stamps tiles
    with poses the runner only planned (FIB-706). Only the beam tab used to answer.
    """

    def test_neither_is_acquiring_when_idle(self, tab, fm_tab):
        assert tab.is_acquiring is False
        assert fm_tab.is_acquiring is False

    def test_a_tab_with_no_widget_is_not_acquiring(self, tab):
        tab._drop_overview()
        assert tab.is_acquiring is False


@pytest.fixture
def locked_pair(tab, fm_tab):
    """Both tabs under a host that derives their locks the way the window does.

    The window itself cannot be built here (napari), so its two decision methods are
    borrowed rather than reimplemented -- if the rule changes, this moves with it. That
    the real window *connects* the signal to that rule is checked separately, in
    `test_overview_tab_wiring.py`, because a fixture wiring it by hand would go on
    passing if production stopped.
    """
    from fibsem.applications.autolamella.ui.AutoLamellaMainUI import (
        AutoLamellaSingleWindowUI as _Real,
    )

    class _Host:
        _overviews_allowed = _Real._overviews_allowed
        _apply_overview_locks = _Real._apply_overview_locks
        _overview_may_work = _Real._overview_may_work
        _set_overviews_allowed = _Real._set_overviews_allowed

        def __init__(self, beam, fluorescence):
            self.beam_overview_tab = beam
            self.fm_overview_tab = fluorescence
            for one in (beam, fluorescence):
                one.acquiring_changed.connect(self._apply_overview_locks)

    return _Host(tab, fm_tab), tab, fm_tab


class TestOneOverviewDoesNotDriveTheStageWhileTheOtherAcquires:
    """The failure this prevents does not announce itself.

    A click-to-move on one overview during the other's tileset drives the stage away,
    and the run carries on placing tiles at the poses it planned -- so the mosaic comes
    out looking entirely plausible and is wrong (FIB-706).
    """

    def test_a_fluorescence_run_locks_the_beam_tab(self, locked_pair, monkeypatch):
        host, beam, fluorescence = locked_pair
        toasts = []
        monkeypatch.setattr(
            "fibsem.ui.widgets.overview_widget.notification_service.show_toast",
            lambda message, level="info", *a, **k: toasts.append(message),
        )
        fluorescence.overview._set_running(True)
        try:
            assert beam.overview._may_move() is False
            assert any("the other overview is acquiring" in m for m in toasts), toasts
        finally:
            fluorescence.overview._set_running(False)

    def test_a_beam_run_locks_the_fluorescence_tab(self, locked_pair, monkeypatch):
        """Asserted on the *reason*, not just the refusal.

        The fluorescence gate also refuses when the stage is away from the fluorescence
        pose, which it is here -- so "it said no" is a test that passes with the lock
        removed entirely. It did, until this was written this way.
        """
        host, beam, fluorescence = locked_pair
        toasts = []
        monkeypatch.setattr(
            "fibsem.ui.fm.widgets.fm_overview_widget.notification_service.show_toast",
            lambda message, level="info", *a, **k: toasts.append(message),
        )
        beam.overview._set_running(True)
        try:
            assert fluorescence.overview._may_move() is False
            assert any("the other overview is acquiring" in m for m in toasts), toasts
        finally:
            beam.overview._set_running(False)

    def test_the_lock_lifts_when_the_run_ends(self, locked_pair):
        host, beam, fluorescence = locked_pair
        fluorescence.overview._set_running(True)
        fluorescence.overview._set_running(False)
        assert beam.overview._interactive is True

    def test_a_run_does_not_lock_the_tab_running_it(self, locked_pair):
        """Its own controls are managed by the run itself -- Cancel above all, which a
        lock must never take away."""
        host, beam, fluorescence = locked_pair
        fluorescence.overview._set_running(True)
        try:
            assert fluorescence.overview._interactive is True
        finally:
            fluorescence.overview._set_running(False)

    def test_a_workflow_ending_mid_run_does_not_unlock_the_other_tab(self, locked_pair):
        """Both facts, derived together. Two callers each setting the flag from their
        own half of the truth is how a control gets stuck on: the workflow says "you may
        work again" while a tileset is still walking the grid."""
        host, beam, fluorescence = locked_pair
        host._set_overviews_allowed(False)
        fluorescence.overview._set_running(True)
        try:
            host._set_overviews_allowed(True)
            assert beam.overview._may_move() is False, (
                "the workflow finishing unlocked a tab the other run still owns"
            )
        finally:
            fluorescence.overview._set_running(False)

    def test_dropping_a_running_tab_does_not_strand_the_lock(self, locked_pair):
        """A reconnection rebuilds the widget mid-run. The widget that would have said
        the run ended is the one being destroyed, so the tab says it instead -- without
        which the other overview stays locked for the rest of the session."""
        host, beam, fluorescence = locked_pair
        fluorescence.overview._set_running(True)
        assert beam.overview._may_move() is False
        fluorescence._drop_overview()
        assert beam.overview._may_move() is True
