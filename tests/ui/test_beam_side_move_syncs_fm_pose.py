"""Moving a lamella on the beam side derives its fluorescence pose -- when Link is on.

A lamella describes one piece of sample from two sides. Every beam-side way to say "this
lamella is somewhere else" -- dragging it on the FIB/SEM overview, saving the current
stage position onto it from the lamella list, and setting its milling pose from the
Selected Lamella panel's pose rows (in the main window and in the coincidence viewer) --
sets the milling pose, and then the *Link fluorescence position* preference decides:
derive the fluorescence pose from it (the default), or leave it and mark it stale. The
transform is a guess, so a pose somebody centred by hand is not rewritten without their
say.

`tests/autolamella/test_poses.py` covers what `derive_fluorescence_pose` computes. This
covers the other half: that these callers actually consult the preference and act on
it. A correct helper nothing invokes is not a fix.

The methods are borrowed onto stubs rather than driven through real widgets, because
standing up an overview widget needs a microscope, a canvas and an experiment to assert
one call.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QMessageBox

import fibsem.applications.autolamella.ui.AutoLamellaUI  # noqa: F401  (for sys.modules)
from fibsem.applications.autolamella.poses import (
    FLUORESCENCE_ORIENTATION,
    MILLING_ORIENTATION,
    build_lamella_poses,
)
from fibsem.applications.autolamella.structures import PoseProvenance
from fibsem.structures import FibsemStagePosition

_app = QApplication.instance() or QApplication(sys.argv)


def _microscope():
    from fibsem import utils
    from fibsem.fm.microscope import FluorescenceMicroscope

    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.stage_is_compustage = True
    microscope.system.stage.shuttle_pre_tilt = 0
    microscope._update_orientations()
    if microscope.fm is None:
        microscope.fm = FluorescenceMicroscope(parent=microscope)
    return microscope


def _at(microscope, orientation, x, y):
    pose = microscope.get_orientation(orientation)
    return FibsemStagePosition(x=x, y=y, z=0.0, r=pose.r, t=pose.t)


def _lamella(microscope, tmp_path, x=100e-6, y=50e-6):
    from fibsem.applications.autolamella.structures import Lamella

    poses = build_lamella_poses(microscope, _at(microscope, MILLING_ORIENTATION, x, y))
    lamella = Lamella(petname="Lamella-01", path=str(tmp_path / "Lamella-01"), number=1)
    lamella.milling_pose = poses.milling
    lamella.fluorescence_pose = poses.fluorescence
    return lamella


class _Experiment:
    def __init__(self, positions):
        # An EventedList, because `update_lamella_position_ui` emits `changed` on it --
        # which is how the FM overview canvas hears about the move.
        from psygnal.containers import EventedList

        self.positions = EventedList(positions)
        self.saves = 0

    def save(self):
        self.saves += 1


class _SelectedList:
    selected_index = 0

    def select(self, name):
        pass


class _SelectedLamellaPanel:
    """The Selected Lamella panel, which refreshes its pose rows in place.

    Recorded rather than ignored: a synced pose whose row is not redrawn leaves the
    panel displaying a position that pose no longer has, which is the same wrong answer
    the sync exists to remove -- just on screen instead of on disk.
    """

    def __init__(self):
        self.refreshed = {}

    def refresh_pose(self, pose_name, pretty):
        self.refreshed[pose_name] = pretty


def test_dragging_a_lamella_on_the_overview_moves_its_fluorescence_pose(tmp_path):
    """`AutoLamellaOverviewTab._on_move_requested` -- drag a marker on the FIB/SEM
    overview canvas.

    This used to be asked of `FibsemMinimapWidget._update_selected_position`, which was
    the same gesture on the napari tab that this one replaced.

    `test_overview_tab_host.py` also drives this handler, but only checks that the
    fluorescence pose *moved*. The orientation assertion at the end is what this test is
    for: x and y moving is equally true of the milling pose copied across, and on a
    compustage the two poses differ only in tilt (FIB-709).
    """
    from fibsem.applications.autolamella.ui.autolamella_overview_tab import (
        AutoLamellaOverviewTab,
    )

    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)
    before = lamella.fluorescence_pose.stage_position.x

    class _Tab:
        _on_move_requested = AutoLamellaOverviewTab._on_move_requested

        def __init__(self):
            self.microscope = microscope
            self.experiment = _Experiment([lamella])

        def refresh_positions(self):
            pass

    _Tab()._on_move_requested(
        lamella.name, _at(microscope, MILLING_ORIENTATION, 400e-6, -200e-6)
    )

    assert lamella.milling_pose.stage_position.x == pytest.approx(400e-6)
    assert lamella.fluorescence_pose.stage_position.x == pytest.approx(400e-6)
    assert lamella.fluorescence_pose.stage_position.x != pytest.approx(before)
    # ...and it is still a fluorescence pose, not the milling one copied across
    assert (
        microscope.get_stage_orientation(lamella.fluorescence_pose.stage_position)
        == FLUORESCENCE_ORIENTATION
    )


def test_saving_a_new_position_from_the_lamella_list_moves_it_too(
    tmp_path, monkeypatch
):
    """`AutoLamellaUI.update_lamella_position_ui` -- the list row's Update Position,
    which records wherever the stage is now onto the selected lamella."""
    module = sys.modules["fibsem.applications.autolamella.ui.AutoLamellaUI"]

    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)
    before = lamella.fluorescence_pose.stage_position.x
    microscope.move_stage_absolute(
        _at(microscope, MILLING_ORIENTATION, 400e-6, -200e-6)
    )
    monkeypatch.setattr(
        module.QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes)
    )

    class _UI:
        update_lamella_position_ui = module.AutoLamellaUI.update_lamella_position_ui

        def __init__(self):
            self.microscope = microscope
            self.protocol = object()
            self.experiment = _Experiment([lamella])
            self.lamella_list = _SelectedList()

        def update_lamella_combobox(self, latest=False):
            pass

        def update_ui(self):
            pass

    _UI().update_lamella_position_ui()

    assert lamella.milling_pose.stage_position.x == pytest.approx(400e-6)
    assert lamella.fluorescence_pose.stage_position.x == pytest.approx(400e-6)
    assert lamella.fluorescence_pose.stage_position.x != pytest.approx(before)
    assert (
        microscope.get_stage_orientation(lamella.fluorescence_pose.stage_position)
        == FLUORESCENCE_ORIENTATION
    )


def test_declining_the_confirmation_moves_neither_pose(tmp_path, monkeypatch):
    """The sync must sit after the confirmation, not before it."""
    module = sys.modules["fibsem.applications.autolamella.ui.AutoLamellaUI"]

    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)
    before = (
        lamella.milling_pose.stage_position.x,
        lamella.fluorescence_pose.stage_position.x,
    )
    microscope.move_stage_absolute(
        _at(microscope, MILLING_ORIENTATION, 400e-6, -200e-6)
    )
    monkeypatch.setattr(
        module.QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No)
    )

    class _UI:
        update_lamella_position_ui = module.AutoLamellaUI.update_lamella_position_ui

        def __init__(self):
            self.microscope = microscope
            self.protocol = object()
            self.experiment = _Experiment([lamella])
            self.lamella_list = _SelectedList()

        def update_lamella_combobox(self, latest=False):
            pass

        def update_ui(self):
            pass

    _UI().update_lamella_position_ui()

    assert (
        lamella.milling_pose.stage_position.x,
        lamella.fluorescence_pose.stage_position.x,
    ) == before


def _pose_row_ui(module, microscope, lamella):
    """A stub carrying `AutoLamellaUI._set_current_position_as_pose` and nothing else."""

    class _UI:
        _set_current_position_as_pose = (
            module.AutoLamellaUI._set_current_position_as_pose
        )

        def __init__(self):
            self.microscope = microscope
            self.experiment = _Experiment([lamella])
            self.lamella_list = _SelectedList()
            self.selected_lamella_widget = _SelectedLamellaPanel()

    return _UI()


def test_setting_the_milling_pose_from_the_pose_rows_moves_the_fluorescence_pose(
    tmp_path, monkeypatch
):
    """`AutoLamellaUI._set_current_position_as_pose` -- the Selected Lamella panel's
    per-pose Update Position button, a third door onto the same move."""
    module = sys.modules["fibsem.applications.autolamella.ui.AutoLamellaUI"]

    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)
    before = lamella.fluorescence_pose.stage_position.x
    microscope.move_stage_absolute(
        _at(microscope, MILLING_ORIENTATION, 400e-6, -200e-6)
    )
    monkeypatch.setattr(
        module.QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes)
    )

    ui = _pose_row_ui(module, microscope, lamella)
    ui._set_current_position_as_pose("MILLING")

    assert lamella.milling_pose.stage_position.x == pytest.approx(400e-6)
    assert lamella.fluorescence_pose.stage_position.x == pytest.approx(400e-6)
    assert lamella.fluorescence_pose.stage_position.x != pytest.approx(before)
    assert (
        microscope.get_stage_orientation(lamella.fluorescence_pose.stage_position)
        == FLUORESCENCE_ORIENTATION
    )
    assert "FLUORESCENCE" in ui.selected_lamella_widget.refreshed


def _link(monkeypatch, fluorescence: bool = True, milling: bool = True):
    """Pin the Link preferences for a test, whatever the machine's file says."""
    import fibsem.config as fibsem_cfg

    preferences = fibsem_cfg.UserPreferences()
    preferences.poses.link_fluorescence_position = fluorescence
    preferences.poses.link_milling_position = milling
    monkeypatch.setattr(fibsem_cfg, "load_user_preferences", lambda: preferences)
    return preferences


def test_setting_the_fluorescence_pose_from_the_pose_rows_derives_the_milling_pose(
    tmp_path, monkeypatch
):
    """The other direction, under its own preference: with *Link milling position*
    on, a fluorescence pose set by hand derives the milling pose from itself."""
    module = sys.modules["fibsem.applications.autolamella.ui.AutoLamellaUI"]

    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)
    milling_before = lamella.milling_pose.stage_position.x
    target = _at(microscope, FLUORESCENCE_ORIENTATION, 400e-6, -200e-6)
    microscope.move_stage_absolute(target)
    monkeypatch.setattr(
        module.QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes)
    )
    _link(monkeypatch, milling=True)

    ui = _pose_row_ui(module, microscope, lamella)
    ui._set_current_position_as_pose("FLUORESCENCE")

    assert lamella.fluorescence_pose.stage_position.x == pytest.approx(400e-6)
    assert lamella.milling_pose.stage_position.x == pytest.approx(400e-6)
    assert lamella.milling_pose.stage_position.x != pytest.approx(milling_before)
    assert lamella.provenance_of("MILLING") is PoseProvenance.DERIVED
    assert "MILLING" in ui.selected_lamella_widget.refreshed


def test_setting_the_fluorescence_pose_with_link_off_leaves_the_milling_pose(
    tmp_path, monkeypatch
):
    """Link off: the fluorescence pose is the one thing they came to change, and the
    milling pose -- observed, by someone -- is left where it was and marked stale."""
    module = sys.modules["fibsem.applications.autolamella.ui.AutoLamellaUI"]

    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)
    milling_before = lamella.milling_pose.stage_position.x
    target = _at(microscope, FLUORESCENCE_ORIENTATION, 400e-6, -200e-6)
    microscope.move_stage_absolute(target)
    monkeypatch.setattr(
        module.QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes)
    )
    _link(monkeypatch, milling=False)

    ui = _pose_row_ui(module, microscope, lamella)
    ui._set_current_position_as_pose("FLUORESCENCE")

    assert lamella.fluorescence_pose.stage_position.x == pytest.approx(400e-6)
    assert lamella.milling_pose.stage_position.x == pytest.approx(milling_before)
    assert lamella.provenance_of("MILLING") is PoseProvenance.STALE


def test_moving_the_milling_pose_with_link_off_leaves_the_fluorescence_pose(
    tmp_path, monkeypatch
):
    """The beam-side drag with *Link fluorescence position* off: an observed
    fluorescence pose stays put and is marked stale, rather than being rewritten from
    a transform nobody asked to trust."""
    from fibsem.applications.autolamella.ui.autolamella_overview_tab import (
        AutoLamellaOverviewTab,
    )

    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)
    lamella.pose_provenance["FLUORESCENCE"] = PoseProvenance.OBSERVED
    before = lamella.fluorescence_pose.stage_position.x
    _link(monkeypatch, fluorescence=False)

    class _Tab:
        _on_move_requested = AutoLamellaOverviewTab._on_move_requested

        def __init__(self):
            self.microscope = microscope
            self.experiment = _Experiment([lamella])

        def refresh_positions(self):
            pass

    _Tab()._on_move_requested(
        lamella.name, _at(microscope, MILLING_ORIENTATION, 400e-6, -200e-6)
    )

    assert lamella.milling_pose.stage_position.x == pytest.approx(400e-6)
    assert lamella.fluorescence_pose.stage_position.x == pytest.approx(before)
    assert lamella.provenance_of("FLUORESCENCE") is PoseProvenance.STALE


def test_declining_the_pose_row_confirmation_moves_neither_pose(tmp_path, monkeypatch):
    """The sync sits after this confirmation too."""
    module = sys.modules["fibsem.applications.autolamella.ui.AutoLamellaUI"]

    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)
    before = (
        lamella.milling_pose.stage_position.x,
        lamella.fluorescence_pose.stage_position.x,
    )
    microscope.move_stage_absolute(
        _at(microscope, MILLING_ORIENTATION, 400e-6, -200e-6)
    )
    monkeypatch.setattr(
        module.QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.No)
    )

    _pose_row_ui(module, microscope, lamella)._set_current_position_as_pose("MILLING")

    assert (
        lamella.milling_pose.stage_position.x,
        lamella.fluorescence_pose.stage_position.x,
    ) == before


def test_setting_the_milling_pose_in_the_coincidence_viewer_moves_it_too(
    tmp_path, monkeypatch
):
    """`FluorescenceCoincidenceViewerWidget._on_slw_pose_update` -- the same panel, in
    the other window, with its own copy of the handler."""
    from fibsem.applications.autolamella.ui import (
        fluorescence_coincidence_viewer_widget as module,
    )

    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)
    before = lamella.fluorescence_pose.stage_position.x
    lamella.milling_angle = None
    microscope.move_stage_absolute(
        _at(microscope, MILLING_ORIENTATION, 400e-6, -200e-6)
    )
    monkeypatch.setattr(
        module.QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes)
    )

    class _Viewer:
        _on_slw_pose_update = (
            module.FluorescenceCoincidenceViewerWidget._on_slw_pose_update
        )

        def __init__(self):
            self.microscope = microscope
            self._selected_lamella = lamella
            self.experiment = _Experiment([lamella])
            self.selected_lamella_widget = _SelectedLamellaPanel()

    viewer = _Viewer()
    viewer._on_slw_pose_update("MILLING")

    assert lamella.milling_pose.stage_position.x == pytest.approx(400e-6)
    assert lamella.fluorescence_pose.stage_position.x == pytest.approx(400e-6)
    assert lamella.fluorescence_pose.stage_position.x != pytest.approx(before)
    assert (
        microscope.get_stage_orientation(lamella.fluorescence_pose.stage_position)
        == FLUORESCENCE_ORIENTATION
    )
    assert "FLUORESCENCE" in viewer.selected_lamella_widget.refreshed
    # this copy never recomputed the milling angle either, so the lamella went on
    # reporting the angle it was marked at rather than the one it now sits at
    assert lamella.milling_angle is not None
