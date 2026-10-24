"""Moving a lamella on the beam side has to carry its fluorescence pose along.

A lamella describes one piece of sample from two sides. Every beam-side way to say "this
lamella is somewhere else" -- dragging it on the FIB/SEM overview, saving the current
stage position onto it from the lamella list, and setting its milling pose from the
Selected Lamella panel's pose rows (in the main window and in the coincidence viewer) --
sets the milling pose and historically stopped there, leaving the fluorescence pose
naming where the lamella used to be. Nothing about a stale pose looks wrong, which is
what made it worth pinning.

`tests/autolamella/test_poses.py` covers what `sync_fluorescence_pose` computes. This
covers the other half: that these two callers actually call it. A correct helper nothing
invokes is not a fix.

The methods are borrowed onto stubs rather than driven through real widgets, because
standing up an overview widget needs a microscope, a canvas and an experiment to assert
one call.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QMessageBox

import fibsem.applications.autolamella.ui.panel.experiment  # noqa: F401  (for sys.modules)
from fibsem.applications.autolamella.poses import (
    FLUORESCENCE_ORIENTATION,
    MILLING_ORIENTATION,
    build_lamella_poses,
)
from fibsem.structures.settings import FibsemStagePosition
from tests.fixtures.demo_stage import demo_session

_app = QApplication.instance() or QApplication(sys.argv)


def _microscope():
    from fibsem.drivers.demo.microscope import DemoFluorescenceMicroscope

    microscope, _ = demo_session(compustage=True)
    microscope.system.stage.shuttle_pre_tilt = 0
    microscope.stage_movement.update_orientations()
    if microscope.fm is None:
        microscope.fm = DemoFluorescenceMicroscope(parent=microscope)
    return microscope


def _at(microscope, orientation, x, y):
    pose = microscope.get_orientation(orientation)
    return FibsemStagePosition(x=x, y=y, z=0.0, r=pose.r, t=pose.t)


def _lamella(microscope, tmp_path, x=100e-6, y=50e-6):
    from fibsem.applications.autolamella.structures import Lamella

    poses = build_lamella_poses(microscope, _at(microscope, MILLING_ORIENTATION, x, y))
    lamella = Lamella(petname="Lamella-01", path=str(tmp_path / "Lamella-01"), number=1)
    lamella.milling_pose = poses.beam
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


def test_dragging_a_lamella_on_the_overview_moves_its_fluorescence_pose(tmp_path):
    """`BeamOverviewTab._on_move_requested` -- drag a marker on the FIB/SEM
    overview canvas.

    This used to be asked of `FibsemMinimapWidget._update_selected_position`, which was
    the same gesture on the napari tab that this one replaced.

    `test_overview_tab_host.py` also drives this handler, but only checks that the
    fluorescence pose *moved*. The orientation assertion at the end is what this test is
    for: x and y moving is equally true of the milling pose copied across, and on a
    compustage the two poses differ only in tilt (FIB-709).
    """
    from fibsem.applications.autolamella.ui.overview.beam import BeamOverviewTab

    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)
    before = lamella.fluorescence_pose.stage_position.x

    class _Tab:
        _on_move_requested = BeamOverviewTab._on_move_requested

        def __init__(self):
            self.microscope = microscope
            self.experiment = _Experiment([lamella])

        def refresh_positions(self):
            pass

    tab = _Tab()
    changed = []
    tab.experiment.positions.events.changed.connect(lambda *a: changed.append(a))
    tab._on_move_requested(
        lamella.name, _at(microscope, MILLING_ORIENTATION, 400e-6, -200e-6)
    )

    # One rule for every move (`poses.move_lamella`): saved, and changed emitted
    # so the other views re-mark it.
    assert tab.experiment.saves == 1 and len(changed) == 1
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
    """`ExperimentTab.update_lamella_position_ui` -- the list row's Update Position,
    which records wherever the stage is now onto the selected lamella."""
    module = sys.modules["fibsem.applications.autolamella.ui.panel.experiment"]

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
        update_lamella_position_ui = module.ExperimentTab.update_lamella_position_ui

        def __init__(self):
            self.microscope = microscope
            self.protocol = object()
            self.experiment = _Experiment([lamella])
            self.lamella_list = _SelectedList()
            self.session = SimpleNamespace(
                runner=SimpleNamespace(run_refusal=lambda what: "")
            )

        def update_items(self, latest=False):
            pass

        def refresh(self):
            pass

    _UI().update_lamella_position_ui()

    assert lamella.milling_pose.stage_position.x == pytest.approx(400e-6)
    assert lamella.fluorescence_pose.stage_position.x == pytest.approx(400e-6)
    assert lamella.fluorescence_pose.stage_position.x != pytest.approx(before)
    assert (
        microscope.get_stage_orientation(lamella.fluorescence_pose.stage_position)
        == FLUORESCENCE_ORIENTATION
    )


def test_saving_a_new_position_keeps_the_objective_position(tmp_path, monkeypatch):
    """The microscope state does not capture the FM objective, so Update
    Position keeps the milling pose's stored focus, as Set pose does (FIB-1456)."""
    module = sys.modules["fibsem.applications.autolamella.ui.panel.experiment"]

    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)
    lamella.milling_pose.objective_position = 0.0123
    monkeypatch.setattr(
        module.QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes)
    )

    class _UI:
        update_lamella_position_ui = module.ExperimentTab.update_lamella_position_ui

        def __init__(self):
            self.microscope = microscope
            self.protocol = object()
            self.experiment = _Experiment([lamella])
            self.lamella_list = _SelectedList()
            self.session = SimpleNamespace(
                runner=SimpleNamespace(run_refusal=lambda what: "")
            )

        def update_items(self, latest=False):
            pass

        def refresh(self):
            pass

    _UI().update_lamella_position_ui()

    assert lamella.milling_pose.objective_position == pytest.approx(0.0123)


def test_declining_the_confirmation_moves_neither_pose(tmp_path, monkeypatch):
    """The sync must sit after the confirmation, not before it."""
    module = sys.modules["fibsem.applications.autolamella.ui.panel.experiment"]

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
        update_lamella_position_ui = module.ExperimentTab.update_lamella_position_ui

        def __init__(self):
            self.microscope = microscope
            self.protocol = object()
            self.experiment = _Experiment([lamella])
            self.lamella_list = _SelectedList()
            self.session = SimpleNamespace(
                runner=SimpleNamespace(run_refusal=lambda what: "")
            )

        def update_items(self, latest=False):
            pass

        def refresh(self):
            pass

    _UI().update_lamella_position_ui()

    assert (
        lamella.milling_pose.stage_position.x,
        lamella.fluorescence_pose.stage_position.x,
    ) == before


def _pose_rows(microscope, lamella):
    """The Selected Lamella panel's controller over *lamella*, its pose rows'
    refreshes recorded on ``widget.refreshed``."""
    from fibsem.applications.autolamella.ui.lamella.selected import (
        SelectedLamellaController,
        SelectedLamellaWidget,
    )

    widget = SelectedLamellaWidget()
    widget.refreshed = {}
    widget.refresh_pose = widget.refreshed.__setitem__
    experiment = _Experiment([lamella])
    controller = SelectedLamellaController(
        widget,
        experiment=lambda: experiment,
        microscope=lambda: microscope,
        lamella=lambda: lamella,
        move_stage=lambda position: None,
    )
    controller.experiment = experiment
    return controller


def test_setting_the_milling_pose_from_the_pose_rows_moves_the_fluorescence_pose(
    tmp_path, monkeypatch
):
    """`SelectedLamellaController.set_pose` -- the Selected Lamella panel's per-pose
    Update Position button, in the main window and the coincidence viewer, a third
    door onto the same move."""
    from fibsem.applications.autolamella.ui.lamella import selected as module

    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)
    before = lamella.fluorescence_pose.stage_position.x
    microscope.move_stage_absolute(
        _at(microscope, MILLING_ORIENTATION, 400e-6, -200e-6)
    )
    monkeypatch.setattr(
        module.QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes)
    )

    lamella.milling_angle = None
    ui = _pose_rows(microscope, lamella)
    ui.set_pose("MILLING")

    assert lamella.milling_pose.stage_position.x == pytest.approx(400e-6)
    assert lamella.fluorescence_pose.stage_position.x == pytest.approx(400e-6)
    assert lamella.fluorescence_pose.stage_position.x != pytest.approx(before)
    assert (
        microscope.get_stage_orientation(lamella.fluorescence_pose.stage_position)
        == FLUORESCENCE_ORIENTATION
    )
    assert "FLUORESCENCE" in ui.widget.refreshed
    assert ui.experiment.saves == 1
    assert lamella.milling_angle is not None


def test_setting_the_fluorescence_pose_from_the_pose_rows_is_not_undone(
    tmp_path, monkeypatch
):
    """Only the milling pose fans out.

    Re-deriving the fluorescence pose after a user has just set it by hand would
    overwrite the one thing they came to that button to change.
    """
    from fibsem.applications.autolamella.ui.lamella import selected as module

    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)
    milling_before = lamella.milling_pose.stage_position.x
    target = _at(microscope, FLUORESCENCE_ORIENTATION, 400e-6, -200e-6)
    microscope.move_stage_absolute(target)
    monkeypatch.setattr(
        module.QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes)
    )

    _pose_rows(microscope, lamella).set_pose("FLUORESCENCE")

    assert lamella.fluorescence_pose.stage_position.x == pytest.approx(400e-6)
    assert lamella.milling_pose.stage_position.x == pytest.approx(milling_before)


def test_declining_the_pose_row_confirmation_moves_neither_pose(tmp_path, monkeypatch):
    """The sync sits after this confirmation too."""
    from fibsem.applications.autolamella.ui.lamella import selected as module

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

    _pose_rows(microscope, lamella).set_pose("MILLING")

    assert (
        lamella.milling_pose.stage_position.x,
        lamella.fluorescence_pose.stage_position.x,
    ) == before
