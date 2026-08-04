"""Moving a lamella on the beam side has to carry its fluorescence pose along.

A lamella describes one piece of sample from two sides. Both of the beam-side ways to
say "this lamella is somewhere else" -- dragging it on the FIB/SEM overview, and saving
the current stage position onto it from the lamella list -- set the milling pose and
historically stopped there, leaving the fluorescence pose naming where the lamella used
to be. Nothing about a stale pose looks wrong, which is what made it worth pinning.

`tests/autolamella/test_poses.py` covers what `sync_fluorescence_pose` computes. This
covers the other half: that these two callers actually call it. A correct helper nothing
invokes is not a fix.

The methods are borrowed onto stubs rather than driven through real widgets, because the
minimap needs a napari viewer and those segfault under the offscreen platform.
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


def test_dragging_a_lamella_on_the_minimap_moves_its_fluorescence_pose(tmp_path):
    """`FibsemMinimapWidget._update_selected_position` -- right-click the FIB/SEM
    overview, Move Selected Position Here."""
    from fibsem.ui.FibsemMinimapWidget import FibsemMinimapWidget

    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)
    before = lamella.fluorescence_pose.stage_position.x

    class _Minimap:
        _update_selected_position = FibsemMinimapWidget._update_selected_position

        def __init__(self):
            self.microscope = microscope
            self.lamella_list = _SelectedList()
            self.parent_widget = type(
                "_UI", (), {"experiment": _Experiment([lamella])}
            )()

        def _update_position_display(self):
            pass

    _Minimap()._update_selected_position(
        _at(microscope, MILLING_ORIENTATION, 400e-6, -200e-6)
    )

    assert lamella.milling_pose.stage_position.x == pytest.approx(400e-6)
    assert lamella.fluorescence_pose.stage_position.x == pytest.approx(400e-6)
    assert lamella.fluorescence_pose.stage_position.x != pytest.approx(before)
    # ...and it is still a fluorescence pose, not the milling one copied across
    assert (
        microscope.get_stage_orientation(lamella.fluorescence_pose.stage_position)
        == FLUORESCENCE_ORIENTATION
    )


def test_saving_a_new_position_from_the_lamella_list_moves_it_too(tmp_path, monkeypatch):
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
