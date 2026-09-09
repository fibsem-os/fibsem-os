"""The lamella details' *Derive* action and the provenance marks (FIB-831).

Each pose row shows where its pose came from -- nothing for observed, "derived" or
"stale" otherwise -- and offers to derive it from the other pose. The fluorescence row
offers the orientations the FM images from. Every derivation confirms, since every one
overwrites, and the result is marked derived.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QMessageBox

import fibsem.applications.autolamella.ui.AutoLamellaUI  # noqa: F401  (for sys.modules)
import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.poses import (
    FLUORESCENCE_POSE,
    MILLING_POSE,
    build_lamella_poses,
)
from fibsem.applications.autolamella.structures import Lamella, PoseProvenance
from fibsem.applications.autolamella.ui.lamella_name_list_widget import (
    _lamella_pose_icon,
)
from fibsem.applications.autolamella.ui.lamella_pose_list_widget import (
    LamellaPoseListWidget,
    LamellaPoseRowWidget,
)
from fibsem.structures import FibsemStagePosition

_app = QApplication.instance() or QApplication(sys.argv)

ARCTIS_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")


def _microscope():
    microscope, _ = utils.setup_session(config_path=ARCTIS_CONFIG)
    return microscope


def _at(microscope, orientation, x=100e-6, y=50e-6):
    pose = microscope.get_orientation(orientation)
    return FibsemStagePosition(x=x, y=y, z=0.0, r=pose.r, t=pose.t)


def _lamella(microscope, tmp_path):
    poses = build_lamella_poses(microscope, _at(microscope, "MILLING"))
    lamella = Lamella(petname="Lamella-01", path=str(tmp_path / "Lamella-01"), number=1)
    lamella.set_pose(MILLING_POSE, poses.milling, PoseProvenance.OBSERVED)
    lamella.set_pose(FLUORESCENCE_POSE, poses.fluorescence, PoseProvenance.DERIVED)
    return lamella


# ── the rows ─────────────────────────────────────────────────────────────


def test_an_observed_pose_shows_no_chip():
    row = LamellaPoseRowWidget("MILLING", "x", provenance=PoseProvenance.OBSERVED)
    assert not row.provenance_label.isVisibleTo(row)


@pytest.mark.parametrize(
    "provenance, word",
    [(PoseProvenance.DERIVED, "derived"), (PoseProvenance.STALE, "stale")],
)
def test_a_pose_with_something_to_say_says_it(provenance, word):
    row = LamellaPoseRowWidget("FLUORESCENCE", "x", provenance=provenance)
    assert row.provenance_label.isVisibleTo(row)
    assert row.provenance_label.text() == word


def test_a_single_orientation_derives_straight_away():
    row = LamellaPoseRowWidget("FLUORESCENCE", "x", derive_orientations=["FIB"])
    received = []
    row.derive_clicked.connect(lambda n, o: received.append((n, o)))

    row.btn_derive.click()

    assert received == [("FLUORESCENCE", "FIB")]


def test_the_milling_row_derives_with_no_orientation_to_choose():
    row = LamellaPoseRowWidget("MILLING", "x")
    received = []
    row.derive_clicked.connect(lambda n, o: received.append((n, o)))

    row.btn_derive.click()

    assert received == [("MILLING", None)]


def test_several_orientations_are_offered_as_a_menu():
    row = LamellaPoseRowWidget("FLUORESCENCE", "x", derive_orientations=["FM", "SEM"])
    received = []
    row.derive_clicked.connect(lambda n, o: received.append((n, o)))

    menu = row._build_derive_menu()
    assert [a.text() for a in menu.actions()] == [
        "Derive into the FM orientation",
        "Derive into the SEM orientation",
    ]
    menu.actions()[1].trigger()

    assert received == [("FLUORESCENCE", "SEM")]


def test_the_list_hands_the_fm_orientations_to_the_fluorescence_row_only(tmp_path):
    microscope = _microscope()
    widget = LamellaPoseListWidget()
    widget.set_fluorescence_orientations(["FM", "SEM"])
    widget.set_lamella(_lamella(microscope, tmp_path))

    rows = {
        widget._list.itemWidget(
            widget._list.item(i)
        ).pose_name: widget._list.itemWidget(widget._list.item(i))
        for i in range(widget._list.count())
    }
    assert rows["FLUORESCENCE"].derive_orientations == ["FM", "SEM"]
    assert rows["MILLING"].derive_orientations == []
    assert rows["FLUORESCENCE"].provenance_label.text() == "derived"
    assert not rows["MILLING"].provenance_label.isVisibleTo(rows["MILLING"])


# ── the list row's mark ──────────────────────────────────────────────────


def test_the_lamella_row_marks_stale_over_derived(tmp_path):
    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)

    icon, _, tooltip = _lamella_pose_icon(lamella)
    assert icon == "mdi:link-variant"
    assert "fluorescence" in tooltip

    lamella.pose_provenance[MILLING_POSE] = PoseProvenance.STALE
    icon, _, tooltip = _lamella_pose_icon(lamella)
    assert icon == "mdi:link-variant-off"
    assert "milling" in tooltip

    lamella.pose_provenance[MILLING_POSE] = PoseProvenance.OBSERVED
    lamella.pose_provenance[FLUORESCENCE_POSE] = PoseProvenance.OBSERVED
    assert _lamella_pose_icon(lamella) is None


# ── the main window's handler ────────────────────────────────────────────


class _Experiment:
    def __init__(self, positions, path):
        from psygnal.containers import EventedList

        self.positions = EventedList(positions)
        self.path = path
        self.saves = 0

    def save(self):
        self.saves += 1


class _SelectedList:
    selected_index = 0


class _Panel:
    def __init__(self):
        self.shown = []

    def set_lamella(self, lamella):
        self.shown.append(lamella)

    def refresh_pose(self, *a, **k):
        pass


def _ui(module, microscope, lamella, tmp_path):
    class _UI:
        _derive_lamella_pose = module.AutoLamellaUI._derive_lamella_pose

        def __init__(self):
            self.microscope = microscope
            self.experiment = _Experiment([lamella], tmp_path)
            self.lamella_list = _SelectedList()
            self.selected_lamella_widget = _Panel()

    return _UI()


def _yes(module, monkeypatch, answer=QMessageBox.Yes):
    asked = []
    monkeypatch.setattr(
        module.QMessageBox,
        "question",
        staticmethod(lambda *a, **k: asked.append(a) or answer),
    )
    return asked


def test_deriving_the_fluorescence_pose_into_an_orientation(tmp_path, monkeypatch):
    module = sys.modules["fibsem.applications.autolamella.ui.AutoLamellaUI"]
    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)
    lamella.fluorescence_pose.objective_position = 7.7e-3
    asked = _yes(module, monkeypatch)
    ui = _ui(module, microscope, lamella, tmp_path)

    ui._derive_lamella_pose(FLUORESCENCE_POSE, "SEM")

    assert len(asked) == 1
    assert "into the SEM orientation" in asked[0][2]
    fluorescence = lamella.fluorescence_pose.stage_position
    assert microscope.get_stage_orientation(fluorescence) == "SEM"
    assert lamella.provenance_of(FLUORESCENCE_POSE) is PoseProvenance.DERIVED
    assert lamella.fluorescence_pose.objective_position == pytest.approx(7.7e-3)
    assert ui.experiment.saves == 1
    assert ui.selected_lamella_widget.shown == [lamella]


def test_overwriting_an_observed_pose_says_so(tmp_path, monkeypatch):
    module = sys.modules["fibsem.applications.autolamella.ui.AutoLamellaUI"]
    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)
    lamella.pose_provenance[FLUORESCENCE_POSE] = PoseProvenance.OBSERVED
    asked = _yes(module, monkeypatch, QMessageBox.No)
    ui = _ui(module, microscope, lamella, tmp_path)
    before = lamella.fluorescence_pose.stage_position.t

    ui._derive_lamella_pose(FLUORESCENCE_POSE, "SEM")

    assert "set by hand" in asked[0][2]
    assert lamella.fluorescence_pose.stage_position.t == pytest.approx(before)
    assert lamella.provenance_of(FLUORESCENCE_POSE) is PoseProvenance.OBSERVED
    assert ui.experiment.saves == 0


def test_deriving_the_milling_pose_from_the_fluorescence_pose(tmp_path, monkeypatch):
    module = sys.modules["fibsem.applications.autolamella.ui.AutoLamellaUI"]
    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)
    lamella.set_pose_position(FLUORESCENCE_POSE, _at(microscope, "FM", x=400e-6))
    _yes(module, monkeypatch)
    ui = _ui(module, microscope, lamella, tmp_path)

    ui._derive_lamella_pose(MILLING_POSE)

    milling = lamella.milling_pose.stage_position
    assert milling.x == pytest.approx(400e-6)
    assert microscope.get_stage_orientation(milling) == "MILLING"
    assert lamella.provenance_of(MILLING_POSE) is PoseProvenance.DERIVED
    assert lamella.milling_angle is not None


def test_a_derivation_the_instrument_refuses_writes_nothing(tmp_path, monkeypatch):
    """A fluorescence pose the objective cannot see the sample from is not one to
    derive a milling pose from; the refusal is a toast and the pose is left alone."""
    module = sys.modules["fibsem.applications.autolamella.ui.AutoLamellaUI"]
    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)
    lamella.set_pose_position(FLUORESCENCE_POSE, _at(microscope, "MILLING", x=9e-6))
    _yes(module, monkeypatch)
    ui = _ui(module, microscope, lamella, tmp_path)
    before = lamella.milling_pose.stage_position.x

    ui._derive_lamella_pose(MILLING_POSE)

    assert lamella.milling_pose.stage_position.x == pytest.approx(before)
    assert lamella.provenance_of(MILLING_POSE) is PoseProvenance.OBSERVED
    assert ui.experiment.saves == 0
