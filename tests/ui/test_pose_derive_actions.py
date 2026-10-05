"""The pose rows' *Derive* action, and what each row says about its pose (FIB-831).

A row says nothing for an observed pose, "derived" for one worked out from the other,
and how far off it is once the two poses disagree by more than a hand-centring could
explain. The fluorescence row offers the orientations the FM images from. Every
derivation confirms, since every one overwrites, and the result is marked derived.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QMessageBox

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.poses import (
    DISAGREEMENT_WARNING_M,
    FLUORESCENCE_POSE,
    MILLING_POSE,
    PoseProvenance,
    build_lamella_poses,
    derivation_question,
    derive_pose,
    move_pose,
    pose_disagreement,
)
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
    Lamella,
)
from fibsem.applications.autolamella.ui import pose_actions
from fibsem.applications.autolamella.ui.lamella_pose_list_widget import (
    LamellaPoseListWidget,
    LamellaPoseRowWidget,
)
from fibsem.structures import FibsemStagePosition

_app = QApplication.instance() or QApplication(sys.argv)

ARCTIS_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")
IFLM_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-iflm-configuration.yaml")


def _microscope(config=ARCTIS_CONFIG):
    microscope, _ = utils.setup_session(config_path=config)
    return microscope


def _at(microscope, orientation, x=100e-6, y=50e-6):
    pose = microscope.get_orientation(orientation)
    return FibsemStagePosition(x=x, y=y, z=0.0, r=pose.r, t=pose.t)


def _lamella(microscope, tmp_path):
    """Marked at the beams: the milling pose observed, the fluorescence one derived."""
    poses = build_lamella_poses(microscope, _at(microscope, "MILLING"))
    lamella = Lamella(petname="Lamella-01", path=str(tmp_path / "Lamella-01"), number=1)
    poses.write_to(lamella)
    return lamella


def _experiment(tmp_path, lamella):
    experiment = Experiment(path=tmp_path, name="derive-exp")
    os.makedirs(experiment.path, exist_ok=True)
    experiment.task_protocol = AutoLamellaTaskProtocol()
    experiment.positions.append(lamella)
    return experiment


def _rows(widget):
    return {
        row.pose_name: row
        for row in (
            widget._list.itemWidget(widget._list.item(i))
            for i in range(widget._list.count())
        )
    }


# ── how far apart the two poses are ──────────────────────────────────────


@pytest.mark.parametrize("config", [ARCTIS_CONFIG, IFLM_CONFIG])
def test_a_freshly_marked_lamella_does_not_disagree_with_itself(tmp_path, config):
    microscope = _microscope(config)
    lamella = _lamella(microscope, tmp_path)

    assert pose_disagreement(microscope, lamella) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("config", [ARCTIS_CONFIG, IFLM_CONFIG])
def test_a_pose_left_behind_says_how_far(tmp_path, config):
    """The FIB-954 shape: the milling pose re-recorded 30 um away by the plain setter
    a workflow task uses, which moves nothing else. No writer had to flag it."""
    microscope = _microscope(config)
    lamella = _lamella(microscope, tmp_path)
    moved = lamella.milling_pose
    moved.stage_position = _at(microscope, "MILLING", x=130e-6)
    lamella.milling_pose = moved

    assert pose_disagreement(microscope, lamella) == pytest.approx(30e-6, rel=1e-3)


def test_there_is_nothing_to_compare_without_both_poses(tmp_path):
    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)
    del lamella.poses[FLUORESCENCE_POSE]

    assert pose_disagreement(microscope, lamella) is None


# ── the rows ─────────────────────────────────────────────────────────────


def test_an_observed_pose_shows_no_chip():
    row = LamellaPoseRowWidget("MILLING", None, provenance=PoseProvenance.OBSERVED)

    assert row.provenance_label.isHidden()


def test_a_derived_pose_says_so():
    row = LamellaPoseRowWidget("FLUORESCENCE", None, provenance=PoseProvenance.DERIVED)

    assert not row.provenance_label.isHidden()
    assert row.provenance_label.text() == "derived"


def test_a_small_disagreement_says_nothing():
    """Two hand-centred poses legitimately differ by a few microns."""
    row = LamellaPoseRowWidget("FLUORESCENCE", None)

    row.set_disagreement(DISAGREEMENT_WARNING_M / 2)

    assert row.provenance_label.isHidden()


def test_a_large_disagreement_replaces_the_chip_and_clears_again():
    row = LamellaPoseRowWidget("FLUORESCENCE", None, provenance=PoseProvenance.DERIVED)

    row.set_disagreement(32e-6)
    assert row.provenance_label.text() == "32 µm off"

    row.set_disagreement(None)
    assert row.provenance_label.text() == "derived"


def test_the_list_puts_the_disagreement_on_the_derived_pose(tmp_path):
    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)
    widget = LamellaPoseListWidget()
    widget.set_lamella(lamella)

    widget.set_pose_disagreement(32e-6)

    rows = _rows(widget)
    assert rows[FLUORESCENCE_POSE].provenance_label.text() == "32 µm off"
    assert rows[MILLING_POSE].provenance_label.isHidden()


def test_a_single_orientation_derives_straight_away():
    row = LamellaPoseRowWidget("FLUORESCENCE", None, derive_orientations=["FIB"])
    asked = []
    row.derive_clicked.connect(lambda name, o: asked.append((name, o)))

    row.btn_derive.click()

    assert asked == [("FLUORESCENCE", "FIB")]


def test_the_milling_row_derives_with_no_orientation_to_choose():
    row = LamellaPoseRowWidget("MILLING", None)
    asked = []
    row.derive_clicked.connect(lambda name, o: asked.append((name, o)))

    row.btn_derive.click()

    assert asked == [("MILLING", None)]


def test_several_orientations_are_offered_as_a_menu():
    row = LamellaPoseRowWidget("FLUORESCENCE", None, derive_orientations=["FM", "SEM"])
    asked = []
    row.derive_clicked.connect(lambda name, o: asked.append((name, o)))

    actions = row._build_derive_menu().actions()
    assert [a.text() for a in actions] == [
        "Derive into the FM orientation",
        "Derive into the SEM orientation",
    ]
    actions[1].trigger()

    assert asked == [("FLUORESCENCE", "SEM")]


def test_a_pose_with_no_counterpart_cannot_be_derived():
    row = LamellaPoseRowWidget("LANDING", None)

    assert not row.btn_derive.isEnabled()


def test_the_list_hands_the_fm_orientations_to_the_fluorescence_row_only(tmp_path):
    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)
    widget = LamellaPoseListWidget()
    widget.set_fluorescence_orientations(["FM", "SEM"])
    widget.set_lamella(lamella)

    rows = _rows(widget)
    assert rows[FLUORESCENCE_POSE].derive_orientations == ["FM", "SEM"]
    assert rows[MILLING_POSE].derive_orientations == []


# ── the action ───────────────────────────────────────────────────────────


def _answer(monkeypatch, answer=QMessageBox.Yes):
    asked = []

    def question(parent, title, text, *args, **kwargs):
        asked.append(text)
        return answer

    monkeypatch.setattr(pose_actions.QMessageBox, "question", question)
    return asked


def test_the_question_says_when_it_overwrites_a_pose_set_by_hand(tmp_path):
    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)

    assert "set by hand" not in derivation_question(lamella, FLUORESCENCE_POSE)
    assert "set by hand" in derivation_question(lamella, MILLING_POSE)


def test_deriving_overwrites_an_observed_pose_once_asked(tmp_path, monkeypatch):
    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)
    experiment = _experiment(tmp_path, lamella)
    centred = _at(microscope, "FM", x=103e-6)
    move_pose(microscope, lamella, FLUORESCENCE_POSE, position=centred)
    announced = []
    experiment.positions.events.changed.connect(lambda *a: announced.append(True))
    asked = _answer(monkeypatch)

    done = pose_actions.derive_lamella_pose(
        None, microscope, experiment, lamella, FLUORESCENCE_POSE
    )

    assert done is True
    assert "set by hand" in asked[0]
    assert lamella.fluorescence_pose.stage_position.x == pytest.approx(100e-6)
    assert lamella.provenance_of(FLUORESCENCE_POSE) is PoseProvenance.DERIVED
    assert announced, "the overview canvases re-mark from this"
    assert os.path.exists(os.path.join(str(experiment.path), "experiment.yaml"))


def test_saying_no_writes_nothing(tmp_path, monkeypatch):
    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)
    centred = _at(microscope, "FM", x=103e-6)
    move_pose(microscope, lamella, FLUORESCENCE_POSE, position=centred)
    _answer(monkeypatch, QMessageBox.No)

    done = pose_actions.derive_lamella_pose(
        None, microscope, None, lamella, FLUORESCENCE_POSE
    )

    assert done is False
    assert lamella.fluorescence_pose.stage_position.x == pytest.approx(103e-6)
    assert lamella.provenance_of(FLUORESCENCE_POSE) is PoseProvenance.OBSERVED


def test_deriving_the_milling_pose_updates_the_milling_angle(tmp_path, monkeypatch):
    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)
    lamella.milling_angle = 999.0
    _answer(monkeypatch)

    assert pose_actions.derive_lamella_pose(
        None, microscope, None, lamella, MILLING_POSE
    )

    assert lamella.provenance_of(MILLING_POSE) is PoseProvenance.DERIVED
    assert lamella.milling_angle != 999.0


def test_a_derivation_the_instrument_refuses_writes_nothing(tmp_path, monkeypatch):
    """On an offset mount, a fluorescence pose standing at the beams is not somewhere
    the objective sees the sample from, so no milling pose is derived from it."""
    microscope = _microscope(IFLM_CONFIG)
    lamella = _lamella(microscope, tmp_path)
    lamella.fluorescence_pose.stage_position = _at(microscope, "FIB")
    before = lamella.milling_pose.stage_position.x
    _answer(monkeypatch)

    done = pose_actions.derive_lamella_pose(
        None, microscope, None, lamella, MILLING_POSE
    )

    assert done is False
    assert lamella.milling_pose.stage_position.x == pytest.approx(before)
    assert lamella.provenance_of(MILLING_POSE) is PoseProvenance.OBSERVED


def test_a_lamella_with_nothing_to_derive_from_is_not_asked(tmp_path, monkeypatch):
    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)
    del lamella.poses[FLUORESCENCE_POSE]
    asked = _answer(monkeypatch)

    assert not pose_actions.derive_lamella_pose(
        None, microscope, None, lamella, MILLING_POSE
    )
    assert asked == []


def test_without_an_fm_nothing_is_derived_and_nothing_is_asked(tmp_path, monkeypatch):
    """No second side to derive between. Said before asking, and nothing written --
    a fluorescence pose there is left over from another system, and a milling pose
    derived from it unchecked is the dangerous outcome."""
    microscope = _microscope(IFLM_CONFIG)
    lamella = _lamella(microscope, tmp_path)
    microscope.fm = None
    before = {name: pose.stage_position.x for name, pose in lamella.poses.items()}
    asked = _answer(monkeypatch)

    for name in (MILLING_POSE, FLUORESCENCE_POSE):
        assert not pose_actions.derive_lamella_pose(
            None, microscope, None, lamella, name
        )
        assert not derive_pose(microscope, lamella, name)

    assert asked == []
    assert {n: p.stage_position.x for n, p in lamella.poses.items()} == before


def test_the_question_says_where_the_objective_position_comes_from(tmp_path):
    microscope = _microscope()
    lamella = _lamella(microscope, tmp_path)

    assert "objective position is kept" in derivation_question(
        lamella, FLUORESCENCE_POSE
    )

    lamella.fluorescence_pose.objective_position = None

    question = derivation_question(lamella, FLUORESCENCE_POSE)
    assert "is kept" not in question
    assert "saved focus position" in question


def test_the_list_puts_the_disagreement_on_a_derived_milling_pose(tmp_path):
    """A target found in fluorescence: the milling pose is the guess, so that is the
    row the distance goes on -- it is the one to derive again."""
    microscope = _microscope()
    poses = build_lamella_poses(microscope, _at(microscope, "FM"))
    lamella = Lamella(petname="Lamella-01", path=str(tmp_path / "L01"), number=1)
    poses.write_to(lamella)
    assert lamella.provenance_of(MILLING_POSE) is PoseProvenance.DERIVED
    widget = LamellaPoseListWidget()
    widget.set_lamella(lamella)

    widget.set_pose_disagreement(32e-6)

    rows = _rows(widget)
    assert rows[MILLING_POSE].provenance_label.text() == "32 µm off"
    assert rows[FLUORESCENCE_POSE].provenance_label.isHidden()


# ── the panel after a move made somewhere else ──────────────────────────


@pytest.fixture
def window(tmp_path, monkeypatch):
    """A real AutoLamellaUI on the simulated Arctis, one lamella selected in it.

    Both poses observed, so moving one leaves the other where it was and the two
    disagree."""
    from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI

    ui = AutoLamellaUI(parent_ui=None)
    monkeypatch.setattr(
        ui.system_widget,
        "load_configuration",
        lambda configuration_name=None: ARCTIS_CONFIG,
    )
    ui.system_widget.connect_to_microscope()
    lamella = _lamella(ui.microscope, tmp_path)
    lamella.fluorescence_pose = lamella.fluorescence_pose  # centred by hand
    ui.experiment = _experiment(tmp_path, lamella)
    ui.update_lamella_combobox()
    ui.update_lamella_ui()
    yield ui, lamella
    ui.experiment = None
    ui.microscope.disconnect()
    ui.close()


def test_a_move_on_a_canvas_redraws_the_selected_lamella(window):
    """The overview canvases announce a move through the experiment's `changed`
    event, and the window's handler for it is the one place that hears every move --
    so it redraws the selected lamella's pose rows too, not only the canvases."""
    from copy import deepcopy

    from fibsem.applications.autolamella.ui.AutoLamellaMainUI import (
        AutoLamellaSingleWindowUI,
    )

    ui, lamella = window
    rows = _rows(ui.selected_lamella_widget.pose_list)
    assert rows[FLUORESCENCE_POSE].provenance_label.isHidden()

    moved = deepcopy(lamella.milling_pose.stage_position)
    moved.x += 30e-6
    move_pose(ui.microscope, lamella, MILLING_POSE, position=moved)
    host = type("_Window", (), {"autolamella_ui": ui})()
    AutoLamellaSingleWindowUI._refresh_overview_positions(host)

    rows = _rows(ui.selected_lamella_widget.pose_list)
    assert rows[FLUORESCENCE_POSE].provenance_label.text() == "30 µm off"


def test_a_move_in_the_main_window_redraws_the_coincidence_viewer(
    window, tmp_path, monkeypatch
):
    """The coincidence viewer shows the same pose rows in its own window and does not
    subscribe to the experiment itself, so the main window's handler tells it -- or
    a pose moved in the main window is left showing its old distance there."""
    from copy import deepcopy

    from fibsem.applications.autolamella.ui.AutoLamellaMainUI import (
        AutoLamellaSingleWindowUI,
    )
    from fibsem.applications.autolamella.ui.fluorescence_coincidence_viewer_widget import (
        FluorescenceCoincidenceViewerWidget,
    )

    ui, lamella = window
    # closing the viewer writes its milling configuration; keep that out of the tree
    monkeypatch.setattr(
        cfg,
        "COINCIDENCE_MILLING_CONFIG_PATH",
        str(tmp_path / "coincidence-milling-config.yaml"),
    )
    viewer = FluorescenceCoincidenceViewerWidget(
        microscope=ui.microscope, experiment=ui.experiment
    )
    viewer.show()
    ui._coincidence_viewer_window = viewer
    try:
        rows = _rows(viewer.selected_lamella_widget.pose_list)
        assert rows[FLUORESCENCE_POSE].provenance_label.isHidden()

        moved = deepcopy(lamella.milling_pose.stage_position)
        moved.x += 30e-6
        move_pose(ui.microscope, lamella, MILLING_POSE, position=moved)
        host = type("_Window", (), {"autolamella_ui": ui})()
        AutoLamellaSingleWindowUI._refresh_overview_positions(host)

        rows = _rows(viewer.selected_lamella_widget.pose_list)
        assert rows[FLUORESCENCE_POSE].provenance_label.text() == "30 µm off"
    finally:
        viewer.close()
        ui._coincidence_viewer_window = None
