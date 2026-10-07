"""The quad view's fourth cell: a page selector, and the chamber drawing it opens on."""

import glob
import math
import os

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from fibsem import utils
from fibsem.microscopes._stage import _create_sample_stage
from fibsem.structures import FibsemStagePosition
from fibsem.ui.widgets.canvas.chamber_view import is_half_turn
from fibsem.ui.widgets.canvas.quad_view import (
    LamellaEditorView,
    MicroscopeViewController,
)


def _microscope(compustage: bool):
    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.stage_device.compustage = compustage
    microscope._stage = _create_sample_stage(microscope)
    if compustage:
        microscope.system.stage.shuttle_pre_tilt = 0
    return microscope


@pytest.fixture
def microscope():
    return _microscope(compustage=False)


@pytest.fixture
def controller(qapp):
    return MicroscopeViewController()


def _show(controller, microscope, orientation: str):
    microscope.move_to_orientation(orientation)
    position = microscope.get_stage_position()
    controller.update_info(microscope, stage_position=position)
    return controller.widget.chamber_view


def test_cell_opens_on_the_chamber_page_with_nothing_drawn(controller):
    cell = controller.widget.page_cell
    assert cell.page == "chamber"
    assert cell.selector.currentData() == "chamber"
    assert not controller.widget.chamber_view.has_position


@pytest.mark.parametrize(
    "orientation, mirrored, milling_angle",
    [("SEM", False, 38.0), ("MILLING", False, 15.0), ("FIB", True, 90.0)],
)
def test_chamber_follows_the_stage_update(
    controller, microscope, orientation, mirrored, milling_angle
):
    chamber = _show(controller, microscope, orientation)
    diagram = chamber.diagram

    assert chamber.has_position
    assert diagram._orientation == orientation
    assert diagram._mirrored is mirrored
    assert diagram._pre_tilt == microscope.system.stage.shuttle_pre_tilt
    assert diagram.milling_angle() == pytest.approx(milling_angle, abs=0.5)
    assert not diagram._show_readout
    assert "Schematic" in chamber.scene.toolTip()


def test_sem_orientation_is_square_to_the_electron_beam(controller, microscope):
    chamber = _show(controller, microscope, "SEM")
    assert chamber.diagram.surface_tilt() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("orientation", ["SEM", "MILLING", "FIB"])
def test_compustage_never_mirrors(controller, qapp, orientation):
    """No rotation axis: the FIB orientation is a tilt, and the beam meets the back."""
    microscope = _microscope(compustage=True)
    chamber = _show(controller, microscope, orientation)
    assert chamber.diagram._mirrored is False
    if orientation == "FIB":
        assert chamber.diagram.milling_angle() == pytest.approx(90.0, abs=0.5)


@pytest.mark.parametrize("orientation, mirrored", [("SEM", False), ("FIB", True)])
def test_mirroring_is_measured_from_the_rotation_reference(
    controller, microscope, orientation, mirrored
):
    """The reference is a system setting, not zero: SEM sits at it, FIB half a turn on."""
    microscope.system.stage.rotation_reference = 120
    microscope._update_orientations()
    chamber = _show(controller, microscope, orientation)
    assert chamber.diagram._mirrored is mirrored


_CONFIGS = sorted(
    glob.glob(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "fibsem",
            "config",
            "*-configuration.yaml",
        )
    )
)


@pytest.mark.parametrize("path", _CONFIGS, ids=os.path.basename)
def test_every_shipped_configuration_draws_its_orientations(qapp, path):
    """Each vendor's numbers -- Tescan's 55 degree column and 180 degree reference, a
    compustage's FIB by tilting -- give the angles the orientations are defined by."""
    microscope, _ = utils.setup_session(config_path=path, manufacturer="Demo")
    controller = MicroscopeViewController()
    angles = {}
    for orientation in ("SEM", "MILLING", "FIB"):
        diagram = _show(controller, microscope, orientation).diagram
        assert diagram.FIB_ANGLE == microscope.system.ion.column_tilt
        angles[orientation] = diagram.milling_angle()
    assert angles["MILLING"] == pytest.approx(
        microscope.system.stage.milling_angle, abs=0.5
    )
    assert angles["FIB"] == pytest.approx(90.0, abs=0.5)


def test_ion_column_drawn_at_the_systems_angle(controller, microscope):
    microscope.system.ion.column_tilt = 55
    chamber = _show(controller, microscope, "SEM")
    assert chamber.diagram.FIB_ANGLE == 55.0
    # The class default is untouched for the wizard's own diagram.
    assert type(chamber.diagram).FIB_ANGLE == 52.0


def test_a_position_without_tilt_keeps_the_last_drawing(controller, microscope):
    chamber = _show(controller, microscope, "MILLING")
    before = chamber.diagram.effective_stage_tilt()
    chamber.set_stage(
        FibsemStagePosition(x=0, y=0, z=0, r=0, t=None),
        orientation="NONE",
        pre_tilt=35,
        column_tilt=52,
        rotation_reference=0,
    )
    assert chamber.diagram.effective_stage_tilt() == before


def test_lamella_editor_view_has_no_chamber(qapp, microscope):
    controller = MicroscopeViewController(view=LamellaEditorView())
    controller.update_info(microscope, stage_position=microscope.get_stage_position())


@pytest.mark.parametrize(
    "r_degrees, reference, expected",
    [
        (0, 0, False),
        (180, 0, True),
        (350, 0, False),
        (120, 120, False),
        (300, 120, True),
        (0, 120, True),
        (91, 0, True),
    ],
)
def test_is_half_turn_wraps(r_degrees, reference, expected):
    assert is_half_turn(math.radians(r_degrees), reference) is expected
