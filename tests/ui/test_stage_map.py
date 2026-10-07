"""The chamber view's stage map: placement, zoom, the inset and its swap."""

import math
from copy import deepcopy
from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from PyQt5.QtCore import QRectF

from fibsem import utils
from fibsem.applications.autolamella.ui.AutoLamellaMainUI import (
    AutoLamellaSingleWindowUI,
)
from fibsem.structures import FibsemStagePosition, SlotCalibration
from fibsem.ui.widgets.canvas.chamber_view import MAP, SIDE
from fibsem.ui.widgets.canvas.overlays.stage_context import holder_slots
from fibsem.ui.widgets.canvas.quad_view import (
    LamellaEditorView,
    MicroscopeViewController,
)
from fibsem.ui.widgets.canvas.stage_map import ZOOM_GRID, ZOOM_HOLDER, ZOOM_TRAVEL

RECT = QRectF(0, 0, 400, 300)


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo")
    return microscope


def _calibrate(microscope, xs=(-2.5e-3, 2.5e-3)):
    stage = microscope.system.stage
    z = microscope.get_stage_position().z
    for slot, x in zip(holder_slots(microscope), xs):
        slot.position = FibsemStagePosition(name=slot.name, x=x, y=0.0, z=z)
        slot.calibration = SlotCalibration(
            orientation="SEM",
            pre_tilt=stage.shuttle_pre_tilt,
            rotation_reference=stage.rotation_reference,
        )


@pytest.fixture
def controller(qapp, microscope):
    controller = MicroscopeViewController()
    controller.widget.resize(900, 700)
    return controller


def _update(controller, microscope):
    controller.update_info(microscope, stage_position=microscope.get_stage_position())
    return controller.widget.chamber_view


def test_the_map_follows_the_stage_update(controller, microscope):
    chamber = _update(controller, microscope)
    stage_map = chamber.map
    assert stage_map._microscope is microscope
    assert stage_map._stage == microscope.get_stage_position()
    # The origin's z is fixed by the first position, and the frame is the SEM pose.
    assert stage_map._origin.z == microscope.get_stage_position().z
    assert stage_map._origin.r == microscope.get_orientation("SEM").r


def test_slots_only_for_a_calibrated_holder(controller, microscope):
    stage_map = _update(controller, microscope).map
    assert stage_map._slot_points() == []
    _calibrate(microscope)
    assert len(stage_map._slot_points()) == 2


def test_travel_zoom_fits_the_limits_and_closer_steps_follow_the_stage(
    controller, microscope
):
    _calibrate(microscope)
    microscope.move_stage_absolute(
        FibsemStagePosition(x=1e-3, y=-0.5e-3, z=microscope.get_stage_position().z)
    )
    stage_map = _update(controller, microscope).map

    box = stage_map._travel_box()
    centre, travel_scale = stage_map._view(RECT, ZOOM_TRAVEL)
    assert centre == pytest.approx(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))

    stage = stage_map._plane(stage_map._stage)
    scales = [travel_scale]
    for zoom in (ZOOM_HOLDER, ZOOM_GRID):
        centre, scale = stage_map._view(RECT, zoom)
        assert centre == pytest.approx(stage)
        scales.append(scale)
    assert scales == sorted(scales), "each step closer than the last"


def test_zoom_steps_clamp(controller, microscope):
    stage_map = _update(controller, microscope).map
    stage_map.zoom_out()
    assert stage_map.zoom == ZOOM_TRAVEL
    for _ in range(5):
        stage_map.zoom_in()
    assert stage_map.zoom == ZOOM_GRID


def test_a_lamella_from_the_other_side_is_drawn_under_the_stage(controller, microscope):
    """A lamella saved at MILLING, the stage at FIB over it: one place on the map, not
    one each side of the grid."""
    _calibrate(microscope)
    lamella = deepcopy(microscope.get_orientation("MILLING"))
    lamella.x, lamella.y = -2.8e-3, 0.2e-3
    lamella.z = microscope.get_stage_position().z
    microscope.move_stage_absolute(microscope.get_target_position(lamella, "FIB"))
    stage_map = _update(controller, microscope).map

    at_fib = microscope.get_stage_position()
    assert math.degrees(at_fib.r) == pytest.approx(180, abs=1)
    assert math.dist(stage_map._plane(lamella), stage_map._plane(at_fib)) < 1e-6


def test_the_inset_swaps_the_views(controller, microscope):
    chamber = _update(controller, microscope)
    scene = chamber.scene
    assert scene.main == SIDE
    assert scene.zoom_in_button.isHidden()

    scene.swap()
    assert scene.main == MAP
    assert not scene.map.isHidden() and scene.diagram.isHidden()
    assert not scene.zoom_in_button.isHidden()

    scene.swap()
    assert scene.main == SIDE


@pytest.mark.parametrize("main", [SIDE, MAP])
def test_both_views_draw_in_both_places(controller, microscope, main):
    _calibrate(microscope)
    chamber = _update(controller, microscope)
    chamber.set_positions([microscope.get_stage_position()])
    chamber.scene.resize(600, 450)
    if main == MAP:
        chamber.scene.swap()
        chamber.map.set_zoom(ZOOM_GRID)
    assert not chamber.scene.grab().isNull()
    assert not chamber.scene.inset.grab().isNull()


def test_the_lamella_editor_has_no_map(qapp, microscope):
    MicroscopeViewController(view=LamellaEditorView()).set_map_positions(
        [microscope.get_stage_position()]
    )


def test_the_window_hands_the_map_the_experiment_s_lamellae(controller, microscope):
    """`_refresh_overview_positions` is the one place that hears the lamellae change;
    the map is marked from it, like the overview tabs."""
    positions = [microscope.get_stage_position(), microscope.get_orientation("FIB")]
    window = SimpleNamespace(
        view_controller=controller,
        autolamella_ui=SimpleNamespace(
            experiment=SimpleNamespace(
                positions=[SimpleNamespace(stage_position=p) for p in positions]
            )
        ),
    )
    AutoLamellaSingleWindowUI._refresh_overview_positions(window)
    assert controller.widget.chamber_view.map._positions == positions

    window.autolamella_ui.experiment = None
    AutoLamellaSingleWindowUI._refresh_overview_positions(window)
    assert controller.widget.chamber_view.map._positions == []
