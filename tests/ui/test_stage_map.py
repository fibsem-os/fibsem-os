"""The chamber view's stage map: placement, zoom, the inset and its swap."""

import math
import os
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
from fibsem.ui.widgets.canvas.overlays.minimap_overlays import GRID_BOUNDARY_RADIUS_M
from fibsem.ui.widgets.canvas.overlays.stage_context import holder_slots
from fibsem.ui.widgets.canvas.quad_view import (
    LamellaEditorView,
    MicroscopeViewController,
)
from fibsem.ui.widgets.canvas.stage_map import ZOOM_GRID, ZOOM_HOLDER, ZOOM_TRAVEL

RECT = QRectF(0, 0, 400, 300)


def _config(name: str) -> str:
    return os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "fibsem",
        "config",
        f"{name}-configuration.yaml",
    )


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


def _visible(stage_map, zoom):
    """The plane rectangle a zoom step shows RECT as: (min x, min y, max x, max y)."""
    (cx, cy), scale = stage_map._view(RECT, zoom)
    half_w, half_h = RECT.width() / 2 / scale, RECT.height() / 2 / scale
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h


def _inside(point, box, pad=0.0):
    return (
        box[0] <= point[0] - pad
        and point[0] + pad <= box[2]
        and box[1] <= point[1] - pad
        and point[1] + pad <= box[3]
    )


def test_closer_steps_follow_the_stage(controller, microscope):
    _calibrate(microscope)
    microscope.move_stage_absolute(
        FibsemStagePosition(x=1e-3, y=-0.5e-3, z=microscope.get_stage_position().z)
    )
    stage_map = _update(controller, microscope).map

    stage = stage_map._plane(stage_map._stage)
    scales = [stage_map._view(RECT, ZOOM_TRAVEL)[1]]
    for zoom in (ZOOM_HOLDER, ZOOM_GRID):
        centre, scale = stage_map._view(RECT, zoom)
        assert centre == pytest.approx(stage)
        scales.append(scale)
    assert scales == sorted(scales), "each step closer than the last"


def test_travel_zoom_reaches_a_stage_out_at_the_load_position(controller, microscope):
    """The view fits where the stage is, not the limits: out at y = 30 mm, the grids
    and the stage are both in it."""
    _calibrate(microscope)
    z = microscope.get_stage_position().z
    microscope.move_stage_absolute(FibsemStagePosition(x=0.0, y=30e-3, z=z))
    stage_map = _update(controller, microscope).map

    box = _visible(stage_map, ZOOM_TRAVEL)
    assert _inside(stage_map._plane(stage_map._stage), box)
    for slot in stage_map._slot_points():
        assert _inside(slot, box, pad=GRID_BOUNDARY_RADIUS_M)
    assert max(box[2] - box[0], box[3] - box[1]) < 100e-3, "not the +/-100 mm limits"


def test_travel_zoom_is_capped_and_keeps_the_stage_in_view(controller, microscope):
    """Past 100 mm across the view stops growing, and slides to keep the stage in."""
    _calibrate(microscope)
    z = microscope.get_stage_position().z
    microscope.move_stage_absolute(FibsemStagePosition(x=95e-3, y=-70e-3, z=z))
    stage_map = _update(controller, microscope).map

    box = _visible(stage_map, ZOOM_TRAVEL)
    assert max(box[2] - box[0], box[3] - box[1]) == pytest.approx(100e-3)
    assert _inside(stage_map._plane(stage_map._stage), box)


def test_travel_zoom_is_never_closer_than_20_mm(controller, microscope):
    stage_map = _update(controller, microscope).map
    box = _visible(stage_map, ZOOM_TRAVEL)
    assert min(box[2] - box[0], box[3] - box[1]) >= 20e-3 - 1e-9


def test_no_station_where_every_device_shares_the_beams_place(controller, microscope):
    assert _update(controller, microscope).map._device_stations() == []


@pytest.mark.parametrize("pose", ["MILLING", "FIB"])
def test_an_offset_fm_station_is_drawn_where_the_stage_goes_for_it(qapp, pose):
    """The stage sent to the FM, in either pose, is drawn at the station marker: the
    station is a place in the chamber, and turns over with the stage on the map."""
    microscope, _ = utils.setup_session(
        manufacturer="Demo", config_path=_config("sim-iflm")
    )
    controller = MicroscopeViewController()
    lamella = deepcopy(microscope.get_orientation("MILLING"))
    lamella.x, lamella.y = -0.3e-3, 0.2e-3
    lamella.z = microscope.get_stage_position().z
    at_fm = microscope.get_target_position(
        lamella,
        target_orientation=None if pose == "MILLING" else pose,
        target_device="FM",
    )
    microscope.move_stage_absolute(at_fm)
    stage_map = _update(controller, microscope).map
    assert microscope.get_current_device(microscope.get_stage_position()) == "FM"

    stations = stage_map._device_stations()
    assert [name for name, _ in stations] == ["FM"]
    station = stage_map._plane(stations[0][1])
    stage = stage_map._plane(stage_map._stage)
    assert math.dist(station, stage) < 1e-3, "the lamella's offset, not 100 mm"
    assert _inside(station, _visible(stage_map, ZOOM_TRAVEL))


def test_travel_zoom_shows_a_compustage_grid_whole(qapp):
    """A compustage's limits can sit inside its grid; fitting the limits alone cropped
    the grid in the inset."""
    microscope, _ = utils.setup_session(
        manufacturer="Demo",
        config_path=_config("sim-arctis"),
    )
    controller = MicroscopeViewController()
    stage_map = _update(controller, microscope).map
    slots = stage_map._slot_points()
    assert slots, "the compustage's working slot is calibrated as built"

    box = _visible(stage_map, ZOOM_TRAVEL)
    for slot in slots:
        assert _inside(slot, box, pad=GRID_BOUNDARY_RADIUS_M)


def test_the_lamella_under_the_stage_is_the_one_ringed(controller, microscope):
    here = microscope.get_stage_position()
    elsewhere = deepcopy(here)
    elsewhere.x += 0.3e-3
    controller.set_map_positions([elsewhere, here])
    stage_map = _update(controller, microscope).map
    assert stage_map.lamellae_here() == [here]


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


def test_places_outside_the_view_are_pointed_to(controller, microscope):
    """At grid zoom on one slot, the other slot is off the view, and only it."""
    _calibrate(microscope)
    slot = holder_slots(microscope)[0].position
    microscope.move_stage_absolute(FibsemStagePosition(x=slot.x, y=slot.y, z=slot.z))
    stage_map = _update(controller, microscope).map

    names = [name for name, _ in stage_map.offscreen_places(RECT, ZOOM_GRID)]
    assert names == [holder_slots(microscope)[1].name]
    assert stage_map.offscreen_places(RECT, ZOOM_TRAVEL) == []


def test_the_fm_station_is_pointed_to_from_the_beams(qapp):
    microscope, _ = utils.setup_session(
        manufacturer="Demo", config_path=_config("sim-iflm")
    )
    stage_map = _update(MicroscopeViewController(), microscope).map

    held = dict(stage_map.offscreen_places(RECT, ZOOM_HOLDER))
    assert "FM" in held
    # To the right: the station is +48.8 mm in x, and the map is not mirrored here.
    assert held["FM"][0] > stage_map._plane(stage_map._stage)[0]
    assert "FM" not in dict(stage_map.offscreen_places(RECT, ZOOM_TRAVEL))
