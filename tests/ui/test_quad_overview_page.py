"""The quad view's overview page: which grid, which overviews, what is marked."""

import os
from copy import deepcopy
from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from PyQt5.QtWidgets import QApplication, QPushButton

from fibsem import utils
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
    GridRecord,
)
from fibsem.applications.autolamella.ui.AutoLamellaMainUI import (
    AutoLamellaSingleWindowUI,
)
from fibsem.applications.autolamella.ui.quad_overview_page import (
    CURRENT,
    NO_GRID,
    QuadOverviewPage,
)
from fibsem.config import AUTOLAMELLA_TASK_PROTOCOL_PATH
from fibsem.structures import (
    BeamType,
    FibsemStagePosition,
    ImageSettings,
    MicroscopeState,
    SampleGrid,
    SlotCalibration,
)
from fibsem.ui.widgets.canvas.overlays.stage_context import holder_slots

SLOT_X = {"Grid-A": -2.5e-3, "Grid-B": 2.5e-3}


def _calibrate(microscope):
    stage = microscope.system.stage
    z = microscope.get_stage_position().z
    for slot, name in zip(holder_slots(microscope), SLOT_X):
        slot.position = FibsemStagePosition(name=slot.name, x=SLOT_X[name], y=0.0, z=z)
        slot.calibration = SlotCalibration(
            orientation="SEM",
            pre_tilt=stage.shuttle_pre_tilt,
            rotation_reference=stage.rotation_reference,
        )
        slot.loaded_grid = SampleGrid(name=name)


def _at(microscope, x, orientation="SEM"):
    """A stage position over x (at y=0), at an orientation."""
    sem = deepcopy(microscope.get_orientation("SEM"))
    sem.x, sem.y, sem.z = x, 0.0, microscope.get_stage_position().z
    if orientation == "SEM":
        return sem
    return microscope.get_target_position(sem, orientation)


def _overview(microscope, experiment, name, position, beam):
    microscope.move_stage_absolute(position)
    image = microscope.acquire_image(
        ImageSettings(hfw=1e-3, resolution=[64, 48], beam_type=beam, save=False)
    )
    return image.save(os.path.join(str(experiment.path), f"{name}.tif"))


@pytest.fixture
def world(qapp, tmp_path):
    microscope, _ = utils.setup_session(manufacturer="Demo")
    _calibrate(microscope)
    experiment = Experiment.create(path=tmp_path, name="quad-overview")
    experiment.task_protocol = AutoLamellaTaskProtocol.load(
        AUTOLAMELLA_TASK_PROTOCOL_PATH
    )
    grids = {
        name: experiment.add_grid(GridRecord(name=name))
        for name in ("Grid-A", "Grid-B", "Grid-C")
    }
    # Root overviews, as the Overview tab saves them: no grid recorded.
    _overview(
        microscope,
        experiment,
        "overview-image-a-sem",
        _at(microscope, SLOT_X["Grid-A"]),
        BeamType.ELECTRON,
    )
    # Taken with the stage turned a half turn: its raw x is on Grid-B's side.
    _overview(
        microscope,
        experiment,
        "overview-image-a-fib",
        _at(microscope, SLOT_X["Grid-A"], "FIB"),
        BeamType.ION,
    )
    _overview(
        microscope,
        experiment,
        "overview-image-b-sem",
        _at(microscope, SLOT_X["Grid-B"]),
        BeamType.ELECTRON,
    )
    _overview(
        microscope,
        experiment,
        "overview-image-off-grid",
        _at(microscope, 20e-3),
        BeamType.ELECTRON,
    )
    for grid, dx in (("Grid-A", -0.3e-3), ("Grid-A", 0.2e-3), ("Grid-B", 0.1e-3)):
        position = deepcopy(microscope.get_orientation("MILLING"))
        position.x, position.y = SLOT_X[grid] + dx, 0.1e-3
        position.z = microscope.get_stage_position().z
        experiment.add_new_lamella(
            microscope_state=MicroscopeState(stage_position=position),
            task_config=experiment.task_protocol.task_config,
            grid_id=grids[grid].id,
        )
    page = QuadOverviewPage()
    page.resize(600, 450)
    page.show()  # a hidden page only notes that it is out of date
    page.set_microscope(microscope)
    page.set_experiment(experiment)
    return SimpleNamespace(
        microscope=microscope, experiment=experiment, grids=grids, page=page
    )


def _views(page, grid):
    return sorted(o.view.split(" @ ")[0] for o in page._index.get(grid, []))


def test_root_overviews_are_placed_on_the_grid_they_were_taken_on(world):
    """Including the one taken half a turn round, whose raw x is on the other side."""
    page, grids = world.page, world.grids
    assert _views(page, grids["Grid-A"].id) == ["FIB", "SEM"]
    assert _views(page, grids["Grid-B"].id) == ["SEM"]
    assert _views(page, NO_GRID) == ["SEM"], "off the grids: left out, not guessed"


def test_it_follows_the_stage_and_keeps_the_last_grid(world):
    page, grids, microscope = world.page, world.grids, world.microscope
    page.set_stage(_at(microscope, SLOT_X["Grid-B"]))
    assert page.shown_grid() == grids["Grid-B"].id
    assert "Grid-B" in page.grid_selector.itemText(0)

    page.set_stage(_at(microscope, 20e-3))
    assert page.shown_grid() == grids["Grid-B"].id, "on no grid: keep the last"


def test_a_pinned_grid_stays_and_drops_the_stage_marker(world):
    page, grids, microscope = world.page, world.grids, world.microscope
    page.set_stage(_at(microscope, SLOT_X["Grid-A"]))
    assert page.canvas._current is not None

    page.grid_selector.setCurrentIndex(page.grid_selector.findData(grids["Grid-B"].id))
    page._on_grid_chosen(page.grid_selector.currentIndex())
    assert page.shown_grid() == grids["Grid-B"].id
    assert page.canvas._current is None, "the stage is on Grid-A, not this one"

    page.set_stage(_at(microscope, SLOT_X["Grid-A"]))
    assert page.shown_grid() == grids["Grid-B"].id


def test_only_the_shown_grid_s_lamellae_are_marked(world):
    page, microscope, experiment = world.page, world.microscope, world.experiment
    page.set_stage(_at(microscope, SLOT_X["Grid-A"]))
    marked = sorted(p.name for p in page.canvas._positions)
    on_a = sorted(
        lam.name for lam in experiment.get_lamellae_for_grid(world.grids["Grid-A"])
    )
    assert marked == on_a and len(marked) == 2


def test_one_chip_per_view_and_the_kind_of_view_is_kept(world):
    page, microscope = world.page, world.microscope
    page.set_stage(_at(microscope, SLOT_X["Grid-A"]))
    assert sorted(page._view_by_group) == ["FIB", "SEM"]
    fib = page._view_by_group["FIB"]
    page._on_chip_clicked(fib)
    assert page.canvas.view == fib

    page.set_stage(_at(microscope, SLOT_X["Grid-B"]))
    assert page.canvas.view.startswith("SEM"), "Grid-B has no FIB view: falls back"


def test_clicking_a_lamella_selects_it(world):
    page, microscope = world.page, world.microscope
    page.set_stage(_at(microscope, SLOT_X["Grid-A"]))
    lamella = world.experiment.get_lamellae_for_grid(world.grids["Grid-A"])[0]
    chosen = []
    page.lamella_selected.connect(chosen.append)
    page._on_position_selected(lamella.name)
    assert chosen == [lamella]


@pytest.mark.parametrize(
    "case, expected",
    [
        ("no experiment", "No experiment loaded."),
        ("grid without overviews", "No overview of Grid-C yet."),
    ],
)
def test_empty_states(world, case, expected):
    page = world.page
    if case == "no experiment":
        page.set_experiment(None)
    else:
        page._mode = world.grids["Grid-C"].id
        page.refresh()
    assert page._stack.currentWidget() is page.empty
    assert page.empty.text().startswith(expected)


def test_an_experiment_without_grids_shows_its_own_overviews(qapp, tmp_path):
    microscope, _ = utils.setup_session(manufacturer="Demo")
    experiment = Experiment.create(path=tmp_path, name="no-grids")
    page = QuadOverviewPage()
    page.show()
    page.set_microscope(microscope)
    page.set_experiment(experiment)
    assert page.grid_selector.isHidden()
    assert page.empty.text().startswith("No overviews in this experiment yet.")

    _overview(
        microscope,
        experiment,
        "overview-image-x",
        _at(microscope, 0.0),
        BeamType.ELECTRON,
    )
    page.refresh()
    assert page.shown_grid() == NO_GRID
    assert page._stack.currentWidget() is page.canvas


def test_the_window_feeds_the_page_its_experiment(world):
    """`_refresh_overview_positions` hands a new experiment to the page, and only
    re-marks it for the same one."""
    page = QuadOverviewPage()
    window = SimpleNamespace(
        quad_overview_page=page,
        view_controller=None,
        autolamella_ui=SimpleNamespace(experiment=world.experiment),
    )
    AutoLamellaSingleWindowUI._refresh_overview_positions(window)
    assert page.experiment is world.experiment
    assert page._mode == CURRENT


def _visible_chips(page):
    header = page.grid_selector.parentWidget()
    return sorted(
        button.text()
        for button in header.findChildren(QPushButton)
        if button.isVisibleTo(header)
    )


def test_old_chips_are_gone_at_once_not_left_over_the_selector(world):
    """A replaced chip was out of the layout but still a visible child of the header
    until its deletion ran, drawn over the grid selector."""
    page, microscope = world.page, world.microscope
    page.set_stage(_at(microscope, SLOT_X["Grid-A"]))
    assert _visible_chips(page) == ["FIB", "SEM"]
    page.set_stage(_at(microscope, SLOT_X["Grid-B"]))
    assert _visible_chips(page) == ["SEM"]
    page._mode = world.grids["Grid-C"].id
    page.refresh()
    assert _visible_chips(page) == []


def test_no_ruler_on_the_page(world):
    assert world.page.canvas.canvas.btn_toggle_ruler.isHidden()


def test_the_page_s_controls_share_the_cell_s_header_row(world):
    """The page's own controls switch with the page, in the row above it; the page
    cycler is in the cell's bar (FIB-1186)."""
    from fibsem.ui.widgets.canvas.quad_view import MicroscopeViewController

    controller = MicroscopeViewController()  # kept: it owns the cell
    cell = controller.widget.page_cell
    cell.add_page("overview", "Overview", world.page, header=world.page.header)
    assert world.page.header.parentWidget() is cell._headers
    cell.set_page("chamber")
    assert cell._headers.currentWidget() is not world.page.header
    cell.set_page("overview")
    assert cell._headers.currentWidget() is world.page.header


def test_the_header_stays_at_the_top_of_an_empty_page(world):
    from fibsem.ui.widgets.canvas.quad_view import MicroscopeViewController

    controller = MicroscopeViewController()  # kept: it owns the cell
    cell = controller.widget.page_cell
    cell.add_page("overview", "Overview", world.page, header=world.page.header)
    cell.set_page("overview")
    world.page._mode = world.grids["Grid-C"].id
    world.page.refresh()
    controller.widget.resize(900, 700)
    controller.widget.show()
    QApplication.processEvents()
    assert cell.header.mapTo(controller.widget, cell.header.rect().topLeft()).y() < (
        controller.widget.height() / 2 + 10
    ), "the cell's header sits at the top of its cell, not mid-way down"
    assert cell.header.height() < 40


def test_the_page_cycler_steps_and_wraps_round(world):
    from fibsem.ui.widgets.canvas.quad_view import MicroscopeViewController

    controller = MicroscopeViewController()  # kept: it owns the cell
    cell = controller.widget.page_cell
    assert not cell.btn_next.isEnabled(), "one page: nothing to cycle to"
    cell.add_page("overview", "Overview", world.page, header=world.page.header)
    assert cell.btn_next.isEnabled()
    cell.btn_next.click()
    assert (cell.page, cell.label.text()) == ("overview", "Overview")
    assert cell._headers.currentWidget() is world.page.header
    cell.btn_next.click()
    assert cell.page == "chamber", "past the last page: back to the first"
    cell.btn_previous.click()
    assert cell.page == "overview", "before the first page: the last"


def test_a_hidden_page_does_nothing_until_it_is_shown(world):
    """Behind the chamber page it loads no image and builds no chips; shown, it
    catches up."""
    page, microscope, grids = world.page, world.microscope, world.grids
    page.set_stage(_at(microscope, SLOT_X["Grid-A"]))
    page.hide()
    page.set_stage(_at(microscope, SLOT_X["Grid-B"]))
    assert page._shown[0] == grids["Grid-A"].id, "nothing redrawn while hidden"
    page.show()
    assert page._shown[0] == grids["Grid-B"].id
    assert _visible_chips(page) == ["SEM"]


def test_a_move_on_the_same_grid_only_moves_the_stage_marker(world):
    page, microscope = world.page, world.microscope
    page.set_stage(_at(microscope, SLOT_X["Grid-A"]))
    chips = dict(page._chips)
    page.set_stage(_at(microscope, SLOT_X["Grid-A"] + 0.2e-3))
    assert page._chips == chips, "the chips were not rebuilt"
    assert page.canvas._current.x == pytest.approx(SLOT_X["Grid-A"] + 0.2e-3)


def _boundary_centre(page):
    shapes = page.canvas.boundary_overlay._specs
    return [(shape.cx, shape.cy) for shape in shapes]


@pytest.mark.parametrize("view", ["SEM", "FIB"])
def test_the_grid_s_rim_is_centred_where_the_stage_is_at_its_centre(world, view):
    """In the FIB view too, where the image was taken half a turn round: the rim and
    the stage marker are placed by the same frame, so they agree."""
    page, microscope = world.page, world.microscope
    centre = _at(microscope, SLOT_X["Grid-A"])
    page.set_stage(centre)
    page._on_chip_clicked(page._view_by_group[view])
    ((cx, cy),) = _boundary_centre(page)
    sx, sy = page.canvas._frame().to_canvas(centre)
    assert (cx, cy) == pytest.approx((sx, sy), abs=1e-6)


def test_no_rim_for_a_grid_no_slot_holds(world):
    page, microscope, grids = world.page, world.microscope, world.grids
    for slot in holder_slots(microscope):
        if slot.loaded_grid is not None and slot.loaded_grid.name == "Grid-B":
            slot.loaded_grid = None
    page.grid_selector.setCurrentIndex(page.grid_selector.findData(grids["Grid-B"].id))
    page._on_grid_chosen(page.grid_selector.currentIndex())
    assert page._stack.currentWidget() is page.canvas
    assert _boundary_centre(page) == []
