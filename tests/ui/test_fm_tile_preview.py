"""The FM tile preview pane: what a selection of ground would actually acquire.

The pane's job is to be *right* and to be *inert*. Right, because a grid drawn against
the wrong pose still looks like a grid — the arithmetic behind that is pinned in
`tests/fm/test_planning.py`, and what is pinned here is that the widget uses it. Inert,
because it is a preview inside a dialog and the stage must not move while someone is
deciding where to look.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from fibsem.imaging.tiling.geometry import PlaneRegion
from fibsem.projection import BeamStageProjection, surface_foreshortening
from fibsem.structures import BeamType, FibsemStagePosition
from fibsem.ui.fm.widgets.fm_tile_preview import FMTilePreviewWidget

OVERLAP = 0.1


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def microscope(qapp):
    from fibsem.ui.fm.overview_app import build_microscope

    return build_microscope()


@pytest.fixture()
def pane(microscope):
    widget = FMTilePreviewWidget(microscope)
    widget.setFixedSize(316, 386)
    return widget


def beam_view(microscope, orientation="MILLING", beam_type=BeamType.ION):
    projection = BeamStageProjection.from_microscope(microscope, beam_type)
    pose = microscope.get_orientation(orientation)
    return projection, FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=pose.r, t=pose.t)


def regions(*boxes):
    return [PlaneRegion.from_points([(x0, y0), (x1, y1)]) for x0, y0, x1, y1 in boxes]


# A wide, shallow pair in the beam plane. Shallow on purpose: the ion view at the
# milling pose is foreshortened, so a box that looks nearly square there is a strip of
# ground almost four times taller.
TWO_REGIONS = ((-560e-6, -110e-6, -60e-6, -17e-6), (300e-6, 34e-6, 760e-6, 125e-6))


def select(pane, microscope, *boxes, overlap=OVERLAP):
    projection, base = beam_view(microscope)
    return pane.set_selection(regions(*boxes), base, projection, overlap)


# ── inert ────────────────────────────────────────────────────────────────


def test_nothing_in_the_pane_can_move_the_stage(pane):
    """The safety property, and the reason it is worth a test rather than a comment.

    The canvas only *emits* click signals; every stage move in the FM overview tab lives
    in that tab's handlers, connected to those same signals. So the pane is inert only
    for as long as nobody copies the tab's wiring block across — which is exactly the
    kind of thing that gets copied.
    """
    canvas = pane.canvas.canvas
    assert canvas.receivers(canvas.canvas_clicked) == 0
    assert canvas.receivers(canvas.canvas_double_clicked) == 0


def test_the_grid_cannot_be_dragged_out_of_shape(pane):
    """Rows, columns and centre come from the regions. A grid dragged here would spring
    back on the next recompute, which reads as a broken control rather than an absent
    one."""
    assert pane.tile_grid_overlay._geometry_editable is False


def test_selected_tiles_are_filled_not_just_outlined(pane):
    """`ENABLED_ALPHA` is 0 on the canvas, so enabled tiles are outlines. That reads over
    an image and does not read over nothing, and this pane is nothing but the grid."""
    assert pane.tile_grid_overlay._fill_alpha > 0


# ── right ────────────────────────────────────────────────────────────────


def test_the_frame_is_posed_for_fluorescence_wherever_the_stage_is(microscope):
    """Built with the stage at the milling pose, because that is where a beam overview
    was just acquired. Anchoring on the live position draws every tile 1.086x too tall.
    """
    milling = microscope.get_orientation("MILLING")
    microscope.move_stage_absolute(
        FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=milling.r, t=milling.t)
    )
    widget = FMTilePreviewWidget(microscope)

    assert surface_foreshortening(widget._projection, widget._origin) == pytest.approx(
        1.0, abs=1e-9
    )


def test_the_drawn_grid_is_the_planned_grid(pane, microscope):
    """The widget must not re-derive a shape of its own; it draws what the planner said."""
    plan = select(pane, microscope, *TWO_REGIONS)

    drawn = pane.tile_grid_overlay._tiles
    assert plan is not None
    assert max(t.row for t in drawn) + 1 == plan.rows
    assert max(t.col for t in drawn) + 1 == plan.cols
    assert [
        [t.enabled for t in drawn if t.row == r] for r in range(plan.rows)
    ] == plan.mask


def test_a_bigger_selection_needs_a_bigger_grid(pane, microscope):
    one = select(pane, microscope, TWO_REGIONS[0])
    two = select(pane, microscope, *TWO_REGIONS)

    assert two.rows * two.cols > one.rows * one.cols
    assert pane.enabled_tiles > 0


# ── the frame holds still ────────────────────────────────────────────────


def test_the_working_area_is_declared_once(pane, microscope):
    """A stable frame is what lets a growing selection read as growing, rather than as
    the world shrinking around it. `refit=False` cannot deliver that on a canvas with no
    image: `_fit_extent` falls back to the working area when nothing is placed, so
    re-declaring per selection moves the view anyway. So it is never re-declared.
    """
    at_open = pane.canvas.canvas.world_extent
    select(pane, microscope, TWO_REGIONS[0])
    select(pane, microscope, *TWO_REGIONS)
    pane.clear()

    assert pane.canvas.canvas.world_extent == at_open


def view_centre(pane):
    """The middle of what the pane actually shows, read off the axes."""
    xlim = pane.canvas.canvas._ax.get_xlim()
    ylim = pane.canvas.canvas._ax.get_ylim()
    return ((xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2)


@pytest.mark.parametrize("orientation", ["FM", "MILLING"])
def test_the_view_is_framed_on_the_stage_not_on_wherever_it_was_parked(
    microscope, orientation
):
    """The declared area has to be centred on what is drawn in it.

    Canvas (0, 0) is the *frame* origin -- wherever the stage happened to be standing
    when the dialog opened. Everything this pane draws is placed about the *stage*
    origin instead: the travel limits, the grid boundary, and a plan that follows the
    sample rather than the pose. Framing on canvas (0, 0) therefore slid the whole scene
    by the live stage displacement, which on a compustage reaches a full grid radius --
    enough to push the grid off the edge of the pane.

    Both poses, because the dialog is opened from the milling one: that is where the
    beam overview was just acquired. Reading the framing off the live pose rather than
    the fluorescence-posed origin is wrong there and indistinguishable at FM.
    """
    pose = microscope.get_orientation(orientation)
    microscope.move_stage_absolute(
        FibsemStagePosition(x=-450e-6, y=-300e-6, z=0.0, r=pose.r, t=pose.t)
    )
    pane = FMTilePreviewWidget(microscope)

    [circle] = [s for s in pane.stage_overlay._specs if s.kind == "circle"]

    assert (circle.cx, circle.cy) == pytest.approx(view_centre(pane), abs=1e-6), (
        "the grid boundary is drawn somewhere other than the middle of the view"
    )


def test_a_stage_that_moves_under_the_dialog_does_not_take_the_framing_with_it(
    microscope,
):
    """Framing on the stage origin must not become framing on the *live* stage: the
    origin is fixed for the life of the pane, so the answer cannot depend on when it is
    asked."""
    fm = microscope.get_orientation("FM")
    microscope.move_stage_absolute(
        FibsemStagePosition(x=100e-6, y=50e-6, z=0.0, r=fm.r, t=fm.t)
    )
    pane = FMTilePreviewWidget(microscope)
    at_open = pane.canvas.canvas.world_extent

    microscope.move_stage_absolute(
        FibsemStagePosition(x=-400e-6, y=-250e-6, z=0.0, r=fm.r, t=fm.t)
    )
    select(pane, microscope, *TWO_REGIONS)

    assert pane.canvas.canvas.world_extent == at_open


# ── editing by hand ──────────────────────────────────────────────────────


def test_toggling_a_tile_changes_the_mask_but_not_the_plan(pane, microscope):
    plan = select(pane, microscope, *TWO_REGIONS)
    before = pane.enabled_tiles

    pane._on_tile_toggled(0, 0, not pane.mask[0][0])

    assert pane.enabled_tiles != before
    assert pane.plan is plan, "the plan is what the regions said; a toggle is not"


def test_a_new_selection_discards_tiles_toggled_by_hand(pane, microscope):
    """The grid is re-derived, so a toggle recorded against the old one would land on a
    different tile or on none. Nothing marks a surviving toggle — the overlay draws every
    enabled tile alike — so keeping them would leave tiles on for no visible reason."""
    select(pane, microscope, *TWO_REGIONS)
    pane._on_tile_toggled(0, 0, not pane.mask[0][0])
    edited = pane.mask

    select(pane, microscope, *TWO_REGIONS)

    assert pane.mask != edited


def test_a_toggle_outside_the_current_grid_is_ignored(pane, microscope):
    """The overlay emits against the grid it was last given, and a selection change
    between the press and the release would otherwise write outside the mask."""
    select(pane, microscope, TWO_REGIONS[0])
    before = pane.mask

    pane._on_tile_toggled(999, 999, True)

    assert pane.mask == before


# ── nothing selected ─────────────────────────────────────────────────────


def test_a_region_with_no_area_selects_nothing_rather_than_raising(pane, microscope):
    """A click that never became a drag. A state, not a failure."""
    assert select(pane, microscope, (1e-4, 1e-4, 1e-4, 1e-4)) is None
    assert pane.plan is None
    assert pane.enabled_tiles == 0


def test_clearing_a_selection_leaves_the_stage_context_drawn(pane, microscope):
    """The boundary and limits are the pane's only reachability cue, and they do not
    depend on there being a selection."""
    select(pane, microscope, *TWO_REGIONS)
    pane.clear()

    assert pane.tile_grid_overlay._tiles == []
    assert pane.stage_overlay._specs


# ── how far the stage can reach ──────────────────────────────────────────


def test_the_pane_draws_both_the_travel_limits_and_the_grid_boundary(pane):
    """Both, because on a compustage they answer different questions and neither
    contains the other.

    Travel is +/-999.9 um in x and only +/-377.8 um in y, against a grid boundary of
    1000 um radius -- so in y the stage runs out of travel well inside the grid, and a
    selection can be on the grid and still unreachable. The boundary alone would say it
    was fine.
    """
    labels = {spec.label for spec in pane.stage_overlay._specs}

    assert labels == {"Stage limits", "Grid boundary"}


def test_reachability_is_worked_out_without_asking_the_instrument(pane, microscope):
    """A UI event must not read the microscope, and this one would read it per *tile*.

    `project_fm_stable_move` reads `fm.camera_tilt` and the camera transform on every
    call -- 216 ms each on the simulator, so one per tile on a region change is half a
    minute of frozen UI for a 150 tile grid, plus instrument traffic on a mouse event.
    The held `FMStageProjection` is the same arithmetic over the same geometry and agrees
    to 1.4e-20 m.
    """
    from unittest.mock import patch

    with patch.object(
        type(microscope),
        "project_fm_stable_move",
        side_effect=AssertionError("the preview asked the instrument"),
    ):
        plan = select(pane, microscope, *TWO_REGIONS)

    # It got all the way through, which it could not have done by asking.
    assert plan is not None
    assert pane.enabled_tiles > 0
    # And it did the work rather than skipping it: this selection reaches roughly
    # +/-450 um of ground against +/-377.8 um of travel, so some of it is genuinely out
    # of reach and an empty answer here would mean the check never ran.
    assert pane.unreachable_tiles


def test_the_limits_drawn_are_the_stage_it_was_given(pane, microscope):
    """Read from the instrument rather than assumed, so a stage with different travel
    draws a different box."""
    limits = microscope._stage.limits
    [box] = [s for s in pane.stage_overlay._specs if s.kind == "rect"]

    ratio = box.height / box.width
    expected = (limits["y"].max - limits["y"].min) / (limits["x"].max - limits["x"].min)
    assert ratio == pytest.approx(expected, rel=1e-6)


def test_the_limits_box_is_shorter_than_the_grid_is_wide(pane):
    """The property that makes drawing it worth anything: a box the size of the boundary
    would tell the user nothing they could not already see."""
    [box] = [s for s in pane.stage_overlay._specs if s.kind == "rect"]
    [circle] = [s for s in pane.stage_overlay._specs if s.kind == "circle"]

    assert box.height < 2 * circle.radius


# ── what the host reads ──────────────────────────────────────────────────


def test_the_pane_reports_every_change_once(pane, microscope):
    fired = []
    pane.selection_changed.connect(lambda: fired.append(1))

    select(pane, microscope, *TWO_REGIONS)
    pane._on_tile_toggled(0, 0, not pane.mask[0][0])
    pane.clear()

    assert len(fired) == 3


def test_clearing_nothing_reports_nothing(pane):
    """Or a host that recomputes on every signal would do so for no reason."""
    fired = []
    pane.selection_changed.connect(lambda: fired.append(1))

    pane.clear()

    assert fired == []


def test_the_mask_the_host_reads_cannot_be_written_through(pane, microscope):
    select(pane, microscope, *TWO_REGIONS)
    taken = pane.mask

    taken[0][0] = not taken[0][0]

    assert pane.mask != taken
