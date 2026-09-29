"""Drawing the ground to image on a beam overview.

The pane reports rectangles in the overview's own displayed plane, measured from the
pose that overview was acquired at. Both halves of that matter: metres because pixels
belong to a detector and cancel on the other side, and the overview's *own* pose because
the stage has usually moved on by the time anyone is choosing where to look.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from fibsem.structures import BeamType, FibsemStagePosition, ImageSettings
from fibsem.ui.fm.widgets.fm_region_selector import FMRegionSelectorWidget
from fibsem.ui.widgets.overview_widget import OverviewView


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def microscope(qapp):
    from fibsem.ui.fm.overview_app import build_microscope

    return build_microscope()


def shot(microscope, orientation="MILLING", beam_type=BeamType.ION, hfw=900e-6):
    """An overview acquired at one pose, carrying its own metadata."""
    pose = microscope.get_orientation(orientation)
    microscope.move_stage_absolute(
        FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=pose.r, t=pose.t)
    )
    return microscope.acquire_image(
        ImageSettings(
            hfw=hfw, resolution=[1024, 1024], beam_type=beam_type, save=False
        )
    )


@pytest.fixture()
def selector(microscope):
    widget = FMRegionSelectorWidget(microscope)
    widget.setFixedSize(540, 400)
    widget.set_overviews([shot(microscope)])
    return widget


# ── what it is looking at ────────────────────────────────────────────────


def test_it_resolves_against_the_overview_rather_than_the_live_stage(
    microscope, qapp
):
    """A saved overview is exactly the case a live projection gets wrong, and by the
    time someone is selecting on one the stage has usually moved.

    The move happens *before* the overview is handed over, which is the order the real
    workflow has: acquire at the milling pose, go to the FM tab, open the dialog. Moving
    afterwards would prove nothing -- the live and recorded poses would still have agreed
    at the moment everything was read.
    """
    image = shot(microscope, "MILLING")
    recorded = image.metadata.microscope_state.stage_position.t
    microscope.move_to_microscope("FM")
    assert microscope.get_stage_position().t != pytest.approx(recorded), (
        "the stage has to have moved, or this proves nothing"
    )

    widget = FMRegionSelectorWidget(microscope)
    widget.set_overviews([image])

    assert widget.base.t == pytest.approx(recorded)


def test_an_overview_that_does_not_say_where_it_was_taken_is_not_placed(microscope):
    """Refused rather than guessed at: a region drawn against a fabricated pose would
    resolve to somewhere other than what the user was looking at."""
    image = shot(microscope)
    image.metadata.microscope_state.stage_position = None
    widget = FMRegionSelectorWidget(microscope)

    widget.set_overviews([image])

    assert widget.has_overview is False


def test_regions_cannot_be_drawn_before_there_is_an_overview(microscope):
    widget = FMRegionSelectorWidget(microscope)

    assert widget.add_region() is None
    assert widget.regions == []


# ── regions ──────────────────────────────────────────────────────────────


def test_a_region_is_reported_in_metres(selector):
    selector.add_region()

    [region] = selector.regions
    assert 0 < region.width < 1e-2, "a region on a grid is micrometres, not pixels"
    assert region.height > 0


def test_regions_accumulate_and_can_be_removed_one_at_a_time(selector):
    selector.add_region()
    selector.add_region()
    assert len(selector.regions) == 2

    selector.delete_selected_region()

    assert len(selector.regions) == 1


def test_deleting_with_nothing_selected_does_nothing(selector):
    selector.add_region()
    selector._select(None)

    selector.delete_selected_region()

    assert len(selector.regions) == 1


def test_a_new_region_does_not_land_exactly_on_the_last(selector):
    """Two identical rectangles stacked look like one, and like Add having failed."""
    selector.add_region()
    selector.add_region()

    first, second = selector.regions
    assert (first.x0, first.y0) != (second.x0, second.y0)


def test_clearing_removes_every_region(selector):
    selector.add_region()
    selector.add_region()

    selector.clear_regions()

    assert selector.regions == []


def test_clearing_nothing_reports_nothing(selector):
    """Or a host recomputing on every signal would do so for no reason."""
    fired = []
    selector.regions_changed.connect(lambda: fired.append(1))

    selector.clear_regions()

    assert fired == []


def test_every_change_is_reported(selector):
    fired = []
    selector.regions_changed.connect(lambda: fired.append(1))

    selector.add_region()
    selector.add_region()
    selector.delete_selected_region()
    selector.clear_regions()

    assert len(fired) == 4


# ── selection ────────────────────────────────────────────────────────────


def test_adding_a_region_selects_it(selector):
    first = selector.add_region()
    second = selector.add_region()

    assert selector.selected_region == second != first


def test_clicking_a_region_selects_it_and_the_background_deselects(selector):
    selector.add_region()
    inside = selector._overlays[0].get_rect()

    selector._on_left_click(inside["cx"], inside["cy"])
    assert selector.selected_region == 0

    selector._on_left_click(inside["x0"] - 1e6, inside["y0"] - 1e6)
    assert selector.selected_region is None


def test_the_topmost_region_wins_where_two_overlap(selector):
    """Matching what clicking would have hit: the one drawn last is on top."""
    selector.add_region()
    selector.add_region()
    selector._overlays[1].set_rect(
        *[selector._overlays[0].get_rect()[k] for k in ("x0", "y0", "width", "height")]
    )
    rect = selector._overlays[0].get_rect()

    selector._on_left_click(rect["cx"], rect["cy"])

    assert selector.selected_region == 1


def test_dragging_a_region_selects_the_one_dragged(selector):
    """From the sender, not from the pointer -- where two overlap, asking the position
    again could name a different region than the one being moved."""
    selector.add_region()
    selector.add_region()
    selector._select(None)

    selector._overlays[0].rect_changed.emit(selector._overlays[0].get_rect())

    assert selector.selected_region == 0


def test_the_description_says_what_is_selected(selector):
    assert "Right-click" in selector.describe_selected()

    selector.add_region()

    assert selector.describe_selected().startswith("Region 1 of 1")


# ── views ────────────────────────────────────────────────────────────────


def views_of(microscope):
    return {
        OverviewView(BeamType.ELECTRON, "SEM"): [
            shot(microscope, "SEM", BeamType.ELECTRON)
        ],
        OverviewView(BeamType.ION, "MILLING"): [shot(microscope, "MILLING")],
    }


def test_it_opens_on_the_most_recent_view(microscope):
    """Which is the one just acquired, and so the one being looked at."""
    widget = FMRegionSelectorWidget(microscope)
    views = views_of(microscope)

    widget.set_views(views)

    assert widget.current_view == list(views)[-1]


def test_switching_view_starts_a_fresh_selection(microscope):
    """Regions belong to the view they were drawn on: images from different views are
    pictures from different directions and do not register."""
    widget = FMRegionSelectorWidget(microscope)
    views = views_of(microscope)
    widget.set_views(views)
    widget.add_region()

    widget.show_view(list(views)[0])

    assert widget.regions == []


def test_the_chip_strip_stays_hidden_with_one_view(microscope):
    """A control with one option is a label that looks clickable."""
    widget = FMRegionSelectorWidget(microscope)

    widget.set_views({OverviewView(BeamType.ION, "MILLING"): [shot(microscope)]})

    assert widget._chip_strip.isVisibleTo(widget) is False


def test_the_chip_strip_appears_once_there_is_a_choice(microscope):
    widget = FMRegionSelectorWidget(microscope)

    widget.set_views(views_of(microscope))

    assert widget._chip_strip.isVisibleTo(widget) is True
