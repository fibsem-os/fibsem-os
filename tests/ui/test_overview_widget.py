"""The FIB/SEM overview widget: the inversion, and the rules it was built to keep.

The tab this replaces stitched one giant image and then reprojected stage positions onto
its pixels, so everything on screen was gated on the stitch. Here each tile is placed
where it was acquired, and that is what these tests pin -- along with the two rules the
old widget broke: no hardware read on a UI event, and one derivation for both directions
so a click resolves to the position the marker was drawn from.

The payload the placement reads is pinned in `tests/test_tiled_progress_payload.py`
rather than here: it is a contract on shared acquisition code, and this module is gated
on PyQt5, which CI does not install -- so a test of it living here would never run
where it matters.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_overview_widget.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

# CI installs `.[test]`, not `.[ui]`, so PyQt5 is absent there. Without this the
# module-level imports below turn a skip into a collection error.
pytest.importorskip("PyQt5")

from PyQt5.QtCore import QPoint  # noqa: E402
from PyQt5.QtWidgets import QApplication, QDialog  # noqa: E402

from copy import deepcopy  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.structures import (  # noqa: E402
    BeamType,
    FibsemImage,
    FibsemStagePosition,
    ImageSettings,
)
from fibsem.ui.widgets import overview_confirmation_dialog  # noqa: E402
from fibsem.ui.widgets.canvas.overlays.minimap_overlays import (  # noqa: E402
    GRID_BOUNDARY_RADIUS_M,
)
from fibsem.ui.widgets.canvas.real_space_canvas import (  # noqa: E402
    _DEFAULT_DISPLAY_PX,
)
from fibsem.ui.widgets.overview_widget import (  # noqa: E402
    FibsemOverviewWidget,
    OverviewView,
)

_app = QApplication.instance() or QApplication(sys.argv)


@pytest.fixture(autouse=True)
def confirmations(monkeypatch):
    """Every `acquire()` opens a modal dialog, which would hang the run.

    Auto-accepted so the tests below can go on testing what they were written for -- and
    *recorded*, because a fixture that silently says yes would equally hide the dialog
    never being shown, which is the one thing a confirmation has to do. Tests that care
    assert against the list this yields.
    """
    shown = []

    def _exec(dialog):
        shown.append(dialog)
        return QDialog.Accepted

    monkeypatch.setattr(
        overview_confirmation_dialog.OverviewConfirmationDialog, "exec_", _exec
    )
    return shown


@pytest.fixture(scope="module")
def microscope():
    """A simulated Arctis, because half of what is drawn here is compustage-shaped.

    A plain Demo session is not a compustage, and the limits box, the holder slots, the
    grid boundary and the MILLING orientation all behave differently or do not draw at
    all without one -- eight tests, whose assertions read as unrelated ("overlay shapes",
    "view orientations") and are not (FIB-734).

    `sim-arctis-configuration.yaml` is the same configuration `test_overview_tab_host.py`
    and `test_beam_stage_projection.py` use, so the three agree on what an Arctis is.
    """
    import fibsem.config as fibsem_config

    path = os.path.join(
        os.path.dirname(fibsem_config.__file__), "config", "sim-arctis-configuration.yaml"
    )
    scope, _ = utils.setup_session(manufacturer="Demo", config_path=path)
    assert scope.stage_is_compustage, "the config stopped being a compustage"
    return scope


@pytest.fixture
def widget(microscope):
    w = FibsemOverviewWidget(microscope)
    w.resize(900, 700)
    yield w
    w.close()


def _tile(microscope, position: FibsemStagePosition, shape=(64, 64), hfw=100e-6):
    """An acquired-looking image that records where it was taken."""
    image = FibsemImage.generate_blank_image(resolution=(shape[1], shape[0]), hfw=hfw)
    image.data = (np.random.default_rng(0).random(shape) * 255).astype(np.uint8)
    state = microscope.get_microscope_state(beam_type=BeamType.ELECTRON)
    state.stage_position = position
    image.metadata.image_settings = ImageSettings(hfw=hfw, beam_type=BeamType.ELECTRON)
    image.metadata.microscope_state = state
    image.metadata.system_info = microscope.system.info
    image.metadata.hardware_geometry = microscope.hardware_geometry()
    return image


def _at(base: FibsemStagePosition, dx: float = 0.0, dy: float = 0.0, name=None):
    p = FibsemStagePosition(x=base.x + dx, y=base.y + dy, z=base.z, r=base.r, t=base.t)
    p.name = name
    return p


def _hold_at(widget, px, dtype=np.uint8):
    """Set the store budget so an overview is held at most *px* a side.

    Tests state the size they mean rather than inheriting the production budget, which
    is 8000 px a side — big enough that a test image sized from it would spend seconds
    in `filtered_data`'s median filter.
    """
    widget._store_budget_bytes = px * px * (np.dtype(dtype).itemsize + 1)


def _settle(widget):
    """Run the canvas's coalesced detail refresh now rather than waiting out its timer."""
    widget.canvas._detail_timer.stop()
    widget.canvas.refresh_detail()
    _app.processEvents()


def _zoom(widget, fraction, centre=(0.0, 0.0)):
    """Frame *fraction* of the placed image's width, and let the refresh settle."""
    placed = widget.canvas._placed[widget.canvas.placed_keys[0]]
    half = (placed.extent[1] - placed.extent[0]) / 2 * fraction
    widget.canvas._ax.set_xlim(centre[0] - half, centre[0] + half)
    widget.canvas._ax.set_ylim(centre[1] + half, centre[1] - half)
    _settle(widget)


class TestTheInversion:
    """Images are placed where they were acquired, rather than reprojected onto one.

    The tab this replaces stitched a mosaic and then reprojected stage positions onto
    its pixels, so everything on screen was gated on the stitch. Here what is placed
    carries its own stage position and is put there.

    The overview itself is one image, not one per tile. That was a deliberate reversal:
    placing tiles individually put each where the stage actually reached, which the
    stitch buffer cannot express (FIB-399) -- but the buffer is what gets saved, so the
    accuracy never survived a reload, and an artist per tile cost about 2 ms of redraw
    for every tile ever acquired (FIB-627).
    """

    def test_the_mosaic_is_placed_where_the_run_was_centred(self, widget, microscope):
        """It is placed from its own metadata like anything else, so it lands where the
        grid was rather than wherever the display supposes."""
        base = microscope.get_stage_position()
        widget.place_image(_tile(microscope, _at(base)), key="anchor")
        reference = widget.canvas.reference_pixel_size

        away = _at(base, dx=80e-6, dy=-30e-6)
        widget.place_image(_tile(microscope, away), key="mosaic")

        anchor = widget.canvas._placed["anchor"].extent
        mosaic = widget.canvas._placed["mosaic"].extent
        centre_dx = ((mosaic[0] + mosaic[1]) - (anchor[0] + anchor[1])) / 2.0
        assert centre_dx == pytest.approx(80e-6 / reference, rel=1e-6)

    def test_a_tile_is_placed_where_it_landed_not_where_it_was_aimed(
        self, widget, microscope
    ):
        """A real stage does not arrive exactly where it was asked to, and the stitch
        buffer cannot express that -- it copies each tile to an integer pixel offset,
        so the error against the true position accumulates across the grid (FIB-399).

        Placing from the tile's own recorded position is what removes that. Here a tile
        records a position half a field away from its neighbour's nominal grid slot; the
        placement has to follow the recording.
        """
        base = microscope.get_stage_position()
        widget.place_image(_tile(microscope, _at(base)), key="first")
        reference = widget.canvas.reference_pixel_size

        strayed = _at(base, dx=37e-6, dy=-11e-6)
        widget.place_image(_tile(microscope, strayed), key="strayed")

        first = widget.canvas._placed["first"].extent
        second = widget.canvas._placed["strayed"].extent
        # Canvas x runs with stage x; the displacement in canvas pixels is the stage
        # displacement over the reference pixel size.
        centre_dx = ((second[0] + second[1]) - (first[0] + first[1])) / 2.0
        assert centre_dx == pytest.approx(37e-6 / reference, rel=1e-6), (
            "the tile was not placed at the position it recorded"
        )

    def test_a_tile_covers_the_ground_it_images(self, widget, microscope):
        """A tile is drawn at the size it *images*, not the size it is stored at.

        The canvas decimates for display and sizes an image from `shape x pixel_size`
        unless told what it covers. Once tiles started being stored display-reduced —
        so a view switch can re-place them — handing that reduced array over without
        `covers` drew every 1024 px tile as if it imaged 512 px of sample: right
        positions, half size, and a mosaic full of black gaps.

        Asserted as *abutment*, which is what a user sees. A 2x2 at zero overlap has to
        meet, not leave a gap and not overlap.
        """
        base = microscope.get_stage_position()
        # 256 px at this hfw, so the canvas's 512 px cap does not reduce it and only
        # the deliberate reduction below can shrink anything.
        hfw = 256 * 4e-7
        widget.place_image(_tile(microscope, _at(base), shape=(256, 256), hfw=hfw),
                           key="a")
        widget.place_image(
            _tile(microscope, _at(base, dx=hfw), shape=(256, 256), hfw=hfw), key="b"
        )

        left = widget.canvas._placed["a"].extent
        right = widget.canvas._placed["b"].extent
        assert right[0] == pytest.approx(left[1], rel=1e-9), (
            f"tiles do not abut: left ends at {left[1]:.1f}, right starts at "
            f"{right[0]:.1f} canvas px"
        )

    def test_a_reduced_tile_is_still_placed_at_full_size(self, widget, microscope):
        """The same property where the reduction actually bites — an image larger than
        the store cap, which a stitched mosaic always is."""
        base = microscope.get_stage_position()
        cap = 1024
        _hold_at(widget, cap)
        big = cap + cap // 2  # over the cap, so it reduces; not so far over it is slow
        hfw = big * 1e-7
        widget.place_image(_tile(microscope, _at(base), shape=(big, big), hfw=hfw),
                           key="big")

        extent = widget.canvas._placed["big"].extent
        drawn_width_m = (extent[1] - extent[0]) * widget.canvas.reference_pixel_size
        assert drawn_width_m == pytest.approx(hfw, rel=1e-9), (
            f"a {big}px tile imaging {hfw * 1e6:.0f} um was drawn covering "
            f"{drawn_width_m * 1e6:.0f} um"
        )

    def test_an_image_that_does_not_say_where_it_was_taken_is_refused(
        self, widget, microscope
    ):
        """Placed at the origin it would look exactly like a correctly placed image."""
        image = _tile(microscope, microscope.get_stage_position())
        # `metadata.stage_position` is a read-only property over the microscope state,
        # so this is the only place it can be absent from.
        image.metadata.microscope_state.stage_position = None
        assert image.metadata.stage_position is None, "the property stopped delegating"
        assert widget.place_image(image) is None
        assert widget.canvas.placed_keys == []

    def test_an_image_with_no_pixel_size_is_refused_rather_than_raising(
        self, widget, microscope
    ):
        """The half of the guard that is load-bearing rather than defensive.

        Once the canvas has a scale and an origin, a missing *position* is caught
        downstream anyway -- the frame refuses it. A missing *pixel size* is not:
        `add_image` raises `ValueError` from inside the placement, which on a progress
        callback means an exception escaping a Qt slot, and PyQt5 qFatals on that
        (FIB-329). Refusing here turns it into one skipped tile.
        """
        base = microscope.get_stage_position()
        widget.place_image(_tile(microscope, _at(base)))  # establishes scale and origin
        assert widget.canvas.reference_pixel_size is not None

        image = _tile(microscope, _at(base, dx=10e-6))
        image.metadata.pixel_size.x = None

        assert widget.place_image(image) is None  # must not raise
        assert len(widget.canvas.placed_keys) == 1, "the bad image was placed anyway"


class TestTheTwoCaps:
    """How much an overview is *held* at and how much of it is *drawn* are two questions.

    Held too coarse and nothing can recover the detail, because it was thrown away
    before the canvas ever saw it -- which is why "don't downscale the actual image" has
    no answer while one number does both jobs. Drawn too fine and every frame pays for
    it at every zoom, since matplotlib resamples the whole array however little of it is
    on screen.

    They are also measured in different units, which is the part most likely to be
    "tidied" back together: what is held is bounded in **bytes**, because bytes are what
    it costs, and what is drawn is bounded in **pixels**, because a frame's cost is the
    pixels matplotlib resamples (FIB-658).
    """

    def test_the_store_bound_is_bytes_and_scales_with_the_detector(self, widget):
        """Bytes, not pixels, because bytes are what is being spent. A pixel cap prices
        the same setting at 52 MB for a mosaic of 1024 px tiles and 118 MB for one of
        3072 px tiles, and costs half as much again on a uint16 detector without saying
        so."""
        wide = widget._store_cap(np.uint8)
        deep = widget._store_cap(np.uint16)

        assert wide * wide * 2 <= widget._store_budget_bytes
        assert deep * deep * 3 <= widget._store_budget_bytes
        assert deep < wide, "a 16-bit detector was given the same pixel count as an 8-bit"

    def test_the_budget_holds_the_complained_about_mosaic_whole(self, widget, microscope):
        """The case behind FIB-658: a 5x5 of 1024 px tiles is 5120 px, and the point of
        the budget is that it is held at its acquired resolution rather than reduced."""
        assert widget._store_cap(np.uint8) >= 5 * 1024, (
            f"a 5x5 of 1024 px tiles would be reduced at a "
            f"{widget._store_cap(np.uint8)} px cap"
        )

    def test_the_draw_cap_covers_the_canvas_it_draws_on(self, widget):
        """Below the canvas's own width a full-frame draw magnifies rather than
        decimates, whatever is held -- so this is the floor, and it is measured against
        the widget rather than against a constant.

        It was written as "greater than the shared default", on the argument that the
        overview should raise its own rather than the default the fluorescence canvas
        also takes. That argument was overtaken: FIB-658 was fixed twice in parallel, and
        the other fix (#423) raised the shared default to 2048 deliberately, for FM's
        own much larger images. So the overview passing its own is now a statement of
        intent rather than a difference, and the property below is what it is for.
        """
        assert widget.canvas.display_max_px >= widget.canvas.width(), (
            f"a {widget.canvas.display_max_px} px cap on a "
            f"{widget.canvas.width()} px canvas magnifies a full-frame draw"
        )
        assert widget.canvas.display_max_px >= _DEFAULT_DISPLAY_PX, (
            "the overview draws less than a canvas with no detail source at all"
        )

    def test_a_stored_overview_out_resolves_the_canvas_it_is_drawn_on(
        self, widget, microscope
    ):
        """Measured against the widget, not a constant: it is the screen the stored
        pixels are stretched over that decides whether the cap is too low. Below it the
        reduction stops decimating and starts *magnifying* -- a 5x5 of 1024 px tiles at
        512 was drawn across a ~1100 px canvas as a 2x2 block per stored pixel, which is
        what "way too pixelated" was.
        """
        base = microscope.get_stage_position()
        cap = 1024
        _hold_at(widget, cap)
        big = cap + cap // 2  # any mosaic over the cap; it reduces to the cap
        record_id = widget.set_image(
            _tile(microscope, _at(base), shape=(big, big), hfw=500e-6)
        )

        stored = np.asarray(widget._records[record_id].images[0].grey)
        assert max(stored.shape[:2]) >= widget.canvas.width(), (
            f"{max(stored.shape[:2])} stored px drawn across a "
            f"{widget.canvas.width()} px canvas, so each one is magnified"
        )

    def test_what_is_held_follows_the_store_cap_and_not_the_canvas(
        self, widget, microscope
    ):
        """Forced apart, because equal numbers cannot show which one is being read.

        This is the assertion that fails if the two are collapsed back into one, and it
        is the one phase 2 needs to hold before it can keep more than it draws.
        """
        drawn_cap = widget.canvas.display_max_px
        source = drawn_cap + drawn_cap // 4  # over the draw cap, under the store cap
        _hold_at(widget, source * 2)
        base = microscope.get_stage_position()
        record_id = widget.set_image(
            _tile(microscope, _at(base), shape=(source, source), hfw=500e-6)
        )

        held = np.asarray(widget._records[record_id].images[0].grey)
        drawn = np.asarray(widget.canvas._placed[record_id].artist.get_array())

        assert max(held.shape[:2]) == source, "the record was reduced to the draw cap"
        assert max(drawn.shape[:2]) <= widget.canvas.display_max_px, (
            "the canvas drew more than its own cap"
        )
        assert max(drawn.shape[:2]) < max(held.shape[:2]), (
            "the canvas drew everything held, so the caps are not actually separate"
        )


class TestOneDerivationForBothDirections:
    """A click on a marker must resolve to the position that marker was drawn from."""

    def test_a_marked_position_picks_and_resolves_back_to_itself(
        self, widget, microscope
    ):
        base = microscope.get_stage_position()
        widget.place_image(_tile(microscope, _at(base)))
        marks = [
            _at(base, 0.0, 0.0, "A"),
            _at(base, 60e-6, 0.0, "B"),
            _at(base, -40e-6, 30e-6, "C"),
        ]
        widget.set_positions(marks)

        frame = widget._frame()
        for mark in marks:
            x, y = frame.to_canvas(mark)
            assert widget._position_at(x, y) == mark.name, f"{mark.name} not pickable"
            resolved = widget._stage_position_at(x, y)
            assert resolved is not None
            assert resolved.x == pytest.approx(mark.x, abs=1e-12)
            assert resolved.y == pytest.approx(mark.y, abs=1e-12)

    def test_clicking_away_from_every_marker_picks_nothing(self, widget, microscope):
        """A miss must not select the nearest marker on the far side of the canvas."""
        base = microscope.get_stage_position()
        widget.place_image(_tile(microscope, _at(base)))
        widget.set_positions([_at(base, 0.0, 0.0, "A")])
        frame = widget._frame()
        x, y = frame.to_canvas(_at(base, 0.0, 0.0))
        assert widget._position_at(x + 4000, y + 4000) is None


    def test_a_hidden_marker_is_not_a_target(self, widget, microscope):
        """Turned off means gone, not merely invisible.

        With saved positions hidden, `_refresh_position_markers` draws nothing — so a
        pick here selects a lamella from a click on what looks like bare mosaic, and the
        host fans that selection out to every list in the window with nothing on screen
        to explain it.
        """
        base = microscope.get_stage_position()
        widget.place_image(_tile(microscope, _at(base)))
        widget.set_positions([_at(base, 0.0, 0.0, "A")])
        frame = widget._frame()
        x, y = frame.to_canvas(_at(base, 0.0, 0.0))
        assert widget._position_at(x, y) == "A", "not pickable while shown"

        widget.overlay_controls.set_visible("positions", False)

        assert widget._position_at(x, y) is None

    def test_a_marker_shown_again_is_pickable_again(self, widget, microscope):
        """The guard must not outlive the state it reads."""
        base = microscope.get_stage_position()
        widget.place_image(_tile(microscope, _at(base)))
        widget.set_positions([_at(base, 0.0, 0.0, "A")])
        frame = widget._frame()
        x, y = frame.to_canvas(_at(base, 0.0, 0.0))

        widget.overlay_controls.set_visible("positions", False)
        widget.overlay_controls.set_visible("positions", True)

        assert widget._position_at(x, y) == "A"


class TestAnOverviewIsReducedOnce:
    """The reduced arrays are the largest thing this widget holds.

    Placing built one and binding it into the canvas's `detail` closure, then the record
    built a second, equal one — so the reduction was paid twice and both copies were
    kept. At the 128 MB store budget that is up to a quarter of a gigabyte per overview
    instead of an eighth.
    """

    def test_a_loaded_overview_is_reduced_once(self, widget, microscope, monkeypatch):
        base = microscope.get_stage_position()
        widget.place_image(_tile(microscope, _at(base)))  # anchors the view

        calls = []
        original = widget._stored_tile
        monkeypatch.setattr(
            widget, "_stored_tile",
            lambda image: calls.append(image) or original(image),
        )

        widget.set_image(_tile(microscope, _at(base, dx=10e-6)))

        assert len(calls) == 1, f"reduced {len(calls)} times"

    def test_the_record_keeps_the_tile_that_was_placed(
        self, widget, microscope, monkeypatch
    ):
        """Not an equal one. Two equal copies is exactly the waste being removed, and
        `show_view` re-places from the record — so if they ever diverged, a view switch
        would redraw something other than what is on screen.

        Identity against the tile the canvas was actually handed, captured at
        `_place_on_canvas`. Asserting the record merely holds *a* tile would pass just
        as well against the version that built two.
        """
        base = microscope.get_stage_position()
        widget.place_image(_tile(microscope, _at(base)))

        placed = []
        original = widget._place_on_canvas
        monkeypatch.setattr(
            widget, "_place_on_canvas",
            lambda tile, view, **kwargs: placed.append(tile) or original(
                tile, view, **kwargs
            ),
        )

        record_id = widget.set_image(_tile(microscope, _at(base, dx=10e-6)))

        record = widget._records[record_id]
        assert len(placed) == 1
        assert record.images == [placed[0]], "the record kept a second, equal tile"


class TestItDoesNotReadHardwareOnUiEvents:
    """The rule the old tab broke twice, and the reason this widget caches.

    `FibsemMinimapWidget` polled `get_microscope_state()` on every experiment load and
    went through `project_stable_move` -- and so `get_scan_rotation` -- on every
    click-to-move. On a TFS system both take the shared imaging channel, from the GUI
    thread (FIB-544, FIB-600).
    """

    def test_drawing_and_picking_read_nothing(self, widget, microscope, monkeypatch):
        base = microscope.get_stage_position()
        widget.place_image(_tile(microscope, _at(base)))
        widget.set_positions([_at(base, 20e-6, 0.0, "A")])

        reads = []
        for name in ("get_microscope_state", "get_scan_rotation", "get_stage_position"):
            original = getattr(microscope, name)

            def counted(*a, __name=name, __orig=original, **k):
                reads.append(__name)
                return __orig(*a, **k)

            monkeypatch.setattr(microscope, name, counted)

        frame = widget._frame()
        x, y = frame.to_canvas(_at(base, 20e-6, 0.0))
        for _ in range(10):
            widget._refresh_context_overlays()
            widget._on_canvas_clicked(x, y)
            widget._on_settings_changed()

        assert reads == [], f"a UI event read the microscope: {sorted(set(reads))}"

    def test_only_the_click_that_drives_the_stage_re_reads_the_projection(
        self, widget, microscope, monkeypatch
    ):
        """Deliberately not cached: a stale scan rotation here would send the stage to
        the point rotated 180 degrees from where the user clicked. Everything else uses
        the kept projection."""
        base = microscope.get_stage_position()
        widget.place_image(_tile(microscope, _at(base)))
        reads = []
        original = microscope.get_scan_rotation
        monkeypatch.setattr(
            microscope, "get_scan_rotation",
            lambda beam_type: (reads.append(beam_type), original(beam_type))[1],
        )

        widget._refresh_context_overlays()
        assert reads == [], "drawing re-read the scan rotation"

        widget._stage_position_at(0.0, 0.0)
        assert len(reads) == 1, "resolving a click did not re-read the scan rotation"


class TestWhatIsDrawn:
    def test_the_context_overlays_describe_where_the_stage_can_go(
        self, widget, microscope
    ):
        base = microscope.get_stage_position()
        widget.place_image(_tile(microscope, _at(base)))
        labels = {spec.label for spec in widget.context_overlay._specs}
        # The limits and the holder slots are configuration-dependent; on a compustage
        # system all of these are present, and this fixture is one.
        assert {"Stage Limits", "Grid Boundary"} <= labels
        # The planned run is the tile grid, not a shape in this overlay -- it is drawn
        # tile by tile so the seams and the enabled set are visible before the button.
        assert widget.tile_grid_overlay._tiles, "the planned run is not drawn"

    def test_the_selected_position_is_drawn_apart_from_the_others(
        self, widget, microscope
    ):
        """Separate overlays rather than a colour within one: `PointsOverlay` paints
        every point the same, so the selection has to be its own layer."""
        base = microscope.get_stage_position()
        widget.place_image(_tile(microscope, _at(base)))
        widget.set_positions([
            _at(base, 0.0, 0.0, "A"), _at(base, 30e-6, 0.0, "B"),
        ])
        widget.set_selected_position("B")

        assert len(widget.position_overlay._points) == 1
        assert len(widget.selected_position_overlay._points) == 1

        widget.set_selected_position(None)
        assert len(widget.position_overlay._points) == 2
        assert len(widget.selected_position_overlay._points) == 0

    def test_the_planned_footprint_follows_the_settings(self, widget, microscope):
        """It answers "if I press the button now, what do I get?", so it has to change
        when the answer does -- and before anything has been acquired."""
        base = microscope.get_stage_position()
        widget.place_image(_tile(microscope, _at(base)))

        small = widget.tile_grid_overlay._extent()
        settings = widget.settings_widget.get_settings()
        settings.nrows, settings.ncols = settings.nrows + 3, settings.ncols + 3
        widget.settings_widget.update_from_settings(settings)
        widget._refresh_context_overlays()

        grown = widget.tile_grid_overlay._extent()
        assert grown[0] > small[0] and grown[1] > small[1], (
            "the drawn grid ignored the settings"
        )


class TestTheHostContract:
    """This widget emits requests; a host decides what a position means."""

    def test_it_asks_rather_than_creating_anything(self, widget, microscope):
        base = microscope.get_stage_position()
        widget.place_image(_tile(microscope, _at(base)))
        widget.set_positions([_at(base, 0.0, 0.0, "A")])
        widget.set_selected_position("A")

        config = widget._position_menu(0.0, 0.0)
        assert config is not None
        labels = [action.label for action in config.actions]
        assert any("Add" in label for label in labels)
        assert any("Move Selected" in label and "A" in label for label in labels)

    def test_nothing_is_offered_to_move_when_nothing_is_selected(
        self, widget, microscope
    ):
        base = microscope.get_stage_position()
        widget.place_image(_tile(microscope, _at(base)))
        widget.set_selected_position(None)

        config = widget._position_menu(0.0, 0.0)
        assert config is not None
        assert not any("Move Selected" in a.label for a in config.actions)

    def test_a_host_that_has_taken_the_instrument_can_forbid_marking(
        self, widget, microscope
    ):
        """A workflow iterating the experiment's positions is usually why a host takes
        the instrument, and adding one underneath it is not a thing to do quietly."""
        base = microscope.get_stage_position()
        widget.place_image(_tile(microscope, _at(base)))
        widget.set_interactive(False)
        assert widget._position_menu(0.0, 0.0) is None
        assert not widget.button_acquire.isEnabled()

    def test_an_overview_can_be_hidden_and_removed_as_one_thing(
        self, widget, microscope
    ):
        """A run is many tiles but one overview -- that is what a user acquired and what
        they mean when they hide it."""
        base = microscope.get_stage_position()
        widget.set_image(_tile(microscope, _at(base)))
        record_id = widget.overviews[0].id
        assert len(widget.canvas.placed_keys) == 1

        assert widget.set_overview_visible(record_id, False)
        assert all(
            not widget.canvas._placed[k].artist.get_visible()
            for k in widget.canvas.placed_keys
        )

        assert widget.remove_overview(record_id)
        assert widget.canvas.placed_keys == []
        assert widget.remove_overview(record_id) is False


class TestThingsOnlyRunningItFound:
    """Three defects that a green test run said nothing about.

    All three were found by opening the widget and looking at it, which is the argument
    for doing that (`feedback_run_the_app_qt_tests_miss_this`). The tests exist so they
    do not come back, not because they were caught this way.
    """

    def test_the_tile_counts_are_still_drawn_in_the_narrowest_column(self, widget):
        """Squeezed below ~80px, a tile spinbox's +/- buttons take the whole control and
        the digit stops being rendered -- while `lineEdit().text()` still reports it, so
        nothing about the widget's *state* looks wrong and no assertion on its value
        would fail.

        So this measures the laid-out control at the narrowest the window allows, which
        is the thing a user sees, rather than any one of the two mechanisms that keep it
        wide enough.
        """
        from PyQt5.QtWidgets import QScrollArea

        widget.show()
        # As narrow as the layout permits -- Qt clamps this to the minimum.
        widget.resize(widget.minimumSizeHint().width(), 700)
        _app.processEvents()

        scroll = widget.findChild(QScrollArea)
        assert scroll.width() <= 600, "the column did not actually get squeezed"
        for name in ("spin_rows", "spin_cols"):
            spinbox = getattr(widget.settings_widget.grid, name)
            assert spinbox.width() >= 80, (
                f"{name} is {spinbox.width()}px at the narrowest column -- too narrow "
                "to draw its digit"
            )

    def test_the_list_keeps_up_with_a_run_in_progress(self, widget, microscope):
        """The row shows a tile count, so it has to be rebuilt as tiles land -- not only
        when the run finishes. A run showing "0 tiles" while a mosaic fills on the
        canvas beside it is the list contradicting the display.

        Read off the *row widget*, not the record: the record is updated either way, so
        asserting on it proves nothing about what reaches the screen.
        """
        from fibsem.ui.widgets.overview_widget import OverviewRecord

        base = microscope.get_stage_position()
        widget._records["run"] = OverviewRecord("run", "run", [])
        widget._active_record = "run"
        widget._refresh_overview_list()
        assert widget.overview_list._list.count() == 1

        for i in (1, 2, 3):
            widget._apply_progress({
                "msg": "Tile Collected", "counter": i, "total": 3,
                "preview": _tile(microscope, _at(base)),
            })
            row = widget.overview_list._rows["run"]
            assert row.detail_label.text().startswith(f"{i} tile"), (
                f"after {i} tile(s) the row still reads {row.detail_label.text()!r}"
            )

    def test_the_status_line_and_the_progress_bar_do_not_disagree(
        self, widget, microscope
    ):
        """The bar carries the message during a run; the label carries the outcome
        after it. A label reading "Starting…" under a bar reading "4 / 9" is the two of
        them contradicting each other in public."""
        widget.label_status.setText("Acquired 4 tile(s).")
        widget._set_running(True)
        assert widget.label_status.text() == "", "a stale outcome survived into the run"

        widget._apply_progress({"counter": 2, "total": 9, "msg": "Tile Collected"})
        assert widget.label_status.text() == "", "the label competed with the bar"

        widget._on_finished({})
        assert widget.label_status.text(), "the outcome was never reported"


class TestTheRunOwnsItsSettings:
    """Reading the settings must not rewrite what a running acquisition is using.

    Reported from a real run: tile (0,0) acquired, then tile (0,1) died in
    `os.path.join(None, filename)`. The cause was not the acquisition at all --
    `OverviewAcquisitionSettingsWidget.get_settings()` mutates and returns the settings
    widget's *own* `ImageSettings`, resetting `path` from its text box. The stage moving
    between tiles refreshed the overlays, which read the beam type through that call,
    which set the run's output path back to None.
    """

    def test_reading_the_settings_does_not_disturb_a_run(self, widget, microscope,
                                                         monkeypatch, tmp_path):
        """The reported sequence, end to end: start a run, then do everything a stage
        move between tiles does, and check the run's settings still name a path.

        The empty path box is the crucial part of the setup -- that is what
        `get_settings()` reads back as `None`.
        """
        captured = {}

        def fake_worker(fn, *args):
            captured["settings"] = args[0]

            class _W:
                def start(self_inner):
                    pass

                def is_alive(self_inner):
                    return True  # a run is under way

            return _W()

        widget.set_save_directory(str(tmp_path))
        monkeypatch.setattr(
            "fibsem.ui.widgets.overview_widget.FunctionWorker", fake_worker
        )
        widget.acquire()
        running = captured["settings"]
        assert running.image_settings.path, "the run started with no path"

        # Now empty the box, as a widget whose host never filled it would be, and do
        # everything a stage move mid-run triggers.
        widget.settings_widget.path_edit.setText("")
        for _ in range(5):
            _ = widget.beam_type
            widget._refresh_position_markers()
            widget._refresh_context_overlays()
            widget._on_stage_moved(microscope.get_stage_position())

        assert running.image_settings.path, (
            "reading the widget mid-run reset the running acquisition's output path — "
            "the second tile would die in os.path.join(None, filename)"
        )

    def test_the_runner_gets_its_own_copy_of_the_settings(self, widget, microscope,
                                                          monkeypatch, tmp_path):
        """Even with nothing else reading, handing a background runner the widget's own
        instance means any later edit reaches into a run in progress."""
        captured = {}

        def fake_worker(fn, *args):
            captured["settings"] = args[0] if args else None

            class _W:
                def start(self_inner):
                    pass

                def is_alive(self_inner):
                    return False

            return _W()

        widget.set_save_directory(str(tmp_path))
        monkeypatch.setattr(
            "fibsem.ui.widgets.overview_widget.FunctionWorker", fake_worker
        )
        widget.acquire()

        handed_over = captured["settings"]
        assert handed_over is not None
        # The settings widget no longer *has* an `ImageSettings` to hand out by
        # accident -- `get_settings` constructs one per call, so the defect this guards
        # is now impossible by construction rather than avoided by a deep copy. Asserted
        # as that property, which is the thing that must not regress.
        first, second = (
            widget.settings_widget.get_settings(),
            widget.settings_widget.get_settings(),
        )
        assert first.image_settings is not second.image_settings, (
            "the settings widget hands out one shared ImageSettings again"
        )
        assert handed_over.image_settings is not first.image_settings

    def test_the_save_directory_fills_the_path_box(self, widget, tmp_path):
        """From the experiment by default, and visible so a user can see and change it.

        An empty box is also what made `get_settings()` read the path back as None.
        """
        widget.set_save_directory(str(tmp_path))
        assert widget.settings_widget.path_edit.text() == str(
            tmp_path
        )
        assert widget._settings().image_settings.path == str(tmp_path)

    def test_a_run_always_has_somewhere_to_write(self, widget, monkeypatch):
        """No host directory and an empty box still has to produce a usable path --
        finding out at the second tile is the worst possible time."""
        widget.set_save_directory(None)
        widget.settings_widget.path_edit.setText("")
        captured = {}

        def fake_worker(fn, *args):
            captured["settings"] = args[0]

            class _W:
                def start(self_inner):
                    pass

                def is_alive(self_inner):
                    return False

            return _W()

        monkeypatch.setattr(
            "fibsem.ui.widgets.overview_widget.FunctionWorker", fake_worker
        )
        widget.acquire()
        assert captured["settings"].image_settings.path, "the run had nowhere to write"

    def test_a_run_is_called_overview_image_by_default(self, widget, monkeypatch):
        """Not `default_image`, which is what `ImageSettings` alone would give.

        The name is not cosmetic: `TiledAcquisitionRunner._setup` makes the tile
        sub-folder from it, so the default decides where every overview of every
        experiment is written. Asserted on what the *runner* receives rather than on the
        text box, because that is the value that names the directory -- a box seeded
        correctly and then dropped somewhere along the handover would still pass.

        A prefix, not the whole name: the run appends the time it started, so that two
        of them cannot land in one directory. See `TestTwoRunsCannotLandOnEachOther`.
        """
        captured = {}

        def fake_worker(fn, *args):
            captured["settings"] = args[0]

            class _W:
                def start(self_inner):
                    pass

                def is_alive(self_inner):
                    return False

            return _W()

        monkeypatch.setattr(
            "fibsem.ui.widgets.overview_widget.FunctionWorker", fake_worker
        )
        widget.acquire()
        assert captured["settings"].image_settings.filename.startswith("overview-image")
        # And visible, so it can be changed before a run rather than discovered after.
        assert (
            widget.settings_widget.filename_edit.text()
            == "overview-image"
        )

    def test_a_run_starts_from_the_overview_defaults(self, widget, monkeypatch):
        """A 500 um tile with autocontrast, not `ImageSettings`'s generic 150 um.

        The tab referenced the overview defaults nowhere at all, so it opened at
        whatever `ImageSettingsWidget` happens to default to. The tile field, dwell and
        autocontrast below all differ from that, which is the point: a tab that looks
        right and images a ninth of the area asked for is not something the picture
        shows you. The resolution is the exception and deliberately so -- see
        `test_both_overview_tabs_open_at_the_shipped_resolution`.

        Asserted on the runner's copy for the same reason the filename is -- these are
        the numbers that reach the instrument.
        """
        captured = {}

        def fake_worker(fn, *args):
            captured["settings"] = args[0]

            class _W:
                def start(self_inner):
                    pass

                def is_alive(self_inner):
                    return False

            return _W()

        monkeypatch.setattr(
            "fibsem.ui.widgets.overview_widget.FunctionWorker", fake_worker
        )
        widget.acquire()
        settings = captured["settings"]
        assert settings.image_settings.hfw == pytest.approx(500e-6)
        assert settings.image_settings.dwell_time == pytest.approx(1e-6)
        assert settings.image_settings.autocontrast is True
        assert tuple(settings.image_settings.resolution) == (1536, 1024)
        assert (settings.nrows, settings.ncols) == (3, 3)


class TestTheDefaultsAreNotShared:
    """The defaults are a factory, and the one thing a factory must not do is hand back
    the same object twice.

    What it replaced was a module-level `DEFAULT_OVERVIEW_ACQUISITION_SETTINGS`, and the
    minimap assigned an experiment path straight into it. That edits the default: open a
    second experiment and the "default" carries the first one's path. Worse in this
    widget, where `ImageSettingsWidget.update_from_settings` keeps the object it is
    handed and `get_settings` mutates and returns that same one -- so a shared default
    would be rewritten by every keystroke in the tab.
    """

    def test_both_overview_tabs_open_at_the_shipped_resolution(self):
        """One shape for both tabs, and it is the one the shipped tab acquires at.

        The minimap named no `resolution`, so it inherited `ImageSettings`' non-square
        default -- and that is what every overview taken from that tab has been. The
        factory opened square instead (FIB-619), so for a while the two tabs disagreed
        and the minimap pinned its own value on top. They agree now: the factory names
        the standard shape, and the minimap sets nothing.

        Not cosmetic. `hfw` is the *horizontal* field, so the same 500 um tile at
        1024x1024 is 0.49 um/px against 0.33 and half again as tall -- overviews from
        the two tabs would cover different ground under the same nominal field, which
        is no way to compare them across the swap.

        Pinned twice on purpose. The literal is the decision; the comparison with
        `ImageSettings()` is where the number came from. If that default ever moves,
        the literal fires and asks whether the overview default follows -- which is the
        question, not the number.
        """
        from fibsem.structures import ImageSettings
        from fibsem.ui.widgets.overview_acquisition_settings_widget import (
            default_overview_acquisition_settings,
        )

        resolution = tuple(
            default_overview_acquisition_settings().image_settings.resolution
        )
        assert resolution == (1536, 1024)
        assert resolution == tuple(ImageSettings().resolution)

    def test_each_call_hands_back_its_own_settings(self):
        from fibsem.ui.widgets.overview_acquisition_settings_widget import (
            default_overview_acquisition_settings,
        )

        first = default_overview_acquisition_settings()
        first.image_settings.path = "/experiments/one"
        first.image_settings.hfw = 1e-3
        first.nrows = 7

        second = default_overview_acquisition_settings()
        assert second.image_settings.path is None
        assert second.image_settings.hfw == pytest.approx(500e-6)
        assert second.nrows == 3

    def test_the_widget_does_not_hold_the_defaults(self, qapp):
        """Seeding a widget must not leave it editing the shared object either.

        Constructing two and typing into one is the cheapest way to catch a factory
        that returns a cached instance.
        """
        from fibsem.ui.widgets.overview_acquisition_settings_widget import (
            OverviewAcquisitionSettingsWidget,
        )

        first = OverviewAcquisitionSettingsWidget()
        second = OverviewAcquisitionSettingsWidget()
        try:
            first.image_settings_widget.hfw_spinbox.setValue(42.0)
            first.set_grid_size(5, 4)

            settings = second.get_settings()
            assert settings.image_settings.hfw == pytest.approx(500e-6)
            assert (settings.nrows, settings.ncols) == (3, 3)
        finally:
            first.close()
            second.close()


class TestViews:
    """Images from different beams or orientations must not share the canvas.

    A view is (beam, orientation), and two images register only if they share one:
    otherwise they are pictures of the same sample from different directions, and
    compositing them says something untrue. The canvas shows one view at a time.

    These use an **Aquilos2** configuration rather than the module's compustage
    microscope, deliberately. On a compustage every orientation shares one rotation, so
    the views are far less distinguishable — which is exactly how the first version of
    this widget mixed them without anything looking wrong.
    """

    @staticmethod
    def _scope():
        import fibsem.config as fibsem_config

        path = os.path.join(
            os.path.dirname(fibsem_config.__file__),
            "config",
            "tfs-aquilos2-configuration.yaml",
        )
        scope, _ = utils.setup_session(manufacturer="Demo", config_path=path)
        return scope

    @staticmethod
    def _at_orientation(scope, name, dx=0.0):
        pose = scope.get_orientation(name)
        return FibsemStagePosition(x=dx, y=0.0, z=0.0, r=pose.r, t=pose.t)

    def _image(self, scope, orientation, beam_type, dx=0.0):
        position = self._at_orientation(scope, orientation, dx)
        image = FibsemImage.generate_blank_image(resolution=(128, 128), hfw=128 * 2e-7)
        image.data = (np.random.default_rng(0).random((128, 128)) * 255).astype(np.uint8)
        state = scope.get_microscope_state(beam_type=beam_type)
        state.stage_position = position
        image.metadata.image_settings = ImageSettings(
            hfw=128 * 2e-7, beam_type=beam_type
        )
        image.metadata.microscope_state = state
        image.metadata.system_info = scope.system.info
        image.metadata.hardware_geometry = scope.hardware_geometry()
        return image

    @pytest.fixture
    def widget(self):
        w = FibsemOverviewWidget(self._scope())
        w.resize(900, 700)
        yield w
        w.close()

    def test_a_view_is_derived_from_the_image_not_asked_for(self, widget):
        scope = widget.microscope
        view = widget._view_of(self._image(scope, "FIB", BeamType.ION))
        assert view.orientation == "FIB"
        assert view.beam_type is BeamType.ION

    def test_images_from_another_view_do_not_share_the_canvas(self, widget):
        scope = widget.microscope
        widget.set_image(self._image(scope, "SEM", BeamType.ELECTRON))
        widget.set_image(self._image(scope, "SEM", BeamType.ELECTRON, dx=60e-6))
        assert len(widget.canvas.placed_keys) == 2

        widget.set_image(self._image(scope, "FIB", BeamType.ION))
        assert widget.current_view.orientation == "FIB"
        assert len(widget.canvas.placed_keys) == 1, (
            "the SEM images are still on the canvas beneath a FIB one"
        )
        assert len(widget.views) == 2, "both views should be known"

    def test_switching_back_re_places_the_view_that_was_left(self, widget):
        scope = widget.microscope
        widget.set_image(self._image(scope, "SEM", BeamType.ELECTRON))
        widget.set_image(self._image(scope, "SEM", BeamType.ELECTRON, dx=60e-6))
        sem_view = widget.current_view
        widget.set_image(self._image(scope, "FIB", BeamType.ION))

        assert widget.show_view(sem_view) is True
        assert widget.current_view == sem_view
        assert len(widget.canvas.placed_keys) == 2, "the SEM images did not come back"
        assert widget.show_view(sem_view) is False, "switching to the current view is a no-op"

    def test_each_view_keeps_its_own_origin(self, widget):
        """Everything in a view is placed relative to that view's anchor. One shared
        origin would put the FIB tiles wherever the SEM overview happened to start.

        Looked up by view rather than counted: the widget also anchors the view the
        *stage* is in, before anything is acquired, so the dict holds more than the
        views images were placed in — and that provisional anchor is a third entry
        whose pose says nothing about these two.
        """
        scope = widget.microscope
        widget.set_image(self._image(scope, "SEM", BeamType.ELECTRON))
        widget.set_image(self._image(scope, "FIB", BeamType.ION))

        sem = OverviewView(beam_type=BeamType.ELECTRON, orientation="SEM")
        fib = OverviewView(beam_type=BeamType.ION, orientation="FIB")
        assert sem in widget._origins and fib in widget._origins
        assert widget._origins[sem].t != pytest.approx(widget._origins[fib].t), (
            "the two views share an origin pose"
        )
        # Both were fixed by an image, so neither is still the stage's stand-in.
        assert not ({sem, fib} & widget._provisional)

    def test_the_projection_is_per_view(self, widget):
        """The stale-cache bug: after switching beam, drawing kept using the electron
        projection (view tilt 0) while a click resolved through the ion one (0.91 rad),
        so markers were drawn in one projection and clicks answered in another."""
        scope = widget.microscope
        widget.set_image(self._image(scope, "SEM", BeamType.ELECTRON))
        widget.set_image(self._image(scope, "FIB", BeamType.ION))

        tilts = {v.label: widget._projection(v)._view_tilt() for v in widget.views}
        assert len(set(round(t, 6) for t in tilts.values())) == 2, (
            f"both views resolved to the same projection: {tilts}"
        )

    def test_the_planned_footprint_is_only_drawn_where_the_run_would_land(self, widget):
        """Drawn on another view it promises coverage this canvas will never show."""
        scope = widget.microscope
        widget._stage_position = self._at_orientation(scope, "SEM")
        widget.set_image(self._image(scope, "SEM", BeamType.ELECTRON))
        widget._refresh_context_overlays()
        assert widget.tile_grid_overlay._tiles, "the grid is missing in its own view"

        # Same canvas, but now the stage is at FIB: the next run lands elsewhere.
        widget._stage_position = self._at_orientation(scope, "FIB")
        widget._refresh_context_overlays()
        assert not widget.tile_grid_overlay._tiles, (
            "the grid was drawn on a view the run will not appear in"
        )

    def test_it_says_when_the_stage_is_looking_somewhere_else(self, widget):
        """Silent when they agree -- a note that is always there stops being read."""
        scope = widget.microscope
        widget._stage_position = self._at_orientation(scope, "SEM")
        widget.set_image(self._image(scope, "SEM", BeamType.ELECTRON))
        widget._refresh_view_note()
        assert not widget.label_view_note.text()

        widget._stage_position = self._at_orientation(scope, "FIB")
        widget._refresh_view_note()
        assert "the stage is at" in widget.label_view_note.text()

    def test_an_overview_still_reports_itself_from_another_view(self, widget):
        """The list describes what an overview *is*, not what happens to be drawn.
        Deriving the row from the canvas made one report nothing merely because you
        were looking at a different view."""
        scope = widget.microscope
        widget.set_image(self._image(scope, "SEM", BeamType.ELECTRON))
        sem_record = widget.overviews[0]
        described = sem_record.detail
        assert described, "the row said nothing about the overview it lists"

        widget.set_image(self._image(scope, "FIB", BeamType.ION))
        assert sem_record.keys == [], "the SEM record should not be on the canvas now"
        assert sem_record.detail == described, (
            f"reads {sem_record.detail!r} while another view is displayed, "
            f"and {described!r} while it is"
        )

    def test_clicking_a_view_the_stage_is_not_in_is_refused(self, widget, monkeypatch):
        """Resolving a click through another view names a point as *that* view sees it,
        and reaching it would rotate and tilt the stage to match — a far bigger move
        than clicking a picture looks like."""
        scope = widget.microscope
        widget._stage_position = self._at_orientation(scope, "SEM")
        widget.set_image(self._image(scope, "SEM", BeamType.ELECTRON))
        assert widget._stage_position_at(0.0, 0.0) is not None, "blocked in its own view"

        widget._stage_position = self._at_orientation(scope, "FIB")
        assert widget._stage_position_at(0.0, 0.0) is None, (
            "a click resolved through a view the stage is not in"
        )

    def test_the_refusal_says_what_to_do_about_it(self, widget, monkeypatch):
        """Either way out can be the right one — you may have switched view to look at
        something, or moved the stage and not switched — so the message names both."""
        scope = widget.microscope
        widget._stage_position = self._at_orientation(scope, "SEM")
        widget.set_image(self._image(scope, "SEM", BeamType.ELECTRON))
        widget._stage_position = self._at_orientation(scope, "FIB")

        toasts = []
        monkeypatch.setattr(
            "fibsem.ui.widgets.overview_widget.notification_service.show_toast",
            lambda message, *a, **k: toasts.append(message),
        )
        assert widget._stage_position_at(0.0, 0.0) is None
        assert toasts, "the click was blocked silently"
        message = toasts[-1]
        assert "SEM" in message and "FIB" in message, message
        assert "Switch the view" in message and "move the stage" in message, message

    def test_marking_is_refused_from_another_view_too(self, widget, monkeypatch):
        """A position marked through the wrong view would be recorded at an orientation
        the instrument is not in."""
        scope = widget.microscope
        widget._stage_position = self._at_orientation(scope, "SEM")
        widget.set_image(self._image(scope, "SEM", BeamType.ELECTRON))
        widget._stage_position = self._at_orientation(scope, "FIB")
        monkeypatch.setattr(
            "fibsem.ui.widgets.overview_widget.notification_service.show_toast",
            lambda *a, **k: None,
        )
        assert widget._position_menu(0.0, 0.0) is None

    def test_steering_never_reorients_the_stage(self, widget):
        """The click says where on the sample, not which way to look at it.

        Without this the target carries the view *origin's* pose, so a stage sitting at
        a slightly different tilt within the same orientation gets tilted back to it as
        a side effect of clicking.
        """
        scope = widget.microscope
        widget.set_image(self._image(scope, "SEM", BeamType.ELECTRON))
        origin = widget._origins[widget.current_view]

        # Same orientation, but not the exact pose the view was anchored at.
        drifted = FibsemStagePosition(
            x=0.0, y=0.0, z=0.0, r=origin.r, t=origin.t + np.deg2rad(0.4)
        )
        widget._stage_position = drifted

        target = widget._stage_position_at(50.0, 50.0)
        assert target is not None
        assert target.r == pytest.approx(drifted.r), "the click changed the rotation"
        assert target.t == pytest.approx(drifted.t), "the click changed the tilt"

    def test_markers_reproject_when_the_view_changes(self, widget):
        """A stage position is 3-D, so it is drawable in any view -- it just lands
        somewhere else. Switching view has to move the markers with it."""
        scope = widget.microscope
        widget.set_image(self._image(scope, "SEM", BeamType.ELECTRON))
        sem_view = widget.current_view
        mark = self._at_orientation(scope, "SEM", dx=120e-6)
        mark.name = "A"
        widget.set_positions([mark])
        in_sem = list(widget.position_overlay._points)

        widget.set_image(self._image(scope, "FIB", BeamType.ION))
        widget.set_positions([mark])
        in_fib = list(widget.position_overlay._points)

        assert in_sem and in_fib
        assert in_sem != in_fib, "the marker did not move when the view did"

        widget.show_view(sem_view)
        widget.set_positions([mark])
        assert widget.position_overlay._points == pytest.approx(in_sem), (
            "coming back to a view did not restore where its markers sit"
        )


class TestTheCanvasDrawsBeforeAnythingIsAcquired:
    """Opening the tab has to show where you are, not a black rectangle.

    A frame needs a projection, a scale and an origin. The projection comes off the
    instrument, but the other two used to arrive only with the first image -- so the
    tab was blank until a tile landed, and then everything appeared at once. That is
    the wrong way round: a planned footprint, the travel limits and the lamella
    markers are worth the most *before* you acquire.
    """

    @staticmethod
    def _at(scope, orientation):
        pose = scope.get_orientation(orientation)
        return FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=pose.r, t=pose.t)

    def test_a_freshly_opened_tab_has_a_frame(self, widget):
        assert widget._frame() is not None, "nothing can be drawn in stage coordinates"
        assert widget.current_view is not None, "no view to draw in"
        assert widget.canvas.reference_pixel_size, "the canvas has no scale"

    def test_it_draws_the_context_with_nothing_placed(self, widget):
        """The whole point. Every one of these used to wait for the first tile."""
        assert not widget.canvas.placed_keys, "this test is about the empty canvas"
        drawn = {spec.label for spec in widget.context_overlay._specs}
        assert "Stage Limits" in drawn
        assert "Grid Boundary" in drawn
        assert widget.tile_grid_overlay._tiles, "the planned run is not shown"
        assert widget.current_position_overlay._points, "the stage is not marked"

    def test_marked_positions_appear_before_any_image(self, widget, microscope):
        """The host hands over the experiment's lamellae as soon as one is loaded, and
        that is usually before anything has been acquired in this session."""
        mark = self._at(microscope, "SEM")
        mark.name = "L1"
        widget.set_positions([mark])
        assert widget.position_overlay._points, "a marked lamella was not drawn"

    def test_the_scale_comes_from_the_settings(self, widget):
        """A widget read, not a device one — and it tracks the field of view, so the
        planned footprint is drawn at the size the next run would actually cover."""
        settings = widget._settings()
        expected = settings.image_settings.hfw / settings.image_settings.resolution[0]
        assert widget.canvas.reference_pixel_size == pytest.approx(expected)

    def test_the_first_image_replaces_the_stand_in_anchor(self, widget, microscope):
        """The seeded anchor is where the stage happened to be when the tab opened,
        which can be millimetres from the data. Nothing is placed against it yet, so
        the first real image takes the anchor over rather than sitting at an offset
        from a position that means nothing."""
        view = widget.current_view
        assert view in widget._provisional, "the anchor was not marked provisional"

        base = microscope.get_stage_position()
        far = _at(base, dx=4e-3, dy=2e-3)
        widget.place_image(_tile(microscope, far), key="first")

        assert view not in widget._provisional
        assert widget._origins[view].x == pytest.approx(far.x)
        assert widget._origins[view].y == pytest.approx(far.y)

    def test_an_image_is_drawn_at_its_own_size_whatever_the_scale_was_seeded_to(
        self, widget, microscope
    ):
        """The seeded scale is only the canvas unit — how many metres a canvas pixel is
        worth. An image coarser or finer than the plan is still drawn covering the
        ground it images, because `add_image` scales each one by its own pixel size."""
        hfw = 37e-6  # deliberately unlike the settings' field of view
        base = microscope.get_stage_position()
        widget.place_image(
            _tile(microscope, _at(base), shape=(64, 64), hfw=hfw), key="odd"
        )
        extent = widget.canvas._placed["odd"].extent
        drawn = (extent[1] - extent[0]) * widget.canvas.reference_pixel_size
        assert drawn == pytest.approx(hfw, rel=1e-9)


class TestOverlaysAreDrawnInTheView:
    """Anything measured in *stage* space has to be projected before it is drawn.

    A view foreshortens stage y and leaves x alone, by a factor that changes with the
    beam and the pose: 1.00 looking down the pose a beam is named after, 0.26 for the
    ion beam at the milling pose. Sized by the canvas scale alone -- which is what
    `StageFrame.length()` gives -- the travel envelope and the grid boundary come out
    the same in every view, and are therefore right in at most one.

    The gridbar lattice has the identical bug and is deliberately not fixed here: it
    still takes one pitch for both axes. Its *centre* is covered below, which was a
    different fault. See FIB-615.

    Every assertion here is against `frame.to_canvas`, the mapping that puts a *marker*
    on the canvas. That is the point: an overlay and a marker describing the same piece
    of stage have to agree, and the bug was that one of them projected and the other
    did not. A test that recomputed the extent the way `_limit_shapes` does could not
    have failed.

    A compustage, because that is the only stage the limits are drawn on -- and it is
    also where all three orientations share a rotation, so nothing here is really about
    the 180-degree flip.
    """

    # Ion at the milling pose: the view you mill in, and the worst case at 0.259.
    FORESHORTENED = ("MILLING", BeamType.ION)
    # Electron at the SEM pose: the one view that is *not* foreshortened, so it is
    # what the old, unprojected code happened to be right for.
    SQUARE = ("SEM", BeamType.ELECTRON)

    @staticmethod
    def _image(scope, orientation, beam_type):
        pose = scope.get_orientation(orientation)
        position = FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=pose.r, t=pose.t)
        hfw = 128 * 2e-7
        image = FibsemImage.generate_blank_image(resolution=(128, 128), hfw=hfw)
        image.data = (np.random.default_rng(0).random((128, 128)) * 255).astype(np.uint8)
        state = scope.get_microscope_state(beam_type=beam_type)
        state.stage_position = position
        image.metadata.image_settings = ImageSettings(hfw=hfw, beam_type=beam_type)
        image.metadata.microscope_state = state
        image.metadata.system_info = scope.system.info
        image.metadata.hardware_geometry = scope.hardware_geometry()
        return image

    @staticmethod
    def _spec(widget, kind, label):
        for spec in widget.context_overlay._specs:
            if spec.kind == kind and spec.label == label:
                return spec
        return None

    def _show(self, widget, view):
        """Put the canvas in *view* and return its frame."""
        widget.set_image(self._image(widget.microscope, *view))
        return widget._frame()

    @staticmethod
    def _marker_span(widget, frame, dx=0.0, dy=0.0):
        """How far a stage step of (dx, dy) carries a *marker*, in canvas pixels.

        The independent authority. Everything drawn from a stage position goes through
        this same call, so an overlay that disagrees with it is drawn in a frame
        nothing else is using.
        """
        origin = frame.origin
        here = frame.to_canvas(
            FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=origin.r, t=origin.t)
        )
        there = frame.to_canvas(
            FibsemStagePosition(x=dx, y=dy, z=0.0, r=origin.r, t=origin.t)
        )
        return abs(there[0] - here[0]), abs(there[1] - here[1])

    def test_the_limits_box_lands_where_a_marker_at_the_limits_would(
        self, widget, microscope
    ):
        """The strongest statement of the property: put the travel envelope's corner on
        the canvas as a *marked position*, and the corner of the drawn box has to be
        under it. Checked in the foreshortened view, where the two used to differ."""
        assert microscope.stage_is_compustage, "the limits only draw on a compustage"
        frame = self._show(widget, self.FORESHORTENED)
        limits = microscope._stage.limits

        box = self._spec(widget, "rect", "Stage Limits")
        assert box is not None, "the stage limits were not drawn"

        corner = frame.to_canvas(
            widget._landmark(frame, limits["x"].max, limits["y"].max)
        )
        assert abs(corner[0] - box.cx) == pytest.approx(box.width / 2, rel=1e-9)
        assert abs(corner[1] - box.cy) == pytest.approx(box.height / 2, rel=1e-9)

    def test_the_limits_box_is_shorter_in_a_foreshortened_view(self, widget, microscope):
        """The bug, pinned. The box was `frame.length(ymax - ymin)` in every view, so
        it was the same height at the milling pose as at the SEM one -- nearly four
        times too tall in the view you mill in."""
        limits = microscope._stage.limits
        span_y = limits["y"].max - limits["y"].min
        span_x = limits["x"].max - limits["x"].min

        heights, widths = {}, {}
        for view in (self.SQUARE, self.FORESHORTENED):
            frame = self._show(widget, view)
            box = self._spec(widget, "rect", "Stage Limits")
            assert box is not None
            # Against the marker path, per view, so this says "the box matches what a
            # marker would do" rather than "the box matches the formula that drew it".
            expected = self._marker_span(widget, frame, dx=span_x, dy=span_y)
            assert box.width == pytest.approx(expected[0], rel=1e-9)
            assert box.height == pytest.approx(expected[1], rel=1e-9)
            heights[view], widths[view] = box.height, box.width

        assert widths[self.SQUARE] == pytest.approx(widths[self.FORESHORTENED]), (
            "x is not foreshortened, so the box must keep its width"
        )
        assert heights[self.FORESHORTENED] < heights[self.SQUARE] / 2, (
            f"the box is {heights[self.FORESHORTENED]:.1f} px tall at the milling pose "
            f"and {heights[self.SQUARE]:.1f} px at the SEM pose — it did not foreshorten"
        )

    def test_the_grid_boundary_is_an_ellipse_where_the_view_is_not_square(self, widget):
        """A circle on the sample is a circle on screen in two views and an ellipse in
        every other. Drawn as a circle it claimed the grid was round from a direction
        it is not."""
        frame = self._show(widget, self.FORESHORTENED)
        boundary = self._spec(widget, "ellipse", "Grid Boundary")
        assert boundary is not None, "the grid boundary was not drawn as an ellipse"

        expected = self._marker_span(
            widget, frame, dx=GRID_BOUNDARY_RADIUS_M, dy=GRID_BOUNDARY_RADIUS_M
        )
        assert boundary.width == pytest.approx(2 * expected[0], rel=1e-9)
        assert boundary.height == pytest.approx(2 * expected[1], rel=1e-9)
        assert boundary.height < boundary.width / 2, "it is still round"

    def test_the_grid_boundary_stays_round_in_a_square_view(self, widget):
        """The other half of the same statement, and what stops the fix being "make it
        an ellipse and hope": looking down the pose the beam is named after, the grid
        really is round."""
        self._show(widget, self.SQUARE)
        boundary = self._spec(widget, "ellipse", "Grid Boundary")
        assert boundary is not None
        assert boundary.width == pytest.approx(boundary.height, rel=1e-9)

    def test_the_lattice_is_centred_where_the_grid_centre_marker_is(self):
        """Grid centre is a *place*, and a place has no rotation of its own.

        Built with `r=0` it looked, to the projection, like a position recorded half a
        turn from the view — which on a stage that reaches the ion beam by rotating is
        exactly what a FIB overview is. So the compucentric correction fired on a
        landmark that was never anywhere, and centred the whole lattice **1.96 mm**
        away from the origin it names.

        Invisible on a compustage, where every orientation shares one rotation, which
        is why this needs its own microscope. Asserted against a marker at the same
        stage position: the two describe the same point and a user sees them together.
        """
        import fibsem.config as fibsem_config

        path = os.path.join(
            os.path.dirname(fibsem_config.__file__),
            "config",
            "tfs-aquilos2-configuration.yaml",
        )
        scope, _ = utils.setup_session(manufacturer="Demo", config_path=path)
        assert not scope.stage_is_compustage, "this is the standard-stage case"

        widget = FibsemOverviewWidget(scope)
        try:
            widget.set_image(self._image(scope, "FIB", BeamType.ION))
            widget.overlay_controls.set_visible("gridbars", True)

            pose = scope.get_orientation("FIB")
            centre = FibsemStagePosition(
                name="Grid Centre", x=0.0, y=0.0, z=0.0, r=pose.r, t=pose.t
            )
            widget.set_positions([centre])

            marker = widget.position_overlay._points[0]
            assert widget.gridbar_overlay._centre == pytest.approx(marker, abs=1e-6), (
                "the grid bars are not centred on the grid centre"
            )
        finally:
            widget.close()


class TestTheHolderIsDrawnOnEveryStage:
    """The travel envelope and the grid boundaries are separate questions.

    One guard used to answer both: `_limit_shapes` returned nothing at all unless the
    stage was a compustage, so a standard stage lost the *travel box* along with the
    grid circle -- and travel limits have nothing to do with grids.

    And the boundary was one circle at the stage origin, which is a compustage's
    holder: a single grid, at zero. The shipped `default-sample-holder.yaml` is a
    2-slot 35-degree shuttle with grids at x = -5 mm and +5 mm, where a circle at the
    origin marks a place no grid is. A grid is 1 mm in radius whatever holds it, so
    what the boundary needs is where the grids are, and the holder already says.
    """

    @staticmethod
    def _slot(name, x, y=0.0):
        from fibsem.microscopes._stage import GridSlot

        return GridSlot(
            name=name, index=0,
            position=FibsemStagePosition(name=name, x=x, y=y, z=0.0),
        )

    def _two_slots(self, widget, monkeypatch):
        """A multi-grid shuttle, in place of the simulator's single centred slot.

        Set on the instance: `slots` is a dataclass field, so patching the class puts
        a `property` object where a dict belongs.
        """
        holder = widget.microscope._stage.holder
        slots = {"Slot-01": self._slot("Slot-01", -5.0e-3),
                 "Slot-02": self._slot("Slot-02", 5.0e-3)}
        monkeypatch.setattr(holder, "slots", slots)
        widget._refresh_context_overlays()
        return slots

    @staticmethod
    def _specs(widget, kind, label=None):
        return [s for s in widget.context_overlay._specs
                if s.kind == kind and (label is None or s.label == label)]

    def test_the_travel_box_draws_on_a_stage_that_is_not_a_compustage(
        self, widget, microscope, monkeypatch
    ):
        """The half that was collateral damage. Travel limits are a property of the
        stage, and every stage that declares them has them."""
        monkeypatch.setattr(
            type(microscope), "stage_is_compustage", property(lambda self: False)
        )
        widget._refresh_context_overlays()

        assert self._specs(widget, "rect", "Stage Limits"), (
            "a standard stage was left with no travel envelope drawn"
        )

    def test_a_slot_with_no_rotation_still_draws(self, widget, monkeypatch):
        """The defect the boundary work walked into. `default-sample-holder.yaml` gives
        each slot three numbers -- x, y, z -- so `SampleHolder.load` leaves `r` and `t`
        as None, and `frame.to_canvas` raises `TypeError` on them. `_slot_shapes` caught
        that and moved on, so the shipped two-slot shuttle drew **no slot markers at
        all**, silently. Only the simulator's holder escaped it, because `_ensure_slots`
        invents its slot with r=0.
        """
        slots = self._two_slots(widget, monkeypatch)
        assert all(s.position.r is None for s in slots.values()), (
            "this no longer reproduces the shipped holder"
        )
        assert len(self._specs(widget, "crosshair")) >= 2, "the slot markers vanished"

    def test_a_boundary_is_drawn_around_every_slot(self, widget, monkeypatch):
        slots = self._two_slots(widget, monkeypatch)
        boundaries = self._specs(widget, "ellipse", "Grid Boundary")
        assert len(boundaries) == len(slots) == 2

    def test_each_boundary_is_centred_on_its_own_slot(self, widget, monkeypatch):
        """Not on the stage origin, which is where the single circle used to go -- and
        with a shuttle that is 10 mm across, a full grid's width from either of them."""
        self._two_slots(widget, monkeypatch)
        boundaries = self._specs(widget, "ellipse", "Grid Boundary")
        crosshairs = [s for s in self._specs(widget, "crosshair")
                      if s.label.startswith("Slot-")]

        centres = sorted((round(s.cx, 6), round(s.cy, 6)) for s in boundaries)
        markers = sorted((round(s.cx, 6), round(s.cy, 6)) for s in crosshairs)
        assert centres == markers, "a boundary is not concentric with its slot marker"
        assert len({c[0] for c in centres}) == 2, "both circles landed in one place"

    def test_a_slot_carries_the_orientation_it_was_defined_in(self, widget, monkeypatch):
        """The holder file is written in the SEM orientation, so that is the pose a
        slot has to carry. Without it the position is a bare x/y and `to_canvas` has
        nothing to compare against -- which is both why the shipped holder crashed and
        why it could never be re-expressed for a re-posed stage.

        Carrying the pose is what lets `BeamStageProjection` decide: on a compustage
        every orientation shares one rotation and this is the identity, and on a
        standard stage a view half a turn away gets the compucentric flip, exactly as a
        marked position does.
        """
        self._two_slots(widget, monkeypatch)
        sem = widget.microscope.get_orientation("SEM")

        place = widget._slot_landmark(
            list(widget.microscope._stage.holder.slots.values())[0]
        )

        assert place.r == pytest.approx(sem.r)
        assert place.t == pytest.approx(sem.t)
        assert place.x == pytest.approx(-5.0e-3), "the slot's own position was lost"

    def test_a_single_centred_slot_still_draws_the_one_circle(self, widget):
        """The compustage case, unchanged: one slot at the origin, one circle on it.
        The simulator's own holder, so this is the behaviour that shipped."""
        boundaries = self._specs(widget, "ellipse", "Grid Boundary")
        crosshairs = [s for s in self._specs(widget, "crosshair")
                      if s.label.startswith("Slot-")]

        assert len(boundaries) == 1
        assert (boundaries[0].cx, boundaries[0].cy) == pytest.approx(
            (crosshairs[0].cx, crosshairs[0].cy)
        )


class TestLifecycle:
    def test_closing_releases_every_microscope_subscription(self, microscope):
        """psygnal subscriptions outlive the widget -- they belong to the microscope --
        and hold bound methods of Qt objects `close` has torn down on the C++ side. The
        equivalent leak on the FM overview was a hard segfault, not an exception."""
        widget = FibsemOverviewWidget(microscope)
        signals = (
            microscope.tiled_acquisition_signal,
            microscope.stage_position_changed,
        )
        before = [len(s) for s in signals]
        widget.close()
        after = [len(s) for s in signals]

        assert all(a < b for a, b in zip(after, before)), (
            f"a subscription survived close(): {before} -> {after}"
        )

    def test_it_holds_no_napari_viewer(self, microscope):
        """The tab it replaces took a `napari.Viewer` as its first argument."""
        widget = FibsemOverviewWidget(microscope)
        assert not hasattr(widget, "viewer")
        with pytest.raises(TypeError):
            FibsemOverviewWidget(microscope, viewer=object())  # type: ignore[call-arg]
        widget.close()


class TestTheTileMaskSurvivesTheSettingsWidget:
    """A mask is carried by the settings widget, though nothing draws one yet.

    `get_settings()` builds a *new* `OverviewAcquisitionSettings` on every call, and it
    is called on every overlay refresh -- so a mask set anywhere else would be dropped
    almost immediately unless the widget holds it. The control that sets it is the tile
    grid on the canvas (FIB-617); this is the plumbing it lands on.
    """

    @staticmethod
    def _mask(rows, cols, disabled=()):
        disabled = set(disabled)
        return [[(i, j) not in disabled for j in range(cols)] for i in range(rows)]

    def test_setting_both_dimensions_notifies_once(self, qapp):
        """The shared panel's own contract, tested on this side for the first time.

        It emitted twice: the mask resize was done outside the block, and the mask emits
        `changed` when its shape moves, so `set_grid_size` fired once for the mask and
        once for itself -- the very thing the method exists to prevent. Invisible when a
        spin box is nudged by hand; an edge drag on the canvas does it on every motion
        event, refreshing the overlay twice against a size asked for once.

        The fluorescence tab always blocked the mask, and adopting this widget is what
        brought its test to bear (FIB-696). Guarded here too, because the beam tab is
        where the widget lives and where the next edit to it will be made.
        """
        from fibsem.ui.widgets.overview_grid_settings_widget import (
            OverviewGridSettingsWidget,
        )

        grid = OverviewGridSettingsWidget()
        seen = []
        grid.changed.connect(lambda: seen.append((grid.rows, grid.cols)))

        grid.set_grid_size(2, 7)

        assert seen == [(2, 7)]
        assert (grid.tile_mask.grid._rows, grid.tile_mask.grid._cols) == (2, 7)

    def test_no_mask_by_default(self, widget):
        assert widget._settings().tile_mask is None

    def test_a_mask_reaches_the_settings(self, widget):
        settings_widget = widget.settings_widget
        settings_widget.grid.spin_rows.setValue(3)
        settings_widget.grid.spin_cols.setValue(3)
        settings_widget.tile_mask = self._mask(3, 3, disabled=[(1, 1)])

        settings = widget._settings()
        assert settings.tile_mask[1][1] is False
        assert settings.n_enabled_tiles == 8

    def test_reading_twice_does_not_lose_it(self, widget):
        """The failure this exists to prevent: every overlay refresh reads the widget."""
        settings_widget = widget.settings_widget
        settings_widget.tile_mask = self._mask(3, 3, disabled=[(0, 0)])
        for _ in range(3):
            widget._refresh_context_overlays()
        assert widget._settings().tile_mask is not None

    def test_resizing_the_grid_keeps_the_tiles_that_still_exist(self, widget):
        """Rows and columns are spin boxes and the mask is positional, so the two go out
        of step the moment either moves -- and `compute_tile_grid` rejects a mismatched
        mask outright, so it cannot simply be carried on.

        **Remapped, not dropped**, which reverses what this tab used to do. The old rule
        was that growing has to invent whether new tiles are in or out and shrinking
        throws a choice away, so it dropped any mask whose shape no longer matched. That
        was defensible when the only way to resize was to nudge a spin box. It is not
        now that dragging a grid edge on the canvas resizes on *every motion event*:
        under the old rule, tweaking a 3x3 to a 4x3 silently discards a selection built
        by hand. Inventing one column of enabled tiles is the smaller surprise, and it
        is what the fluorescence tab already did -- so the same gesture now means the
        same thing on both.
        """
        settings_widget = widget.settings_widget
        settings_widget.grid.spin_rows.setValue(3)
        settings_widget.grid.spin_cols.setValue(3)
        settings_widget.tile_mask = self._mask(3, 3, disabled=[(0, 0)])
        assert widget._settings().n_enabled_tiles == 8

        settings_widget.grid.spin_cols.setValue(4)
        mask = widget._settings().tile_mask
        assert mask is not None, "the selection was thrown away by a resize"
        assert len(mask) == 3 and all(len(row) == 4 for row in mask), (
            f"the mask did not follow the grid: {mask}"
        )
        assert mask[0][0] is False, "the tile that was turned off came back on"
        assert widget._settings().n_enabled_tiles == 11  # 12 - the one still disabled


class TestThePlannedTileset:
    """The planned run is drawn tile by tile, and can be planned somewhere else.

    A single rectangle said how much ground a run would cover and nothing more. What
    is worth seeing before pressing the button is where the seams fall, which tiles
    are in, and -- once the grid can be dragged -- whether it covers the thing you
    actually want.
    """

    @staticmethod
    def _at(scope, orientation):
        pose = scope.get_orientation(orientation)
        return FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=pose.r, t=pose.t)

    def test_the_grid_is_drawn_as_tiles(self, widget):
        overlay = widget.tile_grid_overlay
        settings = widget._settings()
        assert len(overlay._tiles) == settings.nrows * settings.ncols
        assert overlay._anchor() is not None

    def test_dragging_an_edge_resizes_the_grid(self, widget):
        widget._on_grid_resized(2, 5)
        assert widget._settings().nrows == 2
        assert widget._settings().ncols == 5
        assert len(widget.tile_grid_overlay._tiles) == 10

    def test_clicking_a_tile_takes_it_out_of_the_run(self, widget):
        widget._on_tile_toggled(0, 1, False)

        settings = widget._settings()
        assert settings.tile_mask[0][1] is False
        assert settings.n_enabled_tiles == settings.nrows * settings.ncols - 1
        # And the canvas shows it, rather than the mask and the drawing disagreeing.
        drawn = {(t.row, t.col): t.enabled for t in widget.tile_grid_overlay._tiles}
        assert drawn[(0, 1)] is False

    def test_dragging_the_grid_does_not_move_the_stage(self, widget, microscope):
        """Setting a run up and driving the instrument are separate acts. A drag is
        exploratory -- you push the grid around to see what it would cover."""
        before = (widget._stage_position.x, widget._stage_position.y)
        moved = []
        widget.position_move_requested.connect(lambda *a: moved.append(a))

        widget._on_grid_moved(120.0, -40.0)

        assert widget.target is not None, "the drag did not set a target"
        assert (widget._stage_position.x, widget._stage_position.y) == before
        assert not moved

    def test_a_dragged_grid_keeps_the_stage_pose(self, widget, microscope):
        """The run must not have to re-pose the stage to reach its own grid — the same
        rule the click restriction enforces.

        Against a stage that has *drifted* within its orientation, not one sitting
        exactly at the view's anchor: `frame.to_stage` returns the origin's pose, so
        with the two identical this passes whether or not anything keeps the stage's.
        """
        origin = widget._origins[widget.current_view]
        drifted = FibsemStagePosition(
            x=origin.x, y=origin.y, z=origin.z,
            r=origin.r, t=origin.t + np.deg2rad(0.4),
        )
        widget._stage_position = drifted

        widget._on_grid_moved(120.0, -40.0)
        target = widget.target
        assert target.t == pytest.approx(drifted.t), "the grid re-posed the stage"
        assert target.t != pytest.approx(origin.t), "this test cannot tell them apart"

    def test_the_grid_follows_the_target_not_the_stage(self, widget):
        anchored_at = widget.tile_grid_overlay._anchor()
        widget._on_grid_moved(anchored_at[0] + 200.0, anchored_at[1] + 90.0)
        moved_to = widget.tile_grid_overlay._anchor()
        assert moved_to != anchored_at, "the grid did not follow the drag"

        widget.clear_target()
        assert widget.target is None
        assert widget.tile_grid_overlay._anchor() == pytest.approx(anchored_at)

    def test_the_run_is_planned_around_the_target(self, widget, monkeypatch, tmp_path):
        """The point of all of it: a grid dragged somewhere else has to acquire there.
        Without `centre_position` the runner re-reads the stage and the drag is
        decoration."""
        captured = {}

        def fake_worker(fn, *args):
            captured["args"] = args

            class _W:
                def start(self_inner):
                    pass

                def is_alive(self_inner):
                    return False

            return _W()

        widget.set_save_directory(str(tmp_path))
        widget._on_grid_moved(150.0, 60.0)
        monkeypatch.setattr(
            "fibsem.ui.widgets.overview_widget.FunctionWorker", fake_worker
        )
        widget.acquire()

        centre = captured["args"][1]
        assert centre is not None, "the run was not told where the grid is"
        assert centre.x == pytest.approx(widget.target.x)
        assert centre.y == pytest.approx(widget.target.y)

    def test_an_undragged_run_is_told_where_it_happens(
        self, widget, monkeypatch, tmp_path
    ):
        """Reversed by an instrument, and the old reasoning is worth keeping because it
        sounds right: *"None means 'wherever the stage is', which the runner resolves
        itself. Sending a position instead would freeze the grid at wherever the widget
        last saw the stage, which is not the same thing."*

        It is not the same thing, and that is the defect rather than the argument for it.
        "Wherever the stage is" gets resolved when the **worker starts** -- after the
        settings are read, after the dialog is answered -- while the plan and the dialog
        come from the pose the widget last saw. Anything that re-poses in between makes
        the run happen somewhere the user never authorised: reported from an instrument
        as a dialog reading SEM @ MILLING and an overview coming back SEM @ SEM.

        Freezing at the pose the widget last saw is exactly right, because that is the
        pose the dialog described. If it is stale the run is wrong, but it is wrong in
        the way the user was shown -- see FIB-669 for the staleness itself.
        """
        captured = {}

        def fake_worker(fn, *args):
            captured["args"] = args

            class _W:
                def start(self_inner):
                    pass

                def is_alive(self_inner):
                    return False

            return _W()

        widget.set_save_directory(str(tmp_path))
        monkeypatch.setattr(
            "fibsem.ui.widgets.overview_widget.FunctionWorker", fake_worker
        )
        widget.acquire()
        centre = captured["args"][1]
        assert centre is not None, "the run was left to re-read the stage for itself"
        assert centre.t == pytest.approx(widget._stage_position.t)
        assert centre.r == pytest.approx(widget._stage_position.r)

    def test_the_plan_cannot_be_edited_while_a_run_is_going(self, widget):
        """A run in progress is reading this plan."""
        widget._running = True
        widget._on_grid_resized(5, 5)
        widget._on_tile_toggled(0, 0, False)
        widget._on_grid_moved(200.0, 200.0)

        settings = widget._settings()
        assert (settings.nrows, settings.ncols) != (5, 5)
        assert settings.tile_mask is None
        assert widget.target is None


class TestThePlanHoldsStillWhileTheRunWalksTheGrid:
    """A run visits every tile, and the plan it is running must not follow it there.

    `_on_stage_moved` has always guarded against this, and the guard has always been
    bypassed: it skips the *overlay* refresh but still updates the cached position, and
    the planned tileset is anchored at `_target or _stage_position`. Every preview frame
    placed an image, and placing one refreshed every context overlay on the way out --
    so the plan was re-anchored once per tile and walked the grid alongside the
    acquisition (FIB-647).

    Fixed at the level the issue asked for rather than by special-casing `_running`:
    an image is *drawn in* the frame, it does not decide it, so a placement refreshes
    the overlays only when it moved the frame -- the view's origin or the canvas scale.
    """

    def _run_a_tile(self, widget, microscope, position):
        """One tile of a run: the stage arrives, then a preview lands."""
        widget._on_stage_moved(position)
        widget._apply_progress({
            "msg": "Tile Collected", "counter": 1, "total": 3,
            "preview": _tile(microscope, position),
        })

    def test_the_planned_grid_does_not_follow_the_stage(self, widget, microscope):
        from fibsem.ui.widgets.overview_widget import OverviewRecord

        base = microscope.get_stage_position()
        widget._records["run"] = OverviewRecord("run", "run", [])
        widget._active_record = "run"
        widget._set_running(True)
        anchored_at = widget.tile_grid_overlay._anchor()
        assert anchored_at is not None, "the plan is not drawn, so this proves nothing"

        for step in (1, 2, 3):
            self._run_a_tile(widget, microscope, _at(base, dx=step * 250e-6))

        assert widget.tile_grid_overlay._anchor() == pytest.approx(anchored_at), (
            "the planned tileset moved with the stage during the run"
        )

    def test_the_plan_follows_the_stage_again_once_the_run_is_over(
        self, widget, microscope
    ):
        """The counterpart. Held still *for the run*, not disconnected: where the next
        run would land is still wherever the stage ends up."""
        base = microscope.get_stage_position()
        anchored_at = widget.tile_grid_overlay._anchor()

        widget._on_stage_moved(_at(base, dx=400e-6))

        assert widget.tile_grid_overlay._anchor() != pytest.approx(anchored_at)

    def test_the_readout_still_tracks_the_stage_during_a_run(self, widget, microscope):
        """Not everything on the canvas is the plan. Where the stage *is* -- the marker
        and the numbers under it -- has to keep up, and it used to only because placing
        a preview refreshed it as a side effect."""
        base = microscope.get_stage_position()
        widget._set_running(True)
        before = widget.canvas._info_text

        widget._on_stage_moved(_at(base, dx=750e-6, dy=-300e-6))

        assert widget.canvas._info_text != before, (
            "the stage readout froze for the length of the run"
        )

    def test_a_preview_frame_does_not_redraw_the_context_overlays(
        self, widget, microscope
    ):
        """The mechanism, stated on its own: a redraw per tile is a redraw of the stage
        limits, the holder slots, the gridbars and the view selector too, none of which
        an arriving image changes."""
        base = microscope.get_stage_position()
        widget.place_image(_tile(microscope, _at(base)))  # anchors the view
        calls = []
        widget._refresh_context_overlays = lambda: calls.append(1)

        widget.place_image(_tile(microscope, _at(base, dx=100e-6)), key="preview")
        widget.place_image(_tile(microscope, _at(base, dx=200e-6)), key="preview")

        assert calls == [], f"{len(calls)} overlay refresh(es) for images already framed"

    def test_the_plan_is_pinned_to_what_the_run_is_acquiring(
        self, widget, microscope, monkeypatch, tmp_path
    ):
        """Not redrawn is not enough; redrawn *right* is the requirement.

        The first preview re-anchors the view -- it replaces the provisional origin
        `_seed_frame` invented with where the image really came from -- and that is a
        genuine reason to redraw every overlay. Drawn from the cached stage position,
        the plan then landed on whichever tile the stage had reached by then: measured
        against a real 3x3 run of 80 um tiles, one tile off, and it stayed there for the
        rest of the acquisition. So a run pins the centre it was started with.
        """
        def fake_worker(fn, *args):
            class _W:
                def start(self_inner):
                    pass

                def is_alive(self_inner):
                    return False

            return _W()

        base = microscope.get_stage_position()
        widget.set_save_directory(str(tmp_path))
        monkeypatch.setattr(
            "fibsem.ui.widgets.overview_widget.FunctionWorker", fake_worker
        )
        anchored_at = widget.tile_grid_overlay._anchor()

        widget.acquire()
        # The stage sets off, and the first preview arrives from a tile away -- which is
        # the redraw, because it is what un-provisions the origin.
        widget._on_stage_moved(_at(base, dx=300e-6, dy=300e-6))
        widget._apply_progress({
            "msg": "Tile Collected", "counter": 1, "total": 9,
            "preview": _tile(microscope, _at(base)),
        })

        assert widget.tile_grid_overlay._anchor() == pytest.approx(anchored_at), (
            "the plan was redrawn around the tile being acquired, not around the run"
        )

    def test_a_run_without_a_dragged_grid_is_still_given_its_centre(
        self, widget, microscope, monkeypatch, tmp_path
    ):
        """`None` used to mean "runner, read the stage yourself" -- a second reading, a
        moment later, from a different source than the plan and the dialog."""
        captured = {}

        def fake_worker(fn, *args):
            captured["centre"] = args[1]

            class _W:
                def start(self_inner):
                    pass

                def is_alive(self_inner):
                    return False

            return _W()

        widget.set_save_directory(str(tmp_path))
        monkeypatch.setattr(
            "fibsem.ui.widgets.overview_widget.FunctionWorker", fake_worker
        )
        assert widget.target is None, "this test is about the *undragged* case"

        widget.acquire()

        assert captured["centre"] is not None, "the run was left to re-read the stage"
        assert captured["centre"].t == pytest.approx(widget._stage_position.t)

    def test_a_stage_that_moves_at_the_dialog_does_not_move_the_run(
        self, widget, microscope, monkeypatch, tmp_path
    ):
        """The failure this closes, reported from an instrument: the dialog said
        SEM @ MILLING and the overview came back SEM @ SEM.

        The pose is resolved *before* the dialog, so anything that re-poses between
        authorising the run and the worker starting cannot change where it happens. The
        confirmation stands in for that anything -- it is simply a moment when the
        widget is not in control.
        """
        from fibsem.ui.widgets import overview_confirmation_dialog

        captured = {}
        confirmed_at = deepcopy(widget._stage_position)

        def fake_worker(fn, *args):
            captured["centre"] = args[1]

            class _W:
                def start(self_inner):
                    pass

                def is_alive(self_inner):
                    return False

            return _W()

        def moves_the_stage(dialog):
            widget._stage_position = _at(confirmed_at, dx=900e-6, dy=-400e-6)
            return QDialog.Accepted

        widget.set_save_directory(str(tmp_path))
        monkeypatch.setattr(
            "fibsem.ui.widgets.overview_widget.FunctionWorker", fake_worker
        )
        monkeypatch.setattr(
            overview_confirmation_dialog.OverviewConfirmationDialog,
            "exec_", moves_the_stage,
        )

        widget.acquire()

        assert captured["centre"].x == pytest.approx(confirmed_at.x), (
            "the run followed the stage instead of the pose it was authorised for"
        )

    def test_the_pin_is_let_go_of_when_the_run_ends(self, widget, microscope, tmp_path):
        """Or the plan describes the run that has finished rather than the next one."""
        widget._run_centre = _at(microscope.get_stage_position(), dx=500e-6)
        widget._on_finished({"cancelled": True})
        assert widget._run_centre is None

    def test_the_first_image_in_a_view_still_redraws_them(self, widget, microscope):
        """Because that one *does* move the frame: it replaces the provisional anchor
        `_seed_frame` invented with the position the image was really acquired at, which
        moves every overlay drawn in the view."""
        base = microscope.get_stage_position()
        calls = []
        widget._refresh_context_overlays = lambda: calls.append(1)

        widget.place_image(_tile(microscope, _at(base, dx=50e-6)))

        assert calls, "the re-anchored view was left with overlays in the old frame"


class TestARunDoesNotReFrameTheCanvas:
    """The framing you pressed Acquire with is the framing you keep (FIB-648).

    `auto_fit` refits whenever the placed extent changes, and a run changes it at the
    two moments you are most likely to be looking: the first overview appears, and the
    end swaps the preview for the stitch -- a removal and a placement, so two refits
    back to back. In between it mostly does not fire, because the preview keeps one key
    and one extent, which is what made it read as the canvas being stable and then
    lurching.
    """

    def _run(self, widget, microscope, tiles=3):
        from fibsem.ui.widgets.overview_widget import OverviewRecord

        base = microscope.get_stage_position()
        widget._records["run"] = OverviewRecord("run", "run", [])
        widget._active_record = "run"
        widget._set_running(True)
        for counter in range(1, tiles + 1):
            widget._apply_progress({
                "msg": "Tile Collected", "counter": counter, "total": tiles,
                "preview": _tile(microscope, _at(base)),
            })
        widget._mosaic = _tile(microscope, _at(base), shape=(128, 128))
        widget._on_finished({})

    def test_the_framing_survives_the_whole_run(self, widget, microscope):
        ax = widget.canvas._ax
        framing = (tuple(ax.get_xlim()), tuple(ax.get_ylim()))

        self._run(widget, microscope)

        assert (tuple(ax.get_xlim()), tuple(ax.get_ylim())) == framing

    def test_the_run_actually_put_something_there(self, widget, microscope):
        """Or the test above passes on a canvas where nothing happened."""
        self._run(widget, microscope)
        assert widget.canvas.placed_keys, "the run placed nothing to re-frame for"

    def test_a_run_hands_the_camera_over_for_good(self, widget, microscope):
        """Not restored at the end. There is content on the canvas now, and the framing
        belongs to whoever last set it -- "reset view" is how you ask for it back."""
        self._run(widget, microscope)
        assert widget.canvas.auto_fit is False

        widget.canvas.reset_view()
        assert widget.canvas.auto_fit is True

    def test_the_canvas_still_frames_itself_before_a_run(self, widget, microscope):
        """The one time auto-fit is wanted: opening the tab, where the framing is this
        canvas's guess rather than anyone's choice. Handing the camera over on the first
        run must not be brought forward to construction."""
        assert widget.canvas.auto_fit is True
        ax = widget.canvas._ax
        framing = (tuple(ax.get_xlim()), tuple(ax.get_ylim()))

        widget.settings_widget.set_grid_size(7, 7)

        assert (tuple(ax.get_xlim()), tuple(ax.get_ylim())) != framing, (
            "a bigger plan did not re-frame a canvas nobody has framed by hand"
        )


class TestARunNeedsTiles:
    """Masking every tile off is a state the runner cannot do anything with.

    It only became reachable once tiles could be clicked off the canvas, and left
    alone it crashed *after* reporting success -- the run walked zero tiles, emitted a
    finished payload, then died at stitch time.
    """

    def test_the_button_goes_away_with_the_last_tile(self, widget):
        assert widget.button_acquire.isEnabled()
        widget.settings_widget.tile_mask = [[False] * 3 for _ in range(3)]
        assert not widget.button_acquire.isEnabled()
        assert "No tiles" in widget.button_acquire.toolTip()

    def test_it_comes_back_with_a_tile(self, widget):
        widget.settings_widget.tile_mask = [[False] * 3 for _ in range(3)]
        widget._on_tile_toggled(1, 1, True)
        assert widget.button_acquire.isEnabled()

    def test_acquiring_with_nothing_selected_starts_no_run(
        self, widget, monkeypatch, tmp_path
    ):
        """The button is the affordance, not the guard: a host calling this directly,
        or a mask emptied between the click and here, is what this is for."""
        started = []
        monkeypatch.setattr(
            "fibsem.ui.widgets.overview_widget.FunctionWorker",
            lambda *a, **k: started.append(a) or (_ for _ in ()).throw(AssertionError),
        )
        widget.set_save_directory(str(tmp_path))
        widget.settings_widget.tile_mask = [[False] * 3 for _ in range(3)]

        widget.acquire()
        assert not started
        assert not widget.is_acquiring


class TestTheCanvasFollowsTheBeamYouPlanWith:
    """Choosing a beam says what the *next run* will be, so the canvas has to show
    where that run will land.

    Reported from the tab: switching to the ion beam made the planned grid vanish.
    It was refusing to draw on a view the run would not land in -- which was correct,
    but the answer is to move the canvas rather than to hide the plan. An empty canvas
    already followed, because nothing had pinned the view; one with an overview on it
    did not, which is the normal case.
    """

    @staticmethod
    def _image(scope, beam_type):
        position = scope.get_stage_position()
        hfw = 128 * 2e-7
        image = FibsemImage.generate_blank_image(resolution=(128, 128), hfw=hfw)
        image.data = (np.random.default_rng(0).random((128, 128)) * 255).astype(np.uint8)
        state = scope.get_microscope_state(beam_type=beam_type)
        state.stage_position = position
        image.metadata.image_settings = ImageSettings(hfw=hfw, beam_type=beam_type)
        image.metadata.microscope_state = state
        image.metadata.system_info = scope.system.info
        image.metadata.hardware_geometry = scope.hardware_geometry()
        return image

    def test_switching_beam_keeps_the_plan_on_screen(self, widget, microscope):
        widget.set_image(self._image(microscope, BeamType.ELECTRON))
        assert widget.tile_grid_overlay._tiles

        widget.settings_widget.combo_beam.set_value(BeamType.ION)

        assert widget.current_view.beam_type is BeamType.ION
        assert widget.tile_grid_overlay._tiles, "the planned grid disappeared"

    def test_switching_back_brings_the_overview_with_it(self, widget, microscope):
        """The images are kept per view, so this is a switch and not a loss."""
        widget.set_image(self._image(microscope, BeamType.ELECTRON))
        assert len(widget.canvas.placed_keys) == 1

        widget.settings_widget.combo_beam.set_value(BeamType.ION)
        assert len(widget.canvas.placed_keys) == 0, "the ion view is empty, as it should be"

        widget.settings_widget.combo_beam.set_value(BeamType.ELECTRON)
        assert len(widget.canvas.placed_keys) == 1, "the electron overview did not come back"

    def test_re_posing_the_stage_moves_the_canvas_with_it(self, widget, microscope):
        """Reported from the tab: moving to the milling orientation left the display and
        the planned grid describing the pose the stage had *left*.

        This was once deliberate -- a stage move was held not to be a statement about
        what to look at. The distinction does not survive contact: a move *within* an
        orientation does not change the view at all, so the only move that reaches here
        is a change of orientation, which is as deliberate as choosing a beam.
        """
        widget.set_image(self._image(microscope, BeamType.ELECTRON))
        assert widget.current_view.orientation == "SEM"

        pose = microscope.get_orientation("MILLING")
        widget._on_stage_moved(
            FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=pose.r, t=pose.t)
        )
        assert widget.current_view.orientation == "MILLING"
        assert widget.tile_grid_overlay._tiles, "the planned grid did not come with it"

    def test_a_move_within_an_orientation_leaves_the_view_alone(self, widget, microscope):
        """The other half, and why following an orientation change is safe: steering
        around inside a view -- which is what clicking the canvas does -- must not
        reshuffle what is on screen."""
        widget.set_image(self._image(microscope, BeamType.ELECTRON))
        showing = widget.current_view
        placed = len(widget.canvas.placed_keys)

        here = microscope.get_stage_position()
        widget._on_stage_moved(
            FibsemStagePosition(x=here.x + 250e-6, y=here.y, z=here.z, r=here.r, t=here.t)
        )
        assert widget.current_view == showing
        assert len(widget.canvas.placed_keys) == placed

    def test_a_chosen_view_stands_until_the_run_would_land_elsewhere(
        self, widget, microscope
    ):
        """Following on *change* rather than on every refresh is what leaves room for
        the selector: picking a view has to survive the next redraw."""
        widget.set_image(self._image(microscope, BeamType.ELECTRON))
        sem = widget.current_view
        pose = microscope.get_orientation("MILLING")
        widget._on_stage_moved(
            FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=pose.r, t=pose.t)
        )
        assert widget.current_view != sem

        widget.show_view(sem)
        widget._refresh_context_overlays()
        widget._refresh_context_overlays()
        assert widget.current_view == sem, "the redraw undid the choice"

    def test_the_view_the_run_would_land_in_is_always_selectable(self, widget, microscope):
        """And when a stage move does take the plan elsewhere, there has to be a way to
        go and look at it -- the selector only listed views something had been placed
        in, so an empty one was unreachable."""
        widget.set_image(self._image(microscope, BeamType.ELECTRON))
        pose = microscope.get_orientation("MILLING")
        widget._on_stage_moved(
            FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=pose.r, t=pose.t)
        )

        assert widget.acquisition_view in widget._view_chip_buttons, (
            "the view the next run lands in cannot be selected"
        )


class TestTheViewChips:
    """The view selector rides with the canvas, and says which view is which.

    It selects what you are looking at, so it belongs where you are looking rather than
    in the settings column. And the labels had to change: the orientations are named
    after the beams, so pairing the two read as a tautology one way ("SEM · Electron")
    and a contradiction the other ("SEM · Ion").

    On a strip *above* the canvas rather than painted on it (FIB-649). The set is beams
    x orientations and grows through a session, so it never fitted a corner: eight chips
    measure 709 px, which is wider than the canvas at any window worth having, and they
    ran off the right edge and under the toolbar buttons. A growing set inside a fixed
    region has no offset that rescues it -- which is also how they came to be drawn
    inside the canvas status zone's rectangle (FIB-651).
    """

    @staticmethod
    def _image(scope, orientation, beam_type):
        pose = scope.get_orientation(orientation)
        position = FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=pose.r, t=pose.t)
        hfw = 128 * 2e-7
        image = FibsemImage.generate_blank_image(resolution=(128, 128), hfw=hfw)
        image.data = (np.random.default_rng(0).random((128, 128)) * 255).astype(np.uint8)
        state = scope.get_microscope_state(beam_type=beam_type)
        state.stage_position = position
        image.metadata.image_settings = ImageSettings(hfw=hfw, beam_type=beam_type)
        image.metadata.microscope_state = state
        image.metadata.system_info = scope.system.info
        image.metadata.hardware_geometry = scope.hardware_geometry()
        return image

    @pytest.mark.parametrize(
        "orientation, beam, expected",
        [
            ("SEM", BeamType.ELECTRON, "SEM @ SEM"),
            ("FIB", BeamType.ION, "FIB @ FIB"),
            ("MILLING", BeamType.ION, "FIB @ MILLING"),
            ("MILLING", BeamType.ELECTRON, "SEM @ MILLING"),
            ("SEM", BeamType.ION, "FIB @ SEM"),
        ],
    )
    def test_a_view_names_both_facts_in_the_same_shape(self, orientation, beam, expected):
        """Beam then orientation, always both. Dropping the orientation when it matched
        the beam read well until you met the ones it kept, and then "FIB · SEM pose" had
        to be decoded rather than read."""
        assert OverviewView(beam_type=beam, orientation=orientation).label == expected

    def test_a_view_can_spell_itself_out(self):
        """The chip is glanceable; the tooltip is where the words go."""
        view = OverviewView(beam_type=BeamType.ION, orientation="MILLING")
        assert view.describe == "Ion beam, stage at the MILLING orientation."

    def test_orientations_are_shown_as_the_microscope_names_them(self):
        """Title-casing rendered the two acronyms as "Sem" and "Fib"; casing by rule
        instead means a rule to remember, and a new orientation to add to it."""
        assert OverviewView(beam_type=BeamType.ION, orientation="SEM").label == "FIB @ SEM"
        assert (
            OverviewView(beam_type=BeamType.ION, orientation="MILLING").label
            == "FIB @ MILLING"
        )

    def test_the_view_is_named_even_when_there_is_only_one(self, widget):
        """A lone chip says nothing the info bar does not, and it is still worth drawing:
        a control nobody can see does not exist, and the first time there are two views
        is the worst moment to discover that switching was possible all along."""
        chips = widget._view_chip_buttons
        assert len(chips) == 1
        assert next(iter(chips)) == widget.current_view
        assert next(iter(chips.values())).isChecked()

    def test_a_chip_appears_for_every_view_worth_switching_to(self, widget, microscope):
        widget.set_image(self._image(microscope, "SEM", BeamType.ELECTRON))
        widget.settings_widget.combo_beam.set_value(BeamType.ION)

        labels = {b.text() for b in widget._view_chip_buttons.values()}
        assert {"SEM @ SEM", "FIB @ SEM"} <= labels
        # Laid out, not merely constructed: they have twice been sized from a layout
        # that answered with nothing -- once hand-placed in a container that reported a
        # zero size hint, and once inserted into a live layout that had not activated
        # yet -- and both times they reported themselves visible and drew at zero size.
        assert all(b.width() > 0 for b in widget._view_chip_buttons.values())
        assert all(b.height() > 0 for b in widget._view_chip_buttons.values())
        assert widget.view_strip.height() > 0, "the strip collapsed onto its margins"

    def test_clicking_a_chip_switches_the_canvas(self, widget, microscope):
        widget.set_image(self._image(microscope, "SEM", BeamType.ELECTRON))
        sem = widget.current_view
        widget.settings_widget.combo_beam.set_value(BeamType.ION)
        assert widget.current_view != sem
        assert not widget.canvas.placed_keys

        widget._view_chip_buttons[sem].click()
        assert widget.current_view == sem
        assert len(widget.canvas.placed_keys) == 1, "the overview did not come back"

    def test_the_chip_for_the_displayed_view_is_the_checked_one(self, widget, microscope):
        widget.set_image(self._image(microscope, "SEM", BeamType.ELECTRON))
        widget.settings_widget.combo_beam.set_value(BeamType.ION)

        checked = {b.text() for b in widget._view_chip_buttons.values() if b.isChecked()}
        assert checked == {widget.current_view.label}

    def test_the_view_the_run_would_land_in_is_marked_apart(self, widget, microscope):
        """Which view is displayed and which the next run lands in come apart on this
        tab, and the chips have to say which is which -- otherwise the only way to find
        out is to acquire and see where it went."""
        widget.set_image(self._image(microscope, "SEM", BeamType.ELECTRON))
        widget.settings_widget.combo_beam.set_value(BeamType.ION)
        acquisition = widget.acquisition_view
        widget._view_chip_buttons[widget.views[0]].click()  # look at the SEM one

        assert widget.current_view != acquisition
        marked = widget._view_chip_buttons[acquisition]
        other = widget._view_chip_buttons[widget.current_view]
        assert marked.styleSheet() != other.styleSheet()
        assert "next acquisition" in marked.toolTip()

    def test_the_chips_are_above_the_canvas_not_on_it(self, widget, microscope):
        """The whole of FIB-649 in one assertion: nothing the selector does can occlude
        data or collide with the canvas's own chrome if it is not on the canvas."""
        widget.set_image(self._image(microscope, "SEM", BeamType.ELECTRON))
        widget.show()
        _app.processEvents()

        assert widget.view_strip.isVisible()
        for chip in widget._view_chip_buttons.values():
            assert not widget.canvas.isAncestorOf(chip), (
                f"{chip.text()!r} is drawn on the canvas"
            )
        strip = widget.view_strip.geometry()
        assert strip.bottom() <= widget.canvas.geometry().top(), (
            "the strip overlaps the canvas rather than sitting above it"
        )
        widget.hide()

    def test_eight_views_do_not_set_a_floor_on_the_window(self, widget, monkeypatch):
        """The capacity problem, stated as the thing that would go wrong.

        A row of controls in a plain layout sets the *window's* minimum width. Eight
        chips measure 709 px, so a tab used in every view would refuse to be resized
        narrower than that -- a floor that appears mid-session as views accumulate, on
        a tab whose own minimum is the settings column.
        """
        views = [
            OverviewView(beam_type=beam, orientation=orientation)
            for orientation in ("SEM", "MILLING", "FM", "LANDING")
            for beam in (BeamType.ELECTRON, BeamType.ION)
        ]
        before = widget.minimumSizeHint().width()
        monkeypatch.setattr(
            type(widget), "views", property(lambda self: views)  # read-only
        )
        widget._refresh_view_selector()
        _app.processEvents()

        assert len(widget._view_chip_buttons) == 8
        assert widget.minimumSizeHint().width() == before, (
            "the chips became the narrowest the window can be"
        )

    def test_the_strip_goes_away_when_there_is_nothing_to_choose(
        self, widget, monkeypatch
    ):
        """An empty bar above the canvas is chrome that has not earned its place."""
        monkeypatch.setattr(type(widget), "views", property(lambda self: []))
        monkeypatch.setattr(
            type(widget), "acquisition_view", property(lambda self: None)
        )
        widget._current_view = None
        widget._refresh_view_selector()

        assert not widget._view_chip_buttons
        assert widget.view_strip.isHidden()


class TestTheCursorReadoutUsesTheCanvasStatusZone:
    """The pointer readout goes through the canvas, not a label of this tab's own.

    It used to be hand-placed under the view chips, in "the one free corner" -- which
    stopped being free when the canvas grew a status zone there (FIB-639). Measured
    before the fix: the status label at (4, 4, 104x22) and a chip at (8, 8, 73x20), the
    chip drawn inside it. Nothing showed it, because nothing on this tab ever set a
    hint, a readout or a flash -- so the collision was waiting for the first one that
    did (FIB-651).
    """

    def test_the_readout_goes_to_the_canvas(self, widget):
        widget._set_cursor_readout("x 1.0  y 2.0  z 3.0 um")
        assert widget.canvas._readout_text == "x 1.0  y 2.0  z 3.0 um"

    def test_an_empty_readout_releases_the_zone(self, widget):
        """Rather than leaving a blank plaque over the data. The zone then falls back to
        whatever is underneath on its own."""
        widget._set_cursor_readout("x 1.0  y 2.0  z 3.0 um")
        widget._set_cursor_readout("")
        assert widget.canvas._readout_text is None

    def test_the_tab_keeps_no_label_of_its_own(self, widget):
        """The mechanism, not just the outcome: a second label parented to the canvas is
        what put two things in this corner, and it is what would do so again."""
        assert not hasattr(widget, "cursor_readout")

    def test_the_readout_does_not_repaint_the_figure(self, widget, monkeypatch):
        """It fires on every motion event, and a figure repaint on this canvas costs
        every placed image -- measured at ~1.7 ms each and unbounded, 61.7 ms at 36
        (FIB-650). Still worth pinning now that the chrome around it is all widgets:
        what this catches is the readout being moved back onto an axes artist, which is
        what both of the labels it replaced started life as."""
        draws = []
        monkeypatch.setattr(widget.canvas, "draw_idle", lambda: draws.append(1))

        widget._set_cursor_readout("x 1.0  y 2.0  z 3.0 um")

        assert draws == [], f"{len(draws)} figure repaint(s) for one readout update"


class TestTheMillingAngleIsOnTheBeamTab:
    """What the stage tilt *means* on the beam side: the angle the ion beam makes with
    the sample surface, and the number a milling pose is chosen for.

    The fluorescence tab leaves it out deliberately -- meaningless through a camera --
    and its own comment says it belongs here if anywhere.
    """

    def test_the_info_bar_carries_it(self, widget, microscope):
        expected = microscope.get_current_milling_angle(
            stage_position=widget._stage_position
        )
        assert f"milling {expected:.1f}°" in widget.canvas._info_text

    def test_it_follows_the_stage(self, widget, microscope):
        before = widget.canvas._info_text
        pose = microscope.get_orientation("MILLING")
        widget._on_stage_moved(
            FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=pose.r, t=pose.t)
        )
        assert widget.canvas._info_text != before
        assert "milling" in widget.canvas._info_text

    def test_a_pose_with_no_tilt_drops_the_angle_not_the_line(self, widget, monkeypatch):
        """It can refuse, and the position is the half of the line that always works."""
        monkeypatch.setattr(
            widget.microscope, "get_current_milling_angle",
            lambda **kwargs: (_ for _ in ()).throw(ValueError("no tilt")),
        )
        widget._refresh_stage_info()
        assert widget.canvas._info_text, "the whole info bar went with the angle"
        assert "milling" not in widget.canvas._info_text

    def test_it_costs_no_hardware_read(self, widget, microscope, monkeypatch):
        """This runs on every overlay refresh. `get_current_milling_angle` is arithmetic
        over the pose it is handed -- but only if it is handed one."""
        monkeypatch.setattr(
            microscope, "get_stage_position",
            lambda *a, **k: pytest.fail("the info bar polled the stage"),
        )
        widget._refresh_stage_info()


class TestARunIsConfirmedFirst:
    """Pressing Acquire drives the stage. Two things it will do are set on the canvas
    rather than in the controls, so neither is visible from the settings column at the
    moment it is pressed: where the grid was dragged to, and which tiles are masked off.
    Both survive a tab switch, and both are silent.

    The dialog is where they announce themselves. These check that it is asked, that a
    refusal is honoured, and that what it is handed is what the run gets -- a dialog
    describing a different set of settings than the one that runs is worse than none.
    """

    def _fake_worker(self, captured):
        def factory(fn, *args):
            captured["args"] = args

            class _W:
                def start(self_inner):
                    pass

                def is_alive(self_inner):
                    return False

            return _W()

        return factory

    def test_a_run_asks_before_it_starts(
        self, widget, monkeypatch, tmp_path, confirmations
    ):
        captured = {}
        widget.set_save_directory(str(tmp_path))
        monkeypatch.setattr(
            "fibsem.ui.widgets.overview_widget.FunctionWorker",
            self._fake_worker(captured),
        )
        widget.acquire()
        assert len(confirmations) == 1, "the run started without asking"
        assert "args" in captured, "the run was refused after the dialog was accepted"

    def test_declining_leaves_everything_as_it_was(
        self, widget, monkeypatch, tmp_path
    ):
        """Not merely "no worker": a refused run must leave no record behind either, or
        the overview list grows a row for something that never happened."""
        captured = {}
        widget.set_save_directory(str(tmp_path))
        monkeypatch.setattr(
            "fibsem.ui.widgets.overview_widget.FunctionWorker",
            self._fake_worker(captured),
        )
        monkeypatch.setattr(
            overview_confirmation_dialog.OverviewConfirmationDialog,
            "exec_",
            lambda self: QDialog.Rejected,
        )
        before = len(widget._records)

        widget.acquire()

        assert "args" not in captured, "a declined run started anyway"
        assert len(widget._records) == before, "a declined run left a record behind"
        assert not widget.is_acquiring

    def test_the_dialog_is_shown_the_settings_the_run_gets(
        self, widget, monkeypatch, tmp_path, confirmations
    ):
        """Including the destination, which `acquire` fills in *after* reading the
        widget -- a dialog built before that step would show a run with nowhere to go."""
        captured = {}
        widget.set_save_directory(str(tmp_path))
        widget.settings_widget.set_grid_size(2, 4)
        monkeypatch.setattr(
            "fibsem.ui.widgets.overview_widget.FunctionWorker",
            self._fake_worker(captured),
        )
        widget.acquire()

        shown = confirmations[0].settings
        assert shown is captured["args"][0], (
            "the dialog described a different settings object than the one that ran"
        )
        assert shown.image_settings.path, "the dialog was built before the path was set"
        assert (shown.nrows, shown.ncols) == (2, 4)

    def test_an_undragged_run_says_it_is_on_the_stage(
        self, widget, monkeypatch, tmp_path, confirmations
    ):
        captured = {}
        widget.set_save_directory(str(tmp_path))
        monkeypatch.setattr(
            "fibsem.ui.widgets.overview_widget.FunctionWorker",
            self._fake_worker(captured),
        )
        widget.acquire()
        assert confirmations[0].offset is None
        assert confirmations[0]._centre_text() == "the stage position"

    def test_the_dialog_reports_a_dragged_grid(
        self, widget, monkeypatch, tmp_path, confirmations
    ):
        """The reason this dialog exists. A grid dragged half a millimetre away looks
        identical in the settings column."""
        captured = {}
        widget.set_save_directory(str(tmp_path))
        monkeypatch.setattr(
            "fibsem.ui.widgets.overview_widget.FunctionWorker",
            self._fake_worker(captured),
        )
        widget._on_grid_moved(150.0, 60.0)
        widget.acquire()

        dragged = confirmations[0]
        assert dragged.offset is not None, "a dragged grid was reported as on the stage"
        expected = (
            widget.target.x - widget._stage_position.x,
            widget.target.y - widget._stage_position.y,
        )
        assert dragged.offset == pytest.approx(expected)
        assert "from the stage position" in dragged._centre_text()
        assert dragged._centre_text() != "the stage position"

    def test_opening_the_dialog_costs_no_hardware_read(
        self, widget, monkeypatch, tmp_path, confirmations
    ):
        """It reports the view and the offset, both of which have a cached answer. A
        dialog that polled the stage would do it on the click that starts a run, which
        is the worst moment to add a set-then-read on the shared channel."""
        captured = {}
        widget.set_save_directory(str(tmp_path))
        monkeypatch.setattr(
            "fibsem.ui.widgets.overview_widget.FunctionWorker",
            self._fake_worker(captured),
        )
        monkeypatch.setattr(
            widget.microscope, "get_stage_position",
            lambda *a, **k: pytest.fail("the confirmation dialog polled the stage"),
        )
        widget.acquire()
        assert confirmations[0].view_description


class TestTheAcquisitionButtons:
    """`Cancel | Acquire Overview`, the fluorescence tab's arrangement."""

    @staticmethod
    def _pretend_a_run_is_going(widget):
        """`_set_running(True)` alone is not enough: `cancel()` is gated on
        `is_acquiring`, which asks the worker, not the flag."""

        class _LiveWorker:
            def is_alive(self):
                return True

        widget._worker = _LiveWorker()
        widget._set_running(True)

    def test_stop_sits_beside_go_and_stays_there(self, widget):
        """Enabled and disabled rather than shown and hidden: a button that appears when
        a run starts moves everything below it at the moment the user is least able to
        absorb a moving layout."""
        assert widget.button_acquire.text() == "Acquire Overview"
        assert not widget.button_cancel.isHidden()
        assert not widget.button_cancel.isEnabled()

        widget._set_running(True)
        assert widget.button_cancel.isEnabled()
        assert not widget.button_acquire.isEnabled()
        assert not widget.button_cancel.isHidden()

    def test_cancel_goes_dead_once_it_has_been_asked(self, widget):
        """A run stops at the next tile boundary, so the button is still there for a
        while after it is pressed. Left live, a second press reads as the first one not
        having worked."""
        self._pretend_a_run_is_going(widget)
        widget.cancel()
        assert not widget.button_cancel.isEnabled()

    def test_a_host_lock_cannot_take_away_the_stop(self, widget):
        """`set_interactive(False)` is a host claiming the instrument. It must not
        remove the only way to stop a run that is already under way."""
        widget._set_running(True)
        widget.set_interactive(False)
        assert not widget.button_acquire.isEnabled()
        assert widget.button_cancel.isEnabled()

    def test_the_actions_are_not_in_the_scrolling_part(self, widget):
        """Structural, because the failure is invisible until the window is short.

        Inside the scroll area, a host adding its own section -- the lamella list, which
        is what the AutoLamella tab does -- pushes Acquire, Cancel and the progress bar
        below the fold. A run then reports its progress somewhere nobody is looking, and
        stopping it means scrolling first.
        """
        from PyQt5.QtWidgets import QScrollArea

        scroll = widget.findChild(QScrollArea)
        assert scroll is not None, "the settings column is no longer scrolled"
        scrolled = scroll.widget()
        for name in ("button_acquire", "button_cancel", "progress", "label_status"):
            child = getattr(widget, name)
            assert not scrolled.isAncestorOf(child), f"{name} scrolls away with the column"

    def test_they_stay_on_screen_in_a_window_too_short_for_the_column(
        self, microscope, qapp
    ):
        """The case that motivated it: a host section on top, and a window shorter than
        the controls need. Shown for real -- geometry on an unshown widget is whatever
        the last layout pass left, which is how this passes for the wrong reason."""
        from PyQt5.QtWidgets import QListWidget

        widget = FibsemOverviewWidget(microscope)
        try:
            lamellae = QListWidget()
            for i in range(6):
                lamellae.addItem(f"Lamella-{i + 1:02d}")
            widget.add_settings_section("Lamellae", lamellae)
            widget.resize(1250, 620)
            widget.show()
            qapp.processEvents()

            top_left = widget.button_acquire.mapTo(widget, QPoint(0, 0))
            bottom = top_left.y() + widget.button_acquire.height()
            assert bottom <= widget.height(), (
                f"Acquire runs to {bottom}px in a {widget.height()}px widget"
            )
            assert widget.button_acquire.visibleRegion().boundingRect().height() > 0
        finally:
            widget.close()


class TestAnOverviewDoesNotPaintOverTheOneBeneathIt:
    """A mosaic is mostly zeros until it is finished, and those zeros are not black --
    they are nothing. Placed opaquely they hid whatever was underneath, so a second
    overview blanked the first wherever it had not reached yet (FIB-630).

    The fluorescence canvas solved the same problem with `to_rgba`, where alpha is
    signal strength. That is right for signal over black and wrong here: matplotlib
    draws `colour x alpha + (1 - alpha) x beneath`, so on a dense grayscale image the
    second term brightens everything that happens to have something behind it. Measured
    on a mid-grey region over a textured one, it read 0.772 against a true 0.500. So
    alpha is coverage, which is a step function.
    """

    @staticmethod
    def _partial(rows: int = 1, shape: int = 96) -> np.ndarray:
        """A mosaic with `rows` of three acquired and the rest still zero."""
        data = np.zeros((shape, shape), dtype=np.uint8)
        rng = np.random.default_rng(4)
        filled = (rng.random((rows * shape // 3, shape)) * 110 + 90).astype(np.uint8)
        data[: rows * shape // 3] = filled
        return data

    def test_an_unacquired_region_is_transparent_and_the_rest_is_not(self):
        from fibsem.ui.widgets.overview_widget import (
            _as_colour_and_coverage, _contrast_limits,
        )

        data = self._partial()
        rgba = _as_colour_and_coverage(data, data > 0, _contrast_limits(data.astype(float), data > 0))
        alpha = rgba[..., 3]
        assert (alpha[: data.shape[0] // 3] == 255).all(), "acquired tiles went see-through"
        assert (alpha[data.shape[0] // 3:] == 0).all(), (
            "unacquired ground is still opaque, so it paints over what is beneath"
        )

    def test_coverage_is_a_step_not_a_brightness(self):
        """The distinction that makes this correct over another image. A dark *acquired*
        pixel has to stay opaque, or the overview beneath shows through it and the
        result is neither picture."""
        from fibsem.ui.widgets.overview_widget import (
            _as_colour_and_coverage, _contrast_limits,
        )

        data = np.array([[1, 40, 128, 255]], dtype=np.uint8)
        rgba = _as_colour_and_coverage(data, data > 0, _contrast_limits(data.astype(float), data > 0))
        assert (rgba[..., 3] == 255).all(), (
            f"alpha followed intensity: {rgba[..., 3].tolist()}"
        )

    def test_nothing_acquired_is_wholly_transparent(self):
        """A run that has not produced a tile yet, and a cancelled one that never did."""
        from fibsem.ui.widgets.overview_widget import (
            _as_colour_and_coverage, _contrast_limits,
        )

        data = np.zeros((16, 16), dtype=np.uint8)
        assert (_as_colour_and_coverage(data, data > 0, _contrast_limits(data.astype(float), data > 0))[..., 3] == 0).all()

    def test_the_unacquired_zeros_do_not_set_the_contrast(self):
        """They are most of a part-finished mosaic and they are not drawn, so letting
        them win the minimum squeezes what *is* there into the top of the range. That
        renders a half-done overview visibly bleached."""
        from fibsem.ui.widgets.overview_widget import (
            _as_colour_and_coverage, _contrast_limits,
        )

        data = self._partial()
        acquired = data > 0
        grey = _as_colour_and_coverage(data, acquired, _contrast_limits(data.astype(float), acquired))[..., 0][acquired]
        assert grey.mean() == pytest.approx(128, abs=25), (
            f"the acquired tiles render at a mean of {grey.mean():.0f}/255"
        )
        assert grey.min() < 30 and grey.max() > 225, "the stretch did not use the range"

    def test_a_complete_overview_uses_the_whole_range_too(self):
        """The common case must not change: with nothing to exclude, this is the same
        stretch the canvas applied before."""
        from fibsem.ui.widgets.overview_widget import (
            _as_colour_and_coverage, _contrast_limits,
        )

        rng = np.random.default_rng(11)
        data = (rng.random((64, 64)) * 110 + 90).astype(np.uint8)
        rgba = _as_colour_and_coverage(data, np.ones_like(data, dtype=bool),
                                       _contrast_limits(data.astype(float), np.ones_like(data, dtype=bool)))
        assert (rgba[..., 3] == 255).all()
        assert rgba[..., 0].min() == 0 and rgba[..., 0].max() == 255

    def test_coverage_is_measured_before_the_display_filter(self, widget, microscope):
        """`filtered_data` is a median then a gaussian, so it leaks signal a couple of
        pixels *past* the last acquired row. Testing that array for "greater than zero"
        hands back a mask a filter radius too generous, and admits a fringe of near-zero
        values into the contrast stretch -- which is the bleached rendering again.
        """
        image = _tile(microscope, _at(microscope.get_stage_position()), shape=(96, 96))
        data = np.zeros((96, 96), dtype=np.uint8)
        rng = np.random.default_rng(5)
        data[:32] = (rng.random((32, 96)) * 110 + 90).astype(np.uint8)
        image.data = data

        tile = widget._stored_tile(image)
        acquired = np.asarray(tile.acquired)
        # A third of the image, give or take the boundary block -- not a third plus a
        # filter radius, which is what the filtered array would have given.
        assert acquired.mean() == pytest.approx(1 / 3, abs=0.02), (
            f"coverage came out at {acquired.mean():.3f} of the image"
        )
        rgba = widget._for_display(tile)
        grey = rgba[..., 0][rgba[..., 3] > 0]
        assert grey.mean() == pytest.approx(128, abs=30), (
            f"the acquired region renders at {grey.mean():.0f}/255 -- bleached"
        )

    def test_the_canvas_is_given_the_coverage(self, widget, microscope):
        """End to end: what reaches the artist is RGBA, not a 2-D array the canvas would
        colormap opaquely."""
        image = _tile(microscope, _at(microscope.get_stage_position()), shape=(96, 96))
        data = np.zeros((96, 96), dtype=np.uint8)
        data[:32] = 200
        image.data = data
        widget.set_image(image)

        artist = widget.canvas._placed[list(widget.canvas._placed)[-1]].artist
        shown = np.asarray(artist.get_array())
        assert shown.ndim == 3 and shown.shape[-1] == 4, (
            f"the canvas was handed {shown.shape}, so it composites opaquely"
        )
        assert shown[..., 3].min() == 0, "no part of a half-finished mosaic is transparent"


class TestTwoRunsCannotLandOnEachOther:
    """The filename is not a label, it is a location. `TiledAcquisitionRunner._setup`
    makes the tile sub-folder from it and writes the stitch inside that, both keyed on
    the name alone -- so two runs called the same thing overwrite each other's tiles
    *and* mosaics. The canvas, holding both in memory, still shows two; reloading the
    experiment finds one.

    Seen for real: a simulator session ended with three of four rows in the Overviews
    list called `overview-image`, all sharing one directory.
    """

    def _capture(self, widget, monkeypatch, tmp_path):
        captured = {}

        def fake_worker(fn, *args):
            captured["settings"] = args[0]

            class _W:
                def start(self_inner):
                    pass

                def is_alive(self_inner):
                    return False

            return _W()

        widget.set_save_directory(str(tmp_path))
        monkeypatch.setattr(
            "fibsem.ui.widgets.overview_widget.FunctionWorker", fake_worker
        )
        widget.acquire()
        return captured["settings"].image_settings.filename

    def test_a_run_is_stamped_with_the_time_it_started(
        self, widget, monkeypatch, tmp_path
    ):
        import re

        name = self._capture(widget, monkeypatch, tmp_path)
        assert re.fullmatch(r"overview-image-\d{2}-\d{2}-\d{2}", name), name

    def test_two_runs_do_not_share_a_directory(self, widget, monkeypatch, tmp_path):
        """The stamp is only worth having if it actually differs. Frozen rather than
        raced: two real runs are minutes apart, and a test that acquires twice in the
        same second would pass for the wrong reason either way.

        The clock is read where the stamping lives, which is now shared with the
        fluorescence tab rather than private to this one.
        """
        from fibsem.ui.widgets import overview_acquisition_settings_widget as module

        times = iter(["14-23-05", "14-31-40"])
        monkeypatch.setattr(
            module, "current_timestamp_v3", lambda timeonly=True: next(times)
        )
        first = self._capture(widget, monkeypatch, tmp_path)
        widget._set_running(False)
        second = self._capture(widget, monkeypatch, tmp_path)

        assert first != second
        assert first == "overview-image-14-23-05"
        assert second == "overview-image-14-31-40"

    def test_a_name_someone_typed_is_stamped_too(self, widget, monkeypatch, tmp_path):
        """A memorable name invites reuse, so it is the *more* likely to collide. The
        base is kept as a prefix, so what was typed is still what you look for."""
        widget.settings_widget.filename_edit.setText("grid-2-survey")
        name = self._capture(widget, monkeypatch, tmp_path)
        assert name.startswith("grid-2-survey-")
        assert name != "grid-2-survey"

    def test_the_box_still_shows_the_base_name(self, widget, monkeypatch, tmp_path):
        """Stamped at the run, not in the control: a box that rewrote itself on every
        acquisition would make the name unusable as a thing you set once."""
        self._capture(widget, monkeypatch, tmp_path)
        assert (
            widget.settings_widget.filename_edit.text()
            == "overview-image"
        )

    def test_the_dialog_reports_where_it_will_actually_land(
        self, widget, monkeypatch, tmp_path, confirmations
    ):
        """The one place the stamped name is shown before the run. Without this the
        stamp is invisible until the files appear."""
        name = self._capture(widget, monkeypatch, tmp_path)
        saving_to = dict(confirmations[0]._rows())["Saving to"]
        assert saving_to.endswith(name), f"{saving_to} does not name {name}"

    def test_the_record_carries_the_stamped_name(self, widget, monkeypatch, tmp_path):
        """So the Overviews list can tell two runs apart, which is where the collision
        was noticed."""
        name = self._capture(widget, monkeypatch, tmp_path)
        assert [r.label for r in widget._records.values()] == [name]


class TestContrastActsOnEveryOverview:
    """The beam tab had no contrast control at all: what you saw was whatever the auto
    stretch decided. The canvas has one (`btn_contrast`), but it stays hidden on a
    real-space canvas because the machinery behind it adjusts `imgs[0]` -- meaningful
    when a canvas holds one image, arbitrary when it holds an overview per key.

    So the widget owns it, the way the fluorescence tab owns its layers popover, and it
    acts canvas-wide: a mosaic should render uniformly, and per-image contrast would
    emphasise the seams rather than hide them (FIB-415).
    """

    def _place(self, widget, microscope, key=None, rows=None):
        image = _tile(microscope, _at(microscope.get_stage_position()), shape=(96, 96))
        data = np.zeros((96, 96), dtype=np.uint8)
        rng = np.random.default_rng(8)
        filled = rows if rows is not None else 96
        data[:filled] = (rng.random((filled, 96)) * 110 + 90).astype(np.uint8)
        image.data = data
        return widget.place_image(image, key=key)

    @staticmethod
    def _drawn(widget, key):
        return np.asarray(widget.canvas._placed[key].artist.get_array())

    def _narrow(self, widget):
        """A window well inside the data, so the stretch has something to do."""
        control = widget.contrast_control
        control._min, control._max, control._gamma = 0.3, 0.7, 1.0
        control.changed.emit()

    def test_narrowing_the_window_increases_contrast(self, widget, microscope):
        key = self._place(widget, microscope)
        before = self._drawn(widget, key)[..., 0].astype(float).std()
        self._narrow(widget)
        after = self._drawn(widget, key)[..., 0].astype(float).std()
        assert after > before * 1.2, f"contrast did nothing: {before:.1f} -> {after:.1f}"

    def test_contrast_cannot_change_what_was_acquired(self, widget, microscope):
        """The invariant. Alpha is coverage, not brightness: run the same curve over it
        and a region fades out as the maximum comes down, or unacquired ground appears
        as it goes up. Either is the display inventing data."""
        key = self._place(widget, microscope, rows=32)
        before = self._drawn(widget, key)[..., 3].copy()
        self._narrow(widget)
        after = self._drawn(widget, key)[..., 3]
        assert np.array_equal(before, after), "contrast moved the coverage mask"

    def test_adjusting_twice_adjusts_the_original_twice(self, widget, microscope):
        """Not the adjusted one. If the contrasted array were written back over the
        stored tile, every slider move would compound on the last and the picture would
        run away to black or white."""
        key = self._place(widget, microscope)
        self._narrow(widget)
        once = self._drawn(widget, key).copy()
        widget.contrast_control.changed.emit()
        assert np.array_equal(once, self._drawn(widget, key))

    def test_reset_puts_it_back_exactly(self, widget, microscope):
        key = self._place(widget, microscope)
        original = self._drawn(widget, key).copy()
        self._narrow(widget)
        assert not np.array_equal(original, self._drawn(widget, key))
        widget.contrast_control.reset()
        assert np.array_equal(original, self._drawn(widget, key))

    def test_it_reaches_the_acquisition_preview_too(self, widget, microscope):
        """The preview is the one thing on the canvas with no record behind it, so a
        contrast pass that walked the records would leave a run in progress rendering
        differently from everything around it."""
        from fibsem.ui.widgets.overview_widget import PREVIEW_KEY

        self._place(widget, microscope)
        self._place(widget, microscope, key=PREVIEW_KEY)
        before = self._drawn(widget, PREVIEW_KEY)[..., 0].astype(float).std()
        self._narrow(widget)
        after = self._drawn(widget, PREVIEW_KEY)[..., 0].astype(float).std()
        assert after > before * 1.2, "the preview kept the old contrast"

    def test_adjusting_does_not_reorder_the_canvas(self, widget, microscope):
        """`update_image`, not a re-add: re-adding under the same key destroys and
        recreates the artist, which moves it to the top of the draw order -- so turning
        a slider would silently bring one overview out from under another."""
        key = self._place(widget, microscope)
        artist = widget.canvas._placed[key].artist
        self._narrow(widget)
        assert widget.canvas._placed[key].artist is artist

    def test_the_canvas_own_contrast_button_stays_hidden(self, widget):
        """Two contrast buttons, one of which adjusts an arbitrary image, is worse than
        one. The fluorescence tab hides it for the same reason."""
        assert widget.canvas.btn_contrast.isHidden()
        assert not widget.btn_contrast.isHidden()

    def test_the_button_opens_the_popover(self, widget):
        """`isHidden`, not `isVisible`: the fixture widget is never shown, so every
        descendant is invisible whatever it was told to do, and the assertion would
        pass without the popover ever having been opened."""
        assert widget.contrast_control.isHidden()
        widget.btn_contrast.setChecked(True)
        widget._toggle_contrast()
        assert not widget.contrast_control.isHidden()
        widget.btn_contrast.setChecked(False)
        widget._toggle_contrast()
        assert widget.contrast_control.isHidden()


class TestFocusAndFocusStackAreTwoQuestions:
    """*When to focus while walking the grid* and *whether to take a stack at each tile*
    are unrelated: either can be wanted without the other. They were built as one panel,
    which said otherwise.

    The fluorescence tab has had them apart from the start -- a Focus panel and a Z-Stack
    panel -- so this is the two tabs agreeing as much as it is a correction.
    """

    def test_they_are_separate_panels(self, widget):
        settings = widget.settings_widget
        assert settings.focus_panel is not settings.focus_stack_panel
        assert settings.focus_panel._title_label.text() == "Focus"
        assert settings.focus_stack_panel._title_label.text() == "Stack"

    def test_the_mode_is_independent_of_the_stack(self, widget):
        """The pairing that used to be impossible to express in one panel: focus every
        tile, but do not stack."""
        from fibsem.structures import AutoFocusMode

        settings = widget.settings_widget
        settings.combo_autofocus.set_value(AutoFocusMode.EACH_TILE)
        settings.check_focus_stack.setChecked(False)

        read = settings.get_settings()
        assert read.autofocus_settings.mode is AutoFocusMode.EACH_TILE
        assert read.focus_stack_settings.enabled is False

    def test_the_stack_parameters_grey_out_when_it_is_off(self, widget):
        """Live controls under an unticked box imply they apply. The fluorescence tab's
        Z-Stack does the same."""
        settings = widget.settings_widget
        settings.check_focus_stack.setChecked(False)
        assert not settings.spin_focus_steps.isEnabled()
        settings.check_focus_stack.setChecked(True)
        assert settings.spin_focus_steps.isEnabled()


class TestTheSpiralPromotionIsVisible:
    """`TiledAcquisitionRunner._compute_grid` rewrites EACH_ROW to EACH_TILE for a
    spiral, because a spiral has no rows. Correct, and it was silent -- the setting read
    one thing and the run did another. The fluorescence tab warns about exactly this
    combination; this one acted on it and said nothing.
    """

    def _set(self, widget, mode, order):
        from fibsem.structures import TileOrderStrategy

        widget.settings_widget.combo_autofocus.set_value(mode)
        widget.settings_widget.grid.combo_tile_order.set_value(order)
        widget.settings_widget._refresh_derived()

    def test_the_combination_says_what_will_happen(self, widget):
        from fibsem.structures import AutoFocusMode, TileOrderStrategy

        self._set(widget, AutoFocusMode.EACH_ROW, TileOrderStrategy.SPIRAL)
        note = widget.settings_widget.label_focus_note
        assert not note.isHidden()
        assert "every tile" in note.text()

    def test_it_is_quiet_otherwise(self, widget):
        """A note that is always there stops being read. Both halves have to hold: the
        same mode with another order, and the same order with another mode."""
        from fibsem.structures import AutoFocusMode, TileOrderStrategy

        note = widget.settings_widget.label_focus_note
        self._set(widget, AutoFocusMode.EACH_ROW, TileOrderStrategy.TYPEWRITER)
        assert note.isHidden()
        self._set(widget, AutoFocusMode.EACH_TILE, TileOrderStrategy.SPIRAL)
        assert note.isHidden()

    def test_it_describes_what_the_runner_actually_does(self, widget):
        """Pinned against the runner rather than a remembered rule -- if the promotion
        is ever removed, this note becomes a lie and nothing else would notice."""
        import inspect

        from fibsem.imaging import tiled

        source = inspect.getsource(tiled.TiledAcquisitionRunner._compute_grid)
        assert "AutoFocusMode.EACH_ROW" in source and "SPIRAL" in source, (
            "the runner no longer promotes EACH_ROW for a spiral; the note is stale"
        )


class TestAnOverviewIsDrawnFromWhatIsHeld:
    """The overview supplies the canvas a *source* rather than a finished picture.

    A record keeps the grayscale, the coverage mask and the contrast limits; colour and
    the user's curve are applied to whatever part is being drawn, when it is drawn. Two
    things follow that a finished RGBA cannot give: zooming in shows detail the picture
    would already have thrown away, and a contrast step costs a screenful rather than a
    mosaic (FIB-658).
    """

    def _held(self, widget, microscope, side, store_px=None):
        """One overview held at *store_px*, framed whole."""
        if store_px is not None:
            _hold_at(widget, store_px)
        base = microscope.get_stage_position()
        image = _tile(microscope, _at(base), shape=(side, side), hfw=500e-6)
        rng = np.random.default_rng(3)
        ramp = np.linspace(10, 245, side)
        image.data = np.clip(
            np.add.outer(ramp, ramp) / 2 + rng.normal(0, 8, (side, side)), 1, 255
        ).astype(np.uint8)
        record_id = widget.set_image(image)
        _settle(widget)
        return record_id

    def test_the_record_keeps_ingredients_not_a_picture(self, widget, microscope):
        record_id = self._held(widget, microscope, 512)

        tile = widget._records[record_id].images[0]
        assert np.asarray(tile.grey).ndim == 2, "the record holds a finished picture"
        assert np.asarray(tile.acquired).dtype == np.bool_
        assert tile.clim[0] < tile.clim[1]

    def test_zooming_in_recovers_detail_it_could_not_have_held(self, widget, microscope):
        """The payoff, stated exactly: framed whole you see the *draw* cap's worth, and
        zoomed in you see everything *held*. Under a store-time reduction the two are
        necessarily the same number, so a zoom could only magnify.

        Both caps are lowered here so the test image stays small — `filtered_data` runs a
        median then a gaussian over the whole of it. The canvas's is read-only in
        production because the extents of everything placed were computed against it;
        this sets it before anything is placed.
        """
        widget.canvas._display_max_px = 256
        held_px = 1024  # four times the draw cap, so the effect is unambiguous
        self._held(widget, microscope, held_px, store_px=held_px)
        placed = widget.canvas._placed[widget.canvas.placed_keys[0]]

        def per_ground():
            return max(placed.artist.get_array().shape[:2]) / placed.drawn.width

        wide = per_ground()
        _zoom(widget, 0.1)
        close = per_ground()

        assert wide == pytest.approx(widget.canvas.display_max_px, rel=0.05), (
            f"framed whole, {wide:.0f} px per image-width against a "
            f"{widget.canvas.display_max_px} px draw cap"
        )
        assert close == pytest.approx(held_px, rel=0.05), (
            f"zoomed in, {close:.0f} px per image-width against {held_px} px held — the "
            "zoom is magnifying rather than revealing"
        )

    def test_the_stretch_is_the_whole_overview_not_the_part_on_screen(
        self, widget, microscope
    ):
        """`clim` is measured once and kept. Re-measured per patch it would make the
        picture change brightness as you panned across it — every region would render
        mid-grey, and a dark corner would look identical to a bright one."""
        self._held(widget, microscope, 512)
        placed = widget.canvas._placed[widget.canvas.placed_keys[0]]
        _, xmax, ymax, _ = placed.extent

        _zoom(widget, 0.15, centre=(-xmax * 0.7, -ymax * 0.7))  # the dark corner
        dark = float(np.mean(placed.artist.get_array()[..., 0]))
        _zoom(widget, 0.15, centre=(xmax * 0.7, ymax * 0.7))  # the bright one
        bright = float(np.mean(placed.artist.get_array()[..., 0]))

        assert bright > dark * 2, (
            f"dark corner {dark:.0f}/255, bright corner {bright:.0f}/255 — the stretch "
            "is following the view rather than the overview"
        )

    def test_contrast_never_writes_back_over_what_is_held(self, widget, microscope):
        """The stored grayscale is the one copy of the data. Curving it in place would
        compound every slider move onto the last, and there would be no way back to the
        acquired values."""
        record_id = self._held(widget, microscope, 512)
        before = np.asarray(widget._records[record_id].images[0].grey).copy()

        control = widget.contrast_control
        control._min, control._max, control._gamma = 0.2, 0.8, 1.4
        control.changed.emit()

        after = np.asarray(widget._records[record_id].images[0].grey)
        assert np.array_equal(before, after), "the curve was baked into the record"

    def test_a_contrast_change_redraws_without_reordering_the_overviews(
        self, widget, microscope
    ):
        """Re-placing under the same key destroys and recreates the artist, which moves
        it to the top of the draw order — so adjusting contrast would silently bring a
        buried overview out over the one above it."""
        first = self._held(widget, microscope, 256)
        second = self._held(widget, microscope, 256)
        order = list(widget.canvas.placed_keys)
        artists = [widget.canvas._placed[k].artist for k in order]

        widget.contrast_control._gamma = 1.5
        widget.contrast_control.changed.emit()

        assert list(widget.canvas.placed_keys) == order
        assert [widget.canvas._placed[k].artist for k in order] == artists, (
            "the artists were recreated, so the draw order is whatever add order was"
        )
        assert first != second


@pytest.fixture
def at_sem(microscope):
    """Start at the SEM pose, and put the stage back afterwards.

    The `microscope` fixture is module-scoped, so a test that re-poses the stage leaves
    it re-posed for every test after it -- and inherits whatever the one before it left.
    Both directions bit: these tests are about what a re-pose does, so they cannot start
    from "wherever we happen to be".
    """
    before = microscope.get_stage_orientation()
    microscope.move_to_orientation("SEM")
    yield microscope
    microscope.move_to_orientation(before)


class TestAViewWithNothingInItSaysSo:
    """Re-posing the stage moves the displayed view to wherever the next run would land.

    That is the design (FIB-620), and it is what a planner wants -- but if nothing was
    acquired *there*, the canvas empties while every record is still held. A canvas that
    goes blank moments after a stage move reads as a fault, and was reported as one: the
    beam had to be "re-selected" to get the picture back. The beam was never involved.
    Changing it changes the view, and changing the view back is what restored the image
    (FIB-659).
    """

    def _acquired(self, widget, microscope):
        base = microscope.get_stage_position()
        return widget.set_image(_tile(microscope, _at(base), shape=(64, 64)))

    def test_moving_off_the_only_overview_says_where_it_went(self, widget, at_sem):
        self._acquired(widget, at_sem)
        assert widget.canvas._hint_text is None, "it spoke while the overview was shown"

        at_sem.move_to_orientation("MILLING")

        assert not widget.canvas.placed_keys, "the canvas did not actually empty"
        assert widget._records, "the record was dropped rather than left behind"
        hint = widget.canvas._hint_text
        assert hint and "SEM @ SEM" in hint, f"hint was {hint!r}"

    def test_an_empty_canvas_with_nothing_acquired_stays_silent(self, widget):
        """The blank backdrop already means "nothing was acquired here", and that is
        still true. What it cannot say is "but something was acquired elsewhere", which
        is the only case worth a caption."""
        assert not widget.canvas.placed_keys
        assert widget.canvas._hint_text is None

    def test_coming_back_to_the_view_clears_it(self, widget, at_sem):
        self._acquired(widget, at_sem)
        at_sem.move_to_orientation("MILLING")
        assert widget.canvas._hint_text is not None

        widget.show_view(widget.views[0])

        assert widget.canvas.placed_keys
        assert widget.canvas._hint_text is None

    def test_it_names_the_count_rather_than_one_view_when_there_are_several(
        self, widget, at_sem
    ):
        """Naming every view would run past the room the status zone has, and the chips
        directly above already list them."""
        self._acquired(widget, at_sem)
        at_sem.move_to_orientation("MILLING")
        self._acquired(widget, at_sem)
        at_sem.move_to_orientation("FIB")

        hint = widget.canvas._hint_text
        assert hint and "2 other views" in hint, f"hint was {hint!r}"


class TestTheOverlaysCanBeTurnedOff:
    """Everything drawn *over* the data is optional, from one surface.

    A canvas that always draws the travel envelope, the holder's grids, the slot marks
    and a gridbar lattice is one you read around rather than read. The controls hold the
    answer -- `_refresh_context_overlays` asks them rather than being told what changed,
    so a toggle and a stage move take the same path and cannot disagree (FIB-572).
    """

    @staticmethod
    def _context_shapes(widget) -> int:
        """How many shapes the context overlay was last handed."""
        seen = {}
        real = widget.context_overlay.set_shapes

        def spy(specs, *args, **kwargs):
            seen["n"] = len(specs)
            return real(specs, *args, **kwargs)

        widget.context_overlay.set_shapes = spy
        widget._refresh_context_overlays()
        widget.context_overlay.set_shapes = real
        return seen.get("n", 0)

    @pytest.mark.parametrize("key", ["limits", "boundaries", "slots"])
    def test_turning_one_off_stops_it_being_drawn(self, widget, key):
        before = self._context_shapes(widget)

        widget.overlay_controls.set_visible(key, False)
        after = self._context_shapes(widget)

        assert after == before - 1, f"{key} off changed {before} shapes to {after}"

        widget.overlay_controls.set_visible(key, True)
        assert self._context_shapes(widget) == before, "it did not come back"

    def test_hiding_saved_positions_keeps_where_the_stage_is(self, widget, microscope):
        """The saved marks are annotation; where the stage is now is the one mark that
        says which part of the sample you are looking at. Hiding it would make the
        canvas harder to read rather than cleaner."""
        base = microscope.get_stage_position()
        widget.set_positions([
            FibsemStagePosition(name=f"L{i}", x=base.x + i * 20e-6, y=base.y,
                                z=base.z, r=base.r, t=base.t)
            for i in range(3)
        ])
        assert len(widget.position_overlay._points) == 3

        widget.overlay_controls.set_visible("positions", False)

        assert widget.position_overlay._points == []
        assert len(widget.current_position_overlay._points) == 1

    def test_the_grid_bar_pitch_controls_match_the_checkbox_from_the_start(self, widget):
        """They were live from construction over a lattice that was not drawn, because
        the toggle handler had never run -- so adjusting them appeared to do nothing."""
        assert widget.overlay_controls.is_visible("gridbars") is False
        assert not widget.spin_gridbar_spacing.isEnabled()
        assert not widget.spin_gridbar_width.isEnabled()

        widget.overlay_controls.set_visible("gridbars", True)

        assert widget.spin_gridbar_spacing.isEnabled()
        assert widget.spin_gridbar_width.isEnabled()

    def test_an_unknown_overlay_is_shown_rather_than_hidden(self, widget):
        """An overlay whose control was never added is one nobody chose to hide.
        Defaulting to hidden makes a forgotten entry look like a drawing bug."""
        assert widget.overlay_controls.is_visible("something-nobody-added") is True


class TestTheOverlaySwitchesAreOnTheCanvas:
    """On the canvas toolbar beside contrast, not in the settings column.

    What is drawn *over* the picture is a looking-at-it question; the column is for
    setting up the next run. In the column the switches also landed at the bottom of a
    scroll, which is the least reachable place on the tab (FIB-572).
    """

    def test_the_button_opens_and_closes_the_popover(self, widget):
        """`isVisibleTo`, not `isVisible`: the fixture never shows the widget, so
        `isVisible` is False for every child whatever was set on it -- which makes an
        assertion that something is hidden pass without testing anything."""
        assert not widget.overlay_popover.isVisibleTo(widget.canvas)

        widget.btn_overlays.setChecked(True)
        widget._toggle_overlays()
        assert widget.overlay_popover.isVisibleTo(widget.canvas)

        widget.btn_overlays.setChecked(False)
        widget._toggle_overlays()
        assert not widget.overlay_popover.isVisibleTo(widget.canvas)

    def test_it_is_styled_so_it_is_readable_over_the_picture(self, widget):
        """A bare `QWidget` took none of the shared style, whose selectors key off
        `QFrame` -- which drew the switches as unbacked text straight over the data."""
        from PyQt5.QtWidgets import QFrame

        from fibsem.ui.stylesheets import CANVAS_POPOVER_STYLE

        assert isinstance(widget.overlay_popover, QFrame)
        assert widget.overlay_popover.styleSheet() == CANVAS_POPOVER_STYLE

    def test_the_pitch_controls_moved_with_their_switch(self, widget):
        """They mean nothing while the lattice is off, so several panels away from the
        checkbox that draws it is the one place they should not be."""
        popover = widget.overlay_popover
        assert widget.spin_gridbar_spacing.isAncestorOf is not None
        for spin in (widget.spin_gridbar_spacing, widget.spin_gridbar_width):
            assert popover.isAncestorOf(spin), "a pitch control was left in the column"
        assert popover.isAncestorOf(widget.overlay_controls)

    def test_the_display_section_goes_when_it_has_nothing_to_say(self, widget):
        """With the switches moved out it holds only the view note, which is empty
        unless the displayed view differs from where the next run would land. An empty
        titled panel is chrome that has not earned its place."""
        assert not widget.label_view_note.isVisibleTo(widget)
        assert not widget.display_section.isVisibleTo(widget)

    def test_the_display_section_comes_back_with_the_note(self, widget, at_sem):
        self._note_showing(widget, at_sem)
        assert widget.label_view_note.isVisibleTo(widget)
        assert widget.display_section.isVisibleTo(widget)

    @staticmethod
    def _note_showing(widget, microscope):
        """Look at one view while the stage sits at another."""
        showing = widget.acquisition_view
        microscope.move_to_orientation("MILLING")
        widget.show_view(showing)


class TestTheTileGridHasItsOwnButton:
    """The planned tileset gets a button of its own rather than a row among the switches.

    It is the one overlay you *edit* -- drag it, resize it, click tiles out of it -- so
    it carries colour, fill and a re-centre beside its visibility, which is more than a
    checkbox row holds. Same panel class as the fluorescence tab, so the two tabs cannot
    drift apart on the one overlay they both draw (FIB-572).
    """

    def test_the_panel_drives_the_overlay(self, widget):
        overlay = widget.tile_grid_overlay
        assert overlay.is_grid_visible

        widget.tile_grid_panel.visibility_changed.emit(False)
        assert not overlay.is_grid_visible

        widget.tile_grid_panel.visibility_changed.emit(True)
        assert overlay.is_grid_visible

    def test_re_centring_clears_a_dragged_target(self, widget, microscope):
        """The panel's re-centre is the way back after dragging the grid off the stage,
        and `clear_target` is what this tab already calls it."""
        base = microscope.get_stage_position()
        widget._target = _at(base, dx=250e-6)

        widget.tile_grid_panel.centre_requested.emit()

        assert widget._target is None

    def test_the_button_opens_and_closes_it(self, widget):
        assert not widget.tile_grid_panel.isVisible()

        widget.btn_tile_grid.setChecked(True)
        widget._toggle_tile_grid_panel()
        assert widget.tile_grid_panel.isVisible()

        widget.btn_tile_grid.setChecked(False)
        widget._toggle_tile_grid_panel()
        assert not widget.tile_grid_panel.isVisible()

    def test_it_does_not_write_the_gesture_hint_over_the_view_caption(
        self, widget, at_sem
    ):
        """The fluorescence tab pairs this panel with a hint naming the grid's gestures,
        and that writes to the canvas's status zone -- which on this tab already carries
        the "no overview in this view" caption. Two writers to one zone is what the zone
        exists to prevent, so the hint was deliberately not ported."""
        base = at_sem.get_stage_position()
        widget.set_image(_tile(at_sem, _at(base), shape=(64, 64)))
        at_sem.move_to_orientation("MILLING")
        caption = widget.canvas._hint_text
        assert caption and "No overview in this view" in caption

        widget.btn_tile_grid.setChecked(True)
        widget._toggle_tile_grid_panel()

        assert widget.canvas._hint_text == caption, "the panel overwrote the caption"


class TestSayingWhenARunStarts:
    """`acquiring_changed`, which is how a host learns to lock the other overview.

    A host that hears the signal and then *asks* the widget must get a consistent
    answer, and that is the whole trap here: `acquire()` calls `_set_running(True)`
    before it builds the worker, so a worker-only `is_acquiring` said no while a run was
    starting -- which is exactly when the signal fires (FIB-706).
    """

    def test_the_signal_reports_the_new_state(self, widget):
        seen = []
        widget.acquiring_changed.connect(seen.append)
        widget._set_running(True)
        widget._set_running(False)
        assert seen == [True, False]

    def test_is_acquiring_already_agrees_when_the_signal_arrives(self, widget):
        """A host derives the lock by asking both tabs rather than trusting the bool, so
        the property has to be true by the time the signal says a run began."""
        answers = []
        widget.acquiring_changed.connect(lambda _: answers.append(widget.is_acquiring))
        widget._set_running(True)
        assert answers == [True], "the widget announced a run it does not admit to"
        widget._set_running(False)
        assert answers == [True, False]

    def test_a_finished_run_says_so(self, widget):
        """The finish path used to set the flag by hand, so it would have announced
        nothing and left a host holding the lock for good."""
        seen = []
        widget._set_running(True)
        widget.acquiring_changed.connect(seen.append)
        widget._on_finished({})
        assert seen == [False]
        assert widget.is_acquiring is False


class TestDrivingTheStageFromAHost:
    """`move_to` is what the lamella list calls, and it went through a weaker gate than
    a double-click on the same point -- so a locked tab could still be made to move."""

    def test_a_host_cannot_move_while_the_tab_is_locked(self, widget, monkeypatch):
        moved = []
        monkeypatch.setattr(widget, "_move_worker", moved.append)
        widget.set_interactive(False)
        try:
            widget.move_to(FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=0.0, t=0.0))
            assert moved == [], "moved the stage for a host while locked"
        finally:
            widget.set_interactive(True)

    def test_move_to_asks_the_same_question_a_double_click_does(
        self, widget, monkeypatch
    ):
        """Not merely "it refuses when locked" -- that could be any check. This is the
        one gate, so the two ways of asking for a move cannot come apart again.

        No worker is started: `_may_move` answering False is what stops it, which keeps
        a stage move off a test thread.
        """
        asked = []

        def refuse():
            asked.append(True)
            return False

        monkeypatch.setattr(widget, "_may_move", refuse)
        widget.move_to(FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=0.0, t=0.0))
        assert asked, "move_to did not go through _may_move"


class TestItOnlyDrawsItsOwnRuns:
    """`tiled_acquisition_signal` is about to carry a second producer.

    This widget places the payload's mosaic on its own canvas and counts the tiles into
    its own record, so a fluorescence run reaching here would be drawn as one of this
    tab's overviews. Only `modality` separates them (FIB-725).
    """

    def test_a_fluorescence_run_is_ignored(self, widget):
        from fibsem.imaging.tiling.progress import MODALITY_FLUORESCENCE

        widget._tiles_acquired = 0
        widget._apply_progress({
            "modality": MODALITY_FLUORESCENCE,
            "counter": 7, "total": 9, "msg": "Tile Collected",
        })
        assert widget._tiles_acquired == 0, "a fluorescence run moved this tab's count"

    def test_a_beam_run_is_still_drawn(self, widget):
        widget._tiles_acquired = 0
        widget._apply_progress({
            "modality": "beam",
            "counter": 7, "total": 9, "msg": "Tile Collected",
        })
        assert widget._tiles_acquired == 7

    def test_an_unlabelled_run_is_still_drawn(self, widget):
        """Anything predating the key — including a producer outside this repository
        subscribing to a public signal — must keep working."""
        widget._tiles_acquired = 0
        widget._apply_progress({"counter": 7, "total": 9, "msg": "Tile Collected"})
        assert widget._tiles_acquired == 7
