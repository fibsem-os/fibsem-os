"""The FIB/SEM overview widget: the inversion, and the rules it was built to keep.

The tab this replaces stitched one giant image and then reprojected stage positions onto
its pixels, so everything on screen was gated on the stitch. Here each tile is placed
where it was acquired, and that is what these tests pin -- along with the two rules the
old widget broke: no hardware read on a UI event, and one derivation for both directions
so a click resolves to the position the marker was drawn from.

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

from PyQt5.QtWidgets import QApplication  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.structures import (  # noqa: E402
    BeamType,
    FibsemImage,
    FibsemStagePosition,
    ImageSettings,
    OverviewAcquisitionSettings,
)
from fibsem.ui.widgets.overview_widget import FibsemOverviewWidget  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv)


@pytest.fixture(scope="module")
def microscope():
    scope, _ = utils.setup_session(manufacturer="Demo")
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


class TestTheInversion:
    """Tiles are placed where they were acquired, not assembled into one image first."""

    def test_each_tile_lands_at_its_own_position(self, widget, microscope):
        """The whole point of the move. The old tab replaced one image per progress
        update with the growing stitch buffer, so nothing appeared until the buffer
        existed and every tile shared one placement."""
        base = microscope.get_stage_position()
        widget._record_count += 1
        widget._active_record = "run"
        from fibsem.ui.widgets.overview_widget import OverviewRecord

        widget._records["run"] = OverviewRecord("run", "run", [])

        offsets = [(0.0, 0.0), (50e-6, 0.0), (0.0, 50e-6)]
        for dx, dy in offsets:
            widget._place_tile(_tile(microscope, _at(base, dx, dy)), "run")

        assert len(widget._records["run"].keys) == 3, "tiles replaced instead of placing"
        extents = [widget.canvas._placed[k].extent for k in widget.canvas.placed_keys]
        assert len(set(extents)) == 3, "tiles were placed on top of each other"

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
        assert "Overview FoV" in labels, "the planned run is not drawn"
        # The limits and the holder slots are configuration-dependent; on a compustage
        # system all four are present, and this fixture is one.
        assert {"Stage Limits", "Grid Boundary"} <= labels

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

        def footprint():
            for spec in widget.context_overlay._specs:
                if spec.label == "Overview FoV":
                    return spec.width
            return None

        small = footprint()
        settings = widget.settings_widget.get_settings()
        settings.nrows, settings.ncols = settings.nrows + 3, settings.ncols + 3
        widget.settings_widget.update_from_settings(settings)
        widget._refresh_context_overlays()

        assert footprint() > small, "the drawn footprint ignored the settings"


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
        from fibsem.ui.widgets.overview_widget import OverviewRecord

        widget._records["run"] = OverviewRecord("run", "run", [])
        for dx in (0.0, 50e-6, 100e-6):
            widget._place_tile(_tile(microscope, _at(base, dx)), "run")
        assert len(widget.canvas.placed_keys) == 3

        assert widget.set_overview_visible("run", False)
        assert all(
            not widget.canvas._placed[k].artist.get_visible()
            for k in widget.canvas.placed_keys
        )

        assert widget.remove_overview("run")
        assert widget.canvas.placed_keys == []
        assert widget.remove_overview("run") is False


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
        for name in ("nrows_spinbox", "ncols_spinbox"):
            spinbox = getattr(widget.settings_widget, name)
            assert spinbox.width() >= 80, (
                f"{name} is {spinbox.width()}px at the narrowest column -- too narrow "
                "to draw its digit"
            )

    def test_the_list_keeps_up_with_a_run_in_progress(self, widget, microscope):
        """The row shows a tile count, so it has to be rebuilt as tiles land -- not only
        when the run finishes. A run showing "0 tiles" while nine sit on the canvas
        beside it is the list contradicting the display.

        Read off the *row widget*, not the record: the record is updated either way, so
        asserting on it proves nothing about what reaches the screen.
        """
        from fibsem.ui.widgets.overview_widget import OverviewRecord

        base = microscope.get_stage_position()
        widget._records["run"] = OverviewRecord("run", "run", [])
        widget._refresh_overview_list()
        assert widget.overview_list._list.count() == 1

        for i, dx in enumerate((0.0, 50e-6, 100e-6), start=1):
            widget._place_tile(_tile(microscope, _at(base, dx)), "run")
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


def test_the_tiled_progress_signal_carries_the_tile(microscope):
    """The placement depends on it, and it is an additive key on a shared signal -- so
    a consumer reading only `counter`/`total`/`image` is unaffected, and this is what
    says the key is still there."""
    from fibsem.imaging.tiled import TiledAcquisitionRunner

    payloads = []
    microscope.tiled_acquisition_signal.connect(payloads.append)
    try:
        settings = OverviewAcquisitionSettings(
            image_settings=ImageSettings(
                hfw=100e-6, resolution=(64, 64), beam_type=BeamType.ELECTRON,
                save=False, path="/tmp/overview-test", filename="t",
            ),
            nrows=1, ncols=2,
        )
        TiledAcquisitionRunner(microscope, settings).run()
    finally:
        microscope.tiled_acquisition_signal.disconnect(payloads.append)

    tiles = [p for p in payloads if p.get("tile") is not None]
    assert len(tiles) == 2, "the per-tile update did not carry its tile"
    for payload in tiles:
        tile = payload["tile"]
        assert tile.metadata.stage_position is not None
        assert tile.metadata.pixel_size.x
        # The keys every existing consumer reads, still present.
        assert {"counter", "total", "msg", "image"} <= set(payload)
