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
        the canvas's display cap, which is every real tile."""
        base = microscope.get_stage_position()
        cap = widget.canvas._display_max_px
        big = cap * 2
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
        for name in ("nrows_spinbox", "ncols_spinbox"):
            spinbox = getattr(widget.settings_widget, name)
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
        widget.settings_widget.image_settings_widget.path_edit.setText("")
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
        assert handed_over.image_settings is not (
            widget.settings_widget.image_settings_widget._settings
        ), "the runner was handed the settings widget's own object"

    def test_the_save_directory_fills_the_path_box(self, widget, tmp_path):
        """From the experiment by default, and visible so a user can see and change it.

        An empty box is also what made `get_settings()` read the path back as None.
        """
        widget.set_save_directory(str(tmp_path))
        assert widget.settings_widget.image_settings_widget.path_edit.text() == str(
            tmp_path
        )
        assert widget._settings().image_settings.path == str(tmp_path)

    def test_a_run_always_has_somewhere_to_write(self, widget, monkeypatch):
        """No host directory and an empty box still has to produce a usable path --
        finding out at the second tile is the worst possible time."""
        widget.set_save_directory(None)
        widget.settings_widget.image_settings_widget.path_edit.setText("")
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
            widget.settings_widget.image_settings_widget.filename_edit.text()
            == "overview-image"
        )

    def test_a_run_starts_from_the_overview_defaults(self, widget, monkeypatch):
        """A 500 um tile with autocontrast, not `ImageSettings`'s generic 150 um.

        The tab referenced the overview defaults nowhere at all, so it opened at
        whatever `ImageSettingsWidget` happens to default to. Every value below differs
        from that, which is the point: a tab that looks right and images a ninth of the
        area asked for is not something the picture shows you.

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
        assert tuple(settings.image_settings.resolution) == (1024, 1024)
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
            widget.checkbox_gridbars.setChecked(True)

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

    def test_no_mask_by_default(self, widget):
        assert widget._settings().tile_mask is None

    def test_a_mask_reaches_the_settings(self, widget):
        settings_widget = widget.settings_widget
        settings_widget.nrows_spinbox.setValue(3)
        settings_widget.ncols_spinbox.setValue(3)
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

    def test_a_mask_that_no_longer_fits_the_grid_is_dropped(self, widget):
        """Rows and columns are spin boxes; the mask is positional. `compute_tile_grid`
        rejects a mismatched one outright, so carrying it on would turn resizing the
        grid into a failed run rather than a bigger acquisition."""
        settings_widget = widget.settings_widget
        settings_widget.nrows_spinbox.setValue(3)
        settings_widget.ncols_spinbox.setValue(3)
        settings_widget.tile_mask = self._mask(3, 3, disabled=[(0, 0)])
        assert widget._settings().tile_mask is not None

        settings_widget.ncols_spinbox.setValue(4)
        assert widget._settings().tile_mask is None, "a stale mask was carried on"
        assert widget._settings().n_enabled_tiles == 12


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

    def test_an_undragged_run_still_says_nothing_about_a_centre(
        self, widget, monkeypatch, tmp_path
    ):
        """None means "wherever the stage is", which the runner resolves itself. Sending
        a position instead would freeze the grid at wherever the widget last saw the
        stage, which is not the same thing."""
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
        assert captured["args"][1] is None

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

        widget.settings_widget.beam_type_combo.set_value(BeamType.ION)

        assert widget.current_view.beam_type is BeamType.ION
        assert widget.tile_grid_overlay._tiles, "the planned grid disappeared"

    def test_switching_back_brings_the_overview_with_it(self, widget, microscope):
        """The images are kept per view, so this is a switch and not a loss."""
        widget.set_image(self._image(microscope, BeamType.ELECTRON))
        assert len(widget.canvas.placed_keys) == 1

        widget.settings_widget.beam_type_combo.set_value(BeamType.ION)
        assert len(widget.canvas.placed_keys) == 0, "the ion view is empty, as it should be"

        widget.settings_widget.beam_type_combo.set_value(BeamType.ELECTRON)
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
    """The view selector lives on the canvas, and says which view is which.

    It selects what you are looking at, so it belongs where you are looking rather than
    in the settings column. And the labels had to change: the orientations are named
    after the beams, so pairing the two read as a tautology one way ("SEM · Electron")
    and a contradiction the other ("SEM · Ion").
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
        widget.settings_widget.beam_type_combo.set_value(BeamType.ION)

        labels = {b.text() for b in widget._view_chip_buttons.values()}
        assert {"SEM @ SEM", "FIB @ SEM"} <= labels
        # Placed on the canvas, not merely constructed: they were once sized from a
        # layout that answered with nothing, so they reported themselves visible and
        # drew at zero size.
        assert all(b.width() > 0 for b in widget._view_chip_buttons.values())
        assert all(
            b.parent() is widget.canvas for b in widget._view_chip_buttons.values()
        )

    def test_clicking_a_chip_switches_the_canvas(self, widget, microscope):
        widget.set_image(self._image(microscope, "SEM", BeamType.ELECTRON))
        sem = widget.current_view
        widget.settings_widget.beam_type_combo.set_value(BeamType.ION)
        assert widget.current_view != sem
        assert not widget.canvas.placed_keys

        widget._view_chip_buttons[sem].click()
        assert widget.current_view == sem
        assert len(widget.canvas.placed_keys) == 1, "the overview did not come back"

    def test_the_chip_for_the_displayed_view_is_the_checked_one(self, widget, microscope):
        widget.set_image(self._image(microscope, "SEM", BeamType.ELECTRON))
        widget.settings_widget.beam_type_combo.set_value(BeamType.ION)

        checked = {b.text() for b in widget._view_chip_buttons.values() if b.isChecked()}
        assert checked == {widget.current_view.label}

    def test_the_view_the_run_would_land_in_is_marked_apart(self, widget, microscope):
        """Which view is displayed and which the next run lands in come apart on this
        tab, and the chips have to say which is which -- otherwise the only way to find
        out is to acquire and see where it went."""
        widget.set_image(self._image(microscope, "SEM", BeamType.ELECTRON))
        widget.settings_widget.beam_type_combo.set_value(BeamType.ION)
        acquisition = widget.acquisition_view
        widget._view_chip_buttons[widget.views[0]].click()  # look at the SEM one

        assert widget.current_view != acquisition
        marked = widget._view_chip_buttons[acquisition]
        other = widget._view_chip_buttons[widget.current_view]
        assert marked.styleSheet() != other.styleSheet()
        assert "next acquisition" in marked.toolTip()


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
        rgba = np.asarray(tile.data)
        acquired = rgba[..., 3] > 0
        # A third of the image, give or take the boundary block -- not a third plus a
        # filter radius, which is what the filtered array would have given.
        assert acquired.mean() == pytest.approx(1 / 3, abs=0.02), (
            f"coverage came out at {acquired.mean():.3f} of the image"
        )
        grey = rgba[..., 0][acquired]
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
        same second would pass for the wrong reason either way."""
        from fibsem.ui.widgets import overview_widget as module

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
        widget.settings_widget.image_settings_widget.filename_edit.setText("grid-2-survey")
        name = self._capture(widget, monkeypatch, tmp_path)
        assert name.startswith("grid-2-survey-")
        assert name != "grid-2-survey"

    def test_the_box_still_shows_the_base_name(self, widget, monkeypatch, tmp_path):
        """Stamped at the run, not in the control: a box that rewrote itself on every
        acquisition would make the name unusable as a thing you set once."""
        self._capture(widget, monkeypatch, tmp_path)
        assert (
            widget.settings_widget.image_settings_widget.filename_edit.text()
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
