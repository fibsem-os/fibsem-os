"""A fluorescence image laid over the Overview tab and aligned by hand (FIB-1030).

Two things are pinned. The geometry's part: the map from an FM image's own plane to a
beam view's plane is the composition of the two projections, and its decomposition is
what the overlay is placed with -- checked against a marker, the independent authority
for where a stage position falls. The user's part: a drag is kept as metres along the
sample surface and a turn, the same four numbers in every view, and never written back
by a restore.
"""

import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt5")

import fibsem.config as fibsem_config
from fibsem import utils
from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)
from fibsem.structures import (
    BeamType,
    CameraImageTransform,
    FibsemImage,
    FibsemStagePosition,
    ImageSettings,
)
from fibsem.ui.widgets.canvas.aligned_images import decompose
from fibsem.ui.widgets.overview_widget import FibsemOverviewWidget


@pytest.fixture(scope="module")
def microscope():
    path = os.path.join(
        os.path.dirname(fibsem_config.__file__),
        "config",
        "sim-arctis-configuration.yaml",
    )
    scope, _ = utils.setup_session(manufacturer="Demo", config_path=path)
    assert scope.stage_is_compustage
    return scope


@pytest.fixture(autouse=True)
def _destroy_widgets(destroy_widgets_after_test):
    """See tests/ui/test_overview_widget.py: the widget leaves top-levels behind."""


@pytest.fixture
def widget(microscope):
    w = FibsemOverviewWidget(microscope)
    w.resize(900, 700)
    yield w
    w.close()


def _beam_image(scope, orientation, beam_type):
    pose = scope.get_orientation(orientation)
    position = FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=pose.r, t=pose.t)
    hfw = 128 * 2e-6
    image = FibsemImage.generate_blank_image(resolution=(128, 128), hfw=hfw)
    image.data = (np.random.default_rng(0).random((128, 128)) * 255).astype(np.uint8)
    state = scope.get_microscope_state(beam_type=beam_type)
    state.stage_position = position
    image.metadata.image_settings = ImageSettings(hfw=hfw, beam_type=beam_type)
    image.metadata.microscope_state = state
    image.metadata.system_info = scope.system.info
    image.metadata.hardware_geometry = scope.hardware_geometry()
    return image


def _fm_image(scope, position, size=64, pixel_size=1e-6):
    image = FluorescenceImage(
        data=(np.random.default_rng(2).random((1, 1, size, size)) * 4000).astype(
            np.uint16
        ),
        metadata=FluorescenceImageMetadata(
            acquisition_date="2026-09-23T10:00:00",
            pixel_size_x=pixel_size,
            pixel_size_y=pixel_size,
            stage_position=position,
            channels=[
                FluorescenceChannelMetadata(
                    name="GFP",
                    excitation_wavelength=488.0,
                    power=0.5,
                    exposure_time=0.1,
                    gain=1.0,
                    offset=0.0,
                    color="cyan",
                )
            ],
        ),
    )
    image.metadata.geometry = scope.fm_image_geometry()
    return image


SQUARE = ("SEM", BeamType.ELECTRON)
FORESHORTENED = ("MILLING", BeamType.ION)


def _show(widget, view):
    widget.set_image(_beam_image(widget.microscope, *view))
    return widget._frame()


def _fm_at(scope, dx=0.0, dy=0.0):
    pose = scope.get_orientation("FM")
    return FibsemStagePosition(x=dx, y=dy, z=0.0, r=pose.r, t=pose.t)


# ── the decomposition ─────────────────────────────────────────────────────


class TestDecompose:
    def test_identity(self):
        assert decompose(np.eye(2)) == (False, pytest.approx(0.0), 1.0, 1.0)

    def test_a_squash_is_read_off_the_second_row(self):
        mirror, rotation, squash, scale = decompose(np.diag([1.0, 0.616]))
        assert (mirror, rotation, squash, scale) == (
            False,
            pytest.approx(0.0),
            pytest.approx(0.616),
            1.0,
        )

    def test_a_turn_and_a_scale(self):
        theta = math.radians(30.0)
        c, s = math.cos(theta), math.sin(theta)
        a = 2.0 * np.array([[c, -s], [s, c]])
        mirror, rotation, squash, scale = decompose(a)
        assert not mirror
        assert rotation == pytest.approx(30.0)
        assert squash == pytest.approx(1.0)
        assert scale == pytest.approx(2.0)

    def test_a_mirror_is_a_mirror_and_not_a_turn(self):
        mirror, rotation, squash, scale = decompose(np.diag([-1.0, 1.0]))
        assert mirror and rotation == pytest.approx(0.0)
        assert (squash, scale) == (pytest.approx(1.0), pytest.approx(1.0))

    def test_the_composition_round_trips(self):
        """Build A from the model and get the same four numbers back."""
        for mirror, rot, squash, scale in [
            (False, 12.0, 0.7, 1.3),
            (True, -40.0, 0.259, 0.9),
        ]:
            c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
            a = np.diag([1.0, squash]) @ np.array([[c, -s], [s, c]])
            a = a @ np.diag([-1.0 if mirror else 1.0, 1.0]) * scale
            got = decompose(a)
            assert got[0] is mirror
            assert got[1:] == pytest.approx((rot, squash, scale))


# ── placed from metadata, against a marker ────────────────────────────────


class TestPlacedFromMetadata:
    def test_the_image_centre_lands_where_a_marker_at_its_position_would(
        self, widget, microscope
    ):
        frame = _show(widget, SQUARE)
        position = _fm_at(microscope, dx=30e-6, dy=-20e-6)
        key = widget.add_aligned_image(_fm_image(microscope, position), "fm")
        record = widget.aligned_images.get(key)
        assert record.overlay.is_visible
        marker = frame.to_canvas(position)
        assert record.overlay.centre == pytest.approx(marker, abs=1e-6)

    def test_the_footprint_is_the_images_own_pixels_at_its_own_pixel_size(
        self, widget, microscope
    ):
        frame = _show(widget, SQUARE)
        key = widget.add_aligned_image(
            _fm_image(microscope, _fm_at(microscope), size=64, pixel_size=1e-6), "fm"
        )
        record = widget.aligned_images.get(key)
        assert record.overlay.footprint == pytest.approx(
            (frame.length(64e-6), frame.length(64e-6))
        )

    def test_in_a_tilted_view_the_image_is_squashed_like_the_bars(
        self, widget, microscope
    ):
        frame = _show(widget, FORESHORTENED)
        key = widget.add_aligned_image(_fm_image(microscope, _fm_at(microscope)), "fm")
        record = widget.aligned_images.get(key)
        assert record.overlay.squash == pytest.approx(frame.surface_foreshortening())

    def test_an_image_with_no_geometry_is_refused(self, widget, microscope):
        _show(widget, SQUARE)
        image = _fm_image(microscope, _fm_at(microscope))
        image.metadata.geometry = None
        assert widget.add_aligned_image(image, "bare") is None
        assert widget.aligned_images.keys() == []


# ── the user's part ───────────────────────────────────────────────────────


class TestADragIsKeptOnTheSample:
    def test_a_move_is_metres_from_the_metadata_placement(self, widget, microscope):
        _show(widget, SQUARE)
        key = widget.add_aligned_image(_fm_image(microscope, _fm_at(microscope)), "fm")
        record = widget.aligned_images.get(key)
        cx, cy = record.overlay.centre
        ref = widget.canvas.reference_pixel_size
        seen = []
        widget.image_placement_changed.connect(seen.append)

        record.overlay.moved.emit(cx + 10.0, cy - 4.0)
        record.overlay.drag_finished.emit()

        assert record.placement[:2] == pytest.approx((10.0 * ref, -4.0 * ref))
        assert record.overlay.centre == pytest.approx((cx + 10.0, cy - 4.0))
        assert seen == [key]

    def test_in_a_squashed_view_the_canvas_step_is_unsquashed_before_it_is_kept(
        self, widget, microscope
    ):
        frame = _show(widget, FORESHORTENED)
        key = widget.add_aligned_image(_fm_image(microscope, _fm_at(microscope)), "fm")
        record = widget.aligned_images.get(key)
        cx, cy = record.overlay.centre
        ref = widget.canvas.reference_pixel_size
        squash = frame.surface_foreshortening()

        record.overlay.moved.emit(cx, cy + 10.0)

        assert record.dx == pytest.approx(0.0, abs=1e-12)
        assert record.dy == pytest.approx(10.0 * ref / squash)
        assert record.overlay.centre == pytest.approx((cx, cy + 10.0))

    def test_what_is_kept_is_a_plain_float(self, widget, microscope):
        """The canvas answers in numpy scalars; the record has to hold floats, or
        the experiment file cannot be written (and was emptied trying)."""
        _show(widget, SQUARE)
        key = widget.add_aligned_image(_fm_image(microscope, _fm_at(microscope)), "fm")
        record = widget.aligned_images.get(key)
        cx, cy = record.overlay.centre

        record.overlay.moved.emit(np.float64(cx + 3.0), np.float64(cy + 2.0))
        record.overlay.rotated.emit(np.float64(record.base_rotation + 1.0))
        record.overlay.scaled.emit(np.float64(1.1))

        assert all(type(v) is float for v in record.placement)

    def test_a_turn_is_kept_on_top_of_the_geometrys_turn(self, widget, microscope):
        """A camera transform that flips both axes is a half turn in the view: the
        geometry's part. What the user turns on top is what is kept."""
        from dataclasses import replace

        _show(widget, SQUARE)
        image = _fm_image(microscope, _fm_at(microscope))
        image.metadata.geometry = replace(
            image.metadata.geometry, transform=CameraImageTransform.FLIP_XY
        )
        key = widget.add_aligned_image(image, "fm")
        record = widget.aligned_images.get(key)
        base = record.base_rotation
        assert abs(abs(base) - 180.0) < 1e-6, "the flip did not turn the image"

        record.overlay.rotated.emit(base + 7.5)

        assert record.rotation == pytest.approx(7.5)
        assert record.overlay.rotation == pytest.approx(base + 7.5)

    def test_a_corner_scales_the_pixel_size_about_the_centre(self, widget, microscope):
        """The factor multiplies what is kept, not the file's pixel size: two drags
        compound, the centre stays put, and Reset still has the file's to go back to."""
        frame = _show(widget, SQUARE)
        key = widget.add_aligned_image(
            _fm_image(microscope, _fm_at(microscope), size=64, pixel_size=1e-6), "fm"
        )
        record = widget.aligned_images.get(key)
        centre = record.overlay.centre
        seen = []
        widget.image_placement_changed.connect(seen.append)

        record.overlay.scaled.emit(1.25)
        record.overlay.scaled.emit(1.2)
        record.overlay.drag_finished.emit()

        assert record.scale == pytest.approx(1.5)
        assert record.pixel_size == 1e-6
        assert record.overlay.footprint == pytest.approx(
            (frame.length(1.5 * 64e-6), frame.length(1.5 * 64e-6))
        )
        assert record.overlay.centre == pytest.approx(centre)
        assert seen == [key]

    def test_the_same_placement_draws_in_every_view(self, widget, microscope):
        _show(widget, SQUARE)
        key = widget.add_aligned_image(_fm_image(microscope, _fm_at(microscope)), "fm")
        widget.aligned_images.set_placement(key, 20e-6, 30e-6, 15.0)
        record = widget.aligned_images.get(key)

        frame = _show(widget, FORESHORTENED)

        squash = frame.surface_foreshortening()
        assert record.overlay.centre[0] - record.anchor[0] == pytest.approx(
            frame.length(20e-6), rel=1e-6
        )
        assert record.overlay.centre[1] - record.anchor[1] == pytest.approx(
            frame.length(30e-6) * squash, rel=1e-6
        )
        assert record.overlay.rotation == pytest.approx(record.base_rotation + 15.0)

    def test_reset_announces_and_set_placement_does_not(self, widget, microscope):
        _show(widget, SQUARE)
        key = widget.add_aligned_image(_fm_image(microscope, _fm_at(microscope)), "fm")
        seen = []
        widget.image_placement_changed.connect(seen.append)

        widget.aligned_images.set_placement(key, 1e-6, 2e-6, 3.0, 2.0)
        assert seen == []

        widget.aligned_image_panel.btn_reset.click()
        assert widget.aligned_images.get(key).placement == (0.0, 0.0, 0.0, 1.0)
        assert seen == [key]


class TestTheControls:
    def test_align_hands_the_canvas_to_the_selected_image_and_back(
        self, widget, microscope
    ):
        _show(widget, SQUARE)
        key = widget.add_aligned_image(_fm_image(microscope, _fm_at(microscope)), "fm")
        overlay = widget.aligned_images.get(key).overlay

        widget.aligned_image_panel.btn_align.setChecked(True)
        assert widget.canvas.active_overlay is overlay

        widget.aligned_image_panel.btn_align.setChecked(False)
        assert widget.canvas.active_overlay is None

    def test_one_thing_owns_the_canvas_at_a_time(self, widget, microscope):
        _show(widget, SQUARE)
        widget.add_aligned_image(_fm_image(microscope, _fm_at(microscope)), "fm")
        widget.overlay_controls.set_visible("gridbars", True)
        widget.btn_align_gridbars.setChecked(True)

        widget.aligned_image_panel.btn_align.setChecked(True)

        assert not widget.btn_align_gridbars.isChecked()
        assert widget.canvas.active_overlay is not widget.gridbar_overlay

    def test_remove_takes_the_image_off_the_canvas_and_the_list(
        self, widget, microscope
    ):
        _show(widget, SQUARE)
        key = widget.add_aligned_image(_fm_image(microscope, _fm_at(microscope)), "fm")
        widget.aligned_image_panel.btn_align.setChecked(True)

        widget.aligned_image_panel.btn_remove.click()

        assert widget.aligned_images.keys() == []
        assert widget.aligned_image_panel.combo.count() == 0
        assert widget.canvas.active_overlay is None
        assert not widget.aligned_image_panel.btn_align.isEnabled()
        assert key not in widget.aligned_images.keys()

    def test_the_readout_follows_the_selected_image(self, widget, microscope):
        _show(widget, SQUARE)
        key = widget.add_aligned_image(_fm_image(microscope, _fm_at(microscope)), "fm")
        assert "metadata" in widget.aligned_image_panel.label_placement.text()
        record = widget.aligned_images.get(key)
        record.overlay.moved.emit(
            record.overlay.centre[0] + 3.0, record.overlay.centre[1]
        )
        record.overlay.drag_finished.emit()
        assert "Moved" in widget.aligned_image_panel.label_placement.text()
        assert "pixel" not in widget.aligned_image_panel.label_placement.text()

        record.overlay.scaled.emit(1.25)
        record.overlay.drag_finished.emit()
        assert "pixel 1.25 um (×1.250)" in (
            widget.aligned_image_panel.label_placement.text()
        )


class TestFitFromPoints:
    """Three pairs place the image (FIB-1030). Ground truth comes from the placement
    itself: put the image somewhere by `set_placement`, read where three of its
    pixels land, put it back, and the fit from those pairs must find the same four
    numbers -- in a tilted view as well as looking straight down."""

    PIXELS = [(10.0, 10.0), (50.0, 12.0), (30.0, 55.0)]

    def _targets_for(self, widget, key, dx, dy, rotation, scale=1.0):
        widget.aligned_images.set_placement(key, dx, dy, rotation, scale)
        targets = [widget.aligned_images.pixel_to_canvas(key, *p) for p in self.PIXELS]
        widget.aligned_images.set_placement(key, 0.0, 0.0, 0.0, 1.0)
        return targets

    def test_the_fit_recovers_a_known_placement(self, widget, microscope):
        _show(widget, SQUARE)
        key = widget.add_aligned_image(_fm_image(microscope, _fm_at(microscope)), "fm")
        targets = self._targets_for(widget, key, 20e-6, -15e-6, 12.0, 1.1)
        seen = []
        widget.image_placement_changed.connect(seen.append)

        fit = widget.aligned_images.fit_to_points(key, self.PIXELS, targets)

        record = widget.aligned_images.get(key)
        assert record.placement == pytest.approx((20e-6, -15e-6, 12.0, 1.1), rel=1e-6)
        assert fit.rms == pytest.approx(0.0, abs=1e-6)
        for pixel, target in zip(self.PIXELS, targets):
            assert widget.aligned_images.pixel_to_canvas(key, *pixel) == pytest.approx(
                target, abs=1e-6
            )
        assert seen == [key]
        assert len(record.fit["pairs"]) == 3 and record.fit["rms"] == pytest.approx(0.0)

    def test_in_a_tilted_view_the_fit_is_made_on_the_sample(self, widget, microscope):
        """The pairs are clicked on a squashed canvas; a similarity there is not a
        similarity on the sample. Unsquashed first, the fit asks for the same
        correction it would looking straight down."""
        _show(widget, FORESHORTENED)
        key = widget.add_aligned_image(_fm_image(microscope, _fm_at(microscope)), "fm")
        targets = self._targets_for(widget, key, 20e-6, 30e-6, 15.0)

        widget.aligned_images.fit_to_points(key, self.PIXELS, targets)

        record = widget.aligned_images.get(key)
        assert record.placement == pytest.approx((20e-6, 30e-6, 15.0, 1.0), rel=1e-6)
        for pixel, target in zip(self.PIXELS, targets):
            assert widget.aligned_images.pixel_to_canvas(key, *pixel) == pytest.approx(
                target, abs=1e-6
            )

    def test_a_locked_scale_stays_at_one(self, widget, microscope):
        _show(widget, SQUARE)
        key = widget.add_aligned_image(_fm_image(microscope, _fm_at(microscope)), "fm")
        targets = self._targets_for(widget, key, 5e-6, 5e-6, 3.0, 1.2)

        fit = widget.aligned_images.fit_to_points(
            key, self.PIXELS, targets, fix_scale=True
        )

        assert fit.scale == 1.0
        assert widget.aligned_images.get(key).scale == 1.0
        assert widget.aligned_images.get(key).fit["fix_scale"] is True

    @pytest.mark.parametrize("gesture", ["move", "scale"])
    def test_a_drag_afterwards_forgets_the_fit(self, widget, microscope, gesture):
        _show(widget, SQUARE)
        key = widget.add_aligned_image(_fm_image(microscope, _fm_at(microscope)), "fm")
        targets = self._targets_for(widget, key, 5e-6, 5e-6, 3.0)
        widget.aligned_images.fit_to_points(key, self.PIXELS, targets)
        record = widget.aligned_images.get(key)
        assert record.fit

        cx, cy = record.overlay.centre
        if gesture == "move":
            record.overlay.moved.emit(cx + 1.0, cy)
        else:
            record.overlay.scaled.emit(1.1)

        assert record.fit == {}


class TestTheOverviewUnderTheImage:
    def test_the_reference_is_the_overview_holding_the_images_centre(
        self, widget, microscope
    ):
        _show(widget, SQUARE)
        key = widget.add_aligned_image(_fm_image(microscope, _fm_at(microscope)), "fm")
        reference = widget._reference_tile_for_fit(key)
        assert reference is not None
        canvas_key, tile = reference
        # Its corners are the placed extent's corners.
        (cx, cy), (w, h) = widget._extents[canvas_key]
        top_left = widget.canvas.metres_to_canvas(cx - w / 2, cy - h / 2)
        got = widget._tile_pixel_to_canvas(canvas_key, tile, -0.5, -0.5)
        assert got == pytest.approx(top_left, abs=1e-6)
        height, width = tile.grey.shape[:2]
        bottom_right = widget.canvas.metres_to_canvas(cx + w / 2, cy + h / 2)
        got = widget._tile_pixel_to_canvas(canvas_key, tile, width - 0.5, height - 0.5)
        assert got == pytest.approx(bottom_right, abs=1e-6)

    def test_with_nothing_placed_there_is_no_reference(self, widget, microscope):
        assert widget._reference_tile_for_fit("no-such-key") is None


class TestTargetingThroughTheAlignedImage:
    """The alignment is a targeting input, not a picture (FIB-1030): a double-click on
    a fluorescent spot moves the stage to where that spot is on the sample, and a
    right-click there offers a lamella. Nothing new is wired for it -- clicks resolve
    through the view's frame and the overlay takes none outside Align mode -- so this
    pins that it stays true."""

    def _spot(self, widget, microscope, dx=0.0, dy=0.0, rotation=0.0):
        _show(widget, SQUARE)
        position = _fm_at(microscope, dx=40e-6, dy=-25e-6)
        image = _fm_image(microscope, position, size=64, pixel_size=1e-6)
        key = widget.add_aligned_image(image, "fm")
        widget.aligned_images.set_placement(key, dx, dy, rotation)
        pixel = (50.0, 12.0)
        canvas_xy = widget.aligned_images.pixel_to_canvas(key, *pixel)
        return image, key, pixel, canvas_xy

    def test_a_double_click_on_the_image_names_the_spots_own_stage_position(
        self, widget, microscope
    ):
        """With no correction the image sits where its metadata says, so the stage
        position a click resolves to is the one the FM image's own projection gives
        for that pixel -- the two answers come from different code and must agree."""
        from fibsem.projection import FMStageProjection

        image, key, (px, py), (x, y) = self._spot(widget, microscope)

        target = widget._stage_position_at(x, y)
        assert target is not None, "the click was refused"

        projection = FMStageProjection.from_image(image)
        height, width = 64, 64
        own = projection.from_plane(
            (px + 0.5 - width / 2) * 1e-6,
            (py + 0.5 - height / 2) * 1e-6,
            image.metadata.stage_position,
        )
        assert (target.x, target.y) == pytest.approx((own.x, own.y), abs=2e-7)

    def test_a_correction_moves_the_target_with_the_image(self, widget, microscope):
        """Drag the image 10 um along the surface and a click on the same spot names a
        stage position 10 um over -- the correction is what a click acts on."""
        image, key, pixel, before = self._spot(widget, microscope)
        first = widget._stage_position_at(*before)
        widget.aligned_images.set_placement(key, 10e-6, 0.0, 0.0)
        after = widget.aligned_images.pixel_to_canvas(key, *pixel)

        second = widget._stage_position_at(*after)

        assert second.x - first.x == pytest.approx(10e-6, abs=2e-7)
        assert second.y - first.y == pytest.approx(0.0, abs=2e-7)

    def test_a_right_click_on_the_image_offers_a_lamella_there(
        self, widget, microscope
    ):
        image, key, pixel, (x, y) = self._spot(widget, microscope)
        target = widget._stage_position_at(x, y)

        menu = widget._position_menu(x, y)

        assert menu is not None
        labels = [action.label for action in menu.actions]
        assert "Add New Position Here" in labels
        requested = []
        widget.position_add_requested.connect(lambda pos, rec: requested.append(pos))
        menu.actions[labels.index("Add New Position Here")].callback()
        assert requested and (requested[0].x, requested[0].y) == pytest.approx(
            (target.x, target.y), abs=1e-9
        )

    def test_in_align_mode_a_click_belongs_to_the_image_not_the_stage(
        self, widget, microscope
    ):
        """Align hands the canvas to the overlay: the canvas's own click signals stand
        down, so dragging the image cannot also drive the stage."""
        self._spot(widget, microscope)
        widget.aligned_image_panel.btn_align.setChecked(True)
        assert widget.canvas.active_overlay is not None
        assert not widget.canvas._overlay_input_allowed(None)
        widget.aligned_image_panel.btn_align.setChecked(False)
