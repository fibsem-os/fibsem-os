"""The grid bars can be dragged and turned, and land squashed the way the view is.

Two layers. The overlay on its own, against a bare axes: the lattice's geometry through
its rotation and squash, and the gesture -- what a press, a drag and a release emit,
and that none of it happens unless the canvas has made the overlay active. Then the
Overview tab hosting it: the squash comes off the view's frame (FIB-615), a drag is
kept as metres along the sample surface so the same placement draws in every view, and
the Align button is the canvas's own overlay mode.
"""

import math
import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from fibsem import utils
from fibsem.structures import BeamType, FibsemImage, FibsemStagePosition, ImageSettings
from fibsem.ui.widgets.canvas.canvas_base import ContentRect
from fibsem.ui.widgets.canvas.overlays.gridbar_overlay import GridBarOverlay
from fibsem.ui.widgets.canvas.overlays.transform_overlay import (
    HANDLE_DISTANCE_PX,
    MOVE_DRAG_THRESHOLD_PX,
)
from fibsem.ui.widgets.overview_widget import FibsemOverviewWidget

PITCH = 50.0
WIDTH = 8.0


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class _FakeCanvas:
    """Enough canvas for the overlay to draw against, with a switchable active overlay."""

    def __init__(self):
        self._active_overlay = None
        self._overlay_consuming_event = False
        self.redraws = 0
        self.cursor = None

    def draw_idle(self):
        self.redraws += 1

    def setCursor(self, cursor):  # noqa: N802 - mirrors the Qt API
        self.cursor = cursor

    def unsetCursor(self):  # noqa: N802 - mirrors the Qt API
        self.cursor = None


def build(rotation=0.0, squash=1.0, active=True):
    from matplotlib.figure import Figure

    overlay = GridBarOverlay()
    fig = Figure(figsize=(8, 8), dpi=100)
    overlay._ax = fig.add_subplot(111)
    overlay._ax.set_xlim(0, 800)
    overlay._ax.set_ylim(800, 0)
    overlay._canvas = _FakeCanvas()
    if active:
        overlay._canvas._active_overlay = overlay
    overlay.on_content_changed(ContentRect(0.0, 0.0, 800.0, 800.0))
    overlay.set_visible(True)
    overlay.set_lattice((400.0, 400.0), PITCH, WIDTH, rotation=rotation, squash=squash)
    return overlay


def _event(overlay, x, y, button=1, px=0.0, py=0.0):
    return SimpleNamespace(
        button=button, inaxes=overlay._ax, xdata=x, ydata=y, x=px, y=py
    )


def _drag(overlay, start, end, px_travel=50.0):
    """Press at *start*, move to *end* (far enough on screen to be a drag), release."""
    overlay._on_press(_event(overlay, *start))
    overlay._on_motion(_event(overlay, *end, px=px_travel, py=px_travel))
    overlay._on_release(_event(overlay, *end, px=px_travel, py=px_travel))


def _bars(overlay):
    return [a for a in overlay._artists if getattr(a, "_gridbar", False)]


def _edges(bar):
    """A bar polygon's two edge vectors from its first corner: (along, across)."""
    xy = bar.get_xy()
    e1 = (xy[1][0] - xy[0][0], xy[1][1] - xy[0][1])
    e3 = (xy[3][0] - xy[0][0], xy[3][1] - xy[0][1])
    return (e1, e3) if math.hypot(*e1) >= math.hypot(*e3) else (e3, e1)


def _direction(bar):
    along, _ = _edges(bar)
    length = math.hypot(*along)
    return along[0] / length, along[1] / length


def _width_across(bar):
    _, across = _edges(bar)
    return math.hypot(*across)


# ── the lattice's geometry ─────────────────────────────────────────────────


class TestTheLatticeIsDrawnThroughItsRotationAndSquash:
    def test_unrotated_and_unsquashed_it_is_the_lattice_it_always_was(self, qapp):
        overlay = build()
        directions = {
            (round(abs(dx)), round(abs(dy)))
            for dx, dy in map(_direction, _bars(overlay))
        }
        assert directions == {(1, 0), (0, 1)}
        # A pitch of 50 across an 800 px view, out to the far corner (~566 px): each
        # family reaches past the corners, so the lattice fills the view however it is
        # panned. Two families of the same count.
        assert overlay.bar_count % 2 == 0 and overlay.bar_count > 2 * (800 / PITCH)

    def test_rotation_turns_every_bar_by_that_angle(self, qapp):
        overlay = build(rotation=30.0)
        angles = {
            round(math.degrees(math.atan2(dy, dx))) % 180
            for dx, dy in map(_direction, _bars(overlay))
        }
        assert angles == {30, 120}

    def test_the_squash_shortens_the_pitch_along_canvas_y_only(self, qapp):
        overlay = build(squash=0.25)
        x0, y0 = overlay.to_canvas(0.0, 0.0)
        x1, y1 = overlay.to_canvas(PITCH, 0.0)
        assert (x1 - x0, y1 - y0) == pytest.approx((PITCH, 0.0))
        x2, y2 = overlay.to_canvas(0.0, PITCH)
        assert (x2 - x0, y2 - y0) == pytest.approx((0.0, PITCH * 0.25))

    def test_a_bar_lying_along_the_squashed_axis_is_drawn_thinner(self, qapp):
        overlay = build(squash=0.25)
        widths = sorted({round(_width_across(a), 6) for a in _bars(overlay)})
        assert widths == [pytest.approx(WIDTH * 0.25), pytest.approx(WIDTH)]

    def test_the_bars_stop_at_the_grids_rim(self, qapp):
        """Beyond the rim the lattice said nothing about the sample and covered
        everything else drawn there. Every corner of every bar lies within the disc
        (plus half a bar's width, since a bar is cut square at the rim)."""
        overlay = build()
        radius = 220.0
        overlay.set_lattice((400.0, 400.0), PITCH, WIDTH, radius=radius)
        assert overlay.bar_count < 2 * (2 * radius / PITCH + 1) + 1
        for bar in _bars(overlay):
            for x, y in bar.get_xy():
                u, v = overlay.from_canvas(x, y)
                assert math.hypot(u, v) <= radius + WIDTH / 2 + 1e-6
        # ... and the bar through the centre spans the full diameter.
        assert max(math.hypot(*_edges(b)[0]) for b in _bars(overlay)) == pytest.approx(
            2 * radius
        )

    def test_from_canvas_inverts_to_canvas(self, qapp):
        overlay = build(rotation=37.0, squash=0.6)
        for u, v in [(0, 0), (PITCH, 0), (0, PITCH), (-3 * PITCH, 2.5 * PITCH)]:
            assert overlay.from_canvas(*overlay.to_canvas(u, v)) == pytest.approx(
                (u, v)
            )

    def test_the_handle_sits_up_the_lattice_axis_at_a_fixed_screen_distance(self, qapp):
        overlay = build(rotation=90.0)
        hx, hy = overlay._handle_position()
        cx, cy = overlay.centre
        # Turned a quarter clockwise, "up" points right on screen.
        assert hy == pytest.approx(cy)
        assert hx > cx
        (sx0, sy0), (sx1, sy1) = overlay._ax.transData.transform([(cx, cy), (hx, hy)])
        assert math.hypot(sx1 - sx0, sy1 - sy0) == pytest.approx(HANDLE_DISTANCE_PX)

    def test_the_handle_keeps_its_reach_through_a_zoom(self, qapp):
        """The handle is a control, not a feature of the sample: zoomed in four
        times it is still the same distance from the centre on screen, and its
        marker is sized in points. The canvas zooms without telling an overlay."""
        overlay = build()
        cx, cy = overlay.centre
        before = overlay._handle_position()
        overlay._ax.set_xlim(300, 500)
        overlay._ax.set_ylim(500, 300)
        after = overlay._handle_position()
        assert math.hypot(after[0] - cx, after[1] - cy) == pytest.approx(
            math.hypot(before[0] - cx, before[1] - cy) / 4.0
        )
        knob = overlay._artists[-1]
        assert knob.get_markersize() == pytest.approx(
            2 * 9.0 * 72.0 / overlay._ax.figure.dpi
        )

    def test_the_handles_are_drawn_only_while_the_overlay_is_active(self, qapp):
        active = build(active=True)
        assert len(active._artists) == active.bar_count + 3
        inert = build(active=False)
        assert len(inert._artists) == inert.bar_count


# ── the gesture ───────────────────────────────────────────────────────────


class TestTheGesture:
    def test_a_drag_inside_the_lattice_emits_the_moved_centre(self, qapp):
        overlay = build()
        seen = []
        overlay.moved.connect(lambda x, y: seen.append((x, y)))
        finished = []
        overlay.drag_finished.connect(lambda: finished.append(True))

        _drag(overlay, (100.0, 100.0), (130.0, 80.0))

        assert seen[-1] == pytest.approx((430.0, 380.0))
        assert finished == [True]
        assert overlay._canvas._overlay_consuming_event, (
            "the press was not claimed, so the canvas would pan under the drag"
        )

    def test_a_press_that_does_not_travel_is_not_a_drag(self, qapp):
        overlay = build()
        seen = []
        overlay.moved.connect(lambda x, y: seen.append((x, y)))
        finished = []
        overlay.drag_finished.connect(lambda: finished.append(True))

        overlay._on_press(_event(overlay, 100.0, 100.0))
        small = MOVE_DRAG_THRESHOLD_PX / 2
        overlay._on_motion(_event(overlay, 101.0, 100.0, px=small, py=0.0))
        overlay._on_release(_event(overlay, 101.0, 100.0, px=small, py=0.0))

        assert seen == []
        assert finished == []

    def test_dragging_the_handle_emits_the_rotation(self, qapp):
        overlay = build()
        seen = []
        overlay.rotated.connect(seen.append)
        cx, cy = overlay.centre
        hx, hy = overlay._handle_position()

        # From straight up to straight right about the centre: a quarter turn clockwise.
        _drag(overlay, (hx, hy), (cx + 60.0, cy))

        assert seen[-1] == pytest.approx(90.0)

    def test_the_handle_angle_is_read_through_the_squash(self, qapp):
        """In a squashed view the handle is dragged on the squashed canvas, but the turn
        it asks for is on the sample. Pulling it to 45 degrees on screen in a 0.5 view
        is not 45 degrees on the sample."""
        overlay = build(squash=0.5)
        seen = []
        overlay.rotated.connect(seen.append)
        cx, cy = overlay.centre
        hx, hy = overlay._handle_position()

        _drag(overlay, (hx, hy), (cx + 60.0, cy + 60.0))

        # On the sample the pointer sits at (60, 120): 63.4 degrees below the x axis,
        # which is 153.4 degrees clockwise from "up".
        expected = 90.0 + math.degrees(math.atan2(120.0, 60.0))
        assert seen[-1] == pytest.approx(expected)

    def test_nothing_is_emitted_while_the_overlay_is_not_active(self, qapp):
        overlay = build(active=False)
        seen = []
        overlay.moved.connect(lambda x, y: seen.append((x, y)))
        overlay.rotated.connect(seen.append)

        _drag(overlay, (100.0, 100.0), (130.0, 80.0))

        assert seen == []
        assert not overlay._canvas._overlay_consuming_event

    def test_a_double_click_is_left_to_the_canvas(self, qapp):
        overlay = build()
        seen = []
        overlay.moved.connect(lambda x, y: seen.append((x, y)))
        event = _event(overlay, 100.0, 100.0)
        event.dblclick = True

        overlay._on_press(event)
        overlay._on_motion(_event(overlay, 130.0, 80.0, px=50.0, py=50.0))

        assert seen == []


# ── hosted on the Overview tab ─────────────────────────────────────────────


@pytest.fixture(scope="module")
def microscope():
    import fibsem.config as fibsem_config

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


SQUARE = ("SEM", BeamType.ELECTRON)
FORESHORTENED = ("MILLING", BeamType.ION)


def _show(widget, view):
    widget.set_image(_image(widget.microscope, *view))
    widget.overlay_controls.set_visible("gridbars", True)
    widget._refresh_context_overlays()
    return widget._frame()


class TestTheBarsMatchTheView:
    def test_the_squash_is_the_views_surface_foreshortening(self, widget):
        """The bug pinned (FIB-615): one pitch for both axes, so at the milling pose
        the horizontal bars sat nearly four times too far apart."""
        frame = _show(widget, FORESHORTENED)
        expected = frame.surface_foreshortening()
        assert expected < 0.5, "the milling view stopped being foreshortened"
        assert widget.gridbar_overlay.squash == pytest.approx(expected)

    def test_looking_straight_down_the_lattice_is_square(self, widget):
        _show(widget, SQUARE)
        assert widget.gridbar_overlay.squash == pytest.approx(1.0)

    def test_the_pitch_along_the_squashed_axis_matches_a_marker_step(self, widget):
        """The independent authority: a stage step along y carries a *marker* by the
        same canvas distance the lattice's pitch along v comes out at."""
        frame = _show(widget, FORESHORTENED)
        overlay = widget.gridbar_overlay
        pitch_m = widget.spin_gridbar_spacing.value() * 1e-6
        origin = frame.origin
        here = frame.to_canvas(
            FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=origin.r, t=origin.t)
        )
        there = frame.to_canvas(
            FibsemStagePosition(x=0.0, y=pitch_m, z=0.0, r=origin.r, t=origin.t)
        )
        x0, y0 = overlay.to_canvas(0.0, 0.0)
        x1, y1 = overlay.to_canvas(0.0, overlay.pitch)
        assert abs(y1 - y0) == pytest.approx(abs(there[1] - here[1]), rel=1e-6)


class TestADragIsKeptAlongTheSurface:
    def test_a_move_is_recorded_in_metres_from_the_grid_centre(self, widget):
        _show(widget, SQUARE)
        overlay = widget.gridbar_overlay
        cx, cy = overlay.centre
        ref = widget.canvas.reference_pixel_size

        overlay.moved.emit(cx + 10.0, cy - 4.0)

        dx, dy, rotation = widget.gridbar_placement
        assert (dx, dy) == pytest.approx((10.0 * ref, -4.0 * ref))
        assert rotation == 0.0
        assert overlay.centre == pytest.approx((cx + 10.0, cy - 4.0))

    def test_in_a_squashed_view_the_canvas_step_is_unsquashed_before_it_is_kept(
        self, widget
    ):
        frame = _show(widget, FORESHORTENED)
        overlay = widget.gridbar_overlay
        cx, cy = overlay.centre
        ref = widget.canvas.reference_pixel_size
        squash = frame.surface_foreshortening()

        overlay.moved.emit(cx, cy + 10.0)

        dx, dy, _ = widget.gridbar_placement
        assert dx == pytest.approx(0.0, abs=1e-12)
        assert dy == pytest.approx(10.0 * ref / squash)
        # ... and drawn back squashed, it lands where it was dragged to.
        assert overlay.centre == pytest.approx((cx, cy + 10.0))

    def test_the_same_placement_draws_in_every_view(self, widget):
        """Dragged in the square view, the offset is a place on the sample; shown in
        the milling view it is squashed the way that view squashes everything."""
        _show(widget, SQUARE)
        widget.set_gridbar_placement(20e-6, 30e-6, 15.0)
        square = widget.gridbar_overlay
        anchor_sq = square.to_canvas(0.0, 0.0)

        frame = _show(widget, FORESHORTENED)
        overlay = widget.gridbar_overlay
        assert overlay.rotation == 15.0
        squash = frame.surface_foreshortening()
        anchor = widget._frame().to_canvas(
            widget._landmark(frame, 0.0, 0.0, "Grid Centre")
        )
        assert overlay.centre[0] - anchor[0] == pytest.approx(
            frame.length(20e-6), rel=1e-6
        )
        assert overlay.centre[1] - anchor[1] == pytest.approx(
            frame.length(30e-6) * squash, rel=1e-6
        )
        del anchor_sq

    def test_rotation_and_reset(self, widget):
        _show(widget, SQUARE)
        widget.gridbar_overlay.rotated.emit(-12.5)
        assert widget.gridbar_placement[2] == -12.5
        # The readout waits for the gesture to end: it is not worth a label repaint
        # on every motion event.
        assert "grid centre" in widget.label_gridbar_placement.text()
        widget.gridbar_overlay.drag_finished.emit()
        assert "turned -12.5°" in widget.label_gridbar_placement.text()

        widget.btn_reset_gridbars.click()

        assert widget.gridbar_placement == (0.0, 0.0, 0.0)
        assert widget.gridbar_overlay.rotation == 0.0
        assert "grid centre" in widget.label_gridbar_placement.text()


class TestAlignIsTheCanvasOverlayMode:
    def test_the_button_hands_the_canvas_to_the_bars_and_back(self, widget):
        _show(widget, SQUARE)
        assert widget.canvas.active_overlay is None

        widget.btn_align_gridbars.setChecked(True)
        assert widget.canvas.active_overlay is widget.gridbar_overlay
        assert widget.gridbar_overlay.is_editable()
        assert not widget.canvas.btn_mode.isHidden()

        widget.btn_align_gridbars.setChecked(False)
        assert widget.canvas.active_overlay is None
        assert widget.canvas.btn_mode.isHidden()

    def test_turning_the_bars_off_leaves_the_mode(self, widget):
        _show(widget, SQUARE)
        widget.btn_align_gridbars.setChecked(True)

        widget.overlay_controls.set_visible("gridbars", False)

        assert not widget.btn_align_gridbars.isChecked()
        assert widget.canvas.active_overlay is None
        assert not widget.btn_align_gridbars.isEnabled()

    def test_unchecking_the_canvas_toggle_leaves_the_mode_too(self, widget):
        _show(widget, SQUARE)
        widget.btn_align_gridbars.setChecked(True)

        widget.canvas.btn_mode.setChecked(False)

        assert not widget.btn_align_gridbars.isChecked()
        assert widget.canvas.active_overlay is None

    def test_the_controls_start_matched_to_the_checkbox(self, widget):
        assert not widget.btn_align_gridbars.isEnabled()
        assert not widget.btn_reset_gridbars.isEnabled()
        widget.overlay_controls.set_visible("gridbars", True)
        assert widget.btn_align_gridbars.isEnabled()
        assert widget.btn_reset_gridbars.isEnabled()
