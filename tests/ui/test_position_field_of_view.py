"""A marked position is drawn with the field of view it stands for, and picked by it.

A crosshair says where a lamella is. On a canvas spanning millimetres it says nothing
about how much sample one image covers there, which is what you are actually judging --
whether two markers would land in one frame, whether one sits far enough inside a grid
square. `FieldOfViewOverlay` draws that box, and a click inside it selects the position
it belongs to.

The picking is a **union** with the existing screen-space radius rather than a
replacement for it, and that is the part worth pinning: the box is not always the bigger
target. A 100 um box is under 24 px tall past ~2.8 um per screen pixel -- on a ~1000 px
canvas, a field of about 2.8 mm, which is roughly a whole grid and therefore the view
with the most markers in it. Two tests below hold each half of the union, because either
one alone passes with the other half deleted.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_position_field_of_view.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

# CI installs `.[test]`, not `.[ui]`, so PyQt5 is absent there.
pytest.importorskip("PyQt5")

from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.text import Annotation  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.structures import (  # noqa: E402
    BeamType,
    FibsemImage,
    FibsemStagePosition,
    ImageSettings,
)
from fibsem.ui.widgets.canvas.overlays.point_overlay import (  # noqa: E402
    FOV_BOX_LINEWIDTH,
    FOV_LABEL_FONTSIZE,
    FieldOfViewOverlay,
)
from fibsem.ui.widgets.canvas.real_space_canvas import (  # noqa: E402
    FibsemRealSpaceCanvas,
)
from fibsem.ui.widgets.overview_widget import (  # noqa: E402
    PICK_RADIUS_PX,
    POSITION_FOV_HEIGHT,
    POSITION_FOV_WIDTH,
    FibsemOverviewWidget,
)

_app = QApplication.instance() or QApplication(sys.argv)

PIXEL_SIZE = 1e-6  # 1 um per canvas unit, so metres and canvas units read off directly


def _boxes(overlay):
    return [a for a in overlay._artists if isinstance(a, Rectangle)]


def _labels(overlay):
    return [a for a in overlay._artists if isinstance(a, Annotation)]


def _canvas(pixel_size: float = PIXEL_SIZE) -> FibsemRealSpaceCanvas:
    canvas = FibsemRealSpaceCanvas()
    canvas.set_reference_pixel_size(pixel_size)
    return canvas


# ── the overlay ──────────────────────────────────────────────────────────────


class TestTheBoxIsTheFieldOfView:
    def test_the_box_is_the_extent_it_was_given(self):
        overlay = FieldOfViewOverlay(
            color="#ffffff", marker="+", extent=(100e-6, 60e-6)
        )
        canvas = _canvas()
        canvas.add_overlay(overlay)

        overlay.set_points([(0.0, 0.0)])

        (box,) = _boxes(overlay)
        assert box.get_width() == pytest.approx(100e-6 / PIXEL_SIZE)
        assert box.get_height() == pytest.approx(60e-6 / PIXEL_SIZE)
        assert box.get_xy() == pytest.approx((-50.0, -30.0)), "not centred on the point"

    def test_the_extent_is_metres_not_canvas_units(self):
        """Halving the canvas scale must double the box, not leave it the same size.

        The scale is not fixed -- it arrives with the first image, and the fluorescence
        tab sets it from the camera before then -- so an extent stored in canvas units
        would quietly come to mean a different number of microns.
        """
        sizes = []
        for pixel_size in (1e-6, 0.5e-6):
            overlay = FieldOfViewOverlay(extent=(100e-6, 60e-6))
            canvas = _canvas(pixel_size)
            canvas.add_overlay(overlay)
            overlay.set_points([(0.0, 0.0)])
            sizes.append(_boxes(overlay)[0].get_width())

        assert sizes[1] == pytest.approx(sizes[0] * 2)

    def test_one_box_per_point(self):
        overlay = FieldOfViewOverlay(extent=(100e-6, 60e-6))
        _canvas().add_overlay(overlay)

        overlay.set_points([(0.0, 0.0), (200.0, 0.0), (0.0, 200.0)])

        assert len(_boxes(overlay)) == 3

    def test_it_is_drawn_lightly(self):
        """The box is context for the marker, not the subject. At full weight its edge
        competes with the tile grid behind it and reads as part of that lattice."""
        overlay = FieldOfViewOverlay(color="#26c6da", extent=(100e-6, 60e-6))
        _canvas().add_overlay(overlay)

        overlay.set_points([(0.0, 0.0)], labels=["Lamella-01"])

        (box,) = _boxes(overlay)
        assert box.get_linewidth() == FOV_BOX_LINEWIDTH
        assert box.get_edgecolor()[:3] == pytest.approx(
            (0x26 / 255, 0xC6 / 255, 0xDA / 255), abs=0.01
        )
        assert box.get_fill() is False, "a filled box would hide the sample it frames"
        assert _labels(overlay)[0].get_fontsize() == FOV_LABEL_FONTSIZE

    def test_the_box_sits_under_the_marker(self):
        """Where a neighbouring box crosses a crosshair, the crosshair stays legible."""
        overlay = FieldOfViewOverlay(marker="+", extent=(100e-6, 60e-6))
        _canvas().add_overlay(overlay)

        overlay.set_points([(0.0, 0.0)])

        from matplotlib.lines import Line2D

        marker = next(a for a in overlay._artists if isinstance(a, Line2D))
        assert _boxes(overlay)[0].get_zorder() < marker.get_zorder()


class TestTheLabelIsAboveTheBox:
    def test_the_label_sits_above_the_top_left_corner(self):
        """Beside the crosshair -- where `PointsOverlay` puts it -- lands the text
        inside the box, over the sample the box exists to show.

        The y axis is inverted (origin upper), so the top edge is the *smaller* y.
        """
        overlay = FieldOfViewOverlay(extent=(100e-6, 60e-6))
        _canvas().add_overlay(overlay)

        overlay.set_points([(0.0, 0.0)], labels=["Lamella-01"])

        (label,) = _labels(overlay)
        assert label.xy == pytest.approx((-50.0, -30.0)), "not the top-left corner"
        assert label.get_ha() == "left"
        assert label.get_va() == "bottom", "would put the text inside the box"

    def test_an_empty_label_still_draws_nothing(self):
        """How a caller marks one point unnamed without giving up labels for the rest."""
        overlay = FieldOfViewOverlay(extent=(100e-6, 60e-6))
        _canvas().add_overlay(overlay)

        overlay.set_points([(0.0, 0.0)], labels=[""])

        assert _labels(overlay) == []


class TestWithoutABoxItIsTheOverlayItSubclasses:
    """No extent, or no canvas scale, must degrade to a plain crosshair -- not to a
    box of some guessed size, and not to nothing."""

    def test_no_extent_draws_no_box(self):
        overlay = FieldOfViewOverlay(marker="+")
        _canvas().add_overlay(overlay)

        overlay.set_points([(0.0, 0.0)], labels=["Lamella-01"])

        assert _boxes(overlay) == []
        assert len(_labels(overlay)) == 1

    def test_no_canvas_scale_draws_no_box(self):
        """`metres_to_canvas` answers (0, 0) before the canvas has a reference pixel
        size, which is a real state on the fluorescence tab: the camera geometry has to
        be read before the box can be sized, and that read can fail."""
        overlay = FieldOfViewOverlay(marker="+", extent=(100e-6, 60e-6))
        canvas = FibsemRealSpaceCanvas()  # deliberately no reference pixel size
        canvas.add_overlay(overlay)

        overlay.set_points([(0.0, 0.0)], labels=["Lamella-01"])

        assert overlay.half_extent_canvas() is None
        assert _boxes(overlay) == []

    def test_the_label_falls_back_beside_the_marker(self):
        """With no box there is no corner to hang the name off."""
        overlay = FieldOfViewOverlay(marker="+")
        _canvas().add_overlay(overlay)

        overlay.set_points([(3.0, 4.0)], labels=["Lamella-01"])

        (label,) = _labels(overlay)
        assert label.xy == pytest.approx((3.0, 4.0))


class TestSettingTheSameExtentIsNotAnEvent:
    """Hosts call `set_extent` from the same refresh that redraws the markers.

    Repainting a canvas to tell it what it already knows is the cost that made the grid
    drag stutter (FIB-752), and here it would also throw away the artists picking is
    measured against.
    """

    def test_the_same_extent_does_not_rebuild_the_artists(self):
        overlay = FieldOfViewOverlay(extent=(100e-6, 60e-6))
        _canvas().add_overlay(overlay)
        overlay.set_points([(0.0, 0.0)])
        before = list(overlay._artists)

        overlay.set_extent(100e-6, 60e-6)

        assert overlay._artists == before, "redrew for a size it already had"

    def test_a_different_extent_does(self):
        overlay = FieldOfViewOverlay(extent=(100e-6, 60e-6))
        _canvas().add_overlay(overlay)
        overlay.set_points([(0.0, 0.0)])

        overlay.set_extent(200e-6, 120e-6)

        assert _boxes(overlay)[0].get_width() == pytest.approx(200.0)


class TestCoversIsTheBoxThatIsDrawn:
    def test_a_point_inside_the_box_is_covered(self):
        overlay = FieldOfViewOverlay(extent=(100e-6, 60e-6))
        _canvas().add_overlay(overlay)

        assert overlay.covers((0.0, 0.0), 49.0, 29.0)
        assert overlay.covers((0.0, 0.0), -49.0, -29.0)

    def test_a_point_outside_it_is_not(self):
        overlay = FieldOfViewOverlay(extent=(100e-6, 60e-6))
        _canvas().add_overlay(overlay)

        assert not overlay.covers((0.0, 0.0), 51.0, 0.0)
        assert not overlay.covers((0.0, 0.0), 0.0, 31.0), (
            "tested the width for both axes"
        )

    def test_it_is_measured_from_the_point_it_is_given(self):
        overlay = FieldOfViewOverlay(extent=(100e-6, 60e-6))
        _canvas().add_overlay(overlay)

        assert overlay.covers((500.0, 500.0), 540.0, 520.0)
        assert not overlay.covers((500.0, 500.0), 40.0, 20.0)

    def test_no_box_covers_nothing(self):
        """So a caller's own hit radius stays the only target rather than the whole
        canvas becoming one."""
        overlay = FieldOfViewOverlay()
        _canvas().add_overlay(overlay)

        assert not overlay.covers((0.0, 0.0), 0.0, 0.0)


# ── the beam tab ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def microscope():
    """The same simulated Arctis `test_overview_widget.py` uses."""
    import fibsem.config as fibsem_config

    path = os.path.join(
        os.path.dirname(fibsem_config.__file__),
        "config",
        "sim-arctis-configuration.yaml",
    )
    scope, _ = utils.setup_session(manufacturer="Demo", config_path=path)
    return scope


@pytest.fixture
def widget(microscope):
    w = FibsemOverviewWidget(microscope)
    w.resize(900, 700)
    yield w
    w.close()


def _at(base, dx=0.0, dy=0.0, name=None):
    p = FibsemStagePosition(x=base.x + dx, y=base.y + dy, z=base.z, r=base.r, t=base.t)
    p.name = name
    return p


def _tile(microscope, position, shape=(64, 64), hfw=100e-6):
    image = FibsemImage.generate_blank_image(resolution=(shape[1], shape[0]), hfw=hfw)
    image.data = (np.random.default_rng(0).random(shape) * 255).astype(np.uint8)
    state = microscope.get_microscope_state(beam_type=BeamType.ELECTRON)
    state.stage_position = position
    image.metadata.image_settings = ImageSettings(hfw=hfw, beam_type=BeamType.ELECTRON)
    image.metadata.microscope_state = state
    image.metadata.system_info = microscope.system.info
    image.metadata.hardware_geometry = microscope.hardware_geometry()
    return image


def _loaded(widget, microscope, *marks):
    base = microscope.get_stage_position()
    widget.place_image(_tile(microscope, _at(base)))
    widget.set_positions([_at(base, dx, dy, name) for name, dx, dy in marks])
    return base


def _screen_distance(widget, a, b):
    """Distance in screen pixels between two canvas points."""
    transform = widget.canvas._ax.transData
    pa, pb = transform.transform(a), transform.transform(b)
    return ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5


def _frame_canvas_units(widget, span):
    """Frame *span* canvas units across, and settle."""
    ax = widget.canvas._ax
    ax.set_xlim(-span / 2, span / 2)
    ax.set_ylim(span / 2, -span / 2)  # inverted, origin upper
    _app.processEvents()


class TestTheBeamTabBoxesItsPositions:
    def test_the_box_is_a_hundred_microns_at_the_standard_aspect(
        self, widget, microscope
    ):
        """Fixed rather than the current imaging settings: the box says how much sample
        a lamella occupies, and one that resized itself when somebody changed the HFW
        would look like the positions had moved."""
        _loaded(widget, microscope, ("Lamella-01", 0.0, 0.0))

        (box,) = _boxes(widget.position_overlay)
        reference = widget.canvas.reference_pixel_size
        assert box.get_width() * reference == pytest.approx(100e-6)
        assert box.get_height() * reference == pytest.approx(100e-6 * 1024 / 1536)

    def test_flagged_and_selected_positions_are_boxed_too(self, widget, microscope):
        """Three overlays only because `PointsOverlay` paints every point the same
        colour -- they are the same kind of thing and must be drawn as one."""
        base = _loaded(
            widget, microscope, ("Lamella-01", 0.0, 0.0), ("Lamella-02", 400e-6, 0.0)
        )
        widget.set_positions(
            [_at(base, 0.0, 0.0, "Lamella-01"), _at(base, 400e-6, 0.0, "Lamella-02")],
            flagged=["Lamella-02"],
        )
        widget.set_selected_position("Lamella-01")
        _app.processEvents()

        assert len(_boxes(widget.flagged_position_overlay)) == 1
        assert len(_boxes(widget.selected_position_overlay)) == 1

    def test_the_current_stage_position_is_not_boxed(self, widget, microscope):
        """Where you *are*, not something you are sizing up -- and boxing it would
        double every lamella's box the moment you drove to one."""
        _loaded(widget, microscope, ("Lamella-01", 0.0, 0.0))

        assert not isinstance(widget.current_position_overlay, FieldOfViewOverlay)
        assert _boxes(widget.current_position_overlay) == []


class TestPickingIsTheUnionOfBoxAndRadius:
    """Each half is tested against a click the *other* half cannot explain."""

    def test_a_click_inside_the_box_selects_it(self, widget, microscope):
        base = _loaded(widget, microscope, ("Lamella-01", 0.0, 0.0))
        _frame_canvas_units(widget, 400)  # zoomed in: the box is far bigger than 12 px
        frame = widget._frame()
        centre = frame.to_canvas(_at(base))
        probe = frame.to_canvas(_at(base, 40e-6, 0.0))  # inside the 100 um box

        assert widget.position_overlay.covers(centre, *probe), "probe missed the box"
        assert _screen_distance(widget, centre, probe) > PICK_RADIUS_PX, (
            "the radius alone would explain this hit"
        )
        assert widget._position_at(*probe) == "Lamella-01"

    def test_the_same_click_misses_once_the_box_is_taken_away(self, widget, microscope):
        """The other half of the test above: it must be the box doing the work."""
        base = _loaded(widget, microscope, ("Lamella-01", 0.0, 0.0))
        _frame_canvas_units(widget, 400)
        frame = widget._frame()
        probe = frame.to_canvas(_at(base, 40e-6, 0.0))
        assert widget._position_at(*probe) == "Lamella-01"

        for overlay in (
            widget.position_overlay,
            widget.flagged_position_overlay,
            widget.selected_position_overlay,
        ):
            overlay.set_extent(None, None)

        assert widget._position_at(*probe) is None

    def test_the_radius_still_picks_where_the_box_is_smaller_than_it(
        self, widget, microscope
    ):
        """The reason this is a union. Zoomed out to a whole grid the 100 um box is a
        couple of pixels across -- picking by box alone would make markers hardest to
        hit in the view that has the most of them.
        """
        base = _loaded(widget, microscope, ("Lamella-01", 0.0, 0.0))
        _frame_canvas_units(widget, 100_000)
        frame = widget._frame()
        centre = frame.to_canvas(_at(base))

        # A probe 8 screen pixels away, in canvas units at this zoom.
        transform = widget.canvas._ax.transData
        per_unit = abs(
            transform.transform((centre[0] + 1.0, centre[1]))[0]
            - transform.transform(centre)[0]
        )
        probe = (centre[0] + 8.0 / per_unit, centre[1])

        assert not widget.position_overlay.covers(centre, *probe), (
            "the box is still the bigger target here; zoom out further"
        )
        assert _screen_distance(widget, centre, probe) < PICK_RADIUS_PX
        assert widget._position_at(*probe) == "Lamella-01"

    def test_the_nearest_crosshair_wins_between_overlapping_boxes(
        self, widget, microscope
    ):
        """Lamellae closer together than the field of view overlap, so a click can be
        inside two boxes. The nearer one is deliberately *first* in the list: keeping a
        hit without keeping its distance leaves the last match winning, which would
        agree with the right answer if the nearest happened to come last.
        """
        base = _loaded(
            widget,
            microscope,
            ("Lamella-02", 30e-6, 0.0),
            ("Lamella-01", 0.0, 0.0),
        )
        _frame_canvas_units(widget, 400)
        frame = widget._frame()
        centre_1 = frame.to_canvas(_at(base))
        centre_2 = frame.to_canvas(_at(base, 30e-6, 0.0))
        probe = frame.to_canvas(_at(base, 25e-6, 0.0))

        overlay = widget.position_overlay
        assert overlay.covers(centre_1, *probe) and overlay.covers(centre_2, *probe), (
            "the probe is not inside both boxes"
        )
        assert widget._position_at(*probe) == "Lamella-02"

    def test_a_click_outside_every_box_and_radius_picks_nothing(
        self, widget, microscope
    ):
        base = _loaded(widget, microscope, ("Lamella-01", 0.0, 0.0))
        frame = widget._frame()

        assert widget._position_at(*frame.to_canvas(_at(base, 900e-6, 900e-6))) is None

    def test_a_hidden_marker_is_not_a_target_through_its_box_either(
        self, widget, microscope
    ):
        """Turned off means gone. The box widens the target, so it would widen this
        hole too: a click on bare mosaic selecting a lamella nothing on screen names."""
        base = _loaded(widget, microscope, ("Lamella-01", 0.0, 0.0))
        _frame_canvas_units(widget, 400)
        frame = widget._frame()
        probe = frame.to_canvas(_at(base, 40e-6, 0.0))
        assert widget._position_at(*probe) == "Lamella-01"

        widget.overlay_controls.set_visible("positions", False)

        assert widget._position_at(*probe) is None


# ── the fluorescence tab ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def fm_widget(qapp):
    from fibsem.ui.fm.overview_app import build_microscope
    from fibsem.ui.fm.widgets.fm_overview_widget import FMOverviewWidget

    microscope = build_microscope()
    microscope.fm.objective.insert()
    w = FMOverviewWidget(microscope)
    w.resize(1200, 800)
    w.show()
    qapp.processEvents()
    return w


def _fm_named(name, x, y):
    position = FibsemStagePosition(x=x, y=y, z=0.0, r=0.0, t=np.deg2rad(-180.0))
    position.name = name
    return position


class TestTheFluorescenceTabBoxesItsPositions:
    def test_the_box_is_one_camera_frame(self, qapp, fm_widget):
        """Not a constant: on the fluorescence side the field of view *is* the camera,
        and it moves with binning."""
        fm_widget.set_positions([_fm_named("Lamella-01", 120e-6, 90e-6)])
        qapp.processEvents()

        projection = fm_widget._projection()
        height, width = projection.shape
        assert fm_widget.position_overlay._extent == pytest.approx(
            (width * projection.pixel_size, height * projection.pixel_size)
        )

        fm_widget.set_positions([])

    def test_the_box_is_drawn_and_the_stage_position_is_not_boxed(
        self, qapp, fm_widget
    ):
        fm_widget.set_positions([_fm_named("Lamella-01", 120e-6, 90e-6)])
        qapp.processEvents()

        assert len(_boxes(fm_widget.position_overlay)) == 1
        assert _boxes(fm_widget.current_position_overlay) == []

        fm_widget.set_positions([])

    def test_a_click_inside_the_box_selects_it(self, qapp, fm_widget):
        fm_widget.set_positions([_fm_named("Lamella-01", 120e-6, 90e-6)])
        qapp.processEvents()
        frame = fm_widget._frame()
        centre = frame.to_canvas(_fm_named("centre", 120e-6, 90e-6))
        half_w, _ = fm_widget.position_overlay.half_extent_canvas()
        probe = (centre[0] + half_w * 0.8, centre[1])

        transform = fm_widget.canvas.canvas._ax.transData
        pa, pb = transform.transform(centre), transform.transform(probe)
        distance = ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5
        assert distance > fm_widget.PICK_RADIUS_PX, (
            "the radius alone would explain this hit"
        )
        assert fm_widget._position_at(*probe) == "Lamella-01"

        fm_widget.set_positions([])
