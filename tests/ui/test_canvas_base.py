"""Headless smoke: the FibsemCanvasBase / FibsemImageCanvas seam.

The base owns navigation, overlays, toolbar and chrome but knows nothing about what is
drawn; subclasses answer one question, _content_extent(), and the base fits the view to
whatever rectangle that returns. These pin that contract — the only thing the extraction
changed, since _fit_view previously read the image artist's extent directly.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python fibsem/ui/widgets/tests/test_canvas_base.py
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import numpy as np
import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from PyQt5.QtCore import QSize
from PyQt5.QtGui import QResizeEvent
from PyQt5.QtWidgets import QApplication

from fibsem.ui.widgets.canvas.canvas_base import FibsemCanvasBase
from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas

_app = QApplication.instance() or QApplication(sys.argv)


def _img(h, w):
    return np.zeros((h, w), dtype=np.uint8)


def _resize_event(w, h):
    """Offscreen, `resize()` does not deliver one on its own."""
    return QResizeEvent(QSize(w, h), QSize(w, h))


def test_image_canvas_is_a_base_canvas():
    assert issubclass(FibsemImageCanvas, FibsemCanvasBase)


def test_bare_base_has_no_content():
    """The base draws nothing of its own, so it reports an empty extent."""
    assert FibsemCanvasBase()._content_extent() is None


def test_empty_canvas_fit_is_a_no_op():
    """reset_view() on a canvas with no content must not raise or move the axes."""
    c = FibsemCanvasBase()
    before = (c._ax.get_xlim(), c._ax.get_ylim())
    c.reset_view()
    assert (c._ax.get_xlim(), c._ax.get_ylim()) == before


def test_image_canvas_reports_the_image_extent():
    c = FibsemImageCanvas()
    assert c._content_extent() is None  # nothing set yet
    c.set_array(_img(32, 64))
    # matplotlib's extent for an image drawn at the origin, as set_array supplies it
    assert tuple(c._content_extent()) == (-0.5, 63.5, 31.5, -0.5)


def test_fit_view_frames_the_content_extent():
    """Routing through the hook must frame exactly what reading the artist did."""
    c = FibsemImageCanvas()
    c.set_array(_img(32, 64))
    c.reset_view()
    assert c._ax.get_xlim() == (-0.5, 63.5)
    assert c._ax.get_ylim() == (31.5, -0.5)  # y inverted (origin upper)


def test_view_margin_expands_the_content_extent():
    c = FibsemImageCanvas()
    c.set_array(_img(32, 64))
    c.set_view_margin(0.5)  # half the span of empty space on each side
    x0, x1 = c._ax.get_xlim()
    assert (x0, x1) == (-32.5, 95.5)  # 64-wide extent grown by 32 per side


def test_a_subclass_extent_drives_the_fit():
    """The seam is usable by any subclass, not just the image canvas — this is what
    the real-space canvas will rely on to frame images placed in stage space."""

    class _FakeContentCanvas(FibsemCanvasBase):
        def _content_extent(self):
            return (100.0, 300.0, 250.0, 50.0)

    c = _FakeContentCanvas()
    c.reset_view()
    assert c._ax.get_xlim() == (100.0, 300.0)
    assert c._ax.get_ylim() == (250.0, 50.0)


# ── chrome toggles keep their buttons honest ──────────────────────────────


def test_setting_crosshair_visibility_syncs_its_button():
    """The setter, not only the toggle, has to sync — a widget picking its own
    default at construction would otherwise show a checked button whose tooltip
    offers to hide a crosshair that is not drawn."""
    c = FibsemImageCanvas()
    c.set_array(_img(32, 32))

    c.set_crosshair_visible(False)

    assert c._crosshair_artists == []
    assert c.btn_toggle_crosshair.isChecked() is False
    assert c.btn_toggle_crosshair.toolTip() == "Show crosshair"


def test_setting_scalebar_visibility_syncs_its_button():
    """`set_scalebar_visible` is new: the shared canvas had only a toggle, which
    cannot express "off" without first knowing what it is currently set to."""
    c = FibsemImageCanvas()
    c.set_array(_img(32, 32), pixel_size=1e-8)

    assert c._scalebar_artist is not None  # drawn, because pixel_size is known

    c.set_scalebar_visible(False)

    assert c._scalebar_artist is None  # actually removed, not just flagged
    assert c.btn_toggle_scalebar.isChecked() is False
    assert c.btn_toggle_scalebar.toolTip() == "Show scalebar"


@pytest.mark.parametrize("noun", ["crosshair", "scalebar"])
def test_the_toolbar_toggle_still_round_trips(noun):
    c = FibsemImageCanvas()
    c.set_array(_img(32, 32), pixel_size=1e-8)
    toggle = getattr(c, f"toggle_{noun}")
    button = getattr(c, f"btn_toggle_{noun}")

    toggle()
    assert button.isChecked() is False and button.toolTip() == f"Show {noun}"
    toggle()
    assert button.isChecked() is True and button.toolTip() == f"Hide {noun}"


class TestTheTopStripHasTwoZones:
    """One row, two zones growing from opposite ends: controls right, status left.

    The strip had five tenants and no rule, and produced three collisions in two days --
    the LIVE chip under the buttons, the flash on the chip, and the flash on the chip
    again on a Retina display. Each was fixed by arithmetic keeping one coordinate system
    clear of another. This removes the arithmetic: both zones are Qt-laid-out, in logical
    pixels, by the same pass, so they cannot reach each other however long the text or
    however narrow the pane (FIB-639).
    """

    @staticmethod
    def _canvas(w=460, h=380):
        c = FibsemImageCanvas()
        dpi = c.figure.dpi
        c.figure.set_size_inches(w / dpi, h / dpi)
        c.resize(w, h)
        c.set_array(_img(512, 512), pixel_size=1e-8)
        c.set_live_badge(True)
        c._reposition_overlay_buttons()
        return c

    @pytest.mark.parametrize("width", [300, 360, 420, 480, 560, 900])
    def test_the_zones_never_reach_each_other(self, width):
        c = self._canvas(w=width)
        c.flash_message("OBJ 6000.0 um  (+1.0 um)")

        status, chip = c._status_label, c._live_badge
        if status.isHidden():
            return  # no room at all is a valid answer; the controls win
        assert status.geometry().right() < chip.geometry().left()

    @pytest.mark.parametrize("width", [300, 460, 900])
    def test_a_very_long_hint_is_elided_rather_than_overrunning(self, width):
        c = self._canvas(w=width)
        c.set_hint("Shift+scroll to focus the objective, " * 10)

        status = c._status_label
        if status.isHidden():
            return
        assert status.geometry().right() < c._live_badge.geometry().left()
        assert status.text() != c._status_full_text  # actually truncated
        assert status.text().endswith("…")

    def test_the_flash_outranks_the_hint_and_gives_it_back(self):
        """One zone, two sources. The flash is a value that just changed; the hint is a
        standing instruction still true underneath it."""
        c = self._canvas()
        c.set_hint("Shift+scroll to focus")
        assert c._status_label.text() == "Shift+scroll to focus"

        c.flash_message("OBJ 6000.0 um")
        assert c._status_label.text() == "OBJ 6000.0 um"

        c._clear_flash()
        assert c._status_label.text() == "Shift+scroll to focus"

    def test_the_two_sources_look_different(self):
        """A standing instruction and a value that is about to vanish should not be the
        same chip."""
        c = self._canvas()
        c.set_hint("Shift+scroll to focus")
        hint_style = c._status_label.styleSheet()
        c.flash_message("OBJ 6000.0 um")

        assert c._status_label.styleSheet() != hint_style

    def test_clearing_the_hint_hides_the_zone(self):
        c = self._canvas()
        c.set_hint("Shift+scroll to focus")
        c.set_hint(None)

        assert c._status_label.isHidden()

    def test_a_new_frame_does_not_drop_the_hint(self):
        """It is a widget, so `cla()` never removes it and `set_image` has nothing to
        re-apply -- the same trade the LIVE chip already made."""
        c = self._canvas()
        c.set_hint("Shift+scroll to focus")

        c.set_array(_img(512, 512), pixel_size=1e-8)  # the next streamed frame

        assert not c._status_label.isHidden()
        assert c._status_label.text() == "Shift+scroll to focus"

    def test_clearing_the_canvas_hides_it(self):
        c = self._canvas()
        c.set_hint("Shift+scroll to focus")

        c.clear()

        assert c._status_label.isHidden()

    def test_no_room_at_all_yields_to_the_controls(self):
        """A pane narrower than its own toolbar. Better to drop the status text than to
        draw it under the buttons."""
        c = self._canvas(w=120)
        c.set_hint("Shift+scroll to focus")

        assert c._status_label.isHidden()

    @pytest.mark.parametrize("width", [360, 480, 900])
    def test_the_title_still_clears_the_row(self, width):
        """The caption stays centred and stays an artist -- it labels the image, not the
        session -- so it keeps the inset the status zone no longer needs."""
        c = self._canvas(w=width)
        c.set_title("z 42 / 120")
        c.draw()

        chip = c._live_badge.geometry()
        bb = c._title_artist.get_window_extent(c.get_renderer())
        top, bottom = c.height() - bb.y1, c.height() - bb.y0
        assert not (bottom > chip.y() and top < chip.y() + chip.height())

    @pytest.mark.parametrize("device_ratio", [1, 2], ids=["standard", "retina"])
    def test_the_title_inset_is_logical_pixels_whatever_the_device_ratio(
        self, device_ratio
    ):
        """The title is the last thing positioned by hand, so it keeps the bug the
        status zone no longer can have: `figure.bbox` is in device pixels, the toolbar
        row is laid out in logical ones, and dividing one by the other halves the inset
        on a Retina screen -- every Mac and no CI runner.
        """
        c = FibsemImageCanvas()
        dpi = c.figure.dpi
        c.resize(460, 380)
        c.figure.set_size_inches(460 * device_ratio / dpi, 380 * device_ratio / dpi)

        inset_logical = (1.0 - c._top_chrome_y()) * c.height()

        assert inset_logical == pytest.approx(30.0)


class TestTheLiveChipClearsTheToolbar:
    """The chip used to be an axes artist at ``transAxes`` (0.988, 0.985) while the
    toolbar buttons are laid out in *widget* pixels. The axes are ``aspect="equal"``
    inside an edge-to-edge figure, so they reach the widget's top edge for any frame at
    least as tall in aspect as its pane -- and the chip landed underneath the buttons.
    A square FM frame in a square pane did it every time (FIB-596).

    It is a child widget in the toolbar's own layout pass now, so the two cannot
    collide whatever the aspect ratio is.
    """

    @staticmethod
    def _live_square_canvas(w=420, h=420, img=(512, 512)):
        c = FibsemImageCanvas()
        c.resize(w, h)
        c.set_array(_img(*img), pixel_size=1e-8)
        c.set_live_badge(True)
        return c

    def test_the_chip_does_not_overlap_any_visible_button(self):
        c = self._live_square_canvas()

        chip = c._live_badge.geometry()
        clashes = [
            b for b in c._overlay_buttons
            if not b.isHidden() and chip.intersects(b.geometry())
        ]

        assert not clashes, f"{len(clashes)} toolbar button(s) under the LIVE chip"

    @pytest.mark.parametrize(
        "pane,image",
        [
            ((420, 420), (512, 512)),  # square frame, square pane -- the reported case
            ((420, 300), (512, 512)),  # square frame, wide pane
            ((300, 420), (512, 768)),  # wide frame in a pane taller than its aspect
            ((900, 200), (512, 512)),  # extreme letterbox
        ],
    )
    def test_no_aspect_ratio_puts_it_under_the_buttons(self, pane, image):
        """The old bug was aspect-dependent, which is why it showed on the FM and not
        the beams. Placement must not depend on the aspect at all now."""
        c = self._live_square_canvas(pane[0], pane[1], image)

        chip = c._live_badge.geometry()
        assert not any(
            chip.intersects(b.geometry())
            for b in c._overlay_buttons
            if not b.isHidden()
        )

    def test_the_buttons_stay_put_when_a_stream_starts(self):
        """The buttons are click targets. The chip appearing must not slide one out
        from under the cursor -- which is why the chip goes on their far side."""
        c = FibsemImageCanvas()
        c.resize(420, 420)
        c.set_array(_img(512, 512), pixel_size=1e-8)
        # Settle the layout at the resized width first: offscreen, `resize` lands
        # lazily, so an unsettled `before` would record the default width and the test
        # would blame the chip for the resize.
        c._reposition_overlay_buttons()
        before = [b.geometry() for b in c._overlay_buttons]

        c.set_live_badge(True)

        assert [b.geometry() for b in c._overlay_buttons] == before

    def test_a_new_frame_does_not_take_the_chip_down(self):
        """It is a widget, so `cla()` never removes it and `set_image` has nothing to
        restore -- unlike the info bar and title, which are re-applied per frame."""
        c = self._live_square_canvas()

        c.set_array(_img(512, 512), pixel_size=1e-8)  # the next streamed frame

        assert c._live_on
        assert c._live_badge is not None and not c._live_badge.isHidden()

    def test_it_hides_without_being_destroyed(self):
        c = self._live_square_canvas()

        c.set_live_badge(False)

        assert not c._live_on
        assert c._live_badge.isHidden()

    def test_asking_to_hide_one_that_was_never_shown_is_a_no_op(self):
        c = FibsemImageCanvas()

        c.set_live_badge(False)

        assert c._live_badge is None  # not built just to be hidden


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all passed")


def test_the_scalebar_is_drawn_at_the_smaller_font():
    """The size is pinned because the constructor is wrapped in a bare except.

    A kwarg matplotlib_scalebar does not accept is swallowed there and the bar simply
    stops appearing, with nothing in the log to say why -- so "it still constructs"
    is as much the point of this test as the size itself (FIB-583).
    """
    c = FibsemImageCanvas()
    c.set_array(_img(32, 32), pixel_size=1e-8)

    assert c._scalebar_artist is not None
    assert c._scalebar_artist.font_properties.get_size() == 8.0

    c.draw()  # a bad font spec raises here rather than at construction
