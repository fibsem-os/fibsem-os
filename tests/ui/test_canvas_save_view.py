"""Saving what a canvas is showing.

Nothing under `fibsem/ui/widgets/canvas/` could export anything before `save_view`, so
the only route to a picture of an overview was a separate dialog that re-rendered the
data through a different renderer.

The behaviour worth pinning is the non-obvious half. A `FigureCanvasQTAgg` keeps its
figure the size of the *widget*, so saving the figure as it stands writes a page shaped
like the window rather than like the picture -- and `bbox_inches="tight"` cannot rescue
it, because a bounding box is a rectangle and the empty band is inside it. That is the
same trap the overview export dialog fell into; these tests exist so the canvas does not
fall into it again.
"""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas  # noqa: E402

# A window much taller than it is wide, holding a much wider than tall image. Saving
# the figure directly would produce the window's shape; save_view must produce the
# image's.
_TALL_WIDGET = (400, 900)
_WIDE_IMAGE = (200, 800)  # (h, w) -> 4:1


@pytest.fixture
def canvas(qapp):
    c = FibsemImageCanvas()
    c.resize(*_TALL_WIDGET)
    c.set_array(np.zeros(_WIDE_IMAGE, dtype=np.uint8))
    c.show()
    qapp.processEvents()
    yield c
    c.close()
    c.deleteLater()


def _saved_aspect(path) -> float:
    from PIL import Image

    with Image.open(path) as im:
        return im.width / im.height


class TestTheSavedPageIsTheView:
    def test_the_page_matches_the_image_not_the_window(self, canvas, tmp_path):
        out = tmp_path / "view.png"
        canvas.save_view(str(out), dpi=50)

        assert out.exists()
        # 4:1 image in a 0.44:1 window. Anything near the window's aspect means the
        # figure was saved at the widget's size.
        assert _saved_aspect(out) == pytest.approx(4.0, rel=0.05)

    def test_a_zoomed_view_exports_what_it_shows(self, canvas, tmp_path):
        """Zooming to a square region should give a square page, not the image's."""
        canvas._ax.set_xlim(100, 300)
        canvas._ax.set_ylim(200, 0)  # y stays inverted, as the canvas keeps it

        out = tmp_path / "zoomed.png"
        canvas.save_view(str(out), dpi=50)
        assert _saved_aspect(out) == pytest.approx(1.0, rel=0.05)

    def test_width_and_dpi_decide_the_pixel_size(self, canvas, tmp_path):
        from PIL import Image

        out = tmp_path / "sized.png"
        canvas.save_view(str(out), dpi=100, width_in=6.0)
        with Image.open(out) as im:
            assert im.width == pytest.approx(600, abs=2)


class TestItLeavesTheCanvasAlone:
    def test_the_figure_size_is_restored(self, canvas, tmp_path):
        before = tuple(canvas._fig.get_size_inches())
        canvas.save_view(str(tmp_path / "x.png"), dpi=50)
        assert tuple(canvas._fig.get_size_inches()) == pytest.approx(before)

    def test_the_background_is_restored(self, canvas, tmp_path):
        before_fig = canvas._fig.get_facecolor()
        before_ax = canvas._ax.get_facecolor()
        canvas.save_view(str(tmp_path / "x.png"), dpi=50, facecolor="white")
        assert canvas._fig.get_facecolor() == before_fig
        assert canvas._ax.get_facecolor() == before_ax

    def test_a_failed_save_still_restores(self, canvas, tmp_path):
        """The restore is in a finally, so a bad path does not leave a stretched canvas."""
        before = tuple(canvas._fig.get_size_inches())
        with pytest.raises(Exception):
            canvas.save_view(str(tmp_path / "no-such-dir" / "x.png"), dpi=50)
        assert tuple(canvas._fig.get_size_inches()) == pytest.approx(before)


class TestBackground:
    """`facecolor` paints the ground the content sits on.

    Only visible where there *is* ground: the axes fills the whole figure and the image
    fills the axes, so a view framed tightly on the image shows none of it. These zoom
    out first so there is some.
    """

    @staticmethod
    def _zoom_out(canvas):
        canvas._ax.set_xlim(-800, 1600)
        canvas._ax.set_ylim(800, -600)  # y stays inverted

    def test_white_is_actually_white(self, canvas, tmp_path):
        """The canvas draws on near-black; a document wants the other thing."""
        from PIL import Image

        self._zoom_out(canvas)
        out = tmp_path / "white.png"
        canvas.save_view(str(out), dpi=50, facecolor="white")
        with Image.open(out) as im:
            assert im.convert("RGB").getpixel((2, 2)) == (255, 255, 255)

    def test_the_canvas_background_is_the_default(self, canvas, tmp_path):
        from PIL import Image

        self._zoom_out(canvas)
        out = tmp_path / "dark.png"
        canvas.save_view(str(out), dpi=50)
        with Image.open(out) as im:
            r, g, b = im.convert("RGB").getpixel((2, 2))
        assert (r, g, b) != (255, 255, 255)
