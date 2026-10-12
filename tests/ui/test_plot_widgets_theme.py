"""The app's matplotlib plot widgets wear the shared dark plot theme (FIB-1291).

Each clears and rebuilds its axes on every redraw, so the theme is checked after
data has gone through, not just at construction.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PyQt5")

from matplotlib.colors import to_hex  # noqa: E402

from fibsem.theme.plots import DARK  # noqa: E402
from fibsem.ui.fm.widgets.histogram_widget import HistogramWidget  # noqa: E402
from fibsem.ui.fm.widgets.line_plot_widget import LinePlotWidget  # noqa: E402
from fibsem.ui.widgets.profile_line import _ProfilePlotWidget  # noqa: E402


@pytest.fixture
def shown(qapp):
    widgets = []

    def show(widget):
        widget.resize(400, 300)
        widget.show()
        widgets.append(widget)
        return widget

    yield show
    for w in widgets:
        w.deleteLater()


def _themed(ax, figure):
    assert to_hex(figure.get_facecolor()) == DARK.background
    assert to_hex(ax.get_facecolor()) == DARK.background
    assert to_hex(ax.title.get_color()) == DARK.text_strong
    assert {to_hex(s.get_edgecolor()) for s in ax.spines.values()} == {DARK.grid}


def test_the_histogram_is_themed_after_an_image(shown):
    h = shown(HistogramWidget())
    h._update_histogram_internal(np.random.default_rng(0).integers(0, 255, (64, 64)))
    _themed(h.ax, h.figure)
    assert to_hex(h.histogram_patches[0].get_facecolor()) == DARK.series[0]


def test_the_line_plot_is_themed_after_data(shown):
    p = shown(LinePlotWidget())
    p.update_chart(np.linspace(0, 1, 20), title="intensity")
    _themed(p.ax, p.figure)
    assert to_hex(p.line_artist.get_color()) == DARK.series[0]
    assert {to_hex(t.get_color()) for t in p.legend.get_texts()} == {DARK.text}


def test_the_profile_plot_is_themed_and_keeps_the_drawn_lines_orange(shown):
    w = shown(_ProfilePlotWidget())
    w.update_profile(np.arange(10.0), np.arange(10.0) * 20)
    assert to_hex(w._ax.get_facecolor()) == DARK.background
    (line,) = w._ax.get_lines()
    assert to_hex(line.get_color()) == "#ff8c00", "the colour of the line on the image"
