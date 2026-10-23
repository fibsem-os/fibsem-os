"""The app's matplotlib plot widget wears the shared dark plot theme (FIB-1291).

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
from fibsem.ui.fm.widgets.line_plot_widget import LinePlotWidget  # noqa: E402


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


def test_the_line_plot_is_themed_after_data(shown):
    p = shown(LinePlotWidget())
    for value in np.linspace(0, 1, 20):
        p.append_value(value, title="intensity")
    _themed(p.ax, p.figure)
    assert to_hex(p.line_artist.get_color()) == DARK.series[0]
    assert {to_hex(t.get_color()) for t in p.legend.get_texts()} == {DARK.text}
