"""The matplotlib theme every figure shares (FIB-1288).

A figure built and saved inside ``plot_theme`` has the theme's background and
text; the colours it names stay readable on that background; and the module
imports without Qt, because the figures are saved by code that runs without it.
"""

import subprocess
import sys

import numpy as np
import pytest
from matplotlib.colors import to_hex
from matplotlib.figure import Figure

from fibsem.theme import tokens
from fibsem.theme.plots import DARK, LIGHT, plot_theme


def _rgb(hex_colour):
    return np.array([int(hex_colour[i : i + 2], 16) for i in (1, 3, 5)])


def _contrast(a, b):
    """WCAG contrast ratio between two hex colours."""

    def luminance(c):
        c = _rgb(c) / 255
        c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

    high, low = sorted([luminance(a), luminance(b)], reverse=True)
    return (high + 0.05) / (low + 0.05)


@pytest.mark.parametrize("theme", [DARK, LIGHT], ids=lambda t: t.name)
def test_a_figure_built_and_saved_in_the_theme_wears_it(theme, tmp_path):
    with plot_theme(theme):
        fig = Figure(figsize=(2, 2))
        ax = fig.subplots()
        (line,) = ax.plot([0, 1], [0, 1])
        title = ax.set_title("t")
        fig.savefig(tmp_path / "f.png", dpi=50)

    from PIL import Image

    corner = np.asarray(Image.open(tmp_path / "f.png").convert("RGB"))[0, 0]
    assert tuple(corner) == tuple(_rgb(theme.background))
    assert line.get_color() == theme.series[0], "series come in the theme's order"
    assert title.get_color() == theme.text_strong


def test_the_theme_does_not_outlive_its_block():
    before = Figure().get_facecolor()
    with plot_theme(DARK):
        pass
    assert Figure().get_facecolor() == before


def test_dark_is_the_apps_canvas_and_text():
    assert DARK.background == tokens.CANVAS_BG
    assert DARK.text == tokens.TEXT_COLOR
    assert (DARK.good, DARK.warn, DARK.bad) == (
        tokens.OK_COLOR,
        tokens.WARN_COLOR,
        tokens.ERROR_COLOR,
    )
    assert DARK.crosshair == tokens.CURRENT_POSITION_COLOUR


@pytest.mark.parametrize("theme", [DARK, LIGHT], ids=lambda t: t.name)
def test_what_the_theme_draws_stays_readable_on_its_background(theme):
    """Text at 4.5:1, marks at 3:1 (WCAG). The grid is left out on purpose: it
    is meant to recede."""
    for colour in (theme.text, theme.text_strong, theme.text_muted):
        assert _contrast(colour, theme.background) >= 4.5, colour
    for colour in (*theme.series, theme.good, theme.warn, theme.bad):
        assert _contrast(colour, theme.background) >= 3.0, colour


def test_a_score_takes_good_warn_or_bad():
    assert [DARK.score(s) for s in (0.9, 0.7, 0.6, 0.5, 0.2)] == [
        DARK.good,
        DARK.good,
        DARK.warn,
        DARK.warn,
        DARK.bad,
    ]


def test_each_label_box_is_its_own():
    box = DARK.label_box()
    box["alpha"] = 0
    assert DARK.label_box()["alpha"] == 0.8


def test_the_theme_imports_without_qt():
    # A fresh interpreter: anything this session already imported would hide it.
    code = (
        "import sys; import fibsem.theme.plots; "
        "print(','.join(sorted(m for m in sys.modules if m.split('.')[0] in "
        "('PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'qtpy', 'napari') or m == 'fibsem.ui')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "", f"importing the theme loaded {out.stdout}"


def test_style_axes_themes_an_axes_made_outside_the_block():
    """For a widget that clears its axes on every redraw, where defaults set when
    the figure was made are gone by the next one."""
    from fibsem.theme.plots import style_axes, style_legend

    ax = Figure().subplots()
    ax.plot([0, 1], [0, 1], label="a")
    ax.set_title("t")
    style_axes(ax, DARK)
    legend = ax.legend()
    style_legend(legend, DARK)

    assert to_hex(ax.get_facecolor()) == DARK.background
    assert to_hex(ax.title.get_color()) == DARK.text_strong
    assert {to_hex(s.get_edgecolor()) for s in ax.spines.values()} == {DARK.grid}
    assert to_hex(legend.get_frame().get_facecolor()) == DARK.background
    assert {to_hex(t.get_color()) for t in legend.get_texts()} == {DARK.text}
