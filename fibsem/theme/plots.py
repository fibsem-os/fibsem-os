"""One look for every matplotlib figure fibsem-os draws.

``DARK`` is the app: the image canvas's background and the app's text colours,
so a figure an alignment or an autofocus saves reads as part of the panel that
shows it. ``LIGHT`` is for paper: the PDF report and anything exported to be
printed, where a dark page wastes ink and reads badly.

A figure takes a theme by being built and saved inside ``plot_theme``::

    with plot_theme(DARK) as theme:
        fig = Figure()
        ax = fig.subplots()
        ax.axhline(cy, color=theme.crosshair)
        ax.text(x, y, f"score {s:.2f}", color=theme.score(s), bbox=theme.label_box())
        fig.savefig(path)

matplotlib reads its defaults when an artist is made, so the figure has to be
built inside the block; saving inside it too keeps the saved background the
theme's. A Qt widget that clears and redraws its axes on every update calls
``style_axes`` (and ``style_legend``) after each clear instead.

A colour that means something -- a score that is good, the image centre --
comes from the theme by name, never as a literal, so it stays readable in both
themes.

Qt-free, like ``fibsem.theme.tokens``: the figures are saved by code that runs
without the ``[ui]`` extra.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Tuple

from fibsem.theme.tokens import (
    BORDER_COLOR,
    CANVAS_BG,
    CURRENT_POSITION_COLOUR,
    ERROR_COLOR,
    GRAY_CANVAS_COLOR,
    GRAY_WHITE_COLOR,
    NEUTRAL_300,
    NEUTRAL_650,
    NEUTRAL_850,
    OK_COLOR,
    TEXT_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
    WARN_COLOR,
)


@dataclass(frozen=True)
class PlotTheme:
    """The colours and defaults a figure is drawn with."""

    name: str
    background: str
    text: str
    text_strong: str  # titles
    text_muted: str  # ticks, notes
    grid: str  # axes edges and grid lines: recessive on purpose
    # Line colours in a fixed order, so the first series is always the first
    # colour. The same six hues in both themes, each stepped for its
    # background; checked for contrast and colour-blind separation (FIB-1288).
    series: Tuple[str, ...]
    # A score or a verdict. Always drawn with the number or word it colours,
    # never as colour alone.
    good: str
    warn: str
    bad: str
    # The image centre, where the beam is: drawn on image data in both themes.
    crosshair: str

    def score(self, value: float, good: float = 0.7, warn: float = 0.5) -> str:
        """The colour for a score: good at or above ``good``, a warning at or
        above ``warn``, bad below."""
        if value >= good:
            return self.good
        return self.warn if value >= warn else self.bad

    def label_box(self) -> Dict[str, Any]:
        """A chip behind text drawn over an image, as the canvas's ruler labels
        it; a new dict each call, so a caller can change it."""
        return {
            "boxstyle": "round,pad=0.3",
            "facecolor": self.background,
            "alpha": 0.8,
            "edgecolor": "none",
        }

    def rc(self) -> Dict[str, Any]:
        """The matplotlib defaults for this theme."""
        # matplotlib's own, so cycler stays its dependency rather than ours
        from matplotlib.rcsetup import cycler

        return {
            "figure.facecolor": self.background,
            "figure.edgecolor": self.background,
            "savefig.facecolor": self.background,
            "savefig.edgecolor": self.background,
            "axes.facecolor": self.background,
            "axes.edgecolor": self.grid,
            "axes.labelcolor": self.text,
            "axes.titlecolor": self.text_strong,
            "axes.prop_cycle": cycler(color=list(self.series)),
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "text.color": self.text,
            "xtick.color": self.text_muted,
            "ytick.color": self.text_muted,
            "xtick.labelcolor": self.text_muted,
            "ytick.labelcolor": self.text_muted,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "grid.color": self.grid,
            "legend.facecolor": self.background,
            "legend.edgecolor": self.grid,
            "legend.labelcolor": self.text,
            "legend.fontsize": 8,
            "figure.titlesize": 11,
            "figure.titleweight": "bold",
            "image.cmap": "gray",
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
        }


DARK = PlotTheme(
    name="dark",
    background=CANVAS_BG,
    text=TEXT_COLOR,
    text_strong=TEXT_STRONG_COLOR,
    text_muted=TEXT_MUTED_COLOR,
    grid=BORDER_COLOR,
    series=("#3987e5", "#d95926", "#199e70", "#9085e9", "#c98500", "#d55181"),
    good=OK_COLOR,
    warn=WARN_COLOR,
    bad=ERROR_COLOR,
    crosshair=CURRENT_POSITION_COLOUR,
)

# The app's status colours are tuned for a dark background and are too light to
# read on white, so paper has its own, darker steps.
LIGHT = PlotTheme(
    name="light",
    background=GRAY_WHITE_COLOR,
    text=NEUTRAL_850,
    text_strong=GRAY_CANVAS_COLOR,
    text_muted=NEUTRAL_650,
    grid=NEUTRAL_300,
    series=("#2a78d6", "#d4581f", "#128a60", "#4a3aa7", "#a87400", "#c4416f"),
    good="#2e7d32",
    warn="#b26a00",
    bad="#c62828",
    crosshair=CURRENT_POSITION_COLOUR,
)


def style_axes(ax: Any, theme: PlotTheme = DARK) -> None:
    """Colour an axes in ``theme`` after it was made: for a widget that clears and
    redraws its axes on every update, where defaults set by ``plot_theme`` at
    construction would be gone by the next redraw."""
    ax.set_facecolor(theme.background)
    ax.tick_params(colors=theme.text_muted, which="both")
    for spine in ax.spines.values():
        spine.set_color(theme.grid)
    ax.xaxis.label.set_color(theme.text)
    ax.yaxis.label.set_color(theme.text)
    ax.title.set_color(theme.text_strong)


def style_legend(legend: Any, theme: PlotTheme = DARK) -> None:
    """Colour a legend made outside ``plot_theme``; see ``style_axes``."""
    frame = legend.get_frame()
    frame.set_facecolor(theme.background)
    frame.set_edgecolor(theme.grid)
    for text in legend.get_texts():
        text.set_color(theme.text)


@contextmanager
def plot_theme(theme: PlotTheme = DARK) -> Iterator[PlotTheme]:
    """Build and save a figure in ``theme``; yields it, for the colours that
    mean something."""
    from matplotlib import rc_context

    with rc_context(theme.rc()):
        yield theme
