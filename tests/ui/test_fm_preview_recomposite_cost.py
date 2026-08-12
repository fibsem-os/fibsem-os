"""One composite per preview update, and held planes reduced once (FIB-589 follow-up).

Found by measurement after the FM overview preview went sluggish on hardware. Two things
multiply, and neither is visible from reading either site on its own:

* `set_channel` recomposites on every call, so setting C channels in a loop does C
  composites and discards C-1 of them;
* a recomposite re-renders every *other* held image, reducing all of its planes — and
  those are held at acquisition resolution, so a stitched 10x10 mosaic is 10240px square
  per channel.

Together that is C x (C + held x C) reductions per update. It cost nothing while the
reduction was a strided view; box-averaging made each one real (FIB-589), and the live
preview runs it once a tile for the length of a run. Measured with one 10x10 overview
also placed: 148 ms an update against 37 ms batched, and ~0 with the held planes cached.

What is pinned is the call count, not the timing — a timing assertion would be flaky on
CI and would not say which of the two regressed.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import numpy as np
import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from PyQt5.QtWidgets import QApplication

from fibsem.ui.widgets.canvas.fm_canvas import FMRealSpaceCanvasWidget

_app = QApplication.instance() or QApplication(sys.argv)

CHANNELS = [("red", "red"), ("green", "green"), ("cyan", "cyan"), ("magenta", "magenta")]


def plane(side: int = 900, seed: int = 0) -> np.ndarray:
    """Big enough that the reduction does not early-return."""
    return (np.random.default_rng(seed).random((side, side)) * 255).astype(np.uint8)


@pytest.fixture
def widget(qtbot=None):
    w = FMRealSpaceCanvasWidget()
    w.set_pixel_size(1e-7)
    yield w
    w.deleteLater()


@pytest.fixture
def reductions(widget, monkeypatch):
    """Every plane the widget reduces, counted."""
    calls = []
    original = type(widget)._reduce

    def counting(self, p):
        calls.append(p.shape)
        return original(self, p)

    monkeypatch.setattr(type(widget), "_reduce", counting)
    return calls


class TestOneCompositePerUpdate:
    def test_setting_channels_together_costs_one_composite(self, widget, reductions):
        """C reductions, not C squared. The loop threw C-1 blends away."""
        items = [(name, plane(seed=i), color) for i, (name, color) in enumerate(CHANNELS)]

        widget.set_channels(items)

        assert len(reductions) == len(CHANNELS), (
            f"{len(reductions)} reductions for {len(CHANNELS)} channels — a batched "
            f"update should reduce each layer once, so this is compositing more than once"
        )

    def test_setting_them_one_at_a_time_is_what_it_replaced(self, widget, reductions):
        """The premise, so the number above is read against something."""
        for i, (name, color) in enumerate(CHANNELS):
            widget.set_channel(name, plane(seed=i), color)

        n = len(CHANNELS)
        assert len(reductions) == n * (n + 1) // 2, (
            "`set_channel` is expected to recomposite per call — if this changed, the "
            "batching in `_show_preview` may no longer be buying anything"
        )

    def test_the_picture_is_the_same_either_way(self, widget):
        """Batching is an optimisation; it must not change what is drawn."""
        items = [(name, plane(seed=i), color) for i, (name, color) in enumerate(CHANNELS)]

        widget.set_channels(items)
        batched = widget.canvas._placed[widget._composite_key].artist.get_array().copy()

        widget.clear_overviews()
        for name, data, color in items:
            widget.set_channel(name, data, color)
        looped = widget.canvas._placed[widget._composite_key].artist.get_array()

        assert np.array_equal(np.asarray(batched), np.asarray(looped))


class TestHeldPlanesAreReducedOnce:
    def test_re_rendering_a_held_overview_does_not_reduce_it_again(self, widget, reductions):
        """The dominant cost: held planes are at acquisition resolution and never change.

        Place one overview, then place a second somewhere else — which re-renders the
        first through `_restyle_others`. The first one's planes must come from the cache.
        """
        widget.set_composite_key("first")
        widget.set_channels([(n, plane(seed=i), c) for i, (n, c) in enumerate(CHANNELS)])
        reductions.clear()

        widget.set_composite_key("second")
        widget.set_placement((1e-3, 0.0))
        widget.set_channels([(n, plane(seed=i + 9), c) for i, (n, c) in enumerate(CHANNELS)])

        assert len(reductions) == len(CHANNELS), (
            f"{len(reductions)} reductions placing a second overview — only the new "
            f"one's {len(CHANNELS)} planes should be reduced. The first overview is "
            f"held and unchanged, so re-reducing it recomputes the same answer"
        )

    def test_the_cache_is_dropped_with_the_overview(self, widget):
        """It holds full-size arrays, so a leak here is a memory leak."""
        widget.set_composite_key("first")
        widget.set_channels([(n, plane(seed=i), c) for i, (n, c) in enumerate(CHANNELS)])
        assert widget._held_reduced.get("first")

        widget.forget_overview("first")

        assert "first" not in widget._held_reduced

    def test_clearing_drops_every_cached_reduction(self, widget):
        widget.set_composite_key("first")
        widget.set_channels([(n, plane(seed=i), c) for i, (n, c) in enumerate(CHANNELS)])

        widget.clear_overviews()

        assert widget._held_reduced == {}


class TestThePreviewUsesTheBatchedPath:
    """Structural. The widget's own tests cover the canvas; this pins the caller, which
    is where the cost was actually being paid."""

    def test_show_preview_does_not_set_channels_one_at_a_time(self):
        import ast
        import inspect
        import textwrap

        from fibsem.ui.fm.widgets.fm_overview_widget import FMOverviewWidget

        # dedent: a method's source is indented, which `ast.parse` rejects outright
        source = textwrap.dedent(inspect.getsource(FMOverviewWidget._show_preview))
        tree = ast.parse(source)
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }

        assert "set_channels" in called, "_show_preview no longer batches its channels"
        assert "set_channel" not in called, (
            "_show_preview calls `set_channel`, which recomposites per channel — that is "
            "C composites an update, and it runs once a tile for a whole overview run"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
