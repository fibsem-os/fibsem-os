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

CHANNELS = [
    ("red", "red"),
    ("green", "green"),
    ("cyan", "cyan"),
    ("magenta", "magenta"),
]


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
        items = [
            (name, plane(seed=i), color) for i, (name, color) in enumerate(CHANNELS)
        ]

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
        items = [
            (name, plane(seed=i), color) for i, (name, color) in enumerate(CHANNELS)
        ]

        widget.set_channels(items)
        batched = widget.canvas._placed[widget._composite_key].artist.get_array().copy()

        widget.clear_overviews()
        for name, data, color in items:
            widget.set_channel(name, data, color)
        looped = widget.canvas._placed[widget._composite_key].artist.get_array()

        assert np.array_equal(np.asarray(batched), np.asarray(looped))


class TestHeldOverviewsAreNotRedrawnWholesale:
    """What replaced the reduced-plane cache.

    Held planes used to be reduced once and kept, because `_restyle_others` re-blended
    every held image at the display cap on every layer change. Both are gone: each placed
    image now carries a detail source, and the canvas asks it only for the part of itself
    on screen -- so there is nothing to cache the whole reduction *for*. See
    `test_fm_detail_source.py` for what the source itself guarantees.
    """

    def test_placing_a_second_overview_does_not_re_reduce_the_first(
        self, widget, reductions
    ):
        """The dominant cost: held planes are at acquisition resolution and never change.

        `_reduce` is the whole-image path, so counting it here counts exactly the work
        the source was supposed to remove. The first overview is redrawn -- it has to be,
        the new one may overlap it -- but through `_patch`, which reduces the visible
        region rather than the mosaic.
        """
        widget.set_composite_key("first")
        widget.set_channels(
            [(n, plane(seed=i), c) for i, (n, c) in enumerate(CHANNELS)]
        )
        reductions.clear()

        widget.set_composite_key("second")
        widget.set_placement((1e-3, 0.0))
        widget.set_channels(
            [(n, plane(seed=i + 9), c) for i, (n, c) in enumerate(CHANNELS)]
        )

        assert len(reductions) == len(CHANNELS), (
            f"{len(reductions)} whole-image reductions placing a second overview — only "
            f"the new one's {len(CHANNELS)} planes should take that path"
        )

    def test_a_layer_edit_reduces_nothing_at_all(self, widget, reductions):
        """No pixels changed, so there is nothing to blend up front -- the canvas asks
        each source for what it can show. This is the path a slider drag takes, once per
        mouse move."""
        widget.set_channels(
            [(n, plane(seed=i), c) for i, (n, c) in enumerate(CHANNELS)]
        )
        reductions.clear()

        widget.layers[0].opacity = 0.5
        widget._panel.changed.emit()

        assert reductions == [], (
            "a layer edit re-reduced whole planes; it should only refresh detail"
        )


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
