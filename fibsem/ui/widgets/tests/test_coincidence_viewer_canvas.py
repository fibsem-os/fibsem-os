"""Headless regression tests for the coincidence viewer's two image quadrants (FIB-351).

The viewer is in production and was the only consumer of the legacy
``fibsem/ui/widgets/image_canvas.py``. Retiring that module repointed it at
``fibsem.ui.widgets.canvas`` and deleted its bespoke histogram panel in favour of the
canvas' built-in contrast/gamma control — a naive repoint would have left the widget
with *two* ``mdi:contrast-box`` buttons.

These tests pin the parts that would break silently: exactly one contrast button per
canvas, the rect/arrow overlays, the raw-frame display path (the canvas normalises
internally now, so the widget must not pre-process), and the private canvas members
the widget still reaches into.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python fibsem/ui/widgets/tests/test_coincidence_viewer_canvas.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication, QPushButton

from fibsem.structures import FibsemImage
from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas
from fibsem.ui.widgets.canvas.overlays import RectOverlay, ScanDirectionArrowOverlay
from fibsem.ui.widgets.fluorescence_coincidence_viewer_widget import (
    _FibImageCanvas,
    _FmImageCanvas,
)

_app = QApplication.instance() or QApplication(sys.argv)

_RESOLUTION = (768, 512)  # (x, y) — non-square, so an axis mix-up shows up
_HFW = 80e-6


def _fib_image() -> FibsemImage:
    return FibsemImage.generate_blank_image(resolution=_RESOLUTION, hfw=_HFW, random=True)


class _FakeFmMetadata:
    """FluorescenceImageMetadata uses pixel_size_x, not FibsemImage's pixel_size.x."""

    def __init__(self, pixel_size_x: float) -> None:
        self.pixel_size_x = pixel_size_x


class _FakeFmImage:
    def __init__(self, data: np.ndarray, pixel_size_x: float = 1.0e-7) -> None:
        self.data = data
        self.metadata = _FakeFmMetadata(pixel_size_x)


def _fm_stack(channels: int = 2, z: int = 3, h: int = 512, w: int = 768) -> _FakeFmImage:
    rng = np.random.default_rng(0)
    return _FakeFmImage((rng.random((channels, z, h, w)) * 4095).astype(np.uint16))


# --- the regression this issue exists for --------------------------------------


def test_quadrants_add_no_toolbar_buttons_of_their_own():
    """The canvas supplies contrast/gamma itself, so the widget must add nothing.

    Counting buttons rather than matching tooltips on purpose: the old bespoke button
    was labelled "Histogram Controls" while carrying the same mdi:contrast-box icon,
    so a tooltip filter would not have seen the duplicate at all.
    """
    expected = len(FibsemImageCanvas().findChildren(QPushButton))
    for name, widget in (("FIB", _FibImageCanvas()), ("FM", _FmImageCanvas())):
        buttons = widget.canvas.findChildren(QPushButton)
        assert len(buttons) == expected, (
            f"{name} canvas has {len(buttons)} toolbar buttons, expected the bare "
            f"canvas' {expected}: {[b.toolTip() for b in buttons]}"
        )
        assert widget.canvas.btn_contrast in buttons, f"{name}: lost the contrast button"
        # and it still opens the built-in popover. isHidden(), not isVisible() — the
        # latter is False for any child of a parent that was never shown.
        widget.canvas.btn_contrast.setChecked(True)
        widget.canvas.toggle_contrast()
        assert not widget.canvas._contrast.isHidden(), f"{name}: popover did not open"


def test_widget_hands_the_canvas_raw_data():
    """The canvas normalises internally. If the widget pre-processed to float [0,1]
    as it used to, contrast would be applied twice."""
    widget = _FibImageCanvas()
    image = _fib_image()
    widget.set_image(image)
    base = widget.canvas._display_base
    assert base is not None
    assert base.dtype == image.filtered_data.dtype, (
        f"canvas got {base.dtype}, expected the raw {image.filtered_data.dtype}"
    )
    assert base.max() > 1.0, "looks pre-normalised to [0,1] — double contrast"


def test_contrast_changes_the_displayed_array():
    widget = _FibImageCanvas()
    widget.set_image(_fib_image())
    before = widget.canvas._ax.get_images()[0].get_array().copy()
    widget.canvas._contrast.sld_gamma.setValue(2.5)
    after = widget.canvas._ax.get_images()[0].get_array()
    assert not np.allclose(before, after), "gamma had no effect on the display"
    widget.canvas._contrast.reset()
    assert widget.canvas._contrast.is_default()


# --- canvas wiring the widget depends on ---------------------------------------


def test_both_quadrants_build_on_the_new_canvas():
    for widget in (_FibImageCanvas(), _FmImageCanvas()):
        assert isinstance(widget.canvas, FibsemImageCanvas)
        assert widget.canvas.__class__.__module__ == "fibsem.ui.widgets.canvas.image_canvas"
        assert isinstance(widget.rect_overlay, RectOverlay)


def test_fib_overlays_and_private_reach_ins():
    widget = _FibImageCanvas()
    widget.set_image(_fib_image())

    assert isinstance(widget.arrow_overlay, ScanDirectionArrowOverlay)
    widget.set_scan_direction(200.0, 150.0, 80.0, "TopToBottom")
    widget.set_scan_direction(0.0, 0.0, 0.0, "")  # hides without raising

    # _ax + get_images() back the info panel's set_clim path
    assert widget.canvas._ax.get_images(), "no image artist on the axes"
    widget.canvas._refresh_scalebar()

    widget.rect_overlay.set_rect(10, 10, 100, 80)
    rect = widget.rect_overlay.get_rect()
    assert (rect["x0"], rect["y0"], rect["width"], rect["height"]) == (10, 10, 100, 80)


def test_fib_rect_is_drag_only_and_fm_rect_resizable():
    """The two quadrants deliberately differ; a shared default would silently change one."""
    assert _FibImageCanvas().rect_overlay._resizable is False
    assert _FmImageCanvas().rect_overlay._resizable is True


# --- FM z-stack path ------------------------------------------------------------


def test_fm_set_image_renders_and_sets_the_scalebar():
    widget = _FmImageCanvas()
    widget.set_image(_fm_stack())
    assert widget.canvas._ax.get_images(), "FM canvas rendered nothing"
    assert widget.canvas._img_w == 768 and widget.canvas._img_h == 512
    # pixel_size comes from metadata.pixel_size_x, which the FibsemImage path can't source
    assert widget.canvas._pixel_size == 1.0e-7
    assert "FM  z=0/2" in widget.canvas._ax.get_title()


def test_fm_same_shape_update_is_a_data_swap_not_an_axes_reset():
    """Live FM updates take the fast update_display path so the rectangle overlay is
    left alone. Asserting artist *identity*, not the rect value: RectOverlay._rebuild
    restores the saved rect after a reset too, so comparing get_rect() passes either
    way and would not catch a regression here.
    """
    widget = _FmImageCanvas()
    widget.set_image(_fm_stack())
    widget.rect_overlay.set_rect(20, 30, 120, 90)
    before_rect = widget.rect_overlay.get_rect()
    artist = widget.canvas._ax.get_images()[0]

    widget.set_image(_fm_stack())  # same shape → update_display, no cla()

    assert widget.canvas._ax.get_images()[0] is artist, (
        "axes were rebuilt on a same-shape FM update — overlays got reset"
    )
    assert widget.rect_overlay.get_rect() == before_rect


def test_fm_resolution_change_does_rebuild_the_axes():
    """The counterpart: a shape change must take the set_array path."""
    widget = _FmImageCanvas()
    widget.set_image(_fm_stack())
    artist = widget.canvas._ax.get_images()[0]
    widget.set_image(_fm_stack(h=256, w=384))
    images = widget.canvas._ax.get_images()
    assert images, "resolution change left the axes with no image artist"
    assert images[0] is not artist, "axes were not rebuilt for the new resolution"


def test_fm_resolution_change_resets_overlays():
    widget = _FmImageCanvas()
    widget.set_image(_fm_stack())
    widget.set_image(_fm_stack(h=256, w=384))
    assert widget.canvas._img_w == 384 and widget.canvas._img_h == 256


def test_fm_timelapse_frame_displays_without_disturbing_state():
    widget = _FmImageCanvas()
    widget.set_image(_fm_stack())
    widget.rect_overlay.set_rect(5, 5, 50, 40)
    before = widget.rect_overlay.get_rect()

    rng = np.random.default_rng(1)
    frame = (rng.random((512, 768)) * 4095).astype(np.uint16)
    widget.display_timelapse_frame(frame, idx=3, total=10, ts=0.0)

    assert widget.rect_overlay.get_rect() == before
    assert "(3/9)" in widget.frame_label.text()


def test_scrubbing_to_a_dim_timelapse_frame_rescales():
    """The regression this file exists to prevent recurring.

    A coincidence milling run is exactly an FM intensity drop over time, so a stored
    timelapse frame routinely has a far smaller range than the live one. The old widget
    normalised every frame to [0,1] with clim (0,1); handing raw frames to the canvas
    only preserves that if update_display rescales, otherwise a dim frame is drawn
    against the bright live frame's clim and renders near-black.
    """
    widget = _FmImageCanvas()
    rng = np.random.default_rng(0)
    widget.set_image(_FakeFmImage((rng.random((1, 1, 512, 512)) * 4000).astype(np.uint16)))

    dim = (rng.random((512, 512)) * 400).astype(np.float32)  # a 10x drop
    widget.display_timelapse_frame(dim, idx=1, total=5, ts=0.0)

    artist = widget.canvas._ax.get_images()[0]
    lo, hi = artist.get_clim()
    shown = np.asarray(artist.get_array())
    used = (shown.max() - shown.min()) / max(1e-9, hi - lo)
    assert used > 0.95, (
        f"dim frame uses only {used * 100:.0f}% of the display range "
        f"(clim={lo:.0f}..{hi:.0f}, data={shown.min():.0f}..{shown.max():.0f}) — "
        "it is being drawn against the previous frame's range"
    )


def test_rgb_updates_are_not_rescaled():
    """The rescale must stay scoped to grayscale — imshow doesn't scale RGB either,
    and the FM composite canvas swaps RGB through this same path."""
    canvas = FibsemImageCanvas()
    rng = np.random.default_rng(0)
    canvas.set_array((rng.random((64, 64, 3)) * 255).astype(np.uint8))
    before = canvas._ax.get_images()[0].get_clim()
    canvas.update_display((rng.random((64, 64, 3)) * 40).astype(np.uint8))
    assert canvas._ax.get_images()[0].get_clim() == before


def test_clear_resets_both_quadrants():
    fib = _FibImageCanvas()
    fib.set_image(_fib_image())
    fib.clear()
    assert fib.canvas._display_base is None

    fm = _FmImageCanvas()
    fm.set_image(_fm_stack())
    fm.clear()
    assert fm._image is None and fm._data is None and fm._img_shape is None
    assert fm.canvas._display_base is None


def _main() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001 - standalone runner
            failures += 1
            print(f"FAIL  {name}: {exc}")
    print(f"\n{'FAILED' if failures else 'OK'} — {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())
