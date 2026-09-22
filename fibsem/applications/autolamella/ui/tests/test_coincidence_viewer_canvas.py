"""Headless regression tests for the coincidence viewer's two image quadrants (FIB-351).

The viewer is in production and was the only consumer of the legacy
``fibsem/ui/widgets/image_canvas.py``. Retiring that module repointed it at
``fibsem.ui.widgets.canvas`` and deleted its bespoke histogram panel in favour of the
canvas' built-in contrast/gamma control — a naive repoint would have left the widget
with *two* ``mdi:contrast-box`` buttons.

These tests pin the parts that would break silently: exactly one contrast button per
canvas, the rect/arrow overlays, the raw-frame display path (the canvas normalises
internally now, so the widget must not pre-process), and the public canvas API the
widget drives in place of the private members it used to reach into
(``pixel_size`` and ``set_title``).

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python fibsem/applications/autolamella/ui/tests/test_coincidence_viewer_canvas.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication, QPushButton

from fibsem.applications.autolamella.ui.fluorescence_coincidence_viewer_widget import (
    _FibImageCanvas,
    _FmImageCanvas,
)
from fibsem.structures import FibsemImage
from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas
from fibsem.ui.widgets.canvas.overlays import RectOverlay, ScanDirectionArrowOverlay

_app = QApplication.instance() or QApplication(sys.argv)

_RESOLUTION = (768, 512)  # (x, y) — non-square, so an axis mix-up shows up
_HFW = 80e-6


def _fib_image() -> FibsemImage:
    return FibsemImage.generate_blank_image(
        resolution=_RESOLUTION, hfw=_HFW, random=True
    )


class _FakeFmMetadata:
    """FluorescenceImageMetadata uses pixel_size_x, not FibsemImage's pixel_size.x."""

    def __init__(self, pixel_size_x: float) -> None:
        self.pixel_size_x = pixel_size_x


class _FakeFmImage:
    def __init__(self, data: np.ndarray, pixel_size_x: float = 1.0e-7) -> None:
        self.data = data
        self.metadata = _FakeFmMetadata(pixel_size_x)


def _fm_stack(
    channels: int = 2, z: int = 3, h: int = 512, w: int = 768
) -> _FakeFmImage:
    rng = np.random.default_rng(0)
    return _FakeFmImage((rng.random((channels, z, h, w)) * 4095).astype(np.uint16))


def _coincidence_viewer():
    """Build the whole viewer on a simulated FM microscope.

    The FM configuration is not incidental: ``_connect_signals`` returns immediately when
    ``microscope.fm is None``, so on a plain demo session the viewer constructs with none
    of its signals wired and every interaction test would pass without testing anything.

    The milling config path is redirected at a throwaway file so the viewer always starts
    from its built-in ``DEFAULT_MILLING_TASK_CONFIG``. ``_load_milling_config`` otherwise
    restores whatever this machine last saved to
    ``fibsem/config/coincidence-milling-config.yaml`` — a runtime-written, gitignored file
    inside the package, so a poisoned one is invisible to ``git status`` — which made the
    drag test below depend on ambient state, and let the widget's debounced autosave
    overwrite a developer's real saved config from a test run.

    Left redirected for the rest of the process on purpose: that autosave is a 1s
    ``qdebounced`` timer, so it can fire long after the test that built the widget
    returned, whenever another test spins the event loop.
    """
    import tempfile

    from fibsem import config as cfg
    from fibsem import utils
    from fibsem.applications.autolamella.structures import Experiment
    from fibsem.applications.autolamella.ui.fluorescence_coincidence_viewer_widget import (
        FluorescenceCoincidenceViewerWidget,
    )

    cfg.COINCIDENCE_MILLING_CONFIG_PATH = os.path.join(
        tempfile.mkdtemp(), "coincidence-milling-config.yaml"
    )

    microscope, _ = utils.setup_session(
        config_path=os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")
    )
    assert microscope.fm is not None, "expected an FM on the simulated Arctis"
    return FluorescenceCoincidenceViewerWidget(
        microscope=microscope,
        experiment=Experiment(path=tempfile.mkdtemp(), name="coincidence-canvas-test"),
    )


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
        assert widget.canvas.btn_contrast in buttons, (
            f"{name}: lost the contrast button"
        )
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
        assert (
            widget.canvas.__class__.__module__
            == "fibsem.ui.widgets.canvas.image_canvas"
        )
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
    assert widget.canvas.pixel_size == 1.0e-7
    assert widget.canvas._title_artist in widget.canvas._ax.texts, (
        "z-slice title not drawn"
    )
    assert "FM  z=0/2" in widget.canvas._title_artist.get_text()


def test_fm_title_is_drawn_inside_the_axes_not_as_a_clipped_axes_title():
    """The canvas lays the figure out edge-to-edge (subplots_adjust(top=1)), so a real
    ``Axes.set_title`` renders *above* the figure and is invisible whenever the pane is
    wider in aspect than the image. The title must be an in-axes artist instead."""
    widget = _FmImageCanvas()
    widget.canvas._fig.set_size_inches(10, 2)  # pane far wider than the 768x512 frame
    widget.set_image(_fm_stack())
    widget.canvas.draw()

    assert widget.canvas._ax.get_title() == "", "still using the clipped axes title"
    _, fig_h = widget.canvas.get_width_height()
    bbox = widget.canvas._title_artist.get_window_extent(widget.canvas.get_renderer())
    assert 0 <= bbox.y0 and bbox.y1 <= fig_h, f"title outside the figure: {bbox}"


def test_canvas_title_survives_an_image_change():
    """set_array clears the axes, so the title has to be remembered and re-applied the
    way the hint / info bar / legend are — otherwise it vanishes on the next frame.

    Driven against the bare canvas on purpose: the FM widget re-sets the title after
    every set_array, so going through _FmImageCanvas would pass either way.
    """
    canvas = FibsemImageCanvas()
    canvas.set_array(np.zeros((512, 768)), pixel_size=1.0e-7)
    canvas.set_title("FM  z=0/2")
    assert canvas._title_artist is not None

    canvas.set_array(np.zeros((256, 256)), pixel_size=1.0e-7)  # resolution change
    assert canvas._title_text == "FM  z=0/2", "title text must survive an image change"
    # Membership in _ax.texts, not `is not None`: cla() detaches the artist but leaves
    # the attribute pointing at it, so an identity check passes on a vanished title.
    assert canvas._title_artist in canvas._ax.texts, "title not re-attached after reset"

    canvas.set_title(None)  # explicit clear
    assert canvas._title_artist is None and canvas._title_text is None


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
    widget.set_image(
        _FakeFmImage((rng.random((1, 1, 512, 512)) * 4000).astype(np.uint16))
    )

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


def test_the_viewer_builds_and_its_milling_widget_draws_nothing_itself():
    """The whole widget, not just the two quadrants — nothing else covered this.

    The coincidence viewer holds a ``MillingTaskViewerWidget`` for its config panels only:
    it renders the selected stage itself, as the yellow rect on ``fib_canvas``. It never
    hands that widget a controller, so the widget must display nothing and must not raise
    when a pattern update fires.

    This used to be true by accident rather than design. The widget took a ``viewer``
    argument that defaulted to ``napari.current_viewer()``, so passing ``None`` did not
    mean "no viewer" — it latched whichever napari viewer happened to be open (the
    minimap's) and appended a mouse callback to it. The napari draw stayed inert only
    because ``_fib_image_layer`` was also never set. Two independent accidents; the
    argument is gone now (FIB-407).
    """
    widget = _coincidence_viewer()
    milling = widget.milling_viewer_widget

    assert not hasattr(milling, "viewer"), (
        "the milling widget grew a viewer back — napari.current_viewer() latches the "
        "minimap's viewer when one is open"
    )
    assert milling._controller is None, (
        "coincidence viewer must not attach a controller"
    )

    image = _fib_image()
    widget.set_fib_image(image)
    assert milling._fib_image is image, (
        "the widget lost the frame patterns are placed against"
    )

    # Reaches here through a QTimer.singleShot in the app, where a raise would escape the
    # Qt event loop and abort the process under PyQt5 (FIB-329) rather than fail a test.
    milling._update_pattern_display()
    assert milling._pattern_update_pending is False


def _drag_fib_rect_to(overlay, cx: float, cy: float, size: float = 60.0) -> dict:
    """Drive one drag the way the canvas does: move the rect, then emit rect_changed.

    Returns the emitted rect, so callers compare against the geometry the handler
    actually saw rather than the one they asked for.
    """
    overlay.set_rect(x0=cx - size / 2, y0=cy - size / 2, width=size, height=size)
    info = overlay.get_rect()
    overlay.rect_changed.emit(info)
    return info


def test_dragging_the_fib_rect_still_repositions_the_pattern():
    """The drag is how an operator positions a coincidence mill, and it is the one path
    that runs *through* the milling widget rather than around it
    (``_on_fib_rect_changed`` → ``MillingTaskViewerWidget._move_patterns``).

    Note the FM-enabled configuration: ``_connect_signals`` returns at its first line
    when ``microscope.fm is None``, so on a plain demo session the viewer wires up
    *nothing* and a drag test would pass vacuously against any breakage.

    The first drag is what makes the second one meaningful. ``_move_patterns`` positions
    the pattern absolutely, so the point a given rect produces is fixed — and the viewer
    restores the last-used milling config, which after any previous run of this test
    holds exactly the point the drag below produces. The assertion then compared a value
    to itself and failed however the file was invoked. Establishing the start here (and
    isolating the config in ``_coincidence_viewer``) keeps the target demonstrably
    different from the start no matter what the machine has saved.
    """
    widget = _coincidence_viewer()
    milling = widget.milling_viewer_widget
    widget.set_fib_image(_fib_image())

    overlay = widget.fib_canvas.rect_overlay
    assert overlay.receivers(overlay.rect_changed), (
        "the rect drag is not wired to anything"
    )

    stages = milling.config_widget.milling_stages_widget.get_enabled_stages()
    assert stages, "no enabled milling stage to reposition"

    # Both positions keep the 48x77px pattern well inside the 768x512 frame:
    # _move_patterns rejects the whole move if the pattern would fall outside it.
    first_rect = _drag_fib_rect_to(overlay, cx=300.0, cy=300.0)
    before = milling.config_widget.milling_stages_widget.get_enabled_stages()[
        0
    ].pattern.point

    second_rect = _drag_fib_rect_to(overlay, cx=500.0, cy=180.0)

    after = milling.config_widget.milling_stages_widget.get_enabled_stages()[
        0
    ].pattern.point
    assert (after.x, after.y) != (before.x, before.y), (
        "dragging the FIB rect no longer moves the milling pattern"
    )
    # ...and it tracked the rect, rather than landing somewhere of its own: the same
    # displacement in metres, with Y inverted (image Y grows downward, stage Y upward).
    pixel_size = widget.fib_canvas.canvas.pixel_size
    assert np.isclose(
        after.x - before.x, (second_rect["cx"] - first_rect["cx"]) * pixel_size
    )
    assert np.isclose(
        after.y - before.y, -(second_rect["cy"] - first_rect["cy"]) * pixel_size
    )


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
