"""Headless tests for the FM overview acquisition UI.

Covers the parts where being wrong is silent: what the tile mask means, whether the
settings round-trip into the object the acquisition actually consumes, and whether a
combo box shows the label it was given.

Uses PyQt5 directly with the offscreen platform (no pytest-qt dependency).
"""
import os
from copy import deepcopy

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QLabel

from fibsem.fm.structures import (
    AutoFocusMode,
    ChannelSettings,
    OverviewParameters,
    ZParameters,
)
from fibsem.structures import TileOrderStrategy
from fibsem.ui.fm.widgets.fm_overview_confirmation_dialog import (
    FMOverviewConfirmationDialog,
    format_duration,
)
from fibsem.ui.fm.widgets.fm_overview_settings_widget import FMOverviewSettingsWidget
from fibsem.ui.fm.widgets.fm_overview_widget import (
    PREVIEW_KEY,
    FMOverviewWidget,
    shrink_progress_text,
)
from fibsem.ui.fm.widgets.tile_mask_widget import TileMaskWidget
from fibsem.ui.widgets.custom_widgets import ValueComboBox
from fibsem.ui.widgets.progress_widget import FibsemProgressWidget, ProgressUpdate


@pytest.fixture(autouse=True)
def _restore_fm_overview_flag():
    """`FEATURE_FM_OVERVIEW_TAB_ENABLED` is a module global that several tests here set.

    Restored around every test rather than by whoever set it: the flag has to stay on
    for the whole of a test that uses the tab, not just while the tab is built, so the
    setter cannot also be the restorer.
    """
    import fibsem.config as fibsem_cfg

    previous = fibsem_cfg.FEATURE_FM_OVERVIEW_TAB_ENABLED
    yield
    fibsem_cfg.FEATURE_FM_OVERVIEW_TAB_ENABLED = previous


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def plus_mask(n):
    mid = n // 2
    return [[(i == mid or j == mid) for j in range(n)] for i in range(n)]


# ── tile mask ────────────────────────────────────────────────────────────


def test_a_full_selection_is_reported_as_no_mask(qapp):
    """None, not an all-True mask.

    `OverviewParameters.tile_mask` already spells "acquire everything" as None, and
    two spellings for one state means saved configurations disagree about which is
    canonical.
    """
    widget = TileMaskWidget(rows=3, cols=3)

    assert widget.mask is None
    assert widget.n_enabled == 9


def test_disabling_a_tile_produces_a_mask(qapp):
    widget = TileMaskWidget(rows=3, cols=3)
    widget.mask = plus_mask(3)

    assert widget.mask == plus_mask(3)
    assert widget.n_enabled == 5


def test_resizing_the_grid_keeps_the_tiles_that_still_exist(qapp):
    """Nudging the row count must not silently discard a hand-built selection."""
    widget = TileMaskWidget(rows=3, cols=3)
    widget.mask = plus_mask(3)

    widget.set_grid_size(4, 4)

    mask = widget.mask
    assert mask[0][1] is True and mask[0][0] is False    # carried over
    assert all(mask[3])                                   # new row starts enabled


def test_all_none_and_invert(qapp):
    widget = TileMaskWidget(rows=2, cols=2)

    widget.grid.set_all(False)
    assert widget.n_enabled == 0
    widget.grid.invert()
    assert widget.n_enabled == 4
    widget.grid.set_all(True)
    assert widget.mask is None


def test_the_mask_property_hands_back_a_copy(qapp):
    """Otherwise a caller can mutate the widget's state without it noticing."""
    widget = TileMaskWidget(rows=2, cols=2)
    widget.mask = [[True, False], [False, True]]

    snapshot = widget.mask
    snapshot[0][0] = False

    assert widget.mask[0][0] is True


# ── settings ─────────────────────────────────────────────────────────────


def test_settings_round_trip_every_field(qapp):
    """The widget's output is what the acquisition consumes, verbatim."""
    widget = FMOverviewSettingsWidget(channel_settings=[ChannelSettings(name="GFP"), ChannelSettings(name="RFP")])
    parameters = OverviewParameters(
        rows=5, cols=5, overlap=0.15, use_zstack=True,
        autofocus_mode=AutoFocusMode.EACH_TILE,
        tile_order=TileOrderStrategy.SERPENTINE,
        tile_mask=plus_mask(5),
    )

    widget.parameters = parameters

    assert widget.parameters == parameters


def test_changing_the_grid_size_resizes_the_mask(qapp):
    widget = FMOverviewSettingsWidget()
    widget.spin_rows.setValue(4)
    widget.spin_cols.setValue(6)

    assert widget.parameters.rows == 4
    assert widget.parameters.cols == 6
    assert widget.tile_mask.grid._rows == 4
    assert widget.tile_mask.grid._cols == 6


def test_the_sweep_controls_are_only_live_when_focusing(qapp):
    """Mode is the single switch: it greys the sweep out *and* decides what is returned."""
    widget = FMOverviewSettingsWidget(channel_settings=[ChannelSettings(name="GFP")])

    widget.combo_autofocus_mode.set_value(AutoFocusMode.NONE)
    assert not widget.autofocus_widget.isEnabled()
    assert widget.autofocus_settings is None

    widget.combo_autofocus_mode.set_value(AutoFocusMode.ONCE)
    assert widget.autofocus_widget.isEnabled()
    assert widget.autofocus_settings is not None


def test_a_spiral_with_per_row_focus_says_what_will_actually_happen(qapp):
    """The runner promotes EACH_ROW to EACH_TILE for a spiral; don't make that a surprise."""
    widget = FMOverviewSettingsWidget()
    widget.combo_autofocus_mode.set_value(AutoFocusMode.EACH_ROW)
    widget.combo_tile_order.set_value(TileOrderStrategy.SPIRAL)

    assert "per-tile" in widget.label_summary.text()

    widget.combo_tile_order.set_value(TileOrderStrategy.TYPEWRITER)
    assert "per-tile" not in widget.label_summary.text()


def test_the_selection_is_counted_once(qapp):
    """The count belongs to the mask panel. It used to be repeated in the summary line
    below, where a stale second copy would read as a different number."""
    widget = FMOverviewSettingsWidget()
    widget.parameters = OverviewParameters(rows=3, cols=3, tile_mask=plus_mask(3))

    assert "5/9 tiles" in widget.tile_mask.label_count.text()
    assert "tiles" not in widget.label_summary.text()


@pytest.mark.parametrize(
    ("rows", "cols", "overlap", "expected"),
    [
        (3, 3, 0.0, "300 × 300 µm"),
        (3, 3, 0.1, "280 × 280 µm"),   # (n-1) steps of 90 µm, plus one whole tile
        (2, 5, 0.0, "500 × 200 µm"),   # width from columns, height from rows
        (1, 1, 0.5, "100 × 100 µm"),   # a single tile is its own field of view
    ],
)
def test_the_grid_reports_the_total_field_of_view(qapp, rows, cols, overlap, expected):
    """Area covered comes from rows/cols/overlap alone -- a mask does not shrink it,
    because skipped tiles keep their place in the grid."""
    widget = FMOverviewSettingsWidget()
    widget.set_tile_fov(100e-6, 100e-6)

    widget.parameters = OverviewParameters(rows=rows, cols=cols, overlap=overlap)
    assert widget.label_total_fov.text() == expected

    widget.tile_mask.grid.set_all(False)
    assert widget.label_total_fov.text() == expected


def test_the_total_field_of_view_is_unknown_until_the_camera_is_known(qapp):
    assert FMOverviewSettingsWidget().label_total_fov.text() == "—"


def test_focusing_with_every_pass_disabled_is_called_out(qapp):
    """Two controls that each look right, contradicting each other: the run would
    schedule focusing and then sweep nothing."""
    widget = FMOverviewSettingsWidget(channel_settings=[ChannelSettings(name="GFP")])
    widget.combo_autofocus_mode.set_value(AutoFocusMode.ONCE)

    for sweep_pass in widget.autofocus_widget.autofocus_settings.passes:
        sweep_pass.enabled = False
    widget._refresh_derived()

    assert "every sweep pass is disabled" in widget.label_summary.text()


# ── confirmation dialog ──────────────────────────────────────────────────


def _dialog(parameters, channels=None, zparams=None):
    return FMOverviewConfirmationDialog(
        parameters=parameters,
        channel_settings=channels or [ChannelSettings(name="GFP", exposure_time=0.05)],
        zparams=zparams,
        tile_fov=(100e-6, 100e-6),
    )


def test_the_dialog_counts_acquired_and_skipped(qapp):
    dialog = _dialog(OverviewParameters(rows=5, cols=5, tile_mask=plus_mask(5)))

    shown = " ".join(label.text() for label in dialog.findChildren(QLabel))
    assert "9 to acquire" in shown
    assert "16 skipped" in shown


def test_the_dialog_refuses_an_empty_selection(qapp):
    """The runner rejects this too; saying so here beats failing after the dismiss."""
    empty = [[False] * 3 for _ in range(3)]
    dialog = _dialog(OverviewParameters(rows=3, cols=3, tile_mask=empty))

    assert not dialog.button_start.isEnabled()


def test_the_dialog_estimate_reflects_the_mask(qapp):
    """A sparse run must not be quoted the duration of the full grid."""
    full = _dialog(OverviewParameters(rows=5, cols=5))
    sparse = _dialog(OverviewParameters(rows=5, cols=5, tile_mask=plus_mask(5)))

    assert sparse._estimate()["total_time"] < full._estimate()["total_time"]


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "0s"), (45, "45s"), (60, "1m 00s"), (134, "2m 14s"), (3660, "1h 01m")],
)
def test_duration_formatting(seconds, expected):
    assert format_duration(seconds) == expected


# ── shared widget fix ────────────────────────────────────────────────────


def test_an_explicit_format_fn_beats_the_built_in_rendering(qapp):
    """It used to be consulted last, so enums and numbers ignored it silently.

    Three call sites were affected: two mode combo boxes rendered `EACH_ROW` instead
    of the label they asked for, and emission wavelengths rendered as a bare number
    rather than the `550 nm` their formatter produces.
    """
    enums = ValueComboBox(items=list(AutoFocusMode), format_fn=lambda m: f"<{m.value}>")
    assert enums.itemText(0) == "<none>"

    numbers = ValueComboBox(items=[550.0, 600.0], format_fn=lambda w: f"{int(w)} nm")
    assert numbers.itemText(0) == "550 nm"


def test_without_a_format_fn_the_built_in_rendering_still_applies(qapp):
    assert ValueComboBox(items=list(AutoFocusMode)).itemText(0) == "NONE"


# ── progress bars ────────────────────────────────────────────────────────
#
# One signal carries both scales: the tileset runner's tile counter, and -- from
# inside each tile -- `acquire_z_stack` / `acquire_channels`. `task` decides which
# bar a payload drives, so a payload must never move both.


class _Router:
    """The routing half of FMOverviewWidget, without needing a microscope."""

    def __init__(self):
        self.progress_tiles = FibsemProgressWidget()
        self.progress_tile_detail = FibsemProgressWidget()
        self.status = QLabel()

    _apply_progress = FMOverviewWidget._apply_progress
    _apply_tile_progress = FMOverviewWidget._apply_tile_progress
    _tile_detail_update = FMOverviewWidget._tile_detail_update

    def _show_preview(self, payload):
        pass

    def _finish(self, state, error):
        self.finished = (state, error)


def _bar(widget):
    return widget._bar


def test_a_tileset_payload_moves_only_the_tile_bar(qapp):
    router = _Router()

    router._apply_progress({
        "state": "acquiring", "task": "tileset", "current": 4, "total": 9,
    })

    # value/maximum directly, not a percentage -- the widget counts items.
    assert (_bar(router.progress_tiles).value(),
            _bar(router.progress_tiles).maximum()) == (4, 9)
    assert not router.progress_tile_detail.isVisible()


def test_a_zstack_payload_moves_only_the_detail_bar(qapp):
    router = _Router()

    router._apply_progress({
        "state": "acquiring", "task": "z-stack", "channel": "GFP",
        "zlevel": 7, "total_zlevels": 21,
    })

    assert "z-stack" in _bar(router.progress_tile_detail).format()
    assert (_bar(router.progress_tile_detail).value(),
            _bar(router.progress_tile_detail).maximum()) == (7, 21)
    assert not router.progress_tiles.isVisible()


def test_a_channels_payload_drives_the_detail_bar_too(qapp):
    """Without this the second bar is dead on every run that is not z-stacked."""
    router = _Router()

    router._apply_progress({
        "state": "acquiring", "task": "channels", "channel": "RFP",
        "channel_index": 2, "total_channels": 3,
    })

    assert "RFP" in _bar(router.progress_tile_detail).format()
    assert (_bar(router.progress_tile_detail).value(),
            _bar(router.progress_tile_detail).maximum()) == (2, 3)


def test_a_completed_tile_leaves_the_detail_bar_alone(qapp):
    """It used to reset here, so the bar vanished and returned at every tile boundary
    -- a flicker for the length of the run. The next tile overwrites it a moment later
    anyway, so the stale count is never seen."""
    router = _Router()
    router._apply_progress({
        "state": "acquiring", "task": "z-stack", "channel": "GFP",
        "zlevel": 21, "total_zlevels": 21,
    })

    router._apply_progress({
        "state": "tile", "task": "tileset", "current": 1, "total": 9,
    })

    assert router.progress_tile_detail.isVisible()


def test_an_unknown_task_moves_nothing(qapp):
    router = _Router()

    router._apply_progress({"state": "acquiring", "task": "something-else"})

    assert not router.progress_tiles.isVisible()
    assert not router.progress_tile_detail.isVisible()


def test_the_estimate_is_shown_when_the_payload_carries_one(qapp):
    router = _Router()

    router._apply_progress({
        "state": "acquiring", "task": "tileset", "current": 2, "total": 9,
        "estimated_remaining_time": 134.0, "estimated_total_time": 180.0,
    })

    assert "remaining" in _bar(router.progress_tiles).format()


def test_a_stage_move_does_not_fill_the_bar(qapp):
    """`indeterminate` paints a *full* bar, so every move read as "finished".

    The count stays put -- it is still true between tiles -- and the transient state
    goes to the status label.
    """
    router = _Router()
    router._apply_progress({
        "state": "acquiring", "task": "tileset", "current": 3, "total": 9,
    })

    router._apply_progress({"state": "moving", "task": "tileset"})

    assert (_bar(router.progress_tiles).value(),
            _bar(router.progress_tiles).maximum()) == (3, 9)
    assert router.status.text() == "Moving stage…"


# ── layout stability ─────────────────────────────────────────────────────


def test_the_bar_text_is_smaller_than_the_default(qapp):
    """Pinned because `setFont` on the holder silently does not reach the bar, so the
    obvious way to do this looks right and changes nothing."""
    from PyQt5.QtGui import QFontMetrics

    plain, shrunk = FibsemProgressWidget(), FibsemProgressWidget()
    shrink_progress_text(shrunk)

    sample = "Tiles — 4/9 · 74s remaining"
    assert (QFontMetrics(shrunk._bar.font()).width(sample)
            < QFontMetrics(plain._bar.font()).width(sample))


def test_shrinking_the_text_keeps_the_failed_colouring(qapp):
    """The chunk stylesheet is what distinguishes a failed run from a finished one."""
    progress = FibsemProgressWidget()
    shrink_progress_text(progress)

    progress.update_progress(ProgressUpdate.failed("nope"))

    assert "#99121F" in progress._bar.styleSheet()


# ── channel detail form ──────────────────────────────────────────────────


def test_channel_detail_fields_fill_the_panel(qapp):
    """macOS styles QFormLayout its own way -- fields frozen at their size hint, so
    each row ends up a different width and the block floats in the middle of the
    panel. Pinned behaviourally because the properties that fix it are easy to set
    and easy to set uselessly: this form previously called `setLabelAlignment` with
    its own current value, and `ExpandingFieldsGrow` does nothing here because none
    of these controls carry an Expanding policy."""
    from PyQt5.QtCore import Qt

    from fibsem.fm.microscope import FluorescenceMicroscope
    from fibsem.ui.fm.widgets.channel_settings_widget import ChannelSettingsWidget

    widget = ChannelSettingsWidget(fm=FluorescenceMicroscope())
    widget.set_channel(ChannelSettings(name="Channel-01"))
    widget.resize(440, widget.sizeHint().height())
    widget.show()
    qapp.processEvents()

    fields = (widget.excitation_combo, widget.emission_combo,
              widget.exposure_spin, widget.power_spin, widget.gain_spin)
    widths = {f.width() for f in fields}

    assert len(widths) == 1, f"ragged field widths: {[f.width() for f in fields]}"
    assert all(f.width() > f.sizeHint().width() for f in fields)

    form = widget.excitation_combo.parent().layout()
    assert form.labelAlignment() & Qt.AlignLeft
    assert form.formAlignment() & Qt.AlignLeft


def test_channel_rows_do_not_overflow_a_narrow_panel(qapp):
    """A list item with a zero-width size hint takes the row widget's preferred width,
    which QLineEdit inflates ~190px beyond what the row needs and which does not track
    the host. In the overview's controls column that pushed the excitation and emission
    combos past the right edge -- and the column keeps its horizontal scrollbar off, so
    they were unreachable rather than merely cramped."""
    from fibsem.fm.microscope import FluorescenceMicroscope
    from fibsem.ui.fm.widgets.fm_multi_channel_widget import (
        FluorescenceMultiChannelWidget,
    )

    widget = FluorescenceMultiChannelWidget(
        fm=FluorescenceMicroscope(),
        channel_settings=[ChannelSettings(name="Channel-01")],
    )
    widget.show()

    for width in (460, 900):
        widget.resize(width, 300)
        qapp.processEvents()
        qapp.processEvents()

        inner = widget._list._list
        row = inner.itemWidget(inner.item(0))

        assert row.width() <= inner.viewport().width(), (
            f"row overflows at host width {width}: "
            f"{row.width()} > {inner.viewport().width()}"
        )


def test_collapsed_panels_do_not_stretch(qapp):
    """A panel added with a stretch factor keeps claiming vertical space after it is
    collapsed, so folding it leaves a tall empty box with the title floating in the
    middle. Every panel must fold to the same header-only height."""
    widget = FMOverviewSettingsWidget()
    widget.show()
    widget.resize(500, 1200)  # far taller than the folded panels need

    panels = (widget.focus_panel, widget.zstack_panel,
              widget.grid_panel, widget.mask_panel)
    for panel in panels:
        panel.collapse()
    qapp.processEvents()
    qapp.processEvents()

    heights = {p.height() for p in panels}
    assert len(heights) == 1, f"collapsed panels differ in height: {heights}"


# ── tile grid overlay wiring ─────────────────────────────────────────────


def _mouse(overlay, x, y, px=0.0, py=0.0, button=1):
    """A matplotlib mouse event over the overlay's axes.

    `px`/`py` are screen pixels: the overlay measures drag distance in them, so a
    click and a drag differ only in whether these change between press and release.
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        button=button, inaxes=overlay._ax, xdata=x, ydata=y, x=px, y=py
    )


def _end_gesture(overlay, event):
    """Release the overlay *and* the canvas, as a real release does.

    Both are connected to the same matplotlib event in the app, and it is the canvas's
    handler that clears `_overlay_consuming_event` — the flag that suppresses panning,
    and now refitting, for as long as an overlay owns the gesture. Releasing only the
    overlay leaves the canvas believing a drag is still in flight, which then silently
    disables refitting for every later test sharing the fixture.
    """
    overlay._on_release(event)
    if overlay._canvas is not None:
        overlay._canvas._on_release(event)


def _click(overlay, x, y):
    """Press and release at one point -- the gesture that toggles a tile."""
    overlay._on_press(_mouse(overlay, x, y))
    _end_gesture(overlay, _mouse(overlay, x, y))


@pytest.fixture(scope="module")
def overview_widget(qapp):
    """A real widget against the Demo microscope, for the canvas/settings wiring."""
    from fibsem.ui.fm.overview_app import build_microscope
    from fibsem.ui.fm.widgets.fm_overview_widget import FMOverviewWidget

    microscope = build_microscope()
    # Inserted, so the widget under test is one that could actually run. Left retracted
    # -- which is how `build_microscope` leaves it -- every test that reaches `acquire`
    # is really testing the objective guard rather than whatever it meant to (FIB-417).
    microscope.fm.objective.insert()
    widget = FMOverviewWidget(microscope)
    widget.resize(1200, 800)
    widget.show()
    widget.canvas.set_fm_image(
        microscope.fm.acquire_image(ChannelSettings(name="Channel-01"))
    )
    qapp.processEvents()
    widget._refresh_tile_grid()
    qapp.processEvents()
    return widget


def test_clicking_a_tile_updates_the_mask_the_settings_widget_owns(qapp, overview_widget):
    """One mask, two views. The overlay must not keep its own copy -- a click routes
    through the settings widget, and the redraw follows from that."""
    widget = overview_widget
    widget.settings_widget.tile_mask.mask = None
    qapp.processEvents()

    overlay = widget.tile_grid_overlay
    tile = next(t for t in overlay._tiles if (t.row, t.col) == (0, 0))
    x, y, tw, th = overlay._rect_for(tile)
    _click(overlay, x + tw / 2, y + th / 2)
    qapp.processEvents()

    assert widget.settings_widget.tile_mask.mask[0][0] is False
    assert widget.settings_widget.tile_mask.n_enabled == 8
    # and the overlay redrew from the widget's mask, rather than mutating its own
    assert next(
        t for t in overlay._tiles if (t.row, t.col) == (0, 0)
    ).enabled is False


def test_clicking_the_same_tile_twice_returns_it(qapp, overview_widget):
    widget = overview_widget
    widget.settings_widget.tile_mask.mask = None
    qapp.processEvents()

    overlay = widget.tile_grid_overlay
    tile = next(t for t in overlay._tiles if (t.row, t.col) == (2, 1))
    x, y, tw, th = overlay._rect_for(tile)

    for _ in range(2):
        _click(overlay, x + tw / 2, y + th / 2)
        qapp.processEvents()

    assert widget.settings_widget.tile_mask.n_enabled == 9


def test_resizing_the_grid_redraws_the_overlay(qapp, overview_widget):
    widget = overview_widget
    widget.settings_widget.tile_mask.mask = None
    widget.settings_widget.spin_rows.setValue(5)
    widget.settings_widget.spin_cols.setValue(4)
    qapp.processEvents()

    assert len(widget.tile_grid_overlay._tiles) == 20
    assert len(widget.tile_grid_overlay._artists) == 20

    widget.settings_widget.spin_rows.setValue(3)
    widget.settings_widget.spin_cols.setValue(3)
    qapp.processEvents()

    assert len(widget.tile_grid_overlay._tiles) == 9


# ── position overlay ─────────────────────────────────────────────────────


def _offset(base, dx=0.0, dy=0.0, name=""):
    from fibsem.structures import FibsemStagePosition

    position = FibsemStagePosition(
        x=base.x + dx, y=base.y + dy, z=base.z, r=base.r, t=base.t
    )
    position.name = name
    return position


def test_positions_are_marked_where_they_are(qapp, overview_widget):
    """Markers live in the canvas frame, which is anchored on the stage position the
    overview is built around -- so the origin is (0, 0), not the middle of some image.
    What matters is that a known stage offset comes out as the same offset on screen."""
    widget = overview_widget
    image = widget.microscope.fm.acquire_image(ChannelSettings(name="Channel-01"))
    widget.set_image(image)
    base = image.metadata.stage_position
    pixel_size = image.metadata.pixel_size_x

    widget.set_positions([_offset(base, name="here"),
                          _offset(base, dx=20e-6, name="right")])
    qapp.processEvents()

    points = widget.position_overlay._points
    assert points[0] == pytest.approx((0.0, 0.0))  # the origin the canvas is built on
    assert points[1] == pytest.approx((20e-6 / pixel_size, 0.0))  # 20 um to the right
    assert widget.position_overlay._labels == ["here", "right"]


def test_positions_outside_the_image_are_still_marked(qapp, overview_widget):
    """The canvas is zoomed out past the image for the tile grid, so a position beyond
    the field of view is visible and must not be pulled to the border."""
    widget = overview_widget
    image = widget.microscope.fm.acquire_image(ChannelSettings(name="Channel-01"))
    widget.set_image(image)
    base = image.metadata.stage_position
    width = image.data.shape[-1]

    widget.set_positions([_offset(base, dx=300e-6, name="far")])
    qapp.processEvents()

    assert widget.position_overlay._points[0][0] > width / 2


def test_an_image_without_geometry_can_still_be_marked(qapp, overview_widget):
    """Markers live in the canvas frame, which comes from the microscope, so nothing is
    projected *onto* the image — an image with no recorded geometry no longer blocks it.
    That matters in practice: tiles from the tiled runner come back without one."""
    widget = overview_widget
    image = widget.microscope.fm.acquire_image(ChannelSettings(name="Channel-01"))
    base = image.metadata.stage_position
    image.metadata.geometry = None
    widget.set_image(image)

    widget.set_positions([_offset(base, dx=20e-6, name="markable")])
    qapp.processEvents()

    assert widget.position_overlay._points, "the image's missing geometry blocked marking"


def test_positions_are_dropped_when_no_geometry_is_available_at_all(qapp, overview_widget):
    """Better an empty overlay than markers in plausible-looking wrong places."""
    widget = overview_widget
    original = widget._frame
    widget._frame = lambda: None
    try:
        widget.set_positions([_offset(widget._current_stage_position(), name="nowhere")])
        qapp.processEvents()
        assert widget.position_overlay._points == []
    finally:
        widget._frame = original


def test_clearing_the_positions_clears_the_markers(qapp, overview_widget):
    widget = overview_widget
    image = widget.microscope.fm.acquire_image(ChannelSettings(name="Channel-01"))
    widget.set_image(image)
    widget.set_positions([_offset(image.metadata.stage_position, name="one")])
    qapp.processEvents()
    assert widget.position_overlay._points

    widget.set_positions([])
    qapp.processEvents()

    assert widget.position_overlay._points == []


def test_setting_both_grid_dimensions_notifies_once(qapp):
    """Nudging the two spin boxes emits four times -- twice each, since resizing the
    mask emits again -- and passes through a grid size nobody asked for (new rows,
    old columns). Invisible by hand; an edge drag on the canvas does it per motion
    event, refreshing the overlay repeatedly against a size that was never requested."""
    widget = FMOverviewSettingsWidget()
    seen = []
    widget.changed.connect(
        lambda: seen.append((widget.spin_rows.value(), widget.spin_cols.value()))
    )

    widget.set_grid_size(2, 7)

    assert seen == [(2, 7)]
    assert (widget.tile_mask.grid._rows, widget.tile_mask.grid._cols) == (2, 7)


def test_setting_the_same_grid_size_is_a_no_op(qapp):
    widget = FMOverviewSettingsWidget()
    rows, cols = widget.spin_rows.value(), widget.spin_cols.value()
    seen = []
    widget.changed.connect(lambda: seen.append(1))

    widget.set_grid_size(rows, cols)

    assert seen == []


def test_a_canvas_resize_goes_through_the_settings_widget(qapp, overview_widget):
    """The canvas is a view: a drag asks for a size, it does not hold one."""
    widget = overview_widget
    widget.settings_widget.set_grid_size(3, 3)
    qapp.processEvents()

    widget.tile_grid_overlay.grid_resize_requested.emit(4, 6)
    qapp.processEvents()

    assert widget.settings_widget.parameters.rows == 4
    assert widget.settings_widget.parameters.cols == 6
    assert len(widget.tile_grid_overlay._tiles) == 24


def test_overviews_are_kept_per_acquisition_not_per_position(qapp, overview_widget):
    """A small overview and a wider one taken over the same area at different times are
    both worth keeping. Keying on position would silently drop the first; keying on the
    acquisition timestamp keeps both, and still replaces an image shown twice."""
    import copy

    widget = overview_widget
    # The fixture is shared, and overviews now accumulate by design — so start clean
    # rather than counting whatever earlier tests left behind.
    widget.canvas.clear_overviews()
    widget._records.clear()
    small = widget.microscope.fm.acquire_image(ChannelSettings(name="Channel-01"))
    widget.set_image(small)
    qapp.processEvents()

    wide = copy.deepcopy(small)
    wide.metadata.acquisition_date = "2026-07-31T19:00:00"
    wide.metadata.pixel_size_x = small.metadata.pixel_size_x * 3
    wide.metadata.pixel_size_y = small.metadata.pixel_size_y * 3
    widget.set_image(wide)
    qapp.processEvents()

    def overviews():
        # The widget's own record of what it placed. The fixture also seeds the canvas
        # directly, which lands under the canvas widget's default key and gets no
        # record — so this counts acquisitions rather than artists.
        return [record.id for record in widget.overviews()]

    assert len(overviews()) == 2, "the wider overview replaced the detailed one"

    widths = sorted(
        widget.canvas.canvas._placed[k].extent[1]
        - widget.canvas.canvas._placed[k].extent[0]
        for k in overviews()
    )
    assert widths[1] == pytest.approx(widths[0] * 3)  # each at its own scale

    widget.set_image(small)  # showing one again must not draw a second copy
    qapp.processEvents()
    assert len(overviews()) == 2


def test_the_first_preview_frame_lands_at_the_right_scale(qapp, overview_widget):
    """`set_channel` composites and places immediately, so the key, placement and pixel
    size have to be set before it. Establishing them afterwards applied them a tick late:
    the first frame of a run landed under the previous run's key at the previous run's
    pixel size, drawing it at the wrong size on top of a finished overview."""
    import numpy as np

    widget = overview_widget
    stride = 3
    tile_ps = widget.fm.camera.pixel_size[0]
    preview = np.zeros((1, 64, 128), dtype=np.uint8)  # (C, H, W), already decimated

    widget._show_preview({"image": preview, "preview_stride": stride})
    qapp.processEvents()

    placed = widget.canvas.canvas._placed
    assert "fm-preview" in placed, "the first preview frame was dropped"

    extent = placed["fm-preview"].extent
    reference = widget.canvas.canvas.reference_pixel_size
    # 128 decimated pixels at `stride` times the tile pixel size cover 128 * stride
    # tiles-worth of ground, whatever the canvas reference happens to be.
    expected = 128 * (tile_ps * stride) / reference
    assert extent[1] - extent[0] == pytest.approx(expected)


def test_toggling_a_tile_leaves_the_view_alone(qapp, overview_widget):
    """A tile toggle comes through the same refresh as a grid resize. Re-framing on one
    throws away the user's zoom, which reads on screen as clicking a tile zooming the
    view — the bug #245 fixed, and which declaring the working area on every refresh
    quietly reintroduced."""
    widget = overview_widget
    widget.settings_widget.parameters = OverviewParameters(rows=3, cols=3, overlap=0.1)
    qapp.processEvents()

    ax = widget.canvas.canvas._ax
    ax.set_xlim(-100, 100)
    ax.set_ylim(100, -100)
    zoomed = (tuple(ax.get_xlim()), tuple(ax.get_ylim()))

    widget._on_tile_toggled(0, 0, False)
    qapp.processEvents()

    assert (tuple(ax.get_xlim()), tuple(ax.get_ylim())) == zoomed


def test_resizing_the_grid_does_reframe(qapp, overview_widget):
    """The counterpart: a grid that no longer fits the view is worth re-framing for."""
    widget = overview_widget
    widget.settings_widget.parameters = OverviewParameters(rows=3, cols=3, overlap=0.1)
    qapp.processEvents()

    ax = widget.canvas.canvas._ax
    ax.set_xlim(-100, 100)
    ax.set_ylim(100, -100)
    zoomed = (tuple(ax.get_xlim()), tuple(ax.get_ylim()))

    widget.settings_widget.set_grid_size(5, 5)
    qapp.processEvents()

    assert (tuple(ax.get_xlim()), tuple(ax.get_ylim())) != zoomed


def test_dragging_the_grid_leaves_the_view_alone(qapp, overview_widget):
    """The same class of bug as a tile toggle, through a different door.

    A move drag emits on every motion event, and the refresh it triggers re-declared the
    canvas's working area — which forces a refit. Zoomed out, the very first step of a
    drag collapsed the view onto the grid (measured: a span of 41677 down to 4168) and
    then towed it along. The guard only knew about resize drags.
    """
    widget = overview_widget
    widget.settings_widget.parameters = OverviewParameters(rows=3, cols=3, overlap=0.1)
    qapp.processEvents()

    ax = widget.canvas.canvas._ax
    ax.set_xlim(-5000, 5000)
    ax.set_ylim(5000, -5000)
    zoomed = (tuple(ax.get_xlim()), tuple(ax.get_ylim()))

    overlay = widget.tile_grid_overlay
    anchor = overlay._anchor() or (0.0, 0.0)
    # Far enough to leave the declared working area, which is what zooming out makes
    # easy: a drag that stays inside it never tripped the old refit either, so a short
    # one would pass against the bug it is meant to catch.
    declared_width = widget.canvas.canvas.world_extent[0] / widget.canvas.canvas.reference_pixel_size
    reach = declared_width  # twice the half-width that bounds the area

    overlay._on_press(_mouse(overlay, anchor[0], anchor[1], px=0.0, py=0.0))
    for step in range(1, 4):
        # `px` has to move too: the overlay measures the drag threshold in screen
        # pixels, so a gesture that only moves in data coordinates never starts.
        overlay._on_drag(
            _mouse(overlay, anchor[0] + step * reach, anchor[1], px=step * 20.0, py=0.0)
        )
        qapp.processEvents()
        assert overlay.is_dragging
        assert (tuple(ax.get_xlim()), tuple(ax.get_ylim())) == zoomed
    _end_gesture(overlay, _mouse(overlay, anchor[0] + 3 * reach, anchor[1], px=60.0, py=0.0))
    qapp.processEvents()

    assert (tuple(ax.get_xlim()), tuple(ax.get_ylim())) == zoomed


def test_the_working_area_still_follows_a_dragged_grid(qapp, overview_widget):
    """Not moving the view is not the same as not tracking the grid. The working area is
    what the zoom limiter measures against, so leaving it behind would stretch the
    content extent across the gap and cap how far the view could zoom in — the reason it
    was being re-declared in the first place."""
    widget = overview_widget
    widget.settings_widget.parameters = OverviewParameters(rows=3, cols=3, overlap=0.1)
    qapp.processEvents()

    overlay = widget.tile_grid_overlay
    anchor = overlay._anchor() or (0.0, 0.0)
    before = widget.canvas.canvas.world_extent

    overlay._on_press(_mouse(overlay, anchor[0], anchor[1], px=0.0, py=0.0))
    overlay._on_drag(_mouse(overlay, anchor[0] + 1200, anchor[1], px=60.0, py=0.0))
    _end_gesture(overlay, _mouse(overlay, anchor[0] + 1200, anchor[1], px=60.0, py=0.0))
    qapp.processEvents()

    after = widget.canvas.canvas.world_extent
    assert after is not None and before is not None
    assert after[2] != before[2]  # centre followed the grid
    assert (after[0], after[1]) == (before[0], before[1])  # same size, it only moved


def test_declaring_a_working_area_without_a_refit_does_not_move_the_view(qapp, overview_widget):
    """The canvas-level half of the fix, stated on its own: `refit=False` has to hold
    even on an empty canvas, where `_fit_extent` falls back to the working area and so
    *does* change when it is re-declared."""
    widget = overview_widget
    canvas = widget.canvas.canvas
    ax = canvas._ax
    ax.set_xlim(-5000, 5000)
    ax.set_ylim(5000, -5000)
    zoomed = (tuple(ax.get_xlim()), tuple(ax.get_ylim()))

    canvas.set_world_extent(1e-3, 1e-3, (5e-4, 5e-4), refit=False)
    qapp.processEvents()

    assert (tuple(ax.get_xlim()), tuple(ax.get_ylim())) == zoomed

    canvas.set_world_extent(2e-3, 2e-3, (0.0, 0.0), refit=True)
    qapp.processEvents()

    assert (tuple(ax.get_xlim()), tuple(ax.get_ylim())) != zoomed


def test_the_grid_keeps_its_scale_under_a_decimated_preview(qapp, overview_widget):
    """The live preview is coarser than a tile. The grid is drawn in canvas coordinates,
    whose scale is fixed by the canvas reference — so the preview's stride must not
    reach it. Getting this wrong drew the grid stride times too large over the very
    image it describes."""
    import numpy as np

    widget = overview_widget
    widget.settings_widget.parameters = OverviewParameters(rows=3, cols=3, overlap=0.1)
    qapp.processEvents()
    before = widget.tile_grid_overlay._rect_for(widget.tile_grid_overlay._tiles[0])

    widget._show_preview({"image": np.zeros((1, 256, 256), np.uint8), "preview_stride": 4})
    qapp.processEvents()

    after = widget.tile_grid_overlay._rect_for(widget.tile_grid_overlay._tiles[0])
    assert after[2] == pytest.approx(before[2])
    assert after[3] == pytest.approx(before[3])


def test_the_stage_and_grid_limits_are_drawn_in_the_canvas_frame(qapp, overview_widget):
    """Same context the minimap gives, without its indirection: on a real-space canvas
    stage coordinates map straight to canvas coordinates, so there is no stitched image
    to reproject onto. Sizes must match what the microscope reports."""
    widget = overview_widget
    image = widget.microscope.fm.acquire_image(ChannelSettings(name="Channel-01"))
    widget.set_image(image)
    qapp.processEvents()

    by_label = {s.label: s for s in widget.stage_overlay._specs}
    reference = widget.canvas.canvas.reference_pixel_size
    limits = widget.microscope._stage.limits

    stage = by_label["Stage limits"]
    assert stage.width == pytest.approx((limits["x"].max - limits["x"].min) / reference)
    assert stage.height == pytest.approx((limits["y"].max - limits["y"].min) / reference)

    from fibsem.ui.fm.widgets.fm_overview_widget import GRID_RADIUS_M
    assert by_label["Grid boundary"].radius == pytest.approx(GRID_RADIUS_M / reference)

    assert widget.stage_overlay._artists, "specs were built but nothing was drawn"


def test_stage_metadata_does_not_wait_for_an_image(qapp, overview_widget):
    """The camera and stage can say where things are before anything is acquired, and the
    planned tile grid is already drawn from exactly that. Requiring an image would show
    two halves of the same picture at different times."""
    widget = overview_widget
    widget._records.clear()
    widget._origin = None

    widget._refresh_stage_metadata()

    labels = [spec.label for spec in widget.stage_overlay._specs]
    assert "Stage limits" in labels
    assert "Grid boundary" in labels


def test_stage_metadata_is_dropped_when_the_geometry_is_unknown(qapp, overview_widget):
    """Without a geometry there is no frame, and drawing at a guessed scale would put
    plausible-looking shapes in the wrong places."""
    widget = overview_widget
    widget._records.clear()
    widget._origin = None
    original = widget._frame
    widget._frame = lambda: None
    try:
        widget._refresh_stage_metadata()
        assert widget.stage_overlay._specs == []
    finally:
        widget._frame = original


def test_the_current_position_marker_follows_the_stage(qapp, overview_widget):
    """Drawn separately from the canvas's red origin marker: the origin explains why
    everything sits where it does, this yellow one is what you steer by. They coincide
    until the stage moves, then diverge."""
    from fibsem.structures import FibsemStagePosition

    widget = overview_widget
    widget.canvas.clear_overviews()
    widget._origin = None
    widget._refresh_current_position()
    qapp.processEvents()
    at_origin = list(widget.current_position_overlay._points)

    tilt = widget.microscope.get_stage_position().t
    widget.microscope.move_stage_absolute(
        FibsemStagePosition(x=150e-6, y=-60e-6, z=0.0, r=0.0, t=tilt)
    )
    qapp.processEvents()

    moved = widget.current_position_overlay._points
    assert moved != at_origin, "the marker did not follow the stage"

    # the canvas's own origin marker stays put, which is the point of having both
    origin_marker = widget.canvas.canvas._crosshair_artists[0].get_data()
    assert (origin_marker[0][0], origin_marker[1][0]) == (0.0, 0.0)


def test_the_marker_lands_where_an_image_acquired_there_is_placed(qapp, overview_widget):
    """The marker and the image reach the canvas by different routes — one through
    set_points, the other through add_image — so agreeing is worth checking. If they
    ever diverge, the canvas is telling two stories about the same stage position."""
    from fibsem.structures import FibsemStagePosition

    widget = overview_widget
    widget.canvas.clear_overviews()
    widget.set_image(widget.microscope.fm.acquire_image(ChannelSettings(name="Channel-01")))
    qapp.processEvents()

    tilt = widget.microscope.get_stage_position().t
    widget.microscope.move_stage_absolute(
        FibsemStagePosition(x=150e-6, y=-60e-6, z=0.0, r=0.0, t=tilt)
    )
    qapp.processEvents()
    marker = widget.current_position_overlay._points[0]

    widget.set_image(widget.microscope.fm.acquire_image(ChannelSettings(name="Channel-01")))
    qapp.processEvents()

    placed = widget.canvas.canvas._placed[max(widget.canvas.canvas.placed_keys)]
    xmin, xmax, ymax, ymin = placed.extent
    assert marker[0] == pytest.approx((xmin + xmax) / 2)
    assert marker[1] == pytest.approx((ymin + ymax) / 2)


# ── canvas interaction ───────────────────────────────────────────────────


class _SynchronousWorker:
    """Stands in for FunctionWorker so a test can see the outcome, not a thread."""

    def __init__(self, fn, *args, **kwargs):
        self.fn, self.args, self.kwargs = fn, args, kwargs

    def start(self):
        self.fn(*self.args, **self.kwargs)

    def is_alive(self):
        return False


@pytest.fixture
def interactive_widget(qapp, overview_widget):
    """`overview_widget` with the interaction state reset, and restored afterwards.

    The widget is module-scoped, so a target left behind by one test would silently
    re-centre every grid the next one drew.
    """
    from fibsem.structures import FibsemStagePosition

    widget = overview_widget
    start = widget.microscope.get_stage_position()
    widget.clear_target()
    qapp.processEvents()
    yield widget
    widget.clear_target()
    widget.microscope.move_stage_absolute(
        FibsemStagePosition(x=start.x, y=start.y, z=start.z, r=start.r, t=start.t)
    )
    qapp.processEvents()


def test_a_canvas_point_resolves_to_the_position_it_was_drawn_from(interactive_widget):
    """The two projections have to be exact inverses, not merely close. Everything on
    this canvas is placed by the forward one and clicked through the backward one, so
    any gap between them is the distance between what you point at and what you get."""
    from fibsem.structures import FibsemStagePosition

    widget = interactive_widget
    frame = widget._frame()
    base = widget._current_stage_position()

    for dx, dy in [(0.0, 0.0), (120e-6, 0.0), (0.0, -85e-6), (-250e-6, 375e-6)]:
        marked = FibsemStagePosition(
            x=base.x + dx, y=base.y + dy, z=base.z, r=base.r, t=base.t,
            coordinate_system=base.coordinate_system,
        )
        resolved = frame.to_stage(*frame.to_canvas(marked))

        assert resolved.x == pytest.approx(marked.x, abs=1e-12)
        assert resolved.y == pytest.approx(marked.y, abs=1e-12)


def test_double_clicking_moves_the_stage_to_that_point(qapp, interactive_widget, monkeypatch):
    from fibsem.ui.fm.widgets import fm_overview_widget as module

    widget = interactive_widget
    monkeypatch.setattr(module, "FunctionWorker", _SynchronousWorker)
    frame = widget._frame()
    before = widget._current_stage_position()
    point = frame.to_canvas(before)

    widget._on_canvas_double_clicked(point[0] + 1200.0, point[1] - 500.0, None)
    qapp.processEvents()

    after = widget.microscope.get_stage_position()
    assert (after.x, after.y) != (before.x, before.y), "the stage did not move"
    # and it landed under the click, which is the only claim the gesture makes
    assert frame.to_canvas(after) == pytest.approx(
        (point[0] + 1200.0, point[1] - 500.0), abs=1e-6
    )


def test_the_marker_follows_the_stage_after_a_double_click(qapp, interactive_widget, monkeypatch):
    """The move is confirmed by re-reading the stage rather than assumed, so the marker
    shows where the instrument went rather than where it was asked to go."""
    from fibsem.ui.fm.widgets import fm_overview_widget as module

    widget = interactive_widget
    monkeypatch.setattr(module, "FunctionWorker", _SynchronousWorker)
    before = list(widget.current_position_overlay._points)

    widget._on_canvas_double_clicked(1500.0, 900.0, None)
    qapp.processEvents()

    assert widget.current_position_overlay._points != before


def test_the_stage_cannot_be_driven_during_an_acquisition(qapp, interactive_widget, monkeypatch):
    from fibsem.ui.fm.widgets import fm_overview_widget as module

    widget = interactive_widget
    monkeypatch.setattr(module, "FunctionWorker", _SynchronousWorker)
    before = widget.microscope.get_stage_position()

    class _Running:
        def is_alive(self):
            return True

    widget._worker = _Running()
    try:
        widget._on_canvas_double_clicked(1500.0, 900.0, None)
        qapp.processEvents()
    finally:
        widget._worker = None

    after = widget.microscope.get_stage_position()
    assert (after.x, after.y) == (before.x, before.y)


def test_a_point_outside_the_stage_limits_is_refused(qapp, interactive_widget, monkeypatch):
    """A click can land anywhere on a real-space canvas, including well past where the
    stage can physically go. Asking for it is how a stage ends up against its stop."""
    from fibsem.ui.fm.widgets import fm_overview_widget as module

    widget = interactive_widget
    monkeypatch.setattr(module, "FunctionWorker", _SynchronousWorker)
    before = widget.microscope.get_stage_position()
    limits = getattr(widget.microscope._stage, "limits", None)
    if not limits:
        pytest.skip("this stage declares no limits")

    # far outside any plausible envelope: 1 m of travel at 100 nm/px
    widget._on_canvas_double_clicked(1e7, 1e7, None)
    qapp.processEvents()

    after = widget.microscope.get_stage_position()
    assert (after.x, after.y) == (before.x, before.y)


def test_dragging_the_grid_sets_a_target_without_moving_the_stage(qapp, interactive_widget):
    """Planning a run and driving the instrument are separate acts. A drag is
    exploratory -- you push the grid around to see what it would cover."""
    widget = interactive_widget
    before = widget.microscope.get_stage_position()
    anchor = widget.tile_grid_overlay._anchor()

    widget._on_grid_move(anchor[0] + 800.0, anchor[1] - 300.0)
    qapp.processEvents()

    after = widget.microscope.get_stage_position()
    assert (after.x, after.y) == (before.x, before.y)
    assert widget._target is not None
    assert widget.tile_grid_overlay._anchor() == pytest.approx(
        (anchor[0] + 800.0, anchor[1] - 300.0)
    )


def test_the_planned_grid_is_centred_where_the_run_will_be(qapp, interactive_widget):
    """The grid is a preview of an acquisition, so it has to sit where the acquisition
    would: on the target once one is set, and on the stage until then."""
    widget = interactive_widget
    anchor = widget.tile_grid_overlay._anchor()
    widget._on_grid_move(anchor[0] + 800.0, anchor[1])
    qapp.processEvents()

    assert widget._grid_centre() is widget._target

    widget.clear_target()
    qapp.processEvents()

    centre = widget._grid_centre()
    stage = widget.microscope.get_stage_position()
    assert (centre.x, centre.y) == (stage.x, stage.y)


def test_an_untargeted_grid_follows_the_stage(qapp, interactive_widget, monkeypatch):
    """It used to be pinned to the canvas origin -- where the stage was the first time
    anything was drawn. Correct until the stage moved, and then quietly not."""
    from fibsem.ui.fm.widgets import fm_overview_widget as module

    widget = interactive_widget
    monkeypatch.setattr(module, "FunctionWorker", _SynchronousWorker)
    before = widget.tile_grid_overlay._anchor()

    widget._on_canvas_double_clicked(before[0] + 1500.0, before[1], None)
    qapp.processEvents()

    assert widget.tile_grid_overlay._anchor()[0] == pytest.approx(before[0] + 1500.0, abs=1.0)


def test_a_targeted_grid_stays_put_when_the_stage_moves(qapp, interactive_widget, monkeypatch):
    from fibsem.ui.fm.widgets import fm_overview_widget as module

    widget = interactive_widget
    monkeypatch.setattr(module, "FunctionWorker", _SynchronousWorker)
    anchor = widget.tile_grid_overlay._anchor()
    widget._on_grid_move(anchor[0] + 800.0, anchor[1])
    qapp.processEvents()
    targeted = widget.tile_grid_overlay._anchor()

    widget._on_canvas_double_clicked(anchor[0] - 1500.0, anchor[1], None)
    qapp.processEvents()

    assert widget.tile_grid_overlay._anchor() == pytest.approx(targeted)


def test_centring_the_grid_is_only_offered_once_it_is_off_centre(qapp, interactive_widget):
    """The way back from a drag. A control that is always live invites a click that
    does nothing, and one that is never live strands the state it set."""
    widget = interactive_widget
    assert widget.tile_grid_panel.button_centre.isEnabled() is False

    anchor = widget.tile_grid_overlay._anchor()
    widget._on_grid_move(anchor[0] + 800.0, anchor[1])
    qapp.processEvents()
    assert widget.tile_grid_panel.button_centre.isEnabled() is True

    widget.tile_grid_panel.button_centre.click()
    qapp.processEvents()
    assert widget.tile_grid_panel.button_centre.isEnabled() is False
    assert widget.tile_grid_overlay._anchor() == pytest.approx(anchor)


def test_the_target_reaches_the_acquisition(qapp, interactive_widget, monkeypatch):
    """The drag is only worth anything if the run is actually centred there."""
    from fibsem.ui.fm.widgets import fm_overview_widget as module

    widget = interactive_widget
    recorded = {}

    class _RecordingRunner:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

    monkeypatch.setattr(module, "FMTiledAcquisitionRunner", _RecordingRunner)
    monkeypatch.setattr(module, "FunctionWorker", lambda *a, **k: _SynchronousWorker(lambda: None))
    monkeypatch.setattr(
        module.FMOverviewConfirmationDialog, "exec_", lambda self: module.QDialog.Accepted
    )
    anchor = widget.tile_grid_overlay._anchor()
    widget._on_grid_move(anchor[0] + 800.0, anchor[1])
    qapp.processEvents()

    widget.acquire()
    qapp.processEvents()
    widget._set_running(False)

    assert recorded["centre_position"] is widget._target


def test_the_cursor_readout_names_the_position_under_it(qapp, interactive_widget):
    """A canvas you have to click to get a coordinate out of cannot be used to decide
    whether to click."""
    widget = interactive_widget
    under = widget._frame().to_stage(1500.0, -900.0)

    widget._on_cursor_moved(1500.0, -900.0)

    text = widget.cursor_readout.text()
    assert f"{under.x * 1e6:.1f}" in text
    assert f"{under.y * 1e6:.1f}" in text

    widget._on_cursor_moved(None, None)
    assert widget.cursor_readout.text() == "", "the readout must not freeze off-canvas"


def test_the_reported_offset_is_measured_in_the_frame_it_is_drawn_in(qapp, interactive_widget):
    """Subtracting stage axes describes the same gap in a frame the user cannot see: on
    a compustage at t = -180 it reports y with the sign reversed, so the panel said the
    grid was above the stage while it was drawn below."""
    widget = interactive_widget
    frame = widget._frame()
    anchor = widget.tile_grid_overlay._anchor()

    widget._on_grid_move(anchor[0] + 1000.0, anchor[1] + 1000.0)
    qapp.processEvents()

    here = frame.offset(widget._current_stage_position())
    there = frame.offset(widget._target)
    assert widget._target_offset() == pytest.approx(
        (there[0] - here[0], there[1] - here[1])
    )
    # and the sign follows the canvas: dragging down the screen reads as +y
    assert widget._target_offset()[1] > 0


def test_the_working_area_follows_the_grid_not_the_canvas_origin(qapp, interactive_widget):
    """Pinned to the origin, it framed empty space once the stage travelled far from
    wherever the frame was anchored -- moving to an offset FM mount steps 48 mm -- and
    stretched the content extent across the gap, which capped how far the view could
    zoom in."""
    widget = interactive_widget
    span = widget.canvas.canvas.world_extent[0]

    widget._on_grid_move(*[v + 4 * span / widget.canvas.canvas.reference_pixel_size
                           for v in widget.tile_grid_overlay._anchor()])
    qapp.processEvents()

    _, _, cx, cy = widget.canvas.canvas.world_extent
    assert (cx, cy) == pytest.approx(widget._grid_offset())


def test_a_move_inside_the_working_area_leaves_the_framing_alone(qapp, interactive_widget):
    """A move must not throw away the user's zoom -- a double-click to somewhere 100 µm
    away did, the same failure a tile toggle used to cause.

    It is the *framing* that has to hold still, not the declared working area. This test
    used to assert the latter, because re-declaring the area re-framed the view and the
    only way to hold the view was to leave the area behind. The area now follows the
    grid on every move -- it is what the zoom limiter measures against, so leaving it
    behind was its own bug -- and `refit=False` is what makes that safe.
    """
    widget = interactive_widget
    canvas = widget.canvas.canvas
    span_px = canvas.world_extent[0] / canvas.reference_pixel_size
    anchor = widget.tile_grid_overlay._anchor()
    ax = canvas._ax
    framing = (tuple(ax.get_xlim()), tuple(ax.get_ylim()))

    widget._on_grid_move(anchor[0] + span_px / 4, anchor[1])
    qapp.processEvents()

    assert (tuple(ax.get_xlim()), tuple(ax.get_ylim())) == framing
    # ...and the grid still moved, which is the point of not re-framing for it
    assert widget.tile_grid_overlay._anchor()[0] == pytest.approx(anchor[0] + span_px / 4)


def test_the_working_area_follows_the_grid_on_every_move(qapp, interactive_widget):
    """Even a small one. The area is what caps how far the view can zoom in, and one
    left behind stretches the content extent across the gap between it and the grid."""
    widget = interactive_widget
    canvas = widget.canvas.canvas
    span_px = canvas.world_extent[0] / canvas.reference_pixel_size
    anchor = widget.tile_grid_overlay._anchor()

    widget._on_grid_move(anchor[0] + span_px / 4, anchor[1])
    qapp.processEvents()

    after = canvas.world_extent
    assert (after[2], after[3]) == pytest.approx(widget._grid_offset())


def test_a_grid_that_jumps_clear_of_an_empty_canvas_re_frames(qapp):
    """The one case where the view *should* follow a move: nothing has been acquired, so
    the grid is all there is to look at, and a grid off-screen leaves the canvas blank.
    Moving to an offset FM mount travels 48 mm and does exactly this.

    Only on an empty canvas -- once an image is placed the framing is fitted to the
    image, which a grid move does not change. Suppressed mid-drag either way, which is
    what `test_dragging_the_grid_leaves_the_view_alone` covers.
    """
    widget = _fresh_widget(qapp)
    widget._refresh_tile_grid()
    qapp.processEvents()

    canvas = widget.canvas.canvas
    ax = canvas._ax
    framing = (tuple(ax.get_xlim()), tuple(ax.get_ylim()))
    span_px = canvas.world_extent[0] / canvas.reference_pixel_size
    anchor = widget.tile_grid_overlay._anchor()

    widget._on_grid_move(anchor[0] + 4 * span_px, anchor[1])
    qapp.processEvents()

    assert (tuple(ax.get_xlim()), tuple(ax.get_ylim())) != framing
    after = canvas.world_extent
    assert (after[2], after[3]) == pytest.approx(widget._grid_offset())

    widget.close()


# ── anchoring the frame (FIB-418) ────────────────────────────────────────


def test_the_frame_can_be_anchored_at_a_chosen_position(qapp, interactive_widget):
    """So one canvas can describe the FM mount while another describes the column
    48 mm away, each correctly framed, rather than both taking whatever the stage
    happened to be doing when they woke up."""
    from fibsem.structures import FibsemStagePosition

    widget = interactive_widget
    widget.canvas.clear_overviews()
    stage = widget._current_stage_position()
    elsewhere = FibsemStagePosition(
        x=stage.x + 48e-3, y=stage.y, z=stage.z, r=stage.r, t=stage.t,
        coordinate_system=stage.coordinate_system,
    )
    try:
        widget.set_origin(elsewhere)
        qapp.processEvents()

        assert widget.origin is elsewhere
        # canvas zero is now that position, and the stage is 48 mm the other way
        assert widget._frame().to_canvas(elsewhere) == pytest.approx((0.0, 0.0))
        assert widget.current_position_overlay._points[0][0] == pytest.approx(
            widget._frame().to_canvas(stage)[0]
        )
    finally:
        widget.set_origin(None)
        qapp.processEvents()


def test_anchoring_again_returns_to_following_the_stage(qapp, interactive_widget):
    from fibsem.structures import FibsemStagePosition

    widget = interactive_widget
    widget.canvas.clear_overviews()
    stage = widget._current_stage_position()
    widget.set_origin(FibsemStagePosition(
        x=stage.x + 1e-3, y=stage.y, z=stage.z, r=stage.r, t=stage.t,
        coordinate_system=stage.coordinate_system,
    ))
    qapp.processEvents()

    widget.set_origin(None)
    qapp.processEvents()

    # re-derived from the stage on the next use, rather than left unset
    assert widget._frame() is not None
    assert widget.origin.x == pytest.approx(stage.x)


def test_the_frame_cannot_be_re_anchored_under_placed_images(qapp, interactive_widget):
    """Images are positioned relative to the origin as it stood when they were added,
    so moving it afterwards leaves every one describing a position it was never
    acquired at."""
    widget = interactive_widget
    widget.canvas.clear_overviews()
    widget.set_image(
        widget.microscope.fm.acquire_image(ChannelSettings(name="Channel-01"))
    )
    qapp.processEvents()
    anchored = widget.origin

    with pytest.raises(ValueError, match="already"):
        widget.set_origin(widget._current_stage_position())

    assert widget.origin is anchored, "the origin moved despite the refusal"
    widget.canvas.clear_overviews()


def test_closing_the_widget_lets_go_of_the_stage_signal(qapp):
    """`stage_position_changed` belongs to the microscope and outlives the widget. A
    connection left behind emits into a Qt object already torn down on the C++ side,
    which is a segfault rather than an exception: closing the tab and then moving the
    stage from anywhere else in the application took the whole application down.

    Asserted on the subscriber count rather than by provoking the crash, since a test
    that segfaults takes the runner with it.
    """
    from fibsem.ui.fm.overview_app import build_microscope

    microscope = build_microscope()
    before = len(microscope.stage_position_changed)
    widget = FMOverviewWidget(microscope)
    assert len(microscope.stage_position_changed) == before + 1

    widget.close()
    qapp.processEvents()

    assert len(microscope.stage_position_changed) == before


def test_a_dropped_widget_lets_go_of_the_stage_signal_too(qapp):
    """Not everything gets closed. psygnal holds a bound method weakly and drops it
    when the owner is collected -- but not a Qt signal's `emit`, which PyQt rebuilds on
    every access, so psygnal can neither weakref it nor match it at disconnect time.
    It also says nothing when the disconnect therefore removes nothing."""
    import gc

    from fibsem.ui.fm.overview_app import build_microscope

    microscope = build_microscope()
    before = len(microscope.stage_position_changed)
    widget = FMOverviewWidget(microscope)
    assert len(microscope.stage_position_changed) == before + 1

    del widget
    gc.collect()

    assert len(microscope.stage_position_changed) == before


# ── the host contract: saving, and being locked ──────────────────────────


def _fresh_widget(qapp):
    """A widget of its own, for tests that change state the module fixture shares."""
    from fibsem.ui.fm.overview_app import build_microscope

    widget = FMOverviewWidget(build_microscope())
    qapp.processEvents()
    return widget


def test_nothing_is_saved_until_a_host_says_where(qapp):
    """The standalone default. This app is mostly run against a simulator to exercise
    the UI, and those runs must not leave tiles wherever it was launched from."""
    widget = _fresh_widget(qapp)

    assert widget._save_directory is None
    assert widget._claim_destination() is None

    widget.close()


def test_a_save_directory_gives_every_run_its_own_place(qapp, tmp_path):
    """Not the experiment directory itself: tiles are named by row and column, so two
    overviews written straight into one directory would replace each other tile by
    tile."""
    widget = _fresh_widget(qapp)
    widget.set_save_directory(str(tmp_path))

    first = widget._claim_destination()
    second = widget._claim_destination()

    assert first.tiles_directory != second.tiles_directory
    assert os.path.dirname(first.tiles_directory) == str(tmp_path)

    widget.close()


def test_an_unwritable_directory_does_not_stop_the_run(qapp, tmp_path):
    """An overview you can see is worth more than no overview at all, so a save
    directory that cannot be prepared degrades to not saving rather than refusing."""
    blocked = tmp_path / "file-not-a-directory"
    blocked.write_text("")
    widget = _fresh_widget(qapp)
    widget.set_save_directory(str(blocked))

    assert widget._claim_destination() is None

    widget.close()


def test_a_failed_save_is_not_reported_as_a_successful_one(qapp, tmp_path):
    """Otherwise the difference only turns up later, when someone goes looking for the
    file and it is not there."""
    widget = _fresh_widget(qapp)

    # Nothing was meant to be saved: silent.
    widget._destination, widget._saved_path = None, None
    assert widget._describe_save() == ("", "")

    # A destination was claimed and the mosaic did not land: say so.
    widget.set_save_directory(str(tmp_path))
    widget._destination = widget._claim_destination()
    widget._saved_path = None
    assert "not saved" in widget._describe_save()[0]

    # It landed, and the tooltip says where.
    widget._saved_path = widget._destination.mosaic_path
    suffix, tooltip = widget._describe_save()
    assert suffix == "  ·  saved"
    assert tooltip == widget._destination.mosaic_path

    widget.close()


def test_a_host_can_forbid_starting_work_without_touching_a_run(qapp):
    """For a workflow that owns the instrument for a while. The canvas stays live --
    greying out the whole widget would also stop you reading the overview you just
    acquired, which costs the workflow nothing."""
    widget = _fresh_widget(qapp)

    widget.set_interactive(False)
    assert not widget.button_acquire.isEnabled()
    assert not widget.settings_widget.isEnabled()
    assert not widget.channel_widget.isEnabled()
    assert widget.canvas.isEnabled()

    widget.set_interactive(True)
    assert widget.button_acquire.isEnabled()
    assert widget.settings_widget.isEnabled()

    widget.close()


def test_a_lock_arriving_mid_run_leaves_cancel_available(qapp):
    """Otherwise a workflow starting while an overview is in flight would strand it:
    the stage keeps moving and the only button that stops it is greyed out."""
    widget = _fresh_widget(qapp)
    widget._set_running(True)

    widget.set_interactive(False)

    assert widget.button_cancel.isEnabled()
    assert not widget.button_acquire.isEnabled()

    widget._set_running(False)
    widget.close()


def test_a_workflow_ending_does_not_re_enable_a_tab_that_is_still_acquiring(qapp):
    """The two facts are independent and both have to be false before Acquire comes
    back. Deriving the button from only one of them is how it gets stuck -- or worse,
    lets a second acquisition start on top of the first."""
    widget = _fresh_widget(qapp)
    widget.set_interactive(False)
    widget._set_running(True)

    widget.set_interactive(True)

    assert not widget.button_acquire.isEnabled()

    widget._set_running(False)
    assert widget.button_acquire.isEnabled()

    widget.close()


def test_an_empty_grid_still_forbids_acquiring_after_a_lock_is_lifted(qapp):
    """The third fact. A tab unlocked with no tiles selected has nothing to acquire,
    and the button that starts it should say so."""
    widget = _fresh_widget(qapp)
    widget.settings_widget.tile_mask.mask = [[False, False], [False, False]]
    widget.settings_widget.set_grid_size(2, 2)
    widget.settings_widget.tile_mask.mask = [[False, False], [False, False]]
    widget._on_settings_changed()
    assert not widget.button_acquire.isEnabled()

    widget.set_interactive(False)
    widget.set_interactive(True)

    assert not widget.button_acquire.isEnabled()

    widget.close()


# ── the host side: building and retiring the tab ─────────────────────────


class _StubHost:
    """The tab wiring, without the AutoLamella main window.

    Constructing the real window needs napari viewers, which segfault under the
    offscreen platform -- on an unmodified `main` as well as here, so this stands in for
    a window that cannot be built in this environment rather than papering over
    something introduced with the tab. The methods borrowed below touch only the tab
    widget and the two attributes set in `__init__`.
    """

    from fibsem.applications.autolamella.ui.AutoLamellaMainUI import (
        AutoLamellaSingleWindowUI as _Real,
    )

    add_fm_overview_tab = _Real.add_fm_overview_tab
    _apply_fm_overview_visibility = _Real._apply_fm_overview_visibility
    _on_fm_overview_availability = _Real._on_fm_overview_availability
    _refresh_fm_overview_positions = _Real._refresh_fm_overview_positions
    _on_fm_overview_lamella_selected = _Real._on_fm_overview_lamella_selected
    _set_minimap_workflow_enabled = _Real._set_minimap_workflow_enabled
    _rebuild_lamella_list = _Real._rebuild_lamella_list
    _wire_position_events = _Real._wire_position_events

    def __init__(self, microscope=None, experiment=None, tab_enabled=True):
        import fibsem.config as fibsem_cfg
        from PyQt5.QtWidgets import QTabWidget

        self.tab_widget = QTabWidget()
        self.autolamella_ui = type(
            "_UI", (), {"microscope": microscope, "experiment": experiment}
        )()
        # The tab is behind a preference that is off by default, so these tests turn it
        # on and leave it on -- `_restore_fm_overview_flag` puts it back afterwards.
        fibsem_cfg.FEATURE_FM_OVERVIEW_TAB_ENABLED = tab_enabled
        self.add_fm_overview_tab()

    def set_flag(self, enabled: bool):
        """Toggle the preference and re-apply it, as the dialog does on accept."""
        import fibsem.config as fibsem_cfg

        fibsem_cfg.FEATURE_FM_OVERVIEW_TAB_ENABLED = enabled
        self._apply_fm_overview_visibility()

    @property
    def tab_enabled(self):
        return self.tab_widget.isTabEnabled(
            self.tab_widget.indexOf(self.fm_overview_tab)
        )

    # ── reading through to the tab ───────────────────────────────────────
    # Test scaffolding, not production shims: these say "what the window can see of the
    # tab" so the window-level tests below read as they did before the tab was split
    # out, while still going through the real `AutoLamellaFluorescenceOverviewTab`.

    @property
    def fm_overview_widget(self):
        return self.fm_overview_tab.overview

    def _build_fm_overview_widget(self):
        self.fm_overview_tab.refresh_microscope()

    def _teardown_fm_overview_widget(self):
        self.fm_overview_tab._drop_overview()

    def _update_fm_overview_experiment(self):
        self.fm_overview_tab.refresh_experiment()

    def _update_fm_overview_positions(self):
        self._refresh_fm_overview_positions()

    def _update_fm_overview_selection(self, lamella):
        self.fm_overview_tab.set_selected(lamella)


def _plain_microscope():
    """A Demo microscope with no fluorescence detector attached."""
    from fibsem import utils

    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.fm = None
    return microscope


def test_the_fm_overview_tab_is_reserved_but_dead_without_a_microscope(qapp):
    """The widget needs a camera at construction -- every scale on its canvas comes from
    one -- so the tab exists from startup and fills in on connection."""
    host = _StubHost()

    assert "FM Overview" in [
        host.tab_widget.tabText(i) for i in range(host.tab_widget.count())
    ]
    assert not host.tab_enabled
    assert host.fm_overview_widget is None


def test_the_tab_comes_alive_when_fluorescence_connects(qapp):
    from fibsem.ui.fm.overview_app import build_microscope

    host = _StubHost()
    host.autolamella_ui.microscope = build_microscope()

    host._build_fm_overview_widget()

    assert host.tab_enabled
    assert isinstance(host.fm_overview_widget, FMOverviewWidget)

    host._teardown_fm_overview_widget()


def test_a_microscope_without_fluorescence_hides_the_tab(qapp):
    """A system with no FM will never drive this tab, so showing it means a permanently
    greyed-out tab with nothing to say why. Distinct from *not connected yet*, which is
    temporary and covered below."""
    host = _StubHost(microscope=_plain_microscope())

    host._apply_fm_overview_visibility()

    index = host.tab_widget.indexOf(host.fm_overview_tab)
    assert not host.tab_widget.isTabVisible(index)
    assert host.fm_overview_widget is None


def test_no_microscope_yet_leaves_the_tab_visible_but_disabled(qapp):
    """The other absence. Waiting for a connection is what every other tab does, and a
    tab that vanishes until you connect is harder to find than a dim one."""
    host = _StubHost(microscope=None)

    host._apply_fm_overview_visibility()

    index = host.tab_widget.indexOf(host.fm_overview_tab)
    assert host.tab_widget.isTabVisible(index)
    assert not host.tab_widget.isTabEnabled(index)
    assert host.fm_overview_widget is None


def test_connecting_a_microscope_without_fluorescence_takes_the_tab_away(qapp):
    """Whether the system has an FM is only known once something is connected, so the
    tab starts visible and has to be withdrawn rather than never shown."""
    host = _StubHost(microscope=None)
    index = host.tab_widget.indexOf(host.fm_overview_tab)
    assert host.tab_widget.isTabVisible(index)

    host.autolamella_ui.microscope = _plain_microscope()
    host._apply_fm_overview_visibility()

    assert not host.tab_widget.isTabVisible(index)


def test_a_microscope_with_fluorescence_brings_the_tab_back(qapp):
    """And the reverse, so swapping between systems in one session settles correctly."""
    from fibsem.ui.fm.overview_app import build_microscope

    host = _StubHost(microscope=_plain_microscope())
    host._apply_fm_overview_visibility()
    index = host.tab_widget.indexOf(host.fm_overview_tab)
    assert not host.tab_widget.isTabVisible(index)

    host.autolamella_ui.microscope = build_microscope()
    host._apply_fm_overview_visibility()

    assert host.tab_widget.isTabVisible(index)
    assert host.tab_widget.isTabEnabled(index)
    assert isinstance(host.fm_overview_widget, FMOverviewWidget)

    host._teardown_fm_overview_widget()


def test_the_same_microscope_connecting_again_keeps_the_widget(qapp):
    """Rebuilding would throw away whatever is on the canvas, and a connection signal
    can arrive more than once for one instrument."""
    from fibsem.ui.fm.overview_app import build_microscope

    host = _StubHost(microscope=build_microscope())
    host._build_fm_overview_widget()
    first = host.fm_overview_widget

    host._build_fm_overview_widget()

    assert host.fm_overview_widget is first

    host._teardown_fm_overview_widget()


def test_a_different_microscope_replaces_the_widget(qapp):
    """The widget holds its microscope for life. Left alone across a reconnect it would
    read geometry from an instrument nobody is driving -- FIB-433 is about doing this
    without discarding the view."""
    from fibsem.ui.fm.overview_app import build_microscope

    host = _StubHost(microscope=build_microscope())
    host._build_fm_overview_widget()
    first = host.fm_overview_widget

    host.autolamella_ui.microscope = build_microscope()
    host._build_fm_overview_widget()

    assert host.fm_overview_widget is not first
    assert host.fm_overview_tab._microscope is host.autolamella_ui.microscope

    host._teardown_fm_overview_widget()


def test_retiring_the_widget_releases_the_stage_signal(qapp):
    """The subscription belongs to the microscope and outlives the widget. A stale one
    emits into a Qt object already torn down on the C++ side, which is a segfault rather
    than an exception -- so a reconnect that dropped the widget without closing it would
    take the application down on the next stage move."""
    from fibsem.ui.fm.overview_app import build_microscope

    microscope = build_microscope()
    before = len(microscope.stage_position_changed)
    host = _StubHost(microscope=microscope)
    host._build_fm_overview_widget()
    assert len(microscope.stage_position_changed) == before + 1

    host._teardown_fm_overview_widget()

    assert len(microscope.stage_position_changed) == before
    assert host.fm_overview_widget is None


def test_disconnecting_the_microscope_retires_the_tab(qapp):
    from fibsem.ui.fm.overview_app import build_microscope

    host = _StubHost(microscope=build_microscope())
    host._build_fm_overview_widget()

    host.autolamella_ui.microscope = None
    host._build_fm_overview_widget()

    assert host.fm_overview_widget is None
    assert not host.tab_enabled


def test_the_experiment_directory_reaches_the_widget(qapp, tmp_path):
    """By setter, not by handing over the experiment: the widget knows nothing about
    experiments, which is what lets it open standalone and be built in a test."""
    from fibsem.ui.fm.overview_app import build_microscope

    host = _StubHost(microscope=build_microscope())
    host._build_fm_overview_widget()
    assert host.fm_overview_widget._save_directory is None

    host.autolamella_ui.experiment = type(
        "_Exp", (), {"path": str(tmp_path), "positions": []}
    )()
    host._update_fm_overview_experiment()

    assert host.fm_overview_widget._save_directory == str(tmp_path)

    host._teardown_fm_overview_widget()


def test_a_workflow_lock_reaches_the_fm_tab(qapp):
    """A running workflow owns the instrument; neither overview tab should be able to
    start competing for it."""
    from fibsem.ui.fm.overview_app import build_microscope

    host = _StubHost(microscope=build_microscope())
    host._build_fm_overview_widget()

    host._set_minimap_workflow_enabled(False)
    assert not host.fm_overview_widget.button_acquire.isEnabled()

    host._set_minimap_workflow_enabled(True)
    assert host.fm_overview_widget.button_acquire.isEnabled()

    host._teardown_fm_overview_widget()


def test_the_workflow_lock_survives_there_being_no_fm_tab(qapp):
    """A system with no fluorescence detector still runs workflows."""
    host = _StubHost(microscope=_plain_microscope())
    host._build_fm_overview_widget()

    host._set_minimap_workflow_enabled(False)  # must not raise


def test_the_fm_overview_tab_is_behind_a_preference(qapp):
    """Off by default, and hidden rather than absent -- which is what lets the
    preference take effect immediately instead of at the next launch."""
    from fibsem.ui.fm.overview_app import build_microscope

    host = _StubHost(microscope=build_microscope(), tab_enabled=False)
    index = host.tab_widget.indexOf(host.fm_overview_tab)

    assert not host.tab_widget.isTabVisible(index)
    assert host.fm_overview_widget is None


def test_switching_the_preference_on_shows_the_tab_without_a_restart(qapp):
    """The reason the tab is hidden rather than skipped."""
    from fibsem.ui.fm.overview_app import build_microscope

    host = _StubHost(microscope=build_microscope(), tab_enabled=False)
    index = host.tab_widget.indexOf(host.fm_overview_tab)

    host.set_flag(True)

    assert host.tab_widget.isTabVisible(index)
    assert host.tab_widget.isTabEnabled(index)
    assert isinstance(host.fm_overview_widget, FMOverviewWidget)

    host._teardown_fm_overview_widget()


def test_switching_it_off_again_releases_the_widget(qapp):
    """Hidden is not enough on its own: the widget subscribes to the microscope's stage
    signal for its lifetime, so one left behind keeps working for a feature that has
    been switched off."""
    from fibsem.ui.fm.overview_app import build_microscope

    microscope = build_microscope()
    before = len(microscope.stage_position_changed)
    host = _StubHost(microscope=microscope, tab_enabled=True)
    index = host.tab_widget.indexOf(host.fm_overview_tab)
    assert len(microscope.stage_position_changed) == before + 1

    host.set_flag(False)

    assert not host.tab_widget.isTabVisible(index)
    assert host.fm_overview_widget is None
    assert len(microscope.stage_position_changed) == before


def test_everything_tolerates_the_tab_being_switched_off(qapp):
    """The host calls into the tab from several places -- connection, experiment update,
    workflow lock -- and none of them should have to know whether it exists."""
    from fibsem.ui.fm.overview_app import build_microscope

    host = _StubHost(microscope=build_microscope(), tab_enabled=False)

    host._build_fm_overview_widget()           # must not raise
    host._update_fm_overview_experiment()      # must not raise
    host._set_minimap_workflow_enabled(False)  # must not raise
    host._apply_fm_overview_visibility()       # must not raise

    assert getattr(host, "fm_overview_widget", None) is None


def test_the_preference_reaches_the_config_flag():
    """`apply_feature_flags` is what turns the stored preference into the module
    global the UI reads, and a flag added to one without the other is silently dead."""
    import fibsem.config as fibsem_cfg

    prefs = fibsem_cfg.UserPreferences()
    prefs.features.fm_overview_tab = True
    previous = fibsem_cfg.FEATURE_FM_OVERVIEW_TAB_ENABLED
    try:
        fibsem_cfg.apply_feature_flags(prefs)
        assert fibsem_cfg.FEATURE_FM_OVERVIEW_TAB_ENABLED is True

        prefs.features.fm_overview_tab = False
        fibsem_cfg.apply_feature_flags(prefs)
        assert fibsem_cfg.FEATURE_FM_OVERVIEW_TAB_ENABLED is False
    finally:
        fibsem_cfg.FEATURE_FM_OVERVIEW_TAB_ENABLED = previous


def test_the_preference_survives_a_round_trip_through_yaml():
    """It is stored in user-preferences.yaml, so it has to serialise."""
    import fibsem.config as fibsem_cfg

    prefs = fibsem_cfg.UserPreferences()
    prefs.features.fm_overview_tab = True

    restored = fibsem_cfg.UserPreferences.from_dict(prefs.to_dict())

    assert restored.features.fm_overview_tab is True


# ── layout: the actions live in the column, not across the widget ────────


def test_a_long_message_does_not_widen_the_widget(qapp):
    """A `QLabel` reports the full width of its text as its size hint, and in a layout
    that hint is a minimum — so a 132-character acquisition failure used to drag the
    widget's minimum width from 1030 px to 1728 px and shove the window around."""
    widget = _fresh_widget(qapp)
    widget.resize(1200, 800)
    widget.show()
    qapp.processEvents()
    idle = widget.minimumSizeHint().width()

    widget.status.setText(
        "Failed: Acquisition grid extends beyond stage limits. 9 tile(s) out of "
        "bounds: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)"
    )
    qapp.processEvents()

    assert widget.minimumSizeHint().width() == idle

    widget.close()


def test_the_full_message_is_still_reachable(qapp):
    """Eliding is presentation. `text()` returns what was set — the widget compares
    against it to decide whether the status line is its to overwrite — and the tooltip
    carries the whole thing for anything too long to show."""
    widget = _fresh_widget(qapp)
    message = "Failed: " + "a very long explanation " * 8

    widget.status.setText(message)

    assert widget.status.text() == message
    assert widget.status.toolTip() == message

    widget.close()


def test_the_actions_sit_in_the_column_not_across_the_widget(qapp):
    """The floor was 1030 px, all of it this row — the canvas asked for 10. A row
    spanning the widget cannot give anything up; stacked in the settings column the
    floor is the column's own."""
    widget = _fresh_widget(qapp)
    widget.resize(1200, 800)
    widget.show()
    qapp.processEvents()

    assert widget.minimumSizeHint().width() < 600

    # ...and the actions are inside the right-hand pane, not siblings of the splitter
    assert widget.status_row.parentWidget() is not widget

    widget.close()


def test_the_progress_bars_appear_only_while_there_is_progress(qapp):
    """They are empty most of the time. Stacked in a column nothing sits beside them,
    so they can come and go without moving anything — which is what made this possible.
    """
    widget = _fresh_widget(qapp)
    widget.resize(1200, 800)
    widget.show()
    qapp.processEvents()

    assert not widget.progress_tiles.isVisible()
    assert not widget.progress_tile_detail.isVisible()

    widget._set_running(True)
    qapp.processEvents()

    assert widget.progress_tiles.isVisible()
    assert widget.progress_tile_detail.isVisible()

    widget._set_running(False)
    widget.close()


def test_a_finished_run_keeps_its_outcome_on_screen(qapp):
    """`FibsemProgressWidget` paints finished and failed differently, and that colour is
    the at-a-glance answer — hiding the bar the moment a run ends would throw it away.
    The per-tile bar does go, since no tile is in progress to describe."""
    widget = _fresh_widget(qapp)
    widget.resize(1200, 800)
    widget.show()
    qapp.processEvents()
    widget._set_running(True)

    widget._finish("overview-failed", "nope")
    qapp.processEvents()

    assert widget.progress_tiles.isVisible()
    assert not widget.progress_tile_detail.isVisible()

    widget.close()


def test_showing_the_bars_survives_their_own_reset(qapp):
    """`FibsemProgressWidget.reset()` hides itself, so a start sequence that shows the
    bars and then resets them leaves them hidden for the whole run."""
    widget = _fresh_widget(qapp)
    widget.resize(1200, 800)
    widget.show()
    qapp.processEvents()

    widget._set_running(True)  # resets both bars internally
    qapp.processEvents()

    assert widget.progress_tiles.isVisible()

    widget._set_running(False)
    widget.close()


# ── the canvas says where things are ─────────────────────────────────────


def test_the_cursor_readout_is_drawn_over_the_canvas(qapp, interactive_widget):
    """Not beside it. The only free corner is the top left — the toolbar owns the top
    right, the scalebar the bottom right, the stage info bar the bottom left."""
    widget = interactive_widget

    assert widget.cursor_readout.parent() is widget.canvas.canvas


def test_the_cursor_readout_hides_when_the_pointer_leaves(qapp, interactive_widget):
    """It sits on top of the image, so an empty plaque floating over the data is worse
    than nothing there."""
    widget = interactive_widget

    widget._on_cursor_moved(100.0, 100.0)
    qapp.processEvents()
    assert widget.cursor_readout.isVisible()
    assert widget.cursor_readout.text()

    widget._on_cursor_moved(None, None)
    qapp.processEvents()
    assert not widget.cursor_readout.isVisible()


def test_the_info_bar_states_the_stage_pose(qapp, interactive_widget):
    """The marker shows *where*; this says the numbers, which is what you need to write
    one down. The objective rides along because focus depends on it and it appears
    nowhere else on this tab."""
    widget = interactive_widget

    widget._refresh_current_position()
    qapp.processEvents()

    info = widget.canvas.canvas._info_text
    assert info
    assert "X:" in info and "Y:" in info
    assert "objective" in info


def test_the_info_bar_leaves_out_the_milling_angle(qapp, interactive_widget):
    """A beam-side quantity, meaningless while looking through the camera. It belongs
    on the FIB/SEM overview if anywhere."""
    widget = interactive_widget

    widget._refresh_current_position()

    assert "milling" not in (widget.canvas.canvas._info_text or "")


def test_an_unreadable_objective_does_not_cost_the_stage_position(qapp, interactive_widget):
    """A microscope with no objective, or one that will not answer, should lose that
    field and not the whole line."""
    widget = interactive_widget

    class _Broken:
        @property
        def position(self):
            raise RuntimeError("no objective here")

    original = widget.fm.objective
    widget.fm.objective = _Broken()
    try:
        # `_refresh_objective_info`, not `_refresh_stage_info`: the hardware read moved
        # there when it stopped happening on every stage poll (FIB-517). This is still
        # the same question -- a field that cannot be read is dropped, not the line.
        widget._refresh_objective_info()
        info = widget.canvas.canvas._info_text
    finally:
        widget.fm.objective = original

    assert info and "X:" in info
    assert "objective" not in info


def test_the_stage_marker_carries_no_caption(qapp, interactive_widget):
    """It sits on top of the data, and a caption riding along with the crosshair
    obscures the sample it is pointing at. The info bar says what it is instead."""
    widget = interactive_widget

    widget._refresh_current_position()

    assert widget.current_position_overlay._labels == [""]


# ── the canvas frame follows the stage's pose ────────────────────────────


def _microscope_at(tilt_deg: float):
    """A compustage microscope posed at a given tilt, with fluorescence attached."""
    import numpy as np

    from fibsem import utils
    from fibsem.fm.microscope import FluorescenceMicroscope
    from fibsem.structures import FibsemStagePosition

    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.stage_is_compustage = True
    microscope.system.stage.shuttle_pre_tilt = 0
    microscope._update_orientations()
    if microscope.fm is None:
        microscope.fm = FluorescenceMicroscope(parent=microscope)
    microscope.move_stage_absolute(
        FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=0.0, t=np.deg2rad(tilt_deg))
    )
    return microscope


def test_a_click_never_changes_the_stage_orientation(qapp):
    """The tab is built when a microscope connects, so the origin records whatever pose
    the stage was in then — usually t = 0. Moving to the FM position afterwards left
    every click carrying the old tilt, and the stage flipped 180 degrees to reach it.
    """
    import numpy as np

    microscope = _microscope_at(0.0)
    widget = FMOverviewWidget(microscope)
    widget._refresh_tile_grid()  # fixes the origin, at t = 0
    qapp.processEvents()
    assert np.rad2deg(widget.origin.t) == pytest.approx(0.0)

    microscope.move_to_microscope("FM")
    widget._on_stage_moved(microscope.get_stage_position())
    qapp.processEvents()

    target = widget._frame().to_stage(300.0, 200.0)

    assert np.rad2deg(target.t) == pytest.approx(-180.0)
    widget.close()


def test_a_stale_origin_resolves_a_click_to_the_same_place_as_a_fresh_one(qapp):
    """The tilt was the visible half. The projection's y/z split is computed from the
    pose too, so a click through a stale one also landed in the wrong *place* — with y
    inverted, which on a 180 degree flip is as wrong as it gets."""
    stale_microscope = _microscope_at(0.0)
    stale = FMOverviewWidget(stale_microscope)
    stale._refresh_tile_grid()
    stale_microscope.move_to_microscope("FM")
    stale._on_stage_moved(stale_microscope.get_stage_position())
    qapp.processEvents()

    fresh = FMOverviewWidget(_microscope_at(-180.0))
    fresh._refresh_tile_grid()
    qapp.processEvents()

    a = stale._frame().to_stage(300.0, 200.0)
    b = fresh._frame().to_stage(300.0, 200.0)

    assert (a.x, a.y, a.z) == pytest.approx((b.x, b.y, b.z), abs=1e-12)
    assert a.t == pytest.approx(b.t)

    stale.close()
    fresh.close()


def test_the_anchor_itself_does_not_move_with_the_pose(qapp):
    """Only the pose follows. Canvas zero stays where it was fixed, which is what lets
    images acquired at different times accumulate against one frame."""
    microscope = _microscope_at(0.0)
    widget = FMOverviewWidget(microscope)
    widget._refresh_tile_grid()
    qapp.processEvents()
    anchor = (widget.origin.x, widget.origin.y, widget.origin.z)

    microscope.move_to_microscope("FM")
    widget._on_stage_moved(microscope.get_stage_position())
    qapp.processEvents()

    assert (widget.origin.x, widget.origin.y, widget.origin.z) == pytest.approx(anchor)
    widget.close()


def test_an_acquired_image_lands_where_the_grid_that_planned_it_is_drawn(qapp):
    """The invariant that ties the two halves together.

    Images are placed through their own recorded geometry and the grid through the live
    one, which is right — an image may have been taken under a configuration the
    instrument has since left. But both must measure from the *same* base. Placing
    images against the raw origin while the grid used the current pose put a mosaic
    300 µm from the grid that planned it, mirrored in y.
    """
    import numpy as np

    from fibsem.fm.structures import ChannelSettings
    from fibsem.structures import FibsemStagePosition

    microscope = _microscope_at(0.0)
    widget = FMOverviewWidget(microscope)
    widget._refresh_tile_grid()  # origin fixed at t = 0
    qapp.processEvents()

    microscope.move_to_microscope("FM")
    microscope.move_stage_absolute(
        FibsemStagePosition(x=200e-6, y=150e-6, z=0.0, r=0.0, t=np.deg2rad(-180))
    )
    widget._on_stage_moved(microscope.get_stage_position())
    qapp.processEvents()

    image = microscope.fm.acquire_image(ChannelSettings(name="Channel-01"))
    image.metadata.stage_position = microscope.get_stage_position()

    assert widget._offset_of(image) == pytest.approx(widget._grid_offset(), abs=1e-12)

    widget.close()


def test_the_placement_of_an_image_matches_a_widget_that_never_moved(qapp):
    """Same equivalence as the click test, for placement: whatever the tab's history,
    an image belongs in one place."""
    import numpy as np

    from fibsem.fm.structures import ChannelSettings
    from fibsem.structures import FibsemStagePosition

    somewhere = FibsemStagePosition(
        x=200e-6, y=150e-6, z=0.0, r=0.0, t=np.deg2rad(-180)
    )

    stale_microscope = _microscope_at(0.0)
    stale = FMOverviewWidget(stale_microscope)
    stale._refresh_tile_grid()
    stale_microscope.move_to_microscope("FM")
    stale._on_stage_moved(stale_microscope.get_stage_position())

    fresh_microscope = _microscope_at(-180.0)
    fresh = FMOverviewWidget(fresh_microscope)
    fresh._refresh_tile_grid()
    qapp.processEvents()

    image = fresh_microscope.fm.acquire_image(ChannelSettings(name="Channel-01"))
    image.metadata.stage_position = somewhere

    assert stale._offset_of(image) == pytest.approx(fresh._offset_of(image), abs=1e-12)

    stale.close()
    fresh.close()


# ── marked positions ─────────────────────────────────────────────────────


def _named(name: str, x: float, y: float, tilt_deg: float = -180.0):
    import numpy as np

    from fibsem.structures import FibsemStagePosition

    position = FibsemStagePosition(
        x=x, y=y, z=0.0, r=0.0, t=np.deg2rad(tilt_deg)
    )
    position.name = name
    return position


def test_marked_positions_are_drawn_where_their_stage_positions_are(qapp):
    widget = _fresh_widget(qapp)
    widget._refresh_tile_grid()
    qapp.processEvents()

    widget.set_positions([_named("Lamella-01", 100e-6, 50e-6)])

    frame = widget._frame()
    expected = frame.to_canvas(_named("Lamella-01", 100e-6, 50e-6))
    assert widget.position_overlay._points == [pytest.approx(expected)]
    assert widget.position_overlay._labels == ["Lamella-01"]

    widget.close()


def test_the_selected_position_is_drawn_on_its_own_layer(qapp):
    """`PointsOverlay` paints every point the same colour, so one selected marker among
    many needs a layer of its own rather than per-point colours."""
    widget = _fresh_widget(qapp)
    widget._refresh_tile_grid()
    widget.set_positions(
        [_named("Lamella-01", 100e-6, 50e-6), _named("Lamella-02", -80e-6, 20e-6)]
    )
    qapp.processEvents()

    widget.set_selected_position("Lamella-02")

    assert widget.position_overlay._labels == ["Lamella-01"]
    assert widget.selected_position_overlay._labels == ["Lamella-02"]

    widget.close()


def test_selecting_nothing_puts_every_position_back(qapp):
    widget = _fresh_widget(qapp)
    widget._refresh_tile_grid()
    widget.set_positions(
        [_named("Lamella-01", 100e-6, 50e-6), _named("Lamella-02", -80e-6, 20e-6)]
    )
    widget.set_selected_position("Lamella-01")
    qapp.processEvents()

    widget.set_selected_position(None)

    assert sorted(widget.position_overlay._labels) == ["Lamella-01", "Lamella-02"]
    assert widget.selected_position_overlay._points == []

    widget.close()


def test_a_selection_that_names_nothing_marked_highlights_nothing(qapp):
    """The host's list and this one are rebuilt independently from one experiment, so a
    name can arrive for a lamella this canvas cannot place."""
    widget = _fresh_widget(qapp)
    widget._refresh_tile_grid()
    widget.set_positions([_named("Lamella-01", 100e-6, 50e-6)])
    qapp.processEvents()

    widget.set_selected_position("Lamella-99")

    assert widget.position_overlay._labels == ["Lamella-01"]
    assert widget.selected_position_overlay._points == []

    widget.close()


def test_the_selection_survives_the_positions_being_rebuilt(qapp):
    """Held by name rather than index for exactly this: the host rebuilds the list on
    every experiment change, and an index would point at a different lamella the moment
    one was removed."""
    widget = _fresh_widget(qapp)
    widget._refresh_tile_grid()
    widget.set_positions(
        [_named("Lamella-01", 100e-6, 50e-6), _named("Lamella-02", -80e-6, 20e-6)]
    )
    widget.set_selected_position("Lamella-02")
    qapp.processEvents()

    # Lamella-01 removed; the selected one is now at a different index.
    widget.set_positions([_named("Lamella-02", -80e-6, 20e-6)])
    qapp.processEvents()

    assert widget.selected_position_overlay._labels == ["Lamella-02"]
    assert widget.position_overlay._points == []

    widget.close()


def _lamella_with_poses(name: str, microscope, tmp_path, x: float, y: float):
    """A lamella carrying both poses, built the way the app builds them.

    A real `Lamella`, not a stand-in with the two attributes this used to need. The
    settings column now holds a `LamellaNameListWidget`, whose rows subscribe to
    `lamella.events.description` and read the task and defect state -- which is exactly
    why that list cannot live inside `FMOverviewWidget`, and why the tab that owns it is
    in the autolamella package.
    """
    from fibsem.applications.autolamella.poses import (
        MILLING_ORIENTATION,
        build_lamella_poses,
    )
    from fibsem.applications.autolamella.structures import Lamella
    from fibsem.structures import FibsemStagePosition

    beam = microscope.get_orientation(MILLING_ORIENTATION)
    marked = FibsemStagePosition(x=x, y=y, z=0.0, r=beam.r, t=beam.t)
    poses = build_lamella_poses(microscope, marked)
    lamella = Lamella(petname=name, path=str(tmp_path / name), number=1)
    lamella.milling_pose = poses.milling
    lamella.fluorescence_pose = poses.fluorescence
    return lamella


def test_the_host_marks_lamellae_by_their_fluorescence_pose(qapp, tmp_path):
    """The trap this half exists to avoid. A lamella carries a milling pose and a
    fluorescence pose, and on a compustage the stage flips 180 degrees between them —
    so marking the milling pose would put every lamella somewhere the sample is not.
    """
    from fibsem.ui.fm.overview_app import build_microscope

    host = _StubHost(microscope=build_microscope())
    host._build_fm_overview_widget()
    widget = host.fm_overview_widget
    microscope = host.autolamella_ui.microscope

    lamella = _lamella_with_poses("Lamella-01", microscope, tmp_path, 100e-6, 50e-6)
    host.autolamella_ui.experiment = type(
        "_Exp", (), {"path": "/tmp", "positions": [lamella]}
    )()

    host._update_fm_overview_positions()

    marked = widget._positions[0]
    assert marked.t == pytest.approx(lamella.fluorescence_pose.stage_position.t)
    assert marked.t != pytest.approx(lamella.milling_pose.stage_position.t)
    assert marked.name == "Lamella-01"

    host._teardown_fm_overview_widget()


def test_a_lamella_with_no_fluorescence_pose_is_skipped_not_misplaced(qapp, tmp_path):
    """It cannot be placed on this canvas at all — normal on a system with no FM, and
    possible on an older experiment. Better absent than drawn somewhere invented."""
    from fibsem.ui.fm.overview_app import build_microscope

    host = _StubHost(microscope=build_microscope())
    host._build_fm_overview_widget()
    microscope = host.autolamella_ui.microscope

    placeable = _lamella_with_poses("Lamella-01", microscope, tmp_path, 100e-6, 50e-6)
    orphan = _lamella_with_poses("Lamella-02", microscope, tmp_path, 0.0, 0.0)
    del orphan.poses["FLUORESCENCE"]
    host.autolamella_ui.experiment = type(
        "_Exp", (), {"path": "/tmp", "positions": [placeable, orphan]}
    )()

    host._update_fm_overview_positions()

    assert [p.name for p in host.fm_overview_widget._positions] == ["Lamella-01"]

    host._teardown_fm_overview_widget()


def test_no_experiment_clears_the_marked_positions(qapp, tmp_path):
    from fibsem.ui.fm.overview_app import build_microscope

    host = _StubHost(microscope=build_microscope())
    host._build_fm_overview_widget()
    microscope = host.autolamella_ui.microscope
    host.autolamella_ui.experiment = type(
        "_Exp", (), {
            "path": "/tmp",
            "positions": [_lamella_with_poses("Lamella-01", microscope, tmp_path, 100e-6, 50e-6)],
        },
    )()
    host._update_fm_overview_positions()
    assert host.fm_overview_widget._positions

    host.autolamella_ui.experiment = None
    host._update_fm_overview_positions()

    assert host.fm_overview_widget._positions == []

    host._teardown_fm_overview_widget()


def test_selection_reaches_the_overview_whichever_list_it_came_from(qapp):
    """The three selection handlers guard against selecting each other in circles. This
    highlight emits nothing, so it must run whichever list the selection came from —
    which is exactly what that guard suppresses."""
    from fibsem.ui.fm.overview_app import build_microscope

    host = _StubHost(microscope=build_microscope())
    host._build_fm_overview_widget()
    lamella = type("_Lamella", (), {"name": "Lamella-07"})()

    host._syncing_selection = True  # as it is while the lists sync each other
    host._update_fm_overview_selection(lamella)

    assert host.fm_overview_widget.selected_position == "Lamella-07"

    host._teardown_fm_overview_widget()


def test_the_host_tolerates_no_fm_overview_tab(qapp):
    """Both updates are called from paths that run on every system, including those
    with the tab switched off or no fluorescence detector at all."""
    host = _StubHost(microscope=_plain_microscope())
    host._apply_fm_overview_visibility()

    host._update_fm_overview_positions()   # must not raise
    host._update_fm_overview_selection(None)  # must not raise


# ── right-clicking to place a position ───────────────────────────────────


def _menu_at_the_stage(widget, dx: float = 0.0, dy: float = 0.0):
    """The context menu for a point on the canvas, offset from where the stage is."""
    frame = widget._frame()
    point = frame.to_canvas(widget._current_stage_position())
    return widget._position_menu(point[0] + dx, point[1] + dy)


def _labels(config):
    return [action.label for action in config.actions]


def _fire(config, index: int):
    """Trigger a menu entry the way `ContextMenu` does, without opening one.

    `show_at_cursor` runs a modal event loop, so a test that opened the menu would hang
    rather than fail.
    """
    config.actions[index].callback()


def test_the_canvas_right_click_reaches_the_menu(qapp, interactive_widget, monkeypatch):
    """The one link the rest of these tests step over, by calling `_position_menu`
    directly -- they have to, because opening the menu runs a modal loop. Returning None
    from it here keeps that loop shut while still proving the signal arrives."""
    widget = interactive_widget
    seen = []
    monkeypatch.setattr(
        widget, "_position_menu", lambda x, y: seen.append((x, y)) or None
    )

    widget.canvas.canvas.canvas_right_clicked.emit(120.0, 340.0, None)
    qapp.processEvents()

    assert seen == [(120.0, 340.0)]


def test_right_clicking_offers_to_put_a_position_there(qapp, interactive_widget):
    widget = interactive_widget
    received = []
    widget.position_add_requested.connect(lambda p: received.append(p))

    config = _menu_at_the_stage(widget, dx=800.0, dy=-400.0)
    _fire(config, 0)

    assert _labels(config)[0] == "Add Position Here"
    assert len(received) == 1
    # and it names the point that was clicked, not where the stage happens to be
    frame = widget._frame()
    assert frame.to_canvas(received[0]) == pytest.approx(
        frame.to_canvas(widget._current_stage_position()) + np.array([800.0, -400.0]),
        abs=1e-6,
    )


def test_the_position_offered_is_the_one_a_double_click_would_move_to(
    qapp, interactive_widget, monkeypatch
):
    """Both gestures resolve a canvas point through the same code, so marking somewhere
    and driving there cannot drift apart -- which they would, eventually, being the same
    arithmetic written twice."""
    from fibsem.ui.fm.widgets import fm_overview_widget as module

    widget = interactive_widget
    monkeypatch.setattr(module, "FunctionWorker", _SynchronousWorker)
    frame = widget._frame()
    point = frame.to_canvas(widget._current_stage_position())
    x, y = point[0] + 1500.0, point[1] + 900.0

    received = []
    widget.position_add_requested.connect(lambda p: received.append(p))
    _fire(_menu_at_the_stage(widget, dx=1500.0, dy=900.0), 0)

    widget._on_canvas_double_clicked(x, y, None)
    qapp.processEvents()
    moved_to = widget.microscope.get_stage_position()

    assert received[0].x == pytest.approx(moved_to.x, abs=1e-9)
    assert received[0].y == pytest.approx(moved_to.y, abs=1e-9)


def test_the_marked_position_carries_the_fluorescence_pose(qapp, interactive_widget):
    """It becomes a lamella's fluorescence pose, so its rotation and tilt have to be the
    fluorescence ones. They come from wherever the stage is -- which is why the gesture
    is refused anywhere else."""
    widget = interactive_widget
    received = []
    widget.position_add_requested.connect(lambda p: received.append(p))

    _fire(_menu_at_the_stage(widget, dx=300.0), 0)

    assert widget.microscope.get_stage_orientation(received[0]) == "FM"


def test_moving_is_offered_only_once_something_is_selected(qapp, interactive_widget):
    widget = interactive_widget
    widget.set_selected_position(None)

    assert _labels(_menu_at_the_stage(widget)) == ["Add Position Here"]

    widget.set_positions([_named("Lamella-04", 0.0, 0.0)])
    widget.set_selected_position("Lamella-04")

    assert _labels(_menu_at_the_stage(widget)) == [
        "Add Position Here",
        "Move Selected Position Here (Lamella-04)",
    ]

    widget.set_selected_position(None)
    widget.set_positions([])


def test_moving_requests_the_selected_name_and_the_clicked_point(qapp, interactive_widget):
    widget = interactive_widget
    widget.set_positions([_named("Lamella-04", 0.0, 0.0)])
    widget.set_selected_position("Lamella-04")
    received = []
    widget.position_move_requested.connect(lambda n, p: received.append((n, p)))

    _fire(_menu_at_the_stage(widget, dx=-600.0, dy=250.0), 1)

    assert len(received) == 1
    name, position = received[0]
    assert name == "Lamella-04"
    frame = widget._frame()
    assert frame.to_canvas(position) == pytest.approx(
        frame.to_canvas(widget._current_stage_position()) + np.array([-600.0, 250.0]),
        abs=1e-6,
    )

    widget.set_selected_position(None)
    widget.set_positions([])


def test_nothing_is_offered_outside_the_stage_limits(qapp, interactive_widget):
    """The stage cannot go there, so it is neither somewhere to move nor somewhere to
    put a lamella."""
    widget = interactive_widget
    limits = widget.microscope._stage.limits
    frame = widget._frame()
    # A point comfortably past the x limit, in canvas units.
    beyond = frame.length((limits["x"].max - limits["x"].min) * 2)

    assert _menu_at_the_stage(widget, dx=beyond) is None


def test_nothing_is_offered_during_an_acquisition(qapp, interactive_widget):
    """A host that has taken the instrument is usually a workflow iterating the
    experiment's positions, and adding one underneath it is not a thing to do quietly."""
    widget = interactive_widget
    widget._set_running(True)
    try:
        assert _menu_at_the_stage(widget) is None
    finally:
        widget._set_running(False)


def test_nothing_is_offered_while_a_host_holds_the_tab(qapp, interactive_widget):
    widget = interactive_widget
    widget.set_interactive(False)
    try:
        assert _menu_at_the_stage(widget) is None
    finally:
        widget.set_interactive(True)


def test_nothing_is_offered_from_a_beam_orientation(qapp):
    """The canvas frame takes its rotation and tilt from wherever the stage is, so a
    point resolved at a beam orientation comes back carrying *beam* rotation and tilt on
    fluorescence-side coordinates -- a fluorescence pose that is not in the fluorescence
    orientation."""
    from fibsem.structures import FibsemStagePosition

    microscope = _microscope_at(-180.0)  # FM, on a compustage
    widget = FMOverviewWidget(microscope)
    widget._refresh_tile_grid()
    qapp.processEvents()
    assert _menu_at_the_stage(widget) is not None

    milling = microscope.get_orientation("MILLING")
    microscope.move_stage_absolute(
        FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=milling.r, t=milling.t)
    )
    widget._on_stage_moved(microscope.get_stage_position())
    qapp.processEvents()

    assert _menu_at_the_stage(widget) is None

    widget.close()


# ── the host turns a request into a lamella ──────────────────────────────


def _experiment_with(positions, tmp_path):
    """An experiment stub that records saves, for the host handlers."""
    class _Exp:
        def __init__(self):
            self.path = str(tmp_path)
            self.positions = list(positions)
            self.saves = 0

        def save(self):
            self.saves += 1

    return _Exp()


def _wired_host(qapp, tmp_path, positions=()):
    """A `_StubHost` with an experiment and a recording AutoLamella UI."""
    from fibsem.ui.fm.overview_app import build_microscope

    host = _StubHost(microscope=build_microscope())
    host._build_fm_overview_widget()
    host.autolamella_ui.experiment = _experiment_with(positions, tmp_path)
    host.autolamella_ui.added = []
    host.autolamella_ui.updated = 0

    def add_new_lamella(stage_position=None, name=None, objective_position=None,
                        marked_at=None):
        host.autolamella_ui.added.append(
            {"position": stage_position, "objective_position": objective_position,
             "marked_at": marked_at}
        )
        lamella = type("_Lamella", (), {"name": f"Lamella-{len(host.autolamella_ui.added):02d}"})()
        return lamella

    def update_ui():
        host.autolamella_ui.updated += 1

    host.autolamella_ui.add_new_lamella = add_new_lamella
    host.autolamella_ui.update_ui = update_ui
    return host


def _real_lamella(name, microscope, tmp_path, x=100e-6, y=50e-6):
    """A real `Lamella` with both poses, so the property setters behave as they do live."""
    from fibsem.applications.autolamella.poses import (
        MILLING_ORIENTATION,
        build_lamella_poses,
    )
    from fibsem.applications.autolamella.structures import Lamella
    from fibsem.structures import FibsemStagePosition

    beam = microscope.get_orientation(MILLING_ORIENTATION)
    poses = build_lamella_poses(
        microscope,
        FibsemStagePosition(x=x, y=y, z=0.0, r=beam.r, t=beam.t),
    )
    lamella = Lamella(petname=name, path=str(tmp_path / name), number=1)
    lamella.milling_pose = poses.milling
    lamella.fluorescence_pose = poses.fluorescence
    return lamella


def test_adding_declares_the_fluorescence_orientation(qapp, tmp_path):
    """Not left to be derived. On a compustage the answer would be the same; on an
    offset mount it would not, and the wrong answer there is a lamella with a milling
    pose 48 mm off the beam axis that nothing rejects until something tries to mill it
    (FIB-93). Declaring it turns that into a refusal with a user to tell."""
    from fibsem.applications.autolamella.poses import FLUORESCENCE_ORIENTATION

    host = _wired_host(qapp, tmp_path)
    position = _named("wherever", 120e-6, -60e-6)

    host.fm_overview_tab._on_add_requested(position)

    assert host.autolamella_ui.added[0]["marked_at"] == FLUORESCENCE_ORIENTATION
    assert host.autolamella_ui.added[0]["position"] is position

    host._teardown_fm_overview_widget()


def test_adding_records_where_the_objective_actually_is(qapp, tmp_path):
    """Rather than the objective's configured focus position, which `build_lamella_poses`
    falls back to: that is a property of the instrument and says nothing about this
    sample. Whoever marked this had the feature in focus."""
    host = _wired_host(qapp, tmp_path)
    objective = host.autolamella_ui.microscope.fm.objective
    objective.focus_position = 1.0e-3
    objective.insert()
    objective.move_absolute(objective.position + 50e-6)  # focused by hand
    assert objective.state == "Inserted"

    host.fm_overview_tab._on_add_requested(_named("wherever", 0.0, 0.0))

    assert host.autolamella_ui.added[0]["objective_position"] == pytest.approx(
        objective.position
    )
    assert host.autolamella_ui.added[0]["objective_position"] != pytest.approx(1.0e-3)


def test_a_retracted_objective_is_not_recorded_as_a_focus(qapp, tmp_path):
    """It is parked ~10 mm from anything and nobody focused on it. None hands the
    decision to `build_lamella_poses`' fallback, because `fluorescence_selected` only
    asks whether an objective position exists -- a parked one would read as focused."""
    host = _wired_host(qapp, tmp_path)
    objective = host.autolamella_ui.microscope.fm.objective
    objective.retract()
    assert objective.state == "Retracted"

    host.fm_overview_tab._on_add_requested(_named("wherever", 0.0, 0.0))

    assert host.autolamella_ui.added[0]["objective_position"] is None

    host._teardown_fm_overview_widget()

    host._teardown_fm_overview_widget()


def test_adding_without_an_experiment_is_refused_not_crashed(qapp, tmp_path):
    host = _wired_host(qapp, tmp_path)
    host.autolamella_ui.experiment = None

    host.fm_overview_tab._on_add_requested(_named("wherever", 0.0, 0.0))  # must not raise

    assert host.autolamella_ui.added == []

    host._teardown_fm_overview_widget()


def test_a_refused_add_is_reported_rather_than_raised(qapp, tmp_path):
    """The offset-mount refusal arrives here as a ValueError from `build_lamella_poses`,
    and it is a message for the user, not a traceback."""
    host = _wired_host(qapp, tmp_path)

    def refuse(**kwargs):
        raise ValueError("no transform between the fluorescence and beam positions")

    host.autolamella_ui.add_new_lamella = refuse

    host.fm_overview_tab._on_add_requested(_named("wherever", 0.0, 0.0))  # must not raise

    host._teardown_fm_overview_widget()


def test_moving_asks_before_it_moves_anything(qapp, tmp_path, monkeypatch):
    """Moving one pose moves both, and the milling pose is not visible from this canvas.
    Said rather than assumed."""
    from fibsem.applications.autolamella.ui import autolamella_fluorescence_overview_tab as module
    from fibsem.ui.fm.overview_app import build_microscope

    host = _wired_host(qapp, tmp_path)
    microscope = host.autolamella_ui.microscope
    lamella = _real_lamella("Lamella-01", microscope, tmp_path)
    host.autolamella_ui.experiment.positions = [lamella]
    before = deepcopy(lamella.poses)

    asked = []
    monkeypatch.setattr(
        module, "message_box_ui", lambda **kwargs: asked.append(kwargs) or False
    )

    host.fm_overview_tab._on_move_requested("Lamella-01", _named("target", 400e-6, -200e-6))

    assert len(asked) == 1, "moved without asking"
    assert lamella.poses["MILLING"].stage_position.x == before["MILLING"].stage_position.x
    assert (
        lamella.poses["FLUORESCENCE"].stage_position.x
        == before["FLUORESCENCE"].stage_position.x
    )
    assert host.autolamella_ui.experiment.saves == 0

    host._teardown_fm_overview_widget()
    _ = build_microscope  # keep the import meaningful under -W error


def test_confirming_a_move_moves_both_poses(qapp, tmp_path, monkeypatch):
    from fibsem.applications.autolamella.ui import autolamella_fluorescence_overview_tab as module

    host = _wired_host(qapp, tmp_path)
    microscope = host.autolamella_ui.microscope
    lamella = _real_lamella("Lamella-01", microscope, tmp_path)
    host.autolamella_ui.experiment.positions = [lamella]
    monkeypatch.setattr(module, "message_box_ui", lambda **kwargs: True)

    target = _named("target", 400e-6, -200e-6)
    host.fm_overview_tab._on_move_requested("Lamella-01", target)

    # the fluorescence pose is where the user clicked...
    assert lamella.fluorescence_pose.stage_position.x == pytest.approx(400e-6)
    assert lamella.fluorescence_pose.stage_position.y == pytest.approx(-200e-6)
    # ...and the milling pose followed it, into the milling orientation
    assert (
        microscope.get_stage_orientation(lamella.milling_pose.stage_position)
        == "MILLING"
    )
    assert lamella.milling_pose.stage_position.x == pytest.approx(400e-6)
    assert host.autolamella_ui.experiment.saves == 1

    host._teardown_fm_overview_widget()


def test_a_move_rewrites_the_stage_positions_and_nothing_else(qapp, tmp_path, monkeypatch):
    """A move is a move. The poses are edited in place rather than replaced, so the beam
    settings the lamella was marked with survive -- and so does the objective position,
    because the user moved sideways, they did not refocus."""
    from fibsem.applications.autolamella.ui import autolamella_fluorescence_overview_tab as module

    host = _wired_host(qapp, tmp_path)
    microscope = host.autolamella_ui.microscope
    lamella = _real_lamella("Lamella-01", microscope, tmp_path)
    lamella.milling_pose.electron_detector.brightness = 0.123
    lamella.fluorescence_pose.objective_position = 7.7e-3
    host.autolamella_ui.experiment.positions = [lamella]
    monkeypatch.setattr(module, "message_box_ui", lambda **kwargs: True)

    host.fm_overview_tab._on_move_requested("Lamella-01", _named("target", 400e-6, -200e-6))

    assert lamella.milling_pose.electron_detector.brightness == pytest.approx(0.123)
    assert lamella.fluorescence_pose.objective_position == pytest.approx(7.7e-3)


def test_moving_a_lamella_that_has_no_fluorescence_pose_gives_it_one(
    qapp, tmp_path, monkeypatch
):
    """Possible on a lamella marked before there was an FM. It gets a whole new pose
    rather than an edit, and that one is built on what the lamella itself recorded --
    not on whatever the microscope happens to be set to now."""
    from fibsem.applications.autolamella.ui import autolamella_fluorescence_overview_tab as module

    host = _wired_host(qapp, tmp_path)
    microscope = host.autolamella_ui.microscope
    lamella = _real_lamella("Lamella-01", microscope, tmp_path)
    lamella.milling_pose.electron_detector.brightness = 0.123
    del lamella.poses["FLUORESCENCE"]
    host.autolamella_ui.experiment.positions = [lamella]
    monkeypatch.setattr(module, "message_box_ui", lambda **kwargs: True)

    host.fm_overview_tab._on_move_requested("Lamella-01", _named("target", 400e-6, -200e-6))

    assert lamella.fluorescence_pose is not None
    assert lamella.fluorescence_pose.stage_position.x == pytest.approx(400e-6)
    # built on the lamella's own state, not the live microscope's
    assert lamella.fluorescence_pose.electron_detector.brightness == pytest.approx(0.123)

    host._teardown_fm_overview_widget()


def test_a_move_re_derives_the_milling_angle(qapp, tmp_path, monkeypatch):
    """It is computed from the milling-pose stage tilt, so a pose that moved without it
    would leave the lamella claiming an angle its own pose no longer implies."""
    from fibsem.applications.autolamella.ui import autolamella_fluorescence_overview_tab as module

    host = _wired_host(qapp, tmp_path)
    microscope = host.autolamella_ui.microscope
    lamella = _real_lamella("Lamella-01", microscope, tmp_path)
    lamella.milling_angle = 999.0
    host.autolamella_ui.experiment.positions = [lamella]
    monkeypatch.setattr(module, "message_box_ui", lambda **kwargs: True)

    host.fm_overview_tab._on_move_requested("Lamella-01", _named("target", 400e-6, -200e-6))

    assert lamella.milling_angle != pytest.approx(999.0)
    assert lamella.milling_angle == pytest.approx(
        microscope.get_current_milling_angle(stage_position=lamella.stage_position)
    )

    host._teardown_fm_overview_widget()


def test_a_move_that_names_nothing_does_nothing(qapp, tmp_path, monkeypatch):
    """The canvas holds its selection by name and is rebuilt independently of the host's
    list, so a name can arrive for a lamella that is no longer there."""
    from fibsem.applications.autolamella.ui import autolamella_fluorescence_overview_tab as module

    host = _wired_host(qapp, tmp_path)
    lamella = _real_lamella("Lamella-01", host.autolamella_ui.microscope, tmp_path)
    host.autolamella_ui.experiment.positions = [lamella]
    asked = []
    monkeypatch.setattr(
        module, "message_box_ui", lambda **kwargs: asked.append(kwargs) or True
    )

    host.fm_overview_tab._on_move_requested("Lamella-99", _named("target", 0.0, 0.0))

    assert asked == [], "asked about a lamella that is not there"
    assert host.autolamella_ui.experiment.saves == 0

    host._teardown_fm_overview_widget()


# ── positions marked anywhere reach the FM canvas ────────────────────────


class _ListStub:
    """Stands in for the lamella list and card container in `_rebuild_lamella_list`."""

    def __init__(self):
        self.lamellae = []

    def clear(self):
        self.lamellae.clear()

    def add_lamella(self, lamella):
        self.lamellae.append(lamella)


def _list_host(qapp, tmp_path, experiment=None):
    """A `_StubHost` that can run the real `_rebuild_lamella_list`."""
    from fibsem.ui.fm.overview_app import build_microscope

    host = _StubHost(microscope=build_microscope())
    host._build_fm_overview_widget()
    host.autolamella_ui.experiment = experiment
    host.lamella_list_widget = _ListStub()
    host.lamella_card_container = _ListStub()
    host._lamella_list_experiment = None
    host._on_lamella_card_selected = lambda lamella: None
    host._on_workflow_selection_changed = lambda: None
    host._update_lamella_tab_enabled = lambda: None
    return host


def _real_experiment(tmp_path, name="fm-overview-test"):
    from fibsem.applications.autolamella.structures import Experiment

    return Experiment(path=tmp_path, name=name)


def test_a_lamella_added_anywhere_reaches_the_fm_canvas(qapp, tmp_path):
    """The reported bug: a position marked on the FIB/SEM overview never appeared on the
    FM one. Nothing was wrong with the marking -- the canvas was simply never told, so it
    kept showing whatever it had last been handed.

    Appending is how every add arrives, whatever marked it: the minimap calls
    `add_new_lamella` and trusts `inserted` to refresh the displays.
    """
    experiment = _real_experiment(tmp_path)
    host = _list_host(qapp, tmp_path, experiment)
    host._wire_position_events()
    assert host.fm_overview_widget._positions == []

    microscope = host.autolamella_ui.microscope
    lamella = _real_lamella("Lamella-01", microscope, tmp_path)
    experiment.positions.append(lamella)  # nobody calls the FM refresh by hand

    assert [p.name for p in host.fm_overview_widget._positions] == ["Lamella-01"]
    assert host.lamella_list_widget.lamellae == [lamella]

    host._teardown_fm_overview_widget()


def test_removing_a_lamella_takes_its_marker_away(qapp, tmp_path):
    experiment = _real_experiment(tmp_path)
    host = _list_host(qapp, tmp_path, experiment)
    host._wire_position_events()
    microscope = host.autolamella_ui.microscope
    experiment.positions.append(_real_lamella("Lamella-01", microscope, tmp_path))
    assert host.fm_overview_widget._positions

    experiment.positions.pop()

    assert host.fm_overview_widget._positions == []

    host._teardown_fm_overview_widget()


def test_editing_a_position_in_place_re_marks_the_canvas(qapp, tmp_path):
    """`update_lamella_position_ui` replaces a milling pose and emits `changed` -- no
    insert, no removal, and the list rows have nothing to redraw. The canvas does."""
    experiment = _real_experiment(tmp_path)
    host = _list_host(qapp, tmp_path, experiment)
    host._wire_position_events()
    microscope = host.autolamella_ui.microscope
    lamella = _real_lamella("Lamella-01", microscope, tmp_path)
    experiment.positions.append(lamella)
    before = host.fm_overview_widget._positions[0].x

    lamella.fluorescence_pose.stage_position.x = before + 500e-6
    experiment.positions.events.changed.emit()

    assert host.fm_overview_widget._positions[0].x == pytest.approx(before + 500e-6)

    host._teardown_fm_overview_widget()


def test_closing_an_experiment_clears_the_canvas(qapp, tmp_path):
    """Before the `experiment is None` return in the rebuild, because an experiment
    closing has to clear the canvas as much as one opening has to fill it."""
    experiment = _real_experiment(tmp_path)
    host = _list_host(qapp, tmp_path, experiment)
    microscope = host.autolamella_ui.microscope
    experiment.positions.append(_real_lamella("Lamella-01", microscope, tmp_path))
    host._rebuild_lamella_list()
    assert host.fm_overview_widget._positions

    host.autolamella_ui.experiment = None
    host._rebuild_lamella_list()

    assert host.fm_overview_widget._positions == []

    host._teardown_fm_overview_widget()


def test_switching_experiments_lets_go_of_the_old_one(qapp, tmp_path):
    """The disconnect has to actually disconnect. It did not: the connections were
    lambdas and the disconnects named the methods those lambdas called, which psygnal
    accepts and silently ignores.

    Asserted on the subscriber count rather than on what gets drawn, because drawing
    cannot tell the difference -- `_rebuild_lamella_list` reads the *current* experiment,
    so a leaked subscription only does redundant work. The claim being tested is the one
    the code makes, which is that it stops following the experiment it is leaving.
    """
    first = _real_experiment(tmp_path / "one", name="one")
    host = _list_host(qapp, tmp_path, first)
    host._wire_position_events()
    assert len(first.positions.events.inserted) == 1

    second = _real_experiment(tmp_path / "two", name="two")
    host.autolamella_ui.experiment = second
    host._wire_position_events()

    for signal in ("inserted", "removed", "changed"):
        assert len(getattr(first.positions.events, signal)) == 0, (
            f"still subscribed to the old experiment's {signal}"
        )
        assert len(getattr(second.positions.events, signal)) == 1

    host._teardown_fm_overview_widget()


def test_a_confirmed_move_re_marks_the_canvas(qapp, tmp_path, monkeypatch):
    """Otherwise the marker stays where the lamella used to be until something else
    happens to refresh it."""
    from fibsem.applications.autolamella.ui import autolamella_fluorescence_overview_tab as module

    host = _wired_host(qapp, tmp_path)
    microscope = host.autolamella_ui.microscope
    lamella = _real_lamella("Lamella-01", microscope, tmp_path)
    host.autolamella_ui.experiment.positions = [lamella]
    host._update_fm_overview_positions()
    before = host.fm_overview_widget._positions[0].x
    monkeypatch.setattr(module, "message_box_ui", lambda **kwargs: True)

    host.fm_overview_tab._on_move_requested("Lamella-01", _named("target", 400e-6, -200e-6))

    assert host.fm_overview_widget._positions[0].x == pytest.approx(400e-6)
    assert host.fm_overview_widget._positions[0].x != pytest.approx(before)

    host._teardown_fm_overview_widget()


# ── the lamella list in the settings column ──────────────────────────────


def _tab_with(qapp, tmp_path, positions=()):
    """A `AutoLamellaFluorescenceOverviewTab` with a live overview and an experiment."""
    import fibsem.config as fibsem_cfg
    from fibsem.applications.autolamella.ui.autolamella_fluorescence_overview_tab import (
        AutoLamellaFluorescenceOverviewTab,
    )
    from fibsem.ui.fm.overview_app import build_microscope

    # The tab is behind a preference that is off by default, and `refresh_microscope`
    # reads it. `_restore_fm_overview_flag` puts it back afterwards.
    fibsem_cfg.FEATURE_FM_OVERVIEW_TAB_ENABLED = True
    microscope = build_microscope()
    experiment = _experiment_with(positions, tmp_path)
    ui = type(
        "_UI", (), {"microscope": microscope, "experiment": experiment}
    )()
    ui.update_ui = lambda: None
    tab = AutoLamellaFluorescenceOverviewTab(ui)
    tab.refresh_microscope()
    qapp.processEvents()
    return tab


def test_the_list_sits_in_the_settings_column(qapp, tmp_path):
    """Not in a column of its own beside it: the positions are the subject of this tab,
    and a third pane would read as somewhere else to look."""
    tab = _tab_with(qapp, tmp_path)

    # its way up the tree reaches the overview, not the tab directly
    parents, w = [], tab.lamella_list
    while w is not None:
        parents.append(w)
        w = w.parent()

    assert tab.overview in parents
    assert tab.lamella_list.isVisibleTo(tab.overview)

    tab._drop_overview()


def test_the_list_shows_every_lamella_including_the_unplaceable(qapp, tmp_path):
    """The canvas can only draw the ones with a fluorescence pose. The list is how you
    find out the others exist at all -- dropping them here too would leave the log as the
    only place that says so."""
    tab = _tab_with(qapp, tmp_path)
    microscope = tab.microscope
    placeable = _lamella_with_poses("Lamella-01", microscope, tmp_path, 100e-6, 50e-6)
    orphan = _lamella_with_poses("Lamella-02", microscope, tmp_path, 0.0, 0.0)
    del orphan.poses["FLUORESCENCE"]
    tab.experiment.positions.extend([placeable, orphan])

    tab.refresh_positions()

    from PyQt5.QtCore import Qt

    names = [
        tab.lamella_list._list.item(i).data(Qt.UserRole).name
        for i in range(tab.lamella_list._list.count())
    ]
    assert names == ["Lamella-01", "Lamella-02"]
    # ...while the canvas draws only the one it can place
    assert [p.name for p in tab.overview._positions] == ["Lamella-01"]

    tab._drop_overview()


def test_picking_a_row_highlights_it_on_the_canvas(qapp, tmp_path):
    tab = _tab_with(qapp, tmp_path)
    microscope = tab.microscope
    tab.experiment.positions.extend([
        _lamella_with_poses("Lamella-01", microscope, tmp_path, 100e-6, 50e-6),
        _lamella_with_poses("Lamella-02", microscope, tmp_path, -80e-6, 20e-6),
    ])
    tab.refresh_positions()
    qapp.processEvents()

    tab.lamella_list.select("Lamella-02")
    qapp.processEvents()

    assert tab.overview.selected_position == "Lamella-02"

    tab._drop_overview()


def test_picking_a_row_tells_the_window(qapp, tmp_path):
    """The other lists are synced by the window, not from here -- one place doing it
    beats four talking to each other."""
    tab = _tab_with(qapp, tmp_path)
    microscope = tab.microscope
    tab.experiment.positions.extend([
        _lamella_with_poses("Lamella-01", microscope, tmp_path, 100e-6, 50e-6),
        _lamella_with_poses("Lamella-02", microscope, tmp_path, -80e-6, 20e-6),
    ])
    tab.refresh_positions()
    qapp.processEvents()
    seen = []
    tab.lamella_selected.connect(lambda lamella: seen.append(lamella))

    # the second, because populating restores the selection to the first -- picking a
    # row that is already current emits nothing
    tab.lamella_list.select("Lamella-02")
    qapp.processEvents()

    assert [lamella.name for lamella in seen if lamella is not None] == ["Lamella-02"]

    tab._drop_overview()


def test_a_selection_arriving_from_elsewhere_reaches_the_list(qapp, tmp_path):
    tab = _tab_with(qapp, tmp_path)
    microscope = tab.microscope
    tab.experiment.positions.extend([
        _lamella_with_poses("Lamella-01", microscope, tmp_path, 100e-6, 50e-6),
        _lamella_with_poses("Lamella-02", microscope, tmp_path, -80e-6, 20e-6),
    ])
    tab.refresh_positions()
    qapp.processEvents()

    tab.set_selected(tab.experiment.positions[1])
    qapp.processEvents()

    assert tab.lamella_list.selected_name == "Lamella-02"
    assert tab.overview.selected_position == "Lamella-02"

    tab._drop_overview()


def test_move_to_drives_the_stage_to_the_fluorescence_pose(qapp, tmp_path, monkeypatch):
    """Not `stage_position`, which is the milling pose. This tab looks at the sample from
    the fluorescence side, and moving to the milling pose would swing the stage 180
    degrees away from the view you asked to centre."""
    from fibsem.ui.fm.widgets import fm_overview_widget as module

    tab = _tab_with(qapp, tmp_path)
    monkeypatch.setattr(module, "FunctionWorker", _SynchronousWorker)
    microscope = tab.microscope
    lamella = _lamella_with_poses("Lamella-01", microscope, tmp_path, 300e-6, -150e-6)
    tab.experiment.positions.append(lamella)
    tab.refresh_positions()

    tab._on_move_to_requested(lamella)
    qapp.processEvents()

    arrived = microscope.get_stage_position()
    fm_pose = lamella.fluorescence_pose.stage_position
    assert arrived.x == pytest.approx(fm_pose.x, abs=1e-9)
    assert arrived.y == pytest.approx(fm_pose.y, abs=1e-9)
    assert microscope.get_stage_orientation(arrived) == "FM"
    assert arrived.t != pytest.approx(lamella.milling_pose.stage_position.t)

    tab._drop_overview()


def test_move_to_a_lamella_with_no_fluorescence_pose_is_refused(qapp, tmp_path, monkeypatch):
    from fibsem.ui.fm.widgets import fm_overview_widget as module

    tab = _tab_with(qapp, tmp_path)
    monkeypatch.setattr(module, "FunctionWorker", _SynchronousWorker)
    microscope = tab.microscope
    lamella = _lamella_with_poses("Lamella-01", microscope, tmp_path, 300e-6, -150e-6)
    del lamella.poses["FLUORESCENCE"]
    tab.experiment.positions.append(lamella)
    before = microscope.get_stage_position()

    tab._on_move_to_requested(lamella)  # must not raise
    qapp.processEvents()

    assert microscope.get_stage_position().x == pytest.approx(before.x)

    tab._drop_overview()


def test_removing_a_lamella_takes_it_off_the_list_and_the_canvas(qapp, tmp_path):
    tab = _tab_with(qapp, tmp_path)
    microscope = tab.microscope
    keep = _lamella_with_poses("Lamella-01", microscope, tmp_path, 100e-6, 50e-6)
    drop = _lamella_with_poses("Lamella-02", microscope, tmp_path, -80e-6, 20e-6)
    tab.experiment.positions.extend([keep, drop])
    tab.refresh_positions()

    tab._on_remove_requested(drop)
    tab.refresh_positions()  # the window's rebuild does this off `removed`

    assert list(tab.experiment.positions) == [keep]
    assert [p.name for p in tab.overview._positions] == ["Lamella-01"]
    assert tab.experiment.saves == 1

    tab._drop_overview()


def test_the_list_survives_the_overview_being_rebuilt(qapp, tmp_path):
    """A reconnection replaces the overview. The list belongs to the tab, so it is the
    same object afterwards, still populated, and sitting in the *new* column.

    It does not verify the reparent in `_drop_overview` that makes this safe against Qt
    ownership: the offscreen platform never actually runs the deferred delete, so the
    old overview is still alive either way and the mechanism cannot be reproduced here.
    Removing that reparent leaves this test passing -- checked."""
    tab = _tab_with(qapp, tmp_path)
    microscope = tab.microscope
    tab.experiment.positions.append(
        _lamella_with_poses("Lamella-01", microscope, tmp_path, 100e-6, 50e-6)
    )
    tab.refresh_positions()
    first = tab.lamella_list

    from PyQt5.QtCore import QEvent

    from fibsem.ui.fm.overview_app import build_microscope

    tab.autolamella_ui.microscope = build_microscope()
    tab.refresh_microscope()  # a reconnection: new microscope, new overview
    # `deleteLater` is deferred, so without this the old overview is still alive at
    # assert time and a list left inside it would look fine. A running event loop does
    # this for you; a test has to ask.
    qapp.sendPostedEvents(None, QEvent.DeferredDelete)
    qapp.processEvents()

    assert tab.lamella_list is first, "the list was replaced"
    assert tab.lamella_list.selected_name == "Lamella-01", "the list lost its contents"
    parents, w = [], tab.lamella_list
    while w is not None:
        parents.append(w)
        w = w.parent()
    assert tab.overview in parents, "the list did not move to the new overview"

    tab._drop_overview()


def test_a_lamella_is_complete_before_anyone_is_told_about_it(qapp, tmp_path):
    """The reported bug: a position added from the FIB/SEM minimap was missing from the
    FM overview -- always the newest one, and only until something else forced a refresh.

    `add_lamella` publishes `inserted`, and everything drawing the experiment redraws on
    it. The fluorescence pose used to be assigned three lines *after* that, so every
    listener decided the new lamella had none and skipped it as unplaceable.

    Asserted at the moment the signal fires, because that is the only moment it was ever
    wrong: by the time `add_new_lamella` returned, the pose was there.
    """
    from psygnal.containers import EventedDict

    from fibsem.applications.autolamella.poses import (
        MILLING_ORIENTATION,
        build_lamella_poses,
    )
    from fibsem.applications.autolamella.structures import AutoLamellaTaskProtocol
    from fibsem.structures import FibsemStagePosition
    from fibsem.ui.fm.overview_app import build_microscope

    microscope = build_microscope()
    experiment = _real_experiment(tmp_path)
    experiment.task_protocol = AutoLamellaTaskProtocol()

    seen = []
    experiment.positions.events.inserted.connect(
        lambda *_: seen.append(experiment.positions[-1].fluorescence_pose is not None)
    )

    beam = microscope.get_orientation(MILLING_ORIENTATION)
    poses = build_lamella_poses(
        microscope,
        FibsemStagePosition(x=100e-6, y=50e-6, z=0.0, r=beam.r, t=beam.t),
    )
    experiment.add_new_lamella(
        microscope_state=poses.milling,
        task_config=EventedDict(),
        fluorescence_pose=poses.fluorescence,
    )

    assert seen == [True], "the lamella was published before it had a fluorescence pose"


def test_a_lamella_added_from_elsewhere_lands_on_the_canvas_immediately(qapp, tmp_path):
    """The same bug, end to end: nothing calls the FM refresh by hand on this path, so
    the marker has to be there off the `inserted` event alone."""
    from psygnal.containers import EventedDict

    from fibsem.applications.autolamella.poses import (
        MILLING_ORIENTATION,
        build_lamella_poses,
    )
    from fibsem.applications.autolamella.structures import AutoLamellaTaskProtocol
    from fibsem.structures import FibsemStagePosition

    tab = _tab_with(qapp, tmp_path)
    microscope = tab.microscope
    experiment = _real_experiment(tmp_path / "exp")
    experiment.task_protocol = AutoLamellaTaskProtocol()
    tab.autolamella_ui.experiment = experiment
    experiment.positions.events.inserted.connect(lambda *_: tab.refresh_positions())

    beam = microscope.get_orientation(MILLING_ORIENTATION)
    poses = build_lamella_poses(
        microscope,
        FibsemStagePosition(x=100e-6, y=50e-6, z=0.0, r=beam.r, t=beam.t),
    )
    experiment.add_new_lamella(
        microscope_state=poses.milling,
        task_config=EventedDict(),
        name="Lamella-01",
        fluorescence_pose=poses.fluorescence,
    )
    qapp.processEvents()

    assert [p.name for p in tab.overview._positions] == ["Lamella-01"]

    tab._drop_overview()


# ── the scene follows the stage's pose ───────────────────────────────────


def test_marked_positions_move_when_the_stage_re_poses(qapp):
    """Reported: tilt the stage 180 degrees and the markers stay put, when they should
    move. Everything on this canvas is placed through a frame whose rotation and tilt
    come from wherever the stage is, so a re-pose moves all of it -- but only the current
    position crosshair and the tile grid were being redrawn."""
    from fibsem.structures import FibsemStagePosition

    microscope = _microscope_at(-180.0)  # FM
    widget = FMOverviewWidget(microscope)
    widget._refresh_tile_grid()
    qapp.processEvents()

    marked = _named("Lamella-01", 100e-6, 50e-6)
    widget.set_positions([marked])
    at_fm = list(widget.position_overlay._points)

    milling = microscope.get_orientation("MILLING")
    microscope.move_stage_absolute(
        FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=milling.r, t=milling.t)
    )
    widget._on_stage_moved(microscope.get_stage_position())
    qapp.processEvents()

    after = list(widget.position_overlay._points)
    assert after != at_fm, "the markers did not follow the re-pose"
    # ...and they are where the frame now says they are, not merely somewhere else
    assert after[0] == pytest.approx(widget._frame().to_canvas(marked), abs=1e-6)

    widget.close()


def test_the_selected_marker_re_poses_too(qapp):
    """It is on its own overlay, so it is a second thing that has to be redrawn."""
    from fibsem.structures import FibsemStagePosition

    microscope = _microscope_at(-180.0)
    widget = FMOverviewWidget(microscope)
    widget._refresh_tile_grid()
    widget.set_positions([_named("Lamella-01", 100e-6, 50e-6)])
    widget.set_selected_position("Lamella-01")
    qapp.processEvents()
    before = list(widget.selected_position_overlay._points)

    milling = microscope.get_orientation("MILLING")
    microscope.move_stage_absolute(
        FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=milling.r, t=milling.t)
    )
    widget._on_stage_moved(microscope.get_stage_position())
    qapp.processEvents()

    assert list(widget.selected_position_overlay._points) != before

    widget.close()


def test_translating_the_stage_leaves_the_markers_alone(qapp):
    """The stage is polled constantly and a translation moves none of this: the origin
    everything is measured from does not move with it. Redrawing anyway would be work
    per poll for no visible change."""
    from fibsem.structures import FibsemStagePosition

    microscope = _microscope_at(-180.0)
    widget = FMOverviewWidget(microscope)
    widget._refresh_tile_grid()
    widget.set_positions([_named("Lamella-01", 100e-6, 50e-6)])
    qapp.processEvents()
    before = list(widget.position_overlay._points)

    redraws = []
    original = widget._refresh_positions
    widget._refresh_positions = lambda: redraws.append(1) or original()

    here = microscope.get_stage_position()
    microscope.move_stage_absolute(
        FibsemStagePosition(x=here.x + 200e-6, y=here.y - 90e-6, z=here.z,
                            r=here.r, t=here.t)
    )
    widget._on_stage_moved(microscope.get_stage_position())
    qapp.processEvents()

    assert redraws == [], "redrew the markers for a move that cannot have moved them"
    assert list(widget.position_overlay._points) == before

    widget.close()


def test_the_stage_limits_re_pose_even_with_the_grid_pinned(qapp):
    """Drawn from the same frame, so they had the same bug.

    With the grid free, `_refresh_tile_grid` redraws them on its way past and this
    passes either way. The target is pinned first so that path is skipped -- dragging the
    grid somewhere and then re-posing the stage is exactly when nothing else would.

    The origin is also put *off* the grid centre. Anchored on it, the limits box and the
    grid boundary sit at canvas zero whatever the pose -- a zero offset stays zero
    through any rotation -- so the test would pass by drawing nothing that could move.
    """
    from fibsem.structures import FibsemStagePosition

    microscope = _microscope_at(-180.0)
    fm = microscope.get_orientation("FM")
    microscope.move_stage_absolute(
        FibsemStagePosition(x=250e-6, y=-180e-6, z=0.0, r=fm.r, t=fm.t)
    )
    widget = FMOverviewWidget(microscope)
    widget._refresh_tile_grid()  # fixes the origin, off-centre
    qapp.processEvents()
    before = [(spec.cx, spec.cy) for spec in widget.stage_overlay._specs]
    assert any(cx or cy for cx, cy in before), "nothing drawn that a re-pose could move"

    # Pin the grid, so `_on_stage_moved` will not redraw it -- and with it, the limits.
    widget._target = microscope.get_stage_position()

    milling = microscope.get_orientation("MILLING")
    microscope.move_stage_absolute(
        FibsemStagePosition(x=250e-6, y=-180e-6, z=0.0, r=milling.r, t=milling.t)
    )
    widget._on_stage_moved(microscope.get_stage_position())
    qapp.processEvents()

    assert [(spec.cx, spec.cy) for spec in widget.stage_overlay._specs] != before

    widget.close()


# ── clicking a marker selects it ─────────────────────────────────────────


def test_clicking_a_marker_selects_it(qapp, interactive_widget):
    """As the minimap does. The markers are drawn by a non-interactive overlay, so the
    canvas's own click signal does the picking."""
    widget = interactive_widget
    widget.set_positions([
        _named("Lamella-01", 120e-6, 90e-6),
        _named("Lamella-02", -160e-6, 40e-6),
    ])
    picked = []
    widget.position_selected.connect(lambda name: picked.append(name))
    frame = widget._frame()

    widget._on_canvas_clicked(*frame.to_canvas(_named("Lamella-02", -160e-6, 40e-6)))

    assert picked == ["Lamella-02"]
    assert widget.selected_position == "Lamella-02"
    # ...and it moved onto the selected overlay, so it is drawn as selected too
    assert widget.selected_position_overlay._labels == ["Lamella-02"]

    widget.set_selected_position(None)
    widget.set_positions([])


def test_clicking_empty_space_keeps_the_selection(qapp, interactive_widget):
    """Where this parts company with the minimap's interactive overlay, on purpose: the
    host syncs its other lists from the selection and their handlers ignore None, so a
    clearing click would blank this canvas and nothing else."""
    widget = interactive_widget
    widget.set_positions([_named("Lamella-01", 120e-6, 90e-6)])
    widget.set_selected_position("Lamella-01")
    frame = widget._frame()
    picked = []
    widget.position_selected.connect(lambda name: picked.append(name))

    widget._on_canvas_clicked(*frame.to_canvas(_named("far", 900e-6, -900e-6)))

    assert picked == []
    assert widget.selected_position == "Lamella-01"

    widget.set_selected_position(None)
    widget.set_positions([])


def test_the_pick_radius_is_screen_space(qapp, interactive_widget):
    """In pixels, not microns: how close you have to click must not change with the
    zoom. The same stage point is a hit when zoomed out and a miss when zoomed in."""
    widget = interactive_widget
    widget.set_positions([_named("Lamella-01", 120e-6, 90e-6)])
    frame = widget._frame()
    ax = widget.canvas.canvas._ax
    before = (ax.get_xlim(), ax.get_ylim())
    # 80 um away from the marker, in stage terms
    near = frame.to_canvas(_named("probe", 200e-6, 90e-6))

    ax.set_xlim(-50000, 50000)
    ax.set_ylim(-50000, 50000)
    qapp.processEvents()
    zoomed_out = widget._position_at(*near)

    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)
    qapp.processEvents()
    zoomed_in = widget._position_at(*near)

    assert zoomed_out == "Lamella-01", "should be within 12px when zoomed right out"
    assert zoomed_in is None, "should be far outside 12px when zoomed right in"

    ax.set_xlim(*before[0])
    ax.set_ylim(*before[1])
    widget.set_positions([])


def test_the_nearest_marker_wins(qapp, interactive_widget):
    """Two markers can both be inside the radius when they are close together.

    The nearest one is deliberately *first* in the list: keeping a hit without also
    keeping its distance leaves the last match winning, which would agree with the right
    answer if the nearest happened to come last.
    """
    widget = interactive_widget
    widget.set_positions([
        _named("Lamella-02", 121e-6, 90e-6),   # nearest to the probe, and first
        _named("Lamella-01", 120e-6, 90e-6),
    ])
    frame = widget._frame()

    assert widget._position_at(
        *frame.to_canvas(_named("probe", 120.9e-6, 90e-6))
    ) == "Lamella-02"

    widget.set_positions([])


def test_clicking_a_marker_selects_the_lamella_everywhere(qapp, tmp_path):
    """The canvas knows the name it drew; turning that into a lamella is the tab's job,
    and it goes out on the same path a row click takes."""
    tab = _tab_with(qapp, tmp_path)
    microscope = tab.microscope
    tab.experiment.positions.extend([
        _lamella_with_poses("Lamella-01", microscope, tmp_path, 100e-6, 50e-6),
        _lamella_with_poses("Lamella-02", microscope, tmp_path, -80e-6, 20e-6),
    ])
    tab.refresh_positions()
    qapp.processEvents()
    seen = []
    tab.lamella_selected.connect(lambda lamella: seen.append(lamella))

    frame = tab.overview._frame()
    marker = frame.to_canvas(tab.overview._positions[1])
    tab.overview._on_canvas_clicked(*marker)
    qapp.processEvents()

    assert [lamella.name for lamella in seen] == ["Lamella-02"]
    assert tab.lamella_list.selected_name == "Lamella-02"
    assert tab.overview.selected_position == "Lamella-02"

    tab._drop_overview()


def test_the_tile_grid_panel_grows_with_its_summary(qapp, interactive_widget):
    """FIB-510. The panel is a fixed width with a word-wrapped summary, so its height is
    however many lines that text takes -- and the text grows: dragging the grid adds a
    line saying how far it now sits from the stage. Fitted only at open time, the panel
    kept its old height and clipped the extra line top and bottom.
    """
    widget = interactive_widget
    panel = widget.tile_grid_panel
    widget.btn_tile_grid.setChecked(True)
    widget._toggle_tile_grid_panel()
    qapp.processEvents()
    short = panel.height()
    assert short >= panel.sizeHint().height()

    # drag the grid, which is what adds the offset line
    frame = widget._frame()
    here = frame.to_canvas(widget._current_stage_position())
    widget._on_grid_move(here[0] + 2500.0, here[1] - 400.0)
    qapp.processEvents()

    assert "from the stage" in panel.label_summary.text(), "the summary did not grow"
    assert panel.height() > short, "the panel did not grow with it"
    assert panel.height() >= panel.sizeHint().height(), "the summary is clipped"

    widget.btn_tile_grid.setChecked(False)
    widget._toggle_tile_grid_panel()
    widget.clear_target()


def test_the_tile_grid_summary_is_never_clipped(qapp):
    """FIB-510, at the level it actually breaks.

    A word-wrapped QLabel's height depends on the width it is given, and that dependency
    does not reach the panel's own size hint -- so past a certain length the panel stops
    growing and the label keeps a height computed for fewer lines. The text is centred
    vertically, so the overflow is split top and bottom, which is what the report showed.

    Measured at 196 px wide: the label reported height 36 for text needing 48. Three
    lines was not enough to trigger it, which is why the widget-level test above passes
    either way.
    """
    from fibsem.ui.fm.widgets.tile_grid_options_panel import TileGridOptionsPanel

    panel = TileGridOptionsPanel()
    panel.set_summary("3 × 3  ·  10% overlap  ·  9/9 tiles\n287 × 287 µm")
    panel.adjustSize()
    panel.show()
    qapp.processEvents()

    # what dragging the grid adds, which wraps onto a further line
    panel.set_summary(
        "3 × 3  ·  10% overlap  ·  9/9 tiles\n287 × 287 µm"
        "\nOffset +563, -93 µm from the stage"
    )
    qapp.processEvents()

    label = panel.label_summary
    assert label.height() >= label.heightForWidth(label.width()), (
        f"summary clipped: {label.height()}px for text needing "
        f"{label.heightForWidth(label.width())}px"
    )

    panel.close()


def test_a_hidden_tile_grid_panel_is_not_refitted(qapp, interactive_widget):
    """Opening it fits it anyway, and moving a hidden top-level window around is work
    nobody can see."""
    widget = interactive_widget
    widget.btn_tile_grid.setChecked(False)
    widget._toggle_tile_grid_panel()
    qapp.processEvents()
    assert not widget.tile_grid_panel.isVisible()

    fits = []
    original = widget._fit_tile_grid_panel
    widget._fit_tile_grid_panel = lambda: fits.append(1) or original()

    widget._update_grid_summary()

    assert fits == []

    widget._fit_tile_grid_panel = original


# ── one record per overview on the canvas (FIB-543) ──────────────────────


@pytest.fixture
def blank_canvas(qapp, overview_widget):
    """The shared widget with nothing placed, so counts mean what they say."""
    widget = overview_widget
    # The counter too, not just the store: the widget fixture is module-scoped, so an
    # earlier test's acquisitions would otherwise decide what these ids are called.
    widget.canvas.clear_overviews()
    widget._records.clear()
    widget._record_count = 0
    # The list too. It is rebuilt from the store rather than reading it, so clearing
    # only the store leaves rows from a previous test for the next one to count.
    widget.overview_list.set_records([])
    yield widget
    widget.canvas.clear_overviews()
    widget._records.clear()
    widget.overview_list.set_records([])


def _acquire(widget, date=None):
    image = widget.microscope.fm.acquire_image(ChannelSettings(name="Channel-01"))
    image.metadata.acquisition_date = date
    return image


def test_each_acquisition_gets_its_own_record(qapp, blank_canvas):
    """The hole this fills: a single `_displayed_image` meant the second overview
    overwrote the first's provenance while its pixels stayed on screen."""
    widget = blank_canvas

    widget.set_image(_acquire(widget, "2026-08-07T09:00:00"))
    widget.set_image(_acquire(widget, "2026-08-07T09:30:00"))
    qapp.processEvents()

    records = widget.overviews()
    assert len(records) == 2
    assert records[0].id != records[1].id
    # Oldest first: a list reads in the order things were acquired.
    assert [r.metadata.acquisition_date for r in records] == [
        "2026-08-07T09:00:00",
        "2026-08-07T09:30:00",
    ]


def test_two_overviews_without_a_date_do_not_share_one_record(qapp, blank_canvas):
    """The reason ids are minted. The old key fell back to a bare "overview" when the
    acquisition date was missing, so every dateless overview replaced the last."""
    widget = blank_canvas

    widget.set_image(_acquire(widget, None))
    widget.set_image(_acquire(widget, None))
    qapp.processEvents()

    assert len(widget.overviews()) == 2
    assert len(set(widget.canvas.canvas.placed_keys)) == 2


def test_showing_one_image_twice_replaces_it(qapp, blank_canvas):
    """The one property the old date-derived key had that was worth keeping."""
    widget = blank_canvas
    image = _acquire(widget, "2026-08-07T09:00:00")

    widget.set_image(image)
    widget.set_image(image)
    qapp.processEvents()

    assert len(widget.overviews()) == 1


def test_the_live_preview_never_becomes_a_record(qapp, blank_canvas):
    """The preview is transient -- swapped for the real stitch when the run ends -- so
    it must not appear in a list of what has been acquired. It reaches the canvas
    through `set_channel` under its own key rather than through `set_image`, so the
    exclusion is structural; this pins it, because the two paths look alike."""
    widget = blank_canvas
    import numpy as np

    widget._show_preview({"image": np.zeros((4, 4), dtype=np.uint16), "preview_stride": 1})
    qapp.processEvents()

    assert PREVIEW_KEY in widget.canvas.canvas.placed_keys, "the preview was not drawn"
    assert widget.overviews() == [], "the preview leaked into the record store"


def test_hiding_an_overview_takes_it_out_of_the_drawing(qapp, blank_canvas):
    """Hidden rather than removed: it comes back without being re-acquired. Matplotlib
    skips a hidden artist, so this also drops it from the redraw cost (FIB-414)."""
    widget = blank_canvas
    widget.set_image(_acquire(widget, "2026-08-07T09:00:00"))
    qapp.processEvents()
    record = widget.overviews()[0]

    assert widget.set_overview_visible(record.id, False)

    assert record.visible is False
    assert widget.canvas.canvas._placed[record.id].artist.get_visible() is False
    assert record.id in widget.canvas.canvas.placed_keys, "hiding removed it"

    widget.set_overview_visible(record.id, True)
    assert widget.canvas.canvas._placed[record.id].artist.get_visible() is True


def test_a_hidden_overview_stays_hidden_when_its_frame_is_replaced(qapp, blank_canvas):
    """A *reshaped* frame is placed as a new artist, and a new artist starts visible.

    The same-shape path swaps pixels on the existing artist and keeps its visibility on
    its own, so it does not exercise this -- the frame has to change size, which is why
    the second image here is cropped rather than merely re-set.
    """
    widget = blank_canvas
    stamp = "2026-08-07T09:00:00"
    widget.set_image(_acquire(widget, stamp))
    qapp.processEvents()
    record = widget.overviews()[0]
    widget.set_overview_visible(record.id, False)

    smaller = _acquire(widget, stamp)  # same date -> same record, different size
    smaller.data = smaller.data[..., :64, :64]
    widget.set_image(smaller)
    qapp.processEvents()

    assert widget.canvas.canvas._placed[record.id].artist.get_visible() is False


def test_removing_an_overview_drops_its_record_artist_and_planes(qapp, blank_canvas):
    """All three go together. A record without an artist is a row with nothing under
    it; planes without a record are pixels nothing can reach, and `_restyle_others`
    would keep re-rendering them."""
    widget = blank_canvas
    widget.set_image(_acquire(widget, "2026-08-07T09:00:00"))
    qapp.processEvents()
    record = widget.overviews()[0]

    assert widget.remove_overview(record.id)

    assert widget.overviews() == []
    assert record.id not in widget.canvas.canvas.placed_keys
    assert record.id not in widget.canvas._held


def test_removing_something_that_is_not_there_says_so(qapp, blank_canvas):
    assert blank_canvas.remove_overview("overview-404") is False
    assert blank_canvas.set_overview_visible("overview-404", True) is False


def test_a_record_describes_its_grid_and_scale(qapp, blank_canvas):
    """What the second half of a list row shows. The grid comes back off the mosaic's
    position name, which `stitch_tileset` writes as `stitched_mosaic_3x3`."""
    widget = blank_canvas
    image = _acquire(widget, "2026-08-07T09:00:00")
    image.metadata.stage_position.name = "stitched_mosaic_3x3"
    image.metadata.pixel_size_x = 1.2e-6

    widget.set_image(image)
    qapp.processEvents()

    assert widget.overviews()[0].detail == "3×3 · 1.20 µm/px"


def test_a_saved_overview_is_labelled_by_the_folder_its_tiles_are_in(qapp, blank_canvas):
    """`OverviewDestination` stamps the run's directory name onto the mosaic, which is
    what someone looking for it on disk would search for."""
    widget = blank_canvas
    image = _acquire(widget, "2026-08-07T09:00:00")
    image.metadata.description = "overview-14-02-33"

    widget.set_image(image)

    assert widget.overviews()[0].label == "overview-14-02-33"


def test_an_acquired_overview_appears_in_the_list(qapp, blank_canvas):
    widget = blank_canvas

    widget.set_image(_acquire(widget, "2026-08-07T09:00:00"))
    qapp.processEvents()

    assert widget.overview_list._list.count() == 1


def test_the_eye_in_the_list_hides_the_image_on_the_canvas(qapp, blank_canvas):
    """The whole point of the wiring: the row is not a label, it drives the canvas."""
    widget = blank_canvas
    widget.set_image(_acquire(widget, "2026-08-07T09:00:00"))
    qapp.processEvents()
    record = widget.overviews()[0]
    row = widget.overview_list._list.itemWidget(widget.overview_list._list.item(0))

    row.btn_visible.setChecked(True)  # checked means hidden
    qapp.processEvents()

    assert record.visible is False
    assert widget.canvas.canvas._placed[record.id].artist.get_visible() is False


def test_the_trash_in_the_list_drops_the_image_from_the_canvas(qapp, blank_canvas):
    widget = blank_canvas
    widget.set_image(_acquire(widget, "2026-08-07T09:00:00"))
    qapp.processEvents()
    record_id = widget.overviews()[0].id
    row = widget.overview_list._list.itemWidget(widget.overview_list._list.item(0))

    row.btn_remove.click()
    qapp.processEvents()

    assert widget.overviews() == []
    assert record_id not in widget.canvas.canvas.placed_keys
    assert widget.overview_list._list.count() == 0


def test_the_list_does_not_grow_a_row_for_the_live_preview(qapp, blank_canvas):
    """The structural exclusion, seen from the other end: a run in progress must not
    put a row in the list that vanishes when it finishes."""
    widget = blank_canvas
    import numpy as np

    widget._show_preview({"image": np.zeros((4, 4), dtype=np.uint16), "preview_stride": 1})
    qapp.processEvents()

    assert widget.overview_list._list.count() == 0


# ── loading a saved overview (FIB-438) ───────────────────────────────────


def _saved(widget, tmp_path, date="2026-08-07T09:00:00", name="overview-14-02-33"):
    """An overview written to disk the way an acquisition writes one."""
    image = _acquire(widget, date)
    image.metadata.description = name  # what OverviewDestination.save_mosaic stamps
    path = str(tmp_path / f"{name}.ome.tiff")
    image.save(path)
    return image, path


def test_a_loaded_overview_lands_where_the_acquired_one_did(qapp, blank_canvas, tmp_path):
    """The claim the whole feature rests on. `set_image` places by projecting through
    the image's *own* recorded geometry, and everything that projection needs -- the
    stage position's r and t, the hardware geometry, the pixel size -- survives the
    OME-TIFF. So a file off disk is placed identically to the image it came from."""
    widget = blank_canvas
    image, path = _saved(widget, tmp_path)

    widget.set_image(image)
    qapp.processEvents()
    acquired_extent = widget.canvas.canvas._placed[widget.overviews()[0].id].extent

    widget.canvas.clear_overviews()
    widget._records.clear()
    widget.load_overview(path)
    qapp.processEvents()
    loaded_extent = widget.canvas.canvas._placed[widget.overviews()[0].id].extent

    assert loaded_extent == pytest.approx(acquired_extent)


def test_loading_gives_the_overview_a_record_and_a_row(qapp, blank_canvas, tmp_path):
    widget = blank_canvas
    _, path = _saved(widget, tmp_path)

    record = widget.load_overview(path)
    qapp.processEvents()

    assert record is not None
    assert widget.overviews() == [record]
    assert widget.overview_list._list.count() == 1


def test_a_loaded_overview_is_named_for_the_run_that_made_it(qapp, blank_canvas, tmp_path):
    """`description` survives the round trip, so the row still names the folder the
    tiles are in rather than falling back to a number."""
    widget = blank_canvas
    _, path = _saved(widget, tmp_path, name="overview-11-20-05")

    record = widget.load_overview(path)

    assert record.label == "overview-11-20-05"


def test_a_loaded_overview_keeps_its_scale_but_loses_its_grid(qapp, blank_canvas, tmp_path):
    """Documented, not a bug. `stitch_tileset` carries the grid in
    `stage_position.name`, which the OME-TIFF drops -- so a loaded row shows what it
    still knows rather than guessing (FIB-438)."""
    widget = blank_canvas
    image, path = _saved(widget, tmp_path)
    image.metadata.stage_position.name = "stitched_mosaic_3x3"
    image.save(path)

    record = widget.load_overview(path)

    assert "µm/px" in record.detail
    assert "×" not in record.detail, "the grid came back after all -- update the docs"


def test_a_file_that_cannot_be_read_places_nothing(qapp, blank_canvas, tmp_path):
    """Reported, not raised: a bad file is a thing a user picked, not a bug in the tab."""
    widget = blank_canvas
    bad = tmp_path / "not-an-image.ome.tiff"
    bad.write_text("this is not a tiff")

    assert widget.load_overview(str(bad)) is None
    assert widget.overviews() == []


def test_an_overview_with_no_geometry_is_shown_but_not_trusted(
    qapp, blank_canvas, tmp_path, monkeypatch
):
    """Anything acquired before FIB-416 has no geometry, so it cannot be placed where
    it was taken -- it lands at the canvas origin instead. The pixels are still worth
    seeing, so it is shown; being shown *silently* in the wrong place is the part that
    would mislead, so it is also said."""
    widget = blank_canvas
    image, path = _saved(widget, tmp_path)
    image.metadata.geometry = None
    image.save(path)
    toasts = []
    monkeypatch.setattr(
        "fibsem.ui.fm.widgets.fm_overview_widget.notification_service.show_toast",
        lambda message, level="info", *a, **k: toasts.append((message, level)),
    )

    record = widget.load_overview(path)
    qapp.processEvents()

    assert record is not None
    assert record.id in widget.canvas.canvas.placed_keys
    assert any(level == "warning" for _, level in toasts), "placed with no warning"


def test_loading_the_same_file_twice_does_not_stack_two_copies(qapp, blank_canvas, tmp_path):
    """The acquisition date survives the round trip, so the record match still holds
    across a reload."""
    widget = blank_canvas
    _, path = _saved(widget, tmp_path)

    widget.load_overview(path)
    widget.load_overview(path)
    qapp.processEvents()

    assert len(widget.overviews()) == 1


def test_a_loaded_overview_can_anchor_an_empty_canvas(qapp, blank_canvas, tmp_path):
    """First image placed fixes the origin, loaded or acquired. Everything is drawn
    relative to it, so which one it is only decides where canvas zero sits."""
    widget = blank_canvas
    widget._origin = None
    _, path = _saved(widget, tmp_path)

    widget.load_overview(path)

    assert widget._origin is not None


def test_the_header_button_opens_a_file_dialog(qapp, blank_canvas, monkeypatch):
    """The real wiring, clicked for real.

    Safe because the dialog itself is stubbed: it is modal, so a genuine one would
    hang the run waiting for a human -- which it did, when this test clicked through
    to the real thing.
    """
    widget = blank_canvas
    asked = []
    monkeypatch.setattr(
        "fibsem.ui.fm.widgets.fm_overview_widget.ui_utils.open_existing_file_dialog",
        lambda **kwargs: asked.append(kwargs) or "",
    )

    widget.btn_load_overview.click()

    assert len(asked) == 1
    assert "ome.tiff" in asked[0]["_filter"], "the dialog does not offer overviews"


def test_picking_no_file_places_nothing(qapp, blank_canvas, monkeypatch):
    """Cancelling the dialog is not an error, and must not reach the loader."""
    widget = blank_canvas
    monkeypatch.setattr(
        "fibsem.ui.fm.widgets.fm_overview_widget.ui_utils.open_existing_file_dialog",
        lambda **kwargs: "",
    )
    loaded = []
    monkeypatch.setattr(widget, "load_overview", lambda path: loaded.append(path))

    widget.btn_load_overview.click()

    assert loaded == []
    assert widget.overviews() == []


# ── the start options say what they currently mean (FIB-417) ─────────────


def test_the_start_options_show_where_they_would_put_the_objective(qapp, overview_widget):
    """An option that names a position has to say which one. "Saved focus position" is
    only actionable next to the number, and next to where the objective is now."""
    widget = overview_widget
    widget.fm.objective.focus_position = 8.0e-3

    widget._refresh_objective_start_options()

    combo = widget.settings_widget.combo_objective_start
    labels = [combo.itemText(i) for i in range(combo.count())]
    assert any("mm" in label for label in labels), f"no positions shown: {labels}"
    assert any("8.000 mm" in label for label in labels), f"saved focus missing: {labels}"


def test_an_unsaved_focus_position_says_so_rather_than_showing_nothing(
    qapp, overview_widget
):
    """The refusal comes later; the label is where you would notice first."""
    widget = overview_widget
    widget.fm.objective._focus_position = None

    widget._refresh_objective_start_options()

    combo = widget.settings_widget.combo_objective_start
    labels = [combo.itemText(i) for i in range(combo.count())]
    assert any("none saved" in label for label in labels), labels


def test_relabelling_the_options_does_not_look_like_the_user_changed_one(
    qapp, overview_widget
):
    """It runs on every objective refresh, so emitting `changed` would look like a
    settings edit to anything listening for one.

    Currently free: `setItemText` changes a label, not the selection. The assertion is
    on the requirement rather than on that mechanism, so it still holds an
    implementation that repopulated the items -- which would emit.
    """
    widget = overview_widget
    changes = []
    widget.settings_widget.changed.connect(lambda: changes.append(True))

    widget._refresh_objective_start_options()

    assert changes == []


def test_starting_from_an_unsaved_focus_is_refused_before_the_dialog(
    qapp, overview_widget, monkeypatch
):
    """Refused where the question was asked. The runner refuses too, but by then the
    dialog has said yes and the stage is moving."""
    from fibsem.fm.structures import ObjectiveStartPosition

    widget = overview_widget
    # `acquire` refuses on several counts before reaching this one, and the fixture is
    # module-scoped -- a run left alive by an earlier test bails at the first guard and
    # the status line still reads whatever that test put there. Start from a widget
    # that would otherwise go ahead, or this passes without exercising anything.
    widget._worker = None
    widget.fm.set_acquiring(False)
    widget.status.setText("")
    widget.fm.objective.insert()
    assert widget.at_acquisition_orientation(), "the stage is not where a run needs it"
    assert widget.fm.objective.state == "Inserted", "a different guard would fire first"

    widget.fm.objective._focus_position = None
    widget.settings_widget.combo_objective_start.set_value(ObjectiveStartPosition.FOCUS)
    shown = []
    monkeypatch.setattr(
        "fibsem.ui.fm.widgets.fm_overview_widget.FMOverviewConfirmationDialog",
        lambda **kwargs: shown.append(kwargs) or (_ for _ in ()).throw(AssertionError),
    )

    widget.acquire()

    assert shown == [], "the confirmation dialog was shown anyway"
    assert "focus position" in widget.status.text().lower()


def test_acquiring_with_the_objective_retracted_is_refused_before_the_dialog(
    qapp, overview_widget, monkeypatch
):
    """The runner refuses too, but by then the dialog has said yes."""
    widget = overview_widget
    widget._worker = None
    widget.fm.set_acquiring(False)
    widget.status.setText("")
    assert widget.at_acquisition_orientation()
    monkeypatch.setattr(type(widget.fm.objective), "state", property(lambda self: "Retracted"))
    shown = []
    monkeypatch.setattr(
        "fibsem.ui.fm.widgets.fm_overview_widget.FMOverviewConfirmationDialog",
        lambda **kwargs: shown.append(kwargs) or (_ for _ in ()).throw(AssertionError),
    )

    widget.acquire()

    assert shown == [], "the confirmation dialog was shown anyway"
    assert "retracted" in widget.status.text().lower()


def test_an_objective_that_cannot_be_asked_does_not_block_the_run(
    qapp, overview_widget, monkeypatch
):
    """A driver that cannot answer is not one answering "retracted". Refusing over a
    failed read would be the wrong way round; the runner checks again where it can
    raise."""
    widget = overview_widget

    def unreadable(self):
        raise RuntimeError("detector did not answer")

    monkeypatch.setattr(type(widget.fm.objective), "state", property(unreadable))

    assert widget._objective_state() is None


def test_loading_parameters_leaves_the_objective_start_combo_able_to_notify(qapp):
    """The `parameters` setter blocks every widget it writes and unblocks them from a
    second list. Left out of that list, the combo stays blocked for the rest of the
    widget's life -- so changing "Start from" would silently stop redrawing the grid,
    the summary and everything else downstream of `changed`.

    Latent until something loads a saved configuration, which is why it needs a test
    rather than a bug report: today only tests call the setter.
    """
    from fibsem.fm.structures import OverviewParameters
    from fibsem.ui.fm.widgets.fm_overview_settings_widget import FMOverviewSettingsWidget

    widget = FMOverviewSettingsWidget()
    widget.parameters = OverviewParameters(rows=2, cols=2)

    seen = []
    widget.changed.connect(lambda: seen.append(1))
    combo = widget.combo_objective_start
    combo.setCurrentIndex(1 - combo.currentIndex())

    assert seen, "the objective-start combo was left signal-blocked by the setter"
