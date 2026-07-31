"""Headless tests for the FM overview acquisition UI.

Covers the parts where being wrong is silent: what the tile mask means, whether the
settings round-trip into the object the acquisition actually consumes, and whether a
combo box shows the label it was given.

Uses PyQt5 directly with the offscreen platform (no pytest-qt dependency).
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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
from fibsem.ui.fm.widgets.fm_overview_widget import FMOverviewWidget, progress_slot
from fibsem.ui.fm.widgets.tile_mask_widget import TileMaskWidget
from fibsem.ui.widgets.custom_widgets import ValueComboBox
from fibsem.ui.widgets.progress_widget import FibsemProgressWidget, ProgressUpdate


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


def test_a_reset_bar_keeps_its_space(qapp):
    """The bars must not jitter each other.

    `FibsemProgressWidget.reset()` hides itself, so in a plain row the neighbouring
    bar would shift every time a tile finished. Each sits in a fixed slot instead.
    """
    from PyQt5.QtWidgets import QHBoxLayout, QWidget

    left, right = FibsemProgressWidget(), FibsemProgressWidget()
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.addWidget(progress_slot(left))
    layout.addWidget(progress_slot(right))
    row.resize(600, 40)
    row.show()
    qapp.processEvents()

    left.update_progress(ProgressUpdate.numeric(1, 2, "working"))
    right.update_progress(ProgressUpdate.numeric(1, 2, "working"))
    qapp.processEvents()
    before = left.parentWidget().geometry()

    right.reset()
    qapp.processEvents()

    assert left.parentWidget().geometry() == before


def test_the_bar_text_is_smaller_than_the_default(qapp):
    """Pinned because `setFont` on the holder silently does not reach the bar, so the
    obvious way to do this looks right and changes nothing."""
    from PyQt5.QtGui import QFontMetrics

    plain, shrunk = FibsemProgressWidget(), FibsemProgressWidget()
    slot = progress_slot(shrunk)  # noqa: F841 - keeps the holder alive

    sample = "Tiles — 4/9 · 74s remaining"
    assert (QFontMetrics(shrunk._bar.font()).width(sample)
            < QFontMetrics(plain._bar.font()).width(sample))


def test_shrinking_the_text_keeps_the_failed_colouring(qapp):
    """The chunk stylesheet is what distinguishes a failed run from a finished one."""
    progress = FibsemProgressWidget()
    slot = progress_slot(progress)  # noqa: F841

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


@pytest.fixture(scope="module")
def overview_widget(qapp):
    """A real widget against the Demo microscope, for the canvas/settings wiring."""
    from fibsem.ui.fm.overview_app import build_microscope
    from fibsem.ui.fm.widgets.fm_overview_widget import FMOverviewWidget

    microscope = build_microscope()
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
    overlay._on_canvas_clicked(x + tw / 2, y + th / 2, None)
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
        overlay._on_canvas_clicked(x + tw / 2, y + th / 2, None)
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
    original = widget._live_geometry
    widget._live_geometry = lambda: (None, None, None)
    try:
        widget.set_positions([_offset(widget._current_stage_position(), name="nowhere")])
        qapp.processEvents()
        assert widget.position_overlay._points == []
    finally:
        widget._live_geometry = original


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
        # Only what set_image placed: the fixture seeds the canvas directly, which
        # lands under the widget's default key rather than an overview one.
        return [k for k in widget.canvas.canvas.placed_keys if k.startswith("overview@")]

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
    widget._displayed_image = None
    widget._origin = None

    widget._refresh_stage_metadata()

    labels = [spec.label for spec in widget.stage_overlay._specs]
    assert "Stage limits" in labels
    assert "Grid boundary" in labels


def test_stage_metadata_is_dropped_when_the_geometry_is_unknown(qapp, overview_widget):
    """Without a geometry there is no frame, and drawing at a guessed scale would put
    plausible-looking shapes in the wrong places."""
    widget = overview_widget
    widget._displayed_image = None
    widget._origin = None
    original = widget._live_geometry
    widget._live_geometry = lambda: (None, None, None)
    try:
        widget._refresh_stage_metadata()
        assert widget.stage_overlay._specs == []
    finally:
        widget._live_geometry = original
