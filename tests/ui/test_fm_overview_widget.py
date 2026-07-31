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
from fibsem.ui.fm.widgets.tile_mask_widget import TileMaskWidget
from fibsem.ui.widgets.custom_widgets import ValueComboBox


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


def test_the_summary_reports_the_selection(qapp):
    widget = FMOverviewSettingsWidget()
    widget.parameters = OverviewParameters(rows=3, cols=3, tile_mask=plus_mask(3))

    assert "5 of 9 tiles" in widget.label_summary.text()


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
