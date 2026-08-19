"""The dialog that joins the two panes.

Most of what it does is arithmetic already tested elsewhere. What is only testable here
is the joining: that a change on the left reaches the right, that the counter and the
button quote the same number, and that what comes back out is what the acquisition needs
and nothing it does not.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QDialog

from fibsem.fm.structures import (
    AutoFocusMode,
    ChannelSettings,
    OverviewParameters,
    ZParameters,
)
from fibsem.structures import TileOrderStrategy
from fibsem.ui.fm.widgets.fm_sparse_selection_dialog import FMSparseSelectionDialog


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def microscope(qapp):
    from fibsem.ui.fm.overview_app import build_microscope

    return build_microscope()


@pytest.fixture(scope="module")
def views(microscope):
    """Beam overviews, acquired once -- they cost a couple of seconds each."""
    from fibsem.ui.fm.sparse_selection_app import acquire_beam_overviews

    acquired = acquire_beam_overviews(microscope)
    microscope.move_to_microscope("FM")
    return acquired


CHANNELS = [
    ChannelSettings(
        name="DAPI",
        excitation_wavelength=405,
        emission_wavelength=450,
        power=0.5,
        exposure_time=0.1,
    )
]


def parameters(**kwargs):
    base = dict(
        rows=11, cols=15, overlap=0.1, autofocus_mode=AutoFocusMode.NONE,
        tile_order=TileOrderStrategy.SERPENTINE,
    )
    base.update(kwargs)
    return OverviewParameters(**base)


@pytest.fixture()
def dialog(microscope, views):
    return FMSparseSelectionDialog(
        microscope, views, parameters(), channel_settings=CHANNELS
    )


def with_regions(dialog, count=2):
    for _ in range(count):
        dialog.selector.add_region()
    return dialog


# ── nothing selected ─────────────────────────────────────────────────────


def test_nothing_can_be_accepted_before_anything_is_selected(dialog):
    assert dialog.accept_button.isEnabled() is False
    assert dialog.accept_button.text() == "Use these tiles"


def test_accepting_with_nothing_selected_does_nothing(dialog):
    """The button is disabled, so this is only reachable by a caller. The runner would
    refuse an all-disabled mask anyway; refusing here keeps the result honest rather
    than handing back a plan that does not exist."""
    dialog._accept()

    assert dialog.selection is None
    assert dialog.result() != QDialog.Accepted


# ── the two panes move together ──────────────────────────────────────────


def test_drawing_a_region_reaches_the_other_pane(dialog):
    assert dialog.preview.plan is None

    with_regions(dialog, 1)

    assert dialog.preview.plan is not None
    assert dialog.preview.enabled_tiles > 0


def test_clearing_the_regions_empties_the_other_pane(dialog):
    with_regions(dialog, 2)

    dialog.selector.clear_regions()

    assert dialog.preview.plan is None
    assert dialog.accept_button.isEnabled() is False


def test_the_grid_is_planned_at_the_overlap_it_was_given(microscope, views):
    """The one setting that changes the geometry. Everything else is carried through."""
    sparse = FMSparseSelectionDialog(
        microscope, views, parameters(overlap=0.0), channel_settings=CHANNELS
    )
    dense = FMSparseSelectionDialog(
        microscope, views, parameters(overlap=0.5), channel_settings=CHANNELS
    )
    with_regions(sparse, 1)
    with_regions(dense, 1)

    assert dense.preview.plan.cols > sparse.preview.plan.cols


# ── the button and the counter agree ─────────────────────────────────────


def test_the_button_quotes_the_number_the_counter_does(dialog):
    """One set of numbers for both, deliberately: a count computed in one place and
    restated in the other is how a button comes to offer a different number from the
    line above it."""
    with_regions(dialog, 2)

    enabled = dialog.preview.enabled_tiles
    assert dialog.accept_button.text() == f"Use {enabled} tiles"
    assert f"{enabled} of {dialog.preview.total_tiles} tiles" in dialog._status.text()


def test_toggling_a_tile_moves_both(dialog):
    with_regions(dialog, 1)
    before = dialog.preview.enabled_tiles

    dialog.preview._on_tile_toggled(0, 0, not dialog.preview.mask[0][0])

    assert dialog.preview.enabled_tiles != before
    assert dialog.accept_button.text() == f"Use {dialog.preview.enabled_tiles} tiles"


def test_the_counter_names_the_selected_region(dialog):
    with_regions(dialog, 2)

    assert "Region 2 of 2" in dialog._status.text()


# ── the estimate ─────────────────────────────────────────────────────────


def test_the_counter_reports_how_long_the_run_would_take(dialog):
    with_regions(dialog, 1)

    assert "s" in dialog._status.text().rsplit("·", 1)[-1]


def test_without_channel_settings_it_counts_tiles_and_quotes_no_time(
    microscope, views
):
    """Rather than a duration it cannot stand behind: the time depends on exposure and
    channel count, and a number invented here would be quoted back as if it meant
    something."""
    dialog = FMSparseSelectionDialog(microscope, views, parameters())
    with_regions(dialog, 1)

    enabled, total, duration = dialog._counts()

    assert enabled > 0 and total > 0
    assert duration is None


def test_a_z_stack_is_only_costed_when_the_run_is_one(microscope, views):
    """`use_zstack` is what decides, not the presence of z parameters -- the settings
    keep theirs when the switch is off."""
    zparams = ZParameters(zmin=-5e-6, zmax=5e-6, zstep=1e-6)
    off = FMSparseSelectionDialog(
        microscope, views, parameters(use_zstack=False), CHANNELS, zparams
    )
    on = FMSparseSelectionDialog(
        microscope, views, parameters(use_zstack=True), CHANNELS, zparams
    )
    with_regions(off, 1)
    with_regions(on, 1)

    assert off._counts()[2] < on._counts()[2]


# ── what comes back ──────────────────────────────────────────────────────


def test_accepting_returns_the_grid_the_selection_derived(dialog):
    with_regions(dialog, 2)
    plan = dialog.preview.plan

    dialog._accept()

    result = dialog.selection
    assert (result.parameters.rows, result.parameters.cols) == (plan.rows, plan.cols)
    assert result.parameters.tile_mask == dialog.preview.mask


def test_accepting_keeps_the_tiles_toggled_by_hand(dialog):
    """The result is the *edited* mask, not the one the regions produced.

    They differ only after a hand toggle, so a test that never makes one cannot tell
    them apart -- and taking the plan's mask would silently discard every adjustment at
    the last moment.
    """
    with_regions(dialog, 1)
    plan = dialog.preview.plan
    dialog.preview._on_tile_toggled(0, 0, not dialog.preview.mask[0][0])
    assert dialog.preview.mask != plan.mask, "the edit has to have changed something"

    dialog._accept()

    assert dialog.selection.parameters.tile_mask == dialog.preview.mask


def test_accepting_carries_everything_that_is_not_geometry(dialog):
    """Overlap, autofocus and tile order are settings, not shape. The caller gets one
    object back rather than having to merge two."""
    with_regions(dialog, 1)

    dialog._accept()

    result = dialog.selection.parameters
    assert result.overlap == 0.1
    assert result.autofocus_mode is AutoFocusMode.NONE
    assert result.tile_order is TileOrderStrategy.SERPENTINE


def test_the_centre_that_comes_back_is_posed_for_fluorescence(dialog, microscope):
    """The runner drives the stage to it. At a beam pose it would go somewhere else."""
    with_regions(dialog, 1)

    dialog._accept()

    fm = microscope.get_orientation("FM")
    assert dialog.selection.centre_position.t == pytest.approx(fm.t)
    assert dialog.selection.centre_position.r == pytest.approx(fm.r)


def test_accepting_does_not_mutate_the_parameters_it_was_given(microscope, views):
    """A caller that cancels, or that keeps its own settings object, must not find rows
    and columns rewritten underneath it."""
    given = parameters()
    dialog = FMSparseSelectionDialog(microscope, views, given, CHANNELS)
    with_regions(dialog, 1)

    dialog._accept()

    assert (given.rows, given.cols) == (11, 15)
    assert given.tile_mask is None


def test_cancelling_returns_nothing(dialog):
    with_regions(dialog, 1)

    dialog.reject()

    assert dialog.selection is None
