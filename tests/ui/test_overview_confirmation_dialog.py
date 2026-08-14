"""What the FIB/SEM overview's pre-flight dialog says.

Its job is to state the two things a run carries that are *not* visible in the settings
column: where the grid was dragged to, and which tiles are masked off. Both are set on
the canvas, both survive a tab switch, and neither shows up anywhere else at the moment
Acquire is pressed.

Built here from plain settings objects rather than through the widget -- the wiring is
covered in `test_overview_widget.py`, and what is checked here is the wording.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_overview_confirmation_dialog.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

# CI installs `.[test]`, not `.[ui]`, so PyQt5 is absent there.
pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QLabel  # noqa: E402

from fibsem.structures import (  # noqa: E402
    ImageSettings,
    OverviewAcquisitionSettings,
)
from fibsem.ui.widgets.overview_confirmation_dialog import (  # noqa: E402
    OverviewConfirmationDialog,
)

_app = QApplication.instance() or QApplication(sys.argv)


def _settings(**kwargs) -> OverviewAcquisitionSettings:
    image = ImageSettings(
        resolution=(1024, 1024),
        hfw=500e-6,
        dwell_time=1e-6,
        autocontrast=True,
        filename="overview-image",
    )
    image_overrides = kwargs.pop("image", {})
    for key, value in image_overrides.items():
        setattr(image, key, value)
    return OverviewAcquisitionSettings(image_settings=image, **kwargs)


def _rows(dialog) -> dict:
    return dict(dialog._rows())


@pytest.fixture
def dialog():
    built = []

    def build(**kwargs):
        d = OverviewConfirmationDialog(**kwargs)
        built.append(d)
        return d

    yield build
    for d in built:
        d.close()


class TestWhereTheRunWillHappen:
    """The half of this dialog that nothing else says."""

    def test_no_target_means_the_stage(self, dialog):
        assert dialog(settings=_settings())._centre_text() == "the stage position"

    def test_a_dragged_grid_says_how_far_and_which_way(self, dialog):
        """Components, not a distance: the grid is dragged in x and y and checked
        against the canvas in x and y, and "520 µm away" does not say which way."""
        text = dialog(settings=_settings(), offset=(500e-6, -150e-6))._centre_text()
        assert "+500" in text
        assert "-150" in text
        assert "from the stage position" in text

    def test_floating_point_dust_is_not_a_dragged_grid(self, dialog):
        """A target resolved through the projection lands a few nanometres off where it
        started. Reported literally, that reads as a grid someone moved."""
        text = dialog(settings=_settings(), offset=(2e-9, -5e-9))._centre_text()
        assert text == "the stage position"

    def test_the_view_is_named_when_there_is_one(self, dialog):
        rows = _rows(dialog(
            settings=_settings(),
            view_description="Ion beam, stage at the MILLING orientation.",
        ))
        assert rows["Acquired in"] == "Ion beam, stage at the MILLING orientation."

    def test_no_view_leaves_the_row_out_rather_than_guessing(self, dialog):
        """The tab cannot always name one -- an unrecognised pose gives no orientation.
        A row reading "unknown" is worse than no row."""
        assert "Acquired in" not in _rows(dialog(settings=_settings()))


class TestWhatWillBeAcquired:
    def test_the_meta_line_carries_the_grid_and_what_it_covers(self, dialog):
        line = dialog(settings=_settings())._meta_line()
        assert "3 × 3 grid" in line
        assert "1500 × 1500" in line
        assert "typewriter order" in line

    def test_a_masked_run_counts_only_what_it_will_acquire(self, dialog):
        """The chips are the only place the mask is stated in words."""
        settings = _settings(
            tile_mask=[[True, False, False], [False, True, False], [False, False, True]]
        )
        d = dialog(settings=settings)
        labels = [label.text() for label in d.findChildren(QLabel)]
        assert any("3 to acquire" in text for text in labels), labels
        assert any("6 skipped" in text for text in labels), labels

    def test_a_full_run_does_not_mention_skipping(self, dialog):
        d = dialog(settings=_settings())
        labels = [label.text() for label in d.findChildren(QLabel)]
        assert any("9 to acquire" in text for text in labels), labels
        assert not any("skipped" in text for text in labels), labels

    def test_nothing_selected_cannot_be_started(self, dialog):
        """The runner refuses an empty mask, so the refusal belongs before the dialog is
        dismissed rather than after."""
        settings = _settings(tile_mask=[[False] * 3 for _ in range(3)])
        d = dialog(settings=settings)
        assert not d.button_start.isEnabled()
        assert "No tiles" in d.button_start.toolTip()

    def test_the_tile_row_gives_pixels_and_ground(self, dialog):
        rows = _rows(dialog(settings=_settings()))
        assert "1024 × 1024 px" in rows["Tile"]
        assert "500" in rows["Tile"]

    def test_the_dwell_time_is_in_microseconds_not_micrometres(self, dialog):
        """`MICRON_SYMBOL` is "µm" already, so appending an "s" to it gave "µms" --
        which read as a unit right up until you looked at it."""
        rows = _rows(dialog(settings=_settings()))
        assert rows["Dwell time"] == "1.00 µs"

    def test_where_it_will_be_written_when_that_is_known(self, dialog):
        """The filename names the tile sub-folder, so two runs under one name interleave
        on disk. It is worth a row."""
        rows = _rows(dialog(settings=_settings(image={"path": "/data/exp-1"})))
        assert rows["Saving to"] == "/data/exp-1/overview-image"

    def test_no_destination_leaves_the_row_out(self, dialog):
        assert "Saving to" not in _rows(dialog(settings=_settings()))


class TestTheTimeItQuotes:
    """Scan time, under its own name, because that is the part that is arithmetic.

    A wall-clock estimate would be mostly an unmeasured guess at stage settling -- see
    `OverviewAcquisitionSettings.scan_time`. Quoting that as "Estimated time" is the
    failure this wording avoids.
    """

    def test_it_reports_scan_time_not_a_run_duration(self, dialog):
        rows = _rows(dialog(settings=_settings()))
        assert "Estimated time" not in rows
        assert "before stage movement" in rows["Scan time"]

    def test_it_matches_the_settings_it_was_given(self, dialog):
        settings = _settings()
        rows = _rows(dialog(settings=settings))
        # 9 tiles x 1024^2 px x 1 us = 9.4 s, and 1.0 s each.
        assert settings.scan_time == pytest.approx(9.437184)
        assert rows["Scan time"].startswith("9s")
        assert "1s per tile" in rows["Scan time"]

    def test_a_masked_run_quotes_only_the_tiles_it_scans(self, dialog):
        settings = _settings(
            tile_mask=[[True, False, False], [False, True, False], [False, False, True]]
        )
        rows = _rows(dialog(settings=settings))
        assert rows["Scan time"].startswith("3s")


def test_the_three_preflight_dialogs_share_one_style():
    """The coincidence dialog copied nothing from the FM one and imported instead,
    leaving a note to lift the shared parts out if a third appeared. This is the third,
    and the note was followed rather than the import chain extended."""
    from fibsem.ui.fm.widgets import fm_overview_confirmation_dialog as fm
    from fibsem.ui.widgets import coincidence_milling_confirmation_dialog as milling
    from fibsem.ui.widgets import overview_confirmation_dialog as beam
    from fibsem.ui.widgets import preflight

    for module in (fm, milling, beam):
        assert module.chip is preflight.chip
        assert module.detail_block is preflight.detail_block
        assert module.format_duration is preflight.format_duration
