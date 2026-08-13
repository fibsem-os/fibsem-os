"""The overviews on the canvas, listed in the settings column (FIB-543).

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_overview_list_widget.py
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from fibsem.ui.fm.widgets.fm_overview_widget import PlacedOverviewImageRecord
from fibsem.ui.widgets.overview_list_widget import OverviewListWidget

_app = QApplication.instance() or QApplication(sys.argv)


class _Metadata:
    """Only the three fields a record reads for its label and detail."""

    def __init__(self, grid="stitched_mosaic_3x3", pixel_size=1.2e-6):
        self.stage_position = type("Position", (), {"name": grid})()
        self.pixel_size_x = pixel_size
        self.acquisition_date = "2026-08-07T09:00:00"


def _record(index=1, label=None, visible=True, **kwargs):
    return PlacedOverviewImageRecord(
        id=f"overview-{index}",
        label=label or f"Overview {index}",
        metadata=_Metadata(**kwargs),
        visible=visible,
    )


def _widget(*records):
    widget = OverviewListWidget()
    widget.set_records(list(records))
    return widget


# ── contents ──────────────────────────────────────────────────────────────


def test_a_row_appears_for_each_overview_oldest_first():
    widget = _widget(_record(1), _record(2), _record(3))

    labels = [
        widget._list.itemWidget(widget._list.item(i)).name_label.text()
        for i in range(widget._list.count())
    ]

    assert labels == ["Overview 1", "Overview 2", "Overview 3"]


def test_an_empty_list_says_so_rather_than_showing_an_empty_box():
    widget = _widget()

    assert widget.empty_label.isVisible() or not widget.isVisible()
    assert widget._list.isHidden()


def test_a_row_shows_the_grid_and_scale_while_resting():
    widget = _widget(_record(1))
    row = widget._list.itemWidget(widget._list.item(0))

    assert row.detail_label.text() == "3×3 · 1.20 µm/px"


# ── hover and selection ───────────────────────────────────────────────────


def test_the_buttons_stay_out_of_the_way_until_a_row_is_wanted():
    """A settings column is mostly read, not clicked. Rows full of icons at rest turn
    a list you glance at into one you have to parse."""
    widget = _widget(_record(1))
    row = widget._list.itemWidget(widget._list.item(0))

    assert row.btn_remove.isHidden()
    assert row.detail_label.isHidden() is False


def test_selecting_a_row_brings_its_buttons_out():
    widget = _widget(_record(1), _record(2))

    widget.select("overview-2")

    first = widget._list.itemWidget(widget._list.item(0))
    second = widget._list.itemWidget(widget._list.item(1))
    assert second.btn_remove.isHidden() is False
    assert first.btn_remove.isHidden(), "an unselected row showed its buttons"


def test_a_hidden_overview_keeps_its_eye_while_resting():
    """The row has to say *why* it looks different. A greyed-out name with no icon
    reads as broken rather than hidden."""
    widget = _widget(_record(1, visible=False))
    row = widget._list.itemWidget(widget._list.item(0))

    assert row.btn_visible.isHidden() is False
    assert row.btn_remove.isHidden(), "only the eye earns its place at rest"
    assert row.name_label.isEnabled() is False


# ── what the rows report ──────────────────────────────────────────────────


def test_toggling_the_eye_reports_which_overview_and_which_way():
    widget = _widget(_record(1))
    row = widget._list.itemWidget(widget._list.item(0))
    seen = []
    widget.visibility_toggled.connect(lambda rid, vis: seen.append((rid, vis)))

    row.btn_visible.setChecked(True)   # checked means hidden
    row.btn_visible.setChecked(False)

    assert seen == [("overview-1", False), ("overview-1", True)]


def test_the_trash_asks_rather_than_removing_the_row_itself():
    """The widget does not own the canvas, so it reports and lets the owner decide --
    otherwise a refused removal would leave a list disagreeing with the canvas."""
    widget = _widget(_record(1))
    row = widget._list.itemWidget(widget._list.item(0))
    asked = []
    widget.remove_requested.connect(asked.append)

    row.btn_remove.click()

    assert asked == ["overview-1"]
    assert widget._list.count() == 1, "the row removed itself"


def test_refreshing_in_place_does_not_re_report_what_it_was_told():
    """`refresh` is called *from* the handler for a toggle, so a row that re-emitted on
    every refresh would go round in a loop."""
    widget = _widget(_record(1, visible=True))
    seen = []
    widget.visibility_toggled.connect(lambda rid, vis: seen.append((rid, vis)))

    widget.refresh([_record(1, visible=True)])

    assert seen == []


def test_a_refresh_that_does_change_the_state_moves_the_eye():
    widget = _widget(_record(1, visible=True))
    row = widget._list.itemWidget(widget._list.item(0))

    widget.refresh([_record(1, visible=False)])

    assert row.btn_visible.isChecked() is True
    assert row.name_label.isEnabled() is False


# ── selection across a rebuild ────────────────────────────────────────────


def test_the_selection_survives_a_new_overview_arriving():
    """The list is rebuilt whenever one is acquired, and losing the selection each time
    would make it unusable during a run."""
    widget = _widget(_record(1), _record(2))
    widget.select("overview-1")

    widget.set_records([_record(1), _record(2), _record(3)])

    assert widget.selected_id() == "overview-1"


def test_a_selection_that_was_removed_does_not_come_back():
    widget = _widget(_record(1), _record(2))
    widget.select("overview-2")

    widget.set_records([_record(1)])

    assert widget.selected_id() is None
