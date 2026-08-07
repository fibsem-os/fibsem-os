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
from fibsem.ui.fm.widgets.overview_list_widget import OverviewListWidget

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


# ── overview areas (FIB-532) ─────────────────────────────────────────────


def _area_list(qapp):
    from fibsem.ui.fm.widgets.overview_area_list_widget import OverviewAreaListWidget

    return OverviewAreaListWidget()


def test_an_area_row_says_how_much_of_its_grid_is_enabled(qapp):
    """The canvas is often zoomed somewhere else, so a masked area has to be visible
    from the list -- finding out an overnight run only covered four tiles afterwards is
    the failure this line exists to prevent."""
    from fibsem.fm.structures import OverviewArea
    from fibsem.ui.fm.widgets.overview_area_list_widget import describe_area

    full = OverviewArea(name="whole", rows=3, cols=3)
    masked = OverviewArea(
        name="holes", rows=3, cols=3,
        tile_mask=[[True] * 3, [True, False, True], [True] * 3],
    )

    # No count when there is nothing to say: "3×3 · 9/9 tiles" is noise on every row.
    assert describe_area(full) == "3×3"
    assert describe_area(masked) == "3×3 · 8/9 tiles"


def test_the_last_area_cannot_be_removed(qapp):
    """A run needs somewhere to happen. Disabled rather than hidden: a button that
    vanishes when the list gets short reads as a bug, and its tooltip is where the rule
    is explained."""
    from fibsem.fm.structures import OverviewArea

    widget = _area_list(qapp)
    widget.set_areas([OverviewArea(name="a"), OverviewArea(name="b")], 0)
    assert widget._rows[0].btn_remove.isEnabled()

    widget.set_areas([OverviewArea(name="a")], 0)
    assert not widget._rows[0].btn_remove.isEnabled()
    assert "at least one" in widget._rows[0].btn_remove.toolTip()


def test_rebuilding_the_list_is_not_a_user_selecting_something(qapp):
    """Every rebuild moves Qt's current row, and each move emits. Reported as a
    selection, they would each drive a canvas redraw and -- worse -- a rebuild that
    happened to pass through row 0 would look like the user picking the first area."""
    from fibsem.fm.structures import OverviewArea

    widget = _area_list(qapp)
    seen = []
    widget.selection_changed.connect(seen.append)

    widget.set_areas([OverviewArea(name="a"), OverviewArea(name="b")], 1)
    assert seen == [], "a rebuild reported itself as a selection"
    assert widget.selected_index() == 1

    widget.select(0)
    assert seen == [0], "a real selection went unreported"


def test_a_row_shows_the_state_of_a_run_without_being_rebuilt(qapp):
    """States arrive per region while a batch runs. Rebuilding to show them would drop
    the selection and the hover state on every area boundary."""
    from fibsem.fm.structures import OverviewArea
    from fibsem.ui.fm.widgets.overview_area_list_widget import STATE_DONE, STATE_RUNNING

    widget = _area_list(qapp)
    widget.set_areas([OverviewArea(name="a"), OverviewArea(name="b")], 0)
    rows = dict(widget._rows)

    widget.set_states({0: STATE_DONE, 1: STATE_RUNNING})

    assert widget._rows is not None and widget._rows[0] is rows[0], "the list rebuilt"
    assert widget._rows[0].state_label.text() == STATE_DONE
    assert widget._rows[1].state_label.text() == STATE_RUNNING
    # Nothing to say at rest, rather than three rows all reading "queued".
    widget.set_states({})
    assert widget._rows[0].state_label.text() == ""
    assert not widget._rows[0].state_label.isVisible()


def test_a_drag_reports_where_the_row_actually_landed(qapp):
    """Qt reports the insertion point *before* the source row is taken out, so a
    downward move arrives one past where it lands. Uncorrected, dragging the first area
    to the end would reorder the queue to something nobody asked for."""
    from fibsem.fm.structures import OverviewArea

    widget = _area_list(qapp)
    widget.set_areas(
        [OverviewArea(name="a"), OverviewArea(name="b"), OverviewArea(name="c")], 0
    )
    moves = []
    widget.reordered.connect(lambda a, b: moves.append((a, b)))

    # Downward: row 0 dropped after row 2, which Qt reports as insertion at 3.
    widget._on_rows_moved(None, 0, 0, None, 3)
    # Upward needs no correction: the source is below the insertion point.
    widget._on_rows_moved(None, 2, 2, None, 0)

    assert moves == [(0, 2), (2, 0)]
