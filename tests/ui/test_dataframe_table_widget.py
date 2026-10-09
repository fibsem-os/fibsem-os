"""The task history table sorts by value, and shows a duration as a clock (FIB-1191)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import Qt  # noqa: E402

from fibsem.ui.widgets.dataframe_table_widget import DataFrameTableWidget  # noqa: E402
from fibsem.util.durations import format_duration_as_clock  # noqa: E402


def _column(widget: DataFrameTableWidget, col: int) -> list:
    table = widget.table_widget
    return [table.item(row, col).text() for row in range(table.rowCount())]


def test_a_duration_shows_as_a_clock_and_sorts_by_its_seconds(qapp):
    df = pd.DataFrame({"Task": ["a", "b", "c"], "Duration": [245.0, 720.0, 3725.0]})
    widget = DataFrameTableWidget()
    widget.set_dataframe(df, display_formatters={"Duration": format_duration_as_clock})

    widget.table_widget.sortItems(1, Qt.AscendingOrder)
    # as text, "12:00" < "1:02:05" < "4:05"
    assert _column(widget, 1) == ["4:05", "12:00", "1:02:05"]


def test_numbers_sort_as_numbers(qapp):
    df = pd.DataFrame({"n": np.array([10, 9, 100], dtype=np.int64)})
    widget = DataFrameTableWidget(dataframe=df)
    widget.table_widget.sortItems(0, Qt.AscendingOrder)
    assert _column(widget, 0) == ["9", "10", "100"]
