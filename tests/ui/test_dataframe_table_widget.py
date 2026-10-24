"""The DataFrame table sorts numbers by value."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import Qt  # noqa: E402

from fibsem.ui.widgets.dataframe_table_widget import DataFrameTableWidget  # noqa: E402


def _column(widget: DataFrameTableWidget, col: int) -> list:
    table = widget.table_widget
    return [table.item(row, col).text() for row in range(table.rowCount())]


def test_numbers_sort_as_numbers(qapp):
    df = pd.DataFrame({"n": np.array([10, 9, 100], dtype=np.int64)})
    widget = DataFrameTableWidget(dataframe=df)
    widget.table_widget.sortItems(0, Qt.AscendingOrder)
    assert _column(widget, 0) == ["9", "10", "100"]
