"""
Widget for displaying a pandas DataFrame in a sortable table.
"""
from typing import Any, Optional, Callable
import pandas as pd
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette


class DataFrameTableWidget(QWidget):
    """A widget that displays a pandas DataFrame in a sortable table."""

    def __init__(self, dataframe: Optional[pd.DataFrame] = None,
                 cell_formatter: Optional[Callable[[int, int, Any], dict]] = None,
                 parent=None):
        """
        Initialize the DataFrameTableWidget.

        Args:
            dataframe: The pandas DataFrame to display
            cell_formatter: Optional callback function that takes (row_index, col_index, value)
                          and returns a dict with optional keys: 'color', 'background', 'bold'
            parent: The parent widget
        """
        super().__init__(parent)

        self.dataframe = dataframe
        self.cell_formatter = cell_formatter

        # Create layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create table widget
        self.table_widget = QTableWidget()
        self.table_widget.setSortingEnabled(True)
        self.table_widget.setAlternatingRowColors(True)

        # Set selection behavior
        self.table_widget.setSelectionBehavior(QTableWidget.SelectRows)
        self.table_widget.setSelectionMode(QTableWidget.SingleSelection)

        # Apply dark theme palette (matching autolamella_workflow_widget)
        palette = self.table_widget.palette()
        palette.setColor(QPalette.ColorRole.Base, QColor("#202020"))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#2c2c2c"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#dddddd"))
        palette.setColor(QPalette.ColorRole.Highlight, QColor("#3a6ea5"))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
        self.table_widget.setPalette(palette)

        layout.addWidget(self.table_widget)

        # Populate table if dataframe is provided
        if self.dataframe is not None:
            self.update_table()

    def set_dataframe(self, dataframe: pd.DataFrame,
                      cell_formatter: Optional[Callable[[int, int, Any], dict]] = None):
        """
        Set a new dataframe and update the table.

        Args:
            dataframe: The pandas DataFrame to display
            cell_formatter: Optional callback function for custom cell formatting
        """
        self.dataframe = dataframe
        if cell_formatter is not None:
            self.cell_formatter = cell_formatter
        self.update_table()

    def update_table(self):
        """Update the table widget with the current dataframe."""
        if self.dataframe is None or self.dataframe.empty:
            self.table_widget.setRowCount(0)
            self.table_widget.setColumnCount(0)
            return

        # Disable sorting temporarily for better performance
        self.table_widget.setSortingEnabled(False)

        # Set dimensions
        num_rows, num_cols = self.dataframe.shape
        self.table_widget.setRowCount(num_rows)
        self.table_widget.setColumnCount(num_cols)

        # Set column headers
        self.table_widget.setHorizontalHeaderLabels(self.dataframe.columns.tolist())

        # Populate table
        for i in range(num_rows):
            for j in range(num_cols):
                value = self.dataframe.iloc[i, j]

                # Convert value to string, handling various types
                if pd.isna(value):
                    display_value = ""
                else:
                    display_value = str(value)

                item = QTableWidgetItem(display_value)

                # Store the original value for proper sorting
                # This ensures numeric sorting works correctly
                if isinstance(value, (int, float)) and not pd.isna(value):
                    item.setData(Qt.UserRole, value)

                # Apply custom formatting if provided
                if self.cell_formatter is not None:
                    format_dict = self.cell_formatter(i, j, value)
                    if format_dict:
                        if 'color' in format_dict:
                            item.setForeground(QColor(format_dict['color']))
                        if 'background' in format_dict:
                            item.setBackground(QColor(format_dict['background']))
                        if format_dict.get('bold', False):
                            font = item.font()
                            font.setBold(True)
                            item.setFont(font)

                # Make items read-only
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)

                self.table_widget.setItem(i, j, item)

        # Resize columns to content
        self.table_widget.resizeColumnsToContents()

        # Set column resize mode - allow user to resize but start with content width
        header = self.table_widget.horizontalHeader()
        header.setStretchLastSection(True)
        for i in range(num_cols):
            header.setSectionResizeMode(i, QHeaderView.Interactive)

        # Re-enable sorting
        self.table_widget.setSortingEnabled(True)

    def clear(self):
        """Clear the table."""
        self.table_widget.setRowCount(0)
        self.table_widget.setColumnCount(0)
        self.dataframe = None


def main():
    """Example usage of DataFrameTableWidget."""
    import sys
    from PyQt5.QtWidgets import QApplication

    # Create sample dataframe
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'Age': [25, 30, 35, 28, 32],
        'Score': [95.5, 87.3, 92.1, 88.9, 91.0],
        'City': ['New York', 'London', 'Paris', 'Tokyo', 'Berlin']
    }
    df = pd.DataFrame(data)

    app = QApplication(sys.argv)

    # Create and show widget
    widget = DataFrameTableWidget(dataframe=df)
    widget.setWindowTitle("DataFrame Table Widget Example")
    widget.resize(600, 400)
    widget.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
