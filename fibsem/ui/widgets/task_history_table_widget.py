"""
Widget for displaying experiment task history in a sortable table.
"""
from typing import Optional
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QCheckBox

from fibsem.applications.autolamella.structures import Experiment, AutoLamellaTaskStatus
from fibsem.ui.widgets.dataframe_table_widget import DataFrameTableWidget
from fibsem.ui.widgets.task_summary_formatting import (
    COLUMN_NAME_MAPPING,
    format_duration_short,
    make_status_cell_formatter,
)


class TaskHistoryTableWidget(QWidget):
    """A widget that displays the task history from an Experiment in a sortable table."""

    def __init__(self, experiment: Optional[Experiment] = None, parent=None):
        """
        Initialize the TaskHistoryTableWidget.

        Args:
            experiment: The Experiment object to display task history from
            parent: The parent widget
        """
        super().__init__(parent)

        self.experiment = experiment

        # Create layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Add title
        title_layout = QHBoxLayout()
        self.title_label = QLabel("Task History")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        title_layout.addWidget(self.title_label)

        # Add filter checkboxes
        self.display_status_message_checkbox = QCheckBox("Display Status Message")
        self.display_status_message_checkbox.setChecked(False)
        self.display_status_message_checkbox.stateChanged.connect(self.refresh)
        title_layout.addWidget(self.display_status_message_checkbox)

        self.display_errors_only_checkbox = QCheckBox("Display Errors Only")
        self.display_errors_only_checkbox.setChecked(False)
        self.display_errors_only_checkbox.stateChanged.connect(self.refresh)
        title_layout.addWidget(self.display_errors_only_checkbox)

        # Add refresh button
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)
        title_layout.addWidget(self.refresh_button)
        title_layout.addStretch()

        main_layout.addLayout(title_layout)

        # Create the dataframe table widget
        self.table_widget = DataFrameTableWidget()
        main_layout.addWidget(self.table_widget)

        # Populate table if experiment is provided
        if self.experiment is not None:
            self.refresh()

    def set_experiment(self, experiment: Experiment):
        """
        Set a new experiment and update the table.

        Args:
            experiment: The Experiment object to display task history from
        """
        self.experiment = experiment
        self.refresh()

    def refresh(self):
        """Refresh the table with the latest task history from the experiment."""
        if self.experiment is None:
            self.table_widget.clear()
            return

        # Get the task history dataframe
        df = self.experiment.task_history_dataframe()

        # Filter to only show errors if checkbox is checked
        if self.display_errors_only_checkbox.isChecked():
            if 'task_status' in df.columns:
                # Filter for Failed status
                df = df[df['task_status'] == AutoLamellaTaskStatus.Failed.name]

        # Filter to only include specified columns
        columns_to_include = [
            'lamella_name',
            'task_name',
            # 'task_type',
            'task_status',
        ]

        # Conditionally add status message column
        if self.display_status_message_checkbox.isChecked():
            columns_to_include.append('task_status_message')

        # Add remaining columns
        columns_to_include.extend([
            'completed_at',
            'duration'
        ])

        # Only select columns that exist in the dataframe
        available_columns = [col for col in columns_to_include if col in df.columns]
        df = df[available_columns]

        # Format duration column as MMm:SSs
        if 'duration' in df.columns:
            df['duration'] = df['duration'].apply(format_duration_short)

        # Rename columns to nice titles
        df = df.rename(columns=COLUMN_NAME_MAPPING)

        # Update the table with status-based cell colouring
        self.table_widget.set_dataframe(df, cell_formatter=make_status_cell_formatter(df))

    def clear(self):
        """Clear the table."""
        self.table_widget.clear()
        self.experiment = None


def main():
    """Example usage of TaskHistoryTableWidget."""
    import sys
    import os
    from PyQt5.QtWidgets import QApplication

    # Load example experiment
    PATH = "/home/patrick/github/fibsem/fibsem/applications/autolamella/log/AutoLamella-2025-12-14-16-27"
    if os.path.exists(os.path.join(PATH, "experiment.yaml")):
        exp = Experiment.load(os.path.join(PATH, "experiment.yaml"))

        app = QApplication(sys.argv)

        # Create and show widget
        widget = TaskHistoryTableWidget(experiment=exp)
        widget.setWindowTitle("Task History Table Widget Example")
        widget.resize(1000, 600)
        widget.show()

        sys.exit(app.exec_())
    else:
        print(f"Example experiment not found at {PATH}")


if __name__ == "__main__":
    main()
