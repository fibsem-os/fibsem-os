"""Widget for displaying experiment task summary images across all lamellae."""

import logging
import os
from typing import TYPE_CHECKING, List, Optional

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.tools.reporting import (
    plot_experiment_task_summary, Experiment
)

if TYPE_CHECKING:
    from fibsem.applications.autolamella.ui import AutoLamellaUI


# Stylesheet constants
COMBOBOX_STYLESHEET = """
    QComboBox {
        background-color: #3a3a3a;
        color: white;
        border: 1px solid #555;
        padding: 4px 8px;
        font-size: 10px;
        min-width: 150px;
    }
    QComboBox:hover {
        background-color: #4a4a4a;
    }
    QComboBox::drop-down {
        border: none;
    }
    QComboBox QAbstractItemView {
        background-color: #3a3a3a;
        color: white;
        selection-background-color: #4a4a4a;
    }
"""

BUTTON_STYLESHEET = """
    QPushButton {
        background-color: #3a3a3a;
        color: white;
        border: 1px solid #555;
        padding: 4px 8px;
        font-size: 10px;
    }
    QPushButton:hover {
        background-color: #4a4a4a;
    }
    QPushButton:pressed {
        background-color: #2a2a2a;
    }
"""


class ExperimentTaskSummaryWidget(QWidget):
    """Widget for displaying task summary images across all lamellae in an experiment.

    This widget displays a grid of images showing SEM and FIB views at different resolutions
    for a specific task across all lamellae that have completed that task. Users can navigate
    between different tasks using a dropdown selector.
    """

    def __init__(self, parent: Optional['AutoLamellaUI'] = None):
        """Initialize the experiment task summary widget.

        Args:
            parent: Parent AutoLamellaUI widget (optional)
        """
        super().__init__(parent)
        self.parent_widget = parent
        self.experiment: Optional['Experiment'] = None
        self.current_task: Optional[str] = None
        self.current_figure: Optional[Figure] = None

        self.initUI()

    def initUI(self):
        """Initialize the widget UI components."""
        self.setContentsMargins(0, 0, 0, 0)

        layout = QVBoxLayout()

        # Title and controls layout
        controls_layout = QHBoxLayout()

        # Title label
        title_label = QLabel("Experiment Task Summary")
        title_label.setStyleSheet("color: white; font-size: 12px; font-weight: bold;")
        controls_layout.addWidget(title_label)

        controls_layout.addStretch()

        # Task selector
        self.task_label = QLabel("Task:")
        self.task_label.setStyleSheet("color: #bbbbbb; font-size: 10px;")
        controls_layout.addWidget(self.task_label)

        self.task_selector = QComboBox(self)
        self.task_selector.setStyleSheet(COMBOBOX_STYLESHEET)
        self.task_selector.currentIndexChanged.connect(self._on_task_changed)
        controls_layout.addWidget(self.task_selector)

        layout.addLayout(controls_layout)

        # Create matplotlib canvas (initially without a figure)
        self.canvas = None
        self.figure = None

        # Create a placeholder widget for the canvas
        self.canvas_container = QWidget()
        self.canvas_layout = QVBoxLayout(self.canvas_container)
        self.canvas_layout.setContentsMargins(0, 0, 0, 0)

        # Create initial empty figure
        self._create_empty_canvas()

        # Wrap canvas container in a scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.canvas_container)
        self.scroll_area.setWidgetResizable(False)  # Don't resize widget to fit scroll area
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: #262930;
                border: none;
            }
        """)

        layout.addWidget(self.scroll_area)

        # Button layout
        button_layout = QHBoxLayout()

        # Refresh button
        self.refresh_button = QPushButton("Refresh", self)
        self.refresh_button.setStyleSheet(BUTTON_STYLESHEET)
        self.refresh_button.clicked.connect(self.update_summary)
        button_layout.addWidget(self.refresh_button)

        # Export button
        self.export_button = QPushButton("Export Figure", self)
        self.export_button.setStyleSheet(BUTTON_STYLESHEET)
        self.export_button.clicked.connect(self._on_export_clicked)
        button_layout.addWidget(self.export_button)

        # Clear button
        self.clear_button = QPushButton("Clear", self)
        self.clear_button.setStyleSheet(BUTTON_STYLESHEET)
        self.clear_button.clicked.connect(self.clear_summary)
        button_layout.addWidget(self.clear_button)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Info label
        self.info_label = QLabel("No experiment loaded")
        self.info_label.setStyleSheet("color: #bbbbbb; font-size: 10px;")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.info_label)

        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def _create_empty_canvas(self):
        """Create an empty canvas with placeholder text."""
        # Create empty figure
        self.figure = Figure(figsize=(10, 6), dpi=80)
        self.figure.patch.set_facecolor("#262930")

        # Create canvas from figure
        if self.canvas is not None:
            self.canvas_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()

        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(400, 300)
        self.canvas_layout.addWidget(self.canvas)

        # Add placeholder text
        ax = self.figure.add_subplot(111)
        ax.set_facecolor("black")
        ax.text(
            0.5,
            0.5,
            "No experiment task data available",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=12,
            color="#bbbbbb",
        )
        ax.set_title("Experiment Task Summary", color="white")
        ax.axis("off")
        self.figure.tight_layout()
        self.canvas.draw()

    def _replace_canvas_with_figure(self, new_figure):
        """Replace the current canvas with a new figure.

        Args:
            new_figure: Matplotlib Figure object to display
        """
        # Remove old canvas
        if self.canvas is not None:
            self.canvas_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()

        # Store the new figure
        if self.figure is not None and self.figure != new_figure:
            plt.close(self.figure)

        self.figure = new_figure

        # Apply dark theme to the figure
        self.figure.patch.set_facecolor("#262930")

        # Create new canvas with the figure
        self.canvas = FigureCanvas(self.figure)

        # Set fixed size based on figure size for proper scrolling
        # Get the figure size in inches and DPI
        fig_width, fig_height = self.figure.get_size_inches()
        dpi = self.figure.get_dpi()

        # Set the canvas height to the exact size of the figure
        # Width will match the scroll area viewport width
        canvas_height = int(fig_height * dpi)

        # Get scroll area width (subtract some pixels for potential scrollbar)
        scroll_width = self.scroll_area.viewport().width()

        # Set canvas to fill width, fixed height
        self.canvas.setFixedHeight(canvas_height)
        self.canvas.setMinimumWidth(scroll_width)
        self.canvas.setMaximumWidth(scroll_width)

        # Update the container size to match
        self.canvas_container.setFixedHeight(canvas_height)
        self.canvas_container.setMinimumWidth(scroll_width)
        self.canvas_container.setMaximumWidth(scroll_width)

        self.canvas_layout.addWidget(self.canvas)

        # Draw the canvas
        self.canvas.draw()

    def set_experiment(self, experiment: 'Experiment'):
        """Set the experiment and populate the task selector.

        Args:
            experiment: The Experiment object to display
        """
        self.experiment = experiment

        # Clear and populate task selector
        self.task_selector.blockSignals(True)  # Prevent triggering update during population
        self.task_selector.clear()

        if experiment is not None and hasattr(experiment, 'positions'):
            # Collect all unique completed tasks across all lamellae
            all_tasks = set()
            for lamella in experiment.positions:
                if hasattr(lamella, 'task_history'):
                    all_tasks.update([task.name for task in lamella.task_history])

            # Sort tasks alphabetically
            sorted_tasks = sorted(list(all_tasks))

            if sorted_tasks:
                for task_name in sorted_tasks:
                    # Count how many lamellae have completed this task
                    count = sum(
                        1 for lamella in experiment.positions
                        if hasattr(lamella, 'task_history') and
                        lamella.has_completed_task(task_name)
                    )
                    display_name = f"{task_name} ({count} lamellae)"
                    self.task_selector.addItem(display_name, userData=task_name)

                # Set the first task as current
                self.current_task = sorted_tasks[0]
                self.info_label.setText(
                    f"Experiment: {experiment.name} | "
                    f"{len(sorted_tasks)} unique tasks"
                )
            else:
                self.info_label.setText(f"Experiment: {experiment.name} | No completed tasks found")
                self.current_task = None
        else:
            self.info_label.setText("No experiment loaded")
            self.current_task = None

        self.task_selector.blockSignals(False)

        # Update the display
        self.update_summary()

    def _on_task_changed(self, index: int):
        """Handle task selection change.

        Args:
            index: The index of the selected task in the combo box
        """
        if index >= 0:
            self.current_task = self.task_selector.itemData(index)
            self.update_summary()

    def update_summary(self):
        """Update the experiment task summary display for the current task."""

        if self.current_task is None or self.experiment is None:
            self._create_empty_canvas()
            return

        try:
            # Generate the figure
            fig = plot_experiment_task_summary(
                self.experiment,
                self.current_task,
                show_title=True,
                figsize=(10, 5),
                target_size=512,
                fontsize=12,
                mode="dark",
                show=False
            )

            if fig is None:
                # No valid images found for this task
                self._create_empty_canvas()
                self.info_label.setText(
                    f"Task: {self.current_task} | No valid images found"
                )
                return

            # Store the current figure
            self.current_figure = fig

            # Replace canvas with the new figure
            self._replace_canvas_with_figure(fig)

            # Count lamellae with this task
            count = sum(
                1 for lamella in self.experiment.positions
                if hasattr(lamella, 'task_history') and
                lamella.has_completed_task(self.current_task)
            )

            # Update info label
            self.info_label.setText(
                f"Task: {self.current_task} | "
                f"{count} lamellae completed"
            )

        except ImportError as e:
            logging.error(f"Could not import reporting module: {e}")
            self._create_empty_canvas()
            self.info_label.setText("Error: Reporting module not available")
        except Exception as e:
            logging.error(f"Error updating experiment task summary: {e}")
            import traceback
            traceback.print_exc()
            self._create_empty_canvas()
            self.info_label.setText(f"Error: {str(e)}")

    def clear_summary(self):
        """Clear the current summary display."""
        self.current_task = None
        self.current_figure = None
        self.task_selector.clear()
        self._create_empty_canvas()
        self.info_label.setText("No experiment loaded")

    def _on_export_clicked(self):
        """Handle export button click to save the current figure."""
        if self.current_figure is None:
            return

        # Determine starting directory from experiment path if available
        start_dir = ""
        if self.experiment is not None:
            start_dir = str(self.experiment.path)

        # Default filename
        default_name = "experiment_task_summary.png"
        if self.current_task is not None:
            default_name = f"experiment_task_summary_{self.current_task}.png"

        if start_dir:
            default_path = os.path.join(start_dir, default_name)
        else:
            default_path = default_name

        # Open file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Experiment Task Summary",
            default_path,
            "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*.*)"
        )

        if not file_path:
            # User cancelled
            return

        try:
            # Re-generate the figure with the reporting function for clean export
            if self.current_task is not None and self.experiment is not None:
                fig = plot_experiment_task_summary(
                    self.experiment,
                    self.current_task,
                    show_title=True,
                    figsize=(30, 5),
                    show=False
                )

                if fig is not None:
                    # Save with high DPI
                    fig.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
                    logging.info(f"Exported experiment task summary to: {file_path}")

                    # Clean up
                    plt.close(fig)
                else:
                    logging.warning("No figure to export")
        except Exception as e:
            logging.error(f"Error exporting figure: {e}")
            import traceback
            traceback.print_exc()


def create_experiment_task_summary_widget(experiment: 'Experiment',
                                          parent: Optional['AutoLamellaUI'] = None) -> QDialog:
    """Create and initialize an ExperimentTaskSummaryWidget wrapped in a dialog.

    Args:
        experiment: The Experiment object to display
        parent: Optional parent AutoLamellaUI widget

    Returns:
        QDialog: Dialog containing the initialized widget with the experiment loaded
    """
    # Create dialog
    dialog = QDialog(parent)
    dialog.setWindowTitle(f"Experiment Task Summary - {experiment.name}")
    dialog.setMinimumSize(800, 600)

    # Create layout
    layout = QVBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)

    # Create and add widget
    widget = ExperimentTaskSummaryWidget(parent=parent)
    widget.set_experiment(experiment)
    layout.addWidget(widget)

    dialog.setLayout(layout)

    return dialog


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    # Create dummy dialog for testing
    dialog = QDialog()
    dialog.setWindowTitle("Experiment Task Summary Test")
    dialog.setMinimumSize(800, 600)

    layout = QVBoxLayout()
    widget = ExperimentTaskSummaryWidget()
    PATH = "/home/patrick/github/fibsem/fibsem/applications/autolamella/log/AutoLamella-2025-11-07-12-32/experiment.yaml"
    exp = Experiment.load(PATH)
    widget.set_experiment(exp)
    layout.addWidget(widget)
    dialog.setLayout(layout)

    dialog.show()
    sys.exit(app.exec_())