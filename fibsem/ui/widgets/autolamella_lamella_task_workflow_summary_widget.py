"""Widget for displaying task workflow summary images for completed lamellae."""

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
    plot_lamella_task_workflow_summary,
)
from fibsem.applications.autolamella.structures import Experiment, Lamella

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


class LamellaTaskWorkflowSummaryWidget(QWidget):
    """Widget for displaying task workflow summary images for completed lamellae.

    This widget displays a grid of images showing SEM and FIB views at different resolutions
    for each completed task in a lamella workflow. Users can navigate between different
    lamellae using a dropdown selector.
    """

    def __init__(self, parent: Optional['AutoLamellaUI'] = None):
        """Initialize the task workflow summary widget.

        Args:
            parent: Parent AutoLamellaUI widget (optional)
        """
        super().__init__(parent)
        self.parent_widget = parent
        self.experiment: Optional['Experiment'] = None
        self.current_lamella: Optional['Lamella'] = None
        self.current_figure: Optional[Figure] = None
        self._title_artist = None
        self._ylabel_artists: List = []

        self.initUI()

    def initUI(self):
        """Initialize the widget UI components."""

        layout = QVBoxLayout()

        # Title and controls layout
        controls_layout = QHBoxLayout()

        # Title label
        title_label = QLabel("Lamella Workflow Summary")
        title_label.setStyleSheet("color: white; font-size: 12px; font-weight: bold;")
        controls_layout.addWidget(title_label)

        controls_layout.addStretch()

        # Lamella selector
        self.lamella_label = QLabel("Lamella:")
        self.lamella_label.setStyleSheet("color: #bbbbbb; font-size: 10px;")
        controls_layout.addWidget(self.lamella_label)

        self.lamella_selector = QComboBox(self)
        self.lamella_selector.setStyleSheet(COMBOBOX_STYLESHEET)
        self.lamella_selector.currentIndexChanged.connect(self._on_lamella_changed)
        controls_layout.addWidget(self.lamella_selector)

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

        self.setLayout(layout)

    def _create_empty_canvas(self):
        """Create an empty canvas with placeholder text."""
        # Create empty figure
        self.figure = Figure(figsize=(10, 6), dpi=80)
        self.figure.patch.set_facecolor("#262930")
        self._title_artist = None
        self._ylabel_artists = []

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
            "No task workflow data available",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=12,
            color="#bbbbbb",
        )
        ax.set_title("Lamella Workflow Summary", color="white")
        ax.axis("off")
        self.figure.tight_layout()
        self.canvas.draw()

    def _cache_text_artists(self, fig: Figure):
        """Store references to title and ylabel artists for export color adjustments."""
        self._title_artist = getattr(fig, "_suptitle", None)
        self._ylabel_artists = []
        for ax in fig.axes:
            label = ax.yaxis.get_label()
            if label is not None:
                self._ylabel_artists.append(label)

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
        """Set the experiment and populate the lamella selector.

        Args:
            experiment: The Experiment object containing lamellae to display
        """
        self.experiment = experiment

        # Clear and populate lamella selector
        self.lamella_selector.blockSignals(True)  # Prevent triggering update during population
        self.lamella_selector.clear()

        if experiment is not None and hasattr(experiment, 'positions'):
            # Filter to only show lamellae with completed tasks
            lamellae_with_tasks = [
                lamella for lamella in experiment.positions
                if hasattr(lamella, 'task_history') and len(lamella.task_history) > 0
            ]

            if lamellae_with_tasks:
                for lamella in lamellae_with_tasks:
                    display_name = f"{lamella.name} ({len(lamella.task_history)} tasks)"
                    self.lamella_selector.addItem(display_name, userData=lamella)

                # Set the first lamella as current
                self.current_lamella = lamellae_with_tasks[0]
                self.info_label.setText(
                    f"Experiment: {experiment.name} | "
                    f"{len(lamellae_with_tasks)} lamellae with completed tasks"
                )
            else:
                self.info_label.setText(f"Experiment: {experiment.name} | No completed tasks found")
                self.current_lamella = None
        else:
            self.info_label.setText("No experiment loaded")
            self.current_lamella = None

        self.lamella_selector.blockSignals(False)

        # Update the display
        self.update_summary()

    def _on_lamella_changed(self, index: int):
        """Handle lamella selection change.

        Args:
            index: The index of the selected lamella in the combo box
        """
        if index >= 0:
            self.current_lamella = self.lamella_selector.itemData(index)
            self.update_summary()

    def update_summary(self):
        """Update the task workflow summary display for the current lamella."""

        if self.current_lamella is None:
            self._create_empty_canvas()
            return

        try:
            # Generate the figure
            fig = plot_lamella_task_workflow_summary(
                self.current_lamella,
                show_title=True,
                figsize=(10, 5),
                target_size=512,
                fontsize=12,
                mode="dark",
                show=False
            )

            if fig is None:
                # No valid images found for this lamella
                self._create_empty_canvas()
                self.info_label.setText(
                    f"Lamella: {self.current_lamella.name} | No valid workflow images found"
                )
                return

            # Store the current figure
            self.current_figure = fig
            self._cache_text_artists(fig)

            # Replace canvas with the new figure
            self._replace_canvas_with_figure(fig)

            # Update info label
            self.info_label.setText(
                f"Lamella: {self.current_lamella.name} | "
                f"{len(self.current_lamella.task_history)} completed tasks"
            )

        except ImportError as e:
            logging.error(f"Could not import reporting module: {e}")
            self._create_empty_canvas()
            self.info_label.setText("Error: Reporting module not available")
        except Exception as e:
            logging.error(f"Error updating task workflow summary: {e}")
            import traceback
            traceback.print_exc()
            self._create_empty_canvas()
            self.info_label.setText(f"Error: {str(e)}")

    def clear_summary(self):
        """Clear the current summary display."""
        self.current_lamella = None
        self.current_figure = None
        self._title_artist = None
        self._ylabel_artists = []
        self.lamella_selector.clear()
        self._create_empty_canvas()
        self.info_label.setText("No experiment loaded")

    def _on_export_clicked(self):
        """Handle export button click to save the current figure."""
        if self.current_figure is None:
            return
        
        if self.experiment is None or self.current_lamella is None:
            return

        start_dir = str(self.experiment.path)
        default_name = f"task_workflow_summary_{self.current_lamella.name}.png"
        default_filename = os.path.join(start_dir, default_name)

        # Open file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Lamella Workflow Summary",
            default_filename,
            "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*.*)"
        )

        if not file_path:
            # User cancelled
            return

        try:
            original_title_color = None
            ylabel_original_colors: List[str] = []
            if self._title_artist is not None:
                original_title_color = self._title_artist.get_color()
                self._title_artist.set_color("black")
            for label in self._ylabel_artists:
                ylabel_original_colors.append(label.get_color())
                label.set_color("black")

            # Save with high DPI
            self.current_figure.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
            logging.info(f"Exported lamella task summary to: {file_path}")

        except Exception as e:
            logging.error(f"Error exporting figure: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self._title_artist is not None and original_title_color is not None:
                self._title_artist.set_color(original_title_color)
            for label, prev_color in zip(self._ylabel_artists, ylabel_original_colors):
                label.set_color(prev_color)


def create_lamella_workflow_summary_widget(experiment: 'Experiment',
                                                parent: Optional['AutoLamellaUI'] = None) -> QDialog:
    """Create and initialize a LamellaTaskWorkflowSummaryWidget wrapped in a dialog.

    Args:
        experiment: The Experiment object to display
        parent: Optional parent AutoLamellaUI widget

    Returns:
        QDialog: Dialog containing the initialized widget with the experiment loaded
    """
    # Create dialog
    dialog = QDialog(parent)
    dialog.setWindowTitle(f"Lamella Workflow Summary - {experiment.name}")
    dialog.setMinimumSize(400, 300)

    # Create layout
    layout = QVBoxLayout()

    # Create and add widget
    widget = LamellaTaskWorkflowSummaryWidget(parent=parent)
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
    widget = LamellaTaskWorkflowSummaryWidget()
    PATH = "/home/patrick/github/fibsem/fibsem/applications/autolamella/log/AutoLamella-2025-11-07-12-32/experiment.yaml"
    exp = Experiment.load(PATH)
    widget.set_experiment(exp)
    layout.addWidget(widget)
    dialog.setLayout(layout)

    dialog.show()
    sys.exit(app.exec_())
