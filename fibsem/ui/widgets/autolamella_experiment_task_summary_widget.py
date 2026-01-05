"""Widget for displaying experiment task summary images across all lamellae."""

import logging
import os
from typing import TYPE_CHECKING, List, Optional
import traceback

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.structures import Experiment, Lamella
from fibsem.applications.autolamella.tools.reporting import (
    plot_experiment_task_summary,
    plot_lamella_task_workflow_summary,
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

SUMMARY_MODE_TASK = "task"
SUMMARY_MODE_LAMELLA = "lamella"
SUMMARY_MODE_LABELS = {
    SUMMARY_MODE_TASK: "Task Summary",
    SUMMARY_MODE_LAMELLA: "Lamella Summary",
}


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
        self.current_lamella: Optional['Lamella'] = None
        self.current_figure: Optional[Figure] = None
        self.summary_mode: str = SUMMARY_MODE_TASK
        self._title_artist = None
        self._ylabel_artists: List = []

        self.initUI()

    def initUI(self):
        """Initialize the widget UI components."""

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

        self.summary_mode_label = QLabel("Summary:")
        self.summary_mode_label.setStyleSheet("color: #bbbbbb; font-size: 10px;")

        self.summary_mode_selector = QComboBox(self)
        self.summary_mode_selector.setStyleSheet(COMBOBOX_STYLESHEET)
        self.summary_mode_selector.addItem(SUMMARY_MODE_LABELS[SUMMARY_MODE_TASK], SUMMARY_MODE_TASK)
        self.summary_mode_selector.addItem(SUMMARY_MODE_LABELS[SUMMARY_MODE_LAMELLA], SUMMARY_MODE_LAMELLA)
        self.summary_mode_selector.currentIndexChanged.connect(self._on_summary_mode_changed)

        self.task_label = QLabel("Task:")
        self.task_label.setStyleSheet("color: #bbbbbb; font-size: 10px;")

        self.task_selector = QComboBox(self)
        self.task_selector.setStyleSheet(COMBOBOX_STYLESHEET)
        self.task_selector.currentIndexChanged.connect(self._on_task_changed)

        self.lamella_label = QLabel("Lamella:")
        self.lamella_label.setStyleSheet("color: #bbbbbb; font-size: 10px;")

        self.lamella_selector = QComboBox(self)
        self.lamella_selector.setStyleSheet(COMBOBOX_STYLESHEET)
        self.lamella_selector.currentIndexChanged.connect(self._on_lamella_changed)

        # Button layout
        self.refresh_button = QPushButton("Refresh", self)
        self.refresh_button.setStyleSheet(BUTTON_STYLESHEET)
        self.refresh_button.clicked.connect(self.update_summary)

        self.export_button = QPushButton("Export Figure", self)
        self.export_button.setStyleSheet(BUTTON_STYLESHEET)
        self.export_button.clicked.connect(self._on_export_clicked)

        self.info_label = QLabel("No experiment loaded")
        self.info_label.setStyleSheet("color: #bbbbbb; font-size: 10px;")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # Right side selector panels
        selector_panel = QWidget()
        selector_layout = QVBoxLayout(selector_panel)
        selector_layout.setContentsMargins(10, 0, 0, 0)
        selector_layout.setSpacing(12)
        selector_layout.addWidget(self.summary_mode_label)
        selector_layout.addWidget(self.summary_mode_selector)
        selector_layout.addWidget(self.task_label)
        selector_layout.addWidget(self.task_selector)
        selector_layout.addWidget(self.lamella_label)
        selector_layout.addWidget(self.lamella_selector)

        # Display configuration
        self.config_group = QWidget()
        config_layout = QFormLayout(self.config_group)
        config_layout.setContentsMargins(0, 10, 0, 0)
        config_layout.setSpacing(6)

        self.fontsize_spinbox = QSpinBox()
        self.fontsize_spinbox.setRange(6, 48)
        self.fontsize_spinbox.setValue(10)
        self.fontsize_spinbox.setStyleSheet(COMBOBOX_STYLESHEET)
        self.fontsize_spinbox.editingFinished.connect(self._on_display_config_changed)

        self.image_size_spinbox = QSpinBox()
        self.image_size_spinbox.setRange(128, 1024)
        self.image_size_spinbox.setSingleStep(64)
        self.image_size_spinbox.setValue(512)
        self.image_size_spinbox.setStyleSheet(COMBOBOX_STYLESHEET)
        self.image_size_spinbox.editingFinished.connect(self._on_display_config_changed)

        self.show_title_checkbox = QCheckBox()
        self.show_title_checkbox.setChecked(True)
        self.show_title_checkbox.setStyleSheet("color: white; font-size: 10px;")
        self.show_title_checkbox.stateChanged.connect(self._on_display_config_changed)

        self.show_scalebar_checkbox = QCheckBox()
        self.show_scalebar_checkbox.setChecked(True)
        self.show_scalebar_checkbox.setStyleSheet("color: white; font-size: 10px;")
        self.show_scalebar_checkbox.stateChanged.connect(self._on_display_config_changed)

        config_layout.addRow("Font Size", self.fontsize_spinbox)
        config_layout.addRow("Image Size", self.image_size_spinbox)
        config_layout.addRow("Show Title", self.show_title_checkbox)
        config_layout.addRow("Show Scalebar", self.show_scalebar_checkbox)
        selector_layout.addWidget(self.config_group)
        selector_layout.addStretch()

        # content layout
        content_layout = QHBoxLayout()
        content_layout.addWidget(self.scroll_area, stretch=1)
        content_layout.addWidget(selector_panel, stretch=0)

        # Bottom button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.refresh_button)
        button_layout.addWidget(self.export_button)
        button_layout.addStretch()

        layout = QVBoxLayout()
        layout.addLayout(content_layout)
        layout.addLayout(button_layout)
        layout.addWidget(self.info_label)

        self.setLayout(layout)
        self._update_selector_visibility()

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

    def _update_selector_visibility(self):
        """Show/hide selectors based on the active summary mode."""
        is_task_mode = self.summary_mode == SUMMARY_MODE_TASK
        self.task_label.setVisible(is_task_mode)
        self.task_selector.setVisible(is_task_mode)
        self.lamella_label.setVisible(not is_task_mode)
        self.lamella_selector.setVisible(not is_task_mode)

    def _cache_text_artists(self, fig: Figure):
        """Store references to title and ylabel artists for export adjustments."""
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

        # Get scroll area width (fallback to widget width if viewport not ready yet)
        scroll_width = self.scroll_area.viewport().width()
        if scroll_width <= 0:
            scroll_width = self.scroll_area.width()
        if scroll_width <= 0:
            scroll_width = int(fig_width * dpi)

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
        """Set the experiment and populate selectors for both modes."""
        self.experiment = experiment
        self.current_task = None
        self.current_lamella = None

        if experiment is None:
            self.info_label.setText("No experiment loaded")
        else:
            self.info_label.setText(f"Experiment: {experiment.name}")

        self._populate_task_selector()
        self._populate_lamella_selector()
        self._update_selector_visibility()
        self.update_summary()

    def _populate_task_selector(self):
        """Populate the task selector with completed tasks across lamellae."""
        self.task_selector.blockSignals(True)
        self.task_selector.clear()
        self.current_task = None

        preferred_task = "Rough Milling"
        preferred_index = -1

        if self.experiment is not None:
            all_tasks = set()
            for lamella in self.experiment.positions:
                all_tasks.update([task.name for task in lamella.task_history])

            sorted_tasks = sorted(all_tasks)
            if sorted_tasks:
                default_index = 0
                for idx, task_name in enumerate(sorted_tasks):
                    count = sum(
                        1 for lamella in self.experiment.positions
                        if lamella.has_completed_task(task_name)
                    )
                    display_name = f"{task_name} ({count} lamellae)"
                    self.task_selector.addItem(display_name, userData=task_name)
                    if preferred_index == -1 and task_name == preferred_task and count > 0:
                        preferred_index = idx

                if preferred_index != -1:
                    self.task_selector.setCurrentIndex(preferred_index)
                    self.current_task = self.task_selector.itemData(preferred_index)
                else:
                    self.current_task = sorted_tasks[0]
            else:
                self.info_label.setText(
                    f"Experiment: {self.experiment.name} | No completed tasks found"
                )

        self.task_selector.blockSignals(False)

    def _populate_lamella_selector(self):
        """Populate the lamella selector with lamellae that have tasks."""
        self.lamella_selector.blockSignals(True)
        self.lamella_selector.clear()
        self.current_lamella = None

        if self.experiment is not None:
            lamellae_with_tasks = [
                lamella for lamella in self.experiment.positions
                if len(lamella.task_history) > 0
            ]

            if lamellae_with_tasks:
                for lamella in lamellae_with_tasks:
                    display_name = f"{lamella.name} ({len(lamella.task_history)} tasks)"
                    self.lamella_selector.addItem(display_name, userData=lamella)
                self.current_lamella = lamellae_with_tasks[0]
            else:
                self.info_label.setText(
                    f"Experiment: {self.experiment.name} | No lamella workflows found"
                )

        self.lamella_selector.blockSignals(False)

    def _on_summary_mode_changed(self, index: int):
        """Handle summary mode selection change."""
        if index < 0:
            return
        mode = self.summary_mode_selector.itemData(index)
        if mode and mode != self.summary_mode:
            self.summary_mode = mode
            self._update_selector_visibility()
            self.update_summary()

    def _on_task_changed(self, index: int):
        """Handle task selection change.

        Args:
            index: The index of the selected task in the combo box
        """
        if index >= 0:
            self.current_task = self.task_selector.itemData(index)
            self.update_summary()

    def _on_lamella_changed(self, index: int):
        """Handle lamella selection change."""
        if index >= 0:
            self.current_lamella = self.lamella_selector.itemData(index)
            self.update_summary()

    def _on_display_config_changed(self):
        """Refresh the summary when display parameters change."""
        self.update_summary()

    def update_summary(self):
        """Update the summary view based on the selected mode."""
        if self.summary_mode == SUMMARY_MODE_TASK:
            self._update_task_summary()
        else:
            self._update_lamella_summary()

    def _update_task_summary(self):
        """Render the experiment-level task summary."""
        if self.current_task is None or self.experiment is None:
            self._create_empty_canvas()
            self.info_label.setText("Select a task to view the experiment summary.")
            return

        try:
            fig = plot_experiment_task_summary(
                exp=self.experiment,
                task_name=self.current_task,
                show_title=self.show_title_checkbox.isChecked(),
                show_scalebar=self.show_scalebar_checkbox.isChecked(),
                figsize=(10, 5),
                target_size=self.image_size_spinbox.value(),
                fontsize=self.fontsize_spinbox.value(),
                mode="dark",
                show=False
            )

            if fig is None:
                self._create_empty_canvas()
                self.info_label.setText(
                    f"Task: {self.current_task} | No valid images found"
                )
                return

            self.current_figure = fig
            self._cache_text_artists(fig)
            self._replace_canvas_with_figure(fig)

            count = sum(
                1 for lamella in self.experiment.positions
                if lamella.has_completed_task(self.current_task)
            )

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
            traceback.print_exc()
            self._create_empty_canvas()
            self.info_label.setText(f"Error: {str(e)}")

    def _update_lamella_summary(self):
        """Render the lamella-level workflow summary."""
        if self.current_lamella is None:
            self._create_empty_canvas()
            self.info_label.setText("Select a lamella to view the workflow summary.")
            return

        try:
            fig = plot_lamella_task_workflow_summary(
                self.current_lamella,
                show_title=self.show_title_checkbox.isChecked(),
                show_scalebar=self.show_scalebar_checkbox.isChecked(),
                figsize=(10, 5),
                target_size=self.image_size_spinbox.value(),
                fontsize=self.fontsize_spinbox.value(),
                mode="dark",
                show=False
            )

            if fig is None:
                self._create_empty_canvas()
                self.info_label.setText(
                    f"Lamella: {self.current_lamella.name} | No valid workflow images found"
                )
                return

            self.current_figure = fig
            self._cache_text_artists(fig)
            self._replace_canvas_with_figure(fig)

            self.info_label.setText(
                f"Lamella: {self.current_lamella.name} | "
                f"{len(self.current_lamella.task_history)} completed tasks"
            )

        except ImportError as e:
            logging.error(f"Could not import reporting module: {e}")
            self._create_empty_canvas()
            self.info_label.setText("Error: Reporting module not available")
        except Exception as e:
            logging.error(f"Error updating lamella task summary: {e}")
            traceback.print_exc()
            self._create_empty_canvas()
            self.info_label.setText(f"Error: {str(e)}")

    def _on_export_clicked(self):
        """Handle export button click to save the current figure."""
        if self.current_figure is None:
            return

        # Determine starting directory from experiment path if available
        start_dir = ""
        if self.experiment is not None:
            start_dir = str(self.experiment.path)

        # Default filename
        if self.summary_mode == SUMMARY_MODE_LAMELLA and self.current_lamella is not None:
            default_name = f"lamella_workflow_summary_{self.current_lamella.name}.png"
        elif self.current_task is not None:
            default_name = f"experiment_task_summary_{self.current_task}.png"
        else:
            default_name = "experiment_task_summary.png"

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
            original_title_color = None
            ylabel_original_colors: List[str] = []
            if self._title_artist is not None:
                original_title_color = self._title_artist.get_color()
                self._title_artist.set_color("black")
            for label in self._ylabel_artists:
                ylabel_original_colors.append(label.get_color())
                label.set_color("black")

            self.current_figure.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
            logging.info(f"Exported experiment task summary to: {file_path}")

        except Exception as e:
            logging.error(f"Error exporting figure: {e}")
            traceback.print_exc()
        finally:
            if self._title_artist is not None and original_title_color is not None:
                self._title_artist.set_color(original_title_color)
            for label, prev_color in zip(self._ylabel_artists, ylabel_original_colors):
                label.set_color(prev_color)


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
    dialog.setWindowTitle(f"Experiment Summary - {experiment.name}")
    dialog.setMinimumSize(800, 600)

    # Create layout
    layout = QVBoxLayout()

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

    PATH = "/home/patrick/github/fibsem/fibsem/applications/autolamella/log/AutoLamella-2025-11-07-12-32/experiment.yaml"
    PATH = "/home/patrick/github/fibsem/fibsem/applications/autolamella/log/AutoLamella-2025-12-12-12-49/experiment.yaml"
    exp = Experiment.load(PATH)
    dialog = create_experiment_task_summary_widget(exp)
    dialog.show()
    sys.exit(app.exec_())
