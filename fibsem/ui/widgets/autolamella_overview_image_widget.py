"""Widget for generating final overview images with customizable markers and text."""

import logging
import os
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QLineEdit,
    QWidget,
)
from PyQt5.QtGui import QColor

from fibsem.applications.autolamella.structures import Experiment
from fibsem.imaging.tiled import plot_minimap
from fibsem.structures import FibsemImage
import glob
if TYPE_CHECKING:
    from fibsem.applications.autolamella.ui import AutoLamellaUI


# Stylesheet constants
GROUPBOX_STYLESHEET = """
    QGroupBox {
        border: 1px solid #555;
        border-radius: 5px;
        margin-top: 10px;
        padding-top: 10px;
        color: white;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px 0 5px;
    }
"""

LINEEDIT_STYLESHEET = """
    QLineEdit {
        background-color: #3a3a3a;
        color: white;
        border: 1px solid #555;
        padding: 4px 8px;
        font-size: 10px;
    }
    QLineEdit:hover {
        background-color: #4a4a4a;
    }
"""

SPINBOX_STYLESHEET = """
    QSpinBox {
        background-color: #3a3a3a;
        color: white;
        border: 1px solid #555;
        padding: 4px 8px;
        font-size: 10px;
    }
    QSpinBox:hover {
        background-color: #4a4a4a;
    }
"""

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

CHECKBOX_STYLESHEET = """
    QCheckBox {
        color: white;
        font-size: 10px;
    }
    QCheckBox::indicator {
        width: 15px;
        height: 15px;
        border: 1px solid #555;
        background-color: #3a3a3a;
    }
    QCheckBox::indicator:checked {
        background-color: #4a90e2;
    }
"""


ZOOM_SCALE_FACTOR = 1.2


class OverviewImageWidget(QWidget):
    """Widget for generating final overview images with customizable markers and text.

    This widget allows users to:
    - Select an overview image file
    - Specify output filename
    - Customize marker colors, text size, and names
    - Preview the generated image
    - Export the final image
    """

    def __init__(self, parent: Optional['AutoLamellaUI'] = None):
        """Initialize the overview image widget.

        Args:
            parent: Parent AutoLamellaUI widget (optional)
        """
        super().__init__(parent)
        self.parent_widget = parent
        self.experiment: Optional['Experiment'] = None
        self.overview_image: Optional['FibsemImage'] = None
        self.current_figure: Optional[Figure] = None
        self.marker_color = QColor("cyan")
        self.stage_positions = []
        self._scroll_cid: Optional[int] = None
        self._pan_press_cid: Optional[int] = None
        self._pan_release_cid: Optional[int] = None
        self._pan_motion_cid: Optional[int] = None
        self._pan_active = False
        self._pan_axes = None
        self._pan_start = None
        self._pan_axes_limits = None
        self._initial_axes_limits = []
        self._title_artist = None

        self.initUI()

    def initUI(self):
        """Initialize the widget UI components."""

        # Display Options Panel
        display_group = QGroupBox("Display Options")
        display_group.setStyleSheet(GROUPBOX_STYLESHEET)
        display_layout = QFormLayout()

        # Title Text
        self.title_textbox = QLineEdit()
        self.title_textbox.setStyleSheet(LINEEDIT_STYLESHEET)
        self.title_textbox.setText("Overview Image")

        # Marker color
        self.color_label = QLabel("●")
        self.color_label.setStyleSheet(f"color: {self.marker_color.name()}; font-size: 20px;")
        self.color_button = QPushButton("Choose Color")
        self.color_button.setStyleSheet(BUTTON_STYLESHEET)
        self.color_button.clicked.connect(self._on_color_button_clicked)
        color_layout = QHBoxLayout()
        color_layout.addWidget(self.color_label)
        color_layout.addWidget(self.color_button)
        color_layout.addStretch()

        # Text size
        self.text_size_spinbox = QSpinBox()
        self.text_size_spinbox.setStyleSheet(SPINBOX_STYLESHEET)
        self.text_size_spinbox.setRange(6, 48)
        self.text_size_spinbox.setValue(10)
        self.text_size_spinbox.setKeyboardTracking(False)

        # Marker size
        self.markersize_spinbox = QSpinBox()
        self.markersize_spinbox.setStyleSheet(SPINBOX_STYLESHEET)
        self.markersize_spinbox.setRange(5, 100)
        self.markersize_spinbox.setValue(20)
        self.markersize_spinbox.setKeyboardTracking(False)

        # Show names checkbox
        self.show_names_checkbox = QCheckBox("")
        self.show_names_checkbox.setStyleSheet(CHECKBOX_STYLESHEET)
        self.show_names_checkbox.setChecked(True)

        # Show scalebar checkbox
        self.show_scalebar_checkbox = QCheckBox("")
        self.show_scalebar_checkbox.setStyleSheet(CHECKBOX_STYLESHEET)
        self.show_scalebar_checkbox.setChecked(True)
        
        # display layout
        display_layout.addRow("Title", self.title_textbox)
        display_layout.addRow("Marker Color", color_layout)
        display_layout.addRow("Text Size", self.text_size_spinbox)
        display_layout.addRow("Marker Size", self.markersize_spinbox)
        display_layout.addRow("Show Names", self.show_names_checkbox)
        display_layout.addRow("Show Scalebar", self.show_scalebar_checkbox)
        display_group.setLayout(display_layout)

        # Connect signals to update preview on change
        self.text_size_spinbox.valueChanged.connect(self._on_preview_clicked)
        self.markersize_spinbox.valueChanged.connect(self._on_preview_clicked)
        self.show_names_checkbox.stateChanged.connect(self._on_preview_clicked)
        self.show_scalebar_checkbox.stateChanged.connect(self._on_preview_clicked)
        self.title_textbox.editingFinished.connect(self._on_preview_clicked)


        # Preview canvas
        self.canvas = None
        self.figure = None

        # Create a placeholder widget for the canvas
        self.canvas_container = QWidget()
        self.canvas_layout = QVBoxLayout(self.canvas_container)
        self.canvas_layout.setContentsMargins(0, 0, 0, 0)

        # Create initial empty figure
        self._create_empty_canvas()

        # Action buttons
        self.load_image_button = QPushButton("Load Image", self)
        self.load_image_button.setStyleSheet(BUTTON_STYLESHEET)
        self.load_image_button.clicked.connect(self._on_browse_image_clicked)
        self.load_image_button.setAutoDefault(False)
        self.load_image_button.setDefault(False)

        self.reset_view_button = QPushButton("Reset View", self)
        self.reset_view_button.setStyleSheet(BUTTON_STYLESHEET)
        self.reset_view_button.clicked.connect(self._on_reset_view_clicked)

        self.save_button = QPushButton("Save Image", self)
        self.save_button.setStyleSheet(BUTTON_STYLESHEET)
        self.save_button.clicked.connect(self._on_save_clicked)

        hlayout = QHBoxLayout()
        hlayout.addWidget(self.canvas_container)
        hlayout.addWidget(display_group)

        # button layouts
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_image_button)
        button_layout.addWidget(self.reset_view_button)
        button_layout.addWidget(self.save_button)
        button_layout.addStretch()

        # Info label
        self.info_label = QLabel("No experiment loaded")
        self.info_label.setStyleSheet("color: #bbbbbb; font-size: 10px;")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # main layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(hlayout)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.info_label)

        self.setLayout(main_layout)

    def _create_empty_canvas(self):
        """Create an empty canvas with placeholder text."""
        # Create empty figure
        self.figure = Figure(figsize=(10, 6), dpi=80)
        self.figure.patch.set_facecolor("#262930")

        # Create canvas from figure
        if self.canvas is not None:
            self._disconnect_canvas_events()
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
            "No preview available\nSelect image and generate preview",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=12,
            color="#bbbbbb",
        )
        ax.set_title("Overview Image Preview", color="white")
        ax.axis("off")
        self.figure.tight_layout()
        self.canvas.draw()
        self._enable_canvas_events()
        self._capture_initial_view_limits()

    def _replace_canvas_with_figure(self, new_figure):
        """Replace the current canvas with a new figure.

        Args:
            new_figure: Matplotlib Figure object to display
        """
        # Remove old canvas
        if self.canvas is not None:
            self._disconnect_canvas_events()
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
        self.canvas_layout.addWidget(self.canvas)

        # Draw the canvas
        self.canvas.draw()
        self._enable_canvas_events()
        self._capture_initial_view_limits()

    def _disconnect_canvas_events(self):
        """Disconnect zoom and pan handlers from the current canvas."""
        if self.canvas is None:
            return

        for attr in ("_scroll_cid", "_pan_press_cid", "_pan_release_cid", "_pan_motion_cid"):
            cid = getattr(self, attr)
            if cid is None:
                continue
            try:
                self.canvas.mpl_disconnect(cid)
            except Exception:
                pass
            setattr(self, attr, None)

        self._pan_active = False
        self._pan_axes = None
        self._pan_start = None
        self._pan_axes_limits = None

    def _enable_canvas_events(self):
        """Attach scroll, press, release, and motion handlers to the canvas."""
        if self.canvas is None:
            return

        self._scroll_cid = self.canvas.mpl_connect("scroll_event", self._on_canvas_scroll)
        self._pan_press_cid = self.canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self._pan_release_cid = self.canvas.mpl_connect("button_release_event", self._on_canvas_release)
        self._pan_motion_cid = self.canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)

    def _on_canvas_scroll(self, event):
        """Handle scroll-wheel events to zoom in/out."""
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return

        ax = event.inaxes
        if event.button == "up":
            scale_factor = 1 / ZOOM_SCALE_FACTOR
        elif event.button == "down":
            scale_factor = ZOOM_SCALE_FACTOR
        else:
            return

        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        x_range = cur_xlim[1] - cur_xlim[0]
        y_range = cur_ylim[1] - cur_ylim[0]

        if x_range == 0 or y_range == 0:
            return

        new_width = x_range * scale_factor
        new_height = y_range * scale_factor

        relx = (event.xdata - cur_xlim[0]) / x_range
        rely = (event.ydata - cur_ylim[0]) / y_range

        new_xlim = (
            event.xdata - relx * new_width,
            event.xdata + (1 - relx) * new_width,
        )
        new_ylim = (
            event.ydata - rely * new_height,
            event.ydata + (1 - rely) * new_height,
        )

        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        self.canvas.draw_idle()

    def _on_canvas_press(self, event):
        """Start a pan gesture when the left mouse button is pressed."""
        if event.button != 1 or event.inaxes is None or event.xdata is None or event.ydata is None:
            return

        self._pan_active = True
        self._pan_axes = event.inaxes
        self._pan_start = (event.xdata, event.ydata)
        self._pan_axes_limits = (
            event.inaxes.get_xlim(),
            event.inaxes.get_ylim(),
        )

    def _on_canvas_motion(self, event):
        """Update axes while the mouse moves to create panning."""
        if not self._pan_active or self._pan_axes is None:
            return
        if event.inaxes != self._pan_axes or event.xdata is None or event.ydata is None:
            return
        if self._pan_axes_limits is None or self._pan_start is None:
            return

        start_xlim, start_ylim = self._pan_axes_limits
        dx = event.xdata - self._pan_start[0]
        dy = event.ydata - self._pan_start[1]

        new_xlim = (start_xlim[0] - dx, start_xlim[1] - dx)
        new_ylim = (start_ylim[0] - dy, start_ylim[1] - dy)

        self._pan_axes.set_xlim(new_xlim)
        self._pan_axes.set_ylim(new_ylim)
        self.canvas.draw_idle()

    def _on_canvas_release(self, event):
        """End the pan gesture."""
        if event.button != 1:
            return

        self._pan_active = False
        self._pan_axes = None
        self._pan_start = None
        self._pan_axes_limits = None

    def _capture_initial_view_limits(self):
        """Store axes limits to allow resetting the view."""
        self._initial_axes_limits = []
        if self.figure is None:
            return

        for ax in self.figure.axes:
            self._initial_axes_limits.append((ax, ax.get_xlim(), ax.get_ylim()))

    def _on_reset_view_clicked(self):
        """Reset all axes to their initial limits."""
        if not self.figure or not self.canvas or not self._initial_axes_limits:
            return

        for ax, xlim, ylim in self._initial_axes_limits:
            if ax in self.figure.axes:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

        self.canvas.draw_idle()

    def set_experiment(self, experiment: 'Experiment'):
        """Set the experiment to use for generating overview images.

        Args:
            experiment: The Experiment object containing lamella positions
        """
        self.experiment = experiment


        if experiment is not None:
            self.info_label.setText(
                f"Experiment: {experiment.name} | "
                f"{len(experiment.positions)} lamellae"
            )
            self.title_textbox.setText(f"Experiment: {self.experiment.name}")
            self.title_textbox.setCursorPosition(0)

            # Collect positions (always use MILLING state)
            self.stage_positions.clear()
            for p in self.experiment.positions:
                pstate = p.poses.get("MILLING", p.state.microscope_state)
                if pstate is None or pstate.stage_position is None:
                    continue
                pos = pstate.stage_position
                pos.name = p.name
                self.stage_positions.append(pos)

            filenames = glob.glob(os.path.join(experiment.path, "*overview*.tif"))
            filenames = [f for f in filenames if "autogamma" not in f]
            if filenames:
                self._load_overview_image(filenames[-1])
        else:
            self.info_label.setText("No experiment loaded")

    def _on_browse_image_clicked(self):
        """Handle browse button click for selecting and loading overview image."""
        start_dir = ""
        if self.experiment is not None:
            start_dir = str(self.experiment.path)

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Overview Image",
            start_dir,
            "TIFF Files (*.tif *.tiff);;All Files (*.*)"
        )

        if file_path:
            self._load_overview_image(file_path)

    def _load_overview_image(self, file_path: str):
        """Load the overview image from the specified file path."""
        try:
            # Load the overview image
            self.overview_image = FibsemImage.load(file_path)
            self.info_label.setText(f"Loaded: {os.path.basename(file_path)}")

            # Automatically generate preview when image is selected
            self._on_preview_clicked()
        except Exception as e:
            logging.error(f"Error loading image: {e}")
            self.info_label.setText(f"Error loading image: {str(e)}")
            self.overview_image = None

    def _on_color_button_clicked(self):
        """Handle color button click to select marker color."""
        color = QColorDialog.getColor(
            self.marker_color,
            self,
            "Select Marker Color"
        )

        if color.isValid():
            self.marker_color = color
            self.color_label.setStyleSheet(f"color: {color.name()}; font-size: 20px;")
            self._on_preview_clicked()

    def _on_preview_clicked(self):
        """Generate and display preview of the overview image."""
        if self.experiment is None:
            self.info_label.setText("Error: No experiment loaded")
            return

        if self.overview_image is None:
            self.info_label.setText("Error: No overview image loaded")
            return

        try:

            # Get settings
            show_names = self.show_names_checkbox.isChecked()
            show_scalebar = self.show_scalebar_checkbox.isChecked()
            color = self.marker_color.name()
            fontsize = self.text_size_spinbox.value()
            markersize = self.markersize_spinbox.value()

            if not self.stage_positions:
                self.info_label.setText("Warning: No positions found for MILLING state")
                self._create_empty_canvas()
                return

            # Generate figure
            fig = plot_minimap(
                self.overview_image,
                self.stage_positions,
                current_position=None,
                grid_positions=None,
                color=color,
                fontsize=fontsize,
                markersize=markersize,
                show_scalebar=show_scalebar,
                show_names=show_names,
                figsize=(15, 15)
            )

            # Add title
            title_text = self.title_textbox.text().strip()
            self._title_artist = fig.suptitle(title_text, fontsize=14, color="white")
            fig.tight_layout(rect=(0, 0, 1, 0.98))

            # Store and display figure
            self.current_figure = fig
            self._replace_canvas_with_figure(fig)

            self.info_label.setText(
                f"Preview generated | {len(self.stage_positions)} positions shown"
            )

        except Exception as e:
            logging.error(f"Error generating preview: {e}")
            import traceback
            traceback.print_exc()
            self.info_label.setText(f"Error: {str(e)}")
            self._create_empty_canvas()

    def _on_save_clicked(self):
        """Save the overview image by opening a save file dialog."""
        if self.current_figure is None:
            self.info_label.setText("Error: No preview to save. Generate preview first.")
            return

        if self.overview_image is None or self.experiment is None:
            self.info_label.setText("Error: Missing required data")
            return

        # Open save file dialog
        default_name = f"{self.experiment.name}_overview_final.png"
        start_filename = os.path.join(str(self.experiment.path), default_name)

        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Overview Image",
            start_filename,
            "PNG Files (*.png);;"
        )

        if not output_path:
            # User cancelled
            return

        original_title_color = None
        try:
            if self._title_artist is not None:
                original_title_color = self._title_artist.get_color()
                self._title_artist.set_color("black")

            # Save the current figure directly
            self.current_figure.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

            logging.info(f"Saved overview image to: {output_path}")
            self.info_label.setText(f"Saved to: {output_path}")

        except Exception as e:
            logging.error(f"Error saving image: {e}")
            import traceback
            traceback.print_exc()
            self.info_label.setText(f"Error saving: {str(e)}")
        finally:
            if self._title_artist is not None and original_title_color is not None:
                self._title_artist.set_color(original_title_color)


def create_overview_image_widget(experiment: 'Experiment',
                                  parent: Optional['AutoLamellaUI'] = None) -> QDialog:
    """Create and initialize an OverviewImageWidget wrapped in a dialog.

    Args:
        experiment: The Experiment object to use
        parent: Optional parent AutoLamellaUI widget

    Returns:
        QDialog: Dialog containing the initialized widget with the experiment loaded
    """
    # Create dialog
    dialog = QDialog(parent)
    dialog.setWindowTitle(f"Generate Overview Image - {experiment.name}")
    dialog.setMinimumSize(800, 800)

    # Create layout
    layout = QVBoxLayout()

    # Create and add widget
    widget = OverviewImageWidget(parent=parent)
    widget.set_experiment(experiment)
    layout.addWidget(widget)

    dialog.setLayout(layout)

    return dialog

# TODO: add more options for customization (scalebar size, font type, etc.)
# - show only completed lamellae
# - show defected lamellae in different color


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    layout = QVBoxLayout()
    widget = OverviewImageWidget()
    PATH = "/home/patrick/github/fibsem/fibsem/applications/autolamella/log/AutoLamella-2025-11-07-12-32/experiment.yaml"
    exp = Experiment.load(PATH)
    dialog = create_overview_image_widget(experiment=exp)

    dialog.show()
    sys.exit(app.exec_())
