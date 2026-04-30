import logging
from typing import Optional, TYPE_CHECKING

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.fm.structures import FluorescenceImage
from fibsem.fm.plotting import plot_fluorescence_image

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available. Fluorescence plot widget will be disabled.")

if TYPE_CHECKING:
    pass


class FluorescencePlotWidget(QWidget):
    """Widget for displaying fluorescence images with channel selection and z-stack controls."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.image: Optional[FluorescenceImage] = None
        self.parent_widget = parent

        # Display state
        self.selected_channel_idx: Optional[int] = None  # None = all channels

        self.initUI()

    def initUI(self):
        """Initialize the fluorescence plot widget UI."""
        # self.setContentsMargins(0, 0, 0, 0)

        layout = QVBoxLayout()

        if not MATPLOTLIB_AVAILABLE:
            # Show error message if matplotlib is not available
            error_label = QLabel(
                "Matplotlib not available.\nInstall with: pip install matplotlib", self
            )
            error_label.setStyleSheet("color: red; font-size: 10px;")
            error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(error_label)
        else:
            # Create a container widget for the plot area (toolbar + canvas)
            # This will be replaced each time we update the plot
            self.plot_container = QWidget(self)
            self.plot_layout = QVBoxLayout(self.plot_container)
            # self.plot_layout.setContentsMargins(0, 0, 0, 0)

            # Initialize with empty plot
            self._init_plot_widgets()

            layout.addWidget(self.plot_container)

            # Button layout
            button_layout = QHBoxLayout()

            # Load Image button
            self.load_image_button = QToolButton(self)
            self.load_image_button.setText("Load Image")
            self.load_image_button.setToolTip("Load a fluorescence image file")
            self.load_image_button.setStyleSheet("""
                QToolButton {
                    background-color: #3a3a3a;
                    color: white;
                    border: 1px solid #555;
                    padding: 4px 8px;
                    font-size: 10px;
                }
                QToolButton:hover {
                    background-color: #4a4a4a;
                }
                QToolButton:pressed {
                    background-color: #2a2a2a;
                }
            """)
            self.load_image_button.clicked.connect(self._on_load_image_clicked)
            button_layout.addWidget(self.load_image_button)


            layout.addLayout(button_layout)



            # Channel selection will be created dynamically in update_plot
            self.channel_buttons_layout = QHBoxLayout()
            self.channel_buttons_layout.setContentsMargins(0, 0, 0, 0)
            layout.addLayout(self.channel_buttons_layout)

        # layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def _init_plot_widgets(self):
        """Initialize or recreate the plot widgets (canvas and toolbar)."""
        if not MATPLOTLIB_AVAILABLE:
            return

        # Create matplotlib figure and canvas with napari-style dark theme
        self.figure = Figure(figsize=(6, 6), dpi=80)
        self.figure.patch.set_facecolor("#262930")  # napari dark background

        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(300, 300)

        # Create the subplot with dark styling
        self.ax = self.figure.add_subplot(111)
        self._apply_dark_theme()

        # Create navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("""
            QToolBar {
                background-color: #3a3a3a;
                border: 1px solid #555;
                spacing: 3px;
            }
            QToolButton {
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #555;
                padding: 3px;
            }
            QToolButton:hover {
                background-color: #4a4a4a;
            }
            QToolButton:pressed {
                background-color: #2a2a2a;
            }
        """)

        # Clear the plot layout and add new widgets
        while self.plot_layout.count():
            item = self.plot_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.plot_layout.addWidget(self.toolbar)
        self.plot_layout.addWidget(self.canvas)

        # Initial empty plot
        self._plot_empty()

    def _apply_dark_theme(self):
        """Apply napari-style dark theme to the axes."""
        if not MATPLOTLIB_AVAILABLE:
            return

        self.ax.set_facecolor("black")  # Black background for plot area
        self.ax.tick_params(colors="white", which="both")
        self.ax.spines["bottom"].set_color("white")
        self.ax.spines["top"].set_color("white")
        self.ax.spines["right"].set_color("white")
        self.ax.spines["left"].set_color("white")
        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")
        self.ax.title.set_color("white")

    def _plot_empty(self):
        """Plot an empty canvas with placeholder text."""
        if not MATPLOTLIB_AVAILABLE:
            return

        self.ax.clear()
        self._apply_dark_theme()

        self.ax.text(
            0.5,
            0.5,
            "No fluorescence image loaded",
            horizontalalignment="center",
            verticalalignment="center",
            transform=self.ax.transAxes,
            fontsize=12,
            color="#bbbbbb",
        )
        self.ax.set_title("Fluorescence Image")
        self.ax.axis("off")
        self.figure.tight_layout()
        self.canvas.draw()

    def set_image(self, image: FluorescenceImage):
        """Set the fluorescence image to display.

        Args:
            image: The FluorescenceImage to display
        """
        self.image = image
        self.selected_channel_idx = None  # Default to "All Channels"

        # Update the plot (which will also update channel buttons)
        self.update_plot()

    def _update_channel_buttons(self):
        """Update the channel selection radio buttons based on loaded image."""
        # Clear existing buttons from layout
        while self.channel_buttons_layout.count():
            item = self.channel_buttons_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if self.image is None or self.image.metadata is None:
            return

        # Create new button group
        self.channel_button_group = QButtonGroup(self)

        # Create "All Channels" radio button
        all_channels_button = QRadioButton("All Channels", self)
        all_channels_button.setStyleSheet("""
            QRadioButton {
                color: #bbbbbb;
                font-size: 10px;
            }
            QRadioButton::indicator {
                width: 13px;
                height: 13px;
            }
        """)
        # Set "All Channels" as checked if that's the current selection
        if self.selected_channel_idx is None:
            all_channels_button.setChecked(True)
        # Connect signal with None as channel index
        all_channels_button.toggled.connect(lambda checked: self._on_channel_changed(None, checked))
        self.channel_button_group.addButton(all_channels_button, -1)  # Use -1 as ID for "All"
        self.channel_buttons_layout.addWidget(all_channels_button)

        # Create radio buttons for each individual channel
        num_channels = len(self.image.metadata.channels)
        for i, channel_metadata in enumerate(self.image.metadata.channels):
            channel_name = channel_metadata.name or f"Channel {i+1}"
            radio_button = QRadioButton(channel_name, self)
            radio_button.setStyleSheet("""
                QRadioButton {
                    color: #bbbbbb;
                    font-size: 10px;
                }
                QRadioButton::indicator {
                    width: 13px;
                    height: 13px;
                }
            """)

            # Set as checked if this is the currently selected channel
            if self.selected_channel_idx == i:
                radio_button.setChecked(True)

            # Connect signal
            radio_button.toggled.connect(lambda checked, idx=i: self._on_channel_changed(idx, checked))

            self.channel_button_group.addButton(radio_button, i)
            self.channel_buttons_layout.addWidget(radio_button)

        # Add stretch to push buttons to the left
        self.channel_buttons_layout.addStretch()

    def _on_channel_changed(self, channel_idx: Optional[int], checked: bool):
        """Handle channel selection change.

        Args:
            channel_idx: Channel index (0-based) or None for all channels
            checked: Whether the radio button was checked
        """
        if checked:
            self.selected_channel_idx = channel_idx
            self.update_plot()

    def update_plot(self):
        """Update the fluorescence image plot with current settings."""
        if not MATPLOTLIB_AVAILABLE:
            return

        if self.image is None:
            self._plot_empty()
            return

        try:
            if self.figure is not None:
                self.figure.clf()  # Clear existing figure
                plt.close(self.figure)
            # Determine selected channels to display
            # None = all channels, int = specific channel
            selected_channels = None if self.selected_channel_idx is None else [self.selected_channel_idx]

            # Use plot_fluorescence_image to create the plot with max projection
            # This now returns (fig, ax)
            fig, returned_ax = plot_fluorescence_image(
                image=self.image,
                filename=None,  # Don't save to file
                dpi=80,
                metadata_height=0,  # Don't show metadata bar
                display_metadata=False,
                selected_channels=selected_channels,
                z_index=None,  # Always use max projection
            )

            # Recreate the canvas and toolbar with the new figure
            # This ensures the toolbar works properly
            self.figure = fig
            self.ax = returned_ax

            # Create new canvas with the new figure
            self.canvas = FigureCanvas(self.figure)
            self.canvas.setMinimumSize(300, 300)

            # Create new toolbar with the new canvas
            self.toolbar = NavigationToolbar(self.canvas, self)
            self.toolbar.setStyleSheet("""
                QToolBar {
                    background-color: #3a3a3a;
                    border: 1px solid #555;
                    spacing: 3px;
                }
                QToolButton {
                    background-color: #3a3a3a;
                    color: white;
                    border: 1px solid #555;
                    padding: 3px;
                }
                QToolButton:hover {
                    background-color: #4a4a4a;
                }
                QToolButton:pressed {
                    background-color: #2a2a2a;
                }
            """)

            # Clear the plot layout and add new widgets
            while self.plot_layout.count():
                item = self.plot_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            self.plot_layout.addWidget(self.toolbar)
            self.plot_layout.addWidget(self.canvas)

            # Set title with channel name and mode
            title = "Fluorescence Image (Max Projection)"
            if self.selected_channel_idx is None:
                # All channels selected
                title = "All Channels (Max Projection)"
            elif self.image.metadata and self.selected_channel_idx < len(self.image.metadata.channels):
                # Specific channel selected
                channel_name = self.image.metadata.channels[self.selected_channel_idx].name
                title = f"{channel_name} (Max Projection)"
            self.ax.set_title(title, color='white')

            # Apply dark theme
            self._apply_dark_theme()

            # Set figure background to black
            self.figure.patch.set_facecolor('black')
            self.ax.set_facecolor('black')

            # Draw the canvas
            self.canvas.draw()

            # Recreate channel buttons after plot update
            self._update_channel_buttons()

        except Exception as e:
            logging.error(f"Error updating fluorescence plot: {e}")
            import traceback
            traceback.print_exc()
            self._plot_empty()

    def clear_plot(self):
        """Clear the plot and reset state."""
        self.image = None
        self.selected_channel_idx = None  # Reset to "All Channels"
        self._update_channel_buttons()
        self._plot_empty()

    def _on_load_image_clicked(self):
        """Handle load image button click to select and load a fluorescence image file."""
        if not MATPLOTLIB_AVAILABLE:
            return

        # Open file dialog to select an image file
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Fluorescence Image",
            "",
            "Image Files (*.tif *.tiff *.ome.tif *.ome.tiff);;All Files (*.*)"
        )

        if not file_path:
            # User cancelled the dialog
            return

        try:
            # Load the image using FluorescenceImage.load
            image = FluorescenceImage.load(file_path)

            # Set the image
            self.set_image(image)

            logging.info(f"Loaded fluorescence image from: {file_path}")
            logging.info(f"Image shape: {image.data.shape}, dtype: {image.data.dtype}")

        except Exception as e:
            logging.error(f"Error loading fluorescence image: {e}")
            import traceback
            traceback.print_exc()


def create_widget(parent: Optional[QWidget] = None) -> FluorescencePlotWidget:
    """Create the FluorescencePlotWidget.

    Args:
        parent: Optional parent widget

    Returns:
        FluorescencePlotWidget instance
    """
    widget = FluorescencePlotWidget(parent=parent)
    return widget


def main():
    """Main function to run the widget standalone."""
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    widget = create_widget()
    widget.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
