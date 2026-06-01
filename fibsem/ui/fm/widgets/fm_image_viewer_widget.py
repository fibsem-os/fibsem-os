"""
Simple napari widget for loading and displaying FluorescenceImages from file.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Union

import napari
import numpy as np
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QWidget
from superqt import ensure_main_thread

from fibsem.fm.structures import FluorescenceImage, FluorescenceChannelMetadata, FluorescenceImageMetadata
from fibsem.ui.fm.widgets.load_image_dialog import LoadImageDialog
from fibsem.ui.stylesheets import BLUE_PUSHBUTTON_STYLE


def _image_metadata_to_napari_image_layer(
    metadata: FluorescenceImageMetadata, image_shape: tuple[int, int], channel_index: int = 0
) -> dict:
    """Convert FluorescenceImageMetadata to a dictionary compatible with napari image layer.

    This function extracts relevant metadata from the FluorescenceImageMetadata object
    and formats it into a dictionary that can be used to create a napari image layer.

    Args:
        metadata: FluorescenceImageMetadata object containing image metadata
        image_shape: Shape of the image (height, width)
        channel_index: Index of the channel to extract metadata for (default is 0)

    Returns:
        A dictionary containing the metadata formatted for napari image layer.
    """
    # Convert structured metadata to dictionary for napari compatibility
    metadata_dict = metadata.to_dict() if metadata else {}

    channel_name = metadata.channels[channel_index].name
    colormap = metadata.channels[channel_index].color or "gray"
    pixel_size_x = metadata.pixel_size_x
    pixel_size_y = metadata.pixel_size_y
    pixel_size_z = metadata.pixel_size_z

    return {
        "name": channel_name,
        "description": metadata.description or channel_name,
        "metadata": metadata_dict,
        "colormap": colormap,
        "scale": (pixel_size_y, pixel_size_x),  # yx order for napari
        "pixel_size_z": pixel_size_z,
        "blending": "additive",
    }


class FMImageViewerWidget(QWidget):
    """Simple widget for loading and displaying FluorescenceImages."""

    image_loaded_signal = pyqtSignal(FluorescenceImage)

    def __init__(
        self,
        viewer: napari.Viewer,
        parent: Optional[QWidget] = None,
        start_directory: Optional[Union[str, Path]] = None,
    ):
        super().__init__(parent)

        self.viewer = viewer
        self.start_directory = str(start_directory) if start_directory else None

        self.initUI()

    def initUI(self):
        """Initialize the user interface."""
        # Create load button
        self.pushButton_load_image = QPushButton("Load Image...", self)
        self.pushButton_load_image.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self.pushButton_load_image.clicked.connect(self.show_load_image_dialog)

        # Create layout
        layout = QVBoxLayout()
        layout.addWidget(self.pushButton_load_image)
        layout.addStretch()
        self.setLayout(layout)

        # Configure napari scale bar
        self.viewer.scale_bar.visible = True
        self.viewer.scale_bar.unit = "m"

        # Connect signal
        self.image_loaded_signal.connect(self.display_image)

    def show_load_image_dialog(self):
        """Show the load image dialog and display the loaded image."""
        dialog = LoadImageDialog(self, start_directory=self.start_directory)

        # Connect the dialog's signal to our signal
        dialog.image_loaded_signal.connect(self.image_loaded_signal.emit)

        ret = dialog.exec_()
        if ret:
            logging.info("Image loaded successfully")
        else:
            logging.info("Image loading canceled")

    @ensure_main_thread
    @pyqtSlot(FluorescenceImage)
    def display_image(self, image: FluorescenceImage):
        """Display the loaded image in napari viewer.

        Args:
            image: FluorescenceImage object containing the image data and metadata
        """
        try:
            # Convert structured metadata to dictionary for napari compatibility
            image_height, image_width = image.data.shape[-2:]

            for channel_index in range(image.data.shape[0]):
                metadata_dict = _image_metadata_to_napari_image_layer(
                    image.metadata, (image_height, image_width), channel_index=channel_index
                )
                layer_name = f"{metadata_dict['description']}-{metadata_dict['name']}"
                self._update_napari_image_layer(
                    layer_name, image.data[channel_index], metadata_dict
                )

            logging.info(f"Displayed image: {image.metadata.description}")

        except Exception as e:
            logging.error(f"Error displaying image: {e}")

    def _update_napari_image_layer(
        self, layer_name: str, image: np.ndarray, metadata_dict: dict
    ):
        """Update or create a napari image layer with the given metadata.

        Args:
            layer_name: Name of the napari image layer
            image: Image data array
            metadata_dict: Dictionary containing metadata for the image layer
        """
        # Make sure all images are 3D for napari reasons (required to transform)
        if image.ndim == 2:
            image = image[np.newaxis, ...]

        # Add a singleton dimension for z if needed
        if image.ndim == 3 and len(metadata_dict["scale"]) == 2:
            metadata_dict["scale"] = (metadata_dict["pixel_size_z"], *metadata_dict["scale"])

        if layer_name in self.viewer.layers:
            # If the layer already exists, update it
            self.viewer.layers[layer_name].data = image
            self.viewer.layers[layer_name].metadata = metadata_dict["metadata"]
            self.viewer.layers[layer_name].colormap = metadata_dict["colormap"]
        else:
            # If the layer does not exist, create a new one
            self.viewer.add_image(
                data=image,
                name=layer_name,
                metadata=metadata_dict["metadata"],
                colormap=metadata_dict["colormap"],
                scale=metadata_dict["scale"],
                blending="additive",
            )


def create_widget(
    viewer: napari.Viewer, experiment_path: Optional[Union[str, Path]] = None
) -> FMImageViewerWidget:
    """Factory function to create the widget for napari plugin.

    Args:
        viewer: napari Viewer instance
        experiment_path: Optional path to experiment directory to use as start directory

    Returns:
        FMImageViewerWidget instance
    """
    widget = FMImageViewerWidget(viewer=viewer, start_directory=experiment_path)
    return widget


def main():
    """Standalone application for testing."""
    from fibsem.applications.autolamella.config import LOG_PATH
    viewer = napari.Viewer()
    widget = create_widget(viewer, LOG_PATH)
    viewer.window.add_dock_widget(widget, area="right", name="FM Image Viewer")
    napari.run()


if __name__ == "__main__":
    main()
