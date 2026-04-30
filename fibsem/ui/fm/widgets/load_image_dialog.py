"""
Dialog for loading fluorescence images from OME-TIFF files.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QListWidget,
)

from fibsem.fm.structures import FluorescenceImage
from fibsem.ui.stylesheets import GREEN_PUSHBUTTON_STYLE, RED_PUSHBUTTON_STYLE


class LoadImageDialog(QDialog):
    """Dialog for loading FluorescenceImage from OME-TIFF files."""

    image_loaded_signal = pyqtSignal(FluorescenceImage)

    def __init__(self, parent=None, start_directory: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("Load Fluorescence Image")
        self.setModal(True)
        self.resize(500, 200)

        self.loaded_images: List[FluorescenceImage] = []
        self.selected_file_paths: List[str] = []
        self.start_directory = start_directory or os.path.expanduser("~")

        self.init_ui()

    def init_ui(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout()

        # Title
        title_label = QLabel("Load Fluorescence Images from OME-TIFF Files")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title_label)

        # File selection section
        file_layout = QHBoxLayout()

        file_label = QLabel("Files:")
        file_label.setFixedWidth(50)
        file_layout.addWidget(file_label)

        self.file_list_widget = QListWidget()
        self.file_list_widget.setMaximumHeight(100)
        file_layout.addWidget(self.file_list_widget)

        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_file)
        file_layout.addWidget(self.browse_button)

        layout.addLayout(file_layout)

        # Info label
        info_label = QLabel(
            "Only OME-TIFF files (.ome.tiff, .ome.tif) are supported. Hold Ctrl to select multiple files."
        )
        info_label.setStyleSheet("color: #666666; font-size: 11px;")
        layout.addWidget(info_label)

        # Progress bar (initially hidden)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Buttons
        button_layout = QHBoxLayout()

        self.load_button = QPushButton("Load Images")
        self.load_button.setStyleSheet(GREEN_PUSHBUTTON_STYLE)
        self.load_button.clicked.connect(self.load_images)
        self.load_button.setEnabled(False)
        button_layout.addWidget(self.load_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet(RED_PUSHBUTTON_STYLE)
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def browse_file(self):
        """Open file browser to select OME-TIFF files."""
        file_filter = "OME-TIFF files (*.ome.tiff *.ome.tif)"

        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select OME-TIFF Files (Hold Ctrl for multiple)",
            self.start_directory,
            file_filter,
        )

        if file_paths:
            self.selected_file_paths = file_paths
            self.file_list_widget.clear()
            for file_path in file_paths:
                file_name = Path(file_path).name
                self.file_list_widget.addItem(file_name)
            self.load_button.setEnabled(True)

    def validate_files(self, file_paths: List[str]) -> List[str]:
        """Validate that the selected files are valid OME-TIFF files."""
        valid_files = []
        invalid_files = []

        valid_extensions = [".ome.tiff", ".ome.tif"]

        for file_path in file_paths:
            if not file_path:
                continue

            path = Path(file_path)

            # Check if file exists
            if not path.exists():
                invalid_files.append(f"File not found: {file_path}")
                continue

            # Check file extension
            if not any(file_path.lower().endswith(ext) for ext in valid_extensions):
                invalid_files.append(f"Invalid file type: {path.name}")
                continue

            valid_files.append(file_path)

        # Show warnings for invalid files
        if invalid_files:
            warning_msg = "Some files were skipped:\\n\\n" + "\\n".join(invalid_files)
            if valid_files:
                warning_msg += "\\n\\nValid files will still be loaded."
            QMessageBox.warning(self, "File Validation Issues", warning_msg)

        return valid_files

    def load_images(self):
        """Load the selected image files."""
        if not self.selected_file_paths:
            QMessageBox.warning(
                self, "No Files Selected", "Please select at least one OME-TIFF file."
            )
            return

        valid_file_paths = self.validate_files(self.selected_file_paths)
        if not valid_file_paths:
            return

        try:
            # Show progress bar
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, len(valid_file_paths))
            self.load_button.setEnabled(False)
            self.browse_button.setEnabled(False)

            self.loaded_images = []
            successful_loads = 0
            failed_loads = []

            for i, file_path in enumerate(valid_file_paths):
                try:
                    # Update progress
                    self.progress_bar.setValue(i)

                    # Load the image using FluorescenceImage.load()
                    logging.info(f"Loading fluorescence image from: {file_path}")
                    image = FluorescenceImage.load(file_path)
                    self.loaded_images.append(image)

                    # Emit signal for each loaded image
                    self.image_loaded_signal.emit(image)
                    successful_loads += 1

                except Exception as e:
                    error_msg = f"Failed to load {Path(file_path).name}: {str(e)}"
                    failed_loads.append(error_msg)
                    logging.error(error_msg)

            # Complete progress bar
            self.progress_bar.setValue(len(valid_file_paths))

            # Hide progress bar and re-enable buttons
            self.progress_bar.setVisible(False)
            self.load_button.setEnabled(True)
            self.browse_button.setEnabled(True)

            # Show results
            if successful_loads > 0:
                success_msg = f"Successfully loaded {successful_loads} image(s)."
                if failed_loads:
                    success_msg += (
                        f"\\n\\nFailed to load {len(failed_loads)} file(s):\\n"
                        + "\\n".join(failed_loads)
                    )

                QMessageBox.information(self, "Load Complete", success_msg)
                logging.info(f"Successfully loaded {successful_loads} images")
                self.accept()
            else:
                error_msg = "Failed to load any images:\\n\\n" + "\\n".join(
                    failed_loads
                )
                QMessageBox.critical(self, "Load Error", error_msg)

        except Exception as e:
            # Hide progress bar and re-enable buttons
            self.progress_bar.setVisible(False)
            self.load_button.setEnabled(True)
            self.browse_button.setEnabled(True)

            error_msg = f"Unexpected error during loading: {str(e)}"
            logging.error(error_msg)

            QMessageBox.critical(self, "Load Error", error_msg)
