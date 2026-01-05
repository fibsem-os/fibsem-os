from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QGridLayout,
    QPushButton,
    QCheckBox,
)

from fibsem.ui.stylesheets import (
    BLUE_PUSHBUTTON_STYLE,
    GRAY_PUSHBUTTON_STYLE,
)

from typing import TYPE_CHECKING


class DisplayOptionsDialog(QDialog):
    """Dialog for configuring display overlay options."""

    def __init__(self, parent: 'Widget'):
        super().__init__(parent)
        self.parent_widget = parent
        self.setWindowTitle("Display Options")
        self.setModal(True)
        self.initUI()

    def initUI(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout()

        # Checkboxes for each display option
        self.checkbox_current_fov = QCheckBox("Show Current FOV")
        self.checkbox_current_fov.setChecked(self.parent_widget.show_current_fov)
        layout.addWidget(self.checkbox_current_fov)

        self.checkbox_overview_fov = QCheckBox("Show Overview FOV")
        self.checkbox_overview_fov.setChecked(self.parent_widget.show_overview_fov)
        layout.addWidget(self.checkbox_overview_fov)

        self.checkbox_saved_positions_fov = QCheckBox("Show Saved Positions FOV")
        self.checkbox_saved_positions_fov.setChecked(self.parent_widget.show_saved_positions_fov)
        layout.addWidget(self.checkbox_saved_positions_fov)

        self.checkbox_stage_limits = QCheckBox("Show Stage Limits")
        self.checkbox_stage_limits.setChecked(self.parent_widget.show_stage_limits)
        layout.addWidget(self.checkbox_stage_limits)

        self.checkbox_circle_overlays = QCheckBox("Show Circle Overlays")
        self.checkbox_circle_overlays.setChecked(self.parent_widget.show_circle_overlays)
        layout.addWidget(self.checkbox_circle_overlays)

        self.checkbox_histogram = QCheckBox("Show Image Histogram")
        self.checkbox_histogram.setChecked(self.parent_widget.show_histogram)
        layout.addWidget(self.checkbox_histogram)

        # Buttons
        button_layout = QGridLayout()

        self.button_ok = QPushButton("OK")
        self.button_ok.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        self.button_ok.clicked.connect(self.accept)
        button_layout.addWidget(self.button_ok, 0, 0)

        self.button_cancel = QPushButton("Cancel")
        self.button_cancel.setStyleSheet(GRAY_PUSHBUTTON_STYLE)
        self.button_cancel.clicked.connect(self.reject)
        button_layout.addWidget(self.button_cancel, 0, 1)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def get_display_options(self) -> dict:
        """Get the selected display options."""
        return {
            'show_current_fov': self.checkbox_current_fov.isChecked(),
            'show_overview_fov': self.checkbox_overview_fov.isChecked(),
            'show_saved_positions_fov': self.checkbox_saved_positions_fov.isChecked(),
            'show_stage_limits': self.checkbox_stage_limits.isChecked(),
            'show_circle_overlays': self.checkbox_circle_overlays.isChecked(),
            'show_histogram': self.checkbox_histogram.isChecked(),
        }