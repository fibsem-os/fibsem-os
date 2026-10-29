"""Where an offset FM is from the beams and the pose it images in, edited.

The ``fm`` entry's ``origin`` (x, y) and ``pose`` (r, t), in the configuration window's
Geometry tab. Saved by that window's Save into the configuration file
(`fibsem.utils.write_fm_placement`), and used from the next connect: the session's
stage keeps the place and pose it was built with until then.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QPushButton,
    QWidget,
)

from fibsem import utils
from fibsem.ui import notification_service, stylesheets
from fibsem.ui.widgets.custom_widgets import ValueSpinBox


class FMPlacementForm(QWidget):
    """The FM's origin and pose; ``changed`` fires on every edit."""

    changed = pyqtSignal()

    def __init__(self, microscope, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.microscope = microscope
        device = microscope.system.stage.devices["FM"]
        fib = microscope.get_orientation("FIB")

        self.origin_x = ValueSpinBox("mm", -100.0, 100.0, 0.01, 4)
        self.origin_y = ValueSpinBox("mm", -100.0, 100.0, 0.01, 4)
        self.own_pose = QCheckBox("Images in its own pose, not FIB's")
        self.pose_r = ValueSpinBox("°", -360.0, 360.0, 0.1, 2)
        self.pose_t = ValueSpinBox("°", -90.0, 90.0, 0.1, 2)
        self.button_stage_pose = QPushButton("Use the stage's pose")
        self.button_stage_pose.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.button_stage_pose.setToolTip(
            "Take r and t from where the stage is now: move it to the FM's imaging "
            "pose first."
        )

        self.origin_x.setValue((device.origin.x or 0.0) * 1e3)
        self.origin_y.setValue((device.origin.y or 0.0) * 1e3)
        pose = device.pose
        self.own_pose.setChecked(pose is not None)
        self.pose_r.setValue(
            float(np.degrees(fib.r if pose is None or pose.r is None else pose.r))
        )
        self.pose_t.setValue(
            float(np.degrees(fib.t if pose is None or pose.t is None else pose.t))
        )
        self._saved = self.values()

        pose_row = QHBoxLayout()
        pose_row.setContentsMargins(0, 0, 0, 0)
        pose_row.addWidget(self.pose_r)
        pose_row.addWidget(self.pose_t)
        pose_row.addWidget(self.button_stage_pose)
        pose_widget = QWidget()
        pose_widget.setLayout(pose_row)

        form = QFormLayout(self)
        form.setContentsMargins(8, 6, 8, 6)
        form.addRow("Origin x (CALIB dx)", self.origin_x)
        form.addRow("Origin y (CALIB dy)", self.origin_y)
        form.addRow("", self.own_pose)
        form.addRow("Pose r, t", pose_widget)

        for box in (self.origin_x, self.origin_y, self.pose_r, self.pose_t):
            box.valueChanged.connect(self._on_edit)
        self.own_pose.toggled.connect(self._on_edit)
        self.button_stage_pose.clicked.connect(self.use_stage_pose)
        self._apply_enabled()

    def values(self) -> Tuple[Tuple[float, float], Optional[Tuple[float, float]]]:
        """((x, y) in metres, (r, t) in degrees or None for the FIB pose)."""
        origin = (self.origin_x.value() * 1e-3, self.origin_y.value() * 1e-3)
        pose = (
            (self.pose_r.value(), self.pose_t.value())
            if self.own_pose.isChecked()
            else None
        )
        return origin, pose

    def is_modified(self) -> bool:
        return self.values() != self._saved

    def use_stage_pose(self) -> None:
        position = self.microscope.get_stage_position()
        self.own_pose.setChecked(True)
        self.pose_r.setValue(float(np.degrees(position.r)) % 360.0)
        self.pose_t.setValue(float(np.degrees(position.t)))

    def save_to_configuration(self) -> bool:
        """Write the FM's place and pose into the configuration file. Returns whether
        it was saved."""
        path = getattr(self.microscope, "configuration_path", None)
        if not path:
            notification_service.show_toast(
                "This session was not started from a configuration file, so there is "
                "nowhere to save the FM's place.",
                "warning",
            )
            return False
        origin, pose = self.values()
        try:
            utils.write_fm_placement(path, origin, pose)
        except Exception as e:
            logging.error(f"Could not save the FM's place: {e}")
            notification_service.show_toast(
                f"Could not save the FM's place: {e}", "error"
            )
            return False
        pose_changed = pose != self._saved[1]
        self._saved = self.values()
        self.changed.emit()
        logging.info(f"FM origin {origin} m and pose {pose} deg saved to {path}")
        message = "FM place saved. Reconnect to use it."
        if pose_changed and self.microscope.system.stage.fm_calibrated():
            message += (
                " The SEM to FM calibration was measured in the old pose: run it again."
            )
        notification_service.show_toast(message, "info")
        return True

    def _on_edit(self, *_) -> None:
        self._apply_enabled()
        self.changed.emit()

    def _apply_enabled(self) -> None:
        own = self.own_pose.isChecked()
        self.pose_r.setEnabled(own)
        self.pose_t.setEnabled(own)
