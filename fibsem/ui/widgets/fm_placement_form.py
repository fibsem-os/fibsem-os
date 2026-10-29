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
from PyQt5.QtWidgets import QFormLayout, QPushButton, QWidget

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
        self.origin_x.setToolTip("odemis CALIB dx")
        self.origin_y.setToolTip("odemis CALIB dy")
        self.pose_r = ValueSpinBox("°", -360.0, 360.0, 0.1, 2)
        self.pose_t = ValueSpinBox("°", -90.0, 90.0, 0.1, 2)
        self._fib_pose = (
            round(float(np.degrees(fib.r)), 2),
            round(float(np.degrees(fib.t)), 2),
        )
        self.button_stage_pose = QPushButton("Use stage pose")
        self.button_stage_pose.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.button_stage_pose.setToolTip("Take r and t from the stage's position now.")

        self.origin_x.setValue((device.origin.x or 0.0) * 1e3)
        self.origin_y.setValue((device.origin.y or 0.0) * 1e3)
        pose = device.pose
        self.pose_r.setValue(
            float(np.degrees(fib.r if pose is None or pose.r is None else pose.r))
        )
        self.pose_t.setValue(
            float(np.degrees(fib.t if pose is None or pose.t is None else pose.t))
        )
        self._saved = self.values()

        form = QFormLayout(self)
        form.setContentsMargins(8, 6, 8, 6)
        form.addRow("Origin x", self.origin_x)
        form.addRow("Origin y", self.origin_y)
        form.addRow("Rotation", self.pose_r)
        form.addRow("Tilt", self.pose_t)
        form.addRow("", self.button_stage_pose)

        for box in (self.origin_x, self.origin_y, self.pose_r, self.pose_t):
            box.valueChanged.connect(self.changed)
        self.button_stage_pose.clicked.connect(self.use_stage_pose)

    def values(self) -> Tuple[Tuple[float, float], Optional[Tuple[float, float]]]:
        """((x, y) in metres, (r, t) in degrees, or None while the pose is FIB's)."""
        origin = (self.origin_x.value() * 1e-3, self.origin_y.value() * 1e-3)
        pose = (round(self.pose_r.value(), 2), round(self.pose_t.value(), 2))
        return origin, None if pose == self._fib_pose else pose

    def is_modified(self) -> bool:
        return self.values() != self._saved

    def use_stage_pose(self) -> None:
        position = self.microscope.get_stage_position()
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
        message = "FM origin and pose saved. Reconnect to use them."
        if pose_changed and self.microscope.system.stage.fm_calibrated():
            message += (
                " The SEM to FM calibration was measured in the old pose: run it again."
            )
        notification_service.show_toast(message, "info")
        return True
