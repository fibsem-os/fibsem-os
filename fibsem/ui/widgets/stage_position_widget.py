"""The stage position readout and entry form.

Five spin boxes and a refresh button. It says where the stage is, in the units an
operator types in, and it hands back what has been typed. It does not move the stage,
and it has no opinion about who does -- pressing refresh emits a signal and stops there,
so the host decides what a refresh costs.

Extracted from ``FibsemMovementWidget`` (FIB-783), which does this alongside the
movement actions, their workers, the canvas double-click registration and the progress
reporting. Nothing in this file needs a parent contract, a view controller or an image
widget; all of that belongs to the half that moves.

**Units.** The stage speaks metres and radians, limits included. The form shows
millimetres and degrees, so every axis is converted on the way through, by the unit
the stage device gives it. (Without a stage device the limits come from the stage
model, whose tilt limits are already in degrees, so t is not converted there.)

**Device reads.** Exactly one, at construction, for the axes and their ranges, from
the stage device's metadata. Both are configuration rather than state: they change
when someone reconfigures the microscope, not while it is running. Nothing on a UI
event path touches the device.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QDoubleSpinBox, QGridLayout, QLabel, QWidget
from superqt import ensure_main_thread

from fibsem import constants
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import FibsemStagePosition
from fibsem.ui.utils import install_wheel_blocker
from fibsem.ui.widgets.custom_widgets import IconToolButton

# Five decimals of a millimetre -- ten nanometres. A stage move smaller than that
# rounds to nothing in the box, and stable_move routinely asks for less.
_TRANSLATION_DECIMALS = 5

# The arrow step, in millimetres: one micrometre. A compustage is driven in
# micrometres, so this is its step too.
_TRANSLATION_STEP_MM = 0.001


class StagePositionWidget(QWidget):
    """Where the stage is, and where the operator would like it to be.

    Parameters
    ----------
    microscope:
        Read once, during construction, for the stage's axes and their ranges.
        Never read again.
    """

    #: The refresh button was pressed. The host owns what happens next -- reading the
    #: stage is a device call, and this widget does not make those.
    refresh_requested = pyqtSignal()

    def __init__(
        self, microscope: FibsemMicroscope, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.microscope = microscope
        self._setup_ui()
        self._apply_stage_configuration()

    # --- construction --------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label_x = QLabel("X Coordinate")
        self.spinbox_x = QDoubleSpinBox()
        self.label_y = QLabel("Y Coordinate")
        self.spinbox_y = QDoubleSpinBox()
        self.label_z = QLabel("Z Coordinate")
        self.spinbox_z = QDoubleSpinBox()
        self.label_rotation = QLabel("Rotation")
        self.spinbox_rotation = QDoubleSpinBox()
        self.label_tilt = QLabel("Tilt")
        self.spinbox_tilt = QDoubleSpinBox()

        for row, (label, spinbox) in enumerate(
            (
                (self.label_x, self.spinbox_x),
                (self.label_y, self.spinbox_y),
                (self.label_z, self.spinbox_z),
                (self.label_rotation, self.spinbox_rotation),
                (self.label_tilt, self.spinbox_tilt),
            )
        ):
            layout.addWidget(label, row, 0)
            layout.addWidget(spinbox, row, 1)

        for spinbox in (self.spinbox_x, self.spinbox_y, self.spinbox_z):
            spinbox.setDecimals(_TRANSLATION_DECIMALS)
            spinbox.setSingleStep(_TRANSLATION_STEP_MM)
            spinbox.setSuffix(" mm")

        for spinbox in (self.spinbox_rotation, self.spinbox_tilt):
            spinbox.setSuffix(constants.DEGREE_SYMBOL)

        self.spinbox_rotation.setMinimum(-360.0)
        self.spinbox_rotation.setMaximum(360.0)

        # Guard every box against accidental scroll-to-change: the panel scrolls, and a
        # spinbox under the pointer would otherwise retype itself on the way past.
        for spinbox in self._spinboxes().values():
            install_wheel_blocker(spinbox)

        # Belongs to this widget, but lives in the host's panel header rather than in
        # the grid, so the host positions it. Kept here because refreshing the readout
        # is this widget's job to ask for.
        self.btn_refresh = IconToolButton(
            icon="mdi:refresh", tooltip="Refresh stage position"
        )
        self.btn_refresh.clicked.connect(self.refresh_requested)

    def _apply_stage_configuration(self) -> None:
        """The one device read: which axes the stage has, and their ranges.

        From the stage device's axes, whose limits it read at connect: an axis it does
        not have is hidden with its label, and with no stage every axis is."""
        stage = self.microscope.stage
        axes = {} if stage is None else stage.axes

        for name, spinbox in self._spinboxes().items():
            label = self._labels()[name]
            has_axis = name in axes
            label.setVisible(has_axis)
            spinbox.setVisible(has_axis)
            if not has_axis:
                continue
            axis = axes[name]
            low, high = axis.limits.min, axis.limits.max
            if axis.unit == "rad":
                # Rounded: the stage keeps degrees-from-config in radians, and the
                # round trip would otherwise put 194.99999999 on the box.
                low, high = round(np.degrees(low), 9), round(np.degrees(high), 9)
            else:
                low, high = low * constants.SI_TO_MILLI, high * constants.SI_TO_MILLI
            spinbox.setMinimum(low)
            spinbox.setMaximum(high)

    def _labels(self) -> Dict[str, QLabel]:
        return {
            "x": self.label_x,
            "y": self.label_y,
            "z": self.label_z,
            "r": self.label_rotation,
            "t": self.label_tilt,
        }

    def _spinboxes(self) -> Dict[str, QDoubleSpinBox]:
        return {
            "x": self.spinbox_x,
            "y": self.spinbox_y,
            "z": self.spinbox_z,
            "r": self.spinbox_rotation,
            "t": self.spinbox_tilt,
        }

    # --- the readout ---------------------------------------------------------

    @ensure_main_thread
    def set_position(self, stage_position: FibsemStagePosition) -> None:
        """Show *stage_position*, converting into the units on the boxes.

        Marshalled: a stage move reports where it landed from a worker thread, and Qt
        widgets may only be written from the GUI thread. Already on it, this is a
        direct call.
        """
        self.spinbox_x.setValue(stage_position.x * constants.SI_TO_MILLI)
        self.spinbox_y.setValue(stage_position.y * constants.SI_TO_MILLI)
        self.spinbox_z.setValue(stage_position.z * constants.SI_TO_MILLI)
        self.spinbox_rotation.setValue(np.degrees(stage_position.r))
        self.spinbox_tilt.setValue(np.degrees(stage_position.t))

    def get_position(self) -> FibsemStagePosition:
        """What is currently typed in, in the units the stage takes.

        ``RAW`` because the boxes are labelled with stage axes: what the operator reads
        off the box is the raw axis value, not one re-expressed in a linked frame.
        """
        return FibsemStagePosition(
            x=self.spinbox_x.value() * constants.MILLI_TO_SI,
            y=self.spinbox_y.value() * constants.MILLI_TO_SI,
            z=self.spinbox_z.value() * constants.MILLI_TO_SI,
            r=np.radians(self.spinbox_rotation.value()),
            t=np.radians(self.spinbox_tilt.value()),
            coordinate_system="RAW",
        )
