"""The stage position readout and entry form.

Five spin boxes and a refresh button. It says where the stage is, in the units an
operator types in, and it hands back what has been typed. It does not move the stage,
and it has no opinion about who does -- pressing refresh emits a signal and stops there,
so the host decides what a refresh costs.

Extracted from ``FibsemMovementWidget`` (FIB-783), which does this alongside the
movement actions, their workers, the canvas double-click registration and the progress
reporting. Nothing in this file needs a parent contract, a view controller or an image
widget; all of that belongs to the half that moves.

**Units.** The stage speaks metres and radians. The form shows millimetres and degrees,
except the tilt *limits*, which the stage already reports in degrees -- so x, y and z
are converted on the way through and t is not. That asymmetry is the one thing here
worth reading twice.

**Device reads.** Exactly one, at construction, for the ranges and the stage kind. Both
are configuration rather than state: they change when someone reconfigures the
microscope, not while it is running. Nothing on a UI event path touches the device.
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

# The default arrow step, in millimetres. A compustage overrides it: see
# _apply_stage_configuration.
_TRANSLATION_STEP_MM = 0.001
_COMPUSTAGE_STEP_MM = 1e-6 * constants.SI_TO_MILLI


class StagePositionWidget(QWidget):
    """Where the stage is, and where the operator would like it to be.

    Parameters
    ----------
    microscope:
        Read once, during construction, for the axis ranges and whether this is a
        compustage. Never read again.
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
        """The one device read: axis ranges, and whether this stage rotates."""
        limits = self.microscope._stage.limits

        for axis, spinbox in (
            ("x", self.spinbox_x),
            ("y", self.spinbox_y),
            ("z", self.spinbox_z),
        ):
            spinbox.setMinimum(limits[axis].min * constants.SI_TO_MILLI)
            spinbox.setMaximum(limits[axis].max * constants.SI_TO_MILLI)

        # Not converted: the stage reports x/y/z in metres but t in degrees, which is
        # already the unit on the box.
        self.spinbox_tilt.setMinimum(limits["t"].min)
        self.spinbox_tilt.setMaximum(limits["t"].max)

        if self.microscope.stage_is_compustage:
            # A compustage is driven in micrometres, so a whole-millimetre arrow step
            # is a thousand times too coarse to be useful.
            for spinbox in (self.spinbox_x, self.spinbox_y, self.spinbox_z):
                spinbox.setSingleStep(_COMPUSTAGE_STEP_MM)

            # It does not rotate. The label goes with the box -- hiding only the box
            # leaves a caption over the tilt row.
            self.label_rotation.setVisible(False)
            self.spinbox_rotation.setVisible(False)

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
