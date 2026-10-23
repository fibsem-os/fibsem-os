"""The stage position readout and entry form.

Five spin boxes and a refresh button. It says where the stage is, in the units an
operator types in, and it hands back what has been typed. It does not move the stage,
and it has no opinion about who does -- pressing refresh emits a signal and stops there,
so the host decides what a refresh costs.

Extracted from ``FibsemMovementWidget``, which does this alongside the
movement actions, their workers, the canvas double-click registration and the progress
reporting. Nothing in this file needs a parent contract, a view controller or an image
widget; all of that belongs to the half that moves.

**Units.** The stage speaks metres and radians, limits included. The form shows
millimetres and degrees, so every axis is converted on the way through, by the scale
``FibsemStagePosition`` declares for it. Without a stage device every axis is hidden.

**Device reads.** Exactly one, at construction, for the axes and their ranges, from
the stage device's metadata. Both are configuration rather than state: they change
when someone reconfigures the microscope, not while it is running. Nothing on a UI
event path touches the device.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QDoubleSpinBox, QGridLayout, QLabel, QWidget
from superqt import ensure_main_thread

from fibsem import constants
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import FibsemStagePosition
from fibsem.ui.utils import install_wheel_blocker
from fibsem.ui.widgets.custom_widgets import IconToolButton
from fibsem.ui.widgets.form_builder import configure_spinbox

# Each axis's box is shown as FibsemStagePosition declares it: millimetres to five
# decimals (ten nanometres, since stable_move routinely asks for less) with a
# micrometre arrow step, and degrees.
_SPECS = FibsemStagePosition.params


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

        self.label_x, self.spinbox_x = self._axis_row("x")
        self.label_y, self.spinbox_y = self._axis_row("y")
        self.label_z, self.spinbox_z = self._axis_row("z")
        self.label_rotation, self.spinbox_rotation = self._axis_row("r")
        self.label_tilt, self.spinbox_tilt = self._axis_row("t")

        for row, name in enumerate(FibsemStagePosition.AXES):
            layout.addWidget(self._labels()[name], row, 0)
            layout.addWidget(self._spinboxes()[name], row, 1)

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

    @staticmethod
    def _axis_row(name: str) -> Tuple[QLabel, QDoubleSpinBox]:
        spec = _SPECS[name]
        spinbox = QDoubleSpinBox()
        configure_spinbox(spinbox, spec)
        if spec.suffix == constants.DEGREE_SYMBOL:
            # the stage panel writes an angle as "12.50°", the sign against its number
            spinbox.setSuffix(spec.suffix)
        return QLabel(spec.label), spinbox

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
            scale = _SPECS[name].scale
            # Rounded: the stage keeps degrees-from-config in radians, and the round
            # trip would otherwise put 194.99999999 on the box.
            low = round(axis.limits.min * scale, 9)
            high = round(axis.limits.max * scale, 9)
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
        for name, spinbox in self._spinboxes().items():
            spinbox.setValue(getattr(stage_position, name) * _SPECS[name].scale)

    def get_position(self) -> FibsemStagePosition:
        """What is currently typed in, in the units the stage takes.

        No coordinate system is named: the boxes are the stage's own axes, and the
        stage device knows which frame those are.
        """
        return FibsemStagePosition(
            **{
                name: spinbox.value() / _SPECS[name].scale
                for name, spinbox in self._spinboxes().items()
            }
        )
