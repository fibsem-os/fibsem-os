import logging
import os
from copy import deepcopy
from typing import List, Optional

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QGridLayout, QLabel, QSizePolicy, QSpacerItem

import fibsem
from fibsem.detection import detection
from fibsem.detection import utils as det_utils
from fibsem.detection.detection import DetectedFeatures
from fibsem.segmentation.model import load_model
from fibsem.structures import (
    BeamType,
    FibsemImage,
    Point,
)


class FibsemEmbeddedDetectionUI(QtWidgets.QWidget):
    continue_signal = pyqtSignal(DetectedFeatures)

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent=parent)
        self._setup_ui()

        self.parent = parent
        self.has_user_corrected: bool = False

        # quad-view: a "mask" MaskSpec + a modal "detection" PointsSpec on a beam
        # canvas, owned by the controller.
        self._host_beam = None         # BeamType the detection overlays are hosted on
        self._detection_wired = False  # subscribed to controller.overlay_edited

        self.setup_connections()

    # ------------------------------------------------------------------
    # Quad-view overlays (gated; napari path preserved when no controller)
    # ------------------------------------------------------------------

    def _view_controller(self):
        """Return the quad-view MicroscopeViewController, or None (napari path)."""
        parent_ui = getattr(self.parent, "parent_widget", None)
        return getattr(parent_ui, "view_controller", None)

    def _host_beam_for(self, det):
        """Pick the beam to host detection — the beam of ``det.fibsem_image``
        (fallback ION), or None on the napari path."""
        if self._view_controller() is None:
            return None
        fib = getattr(det, "fibsem_image", None)
        beam = fib.metadata.beam_type if fib is not None and fib.metadata is not None else None
        return beam if beam in (BeamType.ELECTRON, BeamType.ION) else BeamType.ION

    def _detach_detection(self, beam):
        """Remove the detection overlays from *beam* (e.g. when the host beam changes)."""
        controller = self._view_controller()
        if controller is None or beam is None:
            return
        controller.arm_overlay(beam, None)
        controller.remove_overlay(beam, "detection")
        controller.remove_overlay(beam, "mask")
        canvas = controller.get_canvas(beam)
        if canvas is not None:
            canvas.set_hint(None)

    def _update_features(self, beam):
        """Show the detection image + read-only mask + draggable feature points on the
        *beam* canvas via the reducer (quad-view equivalent of update_features_ui)."""
        from fibsem.ui.widgets.canvas_state import MaskSpec, PointsSpec

        controller = self._view_controller()
        if controller is None:
            return
        if self._host_beam is not None and self._host_beam is not beam:
            self._detach_detection(self._host_beam)  # host beam changed
        self._host_beam = beam
        if not self._detection_wired:
            controller.overlay_edited.connect(self._on_detection_edited)
            self._detection_wired = True

        # the detection image: its pixel space matches det.mask + feature.px
        if self.det.fibsem_image is not None:
            controller.set_image(beam, self.det.fibsem_image)
        controller.set_overlay(beam, MaskSpec(mask=self.det.mask))  # display-only
        controller.set_overlay(
            beam,
            PointsSpec(
                id="detection",
                points=[(f.px.x, f.px.y) for f in self.det.features],
                colors=[f.color for f in self.det.features],
                labels=[f.name for f in self.det.features],
                marker="+", size=16, removable=False, add_on_right_click=False, modal=True,
            ),
        )
        controller.arm_overlay(beam, "detection", label="Detection", icon="mdi:vector-point")
        canvas = controller.get_canvas(beam)
        if canvas is not None:
            canvas.set_hint("drag features to correct  ·  Continue when done")
        self.update_info()

    def _on_detection_edited(self, beam, overlay_id, points):
        """A feature point was dragged → update ``feature.px`` (mirrors update_point)."""
        if overlay_id != "detection":
            return
        for i, (x, y) in enumerate(points):
            if i < len(self.det.features):
                self.det.features[i].px = Point(x=x, y=y)
        self.has_user_corrected = True
        self.update_info()

    def closeEvent(self, event) -> None:
        if self._detection_wired:
            controller = self._view_controller()
            if controller is not None:
                try:
                    controller.overlay_edited.disconnect(self._on_detection_edited)
                except (TypeError, RuntimeError):
                    pass
            self._detection_wired = False
        super().closeEvent(event)

    def _setup_ui(self):
        self.gridLayout = QGridLayout(self)

        # title label - bold
        self.label_title = QLabel("Feature Detection")
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        self.label_title.setFont(font)
        self.gridLayout.addWidget(self.label_title, 0, 0, 1, 2)

        # model label
        self.label_model = QLabel("Model:")
        self.gridLayout.addWidget(self.label_model, 1, 0, 1, 2)

        # info label
        self.label_info = QLabel("")
        self.label_info.setWordWrap(True)
        self.gridLayout.addWidget(self.label_info, 2, 0, 1, 2)

        # instructions label
        self.label_instructions = QLabel(
            "Drag to move the features, when finished press confirm."
        )
        self.label_instructions.setWordWrap(True)
        self.gridLayout.addWidget(self.label_instructions, 3, 0, 1, 2)

        # vertical spacer
        self.gridLayout.addItem(
            QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding),
            4, 0, 1, 2,
        )

    def setup_connections(self):
        self.label_instructions.setText(
            """Drag the detected feature positions to move them. Press Continue when finished."""
        )

    def clear_layers(self):
        """Remove the detection overlays from the canvas."""
        controller = self._view_controller()
        if controller is not None:
            if self._host_beam is not None:
                self._detach_detection(self._host_beam)
            self._host_beam = None

    def confirm_button_clicked(self):
        """Confirm the detected features, save the data and clear the overlays."""
        # mask is display-only (no painting), so det.mask stays the model output — but
        # normalise its dtype to uint8 (save_feature_data_to_csv -> PIL.Image.fromarray needs it).
        if self.det.mask is not None:
            self.det.mask = np.asarray(self.det.mask).astype(np.uint8)
        det_utils.save_ml_feature_data(det=self.det,
                                       initial_features=self._intial_det.features)
        self.clear_layers()

    def set_detected_features(self, det_features: DetectedFeatures):
        """Set the detected features and update the UI"""
        self.det = det_features
        self._intial_det = deepcopy(det_features)
        self.has_user_corrected = False

        self.update_features_ui()

    def update_features_ui(self):
        """Update the UI with the detected features"""
        beam = self._host_beam_for(self.det)
        if beam is not None:
            self._update_features(beam)
        if self.det.checkpoint:
            self.label_model.setText(f"Checkpoint: {os.path.basename(self.det.checkpoint)}")

    def update_info(self):
        """Update the info label with the feature information"""
        if len(self.det.features) > 2:
            self.label_info.setText("Info not available.")
            return
        
        if len(self.det.features) == 1:
            self.label_info.setText(
            f"""{self.det.features[0].name}: {self.det.features[0].px}
            User Corrected: {self.has_user_corrected}
            """)
            return
        if len(self.det.features) == 2:
            self.label_info.setText(
                f"""Moving 
                {self.det.features[0].name}: {self.det.features[0].px}
                to 
                {self.det.features[1].name}: {self.det.features[1].px}
                dx={self.det.distance.x*1e6:.2f}um, dy={self.det.distance.y*1e6:.2f}um
                User Corrected: {self.has_user_corrected}
                """
                )
            return

    def _get_detected_features(self):

        from fibsem import conversions

        for feature in self.det.features:
            feature.feature_m = conversions.image_to_microscope_image_coordinates(
                feature.px, self.det.image.data, self.det.pixelsize
            )

        logging.debug({"msg": "get_detected_features", "detected_features": self.det.to_dict()})

        return self.det


def main():
    # load model
    checkpoint = "autolamella-mega-20240107.pt"
    model = load_model(checkpoint=checkpoint)
    
    # load image
    image = FibsemImage.load(os.path.join(os.path.dirname(detection.__file__), "test_image_2.tif"))

    pixelsize = image.metadata.pixel_size.x if image.metadata is not None else 25e-9

    # detect features
    features = [detection.LamellaRightEdge(), detection.LandingPost()]
    det = detection.detect_features(
        deepcopy(image.data), model, features=features, pixelsize=pixelsize
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    det_widget_ui = FibsemEmbeddedDetectionUI()
    det_widget_ui.set_detected_features(det)
    det_widget_ui.show()
    app.exec_()

    det = det_widget_ui.det


if __name__ == "__main__":
    main()