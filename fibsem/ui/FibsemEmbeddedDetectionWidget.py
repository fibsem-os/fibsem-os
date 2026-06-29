import logging
import os
from copy import deepcopy
from typing import List, Optional

import napari
import napari.utils.notifications
import numpy as np
from napari.layers import Image as NapariImageLayer
from napari.layers import Labels as NapariLabelsLayer
from napari.layers import Points as NapariPointsLayer
from napari.layers import Shapes as NapariShapesLayer
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QGridLayout, QLabel, QSizePolicy, QSpacerItem

import fibsem
from fibsem.detection import detection
from fibsem.detection import utils as det_utils
from fibsem.detection.detection import DetectedFeatures
from fibsem.segmentation.config import CLASS_COLORS
from fibsem.segmentation.model import load_model
from fibsem.structures import (
    FibsemImage,
    Point,
)
from fibsem.ui.napari.utilities import add_points_layer


class FibsemEmbeddedDetectionUI(QtWidgets.QWidget):
    continue_signal = pyqtSignal(DetectedFeatures)

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent=parent)
        self._setup_ui()

        self.parent = parent
        if parent is not None:
            viewer = parent.viewer
        else:
            viewer = napari.current_viewer()
        self.viewer: napari.Viewer = viewer

        self.has_user_corrected: bool = False

        self.image_layer: NapariImageLayer = None
        self.mask_layer: NapariLabelsLayer = None
        self.features_layer: NapariPointsLayer = None
        self.cross_hair_layer: NapariShapesLayer = None

        # quad-view: MaskOverlay + a modal features PointOverlay on a beam canvas
        # (created lazily when a controller is present; napari path used otherwise)
        self._mask_overlay = None
        self._features_overlay = None
        self._host_canvas = None

        self.setup_connections()

    # ------------------------------------------------------------------
    # Quad-view overlays (gated; napari path preserved when no controller)
    # ------------------------------------------------------------------

    def _view_controller(self):
        """Return the quad-view MicroscopeViewController, or None (napari path)."""
        parent_ui = getattr(self.parent, "parent_widget", None)
        return getattr(parent_ui, "view_controller", None)

    def _host_canvas_for(self, det):
        """Pick the canvas to host detection — the beam of ``det.fibsem_image``
        (fallback FIB), or None on the napari path."""
        controller = self._view_controller()
        if controller is None:
            return None
        fib = getattr(det, "fibsem_image", None)
        beam = fib.metadata.beam_type if fib is not None and fib.metadata is not None else None
        canvas = controller.get_canvas(beam) if beam is not None else None
        return canvas or controller.fib_canvas

    def _detach_canvas_overlays(self):
        """Remove the detection overlays from the current host canvas (e.g. when the
        host beam changes between detections)."""
        c = self._host_canvas
        if c is not None:
            if self._features_overlay is not None:
                c.exit_overlay_mode(self._features_overlay)
                c.remove_overlay(self._features_overlay)
            if self._mask_overlay is not None:
                c.remove_overlay(self._mask_overlay)
        self._features_overlay = None
        self._mask_overlay = None

    def _update_features_canvas(self, canvas):
        """Show the detection image + read-only mask + draggable feature points on
        *canvas* (quad-view equivalent of the napari ``update_features_ui``)."""
        from fibsem.ui.widgets.image_canvas import PointOverlay
        from fibsem.ui.widgets.mask_overlay import MaskOverlay

        if self._host_canvas is not None and self._host_canvas is not canvas:
            self._detach_canvas_overlays()  # host beam changed
        self._host_canvas = canvas

        # the detection image: its pixel space matches det.mask + feature.px
        if self.det.fibsem_image is not None:
            canvas.set_image(self.det.fibsem_image)

        if self._mask_overlay is None:
            self._mask_overlay = MaskOverlay()
            canvas.add_overlay(self._mask_overlay)
        self._mask_overlay.set_mask(self.det.mask)  # display-only

        if self._features_overlay is None:
            self._features_overlay = PointOverlay(
                marker="+", size=16, removable=False, add_on_right_click=False, modal=True,
            )
            canvas.add_overlay(self._features_overlay)
            self._features_overlay.point_moved.connect(self._on_canvas_feature_moved)
        self._features_overlay.set_points(
            [(f.px.x, f.px.y) for f in self.det.features],
            colors=[f.color for f in self.det.features],
            labels=[f.name for f in self.det.features],
        )
        self._features_overlay.set_visible(True)
        canvas.enter_overlay_mode(self._features_overlay, "Detection", icon="mdi:vector-point")
        canvas.set_hint("drag features to correct  ·  Continue when done")
        self.update_info()

    def _on_canvas_feature_moved(self, idx: int, x: float, y: float):
        """A feature point was dragged → update ``feature.px`` (mirrors update_point)."""
        if 0 <= idx < len(self.det.features):
            self.det.features[idx].px = Point(x=x, y=y)
            self.has_user_corrected = True
            self.update_info()

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
        """Remove the layers added by the detection widget and reshow all other layers."""
        if self._features_overlay is not None or self._mask_overlay is not None:  # quad-view
            c = self._host_canvas
            if c is not None:
                if self._features_overlay is not None:
                    c.exit_overlay_mode(self._features_overlay)  # restore Move
                c.set_hint(None)
            if self._features_overlay is not None:
                self._features_overlay.set_visible(False)
                self._features_overlay.clear_points()
            if self._mask_overlay is not None:
                self._mask_overlay.clear()
            return

        # remove feature detection layers
        if self.image_layer is not None:
            if self.image_layer in self.viewer.layers:
                self.viewer.layers.remove(self.image_layer)
            if self.mask_layer in self.viewer.layers:
                self.viewer.layers.remove(self.mask_layer)
            if self.features_layer in self.viewer.layers:
                self.viewer.layers.remove(self.features_layer)
            if self.cross_hair_layer in self.viewer.layers:
                self.viewer.layers.remove(self.cross_hair_layer)

        # reshow all other layers
        excluded_layers = ["alignment_area"]
        for layer in self.viewer.layers:
            if layer.name in excluded_layers:
                continue
            layer.visible = True

    def confirm_button_clicked(self):
        """Confirm the detected features, save the data and and remove the layers from the viewer."""

        if self._features_overlay is not None:  # quad-view: mask is display-only
            # No painting, so det.mask stays the model output — but normalise its dtype
            # to uint8 (save_feature_data_to_csv -> PIL.Image.fromarray needs it; the
            # napari path got this for free via mask_layer.data.astype(np.uint8)).
            if self.det.mask is not None:
                self.det.mask = np.asarray(self.det.mask).astype(np.uint8)
            det_utils.save_ml_feature_data(det=self.det,
                                           initial_features=self._intial_det.features)
            self.clear_layers()
            return

        # update the mask as the user may edit it
        self.det.mask = self.mask_layer.data.astype(np.uint8) # type: ignore
        
        # log the difference between initial and final detections
        # TODO: move this to outside the widget, into the same place as the non-supervised logging.
        det_utils.save_ml_feature_data(det=self.det, 
                                       initial_features=self._intial_det.features)

        # reset layers and camera
        self.clear_layers()
        self.viewer.reset_view()

    def set_detected_features(self, det_features: DetectedFeatures):
        """Set the detected features and update the UI"""
        self.det = det_features
        self._intial_det = deepcopy(det_features)
        self.has_user_corrected = False

        self.update_features_ui()

    def update_features_ui(self):
        """Update the UI with the detected features"""
        canvas = self._host_canvas_for(self.det)
        if canvas is not None:
            self._update_features_canvas(canvas)
            return

        # hide all other layers?
        for layer in self.viewer.layers:
            layer.visible = False

        self.image_layer = self.viewer.add_image( # type: ignore
            self.det.image,
            name="image",
            opacity=0.7,
            blending="additive",
        )

        # add mask to viewer
        self.mask_layer = self.viewer.add_labels(self.det.mask, 
                                                    name="mask", 
                                                    opacity=0.3,
                                                    blending="additive", 
                                                    )
        if hasattr(self.mask_layer, "colormap"):
            self.mask_layer.colormap = CLASS_COLORS
        else:
            self.mask_layer.color = CLASS_COLORS

        # add points to viewer
        data = []
        for feature in self.det.features:
            x, y = feature.px
            data.append([y, x])

        text = {
            "string": [feature.name for feature in self.det.features],
            "color": "white",
            "translation": np.array([-30, 0]),
        }

        self.features_layer = add_points_layer(
            viewer=self.viewer,
            data=data,
            name="features",
            text=text,
            size=20,
            border_width=7,
            border_width_is_relative=False,
            border_color="transparent",
            face_color=[feature.color for feature in self.det.features],
            blending="translucent",
        )

        # draw cross hairs
        self.cross_hair_layer = None
        self.draw_feature_crosshairs()

        # set points layer to select mode and active
        self.viewer.layers.selection.active = self.features_layer
        self.features_layer.mode = "select"
        
        # when the point is moved update the feature
        self.features_layer.events.data.connect(self.update_point)
        self.update_info()
            
        # set camera
        self.viewer.reset_view()

        if self.det.checkpoint:
            self.label_model.setText(f"Checkpont: {os.path.basename(self.det.checkpoint)}")
        
        napari.utils.notifications.show_info(f"Features ({', '.join([f.name for f in self.det.features])}) Detected")

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

    def update_point(self, event):
        """Update the feature when the point is moved"""
        # TODO: events have been updated so we can tell which point was deleted/moved/etc. Update this function to use that.
        logging.debug(f"{event.source.name} changed its data!")

        layer = self.viewer.layers[f"{event.source.name}"]  # type: ignore

        # get the data
        data = layer.data

        # get which point was moved
        index: List[int] = list(layer.selected_data)  
                
        if len(data) != len(self.det.features):
            # loop backwards to remove the features
            for idx in index[::-1]:
                logging.debug({"msg": "detection_point_deleted",
                               "idx": idx, "data": data[idx], 
                               "feature": self.det.features[idx].name})
                self.det.features.pop(idx)

        else: 
            for idx in index:
                logging.debug({"msg": "detection_point_moved", 
                               "idx": idx, "data": data[idx], 
                               "feature": self.det.features[idx].name})
                
                # update the feature
                self.det.features[idx].px = Point(
                    x=data[idx][1], y=data[idx][0]
                )

        self.draw_feature_crosshairs()
        self.has_user_corrected = True
        self.update_info()

    def draw_feature_crosshairs(self):
        """Draw crosshairs for each feature on the image"""

        data = self.features_layer.data

        # for each data point draw two lines from the edge of the image to the point
        line_data, line_colors = [], []
        for idx, point in enumerate(data):
            y, x = point # already flipped
            vline = [[y, 0], [y, self.det.image.data.shape[1]]]
            hline = [[0, x], [self.det.image.data.shape[0], x]]
            
            line_data += [hline, vline]
            color = self.det.features[idx].color
            line_colors += [color, color]
        try:
            self.cross_hair_layer.data = line_data
            self.cross_hair_layer.edge_color = line_colors
        except Exception as e:
            self.cross_hair_layer = self.viewer.add_shapes(
                data=line_data,
                shape_type="line",
                edge_width=3,
                edge_color=line_colors,
                name="feature_cross_hair",
                opacity=0.7,
                blending="additive",
            )    
    
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

    viewer = napari.Viewer(ndisplay=2)
    det_widget_ui = FibsemEmbeddedDetectionUI()
    det_widget_ui.set_detected_features(det)

    viewer.window.add_dock_widget(
        det_widget_ui, area="right", 
        add_vertical_stretch=False, 
        name=f"OpenFIBSEMv{fibsem.__version__} Feature Detection"
    )
    napari.run()

    det = det_widget_ui.det


if __name__ == "__main__":
    main()