"""CorrelationResultWidget — displays a CorrelationResult in collapsible TitledPanels.

Four panels stacked vertically:
  - FIB Result Overlay: ImagePointCanvas showing FIB image with reprojected points
  - Summary:            scale, RMS error, mean absolute error
  - Transformation:     euler rotation, translation
  - Per-Marker Error:   QTableWidget with dx/dy per marker

Usage
-----
    widget = CorrelationResultWidget()
    widget.set_result(result)                       # summary/table only
    widget.set_result(result, input_data=data)      # also shows overlay canvas
    widget.clear()                                  # reset to "—"
"""
from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFormLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
    CorrelationResult,
    PointType,
    PointXYZ,
)
from fibsem.correlation.ui.widgets.image_point_canvas import ImagePointCanvas
from fibsem.ui.widgets.custom_widgets import TitledPanel

_VALUE_STYLE = "color: #e0e0e0; font-size: 12px;"


def _value_label(text: str = "—") -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(_VALUE_STYLE)
    return lbl


class CorrelationResultWidget(QWidget):
    """Displays a CorrelationResult — overlay canvas, summary, transformation, per-marker error."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._overlay_first = True
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # --- FIB Result Overlay panel (read-only canvas) ---
        self._overlay_canvas = ImagePointCanvas(allowed_point_types=[])
        self._overlay_panel = TitledPanel(
            "FIB Result Overlay", collapsible=True, content=self._overlay_canvas
        )
        self._overlay_panel.setVisible(False)
        layout.addWidget(self._overlay_panel)

        # --- Summary panel ---
        summary_body = QWidget()
        summary_form = QFormLayout(summary_body)
        summary_form.setContentsMargins(8, 4, 8, 4)
        summary_form.setSpacing(4)

        self._lbl_scale = _value_label()
        self._lbl_rms = _value_label()
        self._lbl_mae = _value_label()

        summary_form.addRow("Scale:", self._lbl_scale)
        summary_form.addRow("RMS Error:", self._lbl_rms)
        summary_form.addRow("Mean Abs Error:", self._lbl_mae)

        layout.addWidget(TitledPanel("Summary", content=summary_body))

        # --- Transformation panel ---
        transf_body = QWidget()
        transf_form = QFormLayout(transf_body)
        transf_form.setContentsMargins(8, 4, 8, 4)
        transf_form.setSpacing(4)

        self._lbl_rotation = _value_label()
        self._lbl_translation = _value_label()

        transf_form.addRow("Rotation:", self._lbl_rotation)
        transf_form.addRow("Translation:", self._lbl_translation)

        layout.addWidget(TitledPanel("Transformation", content=transf_body))

        # --- Per-marker error panel ---
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Marker", "dx (px)", "dy (px)"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._table.setMinimumHeight(120)

        layout.addWidget(TitledPanel("Per-Marker Error", content=self._table))
        layout.addStretch(1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_result(
        self,
        result: CorrelationResult,
        input_data: Optional[CorrelationInputData] = None,
    ) -> None:
        """Populate all fields from a CorrelationResult.

        If *input_data* is provided and contains a FIB image, the overlay canvas
        is shown with original FIB fiducials (lime), reprojected FM fiducials
        (cyan), and reprojected POI (magenta).
        """
        self._lbl_scale.setText(f"{result.scale:.2f}")
        self._lbl_rms.setText(f"{result.rms_error:.2f} px")

        if result.mean_absolute_error:
            mae_str = ", ".join(f"{v:.2f}" for v in result.mean_absolute_error) + " px"
        else:
            mae_str = "—"
        self._lbl_mae.setText(mae_str)

        if result.rotation_eulers:
            rot_str = "°, ".join(f"{v:.2f}" for v in result.rotation_eulers) + "°"
        else:
            rot_str = "—"
        self._lbl_rotation.setText(rot_str)

        if result.translation and len(result.translation) >= 2:
            trans_str = ", ".join(f"{v:.1f}" for v in result.translation[:2])
        else:
            trans_str = "—"
        self._lbl_translation.setText(trans_str)

        # Rebuild delta_2d table
        markers = result.delta_2d
        self._table.setRowCount(len(markers))
        self._table.setMinimumHeight(max(120, min(len(markers) * 28 + 28, 300)))
        for i, pt in enumerate(markers):
            self._table.setItem(i, 0, self._ro_item(f"M{i + 1}"))
            self._table.setItem(i, 1, self._ro_item(f"{pt.x:.2f}"))
            self._table.setItem(i, 2, self._ro_item(f"{pt.y:.2f}"))

        self._update_overlay(result, input_data)

    def clear(self) -> None:
        """Reset all fields to placeholder dashes."""
        for lbl in (self._lbl_scale, self._lbl_rms, self._lbl_mae,
                    self._lbl_rotation, self._lbl_translation):
            lbl.setText("—")
        self._table.setRowCount(0)
        self._overlay_panel.setVisible(False)
        self._overlay_first = True

    # ------------------------------------------------------------------
    # Overlay
    # ------------------------------------------------------------------

    def _update_overlay(
        self,
        result: CorrelationResult,
        input_data: Optional[CorrelationInputData],
    ) -> None:
        if input_data is None or input_data.fib_image is None:
            self._overlay_panel.setVisible(False)
            return

        self._overlay_panel.setVisible(True)
        img = input_data.fib_image.filtered_data

        if self._overlay_first:
            self._overlay_canvas.set_image(img)
            self._overlay_first = False
        else:
            self._overlay_canvas.update_display(img)

        # Original FIB fiducials — lime
        orig_fib = list(input_data.fib_coordinates)

        # Reprojected 3D fiducials → shown as FM type (cyan)
        reproj = [
            Coordinate(PointXYZ(r.x, r.y, r.z), PointType.FM)
            for r in result.reprojected_3d
        ]

        # Reprojected POI → magenta
        poi_overlay = [
            Coordinate(PointXYZ(p.image_px.x, p.image_px.y, 0.0), PointType.POI)
            for p in result.poi
        ]

        self._overlay_canvas.set_coordinates(orig_fib + reproj + poi_overlay)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ro_item(text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        return item
