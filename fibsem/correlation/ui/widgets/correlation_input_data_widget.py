"""CorrelationInputDataWidget — four CoordinateListWidgets in collapsible TitledPanels.

One panel per PointType (FIB, FM, POI, SURFACE), stacked vertically.
Accepts a CorrelationInputData and reconstructs it on demand.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.correlation.structures import Coordinate, CorrelationInputData, PointType
from fibsem.correlation.ui.widgets.coordinate_list_widget import CoordinateListWidget
from fibsem.ui.widgets.custom_widgets import TitledPanel

_COUNT_LABEL_STYLE_OK   = "color: #aaa; font-size: 11px; background: transparent; padding: 0 4px;"
_COUNT_LABEL_STYLE_WARN = "color: #e07b39; font-size: 11px; font-weight: bold; background: transparent; padding: 0 4px;"

_MIN_COUNTS: Dict[PointType, int] = {
    PointType.FIB: 4,
    PointType.FM: 4,
    PointType.POI: 1,
    PointType.SURFACE: 1,
}


class CorrelationInputDataWidget(QWidget):
    """Four CoordinateListWidgets stacked in collapsible TitledPanels.

    Signals
    -------
    data_changed : CorrelationInputData — emitted on any coordinate change
    """

    data_changed = pyqtSignal(object)  # CorrelationInputData

    def __init__(
        self,
        data: Optional[CorrelationInputData] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._data_meta: dict = {}
        self._setup_ui()
        self._connect_signals()
        if data is not None:
            self.data = data

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Save / Load toolbar
        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(4)
        self._btn_load = QPushButton("Load")
        self._btn_save = QPushButton("Save")
        btn_layout.addWidget(self._btn_load)
        btn_layout.addWidget(self._btn_save)
        btn_layout.addStretch(1)
        layout.addWidget(btn_row)

        self._fib_list = CoordinateListWidget()
        self._fm_list = CoordinateListWidget()
        self._poi_list = CoordinateListWidget()
        self._surface_list = CoordinateListWidget()

        # Map PointType → (CoordinateListWidget, count QLabel)
        self._panels: Dict[PointType, Tuple[CoordinateListWidget, QLabel]] = {}

        for list_widget, title, pt in (
            (self._fib_list,     "FIB Coordinates",     PointType.FIB),
            (self._fm_list,      "FM Coordinates",      PointType.FM),
            (self._poi_list,     "POI Coordinates",     PointType.POI),
            (self._surface_list, "Surface Coordinate (max 1)", PointType.SURFACE),
        ):
            count_label = QLabel("0")
            count_label.setStyleSheet(_COUNT_LABEL_STYLE_WARN)  # 0 < minimum initially
            panel = TitledPanel(title, content=list_widget)
            panel.add_header_widget(count_label)
            self._panels[pt] = (list_widget, count_label)
            layout.addWidget(panel)

        layout.addStretch(1)

    def _connect_signals(self) -> None:
        for list_widget, _ in self._panels.values():
            list_widget.coordinate_changed.connect(self._on_any_change)
            list_widget.coordinate_removed.connect(self._on_any_change)
            list_widget.order_changed.connect(self._on_any_change)
            list_widget.refit_requested.connect(self._on_any_change)

        self._btn_load.clicked.connect(self._on_load)
        self._btn_save.clicked.connect(self._on_save)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def data(self) -> CorrelationInputData:
        """Reconstruct CorrelationInputData from current sub-widget state."""
        surface = self._surface_list.coordinates
        return CorrelationInputData(
            fib_coordinates=self._fib_list.coordinates,
            fm_coordinates=self._fm_list.coordinates,
            poi_coordinates=self._poi_list.coordinates,
            surface_coordinate=surface[0] if surface else None,
            **self._data_meta,
        )

    @data.setter
    def data(self, value: CorrelationInputData) -> None:
        self._fib_list.coordinates = list(value.fib_coordinates)
        self._fm_list.coordinates = list(value.fm_coordinates)
        self._poi_list.coordinates = list(value.poi_coordinates)
        surf = [value.surface_coordinate] if value.surface_coordinate is not None else []
        self._surface_list.coordinates = surf

        # Preserve non-coordinate fields for roundtrip fidelity
        self._data_meta = {
            "fib_image": value.fib_image,
            "fm_image": value.fm_image,
            "method": value.method,
        }

        # FIB shape: (Y, X) — z unconstrained for 2D image
        if value.fib_image_shape is not None:
            y, x = value.fib_image_shape
            self._fib_list.set_axis_maxima(x_max=x - 1, y_max=y - 1, z_max=None)

        # FM shape: (C, Z, Y, X)
        if value.fm_image_shape is not None:
            _c, z, y, x = value.fm_image_shape
            self._fm_list.set_axis_maxima(x_max=x - 1, y_max=y - 1, z_max=z - 1)

        # data_changed is NOT emitted here: setting coordinates directly does not
        # fire sub-list signals, so callers (e.g. _on_load) must emit it themselves.
        self._update_counts()

    def is_valid(self) -> bool:
        """Return True if all minimums are met and FIB/FM coordinate counts match."""
        counts = {pt: len(lw.coordinates) for pt, (lw, _) in self._panels.items()}
        counts_ok = all(counts[pt] >= _MIN_COUNTS[pt] for pt in _MIN_COUNTS)
        return counts_ok and counts[PointType.FIB] == counts[PointType.FM]

    def add_coordinate(self, coord: Coordinate) -> None:
        """Append a coordinate to the appropriate sub-list.

        Surface coordinates replace any existing one (max 1 allowed).
        """
        list_widget, _ = self._panels[coord.point_type]
        if coord.point_type == PointType.SURFACE:
            list_widget.coordinates = [coord]
        else:
            coords = list_widget.coordinates
            coords.append(coord)
            list_widget.coordinates = coords
        self._update_counts()
        self.data_changed.emit(self.data)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_counts(self) -> None:
        counts = {pt: len(lw.coordinates) for pt, (lw, _) in self._panels.items()}
        fib_fm_mismatch = counts[PointType.FIB] != counts[PointType.FM]
        for pt, (_, count_label) in self._panels.items():
            n = counts[pt]
            count_label.setText(str(n))
            warn = n < _MIN_COUNTS[pt] or (pt in (PointType.FIB, PointType.FM) and fib_fm_mismatch)
            count_label.setStyleSheet(_COUNT_LABEL_STYLE_WARN if warn else _COUNT_LABEL_STYLE_OK)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_any_change(self, *_args) -> None:
        self._update_counts()
        self.data_changed.emit(self.data)

    def _on_load(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Load Coordinates", "", "JSON (*.json)")
        if filename:
            self.data = CorrelationInputData.load(filename)
            self.data_changed.emit(self.data)

    def _on_save(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(self, "Save Coordinates", "", "JSON (*.json)")
        if filename:
            self.data.save(filename)
