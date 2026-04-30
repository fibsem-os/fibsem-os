"""CorrelationPointPickerWidget — dual-canvas point picking with synced coordinate lists.

Layout
------
┌─────────────────────┬─────────────────────┬────────────────────────────┐
│                     │                     │  ┌── FIB Fiducials (N) ───┐ │
│  ImagePointCanvas   │  FMImageDisplay     │  │  [Name] [X] [Y] [Z]   │ │
│  (FIB image)        │  Widget             │  │  FIB 1  … … …  [🗑][⠿]│ │
│                     │  (FM image)         │  └───────────────────────┘ │
│  right-click →      │                     │  ┌── FM Fiducials (N) ────┐ │
│    Add FIB          │  right-click →      │  │  FM 1   … … …  [🗑][⠿]│ │
│    Add Surface      │    Add FM           │  └───────────────────────┘ │
│                     │    Add POI          │  ┌── POI (N) ─────────────┐ │
│                     │                     │  │  POI 1  … … …  [🗑][⠿]│ │
│                     │                     │  └───────────────────────┘ │
│                     │                     │  ┌── Surface (N) ─────────┐ │
│                     │                     │  │  Surface … … … [🗑][⠿]│ │
└─────────────────────┴─────────────────────┴──┴───────────────────────┴─┘

Signals
-------
data_changed : CorrelationInputData — emitted after any coordinate edit

Usage
-----
    picker = CorrelationPointPickerWidget()
    picker.set_fib_image(fib_image)
    picker.set_fm_image(fm_image)
    picker.set_data(correlation_input_data)
    picker.data_changed.connect(my_handler)
"""
from __future__ import annotations

from typing import List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
    PointType,
    PointXYZ,
)
from fibsem.correlation.ui.widgets.coordinate_list_widget import CoordinateListWidget
from fibsem.correlation.ui.widgets.fm_image_display_widget import FMImageDisplayWidget
from fibsem.correlation.ui.widgets.image_point_canvas import ImagePointCanvas
from fibsem.fm.structures import FluorescenceImage
from fibsem.structures import FibsemImage
from fibsem.ui.widgets.custom_widgets import TitledPanel

_FIT_METHODS = ["None", "Hole", "Gaussian"]


def _show_figure_dialog(fig, title: str = "Diagnostic", parent=None) -> None:
    """Show a matplotlib figure in a non-modal QDialog."""
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.resize(600, 400)
    layout = QVBoxLayout(dlg)
    canvas = FigureCanvasQTAgg(fig)
    layout.addWidget(canvas)
    dlg.show()


class CorrelationPointPickerWidget(QWidget):
    """FIB canvas | FM canvas | coordinate lists — fully two-way synced."""

    data_changed = pyqtSignal(object)  # CorrelationInputData

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._fib_image: Optional[FibsemImage] = None
        self._fm_image: Optional[FluorescenceImage] = None

        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left: FIB image canvas (FIB + SURFACE points)
        self.fib_canvas = ImagePointCanvas(
            allowed_point_types=[PointType.FIB, PointType.SURFACE],
        )
        splitter.addWidget(self.fib_canvas)

        # Middle: FM image display (FM + POI points)
        self.fm_display = FMImageDisplayWidget(
            allowed_point_types=[PointType.FM, PointType.POI],
        )
        splitter.addWidget(self.fm_display)

        # Right: coordinate lists panel
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.setStyleSheet("background: #2b2d31; border: none;")

        right_container = QWidget()
        right_container.setStyleSheet("background: #2b2d31;")
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        # Load / Save toolbar
        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(4, 4, 4, 0)
        btn_layout.setSpacing(4)
        self._btn_load = QPushButton("Load")
        self._btn_save = QPushButton("Save")
        btn_layout.addWidget(self._btn_load)
        btn_layout.addWidget(self._btn_save)
        btn_layout.addStretch(1)
        right_layout.addWidget(btn_row)

        # FIB fiducials
        self._fib_list = CoordinateListWidget(point_type=PointType.FIB)
        self._fib_panel = TitledPanel("FIB Fiducials", collapsible=True)
        self._fib_panel.set_content(self._fib_list)
        self._fib_count_label = QLabel("(0)")
        self._fib_count_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self._fib_panel.add_header_widget(self._fib_count_label)
        right_layout.addWidget(self._fib_panel)

        # FM fiducials
        self._fm_list = CoordinateListWidget(point_type=PointType.FM)
        self._fm_panel = TitledPanel("FM Fiducials", collapsible=True)
        self._fm_panel.set_content(self._fm_list)
        self._fm_count_label = QLabel("(0)")
        self._fm_count_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self._fm_panel.add_header_widget(self._fm_count_label)
        right_layout.addWidget(self._fm_panel)

        # POI
        self._poi_list = CoordinateListWidget(point_type=PointType.POI)
        self._poi_panel = TitledPanel("POI", collapsible=True)
        self._poi_panel.set_content(self._poi_list)
        self._poi_count_label = QLabel("(0)")
        self._poi_count_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self._poi_panel.add_header_widget(self._poi_count_label)
        right_layout.addWidget(self._poi_panel)

        # Surface (max 1)
        self._surface_list = CoordinateListWidget(point_type=PointType.SURFACE)
        self._surface_panel = TitledPanel("Surface", collapsible=True)
        self._surface_panel.set_content(self._surface_list)
        self._surface_count_label = QLabel("(0)")
        self._surface_count_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self._surface_panel.add_header_widget(self._surface_count_label)
        right_layout.addWidget(self._surface_panel)

        # Fit Settings
        fit_body = QWidget()
        fit_form = QFormLayout(fit_body)
        fit_form.setContentsMargins(8, 4, 8, 4)
        fit_form.setSpacing(4)

        self._fib_method_combo = QComboBox()
        self._fib_method_combo.addItems(_FIT_METHODS)
        self._fib_method_combo.setCurrentText("Hole")
        fit_form.addRow("FIB method:", self._fib_method_combo)

        self._fm_fid_method_combo = QComboBox()
        self._fm_fid_method_combo.addItems(_FIT_METHODS)
        self._fm_fid_method_combo.setCurrentText("None")
        fit_form.addRow("FM Fid. method:", self._fm_fid_method_combo)

        self._fm_poi_method_combo = QComboBox()
        self._fm_poi_method_combo.addItems(_FIT_METHODS)
        self._fm_poi_method_combo.setCurrentText("Gaussian")
        fit_form.addRow("FM POI method:", self._fm_poi_method_combo)

        self._fm_fid_ch_combo = QComboBox()
        fit_form.addRow("FM Fid. channel:", self._fm_fid_ch_combo)

        self._fm_poi_ch_combo = QComboBox()
        fit_form.addRow("FM POI channel:", self._fm_poi_ch_combo)

        self._show_diag_check = QCheckBox()
        fit_form.addRow("Show diagnostic:", self._show_diag_check)

        self._fit_panel = TitledPanel("Fit Settings", collapsible=True)
        self._fit_panel.set_content(fit_body)
        right_layout.addWidget(self._fit_panel)

        right_layout.addStretch(1)
        right_scroll.setWidget(right_container)
        splitter.addWidget(right_scroll)

        splitter.setSizes([500, 500, 300])

    def _connect_signals(self) -> None:
        # Canvas → list
        self.fib_canvas.point_selected.connect(self._on_fib_canvas_selected)
        self.fib_canvas.point_moved.connect(self._on_fib_canvas_moved)
        self.fib_canvas.point_removed.connect(self._on_fib_canvas_removed)
        self.fib_canvas.point_add_requested.connect(self._on_fib_add_requested)

        self.fm_display.point_selected.connect(self._on_fm_canvas_selected)
        self.fm_display.point_moved.connect(self._on_fm_canvas_moved)
        self.fm_display.point_removed.connect(self._on_fm_canvas_removed)
        self.fm_display.point_add_requested.connect(self._on_fm_add_requested)

        # List → canvas
        self._fib_list.coordinate_selected.connect(self._on_fib_list_selected)
        self._fib_list.coordinate_changed.connect(self._on_fib_list_changed)
        self._fib_list.coordinate_removed.connect(self._on_fib_list_removed)
        self._fib_list.order_changed.connect(lambda _: (self._refresh_fib_canvas(), self.data_changed.emit(self.data)))
        self._fib_list.refit_requested.connect(self._on_refit_requested)

        self._fm_list.coordinate_selected.connect(self._on_fm_list_selected)
        self._fm_list.coordinate_changed.connect(self._on_fm_list_changed)
        self._fm_list.coordinate_removed.connect(self._on_fm_list_removed)
        self._fm_list.order_changed.connect(lambda _: (self._refresh_fm_canvas(), self.data_changed.emit(self.data)))
        self._fm_list.refit_requested.connect(self._on_refit_requested)

        self._poi_list.coordinate_selected.connect(self._on_poi_list_selected)
        self._poi_list.coordinate_changed.connect(self._on_poi_list_changed)
        self._poi_list.coordinate_removed.connect(self._on_poi_list_removed)
        self._poi_list.order_changed.connect(lambda _: (self._refresh_fm_canvas(), self.data_changed.emit(self.data)))
        self._poi_list.refit_requested.connect(self._on_refit_requested)

        self._surface_list.coordinate_selected.connect(self._on_surface_list_selected)
        self._surface_list.coordinate_changed.connect(self._on_surface_list_changed)
        self._surface_list.coordinate_removed.connect(self._on_surface_list_removed)
        self._surface_list.order_changed.connect(lambda _: (self._refresh_fib_canvas(), self.data_changed.emit(self.data)))
        self._surface_list.refit_requested.connect(self._on_refit_requested)

        self._btn_load.clicked.connect(self._on_load)
        self._btn_save.clicked.connect(self._on_save)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_fib_image(self, fib_image: FibsemImage) -> None:
        """Display the FIB image and constrain FIB coordinate spinboxes to its shape."""
        self._fib_image = fib_image
        import numpy as np
        img = fib_image.filtered_data
        arr = img.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / arr.max()
        self.fib_canvas.set_image(arr)
        h, w = img.shape[:2]
        self._fib_list.set_axis_maxima(x_max=w - 1, y_max=h - 1)
        self._surface_list.set_axis_maxima(x_max=w - 1, y_max=h - 1)

    def set_fm_image(self, fm_image: FluorescenceImage) -> None:
        """Display the FM image and constrain FM/POI coordinate spinboxes."""
        self._fm_image = fm_image
        self.fm_display.set_fm_image(fm_image)
        _, n_z, h, w = fm_image.data.shape
        self._fm_list.set_axis_maxima(x_max=w - 1, y_max=h - 1, z_max=n_z - 1)
        self._poi_list.set_axis_maxima(x_max=w - 1, y_max=h - 1, z_max=n_z - 1)
        self._rebuild_channel_combos()

    def set_data(self, data: CorrelationInputData) -> None:
        """Populate all lists and refresh both canvases."""
        fib_coords  = list(data.fib_coordinates)
        fm_coords   = list(data.fm_coordinates)
        poi_coords  = list(data.poi_coordinates)
        surf_coords = [data.surface_coordinate] if data.surface_coordinate is not None else []

        # Populate canvases first so that coordinate_selected (emitted by the
        # list setter's auto-select) can immediately highlight the canvas point.
        self.fib_canvas.set_coordinates(fib_coords + surf_coords)
        self.fm_display.set_coordinates(fm_coords + poi_coords)

        self._fib_list.coordinates     = fib_coords
        self._fm_list.coordinates      = fm_coords
        self._poi_list.coordinates     = poi_coords
        self._surface_list.coordinates = surf_coords
        self._update_headers()

    def load_data(self, path: str) -> None:
        """Load coordinates from a JSON file, preserving current images."""
        loaded = CorrelationInputData.load(path)
        loaded.fib_image = self._fib_image
        loaded.fm_image = self._fm_image
        self.set_data(loaded)
        self.data_changed.emit(self.data)

    def save_data(self, path: str) -> None:
        """Save current coordinates to a JSON file."""
        self.data.save(path)

    @property
    def data(self) -> CorrelationInputData:
        surface = self._surface_list.coordinates
        return CorrelationInputData(
            fib_image=self._fib_image,
            fm_image=self._fm_image,
            fib_coordinates=self._fib_list.coordinates,
            fm_coordinates=self._fm_list.coordinates,
            poi_coordinates=self._poi_list.coordinates,
            surface_coordinate=surface[0] if surface else None,
        )

    # ------------------------------------------------------------------
    # Canvas refresh helpers
    # ------------------------------------------------------------------

    def _refresh_fib_canvas(self) -> None:
        self.fib_canvas.set_coordinates(
            self._fib_list.coordinates + self._surface_list.coordinates
        )

    def _refresh_fm_canvas(self) -> None:
        self.fm_display.set_coordinates(
            self._fm_list.coordinates + self._poi_list.coordinates
        )

    def _update_headers(self) -> None:
        self._fib_count_label.setText(f"({len(self._fib_list.coordinates)})")
        self._fm_count_label.setText(f"({len(self._fm_list.coordinates)})")
        self._poi_count_label.setText(f"({len(self._poi_list.coordinates)})")
        self._surface_count_label.setText(f"({len(self._surface_list.coordinates)})")

    # ------------------------------------------------------------------
    # Channel combo helpers
    # ------------------------------------------------------------------

    def _rebuild_channel_combos(self) -> None:
        for cb in (self._fm_fid_ch_combo, self._fm_poi_ch_combo):
            cb.clear()
            if self._fm_image is None:
                continue
            channels = self._fm_image.metadata.channels or []
            for i, ch in enumerate(channels):
                cb.addItem(ch.name or f"CH {i}")
            if cb.count() == 0:
                n = self._fm_image.data.shape[0]
                for i in range(n):
                    cb.addItem(f"CH {i}")

    # ------------------------------------------------------------------
    # Canvas → list slots
    # ------------------------------------------------------------------

    def _on_fib_canvas_selected(self, coord: Coordinate) -> None:
        if coord.point_type == PointType.SURFACE:
            self._surface_list.select_coordinate_silent(coord)
            self._fib_list.select_coordinate_silent(None)
        else:
            self._fib_list.select_coordinate_silent(coord)
            self._surface_list.select_coordinate_silent(None)
        self._fm_list.select_coordinate_silent(None)
        self._poi_list.select_coordinate_silent(None)
        self.fm_display.set_selected(None)

    def _on_fm_canvas_selected(self, coord: Coordinate) -> None:
        self.fib_canvas.set_selected(None)
        self._fib_list.select_coordinate_silent(None)
        self._surface_list.select_coordinate_silent(None)
        if coord.point_type == PointType.FM:
            self._fm_list.select_coordinate_silent(coord)
            self._poi_list.select_coordinate_silent(None)
        else:
            self._fm_list.select_coordinate_silent(None)
            self._poi_list.select_coordinate_silent(coord)

    def _on_fib_canvas_moved(self, coord: Coordinate) -> None:
        if coord.point_type == PointType.SURFACE:
            self._surface_list.refresh_coordinate(coord)
        else:
            self._fib_list.refresh_coordinate(coord)
        self.data_changed.emit(self.data)

    def _on_fm_canvas_moved(self, coord: Coordinate) -> None:
        if coord.point_type == PointType.FM:
            self._fm_list.refresh_coordinate(coord)
        else:
            self._poi_list.refresh_coordinate(coord)
        self.data_changed.emit(self.data)

    def _on_fib_canvas_removed(self, coord: Coordinate) -> None:
        if coord.point_type == PointType.SURFACE:
            self._surface_list.coordinates = [
                c for c in self._surface_list.coordinates if c is not coord
            ]
        else:
            self._fib_list.coordinates = [
                c for c in self._fib_list.coordinates if c is not coord
            ]
        self._refresh_fib_canvas()
        self._update_headers()
        self.data_changed.emit(self.data)

    def _on_fm_canvas_removed(self, coord: Coordinate) -> None:
        if coord.point_type == PointType.FM:
            self._fm_list.coordinates = [
                c for c in self._fm_list.coordinates if c is not coord
            ]
        else:
            self._poi_list.coordinates = [
                c for c in self._poi_list.coordinates if c is not coord
            ]
        self._refresh_fm_canvas()
        self._update_headers()
        self.data_changed.emit(self.data)

    def _on_fib_add_requested(self, x: float, y: float, pt: PointType) -> None:
        coord = Coordinate(PointXYZ(x, y, 0.0), pt)
        if pt == PointType.SURFACE:
            # Surface is max 1 — replace existing
            self._surface_list.coordinates = [coord]
        else:
            self._fib_list.add_coordinate(coord)
        self._refresh_fib_canvas()
        self._update_headers()
        self.data_changed.emit(self.data)

    def _on_fm_add_requested(self, x: float, y: float, pt: PointType) -> None:
        coord = Coordinate(PointXYZ(x, y, float(self.fm_display.current_z)), pt)
        if pt == PointType.FM:
            self._fm_list.add_coordinate(coord)
        else:
            self._poi_list.add_coordinate(coord)
        self._refresh_fm_canvas()
        self._update_headers()
        self.data_changed.emit(self.data)

    # ------------------------------------------------------------------
    # List → canvas slots
    # ------------------------------------------------------------------

    def _on_fib_list_selected(self, coord: Coordinate) -> None:
        self.fib_canvas.set_selected(coord)
        self.fm_display.set_selected(None)
        self._fm_list.select_coordinate_silent(None)
        self._poi_list.select_coordinate_silent(None)
        self._surface_list.select_coordinate_silent(None)

    def _on_fm_list_selected(self, coord: Coordinate) -> None:
        self.fm_display.set_selected(coord)
        self.fib_canvas.set_selected(None)
        self._fib_list.select_coordinate_silent(None)
        self._poi_list.select_coordinate_silent(None)
        self._surface_list.select_coordinate_silent(None)

    def _on_poi_list_selected(self, coord: Coordinate) -> None:
        self.fm_display.set_selected(coord)
        self.fib_canvas.set_selected(None)
        self._fib_list.select_coordinate_silent(None)
        self._fm_list.select_coordinate_silent(None)
        self._surface_list.select_coordinate_silent(None)

    def _on_surface_list_selected(self, coord: Coordinate) -> None:
        self.fib_canvas.set_selected(coord)
        self.fm_display.set_selected(None)
        self._fib_list.select_coordinate_silent(None)
        self._fm_list.select_coordinate_silent(None)
        self._poi_list.select_coordinate_silent(None)

    def _on_fib_list_changed(self, coord: Coordinate, _field: str, _val: float) -> None:
        self.fib_canvas.refresh_coordinate(coord)
        self.data_changed.emit(self.data)

    def _on_fm_list_changed(self, coord: Coordinate, _field: str, _val: float) -> None:
        self.fm_display.refresh_coordinate(coord)
        self.data_changed.emit(self.data)

    def _on_poi_list_changed(self, coord: Coordinate, _field: str, _val: float) -> None:
        self.fm_display.refresh_coordinate(coord)
        self.data_changed.emit(self.data)

    def _on_surface_list_changed(self, coord: Coordinate, _field: str, _val: float) -> None:
        self.fib_canvas.refresh_coordinate(coord)
        self.data_changed.emit(self.data)

    def _on_fib_list_removed(self, _coord: Coordinate) -> None:
        self._refresh_fib_canvas()
        self._update_headers()
        self.data_changed.emit(self.data)

    def _on_fm_list_removed(self, _coord: Coordinate) -> None:
        self._refresh_fm_canvas()
        self._update_headers()
        self.data_changed.emit(self.data)

    def _on_poi_list_removed(self, _coord: Coordinate) -> None:
        self._refresh_fm_canvas()
        self._update_headers()
        self.data_changed.emit(self.data)

    def _on_surface_list_removed(self, _coord: Coordinate) -> None:
        self._refresh_fib_canvas()
        self._update_headers()
        self.data_changed.emit(self.data)

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------

    def _on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Coordinates", "", "JSON (*.json)"
        )
        if path:
            self.load_data(path)

    def _on_save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Coordinates", "", "JSON (*.json)"
        )
        if path:
            self.save_data(path)

    # ------------------------------------------------------------------
    # Refit
    # ------------------------------------------------------------------

    def _on_refit_requested(self, coord: Coordinate) -> None:
        """Run auto-centroid fitting on the selected coordinate."""
        from fibsem.correlation.util import (
            hole_fitting_FIB,
            hole_fitting_reflection,
            target_fitting_fluorescence,
        )

        x, y, z = coord.point.x, coord.point.y, coord.point.z
        fig = None

        try:
            if coord.point_type in (PointType.FIB, PointType.SURFACE):
                method = self._fib_method_combo.currentText()
                if method == "Hole" and self._fib_image is not None:
                    xr, yr, fig = hole_fitting_FIB(
                        self._fib_image.filtered_data, int(x), int(y)
                    )
                    coord.point.x, coord.point.y = float(xr), float(yr)

            elif coord.point_type in (PointType.FM, PointType.POI):
                is_fid = coord.point_type == PointType.FM
                method = (
                    self._fm_fid_method_combo if is_fid else self._fm_poi_method_combo
                ).currentText()
                ch = (
                    self._fm_fid_ch_combo if is_fid else self._fm_poi_ch_combo
                ).currentIndex()
                if method != "None" and self._fm_image is not None and ch >= 0:
                    img = self._fm_image.data[ch]  # (Z, Y, X)
                    if method == "Hole":
                        xr, yr, zr, fig = hole_fitting_reflection(
                            img, int(x), int(y), z=int(z), cutout=2
                        )
                    elif method == "Gaussian":
                        xr, yr, zr, fig = target_fitting_fluorescence(
                            img, int(x), int(y), int(z), cutout=5
                        )
                    coord.point.x = float(xr)
                    coord.point.y = float(yr)
                    coord.point.z = float(zr)

        except Exception as exc:
            QMessageBox.warning(self, "Refit failed", str(exc))
            return

        # Refresh list row and canvas
        if coord.point_type in (PointType.FIB, PointType.SURFACE):
            list_w = (
                self._fib_list if coord.point_type == PointType.FIB
                else self._surface_list
            )
            list_w.refresh_coordinate(coord)
            self.fib_canvas.refresh_coordinate(coord)
        else:
            list_w = (
                self._fm_list if coord.point_type == PointType.FM else self._poi_list
            )
            list_w.refresh_coordinate(coord)
            self.fm_display.refresh_coordinate(coord)

        self.data_changed.emit(self.data)

        if fig is not None and self._show_diag_check.isChecked():
            _show_figure_dialog(fig, title="Refit Diagnostic", parent=self)
