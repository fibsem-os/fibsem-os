"""CorrelationTabWidget — FIB canvas | FM canvas | tabbed controls.

Layout
------
┌─────────────────────┬─────────────────────┬────────────────────────────┐
│                     │                     │  ┌─[Images][Coords][Results]┐│
│  ImagePointCanvas   │  FMImageDisplay     │  │                         ││
│  (FIB image)        │  Widget             │  │  (tab content)          ││
│  right-click →      │  (FM image)         │  │                         ││
│    Add FIB          │  right-click →      │  │                         ││
│    Add Surface      │    Add FM           │  │                         ││
│                     │    Add POI          │  │                         ││
│                     │                     │  └─────────────────────────┘│
└─────────────────────┴─────────────────────┴────────────────────────────┘

Tabs
----
  Images      — browse / load FIB and FM images
  Coordinates — coordinate lists (FIB, FM, POI, Surface) + fit settings + load/save
  Results     — run correlation + CorrelationResultWidget overlay
  RI          — refractive-index depth correction table

Signals
-------
data_changed   : CorrelationInputData — after any coordinate edit
result_changed : CorrelationResult    — after a successful correlation run
"""

from __future__ import annotations

import copy
import logging
import os
import time
from typing import List, Optional

logging.basicConfig(level=logging.INFO)

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QAction,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from fibsem.constants import DATETIME_FILE
from fibsem.correlation.correlation_v2 import run_correlation_from_data
from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
    CorrelationPointOfInterest,
    CorrelationResult,
    PointType,
    PointXYZ,
)
from fibsem.correlation.ui.widgets.coordinate_list_widget import CoordinateListWidget
from fibsem.ui import stylesheets
from fibsem.correlation.ui.widgets.fm_image_display_widget import FMImageDisplayWidget
from fibsem.correlation.ui.widgets.image_point_canvas import ImagePointCanvas
from fibsem.fm.structures import FluorescenceImage
from fibsem.structures import FibsemImage
from fibsem.ui.widgets.custom_widgets import TitledPanel
from fibsem.correlation.ui.widgets.refractive_index_widget import RefractiveIndexWidget

_FIT_METHODS = ["None", "Hole", "Gaussian"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ro_item(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(Qt.ItemFlag.ItemIsEnabled)
    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
    return item


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


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


class _CorrelationWorker(QThread):
    result_ready = pyqtSignal(object)  # CorrelationResult
    errored = pyqtSignal(str)

    def __init__(
        self, input_data: CorrelationInputData, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._data = input_data

    def run(self) -> None:
        try:
            result = run_correlation_from_data(self._data)
            self.result_ready.emit(result)
        except Exception as exc:
            self.errored.emit(str(exc))


# ---------------------------------------------------------------------------
# Tab 0 — Images
# ---------------------------------------------------------------------------


class _ImagesTab(QWidget):
    """Browse / load FIB and FM image files."""

    fib_image_changed = pyqtSignal(object)  # FibsemImage
    fm_image_changed = pyqtSignal(object)  # FluorescenceImage
    project_dir_changed = pyqtSignal(str)  # directory path

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._fib_image: Optional[FibsemImage] = None
        self._fm_image: Optional[FluorescenceImage] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ---- Project directory section ----
        proj_body = QWidget()
        proj_layout = QHBoxLayout(proj_body)
        proj_layout.setContentsMargins(8, 4, 8, 4)
        proj_layout.setSpacing(4)
        self._btn_proj = QPushButton("Browse…")
        self._btn_proj.setFixedWidth(80)
        self._btn_proj.clicked.connect(self._browse_project_dir)
        self._proj_path = QLineEdit()
        self._proj_path.setReadOnly(True)
        self._proj_path.setPlaceholderText("No project directory set")
        proj_layout.addWidget(self._btn_proj)
        proj_layout.addWidget(self._proj_path, stretch=1)
        layout.addWidget(TitledPanel("Project", content=proj_body, collapsible=False))

        # ---- FIB section ----
        fib_body = QWidget()
        fib_layout = QVBoxLayout(fib_body)
        fib_layout.setContentsMargins(8, 4, 8, 4)
        fib_layout.setSpacing(4)

        fib_browse_row = QWidget()
        fib_browse_layout = QHBoxLayout(fib_browse_row)
        fib_browse_layout.setContentsMargins(0, 0, 0, 0)
        fib_browse_layout.setSpacing(4)
        self._btn_fib = QPushButton("Browse…")
        self._btn_fib.setFixedWidth(80)
        self._btn_fib.clicked.connect(self._browse_fib)
        self._fib_path = QLineEdit()
        self._fib_path.setReadOnly(True)
        self._fib_path.setPlaceholderText("No file loaded")
        fib_browse_layout.addWidget(self._btn_fib)
        fib_browse_layout.addWidget(self._fib_path, stretch=1)
        fib_layout.addWidget(fib_browse_row)

        fib_form = QFormLayout()
        fib_form.setContentsMargins(0, 0, 0, 0)
        fib_form.setSpacing(2)
        self._lbl_fib_shape = QLabel("—")
        self._lbl_fib_shape.setStyleSheet("color: #e0e0e0; font-size: 11px;")
        self._lbl_fib_px = QLabel("—")
        self._lbl_fib_px.setStyleSheet("color: #e0e0e0; font-size: 11px;")
        fib_form.addRow("Shape:", self._lbl_fib_shape)
        fib_form.addRow("Pixel size:", self._lbl_fib_px)
        fib_layout.addLayout(fib_form)

        layout.addWidget(TitledPanel("FIB Image", content=fib_body, collapsible=False))

        # ---- FM section ----
        fm_body = QWidget()
        fm_layout = QVBoxLayout(fm_body)
        fm_layout.setContentsMargins(8, 4, 8, 4)
        fm_layout.setSpacing(4)

        fm_browse_row = QWidget()
        fm_browse_layout = QHBoxLayout(fm_browse_row)
        fm_browse_layout.setContentsMargins(0, 0, 0, 0)
        fm_browse_layout.setSpacing(4)
        self._btn_fm = QPushButton("Browse…")
        self._btn_fm.setFixedWidth(80)
        self._btn_fm.clicked.connect(self._browse_fm)
        self._fm_path = QLineEdit()
        self._fm_path.setReadOnly(True)
        self._fm_path.setPlaceholderText("No file loaded")
        fm_browse_layout.addWidget(self._btn_fm)
        fm_browse_layout.addWidget(self._fm_path, stretch=1)
        fm_layout.addWidget(fm_browse_row)

        fm_form = QFormLayout()
        fm_form.setContentsMargins(0, 0, 0, 0)
        fm_form.setSpacing(2)
        self._lbl_fm_shape = QLabel("—")
        self._lbl_fm_shape.setStyleSheet("color: #e0e0e0; font-size: 11px;")
        self._lbl_fm_ch = QLabel("—")
        self._lbl_fm_ch.setStyleSheet("color: #e0e0e0; font-size: 11px;")
        self._lbl_fm_z = QLabel("—")
        self._lbl_fm_z.setStyleSheet("color: #e0e0e0; font-size: 11px;")
        fm_form.addRow("Shape (C×Z×Y×X):", self._lbl_fm_shape)
        fm_form.addRow("Channels:", self._lbl_fm_ch)
        fm_form.addRow("Z-slices:", self._lbl_fm_z)
        fm_layout.addLayout(fm_form)

        layout.addWidget(TitledPanel("FM Image", content=fm_body, collapsible=False))
        layout.addStretch(1)

    # ------------------------------------------------------------------
    # Browse / load
    # ------------------------------------------------------------------

    def _browse_project_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Project Directory", "")
        if path:
            self._proj_path.setText(path)
            self.project_dir_changed.emit(path)

    @property
    def project_dir(self) -> Optional[str]:
        p = self._proj_path.text().strip()
        return p if p else None

    def _browse_fib(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open FIB Image", "", "TIFF (*.tif *.tiff);;All Files (*)"
        )
        if path:
            self._load_fib(path)

    def _load_fib(self, path: str) -> None:
        try:
            image = FibsemImage.load(path)
        except Exception as exc:
            QMessageBox.warning(self, "Load error", str(exc))
            return
        self._fib_image = image
        self._fib_path.setText(path)
        h, w = image.data.shape[:2]
        self._lbl_fib_shape.setText(f"{h} × {w}")
        px = getattr(
            getattr(getattr(image, "metadata", None), "pixel_size", None), "x", None
        )
        self._lbl_fib_px.setText(f"{px * 1e9:.2f} nm" if px else "—")
        self.fib_image_changed.emit(image)

    def _browse_fm(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open FM Image",
            "",
            "OME-TIFF (*.ome.tiff *.ome.tif);;TIFF (*.tif *.tiff);;All Files (*)",
        )
        if path:
            self._load_fm(path)

    def _load_fm(self, path: str) -> None:
        try:
            image = FluorescenceImage.load(path)
        except Exception as exc:
            QMessageBox.warning(self, "Load error", str(exc))
            return
        self._fm_image = image
        self._fm_path.setText(path)
        c, z, h, w = image.data.shape
        self._lbl_fm_shape.setText(f"{c} × {z} × {h} × {w}")
        meta_channels = image.metadata.channels or []
        ch_names = ", ".join(
            ch.name or f"CH {i}" for i, ch in enumerate(meta_channels)
        ) or str(c)
        self._lbl_fm_ch.setText(ch_names)
        self._lbl_fm_z.setText(str(z))
        self.fm_image_changed.emit(image)

    # ------------------------------------------------------------------
    # Public setters (for pre-loading)
    # ------------------------------------------------------------------

    def set_fib_image(self, image: FibsemImage) -> None:
        self._fib_image = image
        filename = (
            getattr(
                getattr(getattr(image, "metadata", None), "image_settings", None),
                "filename",
                "",
            )
            or ""
        )
        self._fib_path.setText(filename)
        h, w = image.data.shape[:2]
        self._lbl_fib_shape.setText(f"{h} × {w}")
        px = getattr(
            getattr(getattr(image, "metadata", None), "pixel_size", None), "x", None
        )
        self._lbl_fib_px.setText(f"{px * 1e9:.2f} nm" if px else "—")

    def set_fm_image(self, image: FluorescenceImage) -> None:
        self._fm_image = image
        filename = getattr(image.metadata, "filename", "") or ""
        self._fm_path.setText(filename)
        c, z, h, w = image.data.shape
        self._lbl_fm_shape.setText(f"{c} × {z} × {h} × {w}")
        meta_channels = image.metadata.channels or []
        ch_names = ", ".join(
            ch.name or f"CH {i}" for i, ch in enumerate(meta_channels)
        ) or str(c)
        self._lbl_fm_ch.setText(ch_names)
        self._lbl_fm_z.setText(str(z))

    @property
    def fib_image(self) -> Optional[FibsemImage]:
        return self._fib_image

    @property
    def fm_image(self) -> Optional[FluorescenceImage]:
        return self._fm_image


# ---------------------------------------------------------------------------
# Tab 1 — Coordinates
# ---------------------------------------------------------------------------


class _CoordinatesTab(QWidget):
    """Coordinate list widgets + fit settings + load/save."""

    # Forwarded from list widgets — parent connects these
    fib_list: CoordinateListWidget
    fm_list: CoordinateListWidget
    poi_list: CoordinateListWidget
    surface_list: CoordinateListWidget

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: #2b2d31; border: none;")

        container = QWidget()
        container.setStyleSheet("background: #2b2d31;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # FIB fiducials
        self.fib_list = CoordinateListWidget(point_type=PointType.FIB)
        self._fib_panel = TitledPanel("FIB Fiducials", collapsible=True)
        self._fib_panel.set_content(self.fib_list)
        self._fib_count_label = QLabel("(0)")
        self._fib_count_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self._fib_panel.add_header_widget(self._fib_count_label)
        layout.addWidget(self._fib_panel)

        # FM fiducials
        self.fm_list = CoordinateListWidget(point_type=PointType.FM)
        self._fm_panel = TitledPanel("FM Fiducials", collapsible=True)
        self._fm_panel.set_content(self.fm_list)
        self._fm_count_label = QLabel("(0)")
        self._fm_count_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self._fm_panel.add_header_widget(self._fm_count_label)
        layout.addWidget(self._fm_panel)

        # POI
        self.poi_list = CoordinateListWidget(point_type=PointType.POI)
        self._poi_panel = TitledPanel("POI", collapsible=True)
        self._poi_panel.set_content(self.poi_list)
        self._poi_count_label = QLabel("(0)")
        self._poi_count_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self._poi_panel.add_header_widget(self._poi_count_label)
        layout.addWidget(self._poi_panel)

        # Surface (max 1)
        self.surface_list = CoordinateListWidget(point_type=PointType.SURFACE)
        self._surface_panel = TitledPanel("Surface", collapsible=True)
        self._surface_panel.set_content(self.surface_list)
        self._surface_count_label = QLabel("(0)")
        self._surface_count_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self._surface_panel.add_header_widget(self._surface_count_label)
        layout.addWidget(self._surface_panel)

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
        layout.addWidget(self._fit_panel)

        layout.addStretch(1)
        scroll.setWidget(container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(scroll)

    def update_headers(self) -> None:
        self._fib_count_label.setText(f"({len(self.fib_list.coordinates)})")
        self._fm_count_label.setText(f"({len(self.fm_list.coordinates)})")
        self._poi_count_label.setText(f"({len(self.poi_list.coordinates)})")
        self._surface_count_label.setText(f"({len(self.surface_list.coordinates)})")

    def rebuild_channel_combos(self, fm_image: Optional[FluorescenceImage]) -> None:
        for cb in (self._fm_fid_ch_combo, self._fm_poi_ch_combo):
            cb.clear()
            if fm_image is None:
                continue
            channels = fm_image.metadata.channels or []
            for i, ch in enumerate(channels):
                cb.addItem(ch.name or f"CH {i}")
            if cb.count() == 0:
                n = fm_image.data.shape[0]
                for i in range(n):
                    cb.addItem(f"CH {i}")


# ---------------------------------------------------------------------------
# Tab 2 — Results
# ---------------------------------------------------------------------------


class _ResultsTab(QWidget):
    """Correlation result summary and per-marker error table (no canvas)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Summary
        summary_body = QWidget()
        summary_form = QFormLayout(summary_body)
        summary_form.setContentsMargins(8, 4, 8, 4)
        summary_form.setSpacing(4)
        self._lbl_scale = self._val("—")
        self._lbl_rms = self._val("—")
        self._lbl_mae = self._val("—")
        self._lbl_rotation = self._val("—")
        self._lbl_trans = self._val("—")
        summary_form.addRow("Scale:", self._lbl_scale)
        summary_form.addRow("RMS Error:", self._lbl_rms)
        summary_form.addRow("Mean Abs Error:", self._lbl_mae)
        summary_form.addRow("Rotation:", self._lbl_rotation)
        summary_form.addRow("Translation:", self._lbl_trans)
        layout.addWidget(TitledPanel("Summary", content=summary_body))

        # Per-marker error table
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Marker", "dx (px)", "dy (px)"])
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._table.setMinimumHeight(120)
        layout.addWidget(TitledPanel("Per-Marker Error", content=self._table))
        layout.addStretch(1)

    @staticmethod
    def _val(text: str = "—") -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #e0e0e0; font-size: 12px;")
        return lbl

    def set_result(self, result: CorrelationResult) -> None:
        self._lbl_scale.setText(f"{result.scale:.4f}")
        self._lbl_rms.setText(f"{result.rms_error:.2f} px")

        if result.mean_absolute_error:
            mae_str = ", ".join(f"{v:.2f}" for v in result.mean_absolute_error) + " px"
        else:
            mae_str = "—"
        self._lbl_mae.setText(mae_str)

        if result.rotation_eulers:
            self._lbl_rotation.setText(
                "°, ".join(f"{v:.2f}" for v in result.rotation_eulers) + "°"
            )
        else:
            self._lbl_rotation.setText("—")

        if result.translation and len(result.translation) >= 2:
            self._lbl_trans.setText(
                ", ".join(f"{v:.1f}" for v in result.translation[:2])
            )
        else:
            self._lbl_trans.setText("—")

        markers = result.delta_2d
        self._table.setRowCount(len(markers))
        self._table.setMinimumHeight(max(120, min(len(markers) * 28 + 28, 300)))
        for i, pt in enumerate(markers):
            self._table.setItem(i, 0, _ro_item(f"M{i + 1}"))
            self._table.setItem(i, 1, _ro_item(f"{pt.x:.2f}"))
            self._table.setItem(i, 2, _ro_item(f"{pt.y:.2f}"))

    def clear(self) -> None:
        for lbl in (
            self._lbl_scale,
            self._lbl_rms,
            self._lbl_mae,
            self._lbl_rotation,
            self._lbl_trans,
        ):
            lbl.setText("—")
        self._table.setRowCount(0)


# ---------------------------------------------------------------------------
# Tab 3 — RI Correction
# ---------------------------------------------------------------------------


class _RITab(QWidget):
    """Refractive-index depth correction table."""

    correction_applied = pyqtSignal(object)  # CorrelationResult

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._poi: List[CorrelationPointOfInterest] = []
        self._surface_y: Optional[float] = None
        self._result: Optional[CorrelationResult] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        self._ri_widget = RefractiveIndexWidget()
        layout.addWidget(self._ri_widget)

        apply_row = QWidget()
        apply_layout = QHBoxLayout(apply_row)
        apply_layout.setContentsMargins(0, 0, 0, 0)
        apply_layout.setSpacing(8)

        self._btn_apply = QPushButton("Apply")
        self._btn_apply.setFixedWidth(80)
        self._btn_apply.clicked.connect(self._apply)
        apply_layout.addWidget(self._btn_apply)

        self._lbl_warning = QLabel("")
        self._lbl_warning.setStyleSheet("color: #e07b39; font-size: 11px;")
        apply_layout.addWidget(self._lbl_warning)
        apply_layout.addStretch(1)
        layout.addWidget(apply_row)

        self._lbl_distance = QLabel("")
        self._lbl_distance.setStyleSheet("color: #a0c8ff; font-size: 11px;")
        self._lbl_distance.setVisible(False)
        layout.addWidget(self._lbl_distance)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(
            ["POI", "X (px)", "Y original (px)", "Y corrected (px)"]
        )
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._table.setMinimumHeight(120)
        layout.addWidget(TitledPanel("Corrected POI Positions", content=self._table))

        self._lbl_multi_poi = QLabel(
            "Note: correction is applied to POI 1 only. "
            "Additional POIs are shown for reference."
        )
        self._lbl_multi_poi.setStyleSheet("color: #e0c060; font-size: 11px;")
        self._lbl_multi_poi.setWordWrap(True)
        self._lbl_multi_poi.setVisible(False)
        layout.addWidget(self._lbl_multi_poi)
        layout.addStretch(1)

    def set_result(
        self,
        result: Optional[CorrelationResult],
        surface_coordinate: Optional[Coordinate] = None,
    ) -> None:
        self._result = result
        self._poi = result.poi if result else []
        self._surface_y = surface_coordinate.point.y if surface_coordinate else None

        logging.debug(
            f"[RITab.set_result] result={result is not None}, "
            f"surface_coordinate={surface_coordinate}, "
            f"surface_y={self._surface_y}, "
            f"n_poi={len(self._poi)}"
        )

        self._table.setRowCount(0)
        self._lbl_multi_poi.setVisible(len(self._poi) > 1)
        self._update_distance_label()

        factor = result.refractive_index_correction_factor if result else None
        if factor is not None:
            self._ri_widget.set_factor(factor)
            self._lbl_warning.setStyleSheet("color: #6dbf6d; font-size: 11px;")
            self._lbl_warning.setText(
                f"Correction already applied (factor: {factor:.3f})."
            )
        elif not self._poi:
            self._lbl_warning.setStyleSheet("color: #e07b39; font-size: 11px;")
            self._lbl_warning.setText("No POI in result.")
        elif self._surface_y is None:
            self._lbl_warning.setStyleSheet("color: #e07b39; font-size: 11px;")
            self._lbl_warning.setText("No surface coordinate — correction unavailable.")
        else:
            self._lbl_warning.setStyleSheet("color: #e07b39; font-size: 11px;")
            self._lbl_warning.setText("")

    def _update_distance_label(self) -> None:
        if self._surface_y is None or not self._poi:
            self._lbl_distance.setVisible(False)
            return
        dist_px = abs(self._poi[0].image_px.y - self._surface_y)
        self._lbl_distance.setText(f"Surface → POI depth: {dist_px:.1f} px")
        self._lbl_distance.setVisible(True)

    def _apply(self) -> None:
        if not self._poi:
            self._lbl_warning.setText("No POI available.")
            return
        if self._surface_y is None:
            self._lbl_warning.setText(
                "No surface coordinate — cannot apply correction."
            )
            return
        self._lbl_warning.setText("")
        factor = self._ri_widget.get_factor()
        surface_y = self._surface_y

        logging.info(
            f"[RITab._apply] factor={factor:.4f}, surface_y={surface_y:.2f}, n_poi={len(self._poi)}"
        )

        # Populate table (all POIs shown for reference)
        self._table.setRowCount(len(self._poi))
        for i, poi in enumerate(self._poi):
            depth = poi.image_px.y - surface_y
            corrected_y = surface_y + depth * factor
            logging.info(
                f"[RITab._apply] POI {i + 1}: original_y={poi.image_px.y:.2f}, "
                f"depth={depth:.2f}, corrected_y={corrected_y:.2f}"
            )
            self._table.setItem(i, 0, _ro_item(f"POI {i + 1}"))
            self._table.setItem(i, 1, _ro_item(f"{poi.image_px.x:.2f}"))
            self._table.setItem(i, 2, _ro_item(f"{poi.image_px.y:.2f}"))
            self._table.setItem(i, 3, _ro_item(f"{corrected_y:.2f}"))

        if self._result is None:
            logging.warning("[RITab._apply] No result to update with correction.")
            return

        # Update POI 0 in the result and propagate (reads input_data internally)
        self._result.apply_refractive_index_correction(factor)
        logging.info("[RITab._apply] correction applied, emitting correction_applied")
        self.correction_applied.emit(self._result)


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------


class CorrelationTabWidget(QWidget):
    """FIB canvas | FM canvas | tabbed controls.

    Signals
    -------
    data_changed   : CorrelationInputData
    result_changed : CorrelationResult
    """

    data_changed = pyqtSignal(object)  # CorrelationInputData
    result_changed = pyqtSignal(object)  # CorrelationResult
    continue_pressed_signal = pyqtSignal(object)  # CorrelationResult

    def __init__(
        self,
        fib_image: Optional[FibsemImage] = None,
        fm_image: Optional[FluorescenceImage] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._fib_image: Optional[FibsemImage] = None
        self._fm_image: Optional[FluorescenceImage] = None
        self._result: Optional[CorrelationResult] = None
        self._worker: Optional[_CorrelationWorker] = None
        self._project_dir: Optional[str] = None

        self._setup_ui()
        self._connect_signals()
        self._btn_continue.setEnabled(False)

        if fib_image is not None:
            self.set_fib_image(fib_image)
        if fm_image is not None:
            self.set_fm_image(fm_image)

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Menu bar
        menubar = QMenuBar()
        file_menu = menubar.addMenu("File")

        self._action_load_fib = QAction("Load FIB Image", self)
        self._action_load_fm = QAction("Load Fluorescence Image", self)
        self._action_load_coords = QAction("Load Coordinates", self)
        self._action_save_coords = QAction("Save Coordinates", self)
        self._action_load_result = QAction("Load Correlation Result", self)

        file_menu.addAction(self._action_load_fib)
        file_menu.addAction(self._action_load_fm)
        file_menu.addSeparator()
        file_menu.addAction(self._action_load_coords)
        file_menu.addAction(self._action_save_coords)
        file_menu.addSeparator()
        file_menu.addAction(self._action_load_result)
        file_menu.addSeparator()
        self._action_export_csv = QAction("Export CSV", self)
        file_menu.addAction(self._action_export_csv)

        view_menu = menubar.addMenu("View")
        self._action_reset_views = QAction("Reset Views", self)
        self._action_show_scalebar = QAction("Show ScaleBar", self)
        self._action_show_scalebar.setCheckable(True)
        self._action_show_scalebar.setChecked(True)
        view_menu.addAction(self._action_reset_views)
        view_menu.addAction(self._action_show_scalebar)

        test_menu = menubar.addMenu("Test")
        self._action_test_save_plot = QAction("Test Save Plot", self)
        test_menu.addAction(self._action_test_save_plot)

        layout.addWidget(menubar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, stretch=1)

        # Left: FIB image canvas
        self._fib_canvas = ImagePointCanvas(
            allowed_point_types=[PointType.FIB, PointType.SURFACE],
        )
        splitter.addWidget(self._fib_canvas)

        # Middle: FM image display
        self._fm_display = FMImageDisplayWidget(
            allowed_point_types=[PointType.FM, PointType.POI],
        )
        splitter.addWidget(self._fm_display)

        # Right: tab widget stacked above run button
        self._tabs = QTabWidget()
        self._images_tab = _ImagesTab()
        self._coords_tab = _CoordinatesTab()
        self._results_tab = _ResultsTab()
        self._ri_tab = _RITab()

        self._tabs.addTab(self._images_tab, "Images")
        self._tabs.addTab(self._coords_tab, "Coordinates")
        self._tabs.addTab(self._results_tab, "Results")
        self._tabs.addTab(self._ri_tab, "Refractive Index")
        self._tabs.setTabEnabled(3, False)
        self._tabs.setTabToolTip(3, "Run correlation first to enable RI correction")

        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self._tabs, stretch=1)

        run_bar = QWidget()
        run_bar.setStyleSheet("background: #1e2124; border-top: 1px solid #3a3d42;")
        run_layout = QVBoxLayout(run_bar)
        run_layout.setContentsMargins(8, 6, 8, 6)
        run_layout.setSpacing(4)

        self._lbl_status = QLabel("Load images and add ≥ 4 FIB / FM pairs and ≥ 1 POI.")
        self._lbl_status.setStyleSheet("color: #aaa; font-size: 12px;")
        self._lbl_status.setWordWrap(True)
        run_layout.addWidget(self._lbl_status)

        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(8)

        self._btn_run = QPushButton("Run Correlation")
        self._btn_run.setEnabled(False)
        self._btn_run.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        btn_layout.addWidget(self._btn_run)

        self._btn_continue = QPushButton("Continue")
        self._btn_continue.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        btn_layout.addWidget(self._btn_continue)
        btn_layout.addStretch(1)

        run_layout.addWidget(btn_row)
        right_layout.addWidget(run_bar)

        splitter.addWidget(right_pane)
        splitter.setSizes([500, 500, 350])

    def _connect_signals(self) -> None:
        # Images tab → push to canvases
        self._images_tab.fib_image_changed.connect(self._on_fib_image_changed)
        self._images_tab.fm_image_changed.connect(self._on_fm_image_changed)
        self._images_tab.project_dir_changed.connect(self._on_project_dir_changed)

        # Auto-save
        self.data_changed.connect(self._auto_save_data)
        self.result_changed.connect(self._auto_save_result)

        # RI correction
        self._ri_tab.correction_applied.connect(self._on_correction_applied)

        # Canvas → list
        self._fib_canvas.point_selected.connect(self._on_fib_canvas_selected)
        self._fib_canvas.point_moved.connect(self._on_fib_canvas_moved)
        self._fib_canvas.point_removed.connect(self._on_fib_canvas_removed)
        self._fib_canvas.point_add_requested.connect(self._on_fib_add_requested)

        self._fm_display.point_selected.connect(self._on_fm_canvas_selected)
        self._fm_display.point_moved.connect(self._on_fm_canvas_moved)
        self._fm_display.point_removed.connect(self._on_fm_canvas_removed)
        self._fm_display.point_add_requested.connect(self._on_fm_add_requested)

        # List → canvas
        cl = self._coords_tab
        cl.fib_list.coordinate_selected.connect(self._on_fib_list_selected)
        cl.fib_list.coordinate_changed.connect(self._on_fib_list_changed)
        cl.fib_list.coordinate_removed.connect(self._on_fib_list_removed)
        cl.fib_list.order_changed.connect(
            lambda _: (self._refresh_fib_canvas(), self.data_changed.emit(self.data))
        )
        cl.fib_list.refit_requested.connect(self._on_refit_requested)

        cl.fm_list.coordinate_selected.connect(self._on_fm_list_selected)
        cl.fm_list.coordinate_changed.connect(self._on_fm_list_changed)
        cl.fm_list.coordinate_removed.connect(self._on_fm_list_removed)
        cl.fm_list.order_changed.connect(
            lambda _: (self._refresh_fm_canvas(), self.data_changed.emit(self.data))
        )
        cl.fm_list.refit_requested.connect(self._on_refit_requested)

        cl.poi_list.coordinate_selected.connect(self._on_poi_list_selected)
        cl.poi_list.coordinate_changed.connect(self._on_poi_list_changed)
        cl.poi_list.coordinate_removed.connect(self._on_poi_list_removed)
        cl.poi_list.order_changed.connect(
            lambda _: (self._refresh_fm_canvas(), self.data_changed.emit(self.data))
        )
        cl.poi_list.refit_requested.connect(self._on_refit_requested)

        cl.surface_list.coordinate_selected.connect(self._on_surface_list_selected)
        cl.surface_list.coordinate_changed.connect(self._on_surface_list_changed)
        cl.surface_list.coordinate_removed.connect(self._on_surface_list_removed)
        cl.surface_list.order_changed.connect(
            lambda _: (self._refresh_fib_canvas(), self.data_changed.emit(self.data))
        )
        cl.surface_list.refit_requested.connect(self._on_refit_requested)

        # File menu
        self._action_load_fib.triggered.connect(self._menu_load_fib)
        self._action_load_fm.triggered.connect(self._menu_load_fm)
        self._action_load_coords.triggered.connect(self._on_load)
        self._action_save_coords.triggered.connect(self._on_save)
        self._action_load_result.triggered.connect(self._menu_load_result)
        self._action_export_csv.triggered.connect(lambda _: self._menu_export_csv())
        self._action_reset_views.triggered.connect(lambda _: self._reset_views())
        self._action_show_scalebar.toggled.connect(self._on_scalebar_toggled)
        self._on_scalebar_toggled(True)
        self._action_test_save_plot.triggered.connect(lambda _: self.save_plot())

        # Bottom bar run button
        self._btn_run.clicked.connect(self._run)
        self._btn_continue.clicked.connect(self._on_continue_pressed)
        self.data_changed.connect(self._update_run_button)
        self.data_changed.connect(self._on_data_changed)

    def _on_data_changed(self, data: CorrelationInputData) -> None:
        self._ri_tab.set_result(
            self._result, surface_coordinate=data.surface_coordinate
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_fib_image(self, fib_image: FibsemImage) -> None:
        """Load FIB image into canvas and update images tab."""
        self._fib_image = fib_image
        arr = fib_image.filtered_data.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / arr.max()
        self._fib_canvas.set_image(arr)
        if fib_image.metadata and fib_image.metadata.pixel_size:
            self._fib_canvas.set_pixel_size(fib_image.metadata.pixel_size.x)
        h, w = fib_image.data.shape[:2]
        self._coords_tab.fib_list.set_axis_maxima(x_max=w - 1, y_max=h - 1)
        self._coords_tab.surface_list.set_axis_maxima(x_max=w - 1, y_max=h - 1)
        self._images_tab.set_fib_image(fib_image)

    def set_fm_image(self, fm_image: FluorescenceImage) -> None:
        """Load FM image into canvas and update images tab."""
        self._fm_image = fm_image
        self._fm_display.set_fm_image(fm_image)
        px = getattr(fm_image.metadata, "pixel_size_x", None)
        if px:
            self._fm_display.canvas.set_pixel_size(px)
        _, n_z, h, w = fm_image.data.shape
        self._coords_tab.fm_list.set_axis_maxima(
            x_max=w - 1, y_max=h - 1, z_max=n_z - 1
        )
        self._coords_tab.poi_list.set_axis_maxima(
            x_max=w - 1, y_max=h - 1, z_max=n_z - 1
        )
        self._coords_tab.rebuild_channel_combos(fm_image)
        self._images_tab.set_fm_image(fm_image)

    def set_project_dir(self, path: str) -> None:
        """Set the project directory used for auto-save and export."""
        self._project_dir = path
        self._images_tab._proj_path.setText(path)

    def set_data(self, data: CorrelationInputData) -> None:
        """Populate all coordinate lists and refresh canvases."""
        fib_coords = list(data.fib_coordinates)
        fm_coords = list(data.fm_coordinates)
        poi_coords = list(data.poi_coordinates)
        surf_coords = (
            [data.surface_coordinate] if data.surface_coordinate is not None else []
        )

        self._fib_canvas.set_coordinates(fib_coords + surf_coords)
        self._fm_display.set_coordinates(fm_coords + poi_coords)

        cl = self._coords_tab
        cl.fib_list.coordinates = fib_coords
        cl.fm_list.coordinates = fm_coords
        cl.poi_list.coordinates = poi_coords
        cl.surface_list.coordinates = surf_coords
        cl.update_headers()

    def load_data(self, path: str) -> None:
        """Load coordinates from JSON, preserving current images."""
        loaded = CorrelationInputData.load(path)
        loaded.fib_image = self._fib_image
        loaded.fm_image = self._fm_image
        self.set_data(loaded)
        self.data_changed.emit(self.data)

    def save_data(self, path: str) -> None:
        """Save current coordinates to JSON."""
        self.data.save(path)

    @property
    def data(self) -> CorrelationInputData:
        cl = self._coords_tab
        surf = cl.surface_list.coordinates
        return CorrelationInputData(
            fib_image=self._fib_image,
            fm_image=self._fm_image,
            fib_coordinates=cl.fib_list.coordinates,
            fm_coordinates=cl.fm_list.coordinates,
            poi_coordinates=cl.poi_list.coordinates,
            surface_coordinate=surf[0] if surf else None,
        )

    @property
    def result(self) -> Optional[CorrelationResult]:
        return self._result

    # ------------------------------------------------------------------
    # Canvas refresh helpers
    # ------------------------------------------------------------------

    def _refresh_fib_canvas(self) -> None:
        cl = self._coords_tab
        self._fib_canvas.set_coordinates(
            cl.fib_list.coordinates + cl.surface_list.coordinates
        )

    def _refresh_fm_canvas(self) -> None:
        cl = self._coords_tab
        self._fm_display.set_coordinates(
            cl.fm_list.coordinates + cl.poi_list.coordinates
        )

    # ------------------------------------------------------------------
    # Image loaded slots
    # ------------------------------------------------------------------

    def _on_fib_image_changed(self, image: FibsemImage) -> None:
        self.set_fib_image(image)

    def _on_fm_image_changed(self, image: FluorescenceImage) -> None:
        self.set_fm_image(image)

    def _on_project_dir_changed(self, path: str) -> None:
        self._project_dir = path
        self._lbl_status.setText(f"Project: {path}")

    def _auto_save_data(self, data: CorrelationInputData) -> None:
        if not self._project_dir:
            return
        try:
            data.save(os.path.join(self._project_dir, "correlation_data.json"))
        except Exception:
            _logger.exception("Auto-save of correlation data failed")

    def _auto_save_result(self, result: CorrelationResult) -> None:
        if not self._project_dir:
            return
        try:
            result.save(os.path.join(self._project_dir, "correlation_result.json"))
        except Exception:
            _logger.exception("Auto-save of correlation result failed")

    # ------------------------------------------------------------------
    # Run correlation
    # ------------------------------------------------------------------

    def _update_run_button(self, data: Optional[CorrelationInputData] = None) -> None:
        d = data if data is not None else self.data
        running = self._worker is not None and self._worker.isRunning()
        ok = (
            not running
            and self._fib_image is not None
            and self._fm_image is not None
            and len(d.fib_coordinates) >= 4
            and len(d.fm_coordinates) >= 4
            and len(d.poi_coordinates) >= 1
            and len(d.fib_coordinates) == len(d.fm_coordinates)
        )
        self._btn_run.setEnabled(ok)
        if running:
            self._lbl_status.setText("Running…")
        elif self._fib_image is None or self._fm_image is None:
            self._lbl_status.setText("Load FIB and FM images to continue.")
        elif ok:
            self._lbl_status.setText("Ready.")
        else:
            n_fib = len(d.fib_coordinates)
            n_fm = len(d.fm_coordinates)
            n_poi = len(d.poi_coordinates)
            self._lbl_status.setText(
                f"Need ≥4 matched pairs (FIB={n_fib}, FM={n_fm}) and ≥1 POI ({n_poi})."
            )

    def _run(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        self._btn_run.setEnabled(False)
        self._lbl_status.setText("Running…")
        self._worker = _CorrelationWorker(copy.deepcopy(self.data))
        self._worker.result_ready.connect(self._on_result_ready)
        self._worker.errored.connect(self._on_run_error)
        self._worker.start()

    def _on_result_ready(self, result: CorrelationResult) -> None:
        self._result = result
        self._results_tab.set_result(result)
        self._ri_tab.set_result(result, surface_coordinate=self.data.surface_coordinate)
        self._tabs.setTabEnabled(3, True)
        self._overlay_result_on_fib(result)
        self._lbl_status.setText("Done.")
        self._btn_continue.setEnabled(True)
        self._update_run_button()
        self.result_changed.emit(result)

    def _on_correction_applied(self, result: CorrelationResult) -> None:
        self._result = result
        self._overlay_result_on_fib(result)
        self.result_changed.emit(result)

    def _on_run_error(self, msg: str) -> None:
        self._lbl_status.setText(f"Error: {msg}")
        self._update_run_button()

    def _overlay_result_on_fib(self, result: CorrelationResult) -> None:
        """Draw reprojected error markers and POI on the FIB canvas."""
        self._fib_canvas.clear_overlay()

        # Reprojected FM fiducials — red "x", smaller, labeled E1/E2/...
        error_pts = [(r.x, r.y) for r in result.reprojected_3d]
        if error_pts:
            self._fib_canvas.add_overlay_points(
                error_pts,
                color="#ff4444",
                label_prefix="E",
                size=7,
                marker="o",
            )

        # Reprojected POI — magenta circle, labeled P1/P2/...
        poi_pts = [(p.image_px.x, p.image_px.y) for p in result.poi]
        if poi_pts:
            self._fib_canvas.add_overlay_points(
                poi_pts,
                color="#ff00ff",
                label_prefix="P",
                size=9,
                marker="o",
            )

    # ------------------------------------------------------------------
    # Canvas → list slots
    # ------------------------------------------------------------------

    def _on_fib_canvas_selected(self, coord: Coordinate) -> None:
        cl = self._coords_tab
        if coord.point_type == PointType.SURFACE:
            cl.surface_list.select_coordinate_silent(coord)
            cl.fib_list.select_coordinate_silent(None)
        else:
            cl.fib_list.select_coordinate_silent(coord)
            cl.surface_list.select_coordinate_silent(None)
        cl.fm_list.select_coordinate_silent(None)
        cl.poi_list.select_coordinate_silent(None)
        self._fm_display.set_selected(None)

    def _on_fm_canvas_selected(self, coord: Coordinate) -> None:
        cl = self._coords_tab
        self._fib_canvas.set_selected(None)
        cl.fib_list.select_coordinate_silent(None)
        cl.surface_list.select_coordinate_silent(None)
        if coord.point_type == PointType.FM:
            cl.fm_list.select_coordinate_silent(coord)
            cl.poi_list.select_coordinate_silent(None)
        else:
            cl.fm_list.select_coordinate_silent(None)
            cl.poi_list.select_coordinate_silent(coord)

    def _on_fib_canvas_moved(self, coord: Coordinate) -> None:
        cl = self._coords_tab
        if coord.point_type == PointType.SURFACE:
            cl.surface_list.refresh_coordinate(coord)
        else:
            cl.fib_list.refresh_coordinate(coord)
        self.data_changed.emit(self.data)

    def _on_fm_canvas_moved(self, coord: Coordinate) -> None:
        cl = self._coords_tab
        if coord.point_type == PointType.FM:
            cl.fm_list.refresh_coordinate(coord)
        else:
            cl.poi_list.refresh_coordinate(coord)
        self.data_changed.emit(self.data)

    def _on_fib_canvas_removed(self, coord: Coordinate) -> None:
        cl = self._coords_tab
        if coord.point_type == PointType.SURFACE:
            cl.surface_list.coordinates = [
                c for c in cl.surface_list.coordinates if c is not coord
            ]
        else:
            cl.fib_list.coordinates = [
                c for c in cl.fib_list.coordinates if c is not coord
            ]
        self._refresh_fib_canvas()
        cl.update_headers()
        self.data_changed.emit(self.data)

    def _on_fm_canvas_removed(self, coord: Coordinate) -> None:
        cl = self._coords_tab
        if coord.point_type == PointType.FM:
            cl.fm_list.coordinates = [
                c for c in cl.fm_list.coordinates if c is not coord
            ]
        else:
            cl.poi_list.coordinates = [
                c for c in cl.poi_list.coordinates if c is not coord
            ]
        self._refresh_fm_canvas()
        cl.update_headers()
        self.data_changed.emit(self.data)

    def _on_fib_add_requested(self, x: float, y: float, pt: PointType) -> None:
        cl = self._coords_tab
        coord = Coordinate(PointXYZ(x, y, 0.0), pt)
        if pt == PointType.SURFACE:
            cl.surface_list.coordinates = [coord]
        else:
            cl.fib_list.add_coordinate(coord)
        self._refresh_fib_canvas()
        cl.update_headers()
        self.data_changed.emit(self.data)

    def _on_fm_add_requested(self, x: float, y: float, pt: PointType) -> None:
        cl = self._coords_tab
        coord = Coordinate(PointXYZ(x, y, float(self._fm_display.current_z)), pt)
        if pt == PointType.FM:
            cl.fm_list.add_coordinate(coord)
        else:
            cl.poi_list.add_coordinate(coord)
        self._refresh_fm_canvas()
        cl.update_headers()
        self.data_changed.emit(self.data)

    # ------------------------------------------------------------------
    # List → canvas slots
    # ------------------------------------------------------------------

    def _on_fib_list_selected(self, coord: Coordinate) -> None:
        cl = self._coords_tab
        self._fib_canvas.set_selected(coord)
        self._fm_display.set_selected(None)
        cl.fm_list.select_coordinate_silent(None)
        cl.poi_list.select_coordinate_silent(None)
        cl.surface_list.select_coordinate_silent(None)

    def _on_fm_list_selected(self, coord: Coordinate) -> None:
        cl = self._coords_tab
        self._fm_display.set_selected(coord)
        self._fib_canvas.set_selected(None)
        cl.fib_list.select_coordinate_silent(None)
        cl.poi_list.select_coordinate_silent(None)
        cl.surface_list.select_coordinate_silent(None)

    def _on_poi_list_selected(self, coord: Coordinate) -> None:
        cl = self._coords_tab
        self._fm_display.set_selected(coord)
        self._fib_canvas.set_selected(None)
        cl.fib_list.select_coordinate_silent(None)
        cl.fm_list.select_coordinate_silent(None)
        cl.surface_list.select_coordinate_silent(None)

    def _on_surface_list_selected(self, coord: Coordinate) -> None:
        cl = self._coords_tab
        self._fib_canvas.set_selected(coord)
        self._fm_display.set_selected(None)
        cl.fib_list.select_coordinate_silent(None)
        cl.fm_list.select_coordinate_silent(None)
        cl.poi_list.select_coordinate_silent(None)

    def _on_fib_list_changed(self, coord: Coordinate, _f: str, _v: float) -> None:
        self._fib_canvas.refresh_coordinate(coord)
        self.data_changed.emit(self.data)

    def _on_fm_list_changed(self, coord: Coordinate, _f: str, _v: float) -> None:
        self._fm_display.refresh_coordinate(coord)
        self.data_changed.emit(self.data)

    def _on_poi_list_changed(self, coord: Coordinate, _f: str, _v: float) -> None:
        self._fm_display.refresh_coordinate(coord)
        self.data_changed.emit(self.data)

    def _on_surface_list_changed(self, coord: Coordinate, _f: str, _v: float) -> None:
        self._fib_canvas.refresh_coordinate(coord)
        self.data_changed.emit(self.data)

    def _on_fib_list_removed(self, _coord: Coordinate) -> None:
        self._refresh_fib_canvas()
        self._coords_tab.update_headers()
        self.data_changed.emit(self.data)

    def _on_fm_list_removed(self, _coord: Coordinate) -> None:
        self._refresh_fm_canvas()
        self._coords_tab.update_headers()
        self.data_changed.emit(self.data)

    def _on_poi_list_removed(self, _coord: Coordinate) -> None:
        self._refresh_fm_canvas()
        self._coords_tab.update_headers()
        self.data_changed.emit(self.data)

    def _on_surface_list_removed(self, _coord: Coordinate) -> None:
        self._refresh_fib_canvas()
        self._coords_tab.update_headers()
        self.data_changed.emit(self.data)

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------

    def _menu_load_fib(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load FIB Image", "", "TIFF (*.tif *.tiff);;All Files (*)"
        )
        if path:
            self._images_tab._load_fib(path)

    def _menu_load_fm(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Fluorescence Image",
            "",
            "OME-TIFF (*.ome.tiff *.ome.tif);;TIFF (*.tif *.tiff);;All Files (*)",
        )
        if path:
            self._images_tab._load_fm(path)

    def _on_continue_pressed(self) -> None:
        if self._result is None:
            return
        reply = QMessageBox.question(
            self,
            "Finish Correlation",
            "Continue with correlation result and close?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.continue_pressed_signal.emit(self._result)
            self.window().close()

    def _menu_export_csv(self) -> None:
        if self._result is None:
            QMessageBox.warning(self, "Export CSV", "No correlation result to export.")
            return
        start = self._project_dir or ""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Correlation CSV", start, "CSV (*.csv);;All files (*)"
        )
        if path:
            self._result.to_csv(path)

    def _reset_views(self) -> None:
        self._fib_canvas.reset_view()
        self._fm_display.canvas.reset_view()

    def _on_scalebar_toggled(self, visible: bool) -> None:
        self._fib_canvas.set_scalebar_visible(visible)
        self._fm_display.canvas.set_scalebar_visible(visible)

    def save_plot(self, path: Optional[str] = None) -> None:
        """Save FIB + FM canvases as a side-by-side matplotlib figure."""
        import matplotlib.pyplot as plt

        if path is None:
            if not self._project_dir:
                QMessageBox.warning(self, "Save Plot", "No project directory set.")
                return
            ts = time.strftime(DATETIME_FILE)
            path = os.path.join(self._project_dir, f"correlation_plot_{ts}.png")

        self._fib_canvas.reset_view()
        self._fm_display.canvas.reset_view()

        fig, (ax_fib, ax_fm) = plt.subplots(1, 2, figsize=(16, 8), facecolor="#1e2124")
        for ax in (ax_fib, ax_fm):
            ax.set_facecolor("#1e2124")

        self._fib_canvas.render_to_axes(ax_fib)
        ax_fib.set_title("FIB", color="white", fontsize=12)

        self._fm_display.canvas.render_to_axes(ax_fm)
        ax_fm.set_title("FM", color="white", fontsize=12)

        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    def _menu_load_result(self) -> None:
        start = self._project_dir or ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Correlation Result", start, "JSON (*.json);;All files (*)"
        )
        if not path:
            return
        try:
            result = CorrelationResult.load(path)
        except Exception as exc:
            QMessageBox.warning(self, "Load Error", f"Could not load result:\n{exc}")
            return
        self._on_result_ready(result)
        if result.input_data:
            self.set_data(result.input_data)
            self._update_run_button()

    def _on_load(self) -> None:
        start = self._project_dir or ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Coordinates", start, "JSON (*.json);;All files (*)"
        )
        if path:
            self.load_data(path)

    def _on_save(self) -> None:
        start = self._project_dir or ""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Coordinates", start, "JSON (*.json);;All files (*)"
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

        cl = self._coords_tab
        x, y, z = coord.point.x, coord.point.y, coord.point.z
        fig = None

        try:
            if coord.point_type in (PointType.FIB, PointType.SURFACE):
                method = cl._fib_method_combo.currentText()
                if method == "Hole" and self._fib_image is not None:
                    xr, yr, fig = hole_fitting_FIB(
                        self._fib_image.filtered_data, int(x), int(y)
                    )
                    coord.point.x, coord.point.y = float(xr), float(yr)

            elif coord.point_type in (PointType.FM, PointType.POI):
                is_fid = coord.point_type == PointType.FM
                method = (
                    cl._fm_fid_method_combo if is_fid else cl._fm_poi_method_combo
                ).currentText()
                ch = (
                    cl._fm_fid_ch_combo if is_fid else cl._fm_poi_ch_combo
                ).currentIndex()
                if method != "None" and self._fm_image is not None and ch >= 0:
                    img = self._fm_image.data[ch]
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

        if coord.point_type in (PointType.FIB, PointType.SURFACE):
            list_w = (
                cl.fib_list if coord.point_type == PointType.FIB else cl.surface_list
            )
            list_w.refresh_coordinate(coord)
            self._fib_canvas.refresh_coordinate(coord)
        else:
            list_w = cl.fm_list if coord.point_type == PointType.FM else cl.poi_list
            list_w.refresh_coordinate(coord)
            self._fm_display.refresh_coordinate(coord)

        self.data_changed.emit(self.data)

        if fig is not None and cl._show_diag_check.isChecked():
            _show_figure_dialog(fig, title="Refit Diagnostic", parent=self)


# ---------------------------------------------------------------------------
# Dialog wrapper
# ---------------------------------------------------------------------------


class CorrelationTabDialog(QDialog):
    """Modal QDialog wrapping CorrelationTabWidget.

    Replaces the napari-based CorrelationUI window with a self-contained
    dialog. The Continue button inside the widget handles confirmation and
    calls ``self.window().close()``; connecting ``continue_pressed_signal``
    to ``accept()`` ensures ``exec_()`` returns ``QDialog.Accepted``.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("FIB-FM Correlation")
        self.setModal(True)
        self.resize(1500, 900)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.widget = CorrelationTabWidget()
        layout.addWidget(self.widget)

        self.widget.continue_pressed_signal.connect(lambda _: self.accept())

    def set_fib_image(self, fib_image) -> None:
        self.widget.set_fib_image(fib_image)

    def set_fm_image(self, fm_image) -> None:
        self.widget.set_fm_image(fm_image)

    def set_project_dir(self, path: str) -> None:
        self.widget.set_project_dir(path)

    @property
    def result(self):
        return self.widget.result
