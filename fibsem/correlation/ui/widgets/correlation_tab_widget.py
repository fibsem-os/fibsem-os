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
│                     │    Add FM-Surface   │  └─────────────────────────┘│
└─────────────────────┴─────────────────────┴────────────────────────────┘

Tabs
----
  Images      — browse / load FIB and FM images
  Coordinates — coordinate lists (FIB, FM, POI, Surface FIB/FM) + fit settings + load/save
  Results     — run correlation + CorrelationResultWidget overlay
  RI          — refractive-index depth correction; mode follows the surface point:
                FIB surface → post-correlation (corrects the correlated POI 1),
                FM surface  → pre-correlation (corrects every input POI z, applied
                during the run). Only one surface point can exist at a time.

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
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional

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
    scale_about_surface,
)
from fibsem.correlation.ui.widgets.coordinate_list_widget import CoordinateListWidget
from fibsem.ui import stylesheets
from fibsem.correlation.ui.widgets.fm_image_display_widget import FMImageDisplayWidget
from fibsem.correlation.ui.widgets.image_point_canvas import ImagePointCanvas
from fibsem.fm.structures import FluorescenceImage
from fibsem.structures import FibsemImage
from fibsem.ui.widgets.custom_widgets import (
    QDirectoryLineEdit,
    QFileLineEdit,
    TitledPanel,
)
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
        # Last path successfully loaded / shown per field — suppresses reloads
        # when the editable path widget re-emits editingFinished (focus-out).
        self._fib_loaded_path: str = ""
        self._fm_loaded_path: str = ""
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
        self._proj_path = QDirectoryLineEdit()
        self._proj_path.lineEdit.setPlaceholderText("No project directory set")
        self._proj_path.editingFinished.connect(self._on_project_dir_edited)
        proj_layout.addWidget(self._proj_path, stretch=1)
        layout.addWidget(TitledPanel("Project", content=proj_body, collapsible=False))

        # ---- FIB section ----
        fib_body = QWidget()
        fib_layout = QVBoxLayout(fib_body)
        fib_layout.setContentsMargins(8, 4, 8, 4)
        fib_layout.setSpacing(4)

        self._fib_path = QFileLineEdit(filter="TIFF (*.tif *.tiff);;All Files (*)")
        self._fib_path.lineEdit.setPlaceholderText("No file loaded")
        self._fib_path.editingFinished.connect(self._on_fib_path_edited)
        fib_layout.addWidget(self._fib_path)

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

        self._fm_path = QFileLineEdit(
            filter="OME-TIFF (*.ome.tiff *.ome.tif);;TIFF (*.tif *.tiff);;All Files (*)"
        )
        self._fm_path.lineEdit.setPlaceholderText("No file loaded")
        self._fm_path.editingFinished.connect(self._on_fm_path_edited)
        fm_layout.addWidget(self._fm_path)

        fm_form = QFormLayout()
        fm_form.setContentsMargins(0, 0, 0, 0)
        fm_form.setSpacing(2)
        self._lbl_fm_shape = QLabel("—")
        self._lbl_fm_shape.setStyleSheet("color: #e0e0e0; font-size: 11px;")
        self._lbl_fm_ch = QLabel("—")
        self._lbl_fm_ch.setStyleSheet("color: #e0e0e0; font-size: 11px;")
        self._lbl_fm_ch.setWordWrap(True)
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

    def _on_project_dir_edited(self) -> None:
        path = self._proj_path.text().strip()
        if path:
            self.project_dir_changed.emit(path)

    @property
    def project_dir(self) -> Optional[str]:
        p = self._proj_path.text().strip()
        return p if p else None

    def _on_fib_path_edited(self) -> None:
        # Fires on browse-pick and manual entry; skip programmatic/no-op sets
        path = self._fib_path.text().strip()
        if path and path != self._fib_loaded_path:
            self._load_fib(path)

    def _load_fib(self, path: str) -> None:
        try:
            image = FibsemImage.load(path)
        except Exception as exc:
            QMessageBox.warning(self, "Load error", str(exc))
            return
        self._fib_image = image
        self._fib_loaded_path = path
        self._fib_path.setText(path)  # setText → textChanged only, no reload
        h, w = image.data.shape[:2]
        self._lbl_fib_shape.setText(f"{h} × {w}")
        px = getattr(
            getattr(getattr(image, "metadata", None), "pixel_size", None), "x", None
        )
        self._lbl_fib_px.setText(f"{px * 1e9:.2f} nm" if px else "—")
        self.fib_image_changed.emit(image)

    def _on_fm_path_edited(self) -> None:
        path = self._fm_path.text().strip()
        if path and path != self._fm_loaded_path:
            self._load_fm(path)

    def _load_fm(self, path: str) -> None:
        try:
            image = FluorescenceImage.load(path)
        except Exception as exc:
            QMessageBox.warning(self, "Load error", str(exc))
            return
        self._fm_image = image
        self._fm_loaded_path = path
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
        self._fib_loaded_path = filename
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
        self._fm_loaded_path = filename
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
    fm_surface_list: CoordinateListWidget

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

        # Surface (max 1, FIB image; mutually exclusive with FM Surface)
        self.surface_list = CoordinateListWidget(point_type=PointType.SURFACE)
        self._surface_panel = TitledPanel("Surface (FIB)", collapsible=True)
        self._surface_panel.set_content(self.surface_list)
        self._surface_count_label = QLabel("(0)")
        self._surface_count_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self._surface_panel.add_header_widget(self._surface_count_label)
        layout.addWidget(self._surface_panel)

        # FM Surface (max 1, FM volume; mutually exclusive with Surface)
        self.fm_surface_list = CoordinateListWidget(point_type=PointType.SURFACE_FM)
        self._fm_surface_panel = TitledPanel("Surface (FM)", collapsible=True)
        self._fm_surface_panel.set_content(self.fm_surface_list)
        self._fm_surface_count_label = QLabel("(0)")
        self._fm_surface_count_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self._fm_surface_panel.add_header_widget(self._fm_surface_count_label)
        layout.addWidget(self._fm_surface_panel)

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
        self._fm_surface_count_label.setText(
            f"({len(self.fm_surface_list.coordinates)})"
        )

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
    """Refractive-index depth correction.

    Two modes, implied by which surface point exists (mutually exclusive):

    post — FIB surface: corrects the correlated POI 1 in FIB image space,
           in-place on the existing result.
    pre  — FM surface: corrects the z of every input POI (FM space). The
           correction is applied inside ``run_correlation_from_data``, so
           applying it triggers (or requires) a correlation run.
    """

    correction_applied = pyqtSignal(object)             # CorrelationResult (post)
    pre_correction_requested = pyqtSignal(float, bool)  # factor, rerun (pre)

    _POST_HEADERS = ["POI", "X (px)", "Y original (px)", "Y corrected (px)"]
    _PRE_HEADERS = ["POI", "X (px)", "Z original (px)", "Z corrected (px)"]

    _WARNING_STYLES = {
        "error":   "color: #e07b39; font-size: 11px;",
        "success": "color: #6dbf6d; font-size: 11px;",
        "armed":   "color: #e0c060; font-size: 11px;",
    }

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._poi: List[CorrelationPointOfInterest] = []
        self._surface_coordinate: Optional[Coordinate] = None
        self._surface_y: Optional[float] = None
        self._fm_surface_coordinate: Optional[Coordinate] = None
        self._input_poi: List[Coordinate] = []
        self._input_pre_factor: Optional[float] = None
        self._fm_pixel_size_z: Optional[float] = None
        self._result: Optional[CorrelationResult] = None
        # Last factor mirrored into the spinbox; mirroring only on change keeps
        # in-progress manual edits from being reverted by unrelated refreshes.
        self._last_mirrored_factor: Optional[float] = None
        self._setup_ui()

    def _set_warning(self, text: str, level: str = "error") -> None:
        """Set the warning label text and its matching severity color together."""
        self._lbl_warning.setStyleSheet(self._WARNING_STYLES[level])
        self._lbl_warning.setText(text)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        self._lbl_mode = QLabel("")
        self._lbl_mode.setStyleSheet("color: #a0c8ff; font-size: 11px;")
        self._lbl_mode.setWordWrap(True)
        self._lbl_mode.setVisible(False)
        layout.addWidget(self._lbl_mode)

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

        self._chk_rerun = QCheckBox("Re-run on apply")
        self._chk_rerun.setChecked(True)
        self._chk_rerun.setToolTip(
            "Re-run the correlation immediately when applying the correction. "
            "If unchecked, the factor is stored and applied on the next run."
        )
        self._chk_rerun.setVisible(False)
        apply_layout.addWidget(self._chk_rerun)

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
        self._table.setHorizontalHeaderLabels(self._POST_HEADERS)
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

    @property
    def mode(self) -> Optional[str]:
        """'pre' when an FM surface exists, 'post' for a FIB surface, else None."""
        if self._fm_surface_coordinate is not None:
            return "pre"
        if self._surface_coordinate is not None:
            return "post"
        return None

    def set_result(
        self,
        result: Optional[CorrelationResult],
        input_data: Optional[CorrelationInputData] = None,
        fm_pixel_size_z: Optional[float] = None,
    ) -> None:
        self._result = result
        self._poi = result.poi if result else []
        surface = input_data.surface_coordinate if input_data else None
        self._surface_coordinate = surface
        self._surface_y = surface.point.y if surface else None
        self._fm_surface_coordinate = (
            input_data.fm_surface_coordinate if input_data else None
        )
        self._input_poi = list(input_data.poi_coordinates) if input_data else []
        self._input_pre_factor = (
            input_data.ri_pre_correction_factor if input_data else None
        )
        self._fm_pixel_size_z = fm_pixel_size_z

        mode = self.mode
        logging.debug(
            f"[RITab.set_result] result={result is not None}, mode={mode}, "
            f"surface_y={self._surface_y}, "
            f"fm_surface={self._fm_surface_coordinate}, "
            f"n_poi={len(self._poi)}"
        )

        # Mode banner + pre-mode controls
        if mode == "pre":
            self._lbl_mode.setText(
                "FM surface mode: corrects the z of every input POI before "
                "correlation (applied during the run)."
            )
        elif mode == "post":
            self._lbl_mode.setText(
                "FIB surface mode: corrects the correlated POI 1 in the FIB image."
            )
        self._lbl_mode.setVisible(mode is not None)
        self._chk_rerun.setVisible(mode == "pre")
        self._ri_widget.set_tilt_locked(mode == "pre")

        self._table.setHorizontalHeaderLabels(
            self._PRE_HEADERS if mode == "pre" else self._POST_HEADERS
        )
        self._table.setRowCount(0)
        self._lbl_multi_poi.setVisible(mode == "post" and len(self._poi) > 1)
        self._update_distance_label()

        result_factor = result.refractive_index_correction_factor if result else None
        result_mode = result.refractive_index_correction_mode if result else None

        # Factor spinbox mirrors the authoritative stored factor. In pre mode a
        # newly armed input factor is the latest user intent and outranks the
        # (older) result factor. Mirror only when the stored value changes so
        # a factor the user is typing survives unrelated data_changed refreshes.
        if mode == "pre" and self._input_pre_factor is not None:
            stored_factor = self._input_pre_factor
        else:
            stored_factor = result_factor
        if stored_factor is not None and stored_factor != self._last_mirrored_factor:
            self._ri_widget.set_factor(stored_factor)
        self._last_mirrored_factor = stored_factor

        # Pre-mode preview table (stored factor, applied or armed)
        if mode == "pre" and self._input_pre_factor is not None:
            self._populate_pre_table(self._input_pre_factor)

        armed = (
            mode == "pre"
            and self._input_pre_factor is not None
            and self._input_pre_factor != result_factor
        )
        if armed:
            self._set_warning(
                f"Factor {self._input_pre_factor:.3f} stored — applied on the next run.",
                level="armed",
            )
        elif result_factor is not None:
            self._set_warning(
                f"Correction applied ({result_mode or 'post'}, factor: {result_factor:.3f}).",
                level="success",
            )
        elif not self._poi:
            self._set_warning("No POI in result.")
        elif mode is None:
            self._set_warning("No surface coordinate — correction unavailable.")
        else:
            self._set_warning("")

    def _update_distance_label(self) -> None:
        mode = self.mode
        if mode == "pre":
            if self._fm_surface_coordinate is None or not self._input_poi:
                self._lbl_distance.setVisible(False)
                return
            dz = abs(self._input_poi[0].point.z - self._fm_surface_coordinate.point.z)
            text = f"Surface → POI 1 depth: {dz:.1f} slices"
            if self._fm_pixel_size_z:
                text += f" ({dz * self._fm_pixel_size_z * 1e6:.2f} µm)"
            self._lbl_distance.setText(text)
            self._lbl_distance.setVisible(True)
            return
        if self._surface_y is None or not self._poi:
            self._lbl_distance.setVisible(False)
            return
        dist_px = abs(self._poi[0].image_px.y - self._surface_y)
        self._lbl_distance.setText(f"Surface → POI depth: {dist_px:.1f} px")
        self._lbl_distance.setVisible(True)

    def _populate_pre_table(self, factor: float) -> None:
        if self._fm_surface_coordinate is None:
            return
        surface_z = self._fm_surface_coordinate.point.z
        self._table.setRowCount(len(self._input_poi))
        for i, coord in enumerate(self._input_poi):
            z = coord.point.z
            corrected_z = scale_about_surface(z, surface_z, factor)
            self._table.setItem(i, 0, _ro_item(f"POI {i + 1}"))
            self._table.setItem(i, 1, _ro_item(f"{coord.point.x:.2f}"))
            self._table.setItem(i, 2, _ro_item(f"{z:.2f}"))
            self._table.setItem(i, 3, _ro_item(f"{corrected_z:.2f}"))

    def _apply(self) -> None:
        if self.mode == "pre":
            self._apply_pre()
        else:
            self._apply_post()

    def _apply_pre(self) -> None:
        if self._fm_surface_coordinate is None:
            self._set_warning("No FM surface coordinate — cannot apply correction.")
            return
        if not self._input_poi:
            self._set_warning("No POI coordinates to correct.")
            return
        self._set_warning("")
        factor = self._ri_widget.get_factor()
        rerun = self._chk_rerun.isChecked()
        logging.info(
            f"[RITab._apply_pre] factor={factor:.4f}, "
            f"surface_z={self._fm_surface_coordinate.point.z:.2f}, "
            f"n_poi={len(self._input_poi)}, rerun={rerun}"
        )
        # The parent stores the factor and emits data_changed, which refreshes
        # this tab (preview table included) via set_result.
        self.pre_correction_requested.emit(factor, rerun)

    def _apply_post(self) -> None:
        if not self._poi:
            self._set_warning("No POI available.")
            return
        if self._surface_y is None or self._surface_coordinate is None:
            self._set_warning("No surface coordinate — cannot apply correction.")
            return
        if (
            self._result is not None
            and self._result.refractive_index_correction_factor is not None
        ):
            # Guard against double-correcting (post-on-post or post-on-pre)
            self._set_warning(
                "Correction already applied to this result — run correlation again first."
            )
            return
        if self._result is not None and self._result.input_data is None:
            # e.g. a result JSON saved with input_data: null
            self._set_warning(
                "Loaded result has no input data — cannot apply correction."
            )
            return
        self._set_warning("")
        factor = self._ri_widget.get_factor()
        surface_y = self._surface_y

        logging.info(
            f"[RITab._apply_post] factor={factor:.4f}, surface_y={surface_y:.2f}, n_poi={len(self._poi)}"
        )

        # Populate table (all POIs shown for reference)
        self._table.setRowCount(len(self._poi))
        for i, poi in enumerate(self._poi):
            depth = poi.image_px.y - surface_y
            corrected_y = scale_about_surface(poi.image_px.y, surface_y, factor)
            logging.info(
                f"[RITab._apply_post] POI {i + 1}: original_y={poi.image_px.y:.2f}, "
                f"depth={depth:.2f}, corrected_y={corrected_y:.2f}"
            )
            self._table.setItem(i, 0, _ro_item(f"POI {i + 1}"))
            self._table.setItem(i, 1, _ro_item(f"{poi.image_px.x:.2f}"))
            self._table.setItem(i, 2, _ro_item(f"{poi.image_px.y:.2f}"))
            self._table.setItem(i, 3, _ro_item(f"{corrected_y:.2f}"))

        if self._result is None:
            logging.warning("[RITab._apply_post] No result to update with correction.")
            return

        # apply surface_y to the result for internal consistency (in case it wasn't set before)
        self._result.input_data.surface_coordinate = self._surface_coordinate # type: ignore

        # Update POI 0 in the result and propagate (reads input_data internally)
        self._result.apply_refractive_index_correction(factor)
        logging.info("[RITab._apply_post] correction applied, emitting correction_applied")
        self.correction_applied.emit(self._result)


# ---------------------------------------------------------------------------
# Point-type registry
# ---------------------------------------------------------------------------


# Which canvas each point type lives on. Single source of truth: the canvas
# right-click allow-lists and the registry's adapter binding both derive from
# this map (declaration order = add-menu order).
_POINT_TYPE_SIDES: Dict[PointType, str] = {
    PointType.FIB: "fib",
    PointType.SURFACE: "fib",
    PointType.FM: "fm",
    PointType.POI: "fm",
    PointType.SURFACE_FM: "fm",
}


class _CanvasAdapter:
    """Thin seam over a point-display surface (canvas or display widget).

    Outbound canvas calls from the registry-driven handlers all go through
    this adapter. NOTE: inbound signals (point_selected/moved/removed/
    add_requested) still connect to the canvases directly and carry
    identity-based Coordinate payloads — migrating onto the shared canvas
    stack (fibsem.ui.widgets.canvas, PR #111, index-based PointOverlay
    signals) therefore means new adapters PLUS an inbound translation layer;
    what stays untouched is the registry, exclusivity, and lifecycle logic.
    """

    def __init__(
        self,
        surface,
        side: str,
        z_provider: Optional[Callable[[], float]] = None,
    ) -> None:
        self._surface = surface
        self.side = side  # "fib" | "fm"
        self._z_provider = z_provider

    def set_coordinates(self, coords: List[Coordinate]) -> None:
        self._surface.set_coordinates(coords)

    def set_selected(self, coord: Optional[Coordinate]) -> None:
        self._surface.set_selected(coord)

    def refresh_coordinate(self, coord: Coordinate) -> None:
        self._surface.refresh_coordinate(coord)

    def current_z(self) -> float:
        """z for newly added points (FM: current slice; FIB: 0)."""
        return float(self._z_provider()) if self._z_provider is not None else 0.0


@dataclass(frozen=True, eq=False)
class _PointTypeSpec:
    """Registry entry driving all per-point-type canvas/list plumbing.

    Adding a point type = one _POINT_TYPE_SIDES entry + one spec (plus its
    list panel); the canvas add-menus, generic handlers, selection clearing,
    exclusivity, axis maxima, and refit routing all follow from the registry.
    Unregistered point types fail loudly (KeyError) instead of being silently
    misrouted, and inconsistent specs are rejected at construction.
    """

    point_type: PointType
    list_widget: CoordinateListWidget
    adapter: _CanvasAdapter
    max_one: bool = False                            # replace-on-add (surfaces)
    exclusive_group: Optional[str] = None            # mutually exclusive specs
    fm_fit_role: Optional[str] = None                # "fid" | "poi" → refit combos
    on_cleared: Optional[Callable[[], None]] = None  # fired when the spec's
                                                     # last point is removed

    def __post_init__(self) -> None:
        expected_side = _POINT_TYPE_SIDES.get(self.point_type)
        if self.adapter.side != expected_side:
            raise ValueError(
                f"{self.point_type}: adapter side {self.adapter.side!r} does not "
                f"match _POINT_TYPE_SIDES ({expected_side!r})"
            )
        if (self.adapter.side == "fm") != (self.fm_fit_role is not None):
            raise ValueError(
                f"{self.point_type}: fm_fit_role must be set for FM-side specs "
                f"and None for FIB-side specs (got {self.fm_fit_role!r})"
            )


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
        # Pre-correlation RI factor (FM surface mode); set via the RI tab Apply
        self._ri_pre_correction_factor: Optional[float] = None

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
        self._action_show_legend = QAction("Show Legend", self)
        self._action_show_legend.setCheckable(True)
        self._action_show_legend.setChecked(True)
        view_menu.addAction(self._action_reset_views)
        view_menu.addAction(self._action_show_scalebar)
        view_menu.addAction(self._action_show_legend)

        test_menu = menubar.addMenu("Test")
        self._action_test_save_plot = QAction("Test Save Plot", self)
        test_menu.addAction(self._action_test_save_plot)

        layout.addWidget(menubar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, stretch=1)

        # Left: FIB image canvas (add-menu types derived from the registry map)
        self._fib_canvas = ImagePointCanvas(
            allowed_point_types=self._point_types_for_side("fib"),
        )
        splitter.addWidget(self._fib_canvas)

        # Middle: FM image display
        self._fm_display = FMImageDisplayWidget(
            allowed_point_types=self._point_types_for_side("fm"),
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

        self._build_point_registry()

    @staticmethod
    def _point_types_for_side(side: str) -> List[PointType]:
        return [pt for pt, s in _POINT_TYPE_SIDES.items() if s == side]

    def _build_point_registry(self) -> None:
        """One _PointTypeSpec per point type drives all canvas/list plumbing."""
        self._fib_adapter = _CanvasAdapter(self._fib_canvas, side="fib")
        self._fm_adapter = _CanvasAdapter(
            self._fm_display,
            side="fm",
            z_provider=lambda: self._fm_display.current_z,
        )
        self._adapters: Dict[str, _CanvasAdapter] = {
            "fib": self._fib_adapter,
            "fm": self._fm_adapter,
        }
        adapter_for = lambda pt: self._adapters[_POINT_TYPE_SIDES[pt]]  # noqa: E731

        cl = self._coords_tab
        specs = (
            _PointTypeSpec(PointType.FIB, cl.fib_list, adapter_for(PointType.FIB)),
            _PointTypeSpec(
                PointType.SURFACE, cl.surface_list, adapter_for(PointType.SURFACE),
                max_one=True, exclusive_group="surface",
            ),
            _PointTypeSpec(
                PointType.FM, cl.fm_list, adapter_for(PointType.FM), fm_fit_role="fid"
            ),
            _PointTypeSpec(
                PointType.POI, cl.poi_list, adapter_for(PointType.POI),
                fm_fit_role="poi",
            ),
            _PointTypeSpec(
                PointType.SURFACE_FM, cl.fm_surface_list,
                adapter_for(PointType.SURFACE_FM),
                max_one=True, exclusive_group="surface", fm_fit_role="fid",
                on_cleared=self._clear_pre_correction_factor,
            ),
        )
        self._point_specs: Dict[PointType, _PointTypeSpec] = {
            spec.point_type: spec for spec in specs
        }
        if set(self._point_specs) != set(_POINT_TYPE_SIDES):
            missing = set(_POINT_TYPE_SIDES) ^ set(self._point_specs)
            raise ValueError(
                f"Point-type registry and _POINT_TYPE_SIDES disagree on: {missing}"
            )

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
        self._ri_tab.pre_correction_requested.connect(self._on_pre_correction_requested)

        # Canvas → list (registry-driven; handlers resolve the spec by type)
        for canvas in (self._fib_canvas, self._fm_display):
            canvas.point_selected.connect(self._on_canvas_selected)
            canvas.point_moved.connect(self._on_canvas_moved)
            canvas.point_removed.connect(self._on_canvas_removed)
            canvas.point_add_requested.connect(self._on_canvas_add_requested)

        # List → canvas (one identical wiring block per spec)
        for spec in self._point_specs.values():
            lw = spec.list_widget
            lw.coordinate_selected.connect(partial(self._on_list_selected, spec))
            lw.coordinate_changed.connect(partial(self._on_list_changed, spec))
            lw.coordinate_removed.connect(partial(self._on_list_removed, spec))
            lw.order_changed.connect(partial(self._on_list_reordered, spec))
            lw.refit_requested.connect(self._on_refit_requested)

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
        self._action_show_legend.toggled.connect(self._on_legend_toggled)
        self._action_test_save_plot.triggered.connect(lambda _: self.save_plot())

        # Bottom bar run button
        self._btn_run.clicked.connect(self._run)
        self._btn_continue.clicked.connect(self._on_continue_pressed)
        self.data_changed.connect(self._update_run_button)
        self.data_changed.connect(self._on_data_changed)

    def _on_data_changed(self, data: CorrelationInputData) -> None:
        self._ri_tab.set_result(
            self._result,
            input_data=data,
            fm_pixel_size_z=self._fm_pixel_size_z(),
        )

    def _fm_pixel_size_z(self) -> Optional[float]:
        if self._fm_image is None:
            return None
        return getattr(self._fm_image.metadata, "pixel_size_z", None)

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
        for spec in self._point_specs.values():
            if spec.adapter is self._fib_adapter:
                spec.list_widget.set_axis_maxima(x_max=w - 1, y_max=h - 1)
        self._images_tab.set_fib_image(fib_image)

    def set_fm_image(self, fm_image: FluorescenceImage) -> None:
        """Load FM image into canvas and update images tab."""
        self._fm_image = fm_image
        self._fm_display.set_fm_image(fm_image)
        px = self._effective_fm_pixel_size(fm_image)
        if px:
            self._fm_display.canvas.set_pixel_size(px)
        _, n_z, h, w = fm_image.data.shape
        for spec in self._point_specs.values():
            if spec.adapter is self._fm_adapter:
                spec.list_widget.set_axis_maxima(
                    x_max=w - 1, y_max=h - 1, z_max=n_z - 1
                )
        self._coords_tab.rebuild_channel_combos(fm_image)
        self._images_tab.set_fm_image(fm_image)

    @staticmethod
    def _effective_fm_pixel_size(fm_image: FluorescenceImage) -> Optional[float]:
        """Pixel size (m) for the FM scalebar, corrected for any display resize.

        ``metadata.pixel_size_x`` describes one pixel at the acquisition
        resolution (``metadata.resolution``). If the displayed data array was
        resized (e.g. binned/downscaled) without rewriting the metadata, the
        two disagree and the raw value over/under-scales the scalebar by the
        resize ratio — so scale it to the displayed width. A no-op when the
        metadata matches the data (the correctly-authored case).
        """
        meta = getattr(fm_image, "metadata", None)
        px = getattr(meta, "pixel_size_x", None)
        if not px:
            return None
        data_w = fm_image.data.shape[3]
        res = getattr(meta, "resolution", None)
        acq_w = res[0] if res else None
        if acq_w and data_w and acq_w != data_w:
            corrected = px * acq_w / data_w
            logging.warning(
                "FM scalebar: data width %d ≠ metadata resolution %d — pixel "
                "size corrected %.1f→%.1f nm/px for the display resize.",
                data_w, acq_w, px * 1e9, corrected * 1e9,
            )
            return corrected
        return px

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
        fm_surf_coords = (
            [data.fm_surface_coordinate]
            if data.fm_surface_coordinate is not None
            else []
        )
        if surf_coords and fm_surf_coords:
            # Only one surface point is allowed — prefer the FM surface
            logging.warning(
                "Both FIB and FM surface coordinates present — keeping the FM surface."
            )
            surf_coords = []

        cl = self._coords_tab
        cl.fib_list.coordinates = fib_coords
        cl.fm_list.coordinates = fm_coords
        cl.poi_list.coordinates = poi_coords
        cl.surface_list.coordinates = surf_coords
        cl.fm_surface_list.coordinates = fm_surf_coords
        for adapter in self._adapters.values():
            self._refresh_canvas(adapter)
        cl.update_headers()

        # A factor without an FM surface is meaningless — don't arm it
        self._ri_pre_correction_factor = (
            data.ri_pre_correction_factor
            if data.fm_surface_coordinate is not None
            else None
        )

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
        fm_surf = cl.fm_surface_list.coordinates
        return CorrelationInputData(
            fib_image=self._fib_image,
            fm_image=self._fm_image,
            fib_coordinates=cl.fib_list.coordinates,
            fm_coordinates=cl.fm_list.coordinates,
            poi_coordinates=cl.poi_list.coordinates,
            surface_coordinate=surf[0] if surf else None,
            fm_surface_coordinate=fm_surf[0] if fm_surf else None,
            ri_pre_correction_factor=self._ri_pre_correction_factor,
        )

    @property
    def result(self) -> Optional[CorrelationResult]:
        return self._result

    # ------------------------------------------------------------------
    # Canvas refresh helpers
    # ------------------------------------------------------------------

    def _refresh_canvas(self, adapter: _CanvasAdapter) -> None:
        """Push the full coordinate set of every spec shown on this canvas."""
        coords: List[Coordinate] = []
        for spec in self._point_specs.values():
            if spec.adapter is adapter:
                coords += spec.list_widget.coordinates
        adapter.set_coordinates(coords)

    def _select_only(self, spec: _PointTypeSpec, coord: Optional[Coordinate]) -> None:
        """Make coord the sole selection: clear every other list and canvas."""
        for other in self._point_specs.values():
            if other is not spec:
                other.list_widget.select_coordinate_silent(None)
        for adapter in self._adapters.values():
            adapter.set_selected(coord if adapter is spec.adapter else None)

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
            logging.exception("Auto-save of correlation data failed")

    def _auto_save_result(self, result: CorrelationResult) -> None:
        if not self._project_dir:
            return
        try:
            result.save(os.path.join(self._project_dir, "correlation_result.json"))
        except Exception:
            logging.exception("Auto-save of correlation result failed")

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
        self._ri_tab.set_result(
            result, input_data=self.data, fm_pixel_size_z=self._fm_pixel_size_z()
        )
        self._tabs.setTabEnabled(3, True)
        self._overlay_result_on_fib(result)
        self._btn_continue.setEnabled(True)
        # after _update_run_button, which would otherwise overwrite it with "Ready."
        self._update_run_button()
        if (
            result.refractive_index_correction_mode == "pre"
            and result.refractive_index_correction_factor is not None
        ):
            msg = (
                f"Done — RI pre-correction ×{result.refractive_index_correction_factor:.3f} applied"
            )
            shift = self._poi_shift_px(result)
            if shift is not None:
                msg += f", POI 1 shifted {shift:.1f} px"
            self._lbl_status.setText(msg + ".")
        else:
            self._lbl_status.setText("Done.")
        self.result_changed.emit(result)

    @staticmethod
    def _poi_shift_px(result: CorrelationResult) -> Optional[float]:
        """Distance (px) between POI 1 and its uncorrected ghost, if present."""
        if not (result.poi and result.poi_uncorrected):
            return None
        return float(
            np.hypot(
                result.poi[0].image_px.x - result.poi_uncorrected[0].image_px.x,
                result.poi[0].image_px.y - result.poi_uncorrected[0].image_px.y,
            )
        )

    def _on_correction_applied(self, result: CorrelationResult) -> None:
        self._result = result
        self._overlay_result_on_fib(result)
        factor = result.refractive_index_correction_factor
        shift = self._poi_shift_px(result)
        if factor is not None and shift is not None:
            self._lbl_status.setText(
                f"RI post-correction ×{factor:.3f} applied, POI 1 shifted {shift:.1f} px."
            )
        self.result_changed.emit(result)

    def _on_pre_correction_requested(self, factor: float, rerun: bool) -> None:
        """Store the pre-correlation RI factor and optionally re-run."""
        self._ri_pre_correction_factor = factor
        self.data_changed.emit(self.data)  # auto-save + RI tab refresh
        if rerun:
            self._run()
        else:
            self._lbl_status.setText(
                "Pre-correction factor stored — run correlation to apply."
            )

    def _clear_pre_correction_factor(self) -> None:
        """The pre-correction factor's lifecycle is tied to the FM surface point:
        removing the surface disarms the factor so it cannot silently re-apply
        on a later run."""
        self._ri_pre_correction_factor = None

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
                legend_label="FM reprojected (E)",
            )

        # Ghost: where the POI would land without the RI pre-correction —
        # hollow magenta ring, unlabeled, larger than the corrected marker so it
        # stays visible even when the shift is small and the markers overlap
        ghost_pts = [(p.image_px.x, p.image_px.y) for p in result.poi_uncorrected]
        if ghost_pts:
            self._fib_canvas.add_overlay_points(
                ghost_pts,
                color="#ff00ff",
                size=13,
                marker="o",
                alpha=0.7,
                show_labels=False,
                hollow=True,
                legend_label="POI uncorrected",
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
                legend_label="POI (P)",
            )

    # ------------------------------------------------------------------
    # Canvas → list slots
    # ------------------------------------------------------------------

    def _on_canvas_selected(self, coord: Coordinate) -> None:
        spec = self._point_specs[coord.point_type]
        spec.list_widget.select_coordinate_silent(coord)
        self._select_only(spec, coord)

    def _on_canvas_moved(self, coord: Coordinate) -> None:
        spec = self._point_specs[coord.point_type]
        spec.list_widget.refresh_coordinate(coord)
        self.data_changed.emit(self.data)

    def _on_canvas_removed(self, coord: Coordinate) -> None:
        spec = self._point_specs[coord.point_type]
        spec.list_widget.coordinates = [
            c for c in spec.list_widget.coordinates if c is not coord
        ]
        if spec.on_cleared is not None and not spec.list_widget.coordinates:
            spec.on_cleared()
        self._refresh_canvas(spec.adapter)
        self._coords_tab.update_headers()
        self.data_changed.emit(self.data)

    def _on_canvas_add_requested(self, x: float, y: float, pt: PointType) -> None:
        spec = self._point_specs[pt]
        coord = Coordinate(PointXYZ(x, y, spec.adapter.current_z()), pt)
        if spec.max_one:
            spec.list_widget.coordinates = [coord]
            self._clear_exclusive_siblings(spec)
        else:
            spec.list_widget.add_coordinate(coord)
        self._refresh_canvas(spec.adapter)
        self._coords_tab.update_headers()
        self.data_changed.emit(self.data)

    def _clear_exclusive_siblings(self, spec: _PointTypeSpec) -> None:
        """Enforce mutual exclusivity (one surface point at a time): clear the
        other members of the spec's exclusive group and fire their lifecycle
        hooks (e.g. disarming the pre-correction factor)."""
        if spec.exclusive_group is None:
            return
        for other in self._point_specs.values():
            if other is spec or other.exclusive_group != spec.exclusive_group:
                continue
            if other.list_widget.coordinates:
                other.list_widget.coordinates = []
                self._refresh_canvas(other.adapter)
            if other.on_cleared is not None:
                other.on_cleared()

    # ------------------------------------------------------------------
    # List → canvas slots
    # ------------------------------------------------------------------

    def _on_list_selected(self, spec: _PointTypeSpec, coord: Coordinate) -> None:
        self._select_only(spec, coord)

    def _on_list_changed(
        self, spec: _PointTypeSpec, coord: Coordinate, _f: str, _v: float
    ) -> None:
        spec.adapter.refresh_coordinate(coord)
        self.data_changed.emit(self.data)

    def _on_list_removed(self, spec: _PointTypeSpec, _coord: Coordinate) -> None:
        # on_cleared = "the spec's LAST point is gone" (the list widget removes
        # the row before emitting, so the check sees the post-removal state)
        if spec.on_cleared is not None and not spec.list_widget.coordinates:
            spec.on_cleared()
        self._refresh_canvas(spec.adapter)
        self._coords_tab.update_headers()
        self.data_changed.emit(self.data)

    def _on_list_reordered(self, spec: _PointTypeSpec, _coords: list) -> None:
        self._refresh_canvas(spec.adapter)
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

    def _on_legend_toggled(self, visible: bool) -> None:
        self._fib_canvas.set_legend_visible(visible)
        self._fm_display.canvas.set_legend_visible(visible)

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
        self._load_result(result)

    def _load_result(self, result: CorrelationResult) -> None:
        """Adopt a loaded correlation result (and its input data, if any)."""
        # Populate the lists first so the RI tab sees the loaded surface points.
        # _on_result_ready refreshes the run button and sets the final status
        # text itself — no trailing update here, it would overwrite the status.
        if result.input_data:
            self.set_data(result.input_data)
        self._on_result_ready(result)

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
        spec = self._point_specs[coord.point_type]
        x, y, z = coord.point.x, coord.point.y, coord.point.z
        fig = None

        try:
            if spec.adapter.side == "fib":
                method = cl._fib_method_combo.currentText()
                if method == "Hole" and self._fib_image is not None:
                    xr, yr, fig = hole_fitting_FIB(
                        self._fib_image.filtered_data, int(x), int(y)
                    )
                    coord.point.x, coord.point.y = float(xr), float(yr)
            else:
                # The FM surface is typically picked in the reflection channel,
                # so it shares the fiducial method/channel settings.
                is_fid = spec.fm_fit_role == "fid"
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

        spec.list_widget.refresh_coordinate(coord)
        spec.adapter.refresh_coordinate(coord)
        self.data_changed.emit(self.data)

        if fig is not None and cl._show_diag_check.isChecked():
            _show_figure_dialog(fig, title="Refit Diagnostic", parent=self)


# ---------------------------------------------------------------------------
# Dialog wrapper
# ---------------------------------------------------------------------------


class CorrelationTabDialog(QDialog):
    """Modal QDialog wrapping CorrelationTabWidget.

    A self-contained correlation dialog. The Continue button inside the widget
    handles confirmation and
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


def main() -> None:
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(stylesheets.NAPARI_STYLE)

    widget = CorrelationTabWidget()
    widget.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
