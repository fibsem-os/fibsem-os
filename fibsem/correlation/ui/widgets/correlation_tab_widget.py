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
from typing import Callable, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)

import numpy as np
from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QAction,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenuBar,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtGui import QKeySequence
from fibsem.constants import DATETIME_FILE
from fibsem.correlation.correlation_v2 import run_correlation_from_data
from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
    CorrelationPointOfInterest,
    CorrelationState,
    CorrelationResult,
    PointType,
    PointXYZ,
    load_correlation_file,
    scale_about_surface,
)
from fibsem.correlation.ui.widgets.coordinate_list_widget import CoordinateListWidget
from fibsem.correlation.ui.widgets.fit_confirmation_dialog import (
    FitConfirmationDialog,
    FitStatus,
    PointFitResult,
    humanize_fit_error,
)
from fibsem.ui import notification_service, stylesheets
from fibsem.ui.icon import fibsem_icon
from fibsem.correlation.ui.widgets.fm_image_display_widget import (
    IMAGE_HEADER_STYLE,
    FMImageDisplayWidget,
)
from fibsem.correlation.ui.widgets.image_point_canvas import ImagePointCanvas
from fibsem.fm.structures import FluorescenceImage
from fibsem.structures import FibsemImage
from fibsem.ui.widgets.custom_widgets import (
    QDirectoryLineEdit,
    QFileLineEdit,
    TitledPanel,
    ValueComboBox,
)
from fibsem.correlation.ui.widgets.refractive_index_widget import RefractiveIndexWidget

_FIT_METHODS = ["None", "Hole", "Gaussian"]

# Consolidated project file (FIB-264). The two legacy names are still *read* for
# back-compat with projects saved by earlier versions; nothing writes them now.
CORRELATION_FILENAME = "correlation.json"
_LEGACY_DATA_FILENAME = "correlation_data.json"
_LEGACY_RESULT_FILENAME = "correlation_result.json"

# Auto-accept outlier fallback: a fit is a refinement of the user's click, so a
# large jump usually means it locked onto the wrong feature. Beyond these, the
# confirm dialog is shown even in auto-accept mode. Generous — the goal is to
# catch gross mis-locks, not sub-pixel disagreement.
_AUTO_ACCEPT_MAX_XY_PX = 25.0  # XY displacement (pixels)
_AUTO_ACCEPT_MAX_Z = 5.0       # Z displacement (slices)

# Run-bar RMS cue. The badge FLAGS problems; it never certifies quality. The RMS
# is a fit residual over the fiducials, so a low value cannot promise the POI —
# which is extrapolated from that fit — lands where you want it. A green light
# would read as exactly that promise, so there is no green: neutral means "no
# detected problem", not "good".
#
# The two relative triggers need no calibration and hold at any HFW. Only the
# absolute limits carry numbers, and they are set far enough out to be statements
# of breakage rather than judgements of quality (500 nm is ~7.7 px at 100 µm HFW
# on a 1536 px-wide image; 1 µm is ~15 px).
_RMS_LARGE_NM = 500.0
_RMS_BROKEN_NM = 1000.0
_RMS_OUTLIER_RATIO = 2.0
# Fewest fiducial pairs the run button accepts. The rigid 3D->2D transform has
# ~7 degrees of freedom, so a fit on this many is barely over-determined and its
# residual is small by construction rather than by agreement.
_RMS_MIN_FIDUCIALS = 4

_RMS_NEUTRAL = "#9aa0a6"
_RMS_WARN = "#ffb300"
_RMS_BAD = "#e53935"

# Form-row labels: muted and one step below the value they describe.
_FORM_LABEL_COLOR = "#9aa0a6"


def _rms_concern(
    rms_nm: Optional[float],
    n_fiducials: Optional[int] = None,
    worst_ratio: Optional[float] = None,
) -> Tuple[str, Optional[str]]:
    """Flag detectable problems with a fit. Returns (colour, reason or None).

    A residual can demonstrate that a correlation is wrong; it can never
    demonstrate that one is right. So this only ever reports suspicion — the
    neutral case carries no verdict at all.

    The relative checks still apply when ``rms_nm`` is None (a result loaded from
    JSON has no pixel size): they are ratios and counts, not distances.
    """
    if rms_nm is not None and rms_nm > _RMS_BROKEN_NM:
        return _RMS_BAD, (
            f"RMS exceeds {_format_distance_nm(_RMS_BROKEN_NM)} — the fit has not "
            "converged on anything usable."
        )

    reasons: List[str] = []
    if rms_nm is not None and rms_nm > _RMS_LARGE_NM:
        reasons.append(f"RMS is above {_format_distance_nm(_RMS_LARGE_NM)}.")
    if n_fiducials and n_fiducials <= _RMS_MIN_FIDUCIALS:
        reasons.append(
            f"Only {n_fiducials} fiducial pairs — the fit has almost no redundancy, "
            "so this residual understates the true error."
        )
    if worst_ratio is not None and worst_ratio > _RMS_OUTLIER_RATIO:
        reasons.append(
            f"One fiducial is {worst_ratio:.1f}× the RMS — more likely a misplaced "
            "correspondence than general noise."
        )
    return (_RMS_WARN, " ".join(reasons)) if reasons else (_RMS_NEUTRAL, None)


def _format_distance_nm(nm: float) -> str:
    """Render a nanometre distance, switching to µm once it stops being readable."""
    return f"{nm / 1000:.2f} µm" if nm >= 1000 else f"{nm:.0f} nm"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ro_item(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(Qt.ItemFlag.ItemIsEnabled)
    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
    return item


def _form_label(text: str) -> QLabel:
    """A form-row label: small and muted, so the value reads louder than its name.

    ``QFormLayout.addRow("Name:", w)`` builds the label at the default app font
    size, which renders *larger* than the 11-12px values in these panels — the
    field name ends up shouting over its own data. Pass this instead.
    """
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color: {_FORM_LABEL_COLOR}; font-size: 11px;")
    return lbl


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


class _ProgressRelay(QObject):
    """Bounces a worker-thread progress callback onto the GUI thread.

    ``interpolate_fm_volume`` calls ``emit_progress(done, total)`` from the worker
    thread; because this object is created on the GUI thread, emitting its signal
    there delivers it through the GUI event loop, so slots may touch widgets.
    """

    progress = pyqtSignal(int, int)  # (channels_done, channels_total)

    def emit_progress(self, done: int, total: int) -> None:
        self.progress.emit(done, total)


# ---------------------------------------------------------------------------
# Tab 0 — Images
# ---------------------------------------------------------------------------


class _ImagesTab(QWidget):
    """Browse / load FIB and FM image files."""

    fib_image_changed = pyqtSignal(object)  # FibsemImage
    fm_image_changed = pyqtSignal(object)  # FluorescenceImage
    project_dir_changed = pyqtSignal(str)  # directory path
    interpolate_requested = pyqtSignal()  # interpolate the loaded FM z-stack

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
        fib_form.addRow(_form_label("Shape:"), self._lbl_fib_shape)
        fib_form.addRow(_form_label("Pixel size:"), self._lbl_fib_px)
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
        fm_form.addRow(_form_label("Shape (C×Z×Y×X):"), self._lbl_fm_shape)
        fm_form.addRow(_form_label("Channels:"), self._lbl_fm_ch)

        # The interpolate action rides the Z-slices row — it acts on the z axis,
        # so it reads as the action on that number rather than a stray button.
        # (Data transform, deliberately not on the canvas toolbar.)
        z_row = QWidget()
        z_row_layout = QHBoxLayout(z_row)
        z_row_layout.setContentsMargins(0, 0, 0, 0)
        z_row_layout.setSpacing(6)
        z_row_layout.addWidget(self._lbl_fm_z)
        z_row_layout.addStretch(1)
        self._btn_interpolate = QPushButton(" Interpolate…")
        self._btn_interpolate.setIcon(fibsem_icon("mdi:arrow-expand-vertical"))
        self._btn_interpolate.setToolTip(
            "Interpolate the z-stack toward an isotropic voxel size"
        )
        self._btn_interpolate.setEnabled(False)  # enabled once a z-stack is loaded
        self._btn_interpolate.clicked.connect(
            lambda: self.interpolate_requested.emit()
        )
        z_row_layout.addWidget(self._btn_interpolate)
        fm_form.addRow(_form_label("Z-slices:"), z_row)

        fm_layout.addLayout(fm_form)

        # Embedded, non-modal progress — shown only while interpolating.
        self._interp_progress = QProgressBar()
        self._interp_progress.setTextVisible(True)
        self._interp_progress.setVisible(False)
        fm_layout.addWidget(self._interp_progress)

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
            self._fib_path.setText(self._fib_loaded_path)  # revert to last-good path
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
            self._fm_path.setText(self._fm_loaded_path)  # revert to last-good path
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
        # interpolation needs a multi-slice stack with a known z step
        self._btn_interpolate.setEnabled(
            z > 1 and bool(getattr(image.metadata, "pixel_size_z", None))
        )

    def set_interpolating(self, running: bool, n_channels: int = 0) -> None:
        """Reflect an in-flight interpolation: the button locks and an embedded
        progress bar appears. Non-modal — the rest of the GUI stays live."""
        self._btn_interpolate.setEnabled(not running and self._fm_image is not None)
        if running:
            self._interp_progress.setRange(0, max(n_channels, 1))
            self._interp_progress.setValue(0)
            self._interp_progress.setFormat("Interpolating z-stack… %v/%m")
        self._interp_progress.setVisible(running)

    def set_interpolation_progress(self, done: int, total: int) -> None:
        self._interp_progress.setRange(0, max(total, 1))
        self._interp_progress.setValue(done)

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

        # ValueComboBox installs a WheelBlocker, so scrolling this panel can't
        # silently change a fit setting on the way past. The channel combos start
        # empty and are refilled by rebuild_channel_combos — the blocker lives on
        # the widget, so it survives clear()/addItem().
        self._fib_method_combo = ValueComboBox(_FIT_METHODS, value="Hole")
        fit_form.addRow(_form_label("FIB method:"), self._fib_method_combo)

        self._fm_fid_method_combo = ValueComboBox(_FIT_METHODS, value="None")
        fit_form.addRow(_form_label("FM Fid. method:"), self._fm_fid_method_combo)

        self._fm_poi_method_combo = ValueComboBox(_FIT_METHODS, value="Gaussian")
        fit_form.addRow(_form_label("FM POI method:"), self._fm_poi_method_combo)

        self._fm_fid_ch_combo = ValueComboBox([])
        fit_form.addRow(_form_label("FM Fid. channel:"), self._fm_fid_ch_combo)

        self._fm_poi_ch_combo = ValueComboBox([])
        fit_form.addRow(_form_label("FM POI channel:"), self._fm_poi_ch_combo)

        self._show_diag_check = QCheckBox()
        fit_form.addRow(_form_label("Show diagnostic:"), self._show_diag_check)

        # Opt-in: apply fits without the confirm dialog. Off by default (the
        # confirm-first behaviour of FIB-252). Errors and far-off "surprising"
        # fits still surface the dialog — see _on_refit_requested.
        self._auto_accept_check = QCheckBox()
        self._auto_accept_check.setToolTip(
            "Apply fits immediately without the confirm dialog.\n"
            "Failed or far-off fits still ask for confirmation."
        )
        fit_form.addRow(_form_label("Auto-accept fits:"), self._auto_accept_check)

        fit_help = QLabel(
            "Select a point and press <b>F</b> to fit it. Each fit opens a "
            "confirmation to accept or reject — unless <b>Auto-accept</b> is on "
            "(failed or far-off fits still ask)."
        )
        fit_help.setWordWrap(True)
        fit_help.setStyleSheet("color: #8a8d93; font-size: 11px; padding-top: 4px;")
        fit_form.addRow(fit_help)

        self._fit_panel = TitledPanel("Fit Settings", collapsible=True)
        self._fit_panel.set_content(fit_body)
        layout.addWidget(self._fit_panel)

        # Advanced / set-once panels start collapsed to keep the tab compact.
        self._surface_panel.collapse()
        self._fm_surface_panel.collapse()
        self._fit_panel.collapse()

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
        summary_form.addRow(_form_label("Scale:"), self._lbl_scale)
        summary_form.addRow(_form_label("RMS Error:"), self._lbl_rms)
        summary_form.addRow(_form_label("Mean Abs Error:"), self._lbl_mae)
        summary_form.addRow(_form_label("Rotation:"), self._lbl_rotation)
        summary_form.addRow(_form_label("Translation:"), self._lbl_trans)
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
        # FM z-stack interpolation (background) — held for the op's lifetime
        self._interp_worker = None
        self._interp_relay: Optional[_ProgressRelay] = None
        self._project_dir: Optional[str] = None
        # Pre-correlation RI factor (FM surface mode); set via the RI tab Apply
        self._ri_pre_correction_factor: Optional[float] = None

        self._setup_ui()
        self._connect_signals()
        self._setup_shortcuts()
        self._set_result_live(False)

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
        # One action for coordinates + result: the file dispatches on its shape
        # (FIB-264), so there is no wrong choice to make — removing the FIB-263
        # misclick by construction rather than guarding against it.
        self._action_load = QAction("Load Correlation…", self)
        self._action_save = QAction("Save Correlation…", self)

        file_menu.addAction(self._action_load_fib)
        file_menu.addAction(self._action_load_fm)
        file_menu.addSeparator()
        file_menu.addAction(self._action_load)
        file_menu.addAction(self._action_save)
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
        self._action_show_labels = QAction("Show Labels", self)
        self._action_show_labels.setCheckable(True)
        self._action_show_labels.setChecked(True)
        self._action_save_plot = QAction("Save Plot", self)
        view_menu.addAction(self._action_reset_views)
        view_menu.addAction(self._action_show_scalebar)
        view_menu.addAction(self._action_show_legend)
        view_menu.addAction(self._action_show_labels)
        view_menu.addSeparator()
        view_menu.addAction(self._action_save_plot)

        layout.addWidget(menubar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, stretch=1)

        # Left: FIB image canvas with a filename header (mirrors the FM display)
        fib_pane = QWidget()
        fib_layout = QVBoxLayout(fib_pane)
        fib_layout.setContentsMargins(0, 0, 0, 0)
        fib_layout.setSpacing(0)

        self._fib_name_label = QLabel("")
        self._fib_name_label.setStyleSheet(IMAGE_HEADER_STYLE)
        self._fib_name_label.setTextFormat(Qt.TextFormat.PlainText)
        self._fib_name_label.setVisible(False)
        fib_layout.addWidget(self._fib_name_label)

        self._fib_canvas = ImagePointCanvas(
            allowed_point_types=self._point_types_for_side("fib"),
        )
        fib_layout.addWidget(self._fib_canvas, stretch=1)
        splitter.addWidget(fib_pane)

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

        # Compact result summary beside Continue (RMS quality-coloured + RI/POI),
        # shown after a run — keeps the status line free for state only.
        self._lbl_result = QLabel("")
        self._lbl_result.setTextFormat(Qt.TextFormat.RichText)
        self._lbl_result.setStyleSheet("color: #9aa0a6; font-size: 12px;")
        self._lbl_result.setVisible(False)
        btn_layout.addWidget(self._lbl_result)
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

        # Auto-save — one file, rewritten whole on either signal (FIB-264)
        self.data_changed.connect(self._auto_save_state)
        self.result_changed.connect(self._auto_save_state)

        # RI correction
        self._ri_tab.correction_applied.connect(self._on_correction_applied)
        self._ri_tab.pre_correction_requested.connect(self._on_pre_correction_requested)

        # Canvas → list (registry-driven; handlers resolve the spec by type)
        for canvas in (self._fib_canvas, self._fm_display):
            canvas.point_selected.connect(self._on_canvas_selected)
            canvas.point_moved.connect(self._on_canvas_moved)
            canvas.point_removed.connect(self._on_canvas_removed)
            canvas.point_add_requested.connect(self._on_canvas_add_requested)

        # FM z-stack interpolation (entry point lives in the Images tab)
        self._images_tab.interpolate_requested.connect(self._on_interpolate_fm)

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
        self._action_load.triggered.connect(self._menu_load_correlation)
        self._action_save.triggered.connect(self._on_save)
        self._action_export_csv.triggered.connect(lambda _: self._menu_export_csv())
        self._action_reset_views.triggered.connect(lambda _: self._reset_views())
        self._action_show_scalebar.toggled.connect(self._on_scalebar_toggled)
        self._on_scalebar_toggled(True)
        self._action_show_legend.toggled.connect(self._on_legend_toggled)
        self._action_show_labels.toggled.connect(self._on_labels_toggled)
        self._action_save_plot.triggered.connect(lambda _: self._on_save_plot_clicked())

        # Bottom bar run button
        self._btn_run.clicked.connect(self._run)
        self._btn_continue.clicked.connect(self._on_continue_pressed)
        self.data_changed.connect(self._update_run_button)
        self.data_changed.connect(self._on_data_changed)

    def _set_result_live(self, live: bool) -> None:
        """Reflect whether the displayed result still describes the current points.

        Continue commits ``result.poi[0].px_m`` to the protocol editor, so it must
        not stay armed once an edit has invalidated the fit. The RMS badge and the
        Run/Continue emphasis answer that same question, so they move together
        here rather than drifting apart across handlers.
        """
        self._btn_continue.setEnabled(live)
        self._btn_continue.setStyleSheet(
            stylesheets.PRIMARY_BUTTON_STYLESHEET
            if live
            else stylesheets.SECONDARY_BUTTON_STYLESHEET
        )
        self._btn_run.setStyleSheet(
            stylesheets.SECONDARY_BUTTON_STYLESHEET
            if live
            else stylesheets.PRIMARY_BUTTON_STYLESHEET
        )
        if not live:
            self._lbl_result.setVisible(False)  # text is set again by a fresh run

    def _on_data_changed(self, data: CorrelationInputData) -> None:
        # Any coordinate edit invalidates the last run: the transform no longer
        # fits the points it is displayed against.
        self._set_result_live(False)
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
        self._update_fib_name_label(fib_image)

    def _update_fib_name_label(self, image: FibsemImage) -> None:
        """Show the loaded FIB image's filename in the header (mirrors FM)."""
        iset = getattr(getattr(image, "metadata", None), "image_settings", None)
        filename = getattr(iset, "filename", "") or ""
        base = os.path.basename(filename) if filename else ""
        self._fib_name_label.setText(base)
        self._fib_name_label.setToolTip(filename)
        self._fib_name_label.setVisible(bool(base))

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

    def load_correlation(self, path: str) -> None:
        """Load any correlation JSON — consolidated file or either legacy file.

        Dispatches on the file's shape (:func:`load_correlation_file`), so the
        caller never picks the wrong loader — the FIB-263 crash can't recur.
        """
        logging.info("Loading correlation from %s", path)
        self._adopt_state(load_correlation_file(path))

    def _adopt_state(self, state: CorrelationState) -> None:
        """Apply a loaded correlation state: points first, then the result (if any).

        Points are loaded before the result so that, when a result is present,
        its staleness is judged against the freshly-loaded coordinates and its
        own snapshot is not replayed over them — the FIB-295 ordering, now the
        only ordering because both come from one file.
        """
        data = state.input_data
        data.fib_image = self._fib_image
        data.fm_image = self._fm_image
        self.set_data(data)
        self.data_changed.emit(self.data)
        if state.result is not None:
            self._load_result(state.result, adopt_inputs=False)

    def load_data(self, path: str) -> None:
        """Load a legacy coordinates file, preserving current images.

        Retained for the quickstart fallback and as the type-specific loader; new
        code and the File menu use :meth:`load_correlation`.
        """
        logging.info("Loading correlation coordinates from %s", path)
        loaded = CorrelationInputData.load(path)
        loaded.fib_image = self._fib_image
        loaded.fm_image = self._fm_image
        self.set_data(loaded)
        self.data_changed.emit(self.data)

    def save_correlation(self, path: str) -> None:
        """Save the whole correlation state (points + result) to one JSON."""
        CorrelationState(input_data=self.data, result=self._result).save(path)

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

    def _auto_save_state(self, *_) -> None:
        """Persist the whole correlation state to a single correlation.json.

        Wired to both ``data_changed`` and ``result_changed`` — either fires,
        the full project is rewritten from the live widget state, so the two can
        never drift apart on disk (which the split data/result files could;
        FIB-264, FIB-295). The emitted payload is ignored: ``self.data`` and
        ``self._result`` are the source of truth.

        A post-run edit rewrites with the (now-stale) result still attached — by
        design. The result carries its own ``computed_from`` snapshot, so
        staleness stays derivable on load (``matches_inputs``) and the result
        remains available to inspect; it is not silently discarded here.
        """
        if not self._project_dir:
            return
        try:
            CorrelationState(input_data=self.data, result=self._result).save(
                os.path.join(self._project_dir, CORRELATION_FILENAME)
            )
        except Exception:
            logging.exception("Auto-save of correlation project failed")

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
        # a run in flight has no live result yet; a failed run leaves it that way
        self._set_result_live(False)
        self._worker = _CorrelationWorker(copy.deepcopy(self.data))
        self._worker.result_ready.connect(self._on_result_ready)
        self._worker.errored.connect(self._on_run_error)
        self._worker.start()

    def _on_result_ready(self, result: CorrelationResult, live: bool = True) -> None:
        """Adopt a result. ``live`` is False for a *loaded* result that no longer
        describes the current points (FIB-295) — a fresh run is always live."""
        self._result = result
        self._results_tab.set_result(result)
        self._ri_tab.set_result(
            result, input_data=self.data, fm_pixel_size_z=self._fm_pixel_size_z()
        )
        self._tabs.setTabEnabled(3, True)
        self._overlay_result_on_fib(result)
        self._set_result_live(live)
        # after _update_run_button, which would otherwise overwrite it with "Ready."
        self._update_run_button()
        # RMS beside Continue (coloured only if something looks wrong); compact
        # RI / POI note on status. Suppressed for a stale result — an RMS is a
        # statement about points these no longer are.
        rms = result.rms_error
        if rms is not None and live:
            px_m = self._fib_pixel_size_m()
            rms_nm = rms * px_m * 1e9 if px_m else None
            shown = _format_distance_nm(rms_nm) if rms_nm is not None else f"{rms:.2f} px"
            worst = self._worst_fiducial_px(result)
            color, concern = _rms_concern(
                rms_nm,
                len(result.reprojected_3d),
                worst / rms if worst is not None and rms > 0 else None,
            )
            self._lbl_result.setText(
                f'<span style="color:{color}">RMS {shown}</span>'
            )
            self._lbl_result.setToolTip(self._rms_tooltip(result, rms, px_m, concern))
            self._lbl_result.setVisible(True)
        # Staleness outranks the RI note: this is the only line explaining why
        # Continue is greyed out, so "Done — RI ×1.500" must not mask it.
        if not live:
            self._lbl_status.setText(
                "Loaded result — the points have changed since this run. Re-run to update."
            )
        elif (
            result.refractive_index_correction_mode == "pre"
            and result.refractive_index_correction_factor is not None
        ):
            msg = f"Done — RI ×{result.refractive_index_correction_factor:.3f}"
            shift = self._poi_shift_px(result)
            if shift is not None:
                msg += f", POI Δ{shift:.1f} px"
            self._lbl_status.setText(msg)
        else:
            self._lbl_status.setText("Done.")
        self.result_changed.emit(result)

    def _fib_pixel_size_m(self) -> Optional[float]:
        """FIB pixel size in metres, or None. A result loaded from JSON has no
        images restored, so this is legitimately absent rather than an error."""
        return getattr(
            getattr(getattr(self._fib_image, "metadata", None), "pixel_size", None),
            "x",
            None,
        )

    @staticmethod
    def _worst_fiducial_px(result: CorrelationResult) -> Optional[float]:
        """Largest single-fiducial reprojection error, which the RMS averages away.

        One badly-placed correspondence among several good ones barely moves the
        RMS but can still ruin the fit, so it is worth reporting separately.
        """
        if not result.delta_2d:
            return None
        return max(float(np.hypot(d.x, d.y)) for d in result.delta_2d)

    def _rms_tooltip(
        self,
        result: CorrelationResult,
        rms_px: float,
        px_m: Optional[float],
        concern: Optional[str] = None,
    ) -> str:
        """Explain what the RMS badge is measuring, and what it is not."""
        n = len(result.reprojected_3d)
        lines = []

        if px_m:
            lines.append(
                f"Registration RMS error — {_format_distance_nm(rms_px * px_m * 1e9)}"
            )
            lines.append(f"{rms_px:.2f} px × {px_m * 1e9:.1f} nm/px")
        else:
            lines.append(f"Registration RMS error — {rms_px:.2f} px")
            lines.append("FIB pixel size unknown — load the FIB image to see this in nm.")

        detail = (
            "Root-mean-square residual of the rigid 3D→2D fit"
            f"{f' across {n} fiducial pairs' if n else ''}."
        )
        worst = self._worst_fiducial_px(result)
        if worst is not None:
            shown = f"{worst:.2f} px"
            if px_m:
                shown += f" ({_format_distance_nm(worst * px_m * 1e9)})"
            detail += f" Worst single fiducial: {shown}."
        lines += ["", detail]

        if concern:
            lines += ["", f"⚠  {concern}"]

        lines += [
            "",
            "Measures how well the FM fiducials land on their FIB counterparts — "
            "not POI accuracy. The POI is extrapolated from this fit, so one "
            "outside the fiducial spread can be worse than this suggests. A low "
            "residual cannot confirm the correlation is right; only a high one "
            "can show it is wrong.",
        ]
        return "\n".join(lines)

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
                size=4,
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
                size=7,
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
                size=5,
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
        coord.fitted = False  # a manual drag supersedes any accepted fit
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
        coord.fitted = False  # a manual edit supersedes any accepted fit
        spec.adapter.refresh_coordinate(coord)
        spec.list_widget.refresh_coordinate(coord)  # drop the fitted indicator
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
            # Auto-save the result plot alongside the auto-saved JSON.
            if self._project_dir:
                try:
                    self.save_plot()
                except Exception:
                    logging.exception("Auto-save of correlation plot failed")
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

    def _on_labels_toggled(self, visible: bool) -> None:
        self._fib_canvas.set_labels_visible(visible)
        self._fm_display.canvas.set_labels_visible(visible)

    def _on_save_plot_clicked(self) -> None:
        """Prompt for a path and save the side-by-side FIB + FM plot."""
        start = self._project_dir or ""
        default = (
            os.path.join(start, f"correlation_plot_{time.strftime(DATETIME_FILE)}.png")
            if start
            else ""
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Correlation Plot", default, "PNG (*.png);;All files (*)"
        )
        if path:
            self.save_plot(path)

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

    def load_result(self, path: str, *, adopt_inputs: bool = True) -> None:
        """Load a correlation result from JSON and adopt it (mirrors load_data)."""
        logging.info("Loading correlation result from %s", path)
        self._load_result(CorrelationResult.load(path), adopt_inputs=adopt_inputs)

    def _load_result(
        self, result: CorrelationResult, *, adopt_inputs: bool = True
    ) -> None:
        """Adopt a loaded correlation result (and its input data, if any).

        ``adopt_inputs=False`` when the caller has already loaded fresher
        coordinates: ``result.input_data`` is a record of what the transform was
        fitted to, not the current truth, so applying it would silently discard
        every edit made after that run (FIB-295).
        """
        # Populate the lists first so the RI tab sees the loaded surface points.
        # _on_result_ready refreshes the run button and sets the final status
        # text itself — no trailing update here, it would overwrite the status.
        if adopt_inputs and result.input_data:
            self.set_data(result.input_data)
        # Continue commits result.poi[0].px_m to the protocol editor, so a result
        # that predates the current points must not arm it.
        self._on_result_ready(result, live=result.matches_inputs(self.data))

    def _menu_load_correlation(self) -> None:
        start = self._project_dir or ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Correlation", start, "JSON (*.json);;All files (*)"
        )
        if not path:
            return
        # One entry point for every correlation JSON; load_correlation dispatches
        # on shape, so a wrong pick is no longer possible — only a genuinely
        # unreadable/foreign file reaches this warning (FIB-264).
        try:
            self.load_correlation(path)
        except Exception as exc:
            logging.exception("Failed to load correlation from %s", path)
            QMessageBox.warning(self, "Load Error", f"Could not load correlation:\n{exc}")

    def _on_save(self) -> None:
        start = os.path.join(self._project_dir or "", CORRELATION_FILENAME)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Correlation", start, "JSON (*.json);;All files (*)"
        )
        if path:
            self.save_correlation(path)

    # ------------------------------------------------------------------
    # FM z-stack interpolation (FIB-253)
    # ------------------------------------------------------------------

    def _fm_side_lists(self) -> List["CoordinateListWidget"]:
        """The coordinate lists whose points live in the FM volume (carry z)."""
        cl = self._coords_tab
        return [cl.fm_list, cl.poi_list, cl.fm_surface_list]

    def _fm_point_count(self) -> int:
        return sum(len(lst.coordinates) for lst in self._fm_side_lists())

    def _rescale_fm_z(self, scale: float) -> None:
        """Scale every FM-side point's z index so its physical depth is preserved
        after the z axis is resampled (depth = z_index * pixel_size_z)."""
        for lst in self._fm_side_lists():
            for coord in lst.coordinates:
                coord.point.z *= scale

    def _on_interpolate_fm(self) -> None:
        if self._fm_image is None:
            return
        if self._interp_worker is not None and self._interp_worker.is_alive():
            return  # one at a time
        from fibsem.correlation.ui.widgets.fm_interpolate_dialog import (
            InterpolateZDialog,
        )

        dlg = InterpolateZDialog(
            self._fm_image, parent=self, fm_point_count=self._fm_point_count()
        )
        if dlg.exec_() != QDialog.Accepted:
            return
        target_m, method = dlg.result_params()
        self._start_fm_interpolation(target_m, method)

    def _start_fm_interpolation(self, target_m: float, method: str) -> None:
        from fibsem.correlation.util import interpolate_fm_volume
        from fibsem.ui.qt.threading import FunctionWorker

        src = self._fm_image
        old_nz = src.data.shape[1]
        n_channels = src.data.shape[0]

        # Non-modal: an embedded progress bar in the Images tab, and the FM display
        # is disabled so a point edit can't race the image/coordinate swap — but
        # the rest of the GUI stays live.
        self._images_tab.set_interpolating(True, n_channels)
        self._fm_display.setEnabled(False)

        relay = self._interp_relay = _ProgressRelay()
        relay.progress.connect(self._images_tab.set_interpolation_progress)

        worker = self._interp_worker = FunctionWorker(
            interpolate_fm_volume, src, target_m, method, relay.emit_progress
        )

        def _finish() -> None:
            self._interp_worker = None
            self._images_tab.set_interpolating(False)
            self._fm_display.setEnabled(True)

        def _done(new_image) -> None:
            _finish()
            self._adopt_interpolated_volume(new_image, old_nz)

        def _fail(exc) -> None:
            _finish()
            logging.exception("FM z-interpolation failed")
            notification_service.show("Z-interpolation failed.", "error")
            QMessageBox.warning(
                self, "Interpolation failed", f"Could not interpolate:\n{exc}"
            )

        worker.returned.connect(_done)
        worker.errored.connect(_fail)
        worker.start()

    def _adopt_interpolated_volume(self, new_image, old_nz: int) -> None:
        """Swap in the resampled volume and keep FM coordinates + metadata coherent.

        Matched pair: rescale FM-point z by the ACTUAL slice ratio, then adopt the
        new volume whose pixel_size_z was derived from that same ratio — so each
        point's physical depth (z_index * pixel_size_z) is preserved.
        """
        new_nz = new_image.data.shape[1]
        self._rescale_fm_z(new_nz / old_nz)
        self.set_fm_image(new_image)
        self.set_data(self.data)  # redraw lists/canvas at the rescaled z
        self.data_changed.emit(self.data)  # auto-save + RI refresh (new z step)
        notification_service.show(
            f"Z-interpolation complete — {old_nz} → {new_nz} slices "
            f"({new_image.metadata.pixel_size_z * 1e9:.0f} nm z step)",
            "info",
        )

    # ------------------------------------------------------------------
    # Refit
    # ------------------------------------------------------------------

    def _setup_shortcuts(self) -> None:
        """`F` fits the selected coordinate — same as the header refit button,
        without hunting for it. Scoped to this widget and its children so it
        only fires while the correlation UI has focus."""
        self._fit_shortcut = QShortcut(QKeySequence("F"), self)
        self._fit_shortcut.setContext(
            Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self._fit_shortcut.activated.connect(self._fit_selected_coordinate)

    def _selected_coordinate(self) -> Optional[Coordinate]:
        """The single selected coordinate across the lists (``_select_only``
        guarantees at most one), or None."""
        for spec in self._point_specs.values():
            coord = spec.list_widget.selected_coordinate
            if coord is not None:
                return coord
        return None

    def _fit_selected_coordinate(self) -> None:
        """`F` hotkey: fit the currently-selected coordinate."""
        # Don't hijack the key while a value is being edited in a field.
        if isinstance(QApplication.focusWidget(), (QLineEdit, QAbstractSpinBox)):
            return
        coord = self._selected_coordinate()
        if coord is not None:
            self._on_refit_requested(coord)

    def _on_refit_requested(self, coord: Coordinate) -> None:
        """Auto-fit the coordinate, then confirm the result before applying it."""
        result = self._run_point_fit(coord)
        if result is None:
            return  # no applicable method / image — nothing to fit

        if self._should_auto_accept(result):
            self._auto_accept_fit(result)
            return

        show_fig = self._coords_tab._show_diag_check.isChecked()
        dialog = FitConfirmationDialog(result, show_figure=show_fig, parent=self)
        # The dialog renders its own figure (OO API, no pyplot) and frees it with
        # its canvas, so there's nothing to plt.close here anymore.
        if dialog.exec_() == QDialog.Accepted:
            self._apply_fit_result(result)

    def _should_auto_accept(self, result: PointFitResult) -> bool:
        """Auto-accept only when opted in, the fit succeeded, and it isn't a
        far-off outlier — failures and surprising jumps still get the dialog."""
        if not self._coords_tab._auto_accept_check.isChecked():
            return False
        if result.status is FitStatus.ERROR:
            return False
        return not self._is_surprising_fit(result)

    @staticmethod
    def _is_surprising_fit(result: PointFitResult) -> bool:
        """A jump larger than a refinement should be — likely the wrong feature."""
        return (
            result.delta_px > _AUTO_ACCEPT_MAX_XY_PX
            or abs(result.delta_z) > _AUTO_ACCEPT_MAX_Z
        )

    def _auto_accept_fit(self, result: PointFitResult) -> None:
        """Apply a fit without the confirm dialog, with a status-bar note."""
        self._apply_fit_result(result)
        # No figure is built on the auto-accept path (the diagnostic is just
        # data), so there's nothing to close.
        # Set the note AFTER applying (apply emits data_changed, which may
        # refresh the status line) so this is the message that sticks.
        name = result.coordinate.point_type.value
        self._lbl_status.setText(
            f"Auto-fit {name}: Δ {result.delta_px:.1f} px, {result.delta_z:+.1f} z"
        )

    def _run_point_fit(self, coord: Coordinate) -> Optional[PointFitResult]:
        """Compute an auto-fit for ``coord`` WITHOUT mutating it.

        Returns None when no fit is applicable (method "None" / image missing);
        otherwise a PointFitResult carrying the proposed position, coarse status,
        and the diagnostic figure.
        """
        from fibsem.correlation.util import (
            hole_fitting_FIB,
            hole_fitting_reflection,
            target_fitting_fluorescence,
        )

        cl = self._coords_tab
        spec = self._point_specs[coord.point_type]
        x, y, z = coord.point.x, coord.point.y, coord.point.z
        initial = PointXYZ(x, y, z)
        method, channel, channel_name = "", None, None
        fitted, diag, error, error_detail, attempted = None, None, None, None, False

        try:
            if spec.adapter.side == "fib":
                method = cl._fib_method_combo.currentText()
                if method == "Hole" and self._fib_image is not None:
                    attempted = True
                    # pass the sub-pixel click (not int) so the diagnostic's
                    # input marker lands where the user clicked — FIB-282.
                    xr, yr, diag = hole_fitting_FIB(
                        self._fib_image.filtered_data, x, y
                    )
                    fitted = PointXYZ(float(xr), float(yr), z)
            else:
                # The FM surface is typically picked in the reflection channel,
                # so it shares the fiducial method/channel settings.
                is_fid = spec.fm_fit_role == "fid"
                method = (
                    cl._fm_fid_method_combo if is_fid else cl._fm_poi_method_combo
                ).currentText()
                ch_combo = cl._fm_fid_ch_combo if is_fid else cl._fm_poi_ch_combo
                channel = ch_combo.currentIndex()
                channel_name = ch_combo.currentText()
                if method != "None" and self._fm_image is not None and channel >= 0:
                    img = self._fm_image.data[channel]
                    if method == "Hole":
                        attempted = True
                        xr, yr, zr, diag = hole_fitting_reflection(
                            img, x, y, z=int(z), cutout=2  # sub-pixel x/y (FIB-282)
                        )
                        fitted = PointXYZ(float(xr), float(yr), float(zr))
                    elif method == "Gaussian":
                        attempted = True
                        xr, yr, zr, diag = target_fitting_fluorescence(
                            img, x, y, int(z), cutout=5  # sub-pixel x/y (FIB-282)
                        )
                        fitted = PointXYZ(float(xr), float(yr), float(zr))
        except Exception as exc:
            logging.exception("Point fit failed")
            error_detail = str(exc)          # raw text -> log + tooltip
            error = humanize_fit_error(exc)  # user-facing, actionable
            attempted = True

        if not attempted:
            return None

        status = PointFitResult.classify(initial, fitted, error=error)
        return PointFitResult(
            coordinate=coord,
            method=method,
            channel=channel,
            channel_name=channel_name,
            initial=initial,
            fitted=fitted,
            status=status,
            message=error,
            detail=error_detail,
            diagnostic=diag,
        )

    def _apply_fit_result(self, result: PointFitResult) -> None:
        """Commit an accepted fit: move the coordinate and flag it as fitted."""
        if result.fitted is None:
            return
        coord = result.coordinate
        coord.point.x = result.fitted.x
        coord.point.y = result.fitted.y
        coord.point.z = result.fitted.z
        coord.fitted = True
        spec = self._point_specs[coord.point_type]
        spec.list_widget.refresh_coordinate(coord)
        spec.adapter.refresh_coordinate(coord)
        self.data_changed.emit(self.data)


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

        # Allow the whole correlation window to be minimised / maximised.
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
        )
        self._min_shortcut = QShortcut(QKeySequence("Ctrl+M"), self)
        self._min_shortcut.activated.connect(self.showMinimized)

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


def _discover_correlation_files(directory: str) -> Dict[str, Optional[str]]:
    """Locate FIB/FM images and saved correlation JSON in a project dir.

    Conventions (matching the widget's auto-save + typical exports):
      - FM image   : ``*.ome.tif`` / ``*.ome.tiff``
      - FIB image  : ``*_ib.tif`` / ``*_ib.tiff`` (else the first non-OME TIFF)
      - correlation: ``correlation.json`` (FIB-264 consolidated file)
      - data       : ``correlation_data.json``   (legacy, still read)
      - result     : ``correlation_result.json`` (legacy, still read)
    """
    import glob

    def _first(patterns: List[str]) -> Optional[str]:
        for pat in patterns:
            hits = sorted(glob.glob(os.path.join(directory, pat)))
            if hits:
                return hits[0]
        return None

    fib = _first(["*_ib.tif", "*_ib.tiff"])
    if fib is None:
        tifs = sorted(
            glob.glob(os.path.join(directory, "*.tif"))
            + glob.glob(os.path.join(directory, "*.tiff"))
        )
        fib = next(
            (t for t in tifs if not t.endswith((".ome.tif", ".ome.tiff"))), None
        )

    def _existing(name: str) -> Optional[str]:
        p = os.path.join(directory, name)
        return p if os.path.exists(p) else None

    return {
        "fib": fib,
        "fm": _first(["*.ome.tif", "*.ome.tiff"]),
        "correlation": _existing(CORRELATION_FILENAME),
        "data": _existing(_LEGACY_DATA_FILENAME),
        "result": _existing(_LEGACY_RESULT_FILENAME),
    }


def load_project(widget: "CorrelationTabWidget", directory: str) -> None:
    """Quickstart-load a correlation project directory into ``widget``.

    Sets the project dir, then loads the FIB + FM images and, if present, the
    saved correlation result (preferred) or coordinate data. Missing or
    unreadable pieces are logged and skipped so a partial project still opens.
    """
    found = _discover_correlation_files(directory)
    logging.info("Quickstart loading correlation project: %s", directory)
    widget.set_project_dir(directory)

    if found["fib"]:
        try:
            widget.set_fib_image(FibsemImage.load(found["fib"]))
            logging.info("  FIB image: %s", os.path.basename(found["fib"]))
        except Exception:
            logging.exception("  failed to load FIB image %s", found["fib"])
    else:
        logging.warning("  no FIB image (*_ib.tif) found in %s", directory)

    if found["fm"]:
        try:
            widget.set_fm_image(FluorescenceImage.load(found["fm"]))
            logging.info("  FM image: %s", os.path.basename(found["fm"]))
        except Exception:
            logging.exception("  failed to load FM image %s", found["fm"])
    else:
        logging.warning("  no FM image (*.ome.tiff) found in %s", directory)

    # The consolidated file is authoritative when present — it carries both the
    # points and the result, staleness self-described (FIB-264/FIB-295).
    if found["correlation"]:
        try:
            widget.load_correlation(found["correlation"])
            logging.info("  correlation: %s", os.path.basename(found["correlation"]))
            return
        except Exception:
            logging.exception(
                "  failed to load %s; falling back to legacy files",
                found["correlation"],
            )

    # Legacy fallback. Coordinates first: correlation_data.json is rewritten on
    # every edit, so it is the freshest record of the points; the result's
    # embedded snapshot is from its last run and must not overwrite later edits
    # (FIB-295).
    loaded_data = False
    if found["data"]:
        try:
            widget.load_data(found["data"])
            loaded_data = True
            logging.info("  data: %s", os.path.basename(found["data"]))
        except Exception:
            logging.exception("  failed to load data %s", found["data"])

    if found["result"]:
        try:
            # With no data file the result's snapshot is the only record of the
            # points, so adopt it; otherwise keep the coordinates just loaded.
            widget.load_result(found["result"], adopt_inputs=not loaded_data)
            logging.info("  result: %s", os.path.basename(found["result"]))
        except Exception:
            logging.exception("  failed to load result %s", found["result"])


def main() -> None:
    import argparse
    import sys

    from PyQt5.QtWidgets import QApplication

    parser = argparse.ArgumentParser(description="FIB-FM correlation widget")
    parser.add_argument(
        "project",
        nargs="?",
        default=None,
        help="Optional correlation project directory to quickstart-load "
        "(FIB/FM images + correlation.json).",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    app.setStyleSheet(stylesheets.NAPARI_STYLE)

    widget = CorrelationTabWidget()
    if args.project:
        load_project(widget, args.project)
    widget.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
