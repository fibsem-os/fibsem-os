"""CorrelationSetupDialog — the "Set up Correlation" pre-dialog (FIB-302).

Shown before the correlation window when opening correlation for a lamella. It
confirms the images and captures, in one place, the choices that were previously
made implicitly in ``_open_correlation_dialog`` (which source to seed from) plus
the FM-interpolation preference — then hands back a :class:`CorrelationSetup` the
caller uses to open the correlation window already seeded.

Layout mirrors the correlation widget: **previews on the left** (where the canvas
lives), **controls on the right** (where the panels live). The previews are
functional — they overlay *the coordinates that would be seeded* for the selected
starting-coordinates source, so the user confirms "right points, right places"
before opening.

This folds in the standalone history picker (FIB-301): the "Previous correlation"
radio + run dropdown is that chooser, inline.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from fibsem import constants
from fibsem.correlation.config import CorrelationConfig
from fibsem.correlation.history import CorrelationRun, LamellaCorrelation
from fibsem.ui import stylesheets
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.widgets.custom_widgets import ValueComboBox, ValueSpinBox

# Starting-coordinates sources (mutually exclusive; see the design doc).
SEED_NONE = "none"
SEED_SPOT_BURNS = "spot_burns"
SEED_PREVIOUS = "previous"

XY = Tuple[float, float]  # a point in normalised image coordinates, (x, y) in [0, 1]

_MUTED = "#9aa0a6"
_HDR = "#b0b3b8"  # muted grey — headers read via bold + letter-spacing, not colour
_FIDUCIAL = "#37c0ff"
_POI = "#ff00ff"  # magenta — matches the POI layer elsewhere in the app
_THUMB = 250  # preview thumbnail size (px)


@dataclass
class CorrelationSetup:
    """The user's confirmed choices from the pre-dialog."""

    fib_filename: str
    fm_filename: str
    seed_source: str                        # SEED_NONE | SEED_SPOT_BURNS | SEED_PREVIOUS
    previous_run: Optional[CorrelationRun]  # set iff seed_source == SEED_PREVIOUS
    interpolate: bool
    isotropic: bool
    target_z_nm: Optional[float]
    interp_method: str


@dataclass
class PreviewData:
    """Base thumbnails + the normalised overlay coordinates each source would seed.

    Supplied by the caller so the dialog stays decoupled from image loading. All
    coordinates are normalised (0-1, top-left origin) so they draw correctly at
    any thumbnail size. Any field may be empty/None — the previews degrade to the
    bare thumbnail (or a placeholder when a thumbnail is absent).
    """

    fib_thumb: Optional[QPixmap] = None
    fm_thumb: Optional[QPixmap] = None
    fib_caption: str = ""
    fm_caption: str = ""
    spot_burns: List[XY] = field(default_factory=list)   # FIB overlays, spot-burn source
    prev_fib: List[XY] = field(default_factory=list)     # FIB overlays, previous source
    prev_fm: List[XY] = field(default_factory=list)      # FM overlays, previous source
    prev_poi: Optional[XY] = None                        # FM POI overlay, previous source


def _format_timestamp(name: str) -> str:
    """Reformat a run folder name (a ``DATETIME_FILE`` stamp) for display.

    Falls back to the raw name for a folder that doesn't parse (e.g. a legacy or
    hand-named directory), so an odd name still shows rather than disappearing.
    """
    try:
        dt = datetime.datetime.strptime(name, constants.DATETIME_FILE)
    except ValueError:
        return name
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _header(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"color:{_HDR};font-size:11px;font-weight:bold;letter-spacing:1px;padding-top:4px;"
    )
    return lbl


def _caption(text: str, color: str = _MUTED, indent: int = 0) -> QLabel:
    lbl = QLabel(text)
    style = f"color:{color};font-size:11px;"
    if indent:
        style += f"margin-left:{indent}px;"
    lbl.setStyleSheet(style)
    lbl.setWordWrap(True)
    return lbl


def _flabel(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color:{_MUTED};font-size:11px;")
    return lbl


def _hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet("color:#2f333a;")
    return line


def _overlay(base: QPixmap, dots: List[XY], cross: Optional[XY]) -> QPixmap:
    """Draw fiducial dots and an optional POI cross on a copy of ``base``."""
    out = QPixmap(base)
    painter = QPainter(out)
    painter.setRenderHint(QPainter.Antialiasing)
    w, h = out.width(), out.height()

    # fiducials: small filled dots
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(_FIDUCIAL))
    r = 3
    for fx, fy in dots:
        painter.drawEllipse(int(fx * w - r), int(fy * h - r), 2 * r, 2 * r)

    if cross is not None:
        pen = QPen(QColor(_POI))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        cx, cy = cross[0] * w, cross[1] * h
        s = 8
        painter.drawLine(int(cx - s), int(cy), int(cx + s), int(cy))
        painter.drawLine(int(cx), int(cy - s), int(cx), int(cy + s))

    painter.end()
    return out


def _interpolation_chip(
    z_slices: Optional[int],
    pixel_size_z_nm: Optional[float],
    isotropic: bool,
    target_z_nm: Optional[float],
    pixel_size_xy_nm: Optional[float],
) -> str:
    """Best-effort '21 → 105 slices' summary; falls back to the target alone."""
    target = pixel_size_xy_nm if isotropic else target_z_nm
    if not z_slices or not pixel_size_z_nm or not target:
        if target:
            return f"target ≈ {target:.0f} nm/slice"
        return "isotropic (match XY)" if isotropic else "custom target z"
    new_slices = max(1, round(z_slices * pixel_size_z_nm / target))
    kind = "isotropic" if isotropic else f"{target:.0f} nm"
    return f"{z_slices} → {new_slices} slices · {kind}"


class CorrelationSetupDialog(QDialog):
    """Modal "Set up Correlation" pre-dialog.

    ``exec_()`` returns ``QDialog.Accepted`` when the user clicks Open Correlation;
    :attr:`setup` then holds the choices (``None`` until accepted / on cancel).
    """

    def __init__(
        self,
        *,
        lamella_name: str,
        fib_options: List[str],
        fib_current: str,
        fm_options: List[str],
        fm_current: str,
        spot_burn_count: int = 0,
        history: Optional[LamellaCorrelation] = None,
        config: Optional[CorrelationConfig] = None,
        preview: Optional[PreviewData] = None,
        fm_z_slices: Optional[int] = None,
        fm_pixel_size_z_nm: Optional[float] = None,
        fm_pixel_size_xy_nm: Optional[float] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._config = config or CorrelationConfig()
        self._preview = preview or PreviewData()
        self._spot_burn_count = spot_burn_count
        self._fm_z_slices = fm_z_slices
        self._fm_pixel_size_z_nm = fm_pixel_size_z_nm
        self._fm_pixel_size_xy_nm = fm_pixel_size_xy_nm
        # Newest first for the dropdown; index i -> self._prev_runs[i].
        runs = list(history.runs) if history is not None else []
        self._prev_runs: List[CorrelationRun] = list(reversed(runs))
        self._setup: Optional[CorrelationSetup] = None

        self.setWindowTitle("Set up Correlation")
        self.setModal(True)
        self.setStyleSheet("background:#1e2124;color:#d0d0d0;")

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 14)
        root.setSpacing(9)

        title = QLabel(f"Set up Correlation — {lamella_name}")
        title.setStyleSheet("font-size:15px;font-weight:bold;color:#ffffff;")
        root.addWidget(title)
        root.addWidget(
            _caption(
                "Confirm the images and choose what to carry over. Loaded points are "
                "placed as seeds — refine them after opening."
            )
        )
        root.addWidget(_hline())

        cols = QHBoxLayout()
        cols.setSpacing(22)
        cols.addWidget(self._build_previews())
        cols.addWidget(
            self._build_controls(fib_options, fib_current, fm_options, fm_current)
        )
        root.addLayout(cols)

        root.addWidget(_hline())
        root.addLayout(self._build_buttons())

        # Never let the dialog shrink below its content — otherwise the two columns
        # differ in height and the shorter side squeezes controls (the interpolation
        # spinbox/combo) below their minimum, collapsing them to slivers.
        root.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

        # Connect only once everything is built, so setting the radio/interp
        # defaults above doesn't fire handlers that touch not-yet-created widgets.
        self._connect_signals()
        self._apply_enabled_state()
        self._redraw_previews()

    def _connect_signals(self) -> None:
        for rb in (self._rb_none, self._rb_burns, self._rb_prev):
            rb.toggled.connect(self._on_source_changed)
        self._run_combo.currentIndexChanged.connect(lambda _i: self._redraw_previews())
        self._chk_interp.toggled.connect(self._on_interp_toggled)
        self._chk_iso.toggled.connect(self._on_interp_toggled)

    # ---- construction -------------------------------------------------------

    def _build_previews(self) -> QWidget:
        col = QVBoxLayout()
        col.setSpacing(12)
        col.addWidget(_header("PREVIEW"))

        self._fib_preview = self._preview_label(self._preview.fib_thumb)
        col.addWidget(self._fib_preview, alignment=Qt.AlignmentFlag.AlignHCenter)
        col.addWidget(_caption(self._preview.fib_caption or "FIB"))

        self._fm_preview = self._preview_label(self._preview.fm_thumb)
        col.addWidget(self._fm_preview, alignment=Qt.AlignmentFlag.AlignHCenter)
        col.addWidget(_caption(self._preview.fm_caption or "FM"))

        legend = QLabel(
            f"<span style='color:{_FIDUCIAL}'>●</span> fiducials    "
            f"<span style='color:{_POI}'>✛</span> POI    "
            f"<span style='color:{_MUTED}'>— from the selection</span>"
        )
        legend.setStyleSheet("font-size:11px;")
        legend.setWordWrap(True)
        col.addWidget(legend)
        col.addStretch(1)

        wrapper = QWidget()
        wrapper.setLayout(col)
        wrapper.setFixedWidth(_THUMB + 6)
        return wrapper

    def _preview_label(self, base: Optional[QPixmap]) -> QLabel:
        """A preview frame sized to its thumbnail's aspect (square when absent), so
        a landscape FIB image isn't letterboxed in a forced-square box."""
        lbl = QLabel()
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("border:1px solid #3a3d42;border-radius:5px;")
        if base is not None and not base.isNull():
            lbl.setFixedSize(base.width(), base.height())
        else:
            lbl.setFixedSize(_THUMB, _THUMB)
        return lbl

    def _build_controls(
        self,
        fib_options: List[str],
        fib_current: str,
        fm_options: List[str],
        fm_current: str,
    ) -> QWidget:
        col = QVBoxLayout()
        col.setSpacing(9)

        col.addWidget(_header("IMAGES"))
        img_form = QFormLayout()
        img_form.setSpacing(6)
        img_form.setContentsMargins(2, 0, 2, 0)
        self._fib_combo = ValueComboBox(fib_options or [fib_current], value=fib_current)
        self._fm_combo = ValueComboBox(fm_options or [fm_current], value=fm_current)
        img_form.addRow(_flabel("FIB"), self._fib_combo)
        img_form.addRow(_flabel("FM"), self._fm_combo)
        col.addLayout(img_form)
        col.addWidget(_hline())

        col.addWidget(_header("STARTING COORDINATES"))
        self._seed_group = QButtonGroup(self)
        self._rb_none = QRadioButton("None — pick everything fresh")
        burn_label = f"Spot-burn fiducials · {self._spot_burn_count} found"
        if not self._prev_runs:
            burn_label += "  (first)"
        self._rb_burns = QRadioButton(burn_label)
        self._rb_prev = QRadioButton("Previous correlation")
        for rb in (self._rb_none, self._rb_burns, self._rb_prev):
            self._seed_group.addButton(rb)
            col.addWidget(rb)

        run_row = QWidget()
        run_layout = QHBoxLayout(run_row)
        run_layout.setContentsMargins(22, 0, 2, 0)
        self._run_combo = ValueComboBox([self._run_label(r) for r in self._prev_runs])
        run_layout.addWidget(self._run_combo, 1)
        col.addWidget(run_row)
        self._prev_caption = _caption(
            "Carries the FM POI + fiducials from that run forward.", indent=22
        )
        col.addWidget(self._prev_caption)
        col.addWidget(_hline())

        self._rb_burns.setEnabled(self._spot_burn_count > 0)
        self._rb_prev.setEnabled(bool(self._prev_runs))
        # Default: Previous if a run exists, else spot burns, else none.
        if self._prev_runs:
            self._rb_prev.setChecked(True)
        elif self._spot_burn_count > 0:
            self._rb_burns.setChecked(True)
        else:
            self._rb_none.setChecked(True)

        col.addWidget(self._build_interpolation())
        col.addWidget(_hline())
        col.addWidget(self._build_inherited_section())
        col.addStretch(1)

        wrapper = QWidget()
        wrapper.setLayout(col)
        wrapper.setFixedWidth(320)
        return wrapper

    def _build_interpolation(self) -> QWidget:
        box = QVBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(6)
        box.addWidget(_header("FM INTERPOLATION"))

        interp = self._config.interpolation
        self._chk_interp = QCheckBox("Interpolate FM volume")
        self._chk_interp.setChecked(interp.enabled)
        box.addWidget(self._chk_interp)

        self._chk_iso = QCheckBox("Isotropic (match XY pixel size)")
        self._chk_iso.setChecked(interp.isotropic)
        box.addWidget(self._chk_iso)

        form = QFormLayout()
        form.setSpacing(6)
        form.setContentsMargins(2, 0, 2, 0)
        self._spin_z = ValueSpinBox(
            suffix="nm", minimum=1.0, maximum=100000.0, step=10.0, decimals=1
        )
        self._spin_z.setValue(interp.target_z_nm or 100.0)
        form.addRow(_flabel("Target z"), self._spin_z)
        self._method_combo = ValueComboBox(["linear", "cubic"], value=interp.method)
        form.addRow(_flabel("Method"), self._method_combo)
        box.addLayout(form)

        self._chip = QLabel()
        self._chip.setStyleSheet(
            "color:#cfe0f2;background:#21303f;border:0.5px solid #2f4a63;"
            "border-radius:6px;padding:6px 10px;font-size:12px;"
        )
        box.addWidget(self._chip)
        box.addWidget(
            _caption("Runs in the background after opening; seeded points rescale in.")
        )

        wrapper = QWidget()
        wrapper.setLayout(box)
        return wrapper

    def _build_inherited_section(self) -> QWidget:
        """Read-only summary of the experiment-global CorrelationConfig (fit + RI).

        Edited in the correlation window, not here (keeps setup fast). NA/λ are
        seeded from the FM image's channel metadata (FIB-277), hence the hint.
        """
        fit = self._config.fit
        ri = self._config.ri
        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)
        col.addWidget(_header("INHERITED SETTINGS"))
        col.addWidget(
            _caption(
                f"Fit — FIB {fit.fib_method} · FM POI {fit.fm_poi_method} · "
                f"POI channel {fit.fm_poi_channel or '—'}"
            )
        )
        col.addWidget(
            _caption(
                f"RI — n₂ {ri.n2:.2f} · NA {ri.na:.2f} · "
                f"λ {ri.wavelength_um * 1000:.0f} nm  (from FM metadata)"
            )
        )
        wrapper = QWidget()
        wrapper.setLayout(col)
        return wrapper

    def _build_buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        cancel.setAutoDefault(False)
        cancel.clicked.connect(self.reject)
        row.addWidget(cancel)
        self._open_btn = QPushButton("  Open Correlation  ")
        self._open_btn.setIcon(fibsem_icon("mdi:arrow-right", color="white"))
        self._open_btn.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self._open_btn.setDefault(True)
        self._open_btn.clicked.connect(self._on_accept)
        row.addWidget(self._open_btn)
        return row

    # ---- interaction --------------------------------------------------------

    def _run_label(self, run: CorrelationRun) -> str:
        return _format_timestamp(run.name)

    def seed_source(self) -> str:
        if self._rb_burns.isChecked():
            return SEED_SPOT_BURNS
        if self._rb_prev.isChecked():
            return SEED_PREVIOUS
        return SEED_NONE

    def _selected_run(self) -> Optional[CorrelationRun]:
        idx = self._run_combo.currentIndex()
        if 0 <= idx < len(self._prev_runs):
            return self._prev_runs[idx]
        return self._prev_runs[0] if self._prev_runs else None

    def _on_source_changed(self, _checked: bool = False) -> None:
        self._apply_enabled_state()
        self._redraw_previews()

    def _on_interp_toggled(self, _checked: bool = False) -> None:
        self._apply_enabled_state()
        self._update_chip()

    def _apply_enabled_state(self) -> None:
        is_prev = self._rb_prev.isChecked()
        self._run_combo.setEnabled(is_prev)
        self._prev_caption.setVisible(is_prev)

        interp_on = self._chk_interp.isChecked()
        self._chk_iso.setEnabled(interp_on)
        self._method_combo.setEnabled(interp_on)
        self._spin_z.setEnabled(interp_on and not self._chk_iso.isChecked())
        self._chip.setVisible(interp_on)
        self._update_chip()

    def _update_chip(self) -> None:
        if not self._chk_interp.isChecked():
            return
        self._chip.setText(
            _interpolation_chip(
                self._fm_z_slices,
                self._fm_pixel_size_z_nm,
                self._chk_iso.isChecked(),
                self._spin_z.value(),
                self._fm_pixel_size_xy_nm,
            )
        )

    def _redraw_previews(self) -> None:
        source = self.seed_source()
        fib_circles: List[XY] = []
        fm_circles: List[XY] = []
        fm_cross: Optional[XY] = None
        if source == SEED_SPOT_BURNS:
            fib_circles = self._preview.spot_burns
        elif source == SEED_PREVIOUS:
            fib_circles = self._preview.prev_fib
            fm_circles = self._preview.prev_fm
            fm_cross = self._preview.prev_poi

        self._set_thumb(self._fib_preview, self._preview.fib_thumb, fib_circles, None)
        self._set_thumb(self._fm_preview, self._preview.fm_thumb, fm_circles, fm_cross)

    @staticmethod
    def _set_thumb(
        label: QLabel,
        base: Optional[QPixmap],
        circles: List[XY],
        cross: Optional[XY],
    ) -> None:
        if base is None:
            label.setText("no preview")
            label.setStyleSheet(
                "border:1px solid #3a3d42;border-radius:5px;color:#6a6d72;"
            )
            return
        label.setPixmap(_overlay(base, circles, cross))

    def _on_accept(self) -> None:
        self._setup = CorrelationSetup(
            fib_filename=self._fib_combo.currentText(),
            fm_filename=self._fm_combo.currentText(),
            seed_source=self.seed_source(),
            previous_run=self._selected_run()
            if self.seed_source() == SEED_PREVIOUS
            else None,
            interpolate=self._chk_interp.isChecked(),
            isotropic=self._chk_iso.isChecked(),
            target_z_nm=None if self._chk_iso.isChecked() else self._spin_z.value(),
            interp_method=self._method_combo.currentText(),
        )
        self.accept()

    @property
    def setup(self) -> Optional[CorrelationSetup]:
        """The confirmed choices, or ``None`` until Open Correlation is clicked."""
        return self._setup
