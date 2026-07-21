"""FitConfirmationDialog — accept/reject a per-point auto-fit result.

A point fit (auto-centroid / gaussian) proposes a refined position for a
coordinate. Rather than mutating the coordinate silently, the fit is captured
as a :class:`PointFitResult` and shown here for the user to accept or reject.

The result type is list-ready: a batch fit (multiple selected points) produces
a list of ``PointFitResult``, which a future summary dialog can present together.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.correlation.structures import Coordinate, PointXYZ
from fibsem.ui import stylesheets

# Per-axis displacement below which a fit is treated as "no change" (the fit
# fell back to the input). Well under the 3-decimal coordinate precision, so
# genuine sub-pixel refinements read as real moves, not "no change".
_UNCHANGED_EPS = 0.001


class FitStatus(Enum):
    OK = "ok"                # fit moved the point
    UNCHANGED = "unchanged"  # fit landed on (≈) the input position
    ERROR = "error"          # fit raised / produced no position


@dataclass
class PointFitResult:
    """Outcome of auto-fitting one coordinate (runtime only — holds a live fig)."""

    coordinate: Coordinate
    method: str
    channel: Optional[int]
    initial: PointXYZ
    fitted: Optional[PointXYZ]
    status: FitStatus
    message: Optional[str] = None
    figure: object = None  # matplotlib Figure; not serialised

    @property
    def delta_px(self) -> float:
        """XY displacement between the initial and fitted position (pixels)."""
        if self.fitted is None:
            return 0.0
        dx = self.fitted.x - self.initial.x
        dy = self.fitted.y - self.initial.y
        return (dx * dx + dy * dy) ** 0.5

    @property
    def delta_z(self) -> float:
        if self.fitted is None:
            return 0.0
        return self.fitted.z - self.initial.z

    @property
    def delta(self) -> Optional[tuple]:
        """Per-axis (dx, dy, dz) displacement, or None when there's no fit."""
        if self.fitted is None:
            return None
        return (
            self.fitted.x - self.initial.x,
            self.fitted.y - self.initial.y,
            self.fitted.z - self.initial.z,
        )

    @staticmethod
    def classify(
        initial: PointXYZ,
        fitted: Optional[PointXYZ],
        *,
        error: Optional[str] = None,
    ) -> FitStatus:
        """Coarse status from the initial/fitted positions (no fit-fn changes)."""
        if error is not None or fitted is None:
            return FitStatus.ERROR
        deltas = (
            fitted.x - initial.x,
            fitted.y - initial.y,
            fitted.z - initial.z,
        )
        moved = any(abs(d) >= _UNCHANGED_EPS for d in deltas)
        return FitStatus.OK if moved else FitStatus.UNCHANGED


# (background, foreground) per status — napari-dark chips
_STATUS_STYLE = {
    FitStatus.OK:        ("#1b5e20", "#a5d6a7"),
    FitStatus.UNCHANGED: ("#5d4037", "#ffcc80"),
    FitStatus.ERROR:     ("#5a1f1f", "#ef9a9a"),
}


def _status_text(result: "PointFitResult") -> str:
    if result.status is FitStatus.OK:
        return "Fit ok"
    if result.status is FitStatus.UNCHANGED:
        return "No change — fit ≈ original"
    return f"Fit failed: {result.message or 'no result'}"


def _fmt_xyz(p: PointXYZ) -> str:
    return f"{p.x:.3f}, {p.y:.3f}, {p.z:.3f}"


def _fmt_delta(initial: PointXYZ, fitted: PointXYZ) -> str:
    return (
        f"{fitted.x - initial.x:+.3f}, "
        f"{fitted.y - initial.y:+.3f}, "
        f"{fitted.z - initial.z:+.3f}"
    )


def _kv(key: str, value: str, value_color: str = "#d0d0d0") -> QWidget:
    w = QWidget()
    row = QHBoxLayout(w)
    row.setContentsMargins(0, 0, 0, 0)
    k = QLabel(key)
    k.setStyleSheet("color: #8a8d93; font-size: 12px;")
    v = QLabel(value)
    v.setStyleSheet(f"color: {value_color}; font-size: 12px;")
    v.setTextFormat(Qt.TextFormat.PlainText)
    row.addWidget(k)
    row.addStretch(1)
    row.addWidget(v)
    return w


# --- dark theming for the embedded diagnostic figure ---------------------
# The fit figures (fibsem.correlation.util) are authored light — black text,
# a near-black gaussian-fit line — for standalone/saved use. This dialog is
# napari-dark, so a white figure glares against it. Re-theme the figure to the
# dialog palette at embed time; done here (not in util.py, which lives on main)
# so the change stays wholly within this dialog.
_FIG_BG = "#1e2124"    # == the dialog background: figure margins blend in
_AXES_BG = "#262930"   # a subtle lift so the z line-plot reads as a panel
_FG = "#d0d0d0"        # dialog foreground (labels, ticks, legend text)
_FG_TITLE = "#e0e0e0"  # titles / suptitle, a touch brighter
_SPINE = "#4a4d53"
# A line darker than this is invisible on the dark panel; the only such artist
# is the gaussian-fit overlay (authored at 0.15 grey). Lift it to a light grey
# that still sits below the 0.55 signal line, preserving the fit-vs-signal read.
_DARK_LINE_LUM = 0.25
_DARK_LINE_TARGET = "0.8"


def _luminance(color) -> float:
    from matplotlib.colors import to_rgb

    r, g, b = to_rgb(color)
    return 0.299 * r + 0.587 * g + 0.114 * b


def _apply_dark_theme(fig) -> None:
    """Re-theme a light-authored fit figure onto the napari-dark palette.

    Only theme-agnostic chrome is touched — figure/axes background, text,
    ticks, spines, legend frame, and any near-black line. The saturated
    input(red)/fitted(green) markers and the grayscale image data are left
    exactly as authored (they already read correctly on dark).
    """
    fig.set_facecolor(_FIG_BG)
    for txt in fig.texts:  # figure-level text == the suptitle
        txt.set_color(_FG_TITLE)
    for ax in fig.axes:
        ax.set_facecolor(_AXES_BG)
        ax.title.set_color(_FG_TITLE)
        ax.xaxis.label.set_color(_FG)
        ax.yaxis.label.set_color(_FG)
        ax.tick_params(colors=_FG)
        for spine in ax.spines.values():
            spine.set_color(_SPINE)
        for line in ax.get_lines():
            if _luminance(line.get_color()) < _DARK_LINE_LUM:
                line.set_color(_DARK_LINE_TARGET)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(_AXES_BG)
            legend.get_frame().set_edgecolor(_SPINE)
            for text in legend.get_texts():
                text.set_color(_FG)


class FitConfirmationDialog(QDialog):
    """Modal accept/reject for a single :class:`PointFitResult`.

    ``exec_()`` returns ``QDialog.Accepted`` when the user accepts the fitted
    position. An ERROR result has nothing to apply — only a Close button.
    """

    def __init__(
        self,
        result: PointFitResult,
        show_figure: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._result = result
        self.setWindowTitle("Confirm fit")
        self.setModal(True)
        self.setStyleSheet("background: #1e2124; color: #d0d0d0;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 12)
        layout.setSpacing(10)

        bg, fg = _STATUS_STYLE[result.status]
        status = QLabel(_status_text(result))
        status.setStyleSheet(
            f"background: {bg}; color: {fg}; border-radius: 10px; "
            f"padding: 4px 12px; font-size: 12px;"
        )
        layout.addWidget(status, alignment=Qt.AlignmentFlag.AlignLeft)

        body = QHBoxLayout()
        body.setSpacing(14)

        if show_figure and result.figure is not None:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

            _apply_dark_theme(result.figure)
            try:
                result.figure.tight_layout()
            except Exception:
                pass
            canvas = FigureCanvasQTAgg(result.figure)
            # Size to the figure's aspect so the wide FM figs (z + XY panels
            # at 9x4.5) aren't squished square.
            w_in, h_in = result.figure.get_size_inches()
            disp_h = 340
            disp_w = int(min(disp_h * (w_in / h_in), 900)) if h_in else disp_h
            canvas.setMinimumSize(max(disp_w, 300), disp_h)
            body.addWidget(canvas, stretch=1)

        details = QVBoxLayout()
        details.setSpacing(4)
        details.addWidget(_kv("point", result.coordinate.point_type.value))
        details.addWidget(_kv("method", result.method or "—"))
        details.addWidget(_kv("before", _fmt_xyz(result.initial)))
        if result.fitted is not None:
            details.addWidget(
                _kv("after", _fmt_xyz(result.fitted), value_color="#7fd4ff")
            )
            details.addWidget(
                _kv("Δ x, y, z", _fmt_delta(result.initial, result.fitted))
            )
        details.addStretch()
        details_w = QWidget()
        details_w.setLayout(details)
        details_w.setMinimumWidth(180)
        body.addWidget(details_w)

        layout.addLayout(body)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        if result.status is FitStatus.ERROR:
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(self.reject)
            btn_row.addWidget(close_btn)
        else:
            reject_btn = QPushButton("Reject")
            reject_btn.clicked.connect(self.reject)
            btn_row.addWidget(reject_btn)
            accept_btn = QPushButton("Accept")
            accept_btn.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
            accept_btn.clicked.connect(self.accept)
            btn_row.addWidget(accept_btn)
        layout.addLayout(btn_row)
