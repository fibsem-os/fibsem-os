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

            try:
                result.figure.tight_layout()
            except Exception:
                pass
            canvas = FigureCanvasQTAgg(result.figure)
            # Size to the figure's aspect so wide multi-subplot FM figs
            # (hole_fitting_reflection is 1x3 at 20x5) aren't squished square.
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
