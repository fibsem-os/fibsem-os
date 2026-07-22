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
    QLayout,
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
    """Outcome of auto-fitting one coordinate (runtime only)."""

    coordinate: Coordinate
    method: str
    channel: Optional[int]
    initial: PointXYZ
    fitted: Optional[PointXYZ]
    status: FitStatus
    message: Optional[str] = None
    channel_name: Optional[str] = None  # display label for `channel` (FM only)
    detail: Optional[str] = None        # raw error, surfaced as a tooltip
    diagnostic: object = None  # FitDiagnostic; rendered on demand, not serialised

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
    return "Fit failed"  # the (long) reason goes in a separate wrapped line


def humanize_fit_error(exc: Exception) -> str:
    """Turn a raw fit exception into an actionable, non-jargon message.

    scipy surfaces failures like *"Optimal parameters not found: Number of calls
    to function has reached maxfev = 800"* — meaningless to a user. Map the known
    modes to plain guidance; the raw text is still logged (and shown as a tooltip
    via :attr:`PointFitResult.detail`) so debugging isn't lost.
    """
    msg = str(exc)
    if "maxfev" in msg or "Optimal parameters not found" in msg:
        return (
            "The fit didn't converge — the feature may be too faint, or not "
            "centred in the search region. Try re-centring your click, or a "
            "different channel / method."
        )
    if isinstance(exc, IndexError) or "out of bounds" in msg or "too close" in msg:
        return "The point is too close to the image edge to fit a region around it."
    if isinstance(exc, (ValueError, ZeroDivisionError)) and (
        "empty" in msg or "zero-size" in msg or "float division" in msg
    ):
        return "The search region is empty or featureless — nothing to fit."
    return "The fit failed unexpectedly (see log for details)."


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
        if result.detail:  # raw error kept reachable without cluttering the chip
            status.setToolTip(result.detail)
        layout.addWidget(status, alignment=Qt.AlignmentFlag.AlignLeft)

        # The humanized failure reason can be a couple of sentences — keep it out
        # of the chip (which would force the whole dialog absurdly wide) and show
        # it as a wrapped detail line, capped so it wraps instead of stretching.
        if result.status is FitStatus.ERROR and result.message:
            reason = QLabel(result.message)
            reason.setWordWrap(True)
            reason.setMaximumWidth(440)
            reason.setStyleSheet("color: #d9b3b3; font-size: 12px;")
            if result.detail:
                reason.setToolTip(result.detail)
            layout.addWidget(reason)

        body = QHBoxLayout()
        body.setSpacing(14)

        # The diagnostic figure is always built by the fit, so always embed it
        # and toggle its visibility at runtime (see the "diagnostic" button).
        # `show_figure` is only the INITIAL state — the pre-fit checkbox default.
        self._canvas = None
        self._diag_shown = False
        if result.diagnostic is not None:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

            from fibsem.correlation.fit_diagnostics import plot_fit_diagnostic

            # Render on demand, dark to match this dialog. The figure is built
            # via the OO API (no pyplot), so it's freed with the canvas — no
            # plt.close needed.
            fig = plot_fit_diagnostic(result.diagnostic, dark=True)
            canvas = FigureCanvasQTAgg(fig)
            # Size to the figure's aspect so the wide FM figs (z + XY panels
            # at 9x4.5) aren't squished square.
            w_in, h_in = fig.get_size_inches()
            disp_h = 340
            disp_w = int(min(disp_h * (w_in / h_in), 900)) if h_in else disp_h
            canvas.setMinimumSize(max(disp_w, 300), disp_h)
            canvas.setVisible(show_figure)
            self._canvas = canvas
            self._diag_shown = show_figure
            body.addWidget(canvas, stretch=1)

        details = QVBoxLayout()
        details.setSpacing(4)
        details.addWidget(_kv("point", result.coordinate.point_type.value))
        details.addWidget(_kv("method", result.method or "—"))
        if result.channel is not None:  # FM fits run on a specific channel
            details.addWidget(
                _kv("channel", result.channel_name or f"channel {result.channel}")
            )
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
        # Diagnostic toggle sits on the left, apart from the accept/reject cluster.
        # It's added first, so opt it out of autoDefault or it would become the
        # dialog's default button (Enter would toggle it instead of accepting).
        if self._canvas is not None:
            self._toggle_btn = QPushButton(self._diag_btn_text(self._diag_shown))
            self._toggle_btn.setToolTip("Show or hide the fit diagnostic plot")
            self._toggle_btn.setAutoDefault(False)
            self._toggle_btn.setDefault(False)
            self._toggle_btn.clicked.connect(self._on_toggle_diagnostic)
            btn_row.addWidget(self._toggle_btn)
        btn_row.addStretch(1)
        if result.status is FitStatus.ERROR:
            close_btn = QPushButton("Close")
            close_btn.setDefault(True)
            close_btn.clicked.connect(self.reject)
            btn_row.addWidget(close_btn)
        else:
            reject_btn = QPushButton("Reject")
            reject_btn.setAutoDefault(False)
            reject_btn.clicked.connect(self.reject)
            btn_row.addWidget(reject_btn)
            accept_btn = QPushButton("Accept")
            accept_btn.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
            accept_btn.setAutoDefault(True)
            accept_btn.setDefault(True)  # Enter accepts the fit
            accept_btn.clicked.connect(self.accept)
            btn_row.addWidget(accept_btn)
            accept_btn.setFocus()
        layout.addLayout(btn_row)

        # Fit the dialog to its contents so toggling the diagnostic grows and
        # shrinks it cleanly. Only when a figure is present — the error layout
        # (a word-wrapped reason label) sizes better without a fixed constraint.
        if self._canvas is not None:
            layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)

    @staticmethod
    def _diag_btn_text(shown: bool) -> str:
        return "Hide diagnostic" if shown else "Show diagnostic"

    def _on_toggle_diagnostic(self) -> None:
        """Reveal/collapse the embedded diagnostic figure and re-fit the dialog."""
        self._diag_shown = not self._diag_shown
        self._canvas.setVisible(self._diag_shown)
        self._toggle_btn.setText(self._diag_btn_text(self._diag_shown))
        self.adjustSize()  # immediate re-fit (SetFixedSize also keeps it snug)
