"""The fit panel: one point's fit, with its inputs beside it (FIB-1218, FIB-283).

It replaces the accept/reject Confirm fit dialog. The same diagnostic is shown,
and beside it the inputs the fit ran with: the window, the FM channel, how an
FM fiducial looks (dark or bright), and whether the target's x, y are fitted
too. Changing one fits again at once, so a fit that landed on the wrong feature
can be steered onto the right one instead of only rejected. Accept applies the
fit on screen; Keep click applies nothing.

The panel never fits by itself: the caller passes ``refit``, which runs the
point's fit with the inputs given and returns a
:class:`~fibsem.ui.correlation.widgets.fit_result.PointFitResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from fibsem.correlation.fit_diagnostics import _C_FIT, _C_IN
from fibsem.correlation.structures import PointXYZ
from fibsem.ui import stylesheets
from fibsem.ui.correlation.widgets.fit_result import (
    _STATUS_STYLE,
    FitStatus,
    PointFitResult,
    _fmt_delta,
    _fmt_xyz,
    _kv,
    _status_text,
)
from fibsem.ui.tokens import (
    ACCENT_COLOR,
    BODY_MUTED_STYLE,
    BODY_STYLE,
    CANVAS_BG,
    PANEL_COLOR,
    TEXT_COLOR,
    state_color,
    state_style,
)
from fibsem.ui.widgets.custom_widgets import ValueComboBox

# The window's half-width, in px, the panel offers.
WINDOW_RANGE = (2, 40)
# The figure's area keeps this size with no figure in it.
FIGURE_MIN_SIZE = (560, 340)
# The plane preview: this many px either side of the point, drawn this much bigger.
PREVIEW_HALF, PREVIEW_SCALE = 20, 5
# The method suffix of a result whose z was set by hand in the panel.
Z_BY_HAND = "z by hand"


@dataclass(frozen=True)
class FitInputs:
    """The inputs one point's fit runs with. A field left None doesn't apply to
    that fit and gets no control in the panel."""

    window: int  # half-width of the fit's window, px
    window_label: str = "Window"
    channel: Optional[int] = None  # FM only: index into the stack's channels
    looks: Optional[str] = None  # FM fiducial only: "dark" or "bright"
    fit_xy: Optional[bool] = None  # FM target only: fit x, y too, or keep them
    planes: Optional[Tuple[int, int]] = None  # FM fiducial only: below, above


# exec_() returns this when the user keeps their click (Keep click), so a caller
# can tell it from closing the panel (QDialog.Rejected), which stops a run of
# points.
KEPT = 2


class FitPanelDialog(QDialog):
    """Modal fit panel for one :class:`PointFitResult`.

    ``exec_()`` returns ``QDialog.Accepted`` when the user accepts (the fit to
    apply is then :attr:`result`, the one on screen when Accept was pressed),
    :data:`KEPT` for Keep click, and ``QDialog.Rejected`` when the panel is
    closed. ``position`` is ``(index, count)`` of this point among the ones
    being fitted in turn; while there is a next one, the buttons say so.
    """

    def __init__(
        self,
        result: PointFitResult,
        inputs: FitInputs,
        refit: Callable[[FitInputs], Optional[PointFitResult]],
        channel_names: Sequence[str] = (),
        position: Optional[tuple] = None,
        stack: Optional[Callable[[int], np.ndarray]] = None,
        save_default: Optional[Callable[[FitInputs], None]] = None,
        default_label: str = "this kind of point",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._result = result
        self._fitted = result  # the last fit, under any z set by hand
        self._inputs = inputs
        self._refit = refit
        self._stack = stack  # FM: the z-stack of a channel, for browsing planes
        self._canvas = None
        point = result.coordinate.point_type.value
        self.setWindowTitle(f"Fit {point}")
        self.setModal(True)
        self.setStyleSheet(f"background: {CANVAS_BG}; color: {TEXT_COLOR};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 12)
        layout.setSpacing(10)

        head = QHBoxLayout()
        where = f" {position[0] + 1} of {position[1]}" if position else ""
        title = QLabel(f"Fit · {point}{where} · {result.method}")
        title.setStyleSheet(f"{BODY_STYLE} font-weight: 600;")
        head.addWidget(title)
        head.addStretch(1)
        self._status = QLabel()
        head.addWidget(self._status)
        layout.addLayout(head)

        # Why the fit failed: the plain message, then the fit's own reason (the
        # raw text), which is what to steer by when changing the inputs.
        self._reason = QLabel()
        self._reason.setWordWrap(True)
        self._reason.setStyleSheet(state_style("error"))
        layout.addWidget(self._reason)
        self._detail = QLabel()
        self._detail.setWordWrap(True)
        self._detail.setStyleSheet(BODY_MUTED_STYLE)
        self._detail.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self._detail)

        body = QHBoxLayout()
        body.setSpacing(16)
        # A fixed home for the figure, so a failed fit (no figure) doesn't
        # collapse the panel under the inputs that might fix it.
        figure_box = QWidget()
        figure_box.setMinimumSize(*FIGURE_MIN_SIZE)
        self._figure_slot = QVBoxLayout(figure_box)
        self._figure_slot.setContentsMargins(0, 0, 0, 0)
        self._no_figure = QLabel(
            "No fit to show. Change the inputs, or keep your click."
        )
        self._no_figure.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._no_figure.setStyleSheet(BODY_MUTED_STYLE)
        self._figure_slot.addWidget(self._no_figure)
        body.addWidget(figure_box, stretch=1)

        side = QVBoxLayout()
        side.setSpacing(10)
        caption = QLabel("Inputs · fits again on change")
        caption.setStyleSheet(BODY_MUTED_STYLE)
        side.addWidget(caption)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        self._window_spin = QSpinBox()
        self._window_spin.setRange(*WINDOW_RANGE)
        self._window_spin.setSuffix(" px")
        self._window_spin.setValue(inputs.window)
        self._window_spin.setKeyboardTracking(False)  # one fit per value, not per key
        self._window_spin.valueChanged.connect(self._on_input_changed)
        form.addRow(_muted(inputs.window_label), self._window_spin)

        self._channel_combo = None
        if inputs.channel is not None:
            self._channel_combo = QComboBox()
            self._channel_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
            self._channel_combo.addItems(list(channel_names))
            self._channel_combo.setCurrentIndex(inputs.channel)
            self._channel_combo.currentIndexChanged.connect(self._on_input_changed)
            form.addRow(_muted("Channel"), self._channel_combo)

        self._looks_combo = None
        if inputs.looks is not None:
            self._looks_combo = ValueComboBox(
                ["dark", "bright"],
                value=inputs.looks,
                format_fn={"dark": "Dark", "bright": "Bright"}.get,
            )
            self._looks_combo.currentIndexChanged.connect(self._on_input_changed)
            form.addRow(_muted("Burn looks"), self._looks_combo)

        self._fit_xy_check = None
        if inputs.fit_xy is not None:
            self._fit_xy_check = QCheckBox("Fit x, y too")
            self._fit_xy_check.setChecked(inputs.fit_xy)
            self._fit_xy_check.setToolTip(
                "Off: z is fitted and x, y stay where you clicked."
            )
            self._fit_xy_check.toggled.connect(self._on_input_changed)
            form.addRow(self._fit_xy_check)

        self._planes_spins = None
        if inputs.planes is not None:
            below, above = QSpinBox(), QSpinBox()
            for spin, value in ((below, inputs.planes[0]), (above, inputs.planes[1])):
                spin.setRange(1, 50)
                spin.setValue(value)
                spin.setKeyboardTracking(False)
                spin.valueChanged.connect(self._on_input_changed)
            below.setSuffix(" below")
            above.setSuffix(" above")
            self._planes_spins = (below, above)
            form.addRow(_muted("Planes searched"), below)
            form.addRow(_muted(""), above)
        side.addLayout(form)

        # z by eye: browse the stack's planes around the point, and take one as
        # z when the fit can't (FIB-1218).
        self._plane_slider = None
        if stack is not None and inputs.channel is not None:
            side.addWidget(_muted("z · browse planes"))
            self._preview = QLabel()
            self._preview.setFixedSize(
                (2 * PREVIEW_HALF + 1) * PREVIEW_SCALE,
                (2 * PREVIEW_HALF + 1) * PREVIEW_SCALE,
            )
            side.addWidget(self._preview)
            self._plane_slider = QSlider(Qt.Orientation.Horizontal)
            self._plane_slider.valueChanged.connect(self._on_plane_changed)
            side.addWidget(self._plane_slider)
            self._use_plane_btn = QPushButton()
            self._use_plane_btn.setAutoDefault(False)
            self._use_plane_btn.clicked.connect(self._use_plane)
            side.addWidget(self._use_plane_btn)

        self._details = QVBoxLayout()
        self._details.setSpacing(4)
        side.addLayout(self._details)
        side.addStretch(1)
        side_w = QWidget()
        side_w.setLayout(side)
        side_w.setMinimumWidth(240)
        body.addWidget(side_w)
        layout.addLayout(body)

        # Accept and Keep click both move on to the next point when there is
        # one; closing the panel stops (FIB-1218).
        more = position is not None and position[0] + 1 < position[1]
        nxt = ", next \u203a" if more else ""
        buttons = QHBoxLayout()
        self._save_btn = None
        if save_default is not None:
            # The inputs on screen become the experiment's fit settings for
            # this kind of point (FIB-1218).
            self._save_default = save_default
            self._save_label = f"Save as default for {default_label}"
            self._save_btn = QPushButton(self._save_label)
            self._save_btn.setAutoDefault(False)
            self._save_btn.setToolTip(
                "Fit every point of this kind with these inputs, from now on in "
                "this experiment. The Method panel shows them."
            )
            self._save_btn.clicked.connect(self._on_save_default)
            buttons.addWidget(self._save_btn)
        buttons.addStretch(1)
        keep = QPushButton(f"Keep click{nxt}")
        keep.setToolTip("Leave the point where you placed it")
        keep.setAutoDefault(False)
        keep.clicked.connect(lambda: self.done(KEPT))
        buttons.addWidget(keep)
        self._keep_btn = keep
        self._accept_btn = QPushButton(f"Accept{nxt}")
        self._accept_btn.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self._accept_btn.setDefault(True)  # Enter accepts
        self._accept_btn.clicked.connect(self.accept)
        buttons.addWidget(self._accept_btn)
        layout.addLayout(buttons)

        self._show(result)

    # -- what the panel shows -------------------------------------------------

    @property
    def result(self) -> PointFitResult:
        """The fit on screen: what Accept applies."""
        return self._result

    @property
    def inputs(self) -> FitInputs:
        return self._inputs

    def _show(self, result: PointFitResult) -> None:
        by_hand = result.method.endswith(Z_BY_HAND)
        tone = ACCENT_COLOR if by_hand else _STATUS_STYLE[result.status]
        self._status.setText("z set by hand" if by_hand else _status_text(result))
        self._status.setStyleSheet(
            f"{BODY_STYLE} color: {tone}; background: {PANEL_COLOR}; "
            f"border: 1px solid {tone}; border-radius: 10px; padding: 4px 12px;"
        )
        self._status.setToolTip(result.detail or "")
        failed = result.status is FitStatus.ERROR
        self._reason.setText((result.message or "") if failed else "")
        self._reason.setToolTip(result.detail or "")
        self._reason.setVisible(failed and bool(result.message))
        self._detail.setText((result.detail or "") if failed else "")
        self._detail.setVisible(failed and bool(result.detail))
        self._accept_btn.setEnabled(not failed)

        _clear(self._details)
        if result.channel is not None:
            self._details.addWidget(
                _kv("channel", result.channel_name or f"channel {result.channel}")
            )
        self._details.addWidget(_kv("before", _fmt_xyz(result.initial)))
        if result.fitted is not None:
            self._details.addWidget(
                _kv("after", _fmt_xyz(result.fitted), value_color=state_color("info"))
            )
            self._details.addWidget(
                _kv("Δ x, y, z", _fmt_delta(result.initial, result.fitted))
            )

        self._show_figure(result.diagnostic)
        if self._plane_slider is not None and not by_hand:
            self._follow_plane(result)

    # -- planes ---------------------------------------------------------------

    def _channel_stack(self) -> np.ndarray:
        return np.asarray(self._stack(self._inputs.channel))

    def _follow_plane(self, result: PointFitResult) -> None:
        """Show the plane the fit landed on (or the clicked one)."""
        nz = self._channel_stack().shape[0]
        point = result.fitted or result.initial
        self._plane_slider.blockSignals(True)
        self._plane_slider.setRange(0, nz - 1)
        self._plane_slider.setValue(min(max(int(round(point.z)), 0), nz - 1))
        self._plane_slider.blockSignals(False)
        self._on_plane_changed(self._plane_slider.value())

    def _on_plane_changed(self, plane: int) -> None:
        self._use_plane_btn.setText(f"Use plane {plane} as z")
        stack = self._channel_stack()
        point = self._result.fitted or self._result.initial
        self._preview.setPixmap(
            _plane_pixmap(stack, plane, self._result.initial, point)
        )

    def _use_plane(self) -> None:
        """z = the plane on show; x, y from the last fit, or the click."""
        plane = self._plane_slider.value()
        base = self._fitted.fitted or self._fitted.initial
        fitted = PointXYZ(base.x, base.y, float(plane))
        self._result = replace(
            self._fitted,
            fitted=fitted,
            status=PointFitResult.classify(self._fitted.initial, fitted),
            method=f"{self._fitted.method} · {Z_BY_HAND}",
            message=None,
            detail=None,
        )
        self._show(self._result)

    def _show_figure(self, diagnostic) -> None:
        if self._canvas is not None:
            # hidden now: deleteLater only runs when the event loop gets to it
            self._figure_slot.removeWidget(self._canvas)
            self._canvas.hide()
            self._canvas.deleteLater()
            self._canvas = None
        self._no_figure.setVisible(diagnostic is None)
        if diagnostic is None:
            return
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

        from fibsem.correlation.fit_diagnostics import plot_fit_diagnostic

        fig = plot_fit_diagnostic(diagnostic, dark=True)
        canvas = FigureCanvasQTAgg(fig)
        w_in, h_in = fig.get_size_inches()
        height = 340
        width = int(min(height * (w_in / h_in), 900)) if h_in else height
        canvas.setMinimumSize(max(width, 300), height)
        self._figure_slot.addWidget(canvas)
        self._canvas = canvas

    # -- inputs ---------------------------------------------------------------

    def _read_inputs(self) -> FitInputs:
        return replace(
            self._inputs,
            window=self._window_spin.value(),
            channel=(
                self._channel_combo.currentIndex()
                if self._channel_combo is not None
                else None
            ),
            looks=self._looks_combo.value() if self._looks_combo is not None else None,
            fit_xy=(
                self._fit_xy_check.isChecked()
                if self._fit_xy_check is not None
                else None
            ),
            planes=(
                tuple(s.value() for s in self._planes_spins)
                if self._planes_spins is not None
                else None
            ),
        )

    def _on_save_default(self) -> None:
        self._save_default(self._read_inputs())
        self._save_btn.setText("Saved as default \u2713")
        self._save_btn.setEnabled(False)

    def _on_input_changed(self, *_) -> None:
        """Fit again with the inputs on screen, and show what came back."""
        if self._save_btn is not None:  # these inputs aren't the saved ones
            self._save_btn.setText(self._save_label)
            self._save_btn.setEnabled(True)
        inputs = self._read_inputs()
        result = self._refit(inputs)
        if result is None:  # nothing to fit with these inputs
            return
        self._inputs = inputs
        self._result = self._fitted = result
        self._show(result)


def _plane_pixmap(stack, plane, click: PointXYZ, point: PointXYZ) -> QPixmap:
    """One plane of ``stack`` around ``point``, the click (red) and the answer
    (green) marked. The contrast is the crop's through every plane, so planes
    compare."""
    nz, height, width = stack.shape
    h, s = PREVIEW_HALF, PREVIEW_SCALE
    cx, cy = int(round(point.x)), int(round(point.y))
    x0, y0 = (
        min(max(cx - h, 0), max(width - 2 * h - 1, 0)),
        min(max(cy - h, 0), max(height - 2 * h - 1, 0)),
    )
    crop = np.asarray(stack[:, y0 : y0 + 2 * h + 1, x0 : x0 + 2 * h + 1], dtype=float)
    lo, hi = np.percentile(crop, 1), np.percentile(crop, 99.5)
    view = np.clip((crop[plane] - lo) / (hi - lo or 1.0), 0, 1) * 255
    view = np.ascontiguousarray(
        np.kron(view.astype(np.uint8), np.ones((s, s), np.uint8))
    )
    image = QImage(
        view.data, view.shape[1], view.shape[0], view.shape[1], QImage.Format_Grayscale8
    )
    pixmap = QPixmap.fromImage(image.copy())
    painter = QPainter(pixmap)
    # the diagnostic figure's input and fit marker colours, so the two match
    for p, colour in ((click, _C_IN), (point, _C_FIT)):
        painter.setPen(QPen(QColor(colour), 2))
        c = QPointF((p.x - x0 + 0.5) * s, (p.y - y0 + 0.5) * s)
        painter.drawLine(c - QPointF(8, 0), c + QPointF(8, 0))
        painter.drawLine(c - QPointF(0, 8), c + QPointF(0, 8))
    painter.end()
    return pixmap


def _muted(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(BODY_MUTED_STYLE)
    return label


def _clear(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item.widget() is not None:
            item.widget().hide()  # deleteLater waits for the event loop
            item.widget().deleteLater()
