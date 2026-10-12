"""The fit panel: one point's fit, with its inputs beside it (FIB-1218, FIB-283).

It replaces the accept/reject Confirm fit dialog. The same diagnostic is shown,
and beside it the inputs the fit ran with: the window, the FM channel, how an
FM fiducial looks (dark or bright), and whether the target's x, y are fitted
too. Changing one fits again at once, so a fit that landed on the wrong feature
can be steered onto the right one instead of only rejected. Accept applies the
fit on screen; Keep click applies nothing.

The panel never fits by itself: the caller passes ``refit``, which runs the
point's fit with the inputs given and returns a
:class:`~fibsem.ui.correlation.widgets.fit_confirmation_dialog.PointFitResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional, Sequence

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from fibsem.ui import stylesheets
from fibsem.ui.correlation.widgets.fit_confirmation_dialog import (
    _STATUS_STYLE,
    FitStatus,
    PointFitResult,
    _fmt_delta,
    _fmt_xyz,
    _kv,
    _status_text,
)
from fibsem.ui.tokens import (
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


@dataclass(frozen=True)
class FitInputs:
    """The inputs one point's fit runs with. A field left None doesn't apply to
    that fit and gets no control in the panel."""

    window: int  # half-width of the fit's window, px
    window_label: str = "Window"
    channel: Optional[int] = None  # FM only: index into the stack's channels
    looks: Optional[str] = None  # FM fiducial only: "dark" or "bright"
    fit_xy: Optional[bool] = None  # FM target only: fit x, y too, or keep them


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
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._result = result
        self._inputs = inputs
        self._refit = refit
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
        side.addLayout(form)

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
        tone = _STATUS_STYLE[result.status]
        self._status.setText(_status_text(result))
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
        )

    def _on_input_changed(self, *_) -> None:
        """Fit again with the inputs on screen, and show what came back."""
        inputs = self._read_inputs()
        result = self._refit(inputs)
        if result is None:  # nothing to fit with these inputs
            return
        self._inputs = inputs
        self._result = result
        self._show(result)


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
