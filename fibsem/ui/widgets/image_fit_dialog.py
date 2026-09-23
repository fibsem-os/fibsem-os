"""Point pairs between an image and the overview it is laid over (FIB-1030).

Users find three clicked pairs easier than a drag, and three pairs are enough for the
similarity a flat sample allows. Two canvases from the correlation tab -- the
reference overview on the left, the image on the right -- share one point store, so
the i-th point on each side is a pair. A live readout says how many pairs there are
and, once a fit is possible, what it would give.

The dialog does no geometry. The host hands it the two arrays and a callback that
turns pairs into a preview fit; on accept the host applies the same pairs for real.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from fibsem.correlation.similarity import SimilarityFit
from fibsem.correlation.structures import PointType
from fibsem.ui import stylesheets
from fibsem.ui.correlation.point_store import CorrelationPointStore
from fibsem.ui.correlation.widgets.correlation_canvas_widget import (
    CorrelationCanvasWidget,
)
from fibsem.ui.tokens import CANVAS_BG, TEXT_COLOR

Pairs = List[Tuple[float, float, float, float]]  # image x, y, reference x, y
PreviewFit = Callable[[Pairs, bool], Optional[SimilarityFit]]

MIN_PAIRS = 3


class ImageFitDialog(QDialog):
    """Pick matching points on the overview and on the image; fit on accept."""

    def __init__(
        self,
        reference: np.ndarray,
        image: np.ndarray,
        preview: Optional[PreviewFit] = None,
        reference_label: str = "Overview",
        image_label: str = "Image",
        rms_text: Optional[Callable[[float], str]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Fit the image from points")
        self.setStyleSheet(f"background: {CANVAS_BG}; color: {TEXT_COLOR};")
        self.resize(1200, 700)
        self._preview = preview
        # How to say a residual: the host knows what a canvas unit is in metres,
        # the dialog does not. Canvas units, labelled as such, when not told.
        self._rms_text = rms_text or (lambda rms: f"RMS {rms:.1f} canvas px")
        self._store = CorrelationPointStore(self)

        self.reference_canvas = CorrelationCanvasWidget(
            allowed_point_types=[PointType.FIB], store=self._store, side="fib"
        )
        self.image_canvas = CorrelationCanvasWidget(
            allowed_point_types=[PointType.FM], store=self._store, side="fm"
        )
        for canvas in (self.reference_canvas, self.image_canvas):
            canvas.point_add_requested.connect(self._on_add_requested)
            canvas.set_legend_visible(False)
        self.reference_canvas.set_image(np.asarray(reference), cmap="gray")
        self.image_canvas.set_image(np.asarray(image))
        self._store.structure_changed.connect(self._refresh)
        self._store.points_changed.connect(self._refresh)

        layout = QVBoxLayout(self)
        hint = QLabel(
            "Click the same feature on both sides, at least three times. "
            "The i-th point on each side makes a pair."
        )
        hint.setStyleSheet(stylesheets.LABEL_INSTRUCTIONS_STYLE)
        layout.addWidget(hint)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._pane(reference_label, self.reference_canvas))
        splitter.addWidget(self._pane(image_label, self.image_canvas))
        layout.addWidget(splitter, 1)

        footer = QHBoxLayout()
        self.label_status = QLabel("")
        self.label_status.setStyleSheet(stylesheets.LABEL_INSTRUCTIONS_STYLE)
        self.check_lock_scale = QCheckBox("Lock scale")
        self.check_lock_scale.setToolTip(
            "Fit only the turn and the offset; keep the image at its own pixel size"
        )
        self.check_lock_scale.toggled.connect(self._refresh)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_fit = QPushButton("Fit and apply")
        self.btn_fit.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.btn_fit.clicked.connect(self.accept)
        footer.addWidget(self.label_status, 1)
        footer.addWidget(self.check_lock_scale)
        footer.addWidget(self.btn_cancel)
        footer.addWidget(self.btn_fit)
        layout.addLayout(footer)
        self._refresh()

    @staticmethod
    def _pane(title: str, canvas: QWidget) -> QWidget:
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        label = QLabel(title)
        label.setStyleSheet(stylesheets.IMAGE_HEADER_STYLE)
        layout.addWidget(label)
        layout.addWidget(canvas, 1)
        return pane

    # ── the pairs ─────────────────────────────────────────────────────────

    @property
    def fix_scale(self) -> bool:
        return self.check_lock_scale.isChecked()

    def pairs(self) -> Pairs:
        """(image x, y, reference x, y) per pair, in the arrays' own pixels."""
        references = self._store.of_type(PointType.FIB)
        images = self._store.of_type(PointType.FM)
        return [
            (float(i.point.x), float(i.point.y), float(r.point.x), float(r.point.y))
            for r, i in zip(references, images)
        ]

    def add_pair(self, image_xy: Sequence[float], reference_xy: Sequence[float]):
        """Add a pair without clicking -- for a host seeding from a record, or tests."""
        from fibsem.correlation.structures import Coordinate, PointXYZ

        self._store.add(
            Coordinate(PointXYZ(reference_xy[0], reference_xy[1], 0.0), PointType.FIB)
        )
        self._store.add(
            Coordinate(PointXYZ(image_xy[0], image_xy[1], 0.0), PointType.FM)
        )

    def _on_add_requested(self, x: float, y: float, point_type: PointType) -> None:
        from fibsem.correlation.structures import Coordinate, PointXYZ

        self._store.add(Coordinate(PointXYZ(x, y, 0.0), point_type))

    def _refresh(self, *_args) -> None:
        pairs = self.pairs()
        references = len(self._store.of_type(PointType.FIB))
        images = len(self._store.of_type(PointType.FM))
        enough = len(pairs) >= MIN_PAIRS
        self.btn_fit.setEnabled(enough)
        if not enough:
            self.label_status.setText(
                f"{len(pairs)} of {MIN_PAIRS} pairs ({references} on the overview, "
                f"{images} on the image)"
            )
            return
        text = f"{len(pairs)} pairs"
        if references != images:
            text += f" ({references} on the overview, {images} on the image; the extra is ignored)"
        if self._preview is not None:
            try:
                fit = self._preview(pairs, self.fix_scale)
            except Exception as e:  # noqa: BLE001 - shown, not raised into a slot
                fit = None
                text += f" — cannot fit: {e}"
            if fit is not None:
                text += (
                    f" — {self._rms_text(fit.rms)}, turn {fit.rotation:+.2f}°, "
                    f"scale ×{fit.scale:.3f}"
                )
        self.label_status.setText(text)
