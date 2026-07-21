"""Dialog to configure FM z-stack interpolation (FIB-253).

Collects a target z pixel size and interpolation method for
:func:`fibsem.correlation.util.interpolate_fm_volume`. Interpolation only
resamples z, so the natural default is the XY pixel size — an isotropic volume.
"""
from __future__ import annotations

from typing import Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from fibsem.correlation.util import INTERPOLATION_METHODS
from fibsem.fm.structures import FluorescenceImage
from fibsem.ui.widgets.custom_widgets import ValueComboBox, ValueSpinBox


def _estimate_slices(nz: int, z_step_m: float, target_m: float) -> int:
    """Match scipy.ndimage.zoom's output length (round-half-to-even on the scale)."""
    if target_m <= 0:
        return nz
    return int(round(nz * (z_step_m / target_m)))


class InterpolateZDialog(QDialog):
    """Ask for a target z pixel size + method; expose them after ``exec_()``."""

    def __init__(
        self, fm_image: FluorescenceImage, parent: Optional[QWidget] = None,
        fm_point_count: int = 0,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Interpolate z-stack")
        self.setModal(True)

        meta = fm_image.metadata
        self._nc, self._nz, self._ny, self._nx = fm_image.data.shape
        self._z_step = meta.pixel_size_z  # m; guaranteed non-None by the caller
        self._xy = meta.pixel_size_x  # m
        self._itemsize = fm_image.data.dtype.itemsize

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(12)

        # --- current volume ---
        info = QLabel(
            f"{self._nc} channels · {self._nz} slices · {self._ny} × {self._nx}\n"
            f"XY pixel size: {self._xy * 1e9:.0f} nm    "
            f"Z step: {self._z_step * 1e9:.0f} nm "
            f"({self._z_step / self._xy:.2f}× anisotropic)"
        )
        info.setStyleSheet("color: #b8b8b8; font-size: 12px;")
        root.addWidget(info)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #3a3d42;")
        root.addWidget(line)

        form = QFormLayout()
        form.setSpacing(10)

        self._chk_iso = QCheckBox("Match XY pixel size (isotropic)")
        self._chk_iso.setChecked(True)
        self._chk_iso.toggled.connect(self._on_iso_toggled)
        form.addRow(self._chk_iso)

        self._spin_target = ValueSpinBox(
            suffix="nm", minimum=1.0, maximum=100000.0, step=10.0, decimals=1,
        )
        self._spin_target.setValue(self._xy * 1e9)
        self._spin_target.setEnabled(False)  # isotropic on by default
        self._spin_target.valueChanged.connect(self._update_preview)
        form.addRow("Target z pixel size", self._spin_target)

        self._combo_method = ValueComboBox(list(INTERPOLATION_METHODS), value="linear")
        form.addRow("Method", self._combo_method)
        root.addLayout(form)

        self._preview = QLabel()
        self._preview.setStyleSheet(
            "color: #cfe0f2; background: #21303f; border: 0.5px solid #2f4a63;"
            " border-radius: 6px; padding: 8px 10px; font-size: 12px;"
        )
        root.addWidget(self._preview)

        if fm_point_count:
            warn = QLabel(
                f"⚠  {fm_point_count} FM point"
                f"{'s' if fm_point_count != 1 else ''} carry a z-index — "
                "they'll be rescaled to the new slice count."
            )
            warn.setWordWrap(True)
            warn.setStyleSheet("color: #d6b57a; font-size: 12px;")
            root.addWidget(warn)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        ok_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok_btn.setText("Interpolate")
        # No default button: Enter while editing a field should commit that field,
        # not fire off a tens-of-seconds interpolation. Require a real click.
        for btn in (ok_btn, buttons.button(QDialogButtonBox.StandardButton.Cancel)):
            btn.setAutoDefault(False)
            btn.setDefault(False)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._update_preview()

    def keyPressEvent(self, event) -> None:
        # Don't let Enter auto-accept. autoDefault=False isn't enough — a shown
        # QDialogButtonBox re-promotes its accept button to default — so swallow
        # Return/Enter here. A focused field (e.g. the spinbox) still gets the key
        # first and commits its edit; only the dialog-level default is blocked, so
        # a real click on Interpolate is required. Escape still cancels.
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            event.accept()
            return
        super().keyPressEvent(event)

    # ------------------------------------------------------------------

    def _on_iso_toggled(self, checked: bool) -> None:
        self._spin_target.setEnabled(not checked)
        if checked:
            self._spin_target.setValue(self._xy * 1e9)
        self._update_preview()

    def _update_preview(self) -> None:
        target_m = self.target_z_size_m()
        new_nz = _estimate_slices(self._nz, self._z_step, target_m)
        gb = self._nc * new_nz * self._ny * self._nx * self._itemsize / 1e9
        iso = " · isotropic" if abs(target_m - self._xy) < 1e-12 else ""
        self._preview.setText(
            f"{self._nz} → {new_nz} slices{iso} · ~{gb:.1f} GB"
        )

    # ------------------------------------------------------------------
    # Results (valid after exec_)

    def target_z_size_m(self) -> float:
        """Chosen target z pixel size in metres."""
        if self._chk_iso.isChecked():
            return self._xy
        return self._spin_target.value() * 1e-9

    def method(self) -> str:
        return self._combo_method.value()

    def result_params(self) -> Tuple[float, str]:
        return self.target_z_size_m(), self.method()
