"""Widget for computing the slice scaling factor (zeta) from optical parameters.

The widget provides spinboxes for each of the five optical parameters and
displays the interpolated zeta value live as the user adjusts them.

Usage::

    python fibsem/correlation/ui/widgets/test_refractive_index_widget.py
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from fibsem.correlation.refractive_index import ZetaParams, _LUT_PATH, lookup_zeta
from fibsem.ui.widgets.custom_widgets import IconToolButton, TitledPanel, ValueSpinBox

_LUT_MISSING_MSG = (
    f"Correction factor calculator unavailable: LUT file not found at\n{_LUT_PATH}\n"
    "The correction factor can still be edited manually."
)

# Default values representative of typical cryo-CLEM conditions
_DEFAULTS = ZetaParams(
    tilt_deg=15.0,
    depth_um=4.0,
    NA=0.8,
    n2=1.4,
    wavelength_um=0.515,
)

_DEFAULT_FACTOR = 1.47


class RefractiveIndexWidget(QWidget):
    """Form widget that computes the depth scaling factor zeta from optical parameters.

    The correction-factor spinbox updates automatically whenever any spinbox value
    changes (zeta feeds it directly).

    Signals
    -------
    zeta_computed(float):
        Emitted after each recompute with the new zeta value.
    """

    zeta_computed = pyqtSignal(float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._zeta: Optional[float] = None
        self._setup_ui()
        self._connect_signals()
        self._recompute()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        # --- parameter form ---
        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.setContentsMargins(8, 8, 8, 8)
        form.setSpacing(6)

        self._spin_tilt = ValueSpinBox(
            suffix="°", minimum=0.0, maximum=30.0, step=1.0, decimals=0,
            tooltip="The angle between the FIB column and the sample surface",
        )
        self._spin_depth = ValueSpinBox(
            suffix="µm", minimum=0.5, maximum=14.5, step=0.5, decimals=1,
            tooltip="Estimated depth of the feature below the coverslip (has little effect below ~20 µm)",
        )
        self._spin_na = ValueSpinBox(
            minimum=0.2, maximum=0.9, step=0.1, decimals=2,
            tooltip="Numerical aperture of the objective lens",
        )
        self._spin_n2 = ValueSpinBox(
            minimum=1.22, maximum=1.46, step=0.04, decimals=2,
            tooltip="Refractive index of the sample medium (n=1.0 is vacuum/air)",
        )
        # Wavelength spinbox works in nm; converted to µm for LUT lookup
        self._spin_wl = ValueSpinBox(
            suffix="nm", minimum=420.0, maximum=720.0, step=10.0, decimals=0,
            tooltip="Excitation wavelength of the fluorescence channel",
        )

        form.addRow("Milling Angle", self._spin_tilt)
        form.addRow("Depth", self._spin_depth)
        form.addRow("Numerical Aperture", self._spin_na)
        form.addRow("Refractive Index", self._spin_n2)
        form.addRow("Wavelength (λ)", self._spin_wl)

        self._spin_factor = ValueSpinBox(
            minimum=0.1, maximum=10.0, step=0.01, decimals=3,
            tooltip="Correction factor (ζ) applied to the depth below the surface",
        )
        self._spin_factor.setValue(_DEFAULT_FACTOR)
        form.addRow("Correction Factor (ζ)", self._spin_factor)

        self._btn_reset_factor = IconToolButton(
            "mdi:refresh",
            tooltip=f"Reset correction factor to default ({_DEFAULT_FACTOR:.3f})",
        )
        self._btn_reset_factor.clicked.connect(lambda: self._spin_factor.setValue(_DEFAULT_FACTOR))

        lut_available = _LUT_PATH.exists()
        if not lut_available:
            self._lut_warning = QLabel("LUT not found — calculator disabled. Edit correction factor manually.")
            self._lut_warning.setStyleSheet("color: #f0a500; font-style: italic; padding: 4px 8px;")
            self._lut_warning.setWordWrap(True)
            self._lut_warning.setToolTip(_LUT_MISSING_MSG)
            form.addRow(self._lut_warning)

            for spin in (self._spin_tilt, self._spin_depth, self._spin_na, self._spin_n2, self._spin_wl):
                spin.setEnabled(False)
                spin.setToolTip(_LUT_MISSING_MSG)

        panel = TitledPanel("Optical Parameters", content=form_widget)
        panel.add_header_widget(self._btn_reset_factor)
        if not lut_available:
            panel.setToolTip(_LUT_MISSING_MSG)
        outer.addWidget(panel)
        outer.addStretch(1)

        self.set_params(_DEFAULTS)

    def _connect_signals(self) -> None:
        for spin in (
            self._spin_tilt,
            self._spin_depth,
            self._spin_na,
            self._spin_n2,
            self._spin_wl,
        ):
            spin.valueChanged.connect(self._recompute)

        self.zeta_computed.connect(self._spin_factor.setValue)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_zeta(self) -> Optional[float]:
        """Return the last computed zeta value, or ``None`` if not yet computed."""
        return self._zeta

    def get_factor(self) -> float:
        """Return the current correction factor spinbox value."""
        return self._spin_factor.value()

    def set_factor(self, v: float) -> None:
        """Set the correction factor spinbox without triggering recompute."""
        self._spin_factor.blockSignals(True)
        self._spin_factor.setValue(v)
        self._spin_factor.blockSignals(False)

    def set_params(self, params: ZetaParams) -> None:
        """Populate all spinboxes from *params* without triggering recompute."""
        for spin in (
            self._spin_tilt,
            self._spin_depth,
            self._spin_na,
            self._spin_n2,
            self._spin_wl,
        ):
            spin.blockSignals(True)

        self._spin_tilt.setValue(params.tilt_deg)
        self._spin_depth.setValue(params.depth_um)
        self._spin_na.setValue(params.NA)
        self._spin_n2.setValue(params.n2)
        self._spin_wl.setValue(params.wavelength_um * 1000.0)  # µm → nm

        for spin in (
            self._spin_tilt,
            self._spin_depth,
            self._spin_na,
            self._spin_n2,
            self._spin_wl,
        ):
            spin.blockSignals(False)

    def get_params(self) -> ZetaParams:
        """Return the current spinbox values as a :class:`ZetaParams`."""
        return ZetaParams(
            tilt_deg=self._spin_tilt.value(),
            depth_um=self._spin_depth.value(),
            NA=self._spin_na.value(),
            n2=self._spin_n2.value(),
            wavelength_um=self._spin_wl.value() / 1000.0,  # nm → µm
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _recompute(self) -> None:
        if not _LUT_PATH.exists():
            return
        params = self.get_params()
        try:
            zeta = lookup_zeta(
                params.tilt_deg,
                params.depth_um,
                params.NA,
                params.n2,
                params.wavelength_um,
            )
            self._zeta = zeta
            self.zeta_computed.emit(zeta)
        except Exception:
            self._zeta = None
