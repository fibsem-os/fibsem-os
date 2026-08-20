"""Widget for computing the slice scaling factor (zeta) from optical parameters.

The widget provides spinboxes for each of the five optical parameters and
displays the interpolated zeta value live as the user adjusts them.

Usage::

    python fibsem/ui/correlation/widgets/test_refractive_index_widget.py
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from fibsem.correlation.refractive_index import (
    _LUT_PATH,
    ZetaParams,
    _ensure_lut,
    lookup_zeta,
)
from fibsem.ui.tokens import (
    NEUTRAL_500,
)
from fibsem.ui.widgets.custom_widgets import IconToolButton, TitledPanel, ValueSpinBox

_LUT_MISSING_MSG = (
    f"Correction factor calculator unavailable: LUT file not found at\n{_LUT_PATH}\n"
    "The correction factor can still be edited manually."
)

# The configuration Perez et al. use for depth correction throughout: vitrified
# cytoplasm (n = 1.35) through the Arctis iFLM's NA 0.75 objective, for which the
# paper reports a scaling factor of 1.5 (doi:10.64898/2026.05.11.724418 — the
# source of the lookup table below). Only NA and n2 move zeta appreciably; it is
# near-flat in tilt, depth and wavelength, so those stay at typical values.
_DEFAULTS = ZetaParams(
    tilt_deg=15.0,
    depth_um=4.0,
    NA=0.75,
    n2=1.35,
    wavelength_um=0.515,
)

# Only used when the lookup table cannot be read. Must equal the LUT's value at
# _DEFAULTS — pinned by tests/correlation/test_default_correction_factor.py, so
# change the two together or that test fails.
_FALLBACK_FACTOR = 1.502


def _default_factor() -> float:
    """The correction factor for `_DEFAULTS`, from the lookup table.

    Derived rather than declared. A literal constant beside a lookup table is a
    second source of truth for the same number, and it drifts silently the moment
    either side moves: the previous constant (1.470) described no configuration
    the widget could be in, while its own defaults asked the table for 1.595
    (FIB-593). Deriving means the reset button can only ever hand back the factor
    the parameters above it describe.
    """
    try:
        return lookup_zeta(
            tilt_deg=_DEFAULTS.tilt_deg,
            depth_um=_DEFAULTS.depth_um,
            NA=_DEFAULTS.NA,
            n2=_DEFAULTS.n2,
            wavelength_um=_DEFAULTS.wavelength_um,
        )
    except Exception:
        logging.debug(
            "Lookup table unusable; falling back to the stored default correction factor",
            exc_info=True,
        )
        return _FALLBACK_FACTOR

# The correlation panels run at 11-12px. Controls and QFormLayout's auto-built
# string labels default to the app font, which renders larger than the values
# beside them — the field name ends up shouting over its own data.
_CONTROL_STYLE = "font-size: 12px;"
_FORM_LABEL_STYLE = f"color: {NEUTRAL_500}; font-size: 11px;"


def _form_label(text: str) -> QLabel:
    """A small, muted form-row label — pass instead of a bare string."""
    lbl = QLabel(text)
    lbl.setStyleSheet(_FORM_LABEL_STYLE)
    return lbl


_TILT_TOOLTIP = "The angle between the FIB column and the sample surface"
_TILT_LOCKED_TOOLTIP = (
    "Locked to 0°: the FM-surface correction acts along the optical axis, "
    "which is normal to the sample surface."
)


class RefractiveIndexWidget(QWidget):
    """Form widget that computes the depth scaling factor zeta from optical parameters.

    The correction-factor spinbox updates automatically whenever any spinbox value
    changes (zeta feeds it directly).

    Signals
    -------
    zeta_computed(float):
        Emitted after each recompute with the new zeta value.
    factor_changed(float):
        Emitted when the correction-factor spinbox moves under the user -- typed
        directly, reset, or recomputed from an optical parameter. Deliberately
        silent for :meth:`set_factor`, which mirrors an already-stored value:
        the point of the signal is to tell a consumer that what is on screen is
        no longer what has been applied (FIB-591).
    """

    zeta_computed = pyqtSignal(float)
    factor_changed = pyqtSignal(float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._zeta: Optional[float] = None
        self._tilt_locked = False
        self._tilt_before_lock: Optional[float] = None
        # Params the user has typed by hand; metadata seeding leaves these alone.
        self._user_edited: set = set()
        try:
            _ensure_lut()
        except Exception as e:
            logging.warning(f"Failed to download refractive index LUT: {e}")
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
            tooltip=_TILT_TOOLTIP,
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

        form.addRow(_form_label("Milling Angle"), self._spin_tilt)
        form.addRow(_form_label("Depth"), self._spin_depth)
        form.addRow(_form_label("Numerical Aperture"), self._spin_na)
        form.addRow(_form_label("Refractive Index"), self._spin_n2)
        form.addRow(_form_label("Wavelength (λ)"), self._spin_wl)

        self._spin_factor = ValueSpinBox(
            minimum=0.1, maximum=10.0, step=0.01, decimals=3,
            tooltip="Correction factor (ζ) applied to the depth below the surface",
        )
        # Resolved once, so the initial value, the tooltip that advertises it and
        # the button that restores it cannot disagree.
        self._default_factor = _default_factor()
        self._spin_factor.setValue(self._default_factor)
        form.addRow(_form_label("Correction Factor (ζ)"), self._spin_factor)

        self._btn_reset_factor = IconToolButton(
            "mdi:refresh",
            tooltip=f"Reset correction factor to default ({self._default_factor:.3f})",
        )
        self._btn_reset_factor.clicked.connect(
            lambda: self._spin_factor.setValue(self._default_factor)
        )

        lut_available = _LUT_PATH.exists()
        self._lut_available = lut_available
        if not lut_available:
            self._lut_warning = QLabel("LUT not found — calculator disabled. Edit correction factor manually.")
            self._lut_warning.setStyleSheet(
                "color: #f0a500; font-style: italic; font-size: 11px; padding: 4px 8px;"
            )
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

        for spin in (
            self._spin_tilt,
            self._spin_depth,
            self._spin_na,
            self._spin_n2,
            self._spin_wl,
            self._spin_factor,
        ):
            spin.setStyleSheet(_CONTROL_STYLE)

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

        # Track manual edits so metadata seeding never clobbers a typed value
        # (FIB-277). editingFinished fires only on user interaction, not setValue.
        self._spin_na.editingFinished.connect(lambda: self._user_edited.add("na"))
        self._spin_wl.editingFinished.connect(
            lambda: self._user_edited.add("wavelength")
        )

        self.zeta_computed.connect(self._spin_factor.setValue)

        # Relayed rather than exposed: set_factor blocks this spinbox's signals,
        # so mirroring a stored factor stays silent while every user-facing route
        # to it (typing, reset, a parameter edit feeding zeta through) reports.
        self._spin_factor.valueChanged.connect(self.factor_changed)

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

    def set_tilt_locked(self, locked: bool) -> None:
        """Lock the tilt spinbox to 0° (FM-space correction acts along the optical axis).

        The previous tilt value is restored on unlock. Idempotent.

        The programmatic tilt change recomputes zeta quietly (the correction
        factor spinbox is NOT overwritten): a mode switch must not clobber a
        manually entered factor. User edits to the optical parameters still
        update the factor as usual.
        """
        if locked == self._tilt_locked:
            return
        self._tilt_locked = locked
        self._spin_tilt.blockSignals(True)
        if locked:
            self._tilt_before_lock = self._spin_tilt.value()
            self._spin_tilt.setValue(0.0)
            self._spin_tilt.setEnabled(False)
            self._spin_tilt.setToolTip(_TILT_LOCKED_TOOLTIP)
        else:
            self._spin_tilt.setEnabled(self._lut_available)
            self._spin_tilt.setToolTip(
                _TILT_TOOLTIP if self._lut_available else _LUT_MISSING_MSG
            )
            if self._tilt_before_lock is not None:
                self._spin_tilt.setValue(self._tilt_before_lock)
        self._spin_tilt.blockSignals(False)
        self._recompute(update_factor=False)

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

    def seed_from_metadata(
        self,
        wavelength_um: Optional[float] = None,
        na: Optional[float] = None,
    ) -> None:
        """Populate λ and/or NA from FM metadata (FIB-277), leaving user-typed
        values and absent (``None``) inputs untouched."""
        if wavelength_um is not None and "wavelength" not in self._user_edited:
            self._spin_wl.setValue(wavelength_um * 1000.0)  # µm → nm
        if na is not None and "na" not in self._user_edited:
            self._spin_na.setValue(na)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _recompute(self, _value: object = None, *, update_factor: bool = True) -> None:
        # _value swallows the float from QDoubleSpinBox.valueChanged connections.
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
            if update_factor:
                self.zeta_computed.emit(zeta)
        except Exception:
            self._zeta = None
