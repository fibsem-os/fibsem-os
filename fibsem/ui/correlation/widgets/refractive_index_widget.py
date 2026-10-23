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

from fibsem.correlation.config import RISettings
from fibsem.correlation.refractive_index import (
    _LUT_PATH,
    ZetaParams,
    lookup_zeta,
    zeta_params_from_settings,
)
from fibsem.ui.tokens import (
    CONTROL_STYLE,
    state_style,
)
from fibsem.ui.widgets.custom_widgets import (
    IconToolButton,
    TitledPanel,
    ValueSpinBox,
    form_label,
)
from fibsem.ui.widgets.form_builder import configure_spinbox

_LUT_MISSING_MSG = (
    f"Correction factor calculator unavailable: LUT file not found at\n{_LUT_PATH}\n"
    "The correction factor can still be edited manually."
)

# The one default set lives on RISettings, where the protocol reads it too; see
# its docstring for where the values come from.
_DEFAULTS = zeta_params_from_settings(RISettings())

# Only used when the lookup table cannot be read. Must equal the LUT's value at
# _DEFAULTS — pinned by tests/correlation/test_default_correction_factor.py, so
# change the two together or that test fails.
_FALLBACK_FACTOR = 1.502


def _default_factor() -> float:
    """The correction factor for `_DEFAULTS`, from the lookup table.

    Derived rather than declared. A literal constant beside a lookup table is a
    second source of truth for the same number, and it drifts silently the moment
    either side moves: the previous constant (1.470) described no configuration
    the widget could be in, while its own defaults asked the table for 1.595.
    Deriving means the reset button can only ever hand back the factor
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


# The correlation panels run at 11-12px. Controls default to the app font, which
# renders larger than the values beside them (form-row labels: see form_label).
_CONTROL_STYLE = CONTROL_STYLE


# Each optical parameter's box is shown as RISettings declares it.
_RI = RISettings.params
_TILT_TOOLTIP = _RI["tilt_deg"].doc
_WAVELENGTH_SCALE = _RI["wavelength_um"].scale  # shown in nm, stored in µm


def _ri_spinbox(name: str) -> ValueSpinBox:
    spinbox = ValueSpinBox(no_buttons=True)
    configure_spinbox(spinbox, _RI[name])
    return spinbox


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
        no longer what has been applied.
    """

    zeta_computed = pyqtSignal(float)
    factor_changed = pyqtSignal(float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._zeta: Optional[float] = None
        self._tilt_locked = False
        self._tilt_before_lock: Optional[float] = None
        # Whether the factor on screen is ζ of the parameters shown, rather than
        # typed, reset, or restored from a stored run (set_factor). A programmatic
        # recompute -- the tilt lock -- must not write over an entered factor; a
        # parameter edit does.
        self._factor_computed = False
        # Params the user has typed by hand; metadata seeding leaves these alone.
        self._user_edited: set = set()
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

        self._spin_tilt = _ri_spinbox("tilt_deg")
        self._spin_depth = _ri_spinbox("depth_um")
        self._spin_na = _ri_spinbox("na")
        self._spin_n2 = _ri_spinbox("n2")
        self._spin_wl = _ri_spinbox("wavelength_um")

        for name, spin in (
            ("tilt_deg", self._spin_tilt),
            ("depth_um", self._spin_depth),
            ("na", self._spin_na),
            ("n2", self._spin_n2),
            ("wavelength_um", self._spin_wl),
        ):
            form.addRow(form_label(_RI[name].label), spin)

        self._spin_factor = ValueSpinBox(
            minimum=0.1,
            maximum=10.0,
            step=0.01,
            decimals=3,
            no_buttons=True,
            tooltip="Correction factor (ζ) applied to the depth below the surface",
        )
        # Resolved once, so the initial value, the tooltip that advertises it and
        # the button that restores it cannot disagree.
        self._default_factor = _default_factor()
        self._spin_factor.setValue(self._default_factor)
        form.addRow(form_label("Correction Factor (ζ)"), self._spin_factor)

        self._btn_reset_factor = IconToolButton(
            "mdi:refresh",
            tooltip=f"Reset correction factor to default ({self._default_factor:.3f})",
        )
        self._btn_reset_factor.clicked.connect(self._reset_factor)

        lut_available = _LUT_PATH.exists()
        self._lut_available = lut_available
        if not lut_available:
            self._lut_warning = QLabel(
                "LUT not found — calculator disabled. Edit correction factor manually."
            )
            self._lut_warning.setStyleSheet(
                f"{state_style('warn', 11)} font-style: italic; padding: 4px 8px;"
            )
            self._lut_warning.setWordWrap(True)
            self._lut_warning.setToolTip(_LUT_MISSING_MSG)
            form.addRow(self._lut_warning)

            for spin in (
                self._spin_tilt,
                self._spin_depth,
                self._spin_na,
                self._spin_n2,
                self._spin_wl,
            ):
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

        # Track manual edits so metadata seeding never clobbers a typed value.
        # editingFinished fires only on user interaction, not setValue.
        self._spin_na.editingFinished.connect(lambda: self._user_edited.add("na"))
        self._spin_wl.editingFinished.connect(
            lambda: self._user_edited.add("wavelength")
        )

        self.zeta_computed.connect(self._spin_factor.setValue)
        # editingFinished fires for typing only, not for setValue.
        self._spin_factor.editingFinished.connect(self._on_factor_typed)

        # Relayed rather than exposed: set_factor blocks this spinbox's signals,
        # so mirroring a stored factor stays silent while every user-facing route
        # to it (typing, reset, a parameter edit feeding zeta through) reports.
        self._spin_factor.valueChanged.connect(self.factor_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_factor(self) -> float:
        """Return the current correction factor spinbox value."""
        return self._spin_factor.value()

    def set_factor(self, v: float) -> None:
        """Set the correction factor spinbox without triggering recompute."""
        self._spin_factor.blockSignals(True)
        self._spin_factor.setValue(v)
        self._spin_factor.blockSignals(False)
        self._factor_computed = False

    def set_tilt_locked(self, locked: bool) -> None:
        """Lock the tilt spinbox to 0° (FM-space correction acts along the optical axis).

        The previous tilt value is restored on unlock. Idempotent.

        The factor follows the tilt now shown, unless it was typed by hand: a mode
        switch must not clobber a manually entered factor.
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
        self._recompute(update_factor=self._factor_computed)

    def set_params(self, params: ZetaParams) -> None:
        """Populate all spinboxes from *params*, then recompute ζ from them.

        The values are set with signals blocked so ζ is computed once, from the
        final set, and not once per box. It used not to be computed at all, so a
        protocol's parameters were shown while the factor stayed at the widget's
        own defaults: 1.502 under boxes describing 1.567 on a real run.
        """
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
        self._spin_wl.setValue(params.wavelength_um * _WAVELENGTH_SCALE)

        for spin in (
            self._spin_tilt,
            self._spin_depth,
            self._spin_na,
            self._spin_n2,
            self._spin_wl,
        ):
            spin.blockSignals(False)
        self._recompute()

    def get_params(self) -> ZetaParams:
        """Return the current spinbox values as a :class:`ZetaParams`."""
        return ZetaParams(
            tilt_deg=self._spin_tilt.value(),
            depth_um=self._spin_depth.value(),
            NA=self._spin_na.value(),
            n2=self._spin_n2.value(),
            wavelength_um=self._spin_wl.value() / _WAVELENGTH_SCALE,
        )

    def seed_from_metadata(
        self,
        wavelength_um: Optional[float] = None,
        na: Optional[float] = None,
    ) -> None:
        """Populate λ and/or NA from FM metadata, leaving user-typed
        values and absent (``None``) inputs untouched."""
        if wavelength_um is not None and "wavelength" not in self._user_edited:
            self._spin_wl.setValue(wavelength_um * _WAVELENGTH_SCALE)
        if na is not None and "na" not in self._user_edited:
            self._spin_na.setValue(na)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def factor_is_computed(self) -> bool:
        """Whether the factor on screen is ζ of the parameters shown."""
        return self._factor_computed

    def _on_factor_typed(self) -> None:
        self._factor_computed = False

    def _reset_factor(self) -> None:
        self._spin_factor.setValue(self._default_factor)
        self._factor_computed = False

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
                self._factor_computed = True
        except Exception:
            self._zeta = None
