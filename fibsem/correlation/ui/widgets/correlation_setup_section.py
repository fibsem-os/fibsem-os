"""Lamella setup section for the correlation window's Images tab (FIB-302).

The correlation widget is generic and standalone-launchable; a *lamella* brings
extra context it can be set up from — its spot burns and its previous correlation
runs. This section carries exactly that, and is installed **only** when the
autolamella protocol editor opens the window
(:meth:`CorrelationTabWidget.add_lamella_setup`), so the generic widget never
learns about lamellae.

It lives in the Images tab rather than a tab of its own: choosing *which image*
and choosing *what to start from* are one decision, and the images already live
there. Changing a control drives the **live canvas** through the widget's normal
seeding API — the canvas is the preview, at full fidelity.

Deliberately not here, because the widget already provides it:

* browse/load an arbitrary image  → the Images tab's path fields
* fit / refractive-index settings → their own tabs (summarised read-only here)
* interpolation                   → the Images tab's *Interpolate…* action
"""
from __future__ import annotations

import datetime
from typing import List, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from fibsem import constants
from fibsem.correlation.config import CorrelationConfig
from fibsem.correlation.history import CorrelationRun, LamellaCorrelation
from fibsem.structures import Point
from fibsem.ui.widgets.custom_widgets import TitledPanel, ValueComboBox

# Starting-coordinates sources (mutually exclusive; see the design doc).
SEED_NONE = "none"
SEED_SPOT_BURNS = "spot_burns"
SEED_PREVIOUS = "previous"

_MUTED = "#9aa0a6"


def format_run_timestamp(name: str) -> str:
    """Reformat a run folder name (a ``DATETIME_FILE`` stamp) for display.

    Falls back to the raw name for a folder that doesn't parse (e.g. a legacy or
    hand-named directory), so an odd name still shows rather than disappearing.
    """
    try:
        dt = datetime.datetime.strptime(name, constants.DATETIME_FILE)
    except ValueError:
        return name
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _caption(text: str, indent: int = 0) -> QLabel:
    lbl = QLabel(text)
    style = f"color:{_MUTED};font-size:11px;"
    if indent:
        style += f"margin-left:{indent}px;"
    lbl.setStyleSheet(style)
    lbl.setWordWrap(True)
    return lbl


class CorrelationSetupSection(QWidget):
    """Starting-coordinates chooser + a read-only summary of the inherited config.

    Emits rather than acts, so the widget keeps ownership of the canvas:

    ``seed_requested(source, payload)``
        ``payload`` is a ``CorrelationInputData`` (previous run), a
        ``list[Point]`` (spot burns, normalised 0-1), or ``None``.
    """

    seed_requested = pyqtSignal(str, object)

    def __init__(
        self,
        *,
        spot_burns: Optional[List[Point]] = None,
        history: Optional[LamellaCorrelation] = None,
        config: Optional[CorrelationConfig] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._config = config or CorrelationConfig()
        self._spot_burns: List[Point] = list(spot_burns or [])
        # Newest first, so index i in the dropdown is self._prev_runs[i].
        runs = list(history.runs) if history is not None else []
        self._prev_runs: List[CorrelationRun] = list(reversed(runs))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(
            TitledPanel(
                "Starting Coordinates",
                content=self._build_seed_body(),
                collapsible=False,
            )
        )
        layout.addWidget(
            TitledPanel(
                "Inherited Settings",
                content=self._build_inherited_body(),
                collapsible=False,
            )
        )

        self.rb_burns.setEnabled(bool(self._spot_burns))
        self.rb_prev.setEnabled(bool(self._prev_runs))
        # Default: previous run if one exists, else spot burns, else nothing —
        # the same precedence the editor applied implicitly before this section.
        if self._prev_runs:
            self.rb_prev.setChecked(True)
        elif self._spot_burns:
            self.rb_burns.setChecked(True)
        else:
            self.rb_none.setChecked(True)

        # Connect last, so setting the defaults above doesn't emit a seed request
        # before the caller has connected — the initial seed is explicit, via
        # emit_current_seed().
        for rb in (self.rb_none, self.rb_burns, self.rb_prev):
            rb.toggled.connect(self._on_source_toggled)
        self.run_combo.currentIndexChanged.connect(self._on_run_changed)
        self._apply_enabled_state()

    # ---- construction -------------------------------------------------------

    def _build_seed_body(self) -> QWidget:
        body = QWidget()
        col = QVBoxLayout(body)
        col.setContentsMargins(8, 4, 8, 4)
        col.setSpacing(4)

        self._group = QButtonGroup(self)
        self.rb_none = QRadioButton("None — pick everything fresh")
        burn_label = f"Spot-burn fiducials · {len(self._spot_burns)} found"
        if not self._prev_runs:
            burn_label += "  (first)"
        self.rb_burns = QRadioButton(burn_label)
        self.rb_prev = QRadioButton("Previous correlation")
        for rb in (self.rb_none, self.rb_burns, self.rb_prev):
            self._group.addButton(rb)
            col.addWidget(rb)

        run_row = QWidget()
        run_layout = QHBoxLayout(run_row)
        run_layout.setContentsMargins(20, 0, 0, 0)
        self.run_combo = ValueComboBox(
            [format_run_timestamp(r.name) for r in self._prev_runs]
        )
        run_layout.addWidget(self.run_combo, 1)
        col.addWidget(run_row)

        self._prev_caption = _caption(
            "Carries the FM POI + fiducials from that run forward.", indent=20
        )
        col.addWidget(self._prev_caption)
        col.addWidget(_caption("Seeded points are placed as-is — refine them on the canvas."))
        return body

    def _build_inherited_body(self) -> QWidget:
        fit, ri = self._config.fit, self._config.ri
        body = QWidget()
        col = QVBoxLayout(body)
        col.setContentsMargins(8, 4, 8, 4)
        col.setSpacing(2)
        col.addWidget(
            _caption(
                f"Fit — FIB {fit.fib_method} · FM POI {fit.fm_poi_method} · "
                f"POI channel {fit.fm_poi_channel or '—'}"
            )
        )
        col.addWidget(
            _caption(
                f"RI — n₂ {ri.n2:.2f} · NA {ri.na:.2f} · "
                f"λ {ri.wavelength_um * 1000:.0f} nm  (from FM metadata)"
            )
        )
        col.addWidget(_caption("Experiment defaults; edit them on their own tabs."))
        return body

    # ---- state --------------------------------------------------------------

    def seed_source(self) -> str:
        if self.rb_burns.isChecked():
            return SEED_SPOT_BURNS
        if self.rb_prev.isChecked():
            return SEED_PREVIOUS
        return SEED_NONE

    def selected_run(self) -> Optional[CorrelationRun]:
        idx = self.run_combo.currentIndex()
        if 0 <= idx < len(self._prev_runs):
            return self._prev_runs[idx]
        return self._prev_runs[0] if self._prev_runs else None

    def seed_payload(self):
        """What the current selection would seed: input data, points, or None."""
        source = self.seed_source()
        if source == SEED_SPOT_BURNS:
            return list(self._spot_burns)
        if source == SEED_PREVIOUS:
            run = self.selected_run()
            return run.state.input_data if run is not None else None
        return None

    def emit_current_seed(self) -> None:
        """Apply the current selection — also the initial seed on open."""
        self.seed_requested.emit(self.seed_source(), self.seed_payload())

    # ---- interaction --------------------------------------------------------

    def _apply_enabled_state(self) -> None:
        is_prev = self.rb_prev.isChecked()
        self.run_combo.setEnabled(is_prev)
        self._prev_caption.setVisible(is_prev)

    def _on_source_toggled(self, checked: bool) -> None:
        self._apply_enabled_state()
        if checked:  # only the newly-selected radio, not the one being cleared
            self.emit_current_seed()

    def _on_run_changed(self, _index: int) -> None:
        if self.rb_prev.isChecked():
            self.emit_current_seed()
