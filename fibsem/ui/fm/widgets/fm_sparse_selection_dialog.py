"""Choose the ground a fluorescence overview should cover, on a beam overview.

The two panes joined: regions are drawn on an acquired FIB/SEM overview on the left, and
the fluorescence tiles they select appear on the right, recomputed on every change.
Accepting hands back overview parameters with the grid and mask filled in, plus the stage
position the run should be centred on.

**No host.** It takes what it needs and returns a result; it does not reach into the tab
that opened it, and it cannot drive the stage. That is what lets it be constructed in a
test, and it follows `FMOverviewConfirmationDialog`, which computes its own estimate from
values handed to it for the same reason.

**The regions end here.** They exist to draw with, and to say which tile came from a
region rather than from a click -- both only while the selection is being made. Storing
them beside the mask would be a second representation of the same thing, able to disagree
with it. Selecting again starts over.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

from PyQt5.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.fm.structures import ChannelSettings, OverviewParameters, ZParameters
from fibsem.fm.timing import estimate_tileset_acquisition_time
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import FibsemImage, FibsemStagePosition
from fibsem.ui.fm.widgets.fm_region_selector import FMRegionSelectorWidget
from fibsem.ui.fm.widgets.fm_tile_preview import FMTilePreviewWidget
from fibsem.ui import stylesheets
from fibsem.ui.stylesheets import CANVAS_BG
from fibsem.ui.tokens import (
    BORDER_COLOR,
    PANEL_COLOR,
    STAGE_LIMITS_COLOUR,
    TEXT_COLOR,
    TEXT_MUTED_COLOR,
)
from fibsem.ui.widgets.preflight import format_duration

# Amber, matching the stage-limits box the preview draws it against.
_WARNING = STAGE_LIMITS_COLOUR

_HINT = (
    "right-click the overview to add a region  ·  drag it or its corners  ·  "
    "click a tile on the right to add or drop it"
)


@dataclass(frozen=True)
class SparseSelection:
    """What a completed selection hands back.

    Attributes:
        parameters: The parameters given, with rows, columns and the tile mask replaced
            by what the selection derived. Everything else -- overlap, autofocus, tile
            order -- is carried through untouched, because none of it is geometry.
        centre_position: Where the grid is centred, at the fluorescence orientation.
            `FMTiledAcquisitionRunner` takes this directly.
    """

    parameters: OverviewParameters
    centre_position: FibsemStagePosition


class FMSparseSelectionDialog(QDialog):
    """Draw regions on a beam overview; get an FM grid back."""

    def __init__(
        self,
        microscope: FibsemMicroscope,
        views: Dict[object, Sequence[FibsemImage]],
        parameters: OverviewParameters,
        channel_settings: Optional[List[ChannelSettings]] = None,
        zparams: Optional[ZParameters] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        """
        Args:
            microscope: for the fluorescence geometry the tiles are planned against.
            views: the beam overviews to choose from, keyed by `OverviewView`.
            parameters: the overview settings as they stand. Only `overlap` affects the
                grid; the rest is carried through so the caller gets one object back.
            channel_settings: for the time estimate. Without them the counter reports
                tiles and no duration, rather than a duration it cannot stand behind.
            zparams: likewise, when the run is a z-stack.
        """
        super().__init__(parent)
        self.setWindowTitle("Select the ground to image")
        self.setStyleSheet(f"QDialog {{ background: {CANVAS_BG}; }}")

        self._parameters = parameters
        self._channel_settings = channel_settings
        self._zparams = zparams if parameters.use_zstack else None
        self._selection: Optional[SparseSelection] = None

        self.selector = FMRegionSelectorWidget(microscope)
        self.preview = FMTilePreviewWidget(microscope)
        self.selector.set_views(views)

        self._status = QLabel()
        self._status.setStyleSheet(
            f"color:{TEXT_COLOR};font-size:11px;border:none;background:transparent;"
        )
        hint = QLabel(_HINT)
        hint.setStyleSheet(
            f"color:{TEXT_MUTED_COLOR};font-size:10px;border:none;background:transparent;"
        )

        bar = QFrame()
        bar.setStyleSheet(
            f"QFrame{{background:{PANEL_COLOR};border:1px solid {BORDER_COLOR};"
            "border-radius:4px;}"
        )
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(10, 7, 10, 7)
        bar_layout.addWidget(hint)
        bar_layout.addStretch(1)
        bar_layout.addWidget(self._status)

        # Hand-built rather than a `QDialogButtonBox`: that renders native buttons,
        # which read as light chrome against this app's dark canvases, and puts them in
        # the platform's order rather than the one the rest of the app uses.
        self.accept_button = QPushButton()
        self.accept_button.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.accept_button.clicked.connect(self._accept)
        cancel = QPushButton("Cancel")
        cancel.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        cancel.clicked.connect(self.reject)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(cancel)
        buttons.addWidget(self.accept_button)

        panes = QHBoxLayout()
        panes.setSpacing(8)
        panes.addWidget(self.selector, 3)
        panes.addWidget(self.preview, 2)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addLayout(panes)
        layout.addWidget(bar)
        layout.addLayout(buttons)
        self.resize(1180, 700)

        self.selector.regions_changed.connect(self._recompute)
        self.selector.selection_changed.connect(lambda _index: self._describe())
        self.preview.selection_changed.connect(self._describe)
        self._describe()

    # ── the result ───────────────────────────────────────────────────────

    @property
    def selection(self) -> Optional[SparseSelection]:
        """What was accepted, or None if the dialog was cancelled."""
        return self._selection

    @classmethod
    def choose(
        cls,
        microscope: FibsemMicroscope,
        views: Dict[object, Sequence[FibsemImage]],
        parameters: OverviewParameters,
        channel_settings: Optional[List[ChannelSettings]] = None,
        zparams: Optional[ZParameters] = None,
        parent: Optional[QWidget] = None,
    ) -> Optional[SparseSelection]:
        """Run the dialog and return the selection, or None if it was cancelled."""
        dialog = cls(
            microscope, views, parameters, channel_settings, zparams, parent
        )
        dialog.exec_()
        return dialog.selection

    def _accept(self) -> None:
        plan = self.preview.plan
        if plan is None:
            return
        self._selection = SparseSelection(
            parameters=replace(
                self._parameters,
                rows=plan.rows,
                cols=plan.cols,
                tile_mask=self.preview.mask,
            ),
            centre_position=plan.centre_position,
        )
        self.accept()

    # ── keeping the two panes in step ────────────────────────────────────

    def _recompute(self) -> None:
        regions = self.selector.regions
        if not regions or self.selector.base is None or self.selector.projection is None:
            self.preview.clear()
        else:
            self.preview.set_selection(
                regions,
                self.selector.base,
                self.selector.projection,
                self._parameters.overlap,
            )
        self._describe()

    def _describe(self) -> None:
        """The counter and the button, from one set of numbers.

        Both together, deliberately: a count computed in one place and restated in the
        other is how a button comes to offer a different number from the line above it.
        """
        enabled, total, duration = self._counts()
        unreachable = len(self.preview.unreachable_tiles)
        self.accept_button.setEnabled(enabled > 0 and not unreachable)
        self.accept_button.setText(
            f"Use {enabled} tiles" if enabled else "Use these tiles"
        )

        parts = [self.selector.describe_selected()]
        if total:
            parts.append(f"{enabled} of {total} tiles")
        if duration is not None and not unreachable:
            parts.append(format_duration(duration))
        if unreachable:
            # Said here rather than left to `raise_if_outside_stage_limits`, which
            # refuses once the run starts. Blocking rather than warning, because the
            # runner will refuse anyway and finding out afterwards costs the whole setup.
            # Masking off the offending tiles is a legitimate fix, so the way out is on
            # screen already.
            parts.append(
                f"<span style='color:{_WARNING}'>{unreachable} beyond the stage's "
                "travel — move or shrink the region, or drop those tiles</span>"
            )
        self._status.setText("   ·   ".join(parts))

    def _counts(self) -> Tuple[int, int, Optional[float]]:
        """Enabled tiles, total tiles, and how long the run would take.

        The duration is None without channel settings rather than guessed at: the time
        depends on exposure and channel count, and a number invented here would be quoted
        back as if it meant something.
        """
        plan = self.preview.plan
        enabled, total = self.preview.enabled_tiles, self.preview.total_tiles
        if plan is None or not enabled or not self._channel_settings:
            return enabled, total, None
        try:
            estimate = estimate_tileset_acquisition_time(
                self._channel_settings,
                (plan.rows, plan.cols),
                self._zparams,
                self._parameters.autofocus_mode,
                tile_mask=self.preview.mask,
            )
        except Exception:
            return enabled, total, None
        return enabled, total, estimate.get("total_time")

    # ── convenience for a host ───────────────────────────────────────────

    @property
    def mask(self) -> List[List[bool]]:
        """The mask as it stands, for a caller watching rather than waiting."""
        return self.preview.mask
