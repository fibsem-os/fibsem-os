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

import logging
import os
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
from fibsem.ui import stylesheets
from fibsem.ui.fm.widgets.fm_region_selector import FMRegionSelectorWidget
from fibsem.ui.fm.widgets.fm_tile_preview import FMTilePreviewWidget
from fibsem.ui.stylesheets import CANVAS_BG
from fibsem.ui.tokens import (
    BORDER_COLOR,
    PANEL_COLOR,
    STAGE_LIMITS_COLOUR,
    TEXT_COLOR,
    TEXT_MUTED_COLOR,
)
from fibsem.ui.widgets.custom_widgets import ElidedLabel
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

        # The warning gets a row of its own and an `ElidedLabel`; the counter gets
        # neither. A message whose length depends on what the user drew must not decide
        # how wide the dialog is -- the warning ran past 90 characters and took the whole
        # bar with it, the same fault FIB-470 fixed on the FM overview widget, where one
        # long status dragged the minimum width from 1030 px to 1728.
        #
        # Not both, though: `ElidedLabel` has an Ignored size policy, so it asks for no
        # width at all. Applied to the counter as well, the stretch beside it took
        # everything and the counter rendered as nothing. It is bounded anyway -- a
        # region index and two dimensions.
        self._status = QLabel()
        self._status.setStyleSheet(
            f"color:{TEXT_COLOR};font-size:11px;border:none;background:transparent;"
        )
        self._warning = ElidedLabel()
        self._warning.setStyleSheet(
            f"color:{_WARNING};font-size:11px;border:none;background:transparent;"
        )
        self._warning.hide()
        hint = QLabel(_HINT)
        hint.setStyleSheet(
            f"color:{TEXT_MUTED_COLOR};font-size:10px;border:none;background:transparent;"
        )

        bar = QFrame()
        bar.setStyleSheet(
            f"QFrame{{background:{PANEL_COLOR};border:1px solid {BORDER_COLOR};"
            "border-radius:4px;}"
        )
        bar_layout = QVBoxLayout(bar)
        bar_layout.setContentsMargins(10, 7, 10, 7)
        bar_layout.setSpacing(3)
        top_row = QHBoxLayout()
        top_row.addWidget(hint)
        top_row.addStretch(1)
        top_row.addWidget(self._status)
        bar_layout.addLayout(top_row)
        bar_layout.addWidget(self._warning)

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
        self._status.setText("   ·   ".join(parts))

        # Said here rather than left to `raise_if_outside_stage_limits`, which refuses
        # once the run starts. Blocking rather than warning, because the runner will
        # refuse anyway and finding out afterwards costs the whole setup. Masking off the
        # offending tiles is a legitimate fix, so the way out is on screen already.
        self._warning.setVisible(bool(unreachable))
        if unreachable:
            self._warning.setText(
                f"{unreachable} tile{'s' if unreachable != 1 else ''} beyond the "
                "stage's travel — move or shrink the region, or drop those tiles"
            )

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


# Extensions `FibsemImage.load` reads. Listed rather than "try everything", so a scan
# does not attempt to open the parameters JSON and the tile folders beside them.
_IMAGE_SUFFIXES = (".tif", ".tiff")


def beam_overviews_in(
    directory: Optional[str], microscope: FibsemMicroscope
) -> Dict[object, List[FibsemImage]]:
    """Stitched beam overviews saved under *directory*, grouped by the view they belong to.

    From disk rather than from the beam overview tab, deliberately. That tab keeps
    `_PlacedTile` -- pixels, a position and a pixel size -- and not the metadata a
    projection is read from, so it could not answer this without being changed; and going
    through the files means a selection can be made in a session that did not acquire the
    overview, which is the ordinary case after a restart.

    Top level only. A run writes its tiles into a directory and the stitched mosaic
    *beside* it under the same name, so the mosaics are exactly the files here and the
    tiles are exactly the ones this does not descend into.

    Fluorescence overviews live in the same experiment folder and are silently skipped:
    what is kept is what `BeamStageProjection.from_image` can read, which is the same
    question as whether a region drawn on it could be resolved at all.

    Returns:
        Views to their images, newest first, so a caller offering a default gets the
        overview most recently acquired.
    """
    from fibsem.projection import BeamStageProjection
    from fibsem.ui.widgets.overview_widget import OverviewView

    if not directory or not os.path.isdir(directory):
        return {}

    paths = [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.lower().endswith(_IMAGE_SUFFIXES)
    ]
    paths.sort(key=os.path.getmtime, reverse=True)

    views: Dict[object, List[FibsemImage]] = {}
    for path in paths:
        try:
            image = FibsemImage.load(path)
        except Exception as e:
            logging.debug(f"Not an overview this can use: {path} ({e})")
            continue
        if BeamStageProjection.from_image(image) is None:
            continue
        position = getattr(
            getattr(getattr(image, "metadata", None), "microscope_state", None),
            "stage_position",
            None,
        )
        if position is None:
            continue
        try:
            view = OverviewView(
                image.metadata.image_settings.beam_type,
                microscope.get_stage_orientation(stage_position=position),
            )
        except Exception as e:
            logging.debug(f"Could not tell which view {path} belongs to: {e}")
            continue
        views.setdefault(view, []).append(image)
    return views
