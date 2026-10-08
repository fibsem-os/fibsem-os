"""The quad view's overview page: the newest overview of a grid, its lamellae marked.

A page of the Microscope tab's fourth cell, beside the chamber drawing. It shows
the overviews the experiment already holds and reads nothing live: the stage
position it marks is the one the quad view's info bar is given.

**Which grid.** By default the grid the stage is on (the loaded grid in the holder
slot under the stage, from the cached position), following it from slot to slot
and keeping the last one while the stage is at none. The selector pins a grid
instead. An experiment with no grid records has no selector, and shows its own
overviews.

**Which overviews.** A grid's are the ones its task history recorded. Overviews the
Overview tabs saved into the experiment's root carry no grid, so they are placed
here, best effort, by where they were taken: their stage position, the holder slot
it falls in, the grid loaded there, the record of that name. An overview that can't
be placed that way is left out of every grid rather than guessed into one
(FIB-1195 is to record the link when the overview is saved).

**One image per view.** The canvas shows one view at a time (a beam at a pose, or
the fluorescence microscope), switched with the chips. Within a view, only the
newest run is shown; older runs are on the Grids tab.

**The grid's rim.** Drawn around the slot the shown grid is loaded in, as the
Overview tab draws its grid boundaries: where that slot is *now*, so a grid that is
not loaded gets none, and a rim that misses the imaged grid says the slot's
calibration is off.

**Cost.** Indexing reads each file's metadata, not its pixels: a root overview has to
be read to be placed, and an experiment can hold many. Pixels are read only for the
images on show. Nothing is done while the page is hidden -- behind the chamber page,
or on another tab -- beyond noting that it is out of date; it catches up when shown.
A stage update that stays on the same grid moves the stage marker and nothing else.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import tifffile as tff
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskStatus,
    Experiment,
    GridRecord,
    Lamella,
)
from fibsem.applications.autolamella.ui.grid_results_widget import image_for
from fibsem.applications.autolamella.workflows.tasks.grid.manager import (
    LOAD_ENTRY_NAME,
)
from fibsem.constants import TIME_DISPLAY_AMPM_SHORT
from fibsem.fm.structures import FluorescenceImage
from fibsem.projection import BeamStageProjection
from fibsem.structures import FibsemImage, FibsemImageMetadata, FibsemStagePosition
from fibsem.ui.tokens import CANVAS_BG
from fibsem.ui.widgets.canvas.overlays.stage_context import (
    at_device,
    holder_slots,
    slot_landmark,
)
from fibsem.ui.widgets.canvas.quad_view import CELL_SELECTOR_STYLE
from fibsem.ui.widgets.overview_widget import VIEW_CHIP_SPACING, VIEW_CHIP_STYLE
from fibsem.ui.widgets.stored_overview_canvas import (
    VIEW_FM,
    StoredOverviewCanvas,
    _view_of_beam_image,
)
from fibsem.util.timestamps import format_time, from_posix

logger = logging.getLogger(__name__)

CURRENT = "current"
# The key of the overviews no grid could be found for, in an experiment with grids.
NO_GRID = ""

_EMPTY_STYLE = "color: #777; font-size: 12px;"
_CAPTION_STYLE = (
    "color: #cfcfcf; font-size: 11px; padding: 2px 6px;"
    " background: rgba(20, 20, 20, 0.7); border-radius: 3px;"
)
# Images kept read between switches: a grid's views, and the last grid's.
_IMAGE_CACHE = 8
# The order chips are offered in, and the view picked when the shown one is missing.
_VIEW_ORDER = ("SEM", "FIB", VIEW_FM)


@dataclass(frozen=True)
class _Overview:
    """One overview file, as far as its metadata says."""

    path: str
    view: str
    when: datetime
    label: str


def _is_fluorescence(path: str) -> bool:
    return path.endswith((".ome.tiff", ".ome.tif"))


def _read_metadata(path: str) -> Optional[FibsemImageMetadata]:
    """A beam image's metadata, without reading its pixels."""
    try:
        with tff.TiffFile(path) as tiff:
            raw = tiff.pages[0].tags["ImageDescription"].value
        return FibsemImageMetadata.from_dict(json.loads(raw))
    except Exception:
        logger.debug(f"Could not read the metadata of {path}", exc_info=True)
        return None


def _beam_view(metadata: FibsemImageMetadata) -> Optional[str]:
    """The canvas's view name for a beam image: the beam at its pose."""
    position = getattr(metadata, "stage_position", None)
    if position is None:
        return None
    return _view_of_beam_image(SimpleNamespace(metadata=metadata), position)


def _view_group(view: str) -> str:
    """ "SEM @ r=0° t=35°" -> "SEM": what a chip is labelled, and ordered, by."""
    return view.split(" @ ")[0]


def _load_image(path: str):
    return (
        FluorescenceImage.load(path)
        if _is_fluorescence(path)
        else FibsemImage.load(path)
    )


def grid_of_root_overview(
    microscope, experiment: Experiment, metadata: FibsemImageMetadata
) -> Optional[GridRecord]:
    """The grid record an overview saved outside any grid was taken on, or None.

    Best effort, and never a guess: its stage position, the holder slot that falls
    in, the grid loaded there, the record of that name. A position recorded half a
    turn from the holder's frame (taken at the FIB orientation on a stage that turns
    to reach it) is flipped back first, as the overview markers are, so it looks up
    the slot it was actually on.
    """
    if microscope is None or experiment is None:
        return None
    position = getattr(metadata, "stage_position", None)
    stage = getattr(microscope, "_stage", None)
    if position is None or stage is None:
        return None
    try:
        sem = microscope.get_orientation("SEM")
        reference = FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=sem.r, t=sem.t)
        centre = microscope.hardware_geometry().rotation_centre
        place = BeamStageProjection._compucentric_corrected(position, reference, centre)
        grid = stage.grid_at_position(place)
    except Exception:
        logger.debug("Could not place a root overview on a grid", exc_info=True)
        return None
    if grid is None:
        return None
    return next((g for g in experiment.grids if g.name == grid.name), None)


class QuadOverviewPage(QWidget):
    """The newest overviews of one grid, with its lamellae and the stage marked.

    Signals:
        lamella_selected(Lamella): a lamella was clicked on the overview.
    """

    lamella_selected = pyqtSignal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._experiment: Optional[Experiment] = None
        self._microscope = None
        self._stage: Optional[FibsemStagePosition] = None
        self._mode = CURRENT
        # The grid the stage was last on, kept while it is on none.
        self._last_grid: Optional[str] = None
        # The grid under the stage, worked out once per stage update.
        self._under: Optional[str] = None
        # Something changed while the page was hidden; it catches up when shown.
        self._stale = False
        # Grid key -> its overviews, newest last. Grid keys are record ids, or
        # NO_GRID; an experiment without grids keeps everything under NO_GRID.
        self._index: Dict[str, List[_Overview]] = {}
        # Metadata reads, by (path, mtime): an experiment update re-lists files often.
        self._seen: Dict[Tuple[str, float], Optional[Tuple[Optional[str], str]]] = {}
        self._images: Dict[str, object] = {}
        self._shown: Optional[Tuple[str, Tuple[str, ...]]] = None
        self._view_by_group: Dict[str, str] = {}
        self._selected: Optional[str] = None

        self.grid_selector = QComboBox()
        self.grid_selector.setStyleSheet(CELL_SELECTOR_STYLE)
        # "Current grid (none; showing Grid-A)" is longer than a grid's own name.
        self.grid_selector.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.grid_selector.activated.connect(self._on_grid_chosen)
        self._chips: Dict[str, QPushButton] = {}
        self._chips_layout = QHBoxLayout()
        self._chips_layout.setSpacing(VIEW_CHIP_SPACING)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self.grid_selector)
        header.addStretch(1)
        header.addLayout(self._chips_layout)

        self.canvas = StoredOverviewCanvas()
        self.canvas.set_placing_enabled(False)
        # A glance at where things are, not a place to measure them: the Overview
        # tab has the ruler.
        self.canvas.canvas.btn_toggle_ruler.hide()
        self.canvas.position_selected.connect(self._on_position_selected)
        self.caption = QLabel(self.canvas)
        self.caption.setStyleSheet(_CAPTION_STYLE)
        self.caption.hide()

        self.empty = QLabel(alignment=Qt.AlignCenter)
        self.empty.setStyleSheet(_EMPTY_STYLE)
        self.empty.setWordWrap(True)

        self._stack = QStackedWidget()
        self._stack.addWidget(self.empty)
        self._stack.addWidget(self.canvas)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._stack, 1)
        # The page's controls, for the cell to show in its own header row beside
        # the page selector (`PageCell.add_page(..., header=page.header)`) rather
        # than as a second row here.
        self.header = QWidget()
        self.header.setStyleSheet(f"background: {CANVAS_BG};")
        self.header.setLayout(header)
        self._show_empty("No experiment loaded.")

    # ── inputs ────────────────────────────────────────────────────────────
    @property
    def experiment(self) -> Optional[Experiment]:
        return self._experiment

    def set_microscope(self, microscope) -> None:
        self._microscope = microscope
        self._seen.clear()
        self._reindex(force=True)

    def set_experiment(self, experiment: Optional[Experiment]) -> None:
        if experiment is not self._experiment:
            self._experiment = experiment
            self._mode = CURRENT
            self._last_grid = None
            self._seen.clear()
            self._images.clear()
            self._shown = None
        self._reindex(force=True)

    def set_stage(self, position: Optional[FibsemStagePosition]) -> None:
        """The stage moved: follow it to another grid, and move its marker."""
        self._stage = position
        under = self._grid_under_stage()
        same_grid = under == self._under
        self._under = under
        if under is not None:
            self._last_grid = under
        if not self.isVisible():
            self._stale = True
        elif same_grid and not self._stale:
            self._mark_stage()
        else:
            self._update()

    def set_selected(self, lamella: Optional[Lamella]) -> None:
        self._selected = lamella.name if lamella is not None else None
        self.canvas.set_selected_position(self._selected)

    def refresh(self) -> None:
        """The experiment's overviews may have changed: one was acquired, or a task
        recorded one. Cheap enough to call on any hint -- a listing and a stat per
        file, the metadata already read -- and redraws only when something did."""
        self._reindex(force=False)

    def _reindex(self, force: bool) -> None:
        index = self._build_index()
        changed = index != self._index
        self._index = index
        self._under = self._grid_under_stage()
        if force or changed:
            self._update()

    def refresh_positions(self) -> None:
        """The lamellae changed; the overviews did not."""
        if self.isVisible():
            self._mark_lamellae()
        else:
            self._stale = True

    def _update(self) -> None:
        """Bring the selector and the canvas up to date, or note that they are out of
        date while nobody can see them."""
        if not self.isVisible():
            self._stale = True
            return
        self._stale = False
        self._fill_selector()
        self._show()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().showEvent(event)
        # Back from the Overview tab, say, with a new overview saved meanwhile.
        self._reindex(force=self._stale)

    # ── what to show ──────────────────────────────────────────────────────
    def _has_grids(self) -> bool:
        return self._experiment is not None and bool(self._experiment.grids)

    def _grid_under_stage(self) -> Optional[str]:
        if not self._has_grids() or self._stage is None:
            return None
        try:
            grid = self._microscope._stage.grid_at_position(self._stage)
        except Exception:
            return None
        if grid is None:
            return None
        record = next((g for g in self._experiment.grids if g.name == grid.name), None)
        return record.id if record is not None else None

    def shown_grid(self) -> Optional[str]:
        """The key of the grid on show: a record id, NO_GRID, or None for nothing."""
        if self._experiment is None:
            return None
        if not self._has_grids():
            return NO_GRID
        if self._mode != CURRENT:
            return self._mode
        return self._under or self._last_grid

    def _grid_name(self, key: Optional[str]) -> str:
        if key == NO_GRID:
            return "Not on a grid"
        record = self._experiment.get_grid_by_id(key) if key else None
        return record.name if record is not None else "none"

    # ── index ─────────────────────────────────────────────────────────────
    def _build_index(self) -> Dict[str, List[_Overview]]:
        experiment = self._experiment
        if experiment is None:
            return {}
        index: Dict[str, List[_Overview]] = {}
        has_grids = self._has_grids()
        for grid in experiment.grids:
            for state in grid.task_history:
                if (
                    state.name == LOAD_ENTRY_NAME
                    or state.status is not AutoLamellaTaskStatus.Completed
                ):
                    continue
                path = image_for(experiment, grid, state)
                if path is None:
                    continue
                read = self._read(path)
                if read is None:
                    continue
                stamp = state.end_timestamp or state.start_timestamp or from_posix(0.0)
                index.setdefault(grid.id, []).append(
                    _Overview(path, read[1], stamp, state.name)
                )
        for path in glob.glob(os.path.join(str(experiment.path), "overview*.tif")):
            read = self._read(path, place=True)
            if read is None:
                continue
            key = (read[0] or NO_GRID) if has_grids else NO_GRID
            index.setdefault(key, []).append(
                _Overview(path, read[1], from_posix(os.path.getmtime(path)), "Overview")
            )
        for overviews in index.values():
            overviews.sort(key=lambda o: o.when)
        return index

    def _read(
        self, path: str, place: bool = False
    ) -> Optional[Tuple[Optional[str], str]]:
        """(the grid it was placed on, its view), from the file's metadata alone."""
        try:
            key = (path, os.path.getmtime(path))
        except OSError:
            return None
        if key in self._seen:
            return self._seen[key]
        result = None
        if _is_fluorescence(path):
            result = (None, VIEW_FM)
        else:
            metadata = _read_metadata(path)
            view = _beam_view(metadata) if metadata is not None else None
            if view is not None:
                grid = (
                    grid_of_root_overview(self._microscope, self._experiment, metadata)
                    if place
                    else None
                )
                result = (grid.id if grid is not None else None, view)
        self._seen[key] = result
        return result

    def _newest_per_view(self, key: Optional[str]) -> List[_Overview]:
        newest: Dict[str, _Overview] = {}
        for overview in self._index.get(key, []) if key is not None else []:
            newest[overview.view] = overview
        return sorted(
            newest.values(),
            key=lambda o: (
                _VIEW_ORDER.index(_view_group(o.view))
                if _view_group(o.view) in _VIEW_ORDER
                else len(_VIEW_ORDER),
                o.view,
            ),
        )

    # ── showing ───────────────────────────────────────────────────────────
    def _fill_selector(self) -> None:
        has_grids = self._has_grids()
        self.grid_selector.setVisible(has_grids)
        if not has_grids:
            return
        current = self._under
        if current is not None:
            label = f"Current grid ({self._grid_name(current)})"
        elif self._last_grid is not None:
            label = f"Current grid (none; showing {self._grid_name(self._last_grid)})"
        else:
            label = "Current grid (none)"
        entries = [(label, CURRENT)]
        entries += [(g.name, g.id) for g in self._experiment.grids]
        if self._index.get(NO_GRID):
            entries.append(("Not on a grid", NO_GRID))
        self.grid_selector.blockSignals(True)
        self.grid_selector.clear()
        for text, data in entries:
            self.grid_selector.addItem(text, data)
        index = self.grid_selector.findData(self._mode)
        self.grid_selector.setCurrentIndex(max(index, 0))
        self.grid_selector.blockSignals(False)

    def _on_grid_chosen(self, index: int) -> None:
        self._mode = self.grid_selector.itemData(index)
        self._show()

    def _show(self) -> None:
        if self._experiment is None:
            self._show_empty("No experiment loaded.")
            return
        key = self.shown_grid()
        if key is None:
            self._show_empty("The stage isn't on a grid. Choose one above.")
            return
        overviews = self._newest_per_view(key)
        if not overviews:
            if not self._has_grids():
                self._show_empty(
                    "No overviews in this experiment yet. Acquire one from the "
                    "Overview tab."
                )
            else:
                self._show_empty(
                    f"No overview of {self._grid_name(key)} yet. Acquire one from the "
                    "Overview tab, or run a grid overview task."
                )
            return
        wanted = (key, tuple(o.path for o in overviews))
        if wanted != self._shown:
            self._place(overviews)
            self._shown = wanted
            self._mark_boundary()
        self._stack.setCurrentWidget(self.canvas)
        self._mark_lamellae()
        self._mark_stage()
        self._refresh_chips()
        self._refresh_caption()

    def _place(self, overviews: List[_Overview]) -> None:
        previous = self.canvas.view
        self.canvas.clear()
        self._view_by_group = {}
        for overview in overviews:
            image = self._image(overview.path)
            if image is None:
                continue
            when = format_time(overview.when, TIME_DISPLAY_AMPM_SHORT)
            record = self.canvas.set_image(image, label=f"{overview.label} · {when}")
            if record is not None:
                self._view_by_group.setdefault(
                    _view_group(overview.view), overview.view
                )
        # Keep the kind of view the user was looking at, if this grid has one.
        views = self.canvas.views
        if previous is not None and _view_group(previous) in self._view_by_group:
            self.canvas.show_view(self._view_by_group[_view_group(previous)])
        elif views:
            self.canvas.show_view(views[0])

    def _image(self, path: str):
        if path in self._images:
            return self._images[path]
        try:
            image = _load_image(path)
        except Exception:
            logger.warning(f"Could not read the overview {path}", exc_info=True)
            image = None
        self._images[path] = image
        while len(self._images) > _IMAGE_CACHE:
            self._images.pop(next(iter(self._images)))
        return image

    def _show_empty(self, text: str) -> None:
        self.empty.setText(text)
        self._stack.setCurrentWidget(self.empty)
        self._shown = None
        self.canvas.set_grid_boundary(None)
        self._view_by_group = {}
        self.caption.hide()
        self._clear_chips()

    def _clear_chips(self) -> None:
        """Take the chips down now, not when the deletion runs: a chip out of the
        layout but not yet deleted is still a visible child of the header, drawn at
        whatever size it last had -- over the grid selector."""
        for chip in self._chips.values():
            self._chips_layout.removeWidget(chip)
            chip.hide()
            chip.setParent(None)
            chip.deleteLater()
        self._chips = {}

    def _refresh_chips(self) -> None:
        self._clear_chips()
        groups = list(self._view_by_group)
        for group in groups:
            view = self._view_by_group[group]
            chip = QPushButton(group)
            chip.setToolTip(view)
            chip.setCheckable(True)
            chip.setChecked(view == self.canvas.view)
            chip.setCursor(Qt.PointingHandCursor)
            chip.setStyleSheet(VIEW_CHIP_STYLE)
            chip.clicked.connect(partial(self._on_chip_clicked, view))
            self._chips_layout.addWidget(chip)
            self._chips[view] = chip

    def _on_chip_clicked(self, view: str) -> None:
        self.canvas.show_view(view)
        for name, chip in self._chips.items():
            chip.setChecked(name == self.canvas.view)
        self._mark_boundary()
        self._mark_lamellae()
        self._mark_stage()
        self._refresh_caption()

    def _refresh_caption(self) -> None:
        view = self.canvas.view
        record = next((r for r in self.canvas.overviews if r.view == view), None)
        if record is None:
            self.caption.hide()
            return
        key = self.shown_grid()
        prefix = f"{self._grid_name(key)} · " if self._has_grids() else ""
        self.caption.setText(prefix + record.label)
        self.caption.adjustSize()
        self.caption.move(8, max(self.canvas.height() - self.caption.height() - 8, 0))
        self.caption.show()
        self.caption.raise_()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        if self.caption.isVisible():
            self._refresh_caption()

    # ── marks ─────────────────────────────────────────────────────────────
    def _lamellae(self) -> List[Lamella]:
        if self._experiment is None:
            return []
        key = self.shown_grid()
        if not self._has_grids() or key == NO_GRID:
            return list(self._experiment.positions)
        record = self._experiment.get_grid_by_id(key)
        return self._experiment.get_lamellae_for_grid(record) if record else []

    def _mark_lamellae(self) -> None:
        """The lamellae of the grid on show, each where the shown view puts it: the
        fluorescence pose on the FM view, the milling pose on a beam view."""
        on_fm = self.canvas.view == VIEW_FM
        positions = []
        for lamella in self._lamellae():
            if on_fm:
                pose = lamella.fluorescence_pose
                place = pose.stage_position if pose is not None else None
            else:
                place = lamella.stage_position
            if place is None:
                continue
            position = deepcopy(place)
            position.name = lamella.name
            positions.append(position)
        self.canvas.set_positions(positions, movable=False)
        self.canvas.set_selected_position(self._selected)

    def _mark_stage(self) -> None:
        """The stage, on a beam view of the grid it is on. Not on the fluorescence
        view: the stage is at the beams' position, which that view does not draw."""
        on_this_grid = not self._has_grids() or self._under == self.shown_grid()
        show = (
            self._stage is not None
            and on_this_grid
            and self.canvas.view is not None
            and self.canvas.view != VIEW_FM
        )
        self.canvas.set_current_position(self._stage if show else None)

    def _mark_boundary(self) -> None:
        self.canvas.set_grid_boundary(self._grid_centre())

    def _grid_centre(self) -> Optional[FibsemStagePosition]:
        """The centre of the slot the shown grid is loaded in, as the shown view
        places things: the beams' frame, or the fluorescence microscope's."""
        key = self.shown_grid()
        if not self._has_grids() or not key:
            return None
        record = self._experiment.get_grid_by_id(key)
        if record is None:
            return None
        slot = next(
            (
                slot
                for slot in holder_slots(self._microscope)
                if getattr(getattr(slot, "loaded_grid", None), "name", None)
                == record.name
            ),
            None,
        )
        place = slot_landmark(self._microscope, slot) if slot is not None else None
        if place is not None and self.canvas.view == VIEW_FM:
            place = at_device(self._microscope, "FM", place)
        return place

    def _on_position_selected(self, name: str) -> None:
        self._selected = name
        lamella = next((lam for lam in self._lamellae() if lam.name == name), None)
        if lamella is not None:
            self.lamella_selected.emit(lamella)


__all__ = ["QuadOverviewPage", "grid_of_root_overview", "CURRENT", "NO_GRID"]
