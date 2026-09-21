"""Grids tab · Positions: mark lamellae on a grid's stored overviews.

The screening hand-off. A grid was screened -- on this instrument, or another
one, or last week -- and its overviews are on disk. This view puts them on a
``StoredOverviewCanvas`` (placed from their own metadata, nothing live), marks
the lamellae that belong to the grid, and lets more be marked, moved and
removed. Every lamella made here carries the grid's id, whether or not the grid
is on the stage; that is what the Overview tab, which *is* the stage, cannot do.

Nothing here reads the microscope's state or moves it. The one thing that still
needs an instrument connected is making a lamella: its poses are built with the
instrument's geometry (``build_lamella_poses``), so with nothing connected the
canvas shows and the add is refused with a reason.
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from datetime import datetime
from functools import partial
from typing import Callable, Dict, List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.poses import (
    FLUORESCENCE_POSE,
    MILLING_POSE,
    followed_note,
    move_consequence,
    move_pose,
)
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskStatus,
    Experiment,
    GridRecord,
    Lamella,
)
from fibsem.applications.autolamella.ui.grid_results_widget import image_for
from fibsem.applications.autolamella.ui.lamella_name_list_widget import (
    LamellaNameListWidget,
)
from fibsem.applications.autolamella.workflows.tasks.grid.manager import (
    LOAD_ENTRY_NAME as _LOAD_ENTRY_NAME,
)
from fibsem.constants import TIME_DISPLAY_AMPM_SHORT
from fibsem.fm.structures import FluorescenceImage
from fibsem.structures import FibsemImage, FibsemStagePosition
from fibsem.ui import notification_service, stylesheets
from fibsem.ui.tokens import (
    NEUTRAL_200,
    NEUTRAL_550,
    OK_COLOR,
    PANEL_COLOR,
    WARN_COLOR,
)
from fibsem.ui.utils import message_box_ui
from fibsem.ui.widgets.overview_widget import VIEW_CHIP_SPACING, VIEW_CHIP_STYLE
from fibsem.ui.widgets.stored_overview_canvas import VIEW_FM, StoredOverviewCanvas

logger = logging.getLogger(__name__)

_SIDE_WIDTH = 260


def _load_overview(path: str):
    """The image at *path*, as the type that can place itself."""
    if path.endswith((".ome.tiff", ".ome.tif")):
        return FluorescenceImage.load(path)
    return FibsemImage.load(path)


class GridPositionsWidget(QWidget):
    """One grid's stored overviews, with its lamellae marked on them.

    Signals:
        load_requested(GridRecord): the user asked for the grid on the stage.
        done_requested(): the user is finished marking.
        lamella_selected(Lamella): a marked lamella was picked here.
    """

    load_requested = pyqtSignal(object)
    done_requested = pyqtSignal()
    lamella_selected = pyqtSignal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._autolamella_ui = None
        self._experiment: Optional[Experiment] = None
        self._grid: Optional[GridRecord] = None
        self._loaded = False
        self._can_load = False
        # record id -> the image path it came from, so a results row can ask for
        # its own overview to be shown.
        self._paths: Dict[str, str] = {}
        self._chips: Dict[str, QPushButton] = {}
        # Replaceable so a test can answer the "move it?" question.
        self._confirm: Callable[[str, str], bool] = lambda title, text: message_box_ui(
            title=title, text=text, parent=self
        )
        self._setup_ui()
        self.refresh()

    # -- layout ----------------------------------------------------------------

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # The banner: whose overviews, and whether the grid is on the stage.
        banner = QWidget()
        banner.setStyleSheet(f"background: {PANEL_COLOR};")
        banner_layout = QHBoxLayout(banner)
        banner_layout.setContentsMargins(12, 6, 12, 6)
        banner_layout.setSpacing(8)
        self.title_label = QLabel()
        self.title_label.setStyleSheet(
            f"font-size: 13px; font-weight: bold; color: {NEUTRAL_200}; "
            "background: transparent;"
        )
        banner_layout.addWidget(self.title_label)
        self.state_label = QLabel()
        self.state_label.setStyleSheet(
            f"font-size: 11px; color: {NEUTRAL_550}; background: transparent;"
        )
        banner_layout.addWidget(self.state_label, 1)
        self.chips_row = QWidget()
        self.chips_row.setStyleSheet("background: transparent;")
        self._chips_layout = QHBoxLayout(self.chips_row)
        self._chips_layout.setContentsMargins(0, 0, 0, 0)
        self._chips_layout.setSpacing(VIEW_CHIP_SPACING)
        banner_layout.addWidget(self.chips_row)
        outer.addWidget(banner)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        self.canvas = StoredOverviewCanvas()
        self.canvas.position_add_requested.connect(self._on_add_requested)
        self.canvas.position_move_requested.connect(self._on_move_requested)
        self.canvas.position_selected.connect(self._on_marker_clicked)
        self.canvas.view_changed.connect(self._on_view_changed)
        body.addWidget(self.canvas, 1)

        side = QWidget()
        side.setFixedWidth(_SIDE_WIDTH)
        side.setStyleSheet(f"background: {PANEL_COLOR};")
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(8, 8, 8, 8)
        side_layout.setSpacing(6)
        self.count_label = QLabel()
        self.count_label.setStyleSheet(
            f"font-size: 12px; font-weight: 600; color: {NEUTRAL_200}; "
            "background: transparent;"
        )
        side_layout.addWidget(self.count_label)
        self.lamella_list = LamellaNameListWidget()
        self.lamella_list.enable_remove_button(True)
        self.lamella_list.lamella_selected.connect(self._on_list_selection)
        self.lamella_list.remove_requested.connect(self._on_remove_requested)
        side_layout.addWidget(self.lamella_list, 1)
        self.hint_label = QLabel(
            "Right-click the overview to add a position, or to move the selected "
            "one. Positions are saved as lamellae on this grid; nothing moves the "
            "stage from here."
        )
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet(
            f"font-size: 11px; color: {NEUTRAL_550}; background: transparent;"
        )
        side_layout.addWidget(self.hint_label)
        actions = QHBoxLayout()
        actions.setSpacing(6)
        self.btn_load = QPushButton()
        self.btn_load.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.btn_load.clicked.connect(self._on_load_clicked)
        actions.addWidget(self.btn_load)
        self.btn_done = QPushButton("Done")
        self.btn_done.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.btn_done.clicked.connect(self.done_requested)
        actions.addWidget(self.btn_done)
        side_layout.addLayout(actions)
        body.addWidget(side)
        outer.addLayout(body, 1)

    # -- model -----------------------------------------------------------------

    def set_autolamella_ui(self, autolamella_ui) -> None:
        """The application widget that makes lamellae; None to only look."""
        self._autolamella_ui = autolamella_ui

    def set_experiment(self, experiment: Optional[Experiment]) -> None:
        self._experiment = experiment
        if experiment is None or (
            self._grid is not None and experiment.get_grid_by_id(self._grid.id) is None
        ):
            self._grid = None
        self._load_images()
        self.refresh()

    def set_grid(
        self,
        grid: Optional[GridRecord],
        loaded: bool = False,
        can_load: bool = False,
    ) -> None:
        """Show *grid*'s stored overviews. A new grid starts the canvas afresh;
        the same grid keeps what is placed, and `refresh` adds any overview
        that has landed since."""
        same = grid is not None and self._grid is not None and grid.id == self._grid.id
        self._grid = grid
        self._loaded = loaded
        self._can_load = can_load
        if not same:
            self.canvas.clear()
            self._paths = {}
        self.refresh()

    def set_stage_state(self, loaded: bool, can_load: bool) -> None:
        self._loaded = loaded
        self._can_load = can_load
        self.refresh()

    @property
    def grid(self) -> Optional[GridRecord]:
        return self._grid

    @property
    def microscope(self):
        return getattr(self._autolamella_ui, "microscope", None)

    def _load_images(self) -> None:
        """Place every overview the grid's history records that is not on the
        canvas yet. Each image is read from disk once; a refresh while a grid
        task is running picks up what it saved without re-reading the rest."""
        grid, experiment = self._grid, self._experiment
        if grid is None or experiment is None:
            self.canvas.clear()
            self._paths = {}
            return
        placed = {os.path.normpath(p) for p in self._paths.values()}
        first = not self._paths
        # `set_image` switches the canvas to the image's view; an overview
        # landing mid-session must not pull the user off the view they are on.
        current = self.canvas.view
        for state in grid.task_history:
            if (
                state.name == _LOAD_ENTRY_NAME
                or state.status is not AutoLamellaTaskStatus.Completed
            ):
                continue
            path = image_for(experiment, grid, state)
            if path is None or os.path.normpath(path) in placed:
                continue
            try:
                image = _load_overview(path)
            except Exception as e:  # noqa: BLE001 - one bad file, said, not fatal
                logger.warning(f"Could not read {path}: {e}")
                continue
            stamp = state.end_timestamp or state.start_timestamp
            when = datetime.fromtimestamp(stamp).strftime(TIME_DISPLAY_AMPM_SHORT)
            record_id = self.canvas.set_image(image, label=f"{state.name} · {when}")
            if record_id is None:
                logger.warning(
                    f"{os.path.basename(path)} says nothing about where it was taken; not shown."
                )
                continue
            self._paths[record_id] = path
            placed.add(os.path.normpath(path))
        views = self.canvas.views
        if first and views:
            self.canvas.show_view(views[0])
        elif current in views and self.canvas.view != current:
            self.canvas.show_view(current)

    def show_overview(self, path: str) -> bool:
        """Bring the view holding the overview at *path* to the front."""
        for record_id, known in self._paths.items():
            if os.path.normpath(known) == os.path.normpath(path):
                record = next(
                    (r for r in self.canvas.overviews if r.id == record_id), None
                )
                if record is not None:
                    self.canvas.show_view(record.view)
                    return True
        return False

    @property
    def views(self) -> List[str]:
        return self.canvas.views

    # -- drawing ---------------------------------------------------------------

    def refresh(self) -> None:
        self._load_images()
        grid, experiment = self._grid, self._experiment
        has_grid = grid is not None and experiment is not None
        for widget in (self.chips_row, self.btn_load, self.btn_done):
            widget.setVisible(has_grid)
        if not has_grid:
            self.title_label.setText("No grid selected")
            self.state_label.setText("Select a grid card to mark positions on it.")
            self.count_label.setText("")
            self.lamella_list.set_lamella([])
            self.canvas.set_positions([])
            return

        self.title_label.setText(grid.name)
        parts = ["Stored overview" if self.canvas.views else "No stored overview"]
        if self._loaded:
            parts.append("on the stage")
        else:
            parts += ["not on the stage", "positions are saved to the grid"]
        self.state_label.setText(" · ".join(parts))
        self.state_label.setStyleSheet(
            f"font-size: 11px; color: {OK_COLOR if self._loaded else WARN_COLOR}; "
            "background: transparent;"
        )
        self.btn_load.setText(f"Load {grid.name}")
        self.btn_load.setVisible(self._can_load and not self._loaded)
        self._refresh_chips()

        lamellae = experiment.get_lamellae_for_grid(grid)
        self.count_label.setText(f"Positions on {grid.name} · {len(lamellae)}")
        self.lamella_list.set_lamella(lamellae)
        self._mark(lamellae)

    def _refresh_chips(self) -> None:
        for chip in self._chips.values():
            self._chips_layout.removeWidget(chip)
            chip.deleteLater()
        self._chips = {}
        for view in self.canvas.views:
            first = next(r for r in self.canvas.overviews if r.view == view)
            chip = QPushButton(first.label.split(" · ")[0])
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

    def _on_view_changed(self, _view: str) -> None:
        for name, chip in self._chips.items():
            chip.setChecked(name == self.canvas.view)
        if self._grid is not None and self._experiment is not None:
            self._mark(self._experiment.get_lamellae_for_grid(self._grid))

    def _pose_of(self, lamella: Lamella) -> Optional[FibsemStagePosition]:
        """Where the shown view puts *lamella*: the fluorescence pose on the FM
        view, the milling pose on a beam view."""
        if self.canvas.view == VIEW_FM:
            pose = lamella.fluorescence_pose
            return pose.stage_position if pose is not None else None
        return lamella.stage_position

    def _mark(self, lamellae: List[Lamella]) -> None:
        positions = []
        for lamella in lamellae:
            place = self._pose_of(lamella)
            if place is None:
                continue
            position = deepcopy(place)
            position.name = lamella.name
            positions.append(position)
        self.canvas.set_positions(positions)

    # -- selection -------------------------------------------------------------

    def _on_list_selection(self, lamella) -> None:
        self.canvas.set_selected_position(lamella.name if lamella else None)
        if lamella is not None:
            self.lamella_selected.emit(lamella)

    def _on_marker_clicked(self, name: str) -> None:
        self.lamella_list.select(name)

    # -- edits -----------------------------------------------------------------

    def _lamella_named(self, name: str) -> Optional[Lamella]:
        experiment = self._experiment
        if experiment is None:
            return None
        return next((p for p in experiment.positions if p.name == name), None)

    def _add_kwargs(self) -> dict:
        if self.canvas.view == VIEW_FM:
            return {"observed": FLUORESCENCE_POSE}
        return {}

    def _on_add_requested(self, position, _record_id=None) -> None:
        """A new lamella at a point on the stored overview, on this grid."""
        grid = self._grid
        ui = self._autolamella_ui
        if grid is None or self._experiment is None:
            return
        if ui is None or ui.microscope is None:
            notification_service.show_toast(
                "Connect to a microscope to mark positions: a lamella's poses are "
                "built with the instrument's geometry.",
                "warning",
            )
            return
        try:
            lamella = ui.add_new_lamella(
                stage_position=position, grid_id=grid.id, **self._add_kwargs()
            )
        except Exception as e:  # noqa: BLE001 - said to the user, not raised
            logger.error(f"Could not add a lamella on {grid.name}: {e}")
            notification_service.show_toast(str(e), "error")
            return
        self.refresh()
        self.lamella_list.select(lamella.name)
        notification_service.show_toast(f"Added {lamella.name} on {grid.name}.", "info")

    def _on_move_requested(self, name: str, position) -> None:
        """Move a marked lamella, the way the Overview tab of the shown view
        would: the pose of the shown view moves, and the other does what the rule
        in `poses.move_pose` says. The FM view asks first, since what happens to
        the milling pose is not visible from there."""
        experiment = self._experiment
        lamella = self._lamella_named(name)
        microscope = self.microscope
        if experiment is None or lamella is None:
            return
        if microscope is None:
            notification_service.show_toast(
                "Connect to a microscope to move positions.", "warning"
            )
            return
        if self.canvas.view == VIEW_FM:
            if lamella.milling_pose is None:
                notification_service.show_toast(
                    f"{name} has no milling pose to move.", "warning"
                )
                return
            moved = FLUORESCENCE_POSE
            if not self._confirm(
                f"Move {name}?",
                f"Move {name} to {position.pretty_string}?"
                f"\n\n{move_consequence(lamella, moved)}",
            ):
                return
        else:
            moved = MILLING_POSE
        followed = move_pose(microscope, lamella, moved, position=position)
        experiment.save()
        # A pose written in place emits nothing on its own; the window re-marks
        # the Overview tabs off this.
        experiment.positions.events.changed.emit()
        self.refresh()
        notification_service.show_toast(
            f"Moved {name}. {followed_note(moved, followed)}".strip(), "info"
        )

    def _on_remove_requested(self, lamella) -> None:
        """The row asked first; this does not ask twice."""
        experiment = self._experiment
        if experiment is None or lamella is None:
            return
        try:
            experiment.positions.remove(lamella)
        except ValueError:
            return
        experiment.save()
        self.refresh()
        notification_service.show_toast(f"Removed {lamella.name}.", "info")

    def _on_load_clicked(self) -> None:
        if self._grid is not None:
            self.load_requested.emit(self._grid)
