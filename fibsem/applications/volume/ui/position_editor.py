"""The Positions tab's Position view: what is stored on one position, edited
over a saved FIB image of it. Right-click on the image places the point of
interest; "Edit on the image" drags the alignment area. Its slices and
description sit beside it, under its name, which is fixed. The protocol's
milling is drawn where it mills on this position, to see, not to edit.

Every edit is the experiment's own method, the one the App Server's position
patch calls, with its refusal said on a toast. Nothing here drives the
microscope: what it does now is on the Microscope tab's Positions panel.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from fibsem import conversions
from fibsem.applications.volume.position import FIB_REFERENCE, VolumePosition
from fibsem.applications.volume.position_milling import (
    position_millings,
    slice_depth_text,
)
from fibsem.structures import BeamType, FibsemImage, FibsemRectangle, Point
from fibsem.ui import stylesheets
from fibsem.ui.run.run_refusal import refused
from fibsem.ui.widgets.canvas.canvas_state import (
    AlignmentSpec,
    MillingSpec,
    PointsSpec,
)
from fibsem.ui.widgets.canvas.quad_view import (
    ItemEditorView,
    MicroscopeViewController,
)
from fibsem.ui.widgets.custom_widgets import ContextMenu, ContextMenuConfig
from fibsem.util.timestamps import format_time
from fibsem.workflows.image_positions import latest_image_at, read_image_position
from fibsem.workflows.item import MILLING_POSE
from fibsem.workflows.outputs import all_item_outputs

if TYPE_CHECKING:
    from fibsem.applications.volume.experiment import VolumeExperiment
    from fibsem.workflows.session import AppSession

__all__ = ["PositionEditor", "image_label", "position_images"]

POI_OVERLAY_ID = "position_poi"
ALIGNMENT_OVERLAY_ID = "position_alignment"
MILLING_OVERLAY_ID = "position_milling"
FIB_SUFFIX = "_ib.tif"
FORM_WIDTH = 400


def position_images(
    experiment: "VolumeExperiment", position: VolumePosition
) -> List[str]:
    """The position's saved FIB images: every run's, and those saved in its
    folder outside a run. Not its alignment reference, which is the alignment
    area only."""
    folder = str(experiment.item_path(position))
    paths = [
        p for p in all_item_outputs(experiment, position) if p.endswith(FIB_SUFFIX)
    ]
    if os.path.isdir(folder):
        paths += [
            os.path.join(folder, name)
            for name in sorted(os.listdir(folder))
            if name.endswith(FIB_SUFFIX) and name != FIB_REFERENCE
        ]
    return list(dict.fromkeys(paths))


def image_label(folder: str, path: str) -> str:
    """ "Setup FIB Milling · ref_setup_face", from a run's folder
    (``<task>/<run>/<file>``); a slice's run says its slice."""
    parts = os.path.relpath(path, folder).split(os.sep)
    stem = parts[-1][: -len(FIB_SUFFIX)]
    if len(parts) < 3:
        return stem
    task, run = parts[0], parts[-2]
    if run.isdigit():
        return f"{task} · slice {int(run) + 1} · {stem}"
    return f"{task} · {stem}"


class PositionEditor(QWidget):
    """One position's stored fields, over a saved FIB image of it."""

    # a field of the position changed here
    edited = pyqtSignal()

    def __init__(self, session: "AppSession", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.session = session
        self._name: Optional[str] = None
        self._millings: dict = {}
        self.image: Optional[FibsemImage] = None
        self._image_path: Optional[str] = None

        self.image_combo = QComboBox()
        self.image_combo.setMinimumContentsLength(28)
        self.image_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLength)
        picker = QHBoxLayout()
        picker.addStretch(1)
        picker.addWidget(QLabel("Image"))
        picker.addWidget(self.image_combo)

        self.view_controller = MicroscopeViewController(
            parent=self, view=ItemEditorView()
        )
        self.view_controller.widget.show_beams()
        self.fib_canvas = self.view_controller.get_canvas(BeamType.ION)
        self.hint = QLabel("Right-click on the image to place the point of interest.")
        self.hint.setStyleSheet(stylesheets.LABEL_INSTRUCTIONS_STYLE)
        image_column = QVBoxLayout()
        image_column.addWidget(self.view_controller.widget, 1)
        image_column.addWidget(self.hint)

        self.form = QWidget()
        self.form.setFixedWidth(FORM_WIDTH)
        form = QFormLayout(self.form)
        form.setContentsMargins(0, 0, 0, 0)
        self.name_label = QLabel()
        self.slices_spin = QSpinBox()
        self.slices_spin.setRange(1, 100000)
        self.poi_label = QLabel()
        self.area_check = QCheckBox("Edit on the image")
        self.milling_combo = QComboBox()
        milling_note = QLabel(
            "Shown where it mills on this position. Slice milling: filled is cut, "
            "the solid outline is the next slice, dashed is still to cut. Set on "
            "the Protocol tab."
        )
        milling_note.setWordWrap(True)
        milling_note.setStyleSheet(stylesheets.LABEL_INSTRUCTIONS_STYLE)
        milling = QVBoxLayout()
        milling.setContentsMargins(0, 0, 0, 0)
        milling.addWidget(self.milling_combo)
        milling.addWidget(milling_note)
        self.depth_label = QLabel()
        self.depth_label.setWordWrap(True)
        self.description_edit = QLineEdit()
        self.description_edit.setPlaceholderText("Notes on this position")
        form.addRow("Name", self.name_label)
        form.addRow("Slices", self.slices_spin)
        form.addRow("Point of interest", self.poi_label)
        form.addRow("Alignment area", self.area_check)
        form.addRow("Milling", milling)
        form.addRow("Depth into the face", self.depth_label)
        form.addRow("Description", self.description_edit)

        body = QHBoxLayout()
        body.addLayout(image_column, 1)
        body.addWidget(self.form)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(picker)
        layout.addLayout(body, 1)

        self.image_combo.currentIndexChanged.connect(lambda _: self._show_image())
        self.milling_combo.currentIndexChanged.connect(lambda _: self._draw())
        self.fib_canvas.canvas_right_clicked.connect(self._on_right_click)
        self.view_controller.overlay_edited.connect(self._on_overlay_edited)
        self.slices_spin.editingFinished.connect(self._on_slices)
        self.area_check.toggled.connect(self._on_area_toggled)
        self.description_edit.editingFinished.connect(self._on_description)
        self.show_position(None)

    @property
    def experiment(self) -> Optional["VolumeExperiment"]:
        return self.session.experiment

    @property
    def position(self) -> Optional[VolumePosition]:
        if self._name is None or self.experiment is None:
            return None
        return self.experiment.get_position_by_name(self._name)

    # --- what it shows --------------------------------------------------------

    def show_position(self, name: Optional[str]) -> None:
        """Show *name*'s fields and images; the same one again keeps the image
        picked."""
        same = name == self._name
        self._name = name
        position = self.position
        self.setEnabled(position is not None)
        self._fill_fields(position)
        self._fill_millings(position, keep=same)
        self._fill_images(position, keep=same)

    def _fill_fields(self, position: Optional[VolumePosition]) -> None:
        for widget in (self.slices_spin, self.description_edit):
            widget.blockSignals(True)
        if position is None:
            self.name_label.clear()
            self.description_edit.clear()
            self.poi_label.clear()
        else:
            self.name_label.setText(position.name)
            self.slices_spin.setValue(position.iterations)
            self.description_edit.setText(position.description)
            poi = position.poi
            self.poi_label.setText(f"x {poi.x * 1e6:.2f} µm · y {poi.y * 1e6:.2f} µm")
        for widget in (self.slices_spin, self.description_edit):
            widget.blockSignals(False)

    def _fill_millings(self, position: Optional[VolumePosition], keep: bool) -> None:
        shown = self.milling_combo.currentText() if keep else None
        millings = position_millings(self.experiment, position) if position else []
        self._millings = {m.label: m for m in millings}
        self.milling_combo.blockSignals(True)
        self.milling_combo.clear()
        self.milling_combo.addItems(list(self._millings))
        if shown in self._millings:
            self.milling_combo.setCurrentText(shown)
        self.milling_combo.blockSignals(False)
        self.milling_combo.setEnabled(bool(millings))
        depth = slice_depth_text(self.experiment, position) if position else None
        self.depth_label.setText(depth or "")

    def _fill_images(self, position: Optional[VolumePosition], keep: bool) -> None:
        shown = self.image_combo.currentData() if keep else None
        paths = position_images(self.experiment, position) if position else []
        folder = str(self.experiment.item_path(position)) if position else ""
        self.image_combo.blockSignals(True)
        self.image_combo.clear()
        for path in paths:
            self.image_combo.addItem(image_label(folder, path), path)
            when = format_time(os.path.getmtime(path), "%H:%M")
            self.image_combo.setItemData(
                self.image_combo.count() - 1, f"{path}\n{when}", Qt.ToolTipRole
            )
        pick = shown if shown in paths else self._default_image(position, paths)
        if pick is not None:
            self.image_combo.setCurrentIndex(paths.index(pick))
        self.image_combo.blockSignals(False)
        self.image_combo.setEnabled(bool(paths))
        self._show_image()

    @staticmethod
    def _default_image(
        position: Optional[VolumePosition], paths: List[str]
    ) -> Optional[str]:
        """The newest image taken at the milling pose, else the newest."""
        if not paths:
            return None
        images = [read_image_position(path) for path in paths]
        pose = position.poses.get(MILLING_POSE) if position else None
        here = latest_image_at(images, pose.stage_position if pose else None)
        if here is None:
            here = max(images, key=lambda image: (image.timestamp, image.filename))
        return next(p for p in paths if os.path.basename(p) == here.filename)

    def _show_image(self) -> None:
        path = self.image_combo.currentData()
        if path and path == self._image_path and self.image is not None:
            self._draw()  # a refresh: the same file, its marks redrawn
            return
        self._image_path = path
        self.image = None
        if path:
            try:
                self.image = FibsemImage.load(path)
            except Exception:  # noqa: BLE001 - a file that can't be read shows nothing
                logging.warning(f"could not load {path}", exc_info=True)
        if self.image is None:
            self.view_controller.clear()
            self.hint.setText("No saved FIB image of this position yet.")
            return
        self.hint.setText("Right-click on the image to place the point of interest.")
        self.view_controller.set_image(BeamType.ION, self.image)
        self._draw()

    def _draw(self) -> None:
        position = self.position
        if position is None or self.image is None or self.image.metadata is None:
            return
        poi = conversions.microscope_image_to_image_coordinates(
            point=position.poi,
            image_shape=self.image.data.shape,
            pixel_size=self.image.metadata.pixel_size.x,
        )
        self.view_controller.set_overlay(
            BeamType.ION,
            PointsSpec(
                id=POI_OVERLAY_ID,
                points=[(poi.x, poi.y)],
                color="magenta",
                marker="+",
                size=14,
                edge_width=1.2,
                add_on_right_click=False,
                removable=False,
                modal=True,
                legend_label="Point of Interest",
            ),
        )
        placed = self._millings.get(self.milling_combo.currentText())
        self.view_controller.set_overlay(
            BeamType.ION,
            MillingSpec(
                id=MILLING_OVERLAY_ID,
                stages=placed.milling.stages if placed is not None else (),
                visible=placed is not None,
                stage_styles=placed.styles if placed is not None else (),
                stage_colours=placed.colours if placed is not None else (),
            ),
        )
        self.view_controller.set_overlay(
            BeamType.ION,
            AlignmentSpec(
                id=ALIGNMENT_OVERLAY_ID,
                rect=position.alignment_area,
                display=True,
                editing=self.area_check.isChecked(),
            ),
        )

    # --- edits ----------------------------------------------------------------

    def _edit(self, what: str, **fields) -> None:
        """Set *fields* on the shown position (``patch_position``), unless a
        run has it."""
        position = self.position
        if position is None:
            return
        if not refused(
            self.session.runner.busy_refusal([position.name], what)
        ) and not refused(self.experiment.patch_position(position.name, **fields)):
            self.edited.emit()
        self._fill_fields(position)
        self._draw()

    def _on_slices(self) -> None:
        position = self.position
        value = self.slices_spin.value()
        if position is None or value == position.iterations:
            return
        self._edit("change its slices", iterations=value)

    def _on_description(self) -> None:
        position = self.position
        text = self.description_edit.text()
        if position is None or text.strip() == position.description:
            return
        self._edit("change its description", description=text)

    def _on_right_click(self, x: float, y: float, _modifiers) -> None:
        image = self.image
        if image is None or image.metadata is None or self.position is None:
            return
        height, width = image.data.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            return
        point = conversions.image_to_microscope_image_coordinates(
            coord=Point(x=x, y=y),
            image=image.data,
            pixelsize=image.metadata.pixel_size.x,
        )
        menu = ContextMenuConfig()
        menu.add_action(
            "Move Point of Interest Here", callback=lambda: self.place_poi(point)
        )
        ContextMenu(menu, parent=self).show_at_cursor()

    def place_poi(self, point: Point) -> None:
        """Make *point*, in the image's frame, the point of interest."""
        self._edit("move its point of interest", poi=point)

    def _on_area_toggled(self, editing: bool) -> None:
        self._draw()
        self.view_controller.arm_overlay(
            BeamType.ION,
            ALIGNMENT_OVERLAY_ID if editing else None,
            label="Alignment",
            icon="mdi:vector-rectangle",
        )

    def _on_overlay_edited(self, beam, overlay_id: str, value) -> None:
        if beam == BeamType.ION and overlay_id == ALIGNMENT_OVERLAY_ID:
            self.set_alignment_area(value)

    def set_alignment_area(self, area: FibsemRectangle) -> None:
        self._edit("move its alignment area", alignment_area=area)
