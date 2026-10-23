"""The volume Protocol tab's slice milling settings: what one slice mills, with
only what a slice changes exposed, and a preview of slice 1 and the next two.

The form writes through the shared task editor, as every settings edit does;
"All milling settings…" opens the full milling editor for the rest.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QFormLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.volume.position import VolumePosition
from fibsem.applications.volume.tasks.milling import (
    VolumeFIBMillingTaskConfig,
    slice_milling_at,
)
from fibsem.milling.patterning.patterns import RectanglePattern
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.structures import BeamType, FibsemImage, FibsemMillingSettings
from fibsem.ui import stylesheets
from fibsem.ui.tokens import SEMANTIC_WARNING_COLOR
from fibsem.ui.utils import beam_choices
from fibsem.ui.widgets.canvas.canvas_state import MillingSpec
from fibsem.ui.widgets.canvas.quad_view import (
    LamellaEditorView,
    MicroscopeViewController,
)
from fibsem.ui.widgets.custom_widgets import TitledPanel
from fibsem.ui.widgets.form_builder import Control, build_control

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope

__all__ = ["SliceMillingForm", "SlicePreview", "slice_form_fits", "preview_slices"]

PATTERN_FIELDS = ("width", "height", "depth")
PREVIEW_SLICES = 3
PREVIEW_OVERLAY_ID = "slice_preview"
# the preview frame is this many cut widths across, so the slices part
PREVIEW_WIDTHS = 1.5


def slice_form_fits(config: Any) -> bool:
    """True for a slice milling config this form can show: one rectangle."""
    return (
        isinstance(config, VolumeFIBMillingTaskConfig)
        and len(config.milling.stages) == 1
        and isinstance(config.milling.stages[0].pattern, RectanglePattern)
    )


def preview_slices(config: VolumeFIBMillingTaskConfig) -> List:
    """Slice 1 and the next two, each placed as the milling task places it, on
    a position whose point of interest is the image centre."""
    position = VolumePosition(name="preview")
    stages = []
    for index in range(PREVIEW_SLICES):
        stage = slice_milling_at(config, position, index).stages[0]
        stage.name = f"Slice {index + 1}"
        stages.append(stage)
    return stages


def _label(spec) -> str:
    return (spec.label or "").capitalize()


class SliceMillingForm(QWidget):
    """Slice thickness, then the slice's cut: width, height, depth, milling
    current and field of view."""

    # the config's new values, by the path the task editor writes
    values_changed = pyqtSignal(dict)
    all_settings_clicked = pyqtSignal()

    def __init__(
        self,
        microscope: Optional["FibsemMicroscope"] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.microscope = microscope
        self._config: Optional[VolumeFIBMillingTaskConfig] = None
        self._controls: Dict[str, Control] = {}

        self._slice_form = QFormLayout()
        self._cut_form = QFormLayout()
        thickness_help = QLabel("Each slice is cut this far below the one before.")
        thickness_help.setWordWrap(True)
        thickness_help.setStyleSheet(stylesheets.LABEL_INSTRUCTIONS_STYLE)
        self.warning = QLabel()
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet(f"color: {SEMANTIC_WARNING_COLOR};")
        fixed = QLabel(
            "Scan direction (top to bottom) and the cleaning cross-section "
            "are fixed for a slice."
        )
        fixed.setWordWrap(True)
        fixed.setStyleSheet(stylesheets.LABEL_INSTRUCTIONS_STYLE)
        self.all_settings_button = QPushButton("All milling settings…")
        self.all_settings_button.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)

        slice_box = QWidget()
        slice_column = QVBoxLayout(slice_box)
        slice_column.addLayout(self._slice_form)
        slice_column.addWidget(thickness_help)
        cut_box = QWidget()
        cut_column = QVBoxLayout(cut_box)
        cut_column.addLayout(self._cut_form)
        cut_column.addWidget(self.warning)
        cut_column.addWidget(fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(TitledPanel("Slice", content=slice_box, collapsible=False))
        layout.addWidget(TitledPanel("Cut", content=cut_box, collapsible=False))
        layout.addWidget(self.all_settings_button)
        self.all_settings_button.clicked.connect(self.all_settings_clicked)

    def set_microscope(self, microscope: Optional["FibsemMicroscope"]) -> None:
        self.microscope = microscope

    def _current_choices(self, name: str) -> List[Any]:
        if self.microscope is None:
            return []
        return beam_choices(self.microscope, name, BeamType.ION)

    def show_config(self, config: VolumeFIBMillingTaskConfig) -> None:
        """Rebuild the controls from a copy of *config*, one slice milling
        config: an edit changes the config only through the write."""
        config = deepcopy(config)
        self._config = config
        for form in (self._slice_form, self._cut_form):
            while form.rowCount():
                form.removeRow(0)
        self._controls = {}
        stage = config.milling.stages[0]
        rows: List[tuple] = [
            (
                self._slice_form,
                "slice_thickness",
                VolumeFIBMillingTaskConfig.params["slice_thickness"],
                config.slice_thickness,
            )
        ]
        rows += [
            (
                self._cut_form,
                name,
                RectanglePattern.params[name],
                getattr(stage.pattern, name),
            )
            for name in PATTERN_FIELDS
        ]
        rows += [
            (
                self._cut_form,
                "milling_current",
                FibsemMillingSettings.params["milling_current"],
                stage.milling.milling_current,
            ),
            (
                self._cut_form,
                "field_of_view",
                FibsemMillingTaskConfig.params["field_of_view"],
                config.milling.field_of_view,
            ),
        ]
        for form, name, spec, value in rows:
            control = build_control(spec, value, resolve_choices=self._current_choices)
            label = QLabel(_label(spec))
            label.setToolTip(spec.doc or "")
            form.addRow(label, control.widget)
            control.connect(self._edit_handler(name))
            self._controls[name] = control
        self._show_warning()

    def value(self, name: str) -> Any:
        return self._controls[name].read()

    def _edit_handler(self, name: str) -> Callable[[], None]:
        return lambda: self._on_edit(name)

    def _on_edit(self, name: str) -> None:
        config = self._config
        if config is None:
            return
        value = self._controls[name].read()
        if name == "slice_thickness":
            config.slice_thickness = value
            self.values_changed.emit({"slice_thickness": value})
        else:
            milling = deepcopy(config.milling)
            stage = milling.stages[0]
            if name in PATTERN_FIELDS:
                setattr(stage.pattern, name, value)
            elif name == "milling_current":
                stage.milling.milling_current = value
            else:
                milling.field_of_view = value
            config.milling = milling
            self.values_changed.emit({"milling": milling})
        self._show_warning()

    def _show_warning(self) -> None:
        """Say when the cut is thinner than a slice: material is left between."""
        config = self._config
        if config is None:
            self.warning.setVisible(False)
            return
        height = config.milling.stages[0].pattern.height
        gap = config.slice_thickness - height
        self.warning.setVisible(gap > 1e-12)
        self.warning.setText(
            f"Height is less than the slice thickness: {gap * 1e6:.3f} µm is "
            "left between slices."
        )


class SlicePreview(QWidget):
    """Slice 1 and the next two, drawn on a blank FIB frame zoomed to the cut:
    at the milling's field of view a slice apart is a pixel or two."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.view_controller = MicroscopeViewController(
            parent=self, view=LamellaEditorView()
        )
        self.view_controller.widget.show_beams()
        self.caption = QLabel()
        self.caption.setStyleSheet(stylesheets.LABEL_INSTRUCTIONS_STYLE)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view_controller.widget, 1)
        layout.addWidget(self.caption)

    def show_config(self, config: VolumeFIBMillingTaskConfig) -> None:
        hfw = config.milling.stages[0].pattern.width * PREVIEW_WIDTHS
        image = FibsemImage.generate_blank_image(hfw=hfw)
        self.view_controller.set_image(BeamType.ION, image)
        self.view_controller.set_overlay(
            BeamType.ION,
            MillingSpec(id=PREVIEW_OVERLAY_ID, stages=preview_slices(config)),
        )
        self.caption.setText(
            "Slice 1, then the next two, one thickness apart · "
            f"zoomed to the cut, HFW {hfw * 1e6:.1f} µm"
        )
