"""The volume Protocol tab's volume milling settings, in one piece: the clearing
trench, the redeposition lanes, the start slice and the slices, with only what
each needs exposed, and a preview of where they mill around a point of
interest.

The form writes through the shared task editor, as every settings edit does;
"All milling settings…" opens the full milling editor for the rest.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QFormLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.volume.position import VolumePosition
from fibsem.applications.volume.position_milling import (
    PlacedMilling,
    setup_millings,
    slice_progress,
    whole_volume,
)
from fibsem.applications.volume.tasks.milling import VolumeFIBMillingTaskConfig
from fibsem.milling.patterning.patterns import RectanglePattern
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.structures import BeamType, FibsemImage, FibsemMillingSettings
from fibsem.theme.tokens import SEMANTIC_WARNING_COLOR
from fibsem.ui import stylesheets
from fibsem.ui.utils import beam_choices
from fibsem.ui.widgets.canvas.canvas_state import MillingSpec
from fibsem.ui.widgets.canvas.quad_view import ItemEditorView, MicroscopeViewController
from fibsem.ui.widgets.custom_widgets import TitledPanel
from fibsem.ui.widgets.form_builder import Control, build_control

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope

__all__ = [
    "VolumeMillingForm",
    "VolumePreview",
    "preview_milling",
    "volume_form_fits",
]

# Each group's rows: (milling config field or None for the task's own, field).
GROUPS: Tuple[Tuple[str, Tuple[Tuple[Optional[str], str], ...]], ...] = (
    (
        "Clearing trench",
        (
            (None, "mill_trench"),
            ("trench", "width"),
            ("trench", "height"),
            ("trench", "depth"),
            ("trench", "milling_current"),
        ),
    ),
    (
        "Redeposition lanes",
        (
            (None, "lane_left"),
            (None, "lane_right"),
            ("lanes", "width"),
            ("lanes", "depth"),
            ("lanes", "milling_current"),
        ),
    ),
    (
        "Start slice",
        (
            ("start_slice", "width"),
            ("start_slice", "height"),
            ("start_slice", "depth"),
            ("start_slice", "milling_current"),
        ),
    ),
    (
        "Slices",
        (
            (None, "slice_thickness"),
            (None, "flipped"),
            ("milling", "height"),
            ("milling", "milling_current"),
            ("milling", "field_of_view"),
        ),
    ),
)
PATTERN_FIELDS = ("width", "height", "depth")
MILLINGS = ("trench", "lanes", "start_slice", "milling")
# the preview's position: its slices, from a point of interest at the centre
PREVIEW_SLICES = 50
PREVIEW_OVERLAY_ID = "volume_preview"
# the preview frame is this much wider than the milling
PREVIEW_MARGIN = 1.2


def volume_form_fits(config: Any) -> bool:
    """True for a volume milling config this form can show: one rectangle in
    each of its millings."""
    return isinstance(config, VolumeFIBMillingTaskConfig) and all(
        len(getattr(config, name).stages) == 1
        and isinstance(getattr(config, name).stages[0].pattern, RectanglePattern)
        for name in MILLINGS
    )


def preview_milling(config: VolumeFIBMillingTaskConfig) -> PlacedMilling:
    """What *config* mills on a position of ``PREVIEW_SLICES`` slices whose
    point of interest is the image centre, none cut yet."""
    position = VolumePosition(name="preview", iterations=PREVIEW_SLICES)
    return whole_volume(
        slice_progress(config, position, 0), setup_millings(config, position)
    )


def _key(milling: Optional[str], name: str) -> str:
    return f"{milling}.{name}" if milling else name


def _spec(milling: Optional[str], name: str):
    if milling is None:
        return VolumeFIBMillingTaskConfig.params[name]
    if name in PATTERN_FIELDS:
        return RectanglePattern.params[name]
    if name == "milling_current":
        return FibsemMillingSettings.params[name]
    return FibsemMillingTaskConfig.params[name]


def _value(config: VolumeFIBMillingTaskConfig, milling: Optional[str], name: str):
    if milling is None:
        return getattr(config, name)
    task = getattr(config, milling)
    if name in PATTERN_FIELDS:
        return getattr(task.stages[0].pattern, name)
    if name == "milling_current":
        return task.stages[0].milling.milling_current
    return getattr(task, name)


class VolumeMillingForm(QWidget):
    """A group each for the clearing trench, the redeposition lanes, the start
    slice and the slices."""

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

        self.warning = QLabel()
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet(f"color: {SEMANTIC_WARNING_COLOR};")
        fixed = QLabel(
            "Each piece is placed from the point of interest, the start slice's "
            "centre, and scans the way the slices go. A slice is as wide and deep "
            "as the start slice; the lanes run the slices' whole length."
        )
        fixed.setWordWrap(True)
        fixed.setStyleSheet(stylesheets.LABEL_INSTRUCTIONS_STYLE)
        self.all_settings_button = QPushButton("All milling settings…")
        self.all_settings_button.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._forms: Dict[str, QFormLayout] = {}
        for title, _rows in GROUPS:
            box = QWidget()
            column = QVBoxLayout(box)
            self._forms[title] = QFormLayout()
            column.addLayout(self._forms[title])
            if title == "Slices":
                column.addWidget(self.warning)
            layout.addWidget(TitledPanel(title, content=box, collapsible=False))
        layout.addWidget(fixed)
        layout.addWidget(self.all_settings_button)
        self.all_settings_button.clicked.connect(self.all_settings_clicked)

    def set_microscope(self, microscope: Optional["FibsemMicroscope"]) -> None:
        self.microscope = microscope

    def _current_choices(self, name: str) -> List[Any]:
        if self.microscope is None:
            return []
        return beam_choices(self.microscope, name, BeamType.ION)

    def show_config(self, config: VolumeFIBMillingTaskConfig) -> None:
        """Rebuild the controls from a copy of *config*, one volume milling
        config: an edit changes the config only through the write."""
        config = deepcopy(config)
        self._config = config
        self._controls = {}
        for title, rows in GROUPS:
            form = self._forms[title]
            while form.rowCount():
                form.removeRow(0)
            for milling, name in rows:
                spec = _spec(milling, name)
                control = build_control(
                    spec,
                    _value(config, milling, name),
                    resolve_choices=self._current_choices,
                )
                label = QLabel((spec.label or name).capitalize())
                label.setToolTip(spec.doc or "")
                form.addRow(label, control.widget)
                key = _key(milling, name)
                control.connect(self._edit_handler(milling, name))
                self._controls[key] = control
        self._show_warning()

    def value(self, key: str) -> Any:
        return self._controls[key].read()

    def _edit_handler(self, milling: Optional[str], name: str) -> Callable[[], None]:
        return lambda: self._on_edit(milling, name)

    def _on_edit(self, milling: Optional[str], name: str) -> None:
        config = self._config
        if config is None:
            return
        value = self._controls[_key(milling, name)].read()
        if milling is None:
            setattr(config, name, value)
            self.values_changed.emit({name: value})
        else:
            task = deepcopy(getattr(config, milling))
            stage = task.stages[0]
            if name in PATTERN_FIELDS:
                setattr(stage.pattern, name, value)
            elif name == "milling_current":
                stage.milling.milling_current = value
            else:
                setattr(task, name, value)
            setattr(config, milling, task)
            self.values_changed.emit({milling: task})
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


def _extent(milling: FibsemMillingTaskConfig) -> float:
    """The frame width that holds every stage, centred on the image centre."""
    half_w = half_h = 0.0
    for stage in milling.stages:
        p = stage.pattern
        half_w = max(half_w, abs(p.point.x) + p.width / 2)
        half_h = max(half_h, abs(p.point.y) + p.height / 2)
    return 2 * max(half_w, half_h * 1.5) * PREVIEW_MARGIN


class VolumePreview(QWidget):
    """The volume milling around a point of interest at the centre of a blank
    FIB frame: the setup's pieces and ``PREVIEW_SLICES`` slices."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.view_controller = MicroscopeViewController(
            parent=self, view=ItemEditorView()
        )
        self.view_controller.widget.show_beams()
        self.caption = QLabel()
        self.caption.setStyleSheet(stylesheets.LABEL_INSTRUCTIONS_STYLE)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view_controller.widget, 1)
        layout.addWidget(self.caption)

    def show_config(self, config: VolumeFIBMillingTaskConfig) -> None:
        placed = preview_milling(config)
        hfw = _extent(placed.milling)
        image = FibsemImage.generate_blank_image(hfw=hfw)
        self.view_controller.set_image(BeamType.ION, image)
        self.view_controller.set_overlay(
            BeamType.ION,
            MillingSpec(
                id=PREVIEW_OVERLAY_ID,
                stages=placed.milling.stages,
                stage_styles=placed.styles,
                stage_colours=placed.colours,
            ),
        )
        self.caption.setText(
            f"For {PREVIEW_SLICES} slices; a position sets its own count · "
            f"HFW {hfw * 1e6:.1f} µm"
        )
