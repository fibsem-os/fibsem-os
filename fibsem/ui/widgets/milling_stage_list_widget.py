from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QIconifyIcon

from fibsem.milling.base import FibsemMillingStage
from fibsem.ui import stylesheets
from fibsem.ui.napari.patterns import COLOURS
from fibsem.utils import format_value

_NAME_MIN_WIDTH = 140
_PATTERN_MIN_WIDTH = 100
_DEPTH_MIN_WIDTH = 80
_CURRENT_MIN_WIDTH = 80
_STRATEGY_MIN_WIDTH = 100
_BTN_SIZE = QSize(28, 28)
_BTN_SPACER_WIDTH = _BTN_SIZE.width() * 2 + 8  # color + remove

_SELECTED_BG = "#2d3f5c"
_NORMAL_BG = "transparent"


def _make_color_icon(color_name: str, size: int = 16) -> QIcon:
    px = QPixmap(size, size)
    px.fill(QColor(color_name))
    return QIcon(px)


def _format_depth(stage: FibsemMillingStage) -> str:
    if hasattr(stage.pattern, "depth") and stage.pattern.depth is not None:
        return format_value(stage.pattern.depth, unit="m", precision=1)
    return "—"


def _format_current(stage: FibsemMillingStage) -> str:
    return format_value(stage.milling.milling_current, unit="A", precision=1)


class _DraggableStageList(QListWidget):
    """QListWidget with InternalMove drag-and-drop that emits the new stage order after each drop.

    Qt clears itemWidget associations when items are moved, so the parent must
    listen to ``reordered`` and rebuild the row widgets.
    """

    reordered = pyqtSignal(list)  # List[FibsemMillingStage]

    def dropEvent(self, event) -> None:
        super().dropEvent(event)
        stages = [
            self.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.count())
            if self.item(i).data(Qt.ItemDataRole.UserRole) is not None
        ]
        self.reordered.emit(stages)


class MillingStageRowWidget(QWidget):
    enabled_changed = pyqtSignal(object, bool)   # FibsemMillingStage, enabled
    remove_clicked = pyqtSignal(object)          # FibsemMillingStage
    row_clicked = pyqtSignal(object)             # FibsemMillingStage

    def __init__(
        self,
        stage: FibsemMillingStage,
        index: int,
        enabled: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.stage = stage
        self.index = index

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(8)

        self.checkbox = QCheckBox()
        self.checkbox.setChecked(enabled)
        self.checkbox.setToolTip("Enable/disable stage")
        layout.addWidget(self.checkbox)

        self.name_label = QLabel()
        self.name_label.setMinimumWidth(_NAME_MIN_WIDTH)
        layout.addWidget(self.name_label)

        self.pattern_label = QLabel()
        self.pattern_label.setMinimumWidth(_PATTERN_MIN_WIDTH)
        layout.addWidget(self.pattern_label)

        self.depth_label = QLabel()
        self.depth_label.setMinimumWidth(_DEPTH_MIN_WIDTH)
        layout.addWidget(self.depth_label)

        self.current_label = QLabel()
        self.current_label.setMinimumWidth(_CURRENT_MIN_WIDTH)
        layout.addWidget(self.current_label)

        self.strategy_label = QLabel()
        self.strategy_label.setMinimumWidth(_STRATEGY_MIN_WIDTH)
        layout.addWidget(self.strategy_label, 1)

        self.btn_color = QToolButton()
        self.btn_color.setFixedSize(_BTN_SIZE)
        self.btn_color.setStyleSheet(stylesheets.TOOLBUTTON_ICON_STYLESHEET)
        self.btn_color.setToolTip("Stage color")
        layout.addWidget(self.btn_color)

        self.btn_remove = QToolButton()
        self.btn_remove.setIcon(QIconifyIcon("mdi:trash-can-outline", color=stylesheets.GRAY_ICON_COLOR))
        self.btn_remove.setToolTip("Remove stage")
        self.btn_remove.setFixedSize(_BTN_SIZE)
        self.btn_remove.setStyleSheet(stylesheets.TOOLBUTTON_ICON_STYLESHEET)
        layout.addWidget(self.btn_remove)

        self.checkbox.stateChanged.connect(
            lambda s: self.enabled_changed.emit(self.stage, bool(s))
        )
        self.btn_remove.clicked.connect(lambda: self.remove_clicked.emit(self.stage))

        self.refresh()

    def mousePressEvent(self, event) -> None:
        self.row_clicked.emit(self.stage)
        super().mousePressEvent(event)

    def set_selected(self, selected: bool) -> None:
        bg = _SELECTED_BG if selected else _NORMAL_BG
        self.setStyleSheet(f"background-color: {bg};")

    def refresh(self) -> None:
        self.name_label.setText(self.stage.name)
        self.pattern_label.setText(self.stage.pattern.name)
        self.depth_label.setText(_format_depth(self.stage))
        self.current_label.setText(_format_current(self.stage))
        self.strategy_label.setText(self.stage.strategy.name)
        color = COLOURS[self.index % len(COLOURS)]
        self.btn_color.setIcon(_make_color_icon(color))


class _MillingStageListHeader(QWidget):
    select_all_changed = pyqtSignal(bool)
    add_clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: #1e2124;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)

        self.checkbox_all = QCheckBox("Stage")
        self.checkbox_all.setChecked(True)
        self.checkbox_all.setStyleSheet("font-weight: bold;")
        self.checkbox_all.setMinimumWidth(24 + 8 + _NAME_MIN_WIDTH)
        layout.addWidget(self.checkbox_all)

        for label_text, min_width in [
            ("Pattern", _PATTERN_MIN_WIDTH),
            ("Depth", _DEPTH_MIN_WIDTH),
            ("Current", _CURRENT_MIN_WIDTH),
            ("Strategy", _STRATEGY_MIN_WIDTH),
        ]:
            lbl = QLabel(label_text)
            lbl.setStyleSheet("font-weight: bold;")
            lbl.setMinimumWidth(min_width)
            layout.addWidget(lbl)

        layout.addStretch(1)

        # spacer covers color button; btn_add aligns with remove button
        spacer = QWidget()
        spacer.setFixedWidth(_BTN_SPACER_WIDTH - _BTN_SIZE.width() - 8)
        layout.addWidget(spacer)

        self.btn_add = QToolButton()
        self.btn_add.setIcon(QIconifyIcon("mdi:plus", color=stylesheets.GRAY_ICON_COLOR))
        self.btn_add.setToolTip("Add milling stage")
        self.btn_add.setFixedSize(_BTN_SIZE)
        self.btn_add.setStyleSheet(stylesheets.TOOLBUTTON_ICON_STYLESHEET)
        layout.addWidget(self.btn_add)

        self.checkbox_all.stateChanged.connect(
            lambda s: self.select_all_changed.emit(bool(s))
        )
        self.btn_add.clicked.connect(self.add_clicked)


class MillingStageListWidget(QWidget):
    """Multi-column list widget for FibsemMillingStage objects."""

    stage_selected = pyqtSignal(object)    # FibsemMillingStage
    stage_added = pyqtSignal(object)       # FibsemMillingStage (new stage, distinct from selection)
    stage_removed = pyqtSignal(object)     # FibsemMillingStage
    enabled_changed = pyqtSignal(list)     # List[FibsemMillingStage] (enabled only)
    order_changed = pyqtSignal(list)       # List[FibsemMillingStage] in new order

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._checked: Dict[int, bool] = {}   # id(stage) -> bool
        self._selected_stage: Optional[FibsemMillingStage] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._header = _MillingStageListHeader()
        layout.addWidget(self._header)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #3a3d42;")
        layout.addWidget(sep)

        self._list = _DraggableStageList()
        self._list.setDragDropMode(QAbstractItemView.InternalMove)
        self._list.setDefaultDropAction(Qt.MoveAction)
        self._list.setSpacing(1)
        self._list.setStyleSheet(stylesheets.LIST_WIDGET_STYLESHEET)
        self._list.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(self._list)

        self._empty_label = QLabel("No milling stages. Click + to add one.")
        self._empty_label.setStyleSheet("color: #606060; font-style: italic; padding: 12px;")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._empty_label)

        self._header.select_all_changed.connect(self._on_select_all)
        self._header.add_clicked.connect(self._on_add_stage)
        self._list.reordered.connect(self._on_reordered)
        self._update_empty_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_stages(self, stages: List[FibsemMillingStage]) -> None:
        self._list.clear()
        self._checked.clear()
        self._selected_stage = None
        for stage in stages:
            self.add_stage(stage)
        self._update_empty_state()

    def add_stage(
        self,
        stage: FibsemMillingStage,
        enabled: bool = True,
    ) -> MillingStageRowWidget:
        index = self._list.count()
        self._checked[id(stage)] = enabled
        row = MillingStageRowWidget(stage, index=index, enabled=enabled)
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, stage)
        item.setSizeHint(row.sizeHint())
        self._list.addItem(item)
        self._list.setItemWidget(item, row)
        self._connect_row(row)
        self._sync_select_all()
        self._update_empty_state()
        return row

    def remove_stage(self, stage: FibsemMillingStage) -> None:
        for i in range(self._list.count()):
            if self._row(i).stage is stage:
                self._list.takeItem(i)
                self._checked.pop(id(stage), None)
                break
        if self._selected_stage is stage:
            self._selected_stage = None
        self._reindex_rows()
        self._sync_select_all()
        self._update_empty_state()

    def get_stages(self) -> List[FibsemMillingStage]:
        """Return all stages in current display order."""
        return [self._row(i).stage for i in range(self._list.count())]

    def get_enabled_stages(self) -> List[FibsemMillingStage]:
        """Return only enabled (checked) stages in display order."""
        return [
            self._row(i).stage
            for i in range(self._list.count())
            if self._row(i).checkbox.isChecked()
        ]

    def refresh_stage(self, stage: FibsemMillingStage) -> None:
        for i in range(self._list.count()):
            row = self._row(i)
            if row.stage is stage:
                row.refresh()
                break

    def refresh_all(self) -> None:
        for i in range(self._list.count()):
            self._row(i).refresh()

    def select_stage(self, stage: FibsemMillingStage) -> None:
        self._set_selected(stage)
        self.stage_selected.emit(stage)

    def clear(self) -> None:
        self._list.clear()
        self._checked.clear()
        self._selected_stage = None
        self._update_empty_state()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_empty_state(self) -> None:
        empty = self._list.count() == 0
        self._empty_label.setVisible(empty)

    def _on_add_stage(self) -> None:
        count = self._list.count()
        source = self._selected_stage
        if source is None and count > 0:
            source = self._row(count - 1).stage
        if source is not None:
            new_stage = deepcopy(source)
        else:
            new_stage = FibsemMillingStage()
        new_stage.name = f"Milling Stage {count + 1}"
        self.add_stage(new_stage, enabled=True)
        self._set_selected(new_stage)
        self.stage_added.emit(new_stage)
        self.stage_selected.emit(new_stage)

    def _reindex_rows(self) -> None:
        for i in range(self._list.count()):
            row = self._row(i)
            row.index = i
            row.refresh()

    def _connect_row(self, row: MillingStageRowWidget) -> None:
        row.enabled_changed.connect(self._on_enabled_changed)
        row.remove_clicked.connect(self._on_remove_clicked)
        row.row_clicked.connect(self._on_row_clicked)

    def _row(self, i: int) -> MillingStageRowWidget:
        return self._list.itemWidget(self._list.item(i))  # type: ignore[return-value]

    def _set_selected(self, stage: Optional[FibsemMillingStage]) -> None:
        for i in range(self._list.count()):
            row = self._row(i)
            row.set_selected(row.stage is stage)
        self._selected_stage = stage

    def _on_reordered(self, stages: List[FibsemMillingStage]) -> None:
        """Rebuild row widgets after drag-and-drop (Qt clears itemWidget on move)."""
        for i, stage in enumerate(stages):
            item = self._list.item(i)
            if item is None:
                continue
            enabled = self._checked.get(id(stage), True)
            row = MillingStageRowWidget(stage, index=i, enabled=enabled)
            item.setSizeHint(row.sizeHint())
            self._list.setItemWidget(item, row)
            self._connect_row(row)
            if stage is self._selected_stage:
                row.set_selected(True)
        self._sync_select_all()
        self.order_changed.emit(stages)

    def _on_row_clicked(self, stage: FibsemMillingStage) -> None:
        self._set_selected(stage)
        self.stage_selected.emit(stage)

    def _on_remove_clicked(self, stage: FibsemMillingStage) -> None:
        self.remove_stage(stage)
        self.stage_removed.emit(stage)

    def _on_enabled_changed(self, stage: FibsemMillingStage, enabled: bool) -> None:
        self._checked[id(stage)] = enabled
        self._sync_select_all()
        self.enabled_changed.emit(self.get_enabled_stages())

    def _on_select_all(self, checked: bool) -> None:
        for i in range(self._list.count()):
            row = self._row(i)
            row.checkbox.blockSignals(True)
            row.checkbox.setChecked(checked)
            row.checkbox.blockSignals(False)
            self._checked[id(row.stage)] = checked
        self.enabled_changed.emit(self.get_enabled_stages())

    def _sync_select_all(self) -> None:
        count = self._list.count()
        if count == 0:
            return
        n_checked = sum(self._row(i).checkbox.isChecked() for i in range(count))
        cb = self._header.checkbox_all
        cb.blockSignals(True)
        if n_checked == 0:
            cb.setCheckState(Qt.Unchecked)
        elif n_checked == count:
            cb.setCheckState(Qt.Checked)
        else:
            cb.setTristate(True)
            cb.setCheckState(Qt.PartiallyChecked)
        cb.blockSignals(False)
