from __future__ import annotations

import os
from copy import deepcopy
from typing import Dict, List, Optional

from PyQt5.QtCore import QEvent, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from fibsem.milling.base import FibsemMillingStage, get_strategy
from fibsem.milling.patterning import get_pattern, get_pattern_names
from fibsem.milling.strategy import get_strategy_names
from fibsem.ui import stylesheets
from fibsem.ui.napari.patterns import COLOURS
from fibsem.ui.widgets.custom_widgets import IconToolButton, ValueComboBox, ValueSpinBox

_DRAG_HANDLE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "icons", "drag_handle.svg")
_NAME_MIN_WIDTH = 120
_PATTERN_FIXED_WIDTH = 110
_DEPTH_FIXED_WIDTH = 112
_CURRENT_FIXED_WIDTH = 100
_STRATEGY_FIXED_WIDTH = 110
_BTN_SIZE = QSize(32, 32)
_ROW_HEIGHT = 40
_BTN_SPACER_WIDTH = _BTN_SIZE.width() * 2 + 8  # color + remove


_SI_TO_MICRO = 1e6
_MICRO_TO_SI = 1e-6

_NAME_EDIT_STYLE = (
    "QLineEdit { background: transparent; border: none; }"
    "QLineEdit:focus { background: #1e2124; border: 1px solid #555; border-radius: 2px; }"
)


def _make_color_icon(color_name: str, size: int = 16) -> QIcon:
    px = QPixmap(size, size)
    px.fill(QColor(color_name))
    return QIcon(px)


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
    stage_changed = pyqtSignal(object)           # FibsemMillingStage after inline mutation

    def __init__(
        self,
        stage: FibsemMillingStage,
        index: int,
        pattern_names: List[str],
        strategy_names: List[str],
        current_values: Optional[List[float]] = None,
        enabled: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.stage = stage
        self.index = index
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._pattern_names = pattern_names
        self._strategy_names = strategy_names
        self._current_values = current_values or []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(4)

        self.checkbox = QCheckBox()
        self.checkbox.setChecked(enabled)
        self.checkbox.setToolTip("Enable/disable stage")
        self.checkbox.setStyleSheet("background: transparent;")
        layout.addWidget(self.checkbox)

        self.name_edit = QLineEdit()
        self.name_edit.setMinimumWidth(_NAME_MIN_WIDTH)
        self.name_edit.setStyleSheet(_NAME_EDIT_STYLE)
        self.name_edit.setToolTip("Stage name")
        layout.addWidget(self.name_edit, 1)

        self.pattern_combo = ValueComboBox(items=pattern_names)
        self.pattern_combo.setFixedWidth(_PATTERN_FIXED_WIDTH)
        self.pattern_combo.setToolTip("Pattern type")
        layout.addWidget(self.pattern_combo)

        self.depth_spin = ValueSpinBox(suffix="µm", minimum=0.01, maximum=1000.0, step=0.1, decimals=1)
        self.depth_spin.setFixedWidth(_DEPTH_FIXED_WIDTH)
        self.depth_spin.setToolTip("Depth (µm)")
        layout.addWidget(self.depth_spin)

        _current_items = self._current_values if self._current_values else [stage.milling.milling_current]
        self.current_combo = ValueComboBox(items=_current_items, unit="A")
        self.current_combo.setFixedWidth(_CURRENT_FIXED_WIDTH)
        self.current_combo.setToolTip("Milling current")
        layout.addWidget(self.current_combo)

        self.strategy_combo = ValueComboBox(items=strategy_names)
        self.strategy_combo.setFixedWidth(_STRATEGY_FIXED_WIDTH)
        self.strategy_combo.setToolTip("Strategy")
        layout.addWidget(self.strategy_combo)

        self.btn_color = QToolButton()
        self.btn_color.setFixedSize(_BTN_SIZE)
        self.btn_color.setStyleSheet(stylesheets.TOOLBUTTON_ICON_STYLESHEET)
        self.btn_color.setToolTip("Stage color")
        layout.addWidget(self.btn_color)

        self.btn_remove = IconToolButton(
            icon="mdi:trash-can-outline", tooltip="Remove stage", size=_BTN_SIZE.width()
        )
        layout.addWidget(self.btn_remove)

        drag_icon = QLabel()
        drag_icon.setFixedSize(10, 16)
        drag_icon.setPixmap(QPixmap(_DRAG_HANDLE_PATH).scaled(10, 16, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        drag_icon.setStyleSheet("background: transparent;")
        drag_icon.setCursor(Qt.CursorShape.OpenHandCursor)
        layout.addWidget(drag_icon)

        self.checkbox.stateChanged.connect(
            lambda s: self.enabled_changed.emit(self.stage, bool(s))
        )
        self.btn_remove.clicked.connect(lambda: self.remove_clicked.emit(self.stage))

        for w in (self.name_edit, self.pattern_combo, self.depth_spin,
                  self.current_combo, self.strategy_combo):
            w.installEventFilter(self)

        self._connect_signals()
        self.refresh()

    # ------------------------------------------------------------------
    # Signal connections
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        self.name_edit.editingFinished.connect(self._on_name_changed)
        self.pattern_combo.currentIndexChanged.connect(self._on_pattern_type_changed)
        self.depth_spin.editingFinished.connect(self._on_depth_changed)
        self.current_combo.currentIndexChanged.connect(self._on_current_changed)
        self.strategy_combo.currentIndexChanged.connect(self._on_strategy_changed)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _block_controls(self, block: bool) -> None:
        for w in (self.name_edit, self.pattern_combo, self.depth_spin,
                  self.current_combo, self.strategy_combo):
            w.blockSignals(block)

    def _update_depth_visibility(self) -> None:
        has_depth = hasattr(self.stage.pattern, "depth") and self.stage.pattern.depth is not None
        self.depth_spin.setVisible(has_depth)
        self.depth_spin.setEnabled(has_depth)

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        child = self.childAt(event.pos())
        if child is None or child is self:
            self.row_clicked.emit(self.stage)
        super().mousePressEvent(event)

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.FocusIn:
            self.row_clicked.emit(self.stage)
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_selected(self, selected: bool) -> None:
        pass  # Selection visuals handled by QListWidget item selection

    def refresh(self) -> None:
        self._block_controls(True)
        self.name_edit.setText(self.stage.name)
        self.name_edit.setToolTip(self.stage.summary)
        self.pattern_combo.set_value(self.stage.pattern.name)
        self.depth_spin.setValue(self.stage.pattern.depth * _SI_TO_MICRO)
        self.current_combo.set_value(self.stage.milling.milling_current)
        self.strategy_combo.set_value(self.stage.strategy.name)
        self._update_depth_visibility()
        color = COLOURS[self.index % len(COLOURS)]
        self.btn_color.setIcon(_make_color_icon(color))
        self._block_controls(False)

    # ------------------------------------------------------------------
    # Inline mutation handlers
    # ------------------------------------------------------------------

    def _on_name_changed(self) -> None:
        text = self.name_edit.text().strip()
        if not text:
            self.name_edit.setText(self.stage.name)  # revert empty input
            return
        if text == self.stage.name:
            return
        self.stage.name = text
        self.stage_changed.emit(self.stage)

    def _on_pattern_type_changed(self) -> None:
        new_name = self.pattern_combo.value()
        if new_name is None or new_name == self.stage.pattern.name:
            return
        old_depth = getattr(self.stage.pattern, "depth", None)
        new_pattern = get_pattern(new_name)
        if old_depth is not None and hasattr(new_pattern, "depth"):
            new_pattern.depth = old_depth
        self.stage.pattern = new_pattern
        self._update_depth_visibility()
        self.stage_changed.emit(self.stage)

    def _on_depth_changed(self) -> None:
        if not hasattr(self.stage.pattern, "depth"):
            return
        new_depth = self.depth_spin.value() * _MICRO_TO_SI
        if new_depth == self.stage.pattern.depth:
            return
        self.stage.pattern.depth = new_depth
        self.stage_changed.emit(self.stage)

    def _on_current_changed(self) -> None:
        new_current = self.current_combo.value()
        if new_current is None or new_current == self.stage.milling.milling_current:
            return
        self.stage.milling.milling_current = new_current
        self.stage_changed.emit(self.stage)

    def _on_strategy_changed(self) -> None:
        new_name = self.strategy_combo.value()
        if new_name is None or new_name == self.stage.strategy.name:
            return
        self.stage.strategy = get_strategy(new_name)
        self.stage_changed.emit(self.stage)


class _MillingStageListHeader(QWidget):
    select_all_changed = pyqtSignal(bool)
    add_clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: #1e2124;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        self.checkbox_all = QCheckBox("Stage")
        self.checkbox_all.setChecked(True)
        self.checkbox_all.setStyleSheet("font-weight: bold; background: transparent;")
        self.checkbox_all.setMinimumWidth(24 + 8 + _NAME_MIN_WIDTH)
        layout.addWidget(self.checkbox_all, 1)

        self.lbl_pattern = QLabel("Pattern")
        self.lbl_pattern.setStyleSheet("font-weight: bold; background: transparent;")
        self.lbl_pattern.setFixedWidth(_PATTERN_FIXED_WIDTH)
        layout.addWidget(self.lbl_pattern)

        for label_text, fixed_width in [
            ("Depth", _DEPTH_FIXED_WIDTH),
            ("Current", _CURRENT_FIXED_WIDTH),
            ("Strategy", _STRATEGY_FIXED_WIDTH),
        ]:
            lbl = QLabel(label_text)
            lbl.setStyleSheet("font-weight: bold; background: transparent;")
            lbl.setFixedWidth(fixed_width)
            layout.addWidget(lbl)

        # spacer covers color button; btn_add aligns with remove button
        spacer = QWidget()
        spacer.setFixedWidth(_BTN_SPACER_WIDTH - _BTN_SIZE.width() - 8)
        spacer.setStyleSheet("background: transparent;")
        layout.addWidget(spacer)

        self.btn_add = IconToolButton(
            icon="mdi:plus", tooltip="Add milling stage", size=_BTN_SIZE.width()
        )
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
    stage_changed = pyqtSignal(object)     # FibsemMillingStage (inline field edit)
    enabled_changed = pyqtSignal(list)     # List[FibsemMillingStage] (enabled only)
    order_changed = pyqtSignal(list)       # List[FibsemMillingStage] in new order

    def __init__(self, current_values: Optional[List[float]] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._checked: Dict[int, bool] = {}   # id(stage) -> bool
        self._selected_stage: Optional[FibsemMillingStage] = None
        self._pending_stage: Optional[FibsemMillingStage] = None
        self._stage_change_pending: bool = False
        self._pattern_names: List[str] = get_pattern_names()
        self._strategy_names: List[str] = get_strategy_names()
        self._current_values: List[float] = current_values or []
        self._show_pattern: bool = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._header = _MillingStageListHeader()
        self._header.lbl_pattern.setVisible(self._show_pattern)
        layout.addWidget(self._header)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #3a3d42;")
        layout.addWidget(sep)

        self._list = _DraggableStageList()
        self._list.setDragDropMode(QAbstractItemView.InternalMove)
        self._list.setDefaultDropAction(Qt.MoveAction)
        self._list.setSpacing(0)
        self._list.setMinimumHeight(3 * _ROW_HEIGHT)
        self._list.setStyleSheet(stylesheets.LIST_WIDGET_STYLESHEET)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
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
        row = MillingStageRowWidget(
            stage, index=index,
            pattern_names=self._pattern_names,
            strategy_names=self._strategy_names,
            current_values=self._current_values,
            enabled=enabled,
        )
        row.pattern_combo.setVisible(self._show_pattern)
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, stage)
        item.setSizeHint(QSize(0, _ROW_HEIGHT))
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

    def set_pattern_column_visible(self, visible: bool) -> None:
        """Show or hide the Pattern column in the header and all rows."""
        self._show_pattern = visible
        self._header.lbl_pattern.setVisible(visible)
        for i in range(self._list.count()):
            self._row(i).pattern_combo.setVisible(visible)

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
        row.stage_changed.connect(self._on_row_stage_changed, type=Qt.QueuedConnection)

    def _row(self, i: int) -> MillingStageRowWidget:
        return self._list.itemWidget(self._list.item(i))  # type: ignore[return-value]

    def _set_selected(self, stage: Optional[FibsemMillingStage]) -> None:
        self._selected_stage = stage
        for i in range(self._list.count()):
            row = self._row(i)
            if row.stage is stage:
                self._list.setCurrentRow(i)
                return
        self._list.setCurrentRow(-1)

    def _on_reordered(self, stages: List[FibsemMillingStage]) -> None:
        """Rebuild row widgets after drag-and-drop (Qt clears itemWidget on move)."""
        for i, stage in enumerate(stages):
            item = self._list.item(i)
            if item is None:
                continue
            enabled = self._checked.get(id(stage), True)
            row = MillingStageRowWidget(
                stage, index=i,
                pattern_names=self._pattern_names,
                strategy_names=self._strategy_names,
                current_values=self._current_values,
                enabled=enabled,
            )
            row.pattern_combo.setVisible(self._show_pattern)
            item.setSizeHint(QSize(0, _ROW_HEIGHT))
            self._list.setItemWidget(item, row)
            self._connect_row(row)
            if stage is self._selected_stage:
                row.set_selected(True)
        self._sync_select_all()
        self.order_changed.emit(stages)

    def _on_row_clicked(self, stage: FibsemMillingStage) -> None:
        self._set_selected(stage)
        self.stage_selected.emit(stage)

    def _on_row_stage_changed(self, stage: FibsemMillingStage) -> None:
        # Defer inline row mutation propagation to avoid re-entrant redraw paths.
        self._pending_stage = stage
        if self._stage_change_pending:
            return
        self._stage_change_pending = True
        QTimer.singleShot(0, self._flush_pending_stage_change)

    def _flush_pending_stage_change(self) -> None:
        self._stage_change_pending = False
        stage = self._pending_stage
        self._pending_stage = None
        if stage is not None:
            self.stage_changed.emit(stage)

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
