from __future__ import annotations

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from difflib import get_close_matches
from difflib import get_close_matches
from enum import Enum
from typing import Any, List, Optional, Union

from PyQt5.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QCursor, QIcon, QTransform
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QAction,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QIconifyIcon

from fibsem.ui import stylesheets as stylesheets
from fibsem.ui.utils import WheelBlocker
from fibsem.utils import format_value


class QFilePathLineEdit(QWidget):
    textChanged = pyqtSignal(str)
    editingFinished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        self.lineEdit = QLineEdit(self)
        self.button_browse = QToolButton(self)
        self.button_browse.setText("...")
        self.button_browse.setMaximumWidth(80)
        layout.addWidget(self.lineEdit)
        layout.addWidget(self.button_browse)

        self.setContentsMargins(0, 0, 0, 0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.button_browse.clicked.connect(self.browse_file)
        self.lineEdit.textChanged.connect(self.textChanged.emit)
        self.lineEdit.editingFinished.connect(self.editingFinished.emit)

    def browse_file(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.lineEdit.setText(selected_files[0])
                self.textChanged.emit(selected_files[0])
                self.editingFinished.emit()

    def text(self) -> str:
        return self.lineEdit.text()
    
    def setText(self, text: str) -> None:
        self.lineEdit.setText(text)


class QDirectoryLineEdit(QWidget):
    textChanged = pyqtSignal(str)
    editingFinished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        self.lineEdit = QLineEdit(self)
        self.button_browse = QToolButton(self)
        self.button_browse.setText("...")
        self.button_browse.setMaximumWidth(80)
        layout.addWidget(self.lineEdit)
        layout.addWidget(self.button_browse)

        self.setContentsMargins(0, 0, 0, 0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.button_browse.clicked.connect(self.browse_directory)
        self.lineEdit.textChanged.connect(self.textChanged.emit)
        self.lineEdit.editingFinished.connect(self.editingFinished.emit)

    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory", self.lineEdit.text())
        if directory:
            self.lineEdit.setText(directory)
            self.textChanged.emit(directory)
            self.editingFinished.emit()

    def text(self) -> str:
        return self.lineEdit.text()

    def setText(self, text: str) -> None:
        self.lineEdit.setText(text)


def _find_closest_string_match(value: str, items: list) -> any:
    """Find the closest matching string value from a list of items using fuzzy matching.
    
    Args:
        value: The value to match
        items: List of items to search through
        
    Returns:
        The closest matching item from the list
    """
    if not items:
        return value
    
    str_value = str(value).lower()
    str_items = [str(item).lower() for item in items]
    
    # Try exact match (case-insensitive)
    for item, str_item in zip(items, str_items):
        if str_item == str_value:
            return item
    
    # Try fuzzy matching with difflib
    matches = get_close_matches(str_value, str_items, n=1, cutoff=0.6)
    if matches:
        idx = str_items.index(matches[0])
        return items[idx]
    
    # Try substring match (bidirectional)
    for item, str_item in zip(items, str_items):
        if str_value in str_item or str_item in str_value:
            return item
    
    # No match found
    raise ValueError(f"No matching item found for value '{value}' in items {items}")


def _create_combobox_control(value: Union[str, int, float, Enum], 
                             items: list, 
                             units: Optional[str], 
                             format_fn: Optional[Callable] = None, 
                             control: Optional[QComboBox] = None) -> QComboBox:
    """Create a QComboBox control for selecting from a list of items."""
    if control is None:
        control = QComboBox()
    for item in items:
        if isinstance(item, (float, int)):
            item_str = format_value(val=item, unit=units, precision=1)
        elif isinstance(item, Enum):
            item_str = item.name # TODO: migrate to QEnumComboBox
        elif format_fn is not None:
            item_str = format_fn(item)
        else:
            item_str = str(item)
        control.addItem(item_str, item)

    if isinstance(value, tuple) and len(value) == 2:
        value = list(value)  # Convert tuple to list for easier handling

    # find the closest match to the current value
    idx = control.findData(value)
    if idx == -1:
        # get the closest value
        if items:
            if isinstance(value, (int, float)) and all(isinstance(x, (int, float)) for x in items):
                # numeric comparison
                closest_value = min(items, key=lambda x: abs(x - value))
            else:
                # string comparison - use fuzzy matching for better results
                closest_value = _find_closest_string_match(str(value), items)
        else:
            closest_value = value
        idx = control.findData(closest_value)
    if idx == -1:
        logging.debug(f"Warning: No matching item or nearest found for {items} with value {value}. Using first item.")
        idx = 0
    control.setCurrentIndex(idx)
    
    try:
        control.installEventFilter(WheelBlocker(parent=control))
    except (ImportError, AttributeError, NameError) as e:
        logging.warning(f"Failed to install WheelBlocker event filter: {e}")

    return control


class ValueComboBox(QComboBox):
    """QComboBox that stores raw values as item data and supports closest-match selection."""

    def __init__(
        self,
        items: list,
        value=None,
        unit: Optional[str] = None,
        format_fn: Optional[Callable] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        for item in items:
            if isinstance(item, (float, int)):
                item_str = format_value(val=item, unit=unit, precision=1)
            elif isinstance(item, Enum):
                item_str = item.name
            elif format_fn is not None:
                item_str = format_fn(item)
            else:
                item_str = str(item)
            self.addItem(item_str, item)
        self.installEventFilter(WheelBlocker(parent=self))
        if value is not None:
            self.set_value(value)

    def set_value(self, value) -> None:
        """Select the item matching value; falls back to closest numeric match."""
        idx = self.findData(value)
        if idx == -1 and self.count() > 0:
            items = [self.itemData(i) for i in range(self.count())]
            if items and isinstance(items[0], (int, float)):
                closest = min(items, key=lambda x: abs(x - value))
                idx = self.findData(closest)
        if idx != -1:
            self.setCurrentIndex(idx)

    def value(self):
        """Return the raw value stored as item data for the current selection."""
        return self.currentData()


class ValueSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox with sensible defaults, WheelBlocker, and None-safe configuration."""

    def __init__(
        self,
        suffix: Optional[str] = None,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        step: Optional[float] = None,
        decimals: Optional[int] = None,
        tooltip: Optional[str] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        if suffix:
            self.setSuffix(f" {suffix}")
        self.setRange(minimum if minimum is not None else 0.0, maximum if maximum is not None else 1e6)
        self.setSingleStep(step if step is not None else 0.01)
        self.setDecimals(decimals if decimals is not None else 3)
        if tooltip:
            self.setToolTip(tooltip)
        self.setKeyboardTracking(False)
        self.installEventFilter(WheelBlocker(parent=self))


@dataclass
class ContextMenuAction:
    """Represents a single action in a context menu."""
    label: str
    callback: Optional[Callable] = None
    icon: Optional[QIcon] = None
    tooltip: Optional[str] = None
    enabled: bool = True
    separator_after: bool = False
    data: Optional[Any] = None


@dataclass
class ContextMenuConfig:
    """Configuration for a context menu."""
    actions: list[ContextMenuAction] = field(default_factory=list)

    def add_action(
        self,
        label: str,
        callback: Optional[Callable] = None,
        icon: Optional[QIcon] = None,
        tooltip: Optional[str] = None,
        enabled: bool = True,
        separator_after: bool = False,
        data: Optional[Any] = None,
    ) -> "ContextMenuConfig":
        """Add an action to the menu configuration. Returns self for chaining."""
        self.actions.append(ContextMenuAction(
            label=label,
            callback=callback,
            icon=icon,
            tooltip=tooltip,
            enabled=enabled,
            separator_after=separator_after,
            data=data,
        ))
        return self

    def add_separator(self) -> "ContextMenuConfig":
        """Mark the previous action to have a separator after it."""
        if self.actions:
            self.actions[-1].separator_after = True
        return self


class ContextMenu(QMenu):
    """A reusable context menu widget.

    Usage:
        # Simple usage with callbacks
        config = ContextMenuConfig()
        config.add_action("Set Point of Interest", callback=self.set_poi)
        config.add_action("Move Patterns", callback=self.move_patterns)

        menu = ContextMenu(config, parent=self)
        menu.show_at_cursor()

        # Or pass context data to callbacks
        config = ContextMenuConfig()
        config.add_action("Edit", callback=lambda: self.edit(item), data=item)

        menu = ContextMenu(config, parent=self)
        selected = menu.show_at_cursor()  # Returns the selected ContextMenuAction or None
    """

    actionTriggered = pyqtSignal(object)  # Emits ContextMenuAction when triggered

    def __init__(self, config: ContextMenuConfig, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._config = config
        self._action_map: dict[QAction, ContextMenuAction] = {}
        self._build_menu()

    def _build_menu(self) -> None:
        """Build the menu from the configuration."""
        for menu_action in self._config.actions:
            if menu_action.icon:
                action = self.addAction(menu_action.icon, menu_action.label)
            else:
                action = self.addAction(menu_action.label)

            action.setEnabled(menu_action.enabled)
            if menu_action.tooltip:
                action.setToolTip(menu_action.tooltip)
            self._action_map[action] = menu_action

            if menu_action.separator_after:
                self.addSeparator()

    def show_at_cursor(self) -> Optional[ContextMenuAction]:
        """Show the menu at the current cursor position.

        Returns:
            The selected ContextMenuAction, or None if cancelled.
            If the action has a callback, it will be executed automatically.
        """
        selected_action = self.exec_(QCursor.pos())

        if selected_action is None:
            return None

        menu_action = self._action_map.get(selected_action)
        if menu_action:
            self.actionTriggered.emit(menu_action)
            if menu_action.callback:
                self._invoke_action_callback(menu_action)

        return menu_action

    def show_at_position(self, pos) -> Optional[ContextMenuAction]:
        """Show the menu at a specific position.

        Args:
            pos: QPoint position to show the menu at.

        Returns:
            The selected ContextMenuAction, or None if cancelled.
        """
        selected_action = self.exec_(pos)

        if selected_action is None:
            return None

        menu_action = self._action_map.get(selected_action)
        if menu_action:
            self.actionTriggered.emit(menu_action)
            if menu_action.callback:
                self._invoke_action_callback(menu_action)

        return menu_action

    def _invoke_action_callback(self, menu_action: ContextMenuAction) -> None:
        """Execute callback safely so one action failure does not break caller flow."""
        try:
            if menu_action.callback is not None:
                menu_action.callback()
        except Exception:
            logging.exception("Context menu action '%s' raised an exception.", menu_action.label)
            try:
                from napari.utils import notifications

                notifications.show_warning(f"Action '{menu_action.label}' failed.")
            except Exception:
                pass


def show_context_menu(
    actions: list[tuple[str, Callable]],
    parent: Optional[QWidget] = None,
) -> Optional[str]:
    """Convenience function to show a simple context menu.

    Args:
        actions: List of (label, callback) tuples.
        parent: Parent widget for the menu.

    Returns:
        The label of the selected action, or None if cancelled.

    Usage:
        result = show_context_menu([
            ("Set Point of Interest", self.set_poi),
            ("Move Patterns", self.move_patterns),
        ], parent=self)
    """
    config = ContextMenuConfig()
    for label, callback in actions:
        config.add_action(label, callback=callback)

    menu = ContextMenu(config, parent=parent)
    selected = menu.show_at_cursor()
    return selected.label if selected else None

@dataclass
class FormRow:
    """Shared form-row descriptor for metadata-driven settings widgets (milling, pattern, etc.)."""
    label: QLabel
    control: QWidget
    field: str
    advanced: bool
    scale: Optional[float]      # effective scale (base_scale ** dims); None = no scaling
    mfr: Optional[str] = None   # manufacturer filter; None = show for all


class TitledPanel(QWidget):
    """A styled panel with a dark header row (title label + optional widgets) and a collapsible content area.

    Usage::

        panel = TitledPanel("Milling", content=milling_widget)
        panel.add_header_widget(btn_advanced)   # right-aligned, before collapse button

        fixed = TitledPanel("Setup", content=setup_widget, collapsible=False)
    """

    def __init__(self, title: str, content: Optional[QWidget] = None,
                 collapsible: bool = True, parent=None) -> None:
        super().__init__(parent)
        self._collapsible = collapsible
        self.setObjectName("TitledPanel")
        self.setStyleSheet(
            "TitledPanel { border: 1px solid #3a3d42; border-radius: 4px; }"
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header
        self._header = QWidget()
        self._header.setStyleSheet("background: #1e2124; border-radius: 3px 3px 0 0;")
        self._header_layout = QHBoxLayout(self._header)
        self._header_layout.setContentsMargins(8, 3, 4, 3)
        self._header_layout.setSpacing(4)
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: bold; background: transparent;")
        self._header_layout.addWidget(title_label)
        self._header_layout.addStretch()

        # Collapse toggle — always the last item in the header; checked=expanded
        self._btn_collapse = QToolButton()
        self._btn_collapse.setCheckable(True)
        self._btn_collapse.setChecked(True)
        self._btn_collapse.setStyleSheet(stylesheets.TOOLBUTTON_ICON_STYLESHEET)
        self._btn_collapse.toggled.connect(self._on_collapse_toggled)
        self._header_layout.addWidget(self._btn_collapse)

        if not collapsible:
            self._btn_collapse.setVisible(False)

        outer.addWidget(self._header)

        # Body
        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(4, 4, 4, 4)
        outer.addWidget(self._body)

        self._on_collapse_toggled(True)  # set initial icon + body visibility

        if content is not None:
            self.set_content(content)

    def _on_collapse_toggled(self, expanded: bool) -> None:
        # Non-collapsible panels are always expanded
        if not self._collapsible:
            expanded = True
        self._body.setVisible(expanded)
        icon = "mdi:chevron-up" if expanded else "mdi:chevron-down"
        self._btn_collapse.setIcon(QIconifyIcon(icon, color=stylesheets.GRAY_ICON_COLOR))
        self._btn_collapse.setToolTip("Collapse" if expanded else "Expand")

    def add_header_widget(self, widget: QWidget) -> None:
        """Add a widget to the right side of the header, before the collapse button."""
        # Insert before the collapse button (always the last item)
        self._header_layout.insertWidget(self._header_layout.count() - 1, widget)

    def set_content(self, widget: QWidget) -> None:
        """Replace the body content with widget."""
        while self._body_layout.count():
            self._body_layout.takeAt(0)
        self._body_layout.addWidget(widget)


class _SpinnerLabel(QLabel):
    """Rotating icon label used as a lightweight acquisition progress indicator."""

    def __init__(self, icon_name="mdi:loading", color="#4fc3f7", size=24,
                 step_deg=20, interval_ms=40, parent=None):
        super().__init__(parent)
        self._pixmap = QIconifyIcon(icon_name, color=color).pixmap(size, size)
        self._angle = 0
        self._step = step_deg
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._tick)
        self.setFixedSize(size, size)
        self.setAlignment(Qt.AlignCenter)
        self._render()

    def _tick(self):
        self._angle = (self._angle + self._step) % 360
        self._render()

    def _render(self):
        t = QTransform().rotate(self._angle)
        self.setPixmap(self._pixmap.transformed(t, Qt.SmoothTransformation))

    def start(self):
        self._timer.start()

    def stop(self):
        self._timer.stop()
        self._angle = 0
        self._render()


class IconToolButton(QToolButton):
    """QToolButton with Iconify icon and automatic checked-state icon/color/tooltip swapping.

    Parameters
    ----------
    icon : str
        Iconify icon name for unchecked/default state (e.g. ``"mdi:tune"``).
    color : str, optional
        Icon color for unchecked/default state. Defaults to ``GRAY_ICON_COLOR``.
    checked_icon : str, optional
        Icon name when checked. If ``None``, uses ``icon``.
    checked_color : str, optional
        Icon color when checked. Defaults to ``GRAY_WHITE_COLOR``.
        Only applied when ``checked_icon`` or ``checked_color`` is provided, or
        ``checkable=True``.
    tooltip : str, optional
        Tooltip for unchecked/default state.
    checked_tooltip : str, optional
        Tooltip when checked. Defaults to ``tooltip``.
    checkable : bool, optional
        Whether the button is checkable. Automatically ``True`` when
        ``checked_icon`` or ``checked_color`` are provided.
    checked : bool, optional
        Initial checked state. Defaults to ``False``.
    size : int, optional
        If provided, calls ``setFixedSize(size, size)``.
    parent : QWidget, optional
    """

    def __init__(
        self,
        icon: str,
        color: str = stylesheets.GRAY_ICON_COLOR,
        checked_icon: str | None = None,
        checked_color: str | None = None,
        tooltip: str | None = None,
        checked_tooltip: str | None = None,
        checkable: bool = False,
        checked: bool = False,
        size: int | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._icon = icon
        self._color = color
        self._checked_icon = checked_icon if checked_icon is not None else icon
        self._checked_color = checked_color if checked_color is not None else stylesheets.GRAY_WHITE_COLOR
        self._tooltip = tooltip
        self._checked_tooltip = checked_tooltip if checked_tooltip is not None else tooltip

        self._has_state = checkable or checked_icon is not None or checked_color is not None

        self.setStyleSheet(stylesheets.TOOLBUTTON_ICON_STYLESHEET)
        if size is not None:
            self.setFixedSize(size, size)

        if self._has_state:
            self.setCheckable(True)
            self.toggled.connect(self._on_toggled)
            # Suppress icon-swap during setChecked so _on_toggled drives it once below
            super().setChecked(checked)
            self._on_toggled(checked)
        else:
            self.setIcon(QIconifyIcon(self._icon, color=self._color))
            if tooltip:
                self.setToolTip(tooltip)

    def _on_toggled(self, checked: bool) -> None:
        icon = self._checked_icon if checked else self._icon
        color = self._checked_color if checked else self._color
        self.setIcon(QIconifyIcon(icon, color=color))
        tip = self._checked_tooltip if checked else self._tooltip
        if tip is not None:
            self.setToolTip(tip)

    def set_icon_state(self, checked: bool) -> None:
        """Update icon/color/tooltip to match ``checked`` without emitting ``toggled``."""
        self._on_toggled(checked)


class TaskNameListWidget(QWidget):
    """Task-name list with a styled header containing a label and optional add/remove buttons.

    Emits ``task_selected(str)`` when the selection changes.
    Call ``set_tasks()`` to repopulate; the current selection is preserved if
    still present, otherwise falls back to a preferred name →
    ``"Rough Milling"`` → first row.
    Call ``set_buttons_visible(add, remove)`` to show/hide the header buttons.
    """

    task_selected = pyqtSignal(str)
    add_clicked = pyqtSignal()
    remove_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header
        header = QWidget()
        header.setStyleSheet("background: #1e2124;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 3, 4, 3)
        header_layout.setSpacing(4)
        lbl = QLabel("Task Name")
        lbl.setStyleSheet("font-weight: bold; background: transparent;")
        header_layout.addWidget(lbl)
        header_layout.addStretch()
        self.btn_add = IconToolButton("mdi:plus", tooltip="Add task", size=24)
        self.btn_remove = IconToolButton("mdi:trash-can-outline", tooltip="Remove task", size=24)
        header_layout.addWidget(self.btn_add)
        header_layout.addWidget(self.btn_remove)
        outer.addWidget(header)

        # List
        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(self._list)

        # Wire signals
        self._list.itemSelectionChanged.connect(
            lambda: self.task_selected.emit(self.selected_task)
        )
        self.btn_add.clicked.connect(self.add_clicked)
        self.btn_remove.clicked.connect(self.remove_clicked)

    def set_buttons_visible(self, add: bool, remove: bool) -> None:
        """Show or hide the add and remove header buttons independently."""
        self.btn_add.setVisible(add)
        self.btn_remove.setVisible(remove)

    @property
    def selected_task(self) -> str:
        """Return the currently selected task name, or ``""`` if nothing selected."""
        item = self._list.currentItem()
        return item.text() if item is not None else ""

    def set_tasks(self, names: List[str], preferred: str = "") -> None:
        """Populate the list, restoring selection intelligently.

        Priority: current selection → *preferred* → ``"Rough Milling"`` → row 0.
        Signals are suppressed during population.
        """
        current = self.selected_task or preferred
        self._list.blockSignals(True)
        self._list.clear()
        for name in names:
            self._list.addItem(name)
        self._restore_selection(names, current)
        self._list.blockSignals(False)

    def select(self, name: str) -> None:
        """Select the item with the given name (exact match)."""
        items = self._list.findItems(name, Qt.MatchExactly)  # type: ignore
        if items:
            self._list.setCurrentItem(items[0])

    def _restore_selection(self, names: List[str], preferred: str) -> None:
        if preferred and preferred in names:
            self.select(preferred)
        elif "Rough Milling" in names:
            self.select("Rough Milling")
        elif self._list.count() > 0:
            self._list.setCurrentRow(0)


_LAMELLA_NAME_MIN_WIDTH = 160
_LAMELLA_ROW_HEIGHT = 30
_LAMELLA_BTN_SIZE = QSize(24, 24)


def _lamella_status_text(lamella) -> tuple[str, str]:
    """Return (text, stylesheet) for the status column.

    Gracefully handles objects without task_state (e.g. plain positions).
    """
    ts = getattr(lamella, "task_state", None)
    if ts and getattr(ts, "status", None) is not None:
        try:
            from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus
            if ts.status == AutoLamellaTaskStatus.InProgress:
                return f"{ts.name}", "color: #6aabdf; background: transparent;"
        except ImportError:
            pass
    last = getattr(lamella, "last_completed_task", None)
    if last:
        return last.completed, "color: #909090; background: transparent;"
    return "", "background: transparent;"


def _lamella_defect_icon(lamella) -> tuple[str, str, str]:
    """Return (icon_name, icon_color, tooltip) for the defect indicator.

    Gracefully handles objects without defect field.
    """
    defect = getattr(lamella, "defect", None)
    if defect is None:
        return "mdi:check-circle", stylesheets.GREEN_COLOR, "No defect"
    try:
        from fibsem.applications.autolamella.structures import DefectType
        if defect.state == DefectType.REWORK:
            desc = f": {defect.description}" if defect.description else ""
            return "mdi:refresh-circle", stylesheets.DEFECT_ORANGE_COLOR, f"Rework required{desc}"
        if defect.state == DefectType.FAILURE:
            desc = f": {defect.description}" if defect.description else ""
            return "mdi:close-circle", stylesheets.DEFECT_RED_COLOR, f"Failure{desc}"
    except ImportError:
        pass
    return "mdi:check-circle", stylesheets.GREEN_COLOR, "No defect"


class _LamellaRow(QWidget):
    """Single row: name + status labels + optional toolbuttons."""

    move_to_clicked = pyqtSignal(object)
    edit_clicked = pyqtSignal(object)
    update_clicked = pyqtSignal(object)
    remove_clicked = pyqtSignal(object)
    defect_changed = pyqtSignal(object)

    def __init__(self, lamella, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.lamella = lamella
        self.setAttribute(Qt.WA_TranslucentBackground)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(8)

        self.name_label = QLabel(lamella.name)
        self.name_label.setMinimumWidth(_LAMELLA_NAME_MIN_WIDTH)
        self.name_label.setStyleSheet("background: transparent;")
        layout.addWidget(self.name_label)

        self.status_label = QLabel()
        layout.addWidget(self.status_label, stretch=1)

        # Defect button
        self.btn_defect = QToolButton()
        self.btn_defect.setFixedSize(_LAMELLA_BTN_SIZE)
        self.btn_defect.setStyleSheet(stylesheets.TOOLBUTTON_ICON_STYLESHEET)
        self.btn_defect.clicked.connect(self._on_defect_clicked)
        self.btn_defect.setVisible(False)
        layout.addWidget(self.btn_defect)

        # Actions menu button
        self.btn_actions = QToolButton()
        self.btn_actions.setIcon(QIconifyIcon("mdi:dots-horizontal", color=stylesheets.GRAY_ICON_COLOR))
        self.btn_actions.setToolTip("Actions")
        self.btn_actions.setFixedSize(_LAMELLA_BTN_SIZE)
        self.btn_actions.setStyleSheet(
            stylesheets.TOOLBUTTON_ICON_STYLESHEET
            + " QToolButton::menu-indicator { image: none; }"
        )
        self.btn_actions.setPopupMode(QToolButton.InstantPopup)
        actions_menu = QMenu(self)
        self._action_move = actions_menu.addAction(
            QIconifyIcon("mdi:crosshairs-gps", color=stylesheets.GRAY_ICON_COLOR),
            "Move to Position",
        )
        self._action_edit = actions_menu.addAction(
            QIconifyIcon("mdi:pencil", color=stylesheets.GRAY_ICON_COLOR),
            "Edit Lamella",
        )
        self._action_update = actions_menu.addAction(
            QIconifyIcon("mdi:map-marker-check", color=stylesheets.GRAY_ICON_COLOR),
            "Update Position",
        )
        self.btn_actions.setMenu(actions_menu)
        self.btn_actions.setVisible(False)
        layout.addWidget(self.btn_actions)

        self._action_move.triggered.connect(lambda: self.move_to_clicked.emit(self.lamella))
        self._action_edit.triggered.connect(lambda: self.edit_clicked.emit(self.lamella))
        self._action_update.triggered.connect(lambda: self.update_clicked.emit(self.lamella))

        # Remove button
        self.btn_remove = QToolButton()
        self.btn_remove.setIcon(QIconifyIcon("mdi:trash-can-outline", color=stylesheets.GRAY_ICON_COLOR))
        self.btn_remove.setToolTip("Remove")
        self.btn_remove.setFixedSize(_LAMELLA_BTN_SIZE)
        self.btn_remove.setStyleSheet(stylesheets.TOOLBUTTON_ICON_STYLESHEET)
        self.btn_remove.clicked.connect(self._on_remove_clicked)
        self.btn_remove.setVisible(False)
        layout.addWidget(self.btn_remove)

        self.refresh()

    def _on_defect_clicked(self) -> None:
        try:
            from fibsem.applications.autolamella.structures import DefectState, DefectType
        except ImportError:
            return
        menu = QMenu(self)
        action_none = menu.addAction(
            QIconifyIcon("mdi:check-circle", color=stylesheets.GREEN_COLOR), "No defect"
        )
        action_rework = menu.addAction(
            QIconifyIcon("mdi:refresh-circle", color=stylesheets.DEFECT_ORANGE_COLOR), "Rework required"
        )
        action_failure = menu.addAction(
            QIconifyIcon("mdi:close-circle", color=stylesheets.DEFECT_RED_COLOR), "Failure"
        )
        chosen = menu.exec_(self.btn_defect.mapToGlobal(self.btn_defect.rect().bottomLeft()))
        if chosen == action_none:
            self.lamella.defect = DefectState(state=DefectType.NONE)
        elif chosen == action_rework:
            self.lamella.defect = DefectState(state=DefectType.REWORK)
        elif chosen == action_failure:
            self.lamella.defect = DefectState(state=DefectType.FAILURE)
        else:
            return
        self.refresh()
        self.defect_changed.emit(self.lamella)

    def _on_remove_clicked(self) -> None:
        reply = QMessageBox.question(
            self, "Remove Lamella", f"Remove <b>{self.lamella.name}</b>?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.remove_clicked.emit(self.lamella)

    def refresh(self) -> None:
        self.name_label.setText(self.lamella.name)
        text, style = _lamella_status_text(self.lamella)
        self.status_label.setText(text)
        self.status_label.setStyleSheet(style)
        # Update defect icon if visible
        if self.btn_defect.isVisible():
            icon_name, icon_color, tooltip = _lamella_defect_icon(self.lamella)
            self.btn_defect.setIcon(QIconifyIcon(icon_name, color=icon_color))
            self.btn_defect.setToolTip(tooltip)


class LamellaNameListWidget(QWidget):
    """Single-selection list widget for lamella names with smart restore logic.

    Shows two columns: lamella name and last completed task status.
    Stores the associated object as ``Qt.UserRole`` data on each item so callers
    can retrieve it via ``selected_lamella``.

    Emits ``lamella_selected(object)`` when the selection changes.
    Call ``set_lamella()`` to repopulate; the current selection is preserved by
    name if still present, otherwise falls back to the first row.
    """

    lamella_selected = pyqtSignal(object)
    add_requested = pyqtSignal()
    move_to_requested = pyqtSignal(object)
    edit_requested = pyqtSignal(object)
    update_requested = pyqtSignal(object)
    remove_requested = pyqtSignal(object)
    defect_changed = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._btn_visible = {
            "defect": False,
            "actions": False,
            "move_to": False,
            "edit": False,
            "update": False,
            "remove": False,
        }

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header
        header = QWidget()
        header.setStyleSheet("background: #1e2124;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 3, 4, 3)
        header_layout.setSpacing(8)
        lbl_name = QLabel("Lamella")
        lbl_name.setStyleSheet("font-weight: bold; background: transparent;")
        lbl_name.setMinimumWidth(_LAMELLA_NAME_MIN_WIDTH)
        header_layout.addWidget(lbl_name)
        lbl_status = QLabel("Status")
        lbl_status.setStyleSheet("font-weight: bold; background: transparent;")
        header_layout.addWidget(lbl_status, stretch=1)
        self.btn_add = IconToolButton(icon="mdi:plus", tooltip="Add", size=_LAMELLA_BTN_SIZE.width())
        self.btn_add.setVisible(False)
        self.btn_add.clicked.connect(self.add_requested.emit)
        header_layout.addWidget(self.btn_add)
        outer.addWidget(header)

        # List
        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(self._list)

        # Wire signals
        self._list.itemSelectionChanged.connect(
            lambda: self.lamella_selected.emit(self.selected_lamella)
        )

    @property
    def selected_lamella(self) -> Any:
        """Return the ``Qt.UserRole`` data of the current item, or ``None``."""
        item = self._list.currentItem()
        return item.data(Qt.UserRole) if item is not None else None  # type: ignore

    @property
    def selected_name(self) -> str:
        """Return the display name of the current item, or ``""``."""
        lamella = self.selected_lamella
        return lamella.name if lamella is not None else ""

    @property
    def selected_index(self) -> int:
        """Return the current row index, or -1 if nothing is selected."""
        return self._list.currentRow()

    def set_lamella(self, positions, preferred_name: str = "") -> None:
        """Populate the list from *positions*, restoring selection by name.

        Priority: current selection → *preferred_name* → first row.
        Signals are suppressed during population.
        """
        current = self.selected_name or preferred_name
        self._list.blockSignals(True)
        self._list.clear()
        for pos in positions:
            row = _LamellaRow(pos)
            row.move_to_clicked.connect(self.move_to_requested)
            row.edit_clicked.connect(self.edit_requested)
            row.update_clicked.connect(self.update_requested)
            row.remove_clicked.connect(self.remove_requested)
            row.defect_changed.connect(self.defect_changed)
            self._apply_btn_visibility(row)
            item = QListWidgetItem()
            item.setData(Qt.UserRole, pos)  # type: ignore
            item.setSizeHint(QSize(0, _LAMELLA_ROW_HEIGHT))
            self._list.addItem(item)
            self._list.setItemWidget(item, row)
        self._restore_selection(current)
        self._list.blockSignals(False)

    def refresh_all(self) -> None:
        """Refresh the display of all rows (e.g. after status changes)."""
        for i in range(self._list.count()):
            row = self._list.itemWidget(self._list.item(i))
            if isinstance(row, _LamellaRow):
                row.refresh()

    # ------------------------------------------------------------------
    # Button visibility
    # ------------------------------------------------------------------

    def enable_add_button(self, visible: bool) -> None:
        self.btn_add.setVisible(visible)

    def enable_defect_button(self, visible: bool) -> None:
        self._btn_visible["defect"] = visible
        for row in self._rows():
            row.btn_defect.setVisible(visible)
            if visible:
                row.refresh()  # ensure icon is up to date

    def enable_actions_button(self, visible: bool) -> None:
        self._btn_visible["actions"] = visible
        for row in self._rows():
            row.btn_actions.setVisible(visible)

    def enable_move_to_action(self, visible: bool) -> None:
        self._btn_visible["move_to"] = visible
        for row in self._rows():
            row._action_move.setVisible(visible)

    def enable_edit_action(self, visible: bool) -> None:
        self._btn_visible["edit"] = visible
        for row in self._rows():
            row._action_edit.setVisible(visible)

    def enable_update_action(self, visible: bool) -> None:
        self._btn_visible["update"] = visible
        for row in self._rows():
            row._action_update.setVisible(visible)

    def enable_remove_button(self, visible: bool) -> None:
        self._btn_visible["remove"] = visible
        for row in self._rows():
            row.btn_remove.setVisible(visible)

    def _rows(self):
        """Yield all _LamellaRow widgets."""
        for i in range(self._list.count()):
            row = self._list.itemWidget(self._list.item(i))
            if isinstance(row, _LamellaRow):
                yield row

    def _apply_btn_visibility(self, row: _LamellaRow) -> None:
        row.btn_defect.setVisible(self._btn_visible["defect"])
        row.btn_actions.setVisible(self._btn_visible["actions"])
        row._action_move.setVisible(self._btn_visible["move_to"])
        row._action_edit.setVisible(self._btn_visible["edit"])
        row._action_update.setVisible(self._btn_visible["update"])
        row.btn_remove.setVisible(self._btn_visible["remove"])
        if self._btn_visible["defect"]:
            # Set icon directly — row.refresh() won't work here because
            # isVisible() returns False before the row is parented via setItemWidget
            icon_name, icon_color, tooltip = _lamella_defect_icon(row.lamella)
            row.btn_defect.setIcon(QIconifyIcon(icon_name, color=icon_color))
            row.btn_defect.setToolTip(tooltip)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(self, name: str) -> None:
        """Select the item with the given name (exact match)."""
        for i in range(self._list.count()):
            item = self._list.item(i)
            obj = item.data(Qt.UserRole)
            if obj is not None and obj.name == name:
                self._list.setCurrentItem(item)
                return

    def _restore_selection(self, preferred: str) -> None:
        if preferred:
            for i in range(self._list.count()):
                item = self._list.item(i)
                obj = item.data(Qt.UserRole)
                if obj is not None and obj.name == preferred:
                    self._list.setCurrentItem(item)
                    return
        if self._list.count() > 0:
            self._list.setCurrentRow(0)
