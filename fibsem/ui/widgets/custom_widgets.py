from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QCursor, QIcon, QTransform
from PyQt5.QtWidgets import (
    QAction,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QToolButton,
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

    # find the closest match to the current value (should only be used for numerical values)
    idx = control.findData(value)
    if idx == -1:
        # get the closest value
        closest_value = min(items, key=lambda x: abs(x - value))
        idx = control.findData(closest_value)
    if idx == -1:
        logging.debug(f"Warning: No matching item or nearest found for {items} with value {value}. Using first item.")
        idx = 0
    control.setCurrentIndex(idx)
    control.installEventFilter(WheelBlocker(parent=control))

    return control


@dataclass
class ContextMenuAction:
    """Represents a single action in a context menu."""
    label: str
    callback: Optional[Callable] = None
    icon: Optional[QIcon] = None
    tooltip: Optional[str] = None
    enabled: bool = True
    separator_after: bool = False
    data: Optional[any] = None


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
        data: Optional[any] = None,
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
                menu_action.callback()

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
                menu_action.callback()

        return menu_action


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