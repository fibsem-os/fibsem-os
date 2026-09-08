from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Union

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QCursor, QFontMetrics, QIcon, QPainter
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QAction,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMenu,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.ui import stylesheets as stylesheets
from fibsem.ui.icon import fibsem_icon, qta
from fibsem.ui.tokens import (
    CANVAS_BG,
)
from fibsem.ui.utils import install_wheel_blocker
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
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory", self.lineEdit.text()
        )
        if directory:
            self.lineEdit.setText(directory)
            self.textChanged.emit(directory)
            self.editingFinished.emit()

    def text(self) -> str:
        return self.lineEdit.text()

    def setText(self, text: str) -> None:
        self.lineEdit.setText(text)


class QFileLineEdit(QWidget):
    """Line edit with a browse button that opens a file picker dialog."""

    textChanged = pyqtSignal(str)
    editingFinished = pyqtSignal()

    def __init__(self, filter: str = "YAML files (*.yaml *.yml)", parent=None):
        super().__init__(parent)
        self._filter = filter

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
        start = self.lineEdit.text() or ""
        path, _ = QFileDialog.getOpenFileName(self, "Select File", start, self._filter)
        if path:
            self.lineEdit.setText(path)
            self.textChanged.emit(path)
            self.editingFinished.emit()

    def text(self) -> str:
        return self.lineEdit.text()

    def setText(self, text: str) -> None:
        self.lineEdit.setText(text)


def _create_combobox_control(
    value: Union[str, int, float, Enum],
    items: list,
    units: Optional[str],
    format_fn: Optional[Callable] = None,
    control: Optional[QComboBox] = None,
) -> QComboBox:
    """Create a QComboBox control for selecting from a list of items."""
    if control is None:
        control = QComboBox()
    for item in items:
        if isinstance(item, (float, int)):
            item_str = format_value(val=item, unit=units, precision=1)
        elif isinstance(item, Enum):
            item_str = item.name  # TODO: migrate to QEnumComboBox
        elif format_fn is not None:
            item_str = format_fn(item)
        else:
            item_str = str(item)
        control.addItem(item_str, item)

    if isinstance(value, tuple) and len(value) == 2:
        value = list(value)  # Convert tuple to list for easier handling

    # find the closest match to the current value (should only be used for numerical values)
    idx = control.findData(value)
    if idx == -1 and len(items) > 0:
        # get the closest value
        closest_value = min(items, key=lambda x: abs(x - value))
        idx = control.findData(closest_value)

    if idx == -1:
        if len(items) == 0:
            logging.warning(f"No items available for combobox with value {value}")
        else:
            logging.debug(
                f"Warning: No matching item or nearest found for {items} with value {value}. Using first item."
            )
            idx = 0

    if idx >= 0:
        control.setCurrentIndex(idx)
    install_wheel_blocker(control)

    return control


class ValueComboBox(QComboBox):
    """QComboBox that stores raw values as item data and supports closest-match selection."""

    def __init__(
        self,
        items: Optional[list] = None,
        value=None,
        unit: Optional[str] = None,
        format_fn: Optional[Callable] = None,
        decimals: int = 1,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._unit = unit
        self._format_fn = format_fn
        self._decimals = decimals
        if items:
            self.add_values(items)
        install_wheel_blocker(self)
        if value is not None:
            self.set_value(value)

    def _format_item(self, item) -> str:
        """Render an item for display, using the unit/format_fn given at construction.

        An explicit `format_fn` wins over every built-in rule. It used to be consulted
        last, so a caller that supplied one still got the default rendering for numbers
        and enums -- silently, since nothing errors when a label is merely wrong. Three
        call sites were affected: two mode combo boxes showed `EACH_ROW` in place of
        the label they asked for, and emission wavelengths rendered as `550.0` rather
        than the `550 nm` their formatter produces.
        """
        if self._format_fn is not None:
            return self._format_fn(item)
        if isinstance(item, (float, int)):
            return format_value(val=item, unit=self._unit, precision=self._decimals)
        if isinstance(item, Enum):
            return item.name
        return str(item)

    def add_value(self, item) -> None:
        """Append a single item, storing the raw value as item data."""
        self.addItem(self._format_item(item), item)

    def add_values(self, items) -> None:
        """Append several items."""
        for item in items:
            self.add_value(item)

    def set_values(self, items, value=None, keep_selection: bool = True) -> None:
        """Replace all items.

        By default the current selection is preserved if it survives the swap —
        repopulating a combobox should not silently change what is selected.
        Falls back to *value*, else leaves the default (first) item. Signals are
        suppressed during the swap so listeners see one change at most.
        """
        previous = self.value() if keep_selection else None
        blocked = self.blockSignals(True)
        try:
            self.clear()
            self.add_values(items)
        finally:
            self.blockSignals(blocked)

        target = value if value is not None else previous
        if target is not None:
            self.set_value(target)

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


class IntegerValueSpinBox(QSpinBox):
    """QSpinBox with sensible defaults, WheelBlocker, and None-safe configuration."""

    def __init__(
        self,
        suffix: Optional[str] = None,
        minimum: Optional[int] = None,
        maximum: Optional[int] = None,
        step: Optional[int] = None,
        tooltip: Optional[str] = None,
        no_buttons: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        if suffix:
            self.setSuffix(f" {suffix}")
        self.setRange(
            minimum if minimum is not None else 0,
            maximum if maximum is not None else 1000000,
        )
        self.setSingleStep(step if step is not None else 1)
        if tooltip:
            self.setToolTip(tooltip)
        if no_buttons:
            self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setKeyboardTracking(False)
        install_wheel_blocker(self)


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
        no_buttons: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        if suffix:
            self.setSuffix(f" {suffix}")
        self.setRange(
            minimum if minimum is not None else 0.0,
            maximum if maximum is not None else 1e6,
        )
        self.setSingleStep(step if step is not None else 0.01)
        self.setDecimals(decimals if decimals is not None else 3)
        if tooltip:
            self.setToolTip(tooltip)
        if no_buttons:
            self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setKeyboardTracking(False)
        install_wheel_blocker(self)


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
        self.actions.append(
            ContextMenuAction(
                label=label,
                callback=callback,
                icon=icon,
                tooltip=tooltip,
                enabled=enabled,
                separator_after=separator_after,
                data=data,
            )
        )
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
            logging.exception(
                "Context menu action '%s' raised an exception.", menu_action.label
            )
            try:
                from fibsem.ui import notification_service

                notification_service.show_toast(
                    f"Action '{menu_action.label}' failed.", "warning"
                )
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


class TitledPanel(QWidget):
    """A styled panel with a dark header row (title label + optional widgets) and a collapsible content area.

    Usage::

        panel = TitledPanel("Milling", content=milling_widget)
        panel.add_header_widget(btn_advanced)   # right-aligned, before collapse button

        fixed = TitledPanel("Setup", content=setup_widget, collapsible=False)
    """

    def __init__(
        self,
        title: str,
        content: Optional[QWidget] = None,
        collapsible: bool = True,
        parent=None,
    ) -> None:
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
        self._header.setStyleSheet(
            f"background: {CANVAS_BG}; border-radius: 3px 3px 0 0;"
        )
        self._header_layout = QHBoxLayout(self._header)
        self._header_layout.setContentsMargins(8, 3, 4, 3)
        self._header_layout.setSpacing(4)
        self._title_label = QLabel(title)
        self._title_label.setStyleSheet("font-weight: bold; background: transparent;")
        self._header_layout.addWidget(self._title_label)
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

    def collapse(self) -> None:
        """Collapse the panel body."""
        self._btn_collapse.setChecked(False)

    def expand(self) -> None:
        """Expand the panel body."""
        self._btn_collapse.setChecked(True)

    def _on_collapse_toggled(self, expanded: bool) -> None:
        # Non-collapsible panels are always expanded
        if not self._collapsible:
            expanded = True
        self._body.setVisible(expanded)
        icon = "mdi:chevron-up" if expanded else "mdi:chevron-down"
        self._btn_collapse.setIcon(fibsem_icon(icon, color=stylesheets.GRAY_ICON_COLOR))
        self._btn_collapse.setToolTip("Collapse" if expanded else "Expand")

    def set_title(self, title: str) -> None:
        """Update the panel header title text."""
        self._title_label.setText(title)

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
    """Spinning icon label used as a lightweight acquisition progress indicator.

    Backed by qtawesome's native ``Spin`` animation: the icon engine rotates the
    glyph and repaints this widget on a timer, so there is no manual rotation
    bookkeeping. The animation timer is created lazily on first paint, so we let
    it ``autostart`` and drive visibility via :meth:`start`/:meth:`stop`.
    """

    def __init__(
        self,
        icon_name="mdi:loading",
        color="#4fc3f7",
        size=24,
        step_deg=20,
        interval_ms=40,
        parent=None,
    ):
        super().__init__(parent)
        self._spin = qta.Spin(self, interval=interval_ms, step=step_deg)
        self._icon = fibsem_icon(icon_name, color=color, animation=self._spin)
        self._active = False
        self.setFixedSize(size, size)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: transparent;")

    def paintEvent(self, event):
        # only draw (and thereby drive the animation) while active
        if not self._active:
            return
        painter = QPainter(self)
        try:
            self._icon.paint(painter, self.rect())
        finally:
            painter.end()

    def start(self):
        self._active = True
        # first paint registers the widget with the Spin (autostart=True) and
        # starts its timer; _spin.start() then resumes it on a start-after-stop.
        self.update()
        self._spin.start()

    def stop(self):
        self._active = False
        self._spin.stop()
        self.update()  # repaint blank

    def clear(self):
        # blanking the spinner implies stopping its animation
        self.stop()
        super().clear()


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
        self._checked_color = (
            checked_color if checked_color is not None else stylesheets.GRAY_WHITE_COLOR
        )
        self._tooltip = tooltip
        self._checked_tooltip = (
            checked_tooltip if checked_tooltip is not None else tooltip
        )

        self._has_state = (
            checkable or checked_icon is not None or checked_color is not None
        )

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
            self.setIcon(fibsem_icon(self._icon, color=self._color))
            if tooltip:
                self.setToolTip(tooltip)

    def _on_toggled(self, checked: bool) -> None:
        icon = self._checked_icon if checked else self._icon
        color = self._checked_color if checked else self._color
        self.setIcon(fibsem_icon(icon, color=color))
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
        header.setStyleSheet(f"background: {CANVAS_BG};")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 3, 4, 3)
        header_layout.setSpacing(4)
        lbl = QLabel("Task Name")
        lbl.setStyleSheet("font-weight: bold; background: transparent;")
        header_layout.addWidget(lbl)
        header_layout.addStretch()
        self.btn_add = IconToolButton("mdi:plus", tooltip="Add task", size=24)
        self.btn_remove = IconToolButton(
            "mdi:trash-can-outline", tooltip="Remove task", size=24
        )
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
        elif self._list.count() > 0:
            self._list.setCurrentRow(0)


# ---------------------------------------------------------------------------
# Shared pieces of the hand-built dark-theme dialogs
# ---------------------------------------------------------------------------


def style_with_tooltip(widget: QWidget, css: str) -> None:
    """Set a selector-less stylesheet without swallowing the widget's tooltip.

    A stylesheet with no selector applies to every type Qt renders through the
    widget, the QToolTip included, and beats the application sheet because it
    sits nearer -- so a cell styled ``background: transparent`` shows its tooltip
    as floating text with nothing behind it.

    Use this for any rule that does not name a type. Rules that do
    (``QPushButton {...}``) cannot leak and can be set directly.
    """
    widget.setStyleSheet(css + stylesheets.TOOLTIP_STYLESHEET)


class ElidedLabel(QLabel):
    """A QLabel that elides rather than clipping mid-glyph.

    For a column that stretches, where the width is not known when the row is built and
    any one-off elide would be wrong at the next size. The size policy is Ignored so long
    unbreakable text -- a dotted class path, a file path, a 132-character acquisition
    failure -- cannot drive the column wider, which would defeat the point. One such
    message dragged the FM overview's minimum width from 1030 px to 1728 px.

    `text()` returns what was set, not what is drawn: callers compare against it to
    decide whether the line is theirs to overwrite, and eliding is presentation. The
    tooltip carries the whole string, which is the only way to read one that does not
    fit; a caller wanting a different tooltip sets it after `setText`.
    """

    def __init__(
        self,
        text: str = "",
        parent: Optional[QWidget] = None,
        mode: Qt.TextElideMode = Qt.ElideRight,
    ) -> None:
        """
        Args:
            mode: which end goes. `ElideRight` for prose, where the start carries the
                sense. **`ElideLeft` for a path**, where it is the tail -- the
                experiment and the run -- that answers "am I writing where I meant
                to", and the leading directories are the same on every line.
        """
        super().__init__(parent)
        self._mode = mode
        self._full_text = ""
        # Ignored horizontally: the label neither asks for room nor refuses to shrink,
        # which is the whole point -- its content must not set anyone's minimum.
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setText(text)

    def setText(self, text: Optional[str]) -> None:  # noqa: N802 - Qt naming
        if (text or "") == self._full_text:
            # Re-measuring costs a QFontMetrics and an elidedText per call, and callers
            # that refresh a whole row on a timer re-set the same string every time --
            # the workflow timeline does it for every row of every status update, where
            # this was a third of the cost. Nothing else here depends on width, which
            # resizeEvent and paintEvent handle.
            return
        self._full_text = text or ""
        self.setToolTip(self._full_text)
        self._elide()

    def text(self) -> str:
        return self._full_text

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._elide()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        # Belt and braces with the resize: a label laid out at its final size and never
        # resized after would otherwise keep whatever `setText` computed from a width it
        # did not yet have, and clip.
        self._elide()
        super().paintEvent(event)

    def _elide(self) -> None:
        metrics = QFontMetrics(self.font())
        elided = metrics.elidedText(
            self._full_text, self._mode, max(0, self.width() - 2)
        )
        # `super().text()`, not ours: ours returns the full string, so this would differ
        # on every paint of an elided label and schedule another one forever. QLabel
        # draws the elided string, so the stylesheet colour survives.
        if elided != super().text():
            super().setText(elided)


def chip(text: str, colour: str, font_size: int = 11) -> QLabel:
    """A pill label: text on a tint of its own colour.

    No dot. Once every chip has one it separates nothing, and it costs real width
    in the narrow columns these sit in.

    The minimum width is set explicitly because a QLabel inside a table cell
    reports a size hint that ignores stylesheet padding, so the pill would
    otherwise render clipped at both ends.
    """
    rgb = QColor(colour)
    tint = f"rgba({rgb.red()}, {rgb.green()}, {rgb.blue()}, 0.15)"
    label = QLabel(text)
    style_with_tooltip(
        label,
        f"background-color: {tint}; color: {colour};"
        f"padding: 2px 9px; border-radius: 10px; font-size: {font_size}px;",
    )
    font = label.font()
    font.setPixelSize(font_size)
    label.setMinimumWidth(QFontMetrics(font).horizontalAdvance(text) + 26)
    return label
