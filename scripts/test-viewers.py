"""
Example showing how to embed multiple napari viewers in a tabbed Qt widget.

This demonstrates embedding AutoLamellaUI and Protocol Editor in separate tabs,
each with their own napari viewer instance.
"""

import sys
try:
    sys.modules.pop("PySide6.QtCore")
except Exception:
    pass

import napari
from datetime import datetime
from PyQt5.QtCore import QPropertyAnimation, QTimer, Qt, QPoint, QEasingCurve, pyqtSignal
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenuBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStatusBar,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from superqt.iconify import QIconifyIcon

import fibsem
from fibsem.ui.widgets.autolamella_task_config_editor import (
    AutoLamellaWorkflowWidget,
    AutoLamellaProtocolTaskConfigEditor,
)
from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus


class ToastNotification(QWidget):
    """A toast notification widget that appears in the bottom-right corner."""

    # Notification types with colors
    TYPES = {
        "info": "#50a6ff",      # Blue
        "success": "#4caf50",   # Green
        "warning": "#ff9800",   # Orange
        "error": "#f44336",     # Red
    }

    def __init__(self, parent=None, duration: int = 5000):
        super().__init__(parent)
        self.duration = duration
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)

        # Main container
        self.container = QWidget(self)
        self.container.setObjectName("toast_container")

        # Layout
        layout = QHBoxLayout(self.container)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

        # Icon label (optional)
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(20, 20)
        self.icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.icon_label)

        # Message label
        self.message_label = QLabel()
        self.message_label.setWordWrap(True)
        self.message_label.setMaximumWidth(300)
        layout.addWidget(self.message_label, 1)

        # Close button
        self.close_btn = QPushButton("×")
        self.close_btn.setFixedSize(20, 20)
        self.close_btn.clicked.connect(self.hide_toast)
        self.close_btn.setCursor(Qt.PointingHandCursor)
        layout.addWidget(self.close_btn)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.container)

        # Opacity effect for fade animation
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        self.opacity_effect.setOpacity(1.0)

        # Fade animation
        self.fade_animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_animation.setDuration(200)

        # Auto-hide timer
        self.hide_timer = QTimer(self)
        self.hide_timer.setSingleShot(True)
        self.hide_timer.timeout.connect(self.hide_toast)

        # Apply base styling
        self._apply_style("info")

    def _apply_style(self, notification_type: str):
        """Apply styling based on notification type."""
        color = self.TYPES.get(notification_type, self.TYPES["info"])

        self.container.setStyleSheet(f"""
            #toast_container {{
                background-color: #1e2027;
                border: 1px solid {color};
                border-left: 4px solid {color};
                border-radius: 6px;
            }}
        """)

        self.message_label.setStyleSheet("""
            QLabel {
                color: #d6d6d6;
                font-size: 13px;
            }
        """)

        self.close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: #888;
                border: none;
                font-size: 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                color: {color};
            }}
        """)

        # Set icon based on type
        icons = {
            "info": "ℹ",
            "success": "✓",
            "warning": "⚠",
            "error": "✕",
        }
        self.icon_label.setText(icons.get(notification_type, "ℹ"))
        self.icon_label.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-size: 14px;
                font-weight: bold;
            }}
        """)

    def show_toast(self, message: str, notification_type: str = "info", duration: int = 3000):
        """Show the toast notification."""
        self.message_label.setText(message)
        self._apply_style(notification_type)

        # Adjust size to content
        self.adjustSize()

        # Position in bottom-right corner of parent
        if self.parent():
            parent_rect = self.parent().rect()
            x = parent_rect.width() - self.width() - 20
            y = parent_rect.height() - self.height() - 20
            self.move(x, y)

        # Fade in
        self.opacity_effect.setOpacity(0)
        self.show()
        self.fade_animation.setStartValue(0)
        self.fade_animation.setEndValue(1)
        self.fade_animation.start()

        # Start auto-hide timer
        hide_duration = duration if duration is not None else self.duration
        if hide_duration > 0:
            self.hide_timer.start(hide_duration)

    def hide_toast(self):
        """Hide the toast with fade animation."""
        self.hide_timer.stop()
        self.fade_animation.setStartValue(1)
        self.fade_animation.setEndValue(0)
        self.fade_animation.finished.connect(self._on_fade_finished)
        self.fade_animation.start()

    def _on_fade_finished(self):
        """Called when fade-out animation completes."""
        try:
            self.fade_animation.finished.disconnect(self._on_fade_finished)
        except TypeError:
            pass  # Already disconnected
        self.hide()
        # Call cleanup callback if set
        if hasattr(self, '_cleanup_callback') and self._cleanup_callback:
            self._cleanup_callback()


class NotificationHistoryPopup(QWidget):
    """Popup widget showing notification history."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(350, 500)

        # Container
        self.container = QWidget(self)
        self.container.setObjectName("notification_popup")

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.container)

        # Container layout
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        # Header
        header = QWidget()
        header.setObjectName("notification_header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 10, 12, 10)

        header_label = QLabel("Notifications")
        header_label.setStyleSheet("color: #d6d6d6; font-weight: bold; font-size: 14px;")
        header_layout.addWidget(header_label)

        header_layout.addStretch()

        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #50a6ff;
                border: none;
                font-size: 12px;
            }
            QPushButton:hover {
                color: #7bc0ff;
            }
        """)
        header_layout.addWidget(self.clear_btn)

        container_layout.addWidget(header)

        # Scroll area for notifications
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)

        self.notifications_container = QWidget()
        self.notifications_layout = QVBoxLayout(self.notifications_container)
        self.notifications_layout.setContentsMargins(0, 0, 0, 0)
        self.notifications_layout.setSpacing(0)
        self.notifications_layout.addStretch()

        self.scroll_area.setWidget(self.notifications_container)
        container_layout.addWidget(self.scroll_area)

        # Empty state label
        self.empty_label = QLabel("No notifications")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet("color: #6b6b6b; padding: 20px;")
        self.notifications_layout.insertWidget(0, self.empty_label)

        # Apply styling
        self.container.setStyleSheet("""
            #notification_popup {
                background-color: #1e2027;
                border: 1px solid #3d4251;
                border-radius: 8px;
            }
            #notification_header {
                border-bottom: 1px solid #3d4251;
            }
        """)

    def add_notification(self, message: str, notification_type: str, timestamp: str):
        """Add a notification to the history."""
        self.empty_label.hide()

        # Create notification item
        item = QWidget()
        item.setObjectName("notification_item")
        item_layout = QHBoxLayout(item)
        item_layout.setContentsMargins(12, 10, 12, 10)
        item_layout.setSpacing(10)

        # Icon
        color = ToastNotification.TYPES.get(notification_type, "#50a6ff")
        icons = {"info": "ℹ", "success": "✓", "warning": "⚠", "error": "✕"}
        icon_label = QLabel(icons.get(notification_type, "ℹ"))
        icon_label.setFixedSize(20, 20)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: bold;")
        item_layout.addWidget(icon_label)

        # Message and timestamp
        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)

        msg_label = QLabel(message)
        msg_label.setWordWrap(True)
        msg_label.setStyleSheet("color: #d6d6d6; font-size: 13px;")
        text_layout.addWidget(msg_label)

        time_label = QLabel(timestamp)
        time_label.setStyleSheet("color: #6b6b6b; font-size: 11px;")
        text_layout.addWidget(time_label)

        item_layout.addLayout(text_layout, 1)

        item.setStyleSheet("""
            #notification_item {
                border-bottom: 1px solid #2d313b;
            }
            #notification_item:hover {
                background-color: #262930;
            }
        """)

        # Insert at top (before the stretch)
        self.notifications_layout.insertWidget(0, item)

    def clear_all(self):
        """Clear all notifications."""
        # Remove all notification items (keep the stretch and empty label)
        while self.notifications_layout.count() > 2:
            item = self.notifications_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.empty_label.show()


class NotificationBell(QWidget):
    """Notification bell icon with counter badge."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.count = 0
        self.setFixedSize(40, 40)
        self.setCursor(Qt.PointingHandCursor)

        # Bell icon button using QToolButton with QIconifyIcon
        self.bell_btn = QToolButton()
        self.bell_btn.setIcon(QIconifyIcon("mdi:bell", color="#d6d6d6"))
        self.bell_btn.setFixedSize(36, 36)
        self.bell_btn.setIconSize(self.bell_btn.size() * 0.6)
        self.bell_btn.clicked.connect(self._on_clicked)

        # Badge
        self.badge = QLabel("0")
        self.badge.setFixedSize(18, 18)
        self.badge.setAlignment(Qt.AlignCenter)
        self.badge.hide()

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.bell_btn)

        # Position badge in top-right
        self.badge.setParent(self)
        self.badge.move(22, 0)

        # Popup
        self.popup = NotificationHistoryPopup()
        self.popup.clear_btn.clicked.connect(self._on_clear_all)

        # Apply styling
        self._apply_style()

    def _apply_style(self):
        self.bell_btn.setStyleSheet("""
            QToolButton {
                background-color: transparent;
                border: none;
            }
            QToolButton:hover {
                background-color: #3d4251;
                border-radius: 18px;
            }
        """)

        self.badge.setStyleSheet("""
            QLabel {
                background-color: #f44336;
                color: white;
                font-size: 10px;
                font-weight: bold;
                border-radius: 9px;
            }
        """)

    def add_notification(self, message: str, notification_type: str):
        """Add a notification and increment counter."""
        self.count += 1
        self.badge.setText(str(min(self.count, 99)))
        self.badge.show()

        # Add to popup history
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.popup.add_notification(message, notification_type, timestamp)

    def _on_clicked(self):
        """Show the notification popup."""
        # Position popup below the bell
        global_pos = self.mapToGlobal(QPoint(0, 0))
        popup_x = global_pos.x() - self.popup.width() + self.width()
        popup_y = global_pos.y() + self.height() + 5

        # Make sure it doesn't go off screen horizontally
        if popup_x < 0:
            popup_x = global_pos.x()

        self.popup.move(popup_x, popup_y)
        self.popup.show()

        # Reset counter when viewed
        self.count = 0
        self.badge.hide()

    def _on_clear_all(self):
        """Clear all notifications."""
        self.popup.clear_all()
        self.count = 0
        self.badge.hide()


class ToastManager:
    """Manages toast notifications for a parent widget."""

    def __init__(self, parent: QWidget):
        self.parent = parent
        self.toasts: list[ToastNotification] = []
        self.spacing = 10
        self.notification_bell: NotificationBell = None

    def set_notification_bell(self, bell: NotificationBell):
        """Set the notification bell to update."""
        self.notification_bell = bell

    def show_toast(self, message: str, notification_type: str = "info", duration: int = 5000):
        """Show a toast notification."""
        toast = ToastNotification(self.parent, duration)
        self.toasts.append(toast)

        # Connect to hidden signal for cleanup (not fade_animation.finished which fires on fade-in too)
        toast.hidden_signal_connected = False
        def on_hidden():
            if toast in self.toasts:
                self.toasts.remove(toast)
                toast.deleteLater()
                self._reposition_toasts()
        toast._cleanup_callback = on_hidden

        toast.show_toast(message, notification_type, duration)

        # Reposition all toasts
        self._reposition_toasts()

        # Add to notification bell history
        if self.notification_bell:
            self.notification_bell.add_notification(message, notification_type)

    def _remove_toast(self, toast: ToastNotification):
        """Remove a toast from the list."""
        if toast in self.toasts:
            self.toasts.remove(toast)
            toast.deleteLater()
            self._reposition_toasts()

    def _reposition_toasts(self):
        """Reposition all visible toasts."""
        if not self.parent:
            return

        parent_rect = self.parent.rect()
        y_offset = 60  # Leave room for notification bell

        for toast in reversed(self.toasts):
            if toast.isVisible():
                x = parent_rect.width() - toast.width() - 20
                y = parent_rect.height() - toast.height() - y_offset
                toast.move(x, y)
                y_offset += toast.height() + self.spacing

# Napari-style dark theme stylesheet
NAPARI_STYLE = """
QMainWindow {
    background-color: #262930;
}

QMenuBar {
    background-color: #262930;
    color: #d6d6d6;
    border-bottom: 1px solid #3d4251;
}

QMenuBar::item {
    background-color: transparent;
    padding: 4px 10px;
}

QMenuBar::item:selected {
    background-color: #3d4251;
}

QMenu {
    background-color: #262930;
    color: #d6d6d6;
    border: 1px solid #3d4251;
}

QMenu::item:selected {
    background-color: #3d4251;
}

QTabWidget::pane {
    border: none;
    background-color: #262930;
}

QTabBar::tab {
    background-color: #1e2027;
    color: #d6d6d6;
    padding: 8px 16px;
    border: none;
    border-bottom: 2px solid transparent;
}

QTabBar::tab:selected {
    background-color: #262930;
    border-bottom: 2px solid #50a6ff;
}

QTabBar::tab:hover:!selected {
    background-color: #2d313b;
}

QPushButton {
    background-color: #3d4251;
    color: #d6d6d6;
    border: none;
    padding: 5px 12px;
    border-radius: 3px;
}

QPushButton:hover {
    background-color: #4a5168;
}

QPushButton:pressed {
    background-color: #50a6ff;
}

QPushButton:disabled {
    background-color: #2d313b;
    color: #6b6b6b;
}

QStatusBar {
    background-color: #1e2027;
    color: #d6d6d6;
    border-top: 1px solid #3d4251;
}
"""


class EmbeddedViewerTab(QWidget):
    """A tab containing an embedded napari viewer with an optional side widget."""

    def __init__(self, title: str = "Viewer", parent=None):
        super().__init__(parent)
        self.title = title

        # Create napari viewer without showing its window
        self.viewer = napari.Viewer(show=False, title=title)

        # Get the Qt widget from the viewer
        self._qt_viewer = self.viewer.window._qt_window

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._qt_viewer)

    def add_dock_widget(self, widget: QWidget, area: str = "right", name: str = "Widget"):
        """Add a dock widget to this tab's viewer."""
        self.viewer.window.add_dock_widget(widget, area=area, name=name)

    def close_viewer(self):
        """Clean up the viewer when done."""
        try:
            self.viewer.close()
        except Exception:
            pass


class AutoLamellaEmbeddedExample(QMainWindow):
    """
    Example showing how to embed the actual AutoLamellaUI and Protocol Editor
    in a single window with tabs.

    Note: This requires a valid microscope connection to fully work.
    For demo purposes, you can use manufacturer="Demo".
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"AutoLamella v{fibsem.__version__} ")
        self.resize(1600, 1000)

        # Apply napari-style dark theme
        self.setStyleSheet(NAPARI_STYLE)

        # Central tab widget
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        self.viewers = []
        self.autolamella_ui = None
        self.minimap_widget = None
        self.minimap_viewer = None

        # Toast notification manager
        self.toast_manager = ToastManager(self)

        # Create menu bar (includes notification bell)
        self._create_menu_bar()

        # Create status bar
        self._create_status_bar()

        # Tab 1: Main AutoLamella viewer with AutoLamellaUI docked
        self._create_main_tab()

        # Add buttons to add tabs dynamically
        self._add_tab_buttons()

        # Connect tab change to status bar update
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

    def _create_menu_bar(self):
        """Create the main menu bar."""
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("File")

        self.action_new_experiment = QAction("New Experiment", self)
        self.action_new_experiment.triggered.connect(self._on_new_experiment)
        file_menu.addAction(self.action_new_experiment)

        self.action_load_experiment = QAction("Load Experiment", self)
        self.action_load_experiment.triggered.connect(self._on_load_experiment)
        file_menu.addAction(self.action_load_experiment)

        file_menu.addSeparator()

        self.action_load_protocol = QAction("Load Protocol", self)
        self.action_load_protocol.triggered.connect(self._on_load_protocol)
        file_menu.addAction(self.action_load_protocol)

        self.action_save_protocol = QAction("Save Protocol", self)
        self.action_save_protocol.triggered.connect(self._on_save_protocol)
        file_menu.addAction(self.action_save_protocol)

        file_menu.addSeparator()

        self.action_exit = QAction("Exit", self)
        self.action_exit.triggered.connect(self.close)
        file_menu.addAction(self.action_exit)

        # Test menu for toast notifications
        test_menu = menu_bar.addMenu("Test")

        self.action_toast_info = QAction("Toast: Info", self)
        self.action_toast_info.triggered.connect(lambda: self.show_toast("This is an info message", "info"))
        test_menu.addAction(self.action_toast_info)

        self.action_toast_success = QAction("Toast: Success", self)
        self.action_toast_success.triggered.connect(lambda: self.show_toast("Operation completed successfully!", "success"))
        test_menu.addAction(self.action_toast_success)

        self.action_toast_warning = QAction("Toast: Warning", self)
        self.action_toast_warning.triggered.connect(lambda: self.show_toast("Warning: Check your settings", "warning"))
        test_menu.addAction(self.action_toast_warning)

        self.action_toast_error = QAction("Toast: Error", self)
        self.action_toast_error.triggered.connect(lambda: self.show_toast("Error: Something went wrong", "error"))
        test_menu.addAction(self.action_toast_error)

    def show_toast(self, message: str, notification_type: str = "info", duration: int = 5000):
        """Show a toast notification."""
        self.toast_manager.show_toast(message, notification_type, duration)

    def _on_new_experiment(self):
        """Handle New Experiment action."""
        if self.autolamella_ui is not None:
            self.autolamella_ui.create_experiment()

    def _on_load_experiment(self):
        """Handle Load Experiment action."""
        if self.autolamella_ui is not None:
            self.autolamella_ui.load_experiment()

    def _on_load_protocol(self):
        """Handle Load Protocol action."""
        if self.autolamella_ui is not None:
            self.autolamella_ui.load_protocol()

    def _on_save_protocol(self):
        """Handle Save Protocol action."""
        if self.autolamella_ui is not None:
            self.autolamella_ui.export_protocol_ui()

    def _create_status_bar(self):
        """Create the status bar."""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Hello Microscope Control")

    def _on_tab_changed(self, index: int):
        """Handle tab change and update status bar."""
        tab_name = self.tab_widget.tabText(index)
        self.status_bar.showMessage(f"Hello {tab_name}")

    def _create_main_tab(self):
        """Create the main AutoLamella tab."""
        # Create the embedded viewer container
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create napari viewer for main UI
        self.main_viewer = napari.Viewer(show=False, title="AutoLamella Main")
        self.main_viewer.window._qt_window.menuBar().hide()
        self.main_viewer.window._qt_window.statusBar().hide()
        self.viewers.append(self.main_viewer)

        # Import and create AutoLamellaUI (requires fibsem to be installed)
        try:
            from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI

            # Create the AutoLamellaUI widget
            self.autolamella_ui = AutoLamellaUI(viewer=self.main_viewer)

            # Connect to workflow update signal from AutoLamellaUI
            if self.autolamella_ui is not None:
                self.autolamella_ui.workflow_update_signal.connect(self._on_workflow_update)
                self.autolamella_ui.experiment_update_signal.connect(self._on_experiment_update)

            # Add it as a dock widget to the viewer
            self.main_viewer.window.add_dock_widget(
                widget=self.autolamella_ui,
                area="right",
                add_vertical_stretch=True,
                name="AutoLamella"
            )
        except ImportError as e:
            print(f"Could not import AutoLamellaUI: {e}")
            # Add placeholder
            from PyQt5.QtWidgets import QLabel
            self.main_viewer.window.add_dock_widget(
                QLabel("AutoLamellaUI not available"),
                area="right",
                name="Placeholder"
            )

        # Add the viewer's Qt window to our layout
        layout.addWidget(self.main_viewer.window._qt_window)

        # Add status bar for this tab
        self.main_status_bar = QStatusBar()
        self.main_status_bar.showMessage("Microscope Control: Ready")
        layout.addWidget(self.main_status_bar)

        self.tab_widget.addTab(container, QIconifyIcon("mdi:microscope", color="#d6d6d6"), "Microscope")

    def _on_experiment_update(self):

        self._on_add_minimap_clicked()
        self._on_add_protocol_editor_clicked()
        self._on_add_lamella_clicked()
        self._on_add_workflow_clicked()

    def _add_tab_buttons(self):
        """Add buttons to the tab bar for adding Protocol Editor, Lamella, and Minimap tabs."""
        # Create button container widget
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)
        button_layout.setContentsMargins(5, 0, 5, 0)
        button_layout.setSpacing(5)

        # Protocol Editor button
        self.btn_add_protocol_editor = QPushButton("+ Protocol Editor")
        self.btn_add_protocol_editor.clicked.connect(self._on_add_protocol_editor_clicked)
        button_layout.addWidget(self.btn_add_protocol_editor)

        # Lamella button
        self.btn_add_lamella = QPushButton("+ Lamella")
        self.btn_add_lamella.clicked.connect(self._on_add_lamella_clicked)
        button_layout.addWidget(self.btn_add_lamella)

        # Workflow button
        self.btn_add_workflow = QPushButton("+ Workflow")
        self.btn_add_workflow.clicked.connect(self._on_add_workflow_clicked)
        button_layout.addWidget(self.btn_add_workflow)

        # Minimap button
        self.btn_add_minimap = QPushButton("+ Minimap")
        self.btn_add_minimap.clicked.connect(self._on_add_minimap_clicked)
        button_layout.addWidget(self.btn_add_minimap)

        # Notification bell
        self.notification_bell = NotificationBell(self)
        self.toast_manager.set_notification_bell(self.notification_bell)
        button_layout.addWidget(self.notification_bell)

        # Add to tab widget corner
        self.tab_widget.setCornerWidget(button_widget)

    def _on_add_protocol_editor_clicked(self):
        """Handle click on Add Protocol Editor button."""
        self.add_protocol_editor_tab()
        self.btn_add_protocol_editor.setEnabled(False)
        self.btn_add_protocol_editor.setText("Protocol Editor ✓")

    def _on_add_lamella_clicked(self):
        """Handle click on Add Lamella button."""
        self.add_lamella_tab()
        self.btn_add_lamella.setEnabled(False)
        self.btn_add_lamella.setText("Lamella ✓")

    def _on_add_workflow_clicked(self):
        """Handle click on Add Workflow button."""
        self.add_workflow_tab()
        self.btn_add_workflow.setEnabled(False)
        self.btn_add_workflow.setText("Workflow ✓")

    def _on_add_minimap_clicked(self):
        """Handle click on Add Minimap button."""
        self.add_minimap_tab()
        self.btn_add_minimap.setEnabled(False)
        self.btn_add_minimap.setText("Minimap ✓")

    def add_protocol_editor_tab(self):
        """Add the protocol editor as a separate tab with its own viewer."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create separate napari viewer for protocol editor
        self.editor_viewer = napari.Viewer(show=False, title="Protocol Editor")
        self.editor_viewer.window._qt_window.menuBar().hide()
        self.editor_viewer.window._qt_window.statusBar().hide()
        self.viewers.append(self.editor_viewer)

        try:

            # Create the protocol editor widgets (without lamella widget - it's in its own tab now)
            task_widget = AutoLamellaProtocolTaskConfigEditor(
                viewer=self.editor_viewer,
                parent=self.autolamella_ui
            )

            # Add to viewer dock
            self.editor_viewer.window.add_dock_widget(
                task_widget,
                area='right',
                name='Protocol Editor'
            )

        except Exception as e:
            print(f"Could not create protocol editor: {e}")
            from PyQt5.QtWidgets import QLabel
            self.editor_viewer.window.add_dock_widget(
                QLabel(f"Protocol Editor not available: {e}"),
                area="right",
                name="Placeholder"
            )

        layout.addWidget(self.editor_viewer.window._qt_window)

        # Add status bar for this tab
        self.editor_status_bar = QStatusBar()
        self.editor_status_bar.showMessage("Protocol Editor: No protocol loaded")
        layout.addWidget(self.editor_status_bar)

        self.tab_widget.addTab(container, QIconifyIcon("mdi:file-document-edit", color="#d6d6d6"), "Protocol")

    def add_lamella_tab(self):
        """Add the lamella editor as a separate tab with its own viewer."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create separate napari viewer for lamella editor
        self.lamella_viewer = napari.Viewer(show=False, title="Lamella Editor")
        self.lamella_viewer.window._qt_window.menuBar().hide()
        self.lamella_viewer.window._qt_window.statusBar().hide()
        self.viewers.append(self.lamella_viewer)

        try:
            from fibsem.ui.widgets.autolamella_lamella_protocol_editor import (
                AutoLamellaProtocolEditorWidget
            )

            # Create the lamella editor widget
            lamella_widget = AutoLamellaProtocolEditorWidget(
                viewer=self.lamella_viewer,
                parent=self.autolamella_ui
            )

            # Store reference in the protocol editor widget if it exists
            if hasattr(self.autolamella_ui, 'protocol_editor_widget') and self.autolamella_ui.protocol_editor_widget:
                self.autolamella_ui.protocol_editor_widget.lamella_widget = lamella_widget

            # Add to viewer dock
            self.lamella_viewer.window.add_dock_widget(
                lamella_widget,
                area='right',
                name='Lamella Editor'
            )

        except Exception as e:
            print(f"Could not create lamella editor: {e}")
            import traceback
            traceback.print_exc()
            self.lamella_viewer.window.add_dock_widget(
                QLabel(f"Lamella Editor not available: {e}"),
                area="right",
                name="Placeholder"
            )

        layout.addWidget(self.lamella_viewer.window._qt_window)

        # Add status bar for this tab
        self.lamella_status_bar = QStatusBar()
        self.lamella_status_bar.showMessage("Lamella: No lamella selected")
        layout.addWidget(self.lamella_status_bar)

        self.tab_widget.addTab(container, QIconifyIcon("mdi:layers", color="#d6d6d6"), "Lamella")

    def add_workflow_tab(self):
        """Add an empty workflow tab with just a header."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header label
        header = QLabel("Workflow")
        header.setStyleSheet("color: #d6d6d6; font-size: 18px; font-weight: bold;")
        layout.addWidget(header)

        try:
            self.workflow_widget = AutoLamellaWorkflowWidget(
                experiment=self.autolamella_ui.experiment,
                parent=self.autolamella_ui
            )
            if self.autolamella_ui is not None and self.autolamella_ui.experiment is not None:
                self.workflow_widget.workflow_config_changed.connect(self.autolamella_ui._on_workflow_config_changed)
                self.workflow_widget.workflow_options_changed.connect(self.autolamella_ui._on_workflow_options_changed)
                self.workflow_widget.setStyleSheet(NAPARI_STYLE)

        except Exception as e:
            print(f"Could not create protocol editor: {e}")
            self.workflow_widget = QLabel(f"Workflow widget not available: {e}")

        # add horizontal layout for workflow widget
        grid_layout = QHBoxLayout()
        if hasattr(self, 'workflow_widget') and self.workflow_widget is not None:
            grid_layout.addWidget(self.workflow_widget)
            # add horizontal stretch
            grid_layout.addStretch()
        layout.addLayout(grid_layout)

        # Add stretch to push header to top
        layout.addStretch()

        # Add status bar for this tab
        self.workflow_status_bar = QStatusBar()
        self.workflow_status_bar.showMessage("Workflow: No workflow running")
        layout.addWidget(self.workflow_status_bar)


        self.tab_widget.addTab(container, QIconifyIcon("mdi:play-circle-outline", color="#d6d6d6"), "Workflow")

    def _on_workflow_update(self, info: dict):
        """Handle workflow update signal and update the workflow status bar."""
        status_msg = info.get("status", None)
        if status_msg is not None:
            task_name = status_msg.get("task_name", "Unknown Task")
            lamella_name = status_msg.get("lamella_name", "Unknown Lamella")
            current_lamella_index = status_msg.get("current_lamella_index", None)
            total_lamellae = status_msg.get("total_lamellas", None)
            current_task_index = status_msg.get("current_task_index", None)
            total_tasks = status_msg.get("total_tasks", None)
            error_msg = status_msg.get("error_message", None)
            msg = info.get("msg", "No message")
            status = status_msg.get("status", "info")

            txt = f"Workflow: {task_name} | {lamella_name}"
            if current_lamella_index is not None and total_lamellae is not None:
                txt += f" ({current_lamella_index + 1}/{total_lamellae})"
            if current_task_index is not None and total_tasks is not None:
                txt += f" | Task {current_task_index + 1}/{total_tasks}"

            if hasattr(self, 'workflow_status_bar'):
                self.workflow_status_bar.showMessage(txt)

            # Show toast notification - error if present, otherwise info
            msg_type = "info"
            if status is AutoLamellaTaskStatus.Completed:
                msg_type = "success"
            if status is AutoLamellaTaskStatus.Failed:
                msg_type = "error"
                msg = error_msg if error_msg is not None else msg
            toast_msg = f"{task_name} - {lamella_name}: {msg}"
            self.show_toast(toast_msg, msg_type)

    def add_minimap_tab(self):
        """Add the minimap as a separate tab with its own viewer."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create separate napari viewer for minimap
        self.minimap_viewer = napari.Viewer(show=False, title="AutoLamella Minimap")
        self.minimap_viewer.window._qt_window.menuBar().hide()
        self.minimap_viewer.window._qt_window.statusBar().hide()
        self.viewers.append(self.minimap_viewer)

        try:
            from fibsem.ui import FibsemMinimapWidget

            # Create the minimap widget
            self.minimap_widget = FibsemMinimapWidget(
                viewer=self.minimap_viewer,
                parent=self.autolamella_ui
            )

            # Store reference in the parent AutoLamellaUI
            self.autolamella_ui.minimap_widget = self.minimap_widget
            self.autolamella_ui.viewer_minimap = self.minimap_viewer

            # Add to viewer dock
            self.minimap_viewer.window.add_dock_widget(
                self.minimap_widget,
                area='right',
                add_vertical_stretch=True,
                name='AutoLamella Minimap'
            )

        except Exception as e:
            print(f"Could not create minimap: {e}")
            import traceback
            traceback.print_exc()
            self.minimap_viewer.window.add_dock_widget(
                QLabel(f"Minimap not available: {e}"),
                area="right",
                name="Placeholder"
            )

        layout.addWidget(self.minimap_viewer.window._qt_window)

        # Add status bar for this tab
        self.minimap_status_bar = QStatusBar()
        self.minimap_status_bar.showMessage("Minimap: No overview acquired")
        layout.addWidget(self.minimap_status_bar)

        self.tab_widget.addTab(container, QIconifyIcon("mdi:map", color="#d6d6d6"), "Overview")

    def closeEvent(self, event):
        """Clean up viewers on close."""
        for viewer in self.viewers:
            try:
                viewer.close()
            except Exception:
                pass
        super().closeEvent(event)


def run_autolamella_example():
    """Run the AutoLamella embedded example."""
    app = QApplication.instance() or QApplication(sys.argv)
    window = AutoLamellaEmbeddedExample()
    window.show()
    app.exec_()


if __name__ == "__main__":
    run_autolamella_example()
