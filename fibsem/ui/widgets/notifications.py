from __future__ import annotations

import sys
try:
    sys.modules.pop("PySide6.QtCore")
except Exception:
    pass

from datetime import datetime
from PyQt5.QtCore import QPropertyAnimation, QTimer, Qt, QPoint
from PyQt5.QtWidgets import (
    QApplication,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from fibsem.ui.icon import fibsem_icon
from fibsem.constants import TIME_DISPLAY
from fibsem.ui import stylesheets
from fibsem.ui.tokens import (
    ACCENT_COLOR,
    BORDER_COLOR,
    DISABLED_BG_COLOR,
    DISABLED_TEXT_COLOR,
    PANEL_COLOR,
    SURFACE_COLOR,
    TEXT_COLOR,
)

class ToastNotification(QWidget):
    """A toast notification widget that appears in the bottom-right corner."""

    # Notification types with colors
    TYPES = {
        "info": ACCENT_COLOR,                      # Blue
        "success": stylesheets.GREEN_COLOR,     # Green
        "warning": stylesheets.ORANGE_COLOR,    # Orange
        "error": "#f44336",                     # Red
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
        layout.addWidget(self.icon_label, 0, Qt.AlignTop)

        # Message label
        self.message_label = QLabel()
        self.message_label.setWordWrap(True)
        self.message_label.setMaximumWidth(400)
        layout.addWidget(self.message_label, 1)

        # Close button
        self.close_btn = QToolButton()
        self.close_btn.setIcon(fibsem_icon("mdi:close", color="#888"))
        self.close_btn.setFixedSize(20, 20)
        self.close_btn.clicked.connect(self.hide_toast)
        self.close_btn.setCursor(Qt.PointingHandCursor)
        layout.addWidget(self.close_btn, 0, Qt.AlignTop)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.container)

        # Neutralise the app-global NAPARI_STYLE `QWidget { background-color: #262930 }`
        # bleed: keep every surface transparent so it shows the #toast_container's #1e2027
        # (the id selector wins over this type rule).
        self.setStyleSheet("QWidget { background-color: transparent; }")

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
                background-color: {PANEL_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
            }}
        """)

        self.message_label.setStyleSheet(f"""
            QLabel {{
                background-color: transparent;
                color: {TEXT_COLOR};
                font-size: 13px;
            }}
        """)

        self.close_btn.setStyleSheet(f"""
            QToolButton {{
                background-color: transparent;
                color: #888;
                border: none;
                font-size: 16px;
                font-weight: bold;
            }}
            QToolButton:hover {{
                color: {TEXT_COLOR};
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

        # Position in bottom-right corner of parent (using global coordinates)
        if self.parent():
            parent_rect = self.parent().rect()
            # Convert to global screen coordinates
            global_pos = self.parent().mapToGlobal(QPoint(0, 0))
            x = global_pos.x() + parent_rect.width() - self.width() - 20
            y = global_pos.y() + parent_rect.height() - self.height() - 20
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
        header_label.setStyleSheet(f"color: {TEXT_COLOR}; font-weight: bold; font-size: 14px;")
        header_layout.addWidget(header_label)

        header_layout.addStretch()

        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {ACCENT_COLOR};
                border: none;
                font-size: 12px;
            }}
            QPushButton:hover {{
                color: #7bc0ff;
            }}
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
        self.empty_label.setStyleSheet(f"color: {DISABLED_TEXT_COLOR}; padding: 20px;")
        self.notifications_layout.insertWidget(0, self.empty_label)

        # Apply styling
        self.container.setStyleSheet(f"""
            #notification_popup {{
                background-color: {PANEL_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
            }}
            #notification_header {{
                border-bottom: 1px solid {BORDER_COLOR};
            }}
        """)
        # The app-global NAPARI_STYLE sets `QWidget { background-color: #262930 }`, which
        # otherwise bleeds into every child surface (header / scroll area / rows). Force them
        # transparent so they show the container's #1e2027; the container + item:hover keep
        # their opaque colour via higher-specificity id selectors.
        self.setStyleSheet("QWidget { background-color: transparent; }")

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
        color = ToastNotification.TYPES.get(notification_type, ACCENT_COLOR)
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
        msg_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 13px;")
        text_layout.addWidget(msg_label)

        time_label = QLabel(timestamp)
        time_label.setStyleSheet(f"color: {DISABLED_TEXT_COLOR}; font-size: 11px;")
        text_layout.addWidget(time_label)

        item_layout.addLayout(text_layout, 1)

        item.setStyleSheet(f"""
            #notification_item {{
                border-bottom: 1px solid {DISABLED_BG_COLOR};
            }}
            #notification_item:hover {{
                background-color: {SURFACE_COLOR};
            }}
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
        self.setFixedSize(28, 28)
        self.setCursor(Qt.PointingHandCursor)

        # Bell icon button using QToolButton with a Material Design icon
        self.bell_btn = QToolButton()
        self.bell_btn.setIcon(fibsem_icon("mdi:bell", color=TEXT_COLOR))
        self.bell_btn.setFixedSize(24, 24)
        self.bell_btn.setIconSize(self.bell_btn.size() * 0.7)
        self.bell_btn.clicked.connect(self._on_clicked)

        # Badge
        self.badge = QLabel("0")
        self.badge.setFixedSize(14, 14)
        self.badge.setAlignment(Qt.AlignCenter)
        self.badge.hide()

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.bell_btn)

        # Position badge in top-right
        self.badge.setParent(self)
        self.badge.move(14, 0)

        # Popup
        self.popup = NotificationHistoryPopup()
        self.popup.clear_btn.clicked.connect(self._on_clear_all)

        # Apply styling
        self._apply_style()

    def _apply_style(self):
        self.bell_btn.setStyleSheet(f"""
            QToolButton {{
                background-color: transparent;
                border: none;
            }}
            QToolButton:hover {{
                background-color: {BORDER_COLOR};
                border-radius: 18px;
            }}
        """)

        self.badge.setStyleSheet("""
            QLabel {
                background-color: #f44336;
                color: white;
                font-size: 10px;
                font-weight: bold;
                border-radius: 7px;
            }
        """)

    def add_notification(self, message: str, notification_type: str):
        """Add a notification and increment counter."""
        self.count += 1
        self.badge.setText(str(min(self.count, 99)))
        self.badge.show()

        # Add to popup history
        timestamp = datetime.now().strftime(TIME_DISPLAY)
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
    """Manages toast notifications, on whichever of the app's windows the operator is
    working in.

    A toast belongs to a window: it is that window's child and is stacked in that
    window's bottom-right corner. Anchoring every toast to the host window instead put
    notifications about work done elsewhere -- the coincidence viewer, correlation, the
    FM UI -- in the corner of a window the operator was not looking at, and on Windows
    a toast owned by the host drags the host's whole z-order group in front of the
    window that raised it (the reported "the AutoLamella GUI comes to the front when
    coincidence milling stops").
    """

    def __init__(self, parent: QWidget):
        self.parent = parent  # fallback anchor, and the window the bell lives on
        self.toasts: list[ToastNotification] = []
        self.spacing = 10
        self.notification_bell: NotificationBell = None
        self._last_active: QWidget = None
        app = QApplication.instance()
        if app is not None:
            app.focusChanged.connect(self._on_focus_changed)

    def set_notification_bell(self, bell: NotificationBell):
        """Set the notification bell to update."""
        self.notification_bell = bell

    def _on_focus_changed(self, _old: QWidget, now: QWidget) -> None:
        """Remember the last real window of ours the operator was in.

        Tracked rather than read off ``QApplication.activeWindow()`` at show time,
        because that is ``None`` whenever the app itself is not focused -- exactly the
        case this exists for, a run that finishes while the operator is in the
        microscope software. Only ``Qt.Window`` types qualify, which skips the app's
        own transient chrome (toasts, popovers, tooltips) and short-lived dialogs: a
        toast anchored to a dialog would be destroyed with it mid-fade.
        """
        if now is None:
            return
        window = now.window()
        if window is not None and window.windowType() == Qt.Window:
            self._last_active = window

    def _anchor(self) -> QWidget:
        """The window a new toast belongs to."""
        window = self._last_active
        try:
            # isMinimized: anchoring to a minimized window would place the toast at
            # that window's restored geometry, i.e. over nothing the operator can see.
            if window is None or not window.isVisible() or window.isMinimized():
                return self.parent
        except RuntimeError:  # window closed; the C++ side is already gone
            self._last_active = None
            return self.parent
        return window

    def show_toast(self, message: str, notification_type: str = "info", duration: int = 5000, temporary: bool = False):
        """Show a toast notification."""
        toast = ToastNotification(self._anchor(), duration)
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

        # Add to notification bell history — skip if temporary
        if self.notification_bell and not temporary:
            self.notification_bell.add_notification(message, notification_type)

    def _remove_toast(self, toast: ToastNotification):
        """Remove a toast from the list."""
        if toast in self.toasts:
            self.toasts.remove(toast)
            toast.deleteLater()
            self._reposition_toasts()

    def _reposition_toasts(self):
        """Stack each visible toast up the bottom-right corner of the window it
        belongs to.

        Grouped by anchor rather than run as one column: toasts raised from different
        windows are stacked on their own window, so two windows' notifications never
        share a column (or overlap, when the windows overlap).
        """
        alive = []
        for toast in self.toasts:
            try:
                toast.isVisible()
            except RuntimeError:
                continue  # destroyed along with the window it was anchored to
            alive.append(toast)
        self.toasts = alive

        y_offsets = {}  # anchor -> next free offset up from its bottom edge
        for toast in reversed(self.toasts):
            anchor = toast.parent()
            if not toast.isVisible() or anchor is None or not anchor.isVisible():
                continue
            y_offset = y_offsets.get(anchor, 60)  # leave room for notification bell
            anchor_rect = anchor.rect()
            # Convert to global screen coordinates
            global_pos = anchor.mapToGlobal(QPoint(0, 0))
            x = global_pos.x() + anchor_rect.width() - toast.width() - 20
            y = global_pos.y() + anchor_rect.height() - toast.height() - y_offset
            toast.move(x, y)
            y_offsets[anchor] = y_offset + toast.height() + self.spacing