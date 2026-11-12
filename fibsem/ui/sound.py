"""Sounds the app makes to get the operator's attention."""

from __future__ import annotations

from qtpy.QtWidgets import QApplication


def play_notification_sound():
    """Play a notification sound to alert the user."""
    QApplication.beep()
