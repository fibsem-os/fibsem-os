"""Toasts are how the application talks, so there is no longer a way to turn them off.

``display.toasts_enabled`` defaulted to False, and ``show_toast`` deliberately bypasses
notification history -- so on a default install 165 call sites emitted feedback that
reached nowhere at all, not even the bell.

Removed rather than flipped. ``UserPreferences.to_dict`` is ``dataclasses.asdict``, so
every save writes every key, and merely opening an experiment triggers a save: a new
default would have reached only a machine that had never run the application, while
everyone who had run it kept the ``false`` already pinned in their file (FIB-781).
"""

import dataclasses
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QWidget  # noqa: E402

from fibsem.applications.autolamella.ui.AutoLamellaMainUI import (  # noqa: E402
    AutoLamellaSingleWindowUI,
)
from fibsem.config import DisplayPreferences, UserPreferences  # noqa: E402
from fibsem.ui.widgets.notifications import NotificationBell, ToastManager  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# --- the preference is gone, and gone in a way old files survive ----------------


def test_there_is_no_preference_for_turning_toasts_off():
    names = {f.name for f in dataclasses.fields(DisplayPreferences)}
    assert "toasts_enabled" not in names, sorted(names)


def test_a_save_can_no_longer_pin_the_old_value():
    """Why this is a removal rather than a flipped default.

    ``to_dict`` writes every field, so for as long as the field exists, a file written
    before the change goes on deciding the answer no matter what the default says.
    """
    assert "toasts_enabled" not in UserPreferences().to_dict()["display"]


def test_a_preferences_file_written_before_the_change_still_loads():
    """Every install that has ever opened an experiment has this key on disk."""
    prefs = UserPreferences.from_dict(
        {"display": {"toasts_enabled": False, "sound_enabled": True}}
    )
    assert not hasattr(prefs.display, "toasts_enabled")
    assert prefs.display.sound_enabled is True  # the rest of the file survives


def test_sound_is_still_a_preference():
    """The neighbour this was wrongly grouped with. An audio alert really is a taste,
    and removing the toast switch is not an argument for removing that one too."""
    assert DisplayPreferences().sound_enabled is False


# --- and the messages actually arrive -------------------------------------------


class _Host(QWidget):
    """A real window, a real ToastManager, running the real ``show_toast``.

    The method is borrowed rather than the real window built: that window wants a
    microscope, an experiment and every tab in the application, none of which bears
    on whether a message reaches the manager.
    """

    show_toast = AutoLamellaSingleWindowUI.show_toast

    def __init__(self):
        super().__init__()
        self.toast_manager = ToastManager(self)


@pytest.fixture()
def host(qapp):
    window = _Host()
    window.setGeometry(0, 0, 800, 600)
    window.show()
    yield window
    window.close()


def test_a_message_reaches_the_toast_manager(host):
    host.show_toast("Connected to microscope", "info")

    assert len(host.toast_manager.toasts) == 1


def test_the_bell_still_records_what_the_removed_branch_used_to(host):
    """The old ``elif`` added non-temporary messages to the bell by hand, so that
    history survived while toasts were suppressed. It went with the switch, and the
    claim replacing it is that ToastManager does this itself either way.
    """
    bell = NotificationBell()
    host.toast_manager.set_notification_bell(bell)

    host.show_toast("Milling finished", "info", temporary=False)

    assert bell.count == 1
    bell.deleteLater()


def test_a_temporary_message_stays_out_of_history(host):
    """``notification_service.show_toast`` emits with temporary=True -- the 165 sites
    this whole change is about. They are transient by design and must not fill the
    bell with validation chatter.
    """
    bell = NotificationBell()
    host.toast_manager.set_notification_bell(bell)

    host.show_toast("No microscope connected", "warning", temporary=True)

    assert bell.count == 0
    bell.deleteLater()
