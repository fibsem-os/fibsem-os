import logging
import os
import subprocess
import sys
from typing import Optional

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import QEvent, QObject, Qt
from PyQt5.QtGui import QImage, QWheelEvent
from PyQt5.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QMessageBox,
    QWidget,
)

from fibsem import config as cfg
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import BeamType, Parameter


def beam_choices(microscope: FibsemMicroscope, name: str, beam_type: BeamType) -> list:
    """The values a beam parameter can take: its device's choices, kept on the
    parameter (read when the beam was built and again when a dependency changes), so
    asking makes no instrument call. Empty when there is no such beam or parameter, or
    the parameter has no choices."""
    beam = microscope.beams.get(beam_type)
    parameter = None if beam is None else beam.parameters.get(name)
    if parameter is None or parameter.choices is None:
        return []
    return list(parameter.choices)


def beam_spec(
    spec: Parameter,
    microscope: Optional[FibsemMicroscope] = None,
    beam_type: Optional[BeamType] = None,
) -> Parameter:
    """``bound_spec`` with ``beam_type``'s beam as the beam: a field bound to a beam
    parameter (``device_parameter="beam.<name>"``) takes what that beam reports, else
    what every beam declares."""
    from fibsem.ui.widgets.form_builder import bound_spec

    beam = None if microscope is None else microscope.beams.get(beam_type)
    return bound_spec(spec, {"beam": beam})


def find_data(combo: QtWidgets.QComboBox, value) -> int:
    """The index of the item whose data equals ``value``, or -1.

    ``QComboBox.findData`` compares a Python object such as a resolution tuple by
    identity, so an equal tuple built elsewhere is not found.
    """
    for index in range(combo.count()):
        if combo.itemData(index) == value:
            return index
    return -1


def open_path_in_file_explorer(path: str) -> bool:
    """Open a directory (or file's location) in the system file explorer.

    Returns True on success. Cross-platform: uses ``open`` on macOS,
    ``os.startfile`` on Windows, and ``xdg-open`` elsewhere.
    """
    if not path or not os.path.isdir(path):
        logging.warning("Cannot open path in file explorer (not found): %s", path)
        return False
    try:
        if sys.platform.startswith("darwin"):
            subprocess.Popen(["open", path])
        elif os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", path])
        return True
    except Exception:
        logging.exception("Failed to open path in file explorer: %s", path)
        return False


_WHEEL_GUARD_PROPERTY = "_fibsem_wheel_guarded"


class WheelBlocker(QObject):
    """Event filter that stops the mouse wheel from changing a widget's value.

    Scrolling over a spinbox/combobox no longer changes it. The wheel event is
    forwarded to the enclosing scroll area instead, so the panel keeps scrolling
    rather than dead-zoning over every input.

    Blocking is unconditional, deliberately. An earlier design allowed a *focused*
    widget to still adjust, but Qt gives focus to the first focusable widget in a
    form automatically — so the first field in every panel stayed vulnerable to
    exactly the accidental-change bug this exists to prevent. Use the keyboard
    (arrow keys, or type a value) to adjust a focused widget.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._forwarding = False

    def eventFilter(self, watched, event):
        if event.type() != QEvent.Wheel:
            return super().eventFilter(watched, event)
        self._forward_to_scroll_area(watched, event)
        return True

    def _forward_to_scroll_area(self, watched, event) -> None:
        """Re-send the wheel event to the nearest scrolling ancestor, if any."""
        if self._forwarding:  # guard against re-entrancy
            return

        target = watched.parentWidget()
        while target is not None and not isinstance(target, QAbstractScrollArea):
            target = target.parentWidget()
        if target is None:
            return  # not inside a scroll area; simply swallow the event

        self._forwarding = True
        try:
            QApplication.sendEvent(target.viewport(), _clone_wheel_event(event))
        except Exception:
            logging.debug("Failed to forward wheel event", exc_info=True)
        finally:
            self._forwarding = False


def _clone_wheel_event(event: QWheelEvent) -> QWheelEvent:
    """Copy a wheel event, preserving both scroll axes and modifiers.

    The position stays in the source widget's coordinates; scroll areas key off the
    delta, not the position, so this is fine for scrolling purposes.
    """
    return QWheelEvent(
        event.pos(),
        event.globalPos(),
        event.pixelDelta(),
        event.angleDelta(),
        event.angleDelta().y(),
        Qt.Vertical,
        event.buttons(),
        event.modifiers(),
        event.phase(),
        event.source(),
        event.inverted(),
    )


def install_wheel_blocker(widget: QWidget) -> None:
    """Guard *widget* against accidental scroll-to-change. Idempotent.

    Also sets ``Qt.StrongFocus``: spinboxes and comboboxes default to
    ``Qt.WheelFocus``, so a passing scroll would otherwise move keyboard focus into
    the widget as well.
    """
    if widget.property(_WHEEL_GUARD_PROPERTY):
        return  # already guarded; a second filter would double-forward the event
    widget.setFocusPolicy(Qt.StrongFocus)
    widget.installEventFilter(WheelBlocker(parent=widget))
    widget.setProperty(_WHEEL_GUARD_PROPERTY, True)


def install_wheel_blocker_recursive(root: QWidget) -> None:
    """Guard every spinbox/combobox under *root*. Safe to call more than once.

    Useful for forms whose inputs are built elsewhere (e.g. generated UI code).
    Widgets added *after* this call are not covered — call it again if a form
    populates itself dynamically.
    """
    for child in root.findChildren((QAbstractSpinBox, QComboBox)):
        install_wheel_blocker(child)


def message_box_ui(
    title: str,
    text: str,
    buttons=QMessageBox.Yes | QMessageBox.No,
    parent: Optional[QWidget] = None,
) -> bool:
    msg = QMessageBox(parent=parent)
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setStandardButtons(buttons)
    msg.exec_()

    response = (
        True
        if (msg.clickedButton() == msg.button(QMessageBox.Yes))
        or (msg.clickedButton() == msg.button(QMessageBox.Ok))
        else False
    )

    return response


def open_existing_directory_dialog(
    msg: str = "Select a directory", path: str = cfg.LOG_PATH, parent=None
) -> str:
    path = QtWidgets.QFileDialog.getExistingDirectory(
        parent=parent, caption=msg, directory=path
    )
    return path


def open_existing_file_dialog(
    msg: str = "Select a file",
    path: str = cfg.LOG_PATH,
    _filter: str = "*yaml",
    parent: Optional[QWidget] = None,
) -> str:
    path, _ = QtWidgets.QFileDialog.getOpenFileName(
        parent=parent, caption=msg, directory=path, filter=_filter
    )
    return path


def open_save_file_dialog(
    msg: str = "Select a file",
    path: str = cfg.LOG_PATH,
    _filter: str = "*yaml",
    parent: Optional[QWidget] = None,
) -> str:
    path, _ = QtWidgets.QFileDialog.getSaveFileName(
        parent=parent,
        caption=msg,
        directory=path,
        filter=_filter,
    )
    return path


def open_information_dialog(
    microscope: Optional[FibsemMicroscope] = None,
    parent: Optional[QWidget] = None,
    application: str = "fibsemOS",
):
    """Show the about dialog. Works with or without a connected microscope."""
    from fibsem.ui.widgets.about_dialog import open_about_dialog

    open_about_dialog(microscope=microscope, application=application, parent=parent)


def arr_to_qimage(arr: np.ndarray) -> QImage:
    """An RGB888 QImage of `arr`, independent of it.

    Qt's `QImage(buffer, ...)` documents the buffer as having to outlive the image,
    which would make the cached result of this a dangling read once the ndarray is
    collected. PyQt5 does not work that way: sip copies the Python buffer, so the
    image owns its pixels and the source can go. Verified rather than assumed -- the
    bits pointer differs from the ndarray's, and mutating the ndarray in place leaves
    the image unchanged, including via `sip.voidptr`. An explicit `.copy()` here would
    be a second full-size memcpy for nothing.

    Worth knowing if this ever moves to PySide, where the same constructor can alias.
    """
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=2)
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    ih, iw, c = arr.shape
    return QImage(arr.data, iw, ih, iw * c, QImage.Format_RGB888)
