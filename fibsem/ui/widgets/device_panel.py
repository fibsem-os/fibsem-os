"""Any device, drawn from what it describes about itself.

`DevicePanel` shows one device the way the device server and an agent see it: each
parameter as a control built by the shared form builder from its metadata (its
``Display``, limits and choices), kept current through the device's ``changed``
signal; each command that needs no arguments as a button, run off the GUI thread
after the user confirms it; and its ``facts`` as read-only text. It knows nothing of
any device type, so a plugin's device shows with nothing written for it.

`DevicesPanel` lists every device the microscope built, and shows the selected one.
The configuration window's Devices tab is one.
"""

from __future__ import annotations

import inspect
import json
import logging
import threading
from dataclasses import replace
from typing import Any, Dict, Mapping, Optional

from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from fibsem.ui.widgets.form_builder import Control, build_control, parameter_spec


class _Bridge(QObject):
    """Moves what arrives on other threads (a device's ``changed``, a command's
    result) onto the GUI thread."""

    changed = pyqtSignal(str, object)
    finished = pyqtSignal(str, str)


def no_argument_commands(device: Any) -> Dict[str, Any]:
    """The device's commands that can run with no arguments, by name, with their
    ``CommandInfo``."""
    found = {}
    for name, info in device.commands.items():
        signature = inspect.signature(getattr(device, name))
        if all(
            p.default is not inspect.Parameter.empty
            or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            for p in signature.parameters.values()
        ):
            found[name] = info
    return found


class DevicePanel(QWidget):
    """One device's parameters, commands and facts."""

    def __init__(self, device: Any, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.device = device
        self.controls: Dict[str, Control] = {}
        self.buttons: Dict[str, QPushButton] = {}
        self._bridge = _Bridge(self)
        self._bridge.changed.connect(self._show_value)
        self._bridge.finished.connect(self._show_result)

        layout = QVBoxLayout(self)
        heading = QLabel(f"<b>{device.name}</b>  ·  {_describe(device)}")
        heading.setTextFormat(Qt.RichText)
        layout.addWidget(heading)

        form = QFormLayout()
        for name, parameter in device.parameters.items():
            form.addRow(self._label(parameter), self._control(name, parameter))
        layout.addLayout(form)

        commands = QHBoxLayout()
        for name, info in no_argument_commands(device).items():
            button = QPushButton(name.replace("_", " "))
            button.setToolTip(info.doc)
            button.setEnabled(bool(info.available))
            button.clicked.connect(lambda _=False, name=name: self.run(name))
            commands.addWidget(button)
            self.buttons[name] = button
        commands.addStretch()
        layout.addLayout(commands)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.facts = QPlainTextEdit(_facts_text(device))
        self.facts.setReadOnly(True)
        self.facts.setMaximumHeight(140)
        layout.addWidget(QLabel("Facts"))
        layout.addWidget(self.facts)
        layout.addStretch()

        device.changed.connect(self._on_changed)
        self.destroyed.connect(lambda *_: _disconnect(device, self._on_changed))

    # -- parameters

    def _label(self, parameter: Any) -> str:
        return parameter_spec(parameter).display.label or parameter.name

    def _control(self, name: str, parameter: Any) -> QWidget:
        try:
            value = parameter.get_value()
        except Exception as error:  # a sick read shows as such, not as a crash
            return QLabel(f"unavailable: {error}")
        spec = replace(parameter_spec(parameter), type=parameter.type)
        control = build_control(spec, value)
        if control is None:
            return QLabel(str(value))
        if not parameter.settable:
            control.widget.setEnabled(False)
        elif control.editable:
            control.connect(lambda name=name: self._write(name))
        self.controls[name] = control
        return control.widget

    def _write(self, name: str) -> None:
        control = self.controls[name]
        try:
            self.device.parameters[name].set_value(control.read())
            self.status.setText("")
        except Exception as error:  # refused: say why, and show what it is
            self.status.setText(f"{name}: {error}")
            self._show_value(name, self.device.parameters[name].cached)

    def _on_changed(self, name: str, value: Any) -> None:
        try:
            self._bridge.changed.emit(name, value)
        except RuntimeError:  # the panel is gone
            _disconnect(self.device, self._on_changed)

    def _show_value(self, name: str, value: Any) -> None:
        control = self.controls.get(name)
        if control is None:
            return
        control.set_blocked(True)
        try:
            control.write(value)
        finally:
            control.set_blocked(False)

    # -- commands

    def run(self, name: str, confirm: bool = True) -> None:
        """Run a command off the GUI thread, once the user agrees; its result or
        error shows below the buttons."""
        if confirm:
            asked = QMessageBox.question(
                self, "Run command", f"Run {self.device.name}.{name}()?"
            )
            if asked != QMessageBox.Yes:
                return
        self.status.setText(f"{name}: running…")

        def work() -> None:
            try:
                result = getattr(self.device, name)()
                text = "done" if result is None else _short(result)
            except Exception as error:
                text = f"failed: {error}"
            try:
                self._bridge.finished.emit(name, text)
            except RuntimeError:  # the panel closed while it ran
                logging.info(f"{self.device.name}.{name}: {text}")

        threading.Thread(target=work, name=f"{name}-command", daemon=True).start()

    def _show_result(self, name: str, text: str) -> None:
        self.status.setText(f"{name}: {text}")


class DevicesPanel(QWidget):
    """Every device, listed by name; the selected one shown as a `DevicePanel`,
    built the first time it is selected."""

    def __init__(self, devices: Mapping[str, Any], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.devices = dict(devices)
        self.panels: Dict[str, DevicePanel] = {}
        self.list = QListWidget()
        self.list.addItems(list(self.devices))
        self.list.setMaximumWidth(200)
        self.stack = QStackedWidget()
        layout = QHBoxLayout(self)
        layout.addWidget(self.list)
        layout.addWidget(self.stack, 1)
        self.list.currentTextChanged.connect(self.show_device)
        if self.devices:
            self.list.setCurrentRow(0)

    def show_device(self, name: str) -> Optional[DevicePanel]:
        if name not in self.devices:
            return None
        panel = self.panels.get(name)
        if panel is None:
            panel = DevicePanel(self.devices[name])
            self.panels[name] = panel
            self.stack.addWidget(panel)
        self.stack.setCurrentWidget(panel)
        return panel


def _describe(device: Any) -> str:
    parts = [getattr(device, "entry_type", None) or type(device).__name__]
    driver = getattr(device, "driver", None)
    if driver:
        parts.append(driver)
    roles = getattr(device, "roles", {}) or {}
    parts += [f"{role}: {filled.name}" for role, filled in roles.items()]
    return "  ·  ".join(parts)


def _facts_text(device: Any) -> str:
    try:
        facts = device.facts()
    except Exception as error:
        return f"unavailable: {error}"
    if not facts:
        return "none"
    return json.dumps(facts, indent=2, default=str)


def _short(value: Any, limit: int = 300) -> str:
    text = (
        json.dumps(value, default=str)
        if isinstance(value, (dict, list))
        else str(value)
    )
    return text if len(text) <= limit else text[:limit] + "…"


def _disconnect(device: Any, slot: Any) -> None:
    try:
        device.changed.disconnect(slot)
    except Exception:  # already gone
        pass
