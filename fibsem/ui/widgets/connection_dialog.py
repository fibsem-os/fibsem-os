"""Connect to a microscope from a dialog, rather than from a tab.

Connecting is the first thing every session does, and until now the only way to do
it was the Connection tab -- the leftmost tab of the microscope panel, a gate
sitting in a bar that otherwise means "the instrument needs you here now". A
developer flag exists purely to skip it (``--quickstart`` connects "instead of
waiting for the Connection tab"), and no menu offered the action at all.

This dialog is the other door. It is deliberately dismissable: preferences, a
protocol or the logs are all reasons to open the application without hardware, and
a modal that cannot be closed turns those into a reason not to open it at all.

See FIB-775. The Connection tab is untouched by this -- swapping it out is separate
work, and this has to be usable first.
"""

from __future__ import annotations

import logging
from typing import NamedTuple, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import fibsem
from fibsem import config as cfg
from fibsem import utils
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import MicroscopeSettings
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.stylesheets import (
    PRIMARY_BUTTON_STYLESHEET,
    SECONDARY_BUTTON_STYLESHEET,
)
from fibsem.ui.tokens import (
    BORDER_COLOR,
    ERROR_COLOR,
    PANEL_COLOR,
    SURFACE_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
)
from fibsem.ui.utils import open_existing_file_dialog

DIALOG_WIDTH = 400


class ConnectionDialog(QDialog):
    """Pick a configuration, see what it points at, and connect to it.

    On success ``microscope`` and ``settings`` hold the session, and the dialog
    accepts. On failure it stays open carrying the reason, because the alternative
    is closing onto an application that cannot do anything.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        microscope: Optional[FibsemMicroscope] = None,
        settings: Optional[MicroscopeSettings] = None,
    ) -> None:
        """
        Args:
            microscope: the session already in progress, if there is one. The dialog
                is reached from the header chip as well as from an empty start, and
                changing the connection is most of what it is for once something is
                connected.
        """
        super().__init__(parent)
        self.setModal(True)
        self.setMinimumWidth(DIALOG_WIDTH)

        self.microscope = microscope
        self.settings = settings
        # Whether the session handed in is not the session handed back. Tells
        # "closed without touching anything" from "disconnected", which look
        # identical if the answer is only ever a microscope-or-None.
        self.changed = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        self.title_label = QLabel()
        self.title_label.setStyleSheet(f"color: {TEXT_STRONG_COLOR}; font-size: 14px;")
        # The plain version, not `get_version_string()` -- its git-describe suffix
        # belongs in Help > About, not on the first thing a session sees.
        version = QLabel(f"AutoLamella {fibsem.__version__}")
        version.setStyleSheet(f"color: {TEXT_MUTED_COLOR}; font-size: 12px;")
        layout.addWidget(self.title_label)
        layout.addWidget(version)

        layout.addWidget(self._build_configuration_row())
        layout.addWidget(self._build_details_panel())

        # Empty until something goes wrong, and it stays visible afterwards so the
        # reason is still there while you change the configuration and retry.
        self.message_label = QLabel()
        self.message_label.setWordWrap(True)
        self.message_label.setVisible(False)
        layout.addWidget(self.message_label)

        layout.addLayout(self._build_buttons())

        self._populate_configurations()
        self._apply_connection_state()

    # ── construction ────────────────────────────────────────────────────────

    def _build_configuration_row(self) -> QWidget:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)

        label = QLabel("Configuration")
        label.setStyleSheet(f"color: {TEXT_MUTED_COLOR}; font-size: 11px;")
        row_layout.addWidget(label)

        self.configuration_combo = QComboBox()
        self.configuration_combo.currentTextChanged.connect(
            self._on_configuration_changed
        )
        row_layout.addWidget(self.configuration_combo, stretch=1)

        self.import_button = QToolButton()
        self.import_button.setIcon(fibsem_icon("mdi:plus", color=TEXT_MUTED_COLOR))
        self.import_button.setToolTip("Import a configuration file")
        self.import_button.clicked.connect(self._on_import_clicked)
        row_layout.addWidget(self.import_button)

        return row

    def _build_details_panel(self) -> QFrame:
        """What this configuration points at, before committing to it.

        The tab never said: it offered a name and a Connect button, so which
        instrument a name meant was something you learned by connecting to it.
        """
        frame = QFrame()
        frame.setStyleSheet(
            f"background: {PANEL_COLOR}; border: 1px solid {BORDER_COLOR}; "
            "border-radius: 4px;"
        )
        frame_layout = QGridLayout(frame)
        frame_layout.setContentsMargins(10, 8, 10, 8)
        frame_layout.setHorizontalSpacing(12)
        frame_layout.setVerticalSpacing(2)

        self.manufacturer_label = QLabel()
        self.address_label = QLabel()
        # A grid rather than tab stops in one label: the columns have to line up
        # whatever the manufacturer is called.
        for row, (name, value) in enumerate(
            (("Manufacturer", self.manufacturer_label), ("Address", self.address_label))
        ):
            key = QLabel(name)
            for label in (key, value):
                label.setStyleSheet(
                    f"color: {TEXT_MUTED_COLOR}; font-size: 11px; border: none;"
                )
            frame_layout.addWidget(key, row, 0)
            frame_layout.addWidget(value, row, 1)
        frame_layout.setColumnStretch(1, 1)

        return frame

    def _build_buttons(self) -> QHBoxLayout:
        buttons = QHBoxLayout()
        buttons.setSpacing(8)

        # "Not now", not "Work offline": there is no offline mode to promise. With
        # no microscope, creating and loading an experiment are both disabled and
        # every tab stays shut, so this leaves you with preferences and the logs.
        # It dismisses the dialog; it does not open a way of working.
        self.offline_button = QPushButton("Not now")
        self.offline_button.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.offline_button.setToolTip(
            "Start without a microscope. Most of the application stays unavailable "
            "until you connect, from File → Connect to Microscope."
        )
        self.offline_button.clicked.connect(self.reject)
        buttons.addWidget(self.offline_button)

        buttons.addStretch(1)

        self.disconnect_button = QPushButton("Disconnect")
        self.disconnect_button.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.disconnect_button.setToolTip("Release the microscope and work with none")
        self.disconnect_button.clicked.connect(self.disconnect_from_microscope)
        buttons.addWidget(self.disconnect_button)

        self.connect_button = QPushButton("Connect")
        self.connect_button.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self.connect_button.setDefault(True)
        self.connect_button.clicked.connect(self.connect_to_microscope)
        buttons.addWidget(self.connect_button)

        return buttons

    # ── configuration ───────────────────────────────────────────────────────

    def _populate_configurations(self) -> None:
        self.configuration_combo.blockSignals(True)
        self.configuration_combo.clear()
        self.configuration_combo.addItems(cfg.USER_CONFIGURATIONS.keys())
        self.configuration_combo.setCurrentText(cfg.DEFAULT_CONFIGURATION_NAME)
        self.configuration_combo.blockSignals(False)
        self._on_configuration_changed(self.configuration_combo.currentText())

    def configuration_path(self) -> Optional[str]:
        """Where the selected configuration lives, or None if it resolves nowhere."""
        configuration = cfg.USER_CONFIGURATIONS.get(
            self.configuration_combo.currentText()
        )
        return configuration.get("path") if configuration else None

    def _on_configuration_changed(self, _name: str = "") -> None:
        """Read the selected configuration and show what it points at.

        Reads the file, not the microscope -- there is nothing connected yet, and
        the whole point is to say where Connect is about to go.
        """
        manufacturer, address = self._describe_configuration()
        self.manufacturer_label.setText(manufacturer)
        self.address_label.setText(address)
        self.connect_button.setEnabled(self.configuration_path() is not None)

    def _describe_configuration(self) -> Tuple[str, str]:
        path = self.configuration_path()
        if path is None:
            return "unknown", "unknown"
        try:
            settings = utils.load_microscope_configuration(path)
        except Exception as e:  # unreadable or malformed: say so rather than raise
            logging.warning(f"Could not read configuration at {path}: {e}")
            return "unreadable", "unreadable"
        return settings.system.info.manufacturer, settings.system.info.ip_address

    def _on_import_clicked(self) -> None:
        path = open_existing_file_dialog(
            msg="Select microscope configuration file",
            path=cfg.CONFIG_PATH,
            _filter="YAML (*.yaml *.yml)",
            parent=self,
        )
        if not path:
            return

        name = cfg.register_configuration(path=path)
        self._populate_configurations()
        self.configuration_combo.setCurrentText(name)

    # ── connecting ──────────────────────────────────────────────────────────

    def _apply_connection_state(self) -> None:
        """Two states, one dialog: nothing connected, or something already is.

        Opened from the header chip mid-session, the useful actions are leaving,
        swapping to another configuration, and letting the instrument go -- not the
        same dialog worded as though nothing had happened yet.
        """
        connected = self.microscope is not None

        self.disconnect_button.setVisible(connected)
        self.offline_button.setText("Close" if connected else "Not now")

        if not connected:
            self.title_label.setText("Connect to microscope")
            self.connect_button.setText("Connect")
            self.offline_button.setToolTip(
                "Start without a microscope. Most of the application stays "
                "unavailable until you connect, from File \u2192 Connect to Microscope."
            )
            return

        info = self.microscope.system.info
        self.title_label.setText(
            f"Connected to {info.manufacturer} at {info.ip_address}"
        )
        # Reconnect, not Connect: with a session in progress this drops it and takes
        # the selected configuration instead, and the button should say so.
        self.connect_button.setText("Reconnect")
        self.offline_button.setToolTip("Leave the connection as it is")

    def disconnect_from_microscope(self) -> None:
        """Let the instrument go, and say so to whoever opened the dialog."""
        if self.microscope is None:
            return

        try:
            self.microscope.disconnect()
        except Exception as e:  # a failed teardown still ends the session here
            logging.warning(f"Error while disconnecting: {e}")

        self.microscope, self.settings = None, None
        self.changed = True
        self.accept()

    def connect_to_microscope(self) -> None:
        path = self.configuration_path()
        if path is None:
            self._show_message(
                f"Configuration {self.configuration_combo.currentText()!r} could not "
                "be found.",
                is_error=True,
            )
            return

        manufacturer, address = self._describe_configuration()
        self._set_busy(True, f"Connecting to {address}…")

        # A reconnect is a disconnect and a connect. Leaving the old session open
        # would hold the instrument against the new one.
        if self.microscope is not None:
            try:
                self.microscope.disconnect()
            except Exception as e:
                logging.warning(f"Error while disconnecting before reconnect: {e}")
            self.microscope, self.settings = None, None

        try:
            self.microscope, self.settings = utils.setup_session(config_path=path)
        except Exception as e:
            logging.warning(f"Could not connect to microscope at {address}: {e}")
            self.microscope, self.settings = None, None
            self._set_busy(False)
            self._show_message(f"Could not reach {address}.\n{e}", is_error=True)
            self.connect_button.setText("Retry")
            return

        logging.info(f"Connected to microscope at {address}")
        self.changed = True
        self.accept()

    def _set_busy(self, busy: bool, message: str = "") -> None:
        """Say what is happening, then let Qt paint it before the call that blocks.

        `setup_session` runs on this thread, so the window is frozen for its
        duration -- the same as the Connection tab today. Painting first at least
        means the frozen window says what it is waiting for. Doing it off-thread is
        its own change.
        """
        for widget in (
            self.connect_button,
            self.offline_button,
            self.disconnect_button,
            self.configuration_combo,
            self.import_button,
        ):
            widget.setEnabled(not busy)
        if busy:
            self._show_message(message, is_error=False)
            QApplication.processEvents()

    def _show_message(self, message: str, is_error: bool) -> None:
        colour = ERROR_COLOR if is_error else TEXT_MUTED_COLOR
        self.message_label.setStyleSheet(
            f"color: {colour}; font-size: 11px; background: {SURFACE_COLOR}; "
            f"border-left: 3px solid {colour}; padding: 6px 8px;"
        )
        self.message_label.setText(message)
        self.message_label.setVisible(bool(message))


class ConnectionResult(NamedTuple):
    """What the dialog did, which is not the same as what it is holding.

    `changed` is the part a caller acts on: a disconnect and a cancel both end with
    no microscope, and applying the first while ignoring the second is the whole
    difference between letting the instrument go and being left holding it.
    """

    changed: bool
    microscope: Optional[FibsemMicroscope]
    settings: Optional[MicroscopeSettings]


def connect_to_microscope_dialog(
    parent: Optional[QWidget] = None,
    microscope: Optional[FibsemMicroscope] = None,
    settings: Optional[MicroscopeSettings] = None,
) -> ConnectionResult:
    """Offer the connection, starting from whatever session is already in progress."""
    dialog = ConnectionDialog(parent=parent, microscope=microscope, settings=settings)
    dialog.exec_()
    if not dialog.changed:
        return ConnectionResult(False, microscope, settings)
    return ConnectionResult(True, dialog.microscope, dialog.settings)
