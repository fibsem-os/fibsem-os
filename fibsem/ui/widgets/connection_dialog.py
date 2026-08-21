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
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from fibsem import config as cfg
from fibsem import guided_setup, utils
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import MicroscopeSettings
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.stylesheets import (
    PRIMARY_BUTTON_STYLESHEET,
    PRIMARY_COLOR,
    SECONDARY_BUTTON_STYLESHEET,
)
from fibsem.ui.tokens import (
    BORDER_COLOR,
    ERROR_COLOR,
    OK_COLOR,
    PANEL_COLOR,
    SURFACE_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
)
from fibsem.ui.utils import message_box_ui, open_existing_file_dialog

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
        workflow_running: bool = False,
    ) -> None:
        """
        Args:
            microscope: the session already in progress, if there is one. The dialog
                is reached from the header chip as well as from an empty start, and
                changing the connection is most of what it is for once something is
                connected.
            workflow_running: whether the instrument is in the middle of doing
                something. Disconnecting is still allowed -- an instrument that has
                already gone away needs releasing precisely when a workflow thinks
                it is using it -- but it is not allowed quietly.
        """
        super().__init__(parent)
        # The window bar already says what the dialog is; the heading inside says
        # what to do in it, and neither repeats the other.
        self.setWindowTitle("Connect to Microscope")
        self.setModal(True)
        self.setMinimumWidth(DIALOG_WIDTH)

        self.microscope = microscope
        self.settings = settings
        self._workflow_running = workflow_running
        # Whether the session handed in is not the session handed back. Tells
        # "closed without touching anything" from "disconnected", which look
        # identical if the answer is only ever a microscope-or-None.
        self.changed = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        self.title_label = QLabel("Select Configuration")
        self.title_label.setStyleSheet(f"color: {TEXT_STRONG_COLOR}; font-size: 14px;")
        layout.addWidget(self.title_label)

        # Above the configuration row on purpose: on a fresh install there is no
        # configuration worth choosing yet, and the walkthrough is what makes one.
        layout.addWidget(self._build_first_run_offer())
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
        self.refresh_first_run_offer()

    # ── construction ────────────────────────────────────────────────────────

    def _build_first_run_offer(self) -> QFrame:
        """The guided-setup offer, which used to live on the Connection tab.

        It came with the tab (FIB-740, #472) and would have gone with it: a fresh
        install would have been left with Tools > Guided Setup, which is not
        somewhere a first-time user thinks to look. This dialog is what they reach
        first instead, so the offer follows it here.
        """
        frame = QFrame()
        frame.setObjectName("frame_first_run")
        frame.setStyleSheet(f"""
            QFrame#frame_first_run {{
                background-color: {PANEL_COLOR};
                border-radius: 6px;
                border: 1px solid {PRIMARY_COLOR};
            }}
        """)
        frame_layout = QHBoxLayout(frame)
        frame_layout.setContentsMargins(12, 10, 8, 10)
        frame_layout.setSpacing(10)

        icon = QLabel()
        icon.setStyleSheet("background: transparent; border: none;")
        icon.setFixedSize(20, 20)
        icon.setPixmap(fibsem_icon("mdi:auto-fix", color=PRIMARY_COLOR).pixmap(20, 20))
        frame_layout.addWidget(icon, 0, Qt.AlignTop)

        text = QVBoxLayout()
        text.setSpacing(2)
        title = QLabel("First time here?")
        title.setStyleSheet(
            f"background: transparent; border: none; color: {PRIMARY_COLOR}; "
            "font-weight: bold; font-size: 11px;"
        )
        subtitle = QLabel(
            "A guided walkthrough that configures fibsemOS to work with your "
            "microscope."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(
            f"background: transparent; border: none; color: {TEXT_MUTED_COLOR}; "
            "font-size: 10px;"
        )
        text.addWidget(title)
        text.addWidget(subtitle)
        frame_layout.addLayout(text, 1)

        self.guided_setup_button = QPushButton("Start Guided Setup")
        self.guided_setup_button.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {PRIMARY_COLOR};
                border: 1px solid {PRIMARY_COLOR};
                border-radius: 4px;
                padding: 5px 12px;
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background-color: rgba(0, 122, 204, 0.20); }}
        """)
        self.guided_setup_button.clicked.connect(self.run_guided_setup)
        frame_layout.addWidget(self.guided_setup_button, 0, Qt.AlignVCenter)

        self.dismiss_first_run_button = QToolButton()
        self.dismiss_first_run_button.setIcon(
            fibsem_icon("mdi:close", color=TEXT_MUTED_COLOR)
        )
        self.dismiss_first_run_button.setToolTip("Do not offer this again")
        self.dismiss_first_run_button.setAutoRaise(True)
        self.dismiss_first_run_button.setStyleSheet(
            "border: none; background: transparent;"
        )
        self.dismiss_first_run_button.clicked.connect(self._dismiss_first_run)
        frame_layout.addWidget(self.dismiss_first_run_button, 0, Qt.AlignTop)

        self.first_run_frame = frame
        frame.setVisible(False)
        return frame

    def refresh_first_run_offer(self, preferences=None) -> None:
        """Show the offer when the flag is on, nothing is configured, and it stands.

        The same three conditions the Connection tab applied, and separate for the
        same reason: the flag lives in the file whose absence means "fresh install",
        so folding the first two together suppresses the offer it enables.
        """
        if preferences is None:
            preferences = cfg.load_user_preferences()
        self.first_run_frame.setVisible(
            preferences.features.guided_setup_enabled
            and not preferences.display.guided_setup_dismissed
            and guided_setup.is_first_run()
        )

    def _dismiss_first_run(self) -> None:
        """Hide the offer, and record that it was declined."""
        self.first_run_frame.setVisible(False)
        guided_setup.dismiss_first_run()

    def run_guided_setup(self) -> Optional[str]:
        """Open the wizard, then select whatever it saved.

        The live microscope is handed over so the wizard can read the stage without
        opening a second client against the same instrument.
        """
        from fibsem.ui.widgets.guided_setup_dialog import open_guided_setup

        name = open_guided_setup(parent=self, microscope=self.microscope)
        if name is None:
            # Backing out is not declining, so the offer stays where it was.
            return None

        # Finishing writes the file `is_first_run` reads, so the offer goes now
        # rather than at the next start -- and the new configuration is the one
        # this dialog should be pointing at.
        self._populate_configurations()
        self.configuration_combo.setCurrentText(name)
        self.refresh_first_run_offer()
        return name

    def _build_configuration_row(self) -> QWidget:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)

        # No "Configuration" label in front of it: the heading above already says
        # Select Configuration, and the combo box holds configuration names.
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

        Laid out as the Connection tab's status card is -- framed panel, icon,
        bold line, muted line, and its green tick on a connection that succeeded --
        so this reads as part of the application rather than a second way of saying
        the same thing.
        """
        frame = QFrame()
        frame.setObjectName("frame_connection_details")
        frame.setStyleSheet(f"""
            QFrame#frame_connection_details {{
                background-color: {PANEL_COLOR};
                border-radius: 6px;
                border: 1px solid {BORDER_COLOR};
            }}
        """)

        frame_layout = QHBoxLayout(frame)
        frame_layout.setContentsMargins(12, 12, 12, 12)
        frame_layout.setSpacing(10)

        self.details_icon = QLabel()
        self.details_icon.setStyleSheet("border: none;")
        self.details_icon.setFixedSize(20, 20)
        frame_layout.addWidget(self.details_icon)

        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        self.details_title = QLabel()
        self.details_title.setStyleSheet(
            f"background-color: transparent; color: {TEXT_STRONG_COLOR}; "
            "font-weight: bold; font-size: 11px; border: none;"
        )
        self.details_subtitle = QLabel()
        self.details_subtitle.setStyleSheet(
            f"background-color: transparent; color: {TEXT_MUTED_COLOR}; "
            "font-size: 10px; border: none;"
        )
        text_layout.addWidget(self.details_title)
        text_layout.addWidget(self.details_subtitle)
        frame_layout.addLayout(text_layout)
        frame_layout.addStretch()

        return frame

    def _set_details(self, title: str, subtitle: str, icon: str, colour: str) -> None:
        """Fill the panel, in the Connection tab's own status-card language."""
        self.details_icon.setPixmap(fibsem_icon(icon, color=colour).pixmap(20, 20))
        self.details_title.setText(title)
        self.details_subtitle.setText(subtitle)

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
        resolves = self.configuration_path() is not None
        self.connect_button.setEnabled(resolves)
        self._refresh_details()

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

    def _refresh_details(self) -> None:
        """Say what is attached, or where Connect is about to go."""
        if self.microscope is not None:
            # A green tick, as the Connection tab's card has: the session was
            # established, and reporting that outcome is what this panel is for.
            # (The header chip in #515 stays wordless about it for a different
            # reason -- it is on screen for hours, and nothing watches the link.)
            info = self.microscope.system.info
            self._set_details(
                "Microscope connected",
                f"{info.manufacturer} {info.model} at {info.ip_address}".strip(),
                "mdi:check-circle",
                OK_COLOR,
            )
            self.details_subtitle.setToolTip(f"Serial number: {info.serial_number}")
            return

        if self.configuration_path() is None:
            self._set_details(
                "No configuration",
                f"{self.configuration_combo.currentText()} could not be found",
                "mdi:alert-circle",
                ERROR_COLOR,
            )
            return

        manufacturer, address = self._describe_configuration()
        self._set_details(
            "No microscope connected",
            f"{manufacturer} at {address}",
            "mdi:close-circle",
            TEXT_MUTED_COLOR,
        )

    def _apply_connection_state(self) -> None:
        """Two states, one dialog: nothing connected, or something already is.

        Opened from the header chip mid-session, the useful actions are leaving,
        swapping to another configuration, and letting the instrument go -- not the
        same dialog worded as though nothing had happened yet.
        """
        connected = self.microscope is not None

        self.disconnect_button.setVisible(connected)
        self.offline_button.setText("Close" if connected else "Not now")
        self._refresh_details()

        if not connected:
            self.connect_button.setText("Connect")
            self.offline_button.setToolTip(
                "Start without a microscope. Most of the application stays "
                "unavailable until you connect, from File \u2192 Connect to Microscope."
            )
            return

        # Reconnect, not Connect: with a session in progress this drops it and takes
        # the selected configuration instead, and the button should say so.
        self.connect_button.setText("Reconnect")
        self.offline_button.setToolTip("Leave the connection as it is")

    def disconnect_from_microscope(self) -> None:
        """Let the instrument go, and say so to whoever opened the dialog."""
        if self.microscope is None:
            return

        if not self._confirm_losing_the_session("Disconnect from"):
            return

        try:
            self.microscope.disconnect()
        except Exception as e:  # a failed teardown still ends the session here
            logging.warning(f"Error while disconnecting: {e}")

        self.microscope, self.settings = None, None
        self.changed = True
        self.accept()

    def _confirm_losing_the_session(self, verb: str) -> bool:
        """Ask before the session in progress ends.

        Both paths here end it: disconnecting outright, and reconnecting, which
        drops the old one to take the selected configuration. Neither is undoable
        from this dialog, and mid-workflow both pull the instrument out from under
        whatever is running -- so the question is the same one, worded for whichever
        button was pressed.
        """
        info = self.microscope.system.info
        text = f"{verb} {info.manufacturer} at {info.ip_address}?"
        if self._workflow_running:
            text += (
                "\n\nA workflow is running. This will not stop it cleanly, and "
                "anything it has not finished will fail."
            )
        return message_box_ui(title=f"{verb} microscope?", text=text, parent=self)

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
        # would hold the instrument against the new one -- and losing it is worth
        # the same question as losing it deliberately.
        if self.microscope is not None:
            if not self._confirm_losing_the_session("Reconnect to"):
                self._set_busy(False)
                self._show_message("", is_error=False)
                return
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
    workflow_running: bool = False,
) -> ConnectionResult:
    """Offer the connection, starting from whatever session is already in progress."""
    dialog = ConnectionDialog(
        parent=parent,
        microscope=microscope,
        settings=settings,
        workflow_running=workflow_running,
    )
    dialog.exec_()
    if not dialog.changed:
        return ConnectionResult(False, microscope, settings)
    return ConnectionResult(True, dialog.microscope, dialog.settings)
