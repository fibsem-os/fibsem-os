"""The tab corner: which instrument and experiment this session is on, a newer
release when there is one, and the notification bell."""

from __future__ import annotations

import html
from typing import Callable, List, Tuple

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QIcon, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QHBoxLayout,
    QMenu,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from fibsem.applications.autolamella.structures import Experiment
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.stylesheets import (
    ACCENT_CHIP_BUTTON_STYLESHEET,
    MENU_BUTTON_STYLESHEET,
    PRIMARY_BUTTON_STYLESHEET,
)
from fibsem.ui.tokens import ACCENT_COLOR, ERROR_COLOR, TEXT_MUTED_COLOR
from fibsem.ui.widgets.notifications import NotificationBell, ToastManager
from fibsem.util.timestamps import format_time

# How wide the experiment name button in the tab corner is allowed to grow. Wide
# enough that a default name -- "AutoLamella-" plus a date stamp -- is never elided;
# the point of the button is to say which experiment is open.
EXPERIMENT_MENU_MAX_WIDTH = 360
# Icons sat on a button, rather than in a menu, are drawn at this size.
BUTTON_ICON_SIZE = 16


def experiment_tooltip(experiment: Experiment) -> str:
    """The hover card for the tab-corner experiment button.

    Rich text, because Qt renders a tooltip as HTML the moment it contains a tag:
    that is what lines the rows up and keeps the directory on a line of its own
    instead of letting it set the width of everything above it.

    Every row here is a fact the experiment already holds. Rows whose fact is
    missing are dropped rather than rendered empty -- a protocol is set after
    construction and `created_at` is absent from experiments written before it was
    recorded, so both are genuinely unknown rather than blank.
    """
    rows: List[Tuple[str, str]] = []

    created = format_time(experiment.created_at, "%d %b %Y, %H:%M")
    if created:
        rows.append(("Created", html.escape(created)))

    lamella = str(len(experiment.positions))
    # `is_failure` is a human's judgement that a lamella is defective, not a record
    # of a task that failed, so the count says defective. See Lamella.is_failure.
    defective = len(experiment.at_failure())
    if defective:
        lamella += (
            f"&nbsp;&nbsp;<span style='color: {ERROR_COLOR}'>"
            f"{defective} defective</span>"
        )
    rows.append(("Lamella", lamella))

    if experiment.task_protocol is not None:
        rows.append(("Protocol", html.escape(experiment.task_protocol.name)))

    body = "".join(
        f"<tr><td style='color: {TEXT_MUTED_COLOR}; padding-right: 10px;'>{label}</td>"
        f"<td>{value}</td></tr>"
        for label, value in rows
    )
    return (
        f"<b>{html.escape(experiment.name)}</b>"
        f"<table cellspacing='0' cellpadding='0'>{body}</table>"
        f"<div style='color: {TEXT_MUTED_COLOR}; margin-top: 4px;'>"
        f"{html.escape(str(experiment.path))}</div>"
    )


def set_button_icon(
    button: QPushButton, key: str, gap: int = 5, color: str = TEXT_MUTED_COLOR
) -> None:
    """Put an icon on a button with breathing room between it and the text.

    Qt spaces a button's icon from its label by about four pixels and offers no way
    to ask for more, which next to a name reads as the two being squashed together.
    Padding the pixmap on its right buys the rest -- and the icon size has to grow
    with the padding, or Qt scales the wider pixmap back into the old box and
    shrinks the glyph instead of moving the text.

    Muted by default: at the label's own weight the icon competes with the name
    rather than introducing it.
    """
    pixmap = fibsem_icon(key, color=color).pixmap(BUTTON_ICON_SIZE, BUTTON_ICON_SIZE)
    ratio = pixmap.devicePixelRatio() or 1.0
    padded = QPixmap(pixmap.width() + round(gap * ratio), pixmap.height())
    padded.setDevicePixelRatio(ratio)
    padded.fill(Qt.transparent)
    painter = QPainter(padded)
    painter.drawPixmap(0, 0, pixmap)
    painter.end()

    button.setIcon(QIcon(padded))
    button.setIconSize(QSize(BUTTON_ICON_SIZE + gap, BUTTON_ICON_SIZE))


class ExperimentHeader(QWidget):
    """The window's tab-corner widget. It is told the session's experiment and
    microscope (``refresh``); what its buttons do is the window's."""

    def __init__(
        self,
        *,
        new_experiment: QAction,
        load_experiment: QAction,
        open_experiment_directory: QAction,
        on_new_experiment: Callable[[], None],
        on_load_experiment: Callable[[], None],
        on_about: Callable[[], None],
        toast_manager: ToastManager,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 0, 5, 0)
        layout.setSpacing(5)

        # Create / Load experiment buttons
        self.btn_create_experiment = QPushButton("Create Experiment")
        self.btn_create_experiment.setToolTip("Create a new experiment")
        self.btn_create_experiment.setEnabled(False)
        self.btn_create_experiment.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self.btn_create_experiment.clicked.connect(on_new_experiment)

        self.btn_load_experiment = QPushButton("Load Experiment")
        self.btn_load_experiment.setToolTip("Load an existing experiment")
        self.btn_load_experiment.setEnabled(False)
        self.btn_load_experiment.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self.btn_load_experiment.clicked.connect(on_load_experiment)

        # The experiment name, once there is one, is itself the control that reaches
        # the create/load actions -- see update_experiment_header.
        self.btn_experiment_menu = QPushButton()
        set_button_icon(self.btn_experiment_menu, "mdi:flask-outline")
        self.btn_experiment_menu.setStyleSheet(MENU_BUTTON_STYLESHEET)
        self.btn_experiment_menu.setMaximumWidth(EXPERIMENT_MENU_MAX_WIDTH)
        # Hug the name. A QPushButton's default policy lets it grow to fill the
        # layout, which here means straight back out to the maximum width above --
        # the corner widget would be no narrower than the pair it replaces.
        self.btn_experiment_menu.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.btn_experiment_menu.setVisible(False)
        experiment_menu = QMenu(self.btn_experiment_menu)
        # The File menu's own QAction objects, not copies of them, so the two places
        # that offer these actions cannot drift apart.
        experiment_menu.addAction(new_experiment)
        experiment_menu.addAction(load_experiment)
        experiment_menu.addSeparator()
        experiment_menu.addAction(open_experiment_directory)
        self.btn_experiment_menu.setMenu(experiment_menu)

        # A newer release, once the startup check has found one -- see
        # update_version_chip. Hidden until then, and for good when the check is off,
        # offline or finds nothing: none of those is anything to act on.
        self.btn_update_available = QPushButton()
        set_button_icon(
            self.btn_update_available, "mdi:arrow-up-circle-outline", color=ACCENT_COLOR
        )
        self.btn_update_available.setStyleSheet(ACCENT_CHIP_BUTTON_STYLESHEET)
        self.btn_update_available.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.btn_update_available.setVisible(False)
        # The About dialog already says what is installed, what is out and how to
        # update; the chip only points there.
        self.btn_update_available.clicked.connect(on_about)

        # Notification bell
        self.notification_bell = NotificationBell(self)
        toast_manager.set_notification_bell(self.notification_bell)

        # Add widgets to layout. The update chip leads, so it never sits beside the
        # experiment it says nothing about.
        layout.addWidget(self.btn_update_available)
        layout.addWidget(self.btn_experiment_menu)
        layout.addWidget(self.btn_create_experiment)
        layout.addWidget(self.btn_load_experiment)
        layout.addWidget(self.notification_bell)

    def refresh(self, experiment, microscope) -> None:
        self.update_experiment_header(experiment, microscope)

    def update_experiment_header(self, experiment, microscope) -> None:
        """Show either the create/load buttons or the experiment menu, never both.

        Until an experiment exists those two buttons are the only thing to do, so they
        stay primary-coloured and take the room. Once one is loaded the experiment name
        replaces them, and opens a menu holding the same actions: the greyed-out pair
        cost a third of the tab bar to say nothing.
        """
        is_connected = microscope is not None

        # Before a microscope connects they stay visible but disabled: the Connection
        # tab is where connecting happens.
        show = experiment is None
        for button in (self.btn_create_experiment, self.btn_load_experiment):
            button.setVisible(show)
            button.setEnabled(experiment is None and is_connected)

        self.btn_experiment_menu.setVisible(experiment is not None)
        if experiment is None:
            return

        # Experiment names carry a date stamp and run long; elided here rather than
        # left to stretch the corner widget back to the width this change reclaims.
        # Measured off the button rather than against a guessed allowance for the
        # icon, padding and chevron -- an allowance set too generously elides names
        # that would have fitted.
        self.btn_experiment_menu.setText(experiment.name)
        overflow = (
            self.btn_experiment_menu.sizeHint().width() - EXPERIMENT_MENU_MAX_WIDTH
        )
        if overflow > 0:
            metrics = self.btn_experiment_menu.fontMetrics()
            self.btn_experiment_menu.setText(
                metrics.elidedText(
                    experiment.name,
                    Qt.ElideMiddle,
                    metrics.horizontalAdvance(experiment.name) - overflow,
                )
            )
        self.btn_experiment_menu.setToolTip(experiment_tooltip(experiment))

    def update_version_chip(self) -> None:
        """Show the newer release the update check found, or nothing.

        Reads the cached answer, so a check that failed or never ran lands in the same
        place as "up to date": the chip hidden.
        """
        from fibsem import update_check

        latest = update_check.get_available_update()
        self.btn_update_available.setVisible(latest is not None)
        if latest is not None:
            self.btn_update_available.setText(f"{latest} available")
            self.btn_update_available.setToolTip(
                f"fibsemOS {latest} is available. Click for details."
            )
