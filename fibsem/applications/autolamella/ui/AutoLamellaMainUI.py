from __future__ import annotations

import argparse
import html
import logging
import sys
import time
from typing import List, Optional, Tuple

try:
    sys.modules.pop("PySide6.QtCore")
except Exception:
    pass

import warnings
from datetime import datetime

import napari
from PyQt5.QtCore import QSize, Qt, QTimer
from PyQt5.QtGui import QIcon, QKeySequence, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from superqt import ensure_main_thread

import fibsem
import fibsem.config as fibsem_cfg
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskStatus,
    Experiment,
    Lamella,
)
from fibsem.applications.autolamella.ui.autolamella_lamella_protocol_editor import (
    AutoLamellaProtocolEditorWidget,
)
from fibsem.applications.autolamella.ui.autolamella_task_config_editor import (
    AutoLamellaProtocolTaskConfigEditor,
)
from fibsem.applications.autolamella.ui.AutoLamellaUI import INSTRUCTIONS, AutoLamellaUI
from fibsem.applications.autolamella.ui.lamella_card_widget import LamellaCardContainer
from fibsem.applications.autolamella.ui.lamella_task_image_widget import (
    LamellaTaskImageWidget,
)
from fibsem.applications.autolamella.ui.lamella_workflow_widget import (
    LamellaWorkflowWidget,
)
from fibsem.applications.autolamella.ui.overview_container_tab import (
    AutoLamellaOverviewContainerTab,
)
from fibsem.applications.autolamella.ui.workflow_preflight_dialog import (
    WorkflowPreflightDialog,
)
from fibsem.applications.autolamella.ui.workflow_timeline_widget import (
    WorkflowProgressWidget,
)
from fibsem.applications.autolamella.workflows.tasks.queue import QueueOp, QueueResult
from fibsem.applications.autolamella.workflows.tasks.status import (
    WorkflowStatusEvent,
    WorkflowStatusUpdate,
)
from fibsem.applications.autolamella.workflows.tasks.tasks import get_task_supervision
from fibsem.applications.autolamella.workflows.workflow_estimate import (
    AdditionEstimate,
    estimate_addition,
    estimate_workflow,
)
from fibsem.imaging.spot import SpotBurnProgress
from fibsem.imaging.tiling.progress import TiledProgress, TiledStatus
from fibsem.milling.progress import (
    MillingMessageTracker,
    MillingProgress,
    MillingProgressStatus,
)
from fibsem.structures import BeamType
from fibsem.ui import notification_service
from fibsem.ui.FibsemMinimapWidget import FibsemMinimapWidget
from fibsem.ui.FibsemSpotBurnWidget import build_spot_burn_progress_update
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.qt.gc import install_main_thread_gc
from fibsem.ui.stylesheets import (
    DANGER_BUTTON_STYLESHEET,
    GRAY_ICON_COLOR,
    MENU_BUTTON_STYLESHEET,
    NAPARI_STYLE,
    PRIMARY_BUTTON_STYLESHEET,
    PROGRESS_BAR_STYLESHEET,
    SECONDARY_BUTTON_STYLESHEET,
    STATUS_BAR_STYLESHEET,
    SUPERVISION_STATUS_AGENT_STYLESHEET,
    SUPERVISION_STATUS_AUTOMATED_STYLESHEET,
    SUPERVISION_STATUS_SUPERVISED_STYLESHEET,
    USER_ATTENTION_BUTTON_STYLESHEET,
    border_stylesheet,
)
from fibsem.ui.tokens import (
    ERROR_COLOR,
    SURFACE_COLOR,
    TEXT_MUTED_COLOR,
)
from fibsem.ui.widgets import preflight
from fibsem.ui.widgets.canvas.quad_view import MicroscopeViewController
from fibsem.ui.widgets.connection_dialog import connect_to_microscope_dialog
from fibsem.ui.widgets.notifications import NotificationBell, ToastManager
from fibsem.ui.widgets.progress_widget import FibsemProgressWidget, ProgressUpdate
from fibsem.utils import format_duration
from fibsem.versioning import get_version_string

# Suppress a specific upstream Napari/NumPy warning from shapes miter computation.
warnings.filterwarnings(
    "ignore",
    message=r"'where' used without 'out', expect unit?ialized memory in output\. If this is intentional, use out=None\.",
    category=UserWarning,
    module=r"napari\.layers\.shapes\._shapes_utils",
)

# How wide the experiment name button in the tab corner is allowed to grow. Wide
# enough that a default name -- "AutoLamella-" plus a date stamp -- is never elided;
# the point of the button is to say which experiment is open.
EXPERIMENT_MENU_MAX_WIDTH = 360
# The connection chip: a manufacturer and an address, and no more room than that.
CONNECTION_CHIP_MAX_WIDTH = 220

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

    if experiment.created_at:
        created = datetime.fromtimestamp(experiment.created_at)
        rows.append(("Created", html.escape(created.strftime("%d %b %Y, %H:%M"))))

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


def set_menu_icon(action: QAction, key: str) -> None:
    """Give a menu action an icon that actually renders.

    Qt turns menu icons off wholesale on macOS (AA_DontShowIconsInMenus), so an
    icon set the usual way shows on Windows and Linux and silently vanishes here.
    """
    action.setIcon(fibsem_icon(key, color=GRAY_ICON_COLOR))
    action.setIconVisibleInMenu(True)


# How long a question addressed to the agent may stand unanswered before it
# escalates to the operator. A preference later; a constant until the arming
# dialog gives it a home.
AGENT_WATCHDOG_MS = 5 * 60 * 1000
# A live supervising agent long-polls the event stream every ~25 s, so its
# token is heard from continuously. Silence this long means nobody is on the
# other end — a parked question is handed to the operator immediately instead
# of waiting out the full hand-over time above.
AGENT_PRESUMED_GONE_S = 90
AGENT_LIVENESS_CHECK_MS = 15 * 1000


def play_notification_sound():
    """Play a notification sound to alert the user."""
    QApplication.beep()


def confirm_run_workflow_dialog(
    experiment: Experiment,
    lamella_names: list,
    task_names: list,
    parent=None,
) -> bool:
    """Show the pre-flight estimate before starting the workflow.

    Returns True if confirmed. Replaces two columns of bullets that restated the
    selection the user had just made; see WorkflowPreflightDialog. Takes the
    experiment because the estimate reads each lamella's task configs.
    """
    estimate = estimate_workflow(experiment, task_names, lamella_names)
    dlg = WorkflowPreflightDialog(estimate, parent=parent)
    return dlg.exec_() == QDialog.Accepted


def confirm_add_to_queue_dialog(
    lamella_names: list,
    task_names: list,
    run_next: bool,
    already_queued: list,
    estimate: Optional[AdditionEstimate] = None,
    parent=None,
) -> bool:
    """Confirm adding work to a queue that is already running.

    Two figures, and they are allowed to disagree. "Adds" is machine time the addition
    costs; "Expected finish" is what that does to the workflow -- and work slotted in
    ahead of a scheduled hold is absorbed into dead time the workflow was going to spend
    anyway, so the first can be an hour while the second does not move. Quoting only the
    work would overstate the cost; quoting only the finish would hide that the machine
    is now busy for an hour it was not.

    ``already_queued`` is stated rather than acted on: a task that runs twice mills
    twice — the same mechanism that makes "Run again" useful — so the consequence is
    named and the operator decides. Silently adding four of the six pairs asked for
    would be its own surprise.

    ``estimate`` is optional and may be unpriced. A protocol whose configs cannot be
    estimated still has to be able to queue work, so the figures are dropped rather than
    shown as a confident zero.
    """
    dlg = QDialog(parent)
    dlg.setWindowTitle("Add to Queue")
    dlg.setMinimumWidth(560)
    dlg.setStyleSheet(f"QDialog {{ background: {preflight.BACKGROUND}; }}")

    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(18, 16, 18, 14)
    layout.setSpacing(12)

    total = len(lamella_names) * len(task_names)
    heading = QLabel(f"Add {total} task(s) to a workflow that is already running?")
    heading.setStyleSheet(f"color: {preflight.TEXT_STRONG}; font-size: 14px;")
    layout.addWidget(heading)

    if estimate is not None and estimate.is_priced:
        metrics = QHBoxLayout()
        metrics.setSpacing(10)
        metrics.addWidget(
            preflight.metric(
                "Adds",
                preflight.format_duration(estimate.work_seconds),
                _priced_note(estimate),
            ),
            1,
        )
        reference = estimate.finish_before
        metrics.addWidget(
            preflight.metric(
                "Expected finish",
                preflight.format_clock(estimate.finish_after, reference),
                f"was {preflight.format_clock(estimate.finish_before, reference)}",
            ),
            1,
        )
        layout.addLayout(metrics)

    layout.addWidget(
        preflight.detail_block(
            [
                ("Position", "Run next" if run_next else "At the end of the queue"),
                ("Lamella", ", ".join(lamella_names)),
                ("Tasks", ", ".join(task_names)),
            ]
        )
    )

    absorbed = _absorbed_note(estimate)
    if absorbed:
        note = QLabel(absorbed)
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {preflight.TEXT_MUTED}; font-size: 11px;")
        layout.addWidget(note)

    if already_queued:
        # Body text, no accent colour anywhere. This is a consequence to notice rather
        # than an alarm: the count is bold enough to catch and Cancel is right there,
        # while three lines in a warning colour out-shouted the two figures that are
        # supposed to lead the dialog. Body rather than muted because it is content, not
        # the secondary shade the detail-block labels use.
        warning = QLabel(
            f"<b>{len(already_queued)} already queued</b> and will be added again, "
            f"so those tasks will run twice:<br>"
            + "<br>".join(f"• {n}" for n in already_queued[:6])
            + ("<br>• …" if len(already_queued) > 6 else "")
        )
        warning.setWordWrap(True)
        warning.setStyleSheet(f"color: {preflight.TEXT}; font-size: 11px;")
        layout.addWidget(warning)

    btn_row = QHBoxLayout()
    btn_row.addStretch()
    yes_btn = QPushButton(f"Add {total} task(s)")
    no_btn = QPushButton("Cancel")
    yes_btn.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
    no_btn.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
    yes_btn.clicked.connect(dlg.accept)
    no_btn.clicked.connect(dlg.reject)
    no_btn.setDefault(True)
    btn_row.addWidget(no_btn)
    btn_row.addWidget(yes_btn)
    layout.addLayout(btn_row)

    return dlg.exec_() == QDialog.Accepted


def _priced_note(estimate: AdditionEstimate) -> str:
    """The task count, and how much of it the figure above actually covers.

    A partly-priced addition would otherwise read as a complete one that happens to be
    quick — a lamella missing a config for one of the tasks contributes nothing, by the
    same rule the pre-flight dialog and the timeline follow.
    """
    tasks = f"{estimate.total_count} task(s)"
    if estimate.priced_count == estimate.total_count:
        return tasks
    return f"{tasks} · {estimate.priced_count} estimated"


def _absorbed_note(estimate: Optional[AdditionEstimate]) -> str:
    """Said only when the two figures disagree, because then they look wrong.

    An hour of work that moves the finish by nothing is the correct answer and reads as
    an error, so the reason goes on screen with it.

    Two guards, and both are needed. **There has to be a scheduled wait**, because that
    is what the sentence claims -- and `delay_seconds` comes back through a datetime
    (microseconds) while `work_seconds` is a difference of two float sums, so for the
    same quantity they can disagree in the twelfth decimal place. That was enough to
    explain a wait that did not exist on a queue with nothing scheduled at all.

    **And the disagreement has to be one the reader can see.** Both figures are rendered
    with `format_duration`, so a gap that rounds away is not a discrepancy needing an
    explanation -- it is two identical numbers with a paragraph between them.
    """
    if estimate is None or not estimate.is_priced:
        return ""
    if estimate.hold_seconds <= 0 or estimate.work_seconds <= 0:
        return ""
    if preflight.format_duration(estimate.delay_seconds) == preflight.format_duration(
        estimate.work_seconds
    ):
        return ""
    if estimate.delay_seconds <= 0:
        return (
            "The workflow is already waiting for a scheduled task, and this work "
            "fits inside that wait — so it costs no extra time overall."
        )
    return (
        f"Only {preflight.format_duration(estimate.delay_seconds)} of this lands after "
        "the workflow's scheduled wait; the rest fits inside it."
    )


class AutoLamellaSingleWindowUI(QMainWindow):
    """Main window for AutoLamella UI with embedded napari viewers."""

    # Whether a workflow currently permits the overview tabs to work. Held rather than
    # applied where it arrives, because each tab's interactivity is derived from this
    # *and* from whether the other tab is acquiring -- and two callers each setting one
    # control from their own half of the truth is how a control gets stuck on.
    _overviews_allowed = True

    def __init__(self):
        super().__init__()
        # The last words a producer supplied, so a backend's messageless tick still
        # has a label to show. See `MillingMessageTracker`.
        self._milling_label = MillingMessageTracker()
        self.setWindowTitle(f"AutoLamella v{get_version_string()} ")
        self.resize(1600, 1000)

        # Apply napari-style dark theme. Border state rules live here (on the parent)
        # so that setProperty + unpolish/polish on _border_frame re-evaluates them.
        self.setStyleSheet(NAPARI_STYLE + border_stylesheet("workflow_border_frame"))

        # Central tab widget wrapped in a QFrame so the border renders reliably
        self.tab_widget = QTabWidget()
        self._border_frame = QFrame()
        self._border_frame.setObjectName("workflow_border_frame")
        self._border_frame.setProperty("borderState", "idle")
        _frame_layout = QVBoxLayout(self._border_frame)
        _frame_layout.setContentsMargins(0, 0, 0, 0)
        _frame_layout.setSpacing(0)
        _frame_layout.addWidget(self.tab_widget)
        self.setCentralWidget(self._border_frame)

        self.viewers: list[napari.Viewer] = []
        self.autolamella_ui: AutoLamellaUI
        self.minimap_widget: FibsemMinimapWidget
        self.minimap_viewer: napari.Viewer

        # Toast notification manager
        self.toast_manager = ToastManager(self)

        # Load user preferences
        self._preferences = fibsem_cfg.load_user_preferences()
        fibsem_cfg.apply_feature_flags(self._preferences)

        # Read once, here, because both the menu entry and the header chip are gated
        # on it and the menu is built before the tabs are.
        self._connection_chip_enabled = self._preferences.features.connection_chip

        # User attention tracking
        self._user_interaction_sound_played = False  # Track if sound was played
        self._sound_enabled = self._preferences.display.sound_enabled
        self._border_enabled = self._preferences.display.border_enabled
        # First assigned here rather than on the Run click: the workflow handlers
        # read it unconditionally, and an AttributeError in a queued slot is a
        # process abort (FIB-329), not a missing border. The attribute alone -- the
        # frame is unstyled until _set_border_state changes it, which is idle's look.
        self._border_state = "idle"
        self.dev_mode = self._preferences.display.dev_mode

        # create menus, status bar, and tabs
        self._create_menu_bar()
        self._create_test_menu()
        self._create_status_bar()
        self.create_tabs()
        self._apply_preferences()
        self._update_instructions()

        # Connect tab change to status bar update
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

    def _create_menu_bar(self):
        """Create the main menu bar."""
        menu_bar = self.menuBar()

        if menu_bar is None:
            raise RuntimeError("Failed to create menu bar for AutoLamella UI.")

        file_menu = menu_bar.addMenu("File")
        if file_menu is None:
            raise RuntimeError("Failed to create File menu in AutoLamella UI.")

        edit_menu = menu_bar.addMenu("Edit")
        if edit_menu is None:
            raise RuntimeError("Failed to create Edit menu in AutoLamella UI.")

        view_menu = menu_bar.addMenu("View")
        if view_menu is None:
            raise RuntimeError("Failed to create View menu in AutoLamella UI.")

        # These three carry icons because they are also the tab-corner experiment
        # menu (see create_notification_button), where the icons do the work of
        # telling create from load at a glance. Qt hides action icons in menus on
        # macOS unless each action asks for them, hence set_menu_icon.
        # Connecting had no menu entry at all -- the Connection tab was the only
        # door, which is what made that tab impossible to remove (FIB-775). Built
        # unconditionally, offered only when the feature flag is on: see below.
        self.action_connect_microscope = QAction("Connect to Microscope...", self)
        set_menu_icon(self.action_connect_microscope, "mdi:connection")
        self.action_connect_microscope.triggered.connect(self._on_connect_microscope)

        self.action_new_experiment = QAction("New Experiment", self)
        set_menu_icon(self.action_new_experiment, "mdi:plus")
        self.action_new_experiment.triggered.connect(self._on_new_experiment)

        self.action_load_experiment = QAction("Load Experiment", self)
        set_menu_icon(self.action_load_experiment, "mdi:folder-open")
        self.action_load_experiment.triggered.connect(self._on_load_experiment)

        self.action_open_experiment_directory = QAction(
            "Open Experiment Directory", self
        )
        set_menu_icon(self.action_open_experiment_directory, "mdi:folder")
        self.action_open_experiment_directory.triggered.connect(
            self._on_open_experiment_directory
        )

        self.action_load_protocol = QAction("Load Protocol", self)
        self.action_load_protocol.triggered.connect(self._on_load_protocol)
        self.action_save_protocol = QAction("Save Protocol", self)
        self.action_save_protocol.triggered.connect(self._on_save_protocol)
        self.action_exit = QAction("Exit", self)
        self.action_exit.triggered.connect(self.close)  # type: ignore

        # Behind the same flag as the chip. The action is built either way -- it is
        # the thing the tab removal will depend on -- but nothing offers it until
        # the feature is switched on, so a release carrying this changes nothing.
        if self._connection_chip_enabled:
            file_menu.addAction(self.action_connect_microscope)
            file_menu.addSeparator()
        file_menu.addAction(self.action_new_experiment)
        file_menu.addAction(self.action_load_experiment)
        file_menu.addAction(self.action_open_experiment_directory)
        file_menu.addSeparator()
        file_menu.addAction(self.action_load_protocol)
        file_menu.addAction(self.action_save_protocol)
        file_menu.addSeparator()
        file_menu.addAction(self.action_exit)

        # Edit menu
        self.action_preferences = QAction("Preferences...", self)
        self.action_preferences.triggered.connect(self._on_open_preferences)
        edit_menu.addAction(self.action_preferences)

        # View menu
        self.action_show_minimap = QAction("Show Minimap Widget", self)
        self.action_show_minimap.setCheckable(True)
        self.action_show_minimap.setChecked(False)
        self.action_show_minimap.triggered.connect(self._on_toggle_minimap_widget)
        # Hidden pending rework of the minimap widget itself.
        self.action_show_minimap.setVisible(False)

        layer_controls_menu = view_menu.addMenu("Show Layer Controls")

        self.action_layer_controls_overview = QAction("Overview", self)
        self.action_layer_controls_overview.setCheckable(True)
        self.action_layer_controls_overview.setChecked(True)
        self.action_layer_controls_overview.triggered.connect(
            lambda checked: self._on_toggle_viewer_layer_controls(checked, "overview")
        )

        # Overview is the only entry left: the Lamella Editor and now the Microscope tab
        # both render on the matplotlib canvas and have no napari layer docks to show.
        # The submenu goes entirely when the minimap migrates (FIB-405).
        layer_controls_menu.addAction(self.action_layer_controls_overview)

        view_menu.addAction(self.action_show_minimap)

        # Quad-view display controls. The F5/Esc shortcuts live on these QActions — one
        # source of truth for the menu item and its keybinding (Qt renders the shortcut
        # text in the menu automatically). They are scoped to the Microscope tab via
        # setShortcutContext + container.addAction (see _create_main_tab), so they only
        # fire when focus is inside that tab.
        view_menu.addSeparator()
        self.action_toggle_fullscreen = QAction("Full Screen", self)
        self.action_toggle_fullscreen.setCheckable(True)
        self.action_toggle_fullscreen.setShortcut(QKeySequence(Qt.Key_F5))
        self.action_toggle_fullscreen.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.action_toggle_fullscreen.triggered.connect(self._hotkey_toggle_fullscreen)
        view_menu.addAction(self.action_toggle_fullscreen)

        self.action_exit_fullscreen = QAction("Exit Full Screen", self)
        self.action_exit_fullscreen.setShortcut(QKeySequence(Qt.Key_Escape))
        self.action_exit_fullscreen.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.action_exit_fullscreen.triggered.connect(self._hotkey_exit_fullscreen)
        view_menu.addAction(self.action_exit_fullscreen)

        fullscreen_menu = view_menu.addMenu("Full Screen View")
        for label, key in (
            ("SEM", BeamType.ELECTRON),
            ("FIB", BeamType.ION),
            ("Fluorescence", "fm"),
        ):
            act = QAction(label, self)
            act.triggered.connect(
                lambda _checked=False, k=key: self.view_controller.set_fullscreen(k)
            )
            fullscreen_menu.addAction(act)

        # keep the checkable / enabled state honest each time the menu opens
        view_menu.aboutToShow.connect(self._sync_view_menu)

        # Imaging hotkeys (Microscope tab; scoped via container.addAction). Live / auto
        # contrast / auto focus act on the selected beam — the view <-> radio sync keeps
        # dual_beam_widget.beam_type aligned with the selected view.
        imaging_menu = menu_bar.addMenu("Imaging")
        if imaging_menu is None:
            raise RuntimeError("Failed to create Imaging menu in AutoLamella UI.")
        self._imaging_actions: list = []  # added to the Microscope-tab container for scoping
        for label, key, handler in (
            ("Acquire", Qt.Key_F2, self._hotkey_acquire),
            ("Live Acquisition", Qt.Key_F6, self._hotkey_toggle_live),
            ("Auto Contrast", Qt.Key_F9, self._hotkey_autocontrast),
            ("Auto Focus", Qt.Key_F11, self._hotkey_autofocus),
        ):
            action = QAction(label, self)
            action.setShortcut(QKeySequence(key))
            action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
            action.triggered.connect(handler)
            imaging_menu.addAction(action)
            self._imaging_actions.append(action)

        # add tools menu, reporting submenu
        tools_menu = menu_bar.addMenu("Tools")
        if tools_menu is None:
            raise RuntimeError("Failed to create Tools menu in AutoLamella UI.")
        reporting_menu = tools_menu.addMenu("Reporting")
        if reporting_menu is None:
            raise RuntimeError("Failed to create Reporting submenu in AutoLamella UI.")
        self.action_generate_report = QAction("Generate Report", self)
        self.action_generate_report.triggered.connect(self._on_generate_report)
        self.action_generate_overview_plot = QAction("Generate Overview Plot", self)
        self.action_generate_overview_plot.triggered.connect(
            self._on_generate_overview_plot
        )
        reporting_menu.addAction(self.action_generate_report)
        reporting_menu.addAction(self.action_generate_overview_plot)

        # user scripts (FIB-338). The menu itself is application-agnostic; this
        # supplies only the folder, the context, and how to notify.
        #
        # Built unconditionally and hidden by _apply_preferences when the
        # features.scripts_enabled flag is off, which is the default. Hidden rather
        # than absent so toggling the preference takes effect without a restart, the
        # same way the coincidence viewer and bug reporter do. Constructing the
        # controller costs nothing -- it adds two fixed actions and never touches the
        # scripts folder until the dialog is opened.
        from fibsem.applications.autolamella.scripting import get_scripts_directory
        from fibsem.ui.widgets.script_menu import ScriptMenuController

        self.scripts_menu = tools_menu.addMenu("Scripts")
        if self.scripts_menu is None:
            raise RuntimeError("Failed to create Scripts submenu in AutoLamella UI.")
        self.script_menu_controller = ScriptMenuController(
            menu=self.scripts_menu,
            scripts_directory=get_scripts_directory,
            context_factory=self._script_context,
            notify=self.show_toast,
            parent=self,
        )

        # Session controls for the embedded agent server: status, token, and
        # scope arming (arming is per-session consent; the durable policy is in
        # Preferences -> Agent). Visible only with the feature enabled, next to
        # Scripts — its future Automation-menu sibling (FIB-863).
        self.action_agent_server = QAction("Agent Server...", self)
        self.action_agent_server.setMenuRole(QAction.NoRole)
        self.action_agent_server.setToolTip(
            "Session status, access token, and what the connected agent may do"
        )
        self.action_agent_server.triggered.connect(self._on_agent_server_dialog)
        tools_menu.addAction(self.action_agent_server)

        # Below the separator because these act on the install rather than on the
        # experiment. The wizard needs a home here as well as on the connection tab:
        # the callout there appears once and is dismissible, so without this entry a
        # second microscope could never be set up through it.
        tools_menu.addSeparator()
        self.action_guided_setup = QAction("Guided Setup...", self)
        # NoRole stays even though "Guided Setup..." clears the trap by itself. Qt's
        # default MenuRole is TextHeuristicRole, and the heuristic moves any action
        # whose text *starts with* "about", "config", "preference", "options",
        # "setting" or "setup" into the application menu. That is how the earlier name,
        # "Setup Wizard...", displaced Edit -> Preferences and opened this instead --
        # nothing about the connection was wrong, the action was simply no longer in
        # the menu it had been added to. This name is safe; the next one might not be.
        self.action_guided_setup.setMenuRole(QAction.NoRole)
        self.action_guided_setup.setToolTip(
            "A guided walkthrough that configures fibsemOS to work with your microscope"
        )
        self.action_guided_setup.triggered.connect(self._on_guided_setup)
        tools_menu.addAction(self.action_guided_setup)

        self.action_show_plugins = QAction("Plugins...", self)
        self.action_show_plugins.setToolTip(
            "Show every registered pattern, strategy and task, and where it came from"
        )
        self.action_show_plugins.triggered.connect(self._on_show_plugins)
        tools_menu.addAction(self.action_show_plugins)

        self.action_create_desktop_shortcut = QAction(
            "Create Desktop Shortcut...", self
        )
        self.action_create_desktop_shortcut.setToolTip(
            "Place a shortcut on your desktop that launches AutoLamella from this "
            "environment"
        )
        self.action_create_desktop_shortcut.triggered.connect(
            self._on_create_desktop_shortcut
        )
        tools_menu.addAction(self.action_create_desktop_shortcut)

        # add help menu
        help_menu = menu_bar.addMenu("Help")
        if help_menu is None:
            raise RuntimeError("Failed to create Help menu in AutoLamella UI.")
        self.action_report_issue = QAction("Report an Issue...", self)
        self.action_report_issue.triggered.connect(self._on_report_issue)
        help_menu.addAction(self.action_report_issue)
        self.action_about = QAction("About", self)
        self.action_about.triggered.connect(self._show_about_dialog)
        help_menu.addAction(self.action_about)

        # add development menu
        dev_menu = menu_bar.addMenu("Development")
        if dev_menu is None:
            raise RuntimeError("Failed to create Development menu in AutoLamella UI.")
        self.action_print_hello = QAction("Print Hello", self)
        self.action_print_hello.triggered.connect(lambda: print("Hello"))
        dev_menu.addAction(self.action_print_hello)

        self._action_coincidence_separator = dev_menu.addSeparator()
        self.action_open_coincidence_viewer = QAction(
            "Open Coincidence Milling Viewer", self
        )
        self.action_open_coincidence_viewer.triggered.connect(
            self._open_coincidence_milling_viewer
        )
        dev_menu.addAction(self.action_open_coincidence_viewer)

        self._dev_menu = dev_menu
        self._dev_menu.menuAction().setVisible(self.dev_mode)

        action_open_fm_image_viewer = QAction("Open Fluorescence Image Viewer", self)
        action_open_fm_image_viewer.triggered.connect(self._open_fm_image_viewer)
        dev_menu.addAction(action_open_fm_image_viewer)

        dev_menu.addSeparator()

        action_load_fm_configuration = QAction("Load Fluorescence Configuration", self)
        action_load_fm_configuration.triggered.connect(self._import_fm_configuration)
        dev_menu.addAction(action_load_fm_configuration)

        action_save_fm_configuration = QAction("Save Fluorescence Configuration", self)
        action_save_fm_configuration.triggered.connect(self._export_fm_configuration)
        dev_menu.addAction(action_save_fm_configuration)

        dev_menu.addSeparator()

        self.action_export_targeting_ml_data = QAction(
            "Export Targeting ML Data...", self
        )
        self.action_export_targeting_ml_data.triggered.connect(
            self._on_export_targeting_ml_data
        )
        dev_menu.addAction(self.action_export_targeting_ml_data)

    def _create_test_menu(self):
        """Create a test menu for toast notifications and sounds."""

        self.action_toast_info = QAction("Toast: Info", self)
        self.action_toast_info.triggered.connect(
            lambda: self.show_toast("This is an info message", "info")
        )

        self.action_toast_success = QAction("Toast: Success", self)
        self.action_toast_success.triggered.connect(
            lambda: self.show_toast("Operation completed successfully!", "success")
        )

        self.action_toast_warning = QAction("Toast: Warning", self)
        self.action_toast_warning.triggered.connect(
            lambda: self.show_toast("Warning: Check your settings", "warning")
        )

        self.action_toast_error = QAction("Toast: Error", self)
        self.action_toast_error.triggered.connect(
            lambda: self.show_toast("Error: Something went wrong", "error")
        )

        self.action_beep = QAction("Play Beep", self)
        self.action_beep.triggered.connect(play_notification_sound)

        self.action_sound_toggle = QAction("Sound Enabled", self)
        self.action_sound_toggle.setCheckable(True)
        self.action_sound_toggle.setChecked(self._sound_enabled)
        self.action_sound_toggle.triggered.connect(self._on_sound_toggle)

        # Border state test actions
        self.action_border_toggle = QAction("Show Workflow Border", self)
        self.action_border_toggle.setCheckable(True)
        self.action_border_toggle.setChecked(self._border_enabled)
        self.action_border_toggle.triggered.connect(self._on_border_toggle)

        self.action_border_automated = QAction("Automated (green)", self)
        self.action_border_automated.triggered.connect(
            lambda: self._set_border_state("automated")
        )

        self.action_border_supervised = QAction("Supervised (blue)", self)
        self.action_border_supervised.triggered.connect(
            lambda: self._set_border_state("supervised")
        )

        self.action_border_waiting = QAction("Waiting for User (orange)", self)
        self.action_border_waiting.triggered.connect(
            lambda: self._set_border_state("waiting")
        )
        self.action_border_pending = QAction("Pending (grey)", self)
        self.action_border_pending.triggered.connect(
            lambda: self._set_border_state("pending")
        )
        self.action_border_stopping = QAction("Stopping (red)", self)
        self.action_border_stopping.triggered.connect(
            lambda: self._set_border_state("stopping")
        )

        self.action_border_idle = QAction("Idle (no border)", self)
        self.action_border_idle.triggered.connect(
            lambda: self._set_border_state("idle")
        )
        self.action_border_agent = QAction("Agent (electric purple)", self)
        self.action_border_agent.triggered.connect(
            lambda: self._set_border_state("agent")
        )

        # add to menu bar
        menu_bar = self.menuBar()
        test_menu = menu_bar.addMenu("Test")  # type: ignore

        toast_menu = test_menu.addMenu("Toast")  # type: ignore
        toast_menu.addAction(self.action_toast_info)  # type: ignore
        toast_menu.addAction(self.action_toast_success)  # type: ignore
        toast_menu.addAction(self.action_toast_warning)  # type: ignore
        toast_menu.addAction(self.action_toast_error)  # type: ignore

        border_menu = test_menu.addMenu("Border State")  # type: ignore
        border_menu.addAction(self.action_border_toggle)  # type: ignore
        border_menu.addSeparator()  # type: ignore
        border_menu.addAction(self.action_border_automated)  # type: ignore
        border_menu.addAction(self.action_border_supervised)  # type: ignore
        border_menu.addAction(self.action_border_waiting)  # type: ignore
        border_menu.addAction(self.action_border_pending)  # type: ignore
        border_menu.addAction(self.action_border_stopping)  # type: ignore
        border_menu.addAction(self.action_border_idle)  # type: ignore
        border_menu.addAction(self.action_border_agent)  # type: ignore

        test_menu.addSeparator()  # type: ignore
        test_menu.addAction(self.action_beep)  # type: ignore
        test_menu.addAction(self.action_sound_toggle)  # type: ignore

        self._test_menu = test_menu
        self._test_menu.menuAction().setVisible(self.dev_mode)

    def _on_sound_toggle(self, checked: bool):
        """Handle sound toggle."""
        self._sound_enabled = checked
        self._preferences.display.sound_enabled = checked
        fibsem_cfg.save_user_preferences(self._preferences)

    def _on_border_toggle(self, checked: bool):
        """Handle workflow border toggle."""
        self._border_enabled = checked
        self._preferences.display.border_enabled = checked
        fibsem_cfg.save_user_preferences(self._preferences)
        self._set_border_state("idle")

    def _on_open_preferences(self):
        """Open the preferences dialog."""
        from fibsem.ui.widgets.preferences_dialog import PreferencesDialog

        dialog = PreferencesDialog(self._preferences, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            self._preferences = dialog.get_preferences()
            fibsem_cfg.save_user_preferences(self._preferences)
            fibsem_cfg.apply_feature_flags(self._preferences)
            self._apply_preferences()

    def _apply_preferences(self):
        """Apply current preferences to UI state."""
        # The embedded agent server follows its flag immediately: ticking the
        # box mid-session starts it against the connected microscope (or stops
        # it), instead of waiting for the next connect.
        if getattr(self, "autolamella_ui", None) is not None:
            self.autolamella_ui.sync_agent_server_with_preference()
        d = self._preferences.display
        self._sound_enabled = d.sound_enabled
        self._border_enabled = d.border_enabled
        self.dev_mode = d.dev_mode
        # The lamella strip's density. Guarded because _apply_preferences also runs
        # before the Lamella tab is built.
        if hasattr(self, "lamella_card_container"):
            self.lamella_card_container.set_mode(d.lamella_card_mode)
        # Sync Test menu toggle actions
        self.action_sound_toggle.setChecked(d.sound_enabled)
        self.action_border_toggle.setChecked(d.border_enabled)
        # Toggle dev/test menu visibility
        self._dev_menu.menuAction().setVisible(d.dev_mode)
        self._test_menu.menuAction().setVisible(d.dev_mode)
        # Toggle coincidence milling viewer action
        coincidence_enabled = self._preferences.features.coincidence_milling_enabled
        self.action_open_coincidence_viewer.setVisible(coincidence_enabled)
        self._action_coincidence_separator.setVisible(coincidence_enabled)
        # Toggle the "Report an Issue..." Help menu action
        self.action_report_issue.setVisible(
            self._preferences.features.bug_report_enabled
        )
        # Show or hide the old napari Minimap tab. The Overview tab is not here: it
        # ships to everyone, and which of its modalities can be reached follows the
        # instrument rather than a flag.
        self._apply_napari_overview_visibility()
        # Toggle Tools -> Scripts. Hiding the menu hides the whole feature: it is the
        # only route to the manager dialog, and the dialog is the only thing that runs
        # a script. If a script is mid-run, leave it visible -- taking away the only
        # Stop button while the microscope is moving would be worse than the flag
        # being briefly wrong.
        self.scripts_menu.menuAction().setVisible(
            self._preferences.features.scripts_enabled
            or self.script_menu_controller.runner.is_running
        )
        # Same rule as the rest of the agent chrome: invisible unless enabled.
        self.action_agent_server.setVisible(
            self._preferences.features.agent_server_enabled
        )
        # getattr because this also runs from the preferences dialog, and the tab it
        # reaches into is built by AutoLamellaUI rather than here.
        system_widget = getattr(self.autolamella_ui, "system_widget", None)
        if system_widget is not None:
            system_widget.refresh_first_run_offer(self._preferences)

    #### USER SCRIPTS (FIB-338)

    def _script_context(self):
        """Build the context a script runs against, or explain why it cannot.

        Returns (context, reason). A reason means the menu entries are disabled and
        that string is shown instead.
        """
        from fibsem.applications.autolamella.scripting import ScriptContext

        experiment = self.autolamella_ui.experiment
        if experiment is None:
            return None, "Load an experiment to run scripts"
        if self.autolamella_ui.is_workflow_running:
            # tasks mutate lamella state on a worker thread, so a script reading
            # mid-workflow would get a torn snapshot.
            return None, "Unavailable while a workflow is running"

        return ScriptContext(
            experiment=experiment,
            log=logging.info,
            microscope=self.autolamella_ui.microscope,
        ), ""

    def show_toast(
        self,
        message: str,
        notification_type: str = "info",
        duration: int = 5000,
        temporary: bool = False,
    ):
        """Show a toast notification.

        Unconditional: there is no longer a preference for turning toasts off, so the
        branch that used to route a suppressed message to the notification bell has
        gone with it. Nothing is lost -- `ToastManager.show_toast` records every
        non-temporary message in the bell itself.
        """
        self.toast_manager.show_toast(
            message, notification_type, duration, temporary=temporary
        )

    def _on_connect_microscope(self):
        """Connect from the dialog, then hand the session to the system widget.

        The widget stays the one place that owns the connection: everything else in
        the application follows its `connected_signal`, so routing through it means
        the dialog needs to know nothing about who cares.
        """
        if self.autolamella_ui is None:
            return

        system_widget = self.autolamella_ui.system_widget
        result = connect_to_microscope_dialog(
            parent=self,
            microscope=system_widget.microscope,
            settings=system_widget.settings,
            workflow_running=self.autolamella_ui.is_workflow_running,
        )
        if not result.changed:
            return

        # Including a disconnect, where the new session is None. Handing that back
        # through the widget is what tells the rest of the application, which
        # follows its signals rather than knowing about this dialog.
        system_widget.microscope = result.microscope
        system_widget.settings = result.settings
        system_widget.update_ui()

    def _on_new_experiment(self):
        """Handle New Experiment action."""
        if self.autolamella_ui is not None:
            self.autolamella_ui.create_experiment()

    def _on_load_experiment(self):
        """Handle Load Experiment action."""
        if self.autolamella_ui is not None:
            self.autolamella_ui.load_experiment()

    def _on_open_experiment_directory(self):
        """Handle Open Experiment Directory action."""
        if self.autolamella_ui is not None:
            self.autolamella_ui._open_experiment_directory()

    def _on_load_protocol(self):
        """Handle Load Protocol action."""
        if self.autolamella_ui is not None:
            self.autolamella_ui.load_protocol()

    def _on_save_protocol(self):
        """Handle Save Protocol action."""
        if self.autolamella_ui is not None:
            self.autolamella_ui.export_protocol_ui()

    def _show_about_dialog(self):
        """Show the About dialog."""
        if self.autolamella_ui is not None:
            self.autolamella_ui.open_information_dialog()

    def _on_report_issue(self):
        """Open the Report an Issue dialog."""
        from fibsem.ui.widgets.bug_report_widget import open_bug_report_dialog

        experiment = getattr(self.autolamella_ui, "experiment", None)
        microscope = getattr(self.autolamella_ui, "microscope", None)
        open_bug_report_dialog(
            experiment_path=str(experiment.path) if experiment is not None else None,
            microscope=microscope,
            parent=self,
        )

    def _on_guided_setup(self):
        """Run the microscope guided setup.

        Delegated to the connection tab rather than opened here, so that whatever the
        wizard saves is selected in the one combo box that decides what the next
        connection uses -- and so the live microscope is handed over, rather than the
        wizard opening a second client against the same instrument.
        """
        self.autolamella_ui.system_widget.run_guided_setup()

    def _on_show_plugins(self):
        """Open the read-only listing of registered extensions."""
        from fibsem.ui.widgets.plugins_dialog import PluginsDialog

        PluginsDialog(parent=self).exec_()

    def _on_create_desktop_shortcut(self):
        """Create a launcher shortcut for the AutoLamella UI, confirming overwrites.

        The location is chosen by the user, defaulting to the Desktop.
        """
        from pathlib import Path

        from PyQt5.QtWidgets import QFileDialog

        from fibsem.tools import desktop_shortcut

        chosen = QFileDialog.getExistingDirectory(
            self,
            "Choose Shortcut Location",
            str(desktop_shortcut.get_desktop_directory()),
        )
        if not chosen:
            return
        directory = Path(chosen)
        try:
            path = desktop_shortcut.create_desktop_shortcut(directory=directory)
        except FileExistsError as exc:
            reply = QMessageBox.question(
                self,
                "Create Desktop Shortcut",
                f"A shortcut already exists at:\n{exc}\n\nReplace it?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
            try:
                path = desktop_shortcut.create_desktop_shortcut(
                    overwrite=True, directory=directory
                )
            except Exception as exc2:
                self._show_shortcut_error(exc2)
                return
        except Exception as exc:
            self._show_shortcut_error(exc)
            return
        self.show_toast(f"Shortcut created: {path}", "info")

    def _show_shortcut_error(self, exc: Exception) -> None:
        logging.error(f"Failed to create desktop shortcut: {exc}")
        QMessageBox.warning(
            self,
            "Create Desktop Shortcut",
            f"Could not create the desktop shortcut:\n{exc}",
        )

    def _on_toggle_minimap_widget(self, checked: bool):
        """Toggle the minimap plot widget visibility."""
        if self.autolamella_ui is not None and hasattr(
            self.autolamella_ui, "minimap_plot_widget"
        ):
            self.autolamella_ui.minimap_plot_widget.setVisible(checked)
            self.autolamella_ui.minimap_plot_widget.activateWindow()

    def _sync_view_menu(self) -> None:
        """Refresh the View menu's dynamic state right before it opens: reflect whether a
        view is full-screened (check + Exit-enabled)."""
        if getattr(self, "view_controller", None) is None:
            return
        fullscreen = self.view_controller.fullscreen is not None
        self.action_toggle_fullscreen.setChecked(fullscreen)
        self.action_exit_fullscreen.setEnabled(fullscreen)

    def _hotkey_toggle_fullscreen(self) -> None:
        """F5: toggle full screen for the selected view."""
        self.view_controller.toggle_fullscreen()

    def _hotkey_exit_fullscreen(self) -> None:
        """Esc: exit full screen (no-op when already showing the grid)."""
        self.view_controller.set_fullscreen(None)

    def _hotkey_acquire(self) -> None:
        """F2: acquire the selected view (SEM / FIB / FM), if a microscope is connected
        and an acquisition isn't already running."""
        view = self.view_controller.selected_view
        ui = self.autolamella_ui
        if view in (BeamType.ELECTRON, BeamType.ION):
            image_widget = getattr(ui, "image_widget", None)
            if image_widget is None:
                logging.info("F2 acquire: no image widget (microscope not connected)")
                return
            if getattr(image_widget, "is_acquiring", False):
                logging.info("F2 acquire: acquisition already in progress")
                return
            if view is BeamType.ELECTRON:
                image_widget.acquire_sem_image()
            else:
                image_widget.acquire_fib_image()
        elif view == "fm":
            fm_widget = getattr(ui, "fm_control_widget", None)
            if fm_widget is None:
                logging.info("F2 acquire: no fluorescence widget")
                return
            # `is_acquisition_active`, not `is_acquiring` — the latter does not exist on
            # this widget, so a getattr default of False would make the guard inert and
            # let F2 start a second FM acquisition mid-run (cf. FIB-441 / FIB-436).
            if fm_widget.is_acquisition_active:
                logging.info("F2 acquire: fluorescence acquisition already in progress")
                return
            fm_widget.acquire_image()

    def _selected_em_image_widget(self):
        """The image widget when a SEM/FIB view is selected, else None (FM / not
        connected). Backs the EM-only imaging hotkeys (live / auto contrast / auto focus)."""
        if self.view_controller.selected_view not in (BeamType.ELECTRON, BeamType.ION):
            return None
        return getattr(self.autolamella_ui, "image_widget", None)

    def _hotkey_toggle_live(self) -> None:
        """F6: toggle live acquisition on the selected SEM/FIB beam."""
        image_widget = self._selected_em_image_widget()
        if image_widget is None:
            logging.info("F6 live: only SEM/FIB support live acquisition")
            return
        image_widget.toggle_live_acquisition()

    def _hotkey_autocontrast(self) -> None:
        """F9: auto-contrast the selected SEM/FIB beam."""
        image_widget = self._selected_em_image_widget()
        if image_widget is None:
            logging.info("F9 autocontrast: only SEM/FIB supported")
            return
        image_widget.run_autocontrast()

    def _hotkey_autofocus(self) -> None:
        """F11: auto-focus the selected SEM/FIB beam."""
        image_widget = self._selected_em_image_widget()
        if image_widget is None:
            logging.info("F11 autofocus: only SEM/FIB supported")
            return
        image_widget.run_autofocus()

    def _on_toggle_viewer_layer_controls(self, checked: bool, viewer_key: str):
        """Toggle the layer list and layer controls for a specific viewer.

        getattr, not attribute access: tabs move off napari one at a time, so a tab that
        has already migrated never sets its viewer attribute. A dict literal would
        dereference all three eagerly and raise AttributeError for *every* entry, not
        just the migrated one — and an exception escaping a Qt slot aborts the process
        under PyQt5 (FIB-329).
        """
        viewer_map = {
            "overview": getattr(self, "minimap_viewer", None),
        }
        viewer = viewer_map.get(viewer_key)
        if viewer is not None:
            qt_viewer = viewer.window._qt_viewer
            qt_viewer.dockLayerList.setVisible(checked)
            qt_viewer.dockLayerControls.setVisible(checked)

    def _on_generate_report(self):
        """Handle Generate Report action."""
        if self.autolamella_ui is not None:
            self.autolamella_ui.action_generate_report()

    def _on_generate_overview_plot(self):
        """Handle Generate Overview Plot action."""
        if self.autolamella_ui is not None:
            self.autolamella_ui.action_generate_overview_plot()

    def _on_export_targeting_ml_data(self):
        """Handle Export Targeting ML Data action."""
        if self.autolamella_ui is not None:
            self.autolamella_ui.export_targeting_ml_data()

    def _open_fm_image_viewer(self):
        """Open the Fluorescence Image Viewer widget."""
        if self.autolamella_ui is not None:
            self.autolamella_ui._open_fm_image_viewer()

    def _import_fm_configuration(self):
        """Load a fluorescence microscope configuration."""
        if self.autolamella_ui is not None:
            self.autolamella_ui.import_fm_configuration()

    def _export_fm_configuration(self):
        """Save the current fluorescence microscope configuration."""
        if self.autolamella_ui is not None:
            self.autolamella_ui.export_fm_configuration()

    def _open_coincidence_milling_viewer(self):
        """Open the Coincidence Milling Viewer dialog."""
        if self.autolamella_ui is not None:
            self.autolamella_ui._open_coincidence_milling_viewer()

    def _create_status_bar(self):
        """Create the status bar."""
        status_bar = self.statusBar()
        if status_bar is None:
            raise RuntimeError("Failed to create status bar for AutoLamella UI.")
        self.status_bar = status_bar
        self.status_bar.setStyleSheet(STATUS_BAR_STYLESHEET)

        # Add generic progress widget (tile acquisition, etc.)
        self.progress_widget = FibsemProgressWidget(self.status_bar)
        self.progress_widget.setMaximumWidth(400)
        self.status_bar.addPermanentWidget(self.progress_widget)

        # Add milling progress bar
        self.milling_progress_bar = QProgressBar(self.status_bar)
        self.milling_progress_bar.setMaximumWidth(400)
        self.milling_progress_bar.setMaximum(100)
        self.milling_progress_bar.setValue(0)
        self.milling_progress_bar.setTextVisible(True)
        self.milling_progress_bar.setAlignment(Qt.AlignCenter)
        self.milling_progress_bar.setStyleSheet(PROGRESS_BAR_STYLESHEET)
        self.milling_progress_bar.hide()  # Hidden by default
        self.status_bar.addPermanentWidget(self.milling_progress_bar)

        # Add user attention button (shown when waiting for user interaction)
        self.user_attention_btn = QPushButton("Attention Required")
        self.user_attention_btn.setStyleSheet(USER_ATTENTION_BUTTON_STYLESHEET)
        self.user_attention_btn.setIcon(
            fibsem_icon("mdi:alert-circle", color=GRAY_ICON_COLOR)
        )
        self.user_attention_btn.hide()  # Hidden by default
        self.user_attention_btn.setToolTip(
            "User Input Required - Click to go to Microscope tab"
        )
        self.user_attention_btn.clicked.connect(self._on_user_attention_clicked)
        self.status_bar.addPermanentWidget(self.user_attention_btn)

        # Add supervised status chip (shown during workflow to indicate supervision mode)
        self._current_task_name = None  # Track current task for supervision toggle
        # The agent watchdog: a question addressed to the agent that goes
        # unanswered this long stops being the agent's and becomes yours —
        # the ordinary waiting chrome (orange border, attention button, sound)
        # takes over, with a toast saying why. Lives in the app, so it
        # survives the agent's own loop dying — the whole point.
        self._agent_watchdog = QTimer(self)
        self._agent_watchdog.setSingleShot(True)
        self._agent_watchdog.timeout.connect(self._on_agent_watchdog_expired)
        self._agent_watchdog_expired = False
        # The companion check: while a question parks on the agent's clock,
        # confirm someone is actually on the other end (the agent's token is
        # heard from continuously while it watches). An agent that dies
        # mid-question hands over in seconds, not the full deadline above.
        self._agent_liveness_check = QTimer(self)
        self._agent_liveness_check.setInterval(AGENT_LIVENESS_CHECK_MS)
        self._agent_liveness_check.timeout.connect(self._on_agent_liveness_check)
        self.supervised_status_btn = QPushButton("Supervised")
        self.supervised_status_btn.setCursor(Qt.PointingHandCursor)  # type: ignore
        self.supervised_status_btn.setToolTip("Click to toggle supervision")
        self.supervised_status_btn.clicked.connect(self._on_supervised_status_clicked)
        self.supervised_status_btn.hide()  # Hidden by default
        self.status_bar.addPermanentWidget(self.supervised_status_btn)

        # Add run workflow button (visible when workflow is not running)
        self.run_workflow_btn = QPushButton("Run Workflow")
        self.run_workflow_btn.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self.run_workflow_btn.setIcon(
            fibsem_icon("mdi:play-circle", color=GRAY_ICON_COLOR)
        )
        self.run_workflow_btn.setEnabled(False)
        self.run_workflow_btn.setToolTip("Run the AutoLamella workflow.")
        self.run_workflow_btn.clicked.connect(self._on_run_workflow_clicked)
        self.status_bar.addPermanentWidget(self.run_workflow_btn)

        # Add stop workflow button
        self.stop_workflow_btn = QPushButton("Stop Workflow")
        self.stop_workflow_btn.setStyleSheet(DANGER_BUTTON_STYLESHEET)
        self.stop_workflow_btn.setIcon(
            fibsem_icon("mdi:stop-circle", color=GRAY_ICON_COLOR)
        )
        self.stop_workflow_btn.hide()  # Hidden by default
        self.stop_workflow_btn.setToolTip(
            "Stop the current workflow. You will be asked to confirm."
        )
        self.stop_workflow_btn.clicked.connect(self._on_stop_workflow_clicked)
        self.status_bar.addPermanentWidget(self.stop_workflow_btn)

    def _on_stop_workflow_clicked(self):
        """Handle stop workflow button click with confirmation."""
        reply = QMessageBox.question(
            self,
            "Stop Workflow",
            "Are you sure you want to stop the workflow?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes and self.autolamella_ui is not None:
            self.autolamella_ui.stop_task_workflow()
            self._set_border_state("stopping")

    def _on_user_attention_clicked(self):
        """Handle user attention button click - switch to Microscope tab."""
        self.tab_widget.setCurrentIndex(0)  # Microscope tab is index 0

    def _on_run_workflow_clicked(self):
        """Run the workflow using the lamella and task selections from the workflow widget."""
        ui = self.autolamella_ui
        if ui is None:
            return
        if ui.is_workflow_running:
            return
        if (
            ui.microscope is None
            or ui.experiment is None
            or ui.experiment.task_protocol is None
        ):
            return

        selected_tasks = self.lamella_workflow_widget.get_selected_tasks()
        selected_lamella = self.lamella_workflow_widget.get_selected_lamella()

        if not selected_tasks or not selected_lamella:
            return

        task_names = [t.name for t in selected_tasks]
        lamella_names = [lam.name for lam in selected_lamella]

        if not confirm_run_workflow_dialog(
            ui.experiment, lamella_names, task_names, parent=self
        ):
            return

        self._set_border_state(self._running_border_state(selected_tasks[0].name))
        self._push_timeline_estimates(
            [(ln, tn) for tn in task_names for ln in lamella_names]
        )
        # The run writes the experiment from its own thread. Land any edit still
        # waiting in the editor first, so there is only ever one writer (FIB-683).
        self.lamella_widget.flush_pending_save()
        ui._start_run_workflow_thread(task_names, lamella_names)
        # Show the Stop button immediately so the run is cancellable even while
        # waiting for a scheduled first task (before any task status arrives).
        self.set_workflow_running()
        # Clear selections after starting workflow
        self.lamella_workflow_widget.lamella_list.set_all_selected(False)
        self.lamella_workflow_widget.workflow.set_all_selected(False)

    def _on_workflow_selection_changed(self, _=None) -> None:
        """Enable the run button only when at least one lamella and one task are selected."""
        n_lam = len(self.lamella_workflow_widget.get_selected_lamella())
        n_task = len(self.lamella_workflow_widget.get_selected_tasks())
        valid = n_lam > 0 and n_task > 0
        self.run_workflow_btn.setEnabled(valid)
        if valid:
            self.run_workflow_btn.setToolTip(
                f"Run workflow: {n_lam} lamella, {n_task} task{'s' if n_task != 1 else ''}"
            )
        else:
            missing = []
            if n_lam == 0:
                missing.append("a lamella")
            if n_task == 0:
                missing.append("a task")
            self.run_workflow_btn.setToolTip(
                f"Select {' and '.join(missing)} to run the workflow"
            )

        # The timeline's Add button commits the same selection, so it follows the
        # same rule — there is nothing to add until something is ticked.
        if hasattr(self, "workflow_timeline"):
            if valid:
                tip = (
                    f"Add to queue: {n_lam} lamella, "
                    f"{n_task} task{'s' if n_task != 1 else ''}"
                )
            else:
                tip = f"Select {' and '.join(missing)} to add to the queue"
            self.workflow_timeline.set_add_enabled(valid, tip)

    def set_workflow_running(self, message: str | None = None):
        """Show stop button and update status message."""
        self.run_workflow_btn.hide()
        self.stop_workflow_btn.show()
        if message and self.status_bar is not None:
            self.status_bar.showMessage(message)
        self._set_minimap_workflow_enabled(False)
        # A live run is exactly when there is a queue to edit, so the actions come
        # on with it — and go off again in hide_workflow_running.
        if hasattr(self, "workflow_timeline"):
            self.workflow_timeline.set_actions_enabled(True)

    def hide_workflow_running(self):
        """Hide the stop button and show run button."""
        self.stop_workflow_btn.hide()
        self.supervised_status_btn.hide()
        self.run_workflow_btn.show()
        self._set_minimap_workflow_enabled(True)
        # The timeline stays on screen after a run, but there is no longer a
        # queue behind it — offering to reorder one would be a lie.
        if hasattr(self, "workflow_timeline"):
            self.workflow_timeline.set_actions_enabled(False)

    def _set_minimap_workflow_enabled(self, enabled: bool):
        """Record whether a workflow currently permits the overviews to work.

        A running workflow owns the instrument, so neither tab should be able to start
        competing for it. Both tabs keep their Cancel button live regardless -- see
        `set_interactive` on either widget -- so a lock arriving mid-run cannot strand
        an acquisition with no way to stop it.

        The answer is recorded rather than applied: `_apply_overview_locks` is what
        reaches the tabs, because it is not the only fact that decides.
        """
        if hasattr(self, "minimap_widget"):
            self.minimap_widget.pushButton_run_tile_collection.setEnabled(enabled)
            self.minimap_widget.pushButton_load_image.setEnabled(enabled)
        self._overviews_allowed = bool(enabled)
        self._apply_overview_locks()

    def _apply_overview_locks(self, *_) -> None:
        """Decide, for each overview tab, whether it may work right now.

        Two facts, and both have to be answered in one place. A workflow owning the
        instrument is one; the *other* overview being mid-run is the other, and it was
        not asked at all. Each tab locked itself while it ran, so nothing stopped a
        click-to-move on the beam overview during a fluorescence tileset -- and that
        does not fail loudly. The runner goes on placing tiles at the poses it planned,
        so the mosaic comes out plausible and wrong (FIB-706).

        Derived from both every time rather than each caller setting the flag it knows
        about: a workflow finishing while an overview runs must not re-enable the other
        tab, and a run finishing inside a locked window must not either.

        Takes and ignores an argument so it can be connected straight to
        `acquiring_changed`, whose bool it deliberately does not read -- it asks the tabs
        instead, so a missed or duplicated signal cannot leave this holding a state the
        tabs disagree with.

        Written out per tab rather than looped, so that `test_overview_tab_wiring.py`
        can still see `self.<tab>.set_interactive` in the window's source. That test is
        the parity check between the two tabs and one of the few CI runs without PyQt5;
        a loop over local variables hides the calls from it.
        """
        fm = getattr(self, "fm_overview_tab", None)
        beam = getattr(self, "beam_overview_tab", None)
        if fm is not None:
            allowed, reason = self._overview_may_work(beam)
            self.fm_overview_tab.set_interactive(allowed, reason)
        if beam is not None:
            allowed, reason = self._overview_may_work(fm)
            self.beam_overview_tab.set_interactive(allowed, reason)

    def _overview_may_work(self, other) -> Tuple[bool, str]:
        """Whether an overview tab may work given what *other* is doing, and why not.

        The reason travels with the answer because the widget cannot work it out: both
        causes are the same `False` by the time they reach it, and a refusal that blames
        a workflow when the real reason is the other overview sends someone looking in
        the wrong place.

        The workflow is named first when both hold. It is the outer authority -- the
        overview run would not have started under it -- so it is the fact that has to
        change first.
        """
        if not self._overviews_allowed:
            return False, "a workflow is running"
        if other is not None and other.is_acquiring:
            return False, "the other overview is acquiring"
        return True, ""

    def _set_border_state(self, state: str):
        """Update the tab widget border to reflect current workflow state.

        States: 'waiting', 'supervised', 'automated', 'idle'
        """
        if state == getattr(self, "_border_state", None):
            return
        self._border_state = state
        effective = state if self._border_enabled else "idle"
        self._border_frame.setProperty("borderState", effective)
        style = self._border_frame.style()
        if style is not None:
            style.unpolish(self._border_frame)
            style.polish(self._border_frame)
        self._border_frame.update()

    def _agent_supervision_active(self, task_name: str) -> bool:
        """Whether ``task_name``'s questions are addressed to a connected agent.

        Gated on the agent server actually running — with no server there is
        nobody the designation could refer to, so all the agent chrome stays
        invisible and a designated task behaves as plain supervised.
        """
        ui = self.autolamella_ui
        if ui is None or task_name is None:
            return False
        host = getattr(ui, "_agent_server_host", None)
        if host is None or not getattr(host, "running", False):
            return False
        protocol = ui.protocol
        if protocol is None:
            return False
        return protocol.get_supervisor(task_name) == "agent"

    def _running_border_state(self, task_name: Optional[str]) -> str:
        """The border for a running workflow: automated, supervised, or agent."""
        if task_name is None or not get_task_supervision(
            task_name, self.autolamella_ui
        ):
            return "automated"
        if self._agent_supervision_active(task_name):
            return "agent"
        return "supervised"

    def _update_supervised_status(self) -> bool:
        """Update the supervised status chip for the current task."""
        task_name = self._current_task_name
        if task_name is None or self.autolamella_ui is None:
            return False
        supervised = get_task_supervision(task_name, self.autolamella_ui)
        if supervised and self._agent_supervision_active(task_name):
            self.supervised_status_btn.setIcon(
                fibsem_icon("mdi:star-four-points", color="white")
            )
            self.supervised_status_btn.setText("Agent")
            self.supervised_status_btn.setToolTip(
                f"{task_name} is supervised by the connected agent. "
                "You can still answer any question first. Click to toggle "
                "supervision."
            )
            self.supervised_status_btn.setStyleSheet(
                SUPERVISION_STATUS_AGENT_STYLESHEET
            )
        elif supervised:
            self.supervised_status_btn.setIcon(
                fibsem_icon("mdi:account-hard-hat", color="white")
            )
            self.supervised_status_btn.setText("Supervised")
            self.supervised_status_btn.setToolTip(
                f"{task_name} is running in supervised mode. Your input will be required. Click to toggle."
            )
            self.supervised_status_btn.setStyleSheet(
                SUPERVISION_STATUS_SUPERVISED_STYLESHEET
            )
        else:
            self.supervised_status_btn.setIcon(
                fibsem_icon("mdi:lightning-bolt", color="white")
            )
            self.supervised_status_btn.setText("Automated")
            self.supervised_status_btn.setToolTip(
                f"{task_name} is running in automated mode. Click to toggle."
            )
            self.supervised_status_btn.setStyleSheet(
                SUPERVISION_STATUS_AUTOMATED_STYLESHEET
            )
        self.supervised_status_btn.show()

        return supervised

    def _on_supervised_status_clicked(self):
        """Toggle supervision for the current task in the protocol."""
        if self._current_task_name is None or self.autolamella_ui is None:
            return
        protocol = self.autolamella_ui.protocol
        if protocol is None:
            return
        for task in protocol.workflow_config.tasks:
            if task.name == self._current_task_name:
                task.supervise = not task.supervise
                break
        self._update_supervised_status()
        if self.autolamella_ui.is_workflow_running:
            self._set_border_state(self._running_border_state(self._current_task_name))
        # Refresh the workflow widget to reflect the toggled supervise state
        if hasattr(self, "lamella_workflow_widget"):
            self.lamella_workflow_widget.workflow.refresh_all()

    def _update_instructions(self):
        """Update the status bar with the current instruction based on application state."""
        if self.autolamella_ui is None or self.status_bar is None:
            return
        is_connected = self.autolamella_ui.microscope is not None
        experiment = self.autolamella_ui.experiment
        is_experiment_loaded = experiment is not None
        # Carried here from the panel, which used to render the same ladder into a
        # label of its own: an experiment with no protocol cannot run anything, and
        # this was the one rung the status bar did not have.
        is_protocol_loaded = self.autolamella_ui.protocol is not None
        has_positions = is_experiment_loaded and len(experiment.positions) > 0

        if not is_connected:
            msg = INSTRUCTIONS["NOT_CONNECTED"]
        elif not is_experiment_loaded:
            msg = INSTRUCTIONS["NO_EXPERIMENT"]
        elif not is_protocol_loaded:
            msg = INSTRUCTIONS["NO_PROTOCOL"]
        elif not has_positions:
            msg = INSTRUCTIONS["NO_LAMELLA"]
        else:
            msg = INSTRUCTIONS["AUTOLAMELLA_READY"]

        self.status_bar.showMessage(msg)

    def _on_microscope_connected(self):
        """Handle microscope connection and connect milling progress signal."""
        # Before the signal wiring below, which returns early on a disconnect: both
        # overview modalities have to hear about that case too, to let go of the old
        # microscope. They hold it for life, so each has to be handed the new one. It
        # also re-answers whether the tab can be used, which is only knowable now --
        # whether this system has a fluorescence detector at all.
        self._refresh_overview_microscope()
        if (
            self.autolamella_ui is not None
            and self.autolamella_ui.microscope is not None
        ):
            try:
                self.autolamella_ui.microscope.milling_progress_signal.disconnect(
                    self._on_milling_progress
                )
            except Exception:
                pass
            self.autolamella_ui.microscope.milling_progress_signal.connect(
                self._on_milling_progress
            )
            try:
                self.autolamella_ui.microscope.tiled_acquisition_signal.disconnect(
                    self._on_tile_acquisition_progress
                )
            except Exception:
                pass
            self.autolamella_ui.microscope.tiled_acquisition_signal.connect(
                self._on_tile_acquisition_progress
            )
            try:
                self.autolamella_ui.microscope.spot_burn_progress_signal.disconnect(
                    self._on_spot_burn_progress
                )
            except Exception:
                pass
            self.autolamella_ui.microscope.spot_burn_progress_signal.connect(
                self._on_spot_burn_progress
            )
        self._update_connection_chip()
        self._update_experiment_header()
        self._update_instructions()

    @ensure_main_thread
    def _on_milling_progress(self, payload: object):
        """Handle milling progress updates from the microscope."""
        # Total-by-construction decode. Every in-tree producer emits a
        # `MillingProgress` and this is a no-op for them; it stands because a
        # plugin-loaded strategy is a producer too, and psygnal hands whatever it
        # emits to this slot unchanged (FIB-797).
        report = MillingProgress.from_payload(payload)
        if report.status.is_terminal:
            self.milling_progress_bar.setVisible(False)
            return

        label = self._milling_label.label(report)

        if report.status is MillingProgressStatus.STAGE_STARTED:
            # `or 1` rather than a `.get` default: a producer that sends 0 total stages
            # is as much a division by zero as one that sends nothing.
            total_stages = report.total_stages or 1
            stage = report.display_stage or 1
            stage_name = report.stage_name or f"Stage {stage}"
            self.milling_progress_bar.setVisible(True)
            self.milling_progress_bar.setValue(0)
            self.milling_progress_bar.setFormat(label)
            self.milling_progress_bar.setToolTip(
                f"Milling Stage: {stage}/{total_stages} - {stage_name}"
            )

        elif report.status is MillingProgressStatus.STAGE_UPDATE:
            remaining_time = report.remaining_time
            if remaining_time is not None and report.estimated_time:
                percent_complete = int(
                    (1 - (remaining_time / report.estimated_time)) * 100
                )
                self.milling_progress_bar.setValue(percent_complete)
                self.milling_progress_bar.setFormat(
                    f"{label} - {format_duration(remaining_time)} remaining"
                )
            else:
                # No countdown to draw, but the producer's words are still worth showing:
                # this is the branch a strategy's own report lands in, and it used to
                # match nothing at all and render nowhere.
                self.milling_progress_bar.setFormat(label)

    @ensure_main_thread
    def _on_spot_burn_progress(self, report: SpotBurnProgress) -> None:
        """Handle spot burn progress updates from the microscope (supervised + unsupervised)."""
        self.progress_widget.update_progress(build_spot_burn_progress_update(report))
        if report.status.is_terminal:
            # hide the Done/Failed state after a moment; reset_if_finished leaves the
            # widget alone if another operation has started rendering progress since
            QTimer.singleShot(2000, self.progress_widget.reset_if_finished)

    # What the status bar calls each state of a run. One table for both modalities and
    # deliberately generic: this is read at a glance from another tab, so it says what
    # kind of thing is happening rather than repeating the producer's own wording. It is
    # also the seam FIB-742 needs -- saying *which* run is going is a modality prefix on
    # these, which is only possible now the words live here instead of arriving baked
    # into the report.
    _STATUS_LABELS = {
        TiledStatus.STARTING: "Collecting tiles",
        TiledStatus.MOVING: "Moving stage",
        TiledStatus.TILE_STARTED: "Collecting tiles",
        TiledStatus.TILE_COLLECTED: "Collecting tiles",
        TiledStatus.TILES_ACQUIRED: "Collecting tiles",
        TiledStatus.STITCHING: "Stitching tiles",
        TiledStatus.SAVING: "Saving overview",
        TiledStatus.FINISHED: "Complete",
        TiledStatus.CANCELLED: "Cancelled",
        TiledStatus.FAILED: "Failed",
    }

    @ensure_main_thread
    def _on_tile_acquisition_progress(self, event: TiledProgress) -> None:
        """Handle tiled acquisition progress updates from the microscope.

        Deliberately **not** filtered by modality,
        unlike the two overview widgets: they each drive one modality's canvas and must
        ignore the other's run, while the status bar is the one consumer that wants both
        (FIB-725).
        """
        message = self._STATUS_LABELS.get(event.status, "Collecting tiles")

        if event.status.is_terminal:
            self.progress_widget.update_progress(
                self._overview_outcome(event.status, message)
            )
            # Hide the Done state after a moment, the same way spot burn does above.
            QTimer.singleShot(2000, self.progress_widget.reset_if_finished)
            return

        if event.completed is None or not event.total:
            # A state that carries no counts: a stage move, a stitch, a save. Nothing is
            # drawn for them, which leaves the last real count standing -- still true
            # while the stage moves or the mosaic is written.
            return

        if event.completed >= event.total:
            self.progress_widget.update_progress(ProgressUpdate.indeterminate(message))
        else:
            self.progress_widget.update_progress(
                ProgressUpdate.numeric(event.completed, event.total, message)
            )

    @staticmethod
    def _overview_outcome(status: TiledStatus, message: str) -> ProgressUpdate:
        """How a finished tiled acquisition reads once the bar is full.

        A cancel is deliberately not `failed`, which paints the bar red: it is someone
        getting what they asked for.
        """
        if status is TiledStatus.FAILED:
            return ProgressUpdate.failed(message)
        if status is TiledStatus.CANCELLED:
            return ProgressUpdate(finished=True, message=message)
        return ProgressUpdate.done()

    def _on_tile_acquisition_finished(self, result: dict) -> None:
        self.progress_widget.reset()
        tiles = result.get("tiles", 0)
        total = result.get("total", 0)
        elapsed = result.get("elapsed", 0.0)
        cancelled = result.get("cancelled", False)
        error: Exception | None = result.get("error", None)

        tile_info = f"{tiles}/{total} tiles" if total else ""
        elapsed_info = f" in {format_duration(elapsed)}" if elapsed else ""

        if error is not None:
            if cancelled:
                self.show_toast(
                    f"Tile acquisition cancelled. {tile_info} collected.", "warning"
                )
            else:
                self.show_toast(
                    f"Tile acquisition failed. {tile_info} collected. {error}", "error"
                )
        else:
            self.show_toast(
                f"Tile acquisition complete. {tile_info}{elapsed_info}.", "success"
            )

    def _on_tab_changed(self, index: int):
        """Handle tab change and update status bar."""
        self.status_bar.setStyleSheet(STATUS_BAR_STYLESHEET)

    def _create_main_tab(self):
        """Create the main AutoLamella tab."""
        # Create the embedded viewer container
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        # Viewer-less: the quad-view controller is the display, so no napari viewer is
        # created here. (The minimap keeps its own until FIB-405.)
        self.main_viewer = None

        # Create the AutoLamellaUI widget
        self.autolamella_ui = AutoLamellaUI(parent_ui=self)

        # Everything the workflow says without needing an answer arrives here
        self.autolamella_ui.workflow_status_signal.connect(self._on_workflow_status)
        self.autolamella_ui.queue_changed_signal.connect(self._on_queue_changed)
        self.autolamella_ui.step_update_signal.connect(self._on_step_update)
        self.autolamella_ui.experiment_update_signal.connect(self._on_experiment_update)
        self.autolamella_ui._workflow_finished_signal.connect(
            self._on_workflow_finished
        )
        self.autolamella_ui._hook_toast_signal.connect(self.show_toast)
        # Question lifecycle drives the agent watchdog (armed per prompt on
        # agent-designated tasks, disarmed by any answer). GUI thread: the
        # responder emits these where the lifecycle already runs.
        self.autolamella_ui.ui_responder.add_question_observer(self._on_question_event)
        notification_service._get_service().toast.connect(self._on_notification_service)
        self.autolamella_ui.system_widget.connected_signal.connect(
            self._on_microscope_connected
        )
        self.autolamella_ui.lamella_list.defect_changed.connect(
            self._on_lamella_defect_changed
        )
        self.autolamella_ui.lamella_list.lamella_selected.connect(
            self._on_experiment_lamella_selected
        )

        # hide menu bar
        self.autolamella_ui.menuBar().setVisible(False)
        self.autolamella_ui.setMinimumWidth(550)

        # Layout: quad view (left) | autolamella controls (right) via splitter.
        # The controller's SEM/FIB/FM canvases are the display and drive the control
        # widgets, which resolve it through parent -> parent_widget -> view_controller.
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        self.view_controller = MicroscopeViewController(parent=self)
        splitter.addWidget(self.view_controller.widget)
        splitter.addWidget(self.autolamella_ui)

        splitter.setSizes([700, 550])
        # set minimum width of right panel to 500
        splitter.widget(1).setMinimumWidth(500)
        layout.addWidget(splitter)
        self.tab_widget.addTab(
            container,
            fibsem_icon("mdi:microscope", color=GRAY_ICON_COLOR),
            "Microscope",
        )
        # The F5/Esc (View) and F2/F6/F9/F11 (Imaging) shortcuts are defined on QActions —
        # one source of truth for the menu item and its keybinding. Adding those actions to
        # the Microscope-tab container makes their WidgetWithChildrenShortcut scope resolve
        # against this tab, so the keys only fire when focus is inside it (not the minimap,
        # editor or workflow tabs).
        for action in (
            self.action_toggle_fullscreen,
            self.action_exit_fullscreen,
            *self._imaging_actions,
        ):
            container.addAction(action)

    def create_tabs(self):
        """Create the tabs for the AutoLamella UI."""
        self._create_main_tab()
        self.add_minimap_tab()
        self.add_overview_tab()
        self.add_protocol_editor_tab()
        self.add_lamella_editor_tab()
        self.add_workflow_tab()

        # add notification button to tab bar
        self.create_notification_button()

    def _on_experiment_update(self):
        """Handle experiment update signal and propagate to tabs."""

        if self.autolamella_ui is None:
            return
        if self.autolamella_ui.experiment is None:
            return

        self.minimap_widget.set_experiment()
        self.fm_overview_tab.refresh_experiment()
        self.beam_overview_tab.refresh_experiment()
        self.task_widget.set_experiment(self.autolamella_ui.experiment)
        self.lamella_widget.set_experiment()
        experiment = self.autolamella_ui.experiment
        if experiment is not None and experiment.task_protocol is not None:
            self.lamella_workflow_widget.set_experiment(experiment)
            self.lamella_workflow_widget.set_workflow_config(
                experiment.task_protocol.workflow_config
            )
            self.lamella_workflow_widget.set_options(experiment.task_protocol.options)

        # Set widget minimum widths (allows resize)
        self.autolamella_ui.setMinimumWidth(500)
        self.task_widget.setMinimumWidth(500)
        self.lamella_widget.setMinimumWidth(500)
        self.lamella_workflow_widget.setMinimumWidth(600)

        # Update the experiment name button
        self._update_experiment_header()

        # Show run workflow button when experiment is loaded
        self.run_workflow_btn.show()

        # enable all the tabs (except lamella tab, which is managed by _update_lamella_tab_enabled)
        lamella_tab_index = (
            self.tab_widget.indexOf(self._lamella_tab_container)
            if hasattr(self, "_lamella_tab_container")
            else -1
        )
        # The Overview tab is enabled by what the instrument has, not by an experiment,
        # so loading one must not switch it on for a system where neither modality can be
        # reached -- it would open onto an empty container.
        overview_tab_index = (
            self.tab_widget.indexOf(self.overview_tab)
            if getattr(self, "overview_tab", None) is not None
            and not self.overview_tab.is_available
            else -1
        )
        for i in range(self.tab_widget.count()):
            if i in (lamella_tab_index, overview_tab_index):
                continue
            self.tab_widget.setTabEnabled(i, True)

        self._update_instructions()

        # Rebuild lamella list and wire position events for the new experiment
        self._rebuild_lamella_list()
        self._on_workflow_selection_changed()  # evaluate after lamella are populated
        self.lamella_workflow_widget._update_summary()
        self._wire_position_events()

    def _wire_position_events(self):
        """Follow the current experiment's position list, and stop following the last.

        Everything that draws the set of lamellae -- the list rows, the cards, the FM
        overview canvas -- is rebuilt from these, so a display that is not reachable from
        here is a display that goes stale the moment a lamella is added anywhere else.
        That was how a position marked on the FIB/SEM overview never appeared on the FM
        one.
        """
        experiment = self.autolamella_ui.experiment if self.autolamella_ui else None
        if experiment is self._lamella_list_experiment:
            return

        if self._lamella_list_experiment is not None:
            events = self._lamella_list_experiment.positions.events
            try:
                events.inserted.disconnect(self._rebuild_lamella_list)
                events.removed.disconnect(self._rebuild_lamella_list)
                events.changed.disconnect(self._refresh_overview_positions)
            except Exception as e:
                logging.debug(f"Could not disconnect position events: {e}")

        # Bound methods rather than lambdas, so the disconnects above can find them.
        # psygnal passes each callback only as many arguments as it accepts, so a
        # no-argument handler needs no wrapper -- and a lambda cannot be disconnected by
        # naming the method it calls, which made the disconnect a silent no-op (psygnal
        # does not complain about disconnecting something that was never connected).
        # Nothing broke, because the experiment being disconnected from is on its way out
        # and never emits again, but it did not do what it said.
        if experiment is not None:
            events = experiment.positions.events
            events.inserted.connect(self._rebuild_lamella_list)
            events.removed.connect(self._rebuild_lamella_list)
            # `changed` means a lamella was edited in place rather than added or removed
            # -- `update_lamella_position_ui` emits it after replacing a milling pose.
            # The list rows do not care, but the canvas draws the positions themselves,
            # so it has to be told.
            events.changed.connect(self._refresh_overview_positions)
        self._lamella_list_experiment = experiment

    def _update_connection_chip(self):
        """Name the instrument this session attached to, or offer to attach one.

        Identity, not liveness. Nothing watches the link -- `connected_signal` is
        emitted off `bool(self.microscope)`, a Python reference (FIB-777) -- so a
        chip claiming "Connected" would keep claiming it through a pulled cable.
        What it connected to is a fact established once, and it does not decay.
        """
        if not self._connection_chip_enabled:
            return

        autolamella_ui = getattr(self, "autolamella_ui", None)
        microscope = autolamella_ui.microscope if autolamella_ui else None

        if microscope is None:
            # The same words as the Connection tab's button and the File menu
            # entry: three doors to one action should not each name it differently.
            self.btn_connection.setText("Connect to Microscope")
            self.btn_connection.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
            self.btn_connection.setToolTip("Choose a configuration and connect")
            return

        # `system.info` is a stored record, not a question put to the instrument.
        info = microscope.system.info
        self.btn_connection.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        self.btn_connection.setToolTip(
            f"{info.manufacturer} {info.model} at {info.ip_address}\n"
            f"Serial number: {info.serial_number}\n"
            "Click to change the connection."
        )

        label = f"{info.manufacturer}  {info.ip_address}"
        self.btn_connection.setText(label)
        overflow = self.btn_connection.sizeHint().width() - CONNECTION_CHIP_MAX_WIDTH
        if overflow > 0:
            metrics = self.btn_connection.fontMetrics()
            self.btn_connection.setText(
                metrics.elidedText(
                    label, Qt.ElideRight, metrics.horizontalAdvance(label) - overflow
                )
            )

    def _update_experiment_header(self):
        """Show either the create/load buttons or the experiment menu, never both.

        Until an experiment exists those two buttons are the only thing to do, so they
        stay primary-coloured and take the room. Once one is loaded the experiment name
        replaces them, and opens a menu holding the same actions: the greyed-out pair
        cost a third of the tab bar to say nothing.
        """
        autolamella_ui = getattr(self, "autolamella_ui", None)
        experiment = autolamella_ui.experiment if autolamella_ui else None
        is_connected = (
            autolamella_ui is not None and autolamella_ui.microscope is not None
        )

        # With the chip, neither button is on screen before there is a microscope:
        # a greyed-out pair beside a Connect chip is the same spent chrome the
        # loaded state used to carry. Without it there is nothing else in the header
        # to connect from, so they stay visible-but-disabled as they were. Both
        # branches go when the flag does (FIB-775).
        show = experiment is None and (
            is_connected or not self._connection_chip_enabled
        )
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

    def create_notification_button(self):
        """Add buttons to the tab bar for adding Protocol Editor, Lamella, and Minimap tabs."""
        # Create button container widget
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)
        button_layout.setContentsMargins(5, 0, 5, 0)
        button_layout.setSpacing(5)

        # Create / Load experiment buttons
        self.btn_create_experiment = QPushButton("Create Experiment")
        self.btn_create_experiment.setToolTip("Create a new experiment")
        self.btn_create_experiment.setEnabled(False)
        self.btn_create_experiment.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self.btn_create_experiment.clicked.connect(self._on_new_experiment)

        self.btn_load_experiment = QPushButton("Load Experiment")
        self.btn_load_experiment.setToolTip("Load an existing experiment")
        self.btn_load_experiment.setEnabled(False)
        self.btn_load_experiment.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        self.btn_load_experiment.clicked.connect(self._on_load_experiment)

        # Which instrument this session is attached to, and the way back to the
        # connection dialog -- see _update_connection_chip. Behind a flag while the
        # Connection tab is still how people connect (FIB-775); the dialog itself is
        # not gated, so File -> Connect to Microscope reaches it either way.
        self.btn_connection = QPushButton()
        self.btn_connection.setMaximumWidth(CONNECTION_CHIP_MAX_WIDTH)
        self.btn_connection.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.btn_connection.clicked.connect(self._on_connect_microscope)

        # The experiment name, once there is one, is itself the control that reaches
        # the create/load actions -- see _update_experiment_header.
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
        experiment_menu.addAction(self.action_new_experiment)
        experiment_menu.addAction(self.action_load_experiment)
        experiment_menu.addSeparator()
        experiment_menu.addAction(self.action_open_experiment_directory)
        self.btn_experiment_menu.setMenu(experiment_menu)

        # Notification bell
        self.notification_bell = NotificationBell(self)
        self.toast_manager.set_notification_bell(self.notification_bell)

        # Add widgets to layout
        if self._connection_chip_enabled:
            button_layout.addWidget(self.btn_connection)
        button_layout.addWidget(self.btn_experiment_menu)
        button_layout.addWidget(self.btn_create_experiment)
        button_layout.addWidget(self.btn_load_experiment)
        button_layout.addWidget(self.notification_bell)

        # Add to tab widget corner
        self.tab_widget.setCornerWidget(button_widget)

        # The chip has a state before anything connects, and only the connect
        # handler would otherwise ever set one.
        self._update_connection_chip()

    def add_protocol_editor_tab(self):
        """Add the protocol editor as a separate tab with its own viewer."""
        container = QWidget(parent=self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create the protocol editor widget (viewer is created internally)
        self.task_widget = AutoLamellaProtocolTaskConfigEditor(
            parent=self.autolamella_ui
        )
        self.autolamella_ui.system_widget.connected_signal.connect(
            self.task_widget._on_microscope_connected
        )
        layout.addWidget(self.task_widget)
        self.tab_widget.addTab(
            container,
            fibsem_icon("mdi:file-document-edit", color=GRAY_ICON_COLOR),
            "Protocol",
        )

        # disable the tab by default
        self.tab_widget.setTabEnabled(self.tab_widget.indexOf(container), False)

    def add_lamella_editor_tab(self):
        """Consolidated Lamella tab: 1-column card strip (left) + Images/Protocol sub-tabs (right)."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        outer_splitter = QSplitter(Qt.Horizontal)
        outer_splitter.setChildrenCollapsible(True)

        # ── Left: 1-column card strip ──────────────────────────────────────
        self.lamella_card_container = LamellaCardContainer(
            columns=1, mode=self._preferences.display.lamella_card_mode
        )
        self.lamella_card_container.defect_changed.connect(
            self._on_lamella_defect_changed
        )
        self.lamella_card_container.lamella_selected.connect(
            self._on_lamella_card_selected
        )
        self.lamella_card_container.move_to_requested.connect(self._on_lamella_move_to)
        self.lamella_card_container.update_position_requested.connect(
            self._on_lamella_card_update_position
        )
        self.lamella_card_container.remove_requested.connect(
            self._on_lamella_remove_requested
        )

        card_scroll = QScrollArea()
        card_scroll.setWidget(self.lamella_card_container)
        card_scroll.setWidgetResizable(True)
        card_scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )

        card_scroll.setMaximumWidth(340)

        outer_splitter.addWidget(card_scroll)
        outer_splitter.setStretchFactor(0, 0)

        # ── Right: sub-tab widget ──────────────────────────────────────────
        right_tabs = QTabWidget()

        # Review tab
        self.lamella_task_image_widget = LamellaTaskImageWidget()

        # Protocol tab: matplotlib canvas (left) + editor (right). The editor owns its
        # own controller, built eagerly so the splitter can embed the canvas before a
        # microscope connects.
        self.lamella_widget = AutoLamellaProtocolEditorWidget(
            parent=self.autolamella_ui,
        )
        self.autolamella_ui.system_widget.connected_signal.connect(
            self.lamella_widget._on_microscope_connected
        )
        self.lamella_widget.setMinimumWidth(550)

        protocol_splitter = QSplitter(Qt.Horizontal)
        protocol_splitter.setChildrenCollapsible(False)
        protocol_splitter.addWidget(self.lamella_widget.view_controller.widget)
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.lamella_widget)
        scroll_area.setWidgetResizable(True)
        protocol_splitter.addWidget(scroll_area)
        protocol_splitter.setSizes([700, 550])

        right_tabs.addTab(protocol_splitter, "Protocol")
        right_tabs.addTab(self.lamella_task_image_widget, "Review")

        outer_splitter.addWidget(right_tabs)
        outer_splitter.setStretchFactor(1, 1)
        outer_splitter.setSizes([340, 99999])

        layout.addWidget(outer_splitter)
        self.tab_widget.addTab(
            container, fibsem_icon("mdi:layers", color=GRAY_ICON_COLOR), "Lamella"
        )
        self._lamella_tab_container = container

        index = self.tab_widget.indexOf(container)
        self.tab_widget.setTabEnabled(index, False)
        self.tab_widget.setTabToolTip(index, "Add lamella positions to enable this tab")

        self._on_lamella_card_selected(None)

    def add_workflow_tab(self):
        """Add the workflow tab with the combined lamella + workflow widget."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        self.lamella_workflow_widget = LamellaWorkflowWidget()
        self.lamella_workflow_widget.lamella_move_to_requested.connect(
            self._on_lamella_move_to
        )
        self.lamella_workflow_widget.lamella_edit_requested.connect(
            self._on_lamella_edit
        )
        self.lamella_workflow_widget.lamella_remove_requested.connect(
            self._on_lamella_remove_requested
        )
        self.lamella_workflow_widget.lamella_defect_changed.connect(
            self._on_lamella_defect_changed
        )

        # Alias so existing methods (_rebuild_lamella_list etc.) keep working unchanged
        self.lamella_list_widget = self.lamella_workflow_widget.lamella_list

        # Workflow task signals — each change persists the updated config to disk
        self.lamella_workflow_widget.task_supervised_changed.connect(
            self._save_workflow_config
        )
        self.lamella_workflow_widget.task_edited.connect(self._save_workflow_config)
        self.lamella_workflow_widget.task_remove_requested.connect(
            self._save_workflow_config
        )
        self.lamella_workflow_widget.task_order_changed.connect(
            self._save_workflow_config
        )
        self.lamella_workflow_widget.task_added.connect(self._save_workflow_config)

        # Workflow info signals — name/description/options changes also persist
        self.lamella_workflow_widget.workflow_name_changed.connect(
            self._save_workflow_config
        )
        self.lamella_workflow_widget.workflow_description_changed.connect(
            self._save_workflow_config
        )
        self.lamella_workflow_widget.workflow_options_changed.connect(
            self._save_workflow_config
        )

        # Selection signals — update run button enabled state
        self.lamella_workflow_widget.lamella_selection_changed.connect(
            self._on_workflow_selection_changed
        )
        self.lamella_workflow_widget.task_selection_changed.connect(
            self._on_workflow_selection_changed
        )

        self.workflow_right_panel = QWidget()
        self.workflow_right_panel.setStyleSheet(f"background: {SURFACE_COLOR};")

        _rp_layout = QVBoxLayout(self.workflow_right_panel)
        _rp_layout.setContentsMargins(0, 0, 0, 0)
        _rp_layout.setSpacing(0)
        self.workflow_timeline = WorkflowProgressWidget()
        self.workflow_timeline.queue_action_requested.connect(self._on_queue_action)
        self.workflow_timeline.add_to_queue_requested.connect(self._on_add_to_queue)
        _rp_layout.addWidget(self.workflow_timeline)

        splitter.addWidget(self.lamella_workflow_widget)
        splitter.addWidget(self.workflow_right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)
        self.tab_widget.addTab(
            container,
            fibsem_icon("mdi:play-circle-outline", color=GRAY_ICON_COLOR),
            "Workflow",
        )

        # disable the workflow tab by default
        self.tab_widget.setTabEnabled(self.tab_widget.indexOf(container), False)

        # Track which experiment's position events we're connected to
        self._lamella_list_experiment = None

        # Guard against bidirectional selection sync loops
        self._syncing_selection = False

        # Connect protocol editor → workflow tab (deferred here since task_widget is created first)
        self.task_widget.workflow_config_changed.connect(
            self.lamella_workflow_widget.set_workflow_config
        )

    def _on_lamella_card_selected(self, lamella: Lamella | None):
        """Update task image panel and protocol editor for selected lamella card."""
        if not hasattr(self, "lamella_task_image_widget"):
            return

        self._selected_card_lamella = lamella
        self.fm_overview_tab.set_selected(lamella)
        self.beam_overview_tab.set_selected(lamella)
        self.lamella_task_image_widget.set_lamella(lamella)
        if lamella is not None:
            self.lamella_widget.select_lamella(lamella.name)
            if not self._syncing_selection:
                self._syncing_selection = True
                try:
                    self.autolamella_ui.lamella_list.select(lamella.name)
                    if hasattr(self, "minimap_widget"):
                        self.minimap_widget.lamella_list.select(lamella.name)
                finally:
                    self._syncing_selection = False

    def _on_experiment_lamella_selected(self, lamella):
        """Sync card container and minimap when experiment-tab list selection changes."""
        self.fm_overview_tab.set_selected(lamella)
        self.beam_overview_tab.set_selected(lamella)
        if getattr(self, "_syncing_selection", False) or lamella is None:
            return
        if not hasattr(self, "lamella_card_container"):
            return
        self._syncing_selection = True
        try:
            self.lamella_card_container.select_lamella(lamella.name)
            if hasattr(self, "minimap_widget"):
                self.minimap_widget.lamella_list.select(lamella.name)
        finally:
            self._syncing_selection = False

    def _on_minimap_lamella_selected(self, lamella):
        """Sync experiment list and card container when minimap list selection changes."""
        self.fm_overview_tab.set_selected(lamella)
        self.beam_overview_tab.set_selected(lamella)
        if getattr(self, "_syncing_selection", False) or lamella is None:
            return
        self._syncing_selection = True
        try:
            self.autolamella_ui.lamella_list.select(lamella.name)
            if hasattr(self, "lamella_card_container"):
                self.lamella_card_container.select_lamella(lamella.name)
        finally:
            self._syncing_selection = False

    def _save_workflow_config(self, *_args):
        """Persist the current task list to the experiment protocol after any task change."""
        if self.autolamella_ui is None or self.autolamella_ui.experiment is None:
            return
        protocol = self.autolamella_ui.experiment.task_protocol
        if protocol is None:
            return
        protocol.workflow_config.tasks[:] = self.lamella_workflow_widget.get_tasks()
        self.autolamella_ui._on_workflow_config_changed(protocol.workflow_config)

    def _on_add_to_queue(self, run_next: bool) -> None:
        """Add the left panel's current selection to the running queue.

        The selection UI is LamellaWorkflowWidget, which stays live during a run
        and is cleared when one is launched — so it is empty and free mid-run.
        Reusing it avoids a second lamella-and-task picker that would have to be
        kept in step with the first.
        """
        manager = getattr(self.autolamella_ui, "_task_manager", None)
        if manager is None:
            return

        lamellae = self.lamella_workflow_widget.get_selected_lamella()
        tasks = self.lamella_workflow_widget.get_selected_tasks()
        if not lamellae or not tasks:
            self._show_queue_message(
                "Select at least one lamella and one task to add to the queue."
            )
            return

        lamella_names = [lam.name for lam in lamellae]
        task_names = [t.name for t in tasks]

        pending = {(i.lamella_name, i.task_name) for i in manager.queue.pending}
        already = [
            f"{t} for {ln}"
            for t in task_names
            for ln in lamella_names
            if (ln, t) in pending
        ]

        # Task-outer, lamella-inner: the order the items are really inserted in below,
        # so what is priced is what will be queued.
        pairs = [(ln, tn) for tn in task_names for ln in lamella_names]
        if not confirm_add_to_queue_dialog(
            lamella_names,
            task_names,
            run_next,
            already,
            estimate=self._estimate_addition(manager, pairs, run_next),
            parent=self,
        ):
            return

        # A lamella created before a task joined the protocol has no config for
        # it, and run_task raises on that. Backfill from the base protocol first.
        experiment = self.autolamella_ui.experiment
        missing = [
            ln
            for ln, lam in zip(lamella_names, lamellae)
            if any(t not in lam.task_config for t in task_names)
        ]
        if missing and experiment is not None:
            experiment.apply_lamella_config(missing, task_names)

        # Task-outer, lamella-inner, matching how build_from_matrix lays out the
        # original queue, so added work interleaves the same way.
        #
        # For "run next" each item is anchored after the previous one rather than
        # to the front: front=True on every add would put each new item ahead of
        # the last, reversing the batch.
        added, anchor = 0, None
        for task_name in task_names:
            for lamella_name in lamella_names:
                if not run_next:
                    item = manager.queue.add(lamella_name, task_name)
                elif anchor is None:
                    item = manager.queue.add(lamella_name, task_name, front=True)
                else:
                    item = manager.queue.add(lamella_name, task_name, after=anchor)
                if item is not None:
                    anchor = item.id
                    added += 1

        manager.notify_queue_changed()
        where = "to run next" if run_next else "to the queue"
        self._show_queue_message(f"Added {added} task(s) {where}.")

        # Clear the selection, as starting a workflow does — the work is committed,
        # and leaving it ticked invites adding the same batch twice.
        self.lamella_workflow_widget.lamella_list.set_all_selected(False)
        self.lamella_workflow_widget.workflow.set_all_selected(False)

    def _on_queue_changed(self, info: dict) -> None:
        """The queue was edited between tasks — no task lifecycle to hang it off."""
        items = info.get("queue_items", [])
        # An added item can be a (lamella, task) pair the launch matrix never held, so
        # the estimates are recomputed here rather than only at the start.
        self._push_timeline_estimates([(i.lamella_name, i.task_name) for i in items])
        self.workflow_timeline.refresh_queue(items)

    def _estimate_addition(self, manager, pairs: list, run_next: bool):
        """What adding `pairs` would cost the running queue, or None if it cannot say.

        Priced from the same `estimated_duration` the pre-flight dialog and the timeline
        use, over a snapshot of the live queue — `queue.items` hands out copies, so the
        hypothetical is built without touching what is running.
        """
        experiment = getattr(self.autolamella_ui, "experiment", None)
        if experiment is None:
            return None

        lamella_by_name = {lam.name: lam for lam in experiment.positions}

        def seconds_for(item):
            lamella = lamella_by_name.get(item.lamella_name)
            if lamella is None:
                return None
            config = lamella.task_config.get(item.task_name)
            if config is None:
                return None
            try:
                return config.estimated_duration
            except Exception:
                # `estimated_time` divides by the sputter rate (milling/base.py:292), so
                # a hand-edited protocol can raise. Losing the figure is a great deal
                # better than losing the dialog that queues the work.
                logging.warning(
                    "Could not estimate duration for %s on %s.",
                    item.task_name,
                    item.lamella_name,
                    exc_info=True,
                )
                return None

        schedule = {}
        protocol = experiment.task_protocol
        if protocol is not None:
            schedule = {
                task.name: task.scheduled_at
                for task in protocol.workflow_config.tasks
                if task.scheduled_at is not None
            }

        # Wall-clock since the running task started, which is what the recorded duration
        # measures too. It counts any supervised wait, so the absolute finish can run a
        # little optimistic — but both walks share it, so the *delay* is unaffected.
        active_elapsed = None
        active = manager.queue.active
        if active is not None:
            lamella = lamella_by_name.get(active.lamella_name)
            if lamella is not None and lamella.task_state.start_timestamp:
                active_elapsed = max(
                    0.0, time.time() - lamella.task_state.start_timestamp
                )

        return estimate_addition(
            manager.queue.items,
            pairs,
            seconds_for,
            run_next=run_next,
            schedule=schedule,
            active_elapsed=active_elapsed,
        )

    def _push_timeline_estimates(self, pairs: list) -> None:
        """Give the timeline the per-item durations, and the schedule they walk past.

        Computed here rather than inside the widget: an estimate comes from a lamella's
        task config, and the timeline is generic — it has no business reaching for an
        Experiment. Recomputed on workflow start and on a queue edit only, never per
        status update: `estimated_duration` walks every milling stage of every task, and
        the status path is the one already watched for latency (FIB-683).
        """
        experiment = getattr(self.autolamella_ui, "experiment", None)
        if experiment is None:
            return

        lamella_by_name = {lam.name: lam for lam in experiment.positions}
        estimates = {}
        for lamella_name, task_name in pairs:
            lamella = lamella_by_name.get(lamella_name)
            if lamella is None:
                continue
            config = lamella.task_config.get(task_name)
            # A lamella with no config for the task has no duration to offer, and
            # inventing one would be worse than leaving the column empty.
            if config is None:
                continue
            try:
                estimates[(lamella_name, task_name)] = config.estimated_duration
            except Exception:
                # Deliberately caught, against this codebase's fail-fast default. This
                # runs in a slot, and PyQt5 aborts the process on an unhandled exception
                # in one (FIB-329) — which here would kill the worker thread mid-mill
                # over a decorative number. `estimated_time` divides by the sputter rate
                # (milling/base.py:292), so a hand-edited protocol can reach a
                # ZeroDivisionError. The column already has a well-defined "no estimate
                # to offer" state, so degrading into it is the graceful failure.
                logging.warning(
                    "Could not estimate duration for %s on %s; the timeline will show "
                    "no estimate for it.",
                    task_name,
                    lamella_name,
                    exc_info=True,
                )
        self.workflow_timeline.set_estimates(estimates)

        protocol = experiment.task_protocol
        schedule = {}
        if protocol is not None:
            schedule = {
                task.name: task.scheduled_at
                for task in protocol.workflow_config.tasks
                if task.scheduled_at is not None
            }
        self.workflow_timeline.set_schedule(schedule)

    def _on_queue_action(self, action: str, item_id: str) -> None:
        """Apply a timeline row action to the live queue.

        The timeline emits an intent keyed on WorkItem.id and nothing more; the
        queue is reached from here because the widget is generic and has no
        business knowing what a queue is.
        """
        manager = getattr(self.autolamella_ui, "_task_manager", None)
        if manager is None:
            return

        queue = manager.queue
        item = next((i for i in queue.items if i.id == item_id), None)
        if item is None:
            self._show_queue_message("That task is no longer in the queue.")
            return
        label = f"{item.task_name} for {item.lamella_name}"

        if action == "stop_task":
            # Confirmed, unlike Remove: that edits a list, this interrupts an
            # operation already under way on the sample.
            # Built rather than QMessageBox.question(...): the difference from Stop
            # is the point of asking at all, and everything in the text slot renders
            # at question weight. The explanation belongs in the informative slot.
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Question)
            box.setWindowTitle("Stop Task")
            box.setText(f"Stop <b>{label}</b>?")
            box.setInformativeText(
                "It will be marked cancelled and the workflow will carry on with "
                "the next queued task. Use Stop to end the whole run instead."
            )
            box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            box.setDefaultButton(QMessageBox.No)
            if box.exec_() != QMessageBox.Yes:
                return
            # Two halves, as Stop Workflow has: tell the queue this task is being
            # abandoned, and bring the hardware to a halt. Without the second, the
            # task stops waiting for the mill but the mill keeps going.
            manager.stop_task()
            self.autolamella_ui.stop_current_operations()
            # No queue mutation here: the task unwinds on the worker thread and the
            # status it emits on the way out is what updates the timeline.
            self._show_queue_message(f"Stopping {label}...")
            return

        if action == "run_again":
            # A fresh item rather than a rewind: the original attempt still
            # happened and stays in the run record.
            added = queue.add(item.lamella_name, item.task_name, front=True)
            message = (
                f"Queued {label} to run next."
                if added is not None
                else f"Could not queue {label}."
            )
        else:
            call = {
                "move_up": lambda: queue.nudge(item_id, -1),
                "move_down": lambda: queue.nudge(item_id, +1),
                "run_next": lambda: queue.move_to_front(item_id),
                "remove": lambda: queue.remove(item_id),
            }.get(action)
            if call is None:
                logging.warning(f"Unknown queue action from the timeline: {action}")
                return
            message = self._queue_action_message(action, label, call())

        manager.notify_queue_changed()
        self._show_queue_message(message)

    @staticmethod
    def _queue_action_message(action: str, label: str, result: QueueResult) -> str:
        """Describe the outcome of a queue action, or "" to say nothing.

        The row may have started running between the menu opening and the click,
        which is the whole reason the queue returns a structured result rather
        than a bool — say so instead of appearing to do nothing.
        """
        if result.op is QueueOp.NO_OP:
            return ""  # already first or last; the user can see that
        if result.op is QueueOp.NOT_PENDING:
            return f"{label} is already running."
        if not result.ok:
            return f"{label} is no longer in the queue."
        return {
            "move_up": f"Moved {label} up.",
            "move_down": f"Moved {label} down.",
            "run_next": f"{label} will run next.",
            "remove": f"Removed {label} from the queue.",
        }.get(action, "")

    def _show_queue_message(self, message: str) -> None:
        """Transient confirmation for a queue edit — no modal, no interruption."""
        if message and self.status_bar is not None:
            self.status_bar.showMessage(message, 4000)

    def _on_workflow_status(self, event: "WorkflowStatusEvent"):
        """Handle a fire-and-forget status update, from workflow_status_signal.

        The one channel for everything the workflow says without needing an
        answer — the dict workflow_update_signal is gone. The run indicators
        refresh on every event because any of them can change what they show: a
        lifecycle report starts or finishes the run, and the responder pings
        this signal when a question flips the waiting state.
        """
        # transient status-bar messages (e.g. scheduled-start countdown)
        if event.status_bar is not None and self.status_bar is not None:
            self.status_bar.showMessage(event.status_bar)

        if event.report is not None:
            self._apply_status_report(event.report)

        if self.autolamella_ui is None:
            return
        self._refresh_workflow_indicators()

    def _apply_status_report(self, report: "WorkflowStatusUpdate") -> None:
        """Render one task-lifecycle report: timeline, run message, list refreshes."""
        # No build-once guard: update_from_status reconciles rows against the
        # snapshot by WorkItem id, so first paint and every later change go
        # through the same path.
        self.workflow_timeline.update_from_status(report)

        task_name = report.task_name
        lamella_name = report.item_name
        queue_position = report.queue_position
        queue_total = report.queue_total
        status = report.status

        # Position in the live queue, not the launch matrix — stays correct
        # when the queue is added to or reordered mid-run.
        txt = f"Workflow: {task_name} | {lamella_name}"
        # `queue_total` is a count, so 0 means "nothing to be in the middle of"
        # rather than "unknown" -- truthiness, not `is not None`, or a malformed
        # payload renders "3/0".
        if queue_position is not None and queue_total:
            txt += f" | {queue_position}/{queue_total}"

        self.set_workflow_running(txt)

        # update current task
        self._current_task_name = task_name

        # Lock editor when the active lamella/task is being processed
        if status is AutoLamellaTaskStatus.InProgress:
            self.lamella_widget.set_active_lamella_name(lamella_name, task_name)
        else:
            self.lamella_widget.set_active_lamella_name(None)

        # Refresh only the affected lamella if we can identify it
        lamella = None
        experiment = self.autolamella_ui.experiment
        if experiment is not None and lamella_name is not None:
            lamella = experiment.get_lamella_by_name(lamella_name)

        # The name list is refreshed from here rather than subscribing per row.
        # `_LamellaRow` used to connect to `task_state.events.name`/`.status`, which
        # the workflow writes from its worker thread; the marshalled refresh then
        # arrived after the row had been replaced and its widgets destroyed
        # (`RuntimeError: wrapped C/C++ object of type QLabel has been deleted`).
        # It joins the other two on the same named-lamella path, so it costs one row
        # -- unlike them, though, nothing else was refreshing this list mid-workflow.
        if lamella is not None:
            self.lamella_list_widget.refresh_lamella(lamella)
            self.lamella_card_container.refresh_lamella(lamella)
            self.autolamella_ui.lamella_list.refresh_lamella(lamella)
        else:
            self.lamella_list_widget.refresh_all()
            self.lamella_card_container.refresh_all()
            self.autolamella_ui.lamella_list.refresh_all()
        self._on_lamella_card_selected(getattr(self, "_selected_card_lamella", None))

    def _on_agent_server_dialog(self) -> None:
        """Open the session dialog: status, token, and scope arming."""
        from fibsem.applications.autolamella.ui.agent_server_dialog import (
            AgentServerDialog,
        )

        dialog = AgentServerDialog(
            host_provider=lambda: getattr(
                self.autolamella_ui, "_agent_server_host", None
            ),
            parent=self,
        )
        dialog.exec_()

    def _watchdog_ms(self) -> int:
        """The watchdog deadline, from Preferences → Agent (constant fallback)."""
        try:
            minutes = fibsem_cfg.load_user_preferences().agent.watchdog_minutes
            return max(1, int(minutes)) * 60 * 1000
        except Exception:
            return AGENT_WATCHDOG_MS

    def _agent_seconds_since_seen(self) -> Optional[float]:
        """Seconds since the agent's token last made a request, or None."""
        host = getattr(self.autolamella_ui, "_agent_server_host", None)
        if host is None:
            return None
        try:
            return host.agent_seconds_since_seen()
        except Exception:
            return None

    def _agent_presumed_gone(self) -> bool:
        """Nobody is on the other end: never connected, or silent too long."""
        age = self._agent_seconds_since_seen()
        return age is None or age > AGENT_PRESUMED_GONE_S

    def _on_question_event(self, kind: str, payload: dict) -> None:
        """GUI thread, from the responder: arm/disarm the agent watchdog."""
        if kind == "prompt_raised":
            self._agent_watchdog_expired = False
            if self._agent_supervision_active(self._current_task_name):
                if self._agent_presumed_gone():
                    # Don't park a question for an agent that isn't there.
                    self._hand_question_to_operator(
                        "The agent hasn't been in touch — this question is yours."
                    )
                    return
                self._agent_watchdog.start(self._watchdog_ms())
                self._agent_liveness_check.start()
            else:
                self._agent_watchdog.stop()
                self._agent_liveness_check.stop()
        elif kind in ("prompt_answered", "prompt_cancelled"):
            self._agent_watchdog.stop()
            self._agent_liveness_check.stop()
            self._agent_watchdog_expired = False
        self._refresh_workflow_indicators()

    def _on_agent_watchdog_expired(self) -> None:
        """The agent went quiet: the standing question becomes the operator's."""
        minutes = max(1, self._watchdog_ms() // 60000)
        self._hand_question_to_operator(
            f"The agent hasn't answered in {minutes} minutes — "
            "this question is now yours."
        )

    def _on_agent_liveness_check(self) -> None:
        """Periodic while a question parks on the agent's clock: hand over the
        moment the agent stops being heard from, not at the full deadline."""
        if self._agent_presumed_gone():
            self._hand_question_to_operator(
                "The agent hasn't been in touch — this question is yours."
            )

    def _hand_question_to_operator(self, message: str) -> None:
        """Escalate the standing question: ordinary waiting chrome + a toast
        saying why. Informs, never revokes — a late agent answer still applies,
        and the first writer still wins."""
        self._agent_watchdog.stop()
        self._agent_liveness_check.stop()
        if not self.autolamella_ui.WAITING_FOR_USER_INTERACTION:
            return  # the answer raced the escalation; nothing is standing
        self._agent_watchdog_expired = True
        notification_service.show_toast(message, "warning")
        # The ordinary waiting chrome (orange border, attention button, sound)
        # takes over below, now that agent_holding no longer suppresses it.
        self._refresh_workflow_indicators()

    def _refresh_workflow_indicators(self) -> None:
        """Re-read the waiting/supervised/running state into the window chrome."""
        # refresh the supervised status chip
        self._update_supervised_status()

        waiting = self.autolamella_ui.WAITING_FOR_USER_INTERACTION
        # A question addressed to a running agent is not (yet) a wait for the
        # operator: the chrome stays agent-purple and quiet while the watchdog
        # counts down. Expiry — or a human-designated question — is what turns
        # this into the ordinary waiting state.
        agent_holding = (
            waiting
            and not self._agent_watchdog_expired
            and self._agent_supervision_active(self._current_task_name)
        )
        # The timeline freezes its countdown on this: a wait for an answer is not
        # machine time whoever is answering, and left running it would spend the
        # estimate while nothing is happening.
        self.workflow_timeline.set_waiting_for_user(waiting)
        if waiting and not agent_holding:
            # Show user attention button and change status bar color
            self.user_attention_btn.show()
            # Play notification sound once when entering waiting state
            if not self._user_interaction_sound_played and self._sound_enabled:
                play_notification_sound()
                self._user_interaction_sound_played = True
        else:
            # Hide user attention button and reset to original dark theme
            self.user_attention_btn.hide()
            self._user_interaction_sound_played = False  # Reset for next time

        # Update border to reflect current workflow state
        if self._border_state == "stopping":
            pass  # Keep the red border until the workflow finishes unwinding
        elif waiting and agent_holding:
            self._set_border_state("agent")
        elif waiting:
            self._set_border_state("waiting")
        elif self.autolamella_ui.WORKFLOW_PENDING:
            self._set_border_state("pending")
        elif self.autolamella_ui.is_workflow_running:
            self._set_border_state(self._running_border_state(self._current_task_name))
        else:
            self._set_border_state("idle")

    def _rebuild_lamella_list(self):
        """Clear and repopulate the lamella list and card container from the current experiment."""
        if not hasattr(self, "lamella_list_widget"):
            return
        experiment = self.autolamella_ui.experiment if self.autolamella_ui else None
        self.lamella_list_widget.clear()
        self.lamella_card_container.clear()
        self._on_lamella_card_selected(None)
        # The overview canvases are further displays of the same set, so they are
        # rebuilt here rather than from subscriptions of their own -- one handler for
        # "the lamellae changed" means the displays cannot end up disagreeing about what
        # the experiment holds. Before the `experiment is None` return, because an
        # experiment closing has to clear them as much as an experiment opening has to
        # fill them.
        self._refresh_overview_positions()
        if experiment is None:
            return
        for lamella in experiment.positions:
            self.lamella_list_widget.add_lamella(lamella)
            self.lamella_card_container.add_lamella(lamella)
        self._on_workflow_selection_changed()
        self._update_lamella_tab_enabled()

    def _update_lamella_tab_enabled(self):
        """Enable or disable the Lamella tab based on whether positions exist."""
        if not hasattr(self, "_lamella_tab_container"):
            return
        index = self.tab_widget.indexOf(self._lamella_tab_container)
        if index < 0:
            return
        experiment = self.autolamella_ui.experiment if self.autolamella_ui else None
        has_positions = experiment is not None and len(experiment.positions) > 0
        self.tab_widget.setTabEnabled(index, has_positions)
        self.tab_widget.setTabToolTip(
            index, "" if has_positions else "Add lamella positions to enable this tab"
        )

    def _on_lamella_move_to(self, lamella: "Lamella"):
        """Move the stage to the given lamella's milling position."""
        if self.autolamella_ui is None or self.autolamella_ui.experiment is None:
            return
        self.autolamella_ui.lamella_list.select(lamella.name)
        self.autolamella_ui.move_to_lamella_position()

    def _on_lamella_card_update_position(self, lamella: "Lamella"):
        """Update the stage position of the given lamella to the current stage position."""
        if self.autolamella_ui is None:
            return
        self.autolamella_ui.lamella_list.select(lamella.name)
        self.autolamella_ui.update_lamella_position_ui()

    def _on_lamella_edit(self, lamella: "Lamella"):
        """Switch to the Lamella tab and select the given lamella in the protocol editor."""
        if self.autolamella_ui is None or self.autolamella_ui.experiment is None:
            return
        self.autolamella_ui.lamella_list.select(lamella.name)

        # Select the lamella in the protocol editor
        self.lamella_widget.select_lamella(lamella.name)

        # Switch to the Lamella tab
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == "Lamella":
                self.tab_widget.setCurrentIndex(i)
                break

    def _on_lamella_defect_changed(self, lamella: "Lamella"):
        """Persist defect state change to disk and sync all widgets."""
        if self.autolamella_ui is None or self.autolamella_ui.experiment is None:
            return
        self.autolamella_ui.experiment.save()
        # Sync defect icon across all widgets
        self.autolamella_ui.lamella_list.refresh_all()
        self.lamella_list_widget.refresh_lamella(lamella)
        self.lamella_card_container.refresh_lamella(lamella)

    def _on_lamella_remove_requested(self, lamella: "Lamella"):
        """Remove the given lamella from the experiment after the list widget has already removed its row."""
        if self.autolamella_ui is None or self.autolamella_ui.experiment is None:
            return
        try:
            idx = self.autolamella_ui.experiment.positions.index(lamella)
        except ValueError:
            return
        self.autolamella_ui.experiment.positions.pop(idx)
        self.autolamella_ui.experiment.save()
        self.autolamella_ui.update_lamella_combobox(latest=True)
        self.autolamella_ui.update_ui()

    def _on_step_update(self, label: str) -> None:
        """Handle per-step update from the workflow worker thread."""
        self.workflow_timeline.update_step(label)

    def _on_workflow_finished(self, cancelled: bool = False):
        """Handle workflow finished signal."""
        # The next run's items carry new ids, so the timeline rebuilds itself on
        # its first status update — nothing to reset here.
        # Resolve any outer row left in ACTIVE state (e.g. if workflow was cancelled)
        self.workflow_timeline.finish_current_step(failed=cancelled)
        self.workflow_timeline.clear_steps()
        self.hide_workflow_running()
        self.lamella_widget.set_active_lamella_name(None)
        self.user_attention_btn.hide()
        self.lamella_list_widget.refresh_all()
        self.lamella_card_container.refresh_all()
        if self.status_bar is not None:
            self.status_bar.showMessage("Workflow: Finished")
            self.status_bar.setStyleSheet(STATUS_BAR_STYLESHEET)
        self._set_border_state("idle")

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

        # Create the minimap widget
        self.minimap_widget = FibsemMinimapWidget(
            viewer=self.minimap_viewer, parent=self.autolamella_ui
        )
        self.minimap_widget.setMinimumWidth(500)
        self.minimap_widget._acquisition_finished.connect(
            self._on_tile_acquisition_finished
        )
        self.minimap_widget.lamella_list.lamella_selected.connect(
            self._on_minimap_lamella_selected
        )

        # Layout: napari viewer (left) | minimap controls (right) via splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        splitter.addWidget(self.minimap_viewer.window._qt_window)
        splitter.addWidget(self.minimap_widget)

        splitter.setSizes([700, 500])
        layout.addWidget(splitter)
        # "Minimap", not "Overview" any more: the canvas Overview tab replaced this one
        # and took the name. Two tabs both called Overview would leave a user guessing
        # which is which for exactly as long as this tab survives, and the internal name
        # for it has always been the minimap (`add_minimap_tab`, `FibsemMinimapWidget`).
        self.tab_widget.insertTab(
            1, container, fibsem_icon("mdi:map", color=GRAY_ICON_COLOR), "Minimap"
        )
        # Kept on the window so `_apply_napari_overview_visibility` can find the tab. It
        # goes with the tab (FIB-405).
        self._minimap_tab_container = container

        # disable the tab by default
        self.tab_widget.setTabEnabled(self.tab_widget.indexOf(container), False)
        self._apply_napari_overview_visibility()

    def add_overview_tab(self):
        """Reserve the Overview tab: both modalities, one tab.

        The FIB/SEM and fluorescence overviews were two tabs here, and are now two pages
        of one -- `AutoLamellaOverviewContainerTab` holds both host tabs and a chip strip
        that chooses between them (FIB-780). Both host tabs stay built and stay
        subscribed while the other is showing, so everything the window tells them is
        still told to both.

        The tab is created on every system and never hidden, so the tab bar keeps the
        same shape whatever the instrument turned out to be. When neither modality has
        anything to drive it is greyed out with a tooltip saying why -- see
        :meth:`_on_overview_availability`. The widgets inside are built and destroyed to
        match, so a dead tab costs nothing but the container.

        The tab is created empty. Both overview widgets require a microscope at
        construction -- every scale on their canvases comes from the instrument -- and at
        this point there may be no microscope at all. The container fills itself in on
        connection and says so through `availability_changed`.
        """
        self.overview_tab = AutoLamellaOverviewContainerTab(self.autolamella_ui)
        # The two host tabs, under the names the window has always used for them. Aliases
        # rather than a rename: these are the same objects, every lifecycle call the
        # window makes still goes to the tab that owns the answer, and
        # `test_overview_tab_wiring.py` still reads the calls out of this source.
        self.fm_overview_tab = self.overview_tab.fm_tab
        self.beam_overview_tab = self.overview_tab.beam_tab

        self.overview_tab.availability_changed.connect(self._on_overview_availability)
        self.fm_overview_tab.lamella_selected.connect(
            self._on_fm_overview_lamella_selected
        )
        self.beam_overview_tab.lamella_selected.connect(
            self._on_beam_overview_lamella_selected
        )
        # Per host tab rather than through the container's own `acquiring_changed`: the
        # lock is derived from both tabs every time it is applied, and connecting to each
        # keeps the two sources of that derivation the same two objects it reads.
        self.fm_overview_tab.acquiring_changed.connect(self._apply_overview_locks)
        self.beam_overview_tab.acquiring_changed.connect(self._apply_overview_locks)

        self.tab_widget.insertTab(
            2,
            self.overview_tab,
            fibsem_icon("mdi:map-search-outline", color=GRAY_ICON_COLOR),
            "Overview",
        )
        self.tab_widget.setTabEnabled(self.tab_widget.indexOf(self.overview_tab), False)
        # Emits availability either way, which is what puts the reason on the tab.
        self._refresh_overview_microscope()

    def _apply_napari_overview_visibility(self) -> None:
        """Show or hide the old napari Minimap tab.

        `features.napari_overview_tab`, off by default and on its way out: the canvas
        Overview replaced this tab, and the flag is what brings the old one back for
        anyone who needs it for the one release before it goes. Both it and this method
        go with the tab before the full release (FIB-405, FIB-413).

        Visibility only, unlike the flag it replaced. The overview host tabs are built
        and dropped because their widgets subscribe to the microscope for their lifetime;
        this one owns a `napari.Viewer` that cannot be rebuilt safely mid-session, and it
        is the tab that is going away rather than the one being staged in, so hidden is
        the whole of what off has to mean.

        Read straight off `self._preferences` rather than through a module-level
        `FEATURE_*` global. FIB-609 removed five of those and kept the one whose caller
        is a widget constructor with no preferences to hand; this one is a method on the
        window that owns them, so a global would only be a second copy to keep in step.
        """
        container = getattr(self, "_minimap_tab_container", None)
        if container is None:
            return
        self.tab_widget.setTabVisible(
            self.tab_widget.indexOf(container),
            self._preferences.features.napari_overview_tab,
        )

    def _on_overview_availability(self, available: bool) -> None:
        """Enable the tab when either modality has something to drive, and say why not.

        The one thing about the tab that is not the tab's own business: it has no
        business reaching out to the tab bar it happens to sit in.

        The tab is never hidden. A greyed tab that explains itself is easier to live
        with than one that appears and vanishes depending on what the microscope turned
        out to be -- but that is only true *because* of the tooltip, so the two are set
        together here rather than in separate passes.

        Qt does show a tooltip on a disabled tab: the `QTabBar` stays enabled and only
        the tab within it is disabled, so the hover still lands. Worth stating because
        disabled *widgets* do swallow tooltips, which makes this look doubtful.
        """
        index = self.tab_widget.indexOf(self.overview_tab)
        self.tab_widget.setTabEnabled(index, available)
        _, reason = self.overview_tab.unavailable_summary()
        self.tab_widget.setTabToolTip(index, "" if available else reason)

    def _refresh_overview_microscope(self):
        """Build or drop both overview widgets to match the instrument.

        The widgets are built and destroyed rather than left behind hidden. Each
        subscribes to the microscope's stage signal for its lifetime, so one kept around
        would go on doing work on every poll for a page nobody can reach -- and would
        still be holding a psygnal reference to tear down later.

        Whether the tab can be *used* is a separate question, answered by
        `availability_changed` coming back through :meth:`_on_overview_availability`.
        This method does not touch the tab bar.
        """
        if getattr(self, "overview_tab", None) is None:
            return
        self.overview_tab.refresh_microscope()

    def _on_beam_overview_lamella_selected(self, lamella):
        """Sync the other lists when the rebuilt Overview tab's list changes.

        The same shape as `_on_minimap_lamella_selected`, minus the call back into the
        tab that raised it -- that list already shows the selection.
        """
        self.fm_overview_tab.set_selected(lamella)
        if getattr(self, "_syncing_selection", False) or lamella is None:
            return
        self._syncing_selection = True
        try:
            self.autolamella_ui.lamella_list.select(lamella.name)
            if hasattr(self, "lamella_card_container"):
                self.lamella_card_container.select_lamella(lamella.name)
            if hasattr(self, "minimap_widget"):
                self.minimap_widget.lamella_list.select(lamella.name)
        finally:
            self._syncing_selection = False

    def _on_fm_overview_lamella_selected(self, lamella):
        """Sync the other lists when the FM overview's list selection changes.

        The same shape as `_on_minimap_lamella_selected`, minus the call back into the
        FM tab: it raised this, and it has already highlighted what was clicked.
        """
        if getattr(self, "_syncing_selection", False) or lamella is None:
            return
        self._syncing_selection = True
        try:
            self.autolamella_ui.lamella_list.select(lamella.name)
            if hasattr(self, "lamella_card_container"):
                self.lamella_card_container.select_lamella(lamella.name)
        finally:
            self._syncing_selection = False

    def _refresh_overview_positions(self):
        """Re-mark **both** overview canvases, tolerating either tab being absent.

        A method on the window rather than connecting each tab's `refresh_positions`
        straight to the experiment's signal: a tab is rebuilt when the microscope
        changes, and a subscription holding the old one's bound method would be both
        stale and undisconnectable. This one is stable for the window's lifetime, which
        is what `_wire_position_events`' disconnect needs.

        Both, because for a long time it was only the fluorescence one -- so a lamella
        marked on the FIB/SEM overview appeared on the FM canvas, and one marked on the
        FM overview never appeared on the beam canvas at all. The beam tab papered over
        its own edits by re-marking inline and went stale for everything else: a lamella
        added from the FM tab, from the Microscope tab, or by a workflow, and any pose
        replaced in place (FIB-709). Neither tab subscribes to the experiment itself, so
        this is the only place that can see all of it.
        """
        for name in ("fm_overview_tab", "beam_overview_tab"):
            tab = getattr(self, name, None)
            if tab is not None:
                tab.refresh_positions()

    def _on_notification_service(
        self, message: str, notification_type: str, temporary: bool
    ) -> None:
        self.show_toast(message, notification_type, temporary=temporary)

    def closeEvent(self, event):
        """Clean up viewers on close."""
        # The editor holds edits for a moment before writing them (FIB-683); this is
        # the last chance to get the final one onto disk.
        if getattr(self, "lamella_widget", None) is not None:
            try:
                self.lamella_widget.flush_pending_save()
            except Exception as e:
                logging.warning(
                    f"Could not flush a pending experiment save on close: {e}"
                )
        # persist the FM working state (channels / camera transform / objective)
        if (
            self.autolamella_ui is not None
            and self.autolamella_ui.fm_control_widget is not None
        ):
            try:
                self.autolamella_ui.fm_control_widget.save_fm_configuration()
            except Exception as e:
                logging.warning(f"Could not save FM working state on close: {e}")
        try:
            notification_service._get_service().toast.disconnect(
                self._on_notification_service
            )
        except RuntimeError:
            pass
        for viewer in self.viewers:
            try:
                viewer.close()
            except Exception:
                pass
        super().closeEvent(event)
        # Force the event loop to exit even if another top-level window (e.g. a stray
        # matplotlib pyplot figure) would otherwise keep the app alive once this window
        # closes. quit() — not os._exit — so atexit / experiment autosave still run.
        app = QApplication.instance()
        if app is not None:
            app.quit()


def _start_update_check() -> None:
    """Ask PyPI about newer releases on a worker thread, once, at startup.

    Startup rather than on dialog open: the about dialog reads the cached result,
    so it stays instant and offline-safe by construction, and the latency lands
    where nothing is waiting on it. A no-op for source installs and when the user
    has opted out, so the common developer case makes no network call at all.
    """
    from fibsem import update_check
    from fibsem.ui.qt.threading import thread_worker

    if not update_check.is_enabled():
        return

    @thread_worker
    def _check() -> None:
        update_check.refresh()

    _check().start()


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the command-line flags accepted by the AutoLamella UI."""
    parser = argparse.ArgumentParser(
        prog="fibsem-autolamella-ui",
        description="Launch the AutoLamella user interface.",
    )
    parser.add_argument(
        "--quickstart",
        action="store_true",
        help="Connect to the microscope with the default configuration as soon as the "
        "window is up, instead of waiting for the Connection tab.",
    )
    parser.add_argument(
        "--quickload",
        action="store_true",
        help="Connect as --quickstart does, then reopen the most recent experiment. "
        "Implies --quickstart -- the experiment tabs are built at connection time.",
    )
    return parser.parse_args(argv)


def run_ui(argv: Optional[List[str]] = None):
    """Run the AutoLamella embedded example."""
    import faulthandler
    import signal

    args = _parse_args(argv)

    from fibsem.tools.bug_report import init_sentry

    # Ctrl+C: PyQt never runs Python's SIGINT handler while blocked in the C++ event
    # loop — and if the GUI thread wedges, no Python runs at all — so restore the OS
    # default disposition: SIGINT then terminates the process at the kernel level,
    # making the app always killable with Ctrl+C. (A hard stop, not a graceful one;
    # use Cmd+Q / File→Exit for a clean shutdown when the loop is responsive.)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    # Diagnostics: dump every thread's Python stack on a fatal signal, and on demand
    # via `kill -USR1 <pid>` — the fastest way to see *where* a frozen GUI thread is
    # stuck (it does not terminate the process, so you can dump repeatedly).
    faulthandler.enable()
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True)

    init_sentry()  # inert unless crash reporting is enabled in preferences
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    # Cyclic garbage must only ever be collected on this thread: worker-thread GC
    # finalizes Qt/vispy objects off the GUI thread and hard-crashes Windows GL
    # drivers (access violation in glDrawArrays). See fibsem/ui/qt/gc.py.
    gc_collector = install_main_thread_gc(parent=app)  # noqa: F841 — kept alive for app lifetime
    window = AutoLamellaSingleWindowUI()
    window.show()
    _start_update_check()
    if args.quickstart or args.quickload:
        # Once the event loop is running, not inline here: connecting blocks for
        # seconds against real hardware, and doing it before exec_() would hold up
        # the first paint — an empty window frame for the whole wait. A zero timer
        # fires as soon as the window system's queue, that first paint included,
        # has drained.
        QTimer.singleShot(
            0, lambda: window.autolamella_ui.quickstart(load_experiment=args.quickload)
        )
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_ui()
