"""Preferences dialog for AutoLamella user preferences."""

from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from fibsem.config import (
    CARD_MODES,
    DisplayPreferences,
    UserPreferences,
    card_mode_label,
)
from fibsem.ui.widgets.custom_widgets import QDirectoryLineEdit, QFileLineEdit

# ---------------------------------------------------------------------------
# Labels and tooltips — edit here to update both text and hover description
# ---------------------------------------------------------------------------

# Display
_LBL_SOUND = "Enable Sound Notifications"
_TIP_SOUND = "Play an audio alert when the workflow requires the user's attention."
_LBL_BORDER = "Enable Workflow Border"
_TIP_BORDER = "Highlight the viewport border while an automated workflow is running."
_LBL_CARD_MODE = "Lamella Card Layout"
_TIP_CARD_MODE = (
    "How each lamella is drawn in the Lamella tab's strip: a large thumbnail, a "
    "compact row, or a single line with no thumbnail."
)
_LBL_DEV_MODE = "Enable Development Mode"
_TIP_DEV_MODE = (
    "Show advanced developer tools and diagnostic menus. Intended for developers only."
)

# Features
_LBL_COINCIDENCE = "Enable Coincidence Milling Viewer"
_TIP_COINCIDENCE = (
    "Enable the coincidence milling viewer for simultaneous FIB milling and FM acquisition. "
    "Restricted to ThermoFisher Arctis with the modified sample holder."
)
_LBL_BUG_REPORT = "Enable Bug Reporter"
_TIP_BUG_REPORT = (
    "Show the 'Report an Issue...' option in the Help menu, for reporting bugs and "
    "optionally submitting experiment data privately to the maintainers."
)
_LBL_CONNECTION_CHIP = "Enable Connection Chip"
_TIP_CONNECTION_CHIP = (
    "Show the connected instrument in the tab bar, beside the experiment, and add "
    "File > Connect to Microscope, which opens a dialog for connecting, "
    "reconnecting and disconnecting. The Connection tab still works and is still "
    "where connecting happens; this is the header half of replacing it."
)
_LBL_GRID_WORKFLOW = "Enable Grid Workflow"
_TIP_GRID_WORKFLOW = (
    "Show the Grids tab and the Workflow tab's Grids view: inventory the grids in "
    "the holder or autoloader, and acquire SEM, FIB and fluorescence overviews of "
    "each. In development; the Microscope tab's Sample view is available either way."
)
_LBL_PROPOSE_REVIEW = "Enable Propose and Review"
_TIP_PROPOSE_REVIEW = (
    "Let a task finish and leave its answer -- the point of interest, to start "
    "with -- as a proposal for you to confirm or reject later in the Review tab, "
    "instead of waiting at the beam for you to answer. In development."
)
_LBL_SCRIPTS = "Enable User Scripts"
_TIP_SCRIPTS = (
    "Show Tools > Scripts, for running your own .py files against the open "
    "experiment. A script has the same access to the microscope as the application "
    "itself and none of its safety checks — nothing validates what it does."
)
_LBL_AGENT_SERVER = "Enable Agent Server"
_LBL_WATCHDOG = "Hand questions to me after"
_TIP_WATCHDOG = (
    "If the agent leaves a question unanswered this long, it becomes yours — "
    "orange border, attention button, sound. Applies only to tasks the agent "
    "supervises."
)
_TIP_AGENT_SERVER = (
    "Host a local, token-protected API over this session when a microscope "
    "connects, so an AI agent (via the fibsem-mcp sidecar) can observe it. "
    "Read-only until you grant permissions for the session in Tools → Agent "
    "Server, which also shows the session token."
)

# Experiment defaults
_LBL_EXP_DIR = "Default Experiment Directory"
_TIP_EXP_DIR = "Directory where new experiments will be saved. Pre-fills the directory field when creating a new experiment."
_LBL_EXP_PROTOCOL = "Default Protocol File"
_TIP_EXP_PROTOCOL = (
    "Protocol file (.yaml) to load automatically when creating a new experiment."
)
_LBL_EXP_USER = "Default User"
_TIP_EXP_USER = (
    "User name pre-filled in the metadata fields when creating a new experiment."
)
_LBL_EXP_PROJECT = "Default Project"
_TIP_EXP_PROJECT = (
    "Project name pre-filled in the metadata fields when creating a new experiment."
)
_LBL_EXP_ORG = "Default Organisation"
_TIP_EXP_ORG = "Organisation name pre-filled in the metadata fields when creating a new experiment."

# Movement
_LBL_ACQ_SEM = "Acquire SEM After Stage Movement"
_TIP_ACQ_SEM = "Automatically acquire a new SEM image after each stage movement."
_LBL_ACQ_FIB = "Acquire FIB After Stage Movement"
_TIP_ACQ_FIB = "Automatically acquire a new FIB image after each stage movement."


class PreferencesDialog(QDialog):
    """Dialog for editing user preferences."""

    def __init__(self, preferences: UserPreferences, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(500)
        self._preferences = preferences
        self._setup_ui()
        self._load_from_preferences(preferences)
        # Connected after loading, so opening this dialog with a flag already on does
        # not fire its warning.
        self._chk_coincidence_milling.toggled.connect(
            self._on_coincidence_milling_toggled
        )
        self._chk_scripts.toggled.connect(self._on_scripts_toggled)

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Sidebar + stack
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)

        self._sidebar = QListWidget()
        self._sidebar.setFixedWidth(120)
        self._sidebar.addItems(
            ["Display", "Features", "Experiment", "Movement", "Agent"]
        )
        self._sidebar.setCurrentRow(0)

        self._stack = QStackedWidget()
        body_layout.addWidget(self._sidebar)
        body_layout.addWidget(self._stack)
        layout.addWidget(body)

        # --- Display ---
        display_page = QWidget()
        display_form = QFormLayout(display_page)
        self._chk_sound = QCheckBox()
        self._chk_sound.setToolTip(_TIP_SOUND)
        self._chk_border = QCheckBox()
        self._chk_border.setToolTip(_TIP_BORDER)
        self._chk_dev_mode = QCheckBox()
        self._chk_dev_mode.setToolTip(_TIP_DEV_MODE)
        # The one display preference that is not a yes/no. Its values are carried as
        # item data, not as the visible text, so the labels can be reworded without
        # changing what lands in the preferences file.
        self._combo_card_mode = QComboBox()
        self._combo_card_mode.setToolTip(_TIP_CARD_MODE)
        for mode in CARD_MODES:
            self._combo_card_mode.addItem(card_mode_label(mode), mode)
        display_form.addRow(_LBL_SOUND, self._chk_sound)
        display_form.addRow(_LBL_BORDER, self._chk_border)
        display_form.addRow(_LBL_CARD_MODE, self._combo_card_mode)
        display_form.addRow(_LBL_DEV_MODE, self._chk_dev_mode)
        self._stack.addWidget(display_page)

        # --- Feature Flags ---
        features_page = QWidget()
        features_form = QFormLayout(features_page)
        self._chk_coincidence_milling = QCheckBox()
        self._chk_coincidence_milling.setToolTip(_TIP_COINCIDENCE)
        self._chk_bug_report = QCheckBox()
        self._chk_bug_report.setToolTip(_TIP_BUG_REPORT)
        self._chk_scripts = QCheckBox()
        self._chk_scripts.setToolTip(_TIP_SCRIPTS)
        self._chk_connection_chip = QCheckBox()
        self._chk_connection_chip.setToolTip(_TIP_CONNECTION_CHIP)
        features_form.addRow(_LBL_COINCIDENCE, self._chk_coincidence_milling)
        features_form.addRow(_LBL_BUG_REPORT, self._chk_bug_report)
        features_form.addRow(_LBL_SCRIPTS, self._chk_scripts)
        features_form.addRow(_LBL_CONNECTION_CHIP, self._chk_connection_chip)
        self._chk_grid_workflow = QCheckBox()
        self._chk_grid_workflow.setToolTip(_TIP_GRID_WORKFLOW)
        features_form.addRow(_LBL_GRID_WORKFLOW, self._chk_grid_workflow)
        self._chk_propose_review = QCheckBox()
        self._chk_propose_review.setToolTip(_TIP_PROPOSE_REVIEW)
        features_form.addRow(_LBL_PROPOSE_REVIEW, self._chk_propose_review)
        self._stack.addWidget(features_page)

        # --- Experiment Defaults ---
        experiment_page = QWidget()
        experiment_form = QFormLayout(experiment_page)
        self._dir_experiment = QDirectoryLineEdit()
        self._dir_experiment.setToolTip(_TIP_EXP_DIR)
        self._dir_protocol = QFileLineEdit()
        self._dir_protocol.setToolTip(_TIP_EXP_PROTOCOL)
        self._edit_exp_user = QLineEdit()
        self._edit_exp_user.setToolTip(_TIP_EXP_USER)
        self._edit_exp_project = QLineEdit()
        self._edit_exp_project.setToolTip(_TIP_EXP_PROJECT)
        self._edit_exp_organisation = QLineEdit()
        self._edit_exp_organisation.setToolTip(_TIP_EXP_ORG)
        experiment_form.addRow(_LBL_EXP_DIR, self._dir_experiment)
        experiment_form.addRow(_LBL_EXP_PROTOCOL, self._dir_protocol)
        experiment_form.addRow(_LBL_EXP_USER, self._edit_exp_user)
        experiment_form.addRow(_LBL_EXP_PROJECT, self._edit_exp_project)
        experiment_form.addRow(_LBL_EXP_ORG, self._edit_exp_organisation)
        self._stack.addWidget(experiment_page)

        # --- Movement ---
        movement_page = QWidget()
        movement_form = QFormLayout(movement_page)
        self._chk_acquire_sem = QCheckBox()
        self._chk_acquire_sem.setToolTip(_TIP_ACQ_SEM)
        self._chk_acquire_fib = QCheckBox()
        self._chk_acquire_fib.setToolTip(_TIP_ACQ_FIB)
        movement_form.addRow(_LBL_ACQ_SEM, self._chk_acquire_sem)
        movement_form.addRow(_LBL_ACQ_FIB, self._chk_acquire_fib)
        self._stack.addWidget(movement_page)

        # --- Agent ---
        # Durable policy only. Scope ARMING is deliberately absent: arming is
        # consent, granted per session in Tools -> Agent Server, and must not
        # survive a restart via this file.
        agent_page = QWidget()
        agent_form = QFormLayout(agent_page)
        self._chk_agent_server = QCheckBox()
        self._chk_agent_server.setToolTip(_TIP_AGENT_SERVER)
        self._spin_watchdog = QSpinBox()
        self._spin_watchdog.setRange(1, 120)
        self._spin_watchdog.setSuffix(" min")
        self._spin_watchdog.setToolTip(_TIP_WATCHDOG)
        agent_intro = QLabel(
            "Let an AI agent watch this session — and, with permission you "
            "grant per session in Tools → Agent Server, act on it — over a "
            "local, token-protected connection."
        )
        agent_intro.setWordWrap(True)
        agent_intro.setStyleSheet("color: #868e93; font-size: 11px;")
        agent_form.addRow(agent_intro)
        agent_form.addRow(_LBL_AGENT_SERVER, self._chk_agent_server)
        agent_form.addRow(_LBL_WATCHDOG, self._spin_watchdog)
        self._stack.addWidget(agent_page)

        self._sidebar.currentRowChanged.connect(self._stack.setCurrentIndex)

        # Buttons
        btn_layout = QHBoxLayout()
        self._btn_restore = QPushButton("Restore Defaults")
        self._btn_restore.clicked.connect(self._on_restore_defaults)
        btn_layout.addWidget(self._btn_restore)
        btn_layout.addStretch()
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        btn_layout.addWidget(buttons)
        layout.addLayout(btn_layout)

    def _load_from_preferences(self, prefs: UserPreferences):
        """Populate widgets from a UserPreferences instance."""
        d = prefs.display
        self._chk_sound.setChecked(d.sound_enabled)
        self._chk_border.setChecked(d.border_enabled)
        self._chk_dev_mode.setChecked(d.dev_mode)
        card_index = self._combo_card_mode.findData(d.lamella_card_mode)
        if card_index == -1:
            # A value this build does not offer -- a preferences file from an older
            # build, or a hand-edited one. Fall back to the same default the card
            # widget falls back to, so the dialog shows the layout actually on screen.
            # Falling back to index 0 instead would put "Cozy" in the box over a strip
            # drawing Standard, and OK would then apply a layout the user never chose.
            card_index = self._combo_card_mode.findData(
                DisplayPreferences().lamella_card_mode
            )
        self._combo_card_mode.setCurrentIndex(card_index)

        f = prefs.features
        self._chk_coincidence_milling.setChecked(f.coincidence_milling_enabled)
        self._chk_bug_report.setChecked(f.bug_report_enabled)
        self._chk_scripts.setChecked(f.scripts_enabled)
        self._chk_agent_server.setChecked(f.agent_server_enabled)
        self._chk_connection_chip.setChecked(f.connection_chip)
        self._chk_grid_workflow.setChecked(f.grid_workflow)
        self._chk_propose_review.setChecked(f.proposer_reviewer_workflow_enabled)

        self._spin_watchdog.setValue(prefs.agent.watchdog_minutes)

        e = prefs.experiment
        self._dir_experiment.setText(e.default_experiment_directory)
        self._dir_protocol.setText(e.default_protocol_path)
        self._edit_exp_user.setText(e.user)
        self._edit_exp_project.setText(e.project)
        self._edit_exp_organisation.setText(e.organisation)

        m = prefs.movement
        self._chk_acquire_sem.setChecked(m.acquire_sem_after_stage_movement)
        self._chk_acquire_fib.setChecked(m.acquire_fib_after_stage_movement)

    def _on_restore_defaults(self):
        reply = QMessageBox.question(
            self,
            "Restore Defaults",
            "Reset all preferences to their default values?",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if reply == QMessageBox.Yes:
            self._load_from_preferences(UserPreferences())

    def _on_coincidence_milling_toggled(self, checked: bool):
        if not checked:
            return
        QMessageBox.warning(
            self,
            "Coincidence Milling — Restricted Use",
            "This mode can only be used on the ThermoFisher Arctis that has the modified "
            "sample holder. It also requires disabling the software restrictions related to "
            "running the fluorescence microscope while milling.",
        )

    def _on_scripts_toggled(self, checked: bool):
        """Same shape as the coincidence-milling warning: state the consequence once,
        on the way in, and never on the way out."""
        if not checked:
            return
        QMessageBox.warning(
            self,
            "User Scripts — No Safety Checks",
            "A script you run from Tools > Scripts has the same access to the "
            "microscope as the application itself, with none of its limits or "
            "interlocks, and nothing validates what it does before it runs.\n\n"
            "Scripts you did not write yourself should be read before they are run.",
        )

    def get_preferences(self) -> UserPreferences:
        """Build a UserPreferences instance from current widget state."""
        from fibsem.config import (
            AgentPreferences,
            DisplayPreferences,
            ExperimentPreferences,
            FeatureFlags,
            MovementPreferences,
        )

        return UserPreferences(
            display=DisplayPreferences(
                sound_enabled=self._chk_sound.isChecked(),
                border_enabled=self._chk_border.isChecked(),
                dev_mode=self._chk_dev_mode.isChecked(),
                lamella_card_mode=self._combo_card_mode.currentData(),
            ),
            features=FeatureFlags(
                coincidence_milling_enabled=self._chk_coincidence_milling.isChecked(),
                bug_report_enabled=self._chk_bug_report.isChecked(),
                scripts_enabled=self._chk_scripts.isChecked(),
                agent_server_enabled=self._chk_agent_server.isChecked(),
                connection_chip=self._chk_connection_chip.isChecked(),
                grid_workflow=self._chk_grid_workflow.isChecked(),
                proposer_reviewer_workflow_enabled=self._chk_propose_review.isChecked(),
            ),
            movement=MovementPreferences(
                acquire_sem_after_stage_movement=self._chk_acquire_sem.isChecked(),
                acquire_fib_after_stage_movement=self._chk_acquire_fib.isChecked(),
            ),
            agent=AgentPreferences(
                watchdog_minutes=self._spin_watchdog.value(),
            ),
            experiment=ExperimentPreferences(
                default_experiment_directory=self._dir_experiment.text(),
                default_protocol_path=self._dir_protocol.text(),
                last_experiment_path=self._preferences.experiment.last_experiment_path,
                recent_experiments=self._preferences.experiment.recent_experiments,
                user=self._edit_exp_user.text(),
                project=self._edit_exp_project.text(),
                organisation=self._edit_exp_organisation.text(),
            ),
            # Preserve sections not managed by this dialog.
            reporting=self._preferences.reporting,
        )
