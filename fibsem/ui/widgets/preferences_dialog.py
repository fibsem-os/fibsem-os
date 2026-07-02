"""Preferences dialog for AutoLamella user preferences."""

from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from fibsem.config import UserPreferences
from fibsem.ui.widgets.custom_widgets import QDirectoryLineEdit, QFileLineEdit

# ---------------------------------------------------------------------------
# Labels and tooltips — edit here to update both text and hover description
# ---------------------------------------------------------------------------

# Display
_LBL_SOUND         = "Enable Sound Notifications"
_TIP_SOUND         = "Play an audio alert when the workflow requires the user's attention."
_LBL_TOASTS        = "Enable Toast Notifications"
_TIP_TOASTS        = "Show brief pop-up messages in the corner of the screen for workflow events."
_LBL_BORDER        = "Enable Workflow Border"
_TIP_BORDER        = "Highlight the viewport border while an automated workflow is running."
_LBL_DEV_MODE      = "Enable Development Mode"
_TIP_DEV_MODE      = "Show advanced developer tools and diagnostic menus. Intended for developers only."

# Features
_LBL_COINCIDENCE   = "Enable Coincidence Milling Viewer"
_TIP_COINCIDENCE   = (
    "Enable the coincidence milling viewer for simultaneous FIB milling and FM acquisition. "
    "Restricted to ThermoFisher Arctis with the modified sample holder."
)
_LBL_SAMPLE_HOLDER = "Enable Sample Holder Widget"
_TIP_SAMPLE_HOLDER = "Show the sample holder navigation widget in the main interface."

# Experiment defaults
_LBL_EXP_DIR       = "Default Experiment Directory"
_TIP_EXP_DIR       = "Directory where new experiments will be saved. Pre-fills the directory field when creating a new experiment."
_LBL_EXP_PROTOCOL  = "Default Protocol File"
_TIP_EXP_PROTOCOL  = "Protocol file (.yaml) to load automatically when creating a new experiment."
_LBL_EXP_USER      = "Default User"
_TIP_EXP_USER      = "User name pre-filled in the metadata fields when creating a new experiment."
_LBL_EXP_PROJECT   = "Default Project"
_TIP_EXP_PROJECT   = "Project name pre-filled in the metadata fields when creating a new experiment."
_LBL_EXP_ORG       = "Default Organisation"
_TIP_EXP_ORG       = "Organisation name pre-filled in the metadata fields when creating a new experiment."

# Movement
_LBL_ACQ_SEM       = "Acquire SEM After Stage Movement"
_TIP_ACQ_SEM       = "Automatically acquire a new SEM image after each stage movement."
_LBL_ACQ_FIB       = "Acquire FIB After Stage Movement"
_TIP_ACQ_FIB       = "Automatically acquire a new FIB image after each stage movement."


class PreferencesDialog(QDialog):
    """Dialog for editing user preferences."""

    def __init__(self, preferences: UserPreferences, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(500)
        self._preferences = preferences
        self._setup_ui()
        self._load_from_preferences(preferences)
        self._chk_coincidence_milling.toggled.connect(self._on_coincidence_milling_toggled)

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Sidebar + stack
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)

        self._sidebar = QListWidget()
        self._sidebar.setFixedWidth(120)
        self._sidebar.addItems(["Display", "Features", "Experiment", "Movement"])
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
        self._chk_toasts = QCheckBox()
        self._chk_toasts.setToolTip(_TIP_TOASTS)
        self._chk_border = QCheckBox()
        self._chk_border.setToolTip(_TIP_BORDER)
        self._chk_dev_mode = QCheckBox()
        self._chk_dev_mode.setToolTip(_TIP_DEV_MODE)
        display_form.addRow(_LBL_SOUND, self._chk_sound)
        display_form.addRow(_LBL_TOASTS, self._chk_toasts)
        display_form.addRow(_LBL_BORDER, self._chk_border)
        display_form.addRow(_LBL_DEV_MODE, self._chk_dev_mode)
        self._stack.addWidget(display_page)

        # --- Feature Flags ---
        features_page = QWidget()
        features_form = QFormLayout(features_page)
        self._chk_coincidence_milling = QCheckBox()
        self._chk_coincidence_milling.setToolTip(_TIP_COINCIDENCE)
        self._chk_sample_holder = QCheckBox()
        self._chk_sample_holder.setToolTip(_TIP_SAMPLE_HOLDER)
        features_form.addRow(_LBL_COINCIDENCE, self._chk_coincidence_milling)
        features_form.addRow(_LBL_SAMPLE_HOLDER, self._chk_sample_holder)
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
        self._chk_toasts.setChecked(d.toasts_enabled)
        self._chk_border.setChecked(d.border_enabled)
        self._chk_dev_mode.setChecked(d.dev_mode)

        f = prefs.features
        self._chk_coincidence_milling.setChecked(f.coincidence_milling_enabled)
        self._chk_sample_holder.setChecked(f.sample_holder_widget)

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

    def get_preferences(self) -> UserPreferences:
        """Build a UserPreferences instance from current widget state."""
        from fibsem.config import (
            DisplayPreferences,
            ExperimentPreferences,
            FeatureFlags,
            MovementPreferences,
        )

        return UserPreferences(
            display=DisplayPreferences(
                sound_enabled=self._chk_sound.isChecked(),
                toasts_enabled=self._chk_toasts.isChecked(),
                border_enabled=self._chk_border.isChecked(),
                dev_mode=self._chk_dev_mode.isChecked(),
            ),
            features=FeatureFlags(
                coincidence_milling_enabled=self._chk_coincidence_milling.isChecked(),
                sample_holder_widget=self._chk_sample_holder.isChecked(),
            ),
            movement=MovementPreferences(
                acquire_sem_after_stage_movement=self._chk_acquire_sem.isChecked(),
                acquire_fib_after_stage_movement=self._chk_acquire_fib.isChecked(),
            ),
            experiment=ExperimentPreferences(
                default_experiment_directory=self._dir_experiment.text(),
                default_protocol_path=self._dir_protocol.text(),
                last_experiment_path=self._preferences.experiment.last_experiment_path,
                user=self._edit_exp_user.text(),
                project=self._edit_exp_project.text(),
                organisation=self._edit_exp_organisation.text(),
            ),
        )
