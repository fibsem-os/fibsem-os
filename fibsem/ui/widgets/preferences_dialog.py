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
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from fibsem.config import UserPreferences
from fibsem.ui.widgets.custom_widgets import QDirectoryLineEdit, QFileLineEdit


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
        self._chk_toasts = QCheckBox()
        self._chk_border = QCheckBox()
        self._chk_dev_mode = QCheckBox()
        display_form.addRow("Sound notifications", self._chk_sound)
        display_form.addRow("Toast notifications", self._chk_toasts)
        display_form.addRow("Workflow border", self._chk_border)
        display_form.addRow("Development mode", self._chk_dev_mode)
        self._stack.addWidget(display_page)

        # --- Feature Flags ---
        features_page = QWidget()
        features_form = QFormLayout(features_page)
        self._chk_lamella_live = QCheckBox()
        self._chk_coincidence_milling = QCheckBox()
        self._chk_sample_holder = QCheckBox()
        features_form.addRow("Lamella position on live view", self._chk_lamella_live)
        features_form.addRow("Coincidence milling viewer", self._chk_coincidence_milling)
        features_form.addRow("Sample holder widget", self._chk_sample_holder)
        self._stack.addWidget(features_page)

        # --- Experiment Defaults ---
        experiment_page = QWidget()
        experiment_form = QFormLayout(experiment_page)
        self._dir_experiment = QDirectoryLineEdit()
        self._dir_protocol = QFileLineEdit()
        self._edit_exp_user = QLineEdit()
        self._edit_exp_project = QLineEdit()
        self._edit_exp_organisation = QLineEdit()
        experiment_form.addRow("Default experiment directory", self._dir_experiment)
        experiment_form.addRow("Default protocol path", self._dir_protocol)
        experiment_form.addRow("User", self._edit_exp_user)
        experiment_form.addRow("Project", self._edit_exp_project)
        experiment_form.addRow("Organisation", self._edit_exp_organisation)
        self._stack.addWidget(experiment_page)

        # --- Movement ---
        movement_page = QWidget()
        movement_form = QFormLayout(movement_page)
        self._chk_acquire_sem = QCheckBox()
        self._chk_acquire_fib = QCheckBox()
        movement_form.addRow("Acquire SEM after movement", self._chk_acquire_sem)
        movement_form.addRow("Acquire FIB after movement", self._chk_acquire_fib)
        self._stack.addWidget(movement_page)

        self._sidebar.currentRowChanged.connect(self._stack.setCurrentIndex)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _load_from_preferences(self, prefs: UserPreferences):
        """Populate widgets from a UserPreferences instance."""
        d = prefs.display
        self._chk_sound.setChecked(d.sound_enabled)
        self._chk_toasts.setChecked(d.toasts_enabled)
        self._chk_border.setChecked(d.border_enabled)
        self._chk_dev_mode.setChecked(d.dev_mode)

        f = prefs.features
        self._chk_lamella_live.setChecked(f.lamella_position_on_live_view)
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
                lamella_position_on_live_view=self._chk_lamella_live.isChecked(),
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
