"""Preferences dialog for AutoLamella user preferences."""

from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from fibsem.config import UserPreferences
from fibsem.ui.widgets.custom_widgets import QDirectoryLineEdit, TitledPanel


class PreferencesDialog(QDialog):
    """Dialog for editing user preferences."""

    def __init__(self, preferences: UserPreferences, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(500)
        self._preferences = preferences
        self._setup_ui()
        self._load_from_preferences(preferences)

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)

        # --- Display ---
        display_content = QWidget()
        display_form = QFormLayout(display_content)
        self._chk_sound = QCheckBox()
        self._chk_toasts = QCheckBox()
        self._chk_border = QCheckBox()
        self._chk_timeline = QCheckBox()
        self._chk_dev_mode = QCheckBox()
        display_form.addRow("Sound notifications", self._chk_sound)
        display_form.addRow("Toast notifications", self._chk_toasts)
        display_form.addRow("Workflow border", self._chk_border)
        display_form.addRow("Workflow timeline", self._chk_timeline)
        display_form.addRow("Development mode", self._chk_dev_mode)
        content_layout.addWidget(TitledPanel("Display", content=display_content, collapsible=False))

        # --- Feature Flags ---
        features_content = QWidget()
        features_form = QFormLayout(features_content)
        self._chk_minimap = QCheckBox()
        self._chk_lamella_live = QCheckBox()
        self._chk_pose = QCheckBox()
        self._chk_grid_marker = QCheckBox()
        self._chk_right_click = QCheckBox()
        self._chk_poi = QCheckBox()
        features_form.addRow("Minimap plot widget", self._chk_minimap)
        features_form.addRow("Lamella position on live view", self._chk_lamella_live)
        features_form.addRow("Pose controls", self._chk_pose)
        features_form.addRow("Grid center marker", self._chk_grid_marker)
        features_form.addRow("Right-click context menu", self._chk_right_click)
        features_form.addRow("Point of interest display", self._chk_poi)
        content_layout.addWidget(TitledPanel("Feature Flags", content=features_content, collapsible=False))

        # --- Paths ---
        paths_content = QWidget()
        paths_form = QFormLayout(paths_content)
        self._dir_experiment = QDirectoryLineEdit()
        self._dir_protocol = QDirectoryLineEdit()
        paths_form.addRow("Default experiment directory", self._dir_experiment)
        paths_form.addRow("Default protocol path", self._dir_protocol)
        content_layout.addWidget(TitledPanel("Paths", content=paths_content, collapsible=False))

        # --- Movement ---
        movement_content = QWidget()
        movement_form = QFormLayout(movement_content)
        self._chk_acquire_sem = QCheckBox()
        self._chk_acquire_fib = QCheckBox()
        movement_form.addRow("Acquire SEM after movement", self._chk_acquire_sem)
        movement_form.addRow("Acquire FIB after movement", self._chk_acquire_fib)
        content_layout.addWidget(TitledPanel("Movement", content=movement_content, collapsible=False))

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

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
        self._chk_timeline.setChecked(d.workflow_timeline_enabled)
        self._chk_dev_mode.setChecked(d.dev_mode)

        f = prefs.features
        self._chk_minimap.setChecked(f.minimap_plot_widget)
        self._chk_lamella_live.setChecked(f.lamella_position_on_live_view)
        self._chk_pose.setChecked(f.pose_controls)
        self._chk_grid_marker.setChecked(f.display_grid_center_marker)
        self._chk_right_click.setChecked(f.right_click_context_menu)
        self._chk_poi.setChecked(f.display_point_of_interest)

        p = prefs.paths
        self._dir_experiment.setText(p.default_experiment_directory)
        self._dir_protocol.setText(p.default_protocol_path)

        m = prefs.movement
        self._chk_acquire_sem.setChecked(m.acquire_sem_after_stage_movement)
        self._chk_acquire_fib.setChecked(m.acquire_fib_after_stage_movement)

    def get_preferences(self) -> UserPreferences:
        """Build a UserPreferences instance from current widget state."""
        from fibsem.config import (
            DisplayPreferences,
            FeatureFlags,
            MovementPreferences,
            PathPreferences,
        )

        return UserPreferences(
            display=DisplayPreferences(
                sound_enabled=self._chk_sound.isChecked(),
                toasts_enabled=self._chk_toasts.isChecked(),
                border_enabled=self._chk_border.isChecked(),
                workflow_timeline_enabled=self._chk_timeline.isChecked(),
                dev_mode=self._chk_dev_mode.isChecked(),
            ),
            features=FeatureFlags(
                minimap_plot_widget=self._chk_minimap.isChecked(),
                lamella_position_on_live_view=self._chk_lamella_live.isChecked(),
                pose_controls=self._chk_pose.isChecked(),
                display_grid_center_marker=self._chk_grid_marker.isChecked(),
                right_click_context_menu=self._chk_right_click.isChecked(),
                display_point_of_interest=self._chk_poi.isChecked(),
            ),
            paths=PathPreferences(
                default_experiment_directory=self._dir_experiment.text(),
                last_experiment_path=self._preferences.paths.last_experiment_path,
                default_protocol_path=self._dir_protocol.text(),
            ),
            movement=MovementPreferences(
                acquire_sem_after_stage_movement=self._chk_acquire_sem.isChecked(),
                acquire_fib_after_stage_movement=self._chk_acquire_fib.isChecked(),
            ),
        )
