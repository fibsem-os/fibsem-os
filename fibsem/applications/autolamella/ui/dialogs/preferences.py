"""Preferences dialog for AutoLamella user preferences."""

import copy

from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
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

from fibsem import update_check, vendor_libraries
from fibsem.config import (
    CARD_MODES,
    QtPreferences,
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
_LBL_TOAST_PROBLEMS = "Toasts for Problems Only"
_TIP_TOAST_PROBLEMS = (
    "Pop up only errors, warnings and the end of a run. Everything else still goes "
    "to the notification bell. Useful while an agent or a long run is working."
)
_LBL_CARD_MODE = "Lamella Card Layout"
_TIP_CARD_MODE = (
    "How each lamella is drawn in the Lamella tab's strip: a large thumbnail, a "
    "compact row, or a single line with no thumbnail."
)
_LBL_DEV_MODE = "Enable Development Mode"
_TIP_DEV_MODE = (
    "Show advanced developer tools and diagnostic menus. Intended for developers only."
)

_LBL_UPDATE_CHECK = "Check for Updates at Start-up"
_TIP_UPDATE_CHECK = (
    "Ask pypi.org for the latest fibsemOS release each time the application starts, "
    "and show a chip in the tab bar when a newer one is out. Takes effect at the next "
    "start. Off by default: instrument PCs are often on networks where an unannounced "
    "connection is a question for the site. The check only downloads the public list "
    "of releases. Help > About can check on request either way."
)
_TIP_UPDATE_CHECK_SOURCE = (
    "This is a source install, so the application never checks for releases: "
    "update it with git instead."
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

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Sidebar + stack
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)

        self._sidebar = QListWidget()
        self._sidebar.setFixedWidth(120)
        self._sidebar.addItems(["Display", "Experiment", "Movement"])
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
        self._chk_toast_problems = QCheckBox()
        self._chk_toast_problems.setToolTip(_TIP_TOAST_PROBLEMS)
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
        display_form.addRow(_LBL_TOAST_PROBLEMS, self._chk_toast_problems)
        display_form.addRow(_LBL_CARD_MODE, self._combo_card_mode)
        display_form.addRow(_LBL_DEV_MODE, self._chk_dev_mode)
        self._chk_update_check = QCheckBox()
        self._chk_update_check.setToolTip(_TIP_UPDATE_CHECK)
        if not update_check.is_supported():
            # Shown rather than hidden, so nobody goes looking for it; disabled because
            # a source install never checks, whatever this says. Its value is still
            # loaded and handed back, so the file keeps what was there.
            self._chk_update_check.setEnabled(False)
            self._chk_update_check.setToolTip(_TIP_UPDATE_CHECK_SOURCE)
        display_form.addRow(_LBL_UPDATE_CHECK, self._chk_update_check)
        self._stack.addWidget(display_page)

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

        # --- Libraries: where each driver's vendor library is on this PC ---
        self._library_edits = {}
        libraries = vendor_libraries.libraries()
        if libraries:
            libraries_page = QWidget()
            libraries_form = QFormLayout(libraries_page)
            for driver, library in libraries.items():
                edit = QDirectoryLineEdit()
                edit.setToolTip(
                    f"The folder that holds {', '.join(library.packages)}, for "
                    f"{driver}. Kept for this computer only."
                )
                found = vendor_libraries.resolve(driver, library)
                if found.folder and found.source != vendor_libraries.SOURCE_PREFERENCE:
                    edit.setPlaceholderText(f"{found.folder} ({found.source})")
                self._library_edits[driver] = edit
                libraries_form.addRow(
                    library.label[:1].upper() + library.label[1:], edit
                )
            self._sidebar.addItem("Libraries")
            self._stack.addWidget(libraries_page)

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
        q = prefs.qt
        self._chk_sound.setChecked(q.sound_enabled)
        self._chk_border.setChecked(q.border_enabled)
        self._chk_toast_problems.setChecked(q.toast_problems_only)
        self._chk_dev_mode.setChecked(q.dev_mode)
        card_index = self._combo_card_mode.findData(q.lamella_card_mode)
        if card_index == -1:
            # A value this build does not offer -- a preferences file from an older
            # build, or a hand-edited one. Fall back to the same default the card
            # widget falls back to, so the dialog shows the layout actually on screen.
            # Falling back to index 0 instead would put "Cozy" in the box over a strip
            # drawing Standard, and OK would then apply a layout the user never chose.
            card_index = self._combo_card_mode.findData(
                QtPreferences().lamella_card_mode
            )
        self._combo_card_mode.setCurrentIndex(card_index)
        self._chk_acquire_sem.setChecked(q.acquire_sem_after_stage_movement)
        self._chk_acquire_fib.setChecked(q.acquire_fib_after_stage_movement)

        self._chk_update_check.setChecked(prefs.install.update_check_enabled)
        folders = prefs.install.vendor_libraries or {}
        for driver, edit in self._library_edits.items():
            edit.setText(folders.get(driver, ""))

        e = prefs.shared
        self._dir_experiment.setText(e.default_experiment_directory)
        self._dir_protocol.setText(e.default_protocol_path)
        self._edit_exp_user.setText(e.user)
        self._edit_exp_project.setText(e.project)
        self._edit_exp_organisation.setText(e.organisation)

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

    def get_preferences(self) -> UserPreferences:
        """Build a UserPreferences instance from current widget state.

        Starts from a copy of the preferences the dialog was opened with and overwrites
        only the fields it has a widget for. Everything else -- the App Server, the
        info bar fields, the contact email -- is handed back as it came in, and the
        caller saves the result to disk. Building a fresh UserPreferences instead would
        reset any field not copied across by hand.

        A copy rather than self._preferences itself, which the caller still holds: a
        cancelled dialog must leave it untouched.
        """
        prefs = copy.deepcopy(self._preferences)

        q = prefs.qt
        q.sound_enabled = self._chk_sound.isChecked()
        q.border_enabled = self._chk_border.isChecked()
        q.toast_problems_only = self._chk_toast_problems.isChecked()
        q.dev_mode = self._chk_dev_mode.isChecked()
        q.lamella_card_mode = self._combo_card_mode.currentData()
        q.acquire_sem_after_stage_movement = self._chk_acquire_sem.isChecked()
        q.acquire_fib_after_stage_movement = self._chk_acquire_fib.isChecked()

        prefs.install.update_check_enabled = self._chk_update_check.isChecked()
        folders = dict(prefs.install.vendor_libraries or {})
        for driver, edit in self._library_edits.items():
            folders.pop(driver, None)
            if edit.text().strip():
                folders[driver] = edit.text().strip()
        prefs.install.vendor_libraries = folders

        e = prefs.shared
        e.default_experiment_directory = self._dir_experiment.text()
        e.default_protocol_path = self._dir_protocol.text()
        e.user = self._edit_exp_user.text()
        e.project = self._edit_exp_project.text()
        e.organisation = self._edit_exp_organisation.text()

        return prefs
