"""Dialog for reporting issues / submitting bug reports (optionally with data)."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import fibsem.config as fibsem_cfg
from fibsem.applications.autolamella.tools import bug_report
from fibsem.applications.autolamella.tools.bug_report import BugReportContent
from fibsem.ui import stylesheets
from fibsem.ui.utils import open_path_in_file_explorer
from fibsem.ui.widgets.custom_widgets import TitledPanel

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import Experiment
    from fibsem.microscope import FibsemMicroscope

SEVERITY_OPTIONS = ["Low", "Normal", "High", "Crash"]


class BugReportDialog(QDialog):
    """Collect a bug report and submit it publicly (GitHub) or privately (email)."""

    def __init__(
        self,
        experiment: Optional["Experiment"] = None,
        microscope: Optional["FibsemMicroscope"] = None,
        traceback_text: str = "",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.experiment = experiment
        self.microscope = microscope
        self._system_context = bug_report.collect_system_context(microscope)

        self.setWindowTitle("Report an Issue")
        self.setModal(True)
        self.setMinimumWidth(520)

        self._create_widgets(traceback_text)
        self._setup_layout()
        self._update_preview()

    def _create_widgets(self, traceback_text: str):
        self.label_description = QLabel(
            "Report a bug or issue. Use <b>Report on GitHub</b> for a public "
            "report (no data), or <b>Create Bundle &amp; Email</b> to send "
            "experiment data privately to the maintainers."
        )
        self.label_description.setWordWrap(True)
        self.label_description.setStyleSheet("font-style: italic; margin-bottom: 8px;")

        # --- Report details ---
        details = QWidget()
        form = QFormLayout(details)
        form.setContentsMargins(8, 8, 8, 8)

        self.lineEdit_title = QLineEdit()
        self.lineEdit_title.setPlaceholderText("Short summary of the issue")

        self.combo_severity = QComboBox()
        self.combo_severity.addItems(SEVERITY_OPTIONS)
        self.combo_severity.setCurrentText("Crash" if traceback_text else "Normal")

        self.lineEdit_email = QLineEdit()
        self.lineEdit_email.setPlaceholderText("you@example.com (optional)")
        prefs = fibsem_cfg.load_user_preferences()
        self.lineEdit_email.setText(getattr(prefs.reporting, "contact_email", ""))

        self.textEdit_description = QTextEdit()
        self.textEdit_description.setPlaceholderText("What happened?")
        if traceback_text:
            self.textEdit_description.setPlainText(
                f"An unexpected error occurred:\n\n{traceback_text}"
            )
        self.textEdit_description.setMinimumHeight(90)

        self.textEdit_steps = QTextEdit()
        self.textEdit_steps.setPlaceholderText("1. ...\n2. ...")
        self.textEdit_steps.setMinimumHeight(60)

        form.addRow("Title", self.lineEdit_title)
        form.addRow("Severity", self.combo_severity)
        form.addRow("Contact email", self.lineEdit_email)
        form.addRow("Description", self.textEdit_description)
        form.addRow("Steps to reproduce", self.textEdit_steps)
        self.details_panel = TitledPanel("Details", content=details, collapsible=False)

        # --- Data to include (private bundle only) ---
        data = QWidget()
        data_layout = QVBoxLayout(data)
        data_layout.setContentsMargins(8, 8, 8, 8)

        self.checkbox_logfile = QCheckBox("Log file (logfile.log)")
        self.checkbox_experiment = QCheckBox("Experiment file (experiment.yaml)")
        self.checkbox_protocol = QCheckBox("Protocol file (protocol.yaml)")
        self.checkbox_screenshots = QCheckBox("Task screenshots (.png)")
        self.checkbox_images = QCheckBox("Image data (.tiff) — may be large")
        self.checkbox_logfile.setChecked(True)
        self.checkbox_experiment.setChecked(True)
        self.checkbox_protocol.setChecked(True)

        has_experiment = self.experiment is not None
        for cb in (
            self.checkbox_logfile,
            self.checkbox_experiment,
            self.checkbox_protocol,
            self.checkbox_screenshots,
            self.checkbox_images,
        ):
            cb.setEnabled(has_experiment)
            cb.stateChanged.connect(lambda _: self._update_preview())
            data_layout.addWidget(cb)

        if not has_experiment:
            no_exp = QLabel("No experiment loaded — only text will be sent.")
            no_exp.setStyleSheet("color: orange; font-style: italic;")
            data_layout.addWidget(no_exp)

        self.data_panel = TitledPanel(
            "Data to include", content=data, collapsible=False
        )

        # --- Preview + privacy note ---
        self.label_preview = QLabel()
        self.label_preview.setWordWrap(True)
        self.label_preview.setStyleSheet("color: gray; font-style: italic;")

        # --- Buttons ---
        self.button_box = QDialogButtonBox(self)
        self.pushButton_github = QPushButton("Report on GitHub")
        self.pushButton_github.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_github.setAutoDefault(False)
        self.pushButton_email = QPushButton("Create Bundle && Email")
        self.pushButton_email.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.pushButton_email.setAutoDefault(False)
        self.pushButton_cancel = QPushButton("Cancel")
        self.pushButton_cancel.setAutoDefault(False)

        self.button_box.addButton(
            self.pushButton_github, QDialogButtonBox.ActionRole
        )
        self.button_box.addButton(self.pushButton_email, QDialogButtonBox.AcceptRole)
        self.button_box.addButton(self.pushButton_cancel, QDialogButtonBox.RejectRole)
        self.pushButton_github.clicked.connect(self._on_report_github)
        self.pushButton_email.clicked.connect(self._on_create_bundle)
        self.pushButton_cancel.clicked.connect(self.reject)

    def _setup_layout(self):
        layout = QVBoxLayout()
        layout.addWidget(self.label_description)
        layout.addWidget(self.details_panel)
        layout.addWidget(self.data_panel)
        layout.addWidget(self.label_preview)
        layout.addStretch()
        layout.addWidget(self.button_box)
        self.setLayout(layout)

    def _content(self) -> BugReportContent:
        return BugReportContent(
            title=self.lineEdit_title.text().strip(),
            description=self.textEdit_description.toPlainText().strip(),
            steps=self.textEdit_steps.toPlainText().strip(),
            severity=self.combo_severity.currentText(),
            contact_email=self.lineEdit_email.text().strip(),
            include_logfile=self.checkbox_logfile.isChecked(),
            include_experiment_yaml=self.checkbox_experiment.isChecked(),
            include_protocol=self.checkbox_protocol.isChecked(),
            include_screenshots=self.checkbox_screenshots.isChecked(),
            include_images=self.checkbox_images.isChecked(),
            system_context=self._system_context,
        )

    def _update_preview(self):
        content = self._content()
        size = bug_report.estimate_bundle_size(self.experiment, content)
        env = ", ".join(f"{k}={v}" for k, v in self._system_context.items())
        size_str = f"{size / 1e6:.1f} MB" if size else "0 MB"
        self.label_preview.setText(
            f"Bundle size (approx): {size_str}. "
            f"Text is scrubbed of your home directory / username before sending. "
            f"Included environment info: {env}"
        )

    def _persist_email(self, content: BugReportContent):
        """Remember the contact email for next time."""
        try:
            prefs = fibsem_cfg.load_user_preferences()
            prefs.reporting.contact_email = content.contact_email
            fibsem_cfg.save_user_preferences(prefs)
        except Exception as e:
            logging.debug("Could not persist contact email: %s", e)

    def _validate(self, content: BugReportContent) -> bool:
        if not content.title and not content.description:
            QMessageBox.warning(
                self, "Missing information", "Please add a title or description."
            )
            return False
        return True

    def _on_report_github(self):
        content = self._content()
        if not self._validate(content):
            return
        try:
            bug_report.open_github_issue(content)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open GitHub issue:\n{e}")
            return
        self.accept()

    def _on_create_bundle(self):
        content = self._content()
        if not self._validate(content):
            return
        self._persist_email(content)
        try:
            zip_path = bug_report.build_bug_report_bundle(content, self.experiment)
            bug_report.compose_support_email(content, zip_path)
        except Exception as e:
            logging.exception("Failed to create bug report bundle.")
            QMessageBox.critical(
                self, "Error", f"Could not create the bug report bundle:\n{e}"
            )
            return
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Bug report created")
        msg.setText("Bug report created")
        msg.setInformativeText(
            f"A data bundle was saved to:\n\n{zip_path}\n\n"
            f"An email to {bug_report.SUPPORT_EMAIL} has been opened — "
            f"please attach this file before sending."
        )
        open_button = msg.addButton("Open Directory", QMessageBox.ActionRole)
        open_button.setMinimumWidth(
            open_button.fontMetrics().boundingRect(open_button.text()).width() + 40
        )
        msg.addButton(QMessageBox.Ok)
        msg.exec_()
        if msg.clickedButton() is open_button:
            open_path_in_file_explorer(os.path.dirname(zip_path))
        self.accept()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):  # type: ignore
            event.ignore()
        else:
            super().keyPressEvent(event)


def open_bug_report_dialog(
    experiment: Optional["Experiment"] = None,
    microscope: Optional["FibsemMicroscope"] = None,
    traceback_text: str = "",
    parent: Optional[QWidget] = None,
) -> None:
    """Convenience helper to construct and show the bug report dialog."""
    dialog = BugReportDialog(
        experiment=experiment,
        microscope=microscope,
        traceback_text=traceback_text,
        parent=parent,
    )
    dialog.exec_()
