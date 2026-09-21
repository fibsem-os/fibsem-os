"""Dialog for reporting issues / submitting bug reports (optionally with data)."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import fibsem.config as fibsem_cfg
from fibsem.tools import bug_report
from fibsem.tools.bug_report import BugReportContent
from fibsem.ui import stylesheets
from fibsem.ui.utils import open_path_in_file_explorer
from fibsem.ui.widgets.custom_widgets import (
    TitledPanel,
    ValueComboBox,
)

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope

SEVERITY_OPTIONS = ["Low", "Normal", "High", "Crash"]


def _copy_to_clipboard(text: str) -> bool:
    """Put ``text`` on the system clipboard. False when there is no clipboard."""
    clipboard = QApplication.clipboard()
    if clipboard is None:
        return False
    clipboard.setText(text)
    return True


class BugReportDialog(QDialog):
    """Collect a bug report and submit it publicly (GitHub) or privately (email)."""

    def __init__(
        self,
        experiment_path: Optional[str] = None,
        microscope: Optional["FibsemMicroscope"] = None,
        traceback_text: str = "",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.experiment_path = experiment_path
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
            "Report a bug or issue. <b>Create Bundle</b> saves a scrubbed "
            "<code>.zip</code> of your experiment data to disk, which you can "
            "send to the maintainers from any machine. Use <b>Report on "
            "GitHub</b> instead for a public report with no data attached."
        )
        self.label_description.setWordWrap(True)
        self.label_description.setStyleSheet("font-style: italic; margin-bottom: 8px;")

        # --- Report details ---
        details = QWidget()
        form = QFormLayout(details)
        form.setContentsMargins(8, 8, 8, 8)

        self.lineEdit_title = QLineEdit()
        self.lineEdit_title.setPlaceholderText("Short summary of the issue")

        self.combo_severity = ValueComboBox()
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

        has_experiment = self.experiment_path is not None
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
        self.pushButton_bundle = QPushButton("Create Bundle")
        self.pushButton_bundle.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.pushButton_bundle.setAutoDefault(False)
        self.pushButton_cancel = QPushButton("Cancel")
        self.pushButton_cancel.setAutoDefault(False)

        self.button_box.addButton(self.pushButton_github, QDialogButtonBox.ActionRole)
        self.button_box.addButton(self.pushButton_bundle, QDialogButtonBox.AcceptRole)
        self.button_box.addButton(self.pushButton_cancel, QDialogButtonBox.RejectRole)
        self.pushButton_github.clicked.connect(self._on_report_github)
        self.pushButton_bundle.clicked.connect(self._on_create_bundle)
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
        size = bug_report.estimate_bundle_size(self.experiment_path, content)
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
            result = bug_report.open_github_issue(content)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open GitHub issue:\n{e}")
            return

        # A prefilled issue is a convenience, not the report. When the browser
        # never opened, or the body had to be trimmed to fit the URL, the full
        # text goes to the clipboard so nothing the user typed is lost.
        if not result.opened:
            _copy_to_clipboard(result.full_text)
            QMessageBox.warning(
                self,
                "Could not open a browser",
                "No browser opened on this machine. The report was copied to "
                "your clipboard — paste it into a new issue at\n\n"
                f"{bug_report.GITHUB_NEW_ISSUE_URL}",
            )
        elif result.truncated:
            _copy_to_clipboard(result.full_text)
            QMessageBox.information(
                self,
                "Report shortened",
                "The report was too long to prefill, so the issue shows a "
                "trimmed copy. The full text was copied to your clipboard — "
                "paste it over the issue body before submitting.",
            )
        self.accept()

    def _on_create_bundle(self):
        content = self._content()
        if not self._validate(content):
            return
        self._persist_email(content)
        try:
            zip_path = bug_report.build_bug_report_bundle(content, self.experiment_path)
        except Exception as e:
            logging.exception("Failed to create bug report bundle.")
            QMessageBox.critical(
                self, "Error", f"Could not create the bug report bundle:\n{e}"
            )
            return

        BundleCreatedDialog(zip_path, content, parent=self).exec_()
        self.accept()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):  # type: ignore
            event.ignore()
        else:
            super().keyPressEvent(event)


class BundleCreatedDialog(QDialog):
    """Confirms the bundle on disk and offers the ways of getting it to support.

    The bundle is the deliverable and it already exists by the time this opens.
    Email, the clipboard and the file explorer are conveniences layered on top,
    and each reports what it actually did rather than asserting success — an
    instrument PC often has no mail client, and telling the user an email is
    waiting when none opened is how a report gets silently dropped.
    """

    def __init__(
        self,
        zip_path: str,
        content: BugReportContent,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.zip_path = zip_path
        self.content = content

        self.setWindowTitle("Bundle created")
        self.setModal(True)
        self.setMinimumWidth(520)

        self._create_widgets()
        self._setup_layout()

    def _create_widgets(self):
        self.label_title = QLabel("Bundle created")
        self.label_title.setStyleSheet("font-weight: bold;")

        self.label_instructions = QLabel(
            f"Email it to <b>{bug_report.SUPPORT_EMAIL}</b>, or copy it to a USB "
            "stick or network share and send it from another machine. This "
            "computer does not need email or internet access."
        )
        self.label_instructions.setWordWrap(True)
        self.label_instructions.setStyleSheet(stylesheets.LABEL_INSTRUCTIONS_STYLE)

        self.lineEdit_path = QLineEdit(self.zip_path)
        self.lineEdit_path.setReadOnly(True)
        self.lineEdit_path.setCursorPosition(0)

        self.pushButton_copy_path = QPushButton("Copy Path")
        self.pushButton_copy_path.setAutoDefault(False)
        self.pushButton_copy_path.clicked.connect(self._on_copy_path)

        self.pushButton_open_folder = QPushButton("Open Folder")
        self.pushButton_open_folder.setAutoDefault(False)
        self.pushButton_open_folder.clicked.connect(self._on_open_folder)

        self.pushButton_copy_report = QPushButton("Copy Report Text")
        self.pushButton_copy_report.setAutoDefault(False)
        self.pushButton_copy_report.clicked.connect(self._on_copy_report)

        self.pushButton_email = QPushButton("Open Email...")
        self.pushButton_email.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_email.setAutoDefault(False)
        self.pushButton_email.clicked.connect(self._on_open_email)

        self.label_status = QLabel("")
        self.label_status.setWordWrap(True)
        self.label_status.setStyleSheet(stylesheets.LABEL_INSTRUCTIONS_STYLE)
        # Reserve the room up front. A word-wrapped QLabel does not reliably
        # grow its dialog once the dialog has been laid out, and the message
        # that matters most -- no mail client on this PC -- is the longest one,
        # so without this it is the one that gets clipped.
        self.label_status.setMinimumHeight(
            3 * self.label_status.fontMetrics().lineSpacing()
        )

        self.pushButton_close = QPushButton("Close")
        self.pushButton_close.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.pushButton_close.setAutoDefault(False)
        self.pushButton_close.clicked.connect(self.accept)

    def _setup_layout(self):
        path_row = QHBoxLayout()
        path_row.addWidget(self.lineEdit_path)
        path_row.addWidget(self.pushButton_copy_path)

        action_row = QHBoxLayout()
        action_row.addWidget(self.pushButton_open_folder)
        action_row.addWidget(self.pushButton_copy_report)
        action_row.addWidget(self.pushButton_email)
        action_row.addStretch()

        layout = QVBoxLayout()
        layout.addWidget(self.label_title)
        layout.addWidget(self.label_instructions)
        layout.addLayout(path_row)
        layout.addLayout(action_row)
        layout.addWidget(self.label_status)
        layout.addStretch()

        close_row = QHBoxLayout()
        close_row.addStretch()
        close_row.addWidget(self.pushButton_close)
        layout.addLayout(close_row)

        self.setLayout(layout)

    def _set_status(self, text: str, warn: bool = False):
        style = (
            "color: orange; font-style: italic;"
            if warn
            else stylesheets.LABEL_INSTRUCTIONS_STYLE
        )
        self.label_status.setStyleSheet(style)
        self.label_status.setText(text)

    def _on_copy_path(self):
        if _copy_to_clipboard(self.zip_path):
            self._set_status("Path copied to the clipboard.")
        else:
            self._set_status("Could not access the clipboard.", warn=True)

    def _on_copy_report(self):
        text = bug_report.render_report_text(self.content)
        text += f"\n\nData bundle: {self.zip_path}"
        if _copy_to_clipboard(text):
            self._set_status(
                "Report text copied to the clipboard — paste it into an email "
                "or a message, and attach the bundle."
            )
        else:
            self._set_status("Could not access the clipboard.", warn=True)

    def _on_open_folder(self):
        if open_path_in_file_explorer(os.path.dirname(self.zip_path)):
            self._set_status("Opened the bundle's folder.")
        else:
            self._set_status(
                "Could not open a file explorer. Use the path above.", warn=True
            )

    def _on_open_email(self):
        try:
            result = bug_report.compose_support_email(self.content, self.zip_path)
        except Exception as e:
            logging.exception("Failed to compose the support email.")
            self._set_status(f"Could not open a mail client: {e}", warn=True)
            return

        if result.opened:
            self._set_status("Mail client opened — attach the bundle before sending.")
        else:
            _copy_to_clipboard(result.full_text)
            self._set_status(
                "No mail client opened on this computer. The report text was "
                f"copied to your clipboard — email it to "
                f"{bug_report.SUPPORT_EMAIL} from another machine, with the "
                "bundle attached.",
                warn=True,
            )


def open_bug_report_dialog(
    experiment_path: Optional[str] = None,
    microscope: Optional["FibsemMicroscope"] = None,
    traceback_text: str = "",
    parent: Optional[QWidget] = None,
) -> None:
    """Convenience helper to construct and show the bug report dialog."""
    dialog = BugReportDialog(
        experiment_path=experiment_path,
        microscope=microscope,
        traceback_text=traceback_text,
        parent=parent,
    )
    dialog.exec_()
