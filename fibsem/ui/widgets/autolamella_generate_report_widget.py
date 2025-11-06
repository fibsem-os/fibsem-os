"""Widget for generating AutoLamella experiment reports with configurable sections."""

import logging
import os
from pathlib import Path
from typing import Optional

from PyQt5 import QtWidgets, QtCore

from fibsem.applications.autolamella.structures import Experiment
from fibsem.applications.autolamella.tools.reporting import generate_report2
from fibsem.ui.stylesheets import (
    BLUE_PUSHBUTTON_STYLE,
    GREEN_PUSHBUTTON_STYLE,
    RED_PUSHBUTTON_STYLE,
)


class ReportGeneratorWorker(QtCore.QThread):
    """Worker thread for generating reports asynchronously."""

    finished = QtCore.pyqtSignal(str)  # Emits the output path on success
    error = QtCore.pyqtSignal(str)  # Emits error message on failure

    def __init__(self, experiment: Experiment, output_path: str, sections: dict):
        super().__init__()
        self.experiment = experiment
        self.output_path = output_path
        self.sections = sections

    def run(self):
        """Generate the report in a background thread."""
        try:
            generate_report2(
                experiment=self.experiment,
                output_filename=self.output_path,
                sections=self.sections
            )
            self.finished.emit(self.output_path)
        except Exception as e:
            logging.error(f"Failed to generate report: {e}", exc_info=True)
            self.error.emit(str(e))


class AutoLamellaGenerateReportWidget(QtWidgets.QDialog):
    """Dialog for generating AutoLamella experiment reports.

    Allows users to:
    - Choose output location and filename for the PDF report
    - Configure which sections to include in the report

    Returns:
        str: Path to the generated PDF report, or None if cancelled
    """

    def __init__(
        self,
        experiment: Experiment,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)

        self.experiment: Optional[Experiment] = None
        self.report_path: Optional[str] = None
        self.worker: Optional[ReportGeneratorWorker] = None
        self.progress_dialog: Optional[QtWidgets.QProgressDialog] = None

        self.setMinimumWidth(600)

        self._setup_ui()
        self._connect_signals()

        self._update_window_title(None)

        if experiment is None:
            raise ValueError("AutoLamellaGenerateReportWidget requires an experiment instance.")

        self.set_experiment(experiment)

    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QtWidgets.QVBoxLayout()

        # Report Output
        output_group = QtWidgets.QGroupBox("Report Output")
        output_layout = QtWidgets.QVBoxLayout()

        # Select Output button at top
        output_button_layout = QtWidgets.QHBoxLayout()
        output_button_layout.addStretch()
        self.btn_select_output = QtWidgets.QPushButton("Select Output File")
        self.btn_select_output.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        output_button_layout.addWidget(self.btn_select_output)
        output_layout.addLayout(output_button_layout)

        # Output form fields
        output_form_layout = QtWidgets.QFormLayout()

        # Output File Path
        self.lineEdit_output_path = QtWidgets.QLineEdit()
        self.lineEdit_output_path.setReadOnly(True)
        self.lineEdit_output_path.setPlaceholderText("No output file selected (will default to experiment directory)")
        self.lineEdit_output_path.setCursorPosition(0)

        output_form_layout.addRow("Output File", self.lineEdit_output_path)

        output_layout.addLayout(output_form_layout)

        output_group.setLayout(output_layout)
        main_layout.addWidget(output_group)

        # Report Sections
        sections_group = QtWidgets.QGroupBox("Report Sections")
        sections_layout = QtWidgets.QVBoxLayout()

        # Checkboxes for sections
        self.checkbox_overview = QtWidgets.QCheckBox("Overview Image with Positions")
        self.checkbox_overview.setChecked(True)
        self.checkbox_overview.setToolTip("Include overview image showing all lamella positions")

        self.checkbox_task_history = QtWidgets.QCheckBox("Task History Table")
        self.checkbox_task_history.setChecked(True)
        self.checkbox_task_history.setToolTip("Include task history and timing information")

        self.checkbox_detection = QtWidgets.QCheckBox("Detection Data")
        self.checkbox_detection.setChecked(True)
        self.checkbox_detection.setToolTip("Include detection results and statistics")

        self.checkbox_lamella_workflow = QtWidgets.QCheckBox("Per-Lamella Workflow Tables")
        self.checkbox_lamella_workflow.setChecked(True)
        self.checkbox_lamella_workflow.setToolTip("Include workflow duration tables for each lamella")

        self.checkbox_lamella_images = QtWidgets.QCheckBox("Per-Lamella Workflow Images")
        self.checkbox_lamella_images.setChecked(True)
        self.checkbox_lamella_images.setToolTip("Include SEM/FIB images for each lamella workflow stage")

        self.checkbox_lamella_milling = QtWidgets.QCheckBox("Per-Lamella Milling Data and Patterns")
        self.checkbox_lamella_milling.setChecked(True)
        self.checkbox_lamella_milling.setToolTip("Include milling patterns and data for each lamella")

        sections_layout.addWidget(self.checkbox_overview)
        sections_layout.addWidget(self.checkbox_task_history)
        sections_layout.addWidget(self.checkbox_detection)
        sections_layout.addWidget(self.checkbox_lamella_workflow)
        sections_layout.addWidget(self.checkbox_lamella_images)
        sections_layout.addWidget(self.checkbox_lamella_milling)

        # Select All / Deselect All buttons
        sections_button_layout = QtWidgets.QHBoxLayout()
        sections_button_layout.addStretch()

        self.btn_select_all = QtWidgets.QPushButton("Select All")
        self.btn_select_all.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        sections_button_layout.addWidget(self.btn_select_all)

        self.btn_deselect_all = QtWidgets.QPushButton("Deselect All")
        self.btn_deselect_all.setStyleSheet(BLUE_PUSHBUTTON_STYLE)
        sections_button_layout.addWidget(self.btn_deselect_all)

        sections_layout.addLayout(sections_button_layout)

        sections_group.setLayout(sections_layout)
        main_layout.addWidget(sections_group)

        # Dialog buttons (Generate/Cancel)
        button_box = QtWidgets.QDialogButtonBox()

        self.btn_generate = QtWidgets.QPushButton("Generate Report")
        self.btn_generate.setStyleSheet(GREEN_PUSHBUTTON_STYLE)
        self.btn_generate.setDefault(True)
        self.btn_generate.setEnabled(False)  # Disabled until an experiment is attached

        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setStyleSheet(RED_PUSHBUTTON_STYLE)

        button_box.addButton(self.btn_generate, QtWidgets.QDialogButtonBox.AcceptRole)
        button_box.addButton(self.btn_cancel, QtWidgets.QDialogButtonBox.RejectRole)

        main_layout.addWidget(button_box)

        self.setLayout(main_layout)

    def _connect_signals(self):
        """Connect UI signals."""
        self.btn_select_output.clicked.connect(self._select_output_file)
        self.btn_select_all.clicked.connect(self._select_all_sections)
        self.btn_deselect_all.clicked.connect(self._deselect_all_sections)
        self.btn_generate.clicked.connect(self._on_generate_clicked)
        self.btn_cancel.clicked.connect(self.reject)

    def set_experiment(
        self,
        experiment: Optional[Experiment],
        experiment_file_path: Optional[str] = None,
    ):
        """Set the experiment directly and update the UI to match it."""
        self.experiment = experiment
        if experiment is None:
            self.lineEdit_output_path.setText("")
            self.btn_generate.setEnabled(False)
            self._update_window_title(None)
            return

        resolved_file_path = experiment_file_path or self._resolve_experiment_file_path(experiment)

        experiment_dir = self._get_experiment_directory(experiment, resolved_file_path)
        if experiment_dir:
            default_filename = f"{experiment.name}_report.pdf"
            default_output = os.path.join(experiment_dir, default_filename)
            self.lineEdit_output_path.setText(default_output)
            self.lineEdit_output_path.setCursorPosition(0)
        else:
            self.lineEdit_output_path.setText("")

        self.btn_generate.setEnabled(True)
        self._update_window_title(experiment.name)

    def _update_window_title(self, experiment_name: Optional[str]):
        """Refresh the dialog title to include the experiment name."""
        base_title = "Generate Report"
        if experiment_name:
            self.setWindowTitle(f"{base_title} for {experiment_name}")
        else:
            self.setWindowTitle(base_title)

    def _resolve_experiment_file_path(self, experiment: Experiment) -> Optional[str]:
        """Try to resolve the on-disk experiment.yaml path for display."""
        experiment_dir = str(getattr(experiment, "path", "") or "")
        if not experiment_dir:
            return None

        candidate = os.path.join(experiment_dir, "experiment.yaml")
        if os.path.exists(candidate):
            return candidate

        return experiment_dir

    @staticmethod
    def _get_experiment_directory(
        experiment: Experiment,
        experiment_file_path: Optional[str],
    ) -> Optional[str]:
        """Determine the directory to use for default outputs."""
        if experiment_file_path:
            return os.path.dirname(experiment_file_path)
        experiment_dir = getattr(experiment, "path", None)
        if experiment_dir:
            return str(experiment_dir)
        return None

    def _select_output_file(self):
        """Open dialog to select output file path."""
        # Start at the current output path if available, otherwise experiment directory
        start_path = str(Path.home())

        if self.lineEdit_output_path.text():
            start_path = self.lineEdit_output_path.text()
        elif self.experiment:
            default_filename = f"{self.experiment.name}_report.pdf"
            start_path = os.path.join(str(self.experiment.path), default_filename)

        # Open save file dialog
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Report As",
            start_path,
            "PDF Files (*.pdf);;All Files (*)"
        )

        if file_path and file_path != "":
            # Add .pdf extension if not present
            if not file_path.lower().endswith('.pdf'):
                file_path += '.pdf'

            self.lineEdit_output_path.setText(file_path)
            self.lineEdit_output_path.setCursorPosition(0)

    def _select_all_sections(self):
        """Select all report sections."""
        self.checkbox_overview.setChecked(True)
        self.checkbox_task_history.setChecked(True)
        self.checkbox_detection.setChecked(True)
        self.checkbox_lamella_workflow.setChecked(True)
        self.checkbox_lamella_images.setChecked(True)
        self.checkbox_lamella_milling.setChecked(True)

    def _deselect_all_sections(self):
        """Deselect all report sections."""
        self.checkbox_overview.setChecked(False)
        self.checkbox_task_history.setChecked(False)
        self.checkbox_detection.setChecked(False)
        self.checkbox_lamella_workflow.setChecked(False)
        self.checkbox_lamella_images.setChecked(False)
        self.checkbox_lamella_milling.setChecked(False)

    def _get_sections_config(self) -> dict:
        """Get the sections configuration from checkboxes."""
        return {
            "overview": self.checkbox_overview.isChecked(),
            "task_history": self.checkbox_task_history.isChecked(),
            "detection": self.checkbox_detection.isChecked(),
            "lamella_workflow": self.checkbox_lamella_workflow.isChecked(),
            "lamella_images": self.checkbox_lamella_images.isChecked(),
            "lamella_milling": self.checkbox_lamella_milling.isChecked(),
        }

    def _on_generate_clicked(self):
        """Handle Generate button click - validate inputs and generate report."""
        # Validate experiment
        if self.experiment is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No Experiment",
                "No experiment is available to generate a report."
            )
            return

        # Get output path
        output_path = self.lineEdit_output_path.text().strip()
        if not output_path:
            # Default to experiment directory with experiment name
            default_filename = f"{self.experiment.name}_report.pdf"
            output_path = os.path.join(str(self.experiment.path), default_filename)
            self.lineEdit_output_path.setText(output_path)

        # Add .pdf extension if not present
        if not output_path.lower().endswith('.pdf'):
            output_path += '.pdf'
            self.lineEdit_output_path.setText(output_path)

        # Validate output directory exists
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Directory",
                f"Output directory does not exist: {output_dir}"
            )
            return

        # Check if file exists
        if os.path.exists(output_path):
            filename = os.path.basename(output_path)
            reply = QtWidgets.QMessageBox.question(
                self,
                "File Exists",
                f"A file named '{filename}' already exists.\n\nDo you want to overwrite it?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No
            )
            if reply == QtWidgets.QMessageBox.No:
                return

        # Check if at least one section is selected
        sections = self._get_sections_config()
        if not any(sections.values()):
            QtWidgets.QMessageBox.warning(
                self,
                "No Sections Selected",
                "Please select at least one section to include in the report."
            )
            return

        # Start generating the report in a worker thread
        self._start_report_generation(output_path, sections)

    def _start_report_generation(self, output_path: str, sections: dict):
        """Start the report generation in a worker thread."""
        # Validate experiment is not None
        if self.experiment is None:
            return

        # Disable the generate button while processing
        self.btn_generate.setEnabled(False)

        # Create and configure progress dialog
        self.progress_dialog = QtWidgets.QProgressDialog(
            "Generating report, please wait...",
            "",  # Text for cancel button
            0,
            0,  # Indeterminate progress
            self
        )
        self.progress_dialog.setWindowTitle("Generating Report")
        self.progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)  # Show immediately
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.setAutoReset(False)
        self.progress_dialog.setCancelButton(None)  # Remove cancel button
        self.progress_dialog.show()

        # Log generation start
        logging.info(f"Generating report for experiment: {self.experiment.name}")
        logging.info(f"Output path: {output_path}")
        logging.info(f"Sections: {sections}")

        # Create and start worker thread
        self.worker = ReportGeneratorWorker(self.experiment, output_path, sections)
        self.worker.finished.connect(self._on_report_generated)
        self.worker.error.connect(self._on_report_error)
        self.worker.start()

    def _on_report_generated(self, output_path: str):
        """Handle successful report generation."""
        # Close progress dialog
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None

        # Clean up worker
        if self.worker:
            self.worker.quit()
            self.worker.wait()
            self.worker = None

        # Re-enable generate button
        self.btn_generate.setEnabled(True)

        # Store the report path
        self.report_path = output_path

        logging.info(f"Report generated successfully: {output_path}")

        # Show success message
        QtWidgets.QMessageBox.information(
            self,
            "Report Generated",
            f"Report generated successfully!\n\nSaved to:\n{output_path}"
        )

        # Accept the dialog
        self.accept()

    def _on_report_error(self, error_message: str):
        """Handle report generation error."""
        # Close progress dialog
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None

        # Clean up worker
        if self.worker:
            self.worker.quit()
            self.worker.wait()
            self.worker = None

        # Re-enable generate button
        self.btn_generate.setEnabled(True)

        logging.error(f"Failed to generate report: {error_message}")

        # Show error message
        QtWidgets.QMessageBox.critical(
            self,
            "Error",
            f"Failed to generate report:\n\n{error_message}"
        )

    def get_report_path(self) -> Optional[str]:
        """Return the path to the generated report, or None if cancelled."""
        return self.report_path


def generate_report_dialog(
    experiment: Experiment,
    parent: Optional[QtWidgets.QWidget] = None,
) -> Optional[str]:
    """Create and execute the report generation dialog.

    Args:
        experiment: Experiment instance to generate a report for
        parent: Parent widget for the dialog

    Returns:
        str: Path to the generated PDF report, or None if cancelled
    """
    if experiment is None:
        raise ValueError("generate_report_dialog requires an experiment instance.")

    # Create and show the dialog
    dialog = AutoLamellaGenerateReportWidget(
        experiment=experiment,
        parent=parent,
    )
    result = dialog.exec_()

    # Handle the result
    if result == QtWidgets.QDialog.Accepted:
        report_path = dialog.get_report_path()
        if report_path:
            logging.info(f"Report generated: {report_path}")
        return report_path
    else:
        logging.info("Report generation cancelled")
        return None


def main():
    """Test the AutoLamellaGenerateReportWidget."""
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    PATH = "/home/patrick/github/fibsem/fibsem/applications/autolamella/log/AutoLamella-2025-10-15-13-22/experiment.yaml"
    exp = Experiment.load(PATH)

    if exp is None:
        logging.error("Provide an Experiment instance to test AutoLamellaGenerateReportWidget.")
        sys.exit(1)
        
    generate_report_dialog(exp)




if __name__ == "__main__":
    main()
