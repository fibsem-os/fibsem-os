"""Test script for confirm_run_workflow_dialog.

Run directly:
    python fibsem/applications/autolamella/ui/tests/test_confirm_run_workflow_dialog.py
"""

import sys

from PyQt5.QtWidgets import QApplication, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from fibsem.applications.autolamella.ui.AutoLamellaMainUI import confirm_run_workflow_dialog
from fibsem.ui import stylesheets

SCENARIOS = [
    {
        "label": "Single lamella, single task",
        "lamella_names": ["01-frank-orca"],
        "task_names": ["Setup Lamella Position"],
    },
    {
        "label": "Few lamella, multiple tasks",
        "lamella_names": ["01-frank-orca", "02-rapid-weevil", "03-moving-dove"],
        "task_names": ["Mill Fiducial", "Rough Milling", "Polishing"],
    },
    {
        "label": "Many lamella (20+)",
        "lamella_names": [f"{i:02d}-lamella-{i}" for i in range(1, 22)],
        "task_names": ["Setup Lamella Position", "Mill Fiducial", "Rough Milling"],
    },
]


def main():
    app = QApplication(sys.argv)

    window = QWidget()
    window.setWindowTitle("confirm_run_workflow_dialog — test launcher")
    window.resize(400, 200)
    layout = QVBoxLayout(window)

    btn_row = QHBoxLayout()
    for scenario in SCENARIOS:
        btn = QPushButton(scenario["label"])
        btn.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)

        def _launch(checked=False, s=scenario):
            confirmed = confirm_run_workflow_dialog(
                lamella_names=s["lamella_names"],
                task_names=s["task_names"],
                parent=window,
            )
            print(f"[{s['label']}] confirmed={confirmed}")

        btn.clicked.connect(_launch)
        btn_row.addWidget(btn)

    layout.addLayout(btn_row)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
