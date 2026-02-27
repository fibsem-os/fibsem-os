from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QProgressBar, QPushButton, QLabel
from PyQt5.QtCore import Qt
import sys
from fibsem.ui.stylesheets import INDETERMINATE_PROGRESS_BAR_STYLESHEET



class IndeterminateProgressExample(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: #262930; color: #d6d6d6;")
        self.setWindowTitle("Indeterminate Progress")
        self.resize(400, 120)

        layout = QVBoxLayout(self)

        self.label = QLabel("Idle")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)   # <-- indeterminate mode
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(INDETERMINATE_PROGRESS_BAR_STYLESHEET)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.btn_toggle = QPushButton("Start")
        self.btn_toggle.clicked.connect(self._toggle)
        layout.addWidget(self.btn_toggle)

    def _toggle(self):
        if self.progress_bar.isVisible():
            self.progress_bar.hide()
            self.label.setText("Idle")
            self.btn_toggle.setText("Start")
        else:
            self.progress_bar.show()
            self.label.setText("Working...")
            self.btn_toggle.setText("Stop")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = IndeterminateProgressExample()
    w.show()
    sys.exit(app.exec_())
