"""Manual test script for toast notifications (FIB-100)."""
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from fibsem.ui.widgets.notifications import ToastManager, NotificationBell

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Toast Test - FIB-100")
        self.resize(600, 400)
        self.setStyleSheet("background-color: #262930;")

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.toast_manager = ToastManager(self)
        bell = NotificationBell(self)
        self.toast_manager.set_notification_bell(bell)
        layout.addWidget(bell)

        for ntype in ["info", "success", "warning", "error"]:
            btn = QPushButton(f"Show {ntype} toast")
            btn.setStyleSheet("color: #d6d6d6; background-color: #3d4251; padding: 8px; border-radius: 3px;")
            btn.clicked.connect(lambda checked, t=ntype: self.toast_manager.show_toast(f"This is a {t} message", t))
            layout.addWidget(btn)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = TestWindow()
    w.show()
    sys.exit(app.exec_())
