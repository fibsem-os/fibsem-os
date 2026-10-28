"""A folder field for a driver's vendor library, with a Find button that asks the
other Pythons on PATH where they have it (``fibsem.vendor_libraries.search``)."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QInputDialog, QMessageBox, QToolButton

from fibsem import vendor_libraries
from fibsem.ui.widgets.custom_widgets import QDirectoryLineEdit


def _choice(found: vendor_libraries.LibraryFolder) -> str:
    return f"{found.folder} ({found.version})" if found.version else found.folder


class LibraryFolderEdit(QDirectoryLineEdit):
    def __init__(self, library=None, parent=None):
        """*library* is a driver's ``SetupLibrary``, or None until one is chosen."""
        super().__init__(parent)
        self.library = library
        self.button_find = QToolButton(self)
        self.button_find.setText("Find")
        self.button_find.setToolTip(
            "Ask each Python on PATH where it has this library."
        )
        self.button_find.clicked.connect(self.find)
        self.layout().addWidget(self.button_find)

    def find(self) -> None:
        """Fill the field with a folder found, choosing where there are several."""
        if self.library is None:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            found = vendor_libraries.search(self.library)
        finally:
            QApplication.restoreOverrideCursor()
        if not found:
            QMessageBox.information(
                self,
                "Find",
                f"No Python on PATH has the {self.library.label}. Choose its "
                "folder with the ... button.",
            )
            return
        folder = found[0].folder
        if len(found) > 1:
            choices = [_choice(f) for f in found]
            picked, ok = QInputDialog.getItem(
                self, "Find", f"Use the {self.library.label} in:", choices, 0, False
            )
            if not ok:
                return
            folder = found[choices.index(picked)].folder
        self.setText(folder)
