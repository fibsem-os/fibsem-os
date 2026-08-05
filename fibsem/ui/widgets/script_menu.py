"""A reusable "Scripts" menu for a Qt application.

The menu is a way *in*, not a way to run. Running happens in the manager dialog
only, because a script can now drive the microscope on a worker thread and the
dialog is the only surface with a Stop button — a menu item that starts a
ten-minute stage survey with no way to interrupt it is a trap.

That also settles what the menu should list: nothing. Names alone cannot express
the folder, a load error, or that a script writes or needs hardware, and the
dialog shows all of it beside the button that starts it.

Knows nothing about experiments, microscopes or AutoLamella. See FIB-338, FIB-340.
"""

from pathlib import Path
from typing import Callable, Optional

from PyQt5.QtWidgets import QAction, QMenu, QWidget

from fibsem.ui.widgets.script_runner import (
    Confirmer,
    ContextFactory,
    Notifier,
    ScriptRunner,
)

MANAGE_LABEL = "Manage scripts…"
OPEN_FOLDER_LABEL = "Open scripts folder"


class ScriptMenuController:
    """Puts the way into user scripts on an existing QMenu.

    Attached to a menu rather than subclassing QMenu so a host can place it
    wherever it likes in its own menu tree.
    """

    def __init__(
        self,
        menu: QMenu,
        scripts_directory: Callable[[], Path],
        context_factory: ContextFactory,
        notify: Notifier,
        parent: Optional[QWidget] = None,
        confirm: Optional[Confirmer] = None,
    ) -> None:
        self.menu = menu
        self.parent = parent
        self.runner = ScriptRunner(
            scripts_directory=scripts_directory,
            context_factory=context_factory,
            notify=notify,
            parent=parent,
            confirm=confirm,
        )
        self.build()

    # --- menu ---

    def build(self) -> None:
        """Two fixed entries.

        Built once, not on ``aboutToShow``: nothing here depends on what is in
        the folder any more, so there is nothing to rebuild and no reason to
        import every script each time the user opens the Tools menu.
        """
        self.menu.clear()

        manage = QAction(MANAGE_LABEL, self.menu)
        manage.triggered.connect(self.open_manager)
        self.menu.addAction(manage)

        open_folder = QAction(OPEN_FOLDER_LABEL, self.menu)
        open_folder.triggered.connect(self.runner.open_folder)
        self.menu.addAction(open_folder)

    # --- manager ---

    def open_manager(self) -> None:
        from fibsem.ui.widgets.script_manager_dialog import ScriptManagerDialog

        dialog = ScriptManagerDialog(runner=self.runner, parent=self.parent)
        dialog.exec_()
