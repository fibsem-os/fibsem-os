"""A reusable "Scripts" menu for running user scripts from a Qt application.

The menu is for running a known-good script in one click. Everything a menu
cannot express — the folder path, per-script load errors, the ``writes`` warning,
source and hash — lives in the manager dialog it opens. A failed script would
otherwise simply be absent from the menu, which is the silent-omission failure
this is meant to avoid.

Knows nothing about experiments, microscopes or AutoLamella. See FIB-338.
"""

from pathlib import Path
from typing import Callable, Optional

from PyQt5.QtWidgets import QAction, QMenu, QWidget

from fibsem.scripting import DiscoveredScript
from fibsem.ui.widgets.script_runner import (
    Confirmer,
    ContextFactory,
    Notifier,
    ScriptRunner,
)

MANAGE_LABEL = "Manage scripts…"


class ScriptMenuController:
    """Populates a QMenu with the scripts in a directory, and runs them.

    Attached to an existing menu rather than subclassing QMenu so a host can place
    it wherever it likes in its own menu tree.
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

        self.menu.setToolTipsVisible(True)
        # Rebuilt on every open: scripts are not cached, so editing a file and
        # running it again picks up the change with no reload step.
        self.menu.aboutToShow.connect(self.rebuild)

    # --- menu ---

    def rebuild(self) -> None:
        self.menu.clear()
        scripts = self.runner.discover()
        host_ready, reason = self.runner.availability()

        runnable = [s for s in scripts if s.is_runnable]
        failed = len(scripts) - len(runnable)

        if not runnable:
            self._add_disabled("No scripts found" if not scripts else "No runnable scripts")

        for script in runnable:
            self.menu.addAction(self._build_action(script, host_ready))

        self.menu.addSeparator()
        if reason:
            self._add_disabled(reason)
        if failed:
            # named here, detail in the dialog -- a menu has nowhere to put a reason
            self._add_disabled(f"{failed} script(s) failed to load — see {MANAGE_LABEL}")

        manage = QAction(MANAGE_LABEL, self.menu)
        manage.triggered.connect(self.open_manager)
        self.menu.addAction(manage)

        open_folder = QAction("Open scripts folder", self.menu)
        open_folder.triggered.connect(self.runner.open_folder)
        self.menu.addAction(open_folder)

    def _add_disabled(self, text: str) -> None:
        action = QAction(text, self.menu)
        action.setEnabled(False)
        self.menu.addAction(action)

    def _build_action(self, script: DiscoveredScript, host_ready: bool) -> QAction:
        label = script.name
        # writes only. on_workflow_completed is not connected to anything yet, and
        # a menu has nowhere to say so -- an "auto" label here would read as a
        # promise. The dialog carries the flag and the caveat together.
        if script.writes:
            label += "  (writes)"

        action = QAction(label, self.menu)
        action.setToolTip(script.description)
        action.setEnabled(host_ready)
        action.triggered.connect(lambda _checked=False, s=script: self.runner.run(s))
        return action

    # --- manager ---

    def open_manager(self) -> None:
        from fibsem.ui.widgets.script_manager_dialog import ScriptManagerDialog

        dialog = ScriptManagerDialog(runner=self.runner, parent=self.parent)
        dialog.exec_()
