"""A reusable "Scripts" menu for running user scripts from a Qt application.

Knows nothing about experiments, microscopes or AutoLamella. A host supplies:

* where the scripts live
* a context factory — what to hand the script, or ``None`` if it cannot run now
* a notifier — how to tell the user something

so the same menu can be dropped into any fibsem application, and tested without
one. See FIB-338.
"""

import logging
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import pandas as pd
from PyQt5.QtWidgets import QAction, QDialog, QMenu, QVBoxLayout, QWidget

from fibsem.scripting import DiscoveredScript, ScriptResult, discover_scripts, run_script
from fibsem.ui.utils import open_path_in_file_explorer

# (context, reason). A context of None means "cannot run", and reason says why.
ContextFactory = Callable[[], Tuple[Optional[Any], str]]
# (message, level) where level is info/success/warning/error
Notifier = Callable[[str, str], None]


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
    ) -> None:
        self.menu = menu
        self.scripts_directory = scripts_directory
        self.context_factory = context_factory
        self.notify = notify
        self.parent = parent

        self.menu.setToolTipsVisible(True)
        # Rebuilt on every open: scripts are not cached, so editing a file and
        # running it again picks up the change with no reload step.
        self.menu.aboutToShow.connect(self.rebuild)

    # --- menu ---

    def rebuild(self) -> None:
        self.menu.clear()
        scripts = self.discover()
        _context, reason = self.context_factory()
        runnable = not reason

        if not scripts:
            self._add_disabled("No scripts found")

        for script in scripts:
            self.menu.addAction(self._build_action(script, runnable))

        self.menu.addSeparator()
        if reason:
            self._add_disabled(reason)

        open_folder = QAction("Open scripts folder", self.menu)
        open_folder.triggered.connect(self.open_folder)
        self.menu.addAction(open_folder)

    def discover(self) -> List[DiscoveredScript]:
        return discover_scripts(self.scripts_directory())

    def _add_disabled(self, text: str) -> None:
        action = QAction(text, self.menu)
        action.setEnabled(False)
        self.menu.addAction(action)

    def _build_action(self, script: DiscoveredScript, host_ready: bool) -> QAction:
        label = script.name
        flags = [name for name, on in (("writes", script.writes),
                                       ("auto", script.on_workflow_completed)) if on]
        if flags:
            label += f"  ({', '.join(flags)})"

        action = QAction(label, self.menu)
        if not script.is_runnable:
            # A script that failed to load is shown, not omitted. Silently hiding
            # it leaves the author with nothing to diagnose.
            action.setEnabled(False)
            action.setToolTip(f"Cannot load: {script.error}")
            return action

        action.setToolTip(script.description)
        action.setEnabled(host_ready)
        action.triggered.connect(lambda _checked=False, s=script: self.run(s))
        return action

    # --- running ---

    def open_folder(self) -> None:
        """Create the scripts folder if needed and reveal it."""
        directory = self.scripts_directory()
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.notify(f"Could not create {directory}: {e}", "error")
            return
        open_path_in_file_explorer(str(directory))

    def run(self, script: DiscoveredScript) -> Optional[ScriptResult]:
        """Run one script and render whatever it returned."""
        context, reason = self.context_factory()
        if context is None:
            self.notify(reason or "Cannot run scripts right now.", "warning")
            return None

        if script.uses_microscope:
            # The strict runner these need -- worker thread, hardware exclusion,
            # cancellation, state restoration -- is FIB-340 and not built yet.
            self.notify(
                f"'{script.name}' declares uses_microscope, which is not supported yet.",
                "warning",
            )
            return None
        if script.background:
            logging.warning(
                "Script %s declares background=True; running inline for now.", script.name
            )

        result = run_script(script, context)

        if not result.ok:
            self.notify(f"{script.name} failed: {result.error}", "error")
            return result

        self.show_result(script, result.value)
        return result

    def show_result(self, script: DiscoveredScript, value: Any) -> None:
        """Render a script's return value.

        The return value carries the output because there is typically no log
        console in the app -- print() goes to a terminal the user may not have,
        and a packaged Windows build has none at all. Without this a working
        script looks like it did nothing.
        """
        if value is None:
            self.notify(f"{script.name} finished.", "success")
            return

        if isinstance(value, pd.DataFrame):
            self.show_dataframe(script.name, value)
            return

        if isinstance(value, Path):
            self.notify(f"{script.name} wrote {value.name}", "success")
            if value.exists():
                open_path_in_file_explorer(str(value.parent))
            return

        self.notify(f"{script.name}: {value}", "success")

    def show_dataframe(self, title: str, dataframe: pd.DataFrame) -> None:
        from fibsem.ui.widgets.dataframe_table_widget import DataFrameTableWidget

        dialog = QDialog(self.parent)
        dialog.setWindowTitle(title)
        dialog.resize(720, 420)
        layout = QVBoxLayout(dialog)
        layout.addWidget(DataFrameTableWidget(dataframe=dataframe, parent=dialog))
        dialog.exec_()
