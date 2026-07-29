"""Shared glue between user scripts and a Qt host.

Owns everything the Scripts menu and the Scripts manager dialog both need —
finding scripts, deciding whether one may run, running it, and rendering what it
returned — so neither surface duplicates it.

Knows nothing about experiments, microscopes or AutoLamella. A host supplies:

* where the scripts live
* a context factory — what to hand the script, or ``None`` if it cannot run now
* a notifier — how to tell the user something

See FIB-338.
"""

import logging
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import pandas as pd
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QWidget

from fibsem.scripting import DiscoveredScript, ScriptResult, discover_scripts, run_script
from fibsem.ui.utils import open_path_in_file_explorer

# (context, reason). A context of None means "cannot run", and reason says why.
ContextFactory = Callable[[], Tuple[Optional[Any], str]]
# (message, level) where level is info/success/warning/error
Notifier = Callable[[str, str], None]


class ScriptRunner:
    """Finds, runs and renders user scripts on behalf of a Qt host."""

    def __init__(
        self,
        scripts_directory: Callable[[], Path],
        context_factory: ContextFactory,
        notify: Notifier,
        parent: Optional[QWidget] = None,
    ) -> None:
        self._scripts_directory = scripts_directory
        self._directory_override: Optional[Path] = None
        self.context_factory = context_factory
        self.notify = notify
        self.parent = parent

    def scripts_directory(self) -> Path:
        """Where to look for scripts — the host's folder, or a chosen override."""
        return self._directory_override or Path(self._scripts_directory())

    def set_directory(self, directory: Optional[Path]) -> None:
        """Point at a different folder for this session. ``None`` restores the host's."""
        self._directory_override = Path(directory) if directory is not None else None

    # --- discovery ---

    def discover(self) -> List[DiscoveredScript]:
        """Load the scripts fresh.

        Not cached anywhere: editing a file and running it again picks up the
        change, so "reload" is not a feature that needs to exist.
        """
        return discover_scripts(self.scripts_directory())

    def availability(self) -> Tuple[bool, str]:
        """Whether the host can run scripts right now, and why not if it cannot."""
        _context, reason = self.context_factory()
        return (not reason), reason

    def open_folder(self) -> None:
        """Create the scripts folder if needed and reveal it."""
        directory = self.scripts_directory()
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.notify(f"Could not create {directory}: {e}", "error")
            return
        open_path_in_file_explorer(str(directory))

    # --- running ---

    def run(self, script: DiscoveredScript) -> Optional[ScriptResult]:
        """Run one script and render whatever it returned.

        Returns ``None`` when the script was refused rather than run.
        """
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

    # --- rendering ---

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
