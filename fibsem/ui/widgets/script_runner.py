"""Shared glue between user scripts and a Qt host.

Owns everything the Scripts menu and the Scripts manager dialog both need —
finding scripts, deciding whether one may run, running it, and rendering what it
returned — so neither surface duplicates it.

Knows nothing about experiments, microscopes or AutoLamella. A host supplies:

* where the scripts live
* a context factory — what to hand the script, or ``None`` if it cannot run now
* a notifier — how to tell the user something

Data scripts run inline on the GUI thread; scripts that declare ``uses_microscope``
run on a worker thread, because hardware operations are far too slow to freeze the
window for. See FIB-338 and FIB-340.
"""

import logging
import threading
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QMessageBox,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from fibsem.cancellation import OperationCancelledError
from fibsem.scripting import DiscoveredScript, ScriptResult, discover_scripts, run_script
from fibsem.ui.qt.threading import FunctionWorker
from fibsem.ui.stylesheets import (
    MESSAGE_BOX_STYLESHEET,
    PRIMARY_BUTTON_STYLESHEET,
    SECONDARY_BUTTON_STYLESHEET,
)
from fibsem.ui.utils import open_path_in_file_explorer

# (context, reason). A context of None means "cannot run", and reason says why.
ContextFactory = Callable[[], Tuple[Optional[Any], str]]
# (message, level) where level is info/success/warning/error
Notifier = Callable[[str, str], None]
# (question, detail) -> True to go ahead. Injectable so the gate can be driven in tests.
Confirmer = Callable[[str, str], bool]


class ScriptRunner:
    """Finds, runs and renders user scripts on behalf of a Qt host."""

    def __init__(
        self,
        scripts_directory: Callable[[], Path],
        context_factory: ContextFactory,
        notify: Notifier,
        parent: Optional[QWidget] = None,
        confirm: Optional[Confirmer] = None,
    ) -> None:
        self._scripts_directory = scripts_directory
        self._directory_override: Optional[Path] = None
        self.context_factory = context_factory
        self.notify = notify
        self.parent = parent
        self.confirm = confirm or self._default_confirm
        self._worker: Optional[FunctionWorker] = None
        self._stop_event = threading.Event()

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

    def _default_confirm(self, question: str, detail: str) -> bool:
        """Yes/No box defaulting to Cancel, since this gates a destructive action."""
        # parented to whatever the user is actually looking at, so it centres over
        # the manager dialog when the run came from there rather than over the
        # main window behind it
        box = QMessageBox(QApplication.activeWindow() or self.parent)
        box.setWindowTitle("Run script")
        box.setText(question)
        box.setInformativeText(detail)
        box.setIcon(QMessageBox.Warning)
        box.setStyleSheet(MESSAGE_BOX_STYLESHEET)
        run = box.addButton("Run script", QMessageBox.AcceptRole)
        cancel = box.addButton("Cancel", QMessageBox.RejectRole)
        # same weighting as the dialog's own footer: Run primary, Cancel secondary
        run.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        cancel.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        box.setDefaultButton(cancel)

        # QMessageBox sizes itself from its own layout and ignores a stylesheet
        # min-width, so the width has to come from the layout too. Without this it
        # comes up at ~280px and wraps two short sentences over three lines.
        layout = box.layout()
        layout.addItem(
            QSpacerItem(400, 0, QSizePolicy.Minimum, QSizePolicy.Fixed),
            layout.rowCount(), 0, 1, layout.columnCount(),
        )

        box.exec_()
        return box.clickedButton() is run

    def _consequence(self, script: DiscoveredScript) -> str:
        """What running this script will do, for the confirmation box. Empty = no gate."""
        parts = []
        if script.uses_microscope:
            parts.append("drives the microscope, and nothing checks what it does")
        if script.writes:
            parts.append("modifies the experiment and saves it when it finishes")
        if not parts:
            return ""
        return "It " + " and ".join(parts) + ". There is no undo."

    @property
    def is_running(self) -> bool:
        """Whether a script is running on a worker thread right now."""
        return self._worker is not None and self._worker.is_alive()

    def stop(self) -> None:
        """Ask the running script to stop.

        Cooperative: a Python thread cannot be killed, so this only sets the event.
        A script that never checks it runs to completion regardless.
        """
        if self.is_running:
            logging.info("Stop requested for the running script.")
            self._stop_event.set()

    def run(
        self,
        script: DiscoveredScript,
        on_finished: Optional[Callable[[ScriptResult], None]] = None,
    ) -> Optional[ScriptResult]:
        """Run one script and render whatever it returned.

        Returns the result for a script that ran inline, and ``None`` both when the
        script was refused and when it went to a worker thread. ``on_finished`` is
        called with the result either way, so a caller that needs the outcome does
        not have to care which happened.
        """
        if self.is_running:
            self.notify("A script is already running.", "warning")
            return None

        context, reason = self.context_factory()
        if context is None:
            self.notify(reason or "Cannot run scripts right now.", "warning")
            return None

        if script.background and not script.uses_microscope:
            logging.warning(
                "Script %s declares background=True; running inline for now.", script.name
            )

        # Last gate before running. A writes script saves the moment it finishes and a
        # microscope script has already moved the hardware by then -- neither has an
        # undo. Read-only scripts, the common case, go straight through.
        consequence = self._consequence(script)
        if consequence and not self.confirm(f"Run '{script.name}'?", consequence):
            logging.info("Script %s cancelled at the confirmation.", script.name)
            return None

        if script.uses_microscope:
            self._run_threaded(script, context, on_finished)
            return None

        # Inline, on the GUI thread, so a slow one freezes the window -- the cursor is
        # the only sign it is working. Set after the confirmation, or the modal box
        # above would come up under a wait cursor.
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            result = run_script(script, context)
        finally:
            QApplication.restoreOverrideCursor()

        self._report(script, result)
        if on_finished is not None:
            on_finished(result)
        return result

    def _run_threaded(
        self,
        script: DiscoveredScript,
        context: Any,
        on_finished: Optional[Callable[[ScriptResult], None]],
    ) -> None:
        """Run a microscope script on a worker thread.

        Hardware operations take seconds to minutes. Inline they would freeze the
        window for the whole run, which is indistinguishable from a hang -- so these
        go to a thread even though data scripts do not. The consequence is that the
        script must not touch ``ctx.experiment``: it holds evented containers whose
        handlers fire into widgets, and those must only be touched from the GUI
        thread. Nothing enforces that; it is documented in SCRIPTING.md.
        """
        self._stop_event = threading.Event()
        if hasattr(context, "stop_event"):
            context.stop_event = self._stop_event

        worker = self._worker = FunctionWorker(run_script, script, context)

        def _done(result: object) -> None:
            # FunctionWorker delivers this on the GUI thread, so touching widgets
            # (toasts, dialogs) from here is safe. run_script never raises, so
            # `returned` always fires and `errored` never does.
            self._worker = None
            if isinstance(result, ScriptResult):
                self._report(script, result)
                if on_finished is not None:
                    on_finished(result)

        worker.returned.connect(_done)
        self.notify(f"{script.name} started…", "info")
        worker.start()

    def _report(self, script: DiscoveredScript, result: ScriptResult) -> None:
        """Surface the outcome of a finished run."""
        if isinstance(result.error, OperationCancelledError):
            self.notify(f"{script.name} cancelled.", "warning")
            return
        if not result.ok:
            self.notify(f"{script.name} failed: {result.error}", "error")
            return
        self.show_result(script, result.value)

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
