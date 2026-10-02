"""Connecting to the microscope without freezing the window.

`utils.setup_session` blocks until the instrument is connected and has done what its
configuration asks for at connect -- and turning a ThermoFisher column on can take a
while (FIB-1153). Run on the GUI thread, nothing repaints for that long and the
application looks dead. `SessionConnector` runs it on a worker thread and reports
back on the GUI thread: each step as it starts, then the session or the failure.

`setup_session` itself stays synchronous, for scripts.
"""

import logging
from typing import Optional

from PyQt5.QtCore import QCoreApplication, QObject, pyqtSignal

from fibsem import utils
from fibsem.ui.qt.threading import FunctionWorker


class SessionConnector(QObject):
    """Connects with one configuration file, off the GUI thread.

    Signals arrive on the thread the connector was made on (the GUI thread):

    * ``progress(str)`` -- the step starting now ("Turning the beams on…").
    * ``connected(microscope, settings)`` -- the session, once it is ready.
    * ``failed(str)`` -- why it could not connect.
    """

    progress = pyqtSignal(str)
    connected = pyqtSignal(object, object)
    failed = pyqtSignal(str)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._worker: Optional[FunctionWorker] = None
        # Until the outcome is delivered, not until the thread ends: the thread can
        # be done while its result is still queued for this thread.
        self._connecting = False

    def is_connecting(self) -> bool:
        return self._connecting

    def start(self, config_path: str) -> None:
        """Start connecting. Ignored while an attempt is already running."""
        if self.is_connecting():
            return
        # Looked up now, not at import, so a test or a caller can stand in for it.
        worker = FunctionWorker(
            utils.setup_session, config_path=config_path, progress=self.progress.emit
        )
        worker.returned.connect(self._on_returned)
        worker.errored.connect(self._on_errored)
        self._worker = worker
        self._connecting = True
        worker.start()

    def _on_returned(self, session) -> None:
        self._connecting = False
        self.connected.emit(*session)

    def _on_errored(self, error: Exception) -> None:
        self._connecting = False
        self.failed.emit(str(error))

    def wait(self, timeout: Optional[float] = None) -> None:
        """Block until the attempt ends and its signals are delivered. For tests and
        for code that must not go on until it has -- never from a slot the user is
        waiting on."""
        if self._worker is None:
            return
        self._worker.join(timeout)
        if self._worker.is_alive():
            logging.warning("Still connecting after waiting %s s", timeout)
            return
        for _ in range(3):  # the result hops worker -> connector -> its listeners
            QCoreApplication.processEvents()
