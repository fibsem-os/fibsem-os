"""Sample staging widget: the holder (working slot) + optional loader magazine.

Pure hardware staging — owns the holder/magazine sub-widgets and drives grid
load/unload on the microscope's stage, with no ``Experiment`` or lamella
coupling. The exchange itself is a stage operation (``Stage.ensure_loaded`` /
``loader.unload_grid``); this widget only runs it off the GUI thread and keeps
the views in sync.

Hosts wanting to react to a changed working slot (e.g. to refresh experiment
views) connect to :attr:`state_changed`.
"""

import threading
from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QGridLayout, QSplitter, QWidget

from fibsem.microscopes._stage import GridExchangeError
from fibsem.ui import notification_service
from fibsem.ui.widgets.loader_magazine_widget import LoaderMagazineWidget
from fibsem.ui.widgets.sample_holder_widget import SampleHolderWidget


class SampleWidget(QWidget):
    """Holder + optional magazine, with threaded grid load/unload."""

    # emitted after a load/unload completes (the loaded grid may have changed)
    state_changed = pyqtSignal()
    # private thread -> GUI bridge: (action, grid_name, error|None)
    _exchange_finished = pyqtSignal(str, str, object)

    def __init__(self, microscope, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.microscope = microscope
        self.holder_widget: Optional[SampleHolderWidget] = None
        self.magazine_widget: Optional[LoaderMagazineWidget] = None
        self._worker_thread: Optional[threading.Thread] = None

        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)

        # magazine (storage / source) — autoloader only, shown on top
        if microscope._stage.loader is not None:
            self.magazine_widget = LoaderMagazineWidget(microscope=microscope)
            self.magazine_widget.set_microscope(microscope)
            self.magazine_widget.magazine_changed.connect(self._on_magazine_changed)
            self.magazine_widget.load_requested.connect(self.request_load)
            self.magazine_widget.unload_requested.connect(self.request_unload)
            splitter.addWidget(self.magazine_widget)

        # holder (working slot / destination) — always present
        self.holder_widget = SampleHolderWidget(microscope=microscope)
        self.holder_widget.set_holder(microscope._stage.holder)
        splitter.addWidget(self.holder_widget)

        self._exchange_finished.connect(self._on_exchange_finished)

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    # --- public API ---------------------------------------------------------

    def request_load(self, grid_name: str) -> None:
        """Load a grid from the magazine into the working slot (off the GUI thread)."""
        if self._busy():
            return
        self.set_busy(True)
        self._start_worker("load", grid_name)

    def request_unload(self) -> None:
        """Retract whatever grid occupies the working slot (off the GUI thread)."""
        if self._busy():
            return
        self.set_busy(True)
        self._start_worker("unload", "")

    def refresh(self) -> None:
        """Refresh the holder + magazine views to match the working slot."""
        if self.holder_widget is not None:
            self.holder_widget.refresh()
        if self.magazine_widget is not None:
            self.magazine_widget.refresh_rows()

    def set_busy(self, busy: bool) -> None:
        """Show the magazine spinner + block its controls during an exchange."""
        if self.magazine_widget is not None:
            self.magazine_widget.set_busy(busy)

    # --- internals ----------------------------------------------------------

    def _busy(self) -> bool:
        return self._worker_thread is not None and self._worker_thread.is_alive()

    def _start_worker(self, action: str, grid_name: str) -> None:
        self._worker_thread = threading.Thread(
            target=self._exchange_worker, args=(action, grid_name), daemon=True,
        )
        self._worker_thread.start()

    def _on_magazine_changed(self) -> None:
        # a magazine grid and the working slot can reference the same SampleGrid
        # object once loaded — keep the holder view in sync
        if self.holder_widget is not None:
            self.holder_widget.refresh()

    def _exchange_worker(self, action: str, grid_name: str) -> None:
        """Worker thread: drive the stage, then signal back to the GUI thread."""
        error: Optional[str] = None
        try:
            if action == "load":
                self.microscope._stage.ensure_loaded(grid_name)
            else:
                self.microscope._stage.unload()
        except GridExchangeError as e:
            error = str(e)
        except Exception as e:  # noqa: BLE001 - surface unexpected failures too
            error = str(e)
        self._exchange_finished.emit(action, grid_name, error)

    def _on_exchange_finished(
        self, action: str, grid_name: str, error: Optional[str]
    ) -> None:
        """GUI thread: refresh views, re-enable controls, toast, notify host."""
        self.set_busy(False)
        self.refresh()
        if error is not None:
            notification_service.show_toast(f"Grid exchange failed: {error}", "error")
        elif action == "load":
            notification_service.show_toast(f"Loaded '{grid_name}' into the beam.", "info")
        self.state_changed.emit()
