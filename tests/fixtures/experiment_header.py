"""A real ExperimentHeader over a stand-in session, for the header tests."""

from __future__ import annotations

from PyQt5.QtWidgets import QAction, QWidget

from fibsem.applications.autolamella.ui.header import ExperimentHeader
from fibsem.ui.widgets.notifications import ToastManager


class HeaderHost:
    """The header, and the experiment and microscope the window would tell it."""

    def __init__(self) -> None:
        self.experiment = None
        self.microscope = None
        self._parent = QWidget()
        self.actions = [
            QAction(name, self._parent)
            for name in ("New Experiment", "Load Experiment", "Open Directory")
        ]
        self.header = ExperimentHeader(
            new_experiment=self.actions[0],
            load_experiment=self.actions[1],
            open_experiment_directory=self.actions[2],
            on_new_experiment=lambda: None,
            on_load_experiment=lambda: None,
            on_about=lambda: None,
            toast_manager=ToastManager(self._parent),
            parent=self._parent,
        )

    def __getattr__(self, name):
        # The header's buttons, by the names the tests read.
        return getattr(self.__dict__["header"], name)

    def update_experiment_header(self) -> None:
        self.header.update_experiment_header(self.experiment, self.microscope)

    def close(self) -> None:
        self._parent.deleteLater()
