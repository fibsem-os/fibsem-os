"""The main window over an experiment with one lamella, for tests that run a
task's proposals through it.

A proposal a task asks on the record is put in front of the operator by the
main window (``AttentionRouter``): a state or a loop's pass on the prompt
bar. A test that decides a pass on the prompt bar needs that window.
"""

import os
from contextlib import contextmanager
from copy import deepcopy
from typing import Dict, Iterator, Optional, Sequence

import pytest
from psygnal.containers import EventedDict

from tests.protocols import planned, task_protocol


@contextmanager
def main_window(
    qapp,
    tmp_path,
    name: str,
    configs: Dict[str, object],
    supervised: Sequence[str] = (),
) -> Iterator[object]:
    """The window (Demo), its experiment holding one lamella with ``configs``
    as its tasks, those named in ``supervised`` supervised. Closed after."""
    from fibsem.applications.autolamella.structures import (
        Experiment,
    )
    from fibsem.applications.autolamella.ui import window as module
    from fibsem.workflows.attention import Attention

    win = module.AutoLamellaWindow()
    ui = win
    ui.system_widget.connect_to_microscope()
    experiment = Experiment(path=tmp_path, name=name)
    os.makedirs(experiment.path, exist_ok=True)
    experiment.task_protocol = task_protocol(
        *[
            planned(
                name=task_name,
                required=True,
                attention=Attention.supervised
                if task_name in supervised
                else Attention.automated,
                config=deepcopy(config),
            )
            for task_name, config in configs.items()
        ]
    )
    experiment.add_new_lamella(
        ui.microscope.get_microscope_state(), EventedDict(dict(configs))
    )
    experiment.positions[0].path.mkdir(parents=True, exist_ok=True)
    ui.experiment = experiment
    win._on_experiment_update()
    try:
        yield win
    finally:
        ui.session.runner.manager = None
        ui.session.disconnect()
        with quit_stubbed(qapp):
            win.close()
        from PyQt5.QtCore import QCoreApplication, QEvent

        win.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        qapp.processEvents()


@contextmanager
def quit_stubbed(app) -> Iterator[None]:
    """``app.quit()`` does nothing inside the block. A window's closeEvent ends in
    app.quit(), which on the shared test QApplication latches an interrupt every
    later QEventLoop.exec_() returns from."""
    original_quit = app.quit
    app.quit = lambda: None
    try:
        yield
    finally:
        app.quit = original_quit


def close_window(win) -> None:
    """Close the window as a test ends, with quit stubbed for the close."""
    from PyQt5.QtWidgets import QApplication

    with quit_stubbed(QApplication.instance()):
        win.close()


def main_window_fixture(
    connect: bool = False,
    config: Optional[str] = None,
    scope: str = "function",
    name: str = "main_ui",
):
    """A ``main_ui`` (or ``name``) fixture: the AutoLamella window, connected to the Demo
    microscope when ``connect`` (on ``config``, a file name in ``fibsem/config``,
    if given), disconnected and closed after."""

    @pytest.fixture(scope=scope, name=name)
    def _main_ui(qapp):
        import fibsem.config as cfg
        from fibsem.applications.autolamella.ui import window as module

        window = module.AutoLamellaWindow()
        with pytest.MonkeyPatch.context() as mp:
            if config is not None:
                path = os.path.join(cfg.SHIPPED_CONFIG_PATH, config)
                mp.setattr(
                    window.system_widget,
                    "load_configuration",
                    lambda configuration_name=None: path,
                )
            if connect:
                window.system_widget.connect_to_microscope()
            yield window
            if window.microscope is not None:
                window.microscope.disconnect()
            close_window(window)

    return _main_ui
