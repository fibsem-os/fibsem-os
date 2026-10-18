"""The main window over an experiment with one lamella, for tests that run a
task's questions through it.

A question a task asks on the record is put in front of the operator by the
main window (``_on_question_asked``): a state or a loop's pass on the prompt
bar, anything else in the Review tab. A test that answers a pass on the
prompt bar needs that window, not the bare ``AutoLamellaUI`` inside it.
"""

import os
from contextlib import contextmanager
from typing import Dict, Iterator, Sequence

from psygnal.containers import EventedDict
from PyQt5.QtCore import QCoreApplication, QEvent


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
        AutoLamellaTaskDescription,
        AutoLamellaTaskProtocol,
        AutoLamellaWorkflowConfig,
        Experiment,
    )
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module
    from fibsem.workflows.attention import Attention

    win = module.AutoLamellaSingleWindowUI()
    ui = win.autolamella_ui
    ui.system_widget.connect_to_microscope()
    experiment = Experiment(path=tmp_path, name=name)
    os.makedirs(experiment.path, exist_ok=True)
    experiment.task_protocol = AutoLamellaTaskProtocol(
        workflow_config=AutoLamellaWorkflowConfig(
            tasks=[
                AutoLamellaTaskDescription(
                    name=task_name,
                    required=True,
                    attention=Attention.supervised
                    if task_name in supervised
                    else Attention.automated,
                )
                for task_name in configs
            ]
        )
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
        microscope = ui.microscope
        ui.disconnect_from_microscope()
        if microscope is not None:
            microscope.disconnect()
        original_quit = qapp.quit
        qapp.quit = lambda: None
        try:
            win.close()
        finally:
            qapp.quit = original_quit
        win.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        qapp.processEvents()
