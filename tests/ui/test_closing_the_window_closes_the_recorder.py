"""Closing the main window closes the event recorder: what is queued reaches
events.jsonl -- the edit the close flushes among it -- and the writer thread
stops, rather than running on until the process ends.

The real main window on Demo, with the recorder it builds on connecting.
"""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from psygnal.containers import EventedDict  # noqa: E402

from fibsem.applications.autolamella.event_recording import (  # noqa: E402
    EVENTS_FILENAME,
    read_events,
    recorder_for,
)
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks.rough import (  # noqa: E402
    MillRoughTaskConfig,
)
from fibsem.structures import MicroscopeState  # noqa: E402

TASK = "Rough Milling"


@pytest.fixture
def window(qapp):
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module

    win = module.AutoLamellaSingleWindowUI()
    win.autolamella_ui.system_widget.connect_to_microscope()
    microscope = win.autolamella_ui.microscope
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        yield win
    finally:
        qapp.quit = original_quit
        win.autolamella_ui._stop_event_recorder()  # if the test did not get to close
        microscope.disconnect()


def test_closing_the_window_writes_the_last_edit_and_stops_the_writer(window, tmp_path):
    ui = window.autolamella_ui
    experiment = Experiment(path=tmp_path, name="closing")
    os.makedirs(experiment.path, exist_ok=True)
    experiment.task_protocol = AutoLamellaTaskProtocol()
    experiment.task_protocol.task_config[TASK] = MillRoughTaskConfig(task_name=TASK)
    experiment.add_new_lamella(MicroscopeState(), EventedDict())
    experiment.positions[0].task_config[TASK] = MillRoughTaskConfig(task_name=TASK)
    ui._adopt_experiment(experiment)
    editor = window.lamella_widget
    editor.set_experiment()
    editor.listWidget_selected_task.set_tasks([TASK], preferred=TASK)
    recorder = ui._event_recorder
    microscope = ui.microscope
    # an edit still settling when the window closes
    editor._on_task_parameters_config_changed("reacquire_alignment_reference", True)

    window.close()

    assert ui._event_recorder is None
    assert not recorder.writer.alive
    assert recorder_for(microscope) is None
    records = list(read_events(Path(experiment.path) / EVENTS_FILENAME))
    ((edit),) = [r for r in records if r["kind"] == "edit"]
    assert edit["payload"]["target"] == "parameters.reacquire_alignment_reference"


def test_closing_the_window_on_its_own_stops_the_writer(qapp):
    """``AutoLamellaUI`` by itself, as tests and scripts open it."""
    from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI

    ui = AutoLamellaUI(parent_ui=None)
    ui.system_widget.connect_to_microscope()
    microscope = ui.microscope
    recorder = ui._event_recorder
    try:
        assert recorder is not None and recorder.writer.alive

        ui.close()

        assert ui._event_recorder is None
        assert not recorder.writer.alive
        assert recorder_for(microscope) is None
    finally:
        ui._stop_event_recorder()
        microscope.disconnect()
        ui.deleteLater()
        qapp.processEvents()
