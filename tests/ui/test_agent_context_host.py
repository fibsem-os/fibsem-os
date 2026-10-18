"""AutoLamellaAppContext against the real window.

The unit tests (tests/autolamella/test_agent_context.py) drive the facade over
real domain objects in a plain holder; this file proves the production host —
an actual AutoLamellaUI — satisfies the same contract, in the states the facade
will actually meet it: freshly constructed (nothing connected, nothing loaded)
and with an experiment adopted."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json

import pytest

pytest.importorskip("PyQt5")

from psygnal.containers import EventedDict

from fibsem.applications.autolamella.server import AutoLamellaAppContext
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI
from fibsem.structures import MicroscopeState


@pytest.fixture
def ui(qapp):
    window = AutoLamellaUI(parent_ui=None)
    try:
        yield window
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_a_fresh_window_satisfies_the_host_contract(ui):
    ctx = AutoLamellaAppContext(ui)
    status = ctx.status()
    json.dumps(status)
    assert status["microscope_connected"] is False
    assert status["experiment"] is None
    assert status["workflow"]["running"] is False
    assert ctx.queue()["available"] is False
    assert ctx.run_summary()["available"] is False
    # No image widget until a microscope connects: unavailable, not an error.
    assert ctx.display_images() == {"available": False, "sem": None, "fib": None}


def test_display_images_are_the_sessions_latest_images(ui):
    """FIB-1451: the session holds the latest image of each beam; the server
    reads them there, once the window shows images at all."""
    import numpy as np

    from fibsem.structures import BeamType, FibsemImage

    sem = FibsemImage.generate_blank_image(resolution=[128, 128], hfw=1e-6)
    sem.data[:] = np.random.randint(0, 255, sem.data.shape, dtype=np.uint8)
    ui.session.images.set(sem, BeamType.ELECTRON)
    ui.image_widget = object()
    ctx = AutoLamellaAppContext(ui)

    payload = ctx.display_images()
    json.dumps(payload)
    assert payload["available"] is True
    assert payload["fib"] is None  # nothing displayed on that side yet
    sem = payload["sem"]
    assert sem["width"] > 0 and sem["height"] > 0
    assert sem["image_b64_jpeg"]
    # Blank placeholders carry no acquisition date; the key degrades to None
    # rather than the whole payload failing.
    assert "acquired_at" in sem


def test_an_adopted_experiment_is_visible_through_the_facade(ui, tmp_path):
    exp = Experiment(path=tmp_path / "exp", name="host-exp")
    exp.task_protocol = AutoLamellaTaskProtocol()
    (tmp_path / "exp").mkdir(parents=True, exist_ok=True)
    exp.add_new_lamella(MicroscopeState(), EventedDict())

    ctx = AutoLamellaAppContext(
        ui
    )  # built BEFORE the experiment exists — call-time wins
    ui.experiment = exp

    status = ctx.status()
    assert status["experiment"]["name"] == "host-exp"
    assert status["experiment"]["num_items"] == 1
    assert ctx.task_outputs(exp.positions[0].name)["available"] is True


class _Dialog:
    """Stands in for WorkflowSummaryDialog: records what opened, and how."""

    opened = []

    def __init__(self, summary, note="", parent=None):
        _Dialog.opened.append((summary, note))

    def setAttribute(self, *args):
        pass

    def show(self):
        _Dialog.opened.append("show")

    def exec_(self):
        raise AssertionError("the run summary must not block the window")


@pytest.fixture
def toasts(monkeypatch):
    from fibsem.applications.autolamella.ui import AutoLamellaUI as ui_module

    _Dialog.opened = []
    monkeypatch.setattr(ui_module, "WorkflowSummaryDialog", _Dialog)
    shown = []
    monkeypatch.setattr(
        ui_module.notification_service,
        "show",
        lambda message, notification_type="info", temporary=False, action=None: (
            shown.append((message, notification_type, action))
        ),
    )
    return shown


def test_run_summary_survives_the_run_end_toast(ui, toasts):
    # The live-run regression: _show_workflow_summary used to consume
    # the last run summary to make the dialog once-only, so a remote read after
    # the run always found nothing. The run's end is shown once; the record
    # stays readable.
    import pandas as pd

    ui.session.runner.last_summary = pd.DataFrame(
        [
            {
                "lamella_name": "01",
                "task_name": "Rough Milling",
                "task_status": "Completed",
            }
        ]
    )

    ui._show_workflow_summary()
    ui._show_workflow_summary()  # a second finished-signal must not re-show
    assert len(toasts) == 1
    message, notification_type, action = toasts[0]
    assert message == "Run ended: 1 completed"
    assert notification_type == "success"
    assert action[0] == "Show summary"
    # No experiment, so no option set: the summary opens, as it always has,
    # but without blocking the window.
    assert _Dialog.opened == [(ui.session.runner.last_summary, ""), "show"]

    payload = AutoLamellaAppContext(ui).run_summary()
    assert payload["available"] is True
    assert payload["items"][0]["task_name"] == "Rough Milling"
    json.dumps(payload)

    # The next run's capture is a new object: it shows again.
    ui.session.runner.last_summary = pd.DataFrame([{"task_name": "Polishing"}])
    ui._show_workflow_summary()
    assert len(toasts) == 2


def test_with_the_summary_off_only_the_toast_shows(ui, toasts, tmp_path):
    # FIB-1376: "Show summary after completion" off leaves the toast, whose
    # link opens the summary on demand.
    import pandas as pd

    exp = Experiment(path=tmp_path / "exp", name="host-exp")
    exp.task_protocol = AutoLamellaTaskProtocol()
    exp.task_protocol.options.show_run_summary = False
    ui.experiment = exp
    summary = pd.DataFrame(
        [{"lamella_name": "01", "task_name": "Polishing", "task_status": "Failed"}]
    )
    ui.session.runner.last_summary = summary
    ui.session.runner.last_note = ""

    ui._show_workflow_summary()
    assert _Dialog.opened == []
    _message, notification_type, (_text, callback) = toasts[0]
    assert notification_type == "warning"  # a failure is not a plain finish

    callback()
    assert _Dialog.opened == [(summary, ""), "show"]
