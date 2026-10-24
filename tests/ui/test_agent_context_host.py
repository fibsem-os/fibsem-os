"""AutoLamellaAppContext against the real window.

The unit tests (tests/autolamella/test_agent_context.py) drive the facade over
real domain objects in a plain holder; this file proves the production host —
an actual window — satisfies the same contract, in the states the facade
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
from fibsem.applications.autolamella.ui.window import AutoLamellaWindow
from fibsem.codec import encode
from fibsem.structures.settings import MicroscopeState
from tests.fixtures.main_window import close_window


@pytest.fixture
def ui(qapp):
    window = AutoLamellaWindow()
    try:
        yield window
    finally:
        close_window(window)
        window.deleteLater()
        qapp.processEvents()


def test_a_fresh_window_satisfies_the_host_contract(ui):
    ctx = AutoLamellaAppContext(ui.session)
    status = encode(ctx.status())
    json.dumps(status)
    assert status["microscope_connected"] is False
    assert status["experiment"] is None
    assert status["workflow"]["running"] is False
    assert encode(ctx.queue()) == {"items": [], "version": None}
    assert ctx.run_summary().error_type == "not_found"
    # No images until a microscope connects.
    assert ctx.display_images().error_type == "not_found"


def test_display_images_are_the_sessions_latest_images(ui):
    """FIB-1451: the session holds the latest image of each beam; the server
    reads them there, once a microscope connects and the window shows them."""
    import numpy as np

    from fibsem.structures.image import FibsemImage
    from fibsem.structures.settings import BeamType

    sem = FibsemImage.generate_blank_image(resolution=[128, 128], hfw=1e-6)
    sem.data[:] = np.random.randint(0, 255, sem.data.shape, dtype=np.uint8)
    ui.session.images.set(sem, BeamType.ELECTRON)
    ctx = AutoLamellaAppContext(ui.session)
    # A stand-in microscope: the window must not hear of it and build for it.
    with ui.session.changed.blocked():
        ui.session.microscope = object()
        try:
            payload = encode(ctx.display_images())
        finally:
            ui.session.microscope = None
    json.dumps(payload)
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

    # built BEFORE the experiment exists — call-time wins
    ctx = AutoLamellaAppContext(ui.session)
    ui.experiment = exp

    status = encode(ctx.status())
    assert status["experiment"]["name"] == "host-exp"
    assert status["experiment"]["num_items"] == 1
    assert ctx.task_outputs(exp.positions[0].name).item_name == exp.positions[0].name


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
    from fibsem.ui.run import view as ui_module

    _Dialog.opened = []
    monkeypatch.setattr(ui_module, "WorkflowSummaryDialog", _Dialog)
    shown = []

    def show(
        message, notification_type="info", temporary=False, action=None, always=False
    ):
        # A run's end pops up even with toasts for problems only.
        assert always, "the run-end toast must be marked always"
        shown.append((message, notification_type, action))

    monkeypatch.setattr(ui_module.notification_service, "show", show)
    return shown


def test_run_summary_survives_the_run_end_toast(ui, toasts):
    # The live-run regression: the run-end summary used to consume
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

    ui.run_view.show_summary()
    ui.run_view.show_summary()  # a second finished-signal must not re-show
    assert len(toasts) == 1
    message, notification_type, action = toasts[0]
    assert message == "Run ended: 1 completed"
    assert notification_type == "success"
    assert action[0] == "Show summary"
    # No experiment, so no option set: the summary opens, as it always has,
    # but without blocking the window.
    assert _Dialog.opened == [(ui.session.runner.last_summary, ""), "show"]

    payload = AutoLamellaAppContext(ui.session).run_summary()
    assert payload["items"][0]["task_name"] == "Rough Milling"
    json.dumps(payload)

    # The next run's capture is a new object: it shows again.
    ui.session.runner.last_summary = pd.DataFrame([{"task_name": "Polishing"}])
    ui.run_view.show_summary()
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

    ui.run_view.show_summary()
    assert _Dialog.opened == []
    _message, notification_type, (_text, callback) = toasts[0]
    assert notification_type == "warning"  # a failure is not a plain finish

    callback()
    assert _Dialog.opened == [(summary, ""), "show"]
