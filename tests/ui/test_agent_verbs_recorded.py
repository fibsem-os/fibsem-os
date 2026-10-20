"""What the agent does through the App Server is the agent's in the record (FIB-1276).

The server marks each request as the agent's, but a mark stays on its thread of
control, and the start verbs run on the GUI thread. The run's worker is started
there and takes its mark from there, so ``workflow_started`` said operator. The
same three-thread shape as the start tests: the request from a thread, the GUI
thread spinning, the run's worker started from the GUI thread.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from psygnal.containers import EventedDict  # noqa: E402

from fibsem.acting import OPERATOR, current_actor  # noqa: E402
from fibsem.applications.autolamella.server import AutoLamellaAppContext  # noqa: E402
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.ui.window import AutoLamellaWindow  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.structures import MicroscopeState  # noqa: E402
from fibsem.workflows.event_recording import (  # noqa: E402
    EventRecorder,
)
from fibsem.workflows.grid.imaging import (  # noqa: E402
    BeamOverviewGridTaskConfig,
)
from fibsem.workflows.grids import GridRecord  # noqa: E402
from fibsem.workflows.runner import GRIDS, ITEMS, RunRequest
from tests.fixtures.main_window import close_window  # noqa: E402
from tests.protocols import add_planned, planned
from tests.run_state import hold_runs

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
TASK_NAME = "Mill Fiducial"


@pytest.fixture
def ui(qapp, monkeypatch, tmp_path):
    import fibsem.config as fibsem_config

    arctis_config = os.path.join(
        os.path.dirname(fibsem_config.__file__),
        "config",
        "sim-arctis-configuration.yaml",
    )
    widget = AutoLamellaWindow()
    monkeypatch.setattr(
        widget.system_widget,
        "load_configuration",
        lambda configuration_name=None: arctis_config,
    )
    widget.system_widget.connect_to_microscope()

    experiment = Experiment(path=tmp_path, name="agent-verbs")
    experiment.task_protocol = AutoLamellaTaskProtocol()
    add_planned(experiment.task_protocol, planned(name=TASK_NAME, required=True))
    experiment.grid_protocol.add(BeamOverviewGridTaskConfig(task_name="overview_sem"))
    experiment.add_grid(GridRecord(name="grid-aspen"))
    experiment.add_new_lamella(MicroscopeState(), EventedDict())
    widget.experiment = experiment

    yield widget
    widget.experiment = None
    if widget.microscope is not None:
        widget.microscope.disconnect()
    close_window(widget)
    widget.deleteLater()
    qapp.processEvents()


@pytest.fixture
def recorder(ui):
    # As the app builds it: anything unmarked is the operator's.
    recorder = EventRecorder(ui.microscope, default_actor=OPERATOR)
    yield recorder
    recorder.close()


def _client(ui, recorder):
    app = build_server(
        ui.microscope,
        app_context=AutoLamellaAppContext(
            ui.session, window=ui, event_buffer=recorder.buffer
        ),
        auth=AuthConfig.generate(level="drive", token=TOKEN),
    )
    return TestClient(app, raise_server_exceptions=False)


def _spin_until(qapp, predicate, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not reached")
        qapp.processEvents()
        time.sleep(0.01)


def _post(qapp, client, path, body):
    """POST from another thread while the GUI thread spins, like production."""
    posted = {}

    def do_post():
        posted["response"] = client.post(path, headers=AUTH, json=body)

    threading.Thread(target=do_post, daemon=True).start()
    _spin_until(qapp, lambda: "response" in posted)
    return posted["response"]


def _actors(recorder, kind):
    return [
        e["actor"]
        for e in recorder.buffer.events_since(0)["events"]
        if e["kind"] == kind
    ]


def _runs_that_announce_themselves(monkeypatch, kind, recorder, seen):
    """Runs that do what a real one does first: on the run's thread, report
    that the run started."""

    def announce():
        seen.append(current_actor())
        recorder.buffer.append("workflow_started", {})

    return hold_runs(monkeypatch, kind, on_execute=announce)


def _wait_for(seen, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while not seen and time.monotonic() < deadline:
        time.sleep(0.01)


def test_a_workflow_the_agent_starts_is_the_agent_s(ui, recorder, qapp, monkeypatch):
    seen = []
    held = _runs_that_announce_themselves(monkeypatch, ITEMS, recorder, seen)
    with _client(ui, recorder) as client:
        started = _post(
            qapp, client, "/app/workflow/start", {"task_names": [TASK_NAME]}
        ).json()
    assert started["started"] is True
    _wait_for(seen)
    held.end(ui.session.runner)
    assert seen == ["agent"]
    assert _actors(recorder, "workflow_started") == ["agent"]


def test_a_grid_run_the_agent_starts_is_the_agent_s(ui, recorder, qapp, monkeypatch):
    seen = []
    held = _runs_that_announce_themselves(monkeypatch, GRIDS, recorder, seen)
    with _client(ui, recorder) as client:
        _post(
            qapp, client, "/app/workflow/grids/start", {"task_names": ["overview_sem"]}
        )
    _wait_for(seen)
    held.end(ui.session.runner)
    assert seen == ["agent"]
    assert _actors(recorder, "workflow_started") == ["agent"]


def test_a_workflow_the_operator_starts_is_still_the_operator_s(
    ui, recorder, qapp, monkeypatch
):
    seen = []
    held = _runs_that_announce_themselves(monkeypatch, ITEMS, recorder, seen)
    # Unmarked, as a click is.
    ui.session.runner.start(
        ui.experiment, RunRequest(kind=ITEMS, task_names=(TASK_NAME,))
    )
    _wait_for(seen)
    held.end(ui.session.runner)
    assert seen == [None]
    assert _actors(recorder, "workflow_started") == ["operator"]


def test_the_agent_s_other_recorded_verbs_are_the_agent_s(ui, recorder, qapp):
    with _client(ui, recorder) as client:
        assert _post(
            qapp, client, "/app/experiment/notes", {"text": "fiducial looks clean"}
        ).json()["recorded"]
        assert _post(
            qapp,
            client,
            "/app/workflow/schedule",
            {"task_name": TASK_NAME, "scheduled_at": "2026-10-12T06:00:00"},
        ).json()["applied"]
    assert _actors(recorder, "note") == ["agent"]
    assert _actors(recorder, "workflow_changed") == ["agent"]
