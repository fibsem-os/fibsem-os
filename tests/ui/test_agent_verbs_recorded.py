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
from fibsem.applications.autolamella.event_recording import (  # noqa: E402
    EventRecorder,
)
from fibsem.applications.autolamella.server import AutoLamellaAppContext  # noqa: E402
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    Experiment,
    GridRecord,
)
from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI  # noqa: E402
from fibsem.applications.autolamella.workflows.tasks.grid.imaging import (  # noqa: E402
    BeamOverviewGridTaskConfig,
)
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.structures import MicroscopeState  # noqa: E402
from fibsem.ui.qt.threading import FunctionWorker  # noqa: E402

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
    widget = AutoLamellaUI(parent_ui=None)
    monkeypatch.setattr(
        widget.system_widget,
        "load_configuration",
        lambda configuration_name=None: arctis_config,
    )
    widget.system_widget.connect_to_microscope()

    experiment = Experiment(path=tmp_path, name="agent-verbs")
    experiment.task_protocol = AutoLamellaTaskProtocol()
    experiment.task_protocol.workflow_config.tasks.append(
        AutoLamellaTaskDescription(name=TASK_NAME, required=True)
    )
    experiment.grid_protocol.add(BeamOverviewGridTaskConfig(task_name="overview_sem"))
    experiment.add_grid(GridRecord(name="grid-aspen"))
    experiment.add_new_lamella(MicroscopeState(), EventedDict())
    widget.experiment = experiment

    yield widget
    widget._task_worker_thread = None
    widget.experiment = None
    if widget.microscope is not None:
        widget.microscope.disconnect()
    widget.close()
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
        app_context=AutoLamellaAppContext(ui, event_buffer=recorder.buffer),
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


def _run_that_announces_itself(ui, recorder, seen):
    """A start that does what the real one does first: start the run's worker
    from the GUI thread, and the worker reports the run started."""

    def start(*_args, **_kwargs):
        seen.append(current_actor())
        worker = FunctionWorker(lambda: recorder.buffer.append("workflow_started", {}))
        worker.start()
        worker.join(5)
        ui._task_worker_thread = worker  # not alive: the next start may go

    return start


def test_a_workflow_the_agent_starts_is_the_agent_s(ui, recorder, qapp):
    seen = []
    ui._start_run_workflow_thread = _run_that_announces_itself(ui, recorder, seen)
    with _client(ui, recorder) as client:
        started = _post(
            qapp, client, "/app/workflow/start", {"task_names": [TASK_NAME]}
        ).json()
    assert started["started"] is False  # the stand-in worker has already ended
    assert seen == ["agent"]
    assert _actors(recorder, "workflow_started") == ["agent"]


def test_a_grid_run_the_agent_starts_is_the_agent_s(ui, recorder, qapp):
    seen = []
    ui._start_run_grid_workflow_thread = _run_that_announces_itself(ui, recorder, seen)
    with _client(ui, recorder) as client:
        _post(
            qapp, client, "/app/workflow/grids/start", {"task_names": ["overview_sem"]}
        )
    assert seen == ["agent"]
    assert _actors(recorder, "workflow_started") == ["agent"]


def test_a_workflow_the_operator_starts_is_still_the_operator_s(ui, recorder, qapp):
    seen = []
    ui._start_run_workflow_thread = _run_that_announces_itself(ui, recorder, seen)
    ui._start_workflow_for_agent([TASK_NAME], None)  # unmarked, as a click is
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
    assert _actors(recorder, "agent_note") == ["agent"]
    assert _actors(recorder, "workflow_changed") == ["agent"]
