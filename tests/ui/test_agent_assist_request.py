"""Assist end to end (FIB-1283): the agent asks over HTTP, the operator approves
on the GUI thread, and the approved start runs through the same marshalled path
as the Run button. Three threads, like production: the agent's HTTP from the
test thread, the GUI spinning, and the request's own worker."""

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

from fibsem.applications.autolamella.server import AutoLamellaAppContext  # noqa: E402
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.server.auth import AgentLevel  # noqa: E402
from fibsem.structures import MicroscopeState  # noqa: E402
from fibsem.workflows.attention import Attention

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
START = {"tool": "workflow_start", "arguments": {"task_names": ["Mill Fiducial"]}}


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

    experiment = Experiment(path=tmp_path, name="assist-start")
    experiment.task_protocol = AutoLamellaTaskProtocol()
    experiment.task_protocol.workflow_config.tasks.append(
        AutoLamellaTaskDescription(
            name="Mill Fiducial", attention=Attention.supervised, required=True
        )
    )
    experiment.add_new_lamella(MicroscopeState(), EventedDict())
    widget.experiment = experiment

    yield widget
    widget.experiment = None
    if widget.microscope is not None:
        widget.microscope.disconnect()
    widget.close()
    widget.deleteLater()
    qapp.processEvents()


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


class _AliveThread:
    @staticmethod
    def is_alive():
        return True


def test_an_approved_request_starts_the_run_from_the_gui_thread(ui, qapp):
    calls = []

    def fake_start(task_names, item_names):
        calls.append((task_names, item_names))
        ui._task_worker_thread = _AliveThread()

    ui._start_run_workflow_thread = fake_start

    auth = AuthConfig(token=TOKEN)
    auth.set_level(AgentLevel.ASSIST)
    app = build_server(ui.microscope, app_context=AutoLamellaAppContext(ui), auth=auth)
    with TestClient(app, raise_server_exceptions=False) as client:
        # Asked to act directly, Assist says to ask.
        refused = _post(qapp, client, "/app/workflow/start", START["arguments"])
        assert refused.status_code == 403
        assert refused.json()["detail"]["error_type"] == "ask_first"
        assert calls == []

        filed = _post(
            qapp,
            client,
            "/app/agent/requests",
            {"actions": [START], "reason": "the fiducial is next"},
        )
        assert filed.status_code == 202, filed.text
        request_id = filed.json()["request_id"]
        assert calls == []

        # The operator's Approve, on the GUI thread: it must not block it, since
        # the start it runs is marshalled back onto this thread.
        store = app.state.requests
        store.approve(request_id)
        _spin_until(qapp, lambda: store.get(request_id).status != "applying")

        done = store.get(request_id)
        assert done.status == "applied", done.results
        assert done.results[0]["result"] == {"available": True, "started": True}
        assert calls == [(["Mill Fiducial"], [ui.experiment.positions[0].name])]

        read = client.get(f"/app/agent/requests/{request_id}", headers=AUTH)
        assert read.json()["status"] == "applied"
