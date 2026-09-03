"""The grid run verbs end to end: a remote start request validated and started on
the GUI thread, sharing the lamella run's worker slot, and the plan that goes
before it. Same three-thread shape as the lamella start test."""

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

from fibsem.applications.autolamella.server import AgentContext  # noqa: E402
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    Experiment,
    GridRecord,
)
from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI  # noqa: E402
from fibsem.applications.autolamella.workflows.tasks.grid.imaging import (  # noqa: E402
    BeamOverviewGridTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.grid.manager import (  # noqa: E402
    LOAD_ENTRY_NAME,
)
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.structures import MicroscopeState  # noqa: E402

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


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

    experiment = Experiment(path=tmp_path, name="grid-start")
    experiment.task_protocol = AutoLamellaTaskProtocol()
    experiment.grid_protocol.add(BeamOverviewGridTaskConfig(task_name="overview_sem"))
    experiment.add_grid(GridRecord(name="grid-aspen"))
    experiment.add_grid(GridRecord(name="grid-birch"))
    experiment.add_new_lamella(MicroscopeState(), EventedDict())
    widget.experiment = experiment

    yield widget
    widget.experiment = None
    if widget.microscope is not None:
        widget.microscope.disconnect()
    widget.close()
    widget.deleteLater()
    qapp.processEvents()


def _client(ui, arm_control):
    app = build_server(
        ui.microscope,
        app_context=AgentContext(ui),
        auth=AuthConfig.generate(arm_control=arm_control, token=TOKEN),
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


class _AliveThread:
    @staticmethod
    def is_alive():
        return True


def test_grid_start_validates_and_starts_on_the_gui_thread(ui, qapp):
    calls = []

    def fake_start(task_names, grid_names, inventory_first):
        calls.append((task_names, grid_names, inventory_first))
        ui._task_worker_thread = _AliveThread()

    ui._start_run_grid_workflow_thread = fake_start

    with _client(ui, arm_control=True) as client:
        # The protocol read names the grid tasks a run can use.
        protocol = client.get("/app/protocol", headers=AUTH).json()
        assert protocol["grid_tasks"] == [
            {"name": "overview_sem", "type": "BEAM_OVERVIEW_GRID"}
        ]

        refused = _post(
            qapp, client, "/app/workflow/grids/start", {"task_names": ["No Such"]}
        ).json()
        assert refused["started"] is False
        assert refused["task_names"] == ["overview_sem"]

        refused = _post(
            qapp,
            client,
            "/app/workflow/grids/start",
            {"task_names": ["overview_sem"], "grid_names": ["grid-oak"]},
        ).json()
        assert refused["started"] is False
        assert refused["grid_names"] == ["grid-aspen", "grid-birch"]

        refused = _post(
            qapp,
            client,
            "/app/workflow/grids/start",
            {
                "task_names": ["overview_sem"],
                "grid_names": ["grid-aspen"],
                "screen_all": True,
            },
        ).json()
        assert refused["started"] is False
        assert "screen_all" in refused["reason"]
        assert calls == []

        # Omitted grids means every recorded grid; the plan comes back with it.
        started = _post(
            qapp, client, "/app/workflow/grids/start", {"task_names": ["overview_sem"]}
        ).json()
        assert started["started"] is True
        assert started["screen_all"] is False
        assert started["plan"] == [
            {"grid": "grid-aspen", "step": LOAD_ENTRY_NAME},
            {"grid": "grid-aspen", "step": "overview_sem"},
            {"grid": "grid-birch", "step": LOAD_ENTRY_NAME},
            {"grid": "grid-birch", "step": "overview_sem"},
        ]
        assert calls == [(["overview_sem"], ["grid-aspen", "grid-birch"], False)]

        # One worker slot: while it is "running", neither kind of run may start.
        again = _post(
            qapp, client, "/app/workflow/grids/start", {"task_names": ["overview_sem"]}
        ).json()
        assert again["started"] is False
        assert "already running" in again["reason"]
        lamella = _post(
            qapp, client, "/app/workflow/start", {"task_names": ["overview_sem"]}
        ).json()
        assert lamella["started"] is False
        assert "already running" in lamella["reason"]
        ui._task_worker_thread = None

        # Screen all grids: inventory first, no grid list, no plan to promise.
        screened = _post(
            qapp,
            client,
            "/app/workflow/grids/start",
            {"task_names": ["overview_sem"], "screen_all": True},
        ).json()
        assert screened == {"available": True, "started": True, "screen_all": True}
        assert calls[-1] == (["overview_sem"], None, True)
        ui._task_worker_thread = None


def test_grid_start_requires_control_and_a_valid_body(ui, qapp):
    with _client(ui, arm_control=False) as client:
        refused = client.post(
            "/app/workflow/grids/start", headers=AUTH, json={"task_names": ["x"]}
        )
        assert refused.status_code == 403
        assert refused.json()["detail"]["scope"] == "control"
        # The plan is a read: it needs no arming.
        plan = client.post(
            "/app/workflow/grids/plan",
            headers=AUTH,
            json={"task_names": ["overview_sem"]},
        )
        assert plan.status_code == 200
        assert plan.json()["valid"] is True
    with _client(ui, arm_control=True) as client:
        for body in (
            {},
            {"task_names": []},
            {"task_names": ["x"], "screen_all": "yes"},
        ):
            rejected = client.post("/app/workflow/grids/start", headers=AUTH, json=body)
            assert rejected.status_code == 422, body
            assert rejected.json()["detail"]["error_type"] == "missing_field"
