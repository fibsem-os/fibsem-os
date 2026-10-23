"""Every /app route answers what its response model says: each answer on the
Demo decodes, strictly, as the route's declared class."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # TestClient transport

from fastapi.routing import APIRoute  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from psygnal.containers import EventedDict  # noqa: E402

from fibsem.applications.autolamella.experiment_setup import (  # noqa: E402
    default_protocol_path,
)
from fibsem.applications.autolamella.server import AutoLamellaAppContext  # noqa: E402
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.codec import decode  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.structures import MicroscopeState  # noqa: E402
from fibsem.workflows.events import EventBuffer  # noqa: E402
from fibsem.workflows.grids import GridRecord  # noqa: E402
from tests.run_state import stub_session  # noqa: E402

TOKEN = "test-token"


def _client(microscope, experiment, folder):
    host = stub_session()
    host.experiments_folder = str(folder)
    host.experiment = experiment
    host.microscope = microscope
    app = build_server(
        microscope,
        app_context=AutoLamellaAppContext(host, event_buffer=EventBuffer()),
        auth=AuthConfig.generate(level="drive", token=TOKEN),
    )
    client = TestClient(app)
    client.headers["Authorization"] = f"Bearer {TOKEN}"
    return client


@pytest.fixture
def client(tmp_path, microscope, monkeypatch):
    # protocols saved, experiments made and the recents list land here, not in
    # the repository
    from fibsem import config as fibsem_cfg
    from fibsem.applications.autolamella import experiment_setup

    config = tmp_path / "config"
    config.mkdir()
    monkeypatch.setattr(fibsem_cfg, "CONFIG_PATH", str(config))
    monkeypatch.setattr(
        fibsem_cfg, "USER_PREFERENCES_PATH", str(config / "user-preferences.yaml")
    )

    protocols = tmp_path / "protocols"
    protocols.mkdir()
    monkeypatch.setattr(experiment_setup, "protocol_directory", lambda: str(protocols))
    exp = Experiment(path=tmp_path / "exp", name="models-exp")
    exp.task_protocol = AutoLamellaTaskProtocol.load(default_protocol_path())
    (tmp_path / "exp").mkdir(parents=True, exist_ok=True)
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    exp.add_grid(GridRecord(name="grid-aspen"))
    exp.save()
    with _client(microscope, exp, tmp_path) as client:
        yield client


def _models(client):
    """Each /app route's declared answer class, by (method, path)."""
    models = {}
    for included in client.app.routes:
        router = getattr(included, "original_router", None)
        for route in router.routes if router is not None else ():
            if isinstance(route, APIRoute) and route.path.startswith("/app"):
                for method in route.methods:
                    models[method, route.path] = route.response_model
    return models


class _Checked:
    """Calls the app, decoding each 200 answer as its route's class."""

    def __init__(self, client):
        self.client = client
        self.models = _models(client)
        self.answered = set()

    def __call__(self, method, path, route=None, **kwargs):
        response = self.client.request(method, path, **kwargs)
        assert response.status_code == 200, (method, path, response.text)
        model = self.models[method, route or path]
        assert model is not None, f"{method} {route or path} declares no answer"
        decoded = decode(model, response.json(), strict=True)
        self.answered.add((method, route or path))
        return decoded


def test_the_app_reads_decode_as_their_declared_class(client):
    call = _Checked(client)
    item = call("GET", "/app/experiment").items[0].item_name
    task = call("GET", "/app/experiment/protocol").tasks[0].name
    call("GET", "/app/workflow/status")
    call("GET", "/app/workflow/queue")
    call("GET", "/app/workflow/summary")
    call("GET", "/app/experiment/task_history")
    call("GET", "/app/log")
    call(
        "GET",
        f"/app/experiment/protocol/task_config/{task}",
        "/app/experiment/protocol/task_config/{task_name}",
    )
    call(
        "GET",
        f"/app/experiment/items/{item}/task_config/{task}",
        "/app/experiment/items/{item_name}/task_config/{task_name}",
    )
    call(
        "GET",
        f"/app/experiment/items/{item}/history",
        "/app/experiment/items/{item_name}/history",
    )
    call("GET", f"/app/experiment/items/{item}", "/app/experiment/items/{item_name}")
    call(
        "GET",
        f"/app/experiment/items/{item}/outputs",
        "/app/experiment/items/{item_name}/outputs",
    )
    call("GET", "/app/experiment/protocols")
    call("GET", "/app/experiment/recent")
    call("GET", "/app/experiment/all")
    call("GET", "/app/experiment/files")
    for path in ("experiment.yaml", item):
        call(
            "GET", f"/app/experiment/files/{path}", "/app/experiment/files/{path:path}"
        )
    call("GET", "/app/workflow/proposals")
    call("GET", "/app/events")
    call("GET", "/app/events/history")

    grids = call("GET", "/app/experiment/grids")
    grid = grids.items[0].name
    call("GET", f"/app/experiment/grids/{grid}", "/app/experiment/grids/{grid_name}")
    call("GET", "/app/experiment/grids/nowhere", "/app/experiment/grids/{grid_name}")
    call(
        "GET",
        f"/app/experiment/grids/{grid}/outputs/none.tif/markers",
        "/app/experiment/grids/{grid_name}/outputs/{filename:path}/markers",
    )
    grid_task = call("GET", "/app/experiment/protocol").grid_tasks[0].name
    call(
        "GET",
        f"/app/experiment/grid_protocol/task_config/{grid_task}",
        "/app/experiment/grid_protocol/task_config/{task_name}",
    )
    call("POST", "/app/workflow/grids/plan", json={"task_names": [grid_task]})


def _png() -> str:
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("L", (8, 8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _flipped(config, path=""):
    """A patch flipping *config*'s first flag, by its dotted path."""
    for key, value in config.items():
        if isinstance(value, bool):
            return {path + key: not value}
        if isinstance(value, dict) and (patch := _flipped(value, f"{path}{key}.")):
            return patch
    return None


def test_the_app_edits_decode_as_their_declared_class(client, tmp_path):
    call = _Checked(client)
    item = call("GET", "/app/experiment").items[0].item_name
    protocol = call("GET", "/app/experiment/protocol")
    order = [task.name for task in protocol.tasks]
    task = order[0]
    grid_task = protocol.grid_tasks[0].name

    call("POST", "/app/experiment/notes", json={"text": "a note", "item_name": item})
    call("POST", "/app/experiment/evidence", json={"upload": _png()})

    # the workflow's tasks
    call(
        "POST",
        "/app/experiment/protocol/supervision",
        json={"task_name": task, "supervise": True},
    )
    call(
        "POST", "/app/workflow/schedule", json={"task_name": task, "scheduled_at": None}
    )
    call("POST", "/app/workflow/options", json={"turn_beams_off": False})
    types = client.post(
        "/app/experiment/protocol/tasks", json={"task_type": "?", "task_name": "x"}
    ).json()["detail"]["task_type_names"]
    call(
        "POST",
        "/app/experiment/protocol/tasks",
        json={"task_type": types[0], "task_name": "Extra"},
    )
    call(
        "POST",
        "/app/experiment/protocol/tasks/Extra/requirements",
        "/app/experiment/protocol/tasks/{task_name}/requirements",
        json={"requires": [task]},
    )
    call(
        "POST",
        "/app/experiment/protocol/tasks/reorder",
        json={"order": [*order, "Extra"]},
    )
    call(
        "DELETE",
        "/app/experiment/protocol/tasks/Extra",
        "/app/experiment/protocol/tasks/{task_name}",
    )
    call(
        "POST", "/app/experiment/protocol/lamella_defaults", json={"use_petname": True}
    )
    call("POST", "/app/experiment/protocol/save", json={"name": "copy"})

    # task configs, each patch flipping a flag and each read its version
    route = "/app/experiment/protocol/task_config/{task_name}"
    read = call("GET", f"/app/experiment/protocol/task_config/{task}", route)
    call(
        "POST",
        f"/app/experiment/protocol/task_config/{task}",
        route,
        json={"patch": _flipped(read.config), "version": read.version},
    )
    milled = "Rough Milling"
    read = call("GET", f"/app/experiment/protocol/task_config/{milled}", route)
    key, milling = next(iter(read.config["milling"].items()))
    stages = [stage["name"] for stage in milling["stages"]]
    call(
        "POST",
        f"/app/experiment/protocol/task_config/{milled}/stages/reorder",
        route + "/stages/reorder",
        json={"milling_key": key, "order": stages, "version": read.version},
    )
    # the item's own copies, from the protocol's
    call(
        "POST",
        f"/app/experiment/items/{item}/apply_protocol",
        "/app/experiment/items/{item_name}/apply_protocol",
        json={"task_names": [task, milled]},
    )
    route = "/app/experiment/items/{item_name}/task_config/{task_name}"
    read = call("GET", f"/app/experiment/items/{item}/task_config/{task}", route)
    call(
        "POST",
        f"/app/experiment/items/{item}/task_config/{task}",
        route,
        json={"patch": _flipped(read.config), "version": read.version},
    )
    read = call("GET", f"/app/experiment/items/{item}/task_config/{milled}", route)
    call(
        "POST",
        f"/app/experiment/items/{item}/task_config/{milled}/stages/reorder",
        route + "/stages/reorder",
        json={"milling_key": key, "order": stages, "version": read.version},
    )
    route = "/app/experiment/grid_protocol/task_config/{task_name}"
    read = call("GET", f"/app/experiment/grid_protocol/task_config/{grid_task}", route)
    call(
        "POST",
        f"/app/experiment/grid_protocol/task_config/{grid_task}",
        route,
        json={"patch": _flipped(read.config), "version": read.version},
    )
    grid_order = [t.name for t in protocol.grid_tasks]
    call(
        "POST",
        "/app/experiment/grid_protocol/tasks/reorder",
        json={"order": grid_order},
    )
    call(
        "POST",
        f"/app/experiment/grid_protocol/tasks/{grid_task}/requirements",
        "/app/experiment/grid_protocol/tasks/{task_name}/requirements",
        json={"requires": []},
    )

    # grids
    grid = call("GET", "/app/experiment/grids").items[0].name
    base = "/app/experiment/grids/{grid_name}"
    call(
        "POST",
        f"/app/experiment/grids/{grid}/quality",
        base + "/quality",
        json={"verdict": "GOOD", "reason": "clean"},
    )
    call(
        "POST",
        f"/app/experiment/grids/{grid}/note",
        base + "/note",
        json={"note": "a note"},
    )
    call(
        "POST",
        f"/app/experiment/grids/{grid}/rename",
        base + "/rename",
        json={"name": "grid-birch"},
    )
    call("DELETE", "/app/experiment/grids/grid-birch", base)

    # items
    added = call("POST", "/app/experiment/items", json={"name": "extra"}).item
    assert added.item_name == "extra"
    base = "/app/experiment/items/{item_name}"
    call(
        "POST",
        "/app/experiment/items/extra/move_to",
        base + "/move_to",
        json={"pose": "MILLING"},
    )
    call(
        "POST", "/app/experiment/items/extra/poses/MILLING", base + "/poses/{pose_name}"
    )
    position = added.poses["MILLING"]
    call(
        "POST",
        "/app/experiment/items/extra/position",
        base + "/position",
        json={"stage_position": position.to_dict()},
    )
    detail = call("GET", "/app/experiment/items/extra", base)
    call(
        "POST",
        "/app/experiment/items/extra",
        base,
        json={"patch": {"description": "moved"}, "version": detail.version},
    )
    call("DELETE", "/app/experiment/items/extra", base)

    # the run
    call("POST", "/app/workflow/stop", json={"reason": "a test"})
    call(
        "POST",
        "/app/workflow/stop_task",
        json={"item_name": item, "task_name": task, "reason": "a test"},
    )
    call("POST", "/app/workflow/grids/plan", json={"task_names": [grid_task]})

    # experiments and protocols
    name = call("GET", "/app/experiment/protocols").protocols[0].name
    call("POST", "/app/experiment/protocol/load", json={"protocol": name})
    created = call("POST", "/app/experiment/create", json={"name": "models-new"})
    assert created.experiment.available

    # what is not done without a run says why, in a 200
    call("POST", "/app/workflow/start", json={"task_names": ["Nothing"]})
    call(
        "POST",
        "/app/workflow/grids/start",
        json={"task_names": [grid_task], "grid_names": ["none"]},
    )
    call(
        "POST",
        "/app/workflow/queue/requeue",
        json={"item_name": item, "task_name": task},
    )
    call(
        "POST",
        "/app/workflow/queue/edit",
        json={"item_name": item, "task_name": task, "action": "remove"},
    )
    call(
        "POST",
        "/app/workflow/queue/grids",
        json={"task_names": [grid_task], "grid_names": ["none"]},
    )

    call(
        "POST",
        "/app/experiment/load",
        json={"path": str(tmp_path / "models-new" / "experiment.yaml")},
    )


def test_every_app_route_declares_its_answer(client):
    """A route answers a declared class, or an image."""
    images = {
        "/app/experiment/items/{item_name}/outputs/{filename}",
        "/app/experiment/grids/{grid_name}/outputs/{filename}",
        "/app/experiment/planes/{path:path}",
    }
    untyped = [
        (method, path)
        for (method, path), model in _models(client).items()
        if model is None and path not in images and not path.startswith("/app/agent")
    ]
    assert untyped == []


def test_each_app_tool_returns_its_route_s_class(client):
    """`ToolSpec.returns` names what the tool's route answers, as the app
    writes it (`answer_class`)."""
    import re

    from fibsem.ai.agent.catalog import CATALOG
    from fibsem.server.responses import answer_class

    def shape(path):
        return re.sub(r"\{[^}]*\}", "{}", path)

    models = {(m, shape(p)): model for (m, p), model in _models(client).items()}
    types = client.app.state.app_context.answer_types
    for spec in CATALOG:
        if spec.router != "app":
            continue
        model = models[spec.method, shape(spec.path)]
        expected = answer_class(spec.answers, types) if spec.returns else None
        assert model == expected, spec.name
