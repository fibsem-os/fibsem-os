"""The app router end to end: a real AgentContext over a real experiment,
mounted into the server, reached over HTTP and through MCP tools."""

import asyncio
import os

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from psygnal.containers import EventedDict  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.applications.autolamella.server import AgentContext  # noqa: E402
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.structures import MicroscopeState  # noqa: E402

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class Host:
    experiment = None
    microscope = None
    _task_manager = None
    is_workflow_running = False


@pytest.fixture(scope="module")
def microscope():
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    return microscope


@pytest.fixture
def host(tmp_path, microscope):
    host = Host()
    exp = Experiment(path=tmp_path / "exp", name="router-exp")
    exp.task_protocol = AutoLamellaTaskProtocol()
    (tmp_path / "exp").mkdir(parents=True, exist_ok=True)
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    host.experiment = exp
    host.microscope = microscope
    return host


@pytest.fixture
def event_buffer():
    from fibsem.applications.autolamella.server.events import EventBuffer

    return EventBuffer()


@pytest.fixture
def client(microscope, host, event_buffer):
    app = build_server(
        microscope,
        app_context=AgentContext(host, event_buffer=event_buffer),
        auth=AuthConfig(token=TOKEN),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def test_capabilities_reports_the_app_router(client):
    body = client.get("/capabilities", headers=AUTH).json()
    assert body["routers"] == {"microscope": True, "app": True}


def test_without_app_context_the_routes_do_not_exist(microscope):
    app = build_server(microscope, auth=AuthConfig(token=TOKEN))
    with TestClient(app, raise_server_exceptions=False) as bare:
        assert bare.get("/capabilities", headers=AUTH).json()["routers"]["app"] is False
        assert bare.get("/app/status", headers=AUTH).status_code == 404


def test_app_routes_require_a_token(client):
    assert client.get("/app/status").status_code == 401


def test_status_and_queue_over_http(client):
    status = client.get("/app/status", headers=AUTH).json()
    assert status["experiment"]["name"] == "router-exp"
    assert status["workflow"]["running"] is False
    queue = client.get("/app/queue", headers=AUTH).json()
    assert queue["available"] is False  # no run yet


def test_task_outputs_round_trip(client, host):
    name = host.experiment.positions[0].name
    payload = client.get(f"/app/task_outputs/{name}", headers=AUTH).json()
    assert payload["available"] is True
    assert payload["item_name"] == name
    missing = client.get("/app/task_outputs/nope", headers=AUTH).json()
    assert missing["available"] is False


def test_output_images_serve_jpeg_and_refuse_unknown_names(client, host):
    import numpy as np
    import tifffile

    from fibsem.applications.autolamella.structures import AutoLamellaTaskState

    lamella = host.experiment.positions[0]
    os.makedirs(lamella.path, exist_ok=True)
    fname = "ref_Rough Milling_final_res_01_ib.tif"
    tifffile.imwrite(
        os.path.join(lamella.path, fname),
        (np.random.rand(64, 96) * 65535).astype(np.uint16),
    )
    lamella.task_history.append(
        AutoLamellaTaskState(name="Rough Milling", outputs={"final_fib": [fname]})
    )

    served = client.get(f"/app/items/{lamella.name}/outputs/{fname}", headers=AUTH)
    assert served.status_code == 200
    assert served.headers["content-type"] == "image/jpeg"
    assert served.content[:2] == b"\xff\xd8"  # JPEG magic

    # An unlisted name is refused with the valid ones — never resolved as a path.
    refused = client.get(
        f"/app/items/{lamella.name}/outputs/../../etc/passwd", headers=AUTH
    )
    assert refused.status_code in (404, 422)
    unknown = client.get(f"/app/items/{lamella.name}/outputs/nope.tif", headers=AUTH)
    assert unknown.status_code == 404
    assert unknown.json()["detail"]["filenames"] == [fname]


def test_task_config_reads_over_http(client, host):
    from fibsem.applications.autolamella.workflows.tasks.rough import (
        MillRoughTaskConfig,
    )

    config = MillRoughTaskConfig(task_name="Rough Milling")
    host.experiment.task_protocol.task_config["Rough Milling"] = config
    item = host.experiment.positions[0]
    item.task_config["Rough Milling"] = config

    doc = client.get("/app/protocol/task_config/Rough Milling", headers=AUTH).json()
    assert doc["available"] is True and doc["level"] == "protocol"
    assert "version" in doc and "config" in doc

    item_doc = client.get(
        f"/app/items/{item.name}/task_config/Rough Milling", headers=AUTH
    ).json()
    assert item_doc["level"] == "item"
    assert item_doc["version"] == doc["version"]

    unknown = client.get("/app/protocol/task_config/Nope", headers=AUTH).json()
    assert "task_names" in unknown


def test_summaries_and_protocol_over_http(client):
    for path in ("/app/experiment_summary", "/app/task_history", "/app/protocol"):
        body = client.get(path, headers=AUTH).json()
        assert body["available"] is True, path
    assert client.get("/app/run_summary", headers=AUTH).json()["available"] is False


def test_sidecar_grows_the_app_tools_from_capabilities(client):
    pytest.importorskip("mcp")
    from fibsem.mcp.sidecar import build_sidecar
    from fibsem.server.catalog import CATALOG

    client.headers["Authorization"] = f"Bearer {TOKEN}"
    capabilities = client.get("/capabilities").json()
    sidecar = build_sidecar(client, capabilities)
    listed = asyncio.run(sidecar.list_tools())
    names = {t.name for t in getattr(listed, "tools", listed)}
    # Control-scope app tools (answer_prompt) need arming; this server is
    # read-only, so only the read-scope app tools must appear.
    assert {t.name for t in CATALOG if t.router == "app" and t.scope == "read"} <= names
    assert "answer_prompt" not in names

    result = asyncio.run(sidecar.call_tool("get_app_status", {}))
    if isinstance(result, tuple):
        result = result[0]
    contents = list(getattr(result, "content", result))
    text = "".join(getattr(c, "text", "") for c in contents)
    assert "router-exp" in text


def test_events_long_poll_over_http(client, event_buffer):
    empty = client.get("/app/events?since=0&timeout=0", headers=AUTH).json()
    assert empty["available"] is True
    assert empty["events"] == []

    event_buffer.append("milling_progress", {"stage_name": "Rough Mill 01"})
    body = client.get("/app/events?since=0", headers=AUTH).json()
    assert body["latest_seq"] == 1
    assert body["events"][0]["kind"] == "milling_progress"

    caught_up = client.get("/app/events?since=1&timeout=0", headers=AUTH).json()
    assert caught_up["events"] == []


def test_events_unavailable_without_a_buffer(microscope, host):
    app = build_server(
        microscope, app_context=AgentContext(host), auth=AuthConfig(token=TOKEN)
    )
    with TestClient(app, raise_server_exceptions=False) as bare:
        body = bare.get("/app/events", headers=AUTH).json()
    assert body["available"] is False


def test_task_schedule_verb_sets_clears_and_refuses(microscope, host, event_buffer):
    from fibsem.applications.autolamella.structures import (
        AutoLamellaTaskDescription,
    )
    from fibsem.applications.autolamella.workflows.tasks.rough import (
        MillRoughTaskConfig,
    )

    # A real workflow entry to schedule.
    protocol = host.experiment.task_protocol
    protocol.task_config["Rough Milling"] = MillRoughTaskConfig(
        task_name="Rough Milling"
    )
    protocol.workflow_config.tasks.append(
        AutoLamellaTaskDescription(name="Rough Milling", supervise=True, required=True)
    )

    armed = build_server(
        microscope,
        app_context=AgentContext(host, event_buffer=event_buffer),
        auth=AuthConfig.generate(arm_configure=True, token=TOKEN),
    )
    with TestClient(armed, raise_server_exceptions=False) as client:
        when = "2026-09-04T06:00:00"
        resp = client.post(
            "/app/workflow/schedule",
            headers=AUTH,
            json={"task_name": "Rough Milling", "scheduled_at": when},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["applied"] is True and body["saved"] is True
        assert body["scheduled_at"] == when
        # The live protocol shows it, and protocol.yaml has it.
        shown = client.get("/app/protocol", headers=AUTH).json()["tasks"]
        assert (
            next(t for t in shown if t["name"] == "Rough Milling")["scheduled_at"]
            == when
        )
        events = client.get("/app/events?since=0", headers=AUTH).json()["events"]
        assert [e for e in events if e["kind"] == "workflow_changed"]

        cleared = client.post(
            "/app/workflow/schedule",
            headers=AUTH,
            json={"task_name": "Rough Milling", "scheduled_at": None},
        )
        assert cleared.json()["scheduled_at"] is None

        bad = client.post(
            "/app/workflow/schedule",
            headers=AUTH,
            json={"task_name": "Rough Milling", "scheduled_at": "6am tomorrow"},
        )
        assert bad.status_code == 422
        assert bad.json()["detail"]["error_type"] == "invalid_value"

        unknown = client.post(
            "/app/workflow/schedule",
            headers=AUTH,
            json={"task_name": "Nope", "scheduled_at": when},
        )
        assert unknown.status_code == 404
        assert "Rough Milling" in unknown.json()["detail"]["task_names"]


def test_task_schedule_needs_the_configure_scope(client):
    resp = client.post(
        "/app/workflow/schedule",
        headers=AUTH,
        json={"task_name": "X", "scheduled_at": None},
    )
    assert resp.status_code in (403, 404)  # unarmed configure scope
    assert resp.status_code == 403


def test_agent_notes_land_on_the_record(microscope, host, event_buffer):
    armed = build_server(
        microscope,
        app_context=AgentContext(host, event_buffer=event_buffer),
        auth=AuthConfig.generate(arm_control=True, token=TOKEN),
    )
    with TestClient(armed, raise_server_exceptions=False) as client:
        item = host.experiment.positions[0].name
        ok = client.post(
            "/app/agent/notes",
            headers=AUTH,
            json={"text": "curtaining on the face, accepted anyway", "item_name": item},
        )
        assert ok.status_code == 200 and ok.json()["recorded"] is True
        events = client.get("/app/events?since=0", headers=AUTH).json()["events"]
        note = [e for e in events if e["kind"] == "agent_note"][-1]
        assert note["payload"]["item_name"] == item

        unknown = client.post(
            "/app/agent/notes",
            headers=AUTH,
            json={"text": "x", "item_name": "nope"},
        )
        assert unknown.status_code == 404
        empty = client.post("/app/agent/notes", headers=AUTH, json={"text": "   "})
        assert empty.status_code == 422


def test_agent_notes_need_the_control_scope(client):
    resp = client.post("/app/agent/notes", headers=AUTH, json={"text": "x"})
    assert resp.status_code == 403


# --- grids: the screening read model (FIB-876) -------------------------------


@pytest.fixture
def grid_with_overview(host, microscope):
    """A grid whose history recorded a real Demo SEM overview, one lamella
    linked to it posed at the overview's centre and one posed far outside."""
    from fibsem.applications.autolamella.structures import (
        AutoLamellaTaskState,
        AutoLamellaTaskStatus,
        GridRecord,
    )
    from fibsem.structures import (
        BeamType,
        FibsemStagePosition,
        ImageSettings,
        MicroscopeState,
    )

    experiment = host.experiment
    grid = experiment.add_grid(GridRecord(name="grid-aspen"))
    outdir = experiment.grid_path(grid) / "overview_sem"
    outdir.mkdir(parents=True)
    image = microscope.acquire_image(
        ImageSettings(resolution=(128, 128), hfw=200e-6, beam_type=BeamType.ELECTRON)
    )
    image.save(str(outdir / "overview.tif"))
    grid.task_history.append(
        AutoLamellaTaskState(
            name="overview_sem",
            status=AutoLamellaTaskStatus.Completed,
            outputs={"overview_sem": ["overview_sem/overview.tif"]},
        )
    )
    centre = image.metadata.microscope_state.stage_position
    inside = experiment.positions[0]
    inside.grid_id = grid.id
    inside.milling_pose = MicroscopeState(stage_position=centre)
    experiment.add_new_lamella(MicroscopeState(), EventedDict())
    outside = experiment.positions[-1]
    outside.grid_id = grid.id
    outside.milling_pose = MicroscopeState(
        stage_position=FibsemStagePosition(
            x=centre.x + 5e-3, y=centre.y, z=centre.z, r=centre.r, t=centre.t
        )
    )
    return grid


def test_grids_list_and_detail_over_http(client, host, grid_with_overview):
    listed = client.get("/app/grids", headers=AUTH).json()
    assert listed["available"] is True
    (summary,) = listed["items"]
    assert summary["name"] == "grid-aspen"
    assert summary["quality"] == "UNASSESSED"
    assert summary["completed_tasks"] == ["overview_sem"]
    assert summary["last_completed_task"] == "overview_sem"
    assert summary["num_items"] == 2
    assert summary["overviews"] == {"overview_sem": "overview.tif"}

    detail = client.get("/app/grids/grid-aspen", headers=AUTH).json()
    assert detail["tasks"] == [
        {
            "name": "overview_sem",
            "status": "Completed",
            "files": {"overview_sem": ["overview.tif"]},
        }
    ]
    assert sorted(detail["items"]) == sorted(p.name for p in host.experiment.positions)

    missing = client.get("/app/grids/nope", headers=AUTH).json()
    assert missing["available"] is False
    assert missing["grid_names"] == ["grid-aspen"]


def test_grid_overview_serves_jpeg_and_refuses_unlisted_names(
    client, grid_with_overview
):
    served = client.get("/app/grids/grid-aspen/outputs/overview.tif", headers=AUTH)
    assert served.status_code == 200
    assert served.headers["content-type"] == "image/jpeg"
    assert served.content[:2] == b"\xff\xd8"

    unknown = client.get("/app/grids/grid-aspen/outputs/nope.tif", headers=AUTH)
    assert unknown.status_code == 404
    assert unknown.json()["detail"]["filenames"] == ["overview.tif"]

    no_grid = client.get("/app/grids/nope/outputs/overview.tif", headers=AUTH)
    assert no_grid.status_code == 404
    assert no_grid.json()["detail"]["grid_names"] == ["grid-aspen"]


def test_grid_markers_place_items_in_source_pixels(client, host, grid_with_overview):
    body = client.get(
        "/app/grids/grid-aspen/outputs/overview.tif/markers", headers=AUTH
    ).json()
    assert body["available"] is True
    assert body["linked"] is True
    assert body["image"]["width"] == 128 and body["image"]["height"] == 128
    assert body["image"]["hfw"] == pytest.approx(200e-6)
    assert body["image"]["pixel_size"] == pytest.approx(200e-6 / 128)
    assert body["unplaced"] == []
    by_name = {m["item_name"]: m for m in body["markers"]}
    inside, outside = host.experiment.positions
    # the item posed where the overview was taken sits at the image centre
    assert by_name[inside.name]["x"] == pytest.approx(64.0)
    assert by_name[inside.name]["y"] == pytest.approx(64.0)
    assert by_name[inside.name]["inside"] is True
    assert by_name[inside.name]["is_failure"] is False
    assert by_name[inside.name]["last_completed_task"] is None
    # 5 mm to the right at 1.5625 um/px is well past the 128 px edge
    assert by_name[outside.name]["inside"] is False
    assert by_name[outside.name]["x"] > 128


def test_grid_markers_report_unplaceable_items(client, host, grid_with_overview):
    from fibsem.structures import MicroscopeState

    host.experiment.add_new_lamella(MicroscopeState(), EventedDict())
    poseless = host.experiment.positions[-1]
    poseless.grid_id = grid_with_overview.id
    poseless.milling_pose = MicroscopeState()
    body = client.get(
        "/app/grids/grid-aspen/outputs/overview.tif/markers", headers=AUTH
    ).json()
    # a default MicroscopeState carries a position with no r: unplaceable, with
    # the reprojection's own reason rather than a silent drop
    (unplaced,) = body["unplaced"]
    assert unplaced["item_name"] == poseless.name
    assert "r coordinate" in unplaced["reason"]
    assert len(body["markers"]) == 2

    unknown = client.get(
        "/app/grids/grid-aspen/outputs/nope.tif/markers", headers=AUTH
    ).json()
    assert unknown["markers"] == []
    assert unknown["filenames"] == ["overview.tif"]
