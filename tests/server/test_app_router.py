"""The app router end to end: a real AutoLamellaAppContext over a real experiment,
mounted into the server, reached over HTTP and through MCP tools."""

import asyncio
import json
import os
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from psygnal.containers import EventedDict  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.applications.autolamella.server import AutoLamellaAppContext  # noqa: E402
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.structures import MicroscopeState  # noqa: E402
from tests.protocols import add_planned, planned  # noqa: E402
from tests.run_state import run_working_on, stub_session

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(scope="module")
def microscope():
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    return microscope


@pytest.fixture
def host(tmp_path, microscope):
    host = stub_session()
    exp = Experiment(path=tmp_path / "exp", name="router-exp")
    exp.task_protocol = AutoLamellaTaskProtocol()
    (tmp_path / "exp").mkdir(parents=True, exist_ok=True)
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    host.experiment = exp
    host.microscope = microscope
    return host


@pytest.fixture
def event_buffer():
    from fibsem.workflows.events import EventBuffer

    return EventBuffer()


@pytest.fixture
def client(microscope, host, event_buffer):
    app = build_server(
        microscope,
        app_context=AutoLamellaAppContext(host, event_buffer=event_buffer),
        auth=AuthConfig(token=TOKEN),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def test_capabilities_reports_the_app_router(client):
    body = client.get("/capabilities", headers=AUTH).json()
    assert body["routers"] == {"microscope": True, "services": True, "app": True}
    assert body["parts"] == ["items", "poses", "grids", "milling_stages", "lamellae"]
    assert body["app_name"] == "AutoLamella"


def test_a_refusal_names_the_app_the_operator_works_in(client):
    """The agent is told where the operator acts: the app's own name."""
    from fibsem.ai.agent.levels import AgentLevel

    client.app.state.auth.set_level(AgentLevel.ASSIST)
    asked = client.post("/app/workflow/stop", headers=AUTH, json={"reason": "x"})
    assert asked.status_code == 403
    assert "in AutoLamella." in asked.json()["detail"]["message"]


class _CoreOnly:
    """An app context with no optional parts."""

    def status(self):
        return {"core": True}


def test_a_part_the_context_lacks_is_not_served(microscope):
    app = build_server(
        microscope, app_context=_CoreOnly(), auth=AuthConfig(token=TOKEN)
    )
    with TestClient(app, raise_server_exceptions=False) as core:
        assert core.get("/capabilities", headers=AUTH).json()["parts"] == []
        assert core.get("/app/workflow/status", headers=AUTH).json() == {"core": True}
        assert core.get("/app/experiment/grids", headers=AUTH).status_code == 404
        assert (
            core.post("/app/experiment/items/a/poses/milling", headers=AUTH).status_code
            == 404
        )


def test_without_app_context_the_routes_do_not_exist(microscope):
    app = build_server(microscope, auth=AuthConfig(token=TOKEN))
    with TestClient(app, raise_server_exceptions=False) as bare:
        assert bare.get("/capabilities", headers=AUTH).json()["routers"]["app"] is False
        assert bare.get("/app/workflow/status", headers=AUTH).status_code == 404


def test_app_routes_require_a_token(client):
    assert client.get("/app/workflow/status").status_code == 401


def test_status_and_queue_over_http(client):
    status = client.get("/app/workflow/status", headers=AUTH).json()
    assert status["experiment"]["name"] == "router-exp"
    assert status["workflow"]["running"] is False
    queue = client.get("/app/workflow/queue", headers=AUTH).json()
    assert queue["available"] is False  # no run yet


def test_an_items_history_comes_through_the_tabs_filter(client, host):
    from fibsem.workflows.status import TaskState, TaskStatus

    lamella = host.experiment.positions[0]
    lamella.task_history.append(
        TaskState(name="Setup", task_id="a", status=TaskStatus.Completed)
    )
    lamella.task_history.append(
        TaskState(name="Rough", task_id="b", status=TaskStatus.Failed)
    )
    url = f"/app/experiment/items/{lamella.name}/history"
    body = client.get(url, headers=AUTH).json()
    assert body["tasks"] == ["Setup", "Rough"] and body["total"] == 2
    assert [r["task_name"] for r in body["runs"]] == ["Setup", "Rough"]
    assert body["filter"]["order"] == "oldest"
    newest = client.get(url, headers=AUTH, params={"order": "newest"}).json()
    assert [r["task_name"] for r in newest["runs"]] == ["Rough", "Setup"]
    failed = client.get(url, headers=AUTH, params={"status": "failed"}).json()
    assert [r["status"] for r in failed["runs"]] == ["Failed"]
    assert failed["total"] == 2


def test_task_outputs_round_trip(client, host):
    name = host.experiment.positions[0].name
    payload = client.get(f"/app/experiment/items/{name}/outputs", headers=AUTH).json()
    assert payload["available"] is True
    assert payload["item_name"] == name
    missing = client.get("/app/experiment/items/nope/outputs", headers=AUTH).json()
    assert missing["available"] is False


def test_output_images_serve_jpeg_and_refuse_unknown_names(client, host):
    import numpy as np
    import tifffile

    from fibsem.workflows.status import TaskState

    lamella = host.experiment.positions[0]
    os.makedirs(lamella.path, exist_ok=True)
    fname = "ref_Rough Milling_final_res_01_ib.tif"
    tifffile.imwrite(
        os.path.join(lamella.path, fname),
        (np.random.rand(64, 96) * 65535).astype(np.uint16),
    )
    lamella.task_history.append(
        TaskState(name="Rough Milling", outputs={"final_fib": [fname]})
    )

    served = client.get(
        f"/app/experiment/items/{lamella.name}/outputs/{fname}", headers=AUTH
    )
    assert served.status_code == 200
    assert served.headers["content-type"] == "image/jpeg"
    assert served.content[:2] == b"\xff\xd8"  # JPEG magic

    # An unlisted name is refused with the valid ones — never resolved as a path.
    refused = client.get(
        f"/app/experiment/items/{lamella.name}/outputs/../../etc/passwd", headers=AUTH
    )
    assert refused.status_code in (404, 422)
    unknown = client.get(
        f"/app/experiment/items/{lamella.name}/outputs/nope.tif", headers=AUTH
    )
    assert unknown.status_code == 404
    assert unknown.json()["detail"]["filenames"] == [fname]


def test_every_listed_output_is_served_and_only_completed_runs_count(client, host):
    # FIB-1330: start images are listed by get_task_outputs, so they are
    # served too; a cancelled run is listed with its status, not as completed.
    import numpy as np
    import tifffile

    from fibsem.workflows.status import TaskState, TaskStatus

    lamella = host.experiment.positions[0]
    os.makedirs(lamella.path, exist_ok=True)
    start = "ref_Rough Milling_start_ib.tif"
    tifffile.imwrite(
        os.path.join(lamella.path, start),
        (np.random.rand(64, 96) * 65535).astype(np.uint16),
    )
    lamella.task_history.append(
        TaskState(
            name="Rough Milling",
            status=TaskStatus.Completed,
            outputs={"start_fib": [start]},
        )
    )
    lamella.task_history.append(
        TaskState(name="Polishing", status=TaskStatus.Cancelled)
    )

    listed = client.get(
        f"/app/experiment/items/{lamella.name}/outputs", headers=AUTH
    ).json()
    assert listed["completed_tasks"] == ["Rough Milling"]
    assert [(t["name"], t["status"]) for t in listed["tasks"]] == [
        ("Rough Milling", "Completed"),
        ("Polishing", "Cancelled"),
    ]
    assert listed["tasks"][0]["files"] == {"start_fib": [start]}
    served = client.get(
        f"/app/experiment/items/{lamella.name}/outputs/{start}", headers=AUTH
    )
    assert served.status_code == 200
    assert served.headers["content-type"] == "image/jpeg"


def test_task_history_is_in_time_order(client, host):
    from datetime import datetime, timedelta

    from fibsem.workflows.status import TaskState

    host.experiment.add_new_lamella(MicroscopeState(), EventedDict())
    first, second = host.experiment.positions[:2]
    t0 = datetime(2026, 10, 12, 9).timestamp()
    # The second item ran first: listing by item would put it last.
    first.task_history.append(TaskState(name="Polishing", start_timestamp=t0 + 60))
    second.task_history.append(TaskState(name="Rough Milling", start_timestamp=t0))

    items = client.get("/app/experiment/task_history", headers=AUTH).json()["items"]
    assert [i["task_name"] for i in items] == ["Rough Milling", "Polishing"]


def test_sidecar_returns_a_saved_output_as_image_content(client, host):
    """The Observer's saved-image gap (FIB-848): the JPEG route reaches the
    agent as MCP image content, with the basename's spaces encoded."""
    pytest.importorskip("mcp")
    import numpy as np
    import tifffile

    from fibsem.ai.agent.mcp.sidecar import build_sidecar
    from fibsem.workflows.status import TaskState

    lamella = host.experiment.positions[0]
    os.makedirs(lamella.path, exist_ok=True)
    fname = "ref_Rough Milling_final_res_01_ib.tif"
    tifffile.imwrite(
        os.path.join(lamella.path, fname),
        (np.random.rand(64, 96) * 65535).astype(np.uint16),
    )
    lamella.task_history.append(
        TaskState(name="Rough Milling", outputs={"final_fib": [fname]})
    )

    client.headers["Authorization"] = f"Bearer {TOKEN}"
    sidecar = build_sidecar(client, client.get("/capabilities").json())

    def call(filename):
        result = asyncio.run(
            sidecar.call_tool(
                "item_get_output_image",
                {"item_name": lamella.name, "filename": filename},
            )
        )
        if isinstance(result, tuple):
            result = result[0]
        return list(getattr(result, "content", result))

    contents = call(fname)
    image = next(c for c in contents if getattr(c, "type", None) == "image")
    assert image.mime_type == "image/jpeg"
    assert fname in "".join(getattr(c, "text", "") for c in contents)

    from mcp.server.mcpserver.exceptions import ToolError

    # A refusal is an error to the MCP client, its text the server's answer.
    with pytest.raises(ToolError) as refused:
        call("nope.tif")
    assert "refused (404)" in str(refused.value)
    assert fname in str(refused.value)  # the valid names come with the refusal


def test_task_config_reads_over_http(client, host):
    from fibsem.applications.autolamella.workflows.tasks.rough import (
        MillRoughTaskConfig,
    )

    config = MillRoughTaskConfig(task_name="Rough Milling")
    host.experiment.task_protocol.task_config["Rough Milling"] = config
    item = host.experiment.positions[0]
    item.task_config["Rough Milling"] = config

    doc = client.get(
        "/app/experiment/protocol/task_config/Rough Milling", headers=AUTH
    ).json()
    assert doc["available"] is True and doc["level"] == "protocol"
    assert "version" in doc and "config" in doc

    item_doc = client.get(
        f"/app/experiment/items/{item.name}/task_config/Rough Milling", headers=AUTH
    ).json()
    assert item_doc["level"] == "item"
    assert item_doc["version"] == doc["version"]

    unknown = client.get(
        "/app/experiment/protocol/task_config/Nope", headers=AUTH
    ).json()
    assert "task_names" in unknown


def test_summaries_and_protocol_over_http(client):
    for path in (
        "/app/experiment",
        "/app/experiment/task_history",
        "/app/experiment/protocol",
    ):
        body = client.get(path, headers=AUTH).json()
        assert body["available"] is True, path
    assert (
        client.get("/app/workflow/summary", headers=AUTH).json()["available"] is False
    )


def test_sidecar_grows_the_app_tools_from_capabilities(client):
    pytest.importorskip("mcp")
    from fibsem.ai.agent.catalog import CATALOG
    from fibsem.ai.agent.mcp.sidecar import build_sidecar

    client.headers["Authorization"] = f"Bearer {TOKEN}"
    capabilities = client.get("/capabilities").json()
    sidecar = build_sidecar(client, capabilities)
    listed = asyncio.run(sidecar.list_tools())
    names = {t.name for t in getattr(listed, "tools", listed)}
    # Every app tool registers whatever the level; the server's 403 is the
    # gate, so the read-only server still lists proposal_decide.
    assert {t.name for t in CATALOG if t.router == "app"} <= names
    assert "Needs the agent at Drive" in next(
        t.description
        for t in getattr(listed, "tools", listed)
        if t.name == "proposal_decide"
    )

    result = asyncio.run(sidecar.call_tool("app_get_status", {}))
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
        microscope,
        app_context=AutoLamellaAppContext(host),
        auth=AuthConfig(token=TOKEN),
    )
    with TestClient(app, raise_server_exceptions=False) as bare:
        body = bare.get("/app/events", headers=AUTH).json()
    assert body["available"] is False


def test_event_history_reads_the_record_before_the_live_buffer(
    client, host, event_buffer
):
    import json
    from pathlib import Path

    from fibsem.workflows.event_recording import EVENTS_FILENAME

    event_buffer.append("task_started", {"task_name": "Polishing"})
    held = event_buffer.events_since(0)["events"][0]
    # The file has the night: an earlier session, this session's records from
    # before the buffer's oldest, the one the buffer still holds, and progress.
    lines = [
        {"session": "night", "seq": 1, "kind": "task_started"},
        {"session": "night", "seq": 2, "kind": "milling_progress"},
        {"session": held.get("session"), "seq": 0, "kind": "agent_note"},
        {**held},
    ]
    path = Path(host.experiment.path) / EVENTS_FILENAME
    path.write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")

    body = client.get("/app/events/history", headers=AUTH).json()
    assert body["available"] is True
    assert [(e["session"], e["seq"]) for e in body["events"]] == [
        ("night", 1),
        (held.get("session"), 0),
    ]
    assert body["start"] == 0

    page = client.get("/app/events/history?limit=1", headers=AUTH).json()
    assert [e["seq"] for e in page["events"]] == [0]
    earlier = client.get(
        f"/app/events/history?limit=1&before={page['start']}", headers=AUTH
    ).json()
    assert [e["seq"] for e in earlier["events"]] == [1]


def test_task_schedule_verb_sets_clears_and_refuses(microscope, host, event_buffer):
    from fibsem.applications.autolamella.workflows.tasks.rough import (
        MillRoughTaskConfig,
    )
    from fibsem.workflows.attention import Attention

    # A real workflow entry to schedule.
    protocol = host.experiment.task_protocol
    protocol.task_config["Rough Milling"] = MillRoughTaskConfig(
        task_name="Rough Milling"
    )
    add_planned(
        protocol,
        planned(name="Rough Milling", attention=Attention.supervised, required=True),
    )

    armed = build_server(
        microscope,
        app_context=AutoLamellaAppContext(host, event_buffer=event_buffer),
        auth=AuthConfig.generate(level="drive", token=TOKEN),
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
        shown = client.get("/app/experiment/protocol", headers=AUTH).json()["tasks"]
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


def test_workflow_options_verb_sets_and_refuses(microscope, host, event_buffer):
    armed = build_server(
        microscope,
        app_context=AutoLamellaAppContext(host, event_buffer=event_buffer),
        auth=AuthConfig.generate(level="drive", token=TOKEN),
    )
    with TestClient(armed, raise_server_exceptions=False) as client:
        resp = client.post(
            "/app/workflow/options", headers=AUTH, json={"show_run_summary": False}
        )
        assert resp.status_code == 200
        assert resp.json()["applied"] is True
        shown = client.get("/app/experiment/protocol", headers=AUTH).json()["options"]
        assert shown["show_run_summary"] is False
        events = client.get("/app/events?since=0", headers=AUTH).json()["events"]
        assert any(
            e["kind"] == "workflow_changed"
            and e["payload"].get("field") == "show_run_summary"
            for e in events
        )

        bad = client.post(
            "/app/workflow/options", headers=AUTH, json={"show_run_summary": "no"}
        )
        assert bad.status_code == 422


def test_workflow_options_need_the_configure_scope(client):
    resp = client.post(
        "/app/workflow/options", headers=AUTH, json={"show_run_summary": False}
    )
    assert resp.status_code == 403


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
        app_context=AutoLamellaAppContext(host, event_buffer=event_buffer),
        auth=AuthConfig.generate(level="drive", token=TOKEN),
    )
    with TestClient(armed, raise_server_exceptions=False) as client:
        item = host.experiment.positions[0].name
        ok = client.post(
            "/app/experiment/notes",
            headers=AUTH,
            json={"text": "curtaining on the face, accepted anyway", "item_name": item},
        )
        assert ok.status_code == 200 and ok.json()["recorded"] is True
        events = client.get("/app/events?since=0", headers=AUTH).json()["events"]
        note = [e for e in events if e["kind"] == "note"][-1]
        assert note["payload"]["item_name"] == item
        assert note["payload"]["item_id"] == host.experiment.positions[0].id

        unknown = client.post(
            "/app/experiment/notes",
            headers=AUTH,
            json={"text": "x", "item_name": "nope"},
        )
        assert unknown.status_code == 404
        empty = client.post("/app/experiment/notes", headers=AUTH, json={"text": "   "})
        assert empty.status_code == 422

        # Its answer to the operator's message names the message (FIB-1423).
        reply = client.post(
            "/app/experiment/notes",
            headers=AUTH,
            json={"text": "skipping 03", "reply_to": "ab12"},
        )
        assert reply.status_code == 200
        events = client.get("/app/events?since=0", headers=AUTH).json()["events"]
        note = [e for e in events if e["kind"] == "note"][-1]
        assert note["payload"]["reply_to"] == "ab12"
        bad = client.post(
            "/app/experiment/notes", headers=AUTH, json={"text": "x", "reply_to": 3}
        )
        assert bad.status_code == 422


def test_agent_notes_need_only_the_read_scope(client):
    # A watching agent notes what it sees (FIB-1035); a note changes nothing.
    resp = client.post("/app/experiment/notes", headers=AUTH, json={"text": "x"})
    assert resp.status_code == 200


# --- grids: the screening read model (FIB-876) -------------------------------


@pytest.fixture
def grid_with_overview(host, microscope):
    """A grid whose history recorded a real Demo SEM overview, one lamella
    linked to it posed at the overview's centre and one posed far outside."""
    from fibsem.structures import (
        BeamType,
        FibsemStagePosition,
        ImageSettings,
        MicroscopeState,
    )
    from fibsem.workflows.grids import GridRecord
    from fibsem.workflows.status import TaskState, TaskStatus

    experiment = host.experiment
    grid = experiment.add_grid(GridRecord(name="grid-aspen"))
    outdir = experiment.grid_path(grid) / "overview_sem"
    outdir.mkdir(parents=True)
    image = microscope.acquire_image(
        ImageSettings(resolution=(128, 128), hfw=200e-6, beam_type=BeamType.ELECTRON)
    )
    image.save(str(outdir / "overview.tif"))
    grid.task_history.append(
        TaskState(
            name="overview_sem",
            status=TaskStatus.Completed,
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
    listed = client.get("/app/experiment/grids", headers=AUTH).json()
    assert listed["available"] is True
    (summary,) = listed["items"]
    assert summary["name"] == "grid-aspen"
    assert summary["quality"] == "UNASSESSED"
    assert summary["completed_tasks"] == ["overview_sem"]
    assert summary["last_completed_task"] == "overview_sem"
    assert summary["num_items"] == 2
    assert summary["overviews"] == {"overview_sem": "overview.tif"}

    detail = client.get("/app/experiment/grids/grid-aspen", headers=AUTH).json()
    assert detail["tasks"] == [
        {
            "name": "overview_sem",
            "status": "Completed",
            "files": {"overview_sem": ["overview.tif"]},
        }
    ]
    assert sorted(detail["items"]) == sorted(p.name for p in host.experiment.positions)

    missing = client.get("/app/experiment/grids/nope", headers=AUTH).json()
    assert missing["available"] is False
    assert missing["grid_names"] == ["grid-aspen"]


def test_grid_overview_serves_jpeg_and_refuses_unlisted_names(
    client, grid_with_overview
):
    served = client.get(
        "/app/experiment/grids/grid-aspen/outputs/overview.tif", headers=AUTH
    )
    assert served.status_code == 200
    assert served.headers["content-type"] == "image/jpeg"
    assert served.content[:2] == b"\xff\xd8"

    unknown = client.get(
        "/app/experiment/grids/grid-aspen/outputs/nope.tif", headers=AUTH
    )
    assert unknown.status_code == 404
    assert unknown.json()["detail"]["filenames"] == ["overview.tif"]

    no_grid = client.get(
        "/app/experiment/grids/nope/outputs/overview.tif", headers=AUTH
    )
    assert no_grid.status_code == 404
    assert no_grid.json()["detail"]["grid_names"] == ["grid-aspen"]


def test_sidecar_returns_a_grid_overview_as_image_content(client, grid_with_overview):
    pytest.importorskip("mcp")
    from fibsem.ai.agent.mcp.sidecar import build_sidecar

    client.headers["Authorization"] = f"Bearer {TOKEN}"
    sidecar = build_sidecar(client, client.get("/capabilities").json())

    def call(filename):
        result = asyncio.run(
            sidecar.call_tool(
                "grid_get_output_image",
                {"grid_name": "grid-aspen", "filename": filename},
            )
        )
        if isinstance(result, tuple):
            result = result[0]
        return list(getattr(result, "content", result))

    contents = call("overview.tif")
    image = next(c for c in contents if getattr(c, "type", None) == "image")
    assert image.mime_type == "image/jpeg"
    # The image's metadata block rides along, so its scale needs no second call.
    meta = json.loads(
        next(c.text for c in contents if getattr(c, "type", None) == "text")
    )
    assert meta["metadata"]["beam_type"] == "ELECTRON"
    assert meta["metadata"]["pixel_size"] == pytest.approx(200e-6 / 128)
    from mcp.server.mcpserver.exceptions import ToolError

    with pytest.raises(ToolError) as refused:
        call("nope.tif")
    assert "refused (404)" in str(refused.value)
    assert "overview.tif" in str(refused.value)


def test_a_saved_beam_image_carries_its_metadata_block(client, grid_with_overview):
    from fibsem.server.app_routes import IMAGE_METADATA_HEADER

    served = client.get(
        "/app/experiment/grids/grid-aspen/outputs/overview.tif", headers=AUTH
    )
    block = json.loads(served.headers[IMAGE_METADATA_HEADER])
    assert block["beam_type"] == "ELECTRON"
    assert block["hfw"] == pytest.approx(200e-6)
    assert block["pixel_size"] == pytest.approx(200e-6 / 128)
    # 128 px wide renders 1:1, so a preview pixel is a source pixel.
    assert block["preview_pixelsize"] == pytest.approx(block["pixel_size"])
    assert set(block["stage_position"]) >= {"x", "y", "z", "r_deg", "t_deg"}
    assert block["beam"]["voltage"] is not None
    assert block["path"].endswith("overview.tif")


def test_a_plain_tiff_has_no_metadata_header(client, host):
    import numpy as np
    import tifffile

    from fibsem.server.app_routes import IMAGE_METADATA_HEADER
    from fibsem.workflows.status import TaskState

    lamella = host.experiment.positions[0]
    os.makedirs(lamella.path, exist_ok=True)
    tifffile.imwrite(
        os.path.join(lamella.path, "plain.tif"), np.zeros((8, 8), np.uint8)
    )
    lamella.task_history.append(
        TaskState(name="Rough Milling", outputs={"final_fib": ["plain.tif"]})
    )
    served = client.get(
        f"/app/experiment/items/{lamella.name}/outputs/plain.tif", headers=AUTH
    )
    assert served.status_code == 200
    assert IMAGE_METADATA_HEADER not in served.headers


def test_image_metadata_reads_files_inside_the_experiment_only(
    client, host, grid_with_overview, tmp_path
):
    import numpy as np
    import tifffile

    path = str(
        host.experiment.grid_path(grid_with_overview) / "overview_sem" / "overview.tif"
    )
    url = "/app/experiment/images/metadata"
    full = client.get(url, params={"path": path}, headers=AUTH)
    assert full.status_code == 200
    body = full.json()
    assert body["path"] == path
    assert body["metadata"]["image"]["beam_type"] == "ELECTRON"
    assert "microscope_state" in body["metadata"]

    outside = tmp_path / "elsewhere.tif"
    tifffile.imwrite(str(outside), np.zeros((8, 8), np.uint8))
    refused = client.get(url, params={"path": str(outside)}, headers=AUTH)
    assert refused.status_code == 403
    assert refused.json()["detail"]["error_type"] == "not_allowed"
    escape = os.path.join(str(host.experiment.path), "..", "elsewhere.tif")
    assert client.get(url, params={"path": escape}, headers=AUTH).status_code == 403

    missing = os.path.join(str(host.experiment.path), "nope.tif")
    assert client.get(url, params={"path": missing}, headers=AUTH).status_code == 404


def test_grid_markers_place_items_in_source_pixels(client, host, grid_with_overview):
    body = client.get(
        "/app/experiment/grids/grid-aspen/outputs/overview.tif/markers", headers=AUTH
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
        "/app/experiment/grids/grid-aspen/outputs/overview.tif/markers", headers=AUTH
    ).json()
    # a default MicroscopeState carries a position with no r: unplaceable, with
    # the reprojection's own reason rather than a silent drop
    (unplaced,) = body["unplaced"]
    assert unplaced["item_name"] == poseless.name
    assert "r coordinate" in unplaced["reason"]
    assert len(body["markers"]) == 2

    unknown = client.get(
        "/app/experiment/grids/grid-aspen/outputs/nope.tif/markers", headers=AUTH
    ).json()
    assert unknown["markers"] == []
    assert unknown["filenames"] == ["overview.tif"]


def test_grid_workflow_plan_is_the_preflight_without_hardware(client, host):
    from fibsem.workflows.grid.imaging import (
        BeamOverviewGridTaskConfig,
    )
    from fibsem.workflows.grid.manager import (
        LOAD_ENTRY_NAME,
    )
    from fibsem.workflows.grids import GridRecord

    experiment = host.experiment
    experiment.grid_protocol.add(BeamOverviewGridTaskConfig(task_name="overview_sem"))
    experiment.grid_protocol.add(BeamOverviewGridTaskConfig(task_name="overview_fib"))
    experiment.add_grid(GridRecord(name="grid-aspen"))
    experiment.add_grid(GridRecord(name="grid-birch"))

    body = client.post(
        "/app/workflow/grids/plan",
        headers=AUTH,
        json={
            "task_names": ["overview_fib", "overview_sem"],
            "grid_names": ["grid-birch"],
        },
    ).json()
    assert body["valid"] is True
    assert body["steps"] == [
        {"grid": "grid-birch", "step": LOAD_ENTRY_NAME},
        {"grid": "grid-birch", "step": "overview_fib"},
        {"grid": "grid-birch", "step": "overview_sem"},
    ]

    every = client.post(
        "/app/workflow/grids/plan", headers=AUTH, json={"task_names": ["overview_sem"]}
    ).json()
    assert every["grid_names"] == ["grid-aspen", "grid-birch"]
    assert len(every["steps"]) == 4

    screen = client.post(
        "/app/workflow/grids/plan",
        headers=AUTH,
        json={"task_names": ["overview_sem"], "screen_all": True},
    ).json()
    assert screen["valid"] is True and screen["screen_all"] is True
    assert "inventory" in screen["note"]

    bad_task = client.post(
        "/app/workflow/grids/plan", headers=AUTH, json={"task_names": ["nope"]}
    ).json()
    assert bad_task["valid"] is False
    assert bad_task["task_names"] == ["overview_sem", "overview_fib"]
    bad_grid = client.post(
        "/app/workflow/grids/plan",
        headers=AUTH,
        json={"task_names": ["overview_sem"], "grid_names": ["grid-oak"]},
    ).json()
    assert bad_grid["valid"] is False
    assert bad_grid["grid_names"] == ["grid-aspen", "grid-birch"]
    malformed = client.post("/app/workflow/grids/plan", headers=AUTH, json={})
    assert malformed.status_code == 422


def test_render_markers_draws_the_items_on_any_beam_image(
    client, host, grid_with_overview
):
    """Any beam image in the experiment with the items on it, as the overview
    plot dialog draws them: saved beside the image for the agent to cite as
    evidence, with a preview in the answer. A grid name draws that grid's
    items; without one, every item. A poseless item is left off, not a
    failure."""
    import base64
    import os
    import shutil

    from fibsem.structures import MicroscopeState

    host.experiment.add_new_lamella(MicroscopeState(), EventedDict())
    poseless = host.experiment.positions[-1]
    poseless.grid_id = grid_with_overview.id
    poseless.milling_pose = MicroscopeState()
    overview = host.experiment.grid_path(grid_with_overview) / "overview_sem"
    url = "/app/experiment/images/markers"
    resp = client.post(
        url,
        headers=AUTH,
        json={"path": str(overview / "overview.tif"), "grid_name": "grid-aspen"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["path"] == str(overview / "markers" / "overview.png")
    assert os.path.getsize(body["path"]) > 0
    assert base64.b64decode(body["plot"]["image_b64_jpeg"])[:2] == b"\xff\xd8"
    assert len(body["items"]) == 3 and body["linked"] is True

    # One of the agent's own overviews, on no grid's record.
    own = os.path.join(str(host.experiment.path), "overviews", "mine.tif")
    os.makedirs(os.path.dirname(own), exist_ok=True)
    shutil.copy(overview / "overview.tif", own)
    mine = client.post(url, headers=AUTH, json={"path": own})
    assert mine.status_code == 200, mine.text
    assert mine.json()["path"].endswith(
        os.path.join("overviews", "markers", "mine.png")
    )
    assert len(mine.json()["items"]) == len(host.experiment.positions)
    assert "linked" not in mine.json()

    assert client.post(url, headers=AUTH, json={}).status_code == 422
    missing = client.post(url, headers=AUTH, json={"path": own + ".nope"})
    assert missing.status_code == 404
    unknown = client.post(url, headers=AUTH, json={"path": own, "grid_name": "x"})
    assert unknown.status_code == 404


def test_fm_overview_previews_in_colour_and_places_markers(
    client, host, grid_with_overview
):
    """A fluorescence overview is an OME-TIFF stack: the preview composites it
    channel-by-colour (a grey max-projection loses the channels), and markers
    go through the FM canvas's own reprojection, which reads the stage
    position and hardware geometry the mosaic recorded."""
    import numpy as np
    from PIL import Image

    from fibsem.fm.structures import (
        FluorescenceChannelMetadata,
        FluorescenceImage,
        FluorescenceImageMetadata,
    )
    from fibsem.structures import FibsemHardwareGeometry, FibsemStagePosition
    from fibsem.workflows.status import TaskState, TaskStatus

    experiment = host.experiment
    grid = grid_with_overview
    centre = FibsemStagePosition(x=1e-3, y=-2e-3, z=0.0, r=0.0, t=0.0)
    outdir = experiment.grid_path(grid) / "overview_fm"
    outdir.mkdir(parents=True)
    # Two channels, red and green, in separate corners: the composite must
    # carry both hues, which a single-plane grey projection cannot.
    data = np.zeros((2, 3, 64, 64), dtype=np.uint16)
    data[0, :, :32, :32] = 4000
    data[1, :, 32:, 32:] = 4000
    channels = [
        FluorescenceChannelMetadata(
            name=f"Channel-{i:02d}",
            excitation_wavelength=488.0,
            power=0.5,
            exposure_time=0.1,
            gain=1.0,
            offset=0.0,
            color=colour,
        )
        for i, colour in enumerate(("red", "green"))
    ]
    metadata = FluorescenceImageMetadata(
        acquisition_datetime="2026-01-01T00:00:00",
        pixel_size_x=1e-7,
        pixel_size_y=1e-7,
        resolution=(64, 64),
        channels=channels,
        z_positions=[-1e-6, 0.0, 1e-6],
        stage_position=centre,
        geometry=FibsemHardwareGeometry(
            column_tilt=0,
            fib_column_tilt=52,
            shuttle_pre_tilt=0,
            rotation_reference=0,
            rotation_180=0,
            fm_is_pose=True,
            mirrors_back_view=True,
            camera_tilt=180.0,
        ),
    )
    # The linked items sit at the beam overview's centre (the stage origin);
    # put one at this mosaic's centre instead so it lands mid-image.
    experiment.positions[0].milling_pose.stage_position = centre
    FluorescenceImage(data=data, metadata=metadata).save(
        str(outdir / "overview.ome.tiff")
    )
    grid.task_history.append(
        TaskState(
            name="overview_fm",
            status=TaskStatus.Completed,
            outputs={"overview_fm": ["overview_fm/overview.ome.tiff"]},
        )
    )

    served = client.get(
        "/app/experiment/grids/grid-aspen/outputs/overview.ome.tiff", headers=AUTH
    )
    assert served.status_code == 200
    import io

    rgb = np.asarray(Image.open(io.BytesIO(served.content)).convert("RGB")).astype(int)
    top_left, bottom_right = rgb[8, 8], rgb[56, 56]
    assert top_left[0] > 150 and top_left[1] < 80  # red corner
    assert bottom_right[1] > 150 and bottom_right[0] < 80  # green corner
    # what the legend needs: the channels as drawn, and the z planes
    from fibsem.server.app_routes import IMAGE_METADATA_HEADER

    fm = json.loads(served.headers[IMAGE_METADATA_HEADER])["fm"]
    assert [c["name"] for c in fm["channels"]] == [c.name for c in metadata.channels]
    assert fm["z_planes"] == data.shape[-3]

    markers = client.get(
        "/app/experiment/grids/grid-aspen/outputs/overview.ome.tiff/markers",
        headers=AUTH,
    ).json()
    assert markers["image"]["beam_type"] == "FM"
    assert markers["image"]["width"] == 64 and markers["image"]["pixel_size"] == 1e-7
    assert markers["unplaced"] == []
    by_name = {m["item_name"]: m for m in markers["markers"]}
    at_centre = by_name[experiment.positions[0].name]
    assert at_centre["x"] == pytest.approx(32.0) and at_centre["y"] == pytest.approx(
        32.0
    )
    assert at_centre["inside"] is True
    assert by_name[experiment.positions[1].name]["inside"] is False
    # The inverse: the mosaic's centre pixel is the stage position it was taken at.
    converted = client.get(
        "/app/experiment/grids/grid-aspen/outputs/overview.ome.tiff/stage_position",
        params={"x": 32, "y": 32},
        headers=AUTH,
    ).json()["stage_position"]
    assert (converted["x"], converted["y"]) == pytest.approx((centre.x, centre.y))

    listed = client.get("/app/experiment/grids", headers=AUTH).json()["items"][0][
        "overviews"
    ]
    assert listed == {
        "overview_sem": "overview.tif",
        "overview_fm": "overview.ome.tiff",
    }


def test_unlinked_items_fall_back_only_on_a_single_grid_experiment(
    client, host, grid_with_overview
):
    from fibsem.workflows.grids import GridRecord

    experiment = host.experiment
    for lamella in experiment.positions:
        lamella.grid_id = None
    # One grid recorded: every posed item is shown, flagged as unlinked.
    body = client.get(
        "/app/experiment/grids/grid-aspen/outputs/overview.tif/markers", headers=AUTH
    ).json()
    assert body["linked"] is False
    assert len(body["markers"]) == 2

    # A second grid: the same items would paint both grids, so neither gets them.
    experiment.add_grid(GridRecord(name="grid-birch"))
    body = client.get(
        "/app/experiment/grids/grid-aspen/outputs/overview.tif/markers", headers=AUTH
    ).json()
    assert body["linked"] is False
    assert body["markers"] == [] and body["unplaced"] == []
    assert "several grids" in body["reason"]


# -- lamella positions: the Lamella tab's buttons, for an agent at Drive ----------


@pytest.fixture
def driver(microscope, host, event_buffer):
    app = build_server(
        microscope,
        app_context=AutoLamellaAppContext(host, event_buffer=event_buffer),
        auth=AuthConfig.generate(level="drive", token=TOKEN),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers["Authorization"] = f"Bearer {TOKEN}"
        yield client


def test_an_item_is_added_where_the_stage_is(driver, host, microscope):
    from fibsem.structures import FibsemStagePosition

    microscope.move_stage_absolute(FibsemStagePosition(x=1e-4, y=-2e-4))
    before = len(host.experiment.positions)
    resp = driver.post("/app/experiment/items", json={"name": "site-a"})
    assert resp.status_code == 200, resp.text
    item = resp.json()["item"]
    assert item["item_name"] == "site-a"
    assert len(host.experiment.positions) == before + 1
    lamella = host.experiment.get_lamella_by_name("site-a")
    assert lamella.milling_pose.stage_position.x == pytest.approx(1e-4)
    assert lamella.milling_angle is not None
    # Saved, as a click saves.
    from fibsem.applications.autolamella.structures import Experiment

    saved = Experiment.load(os.path.join(host.experiment.path, "experiment.yaml"))
    assert "site-a" in [p.name for p in saved.positions]

    taken = driver.post("/app/experiment/items", json={"name": "site-a"})
    assert taken.status_code == 422


def test_an_item_is_added_on_an_overview_the_agent_saved(
    driver, host, grid_with_overview
):
    """Headless round 12 H1: an overview from beam_acquire_overview is in no
    task's outputs, so item_add on it failed after approval."""
    import shutil

    from fibsem.workflows.outputs import saved_file

    experiment = host.experiment
    source = experiment.grid_path(grid_with_overview) / "overview_sem/overview.tif"
    shutil.copy(source, os.path.join(experiment.path, "grid01-overview.tif"))
    base = "/app/experiment/grids/grid-aspen/outputs/grid01-overview.tif"
    assert driver.get(base).status_code == 200
    assert driver.get(f"{base}/markers").json()["markers"] is not None
    resp = driver.post(
        "/app/experiment/items",
        json={
            "name": "on-saved",
            "overview": {
                "grid_name": "grid-aspen",
                "filename": "grid01-overview.tif",
                "px": [40.0, 90.0],
            },
        },
    )
    assert resp.status_code == 200, resp.text
    # Nothing outside the experiment folder.
    assert saved_file(experiment, experiment.path, "../outside.tif") is None


def test_an_item_is_added_under_an_overview_point(driver, host, grid_with_overview):
    # The marker for the new item lands back on the pixel it was added at: the
    # conversion is the markers' inverse.
    point = [40.0, 90.0]
    converted = driver.get(
        "/app/experiment/grids/grid-aspen/outputs/overview.tif/stage_position",
        params={"x": point[0], "y": point[1]},
    )
    assert converted.status_code == 200, converted.text
    assert converted.json()["grid_id"] == grid_with_overview.id
    by_fraction = driver.get(
        "/app/experiment/grids/grid-aspen/outputs/overview.tif/stage_position",
        params={"fx": point[0] / 128, "fy": point[1] / 128},
    ).json()
    assert by_fraction["stage_position"] == pytest.approx(
        converted.json()["stage_position"]
    )

    resp = driver.post(
        "/app/experiment/items",
        json={
            "name": "picked",
            "overview": {
                "grid_name": "grid-aspen",
                "filename": "overview.tif",
                "px": point,
            },
        },
    )
    assert resp.status_code == 200, resp.text
    assert host.experiment.get_lamella_by_name("picked").grid_id == (
        grid_with_overview.id
    )
    markers = driver.get(
        "/app/experiment/grids/grid-aspen/outputs/overview.tif/markers"
    ).json()
    marker = next(m for m in markers["markers"] if m["item_name"] == "picked")
    assert (marker["x"], marker["y"]) == pytest.approx(point, abs=0.5)

    outside = driver.get(
        "/app/experiment/grids/grid-aspen/outputs/overview.tif/stage_position",
        params={"x": 500, "y": 10},
    )
    assert outside.status_code == 422
    unknown = driver.get(
        "/app/experiment/grids/grid-aspen/outputs/nope.tif/stage_position",
        params={"x": 1, "y": 1},
    )
    assert unknown.status_code == 404
    assert unknown.json()["detail"]["filenames"] == ["overview.tif"]


def test_a_pose_is_set_from_the_stage_and_moved_back_to(driver, host, microscope):
    from fibsem.structures import FibsemStagePosition

    name = host.experiment.positions[0].name
    microscope.move_stage_absolute(FibsemStagePosition(x=3e-4, y=1e-4))
    resp = driver.post(f"/app/experiment/items/{name}/poses/MILLING")
    assert resp.status_code == 200, resp.text
    lamella = host.experiment.get_lamella_by_name(name)
    assert lamella.milling_pose.stage_position.x == pytest.approx(3e-4)
    assert lamella.milling_angle is not None

    microscope.move_stage_absolute(FibsemStagePosition(x=0.0, y=0.0))
    moved = driver.post(
        f"/app/experiment/items/{name}/move_to", json={"pose": "MILLING"}
    )
    assert moved.status_code == 200, moved.text
    assert microscope.get_stage_position().x == pytest.approx(3e-4)
    assert moved.json()["stage_position"]["x"] == pytest.approx(3e-4)

    no_pose = driver.post(
        f"/app/experiment/items/{name}/move_to", json={"pose": "NOPE"}
    )
    assert no_pose.status_code == 404
    assert "MILLING" in no_pose.json()["detail"]["pose_names"]
    unknown = driver.post("/app/experiment/items/nope/poses/MILLING")
    assert unknown.status_code == 404
    assert name in unknown.json()["detail"]["item_names"]


def test_an_item_is_moved_to_a_point_and_the_stage_stays(driver, host, microscope):
    """Moving a lamella on an overview: its poses move, the stage doesn't."""
    from fibsem.applications.autolamella.structures import Experiment
    from fibsem.structures import FibsemStagePosition

    name = host.experiment.positions[0].name
    microscope.move_stage_absolute(FibsemStagePosition(x=0.0, y=0.0))
    resp = driver.post(
        f"/app/experiment/items/{name}/position",
        json={"stage_position": {"x": 2e-4, "y": -1e-4, "z": 0, "r": 0, "t": 0}},
    )
    assert resp.status_code == 200, resp.text
    lamella = host.experiment.get_lamella_by_name(name)
    assert lamella.milling_pose.stage_position.x == pytest.approx(2e-4)
    assert lamella.milling_pose.stage_position.y == pytest.approx(-1e-4)
    assert microscope.get_stage_position().x == pytest.approx(0.0)
    saved = Experiment.load(os.path.join(host.experiment.path, "experiment.yaml"))
    moved = next(p for p in saved.positions if p.name == name)
    assert moved.milling_pose.stage_position.x == pytest.approx(2e-4)

    nowhere = driver.post(f"/app/experiment/items/{name}/position", json={})
    assert nowhere.status_code == 422
    there = {"stage_position": {"x": 0, "y": 0, "z": 0, "r": 0, "t": 0}}
    with run_working_on(host.runner, name):
        refused = driver.post(f"/app/experiment/items/{name}/position", json=there)
        assert refused.status_code == 409
        assert f"working on {name}" in refused.json()["detail"]["message"]
    with run_working_on(host.runner, "another"):
        moved = driver.post(f"/app/experiment/items/{name}/position", json=there)
        assert moved.status_code == 200, moved.text


def test_an_item_s_objective_position_is_applied_to_all(driver, host):
    """Apply to All Lamella: one lamella's stored objective position on every
    lamella with a fluorescence pose."""
    from fibsem.structures import MicroscopeState

    host.experiment.add_new_lamella(MicroscopeState(), EventedDict(), name="second")
    positions = host.experiment.positions
    for lamella in positions[:2]:
        lamella.fluorescence_pose = MicroscopeState()
    with_pose = [p for p in positions if p.fluorescence_pose is not None]
    source = with_pose[0]
    source.fluorescence_pose.objective_position = 1.25e-3
    route = f"/app/experiment/items/{source.name}/objective_position/apply_to_all"
    resp = driver.post(route)
    assert resp.status_code == 200, resp.text
    assert resp.json()["items"] == len(with_pose)
    assert all(
        p.fluorescence_pose.objective_position == pytest.approx(1.25e-3)
        for p in with_pose
    )

    # The lamella a run is on is left as it is.
    busy = with_pose[-1]
    busy.fluorescence_pose.objective_position = 2e-3
    source.fluorescence_pose.objective_position = 1.5e-3
    with run_working_on(host.runner, busy.name):
        resp = driver.post(route)
    assert resp.json()["skipped"] == [busy.name]
    assert busy.fluorescence_pose.objective_position == pytest.approx(2e-3)


def test_a_pose_is_not_set_from_the_stage_while_a_run_is_on(driver, host):
    """The run moves the stage, so the pose would be wherever it left it."""
    name = host.experiment.positions[0].name
    before = host.experiment.get_lamella_by_name(name).milling_pose
    host.runner.is_running = True
    try:
        refused = driver.post(f"/app/experiment/items/{name}/poses/MILLING")
        assert refused.status_code == 409
        assert refused.json()["detail"]["error_type"] == "task_running"
    finally:
        host.runner.is_running = False
    assert host.experiment.get_lamella_by_name(name).milling_pose is before


def test_an_item_is_removed_but_not_the_one_a_run_is_on(driver, host):
    host.experiment.add_new_lamella(MicroscopeState(), EventedDict(), name="spare")
    with run_working_on(host.runner, "spare"):
        refused = driver.delete("/app/experiment/items/spare")
        assert refused.status_code == 409
        assert refused.json()["detail"]["error_type"] == "task_running"
    with run_working_on(host.runner, host.experiment.positions[0].name):
        removed = driver.delete("/app/experiment/items/spare")
    assert removed.status_code == 200
    assert host.experiment.get_lamella_by_name("spare") is None


def test_an_item_is_not_added_where_the_stage_is_while_a_run_is_on(driver, host):
    """The run has moved the stage, so "here" is wherever it left it; a given
    stage_position is still taken."""
    host.runner.is_running = True
    try:
        refused = driver.post("/app/experiment/items", json={"name": "here"})
        assert refused.status_code == 409
        assert "stage_position" in refused.json()["detail"]["message"]
        given = driver.post(
            "/app/experiment/items",
            json={
                "name": "given",
                "stage_position": {"x": 1e-4, "y": 0, "z": 0, "r": 0, "t": 0},
            },
        )
        assert given.status_code == 200, given.text
    finally:
        host.runner.is_running = False
    assert host.experiment.get_lamella_by_name("here") is None


def test_position_edits_need_drive(client, host):
    name = host.experiment.positions[0].name
    assert (
        client.post("/app/experiment/items", headers=AUTH, json={}).status_code == 403
    )
    assert (
        client.delete(f"/app/experiment/items/{name}", headers=AUTH).status_code == 403
    )
    assert (
        client.post(
            f"/app/experiment/items/{name}/move_to",
            headers=AUTH,
            json={"pose": "MILLING"},
        ).status_code
        == 403
    )


# --- the experiment's files (D3) ----------------------------------------------

OVERVIEW = "grids/grid-aspen/overview_sem/overview.tif"


def test_experiment_files_list_one_folder_at_watch(client, host, grid_with_overview):
    (Path(host.experiment.path) / "notes.md").write_text("# notes\n")
    top = client.get("/app/experiment/files", headers=AUTH)
    assert top.status_code == 200
    entries = {e["name"]: e for e in top.json()["entries"]}
    assert entries["grids"]["kind"] == "folder"
    assert entries["grids"]["images"] == 1  # the overview, three folders down
    notes = entries["notes.md"]
    assert (notes["kind"], notes["size"], notes["path"]) == ("text", 8, "notes.md")
    assert notes["modified"]
    assert "images" not in notes
    below = client.get(
        "/app/experiment/files",
        params={"dir": "grids/grid-aspen/overview_sem"},
        headers=AUTH,
    ).json()
    assert below["dir"] == "grids/grid-aspen/overview_sem"
    assert [(e["path"], e["kind"]) for e in below["entries"]] == [(OVERVIEW, "image")]
    file = client.get("/app/experiment/files", params={"dir": OVERVIEW}, headers=AUTH)
    assert file.status_code == 422


def test_experiment_files_read_text_and_image_previews(
    client, host, grid_with_overview
):
    (Path(host.experiment.path) / "notes.md").write_text("# notes\n")
    text = client.get("/app/experiment/files/notes.md", headers=AUTH).json()
    assert text["text"] == "# notes\n"
    assert text["truncated"] is False

    image = client.get(f"/app/experiment/files/{OVERVIEW}", headers=AUTH).json()
    assert image["summary"]["kind"] == "image"
    assert image["summary"]["beam_type"] == "ELECTRON"
    assert image["summary"]["pixel_size"] == pytest.approx(200e-6 / 128)
    preview = image["preview"]
    assert (preview["width"], preview["full_width"]) == (128, 128)
    assert preview["image_b64_jpeg"]
    assert preview["preview_pixelsize"] == pytest.approx(200e-6 / 128)
    # A thumbnail: the preview at most this wide.
    small = client.get(
        f"/app/experiment/files/{OVERVIEW}", params={"width": 64}, headers=AUTH
    ).json()["preview"]
    assert (small["width"], small["full_width"]) == (64, 128)
    # An absolute path inside the experiment, as an image answer gives it, too.
    absolute = os.path.join(str(host.experiment.path), OVERVIEW)
    served = client.get("/app/experiment/files/" + absolute, headers=AUTH)
    assert served.json()["path"] == os.path.realpath(absolute)


def test_experiment_files_crop_at_native_resolution(client, grid_with_overview):
    url = f"/app/experiment/files/{OVERVIEW}"
    box = client.get(url, params={"box": "32,16,20,10"}, headers=AUTH).json()
    preview = box["preview"]
    assert preview["region"] == {"x": 32, "y": 16, "width": 20, "height": 10}
    assert (preview["width"], preview["height"]) == (20, 10)
    # Native resolution: a preview pixel is a source pixel; the field is the region's.
    assert preview["preview_pixelsize"] == pytest.approx(200e-6 / 128)
    assert preview["hfw"] == pytest.approx(20 * 200e-6 / 128)
    assert box["summary"]["shape"] == [128, 128]

    fraction = client.get(
        url, params={"fraction": "0.5,0.25,0.25,0.5"}, headers=AUTH
    ).json()
    assert fraction["preview"]["region"] == {
        "x": 64,
        "y": 32,
        "width": 32,
        "height": 64,
    }

    for params in (
        {"box": "500,500,10,10"},  # misses the image
        {"fraction": "0.5,0.5,2,1"},  # not a fraction
        {"box": "1,2,3"},
        {"box": "a,b,c,d"},
        {"box": "0,0,8,8", "fraction": "0,0,0.5,0.5"},
    ):
        refused = client.get(url, params=params, headers=AUTH)
        assert refused.status_code == 422, params


def test_experiment_files_stay_inside_the_experiment(client, host, tmp_path):
    outside = tmp_path / "elsewhere.md"
    outside.write_text("secret")
    (Path(host.experiment.path) / "link.md").symlink_to(outside)
    (Path(host.experiment.path) / "notes.md").write_text("# notes\n")
    for path in ("..%2Felsewhere.md", "link.md", str(outside)):
        refused = client.get(f"/app/experiment/files/{path}", headers=AUTH)
        assert refused.status_code == 403, path
        assert refused.json()["detail"]["error_type"] == "not_allowed"
    escape = client.get("/app/experiment/files", params={"dir": ".."}, headers=AUTH)
    assert escape.status_code == 403
    assert client.get("/app/experiment/files/nope.md", headers=AUTH).status_code == 404
    # Only an image crops.
    text = client.get(
        "/app/experiment/files/notes.md", params={"box": "0,0,1,1"}, headers=AUTH
    )
    assert text.status_code == 422


def test_experiment_files_need_an_open_experiment(client, host):
    host.experiment = None
    for url in ("/app/experiment/files", "/app/experiment/files/notes.md"):
        refused = client.get(url, headers=AUTH)
        assert refused.status_code == 409
        assert refused.json()["detail"]["error_type"] == "no_experiment"


def test_sidecar_reads_experiment_files_as_tools_and_resources(
    client, host, grid_with_overview
):
    pytest.importorskip("mcp")
    from fibsem.ai.agent.mcp.sidecar import build_sidecar

    (Path(host.experiment.path) / "notes.md").write_text("# notes\n")
    client.headers["Authorization"] = f"Bearer {TOKEN}"
    sidecar = build_sidecar(client, client.get("/capabilities").json())

    result = asyncio.run(
        sidecar.call_tool(
            "experiment_crop_image", {"path": OVERVIEW, "box": [0, 0, 16, 16]}
        )
    )
    if isinstance(result, tuple):
        result = result[0]
    contents = list(getattr(result, "content", result))
    image = next(c for c in contents if getattr(c, "type", None) == "image")
    assert image.mime_type == "image/jpeg"
    answer = json.loads(contents[0].text)
    assert answer["preview"]["region"]["width"] == 16
    assert answer["preview"]["image_index"] == 0

    # resources/list: the experiment's top level only, not every file below it.
    listed = {str(r.uri) for r in asyncio.run(sidecar.list_resources())}
    assert {"experiment://grids", "experiment://notes.md"} <= listed
    assert not any(OVERVIEW in uri for uri in listed)
    templates = asyncio.run(sidecar.list_resource_templates())
    assert [t.uri_template for t in templates] == ["experiment://{+path}"]

    text = list(asyncio.run(sidecar.read_resource("experiment://notes.md")))
    assert json.loads(text[0].content)["text"] == "# notes\n"
    read = list(asyncio.run(sidecar.read_resource(f"experiment://{OVERVIEW}")))
    assert [c.mime_type for c in read] == ["application/json", "image/jpeg"]
    assert read[1].content[:2] == b"\xff\xd8"
    assert json.loads(read[0].content)["summary"]["beam_type"] == "ELECTRON"
    folder = list(asyncio.run(sidecar.read_resource("experiment://grids")))
    assert [e["name"] for e in json.loads(folder[0].content)["entries"]] == [
        "grid-aspen"
    ]


def test_thickness_reads_a_saved_fib_image_over_http_and_mcp(client, host):
    """FIB-1434: the band measured off a saved FIB image in the experiment,
    around the item's POI, by the route and by the Watch tool, with the
    result's plot."""
    pytest.importorskip("mcp")
    from fibsem.ai.agent.mcp.sidecar import build_sidecar
    from fibsem.structures import BeamType
    from fibsem.workflows.status import TaskState
    from tests.test_lamella_thickness import band_image

    lamella = host.experiment.positions[0]
    os.makedirs(lamella.path, exist_ok=True)
    fname = "ref_Polishing_final_high_res_ib.tif"
    image = band_image(235, angle_deg=0.6)
    image.metadata.microscope_state = MicroscopeState()  # a saved image has one
    image.metadata.image_settings.beam_type = BeamType.ION
    image.save(os.path.join(lamella.path, fname))
    sem = "ref_Polishing_final_high_res_eb.tif"
    image.metadata.image_settings.beam_type = BeamType.ELECTRON
    image.save(os.path.join(lamella.path, sem))
    lamella.task_history.append(
        TaskState(name="Polishing", outputs={"final_fib": [fname], "final_sem": [sem]})
    )
    route = f"/app/experiment/items/{lamella.name}/thickness"
    image_path = os.path.join(lamella.name, fname)

    answer = client.post(route, json={"path": image_path}, headers=AUTH).json()
    assert answer["thickness_nm"]["mean"] == pytest.approx(235, abs=10)
    assert answer["angle_deg"] == pytest.approx(0.6, abs=0.05)
    # This item has no trench patterns, so the window is the default.
    assert answer["window_um"] == [10.0, 4.0]
    assert answer["window_from"] == "default"
    assert answer["plot"]["image_b64_jpeg"]
    # Saved beside the image, as an autofocus run is.
    saved = answer["saved_to"]
    assert saved == os.path.join(lamella.path, "thickness", fname[: -len(".tif")])
    assert sorted(os.listdir(saved)) == ["data.json", "plot.png", "window.tif"]

    assert answer["warnings"] == []
    narrow = client.post(
        route, json={"path": image_path, "width_um": "5"}, headers=AUTH
    ).json()
    assert narrow["continuity"]["columns"] == 512
    assert narrow["window_from"] == "given"

    # Any FIB image in the experiment, not only a recorded output: the
    # agent's own check image (2026-10-19 run: refused mid-task).
    own = os.path.join(lamella.path, "polish-checks", "check_01_ib.tif")
    os.makedirs(os.path.dirname(own))
    image.metadata.image_settings.beam_type = BeamType.ION
    image.save(own)
    checked = client.post(route, json={"path": own}, headers=AUTH).json()
    assert checked["thickness_nm"]["mean"] == pytest.approx(235, abs=10)
    assert checked["saved_to"] == os.path.join(
        os.path.dirname(own), "thickness", "check_01_ib"
    )

    unknown = client.post(route, json={"path": "nope.tif"}, headers=AUTH)
    assert unknown.status_code == 404
    outside = client.post(route, json={"path": "/etc/hostname"}, headers=AUTH)
    assert outside.status_code == 403
    refused = client.post(
        route, json={"path": os.path.join(lamella.name, sem)}, headers=AUTH
    )
    assert refused.status_code == 422
    assert client.post(route, json={}, headers=AUTH).status_code == 422
    bad = client.post(
        route, json={"path": image_path, "height_um": "tall"}, headers=AUTH
    )
    assert bad.status_code == 422

    client.headers["Authorization"] = f"Bearer {TOKEN}"
    sidecar = build_sidecar(client, client.get("/capabilities").json())
    result = asyncio.run(
        sidecar.call_tool(
            "item_measure_thickness", {"item_name": lamella.name, "path": image_path}
        )
    )
    if isinstance(result, tuple):
        result = result[0]
    contents = list(getattr(result, "content", result))
    assert any(getattr(c, "type", None) == "image" for c in contents)
    text = json.loads(contents[0].text)
    assert text["thickness_nm"]["mean"] == pytest.approx(235, abs=10)
    assert text["plot"]["image_index"] == 0


def test_the_thickness_window_comes_from_the_trench_patterns():
    """Along: nine tenths of the narrowest trench pattern; across: the widest
    gap (the rough slab) and half a micron into each trench."""
    from types import SimpleNamespace

    from fibsem.applications.autolamella.structures import Lamella
    from fibsem.milling.patterning.patterns.trench import TrenchPattern

    def task(sync, *patterns):
        stages = [SimpleNamespace(pattern=p) for p in patterns]
        return SimpleNamespace(
            sync_to_poi=sync, milling={"mill": SimpleNamespace(stages=stages)}
        )

    rough = TrenchPattern(width=12e-6, spacing=2e-6)
    polish = TrenchPattern(width=10e-6, spacing=0.25e-6)
    lamella = SimpleNamespace(
        task_config={
            "Rough Milling": task(True, rough),
            "Polishing": task(True, polish),
            "Elsewhere": task(False, TrenchPattern(width=1e-6, spacing=9e-6)),
        }
    )
    width, height = Lamella.trench_window(lamella)
    assert width == pytest.approx(9e-6)
    assert height == pytest.approx(3e-6)
    assert Lamella.trench_window(SimpleNamespace(task_config={})) is None
