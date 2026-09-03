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


def test_grid_workflow_plan_is_the_preflight_without_hardware(client, host):
    from fibsem.applications.autolamella.structures import GridRecord
    from fibsem.applications.autolamella.workflows.tasks.grid.imaging import (
        BeamOverviewGridTaskConfig,
    )
    from fibsem.applications.autolamella.workflows.tasks.grid.manager import (
        LOAD_ENTRY_NAME,
    )

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


def test_fm_overview_previews_in_colour_and_places_markers(
    client, host, grid_with_overview
):
    """A fluorescence overview is an OME-TIFF stack: the preview composites it
    channel-by-colour (a grey max-projection loses the channels), and markers
    go through the FM canvas's own reprojection, which reads the stage
    position and hardware geometry the mosaic recorded."""
    import numpy as np
    from PIL import Image

    from fibsem.applications.autolamella.structures import (
        AutoLamellaTaskState,
        AutoLamellaTaskStatus,
    )
    from fibsem.fm.structures import (
        FluorescenceChannelMetadata,
        FluorescenceImage,
        FluorescenceImageMetadata,
    )
    from fibsem.structures import FibsemHardwareGeometry, FibsemStagePosition

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
        acquisition_date="2026-01-01T00:00:00",
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
            is_compustage=True,
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
        AutoLamellaTaskState(
            name="overview_fm",
            status=AutoLamellaTaskStatus.Completed,
            outputs={"overview_fm": ["overview_fm/overview.ome.tiff"]},
        )
    )

    served = client.get("/app/grids/grid-aspen/outputs/overview.ome.tiff", headers=AUTH)
    assert served.status_code == 200
    import io

    rgb = np.asarray(Image.open(io.BytesIO(served.content)).convert("RGB")).astype(int)
    top_left, bottom_right = rgb[8, 8], rgb[56, 56]
    assert top_left[0] > 150 and top_left[1] < 80  # red corner
    assert bottom_right[1] > 150 and bottom_right[0] < 80  # green corner

    markers = client.get(
        "/app/grids/grid-aspen/outputs/overview.ome.tiff/markers", headers=AUTH
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

    listed = client.get("/app/grids", headers=AUTH).json()["items"][0]["overviews"]
    assert listed == {
        "overview_sem": "overview.tif",
        "overview_fm": "overview.ome.tiff",
    }


def test_unlinked_items_fall_back_only_on_a_single_grid_experiment(
    client, host, grid_with_overview
):
    from fibsem.applications.autolamella.structures import GridRecord

    experiment = host.experiment
    for lamella in experiment.positions:
        lamella.grid_id = None
    # One grid recorded: every posed item is shown, flagged as unlinked.
    body = client.get(
        "/app/grids/grid-aspen/outputs/overview.tif/markers", headers=AUTH
    ).json()
    assert body["linked"] is False
    assert len(body["markers"]) == 2

    # A second grid: the same items would paint both grids, so neither gets them.
    experiment.add_grid(GridRecord(name="grid-birch"))
    body = client.get(
        "/app/grids/grid-aspen/outputs/overview.tif/markers", headers=AUTH
    ).json()
    assert body["linked"] is False
    assert body["markers"] == [] and body["unplaced"] == []
    assert "several grids" in body["reason"]
