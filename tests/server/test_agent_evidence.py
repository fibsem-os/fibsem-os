"""The agent's evidence: kept in the experiment, an image of the experiment's or
one the agent made, and cited by id on its calls, requests and notes."""

import base64
import io
import os

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402
from psygnal.containers import EventedDict  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.applications.autolamella.server import AutoLamellaAppContext  # noqa: E402
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.server import AuthConfig, build_server, evidence  # noqa: E402
from fibsem.server.calls import EVIDENCE_HEADER, from_headers, headers_for  # noqa: E402
from fibsem.structures import MicroscopeState  # noqa: E402

TOKEN = "test-token"
_NUDGE = {
    "kwargs": {
        "delta": {
            "name": None,
            "x": 1e-6,
            "y": 0.0,
            "z": 0.0,
            "r": 0.0,
            "t": 0.0,
            "coordinate_system": "RAW",
        }
    }
}
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
    exp = Experiment(path=tmp_path / "exp", name="evidence-exp")
    exp.task_protocol = AutoLamellaTaskProtocol()
    (tmp_path / "exp").mkdir(parents=True, exist_ok=True)
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    host.experiment = exp
    host.microscope = microscope
    return host


@pytest.fixture
def client(microscope, host):
    from fibsem.applications.autolamella.server.events import EventBuffer

    app = build_server(
        microscope,
        app_context=AutoLamellaAppContext(host, event_buffer=EventBuffer()),
        auth=AuthConfig.generate(level="drive", token=TOKEN),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def _png(width=40, height=30, fmt="PNG") -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (200, 40, 160)).save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _events(client, kind):
    events = client.get("/app/events?since=0", headers=AUTH).json()["events"]
    return [e for e in events if e["kind"] == kind]


def test_an_upload_is_saved_into_the_experiment(client, host):
    item = host.experiment.positions[0].name
    marks = {"points": [{"x": 10, "y": 12, "label": "edge"}]}
    resp = client.post(
        "/app/experiment/evidence",
        headers=AUTH,
        json={
            "text": "edge 2 px left",
            "upload": _png(),
            "marks": marks,
            "item_name": item,
        },
    )
    assert resp.status_code == 200, resp.text
    record = resp.json()
    folder = evidence.folder(host.experiment.path)
    assert record["image"] == os.path.join(folder, f"{record['evidence_id']}.png")
    assert record["uploaded"] is True
    assert os.path.isfile(os.path.join(folder, f"{record['evidence_id']}.json"))
    with Image.open(record["image"]) as saved:
        assert saved.format == "PNG" and saved.size == (40, 30)
    kept = _events(client, "agent_evidence")[-1]["payload"]
    assert kept["evidence_id"] == record["evidence_id"]
    assert kept["marks"]["points"][0]["label"] == "edge"
    assert kept["item_id"] == host.experiment.positions[0].id


def test_a_large_upload_is_saved_at_most_2048_px(client):
    resp = client.post(
        "/app/experiment/evidence",
        headers=AUTH,
        json={"text": "wide", "upload": _png(3000, 1000, "JPEG")},
    )
    assert resp.status_code == 200, resp.text
    with Image.open(resp.json()["image"]) as saved:
        assert saved.format == "PNG" and max(saved.size) == evidence.MAX_SIDE


@pytest.mark.parametrize(
    "body, status",
    [
        ({"upload": _png()}, 422),  # no text
        ({"text": "x"}, 422),  # no image
        ({"text": "x", "upload": "not base64!"}, 422),
        ({"text": "x", "upload": base64.b64encode(b"GIF89a").decode()}, 422),
        ({"text": "x", "upload": _png(), "marks": {"points": [{"x": "a"}]}}, 422),
        ({"text": "x", "upload": _png(), "item_name": "nope"}, 404),
        ({"text": "x", "image": "/etc/hostname"}, 403),
    ],
)
def test_evidence_that_cannot_be_kept_is_refused(client, body, status):
    resp = client.post("/app/experiment/evidence", headers=AUTH, json=body)
    assert resp.status_code == status, resp.text


def test_an_upload_over_4_mb_is_refused(client):
    big = base64.b64encode(b"\x89PNG" + b"\0" * evidence.MAX_UPLOAD_BYTES).decode()
    resp = client.post(
        "/app/experiment/evidence", headers=AUTH, json={"text": "x", "upload": big}
    )
    assert resp.status_code == 413


def test_a_call_and_a_note_cite_the_evidence(client, microscope):
    kept = client.post(
        "/app/experiment/evidence",
        headers=AUTH,
        json={"text": "face clean", "upload": _png()},
    ).json()
    cite = {**AUTH, EVIDENCE_HEADER: kept["evidence_id"]}
    note = client.post("/app/experiment/notes", headers=cite, json={"text": "accepted"})
    assert note.status_code == 200, note.text
    assert (
        _events(client, "agent_note")[-1]["payload"]["evidence"] == kept["evidence_id"]
    )

    recorded = []

    def record(kind, payload):
        recorded.append((kind, payload))

    microscope.session.record_signal.connect(record)
    try:
        moved = client.post(
            "/microscope/services/stage_movement/commands/move_relative",
            headers={**cite, "X-Agent-Tool": "stage_move_relative"},
            json=_NUDGE,
        )
    finally:
        microscope.session.record_signal.disconnect(record)
    assert moved.status_code == 200, moved.text
    (called,) = [p for kind, p in recorded if kind == "agent_called"]
    assert called["evidence"] == kept["evidence_id"]


def test_the_evidence_header_round_trips():
    headers = {
        k.lower(): v for k, v in headers_for("x", {}, evidence="ab12cd34").items()
    }
    assert from_headers(headers).evidence == "ab12cd34"
    assert from_headers({}).evidence is None


def test_the_sidecar_keeps_a_file_and_the_call_cites_it(client, microscope, tmp_path):
    """An acting tool's evidence: the sidecar reads the file on the agent's
    machine, keeps it in the experiment, and the call names it."""
    pytest.importorskip("mcp")
    import asyncio

    from fibsem.mcp.sidecar import build_sidecar

    overlay = tmp_path / "overlay.png"
    overlay.write_bytes(base64.b64decode(_png()))
    client.headers["Authorization"] = f"Bearer {TOKEN}"
    sidecar = build_sidecar(client, client.get("/capabilities").json())
    recorded = []

    def record(kind, payload):
        recorded.append((kind, payload))

    microscope.session.record_signal.connect(record)
    try:
        asyncio.run(
            sidecar.call_tool(
                "stage_move_relative",
                {
                    "dx": 1e-6,
                    "evidence": {"text": "edge 1 µm left", "file": str(overlay)},
                },
            )
        )
    finally:
        microscope.session.record_signal.disconnect(record)
    (kept,) = _events(client, "agent_evidence")
    assert kept["payload"]["uploaded"] is True
    assert os.path.isfile(kept["payload"]["image"])
    (called,) = [p for kind, p in recorded if kind == "agent_called"]
    assert called["evidence"] == kept["payload"]["evidence_id"]
    # The arguments the server describes the call by leave the evidence out.
    assert "evidence" not in called["arguments"]


def test_every_acting_tool_takes_evidence_and_no_read_does():
    from fibsem.server.catalog import CATALOG

    by_name = {t.name: t for t in CATALOG}
    assert by_name["stage_move_relative"].takes_evidence
    assert by_name["request_create"].takes_evidence
    assert by_name["experiment_add_note"].takes_evidence
    assert not by_name["stage_get_position"].takes_evidence
    assert "image" not in by_name["request_create"].params
