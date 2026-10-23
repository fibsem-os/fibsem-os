"""Evidence: kept in the experiment, an image of the experiment's or one the
agent made, and cited by id on its calls, requests, suggestions, decisions and
notes."""

import asyncio
import base64
import io
import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402
from psygnal.containers import EventedDict  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.ai.agent.calls import (  # noqa: E402
    EVIDENCE_HEADER,
    from_headers,
    headers_for,
)
from fibsem.ai.agent.catalog import CATALOG  # noqa: E402
from fibsem.ai.agent.levels import AgentLevel  # noqa: E402
from fibsem.applications.autolamella.server import AutoLamellaAppContext  # noqa: E402
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.structures import MicroscopeState  # noqa: E402
from fibsem.workflows import evidence  # noqa: E402
from fibsem.workflows.actions import ACTION, NOTE  # noqa: E402
from fibsem.workflows.proposals import (  # noqa: E402
    Decision,
    DecisionOutcome,
    Suggestion,
)
from tests.run_state import stub_session

AGENT = {"Authorization": "Bearer agent"}
MOVE_RELATIVE = "/microscope/services/stage_movement/commands/move_relative"
NUDGE = {
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


class RecordingContext(AutoLamellaAppContext):
    def __init__(self, session):
        super().__init__(session)
        self.recorded = []

    def record_event(self, kind, payload, actor=None):
        self.recorded.append((kind, payload))


@pytest.fixture(scope="module")
def microscope():
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    return microscope


@pytest.fixture
def context(tmp_path, microscope):
    host = stub_session()
    exp = Experiment(path=tmp_path / "exp", name="evidence-exp")
    exp.task_protocol = AutoLamellaTaskProtocol()
    (tmp_path / "exp").mkdir(parents=True, exist_ok=True)
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    host.experiment = exp
    host.microscope = microscope
    return RecordingContext(host)


@pytest.fixture
def client(microscope, context):
    auth = AuthConfig(token="agent", dashboard_token="dashboard")
    auth.set_level(AgentLevel.DRIVE)
    app = build_server(microscope, app_context=context, auth=auth)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def _png(width=40, height=30, fmt="PNG") -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (200, 40, 160)).save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _recorded(context, kind):
    return [p for k, p in context.recorded if k == kind]


def _keep(client, **body):
    return client.post("/app/experiment/evidence", headers=AGENT, json=body)


def test_an_upload_is_saved_into_the_experiment(client, context):
    exp = context._session.experiment
    item = exp.positions[0].name
    marks = {"points": [{"x": 10, "y": 12, "label": "edge"}]}
    resp = _keep(client, upload=_png(), marks=marks, item_name=item)
    assert resp.status_code == 200, resp.text
    record = resp.json()
    folder = evidence.folder(os.path.realpath(exp.path))
    assert record["image"] == os.path.join(folder, f"{record['evidence_id']}.png")
    assert record["uploaded"] is True
    with Image.open(record["image"]) as saved:
        assert saved.format == "PNG" and saved.size == (40, 30)
    (kept,) = _recorded(context, evidence.EVIDENCE)
    assert kept["evidence_id"] == record["evidence_id"]
    assert kept["marks"]["points"][0]["label"] == "edge"
    assert kept["item_id"] == exp.positions[0].id


def test_paths_are_kept_relative_so_a_moved_experiment_finds_them(client, context):
    exp = context._session.experiment
    record = _keep(client, upload=_png()).json()
    with open(
        os.path.join(evidence.folder(exp.path), f"{record['evidence_id']}.json")
    ) as f:
        assert f.read().count('"evidence/') == 1


def test_a_large_upload_is_saved_at_most_2048_px(client):
    resp = _keep(client, upload=_png(3000, 1000, "JPEG"))
    assert resp.status_code == 200, resp.text
    with Image.open(resp.json()["image"]) as saved:
        assert saved.format == "PNG" and max(saved.size) == evidence.MAX_SIDE


@pytest.mark.parametrize(
    "body, status",
    [
        ({}, 422),  # no image
        ({"upload": _png(), "image": "/etc/hostname"}, 422),  # both
        ({"upload": "not base64!"}, 422),
        ({"upload": base64.b64encode(b"GIF89a").decode()}, 422),
        ({"upload": _png(), "marks": {"points": [{"x": "a"}]}}, 422),
        ({"upload": _png(), "item_name": "nope"}, 404),
        ({"image": "/etc/hostname"}, 403),
        # A crop is of an image in the experiment.
        ({"upload": _png(), "marks": {"crop": "A1", "points": []}}, 422),
    ],
)
def test_evidence_that_cannot_be_kept_is_refused(client, body, status):
    resp = client.post("/app/experiment/evidence", headers=AGENT, json=body)
    assert resp.status_code == status, resp.text


def test_an_upload_over_4_mb_is_refused(client):
    big = base64.b64encode(b"\x89PNG" + b"\0" * evidence.MAX_UPLOAD_BYTES).decode()
    assert _keep(client, upload=big).status_code == 413


def test_a_call_and_a_note_cite_the_evidence(client, context, microscope):
    kept = _keep(client, upload=_png()).json()
    cite = {**AGENT, EVIDENCE_HEADER: kept["evidence_id"]}
    note = client.post("/app/experiment/notes", headers=cite, json={"text": "accepted"})
    assert note.status_code == 200, note.text
    assert _recorded(context, NOTE)[-1]["evidence"] == kept["evidence_id"]

    recorded = []

    def record(kind, payload):
        recorded.append((kind, payload))

    microscope.session.record_signal.connect(record)
    try:
        moved = client.post(
            MOVE_RELATIVE,
            headers={**cite, "X-Agent-Tool": "stage_move_relative"},
            json=NUDGE,
        )
    finally:
        microscope.session.record_signal.disconnect(record)
    assert moved.status_code == 200, moved.text
    (called,) = [p for kind, p in recorded if kind == ACTION]
    assert called["evidence"] == kept["evidence_id"]


def test_citing_evidence_that_is_not_there_is_refused(client):
    cite = {**AGENT, EVIDENCE_HEADER: "00000000"}
    note = client.post("/app/experiment/notes", headers=cite, json={"text": "x"})
    assert note.status_code == 404


def test_the_evidence_header_round_trips():
    headers = {
        k.lower(): v for k, v in headers_for("x", {}, evidence="ab12cd34").items()
    }
    assert from_headers(headers).evidence == "ab12cd34"
    assert from_headers({}).evidence is None


def test_the_sidecar_keeps_a_file_and_the_call_cites_it(
    client, context, microscope, tmp_path, monkeypatch
):
    """An acting tool's evidence: the sidecar reads the file in the agent's
    working folder, keeps it in the experiment, and the call names it."""
    pytest.importorskip("mcp")
    from fibsem.ai.agent.mcp.sidecar import build_sidecar

    monkeypatch.setenv("FIBSEM_AGENT_WORKDIR", str(tmp_path))
    overlay = tmp_path / "overlay.png"
    overlay.write_bytes(base64.b64decode(_png()))
    client.headers.update(AGENT)
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
                    "evidence": {"file": str(overlay)},
                },
            )
        )
    finally:
        microscope.session.record_signal.disconnect(record)
    (kept,) = _recorded(context, evidence.EVIDENCE)
    assert kept["uploaded"] is True
    assert os.path.isfile(kept["image"])
    (called,) = [p for kind, p in recorded if kind == ACTION]
    assert called["evidence"] == kept["evidence_id"]
    # The arguments the server describes the call by leave the evidence out.
    assert "evidence" not in called.get("arguments", {})


def test_the_sidecar_reads_an_evidence_file_only_in_the_working_folder(
    tmp_path, monkeypatch
):
    """The sidecar runs outside code mode's sandbox, so it reads no file
    outside the agent's working folder, through a link or not."""
    pytest.importorskip("mcp")
    from fibsem.ai.agent.mcp.sidecar import _evidence_body, _Failure

    work, elsewhere = tmp_path / "work", tmp_path / "elsewhere"
    work.mkdir()
    elsewhere.mkdir()
    encoded = _png()
    png = base64.b64decode(encoded)
    (work / "overlay.png").write_bytes(png)
    (elsewhere / "secret.png").write_bytes(png)
    outside = [elsewhere / "secret.png", work / ".." / "x"]
    try:
        (work / "link.png").symlink_to(elsewhere / "secret.png")
        outside.append(work / "link.png")
    except OSError:  # Windows without the privilege to make links
        pass
    monkeypatch.setenv("FIBSEM_AGENT_WORKDIR", str(work))

    body = _evidence_body({"file": str(work / "overlay.png"), "source": "a.tif"})
    assert body == {"source": "a.tif", "upload": encoded}
    for path in outside:
        refused = _evidence_body({"file": str(path)})
        assert isinstance(refused, _Failure), path
        assert "outside your working folder" in str(refused)

    # Without code mode's folder, the working folder is where it runs.
    monkeypatch.delenv("FIBSEM_AGENT_WORKDIR")
    monkeypatch.chdir(elsewhere)
    assert "upload" in _evidence_body({"file": "secret.png"})


def test_every_acting_tool_takes_evidence_and_no_read_does():
    by_name = {t.name: t for t in CATALOG}
    assert by_name["stage_move_relative"].takes_evidence
    assert by_name["request_create"].takes_evidence
    assert by_name["experiment_add_note"].takes_evidence
    assert by_name["proposal_suggest"].takes_evidence
    assert not by_name["stage_get_position"].takes_evidence
    assert "image" not in by_name["request_create"].params
    assert "image" not in by_name["proposal_suggest"].params


def test_a_suggestion_and_a_decision_keep_the_evidence_they_cite():
    suggestion = Suggestion(actor="agent:x", reason="r", evidence="ab12cd34")
    assert Suggestion.from_dict(suggestion.to_dict()).evidence == "ab12cd34"
    decision = Decision(
        outcome=DecisionOutcome.Confirmed, actor="agent:x", evidence="ab12cd34"
    )
    assert decision.to_dict()["evidence"] == "ab12cd34"
    assert Decision.from_dict(decision.to_dict()).evidence == "ab12cd34"
    # Saved only when there is one: old files read the same.
    assert "evidence" not in Decision(DecisionOutcome.Confirmed, "agent:x").to_dict()


def test_a_reference_image_is_cited_as_proposal_list_gives_it(client, context):
    """Round 10 (FIB-1494): proposal_list gives a reference image relative to
    its item's folder, and evidence took only absolute paths."""
    exp = context._session.experiment
    item = exp.positions[0].name
    folder = os.path.join(exp.path, item)
    os.makedirs(folder, exist_ok=True)
    Image.new("L", (16, 16)).save(os.path.join(folder, "ref_final_ib.png"))
    for body in (
        {"image": "ref_final_ib.png", "item_name": item},
        {"image": "ref_final_ib.png"},  # the one item folder that holds it
        {"image": os.path.join(item, "ref_final_ib.png")},
    ):
        resp = _keep(client, **body)
        assert resp.status_code == 200, (body, resp.text)
    missing = _keep(client, image="nowhere.png")
    assert missing.status_code == 422
    assert "nowhere.png" in missing.json()["detail"]["message"]


def test_only_a_beam_image_is_a_frame_to_draw_on():
    """A picture the agent rendered (PNG) is shown as it is, like an upload."""
    from fibsem.workflows.evidence import frame_path

    assert frame_path({"image": "/exp/overview.tif"}) == "/exp/overview.tif"
    assert frame_path({"image": "/exp/markers/overview.png"}) is None
    assert frame_path({"image": "/exp/a.png", "uploaded": True}) is None
    assert frame_path(None) is None


def test_marks_read_off_a_crop_are_kept_in_the_image_s_pixels(client, context):
    exp = context._session.experiment
    path = os.path.join(str(exp.path), "wide.png")
    Image.new("RGB", (3072, 2048)).save(path)
    marks = {"crop": "B2", "points": [{"x": 10, "y": 20}]}
    record = _keep(client, image=path, marks=marks).json()
    point = record["marks"]["points"][0]
    assert (point["x"], point["y"]) == (778, 703)
    bad = _keep(client, image=path, marks={"crop": "Q1", "points": []})
    assert bad.status_code == 422 and "A1 to D3" in bad.json()["detail"]["message"]
