"""The agent server's review surface: GET /app/reviews is the tab's inbox,
POST /app/decide is the tab's Confirm and Reject, through the same
Experiment.decide -- on the main thread, blocking the server worker until
applied, with the agent recorded as author.
"""

import os
import threading
import time

import numpy as np
import pytest
from psygnal.containers import EventedDict

pytest.importorskip("PyQt5")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from fibsem.applications.autolamella.proposals import (  # noqa: E402
    MILLING_SETUP,
    DecisionOutcome,
    Proposal,
)
from fibsem.applications.autolamella.server import AgentContext  # noqa: E402
from fibsem.applications.autolamella.server.events import EventBuffer  # noqa: E402
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaWorkflowConfig,
    Experiment,
    Verdict,
)
from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI  # noqa: E402
from fibsem.applications.autolamella.workflows.tasks.rough import (  # noqa: E402
    MillRoughTaskConfig,
)
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.structures import (  # noqa: E402
    BeamType,
    FibsemImage,
    FibsemImageMetadata,
    FibsemStagePosition,
    ImageSettings,
    MicroscopeState,
    Point,
)

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
SETUP = "Setup Lamella Position"
ROUGH = "Rough Milling"


def _fib_image() -> FibsemImage:
    metadata = FibsemImageMetadata(
        image_settings=ImageSettings(beam_type=BeamType.ION, hfw=512e-7),
        pixel_size=Point(1e-7, 1e-7),
        microscope_state=MicroscopeState(stage_position=FibsemStagePosition()),
    )
    return FibsemImage(data=np.zeros((512, 512), dtype=np.uint8), metadata=metadata)


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
    exp = Experiment(path=tmp_path / "exp", name="review-exp")
    exp.task_protocol = AutoLamellaTaskProtocol(
        workflow_config=AutoLamellaWorkflowConfig(
            tasks=[
                AutoLamellaTaskDescription(
                    name=SETUP, supervise=False, required=True, review=True
                ),
                AutoLamellaTaskDescription(
                    name=ROUGH, supervise=False, required=True, requires=[SETUP]
                ),
            ]
        )
    )
    (tmp_path / "exp").mkdir(parents=True, exist_ok=True)
    exp.add_new_lamella(
        MicroscopeState(), EventedDict({ROUGH: MillRoughTaskConfig(task_name=ROUGH)})
    )
    lamella = exp.positions[0]
    lamella.path.mkdir(parents=True, exist_ok=True)
    ref = os.path.join(str(lamella.path), "ref_setup_ib")
    _fib_image().save(ref)
    lamella.proposals[SETUP] = Proposal(
        kind=MILLING_SETUP,
        values={"poi": Point(0.0, 0.0)},
        provenance={"proposer": "centre-of-image", "reference_image": ref + ".tif"},
    )
    widget.experiment = exp
    yield widget
    if widget.microscope is not None:
        widget.microscope.disconnect()
    widget.close()
    widget.deleteLater()
    qapp.processEvents()


def _client(ui, buffer=None):
    app = build_server(
        ui.microscope,
        app_context=AgentContext(ui, event_buffer=buffer),
        auth=AuthConfig.generate(arm_control=True, token=TOKEN),
    )
    return TestClient(app, raise_server_exceptions=False)


def _spin_until(qapp, predicate, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not reached")
        qapp.processEvents()
        time.sleep(0.01)


def _post_on_worker(qapp, client, path, body):
    """POST from a worker thread while the GUI loop spins: decide() marshals
    onto the main thread and blocks the worker until it has run."""
    posted = {}

    def target():
        posted["response"] = client.post(path, headers=AUTH, json=body)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    _spin_until(qapp, lambda: "response" in posted)
    return posted["response"]


def test_reviews_lists_the_pending_proposal_with_its_image(ui):
    with _client(ui) as client:
        resp = client.get("/app/reviews", headers=AUTH)
    assert resp.status_code == 200, resp.text
    doc = resp.json()
    assert doc["available"] is True
    (review,) = doc["reviews"]
    lamella = ui.experiment.positions[0]
    assert review["item_id"] == lamella.id
    assert review["item_name"] == lamella.name
    assert review["task_name"] == SETUP
    assert review["kind"] == MILLING_SETUP
    assert review["values"] == {"poi": {"x": 0.0, "y": 0.0}}
    assert review["gating"] is True
    assert review["waiting_on"] == [ROUGH]
    assert review["reference_image"]["width"] > 0
    assert review["reference_image"]["image_b64_jpeg"]


def test_confirm_from_a_worker_writes_through_as_the_agent(ui, qapp):
    buffer = EventBuffer()
    lamella = ui.experiment.positions[0]
    with _client(ui, buffer=buffer) as client:
        resp = _post_on_worker(
            qapp,
            client,
            "/app/decide",
            {
                "item_id": lamella.id,
                "task_name": SETUP,
                "outcome": "Confirmed",
                "values": {"poi": {"x": 2e-6, "y": -1e-6}},
                "author": "test-model",
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["applied"] is True
        assert body["delta"]["poi"] == {"x": 2e-6, "y": -1e-6}
        assert ROUGH in body["synced_tasks"]
        assert lamella.poi == Point(2e-6, -1e-6)
        proposal = lamella.proposals[SETUP]
        assert proposal.current.outcome is DecisionOutcome.Confirmed
        assert proposal.current.author == "agent:test-model"
        assert client.get("/app/reviews", headers=AUTH).json()["reviews"] == []
    kinds = [e["kind"] for e in buffer.events_since(0)["events"]]
    assert "review_decided" in kinds


def test_reject_needs_a_reason_and_retires_the_item(ui, qapp):
    lamella = ui.experiment.positions[0]
    with _client(ui) as client:
        resp = _post_on_worker(
            qapp,
            client,
            "/app/decide",
            {"item_id": lamella.id, "task_name": SETUP, "outcome": "Rejected"},
        )
        assert resp.status_code == 422
        assert lamella.proposals[SETUP].pending

        resp = _post_on_worker(
            qapp,
            client,
            "/app/decide",
            {
                "item_id": lamella.id,
                "task_name": SETUP,
                "outcome": "Rejected",
                "reason": "no usable site",
            },
        )
        assert resp.status_code == 200, resp.text
    assert lamella.is_failure
    assert lamella.quality.verdict is Verdict.FAILED
    assert lamella.quality.author == "agent:remote"


def test_decide_refuses_what_is_not_pending_or_is_running(ui, qapp):
    from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus

    lamella = ui.experiment.positions[0]
    with _client(ui) as client:
        resp = _post_on_worker(
            qapp,
            client,
            "/app/decide",
            {"item_id": lamella.id, "task_name": "Nope", "outcome": "Confirmed"},
        )
        assert resp.status_code == 409
        assert resp.json()["detail"]["error_type"] == "not_pending"

        lamella.task_state.name = ROUGH
        lamella.task_state.status = AutoLamellaTaskStatus.InProgress
        resp = _post_on_worker(
            qapp,
            client,
            "/app/decide",
            {"item_id": lamella.id, "task_name": SETUP, "outcome": "Confirmed"},
        )
        assert resp.status_code == 409
        assert resp.json()["detail"]["error_type"] == "running"
    assert lamella.proposals[SETUP].pending
