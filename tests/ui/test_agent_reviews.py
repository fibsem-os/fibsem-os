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
    POINT_OF_INTEREST,
    Decision,
    DecisionOutcome,
    Proposal,
)
from fibsem.applications.autolamella.server import AgentContext  # noqa: E402
from fibsem.applications.autolamella.server.events import EventBuffer  # noqa: E402
from fibsem.applications.autolamella.structures import (  # noqa: E402
    Attention,
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaTaskState,
    AutoLamellaTaskStatus,
    AutoLamellaWorkflowConfig,
    Experiment,
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
RUN = "run-1"  # the run the fixture's proposal is from
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
                    name=SETUP, required=True, attention=Attention.review
                ),
                AutoLamellaTaskDescription(name=ROUGH, required=True, requires=[SETUP]),
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
        kind=POINT_OF_INTEREST,
        values={"poi": Point(0.0, 0.0)},
        provenance={
            "proposer": "centre-of-image",
            "reference_image": ref + ".tif",
            "task_id": RUN,
        },
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
    assert review["task_id"] == RUN
    assert review["kind"] == POINT_OF_INTEREST
    assert review["values"] == {"poi": {"x": 0.0, "y": 0.0}}
    assert review["gated"] is True
    assert review["waiting_on"] == [ROUGH]
    assert review["reference_image"]["width"] > 0
    assert review["reference_image"]["image_b64_jpeg"]


def test_to_check_is_listed_and_an_agent_look_does_not_clear_it(ui, qapp):
    """An agent's acknowledgement is recorded, writes nothing, and leaves the
    row to check: it is not automatic (a decider acted) and not a person
    (someone may still want to look). A person's look clears it."""
    lamella = ui.experiment.positions[0]
    proposal = lamella.proposals[SETUP]
    proposal.decisions.append(
        Decision(
            outcome=DecisionOutcome.Confirmed,
            author="auto:centre-of-image",
            values=dict(proposal.values),
        )
    )
    with _client(ui) as client:
        doc = client.get("/app/reviews", headers=AUTH).json()
        assert doc["reviews"] == [], "nothing waits on it"
        (check,) = doc["to_check"]
        assert check["item_id"] == lamella.id and check["task_name"] == SETUP
        assert check["decisions"][-1]["author"] == "auto:centre-of-image"

        resp = _post_on_worker(
            qapp,
            client,
            "/app/decide",
            {
                "item_id": lamella.id,
                "task_name": SETUP,
                "task_id": RUN,
                "outcome": "Confirmed",
                "author": "test-model",
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["synced_tasks"] == [], "an acknowledgement writes nothing"
        (check,) = client.get("/app/reviews", headers=AUTH).json()["to_check"]
        assert check["decisions"][-1]["author"] == "agent:test-model"
    assert proposal.to_check, "an agent looked; a person has not"
    assert (
        str(proposal.current.author) == "agent:test-model"
        and proposal.current.values == {}
    )
    ui.experiment._decide(
        lamella.id,
        SETUP,
        Decision(outcome=DecisionOutcome.Confirmed, author="human:op", task_id=RUN),
    )
    assert not proposal.to_check


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
                "task_id": RUN,
                "outcome": "Confirmed",
                "values": {"poi": {"x": 2e-6, "y": -1e-6}},
                "author": "test-model",
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["applied"] is True
        assert body["delta"]["poi"] == {"x": 2e-6, "y": -1e-6}
        assert lamella.proposals[SETUP].current.via == "server"
        assert ROUGH in body["synced_tasks"]
        assert lamella.poi == Point(2e-6, -1e-6)
        proposal = lamella.proposals[SETUP]
        assert proposal.current.outcome is DecisionOutcome.Confirmed
        assert str(proposal.current.author) == "agent:test-model"
        assert client.get("/app/reviews", headers=AUTH).json()["reviews"] == []
    kinds = [e["kind"] for e in buffer.events_since(0)["events"]]
    assert "review_decided" in kinds


def test_reject_needs_a_reason_and_fails_the_task(ui, qapp):
    lamella = ui.experiment.positions[0]
    lamella.task_history.append(
        AutoLamellaTaskState(name=SETUP, status=AutoLamellaTaskStatus.AwaitingDecision)
    )
    with _client(ui) as client:
        resp = _post_on_worker(
            qapp,
            client,
            "/app/decide",
            {
                "item_id": lamella.id,
                "task_name": SETUP,
                "task_id": RUN,
                "outcome": "Rejected",
            },
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
                "task_id": RUN,
                "outcome": "Rejected",
                "reason": "no usable site",
            },
        )
        assert resp.status_code == 200, resp.text
    assert lamella.task_history[-1].status is AutoLamellaTaskStatus.Failed
    assert lamella.task_history[-1].status_message == (
        "Rejected by agent · remote: no usable site"
    )
    assert not lamella.is_failure


def test_decide_refuses_what_is_not_pending_or_is_running(ui, qapp):
    from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus

    lamella = ui.experiment.positions[0]
    with _client(ui) as client:
        resp = _post_on_worker(
            qapp,
            client,
            "/app/decide",
            {
                "item_id": lamella.id,
                "task_name": "Nope",
                "task_id": RUN,
                "outcome": "Confirmed",
            },
        )
        assert resp.status_code == 409
        assert resp.json()["detail"]["error_type"] == "not_pending"

        # a look at an earlier task's result, while a later one runs, lands:
        # only the task being decided, or a value, is refused under a run
        lamella.task_state.name = ROUGH
        lamella.task_state.status = AutoLamellaTaskStatus.InProgress
        resp = _post_on_worker(
            qapp,
            client,
            "/app/decide",
            {
                "item_id": lamella.id,
                "task_name": SETUP,
                "task_id": RUN,
                "outcome": "Confirmed",
                "values": {"poi": {"x": 1e-6, "y": 0.0}},
            },
        )
        assert resp.status_code == 409
        assert resp.json()["detail"]["error_type"] == "running"

        # now the run being decided is the run in progress: a stop, not a
        # decision, even though it carries no values
        lamella.task_state.name = SETUP
        lamella.task_state.task_id = RUN
        resp = _post_on_worker(
            qapp,
            client,
            "/app/decide",
            {
                "item_id": lamella.id,
                "task_name": SETUP,
                "task_id": RUN,
                "outcome": "Confirmed",
            },
        )
        assert resp.status_code == 409
        assert resp.json()["detail"]["error_type"] == "running"
    assert lamella.proposals[SETUP].pending


def test_decide_refuses_a_missing_or_stale_run_and_an_edit_disguised_as_a_look(
    ui, qapp
):
    """FIB-1003 through the route: a missing task_id and an acknowledgement
    carrying values are 422; a run that has been replaced is 409. None writes."""
    lamella = ui.experiment.positions[0]
    base = {"item_id": lamella.id, "task_name": SETUP, "outcome": "Confirmed"}
    poi = {"poi": {"x": 4e-6, "y": 0.0}}
    with _client(ui) as client:
        resp = _post_on_worker(qapp, client, "/app/decide", {**base, "values": poi})
        assert resp.status_code == 422
        assert resp.json()["detail"]["error_type"] == "missing_field"

        resp = _post_on_worker(
            qapp,
            client,
            "/app/decide",
            {**base, "task_id": "run-0", "values": poi},
        )
        assert resp.status_code == 409, resp.text
        assert resp.json()["detail"]["error_type"] == "stale_review"

        resp = _post_on_worker(qapp, client, "/app/decide", {**base, "task_id": RUN})
        assert resp.status_code == 422, resp.text
        assert resp.json()["detail"]["error_type"] == "invalid_value"
        assert "needs its values" in resp.json()["detail"]["message"]

        assert lamella.poi == Point(0.0, 0.0)
        assert lamella.proposals[SETUP].pending

        resp = _post_on_worker(
            qapp, client, "/app/decide", {**base, "task_id": RUN, "values": poi}
        )
        assert resp.status_code == 200, resp.text
        resp = _post_on_worker(
            qapp,
            client,
            "/app/decide",
            {**base, "task_id": RUN, "values": {"poi": {"x": 9e-6, "y": 9e-6}}},
        )
        assert resp.status_code == 422, resp.text
        assert "already decided" in resp.json()["detail"]["message"]
    assert lamella.poi == Point(4e-6, 0.0)
