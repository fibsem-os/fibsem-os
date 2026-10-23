"""The App Server's proposal surface: GET /app/workflow/proposals lists the
questions a task is stopped on, POST
/app/workflow/decide answers one through the same Experiment.decide the
operator's Continue goes through -- on the main thread, blocking the server
worker until applied, with the agent recorded as author.
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

from fibsem.applications.autolamella.server import AutoLamellaAppContext  # noqa: E402
from fibsem.applications.autolamella.structures import (
    Experiment,
)
from fibsem.applications.autolamella.ui.window import AutoLamellaWindow  # noqa: E402
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
from fibsem.workflows.attention import Attention
from fibsem.workflows.events import EventBuffer  # noqa: E402
from fibsem.workflows.proposals import (  # noqa: E402
    POINT_OF_INTEREST,  # noqa: E402
    DecisionOutcome,
    Proposal,
)
from fibsem.workflows.status import TaskStatus  # noqa: E402
from tests.fixtures.main_window import close_window  # noqa: E402
from tests.proposal_records import set_proposals  # noqa: E402
from tests.protocols import planned, task_protocol

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
    widget = AutoLamellaWindow()
    monkeypatch.setattr(
        widget.system_widget,
        "load_configuration",
        lambda configuration_name=None: arctis_config,
    )
    widget.system_widget.connect_to_microscope()
    exp = Experiment(path=tmp_path / "exp", name="review-exp")
    exp.task_protocol = task_protocol(
        planned(name=SETUP, required=True, attention=Attention.supervised),
        planned(name=ROUGH, required=True, requires=[SETUP]),
    )
    (tmp_path / "exp").mkdir(parents=True, exist_ok=True)
    exp.add_new_lamella(
        MicroscopeState(), EventedDict({ROUGH: MillRoughTaskConfig(task_name=ROUGH)})
    )
    lamella = exp.positions[0]
    lamella.path.mkdir(parents=True, exist_ok=True)
    ref = os.path.join(str(lamella.path), "ref_setup_ib")
    _fib_image().save(ref)
    # A question Setup is stopped on: the task is running, asking for its
    # point of interest.
    lamella.task_state.name = SETUP
    lamella.task_state.status = TaskStatus.InProgress
    lamella.task_state.task_id = RUN
    question = Proposal(
        kind=POINT_OF_INTEREST,
        values={"poi": Point(0.0, 0.0)},
        provenance={
            "proposer": "centre-of-image",
            "reference_image": ref + ".tif",
            "task_id": RUN,
        },
    )
    question.asking = True
    set_proposals(lamella, SETUP, [question])
    widget.experiment = exp
    yield widget
    if widget.microscope is not None:
        widget.microscope.disconnect()
    close_window(widget)
    widget.deleteLater()
    qapp.processEvents()


def _finished(lamella) -> None:
    """Setup ended with its question unanswered: the value is open, used as
    proposed, and nobody is stopped on it."""
    lamella.task_state.status = TaskStatus.Completed
    lamella.proposal(SETUP).asking = False


def _client(ui, buffer=None):
    app = build_server(
        ui.microscope,
        app_context=AutoLamellaAppContext(ui.session, event_buffer=buffer),
        auth=AuthConfig.generate(level="drive", token=TOKEN),
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


def test_the_list_has_the_pending_proposal_with_its_image(ui):
    with _client(ui) as client:
        resp = client.get("/app/workflow/proposals", headers=AUTH)
    assert resp.status_code == 200, resp.text
    doc = resp.json()
    assert doc["available"] is True
    (listed,) = doc["proposals"]
    lamella = ui.experiment.positions[0]
    assert listed["item_id"] == lamella.id
    assert listed["item_name"] == lamella.name
    assert listed["task_name"] == SETUP
    assert listed["task_id"] == RUN
    assert listed["kind"] == POINT_OF_INTEREST
    assert listed["values"] == {"poi": {"name": None, "x": 0.0, "y": 0.0}}
    assert listed["reference_image"]["width"] > 0
    assert listed["reference_image"]["image_b64_jpeg"]


def test_confirm_from_a_worker_writes_through_as_the_agent(ui, qapp):
    buffer = EventBuffer()
    lamella = ui.experiment.positions[0]
    with _client(ui, buffer=buffer) as client:
        resp = _post_on_worker(
            qapp,
            client,
            "/app/workflow/decide",
            {
                "item_id": lamella.id,
                "task_name": SETUP,
                "task_id": RUN,
                "outcome": "Confirmed",
                "values": {"poi": {"x": 2e-6, "y": -1e-6}},
                "actor": "test-model",
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["applied"] is True
        assert "delta" not in body
        assert lamella.proposal(SETUP).current.via == "server"
        assert ROUGH in body["synced_tasks"]
        assert lamella.poi == Point(2e-6, -1e-6)
        proposal = lamella.proposal(SETUP)
        assert proposal.current.outcome is DecisionOutcome.Confirmed
        assert str(proposal.current.actor) == "agent:test-model"
        assert (
            client.get("/app/workflow/proposals", headers=AUTH).json()["proposals"]
            == []
        )


def test_reject_needs_a_reason_and_is_handed_to_the_task(ui, qapp):
    """A reject is the answer the stopped task reads; the task fails itself
    on it, so the decision moves no status."""
    lamella = ui.experiment.positions[0]
    with _client(ui) as client:
        resp = _post_on_worker(
            qapp,
            client,
            "/app/workflow/decide",
            {
                "item_id": lamella.id,
                "task_name": SETUP,
                "task_id": RUN,
                "outcome": "Rejected",
            },
        )
        assert resp.status_code == 422
        assert lamella.proposal(SETUP).pending

        resp = _post_on_worker(
            qapp,
            client,
            "/app/workflow/decide",
            {
                "item_id": lamella.id,
                "task_name": SETUP,
                "task_id": RUN,
                "outcome": "Rejected",
                "reason": "no usable site",
            },
        )
        assert resp.status_code == 200, resp.text
    decision = lamella.proposal(SETUP).current
    assert decision.outcome is DecisionOutcome.Rejected
    assert decision.reason == "no usable site"
    assert lamella.task_state.status is TaskStatus.InProgress
    assert not lamella.is_failure


def test_decide_refuses_what_is_not_pending_or_is_running(ui, qapp):
    lamella = ui.experiment.positions[0]
    _finished(lamella)
    with _client(ui) as client:
        resp = _post_on_worker(
            qapp,
            client,
            "/app/workflow/decide",
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
        lamella.task_state.status = TaskStatus.InProgress
        resp = _post_on_worker(
            qapp,
            client,
            "/app/workflow/decide",
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
            "/app/workflow/decide",
            {
                "item_id": lamella.id,
                "task_name": SETUP,
                "task_id": RUN,
                "outcome": "Confirmed",
            },
        )
        assert resp.status_code == 409
        assert resp.json()["detail"]["error_type"] == "running"
    assert lamella.proposal(SETUP).pending


def test_decide_refuses_a_missing_or_stale_run_and_a_second_decision(ui, qapp):
    """FIB-1003 through the route: a missing task_id and a decision on a
    proposal already decided are 422; a run that has been replaced is 409.
    None writes."""
    lamella = ui.experiment.positions[0]
    _finished(lamella)
    base = {"item_id": lamella.id, "task_name": SETUP, "outcome": "Confirmed"}
    poi = {"poi": {"x": 4e-6, "y": 0.0}}
    with _client(ui) as client:
        resp = _post_on_worker(
            qapp, client, "/app/workflow/decide", {**base, "values": poi}
        )
        assert resp.status_code == 422
        assert resp.json()["detail"]["error_type"] == "missing_field"

        resp = _post_on_worker(
            qapp,
            client,
            "/app/workflow/decide",
            {**base, "task_id": "run-0", "values": poi},
        )
        assert resp.status_code == 409, resp.text
        assert resp.json()["detail"]["error_type"] == "stale_proposal"

        resp = _post_on_worker(
            qapp, client, "/app/workflow/decide", {**base, "task_id": RUN}
        )
        assert resp.status_code == 422, resp.text
        assert resp.json()["detail"]["error_type"] == "invalid_value"
        assert "needs its values" in resp.json()["detail"]["message"]

        assert lamella.poi == Point(0.0, 0.0)
        assert lamella.proposal(SETUP).pending

        resp = _post_on_worker(
            qapp,
            client,
            "/app/workflow/decide",
            {**base, "task_id": RUN, "values": poi},
        )
        assert resp.status_code == 200, resp.text
        resp = _post_on_worker(
            qapp,
            client,
            "/app/workflow/decide",
            {**base, "task_id": RUN, "values": {"poi": {"x": 9e-6, "y": 9e-6}}},
        )
        assert resp.status_code == 422, resp.text
        assert "already decided" in resp.json()["detail"]["message"]
    assert lamella.poi == Point(4e-6, 0.0)


# ---------------------------------------------------------------------------
# A decision names the proposal (FIB-1025)
# ---------------------------------------------------------------------------


def test_the_list_gives_the_proposals_own_id_to_pass_back(ui):
    lamella = ui.experiment.positions[0]
    with _client(ui) as client:
        waiting = client.get("/app/workflow/proposals", headers=AUTH).json()[
            "proposals"
        ]

    assert waiting[0]["proposal_id"] == lamella.proposal(SETUP).id
    assert waiting[0]["task_id"] == RUN, "the run is still there, as a fact about it"


def test_decide_by_the_proposals_id_needs_no_run(ui, qapp):
    lamella = ui.experiment.positions[0]
    proposal = lamella.proposal(SETUP)
    body = {
        "item_id": lamella.id,
        "task_name": SETUP,
        "outcome": "Confirmed",
        "proposal_id": proposal.id,
        "values": {"poi": {"x": 4e-6, "y": 0.0}},
    }
    with _client(ui) as client:
        resp = _post_on_worker(qapp, client, "/app/workflow/decide", body)

    assert resp.status_code == 200, resp.text
    # Only what is set (round 11 R7): no empty save_error, reason or error_type.
    assert set(resp.json()) <= {"applied", "available", "synced_tasks"}
    assert lamella.poi == Point(4e-6, 0.0)
    assert proposal.current.proposal_id == proposal.id
    assert proposal.current.task_id == RUN, "filled in from the proposal it named"


def test_decide_by_an_id_that_is_no_longer_on_the_record_is_stale(ui, qapp):
    """What the run could not tell: the same run, another proposal. Naming the
    run as well does not excuse it -- the id decides."""
    lamella = ui.experiment.positions[0]
    shown = lamella.proposal(SETUP).id
    # the task asked again, in the same run
    again = Proposal(
        kind=POINT_OF_INTEREST,
        values={"poi": Point(1e-6, 0.0)},
        provenance=dict(lamella.proposal(SETUP).provenance),
    )
    again.asking = True
    set_proposals(lamella, SETUP, [again])
    body = {
        "item_id": lamella.id,
        "task_name": SETUP,
        "outcome": "Confirmed",
        "proposal_id": shown,
        "task_id": RUN,
        "values": {"poi": {"x": 4e-6, "y": 0.0}},
    }
    with _client(ui) as client:
        resp = _post_on_worker(qapp, client, "/app/workflow/decide", body)

    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"]["error_type"] == "stale_proposal"
    assert "asked again since you looked" in resp.json()["detail"]["message"]
    assert lamella.proposal(SETUP).pending and lamella.poi == Point(0.0, 0.0)


def test_withdrawn_cannot_be_posted_as_a_decision(ui, qapp):
    """The route takes its outcome by name, and Withdrawn is a name."""
    lamella = ui.experiment.positions[0]
    body = {
        "item_id": lamella.id,
        "task_name": SETUP,
        "outcome": "Withdrawn",
        "proposal_id": lamella.proposal(SETUP).id,
        "reason": "taking it back",
    }
    with _client(ui) as client:
        resp = _post_on_worker(qapp, client, "/app/workflow/decide", body)

    assert resp.status_code == 422, resp.text
    assert lamella.proposal(SETUP).pending
