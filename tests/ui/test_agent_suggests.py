"""POST /app/workflow/proposals/suggest: at Assist an agent suggests a decision
on a pending proposal. Nothing is decided; the suggestion is on the proposal,
in proposal_list, for the operator to approve or decline."""

import pytest

pytest.importorskip("PyQt5")

from fastapi.testclient import TestClient  # noqa: E402

from fibsem.applications.autolamella.server import AutoLamellaAppContext  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from tests.ui.test_agent_proposals import (  # noqa: E402,F401 - the fixture
    AUTH,
    SETUP,
    TOKEN,
    _post_on_worker,
    ui,
)

PATH = "/app/workflow/proposals/suggest"


def _client(ui, level):
    app = build_server(
        ui.microscope,
        app_context=AutoLamellaAppContext(ui.session),
        auth=AuthConfig.generate(level=level, token=TOKEN),
    )
    return TestClient(app, raise_server_exceptions=False)


def _body(ui, **kw):
    lamella = ui.experiment.positions[0]
    return {
        "item_id": lamella.id,
        "task_name": SETUP,
        "proposal_id": lamella.proposal(SETUP).id,
        "reason": "the lamella's edge is 1 µm right of centre",
        "values": {"poi": {"x": 1e-6, "y": 0.0}},
        "actor": "claude",
        **kw,
    }


def test_at_assist_the_suggestion_goes_on_the_proposal(qapp, ui):
    client = _client(ui, "assist")

    response = _post_on_worker(qapp, client, PATH, _body(ui))

    assert response.status_code == 200, response.text
    proposal = ui.experiment.positions[0].proposal(SETUP)
    assert proposal.pending
    assert str(proposal.suggestion.actor) == "agent:claude"
    assert response.json()["suggestion_id"] == proposal.suggestion.id
    listed = client.get("/app/workflow/proposals", headers=AUTH).json()
    assert listed["proposals"][0]["suggestion"]["reason"].startswith("the lamella")


def test_at_watch_it_is_refused(qapp, ui):
    response = _post_on_worker(qapp, _client(ui, "watch"), PATH, _body(ui))

    assert response.status_code == 403
    assert ui.experiment.positions[0].proposal(SETUP).suggestion is None


def test_a_suggestion_says_why(qapp, ui):
    response = _post_on_worker(qapp, _client(ui, "assist"), PATH, _body(ui, reason=" "))

    assert response.status_code == 422
    assert response.json()["detail"]["error_type"] == "missing_field"


def test_one_on_a_proposal_no_longer_there_is_stale(qapp, ui):
    response = _post_on_worker(
        qapp, _client(ui, "assist"), PATH, _body(ui, proposal_id="gone")
    )

    assert response.status_code == 409
    assert response.json()["detail"]["error_type"] == "stale_proposal"


def test_a_value_the_kind_does_not_carry_is_refused(qapp, ui):
    response = _post_on_worker(
        qapp,
        _client(ui, "assist"),
        PATH,
        _body(ui, values={"alignment_area": {"left": 0.1}}),
    )

    assert response.status_code == 422
    assert ui.experiment.positions[0].proposal(SETUP).suggestion is None
