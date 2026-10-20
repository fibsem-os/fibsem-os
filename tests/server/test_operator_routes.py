"""The operator token and the operator's routes: what the agent chip and card do,
from the web page or a script, and nothing the agent's token can reach."""

import base64
import os
import socket
import time

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fibsem.acting import OPERATOR  # noqa: E402
from fibsem.ai.agent.levels import AgentLevel  # noqa: E402
from fibsem.applications.autolamella.server.context import (  # noqa: E402
    AutoLamellaAppContext,
)
from fibsem.server import AuthConfig  # noqa: E402
from fibsem.server.hosting import AppServerHost  # noqa: E402
from fibsem.server.operator_token import (  # noqa: E402
    load_operator_token,
    new_operator_token,
)
from fibsem.workflows.actions import NOTE  # noqa: E402
from fibsem.workflows.session import AppSession  # noqa: E402
from tests.autolamella.test_select_position_asks import (  # noqa: E402, F401
    SETUP,
    _experiment,
    microscope,
)

os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def server(microscope, tmp_path):  # noqa: F811
    """An App Server over a windowless session, the agent at Watch."""
    session = AppSession()
    session.microscope = microscope
    session.experiment = _experiment(tmp_path / "exp", microscope)
    session.start_recording()
    session.agent_auth = AuthConfig.generate(level=AgentLevel.WATCH)
    session.app_server = AppServerHost(
        session,
        lambda buffer: AutoLamellaAppContext(session, event_buffer=buffer),
        port=_free_port(),
        discovery_path=tmp_path / "app-server.json",
    )
    assert session.app_server.start()
    with httpx.Client(base_url=session.app_server.url, timeout=30) as client:
        client.session = session
        client.config = session.agent_auth
        client.operator = _bearer(session.agent_auth.operator_token)
        client.agent = _bearer(session.agent_auth.token)
        yield client
    session.app_server.stop()
    session.stop_recording()


def _notes(server):
    events = server.get("/app/events", headers=server.operator).json()["events"]
    return [e for e in events if e["kind"] == NOTE]


def test_the_token_is_kept_between_launches(tmp_path):
    path = tmp_path / "operator-token.json"
    token = load_operator_token(path)
    assert load_operator_token(path) == token
    assert new_operator_token(path) != token
    assert load_operator_token(path) != token


def test_the_server_takes_the_install_token(server, tmp_path):
    assert server.config.operator_token == load_operator_token(
        tmp_path / "operator-token.json"
    )


@pytest.mark.parametrize(
    "who, status", [("agent", 403), ("dashboard", 403), ("nobody", 401)]
)
def test_only_the_operator_reaches_the_operator_routes(server, who, status):
    headers = {
        "agent": server.agent,
        "dashboard": _bearer(server.config.dashboard_token),
        "nobody": _bearer("wrong"),
    }[who]
    answer = server.put("/app/agent/level", headers=headers, json={"level": "drive"})
    assert answer.status_code == status, answer.text
    assert server.config.level is AgentLevel.WATCH


def test_the_operator_reads_but_does_not_reach_the_agents_routes(server):
    assert server.get("/capabilities", headers=server.operator).status_code == 200
    assert server.get("/app/events", headers=server.operator).status_code == 200
    answer = server.post(
        "/app/workflow/stop", headers=server.operator, json={"reason": "test"}
    )
    assert answer.status_code == 403
    assert answer.json()["detail"]["error_type"] == "operator_reach"
    # Not served here: the page reads why.
    answer = server.get("/microscope", headers=server.operator)
    assert answer.json()["detail"]["error_type"] == "microscope_off"
    # Reading is not the agent being heard from.
    assert server.config.seconds_since_seen() is None


def test_the_operator_reads_what_the_microscope_is(microscope, tmp_path):  # noqa: F811
    session = AppSession()
    session.microscope = microscope
    session.agent_auth = AuthConfig.generate(level=AgentLevel.WATCH)
    host = AppServerHost(
        session,
        lambda buffer: AutoLamellaAppContext(session, event_buffer=buffer),
        port=_free_port(),
        discovery_path=tmp_path / "app-server.json",
        serve_microscope=True,
    )
    assert host.start()
    operator = _bearer(session.agent_auth.operator_token)
    try:
        with httpx.Client(base_url=host.url, timeout=30) as client:
            # What it is and its state: the page's Microscope panel.
            assert client.get("/microscope", headers=operator).json()["manufacturer"]
            state = client.get("/microscope/state", headers=operator).json()
            assert "stage_position" in state["microscope_state"]
            # Its devices, and every write and command, stay the agent's.
            for method, path in (
                ("GET", "/microscope/devices"),
                ("GET", "/microscope/devices/stage/values"),
                ("PUT", "/microscope/state"),
            ):
                answer = client.request(method, path, headers=operator, json={})
                assert answer.status_code == 403, path
            assert session.agent_auth.seconds_since_seen() is None
    finally:
        host.stop()


def test_a_message_goes_on_the_record_as_the_operators(server):
    answer = server.post(
        "/app/agent/messages", headers=server.operator, json={"text": "how's 01?"}
    )
    assert answer.status_code == 201, answer.text
    message_id = answer.json()["id"]
    note = _notes(server)[-1]
    assert note["payload"]["id"] == message_id
    assert note["payload"]["text"] == "how's 01?"
    assert str(note["actor"]).startswith(str(OPERATOR))


@pytest.mark.parametrize("body", [{}, {"text": "  "}, {"text": "x" * 2001}])
def test_an_empty_or_long_message_is_refused(server, body):
    answer = server.post("/app/agent/messages", headers=server.operator, json=body)
    assert answer.status_code == 422


def test_the_operator_sets_the_level_and_stops_the_agent(server):
    answer = server.put(
        "/app/agent/level", headers=server.operator, json={"level": "Drive"}
    )
    assert answer.status_code == 200, answer.text
    assert server.config.level is AgentLevel.DRIVE
    state = server.get("/app/agent", headers=server.operator).json()
    assert state["level"] == "drive" and not state["stopped"]

    answer = server.post("/app/agent/stop", headers=server.operator)
    assert answer.status_code == 200, answer.text
    assert answer.json()["level"] == "watch"
    assert server.config.level is AgentLevel.WATCH and server.config.stopped


def test_a_level_that_isnt_one_is_refused(server):
    answer = server.put(
        "/app/agent/level", headers=server.operator, json={"level": "sometimes"}
    )
    assert answer.status_code == 422
    assert answer.json()["detail"]["error_type"] == "bad_level"


SUPERVISE = {
    "tool": "protocol_set_supervision",
    "arguments": {"task_name": SETUP, "supervise": True},
}


def _file(server):
    server.put("/app/agent/level", headers=server.operator, json={"level": "assist"})
    answer = server.post(
        "/app/agent/requests",
        headers=server.agent,
        json={"actions": [SUPERVISE], "reason": "watch the setup"},
    )
    assert answer.status_code == 202, answer.text
    return answer.json()["request_id"]


def test_the_agent_cannot_approve_its_own_request(server):
    request_id = _file(server)
    answer = server.post(
        f"/app/agent/requests/{request_id}/approve", headers=server.agent
    )
    assert answer.status_code == 403
    waiting = server.get("/app/agent", headers=server.operator).json()["request"]
    assert waiting["request_id"] == request_id


def test_the_operator_approves_a_request(server):
    request_id = _file(server)
    answer = server.post(
        f"/app/agent/requests/{request_id}/approve", headers=server.operator
    )
    assert answer.status_code == 200, answer.text
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status = server.get(
            f"/app/agent/requests/{request_id}", headers=server.agent
        ).json()["status"]
        if status != "applying":
            break
        time.sleep(0.05)
    assert status == "applied"


def test_the_operator_rejects_a_request_with_a_reason(server):
    request_id = _file(server)
    answer = server.post(
        f"/app/agent/requests/{request_id}/reject",
        headers=server.operator,
        json={"reason": "not yet"},
    )
    assert answer.status_code == 200, answer.text
    request = server.get(f"/app/agent/requests/{request_id}", headers=server.agent)
    assert request.json()["status"] == "rejected"
    # Deciding it twice is refused, as in the window.
    again = server.post(
        f"/app/agent/requests/{request_id}/reject", headers=server.operator
    )
    assert again.status_code >= 400


def test_a_new_token_revokes_the_old_one(server):
    old = server.operator
    server.session.app_server.new_operator_token()
    assert server.get("/app/agent", headers=old).status_code == 401
    new = _bearer(server.config.operator_token)
    assert server.get("/app/agent", headers=new).status_code == 200


def test_the_card_s_rows_read_as_a_conversation(server):
    server.post(
        "/app/agent/messages", headers=server.operator, json={"text": "how's 01?"}
    )
    message_id = _notes(server)[-1]["payload"]["id"]
    server.session.recorder.buffer.append(
        NOTE, {"text": "01 is **fine**", "reply_to": message_id}, actor="agent"
    )
    rows = server.get("/app/agent/recent", headers=server.operator).json()["rows"]
    assert [(row["kind"], row["text"]) for row in rows[-2:]] == [
        ("you", "how's 01?"),
        ("reply", "01 is **fine**"),
    ]
    assert server.get("/app/agent/recent", headers=server.agent).status_code == 403


def test_the_agent_s_state_names_the_experiment(server):
    state = server.get("/app/agent", headers=server.operator).json()
    assert state["experiment"] == os.path.basename(server.session.experiment.path)
    assert state["request"] is None
    # Doing nothing on the microscope, and the levels as the card's menu has them.
    assert state["now"] is None
    assert [(level["name"], level["tip"]) for level in state["levels"]][1:3] == [
        ("Watch", "Read only"),
        ("Assist", "Asks first"),
    ]


def test_evidence_names_its_image_for_the_files_panel(server):
    import io

    from PIL import Image

    from fibsem.workflows import evidence

    root = str(server.session.experiment.path)
    buffer = io.BytesIO()
    Image.new("L", (32, 16)).save(buffer, "PNG")
    record = evidence.save(
        lambda: root, upload=base64.b64encode(buffer.getvalue()).decode()
    )
    answer = server.get(
        f"/app/agent/evidence/{record['evidence_id']}", headers=server.operator
    ).json()
    assert answer["path"] == f"evidence/{record['evidence_id']}.png"
    assert answer["preview"]["full_width"] == 32


def test_missing_evidence_is_not_found(server):
    answer = server.get("/app/agent/evidence/0123abcd", headers=server.operator)
    assert answer.status_code == 404


def test_the_card_s_picture_as_the_window_drew_it(server):
    # No window, no picture.
    assert (
        server.get("/app/agent/pictures/request", headers=server.operator).status_code
        == 404
    )
    server.session.agent.set_picture("request", b"\x89PNG fake", "Ion beam")
    state = server.get("/app/agent", headers=server.operator).json()
    assert state["pictures"]["request"]["image"] is True
    picture = server.get("/app/agent/pictures/request", headers=server.operator).json()
    assert base64.b64decode(picture["image_b64_png"]) == b"\x89PNG fake"
    assert picture["caption"] == "Ion beam"
    # Only a caption: why there is no picture.
    server.session.agent.set_picture("request", None, "No ion image yet")
    state = server.get("/app/agent", headers=server.operator).json()
    assert state["pictures"]["request"] == {
        "caption": "No ion image yet",
        "image": False,
        "version": state["pictures"]["request"]["version"],
    }
    server.session.agent.set_picture("request", None)
    assert server.get("/app/agent", headers=server.operator).json()["pictures"] == {}


def test_the_chat_page_is_served(server):
    # Static, like the event log: every call it makes needs the operator token.
    answer = server.get("/chat")
    assert answer.status_code == 200
    assert "/app/agent/recent" in answer.text


def test_the_connect_text_is_the_card_s(server):
    from fibsem.server.agent_setup import setup_text

    answer = server.get("/app/agent/connect", headers=server.operator)
    assert answer.json()["text"] == setup_text(server.session.app_server.url)
    assert server.get("/app/agent/connect", headers=server.agent).status_code == 403


def _suggested(server):
    from fibsem.structures import Point
    from fibsem.workflows.proposals import POINT_OF_INTEREST, Proposal, Suggestion
    from fibsem.workflows.status import TaskStatus

    experiment = server.session.experiment
    lamella = experiment.positions[0]
    lamella.task_state.name = SETUP
    lamella.task_state.task_id = "run-1"
    lamella.task_state.status = TaskStatus.InProgress
    proposal = Proposal(
        kind=POINT_OF_INTEREST,
        values={"poi": Point(0.0, 0.0)},
        provenance={"task_id": "run-1"},
    )
    assert experiment.ask_proposal(lamella.id, SETUP, proposal)
    suggestion = Suggestion(
        actor="agent:claude", reason="1 µm right", values={"poi": Point(1e-6, 0.0)}
    )
    assert experiment.suggest(lamella.id, SETUP, proposal.id, suggestion).applied
    shown = server.get("/app/agent", headers=server.operator).json()["suggestion"]
    assert shown["proposal_id"] == proposal.id and shown["reason"] == "1 µm right"
    return proposal, suggestion


def test_the_operator_approves_the_agent_s_suggestion(server):
    from fibsem.workflows.proposals import DecisionOutcome

    proposal, suggestion = _suggested(server)
    path = f"/app/agent/suggestions/{proposal.id}/approve"
    assert server.post(path, headers=server.agent).status_code == 403
    answer = server.post(path, headers=server.operator)
    assert answer.status_code == 200, answer.text
    decision = proposal.current
    assert decision.outcome is DecisionOutcome.Confirmed
    assert decision.actor.kind.value == "operator"
    assert decision.suggestion_id == suggestion.id
    assert (
        server.get("/app/agent", headers=server.operator).json()["suggestion"] is None
    )
    # Approving it twice is refused, as the card's would be.
    assert server.post(path, headers=server.operator).status_code == 409


def test_the_operator_declines_the_agent_s_suggestion(server):
    proposal, _ = _suggested(server)
    answer = server.post(
        f"/app/agent/suggestions/{proposal.id}/decline",
        headers=server.operator,
        json={"reason": "that is the trench"},
    )
    assert answer.status_code == 200, answer.text
    assert proposal.pending
    assert proposal.suggestion.declined_reason == "that is the trench"
