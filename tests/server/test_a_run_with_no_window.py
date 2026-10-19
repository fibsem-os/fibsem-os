"""A run with no window does what it does with one, minus the drawing.

The App Server over a session with nobody but the agent, at Drive: the agent
starts a supervised Select Position, sees on the status that the run is held
and by what, sees the task's images, and answers each question; the run ends
on its decisions. Nothing here imports Qt.
"""

import os
import socket
import time

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fibsem.ai.agent.levels import AgentLevel  # noqa: E402
from fibsem.applications.autolamella.proposals import POINT_OF_INTEREST  # noqa: E402
from fibsem.applications.autolamella.server.context import (  # noqa: E402
    AutoLamellaAppContext,
)
from fibsem.server import AuthConfig  # noqa: E402
from fibsem.server.hosting import AppServerHost  # noqa: E402
from fibsem.structures import BeamType  # noqa: E402
from fibsem.workflows.attention import Attention  # noqa: E402
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


@pytest.fixture
def agent(microscope, tmp_path):  # noqa: F811
    """The agent's client to an App Server over a windowless session at Drive,
    with a supervised Select Position open."""
    session = AppSession()
    session.microscope = microscope
    session.experiment = _experiment(
        tmp_path, microscope, attention=Attention.supervised
    )
    session.agent_auth = AuthConfig.generate(level=AgentLevel.DRIVE)
    session.app_server = AppServerHost(
        session,
        lambda buffer: AutoLamellaAppContext(session, event_buffer=buffer),
        port=_free_port(),
        discovery_path=tmp_path / "agent-server.json",
    )
    assert session.app_server.start()
    with httpx.Client(
        base_url=session.app_server.url,
        headers={"Authorization": f"Bearer {session.agent_auth.token}"},
        timeout=30,
    ) as client:
        client.session = session
        yield client
    session.runner.stop()
    session.runner.join(30)
    session.app_server.stop()


def _ok(answer):
    assert answer.status_code == 200, answer.text
    return answer.json()


def _held(agent, timeout: float = 60):
    """The run's status once it is held, or None once the run has ended."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        workflow = _ok(agent.get("/app/workflow/status"))["workflow"]
        if workflow["held"] is not None:
            return workflow
        if not workflow["running"]:
            return None
        time.sleep(0.1)
    raise AssertionError("the run neither held nor ended")


def test_the_agent_answers_a_supervised_run_with_no_window(agent):
    session = agent.session
    lamella = session.experiment.positions[0]
    started = _ok(agent.post("/app/workflow/start", json={"task_names": [SETUP]}))
    assert started["started"], started

    answered = []
    while (workflow := _held(agent)) is not None:
        held = workflow["held"]
        assert held["items"] == [f"{lamella.name}/{SETUP}"]
        assert held["releases"] == f"answer {SETUP} on {lamella.name}"
        (question,) = held["questions"]
        # What the task acquired is what the agent sees, window or not.
        assert session.images.get(BeamType.ION) is not None
        assert _ok(agent.get("/app/experiment/images"))["available"]
        _ok(
            agent.post(
                "/app/workflow/decide",
                json={
                    "item_id": lamella.id,
                    "task_name": SETUP,
                    "proposal_id": question["proposal_id"],
                    "outcome": "Confirmed",
                },
            )
        )
        answered.append(question["proposal_id"])
        assert len(answered) < 10, "the run kept asking"

    assert session.runner.join(30), "the run did not end"
    assert answered, "the supervised task asked nothing"
    asked = [p for p in lamella.proposals_from(SETUP) if p.id in answered]
    assert len(asked) == len(answered)
    assert all(str(p.current.actor).startswith("agent") for p in asked), (
        "each answer is the agent's"
    )
    # The point of interest is one of the questions, asked while Setup ran,
    # and nothing is left open after it.
    assert POINT_OF_INTEREST in {p.kind for p in asked}
    assert not [p for p in lamella.proposals_from(SETUP) if p.pending]
