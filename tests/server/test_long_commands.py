"""Long commands (map 8.3): a mill or a spot burn an agent starts answers at once
with its run and runs on, in control of the microscope, which the agent may act
beside; ``GET .../run`` says how it ended, and the end is a ``command_ended``
event.

The spot burn service's ``run`` is replaced on the instance by one that waits
for the test, so "still running" is a state the test holds, not a race.
"""

import os
import threading

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # TestClient transport

from fastapi.testclient import TestClient  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.acting import OPERATOR  # noqa: E402
from fibsem.cancellation import OperationCancelledError  # noqa: E402
from fibsem.client import ServerError  # noqa: E402
from fibsem.control import MicroscopeBusy  # noqa: E402
from fibsem.devices.core import DeviceFault  # noqa: E402
from fibsem.imaging.spot import SpotBurnSettings  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.structures import Point  # noqa: E402
from tests.server.test_client import in_process  # noqa: E402

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
RUN = "/microscope/services/spot_burn/commands/run"
RUN_STATE = "/microscope/services/spot_burn/run"
MOVE = "/microscope/services/stage_movement/commands/move_to_orientation"
SETTINGS = SpotBurnSettings(coordinates=[Point(0.5, 0.5)], exposure_time=0.01)


@pytest.fixture(scope="module")
def microscope():
    previous = os.environ.get("FIBSEM_SIM_NO_DELAY")
    os.environ["FIBSEM_SIM_NO_DELAY"] = "1"
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    yield microscope
    if previous is None:
        os.environ.pop("FIBSEM_SIM_NO_DELAY", None)
    else:
        os.environ["FIBSEM_SIM_NO_DELAY"] = previous


@pytest.fixture(scope="module")
def app(microscope):
    return build_server(
        microscope, auth=AuthConfig.generate(level="drive", token=TOKEN)
    )


@pytest.fixture(scope="module")
def client(app):
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers.update(AUTH)
        yield client


@pytest.fixture
def held(microscope, monkeypatch):
    """The spot burn's run, held until the test lets it go, ending as the test
    says: ``outcome`` is a value to return or an exception to raise."""
    go = threading.Event()
    state = {"outcome": "finished"}

    def run(settings: SpotBurnSettings, stop_event=None):
        assert go.wait(10), "the test never let the run go"
        if isinstance(state["outcome"], BaseException):
            raise state["outcome"]
        return state["outcome"]

    monkeypatch.setattr(microscope.spot_burn, "run", run)
    state["go"] = go
    yield state
    go.set()


def _start(client):
    resp = client.post(RUN, json={"kwargs": {"settings": SETTINGS.to_dict()}})
    assert resp.status_code == 200, resp.text
    return resp.json()["result"]


def _wait(client, run_id):
    resp = client.get(RUN_STATE, params={"run_id": run_id, "wait": 10})
    assert resp.status_code == 200, resp.text
    return resp.json()["run"]


def test_a_long_command_answers_at_once_and_runs_on_in_control(
    client, microscope, held
):
    events = []

    def seen(kind, payload):
        events.append(kind)

    microscope.session.record_signal.connect(seen)
    try:
        run = _start(client)
        assert run["state"] == "running" and run["command"] == "run"

        # Still running, and still in control of the microscope as the agent's.
        current = microscope.control.current
        assert current.by == "agent" and current.action == "spot burn run"
        assert client.get(RUN_STATE, params={"wait": 0}).json()["run"]["state"] == (
            "running"
        )
        # Its starter may act beside it (an image or a move during its mill);
        # anyone else is refused.
        beside = client.post(MOVE, json={"kwargs": {"orientation": "SEM"}})
        assert beside.status_code == 200, beside.text
        assert microscope.control.current.action == "spot burn run"
        with pytest.raises(MicroscopeBusy):
            with microscope.control.take(OPERATOR, "acquire"):
                pass

        held["go"].set()
        ended = _wait(client, run["run_id"])
        assert ended["state"] == "done" and ended["result"] == "finished"
        assert ended["ended_at"]
        # Control was given up before the run said it had ended.
        assert microscope.control.current is None
    finally:
        microscope.session.record_signal.disconnect(seen)
    assert events.count("command_started") == 1
    assert events.count("command_ended") == 1


def test_a_failed_run_says_why_and_gives_up_control(client, microscope, held):
    held["outcome"] = RuntimeError("the beam would not blank")
    run = _start(client)
    held["go"].set()
    ended = _wait(client, run["run_id"])
    assert ended["state"] == "failed"
    assert ended["error"]["error_type"] == "RuntimeError"
    assert "would not blank" in ended["error"]["message"]
    assert microscope.control.current is None


def test_a_stopped_run_ends_cancelled(client, held):
    held["outcome"] = "cancelled"
    run = _start(client)
    held["go"].set()
    assert _wait(client, run["run_id"])["state"] == "cancelled"


def test_the_agent_stops_its_own_run(client, microscope, held):
    run = _start(client)
    stopped = client.post(
        "/microscope/services/spot_burn/commands/stop",
        json={"kwargs": {}, "reason": "wrong place"},
    )
    assert stopped.status_code == 200, stopped.text
    held["go"].set()
    assert _wait(client, run["run_id"])["state"] == "done"


def test_waiting_for_an_earlier_run_answers_with_the_latest(client, held):
    held["go"].set()
    first = _start(client)
    _wait(client, first["run_id"])
    second = _start(client)
    latest = _wait(client, first["run_id"])
    assert latest["run_id"] == second["run_id"]


def test_a_service_that_ran_nothing_long_is_404(client):
    resp = client.get("/microscope/services/imaging/run")
    assert resp.status_code == 404
    assert resp.json()["detail"]["error_type"] == "not_found"


def test_the_listing_marks_long_commands(client):
    services = client.get("/microscope/services").json()
    assert services["milling"]["commands"]["run"]["long"] is True
    assert services["spot_burn"]["commands"]["run"]["long"] is True
    assert "long" not in services["milling"]["commands"]["prepare"]


def test_the_client_waits_for_the_run(app, held):
    held["go"].set()
    run = in_process(app).services["spot_burn"].run(settings=SETTINGS)
    assert run.wait(timeout=10) == "finished"
    assert run.state == "done"


@pytest.mark.parametrize(
    "outcome, raised",
    [
        (OperationCancelledError("stopped"), OperationCancelledError),
        (DeviceFault("the beam would not blank"), DeviceFault),
        (KeyError("nothing here"), ServerError),
    ],
)
def test_the_client_raises_what_the_run_raised(app, held, outcome, raised):
    held["outcome"] = outcome
    held["go"].set()
    run = in_process(app).services["spot_burn"].run(settings=SETTINGS)
    with pytest.raises(raised):
        run.wait(timeout=10)


def test_the_client_reads_the_latest_run(app, held):
    client = in_process(app)
    assert client.services["imaging"].latest_run() is None
    run = client.services["spot_burn"].run(settings=SETTINGS)
    assert client.services["spot_burn"].latest_run().run_id == run.run_id
    with pytest.raises(TimeoutError):
        run.wait(timeout=0.1)
    held["go"].set()
    assert run.wait(timeout=10) == "finished"
