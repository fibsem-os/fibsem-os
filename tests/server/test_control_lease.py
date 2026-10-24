"""Control held across requests (PR C): a lease on the App Server's control. A
remote run holds the microscope for its whole length, its calls run inside it,
anyone else is refused naming it, and it lends during a question.

Skipped when the [server] extra is not installed.
"""

import os
import time

import pytest

from tests.fixtures.app_server import serving

pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")

from fastapi.testclient import TestClient  # noqa: E402

from fibsem.acting import AGENT, OPERATOR, WORKFLOW  # noqa: E402
from fibsem.client.remote_microscope import RemoteMicroscope  # noqa: E402
from fibsem.control import MicroscopeBusy, lending  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.server.leases import LEASE_HEADER, Leases  # noqa: E402
from fibsem.structures.settings import FibsemStagePosition  # noqa: E402

TOKEN = "test-token"
OPERATOR_TOKEN = "operator-token"
MOVE = "/microscope/services/stage_movement/commands/move_relative"


@pytest.fixture(scope="module")
def auth():
    auth = AuthConfig.generate(level="drive", token=TOKEN)
    auth.operator_token = OPERATOR_TOKEN
    return auth


@pytest.fixture(scope="module")
def app(microscope, auth):
    return build_server(microscope, auth=auth)


@pytest.fixture
def http(app):
    return TestClient(app)


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


def _as(token):
    return {"Authorization": f"Bearer {token}"}


def _move(http, token, lease=None):
    headers = _as(token)
    if lease is not None:
        headers[LEASE_HEADER] = lease
    return http.post(MOVE, json=NUDGE, headers=headers)


@pytest.fixture(autouse=True)
def nobody_in_control(microscope):
    yield
    assert microscope.control.current is None


def test_a_lease_holds_control_and_its_calls_run_inside_it(http, microscope):
    taken = http.post(
        "/microscope/control", json={"action": "remote run"}, headers=_as(TOKEN)
    )
    assert taken.status_code == 200, taken.text
    lease = taken.json()["lease"]
    try:
        assert microscope.control.current.by == AGENT
        assert microscope.control.current.action == "remote run"
        assert _move(http, TOKEN, lease).status_code == 200
        held = http.get(
            "/microscope/control", headers={**_as(TOKEN), LEASE_HEADER: lease}
        ).json()
        assert held["in_control"]["action"] == "remote run" and held["yours"]
        # Without the lease, the same caller is refused, naming the lease.
        refused = _move(http, TOKEN)
        assert refused.status_code == 409
        assert refused.json()["detail"]["in_control"]["action"] == "remote run"
        # The operator's clicks are refused too.
        with pytest.raises(MicroscopeBusy):
            with microscope.control.take(OPERATOR, "acquire images"):
                pass
    finally:
        assert http.delete(f"/microscope/control/{lease}", headers=_as(TOKEN)).json()[
            "released"
        ]


def test_a_lease_is_refused_while_someone_else_is_in_control(http, microscope):
    with microscope.control.take(WORKFLOW, "run"):
        refused = http.post("/microscope/control", json={}, headers=_as(TOKEN))
    assert refused.status_code == 409
    assert refused.json()["detail"]["in_control"]["by"] == "workflow"


def test_another_callers_lease_is_not_theirs(http):
    lease = http.post(
        "/microscope/control", json={}, headers=_as(OPERATOR_TOKEN)
    ).json()["lease"]
    try:
        gone = _move(http, TOKEN, lease)
        assert gone.status_code == 409
        assert gone.json()["detail"]["error_type"] == "lease_gone"
    finally:
        http.delete(f"/microscope/control/{lease}", headers=_as(OPERATOR_TOKEN))


def test_giving_a_lease_up_is_never_refused_by_the_level(http, auth):
    lease = http.post("/microscope/control", json={}, headers=_as(TOKEN)).json()[
        "lease"
    ]
    auth.set_level("watch")
    try:
        assert (
            http.put(f"/microscope/control/{lease}", headers=_as(TOKEN)).status_code
            == 403
        )
        released = http.delete(f"/microscope/control/{lease}", headers=_as(TOKEN))
        assert released.json() == {"released": True}
    finally:
        auth.set_level("drive")


def test_an_idle_lease_is_given_up():
    from fibsem.control import Control

    control = Control()
    leases = Leases(control, idle_s=0.2)
    lease = leases.take(AGENT, "remote run")
    assert control.current is lease.entry
    deadline = time.monotonic() + 5
    while control.current is not None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert control.current is None
    assert leases.get(lease.id) is None


def test_a_lent_lease_lets_one_other_action_in(http, microscope):
    lease = http.post("/microscope/control", json={}, headers=_as(TOKEN)).json()[
        "lease"
    ]
    try:
        assert (
            http.post(
                f"/microscope/control/{lease}/lend", headers=_as(TOKEN)
            ).status_code
            == 200
        )
        with microscope.control.take(OPERATOR, "acquire images") as entry:
            assert microscope.control.current is entry
        http.delete(f"/microscope/control/{lease}/lend", headers=_as(TOKEN))
        with pytest.raises(MicroscopeBusy):
            with microscope.control.take(OPERATOR, "acquire images"):
                pass
    finally:
        http.delete(f"/microscope/control/{lease}", headers=_as(TOKEN))


# --- RemoteMicroscope.control over a real server -----------------------------


@pytest.fixture(scope="module")
def url(app):
    with serving(app) as url:
        yield url


@pytest.fixture(scope="module")
def remote(url):
    remote = RemoteMicroscope.connect(url, OPERATOR_TOKEN)
    yield remote
    remote.disconnect()


def test_a_remote_run_holds_the_servers_control_for_its_length(remote, microscope):
    with remote.control.take(WORKFLOW, "remote lamellae run"):
        # Held on the server as the operator, whose token this is.
        assert microscope.control.current.by == OPERATOR
        assert remote.control.held.action == "remote lamellae run"
        # The run's own calls go through inside the lease.
        remote.move_stage_relative(FibsemStagePosition(x=1e-6))
        with pytest.raises(MicroscopeBusy):
            with microscope.control.take(AGENT, "move"):
                pass
    assert microscope.control.current is None
    assert remote.control.held is None


def test_a_remote_take_is_refused_naming_who_is_in_control(remote, microscope):
    with microscope.control.take(AGENT, "move the stage"):
        with pytest.raises(MicroscopeBusy) as refused:
            with remote.control.take(OPERATOR, "acquire images"):
                pass
    assert refused.value.in_control.by == AGENT
    assert refused.value.in_control.action == "move the stage"


def test_a_remote_run_lends_during_its_question(remote, microscope):
    with remote.control.take(WORKFLOW, "remote run"):
        with lending():
            # Whoever answers acts, on the server too.
            with microscope.control.take(OPERATOR, "acquire images"):
                pass
        with pytest.raises(MicroscopeBusy):
            with microscope.control.take(OPERATOR, "acquire images"):
                pass


def test_a_click_on_a_remote_microscope_holds_its_lease_in_the_worker_only(
    remote, microscope
):
    pytest.importorskip("PyQt5")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    from fibsem.client import transport
    from fibsem.ui.qt.threading import FunctionWorker

    QApplication.instance() or QApplication([])
    seen = []
    worker = FunctionWorker(
        lambda: seen.append((transport._lease.get(), microscope.control.current.by))
    )
    assert worker.start(control=remote.control, action="acquire images")
    worker.join(10)
    lease, by = seen[0]
    assert lease is not None and by == OPERATOR
    assert transport._lease.get() is None
    assert wait_for(lambda: microscope.control.current is None)


def wait_for(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


def test_a_remote_click_during_another_shares_its_lease(remote, microscope):
    with remote.control.take(OPERATOR, "mill", join=True):
        with remote.control.take(OPERATOR, "acquire images", join=True):
            remote.move_stage_relative(FibsemStagePosition(x=1e-6))
            assert microscope.control.current.action == "mill"
    assert microscope.control.current is None
