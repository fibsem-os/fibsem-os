"""The Hub's simulator: an ordinary gateway on the hub PC with the Demo
microscope, listed as an instrument, that the Hub starts and stops."""

import json

import pytest
from fastapi.testclient import TestClient

from fibsem.gateway.gateway import GatewayRefused
from fibsem.hub.hub import Hub
from fibsem.hub.server import build_hub_app, hub_auth
from fibsem.hub.simulator import NAME, Simulator

OPERATOR = {"Authorization": "Bearer operator-token"}


class FakeProcess:
    def __init__(self, command, ends=None):
        self.command = command
        self.code = ends
        self.pid = 4242

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        return self.code

    def terminate(self):
        self.code = -15

    def kill(self):
        self.code = -9


class Pc:
    """The hub PC: the processes started on it, and its gateway file."""

    def __init__(self):
        self.started = []
        self.gateway = None
        self.listens = True
        self.codes = []
        self.paired = []
        self.now = 0.0

    def spawn(self, command):
        process = FakeProcess(command)
        self.started.append(process)
        if command[-1] == "stop":
            # ``fibsem-gateway stop``: the running gateway quits.
            for other in self.started:
                other.code = 0
            self.gateway = None
        elif self.listens:
            self.gateway = {"url": "http://127.0.0.1:8010", "pid": process.pid}
        return process

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def pc():
    return Pc()


@pytest.fixture
def hub(tmp_path, pc):
    def fetch(url, route, token, timeout):
        raise OSError("connection refused")

    def pair(url, code, name):
        pc.paired.append((url, code))
        return "token", "hub"

    return Hub(tmp_path / "hub.yaml", tmp_path / "hub", fetch=fetch, pair=pair)


@pytest.fixture
def simulator(hub, pc):
    return Simulator(
        hub,
        spawn=pc.spawn,
        running=lambda: pc.gateway,
        port=lambda: 8010,
        python="python",
        code=lambda: pc.codes.pop(0) if pc.codes else None,
        clock=lambda: pc.now,
        sleep=pc.sleep,
    )


@pytest.fixture(autouse=True)
def unreleased(monkeypatch):
    monkeypatch.setenv("FIBSEM_UNRELEASED", "1")


def test_the_simulator_is_a_gateway_here_with_the_demo_microscope(simulator, hub, pc):
    assert simulator.state() == {"name": NAME, "running": False, "listed": False}
    simulator.start()
    (process,) = pc.started
    assert process.command == [
        "python",
        "-m",
        "fibsem.gateway",
        "run",
        "--no-window",
        "--manufacturer",
        "Demo",
    ]
    (row,) = hub.instruments()
    assert (row.name, row.url, row.ssh) == (NAME, "http://127.0.0.1:8010", None)
    assert simulator.state() == {"name": NAME, "running": True, "listed": True}
    # With no code from its gateway it is left for a person to pair.
    assert hub.token(NAME) is None
    simulator.start()
    assert len(pc.started) == 1, "one that runs is not started again"


def test_the_hub_pairs_itself_with_the_simulator_it_started(simulator, hub, pc):
    pc.codes = ["1234", "5678"]
    simulator.start()
    assert pc.paired == [("http://127.0.0.1:8010", "1234")]
    assert hub.token(NAME) == "token"
    simulator.stop()
    simulator.start()
    assert len(pc.paired) == 1, "its pairing is kept: it is not paired again"


def test_the_hub_never_pairs_itself_with_a_gateway_it_did_not_start(simulator, hub, pc):
    pc.codes = ["1234"]
    simulator.start()
    hub.remove(NAME)
    pc.paired.clear()
    pc.codes = ["5678"]
    # Its own simulator ended, and another gateway took its place here.
    pc.started[0].code = 0
    pc.gateway = {"url": "http://127.0.0.1:8010", "pid": 7}
    with pytest.raises(GatewayRefused):
        simulator.start()
    hub.add(NAME, "http://127.0.0.1:8010")
    simulator._pair()
    assert pc.paired == [] and hub.token(NAME) is None


def test_another_gateway_on_this_pc_is_never_taken_for_the_simulator(
    simulator, hub, pc
):
    pc.gateway = {"url": "http://127.0.0.1:8010", "pid": 7}
    with pytest.raises(GatewayRefused) as refused:
        simulator.start()
    assert refused.value.error_type == "gateway_here"
    assert not pc.started and hub.instruments() == []


def test_a_simulator_that_does_not_start_says_so(simulator, hub, pc):
    pc.listens = False
    with pytest.raises(GatewayRefused) as refused:
        simulator.start()
    assert refused.value.error_type == "simulator_failed"
    assert pc.started[0].poll() is not None, "it is not left running"
    assert hub.instruments() == [] and not simulator.state()["running"]


def test_stopping_asks_its_gateway_to_quit_and_keeps_it_listed(simulator, hub, pc):
    simulator.start()
    simulator.stop()
    assert pc.started[-1].command == ["python", "-m", "fibsem.gateway", "stop"]
    assert pc.started[0].poll() is not None
    assert simulator.state() == {"name": NAME, "running": False, "listed": True}
    simulator.stop()
    assert len(pc.started) == 2, "nothing to stop the second time"


def test_the_simulator_is_not_part_of_a_release(simulator, monkeypatch, pc):
    monkeypatch.setenv("FIBSEM_UNRELEASED", "0")
    with pytest.raises(GatewayRefused) as refused:
        simulator.start()
    assert refused.value.error_type == "not_released" and not pc.started


def test_the_page_starts_stops_and_removes_the_simulator(tmp_path, hub, simulator, pc):
    token_file = tmp_path / "operator-token.json"
    token_file.write_text(json.dumps({"token": "operator-token"}))
    auth = hub_auth(token_file, tmp_path / "hub-paired.json")
    page = TestClient(build_hub_app(hub, auth, simulator=simulator))
    assert page.post("/hub/simulator/start").status_code == 401
    assert page.get("/hub", headers=OPERATOR).json()["simulator"] == {
        "name": NAME,
        "running": False,
        "listed": False,
    }
    started = page.post("/hub/simulator/start", headers=OPERATOR).json()
    assert started["simulator"]["running"] and started["instruments"][0]["name"] == NAME
    stopped = page.post("/hub/simulator/stop", headers=OPERATOR).json()
    assert stopped["simulator"] == {"name": NAME, "running": False, "listed": True}
    page.post("/hub/simulator/start", headers=OPERATOR)
    # Removing it from the list stops it too.
    removed = page.delete(f"/hub/instruments/{NAME}", headers=OPERATOR).json()
    assert removed == {
        "instruments": [],
        "simulator": {"name": NAME, "running": False, "listed": False},
    }
    pc.gateway = {"url": "http://127.0.0.1:8010", "pid": 7}
    refused = page.post("/hub/simulator/start", headers=OPERATOR)
    assert refused.status_code == 409
    assert refused.json()["detail"]["error_type"] == "gateway_here"
