"""The gateway's copy of this PC's experiments to a shared folder."""

import json
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from fibsem.gateway.copies import Copies
from fibsem.gateway.gateway import GatewayRefused
from fibsem.gateway.server import GatewayAuth, build_gateway_app
from tests.gateway.test_gateway_routes import FakeGateway, _pair_hub


class Clock:
    def __init__(self):
        self.now = datetime(2026, 11, 13, 20, 0)

    def __call__(self):
        return self.now


@pytest.fixture
def pc(tmp_path):
    """A PC with one experiment, a shared folder, and the copies for them."""
    experiment = tmp_path / "data" / "grid-1"
    experiment.mkdir(parents=True)
    (experiment / "experiment.yaml").write_text("name: grid-1")
    shared = tmp_path / "shared"
    shared.mkdir()
    clock, running = Clock(), [False]

    def make():
        return Copies(
            running=lambda: running[0],
            path=tmp_path / "state" / "copies.json",
            folders=lambda: [experiment],
            now=clock,
            label=lambda: "fib-1",
        )

    return make, shared, clock, running, experiment


def wait(copies):
    copies._thread.join(timeout=10)


def test_nothing_is_copied_until_the_operator_names_a_folder(pc):
    make, shared, clock, _, _ = pc
    copies = make()
    assert copies.state() == {
        "folder": None,
        "available": False,
        "copying": False,
        "last": None,
        "next": None,
    }
    clock.now += timedelta(days=2)
    copies.poll()
    assert copies._thread is None
    with pytest.raises(GatewayRefused, match="No shared folder"):
        copies.copy_now("operator")


def test_copy_now_copies_and_the_last_copy_is_kept_across_a_restart(pc):
    make, shared, _, _, _ = pc
    copies = make()
    assert copies.set_folder(str(shared))["available"]
    copies.copy_now("operator")
    wait(copies)
    assert (
        shared / "fib-1" / "grid-1" / "experiment.yaml"
    ).read_text() == "name: grid-1"
    last = copies.state()["last"]
    assert (last["sent"], last["complete"], last["who"]) == (1, True, "operator")
    again = make().state()
    assert again["folder"] == str(shared) and again["last"] == last


def test_the_nightly_copy_goes_at_its_time_and_waits_for_a_run_to_end(pc):
    make, shared, clock, running, _ = pc
    copies = make()
    copies.set_folder(str(shared))
    assert copies.state()["next"] == "2026-11-14T02:00"
    copies.poll()
    assert copies._thread is None

    clock.now = datetime(2026, 11, 14, 2, 1)
    running[0] = True
    copies.poll()
    assert copies._thread is None

    running[0] = False
    clock.now = datetime(2026, 11, 14, 3, 30)
    copies.poll()
    wait(copies)
    assert copies.state()["last"]["who"] == "nightly"
    assert copies.state()["next"] == "2026-11-15T02:00"


def test_a_shared_folder_that_is_gone_fails_the_copy_and_nothing_else(pc):
    make, shared, clock, _, _ = pc
    copies = make()
    copies.set_folder(str(shared))
    shared.rmdir()
    with pytest.raises(GatewayRefused, match="can't be reached"):
        copies.copy_now("operator")
    clock.now = datetime(2026, 11, 14, 2, 1)
    copies.poll()
    assert "can't be reached" in copies.state()["last"]["why"]
    assert not copies.state()["available"]


def test_a_folder_that_is_not_there_or_not_a_full_path_is_refused(pc, tmp_path):
    make, _, _, _, _ = pc
    copies = make()
    with pytest.raises(GatewayRefused, match="isn't a folder"):
        copies.set_folder(str(tmp_path / "typo"))
    with pytest.raises(GatewayRefused, match="full path"):
        copies.set_folder("shared")
    assert copies.set_folder("")["folder"] is None


def test_two_experiments_of_one_name_are_told_apart_in_the_shared_folder(pc, tmp_path):
    make, shared, _, _, experiment = pc
    other = tmp_path / "elsewhere" / "grid-1"
    other.mkdir(parents=True)
    (other / "experiment.yaml").write_text("name: other")
    copies = make()
    copies._folders = lambda: [experiment, other]
    copies.set_folder(str(shared))
    copies.copy_now("operator")
    wait(copies)
    assert (
        shared / "fib-1" / "grid-1-2" / "experiment.yaml"
    ).read_text() == "name: other"


def test_only_the_operator_names_the_folder_and_a_paired_hub_may_copy_now(pc, tmp_path):
    make, shared, _, _, _ = pc
    copies = make()
    token_file = tmp_path / "operator-token.json"
    token_file.write_text(json.dumps({"token": "operator-token"}))
    auth = GatewayAuth.from_disk(token_file, tmp_path / "gateway-paired.json")
    client = TestClient(build_gateway_app(FakeGateway(), auth, copies=copies))
    operator = {"Authorization": "Bearer operator-token"}
    hub = {"Authorization": f"Bearer {_pair_hub(client, [])['token']}"}

    assert client.get("/gateway/copies").status_code == 401
    assert (
        client.put(
            "/gateway/copies", json={"folder": str(shared)}, headers=hub
        ).status_code
        == 403
    )
    refused = client.put("/gateway/copies", json={"folder": "typo"}, headers=operator)
    assert refused.status_code == 422
    assert client.post("/gateway/copies/now", headers=hub).status_code == 409

    set_ = client.put("/gateway/copies", json={"folder": str(shared)}, headers=operator)
    assert set_.json()["folder"] == str(shared)
    assert client.post("/gateway/copies/now", headers=hub).status_code == 200
    wait(copies)
    assert client.get("/gateway/copies", headers=hub).json()["last"]["sent"] == 1
