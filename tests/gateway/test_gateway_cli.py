"""``fibsem-gateway pair``: a code for the operator's browser, an agent or a
hub, from the gateway that runs here."""

import pytest

from fibsem.gateway import main as gateway_main


@pytest.fixture
def asked(monkeypatch):
    calls = []

    def call(method, route, body=None):
        calls.append((method, route, body))
        return {"code": "123456", "role": body["role"]}

    monkeypatch.setattr(gateway_main, "call", call)
    monkeypatch.setattr(gateway_main.diagnostics, "install", lambda name: None)
    return calls


@pytest.mark.parametrize(
    ("flags", "body"),
    [
        ([], {"role": "operator"}),
        (["--agent"], {"role": "agent"}),
        (["--hub"], {"role": "hub"}),
        (["--hub", "--may-switch"], {"role": "hub", "grants": ["switch_modes"]}),
    ],
)
def test_pair_opens_pairing_for_the_role_asked(asked, capsys, flags, body):
    assert gateway_main.main(["pair", *flags]) == 0
    assert asked == [("POST", "/pairing", body)]
    assert f"123456 (5 minutes, {body['role']})" in capsys.readouterr().out


def test_an_agent_and_a_hub_at_once_is_refused(asked):
    with pytest.raises(SystemExit):
        gateway_main.main(["pair", "--agent", "--hub"])
    assert asked == []


def test_the_gateway_file_is_this_users_alone(tmp_path):
    """The file a program here finds the gateway by: its address and the token
    that only asks to connect, readable by this PC's user only."""
    import json
    import os
    import stat

    from fibsem.gateway.gateway import write_gateway_file

    path = write_gateway_file("http://127.0.0.1:8010", "ask", tmp_path / "g.json")
    found = json.loads(path.read_text())
    assert found == {
        "url": "http://127.0.0.1:8010",
        "pid": os.getpid(),
        "connect_token": "ask",
    }
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert [p.name for p in tmp_path.iterdir()] == ["g.json"]


def test_a_taken_port_stops_the_gateway_before_it_writes_its_file(monkeypatch, capsys):
    """Another program on the port: the gateway says so and ends, and never
    writes gateway.json, which would send agents to that program."""
    import socket

    from fibsem.gateway import server as gateway_server

    taken = socket.socket()
    taken.bind(("127.0.0.1", 0))
    taken.listen()
    port = taken.getsockname()[1]
    written = []

    class Stub:
        def __init__(self, *args, **kwargs):
            self.pairing = type("P", (), {"tokens": None})()
            self.connect_token = "c"
            self.is_connect = lambda token: False

        def start(self):
            pytest.fail("started")

        def close(self):
            pass

    monkeypatch.setattr(gateway_main, "read_gateway_file", lambda: None)
    monkeypatch.setattr(gateway_main, "Gateway", Stub)
    monkeypatch.setattr(gateway_main, "GatewayAgent", Stub)
    monkeypatch.setattr(gateway_main, "HostMicroscope", Stub)
    monkeypatch.setattr(gateway_server.GatewayAuth, "from_disk", Stub)
    monkeypatch.setattr(
        gateway_server, "build_gateway_app", lambda *a, **k: gateway_server.FastAPI()
    )
    monkeypatch.setattr(
        gateway_main, "write_gateway_file", lambda *a: written.append(a)
    )
    try:
        assert gateway_main.run(port, window=False, connection=[]) == 1
    finally:
        taken.close()
    assert written == []
    assert f"could not listen on port {port}" in capsys.readouterr().out
