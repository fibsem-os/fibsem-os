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
