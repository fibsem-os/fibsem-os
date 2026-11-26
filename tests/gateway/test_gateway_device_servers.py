"""The Device Servers the SEM PC's gateway watches: read from the remote
entries of the configuration the app connects with, asked for their health
(through an SSH tunnel when the entry names one), listed in GET /gateway, and
one entry added or removed from the page by the operator only."""

import json
from pathlib import Path

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

import fibsem
from fibsem.devices.wire import DEVICE_API_VERSION
from fibsem.gateway.device_servers import (
    NOT_ANSWERING,
    NOT_PAIRED,
    RUNNING,
    TOO_OLD_OR_NEW,
    TUNNEL_DOWN,
    DeviceServers,
    EntryRefused,
    add_entry,
    configuration_file,
    listed,
    remove_entry,
)
from fibsem.gateway.server import GatewayAuth, build_gateway_app
from fibsem.util.tunnels import Tunnel, TunnelDown
from tests.gateway.test_gateway_routes import (
    OPERATOR_TOKEN,
    FakeGateway,
    _as,
    _pair_hub,
)

CONFIG = """\
info:
  manufacturer: Demo
hardware:
  devices:
    - name: electron
      type: beam
      column_tilt: 0
    - name: fm
      driver: Odemis
      remote: 127.0.0.1:8765
      tunnel: meteor@meteor-pc
    - name: afm
      type: stage
      remote: 192.168.0.30:8766
    - name: spare
      remote: 192.168.0.40:8767
      enabled: false
"""

HEALTH = {
    "ok": True,
    "devices": {"fm": {"ok": True}, "camera": {"ok": False, "detail": "cold"}},
    "program": "fibsem-device-server",
    "version": fibsem.__version__,
    "api_version": DEVICE_API_VERSION,
}


@pytest.fixture
def config(tmp_path) -> Path:
    path = tmp_path / "sem.yaml"
    path.write_text(CONFIG)
    return path


def test_the_list_is_the_configurations_remote_entries(config):
    servers, problem = listed(config)
    assert problem is None
    assert [(s.key, s.serves, s.tunnel) for s in servers] == [
        ("127.0.0.1:8765", ["fm"], "meteor@meteor-pc"),
        ("192.168.0.30:8766", ["afm"], None),
    ]


def test_an_unreadable_configuration_says_why(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("hardware: {devices: [{name: fm, remote: nowhere}]}\n")
    servers, problem = listed(bad)
    assert servers == [] and "remote is <address>:<port>" in problem
    assert listed(None) == ([], None)


def test_which_configuration(tmp_path):
    saved = [
        {"name": "a", "path": str(tmp_path / "a.yaml"), "default": True},
        {"name": "b", "path": str(tmp_path / "b.yaml"), "default": False},
    ]
    # the app's own flags first, then the page's choice, then the default
    assert configuration_file(["--config", "/x.yaml"], "b", saved) == Path("/x.yaml")
    assert configuration_file([], "b", saved) == tmp_path / "b.yaml"
    assert configuration_file([], None, saved) == tmp_path / "a.yaml"
    assert configuration_file([], "gone", saved) == tmp_path / "a.yaml"
    assert configuration_file([], None, []) is None


class FakeTunnels:
    def __init__(self, down=None):
        self.down = down
        self.opened = []
        self.closed = []

    def get(self, name):
        return None

    def ensure(self, name, destination, target=None, port=None, reverse=()):
        if self.down:
            raise TunnelDown(self.down)
        self.opened.append((name, destination, target, port))
        return Tunnel(destination, port, target)

    def close(self, name):
        self.closed.append(name)

    def close_all(self):
        self.closed.append("*")


def _watcher(config, answer=(200, HEALTH), tunnels=None, accepts=False):
    asked = []

    def get(url, token):
        asked.append((url, token))
        if isinstance(answer, Exception):
            raise answer
        return answer

    servers = DeviceServers(
        lambda: config,
        tunnels=tunnels or FakeTunnels(),
        get=get,
        token=lambda url: f"token-for-{url}",
        accepts=lambda port: accepts,
        clock=lambda: 1000.0,
    )
    return servers, asked


def test_each_is_asked_for_its_health_with_its_paired_token(config):
    tunnels = FakeTunnels()
    servers, asked = _watcher(config, tunnels=tunnels)
    servers.check()
    rows = servers.state()
    assert [r["status"] for r in rows] == [RUNNING, RUNNING]
    assert rows[0]["devices"][1] == {"name": "camera", "ok": False, "detail": "cold"}
    assert rows[0]["health"]["api_version"] == DEVICE_API_VERSION
    assert rows[0]["kind"] == "remote" and rows[0]["name"] == "meteor@meteor-pc"
    assert asked[0] == ("http://127.0.0.1:8765", "token-for-http://127.0.0.1:8765")
    # the fm's tunnel is opened on the configuration's own port
    assert tunnels.opened == [("127.0.0.1:8765", "meteor@meteor-pc", 8765, 8765)]
    # and the page sees it up, and since when; the other has no tunnel
    assert (rows[0]["tunnel_state"], rows[0]["tunnel_since"]) == ("open", 1000.0)
    assert rows[1]["tunnel_state"] is None
    servers._clock = lambda: 2000.0
    servers.check()
    assert servers.state()[0]["tunnel_since"] == 1000.0  # up all along


def test_a_port_that_already_answers_is_used_as_it_is(config):
    tunnels = FakeTunnels()
    servers, _ = _watcher(config, tunnels=tunnels, accepts=True)
    servers.check()
    assert tunnels.opened == []
    assert servers.state()[0]["status"] == RUNNING
    assert servers.state()[0]["tunnel_state"] == "by_hand"


def test_a_tunnel_that_wont_open_says_ssh_s_words(config):
    tunnels = FakeTunnels(down="meteor@meteor-pc: Permission denied (publickey).")
    servers, asked = _watcher(config, tunnels=tunnels)
    servers.check()
    row = servers.state()[0]
    assert row["status"] == TUNNEL_DOWN and "Permission denied" in row["detail"]
    assert (row["tunnel_state"], row["tunnel_since"]) == ("down", None)
    assert all("8765" not in url for url, _ in asked)


@pytest.mark.parametrize(
    "answer, status",
    [
        (httpx.ConnectError("refused"), NOT_ANSWERING),
        ((401, {"detail": "no"}), NOT_PAIRED),
        ((500, None), NOT_ANSWERING),
        ((200, {**HEALTH, "api_version": "9.9.0"}), TOO_OLD_OR_NEW),
    ],
)
def test_how_one_that_isnt_well_shows(config, answer, status):
    servers, _ = _watcher(config, answer=answer)
    servers.check()
    assert {r["status"] for r in servers.state()} == {status}
    assert servers.state()[0]["detail"]


def test_one_left_out_of_the_configuration_has_its_tunnel_closed(config):
    tunnels = FakeTunnels()
    servers, _ = _watcher(config, tunnels=tunnels)
    servers.check()
    remove_entry(config, "fm")
    servers.check()
    assert "127.0.0.1:8765" in tunnels.closed
    assert [r["serves"] for r in servers.state()] == [["afm"]]


def _entries(path):
    return yaml.safe_load(path.read_text())["hardware"]["devices"]


def test_add_and_remove_touch_one_entry_only(config):
    before = _entries(config)
    add_entry(config, "manipulator", "192.168.0.50", 8770)
    after = _entries(config)
    assert after[:-1] == before
    assert after[-1] == {"name": "manipulator", "remote": "192.168.0.50:8770"}
    # an existing entry keeps its own keys
    add_entry(config, "afm", "127.0.0.1", 8766, tunnel="afm@afm-pc")
    afm = next(e for e in _entries(config) if e["name"] == "afm")
    assert afm == {
        "name": "afm",
        "type": "stage",
        "remote": "127.0.0.1:8766",
        "tunnel": "afm@afm-pc",
    }
    remove_entry(config, "manipulator")
    remove_entry(config, "afm")
    afm = next(e for e in _entries(config) if e["name"] == "afm")
    assert afm == {"name": "afm", "type": "stage"}
    assert [e["name"] for e in _entries(config)] == [e["name"] for e in before]
    # the file as it was is kept beside it
    assert config.with_name("sem.yaml.bak").read_text() == CONFIG


@pytest.mark.parametrize(
    "device, address, port, tunnel, words",
    [
        ("fm; rm", "x", 1, None, "letters and digits"),
        ("fm", "-oProxyCommand=x", 1, None, "host name"),
        ("fm", "pc", 70000, None, "1 to 65535"),
        ("fm", "pc", "eight", None, "a number"),
        ("fm", "pc", 8765, "-oProxyCommand=x", "user@host"),
        ("fm", "192.168.0.20", 8765, "me@pc", "127.0.0.1"),
    ],
)
def test_a_bad_entry_is_refused(config, device, address, port, tunnel, words):
    with pytest.raises(EntryRefused, match=words):
        add_entry(config, device, address, port, tunnel)
    assert config.read_text() == CONFIG


def test_an_older_shape_is_not_rewritten(tmp_path):
    old = tmp_path / "old.yaml"
    old.write_text("hardware:\n  devices:\n    fm: {enabled: true}\n")
    with pytest.raises(EntryRefused, match="older shape"):
        add_entry(old, "fm", "pc", 8765)


@pytest.fixture
def routes(tmp_path, config):
    token_file = tmp_path / "operator-token.json"
    token_file.write_text(json.dumps({"token": OPERATOR_TOKEN}))
    auth = GatewayAuth.from_disk(token_file, tmp_path / "gateway-paired.json")
    servers, _ = _watcher(config)
    gateway = FakeGateway()
    client = TestClient(build_gateway_app(gateway, auth, remote_servers=servers))
    return client, gateway, config


def test_the_page_adds_and_removes_an_entry(routes):
    client, gateway, config = routes
    r = client.post(
        "/gateway/device-servers",
        json={"device": "manipulator", "address": "pc-5", "port": 8770},
        headers=_as(OPERATOR_TOKEN),
    )
    assert r.status_code == 200
    assert _entries(config)[-1]["remote"] == "pc-5:8770"
    r = client.delete(
        "/gateway/device-servers/manipulator", headers=_as(OPERATOR_TOKEN)
    )
    assert r.status_code == 200
    assert "manipulator" not in [e["name"] for e in _entries(config)]
    assert [a[2] for a in gateway.access] == [
        "added a Device Server for manipulator in sem.yaml",
        "removed the Device Server for manipulator in sem.yaml",
    ]
    r = client.post(
        "/gateway/device-servers",
        json={"device": "fm", "address": "pc", "port": 0},
        headers=_as(OPERATOR_TOKEN),
    )
    assert r.status_code == 422 and "1 to 65535" in r.json()["detail"]["message"]


def test_a_hub_cannot_edit_the_configuration(routes):
    client, _, config = routes
    hub = _pair_hub(client, [])
    r = client.post(
        "/gateway/device-servers",
        json={"device": "x", "address": "pc", "port": 1},
        headers=_as(hub["token"]),
    )
    assert r.status_code == 403
    assert (
        client.delete(
            "/gateway/device-servers/fm", headers=_as(hub["token"])
        ).status_code
        == 403
    )
    assert config.read_text() == CONFIG


def test_an_edit_keeps_every_other_line_and_comment(tmp_path):
    """The repo's own sample: comments, aligned values, four-space indents."""
    sample = Path(fibsem.__file__).parent / "config" / "microscope-configuration.yaml"
    path = tmp_path / "commented.yaml"
    path.write_text(sample.read_text())
    original = path.read_text().splitlines()
    add_entry(path, "fm", "127.0.0.1", 8765, tunnel="meteor@meteor-pc")
    added = path.read_text().splitlines()
    assert [line for line in added if line not in original] == [
        "      - name: fm",
        "        remote: 127.0.0.1:8765",
        "        tunnel: meteor@meteor-pc",
    ]
    assert [line for line in original if line not in added] == []
    servers, problem = listed(path)
    assert problem is None and [s.tunnel for s in servers] == ["meteor@meteor-pc"]
    # pointed elsewhere: only its remote line changes, and the tunnel goes
    add_entry(path, "fm", "192.168.0.20", 8765)
    again = path.read_text().splitlines()
    assert [line for line in again if line not in original] == [
        "      - name: fm",
        "        remote: 192.168.0.20:8765",
    ]
    remove_entry(path, "fm")
    assert path.read_text().splitlines() == original


def test_an_existing_entry_keeps_its_comments(tmp_path):
    path = tmp_path / "fm.yaml"
    path.write_text(
        "hardware:\n"
        "  devices:\n"
        "    - name: fm   # the METEOR\n"
        "      driver: Odemis   # its own driver\n"
        "      remote: old-pc:8765   # where it was\n"
        "    - name: stage\n"
        "      rotation_reference: 0\n"
        "calibration: {}\n"
    )
    add_entry(path, "fm", "new-pc", 8765)
    assert path.read_text() == (
        "hardware:\n"
        "  devices:\n"
        "    - name: fm   # the METEOR\n"
        "      driver: Odemis   # its own driver\n"
        "      remote: new-pc:8765\n"
        "    - name: stage\n"
        "      rotation_reference: 0\n"
        "calibration: {}\n"
    )
    remove_entry(path, "fm")
    assert "remote" not in path.read_text() and "driver: Odemis" in path.read_text()


def test_a_name_that_is_no_device_is_refused(config):
    with pytest.raises(EntryRefused, match="no device type"):
        add_entry(config, "toaster", "pc", 8765)
