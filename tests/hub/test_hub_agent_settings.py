"""What an agent on the hub PC may be given beyond its folder: the data
folders, the allowed sites, connectors and code mode, and every refusal."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from fibsem import config
from fibsem.ai.agent import code_mode, extra_mcp
from fibsem.gateway.gateway import GatewayRefused
from fibsem.hub import agent_settings
from fibsem.hub.hub import Hub
from fibsem.hub.server import build_hub_app, hub_auth
from fibsem.server.pairing import OPERATOR as OPERATOR_ROLE

OPERATOR = {"Authorization": "Bearer operator-token"}


@pytest.fixture
def lab(tmp_path, monkeypatch):
    kept = {"prefs": config.UserPreferences()}
    monkeypatch.setattr(config, "load_user_preferences", lambda: kept["prefs"])
    monkeypatch.setattr(
        config, "update_user_preferences", lambda change: change(kept["prefs"])
    )
    monkeypatch.setattr(
        extra_mcp, "usable", lambda file=None: ({"notebook": object()}, [])
    )
    monkeypatch.setattr(agent_settings.sys, "platform", "linux")
    data = tmp_path / "data"
    data.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    return SimpleNamespace(data=data, other=other, tmp=tmp_path, kept=kept)


def _ready(lab, sites=("pypi.org",)):
    return agent_settings.save(
        {"data_folders": [str(lab.data)], "allowed_sites": list(sites), "offered": True}
    )


def _refused(asked):
    with pytest.raises(GatewayRefused) as refused:
        agent_settings.launch(asked)
    return refused.value


def test_nothing_is_offered_until_it_is_switched_on_at_the_hub_pc(lab):
    shown = agent_settings.read()
    assert shown["offered"] is False and shown["data_folders"] == []
    assert shown["allowed_sites"] == [] and shown["connectors"] == ["notebook"]
    # A start that asks for nothing is as it ever was.
    assert agent_settings.launch(None) == ([], {})
    assert agent_settings.launch({"code": False, "connectors": []}) == ([], {})
    agent_settings.save(
        {"data_folders": [str(lab.data)], "allowed_sites": ["pypi.org"]}
    )
    for asked in (
        {"folder": str(lab.data)},
        {"connectors": ["notebook"]},
        {"code": True},
        {"code": True, "network": True},
    ):
        assert _refused(asked).error_type == "not_offered"


def test_a_start_is_given_what_it_asked_for_from_the_lists(lab):
    _ready(lab, ["pypi.org", "*.example.org"])
    args, said = agent_settings.launch(
        {
            "folder": str(lab.data),
            "connectors": ["notebook"],
            "code": True,
            "network": True,
        }
    )
    assert args == [
        "--code",
        "--allow-site",
        "pypi.org",
        "--allow-site",
        "*.example.org",
        "--read",
        str(lab.data),
        "--connector",
        "notebook",
    ]
    assert said == {
        "code": True,
        "network": True,
        "sites": ["pypi.org", "*.example.org"],
        "folder": str(lab.data),
        "connectors": ["notebook"],
    }
    # What the record says starts it the same way again.
    assert agent_settings.launch(said) == (args, said)
    # Without code mode: files in its own folder, and no network.
    args, said = agent_settings.launch({"folder": str(lab.data)})
    assert args == ["--files", "--read", str(lab.data)]
    assert said == {"folder": str(lab.data)}
    # Code mode alone has no network.
    assert agent_settings.launch({"code": True}) == (["--code"], {"code": True})


def test_a_folder_outside_the_list_is_refused(lab):
    _ready(lab)
    for folder in (str(lab.other), str(lab.data / "deeper"), str(lab.tmp), "/"):
        refused = _refused({"folder": folder})
        assert refused.error_type == "folder_not_listed" and refused.status == 403
    # A listed folder that has gone since is said, not passed on.
    lab.data.rmdir()
    assert _refused({"folder": str(lab.data)}).error_type == "no_folder"


def test_a_site_that_is_not_allowed_is_refused(lab):
    _ready(lab)
    refused = _refused({"code": True, "network": True, "sites": ["github.com"]})
    assert refused.error_type == "site_not_allowed" and refused.status == 403
    assert "github.com" in str(refused)
    # Network is the sandbox's: never without code mode.
    assert _refused({"network": True}).error_type == "network_needs_code"
    agent_settings.save({"allowed_sites": []})
    assert _refused({"code": True, "network": True}).error_type == "no_sites"


@pytest.mark.parametrize(
    "site",
    [
        "localhost",
        "127.0.0.1",
        "10.0.0.5",
        "192.168.1.20",
        "[::1]",
        "scope-1",
        "microscope.local",
        "hub.internal",
        "nas.lan",
        "app.localhost",
    ],
)
def test_a_local_address_is_never_an_allowed_site(lab, site):
    with pytest.raises(GatewayRefused) as refused:
        agent_settings.save({"allowed_sites": ["pypi.org", site]})
    assert refused.value.error_type == "local_address"
    assert agent_settings.read()["allowed_sites"] == []
    # One written into the file by hand is refused at the start too.
    lab.kept["prefs"].install.hub.allowed_sites = [site]
    lab.kept["prefs"].install.hub.offer_agent_settings = True
    assert _refused({"code": True, "network": True}).error_type == "local_address"


@pytest.mark.parametrize(
    "site", ["http://pypi.org", "pypi.org:443", "pypi.org/simple", "", "a b.org"]
)
def test_a_site_is_a_host_name(lab, site):
    with pytest.raises(GatewayRefused) as refused:
        agent_settings.save({"allowed_sites": [site]})
    assert refused.value.error_type == "bad_site"


def test_a_data_folder_is_one_folder_on_this_pc(lab):
    from pathlib import Path

    for folder in (str(lab.tmp / "missing"), "", str(Path.home()), "/"):
        with pytest.raises(GatewayRefused) as refused:
            agent_settings.save({"data_folders": [folder]})
        assert refused.value.error_type == "bad_folder", folder
    with pytest.raises(GatewayRefused) as refused:
        agent_settings.save({"anything": 1})
    assert refused.value.error_type == "unknown_field"


def test_a_connector_that_is_not_usable_and_a_pc_with_no_sandbox_are_refused(
    lab, monkeypatch
):
    _ready(lab)
    assert _refused({"connectors": ["nope"]}).error_type == "no_connector"
    monkeypatch.setattr(agent_settings.sys, "platform", "win32")
    assert agent_settings.read()["code_mode"] is False
    assert _refused({"code": True}).error_type == "no_sandbox"


def test_the_settings_are_changed_only_at_the_hub_pc(lab):
    token_file = lab.tmp / "operator-token.json"
    token_file.write_text('{"token": "operator-token"}')
    auth = hub_auth(token_file, lab.tmp / "paired.json")
    _, paired = auth.pairing.tokens.add("laptop", role=OPERATOR_ROLE)
    browser = {"Authorization": f"Bearer {paired}"}
    hub = Hub(file=lab.tmp / "hub.yaml", tokens=lab.tmp / "hub")
    client = TestClient(build_hub_app(hub, auth))
    assert client.get("/hub/agent-settings").status_code == 401
    body = {
        "data_folders": [str(lab.data)],
        "allowed_sites": ["PyPI.org"],
        "offered": True,
    }
    refused = client.put("/hub/agent-settings", json=body, headers=browser)
    assert refused.status_code == 403
    assert refused.json()["detail"]["error_type"] == "at_this_pc_only"
    saved = client.put("/hub/agent-settings", json=body, headers=OPERATOR)
    assert saved.status_code == 200
    assert saved.json()["allowed_sites"] == ["pypi.org"] and saved.json()["offered"]
    shown = client.get("/hub/agent-settings", headers=browser).json()
    assert shown["data_folders"] == [str(lab.data)] and shown["at_this_pc"] is False
    local = client.put(
        "/hub/agent-settings", json={"allowed_sites": ["localhost"]}, headers=OPERATOR
    )
    assert local.status_code == 422
    assert local.json()["detail"]["error_type"] == "local_address"


def test_the_sandbox_gets_the_folder_read_only_and_only_the_sites(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    given = code_mode.grants([str(data)], ["PyPI.org"], work, True)
    assert given == code_mode.Grants((data.resolve(),), ("pypi.org",))
    made = code_mode.settings(work, {}, True, given)
    sandbox = made["sandbox"]
    assert sandbox["enabled"] and sandbox["failIfUnavailable"]
    assert sandbox["allowUnsandboxedCommands"] is False
    assert sandbox["network"] == {"allowedDomains": ["pypi.org"]}
    assert str(data.resolve()) in sandbox["filesystem"]["allowRead"]
    assert str(data.resolve()) in sandbox["filesystem"]["denyWrite"]
    rule = code_mode._rule(data.resolve())
    assert f"Edit({rule})" in made["permissions"]["deny"]
    assert made["permissions"]["additionalDirectories"] == [str(data.resolve())]
    allowed = code_mode.allowed_tools(work, True, given)
    assert f"Read({rule})" in allowed and f"Edit({rule})" not in allowed
    # With nothing given, no network at all, as before.
    assert code_mode.settings(work, {}, True)["sandbox"]["network"] == {
        "allowedDomains": []
    }
    # Sites are the sandbox's; the folders stay apart.
    with pytest.raises(ValueError, match="add --code"):
        code_mode.grants([], ["pypi.org"], work, False)
    with pytest.raises(ValueError, match="must be apart"):
        code_mode.grants([str(work / "..")], [], work, True)
    with pytest.raises(ValueError, match="local address"):
        code_mode.grants([], ["localhost"], work, True)
