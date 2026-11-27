"""What a paired Hub may ask of this PC's agent through the gateway: reads,
a message, an answer, Stop agent, and the level within this PC's limit. It is
passed on as the operator with the Hub's name; everything else is refused."""

import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from fibsem import config
from fibsem.gateway import settings as gateway_settings
from fibsem.gateway.gateway import Gateway
from fibsem.gateway.hub_remote import HubRemote, remote_limit
from fibsem.gateway.passthrough import PassThrough
from fibsem.gateway.server import GatewayAuth, build_gateway_app
from fibsem.gateway.settings import KeyRefused
from fibsem.server.pairing import (
    HUB,
    OPERATOR,
    SWITCH_MODES,
    PairedTokens,
    Pairing,
)

OPERATOR_TOKEN = "operator-token"


def _as(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def made(tmp_path):
    """A gateway over a stand-in App Server that notes each call it gets."""
    got, state = [], SimpleNamespace(limit="drive", level="assist")

    def app_server(request: httpx.Request) -> httpx.Response:
        got.append(
            SimpleNamespace(
                method=request.method,
                path=request.url.path,
                token=request.headers.get("authorization"),
                hub=request.headers.get("x-fibsem-hub"),
                body=request.content,
            )
        )
        said = {"level": state.level} if request.url.path == "/capabilities" else {}
        return httpx.Response(200, stream=httpx.ByteStream(json.dumps(said).encode()))

    gateway = Gateway(
        home=tmp_path,
        app_registry=lambda: {},
        last=lambda: None,
        recent=lambda: [],
        new_folder=lambda: str(tmp_path),
        configurations=lambda: [],
        discovery=lambda: None,
    )
    auth = GatewayAuth(
        token=OPERATOR_TOKEN,
        pairing=Pairing(PairedTokens(tmp_path / "paired.json"), roles=(OPERATOR, HUB)),
    )
    _, hub = auth.pairing.tokens.add("hub-pc", HUB, grants=(SWITCH_MODES,))
    _, reader = auth.pairing.tokens.add("reader", HUB)
    passthrough = PassThrough(
        lambda: "http://app",
        transport=httpx.MockTransport(app_server),
        secret=lambda: None,
        hub_remote=HubRemote(
            auth.hub, lambda: OPERATOR_TOKEN, limit=lambda: state.limit
        ),
    )
    with TestClient(build_gateway_app(gateway, auth, passthrough=passthrough)) as page:
        yield SimpleNamespace(
            page=page, got=got, state=state, hub=hub, reader=reader, gateway=gateway
        )


def test_a_hubs_read_is_passed_on_as_the_operator_with_its_name(made):
    for token, name in ((made.hub, "hub-pc"), (made.reader, "reader")):
        answer = made.page.get("/app/agent/recent?rows=5", headers=_as(token))
        assert answer.status_code == 200
        sent = made.got[-1]
        assert sent.token == f"Bearer {OPERATOR_TOKEN}" and sent.hub == name


def test_a_caller_cannot_name_itself_a_hub(made):
    made.page.get(
        "/app/agent", headers={**_as(OPERATOR_TOKEN), "X-Fibsem-Hub": "hub-pc"}
    )
    assert made.got[-1].hub is None
    assert made.got[-1].token == f"Bearer {OPERATOR_TOKEN}"


def test_a_token_nobody_paired_goes_on_as_it_came(made):
    made.page.get("/app/agent", headers=_as("stranger"))
    assert made.got[-1].token == "Bearer stranger" and made.got[-1].hub is None


def test_a_hub_never_reads_what_connects_an_agent(made):
    answer = made.page.get("/app/agent/connect", headers=_as(made.hub))
    assert answer.status_code == 403
    assert answer.json()["detail"]["error_type"] == "at_this_pc_only"
    assert made.got == []


@pytest.mark.parametrize(
    "route, said",
    [
        ("/app/agent/messages", "Messaged the agent"),
        ("/app/agent/messages/m1/interrupt", "Interrupted the agent for a message"),
        ("/app/agent/review/r1/answer", "Answered the agent"),
        ("/app/agent/stop", "Stopped the agent"),
    ],
)
def test_a_hub_messages_answers_and_stops_the_agent(made, route, said):
    answer = made.page.post(route, json={"text": "hello"}, headers=_as(made.hub))
    assert answer.status_code == 200
    sent = made.got[-1]
    assert (sent.method, sent.path, sent.hub) == ("POST", route, "hub-pc")
    assert sent.token == f"Bearer {OPERATOR_TOKEN}"
    assert json.loads(sent.body) == {"text": "hello"}
    last = made.gateway.history(1).switches[0]
    assert (last.who, last.via, last.result) == ("hub-pc", "hub", said)


@pytest.mark.parametrize(
    "method, route",
    [
        ("POST", "/app/agent/requests/r1/approve"),
        ("POST", "/app/agent/suggestions/s1/approve"),
        ("POST", "/app/agent/connections/c1/allow"),
        ("POST", "/app/workflow/start"),
        ("POST", "/microscope/stage/move"),
        ("PUT", "/app/library/skills/a"),
        ("DELETE", "/app/library/skills/a"),
        ("POST", "/app/agent/context-reset"),
    ],
)
def test_everything_else_of_a_hubs_is_refused_here(made, method, route):
    answer = made.page.request(method, route, json={}, headers=_as(made.hub))
    assert answer.status_code == 403
    assert answer.json()["detail"]["error_type"] == "not_from_a_hub"
    assert made.got == []


def test_a_hub_that_only_reads_acts_on_nothing(made):
    answer = made.page.post("/app/agent/stop", json={}, headers=_as(made.reader))
    assert answer.status_code == 403
    assert answer.json()["detail"]["error_type"] == "no_grant"
    assert made.got == []


def test_remote_control_switched_off_leaves_a_hub_its_reads(made):
    made.state.limit = "off"
    for route in ("/app/agent/messages", "/app/agent/stop"):
        answer = made.page.post(route, json={"text": "x"}, headers=_as(made.hub))
        assert answer.status_code == 403
        assert answer.json()["detail"]["error_type"] == "remote_off"
    level = made.page.put(
        "/app/agent/level", json={"level": "watch"}, headers=_as(made.hub)
    )
    assert level.json()["detail"]["error_type"] == "remote_off"
    assert made.got == []
    assert made.page.get("/app/agent", headers=_as(made.hub)).status_code == 200


@pytest.mark.parametrize(
    "limit, now, asked, passed",
    [
        ("drive", "watch", "drive", True),
        ("assist", "watch", "assist", True),
        ("assist", "watch", "drive", False),
        ("watch", "off", "watch", True),
        ("watch", "watch", "assist", False),
        # lowered freely, even to a level above the limit
        ("watch", "drive", "assist", True),
        ("watch", "drive", "off", True),
        ("assist", "assist", "ASSIST", True),
    ],
)
def test_a_hub_lowers_the_level_freely_and_raises_it_to_the_limit(
    made, limit, now, asked, passed
):
    made.state.limit, made.state.level = limit, now
    answer = made.page.put(
        "/app/agent/level", json={"level": asked}, headers=_as(made.hub)
    )
    sent = [call for call in made.got if call.method == "PUT"]
    if passed:
        assert answer.status_code == 200
        assert json.loads(sent[0].body) == {"level": asked}
        assert sent[0].hub == "hub-pc"
    else:
        assert answer.status_code == 403
        assert answer.json()["detail"]["error_type"] == "level_limit"
        assert sent == []


def test_a_level_the_app_does_not_tell_is_held_to_the_limit(made):
    made.state.limit, made.state.level = "watch", None
    answer = made.page.put(
        "/app/agent/level", json={"level": "assist"}, headers=_as(made.hub)
    )
    assert answer.status_code == 403
    bad = made.page.put("/app/agent/level", json={"level": "up"}, headers=_as(made.hub))
    assert bad.status_code == 422


def test_the_operator_is_never_held_to_a_hubs_limit(made):
    made.state.limit = "off"
    answer = made.page.put(
        "/app/agent/level", json={"level": "drive"}, headers=_as(OPERATOR_TOKEN)
    )
    assert answer.status_code == 200 and made.got[-1].hub is None


def test_what_a_hub_may_do_is_set_at_this_pc_only(monkeypatch):
    preferences = config.UserPreferences()
    monkeypatch.setattr(config, "load_user_preferences", lambda: preferences)
    monkeypatch.setattr(
        config, "update_user_preferences", lambda change: change(preferences)
    )
    assert remote_limit() == "drive"
    assert gateway_settings.this_pc()["hub_remote"] == "drive"
    with pytest.raises(KeyRefused) as refused:
        gateway_settings.save_this_pc({"hub_remote": "off"}, local=False)
    assert refused.value.error_type == "at_this_pc_only"
    with pytest.raises(KeyRefused):
        gateway_settings.save_this_pc({"hub_remote": "always"}, local=True)
    assert (
        gateway_settings.save_this_pc({"hub_remote": "off"}, local=True)["hub_remote"]
        == "off"
    )
    preferences.install.gateway.hub_remote = "anything"
    assert remote_limit() == "off"
