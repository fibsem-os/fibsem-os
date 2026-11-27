"""The Hub's agents against a gateway's real slot routes: the two sides of an
agent that runs on the hub PC, with a link that goes away and comes back."""

from fibsem.hub.agents import HubAgents
from fibsem.hub.hub import Hub
from fibsem.server.pairing import AGENT, HUB, SWITCH_MODES
from tests.gateway.test_gateway_hub_agent import (  # noqa: F401 - fixtures
    _as,
    _lapse,
    client,
    made,
)
from tests.hub.test_hub_agents import FakeManager


def test_the_hubs_own_agents_against_these_routes(client, made, tmp_path):  # noqa: F811
    c, tokens = client.client, client.auth.pairing.tokens
    _, hub_token = tokens.add("hub-pc", HUB, grants=(SWITCH_MODES,))
    down = [False]

    def send(url, method, route, token, body, timeout):
        if down[0]:
            raise OSError("down")
        r = c.request(method, route, json=body, headers=_as(token))
        return r.status_code, r.json()

    hub = Hub(
        file=tmp_path / "hub.yaml",
        tokens=tmp_path / "hubtokens",
        fetch=lambda *a: (200, {}),
        pair=lambda url, code, name: (hub_token, "hub"),
        send=send,
    )
    hub.add("Scope 1", url="https://scope-1:8010")
    hub.pair("Scope 1", "1")
    now = [0.0]
    manager = FakeManager()
    side = HubAgents(hub, api_key=lambda: "k", clock=lambda: now[0], manager=manager)
    state = side.start("Scope 1", "default", "Mill", "assist", "pat")
    first = manager.started[0][3]["FIBSEM_SERVER_TOKEN"]
    assert tokens.match(first).role == AGENT
    assert made.app.level == "assist"
    # renew
    now[0] += 11
    made.clock.now += 11
    side.poll()
    assert side.state()["Scope 1"]["state"] == "running"
    # link away past the grace, then back: rejoin with a new token
    down[0] = True
    now[0] += 11
    side.poll()
    assert side.state()["Scope 1"]["state"] == "away"
    _lapse(made, client.agent)
    assert made.app.level == "watch" and tokens.match(first) is None
    down[0] = False
    now[0] += 4
    side.poll()
    now[0] += 4
    side.poll()
    assert side.state()["Scope 1"]["state"] == "running", side.state()
    fresh = manager.agents["Scope 1"].env["FIBSEM_SERVER_TOKEN"]
    assert fresh != first and tokens.match(fresh).role == AGENT
    assert made.app.level == "assist"
    # the level changed at the PC while away: final
    down[0] = True
    now[0] += 11
    side.poll()
    _lapse(made, client.agent)
    made.app.level = "suggest"
    down[0] = False
    for _ in range(3):
        now[0] += 4
        side.poll()
    ended = side.state()["Scope 1"]
    assert ended["state"] == "ended", ended
    assert manager.agents["Scope 1"].state == "ended"
    assert state["session_id"] == ended["session_id"]
    assert tokens.agent() is None
