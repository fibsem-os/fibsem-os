"""fibsem-agent's side of an Agent Server (`fibsem.server.agent.host`): finding
it, and the agent's token there.

A host sits behind this PC's gateway, as the App Server does: the gateway
passes calls to it, and allows agents. ``fibsem-agent --host`` with no address
connects through the gateway; with no token set it asks to connect there
(``gateway.json``'s token only asks), the operator allows it on the gateway's
Agent page, and the token it is given is kept (`fibsem.client.pairing`) for the
next start, a resumed session's too.
"""

import sys
from typing import Callable, Optional

import httpx

from fibsem import __version__
from fibsem.client.pairing import agent_token, ask_to_connect, forget_agent_token
from fibsem.client.transport import HOST_TOKEN_ENV
from fibsem.keystore import get_secret
from fibsem.server import discovery

# What --host shows when no host runs on this PC yet (--show-config only).
ON_THIS_PC = "the Agent Server on this PC"
NO_HOST = (
    "fibsem-agent: no Agent Server is running on this PC. Start the gateway and "
    "fibsem-agent-server, or give --host an address."
)
# How long to wait for the operator's Allow.
ALLOW_WAIT_S = 600.0
USER_AGENT = f"fibsem-agent/{__version__}"


def _say(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def host_url(given: Optional[str]) -> Optional[str]:
    """--host's address: as given, else this PC's gateway while a host runs
    behind it; None when no host runs."""
    if given:
        return given.rstrip("/")
    if discovery.read_discovery_file(discovery.HOST_DISCOVERY_FILE) is None:
        return None
    gateway = discovery.read_gateway_file()
    return gateway["url"].rstrip("/") if gateway is not None else None


def _refused(url: str, token: str) -> bool:
    """Whether the host at *url* refuses *token* (it was unpaired, or its file
    of allowed agents was lost). Not reaching the host is not a refusal: the
    harness waits for it."""
    try:
        response = httpx.get(
            f"{url}/capabilities",
            headers={"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT},
            timeout=10.0,
        )
    except httpx.HTTPError:
        return False
    return response.status_code == 401


def host_token(
    url: str, wait_s: float = ALLOW_WAIT_S, say: Callable[[str], None] = _say
) -> str:
    """The agent's token at the host at *url* (this PC's gateway): set
    (``FIBSEM_HOST_TOKEN``, in the environment or the key store), else the one
    kept from an earlier Allow there, else asked for with the gateway's
    connect token, waiting up to *wait_s* for the operator. Raises ``LookupError`` when there is nothing to
    ask with or nobody answers, ``PermissionError`` when the operator denies it."""
    token = get_secret(HOST_TOKEN_ENV)
    if token:
        return token
    kept = agent_token(url)
    if kept is not None:
        if not _refused(url, kept):
            return kept
        forget_agent_token(url)
    found = discovery.read_gateway_file()
    if found is None or not found.get("connect_token"):
        raise LookupError(
            f"no gateway runs on this PC to ask for the Agent Server at {url}: "
            f"start it, or set {HOST_TOKEN_ENV} (the environment or the key "
            "store) to the agent's token."
        )
    say(
        f"fibsem-agent: asking this PC's gateway ({url}) to let this agent "
        f"connect. The operator allows it on the Agent page ({url}/ui/agent)."
    )
    token = ask_to_connect(url, found["connect_token"], USER_AGENT, wait_s=wait_s)
    if token is None:
        raise LookupError(
            f"the operator didn't allow this agent within {wait_s / 60:g} min. "
            "Start it again to ask again."
        )
    say("fibsem-agent: allowed.")
    return token
