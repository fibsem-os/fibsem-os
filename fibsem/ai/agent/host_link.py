"""fibsem-agent's side of an agent host (`fibsem.server.agent.host`): finding
it, and the agent's token there.

A host run on its own writes a discovery file, as the App Server does, with a
token that only asks to connect. ``fibsem-agent --host`` with no address finds
the host there; with no token set it asks to connect, the operator allows it on
the host's Agent page, and the token it is given is kept (`fibsem.client.pairing`)
for the next start, a resumed session's too.
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
ON_THIS_PC = "the agent host on this PC"
NO_HOST = (
    "fibsem-agent: no agent host is running on this PC. Start fibsem-agent-host, "
    "or give --host its address."
)
# How long to wait for the operator's Allow.
ALLOW_WAIT_S = 600.0
USER_AGENT = f"fibsem-agent/{__version__}"


def _say(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def host_url(given: Optional[str]) -> Optional[str]:
    """--host's address: as given, else the one in the discovery file of a host
    running on this PC; None when none runs."""
    if given:
        return given.rstrip("/")
    found = discovery.read_discovery_file(discovery.HOST_DISCOVERY_FILE)
    return found["url"].rstrip("/") if found is not None else None


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
    """The agent's token at the host at *url*: set (``FIBSEM_HOST_TOKEN``, in
    the environment or the key store), else the one kept from an earlier Allow
    there, else asked for with the discovery file's token, waiting up to
    *wait_s* for the operator. Raises ``LookupError`` when there is nothing to
    ask with or nobody answers, ``PermissionError`` when the operator denies it."""
    token = get_secret(HOST_TOKEN_ENV)
    if token:
        return token
    kept = agent_token(url)
    if kept is not None:
        if not _refused(url, kept):
            return kept
        forget_agent_token(url)
    found = discovery.read_discovery_file(discovery.HOST_DISCOVERY_FILE)
    if found is None or not found.get("connect_token"):
        raise LookupError(
            f"the agent host at {url} wrote no discovery file on this PC: set "
            f"{HOST_TOKEN_ENV} (the environment or the key store) to its agent token."
        )
    say(
        f"fibsem-agent: asking the agent host at {url} to let this agent connect. "
        f"The operator allows it on the Agent page ({url}/ui/agent)."
    )
    token = ask_to_connect(url, found["connect_token"], USER_AGENT, wait_s=wait_s)
    if token is None:
        raise LookupError(
            f"the operator didn't allow this agent within {wait_s / 60:g} min. "
            "Start it again to ask again."
        )
    say("fibsem-agent: allowed.")
    return token
