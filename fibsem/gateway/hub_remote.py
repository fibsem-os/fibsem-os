"""What a paired Hub may ask of this PC's app and agent through the gateway.

A person at the Hub sees an instrument's Agent page there (`fibsem.hub`), so
the Hub's calls for it come here with the Hub's own pairing token. The App
Server and the Agent Server know only the operator and the agent, so the
gateway decides, and passes on what it allows as the operator, with the Hub's
name beside it. This PC keeps the final say:

- **Reads** (GET) of what the gateway passes on: any paired Hub. Never the
  text that connects an agent.
- **Message, answer, Stop**: sending the agent a message (and interrupting it
  for one), answering a review item, and Stop agent. A Hub that was given
  "may start and stop apps", while remote control is on at this PC.
- **The level**: lowered freely, raised only as far as this PC allows
  (``install.gateway.hub_remote``: ``off``, ``watch``, ``assist`` or
  ``drive``). ``off`` switches remote control off: reads only.

Everything else is refused: it is the operator's, at this PC or from a browser
paired with it.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional

from fibsem import config
from fibsem.ai.agent.levels import AgentLevel
from fibsem.server.pairing import SWITCH_MODES

logger = logging.getLogger(__name__)

__all__ = ["HEADER", "LIMITS", "HubRefused", "HubRemote", "act_name", "remote_limit"]

# The Hub's name, on a call the gateway passes on for it.
HEADER = "x-fibsem-hub"
OFF = "off"
# What ``install.gateway.hub_remote`` may be: off, or the highest level a Hub
# may raise the agent to.
LIMITS = (OFF, "watch", "assist", "drive")
_RANK = {level.value: rank for rank, level in enumerate(AgentLevel)}

_READS = ("GET", "HEAD")
# Never read by a Hub: it holds what connects an agent to this PC.
_NOT_READ = re.compile(r"^/app/agent/connect$")
LEVEL = ("PUT", "/app/agent/level")
_ACTS = (
    ("POST", re.compile(r"^/app/agent/messages$"), "Messaged the agent"),
    (
        "POST",
        re.compile(r"^/app/agent/messages/[^/]+/interrupt$"),
        "Interrupted the agent for a message",
    ),
    ("POST", re.compile(r"^/app/agent/review/[^/]+/answer$"), "Answered the agent"),
    ("POST", re.compile(r"^/app/agent/stop$"), "Stopped the agent"),
)


def remote_limit() -> str:
    """``install.gateway.hub_remote``; ``off`` for a value it doesn't know."""
    try:
        limit = str(config.load_user_preferences().install.gateway.hub_remote).lower()
    except Exception:  # noqa: BLE001 - unreadable preferences allow nothing
        logger.debug("could not read the hub remote setting", exc_info=True)
        return OFF
    return limit if limit in LIMITS else OFF


def act_name(method: str, path: str) -> Optional[str]:
    """What a Hub's call did, for this PC's History; None for a read."""
    if (method, path) == LEVEL:
        return "Changed the agent's level"
    for verb, route, name in _ACTS:
        if method == verb and route.match(path):
            return name
    return None


class HubRefused(Exception):
    """Why a Hub's call isn't passed on."""

    def __init__(self, status: int, error_type: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.error_type = error_type


class HubRemote:
    """Decides a paired Hub's calls. *hub* is `GatewayAuth.hub`."""

    def __init__(
        self,
        hub: Callable[[Optional[str]], Optional[Any]],
        operator_token: Callable[[], Optional[str]],
        limit: Callable[[], str] = remote_limit,
    ) -> None:
        self._hub = hub
        self.operator_token = operator_token
        self._limit = limit

    def found(self, token: Optional[str]) -> Optional[Any]:
        """The paired Hub whose token this is, or None."""
        return self._hub(token) if token else None

    def check(self, hub: Any, method: str, path: str) -> bool:
        """Whether *hub*'s call sets the level (then `may_set` decides it).
        Raises `HubRefused` for a call a Hub never makes, or may not now."""
        method = method.upper()
        if method in _READS:
            if _NOT_READ.match(path):
                raise HubRefused(
                    403, "at_this_pc_only", "An agent is connected at this PC only."
                )
            return False
        if act_name(method, path) is None:
            raise HubRefused(
                403,
                "not_from_a_hub",
                "A Hub may message the agent, answer it, stop it and change its "
                "level. This is done at the microscope PC.",
            )
        if SWITCH_MODES not in tuple(getattr(hub, "grants", ()) or ()):
            raise HubRefused(
                403,
                "no_grant",
                "This Hub may read this PC but not act on it: the operator "
                "grants that on the gateway's page.",
            )
        if self._limit() == OFF:
            raise HubRefused(
                403,
                "remote_off",
                "Remote control from a Hub is switched off at the microscope PC "
                "(Settings, This PC, on its gateway's page).",
            )
        return (method, path) == LEVEL

    def may_set(self, level: Any, now: Optional[str]) -> None:
        """Whether a Hub may set *level* while the level is *now*: lowered
        freely, raised as far as this PC allows. Raises `HubRefused`."""
        level = str(level or "").lower()
        if level not in _RANK:
            names = ", ".join(_RANK)
            raise HubRefused(422, "bad_level", f"The level is one of {names}.")
        limit = self._limit()
        if now in _RANK and _RANK[level] <= _RANK[now]:
            return
        if _RANK[level] > _RANK.get(limit, 0):
            raise HubRefused(
                403,
                "level_limit",
                f"From a Hub the level goes up to {limit.title()} at most: "
                "raise it at the microscope PC, or change what a Hub may do in "
                "its gateway's settings.",
            )
