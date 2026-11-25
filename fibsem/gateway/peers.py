"""Pairing through the gateways (C1 in the Device Server design, Patrick's yes
on 25 Nov).

**On the device PC** (`PeerAsks`): the SEM PC's gateway asks this PC's gateway
to pair, giving only its name. Nothing comes back until the operator here
allows it on the gateway's page. Allow pairs the SEM PC twice: with this PC's
Device Server (a token in ``device-server-paired.json``, the file ``--pair``
writes), and with this gateway as a "gateway", whose one grant starts and
stops the Device Server. One ask waits at a time and lapses after 10 minutes;
the tokens go only to whoever holds that ask's id. Unpairing the gateway row
in Pairing ends its Device Server pairing too (`forget_orphans`).

**On the SEM PC** (`keep`, `kept`, `forget`): the Device Server's token is
kept where ``fibsem-cli pair-device`` keeps it, so the app's remote driver
uses it as it is, and the gateway's token beside it in its own ``0600`` file.
No token is printed or sent anywhere else.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from fibsem.client.pairing import REMOTE_DIR, _file, _keep, paired_token
from fibsem.server.pairing import OPERATOR, PairedComputer, PairedTokens

logger = logging.getLogger(__name__)

__all__ = [
    "GATEWAY",
    "START_DEVICE_SERVER",
    "PeerAsks",
    "forget",
    "keep",
    "kept",
]

# The role a paired gateway has here, and its one grant.
GATEWAY = "gateway"
START_DEVICE_SERVER = "start_device_server"
# How long an ask waits for the operator, one call's wait, how long the
# tokens wait to be collected after Allow, and how long a Deny holds.
ASK_LAPSE_S = 600.0
CALL_WAIT_S = 20.0
HAND_S = 120.0
DENIED_S = 60.0
# The Device Server pairing's name: the gateway's, marked, so unpairing the
# gateway finds it.
MARK = " (gateway)"
_FIELD_LIMIT = 80
GATEWAY_TOKENS = REMOTE_DIR / "gateway"


def _clean(value: object, default: str) -> str:
    return " ".join(str(value or "").split())[:_FIELD_LIMIT] or default


@dataclass(frozen=True)
class PeerAsk:
    """One gateway asking to pair: the name it gives and where it asks from."""

    id: str
    name: str
    origin: str
    asked_at: float

    def to_dict(self) -> dict:
        # The id collects the tokens: the page gets a short one to answer by.
        return {
            "ask": self.id[:8],
            "name": self.name,
            "origin": self.origin,
            "asked_at": self.asked_at,
        }


class PeerAsks:
    """The one ask waiting on this PC's gateway, and how the last ended."""

    def __init__(
        self,
        gateway_tokens: PairedTokens,
        device_tokens: Callable[[], PairedTokens],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._gateway_tokens = gateway_tokens
        self._device_tokens = device_tokens
        self._clock = clock
        self._cond = threading.Condition()
        self._ask: Optional[PeerAsk] = None
        self._since = 0.0
        self._answers: Dict[str, dict] = {}

    def asking(self) -> List[PeerAsk]:
        with self._cond:
            self._lapse()
            return [self._ask] if self._ask is not None else []

    def _lapse(self) -> None:
        now = self._clock()
        if self._ask is not None and now - self._since > ASK_LAPSE_S:
            self._ask = None
        for key, answer in list(self._answers.items()):
            if now > answer["until"]:
                del self._answers[key]

    def ask(
        self,
        name: str,
        origin: str,
        ask_id: Optional[str] = None,
        wait_s: float = CALL_WAIT_S,
    ) -> dict:
        """Ask to pair, or ask again with *ask_id*, waiting up to *wait_s*.
        ``{"state": "waiting", "id"}``, ``{"state": "allowed",
        "device_token", "gateway_token", "name"}`` (to that id only), or
        ``{"state": "denied", "message"}``. Another ask while one waits is
        refused (``busy``)."""
        with self._cond:
            self._lapse()
            if ask_id:
                answered = self._answers.get(ask_id)
                if answered is not None:
                    return {k: v for k, v in answered.items() if k != "until"}
                if self._ask is None or not secrets.compare_digest(
                    self._ask.id, ask_id
                ):
                    return {"state": "lapsed", "message": "Ask again."}
            elif self._ask is not None:
                return {
                    "state": "busy",
                    "message": "Another computer is asking: try in a minute.",
                }
            else:
                self._ask = PeerAsk(
                    id=secrets.token_urlsafe(24),
                    name=_clean(name, "a gateway"),
                    origin=_clean(origin, "unknown"),
                    asked_at=time.time(),
                )
                self._since = self._clock()
                logger.info(
                    "%s asks to pair with this PC's Device Server", self._ask.name
                )
            current = self._ask.id
            self._cond.wait_for(
                lambda: self._ask is None or self._ask.id != current,
                timeout=max(0.0, min(wait_s, CALL_WAIT_S)),
            )
            answered = self._answers.get(current)
            if answered is not None:
                return {k: v for k, v in answered.items() if k != "until"}
            return {"state": "waiting", "id": current}

    def _take(self, short: str) -> PeerAsk:
        self._lapse()
        if self._ask is None or not self._ask.id.startswith(short or "\0"):
            raise LookupError("Nothing asks to pair with that id.")
        ask, self._ask = self._ask, None
        return ask

    def allow(self, short: str) -> PairedComputer:
        """Pair the asking gateway with the Device Server and with this
        gateway (start and stop only), and hold both tokens for it."""
        with self._cond:
            ask = self._take(short)
            device_tokens = self._device_tokens()
            _, device_token = device_tokens.add(ask.name + MARK, OPERATOR)
            computer, gateway_token = self._gateway_tokens.add(
                ask.name, GATEWAY, grants=(START_DEVICE_SERVER,)
            )
            self._answers[ask.id] = {
                "state": "allowed",
                "name": ask.name,
                "device_token": device_token,
                "gateway_token": gateway_token,
                "until": self._clock() + HAND_S,
            }
            self._cond.notify_all()
        logger.info("Paired %s with this PC's Device Server", ask.name)
        return computer

    def deny(self, short: str) -> PeerAsk:
        with self._cond:
            ask = self._take(short)
            self._answers[ask.id] = {
                "state": "denied",
                "message": "The operator at the device PC denied it.",
                "until": self._clock() + DENIED_S,
            }
            self._cond.notify_all()
        return ask

    def forget_orphans(self) -> List[str]:
        """Unpair from the Device Server each gateway no longer paired here:
        unpairing its gateway row in Pairing ends both."""
        paired = {c.name for c in self._gateway_tokens.computers() if c.role == GATEWAY}
        tokens = self._device_tokens()
        gone = []
        for computer in tokens.computers():
            if (
                computer.name.endswith(MARK)
                and computer.name[: -len(MARK)] not in paired
            ):
                tokens.remove(computer.id)
                gone.append(computer.name)
        return gone


# -- the SEM PC's side ---------------------------------------------------------


def keep(device_server_url: str, device_token: str, gateway_token: str) -> None:
    """Keep the Device Server's token where ``fibsem-cli pair-device`` does,
    and its gateway's beside it (both ``0600``)."""
    _keep(device_server_url, device_token, REMOTE_DIR)
    _keep(device_server_url, gateway_token, GATEWAY_TOKENS)


def kept(device_server_url: str) -> Optional[str]:
    """The device PC gateway's token for the Device Server at that url."""
    return paired_token(device_server_url, GATEWAY_TOKENS)


def forget(device_server_url: str) -> None:
    """Forget both tokens for the Device Server at that url."""
    for folder in (REMOTE_DIR, GATEWAY_TOKENS):
        try:
            Path(_file(device_server_url, folder)).unlink()
        except OSError:
            pass
