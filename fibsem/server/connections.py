"""A local agent asks to connect, and the operator allows or denies it.

The discovery file (`fibsem.server.discovery`) carries a connect token, not an
agent's. It reaches ``POST /connect`` and nothing else, so a program on the PC
that reads the file can only ask. The ask waits on the operator's card (the
window's, or the web app's Agent page). **Allow** pairs it as an agent
(`PairedTokens.add`), listed with Unpair like a computer paired with a code;
the asking program gets the token and keeps it (`fibsem.client.pairing`), so
later connections don't ask. **Deny** refuses it, and the next ask waits a
minute.

One ask at a time: every program asking with the connect token joins it (an
agent's sidecar and its ``fibsem-mcp watch`` are one card), and all of them get
the token an Allow mints.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from fibsem.server.pairing import AGENT, PairedComputer, PairedTokens

__all__ = ["ASK_WAIT_S", "ConnectAsk", "Connections"]

# How long one POST /connect waits for the operator; the asker asks again.
ASK_WAIT_S = 25.0
# How long after an Allow the token is handed to whoever asks with the connect
# token: the agent's other programs, asking at the same time.
HAND_S = 120.0
# How long after a Deny the next ask is refused, so a looping client can't
# keep the card up.
DENIED_S = 60.0
_FIELD_LIMIT = 80


def _clean(value: object, default: str) -> str:
    return " ".join(str(value or "").split())[:_FIELD_LIMIT] or default


@dataclass(frozen=True)
class ConnectAsk:
    """One ask to connect: the client as it names itself, where it connects
    from (this PC, or an address over SSH) and the PC user it runs as."""

    id: str
    client: str
    origin: str
    user: str
    asked_at: float

    @property
    def name(self) -> str:
        """What the paired list calls it: "claude-code on this PC"."""
        return f"{self.client.split()[0]} on {self.origin}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "client": self.client,
            "origin": self.origin,
            "user": self.user,
            "asked_at": self.asked_at,
            "name": self.name,
        }


@dataclass(frozen=True)
class _Decided:
    ask_id: str
    until: float
    token: Optional[str] = None
    computer: Optional[PairedComputer] = None
    reason: str = ""


class Connections:
    """The ask waiting for the operator, and how the last one ended.

    ``on_change`` listeners are called after an ask arrives or is decided,
    from whichever thread did it."""

    def __init__(
        self, tokens: PairedTokens, clock: Callable[[], float] = time.monotonic
    ):
        self.tokens = tokens
        self._clock = clock
        self._changed_cond = threading.Condition()
        self._ask: Optional[ConnectAsk] = None
        self._allowed: Optional[_Decided] = None
        self._denied: Optional[_Decided] = None
        self._listeners: List[Callable[[], None]] = []

    def on_change(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Follow asks and decisions; returns the unsubscribe."""
        self._listeners.append(callback)
        return lambda: self._listeners.remove(callback)

    def _changed(self) -> None:
        for callback in list(self._listeners):
            try:
                callback()
            except Exception:  # noqa: BLE001 - a listener never blocks an ask
                logging.exception("Connections listener failed")

    def asking(self) -> List[ConnectAsk]:
        """The ask waiting for the operator: none or one."""
        with self._changed_cond:
            return [self._ask] if self._ask is not None else []

    def ask(
        self,
        client: str,
        origin: str,
        user: str,
        wait_s: float = ASK_WAIT_S,
    ) -> dict:
        """Ask to connect, or join the ask already waiting, and wait up to
        *wait_s* for the operator. ``{"state": "allowed", "token", "name"}``,
        ``{"state": "denied", "message", "retry_in_s"}`` or ``{"state":
        "waiting", "id"}``, when the asker asks again."""
        with self._changed_cond:
            answered = self._answer()
            if answered is not None:
                return answered
            new = self._ask is None
            if new:
                self._ask = ConnectAsk(
                    id=secrets.token_hex(4),
                    client=_clean(client, "an agent"),
                    origin=_clean(origin, "this PC"),
                    user=_clean(user, "unknown"),
                    asked_at=time.time(),
                )
            ask_id = self._ask.id
        if new:
            self._changed()
        with self._changed_cond:
            self._changed_cond.wait_for(
                lambda: self._ask is None or self._ask.id != ask_id,
                timeout=max(0.0, wait_s),
            )
            answered = self._answer()
        return answered or {"state": "waiting", "id": ask_id}

    def _answer(self) -> Optional[dict]:
        now = self._clock()
        allowed, denied = self._allowed, self._denied
        # An agent unpaired since its Allow asks afresh.
        if (
            allowed is not None
            and now < allowed.until
            and self.tokens.match(allowed.token) is not None
        ):
            return {
                "state": "allowed",
                "token": allowed.token,
                "name": allowed.computer.name,
            }
        if denied is not None and now < denied.until:
            message = "The operator denied this agent."
            if denied.reason:
                message += f" Their reason: {denied.reason}"
            return {
                "state": "denied",
                "message": f"{message} Ask again in a minute.",
                "retry_in_s": round(denied.until - now),
            }
        return None

    def _take(self, ask_id: str) -> ConnectAsk:
        if self._ask is None or self._ask.id != ask_id:
            raise LookupError("No agent is asking to connect with that id.")
        ask, self._ask = self._ask, None
        return ask

    def allow(self, ask_id: str) -> PairedComputer:
        """Pair the asking agent: a token of its own, listed with Unpair, and
        handed to it. Raises ``LookupError`` when nothing asks with that id."""
        with self._changed_cond:
            ask = self._take(ask_id)
            computer, token = self.tokens.add(ask.name, AGENT)
            self._allowed = _Decided(
                ask_id, self._clock() + HAND_S, token=token, computer=computer
            )
            self._denied = None
            self._changed_cond.notify_all()
        self._changed()
        return computer

    def deny(self, ask_id: str, reason: str = "") -> ConnectAsk:
        """Refuse the asking agent, with the operator's reason; the next ask
        waits a minute. Raises ``LookupError`` when nothing asks with that id."""
        with self._changed_cond:
            ask = self._take(ask_id)
            self._denied = _Decided(
                ask_id, self._clock() + DENIED_S, reason=reason.strip()
            )
            self._allowed = None
            self._changed_cond.notify_all()
        self._changed()
        return ask
