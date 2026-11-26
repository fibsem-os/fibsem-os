"""The slot for an agent the gateway did not start: one a paired Hub runs on
the hub PC, reaching this instrument's App Server through the Hub's SSH
tunnel to the gateway.

The Hub asks for the slot for its session (``POST /gateway/agent/slot``). The
gateway runs the instrument's checks (the level, an app or Agent Server to
act through, no other agent holding the slot), makes the agent's token the
way it does for its own agent (`PairedTokens.add`, kept only hashed), sets
the level and writes the slot down. The token goes once, in that answer, to
the Hub, which keeps it in memory only.

The Hub renews the slot every ``lease_s / 3``. While it does, or within
``grace_s`` of the last renewal, the session keeps the slot and its level.
After that the gateway sets Watch, unpairs the token and frees the slot.
The same session can come back (``rejoin``) with that token: it gets a new
one and the level it last ran at, unless the PC changed the level since,
Stop or End was pressed, or another agent took the slot. A session id alone
never comes back, and another session never gets the old level.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from fibsem.ai.agent.levels import AgentLevel
from fibsem.gateway import slot as slot_mod
from fibsem.gateway.gateway import GatewayRefused
from fibsem.gateway.host import call_host
from fibsem.server.pairing import AGENT, PairedTokens
from fibsem.util import timestamps

logger = logging.getLogger(__name__)

__all__ = ["HUB", "LEASE_S", "GRACE_S", "HubAgents"]

HUB = "hub"
# The Hub renews every third of this.
LEASE_S = 30.0
# The default for ``install.gateway.hub_agent_grace_s``.
GRACE_S = 120.0
# While a hub agent holds the slot, the App Server's level is read this often.
LEVEL_READ_S = 5.0
MAX_SESSION_ID = 128
STOPPED = (AgentLevel.WATCH.value, AgentLevel.OFF.value)

# Why a session no longer holds the slot, as the 409's error_type.
LOST_SLOT = "lost_slot"
LEVEL_CHANGED = "level_changed"
STOPPED_REASON = "stopped"
ENDED = "ended"
OTHER_HOLDER = "other_holder"
BUSY = "busy"
NOT_READY = "not_ready"


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _grace_from_preferences() -> float:
    try:
        from fibsem.config import load_user_preferences

        return float(load_user_preferences().install.gateway.hub_agent_grace_s)
    except Exception:  # noqa: BLE001 - an unreadable file is the default
        return GRACE_S


@dataclass
class _Held:
    """The hub session that holds, or last held, the slot."""

    session_id: str
    hub: str
    holder: str
    token_hash: str
    paired_id: str
    playbook: str
    goal: Optional[str]
    level: str
    budget_usd: Optional[float]
    started: str
    renewed: float
    # Why it no longer holds the slot (`LOST_SLOT` and the others), with
    # words the harness can show; None while it holds it.
    lost: Optional[str] = None
    lost_why: str = ""
    # Whether it went quiet past the grace: the only way back is a rejoin.
    lapsed: bool = False


class HubAgents:
    """The hub agent's slot, beside the gateway's own agent (`GatewayAgent`),
    whose checks, level setting and slot record it shares."""

    def __init__(
        self,
        agent: Any,
        tokens: PairedTokens,
        grace_s: Optional[Callable[[], float]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._agent = agent
        self._tokens = tokens
        self._grace = grace_s or _grace_from_preferences
        self._clock = clock
        self._lock = threading.Lock()
        self._held: Optional[_Held] = None
        self._level_read = float("-inf")
        self._forget_old()

    @property
    def grace_s(self) -> float:
        try:
            return max(LEASE_S, float(self._grace()))
        except (TypeError, ValueError):
            return GRACE_S

    def _forget_old(self) -> None:
        """A hub slot written down before this gateway started: its session
        is not known here (its token's proof went with the last gateway), so
        it ends, and its token with it."""
        held = self._agent.slot.read()
        if held.get("came_in") != HUB or held.get("state") == slot_mod.ENDED:
            return
        self._agent.slot.change(state=slot_mod.ENDED, why="the gateway restarted")
        paired = self._tokens.agent()
        if paired is not None and paired.name == held.get("holder"):
            self._tokens.remove(paired.id)

    # -- reading -----------------------------------------------------------

    def running(self) -> Optional[Dict[str, Any]]:
        """The hub session holding the slot, as the page's running agent."""
        held = self._held
        if held is None or held.lost is not None:
            return None
        return {
            "session_id": held.session_id,
            "started": held.started,
            "playbook": held.playbook,
            "goal": held.goal,
            "level": held.level,
            "budget_usd": held.budget_usd,
            "spent_usd": None,
            "resumed": False,
            "restarts": [],
            "state": "running",
            "via": HUB,
            "hub": held.hub,
            "renewed_s": round(self._clock() - held.renewed, 1),
        }

    def holder(self) -> Optional[str]:
        """The hub agent holding the slot, as the slot names it, or None."""
        held = self._held
        return held.holder if held is not None and held.lost is None else None

    # -- the Hub's calls -----------------------------------------------------

    def take(
        self,
        hub: str,
        session_id: str,
        playbook: str,
        goal: Optional[str],
        level: str,
        budget_usd: Optional[float] = None,
        replace: bool = False,
    ) -> Dict[str, Any]:
        """The slot for *session_id*: the checks, a fresh token, the level.
        Raises `GatewayRefused`."""
        session_id = _session(session_id)
        with self._agent._lock, self._lock:
            held = self._held
            if held is not None and held.lost is None:
                raise GatewayRefused(
                    409,
                    BUSY,
                    f"{held.holder} holds this instrument: release it first.",
                )
            level, url = self._ready(level, replace)
            return self._pair_and_take(
                hub,
                session_id,
                playbook,
                goal,
                level,
                url,
                budget_usd,
                why=f"started by {hub} (hub)",
            )

    def rejoin(self, hub: str, session_id: str, token: str) -> Dict[str, Any]:
        """The same session back after its grace ran out, proved by the token
        it was given: a new token, and its last level, unless something on
        the PC changed since."""
        session_id = _session(session_id)
        with self._agent._lock, self._lock:
            held = self._held
            if (
                held is None
                or held.session_id != session_id
                or held.hub != hub
                or not token
                or not hmac.compare_digest(_hash(str(token)), held.token_hash)
            ):
                raise GatewayRefused(
                    409, LOST_SLOT, "This session doesn't hold this instrument."
                )
            if held.lost is None:
                held.renewed = self._clock()
                return self._answer(None)
            if not held.lapsed:
                raise GatewayRefused(409, held.lost, held.lost_why)
            slot = self._agent.slot.read()
            if (
                self._agent.own_running() is not None
                or slot.get("session_id") != session_id
            ):
                raise _lose(
                    held, OTHER_HOLDER, "Another agent took this instrument since."
                )
            if held.level in STOPPED:
                raise _lose(
                    held,
                    STOPPED_REASON,
                    "The agent was stopped at the PC before it went quiet.",
                )
            level, url = self._ready(held.level, replace=False)
            now = self._level_now(url)
            if now is not None and now != AgentLevel.WATCH.value:
                raise _lose(
                    held,
                    LEVEL_CHANGED,
                    f"The level was changed at the PC (to {now.title()}).",
                )
            return self._pair_and_take(
                hub,
                session_id,
                held.playbook,
                held.goal,
                level,
                url,
                held.budget_usd,
                why=f"rejoined by {hub} (hub)",
                started=held.started,
            )

    def renew(self, hub: str, session_id: str) -> Dict[str, Any]:
        with self._lock:
            held = self._held
            if held is None or held.session_id != session_id or held.hub != hub:
                raise GatewayRefused(
                    409, LOST_SLOT, "This session doesn't hold this instrument."
                )
            self._check(held)
            if held.lost is not None:
                raise GatewayRefused(409, held.lost, held.lost_why)
            held.renewed = self._clock()
            return {"lease_s": LEASE_S, "grace_s": self.grace_s}

    def release(self, hub: str, session_id: str) -> None:
        with self._lock:
            held = self._held
            if held is None or held.session_id != session_id or held.hub != hub:
                return
            if held.lost is None:
                self._end(held, ENDED, f"released by {hub}", "The Hub released it.")
            held.lapsed = False

    # -- the gateway's calls ---------------------------------------------------

    def end(self, who: str) -> bool:
        """End on the gateway's page (or window) while a hub agent holds the
        slot: as a release; its next renewal hears `ENDED`."""
        with self._lock:
            held = self._held
            if held is None or held.lost is not None:
                return False
            self._end(held, ENDED, f"ended by {who}", f"Ended at the PC by {who}.")
            return True

    def follow(self) -> None:
        """A gateway watcher: the slot taken or the token unpaired at the
        PC, the level noted, and the grace."""
        with self._lock:
            held = self._held
            if held is None or held.lost is not None:
                return
            self._check(held)
            if held.lost is not None:
                return
            if self._clock() - held.renewed > self.grace_s:
                self._end(
                    held,
                    LOST_SLOT,
                    f"not renewed for {self.grace_s / 60:.0f} min",
                    "The Hub went quiet: the instrument is at Watch.",
                )
                held.lapsed = True
                return
            if self._clock() - self._level_read >= LEVEL_READ_S:
                self._level_read = self._clock()
                now = self._level_now(self._agent._url())
                if now is not None and now != held.level:
                    held.level = now
                    self._agent.slot.change(session_id=held.session_id, level=now)

    # -- inside ----------------------------------------------------------------

    def _ready(self, level: str, replace: bool):
        """The instrument's checks of a start, not the PC's own (its API key,
        code mode): those are the hub PC's."""
        try:
            return self._agent._ready(level, None, replace, hub=True)
        except GatewayRefused as refusal:
            if refusal.error_type in ("agent_running", "agent_paired"):
                raise GatewayRefused(409, BUSY, str(refusal)) from None
            if refusal.error_type in ("no_app", "app_starting"):
                raise GatewayRefused(409, NOT_READY, str(refusal)) from None
            raise

    def _pair_and_take(
        self,
        hub: str,
        session_id: str,
        playbook: str,
        goal: Optional[str],
        level: str,
        url: str,
        budget_usd: Optional[float],
        why: str,
        started: Optional[str] = None,
    ) -> Dict[str, Any]:
        holder = f"{playbook} via {hub}"
        computer, token = self._tokens.add(holder, role=AGENT)
        try:
            self._agent._level(url, level)
        except GatewayRefused:
            self._tokens.remove(computer.id)
            raise
        self._held = _Held(
            session_id=session_id,
            hub=hub,
            holder=holder,
            token_hash=_hash(token),
            paired_id=computer.id,
            playbook=playbook,
            goal=goal,
            level=level,
            budget_usd=budget_usd,
            started=started or timestamps.now().isoformat(),
            renewed=self._clock(),
        )
        self._level_read = self._clock()
        self._agent.slot.take(holder, HUB, why, session_id=session_id, level=level)
        logger.info("Hub %s holds the agent slot (%s, %s)", hub, playbook, level)
        return self._answer(token)

    def _answer(self, token: Optional[str]) -> Dict[str, Any]:
        answer = {
            "lease_s": LEASE_S,
            "grace_s": self.grace_s,
            "state": self._agent.state(),
        }
        if token is not None:
            answer["token"] = token
        return answer

    def _check(self, held: _Held) -> None:
        """Whether the slot is still the session's: its token paired, the
        slot written down as its own."""
        if held.lost is not None:
            return
        slot = self._agent.slot.read()
        if slot.get("session_id") != held.session_id:
            _mark(held, OTHER_HOLDER, "Another agent took this instrument at the PC.")
            self._unpair(held)
            return
        paired = self._tokens.agent()
        if paired is None or paired.id != held.paired_id:
            _mark(held, ENDED, "The agent was disconnected at the PC.")
            self._agent.slot.change(
                session_id=held.session_id,
                state=slot_mod.ENDED,
                why="disconnected at the PC",
            )

    def _end(self, held: _Held, reason: str, why: str, words: str) -> None:
        """Unpair, Watch, the slot free."""
        _mark(held, reason, words)
        self._unpair(held)
        self._agent.slot.change(
            session_id=held.session_id, state=slot_mod.ENDED, why=why
        )
        url = self._agent._url()
        if url is not None:
            try:
                self._agent._level(url, AgentLevel.WATCH.value)
            except GatewayRefused as refusal:
                logger.warning("The level wasn't set back to Watch: %s", refusal)
        logger.info("The hub agent's slot ended (%s)", why)

    def _unpair(self, held: _Held) -> None:
        paired = self._tokens.agent()
        if paired is not None and paired.id == held.paired_id:
            self._tokens.remove(paired.id)

    def _level_now(self, url: Optional[str]) -> Optional[str]:
        if url is None:
            return None
        try:
            level = call_host(url, "GET", "/capabilities").get("level")
        except (GatewayRefused, AttributeError):
            return None
        return level if isinstance(level, str) else None


def _session(session_id: Any) -> str:
    session_id = str(session_id or "").strip()
    if not session_id or len(session_id) > MAX_SESSION_ID:
        raise GatewayRefused(
            422,
            "no_session",
            f"Say which session, in at most {MAX_SESSION_ID} characters.",
        )
    return session_id


def _mark(held: _Held, reason: str, words: str) -> None:
    held.lost, held.lost_why = reason, words


def _lose(held: _Held, reason: str, words: str) -> GatewayRefused:
    """A rejoin refused for good: the session doesn't come back."""
    _mark(held, reason, words)
    held.lapsed = False
    return GatewayRefused(409, reason, words)
