"""The agents this hub PC runs, one for an instrument.

An agent here is the same ``fibsem-agent`` an instrument's gateway starts
beside itself, started the same way and reaching the instrument's servers
through its gateway: only the PC it runs on differs. The gateway still owns
the instrument's one agent slot. The hub asks it for the slot
(``POST /gateway/agent/slot``), and the gateway answers the agent's token,
made as it makes a local agent's, once: it is kept in memory and in the
agent's environment, never in a file. The hub then renews the slot while the
agent runs, joins again as the same session after the connection was away,
and gives the slot back when the agent ends.

The gateway decides everything: a refusal to renew or to join again (the
level was changed at the PC, Stop or End was pressed there, another agent has
the slot) ends the agent here.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fibsem.ai.agent.manager import AgentRunning, AgentSessionManager
from fibsem.gateway.gateway import GatewayRefused
from fibsem.hub.hub import TOKENS_DIR, Hub, crosses_in_clear
from fibsem.util import timestamps

__all__ = ["API_KEY", "HubAgents", "SESSIONS_DIR"]

log = logging.getLogger(__name__)

API_KEY = "ANTHROPIC_API_KEY"
# Each hub agent's session folder, apart from a gateway's on the same PC.
SESSIONS_DIR = TOKENS_DIR / "sessions"
SLOT = "/gateway/agent/slot"
# A renewal, a release: short, so one instrument's dead connection never holds
# the others' renewals for long.
RENEW_S = 5.0
# The least time between two tries while the connection is away.
RETRY_S = 3.0
POLL_S = 1.0
# The gateway's refusals that mean "the slot was freed while you were away":
# the same session may ask to join again. Every other refusal ends the agent.
_AWAY = ("lost_slot",)
# A gateway that is busy or not ready is asked again, as one that didn't answer.
_LATER = ("busy", "not_ready")


@dataclass
class _Held:
    """One instrument's slot, as this hub holds it for its agent."""

    session_id: str
    # The agent's token, in memory only; None for an agent an earlier run of
    # the hub started (it can renew, but never join again).
    token: Optional[str]
    lease_s: float
    renewed: float
    tried: float
    about: Dict[str, Any]
    # "running", or "away" while its gateway doesn't answer or the slot is
    # being asked for again.
    state: str = "running"
    why: Optional[str] = None


class HubAgents:
    """Start, keep and end the agents this PC runs for the hub's instruments.

    *api_key* gives the key the agents use, None when there is none. `poll`
    does the keeping; `start` runs it in the background.
    """

    def __init__(
        self,
        hub: Hub,
        root: Optional[Path] = None,
        command: Optional[List[str]] = None,
        api_key: Callable[[], Optional[str]] = lambda: None,
        clock: Callable[[], float] = time.monotonic,
        manager: Optional[AgentSessionManager] = None,
    ) -> None:
        self._hub = hub
        self._api_key = api_key
        self._clock = clock
        self._lock = threading.RLock()
        self._held: Dict[str, _Held] = {}
        # The last agent of each instrument that ended here, and why.
        self._ended: Dict[str, Dict[str, Any]] = {}
        self._manager = manager or AgentSessionManager(
            root=root or SESSIONS_DIR,
            command=command,
            reachable=lambda name: self._has_slot(name),
            env_for=lambda name: self._env(name),
        )
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- what the page shows -------------------------------------------------

    def state(self) -> Dict[str, Dict[str, Any]]:
        """Each instrument's agent on this PC: the running one, else the last
        that ended here."""
        with self._lock:
            found = {name: dict(about) for name, about in self._ended.items()}
            for name, held in self._held.items():
                found[name] = {
                    **held.about,
                    "session_id": held.session_id,
                    "state": held.state,
                    "why": held.why,
                }
            return found

    def _has_slot(self, name: str) -> bool:
        with self._lock:
            held = self._held.get(name)
            return held is not None and held.state == "running"

    def _env(self, name: str) -> Dict[str, str]:
        with self._lock:
            held = self._held.get(name)
            token = held.token if held is not None else None
        env = {"FIBSEM_SERVER_TOKEN": token} if token else {}
        key = self._api_key()
        if key:
            env[API_KEY] = key
        return env

    # -- acting --------------------------------------------------------------

    def start(
        self,
        name: str,
        playbook: str,
        goal: Optional[str],
        level: str,
        who: str,
        budget_usd: Optional[float] = None,
        replace: bool = False,
    ) -> Dict[str, Any]:
        """Ask *name*'s gateway for its agent slot and start the agent on this
        PC. Each failure undoes what came before it. Raises `GatewayRefused`."""
        instrument = self._hub.instrument(name)
        if crosses_in_clear(instrument):
            raise GatewayRefused(
                409,
                "plain_http",
                f"{name} is reached by plain http, which would send the agent's "
                "token unprotected. Add it by its SSH address (user@host) to "
                "run an agent on this PC.",
            )
        if not self._api_key():
            raise GatewayRefused(
                409, "no_api_key", "This hub PC has no API key for its agents."
            )
        with self._lock:
            if name in self._held:
                raise GatewayRefused(
                    409,
                    "agent_running",
                    f"An agent for {name} already runs on this PC: end it first.",
                )
            session_id = str(uuid.uuid4())
            answer = self._hub.ask(
                name,
                "POST",
                SLOT,
                {
                    "session_id": session_id,
                    "playbook": playbook,
                    "goal": goal or None,
                    "level": level,
                    "budget_usd": budget_usd,
                    "replace": replace,
                },
            )
            token = answer.get("token") if isinstance(answer, dict) else None
            if not isinstance(token, str) or not token:
                self._release(name, session_id)
                raise GatewayRefused(
                    502, "no_token", f"{name}'s gateway gave the slot and no token."
                )
            now = self._clock()
            about = {
                "started": timestamps.now().isoformat(),
                "playbook": playbook,
                "goal": goal or None,
                "level": level,
                "who": who,
                "budget_usd": budget_usd,
            }
            self._held[name] = _Held(session_id, token, _lease(answer), now, now, about)
            self._ended.pop(name, None)
            args = ["--playbook", playbook, "--url", self._hub.reach(instrument)]
            if budget_usd is not None:
                args += ["--max-budget-usd", f"{float(budget_usd):g}"]
            try:
                self._manager.start(
                    name,
                    session_id,
                    args,
                    goal=goal or None,
                    env=self._env(name),
                    about=about,
                )
            except (AgentRunning, OSError) as error:
                del self._held[name]
                self._release(name, session_id)
                raise GatewayRefused(409, "agent_not_started", str(error)) from None
            log.info("The hub started an agent for %s (%s) for %s", name, level, who)
        return self.state()[name]

    def end(self, name: str, who: str) -> Dict[str, Any]:
        """End *name*'s agent on this PC and give its slot back."""
        with self._lock:
            if name not in self._held:
                raise GatewayRefused(
                    409, "no_agent", f"No agent for {name} runs on this PC."
                )
        self._finish(name, f"ended by {who}")
        return self.state().get(name, {})

    def _finish(self, name: str, why: str, release: bool = True) -> None:
        """Stop *name*'s agent, forget its token, and (with *release*) give
        its slot back: never while the gateway said the slot is not ours."""
        self._manager.end(name)
        with self._lock:
            held = self._held.pop(name, None)
            if held is None:
                return
            self._ended[name] = {
                **held.about,
                "session_id": held.session_id,
                "state": "ended",
                "why": why,
                "ended": timestamps.now().isoformat(),
            }
        if release:
            self._release(name, held.session_id)
        log.info("The hub's agent for %s ended: %s", name, why)

    def _release(self, name: str, session_id: str) -> None:
        try:
            self._hub.ask(
                name,
                "POST",
                f"{SLOT}/release",
                {"session_id": session_id},
                timeout=RENEW_S,
            )
        except GatewayRefused as refusal:
            # Its lease runs out there by itself.
            log.info("%s's slot wasn't given back: %s", name, refusal)

    # -- keeping the slot ----------------------------------------------------

    def poll(self) -> None:
        """Renew each slot that is due, join again where the connection came
        back, and end the agents whose slot the gateway took away."""
        self._manager.poll()
        with self._lock:
            names = list(self._held)
        for name in names:
            try:
                self._keep(name)
            except Exception:  # noqa: BLE001 - one instrument never costs the others
                log.debug("Keeping %s's slot failed.", name, exc_info=True)

    def _keep(self, name: str) -> None:
        with self._lock:
            held = self._held.get(name)
        if held is None:
            return
        agent = self._manager.agent(name)
        if agent is None or agent.state == "ended":
            why = (agent.ended if agent is not None else None) or "it ended"
            self._finish(name, why)
            return
        now = self._clock()
        due = held.lease_s / 3 if held.state == "running" else RETRY_S
        if now - max(held.renewed, held.tried) < due:
            return
        held.tried = now
        body: Dict[str, Any] = {"session_id": held.session_id}
        route = f"{SLOT}/renew"
        if held.why == "rejoin":
            route, body = SLOT, {**body, "rejoin": True, "token": held.token}
        try:
            answer = self._hub.ask(
                name, "POST", route, body, timeout=RENEW_S, read_again=False
            )
        except GatewayRefused as refusal:
            self._refused(name, held, refusal)
            return
        with self._lock:
            held.renewed = self._clock()
            held.lease_s = _lease(answer, held.lease_s)
            if held.state != "running":
                log.info("The hub's agent for %s has its slot again.", name)
            held.state, held.why = "running", None

    def _refused(self, name: str, held: _Held, refusal: GatewayRefused) -> None:
        reason = refusal.error_type
        if refusal.status == 503 or reason in _LATER:
            # Not answering: the agent stays, and the gateway's own rule
            # (its grace time, then Watch) covers the instrument meanwhile.
            with self._lock:
                held.state = "away"
                held.why = "rejoin" if held.why == "rejoin" else str(refusal)
            return
        if reason in _AWAY and held.token and held.why != "rejoin":
            # The slot was freed while this hub was away: ask for it again as
            # the same session, on the next round.
            with self._lock:
                held.state, held.why, held.tried = "away", "rejoin", float("-inf")
            return
        self._finish(name, str(refusal), release=False)

    # -- the background keeping ----------------------------------------------

    def start_keeping(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="hub-agents", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop keeping, end every agent this hub runs and give their slots
        back: an agent never runs on with nobody keeping its slot."""
        self._stop.set()
        with self._lock:
            names = list(self._held)
        for name in names:
            self._finish(name, "the hub was stopped")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll()
            except Exception:  # noqa: BLE001 - the keeping goes on
                log.debug("The hub's agent keeping failed.", exc_info=True)
            self._stop.wait(POLL_S)


def _lease(answer: Any, default: float = 30.0) -> float:
    lease = answer.get("lease_s") if isinstance(answer, dict) else None
    return float(lease) if isinstance(lease, (int, float)) and lease > 0 else default
