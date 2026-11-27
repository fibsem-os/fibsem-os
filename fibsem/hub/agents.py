"""The agents this hub PC runs, one for an instrument.

An agent here is the same ``fibsem-agent`` an instrument's gateway starts
beside itself, started the same way and reaching the instrument's servers
through its gateway: only the PC it runs on differs. The gateway still owns
the instrument's one agent slot. The hub asks it for the slot
(``POST /gateway/agent/slot``), and the gateway answers the agent's token,
made as it makes a local agent's, once: it is kept in memory and in the
agent's environment, never in a file. The hub then renews the slot while the
agent runs, joins again as the same session after the connection was away
(the gateway then answers a new token, and the agent carries its session on
with it), and gives the slot back when the agent ends.

An agent outlives a Hub that crashed or was restarted: the next Hub takes it
back (`_adopt`), renews its slot, and where the gateway had freed the slot
meanwhile asks for it again as the same paired Hub, with no token (the
gateway's own checks still decide). A Hub stopped on purpose ends its agents
first. A session that ended is carried on by a person, with `resume`.

The gateway decides everything: a refusal to renew or to join again (the
level was changed at the PC, Stop or End was pressed there, another agent has
the slot) ends the agent here.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fibsem.ai.agent.manager import (
    MANAGED,
    RECORD,
    AgentRunning,
    AgentSessionManager,
    read_json,
    write_json,
)
from fibsem.gateway.gateway import GatewayRefused
from fibsem.hub.history import AGENT, HubHistory
from fibsem.hub.hub import TOKENS_DIR, Hub, crosses_in_clear
from fibsem.keystore import stored_secret
from fibsem.util import timestamps

__all__ = ["API_KEY", "HubAgents", "SESSIONS_DIR", "stored_key"]

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


def stored_key() -> Optional[str]:
    """This PC's API key as a gateway reads its own: the environment's, else
    the one in this PC's key store. Read at each start, never kept."""
    return os.environ.get(API_KEY) or stored_secret(API_KEY)


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
    # When its gateway stopped answering, for the log.
    away_since: float = 0.0
    # What the agent was started with, to start it again on its session.
    args: List[str] = field(default_factory=list)
    # When the slot came back with a new token, until the agent runs again
    # with it: how long carrying on took, for the log.
    back_at: Optional[float] = None


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
        history: Optional[HubHistory] = None,
    ) -> None:
        self._hub = hub
        self._history = history
        self._root = root or SESSIONS_DIR
        self._api_key = api_key
        self._clock = clock
        self._lock = threading.RLock()
        self._held: Dict[str, _Held] = {}
        # The last agent of each instrument that ended here, and why.
        self._ended: Dict[str, Dict[str, Any]] = {}
        self._manager = manager or AgentSessionManager(
            root=self._root,
            command=command,
            reachable=lambda name: self._has_slot(name),
            env_for=lambda name: self._env(name),
        )
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._adopt()

    @property
    def root(self) -> Path:
        """Where this hub PC keeps its agents' session folders."""
        return self._root

    def _note(self, name: str, what: str, who: str, ok: bool = True) -> None:
        if self._history is not None:
            self._history.write(AGENT, what, who, name, ok)

    def _adopt(self) -> None:
        """Take back the agents an earlier run of the Hub left running (it
        crashed or was restarted; one stopped on purpose ended them): each
        slot is renewed from the next round on. Its token went with that
        Hub, so where the gateway freed the slot meanwhile this Hub asks for
        it again as the same paired Hub."""
        for instrument in self._hub.instruments():
            agent = self._manager.agent(instrument.name)
            if agent is None or agent.state == "ended":
                continue
            managed = read_json(agent.folder / MANAGED)
            self._held[instrument.name] = _Held(
                agent.session_id,
                None,
                30.0,
                # Renewed on the first round, not a third of a lease later.
                float("-inf"),
                float("-inf"),
                dict(managed.get("about") or {}),
                state="away",
                why="the Hub was restarted",
                away_since=self._clock(),
                args=list(managed.get("args") or []),
            )
            log.info(
                "The Hub took back its agent for %s (session %s) after a restart.",
                instrument.name,
                agent.session_id,
            )
            self._note(
                instrument.name,
                "Took the agent back after the Hub restarted",
                "the Hub",
            )

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
        instrument = self._may_run(name)
        with self._lock:
            self._free(name)
            session_id = str(uuid.uuid4())
            token, answer = self._take(
                name, session_id, playbook, goal, level, budget_usd, replace
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
            args = ["--playbook", playbook, "--url", self._hub.reach(instrument)]
            if budget_usd is not None:
                args += ["--max-budget-usd", f"{float(budget_usd):g}"]
            self._held[name] = _Held(
                session_id, token, _lease(answer), now, now, about, args=list(args)
            )
            self._ended.pop(name, None)
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
        self._note(name, f"Started an agent on this hub PC ({playbook}, {level})", who)
        return self.state()[name]

    def _may_run(self, name: str):
        """*name*'s instrument, when an agent for it may run on this PC."""
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
                409,
                "no_api_key",
                "No API key on this hub PC: set it in the Hub's Settings, or "
                f"run fibsem-cli key set {API_KEY} here.",
            )
        return instrument

    def _free(self, name: str) -> None:
        if name in self._held:
            raise GatewayRefused(
                409,
                "agent_running",
                f"An agent for {name} already runs on this PC: end it first.",
            )

    def _take(
        self,
        name: str,
        session_id: str,
        playbook: str,
        goal: Optional[str],
        level: str,
        budget_usd: Optional[float],
        replace: bool,
    ):
        """Ask *name*'s gateway for its slot: the agent's token and the
        gateway's answer."""
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
        return token, answer

    def resume(
        self,
        name: str,
        session_id: str,
        level: str,
        who: str,
        budget_usd: Optional[float] = None,
        replace: bool = False,
    ) -> Dict[str, Any]:
        """Carry on a session that ended on this hub PC, for the instrument
        it ran for: its gateway is asked for the slot as for a start (its
        checks, a fresh token, the level), and the agent starts again in the
        session's own folder. Raises `GatewayRefused`."""
        folder = self._manager.folder(str(session_id or ""))
        managed = read_json(folder / MANAGED) if folder else {}
        record = read_json(folder / RECORD) if folder else {}
        if not record.get("session_id"):
            raise GatewayRefused(
                404, "no_session", f"No session {session_id} on this hub PC."
            )
        if managed.get("microscope") != name:
            raise GatewayRefused(
                409,
                "other_instrument",
                f"That session ran for {managed.get('microscope') or 'no instrument'},"
                f" not {name}.",
            )
        instrument = self._may_run(name)
        started = managed.get("about") or {}
        playbook = record.get("playbook")
        playbook = playbook.get("name") if isinstance(playbook, dict) else playbook
        playbook = playbook or started.get("playbook") or ""
        goal = record.get("goal") or started.get("goal")
        with self._lock:
            self._free(name)
            token, answer = self._take(
                name, session_id, playbook, goal, level, budget_usd, replace
            )
            now = self._clock()
            about = {
                "started": timestamps.now().isoformat(),
                "playbook": playbook,
                "goal": goal,
                "level": level,
                "who": who,
                "budget_usd": budget_usd,
                "resumed": True,
            }
            args = ["--playbook", playbook, "--url", self._hub.reach(instrument)]
            for key in ("model", "effort"):
                if record.get(key):
                    args += [f"--{key}", str(record[key])]
            if budget_usd is not None:
                args += ["--max-budget-usd", f"{float(budget_usd):g}"]
            self._held[name] = _Held(
                session_id, token, _lease(answer), now, now, about, args=list(args)
            )
            self._ended.pop(name, None)
            try:
                self._manager.resume(
                    name, session_id, args, env=self._env(name), about=about
                )
            except (AgentRunning, OSError) as error:
                del self._held[name]
                self._release(name, session_id)
                raise GatewayRefused(409, "agent_not_started", str(error)) from None
            log.info(
                "The hub resumed session %s for %s (%s) for %s",
                session_id,
                name,
                level,
                who,
            )
        self._note(name, f"Resumed a session on this hub PC ({playbook}, {level})", who)
        return self.state()[name]

    def end(self, name: str, who: str) -> Dict[str, Any]:
        """End *name*'s agent on this PC and give its slot back."""
        with self._lock:
            if name not in self._held:
                raise GatewayRefused(
                    409, "no_agent", f"No agent for {name} runs on this PC."
                )
        self._finish(name, f"ended by {who}", who=who)
        return self.state().get(name, {})

    def _finish(
        self, name: str, why: str, release: bool = True, who: str = "the Hub"
    ) -> None:
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
        self._note(name, f"The agent on this hub PC ended: {why}", who)

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
        if held.token is None and agent.state == "waiting":
            # Taken back after a restart, and its process is gone: nothing
            # here has a token to start it with again.
            self._finish(
                name, "it stopped while the Hub was not running: resume it in Sessions"
            )
            return
        now = self._clock()
        if held.back_at is not None and agent.state == "running":
            log.info(
                "The hub's agent for %s runs again %.0f s after its slot came "
                "back (its first call follows in its own log).",
                name,
                now - held.back_at,
            )
            held.back_at = None
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
        fresh = answer.get("token") if isinstance(answer, dict) else None
        if isinstance(fresh, str) and fresh and fresh != held.token:
            # Joined again after the slot was freed: the gateway unpaired the
            # old token then and made this one. The agent is stopped and
            # carries the same session on with it.
            log.info(
                "%s's gateway gave the slot back with a new token; the agent "
                "carries its session on with it.",
                name,
            )
            with self._lock:
                held.token = fresh
                held.back_at = self._clock()
            self._reach_again(name, held, agent)
            asked = self._clock()
            parked = self._manager.park(name)
            log.info(
                "The hub's agent for %s stopped %.0f s after it was asked, to "
                "start again with the new token.",
                name,
                self._clock() - asked,
            )
            if parked is agent and agent.state != "ended":
                agent.env = self._env(name)
                self._manager.unpark(name)
            elif not self._carry_on(name, held):
                return
        with self._lock:
            held.renewed = self._clock()
            held.lease_s = _lease(answer, held.lease_s)
            if held.state != "running":
                log.info(
                    "The hub's agent for %s has its slot again, after %.0f s.",
                    name,
                    self._clock() - held.away_since,
                )
            held.state, held.why = "running", None

    def _reach_again(self, name: str, held: _Held, agent: Any) -> None:
        """Where the agent reaches its instrument now, in what it is started
        again with: a tunnel opened by a new Hub may have another port."""
        try:
            url = self._hub.reach(self._hub.instrument(name))
        except (GatewayRefused, OSError):
            return
        args = list(held.args)
        if "--url" in args[:-1] and args[args.index("--url") + 1] != url:
            args[args.index("--url") + 1] = url
            held.args = args
            managed = read_json(agent.folder / MANAGED)
            if managed:
                managed["args"] = list(args)
                write_json(agent.folder / MANAGED, managed)

    def _carry_on(self, name: str, held: _Held) -> bool:
        """Start *name*'s agent again on its session: it didn't stop when
        asked and was ended, and the session is all there in its folder."""
        try:
            self._manager.resume(
                name,
                held.session_id,
                held.args + ["--carry-on"],
                env=self._env(name),
                about=held.about,
            )
        except (AgentRunning, OSError) as error:
            self._finish(name, f"it could not carry on: {error}")
            return False
        log.warning(
            "The hub's agent for %s didn't stop when asked; it was started "
            "again on its session.",
            name,
        )
        return True

    def _refused(self, name: str, held: _Held, refusal: GatewayRefused) -> None:
        reason = refusal.error_type
        if refusal.status == 503 or reason in _LATER:
            # Not answering: the agent stays, and the gateway's own rule
            # (its grace time, then Watch) covers the instrument meanwhile.
            with self._lock:
                if held.state == "running":
                    held.away_since = self._clock()
                    log.warning(
                        "%s's gateway doesn't answer the hub's agent's renewal "
                        "(%s); trying again every %g s.",
                        name,
                        refusal,
                        RETRY_S,
                    )
                held.state = "away"
                held.why = "rejoin" if held.why == "rejoin" else str(refusal)
            return
        if reason in _AWAY and held.why != "rejoin":
            # The slot was freed while this hub was away: ask for it again as
            # the same session, on the next round. A Hub that was restarted
            # has no token to show and asks as the same paired Hub.
            if held.state == "running":
                held.away_since = self._clock()
            log.warning(
                "%s's gateway freed the slot while the hub was away (%s); "
                "asking for it again as the same session.",
                name,
                refusal,
            )
            with self._lock:
                held.state, held.why, held.tried = "away", "rejoin", float("-inf")
            return
        log.info(
            "%s's gateway took the slot from the hub's agent (%s); ending the agent.",
            name,
            refusal,
        )
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
