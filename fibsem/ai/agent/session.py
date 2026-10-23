"""The operator's side of the agent, without Qt.

What the agent chip, its card and its menu do, as calls: set the level, the
brake, allow or deny an agent asking to connect, approve or reject a request,
say how each request ended once, and send the agent a message. Each records what it does on the session's event stream,
as the operator. A window draws it; nothing here needs a widget.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from fibsem.acting import AGENT, OPERATOR
from fibsem.ai.agent.levels import AgentLevel, if_allowed
from fibsem.workflows.actions import ACTION, NOTE, action_record, note_record
from fibsem.workflows.evidence import EVIDENCE
from fibsem.workflows.proposals import DecisionResult

if TYPE_CHECKING:
    from fibsem.workflows.session import AppSession

# A request the operator still has to decide, or one being applied.
OPEN = ("pending", "applying")
# The longest message to the agent, in characters.
MESSAGE_LIMIT = 2000
# How often a waiting request's values are read again.
REQUEST_REFRESH_S = 15.0
# The share of its budget at which the card and the record warn.
BUDGET_WARNING = 0.8


def _no_suggestion(verb: str) -> DecisionResult:
    return DecisionResult(
        applied=False,
        error_type="stale_proposal",
        reason=f"There is no suggestion to {verb}.",
    )


def _dollars(value: float) -> str:
    return f"${value:.0f}" if float(value).is_integer() else f"${value:.2f}"


def budget_used(session: Optional[Dict[str, Any]]) -> Optional[float]:
    """The share of its budget the agent reported spending, or None."""
    if not session:
        return None
    spent, budget = session.get("spent_usd"), session.get("budget_usd")
    if spent is None or not budget:
        return None
    return spent / budget


def budget_low(session: Optional[Dict[str, Any]]) -> bool:
    """Whether the agent has spent `BUDGET_WARNING` of its budget or more."""
    used = budget_used(session)
    return used is not None and used >= BUDGET_WARNING


def session_text(session: Optional[Dict[str, Any]]) -> Optional[str]:
    """ "claude-opus · Mill 01 and 02 · $2.10 of $20" from the agent's reported
    session (``fibsem.server.agent_session``); None when it reported none.
    With the budget low, the spend leads, so a narrow card keeps it."""
    if not session:
        return None
    parts = [session[key] for key in ("model", "goal") if session.get(key)]
    spent, budget = session.get("spent_usd"), session.get("budget_usd")
    if budget_low(session):
        used = round(100 * budget_used(session))
        parts.insert(0, f"{_dollars(spent)} of {_dollars(budget)}, {used}% used")
    elif spent is not None and budget is not None:
        parts.append(f"{_dollars(spent)} of {_dollars(budget)}")
    elif spent is not None:
        parts.append(f"{_dollars(spent)} spent")
    if session.get("ended"):
        reason = session.get("reason")
        parts.append(f"ended: {reason}" if reason else "ended")
    return " · ".join(parts) or None


def session_line(session: Optional[Dict[str, Any]]) -> Optional[str]:
    """ "claude-opus · code mode · started 23:12" for the Agent page, whose
    side panel shows the goal and the budget; None when it reported none."""
    if not session:
        return None
    parts = [session["model"]] if session.get("model") else []
    if session.get("code") is True:
        parts.append("code mode")
    if session.get("started"):
        started = datetime.fromisoformat(session["started"]).astimezone()
        parts.append(f"started {started:%H:%M}")
    if session.get("ended"):
        reason = session.get("reason")
        parts.append(f"ended: {reason}" if reason else "ended")
    return " · ".join(parts) or None


def session_facts(session: Optional[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """What the agent said it can do, as (name, value) rows for the chip's
    details: its harness, profile, model and effort, code mode, skills,
    session id."""
    if not session:
        return []
    facts = []
    if session.get("harness"):
        facts.append(("Harness", session["harness"]))
    if session.get("profile"):
        hash_ = session.get("profile_hash")
        profile = f"{session['profile']} ({hash_})" if hash_ else session["profile"]
        facts.append(("Profile", profile))
    model = ", ".join(
        part
        for part in (
            session.get("model"),
            f"effort {session['effort']}" if session.get("effort") else None,
        )
        if part
    )
    if model:
        facts.append(("Model", model))
    if session.get("code") is True:
        where = f" in {session['workdir']}" if session.get("workdir") else ""
        facts.append(("Code", f"On{where} (sandboxed)"))
    elif session.get("files") is True:
        where = f" in {session['workdir']}" if session.get("workdir") else ""
        facts.append(("Code", f"Off; reads and writes files{where}"))
    elif session.get("code") is False:
        facts.append(("Code", "Off"))
    if session.get("skills"):
        facts.append(("Skills", ", ".join(session["skills"])))
    if session.get("session_id"):
        new = {True: " (resumed)", False: " (new)"}.get(session.get("resumed"), "")
        facts.append(("Session id", f"{session['session_id']}{new}"))
    return facts


class AgentSession:
    """The agent as the operator meets it, over one :class:`AppSession`: the
    App Server it reaches the session through, its level and its requests."""

    def __init__(self, session: "AppSession") -> None:
        self._session = session
        self._store: Optional[Any] = None
        self._unsubscribe: Optional[Callable[[], None]] = None
        # The pictures of the waiting request and the suggestion, for the
        # card and the web app: kind ("request", "suggestion") to its PNG,
        # caption and version. Drawn on a thread of their own (`keep_pictures`).
        self._pictures: Dict[str, Dict[str, Any]] = {}
        self._picture_version = 0
        self._picture_listeners: List[Callable[[], None]] = []
        self._drawer: Optional[Any] = None
        self._drawer_context: Optional[Any] = None
        self._drawing: Optional[threading.Thread] = None
        self._stop_drawing = threading.Event()
        self._draw_now = threading.Event()
        # The Now line, following the session's microscope.
        self._now: Optional[Any] = None
        self._now_microscope: Optional[Any] = None
        self._stop_now: Optional[Callable[[], None]] = None

    @property
    def host(self) -> Optional[Any]:
        """The App Server's host while it runs and serves the agent, else None."""
        host = self._session.app_server
        if host is None or not getattr(host, "running", False):
            return None
        if getattr(host, "auth", None) is None:
            return None
        return host

    @property
    def level(self) -> Optional[AgentLevel]:
        """The agent's level while the App Server serves it, else None."""
        host = self.host
        if host is None:
            return None
        return host.auth.level or AgentLevel.WATCH

    # --- level ---------------------------------------------------------------

    def set_level(self, level: Any) -> Optional[str]:
        """Set the agent's level, effective for its next request; returns why
        not, or None when it is set.

        Assist and Drive act, so they need an experiment to record what is
        done: the open one, or else the session's plain one, made now. Watch
        and Off record nothing of their own.
        """
        level = AgentLevel(level)
        auth = self._session.agent_auth
        if auth is None:
            return "The App Server isn't running."
        if level in (AgentLevel.ASSIST, AgentLevel.DRIVE):
            try:
                self._session.recording_experiment()
            except OSError as exc:
                return (
                    f"There's nowhere to record what the agent does ({exc}). "
                    "Open or create an experiment first."
                )
        previous = auth.level
        auth.set_level(level)
        logging.info("Agent level set to %s by the operator", level.value)
        self._record("agent_level_set", previous=previous, level=level)
        return None

    def stop(self) -> Optional[str]:
        """The brake: down to Watch at once, without asking, and stop what the
        agent is doing on the microscope. A running workflow is left
        running: runs are the app's, with their own Stop. Returns the stopped
        action, or None when the agent was doing nothing or is not served."""
        host = self.host
        if host is None:
            return None
        previous = host.auth.level
        stopped = host.stop_agent(self._session.microscope)
        logging.warning(
            "Agent stopped by the operator: level set to watch%s",
            f"; stopped {stopped}" if stopped else "",
        )
        recorder = self._session.recorder
        if recorder is not None:
            recorder.buffer.append(
                ACTION,
                action_record(
                    "Stop the agent",
                    stops="agent",
                    previous=previous.value if previous is not None else None,
                    level=host.auth.level.value,
                    stopped=stopped,
                ),
                actor=OPERATOR,
            )
        return stopped

    # --- requests --------------------------------------------------------------

    @property
    def requests(self) -> Optional[Any]:
        """The agent's request store while the App Server runs, else None."""
        host = self.host
        return getattr(host, "requests", None) if host is not None else None

    def follow(self, on_change: Callable[[], None]) -> Optional[Any]:
        """Call ``on_change`` whenever a request is filed, decided or applied,
        on whichever thread changed it. Follows the store the App Server has
        now: asked again when the server starts or stops, it moves to the new
        store, and with none it follows nothing. Returns the store."""
        store = self.requests
        if store is self._store:
            return store
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        self._store = store
        if store is None:
            return None
        self._unsubscribe = store.subscribe(lambda _request: on_change())
        return store

    def current(self) -> Optional[Any]:
        """The request waiting for the operator or being applied, or None."""
        store = self.requests
        for request in store.recent() if store is not None else []:
            if request.status in OPEN:
                return request
        return None

    def approve(self, request_id: str) -> None:
        """Run the request; its actions run on their own worker thread, so this
        returns at once. Raises ``RequestRefused`` when it can't."""
        self._decide(request_id, approve=True)

    def reject(self, request_id: str, reason: str = "") -> None:
        """Run none of it; the agent gets the reason. Raises ``RequestRefused``
        when it can't."""
        self._decide(request_id, approve=False, reason=reason)

    def _decide(self, request_id: str, approve: bool, reason: str = "") -> None:
        store = self.requests
        if store is None:
            return
        if approve:
            store.approve(request_id)
        else:
            store.reject(request_id, reason=reason)
        logging.info(
            "Agent's request %s %s by the operator",
            request_id,
            "approved" if approve else "rejected",
        )

    # --- its suggestion on a pending proposal --------------------------------

    def suggestion(self) -> Optional[Dict[str, Any]]:
        """The suggestion the card shows (``suggestions.suggestion_card``),
        with the ``proposal_id`` it is on; None when there is none."""
        from fibsem.ai.agent.suggestions import front_suggestion, suggestion_card

        experiment = self._session.experiment
        front = front_suggestion(experiment)
        if front is None:
            return None
        item, task_name, proposal = front
        card = suggestion_card(item, task_name, proposal, str(experiment.path))
        return {"proposal_id": proposal.id, **card}

    def waiting(self) -> Optional[Dict[str, Any]]:
        """The decision a run waits on, and who can make it, for the chat
        page: ``proposal`` ("01 · Setup · Alignment area") and ``who``; None
        while nothing waits."""
        from fibsem.ai.agent.suggestions import front_suggestion
        from fibsem.workflows.proposals import proposal_label

        experiment = self._session.experiment
        pending = experiment.pending_proposals() if experiment is not None else []
        if not pending:
            return None
        item, task_name, proposal = pending[0]
        level = self.level
        if OPERATOR in self._session.deciders():
            who = "Decide it in the window."
        elif level is AgentLevel.DRIVE:
            who = "The agent decides it."
        elif level is AgentLevel.ASSIST:
            who = (
                "The agent suggested a decision."
                if front_suggestion(experiment) is not None
                else "The agent will suggest a decision."
            )
        else:
            who = (
                "No one here can decide it: raise the agent to Assist or "
                "Drive, or open the window."
            )
        return {
            "proposal": f"{item.name} · {task_name} · {proposal_label(proposal)}",
            "who": who,
        }

    def approve_suggestion(self, proposal_id: str) -> DecisionResult:
        """Decide the proposal as the agent suggested, as the operator's
        decision, naming the suggestion."""
        front = self._front(proposal_id)
        if front is None:
            return _no_suggestion("approve")
        experiment, item, task_name = front
        return experiment.approve_suggestion(
            item.id, task_name, proposal_id, actor=experiment.actor(), via="workflow"
        )

    def decline_suggestion(self, proposal_id: str, reason: str = "") -> DecisionResult:
        """Drop the suggestion, with the reason for the agent; the proposal
        stays pending."""
        front = self._front(proposal_id)
        if front is None:
            return _no_suggestion("decline")
        experiment, item, task_name = front
        return experiment.decline_suggestion(
            item.id, task_name, proposal_id, reason=reason.strip()
        )

    def _front(self, proposal_id: str):
        """``(experiment, item, task_name)`` while the suggestion shown is on
        ``proposal_id``, else None."""
        from fibsem.ai.agent.suggestions import front_suggestion

        experiment = self._session.experiment
        front = front_suggestion(experiment)
        if front is None or front[2].id != proposal_id:
            return None
        return experiment, front[0], front[1]

    # --- an agent asking to connect ------------------------------------------

    @property
    def connections(self) -> Optional[Any]:
        """The asks to connect while the App Server runs
        (``fibsem.server.connections``), else None."""
        host = self.host
        return getattr(host.auth, "connections", None) if host is not None else None

    def asking(self) -> Optional[Dict[str, Any]]:
        """The agent asking to connect, as the card shows it, or None."""
        connections = self.connections
        asks = connections.asking() if connections is not None else []
        if not asks:
            return None
        replaces = self._paired_agent()
        return {
            **asks[0].to_dict(),
            "replaces": replaces,
            "if_allowed": if_allowed(self.level, replaces),
        }

    def _paired_agent(self) -> Optional[str]:
        """The allowed agent's name, which allowing another replaces."""
        connections = self.connections
        agent = connections.tokens.agent() if connections is not None else None
        return agent.name if agent is not None else None

    def allow_connection(self, ask_id: str) -> str:
        """Allow the asking agent: it is paired, with a token of its own,
        listed with Unpair. Returns the name it is listed under. Raises
        ``LookupError`` when nothing asks with that id."""
        connections = self.connections
        if connections is None:
            raise LookupError("The App Server isn't running.")
        replaces = self._paired_agent()
        computer = connections.allow(ask_id)
        logging.info("Operator allowed %s to connect", computer.name)
        self._record_connection(
            f"Allow {computer.name} to connect",
            reason=f"Replaces {replaces}" if replaces else "",
        )
        return computer.name

    def deny_connection(self, ask_id: str, reason: str = "") -> None:
        """Refuse the asking agent, with the operator's reason. Raises
        ``LookupError`` when nothing asks with that id."""
        connections = self.connections
        if connections is None:
            raise LookupError("The App Server isn't running.")
        ask = connections.deny(ask_id, reason)
        logging.info("Operator denied %s", ask.name)
        self._record_connection(f"Deny {ask.name}", reason=reason.strip())

    def _record_connection(self, what: str, reason: str = "") -> None:
        recorder = self._session.recorder
        if recorder is not None:
            recorder.buffer.append(
                ACTION, action_record(what, reason=reason), actor=OPERATOR
            )

    # --- what the agent hears ------------------------------------------------

    def message(
        self, text: str, evidence: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Put the operator's message to the agent on the session's record,
        where the agent's watch hears it as a ``MESSAGE`` line and answers with
        a message naming its id. *evidence*, the record of the
        image the operator marked for it (`fibsem.workflows.evidence.save`),
        goes on the record first and the message cites it. Returns the id, or
        None when nothing is recording, so the agent can't hear it."""
        recorder = self._session.recorder
        if recorder is None:
            return None
        message_id = uuid.uuid4().hex[:8]
        logging.info("Operator message to the agent (%s): %s", message_id, text)
        evidence_id = None
        if evidence is not None:
            evidence_id = evidence["evidence_id"]
            recorder.buffer.append(EVIDENCE, dict(evidence), actor=OPERATOR)
        recorder.buffer.append(
            NOTE,
            note_record(text, to=AGENT, id=message_id, evidence=evidence_id),
            actor=OPERATOR,
        )
        return message_id

    # --- the pictures of what it asks ------------------------------------------

    def set_picture(self, kind: str, png: Optional[bytes], caption: str = "") -> None:
        """The *kind* picture as drawn: a PNG with its legend, or None with
        *caption* saying why there is none; neither takes it down."""
        if png is None and not caption:
            if self._pictures.pop(kind, None) is None:
                return
        else:
            self._picture_version += 1
            self._pictures[kind] = {
                "png": png,
                "caption": caption,
                "version": self._picture_version,
            }
        for listener in list(self._picture_listeners):
            listener()

    def picture(self, kind: str) -> Optional[Dict[str, Any]]:
        """The *kind* picture (`set_picture`), or None."""
        return self._pictures.get(kind)

    def on_picture(self, listener: Callable[[], None]) -> Callable[[], None]:
        """Call *listener* when a picture is drawn or taken down, on the
        drawing thread. Returns how to stop."""
        self._picture_listeners.append(listener)
        return lambda: self._picture_listeners.remove(listener)

    def draw_pictures(self) -> None:
        """Draw the waiting request and the suggestion, each once, on the
        image its values were measured on (``pictures.AgentPictures``); what
        is no longer waiting comes down."""
        drawer = self._picture_drawer()
        current = self.current()
        drawer.show_request(current.to_dict() if current is not None else None)
        drawer.show_suggestion(self._suggestion_request())

    def keep_pictures(self) -> None:
        """Draw the pictures as requests and suggestions come and go, and read
        a waiting request's values again every `REQUEST_REFRESH_S`, on a thread
        of its own until `stop_pictures`: drawing reads images and the
        experiment, the refresh the routes a request describes."""
        if self._drawing is not None and self._drawing.is_alive():
            return
        self._stop_drawing.clear()
        self._drawing = threading.Thread(
            target=self._keep_pictures, name="agent-pictures", daemon=True
        )
        self._drawing.start()

    def stop_pictures(self) -> None:
        self._stop_drawing.set()
        self._draw_now.set()
        if self._drawing is not None:
            self._drawing.join(5.0)
        self._drawing = None
        for kind in list(self._pictures):
            self.set_picture(kind, None)

    def _keep_pictures(self) -> None:
        store, unsubscribe = None, None
        described: tuple = (None, 0.0)
        while not self._stop_drawing.is_set():
            self._draw_now.clear()
            if self.requests is not store:
                if unsubscribe is not None:
                    unsubscribe()
                store = self.requests
                unsubscribe = (
                    store.subscribe(lambda _request: self._draw_now.set())
                    if store is not None
                    else None
                )
            try:
                self.draw_pictures()
                described = self._describe_again(store, described)
            except Exception:
                logging.exception("Drawing the agent's pictures failed")
            self._draw_now.wait(1.0)
        if unsubscribe is not None:
            unsubscribe()

    def _describe_again(self, store, described: tuple) -> tuple:
        """Read the waiting request's values again once it has waited
        `REQUEST_REFRESH_S` since it was filed or last read."""
        current = self.current()
        if store is None or current is None or current.status != "pending":
            return (None, 0.0)
        now = time.monotonic()
        if described[0] != current.id:
            return (current.id, now)
        if now - described[1] < REQUEST_REFRESH_S:
            return described
        store.refresh(current.id)
        return (current.id, now)

    def _picture_drawer(self):
        """The drawer, made again for each App Server's app: what it reads
        from the experiment is the app's."""
        from fibsem.ai.agent.pictures import AgentPictures

        host = self._session.app_server
        context = getattr(host, "app_context", None) if host is not None else None
        if self._drawer is None or context is not self._drawer_context:
            sources = context.picture_sources() if context is not None else None
            reads = {}
            if sources is not None:
                reads = dict(
                    reference_image=sources.proposal_image,
                    task_config=sources.task_config,
                    item_pose=sources.item_pose,
                    item_image=sources.item_image,
                    overview_image=sources.overview_image,
                    item_overview=sources.item_overview,
                    pending_proposal=sources.pending_proposal,
                )
            self._drawer = AgentPictures(
                self._session.images.get, show_picture=self._show_picture, **reads
            )
            self._drawer_context = context
        return self._drawer

    def _show_picture(
        self, png: Optional[bytes], caption: str = "", suggestion: bool = False
    ) -> None:
        self.set_picture("suggestion" if suggestion else "request", png, caption)

    def _suggestion_request(self) -> Optional[Dict[str, Any]]:
        """The suggestion shown, as the request that would decide it."""
        from fibsem.ai.agent.suggestions import front_suggestion, suggestion_request

        experiment = self._session.experiment
        front = front_suggestion(experiment)
        if front is None:
            return None
        item, task_name, proposal = front
        return suggestion_request(item, task_name, proposal, str(experiment.path))

    def now(self) -> Optional[str]:
        """What the agent is doing on the microscope, as the card's Now line
        says it; None when it does nothing."""
        from fibsem.ai.agent.now import AgentNow, follow

        microscope = self._session.microscope
        if microscope is not self._now_microscope:
            if self._stop_now is not None:
                self._stop_now()
            self._now = AgentNow()
            self._stop_now = follow(self._now, microscope) if microscope else None
            self._now_microscope = microscope
        return self._now.text if self._now is not None else None

    def recent(self) -> List[Any]:
        """The agent card's rows (`recent.RecentRow`), read from the session's
        record now: for a reader outside the window."""
        from fibsem.ai.agent.recent import RecentActions

        recorder = self._session.recorder
        rows = RecentActions()
        if recorder is not None:
            for record in recorder.buffer.events_since(0).get("events", []):
                rows.add(record)
        return rows.rows

    def experiment_path(self) -> Optional[str]:
        """Where what the agent does is recorded: the open experiment, or the
        session's plain one; None for neither."""
        experiment = self._session.experiment or self._session.plain_experiment
        path = getattr(experiment, "path", None)
        return str(path) if path is not None else None

    def experiment_name(self) -> Optional[str]:
        """The open experiment's name; None before one is open (not the
        session's plain record)."""
        experiment = self._session.experiment
        return experiment.name if experiment is not None else None

    def seconds_since_seen(self) -> Optional[float]:
        """Seconds since the agent's token last made a request, or None."""
        host = self._session.app_server
        if host is None:
            return None
        try:
            return host.agent_seconds_since_seen()
        except Exception:
            return None

    def reported_session(self) -> Optional[dict]:
        """The agent's session as it reported it: model, goal, budget and
        spend (``fibsem.server.agent_session``), or None."""
        host = self._session.app_server
        if host is None:
            return None
        try:
            return host.agent_session()
        except Exception:
            return None

    def session_text(self) -> Optional[str]:
        """The reported session in one line, as the card and the chat show
        it, or None."""
        return session_text(self.reported_session())

    def session_line(self) -> Optional[str]:
        """The reported session as the Agent page's line shows it, or None."""
        return session_line(self.reported_session())

    def budget_low(self) -> bool:
        """Whether the agent reported spending most of its budget."""
        return budget_low(self.reported_session())

    def session_facts(self) -> List[Tuple[str, str]]:
        """What the agent said it can do, as rows (`session_facts`)."""
        return session_facts(self.reported_session())

    def client(self) -> Optional[str]:
        """The MCP client the agent's sidecar named, or None."""
        host = self._session.app_server
        if host is None:
            return None
        try:
            return host.agent_client()
        except Exception:
            return None

    def _record(self, kind: str, previous, level, **extra) -> None:
        """A level change, on the session's record (events.jsonl), as the
        operator's."""
        recorder = self._session.recorder
        if recorder is None:
            return
        recorder.buffer.append(
            kind,
            {
                "previous": previous.value if previous is not None else None,
                "level": level.value,
                **extra,
            },
            actor=OPERATOR,
        )
