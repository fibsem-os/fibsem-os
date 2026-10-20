"""The operator's side of the agent, without Qt (FIB-1461).

What the agent chip, its card and its menu do, as calls: set the level, the
brake, approve or reject a request, say how each request ended once, and send
the agent a message. Each records what it does on the session's event stream,
as the operator. A window draws it; nothing here needs a widget.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

from fibsem.acting import OPERATOR
from fibsem.ai.agent.levels import AgentLevel
from fibsem.workflows.actions import ACTION, NOTE, action_record, note_record
from fibsem.workflows.proposals import DecisionResult

if TYPE_CHECKING:
    from fibsem.workflows.session import AppSession

# A request the operator still has to decide, or one being applied.
OPEN = ("pending", "applying")
# The longest message to the agent, in characters (FIB-1423).
MESSAGE_LIMIT = 2000


def _no_suggestion(verb: str) -> DecisionResult:
    return DecisionResult(
        applied=False,
        error_type="stale_proposal",
        reason=f"There is no suggestion to {verb}.",
    )


class AgentSession:
    """The agent as the operator meets it, over one :class:`AppSession`: the
    App Server it reaches the session through, its level and its requests."""

    def __init__(self, session: "AppSession") -> None:
        self._session = session
        self._store: Optional[Any] = None
        self._unsubscribe: Optional[Callable[[], None]] = None
        # Requests whose ending has been said, so each is said once.
        self._said: Set[str] = set()
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
        agent is doing on the microscope (FIB-1267). A running workflow is left
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
        # What ended before anyone was following is not news.
        self._said.update(r.id for r in store.recent() if r.status not in OPEN)
        self._unsubscribe = store.subscribe(lambda _request: on_change())
        return store

    def current(self) -> Optional[Any]:
        """The request waiting for the operator or being applied, or None."""
        store = self.requests
        for request in store.recent() if store is not None else []:
            if request.status in OPEN:
                return request
        return None

    def outcomes(self) -> List[Any]:
        """The requests that have ended and not been said yet, each returned
        once: how one ended is said once, so it can go in a toast."""
        store = self.requests
        ended = []
        for request in store.recent() if store is not None else []:
            if request.status in OPEN or request.id in self._said:
                continue
            self._said.add(request.id)
            ended.append(request)
        return ended

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

    # --- what the agent hears ------------------------------------------------

    def message(self, text: str) -> Optional[str]:
        """Put the operator's message to the agent on the session's record,
        where the agent's watch hears it as a ``MESSAGE`` line and answers with
        a note naming its id (FIB-1423). Returns the id, or None when nothing
        is recording, so the agent can't hear it."""
        recorder = self._session.recorder
        if recorder is None:
            return None
        message_id = uuid.uuid4().hex[:8]
        logging.info("Operator message to the agent (%s): %s", message_id, text)
        recorder.buffer.append(NOTE, note_record(text, id=message_id), actor=OPERATOR)
        return message_id

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

    def seconds_since_seen(self) -> Optional[float]:
        """Seconds since the agent's token last made a request, or None."""
        host = self._session.app_server
        if host is None:
            return None
        try:
            return host.agent_seconds_since_seen()
        except Exception:
            return None

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
