"""The agent host: the agent's own routes, served with no microscope and no app.

``fibsem-agent`` hears from, and keeps its record through, a few routes of the
App Server: its events (``/app/events``), its session, transcript and plan
(``/app/agent/...``), the messages it reads and sends, and the operator's
level and Stop. An agent whose instrument is served by something else, or that
has no instrument at all, still needs those. This module serves them on their
own, at the same paths with the same answers, so the harness, the watch and
the web app's Agent page work against it unchanged.

What it holds is the session's, not an instrument's: an event stream of its
own (written to ``events.jsonl`` in its folder), the session the harness
reports, the plan and transcript (in each session's own folder, as the App
Server keeps them while no experiment is open), and the level and Stop. It
acts on nothing: the level gates only this host's routes, and an instrument
the agent uses keeps its own gate.

Build it with :func:`build_agent_host`. The answer is a FastAPI app: served
by itself, or mounted under a path by a server that holds several (the
gateway, the hub). Its tokens are an :class:`~fibsem.server.auth.AuthConfig`,
made by whoever builds it and never passed on a command line.
"""

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, FastAPI

from fibsem import __version__, paths
from fibsem.acting import AGENT, OPERATOR
from fibsem.ai.agent.levels import (
    LEVEL_NAMES,
    LEVEL_TIPS,
    LEVEL_WORDS,
    AgentLevel,
    chip_text,
)
from fibsem.ai.agent.session import (
    MESSAGE_LIMIT,
    budget_low,
    session_facts,
    session_line,
)
from fibsem.server.agent.files import AgentFiles
from fibsem.server.agent.plan import PlanStore, build_agent_plan_router
from fibsem.server.agent.session import build_agent_session_router
from fibsem.server.agent.transcript import build_agent_transcript_router
from fibsem.server.auth import (
    AuthConfig,
    refusal_error,
    require_level,
    require_monitoring,
    require_operator,
)
from fibsem.server.bodies import (
    LevelBody,
    MessageBody,
    NoteBody,
    OperatorMessageBody,
    refuse_invalid_requests,
)
from fibsem.server.discovery import API_VERSION, fibsem_folder
from fibsem.server.responses import (
    AgentStopped,
    AgentView,
    Events,
    LevelOption,
    LevelSet,
    MessageSent,
    Recorded,
    answer,
)
from fibsem.workflows.actions import ACTION, NOTE, action_record, note_record
from fibsem.workflows.events import EventBuffer, split_event_id
from fibsem.workflows.events import kinds as event_kinds
from fibsem.workflows.events.recording import EventFileWriter, read_events_after
from fibsem.workflows.events.text import note_asked, with_text

logger = logging.getLogger(__name__)

EVENTS_FILENAME = "events.jsonl"
# The long-poll's ceiling, as the App Server's: a parked poll holds a worker.
MAX_POLL_S = 30.0


def default_folder() -> Path:
    """Where a host keeps its record when it is given no folder."""
    return paths.state_root() / "agent" / "host"


class AgentHost:
    """What the host holds: its tokens and level, its event stream and the
    file it is written to, and where each session's plan and transcript go."""

    def __init__(
        self,
        auth: AuthConfig,
        folder: Optional[Path] = None,
        name: str = "fibsemOS",
    ) -> None:
        self.auth = auth
        self.name = name
        self.folder = Path(folder) if folder is not None else default_folder()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.events_file = self.folder / EVENTS_FILENAME
        self.buffer = EventBuffer(stream=uuid.uuid4().hex)
        self._writer = EventFileWriter()
        self._writer.set_path(self.events_file)
        # The writer only queues, so handing it each record under the
        # buffer's lock is allowed.
        self._unsubscribe = self.buffer.subscribe(self._writer.write)
        # No experiment: each session's files stay in its own folder.
        self.files = AgentFiles(lambda: None)
        self.plan = PlanStore(lambda: None, files=self.files)

    def record(
        self, kind: str, payload: Dict[str, Any], actor: Optional[str] = None
    ) -> None:
        """Put an event on the host's record, as the App Server's
        ``record_event`` does."""
        self.buffer.append(kind, payload, actor=actor)

    def events(
        self, since: int = 0, timeout: float = 0.0, after: Optional[str] = None
    ) -> Dict[str, Any]:
        """Events after *since*, parking up to *timeout* seconds for news.
        *after*, an event id, reaches back past a restart through the file."""
        if after:
            named = split_event_id(after)
            if named is not None and named[0] == self.buffer.stream:
                return {
                    **self.buffer.wait_for(since=named[1], timeout=timeout),
                    "resumed": True,
                }
            held = self.buffer.events_since(0)
            earlier: List[Dict[str, Any]] = []
            resumed = False
            if self.events_file.is_file():
                stream = self.buffer.stream

                def keep(record: Dict[str, Any]) -> bool:
                    # What the buffer holds, it answers itself.
                    return not str(record.get("id", "")).startswith(f"{stream}-")

                earlier, resumed = read_events_after(
                    self.events_file,
                    after,
                    skip_kinds=event_kinds.LIVE_KINDS,
                    keep=keep,
                )
            return {**held, "events": earlier + held["events"], "resumed": resumed}
        return self.buffer.wait_for(since=since, timeout=timeout)

    # --- the operator's side ---------------------------------------------------

    def message(self, text: str) -> str:
        """The operator's message to the agent, on the record as the App
        Server puts it (`AgentControl.message`); returns its id."""
        message_id = uuid.uuid4().hex[:8]
        logger.info("Operator message to the agent (%s): %s", message_id, text)
        self.record(NOTE, note_record(text, to=AGENT, id=message_id), actor=OPERATOR)
        return message_id

    def set_level(self, level: AgentLevel) -> None:
        previous = self.auth.level
        self.auth.set_level(level)
        logger.info("Agent level set to %s by the operator", level.value)
        self.record(
            event_kinds.AGENT_LEVEL_SET,
            {
                "previous": previous.value if previous is not None else None,
                "level": self.auth.level.value,
            },
            actor=OPERATOR,
        )

    def stop(self) -> None:
        """Stop agent: down to Watch at once. The host runs nothing of the
        agent's, so there is no action to stop; an instrument it uses has its
        own Stop."""
        previous = self.auth.level
        self.auth.stop_agent()
        logger.warning("Agent stopped by the operator: level set to watch")
        self.record(
            ACTION,
            action_record(
                "Stop the agent",
                stops="agent",
                previous=previous.value if previous is not None else None,
                level=self.auth.level.value,
            ),
            actor=OPERATOR,
        )

    def close(self) -> None:
        self._unsubscribe()
        self._writer.close()


def _with_text(answer_: Dict[str, Any]) -> Dict[str, Any]:
    asked: Dict[str, Any] = {}
    described = []
    for record in answer_.get("events") or []:
        note_asked(asked, record)
        described.append(with_text(record, asked))
    return {**answer_, "events": described}


def _text(value: str, key: str) -> str:
    if not value.strip():
        raise refusal_error(422, "missing_field", f"Pass {key!r} as text.")
    return value.strip()


def _no_items(item_name: Optional[str]) -> None:
    if item_name is not None:
        raise refusal_error(
            422, "invalid", "This host keeps no experiment, so no item_name."
        )


def build_agent_host(
    auth: AuthConfig,
    folder: Optional[Path] = None,
    name: str = "fibsemOS",
) -> FastAPI:
    """The agent host as a FastAPI app; its :class:`AgentHost` is
    ``app.state.host``, which its owner closes when done with it."""
    host = AgentHost(auth, folder, name)
    app = FastAPI(title="fibsemOS agent host", version=__version__)
    app.state.host = host
    app.state.auth = auth
    app.state.app_name = name
    # The routes shared with the App Server read these.
    app.state.agent_files = host.files
    app.state.agent_plan = host.plan
    refuse_invalid_requests(app)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    base = APIRouter(dependencies=[Depends(require_monitoring())])

    @base.get("/capabilities")
    def capabilities():
        return {
            "api_version": API_VERSION,
            "routers": {"microscope": False, "services": False, "app": False},
            # What this server is: the agent's routes only.
            "host": "agent",
            "app_name": name,
            "parts": [],
            "answer_types": {},
            "level": auth.level.value,
            "agent": auth.agent_allowed,
            "connected": False,
            "instruments": [],
            "fibsem": {"version": __version__, "path": fibsem_folder()},
            "gateway": None,
        }

    @base.get("/app/events", response_model=Events)
    def events(
        since: int = 0,
        timeout: float = 0.0,
        after: Optional[str] = None,
        text: bool = False,
    ):
        timeout = min(max(timeout, 0.0), MAX_POLL_S)
        found = host.events(since=since, timeout=timeout, after=after)
        return _with_text(found) if text else found

    app.include_router(base)

    # What the agent writes: notes and messages to the operator.
    record = APIRouter(
        prefix="/app/experiment",
        dependencies=[Depends(require_level(AgentLevel.WATCH))],
    )

    @record.post("/notes", response_model=Recorded)
    def add_note(body: NoteBody):
        _no_items(body.item_name)
        text = _text(body.text, "text")
        host.record(NOTE, note_record(text), actor=AGENT)
        return {"recorded": True}

    @record.post("/messages", response_model=Recorded)
    def add_message(body: OperatorMessageBody):
        _no_items(body.item_name)
        text = _text(body.text, "text")
        message_id = uuid.uuid4().hex[:8]
        host.record(
            NOTE,
            note_record(text, to=OPERATOR, id=message_id, reply_to=body.reply_to),
            actor=AGENT,
        )
        return {"recorded": True, "id": message_id}

    app.include_router(record)

    # The agent's session, plan and transcript: the App Server's own routes.
    app.include_router(build_agent_session_router(host.record))
    app.include_router(build_agent_plan_router(lambda: None, host.record))
    app.include_router(build_agent_transcript_router(lambda: None, host.record))

    # The operator's side: what the Agent page does.
    operator = APIRouter(
        prefix="/app/agent", dependencies=[Depends(require_operator())]
    )

    @operator.get("", response_model=AgentView)
    def get_agent():
        session = auth.agent_session
        return answer(
            AgentView(
                experiment=None,
                level=auth.level.value,
                stopped=bool(auth.stopped),
                last_heard_s=auth.seconds_since_seen(),
                client=auth.agent_client,
                session=session_line(session),
                budget_low=budget_low(session),
                request=None,
                chip=chip_text(auth.level, None, False),
                suggestion=None,
                now=None,
                waiting=None,
                pictures={},
                levels=[
                    LevelOption(
                        level.value,
                        LEVEL_NAMES[level],
                        LEVEL_WORDS[level],
                        LEVEL_TIPS[level],
                    )
                    for level in AgentLevel
                ],
                facts=[list(fact) for fact in session_facts(session)],
            )
        )

    @operator.post("/messages", status_code=201, response_model=MessageSent)
    def send_message(body: MessageBody):
        if body.evidence is not None:
            raise refusal_error(
                422,
                "invalid",
                "This host keeps no images, so a message here is text only.",
            )
        text = _text(body.text, "text")
        if len(text) > MESSAGE_LIMIT:
            raise refusal_error(
                422, "too_long", f"A message is at most {MESSAGE_LIMIT} characters."
            )
        return answer(MessageSent(host.message(text)), status_code=201)

    @operator.put("/level", response_model=LevelSet)
    def set_level(body: LevelBody):
        try:
            level = AgentLevel(_text(body.level, "level").lower())
        except ValueError:
            names = ", ".join(level.value for level in AgentLevel)
            raise refusal_error(422, "bad_level", f"The level is one of {names}.")
        if not auth.agent_allowed and level is not AgentLevel.OFF:
            raise refusal_error(
                409, "level_refused", "The agent is not part of this release."
            )
        host.set_level(level)
        return answer(LevelSet(level.value))

    @operator.post("/stop", response_model=AgentStopped)
    def stop_agent():
        host.stop()
        return answer(AgentStopped(False, auth.level.value))

    app.include_router(operator)
    return app
