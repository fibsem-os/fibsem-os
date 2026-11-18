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

With an instrument (:class:`InstrumentLink`), the host follows the
instrument server's events and passes them on, so the agent wakes on what
happens there, and the operator's Stop here is the instrument's Stop too:
the instrument's own level goes down to Watch and what the agent is doing
there stops. A Stop pressed at the instrument reaches the agent the same way,
and lowers the level here. The level itself is set at each place apart: the
instrument's gates the instrument.

Build it with :func:`build_agent_host`. The answer is a FastAPI app: served
by itself, or mounted under a path by a server that holds several (the
gateway, the hub). Its tokens are an :class:`~fibsem.server.auth.AuthConfig`,
made by whoever builds it and never passed on a command line.
"""

import logging
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
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
from fibsem.client.transport import HOST_TOKEN_ENV
from fibsem.keystore import get_secret
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


# The instrument's routes the host calls: the App Server's paths.
INSTRUMENT_EVENTS = "/app/events"
INSTRUMENT_STOP = "/app/agent/stop"
# How long one poll of the instrument's events parks, and the wait after one
# fails before the next.
RELAY_POLL_S = 25.0
RELAY_RETRY_S = 5.0


@dataclass
class InstrumentLink:
    """The instrument server the agent works on, as the host reaches it: its
    address, the operator's token there (from the PC's key store, never a
    file or a command line) and the routes for its events and its Stop."""

    url: str
    token: str
    name: str = "instrument"
    events_path: str = INSTRUMENT_EVENTS
    stop_path: str = INSTRUMENT_STOP

    def client(self, timeout: float) -> httpx.Client:
        return httpx.Client(
            base_url=self.url.rstrip("/"),
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=timeout,
        )


def _is_agent_stop(record: Dict[str, Any]) -> bool:
    payload = record.get("payload") or {}
    return (
        record.get("kind") == ACTION
        and record.get("actor") == OPERATOR
        and payload.get("stops") == "agent"
    )


class _Relay:
    """Follows the instrument's events on a thread of its own and puts each on
    the host's record, from the first one after it starts."""

    def __init__(self, host: "AgentHost", link: InstrumentLink) -> None:
        self._host = host
        self._link = link
        self._stopping = threading.Event()
        # Set once it holds its place in the instrument's stream.
        self.following = threading.Event()
        self._client = link.client(RELAY_POLL_S + 10.0)
        self._thread = threading.Thread(
            target=self._run, name="fibsem-agent-host-relay", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stopping.set()
        try:
            self._client.close()
        except Exception:  # noqa: BLE001 - closing ends a parked poll
            pass
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(2.0)

    def _poll(self, since: int, timeout: float) -> Optional[Dict[str, Any]]:
        try:
            response = self._client.get(
                self._link.events_path, params={"since": since, "timeout": timeout}
            )
            if response.status_code >= 400:
                logger.warning(
                    "%s answered %s for its events",
                    self._link.name,
                    response.status_code,
                )
                return None
            return response.json()
        except (httpx.HTTPError, ValueError, RuntimeError) as error:
            if not self._stopping.is_set():
                logger.warning("%s's events: %s", self._link.name, error)
            return None

    def _run(self) -> None:
        stream: Optional[str] = None
        since = 0
        while not self._stopping.is_set():
            # A new stream (the instrument started again) is followed from
            # its tail, as the watch does.
            answer_ = self._poll(since, 0.0 if stream is None else RELAY_POLL_S)
            if answer_ is None or not answer_.get("available", True):
                self._stopping.wait(RELAY_RETRY_S)
                continue
            if answer_.get("stream") != stream:
                stream, since = answer_.get("stream"), answer_.get("latest_seq") or 0
                self.following.set()
                continue
            for record in answer_.get("events") or []:
                since = max(since, record.get("seq") or since)
                self._host.relayed(record, self._link.name)


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
        instrument: Optional["InstrumentLink"] = None,
    ) -> None:
        self.auth = auth
        self.name = name
        self.instrument = instrument
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
        self._relay = _Relay(self, instrument) if instrument is not None else None
        if self._relay is not None:
            self._relay.start()

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

    @property
    def following(self) -> bool:
        """Whether the host follows the instrument's events now."""
        return self._relay is not None and self._relay.following.is_set()

    def relayed(self, record: Dict[str, Any], source: str) -> None:
        """A record from the instrument's stream, onto the host's: its kind,
        payload and envelope (who acted, the item, the task), with the
        instrument it came from. Progress records are left there. The
        instrument's Stop agent lowers the level here too."""
        kind = record.get("kind")
        if not kind or kind in event_kinds.LIVE_KINDS:
            return
        if _is_agent_stop(record):
            self.auth.stop_agent()
        self.buffer.append(
            kind,
            record.get("payload") or {},
            actor=record.get("actor"),
            envelope={**record, "instrument": source},
        )

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

    def stop(self) -> Optional[str]:
        """Stop agent: down to Watch at once, here and at the instrument.

        The instrument's Stop puts its own record on its stream, which the
        host passes on: the agent hears that one, which says what was
        stopped there. With no instrument, or one that doesn't answer, the
        host records the Stop itself. Returns what the instrument stopped,
        if anything."""
        previous = self.auth.level
        self.auth.stop_agent()
        logger.warning("Agent stopped by the operator: level set to watch")
        reason = ""
        if self.instrument is not None:
            stopped, why_not = self._stop_instrument()
            if why_not is None:
                return stopped
            reason = why_not
        self.record(
            ACTION,
            action_record(
                "Stop the agent",
                reason=reason,
                stops="agent",
                previous=previous.value if previous is not None else None,
                level=self.auth.level.value,
            ),
            actor=OPERATOR,
        )
        return None

    def _stop_instrument(self) -> Tuple[Optional[str], Optional[str]]:
        """(what it stopped, None) once the instrument has its Stop, or
        (None, why not)."""
        link = self.instrument
        try:
            with link.client(10.0) as client:
                response = client.post(link.stop_path)
        except httpx.HTTPError as error:
            logger.error("Stop agent did not reach %s: %s", link.name, error)
            return None, f"{link.name} did not answer, so its Stop was not pressed"
        if response.status_code >= 400:
            logger.error("Stop agent: %s answered %s", link.name, response.status_code)
            return None, (
                f"{link.name} refused its Stop ({response.status_code}), "
                "so press it there"
            )
        try:
            stopped = response.json().get("stopped")
        except (ValueError, AttributeError):
            stopped = None
        return (stopped if isinstance(stopped, str) else None), None

    def close(self) -> None:
        if self._relay is not None:
            self._relay.close()
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
    instrument: Optional["InstrumentLink"] = None,
) -> FastAPI:
    """The agent host as a FastAPI app; its :class:`AgentHost` is
    ``app.state.host``, which its owner closes when done with it."""
    host = AgentHost(auth, folder, name, instrument)
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
            # The instrument server the agent works on, by name; None for none.
            "instrument": instrument.name if instrument is not None else None,
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
        stopped = host.stop()
        return answer(AgentStopped(stopped, auth.level.value))

    app.include_router(operator)
    return app


# --- fibsem-agent-host ---------------------------------------------------------

# The host's tokens, from the environment or the PC's key store (`keystore`):
# the agent's, which fibsem-agent reads by the same name, and the operator's.
HOST_TOKEN = HOST_TOKEN_ENV
HOST_OPERATOR_TOKEN = "FIBSEM_HOST_OPERATOR_TOKEN"
# The instrument's operator token, by default: the App Server's.
INSTRUMENT_TOKEN = "FIBSEM_OPERATOR_TOKEN"
DEFAULT_PORT = 8030


def main(argv: Optional[List[str]] = None) -> int:
    """``fibsem-agent-host``: serve the agent host on this PC."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(
        prog="fibsem-agent-host",
        description="Serve the agent's routes (events, session, plan, "
        "transcript, messages, level and Stop) with no microscope, for "
        "fibsem-agent --host. Its tokens come from the environment or the key "
        f"store: {HOST_TOKEN} (the agent's) and {HOST_OPERATOR_TOKEN}.",
    )
    parser.add_argument("--bind", default="127.0.0.1", help="address to serve on")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--folder", default=None, help="where its record is kept")
    parser.add_argument("--name", default="fibsemOS", help="the name pages show")
    parser.add_argument(
        "--instrument",
        default=None,
        metavar="URL",
        help="the instrument server the agent works on: its events are passed "
        "on, and Stop here presses its Stop",
    )
    parser.add_argument(
        "--instrument-name", default="instrument", help="what to call it"
    )
    parser.add_argument(
        "--instrument-token",
        default=INSTRUMENT_TOKEN,
        metavar="NAME",
        help="the key (environment or key store) holding the operator's token "
        f"at the instrument (default {INSTRUMENT_TOKEN})",
    )
    args = parser.parse_args(argv)

    token, operator = get_secret(HOST_TOKEN), get_secret(HOST_OPERATOR_TOKEN)
    missing = [
        name
        for name, value in ((HOST_TOKEN, token), (HOST_OPERATOR_TOKEN, operator))
        if not value
    ]
    if missing:
        print(
            f"fibsem-agent-host: set {' and '.join(missing)} in the environment "
            "or the key store first.",
            file=sys.stderr,
        )
        return 1
    instrument = None
    if args.instrument:
        instrument_token = get_secret(args.instrument_token)
        if not instrument_token:
            print(
                f"fibsem-agent-host: --instrument needs the operator's token "
                f"there in {args.instrument_token}.",
                file=sys.stderr,
            )
            return 1
        instrument = InstrumentLink(
            args.instrument, instrument_token, name=args.instrument_name
        )
    auth = AuthConfig(token=token, operator_token=operator)
    folder = Path(args.folder).expanduser() if args.folder else None
    app = build_agent_host(auth, folder, args.name, instrument)
    try:
        uvicorn.run(app, host=args.bind, port=args.port, log_level="warning")
    finally:
        app.state.host.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
