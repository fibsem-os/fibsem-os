"""``fibsem-agent``'s wake loop against a real App Server.

The App Server over a windowless Demo session, the real ``watch()`` and the
real events; only the model is stood in, by an agent that answers each turn
the way a model would, over HTTP. Nothing here imports Qt.
"""

import asyncio
import logging
import os
import re
import socket
import threading
import time

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fibsem.ai.agent import runner as runner_mod  # noqa: E402
from fibsem.ai.agent.levels import AgentLevel  # noqa: E402
from fibsem.ai.agent.mcp import watch as watch_mod  # noqa: E402
from fibsem.ai.agent.runner import Runner, TurnEnd, Wakes, kickoff  # noqa: E402
from fibsem.ai.agent.session import AgentSession  # noqa: E402
from fibsem.applications.autolamella.server.context import (  # noqa: E402
    AutoLamellaAppContext,
)
from fibsem.server import AuthConfig  # noqa: E402
from fibsem.server.hosting import AppServerHost  # noqa: E402
from fibsem.workflows.attention import Attention  # noqa: E402
from fibsem.workflows.session import AppSession  # noqa: E402
from tests.autolamella.test_select_position_asks import (  # noqa: E402, F401
    SETUP,
    _experiment,
    microscope,
)

os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")

POLL_S = 0.3
RETRY_S = 0.05


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def session(microscope, tmp_path, monkeypatch, caplog):  # noqa: F811
    """A windowless session with a supervised Select Position open, its App
    Server running at Watch."""
    monkeypatch.delenv("FIBSEM_SERVER_URL", raising=False)
    monkeypatch.delenv("FIBSEM_SERVER_TOKEN", raising=False)
    caplog.set_level(logging.INFO, logger=runner_mod.log.name)
    session = AppSession()
    session.microscope = microscope
    session.experiment = _experiment(
        tmp_path, microscope, attention=Attention.supervised
    )
    session.start_recording()
    session.agent_auth = AuthConfig.generate(level=AgentLevel.WATCH)
    session.app_server = AppServerHost(
        session,
        lambda buffer: AutoLamellaAppContext(session, event_buffer=buffer),
        port=_free_port(),
        discovery_path=tmp_path / "agent-server.json",
    )
    assert session.app_server.start()
    yield session
    session.runner.stop()
    session.runner.join(30)
    session.app_server.stop()
    session.stop_recording()


def _http(session):
    return httpx.Client(
        base_url=session.app_server.url,
        headers={"Authorization": f"Bearer {session.agent_auth.token}"},
        timeout=30,
    )


def _connect(session):
    return lambda: watch_mod.connect(session.app_server.url, session.agent_auth.token)


class ScriptedAgent:
    """The model stood in: it keeps each prompt and answers it with
    ``reply(prompt)``, a coroutine that may call the App Server as the agent
    would. An interrupt ends the turn it is in."""

    def __init__(self, reply=None):
        self.prompts = []
        self.started = self.closed = False
        self.interrupts = 0
        self.session_id = "scripted"
        self._reply = reply
        self._interrupted = None

    async def start(self):
        self.started = True
        self._interrupted = asyncio.Event()

    async def turn(self, prompt):
        self.prompts.append(prompt)
        if self._reply is None:
            return TurnEnd()
        return await self._reply(prompt) or TurnEnd()

    async def interrupt(self):
        self.interrupts += 1
        self._interrupted.set()

    async def until_interrupted(self):
        await self._interrupted.wait()
        return TurnEnd("error_during_execution")

    async def close(self):
        self.closed = True


class Running:
    """The runner on its own thread, as ``fibsem-agent`` runs it."""

    def __init__(self, runner):
        self.runner = runner
        self.code = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        self.code = asyncio.run(self.runner.run())

    def ended(self, timeout=30.0):
        self._thread.join(timeout)
        assert not self._thread.is_alive(), "the runner did not end"
        return self.code

    def stop(self):
        self.runner.stop()
        return self.ended()


def _runner(session, agent, **kwargs):
    kwargs.setdefault("poll_s", POLL_S)
    kwargs.setdefault("retry_s", RETRY_S)
    return Running(Runner(agent, _connect(session), **kwargs))


def _until(predicate, timeout=20.0, what="it"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def _heard(caplog, start):
    return [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith(f"watch: {start}")
    ]


# --- the kickoff and the session's tools ------------------------------------


def test_the_kickoff_reads_the_skill_for_the_level():
    first = kickoff()
    assert "autolamella-agent" in first
    assert "skill for the level" in first
    assert "server_get_capabilities" in first
    assert "don't start one" in first
    # The runner never sets the level: it tells the agent whose it is.
    assert "level is the operator's" in first
    again = kickoff(resumed=True)
    assert "resumed" in again and "server_get_capabilities" in again
    # Which skill goes with which level is the system prompt's.
    for skill in ("observe", "assist", "supervise"):
        assert f"{skill}-autolamella" in runner_mod.RUNNER_PROMPT


def test_the_agent_has_the_fibsem_tools_and_its_skills_and_nothing_else(tmp_path):
    pytest.importorskip("claude_agent_sdk")
    checkout = runner_mod.find_checkout()
    assert checkout is not None, "tests run from a checkout with .claude/skills"
    options = runner_mod.build_options(
        checkout, url="http://127.0.0.1:9", token="t0k", resume="abc"
    )

    assert options.permission_mode == "dontAsk"
    assert options.allowed_tools == ["mcp__fibsem__*", "ToolSearch"] + [
        f"Agent({name})" for name in runner_mod.SPECIALISTS
    ]
    assert options.tools == ["Skill", "ToolSearch", "Agent"]
    for tool in ("Bash", "Read", "Grep", "Glob", "Write", "Edit", "WebFetch"):
        assert tool in options.disallowed_tools
    assert options.effort == "high"
    # The CLI's own agents are refused: only the specialists are called, and
    # each answers inside the turn that called it.
    for agent in ("general-purpose", "claude", "Explore"):
        assert f"Agent({agent})" in options.disallowed_tools
    assert set(options.skills) == {
        "autolamella-agent",
        "observe-autolamella",
        "assist-autolamella",
        "supervise-autolamella",
    }
    # The checkout's skills and nothing from ~/.claude, nor its MCP servers.
    assert options.setting_sources == ["project"]
    assert options.strict_mcp_config
    assert list(options.mcp_servers) == ["fibsem"]
    server = options.mcp_servers["fibsem"]
    assert os.path.isabs(server["command"])
    # Not in the server entry, which goes on the CLI's command line: in the
    # CLI's environment, which the sidecar inherits.
    assert "env" not in server
    assert options.cwd == str(checkout)
    assert options.resume == "abc"
    assert options.verbatim_prompts
    assert options.max_turns == runner_mod.MAX_TURNS
    assert options.max_budget_usd == runner_mod.MAX_BUDGET_USD
    # No developer CLAUDE.md or auto memory in an operator's agent.
    assert options.env == {
        "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
        "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
        "FIBSEM_SERVER_URL": "http://127.0.0.1:9",
        "FIBSEM_SERVER_TOKEN": "t0k",
    }

    # And what the CLI is told, as the SDK builds its command line.
    from claude_agent_sdk._internal.transport.subprocess_cli import (
        SubprocessCLITransport,
    )

    transport = SubprocessCLITransport("", options)
    transport._cli_path = "claude"
    command = transport._build_command()
    assert "--strict-mcp-config" in command
    assert "--setting-sources=project" in command
    assert command[command.index("--permission-mode") + 1] == "dontAsk"
    allowed = command[command.index("--allowedTools") + 1].split(",")
    assert "Skill(autolamella-agent)" in allowed and "mcp__fibsem__*" in allowed
    assert "bypassPermissions" not in command
    assert not any("t0k" in part for part in command), "the token is in argv"


# --- the specialists ------------------------------------------------------------

# Read-only, but it saves its result beside the image it measured.
MEASURES = {"item_measure_thickness"}
# Only the Run Lead decides, starts, stops, mills, edits and keeps the record.
LEAD_ONLY = {
    "proposal_decide",
    "proposal_suggest",
    "request_create",
    "experiment_add_note",
    "item_update_task_config",
    "item_remove",
    "milling_run_stage",
    "milling_run_task",
    "milling_stop",
    "milling_end",
}


def _specialists():
    """Each agent file's frontmatter, by its file name."""
    import yaml

    checkout = runner_mod.find_checkout()
    agents = {}
    for path in sorted((checkout / ".claude" / "agents").glob("autolamella-*.md")):
        _, front, _ = path.read_text(encoding="utf-8").split("---\n", 2)
        agents[path.stem] = yaml.safe_load(front)
    return agents


def _tools(front):
    return [tool.strip() for tool in front["tools"].split(",")]


def test_the_runner_allows_the_specialists_the_checkout_defines():
    agents = _specialists()
    assert set(agents) == set(runner_mod.SPECIALISTS)
    for name, front in agents.items():
        assert front["name"] == name
        assert front["description"]


def test_a_specialist_has_only_tools_that_exist_and_the_runner_can_refuse():
    from fibsem.ai.agent.catalog import CATALOG

    catalog = {spec.name for spec in CATALOG}
    for name, front in _specialists().items():
        for tool in _tools(front):
            if tool.startswith("mcp__fibsem__"):
                assert tool.removeprefix("mcp__fibsem__") in catalog, (name, tool)
            else:
                # An agent gets a tool it lists unless the runner denies it.
                assert tool in runner_mod.DENIED_TOOLS, (name, tool)


def test_only_the_run_lead_decides_and_its_agents_only_read():
    from fibsem.ai.agent.catalog import CATALOG

    read_only = {spec.name for spec in CATALOG if spec.read_only}
    for name, front in _specialists().items():
        fibsem = {
            tool.removeprefix("mcp__fibsem__")
            for tool in _tools(front)
            if tool.startswith("mcp__fibsem__")
        }
        assert not fibsem & LEAD_ONLY, (name, fibsem & LEAD_ONLY)
        assert not {t for t in fibsem if t.startswith(("workflow_", "protocol_"))} - (
            read_only
        ), name
        assert fibsem <= read_only | MEASURES, (name, fibsem - read_only)


# --- turning lines into turns -------------------------------------------------


def test_a_gap_or_a_restart_asks_the_agent_to_re_read():
    gap = Wakes().add([watch_mod.GAP_LINE])
    assert runner_mod.REREAD_PROPOSALS in gap
    restarted = Wakes().add([watch_mod.RESTARTED_LINE])
    assert runner_mod.REREAD_LEVEL in restarted
    level = Wakes().add(["LEVEL drive (was watch): the operator set the agent level"])
    assert runner_mod.REREAD_LEVEL in level
    plain = Wakes().add(["EVENT task_started 01"])
    assert plain == "The watcher printed:\nEVENT task_started 01"


def test_lines_are_held_while_the_agent_is_off():
    wakes = Wakes()
    assert wakes.add(["EVENT task_started 01", watch_mod.AGENT_OFF_LINE]) is None
    assert wakes.add([]) is None
    prompt = wakes.add([watch_mod.AGENT_BACK_LINE, "MESSAGE id=ab12 ..."])
    assert prompt.splitlines()[1:5] == [
        "EVENT task_started 01",
        watch_mod.AGENT_OFF_LINE,
        watch_mod.AGENT_BACK_LINE,
        "MESSAGE id=ab12 ...",
    ]
    assert runner_mod.REREAD_LEVEL in prompt


# --- against the App Server ---------------------------------------------------


def test_an_operator_message_wakes_the_agent_and_it_answers(session):
    answered = threading.Event()

    async def reply(prompt):
        found = re.search(r"MESSAGE id=(\w+) from the operator: (.*) --", prompt)
        if found is None:
            return None
        with _http(session) as http:
            note = http.post(
                "/app/experiment/notes",
                json={"text": "01 looks fine", "reply_to": found.group(1)},
            )
        assert note.status_code == 200, note.text
        answered.set()

    agent = ScriptedAgent(reply)
    running = _runner(session, agent)
    _until(lambda: agent.prompts, what="the kickoff")
    assert agent.started and agent.prompts[0] == kickoff()

    message_id = AgentSession(session).message("is 01 ok?")
    _until(answered.is_set, what="the answer")
    assert running.stop() == 0
    assert agent.closed

    assert agent.prompts[1].startswith(
        f"The watcher printed:\nMESSAGE id={message_id} from the operator: is 01 ok?"
    )
    notes = [
        e
        for e in session.recorder.buffer.events_since(0)["events"]
        if e["kind"] == "note" and e.get("actor") != "operator"
    ]
    assert [n["payload"].get("reply_to") for n in notes] == [message_id]


def test_lines_that_arrive_during_a_turn_are_the_next_turn(session, caplog):
    async def reply(prompt):
        if prompt.startswith("You are"):
            for text in ("first", "second"):
                AgentSession(session).message(text)
            # The turn goes on while the watcher hears both.
            while len(_heard(caplog, "MESSAGE")) < 2:
                await asyncio.sleep(0.05)

    agent = ScriptedAgent(reply)
    running = _runner(session, agent)
    _until(lambda: len(agent.prompts) >= 2, what="the second turn")
    time.sleep(2 * POLL_S)
    assert running.stop() == 0

    assert len(agent.prompts) == 2, "both lines came as one turn"
    lines = agent.prompts[1].splitlines()
    assert lines[0] == "The watcher printed:"
    assert [line.split(": ")[1].split(" --")[0] for line in lines[1:]] == [
        "first",
        "second",
    ]


def test_no_turns_while_the_agent_is_off(session, caplog):
    agent = ScriptedAgent()
    running = _runner(session, agent)
    _until(lambda: agent.prompts, what="the kickoff")
    operator = AgentSession(session)

    assert operator.set_level(AgentLevel.OFF) is None
    _until(lambda: _heard(caplog, "AGENT OFF"), what="AGENT OFF")
    message_id = operator.message("still there?")
    time.sleep(5 * POLL_S)
    assert len(agent.prompts) == 1, "a turn while the agent was Off"

    assert operator.set_level(AgentLevel.WATCH) is None
    # The message may come with AGENT BACK or as the turn after it.
    _until(
        lambda: any(f"MESSAGE id={message_id}" in p for p in agent.prompts),
        what="the message after AGENT BACK",
    )
    assert running.stop() == 0

    back = agent.prompts[1]
    assert watch_mod.AGENT_OFF_LINE in back and watch_mod.AGENT_BACK_LINE in back
    assert runner_mod.REREAD_LEVEL in back


def test_the_agent_answers_a_supervised_run_through_the_runner(session):
    assert AgentSession(session).set_level(AgentLevel.DRIVE) is None
    lamella = session.experiment.positions[0]
    answered, ended = [], threading.Event()

    async def reply(prompt):
        with _http(session) as http:
            if prompt.startswith("You are"):
                started = http.post(
                    "/app/workflow/start", json={"task_names": [SETUP]}
                ).json()
                assert started["started"], started
                return None
            if "PROPOSAL item=" in prompt:
                workflow = http.get("/app/workflow/status").json()["workflow"]
                for question in workflow["asking"]:
                    decided = http.post(
                        "/app/workflow/decide",
                        json={
                            "item_id": lamella.id,
                            "task_name": SETUP,
                            "proposal_id": question["proposal_id"],
                            "outcome": "Confirmed",
                        },
                    )
                    assert decided.status_code == 200, decided.text
                    answered.append(question["proposal_id"])
        if watch_mod.ENDED in prompt:
            ended.set()

    agent = ScriptedAgent(reply)
    running = _runner(session, agent)
    _until(ended.is_set, timeout=90, what="the run's end")
    assert running.stop() == 0

    assert session.runner.join(30), "the run did not end"
    assert answered, "the run asked nothing, or the agent never heard it"
    asked = [p for p in lamella.proposals_from(SETUP) if p.id in answered]
    assert all(str(p.current.actor).startswith("agent") for p in asked)
    assert not [p for p in lamella.proposals_from(SETUP) if p.pending]


def test_the_session_is_reported_to_the_app_after_each_turn_and_at_the_end(session):
    async def reply(prompt):
        return TurnEnd(cost_usd=0.25, session_id="scripted")

    agent = ScriptedAgent(reply)
    agent.model = "a-model"
    running = _runner(session, agent, goal="Mill 01 and 02", budget_usd=5.0)
    _until(lambda: agent.prompts, what="the kickoff")
    assert "Mill 01 and 02" in agent.prompts[0]
    _until(
        lambda: (AgentSession(session).reported_session() or {}).get("turns") == 1,
        what="the report",
    )
    reported = AgentSession(session).reported_session()
    assert reported["session_id"] == "scripted"
    assert reported["model"] == "a-model"
    assert reported["goal"] == "Mill 01 and 02"
    assert (reported["budget_usd"], reported["spent_usd"]) == (5.0, 0.25)
    assert not reported.get("ended")

    assert running.stop() == 0
    ended = AgentSession(session).reported_session()
    assert ended["ended"] and ended["reason"] == "stopped at fibsem-agent"
    kinds = [e["kind"] for e in session.recorder.buffer.events_since(0)["events"]]
    assert kinds.count("agent_session_started") == 1
    assert kinds.count("agent_session_ended") == 1


# --- ending -------------------------------------------------------------------


def test_stopping_interrupts_the_turn_and_closes_the_session(session):
    agent = ScriptedAgent(lambda prompt: agent.until_interrupted())
    running = _runner(session, agent)
    _until(lambda: agent.prompts, what="the kickoff")

    assert running.stop() == 0
    assert agent.interrupts == 1
    assert agent.closed


def test_a_spent_budget_ends_the_run(session):
    async def reply(prompt):
        return TurnEnd("error_max_budget_usd", 5.01, "scripted")

    agent = ScriptedAgent(reply)
    assert _runner(session, agent).ended() == 2
    assert agent.closed and len(agent.prompts) == 1


def test_a_model_that_cannot_sign_in_ends_the_run(session):
    # Not waiting for a wake-up the model could never answer.
    async def reply(prompt):
        return TurnEnd(error="authentication_failed")

    agent = ScriptedAgent(reply)
    assert _runner(session, agent).ended() == 1
    assert agent.closed and len(agent.prompts) == 1


def test_a_turn_with_no_result_ends_the_run(session):
    # The CLI died: the next wake-up would find no one to answer.
    async def reply(prompt):
        return TurnEnd("no_result")

    agent = ScriptedAgent(reply)
    assert _runner(session, agent).ended() == 1
    assert agent.closed


def test_a_cli_that_dies_on_the_ctrl_c_still_stops_cleanly(session):
    # A Ctrl-C at the terminal reaches the CLI too, which may end mid-turn.
    async def reply(prompt):
        await agent.until_interrupted()
        raise RuntimeError("the CLI exited")

    agent = ScriptedAgent(reply)
    running = _runner(session, agent)
    _until(lambda: agent.prompts, what="the kickoff")
    assert running.stop() == 0
    assert agent.closed


def test_the_kickoff_waits_while_the_agent_is_off(session, caplog):
    operator = AgentSession(session)
    assert operator.set_level(AgentLevel.OFF) is None
    agent = ScriptedAgent()
    running = _runner(session, agent)
    _until(lambda: _heard(caplog, "AGENT OFF"), what="AGENT OFF")
    time.sleep(5 * POLL_S)
    assert not agent.prompts, "a kickoff the server would refuse"

    assert operator.set_level(AgentLevel.WATCH) is None
    _until(lambda: agent.prompts, what="the kickoff")
    assert running.stop() == 0
    assert agent.prompts == [kickoff()]


class _NoEvents:
    """The real client, with an event stream that fails: a server that
    answers but can't be watched."""

    def __init__(self, client):
        self._client = client

    def request(self, method, path, *args, **kwargs):
        if path.startswith("/app/events"):
            return httpx.Response(500, json={"detail": "broken"})
        return self._client.request(method, path, *args, **kwargs)

    def close(self):
        self._client.close()


def test_a_server_whose_events_cannot_be_read_gets_no_turns(session, caplog):
    agent = ScriptedAgent()
    connect = _connect(session)
    running = Running(
        Runner(
            agent,
            lambda: _NoEvents(connect()),
            server_wait_s=1.0,
            poll_s=POLL_S,
            retry_s=RETRY_S,
        )
    )
    assert running.ended() == 1
    assert not agent.started and not agent.prompts
    # It gave up on the 500s, not on an error of its own.
    assert _heard(caplog, "SERVER UNAVAILABLE")
    assert not any("watcher failed" in r.getMessage() for r in caplog.records)


def test_with_no_server_the_agent_never_starts():
    agent = ScriptedAgent()
    running = Running(Runner(agent, lambda: None, server_wait_s=0.3, retry_s=0.05))
    assert running.ended() == 1
    assert not agent.started and not agent.prompts


def test_an_app_that_goes_away_and_comes_back_is_watched_again(session, caplog):
    agent = ScriptedAgent()
    running = _runner(session, agent, server_wait_s=30)
    _until(lambda: agent.prompts, what="the kickoff")

    session.app_server.stop()
    _until(
        lambda: any("watcher stopped" in r.getMessage() for r in caplog.records),
        what="the watcher to give up",
    )
    assert session.app_server.start()
    _until(lambda: len(agent.prompts) == 2, what="the turn after the app came back")
    assert running.stop() == 0

    back = agent.prompts[1]
    assert "SERVER GONE" in back
    assert "could not be watched" in back and runner_mod.REREAD_LEVEL in back


def test_an_app_that_stays_gone_ends_the_run(session):
    agent = ScriptedAgent()
    running = _runner(session, agent, server_wait_s=0.5)
    _until(lambda: agent.prompts, what="the kickoff")

    session.app_server.stop()
    assert running.ended() == 1
    assert agent.closed and len(agent.prompts) == 1
