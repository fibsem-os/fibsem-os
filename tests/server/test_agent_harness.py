"""``fibsem-agent``'s wake loop against a real App Server.

The App Server over a windowless Demo session, the real ``watch()`` and the
real events; only the model is stood in, by an agent that answers each turn
the way a model would, over HTTP. Nothing here imports Qt.
"""

import asyncio
import logging
import os
import re
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from fibsem.ai.agent import harness as harness_mod
from fibsem.ai.agent import instruments, playbooks, usage, wake_policy
from fibsem.ai.agent import watch as watch_mod
from fibsem.ai.agent.catalog import CATALOG
from fibsem.ai.agent.harness import Harness, TurnEnd, Wakes, kickoff
from fibsem.ai.agent.levels import AgentLevel
from fibsem.ai.agent.session import AgentControl
from fibsem.ai.agent.wake_lines import fibsem as fibsem_lines
from fibsem.applications.autolamella.server.context import (
    AutoLamellaAppContext,
)
from fibsem.server import AuthConfig
from fibsem.server.hosting import AppServerHost
from fibsem.workflows.attention import Attention
from fibsem.workflows.events import split_event_id
from fibsem.workflows.session import AppSession
from tests.autolamella.test_select_position_asks import (  # noqa: F401
    SETUP,
    _experiment,
    microscope,
)
from tests.waiting import free_port, wait_for

os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")

POLL_S = 0.3
RETRY_S = 0.05


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    """fibsem-agent starts only with an API key: a stand-in one, never sent."""
    monkeypatch.setenv(harness_mod.API_KEY, "sk-test-stand-in")


def _all():
    """What a session started on no particular playbook loads."""
    return harness_mod.session_names(harness_mod.find_checkout())


@pytest.fixture
def session(microscope, tmp_path, monkeypatch, caplog):  # noqa: F811
    """A windowless session with a supervised Select Position open, its App
    Server running at Watch."""
    monkeypatch.delenv("FIBSEM_SERVER_URL", raising=False)
    monkeypatch.delenv("FIBSEM_SERVER_TOKEN", raising=False)
    caplog.set_level(logging.INFO, logger=harness_mod.log.name)
    session = AppSession()
    session.microscope = microscope
    session.experiment = _experiment(
        tmp_path, microscope, attention=Attention.supervised
    )
    session.start_recording()
    session.agent_auth = AuthConfig.generate(level=AgentLevel.WATCH)
    session.app_server = AppServerHost(
        session,
        lambda: AutoLamellaAppContext(session),
        port=free_port(),
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
    """The harness on its own thread, as ``fibsem-agent`` runs it."""

    def __init__(self, harness):
        self.harness = harness
        self.code = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        self.code = asyncio.run(self.harness.run())

    def ended(self, timeout=30.0):
        self._thread.join(timeout)
        assert not self._thread.is_alive(), "the harness did not end"
        return self.code

    def stop(self):
        self.harness.stop()
        return self.ended()


def _harness(session, agent, **kwargs):
    kwargs.setdefault("poll_s", POLL_S)
    kwargs.setdefault("retry_s", RETRY_S)
    return Running(Harness(agent, _connect(session), **kwargs))


def _heard(caplog, start):
    return [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith(f"watch: {start}")
    ]


# --- the kickoff and the session's tools ------------------------------------


# --- the wake-up -------------------------------------------------------------


def test_a_wake_up_rings_once_with_its_reason():
    async def main():
        rang = []
        alarm = harness_mod.Alarm()
        alarm.bind(asyncio.get_running_loop(), rang.append)
        said = alarm.set(0.001, "check drift on 02")
        assert "0.001 min" in said and alarm.reason == "check drift on 02"
        await asyncio.sleep(0.2)
        assert rang == ["WAKE check drift on 02 (you asked 0.001 min ago)"]
        assert alarm.reason is None

    asyncio.run(main())


def test_a_new_wake_up_replaces_the_last_and_zero_cancels():
    async def main():
        rang = []
        alarm = harness_mod.Alarm()
        alarm.bind(asyncio.get_running_loop(), rang.append)
        alarm.set(0.001, "first")
        alarm.set(0.002, "second")
        await asyncio.sleep(0.3)
        assert rang == ["WAKE second (you asked 0.002 min ago)"]
        alarm.set(0.001, "third")
        assert alarm.set(0, "") == "No wake-up is set."
        await asyncio.sleep(0.2)
        assert len(rang) == 1
        # Capped, and cancelled when the run ends.
        assert f"{alarm.max_min:g} min" in alarm.set(10_000, "later")
        alarm.cancel()
        assert alarm.reason is None

    asyncio.run(main())


def test_a_wake_up_before_the_run_says_it_cannot_ring():
    assert "can't wake you" in harness_mod.Alarm().set(5, "x")


def test_the_wake_up_tool_is_the_harness_own_server():
    pytest.importorskip("claude_agent_sdk")
    checkout = harness_mod.find_checkout()
    options = harness_mod.build_options(
        checkout, url="http://127.0.0.1:9", token="t0k", alarm=harness_mod.Alarm()
    )
    assert list(options.mcp_servers) == ["fibsem", harness_mod.HARNESS_SERVER]
    assert options.mcp_servers[harness_mod.HARNESS_SERVER]["type"] == "sdk"
    assert harness_mod.WAKE_TOOL in options.allowed_tools


def test_a_deadline_wakes_the_agent_before_it_and_at_it():
    async def main():
        rang = []
        deadlines = harness_mod.Deadlines(warn_s=0.2)
        deadlines.bind(asyncio.get_running_loop(), rang.append)
        by = (datetime.now().astimezone() + timedelta(seconds=0.4)).isoformat()
        deadlines.set([{"text": "Finish", "by": by}, {"text": "No currents"}])
        # Kept from one report to the next, it keeps its timers.
        deadlines.set([{"text": "Finish", "by": by}])
        await asyncio.sleep(0.8)
        at = datetime.fromisoformat(by).strftime("%H:%M")
        assert rang == [
            f"DEADLINE in 0.00333333 min: Finish (by {at})",
            f"DEADLINE now: Finish (by {at})",
        ]

    asyncio.run(main())


def test_a_deadline_taken_away_or_already_past_rings_nothing():
    async def main():
        rang = []
        deadlines = harness_mod.Deadlines(warn_s=0.1)
        deadlines.bind(asyncio.get_running_loop(), rang.append)
        now = datetime.now().astimezone()
        soon = (now + timedelta(seconds=0.3)).isoformat()
        past = (now - timedelta(minutes=1)).isoformat()
        deadlines.set([{"text": "Finish", "by": soon}, {"text": "Old", "by": past}])
        deadlines.set([])
        await asyncio.sleep(0.5)
        assert rang == [] and deadlines.times == []

    asyncio.run(main())


def test_the_harness_keeps_the_deadlines_in_the_agents_plan(session):
    by = (datetime.now().astimezone() + timedelta(hours=2)).replace(
        second=0, microsecond=0
    )

    async def reply(prompt):
        with _http(session) as http:
            http.post("/app/agent/session", json={"session_id": "scripted"})
            http.post(
                "/app/agent/plan/section",
                json={"section": "Rules", "text": f"- Finish by {by:%H:%M}"},
            )
        return TurnEnd(session_id="scripted")

    agent = ScriptedAgent(reply)
    running = _harness(session, agent)
    assert wait_for(lambda: running.harness._deadlines.times, timeout=20), (
        "timed out waiting for the deadline"
    )
    assert running.harness._deadlines.times == [by.isoformat(timespec="minutes")]
    running.stop()
    assert running.harness._deadlines.times == []


def test_the_first_turn_can_write_the_plan(session):
    written = []

    async def reply(prompt):
        with _http(session) as http:
            written.append(
                http.post(
                    "/app/agent/plan/section",
                    json={"section": "Goal", "text": "Make two lamellae."},
                ).status_code
            )
        return TurnEnd(session_id="scripted")

    running = _harness(session, ScriptedAgent(reply))
    assert wait_for(lambda: written, timeout=20), (
        "timed out waiting for the first turn's plan"
    )
    running.stop()
    assert written == [200]


def test_a_message_is_read_as_the_turn_carrying_it_starts(session):
    agent = ScriptedAgent()
    running = _harness(session, agent)
    assert wait_for(lambda: len(agent.prompts) >= 1, timeout=20), (
        "timed out waiting for the kickoff"
    )
    message_id = AgentControl(session).message("Skip 04.")
    assert wait_for(lambda: len(agent.prompts) >= 2, timeout=20), (
        "timed out waiting for the message's turn"
    )
    assert f"MESSAGE id={message_id}" in agent.prompts[1]
    running.stop()
    read = [
        e["payload"]["ids"]
        for e in session.recorder.buffer.events_since(0)["events"]
        if e["kind"] == "agent_messages_read"
    ]
    # Once, by the harness: its watch leaves it to the turn.
    assert read == [[message_id]]


def test_a_wake_up_rings_into_the_next_turn(session):
    agent = ScriptedAgent()
    running = _harness(session, agent, alarm=(alarm := harness_mod.Alarm()))
    assert wait_for(lambda: len(agent.prompts) >= 1, timeout=20), (
        "timed out waiting for the kickoff"
    )
    running.harness._loop.call_soon_threadsafe(alarm.set, 0.001, "look at 03 again")
    assert wait_for(lambda: len(agent.prompts) >= 2, timeout=20), (
        "timed out waiting for the wake-up turn"
    )
    assert "WAKE look at 03 again" in agent.prompts[1]
    running.stop()


def test_a_turn_that_reaches_its_step_limit_carries_on_at_once(session):
    # Nobody is watching: nothing else would start the next turn.
    async def reply(prompt):
        return TurnEnd("error_max_turns") if len(agent.prompts) == 1 else None

    agent = ScriptedAgent(reply)
    running = _harness(session, agent)
    assert wait_for(lambda: len(agent.prompts) >= 2, timeout=20), (
        "the run stalled after the step limit"
    )
    assert harness_mod.CARRY_ON in agent.prompts[1]
    running.stop()


def test_the_kickoff_reads_the_base_skill_then_the_task_skill_or_asks():
    first = kickoff()
    assert "Read the fibsem-agent and fibsem-microscope skills" in first
    assert "task skill" in first
    assert "what they would like to get done today" in first
    assert "server_get_capabilities" in first
    assert "don't start one" in first
    # The harness never sets the level: it tells the agent whose it is.
    assert "level is the operator's" in first
    again = kickoff(resumed=True)
    assert "resumed" in again and "server_get_capabilities" in again
    assert "LEVEL line" in harness_mod.texts().prompt
    assert "stop and ask" in harness_mod.texts().prompt
    assert "only through fibsem-os" in harness_mod.texts().prompt
    # A goal is said back, with its rule, before the agent acts on it.
    goal = kickoff(goal="mill 01 and 02")
    assert "mill 01 and 02" in goal and "Before acting on it" in goal


def test_the_sidecar_is_told_the_playbooks_instrument_kind():
    pytest.importorskip("claude_agent_sdk")
    from fibsem.ai.agent.catalog import INSTRUMENT_ENV

    checkout = harness_mod.find_checkout()
    playbook = harness_mod.playbooks.load(harness_mod.playbooks.DEFAULT, checkout)
    options = harness_mod.build_options(checkout, playbook=playbook)
    assert options.env[INSTRUMENT_ENV] == "fibsem"
    assert INSTRUMENT_ENV not in harness_mod.build_options(checkout).env


def test_the_agent_has_the_fibsem_tools_and_its_skills_and_nothing_else(tmp_path):
    pytest.importorskip("claude_agent_sdk")
    checkout = harness_mod.find_checkout()
    assert checkout is not None, "tests run from a checkout with .claude/skills"
    options = harness_mod.build_options(
        checkout, url="http://127.0.0.1:9", token="t0k", resume="abc"
    )

    assert options.permission_mode == "dontAsk"
    assert options.allowed_tools == [
        "mcp__fibsem__*",
        "ToolSearch",
        "SendMessage",
        harness_mod.WAKE_TOOL,
        harness_mod.WAKE_ON_TOOL,
        *harness_mod.PLAYBOOK_TOOLS,
        harness_mod.HANDOFF_TOOL,
    ] + [f"Agent({name})" for name in _all().agents]
    # SendMessage carries a specialist on (one Verifier per lamella's passes).
    # The checklist is a fibsem tool, so the operator sees it: no CLI todo list.
    assert options.tools == ["Skill", "ToolSearch", "Agent", "SendMessage"]
    for tool in ("Bash", "Read", "Grep", "Glob", "Write", "Edit", "WebFetch"):
        assert tool in options.disallowed_tools
    assert options.effort == "high"
    # The CLI's own agents are refused: only the specialists are called, and
    # each answers inside the turn that called it.
    for agent in ("general-purpose", "claude", "Explore"):
        assert f"Agent({agent})" in options.disallowed_tools
    assert set(options.skills) == {
        "fibsem-agent",
        "fibsem-microscope",
        "autolamella",
        "autolamella-lead",
        "autolamella-assistant",
        "autolamella-polishing",
        "autolamella-correlative",
        "autolamella-coincidence-milling",
        "screen-grids",
        "fibsem-experiment",
    }
    for skill in options.skills:
        assert (checkout / ".claude" / "skills" / skill / "SKILL.md").is_file()
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
    assert options.max_turns == harness_mod.defaults().max_turns
    # The harness keeps the budget: given one, the CLI says the spend in a
    # reminder on every model call.
    assert options.max_budget_usd is None
    # A tool result with an image in it is more than the SDK's 1 MB default.
    assert options.max_buffer_size >= 16 * 1024 * 1024
    # No developer CLAUDE.md or auto memory in an operator's agent.
    assert options.env == {
        "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
        "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
        # The main conversation's prompt cache lasts an hour (`wake_policy`).
        "CLAUDE_CODE_PROMPT_CACHE_TTL": "1h",
        # The base playbook's compaction window, the context ceiling's backstop.
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "700000",
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
    assert "Skill(autolamella)" in allowed and "mcp__fibsem__*" in allowed
    assert "bypassPermissions" not in command
    assert not any("t0k" in part for part in command), "the token is in argv"


def test_a_labs_extra_mcp_server_is_added_and_its_token_is_never_shown(
    tmp_path, monkeypatch, capsys
):
    pytest.importorskip("claude_agent_sdk")
    import dataclasses
    import json

    from claude_agent_sdk._internal.transport.subprocess_cli import (
        SubprocessCLITransport,
    )

    from fibsem.ai.agent import extra_mcp

    secret = "not-a-real-token"
    monkeypatch.setenv("FIBSEM_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("NOTEBOOK_TOKEN", secret)
    (tmp_path / "home" / "agent").mkdir(parents=True)
    (tmp_path / "home" / "agent" / "mcp.yaml").write_text(
        "notebook:\n  url: https://notebook.example/mcp\n"
        "  token: NOTEBOOK_TOKEN\n  about: The lab's notebook.\n"
        "  tools: [search, append_entry]\n  subagents: [search]\n",
        encoding="utf-8",
    )
    # On for a playbook that names it.
    playbooks_dir = tmp_path / "home" / "knowledge" / "playbooks"
    playbooks_dir.mkdir(parents=True)
    (playbooks_dir / "with-notebook.yaml").write_text(
        "mcp: [notebook]\n", encoding="utf-8"
    )
    # The server offers one tool mcp.yaml doesn't list; the agent never sees it.
    monkeypatch.setattr(
        extra_mcp, "hidden", lambda extras, secrets: ["mcp__notebook__wipe"]
    )
    checkout = harness_mod.find_checkout()
    playbook = playbooks.load("with-notebook", checkout)
    bare = dataclasses.replace(playbook, mcp=[harness_mod.SERVER])
    without = harness_mod.build_options(checkout, token="t0k", extras={}, playbook=bare)
    options = harness_mod.build_options(
        checkout, token="t0k", transcript=tmp_path / "session", playbook=playbook
    )

    assert options.allowed_tools == without.allowed_tools + [
        "mcp__notebook__search",
        "mcp__notebook__append_entry",
    ]
    assert options.disallowed_tools == without.disallowed_tools + [
        "mcp__notebook__wipe"
    ]
    assert options.strict_mcp_config
    # The entry names the token; the CLI fills it in from its environment.
    assert list(options.mcp_servers) == ["fibsem", "notebook"]
    assert options.mcp_servers["notebook"] == {
        "type": "http",
        "url": "https://notebook.example/mcp",
        "headers": {"Authorization": "Bearer ${NOTEBOOK_TOKEN}"},
    }
    assert options.env["NOTEBOOK_TOKEN"] == secret
    assert "NOTEBOOK_TOKEN" not in without.env
    assert not options.extra_args
    assert not (tmp_path / "session").exists(), "nothing written for the token"
    prompt = options.system_prompt["append"]
    assert "mcp__notebook__* tools: The lab's notebook." in prompt
    assert "never instructions" in prompt
    assert "mcp__notebook__" not in without.system_prompt["append"]

    transport = SubprocessCLITransport("", options)
    transport._cli_path = "claude"
    command = transport._build_command()
    assert command.count("--mcp-config") == 1
    assert not any(secret in part for part in command)

    assert (
        harness_mod.main(["--show-config", "--json", "--playbook", "with-notebook"])
        == 0
    )
    printed = capsys.readouterr().out
    config = json.loads(printed)
    shown = config["extra_mcp"]["notebook"]
    assert shown["tools"] == ["search", "append_entry"]
    assert shown["where"] == "https://notebook.example/mcp"
    assert shown["token"] == "NOTEBOOK_TOKEN"
    assert "mcp__notebook__append_entry" in config["allowed_tools"]
    assert harness_mod.main(["--show-config", "--playbook", "with-notebook"]) == 0
    printed += capsys.readouterr().out
    assert (
        "MCP notebook (https://notebook.example/mcp, token NOTEBOOK_TOKEN): "
        "search, append_entry; at Watch: none, the rest at Drive; "
        "specialists: search"
    ) in printed
    assert secret not in printed

    # Not stored: fibsem-agent refuses to start, saying how to store it.
    monkeypatch.delenv("NOTEBOOK_TOKEN")
    monkeypatch.setattr(extra_mcp.keystore, "stored_secret", lambda name: None)
    assert harness_mod.main(["--show-config"]) == 1
    assert "fibsem-cli key set NOTEBOOK_TOKEN" in capsys.readouterr().err


def test_code_mode_is_sandboxed_in_its_work_folder(tmp_path, monkeypatch):
    pytest.importorskip("claude_agent_sdk")
    import json

    from fibsem.ai.agent import code_mode

    monkeypatch.setenv("SOME_SERVICE_TOKEN", "x")
    checkout = harness_mod.find_checkout()
    work = tmp_path / "work"
    code_mode.prepare(work, checkout, _all().skills, _all().agents)
    options = harness_mod.build_options(checkout, token="t0k", work=work)

    # The skills are read from the work folder, which is the CLI's.
    assert options.cwd == str(work)
    for skill in _all().skills:
        assert (work / ".claude" / "skills" / skill / "SKILL.md").is_file()
    assert (work / "notebook" / "INDEX.md").is_file()
    for agent in _all().agents:
        assert (work / ".claude" / "agents" / f"{agent}.md").is_file()

    assert options.permission_mode == "dontAsk"
    for tool in ("Bash", "Read", "Write", "Edit", "Glob", "Grep"):
        assert tool in options.tools
    for tool in ("WebFetch", "WebSearch", "Agent(general-purpose)"):
        assert tool in options.disallowed_tools
    assert "Bash" not in options.disallowed_tools
    # Writes in the work folder only.
    rule = f"/{work.as_posix()}/**"
    assert f"Edit({rule})" in options.allowed_tools
    assert "Write" not in options.allowed_tools and "Edit" not in options.allowed_tools

    settings = json.loads(options.settings)
    assert settings["permissions"]["blockReadsOutsideWorkingDirectories"] is True
    assert (
        f"Edit(/{(work / '.claude').as_posix()}/**)"
        in (settings["permissions"]["deny"])
    )
    sandbox = settings["sandbox"]
    assert sandbox["enabled"] and sandbox["failIfUnavailable"]
    # Its shell can't write the CLI's settings or hooks either.
    assert sandbox["filesystem"]["denyWrite"] == [str(work / ".claude")]
    assert sandbox["allowUnsandboxedCommands"] is False
    assert sandbox["network"]["allowedDomains"] == []
    hidden = {entry["name"] for entry in sandbox["credentials"]["envVars"]}
    assert {
        "ANTHROPIC_API_KEY",
        "CLAUDE_CODE_MESSAGING_TOKEN",
        "FIBSEM_SERVER_TOKEN",
        "SOME_SERVICE_TOKEN",
    } <= hidden
    assert all(entry["mode"] == "deny" for entry in sandbox["credentials"]["envVars"])

    assert options.env["FIBSEM_AGENT_WORKDIR"] == str(work)
    assert options.env["MPLCONFIGDIR"].startswith(str(work))
    prompt = options.system_prompt["append"]
    assert str(work) in prompt and "no network" in prompt
    # Its code runs with fibsem-agent's own environment, read only.
    assert sys.executable in prompt
    readable = sandbox["filesystem"]["allowRead"]
    assert any(Path(sys.prefix).resolve().is_relative_to(path) for path in readable)
    import fibsem

    package = Path(fibsem.__file__).resolve()
    assert any(package.is_relative_to(path) for path in readable)
    assert "There is no shell" not in prompt

    # On the CLI's command line, as flag settings, which a project's own
    # settings can't loosen; the token still isn't in argv.
    from claude_agent_sdk._internal.transport.subprocess_cli import (
        SubprocessCLITransport,
    )

    transport = SubprocessCLITransport("", options)
    transport._cli_path = "claude"
    command = transport._build_command()
    assert json.loads(command[command.index("--settings") + 1]) == settings
    assert not any("t0k" in part for part in command), "the token is in argv"


def test_files_mode_reads_and_writes_its_folder_and_runs_nothing(tmp_path):
    pytest.importorskip("claude_agent_sdk")
    import json

    from fibsem.ai.agent import code_mode

    checkout = harness_mod.find_checkout()
    work = tmp_path / "work"
    code_mode.prepare(work, checkout, _all().skills, _all().agents)
    options = harness_mod.build_options(checkout, work=work, code=False)

    assert options.cwd == str(work)
    for tool in ("Read", "Write", "Edit", "Glob", "Grep"):
        assert tool in options.tools
    # No shell: not given, not allowed, and still denied.
    assert "Bash" not in options.tools
    assert "Bash" not in options.allowed_tools
    assert "Bash" in options.disallowed_tools
    assert f"Edit(/{work.as_posix()}/**)" in options.allowed_tools
    settings = json.loads(options.settings)
    assert settings["permissions"]["blockReadsOutsideWorkingDirectories"] is True
    assert (
        f"Edit(/{(work / '.claude').as_posix()}/**)"
        in (settings["permissions"]["deny"])
    )
    # Nothing runs, so there is no sandbox to need.
    assert "sandbox" not in settings
    prompt = options.system_prompt["append"]
    assert str(work) in prompt and "can't run code" in prompt
    assert "Run Python" not in prompt and "notebook/INDEX.md" in prompt


def test_files_mode_works_on_windows_but_still_needs_a_folder_of_its_own(
    tmp_path, monkeypatch
):
    from fibsem.ai.agent import code_mode

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    checkout = tmp_path / "fibsem-os"
    work = tmp_path / "home" / "fibsem-work"
    monkeypatch.setattr(code_mode.sys, "platform", "win32")
    assert code_mode.refusal(work, checkout, code=False) is None
    for wrong in (tmp_path / "home", tmp_path, checkout / "sub"):
        assert code_mode.refusal(wrong, checkout, code=False), wrong
    assert code_mode.describe(work, code=False) == (
        f"Off; reads and writes files in {work} (--files)"
    )


def test_folder_rules_name_a_windows_folder_as_claude_code_matches_it():
    """Claude Code matches rules against a Windows path in POSIX form, its
    drive a lower-case folder: a rule with C: in it allows nothing."""
    from pathlib import PurePosixPath, PureWindowsPath

    from fibsem.ai.agent import code_mode

    work = PureWindowsPath(r"C:\Users\User\fibsem-work")
    assert code_mode.allowed_tools(work, code=False)[-1] == (
        "Edit(//c/Users/User/fibsem-work/**)"
    )
    deny = code_mode.settings(work, {}, code=False)["permissions"]["deny"]
    assert deny == ["Edit(//c/Users/User/fibsem-work/.claude/**)"]
    assert code_mode.allowed_tools(PurePosixPath("/home/a/w"))[-1] == (
        "Edit(//home/a/w/**)"
    )


def test_code_and_files_are_one_or_the_other(capsys):
    with pytest.raises(SystemExit):
        harness_mod.main(["--code", "a", "--files", "b"])
    assert "use one of them" in capsys.readouterr().err


def test_code_mode_rebuilds_its_claude_folder_every_start(tmp_path):
    """A settings file (or anything else) left in work/.claude doesn't reach
    the next session: only the skills and specialists are there."""
    from fibsem.ai.agent import code_mode

    checkout = harness_mod.find_checkout()
    work = tmp_path / "work"
    skills, agents = _all().skills, _all().agents
    code_mode.prepare(work, checkout, skills, agents)
    planted = [
        work / ".claude" / "settings.json",
        work / ".claude" / "settings.local.json",
        work / ".claude" / "agents" / "other.md",
        work / ".claude" / "commands" / "x.md",
    ]
    for path in planted:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"hooks": {}}')
    (work / ".claude" / "skills" / skills[0] / "SKILL.md").write_text("changed")
    (work / "notebook" / "lesson.md").write_text("kept")

    code_mode.prepare(work, checkout, skills, agents)

    assert not any(path.exists() for path in planted)
    shipped = checkout / ".claude" / "skills" / skills[0] / "SKILL.md"
    copied = work / ".claude" / "skills" / skills[0] / "SKILL.md"
    assert copied.read_text() == shipped.read_text()
    assert {p.name for p in (work / ".claude" / "agents").iterdir()} == {
        f"{name}.md" for name in agents
    }
    # The notebook is the agent's own and stays.
    assert (work / "notebook" / "lesson.md").read_text() == "kept"


def test_code_mode_needs_a_folder_of_its_own(tmp_path, monkeypatch):
    from fibsem.ai.agent import code_mode

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    checkout = tmp_path / "fibsem-os"
    work = tmp_path / "home" / "fibsem-work"
    monkeypatch.setattr(code_mode.sys, "platform", "darwin")
    assert code_mode.refusal(work, checkout) is None
    for wrong in (tmp_path / "home", Path("/"), tmp_path, checkout / "sub"):
        assert code_mode.refusal(wrong, checkout), wrong
    # fibsem-agent's own environment can't be in the agent's folder.
    monkeypatch.setattr(code_mode.sys, "prefix", str(work / "env"))
    assert "Python" in code_mode.refusal(work, checkout)
    monkeypatch.setattr(code_mode.sys, "platform", "win32")
    assert "sandbox" in code_mode.refusal(work, checkout)


def test_fibsem_agent_says_whether_code_mode_is_on(tmp_path, monkeypatch):
    from fibsem.ai.agent import code_mode

    monkeypatch.setattr(code_mode.sys, "platform", "darwin")
    assert code_mode.describe(None) == "Off (--code to turn it on)"
    assert code_mode.describe(tmp_path) == f"Available in {tmp_path} (sandboxed)"
    monkeypatch.setattr(code_mode.sys, "platform", "win32")
    assert code_mode.describe(None).startswith("Not available")


def test_fibsem_agent_starts_with_its_status(tmp_path):
    import argparse

    args = argparse.Namespace(
        url=None,
        model=None,
        goal="mill 01",
        max_budget_usd=5.0,
        max_turns=40,
        resume=None,
    )
    lines = harness_mod.status(
        args, tmp_path, tmp_path / "a.log", tmp_path / "w", "abc-123"
    )
    rows = dict(line.split(":", 1) for line in lines)
    assert rows["Goal"].strip() == "mill 01"
    assert rows["Session"].strip() == "abc-123 (new)"
    assert rows["Code"].strip().startswith("Available in")
    assert len({line.index(line.split()[1]) for line in lines}) == 1, "aligned"


def test_a_bare_resume_carries_on_the_last_session_in_the_same_folder(
    tmp_path, monkeypatch
):
    from fibsem.ai.agent import manager

    monkeypatch.setattr(harness_mod.Path, "home", lambda: tmp_path)
    work = tmp_path / "work"
    manager.SessionFolder.open("first", {"cwd": str(work)})
    assert manager.last_session(work) == "first"
    assert manager.folders()[0].parent == tmp_path / ".fibsem" / "agent" / "sessions"


def test_each_session_has_a_folder_with_its_record_and_log(tmp_path, monkeypatch):
    """No App Server answers, so the agent never starts: the run still leaves
    its record, and how it ended."""
    pytest.importorskip("claude_agent_sdk")
    import uuid

    from fibsem.ai.agent import manager

    monkeypatch.setattr(harness_mod.Path, "home", lambda: tmp_path)
    monkeypatch.delenv("FIBSEM_SERVER_TOKEN", raising=False)
    session_id = str(uuid.uuid4())
    handlers = list(harness_mod.log.handlers)
    try:
        code = harness_mod.main(
            ["--url", "http://127.0.0.1:9", "--server-wait", "0"]
            + ["--session-id", session_id, "--goal", "mill 01"]
        )
    finally:
        for handler in harness_mod.log.handlers[len(handlers) :]:
            handler.close()
            harness_mod.log.removeHandler(handler)
    assert code == 1
    folder = manager.find_folder(session_id)
    assert folder is not None and folder.name.endswith(session_id)
    record = manager.read_json(folder / manager.RECORD)
    assert record["goal"] == "mill 01"
    assert record["url"] == "http://127.0.0.1:9"
    assert record["mode"] == "files"
    (run,) = record["runs"]
    assert run["exit_code"] == 1 and run["resumable"] is True
    assert "Session:" in (folder / manager.LOG).read_text(encoding="utf-8")


def _main_kept_apart(argv):
    """`harness.main`, its log handlers taken off again after."""
    handlers = list(harness_mod.log.handlers)
    try:
        return harness_mod.main(argv)
    finally:
        for handler in harness_mod.log.handlers[len(handlers) :]:
            handler.close()
            harness_mod.log.removeHandler(handler)


def test_with_no_api_key_it_refuses_to_start(tmp_path, monkeypatch, capsys):
    """No key, no start: it never falls back to this computer's Claude login,
    and it opens no session."""
    # Without the [agent] extra it stops earlier, on the missing extra.
    pytest.importorskip("claude_agent_sdk")
    monkeypatch.delenv(harness_mod.API_KEY, raising=False)
    monkeypatch.setattr(harness_mod, "get_secret", lambda name: None)
    from fibsem.ai.agent import manager

    before = set(manager.sessions_dir().glob("*"))

    code = _main_kept_apart(["--url", "http://127.0.0.1:9", "--server-wait", "0"])

    assert code == 1
    assert "No API key: run fibsem-cli key set ANTHROPIC_API_KEY" in (
        capsys.readouterr().err
    )
    assert harness_mod.API_KEY not in os.environ
    assert set(manager.sessions_dir().glob("*")) == before


def test_a_failure_outside_the_run_is_said_and_recorded(tmp_path, monkeypatch, capsys):
    """An error after the session opened, here in rebuilding its folder, says
    why on the console and in the log, and ends the run in its record."""
    pytest.importorskip("claude_agent_sdk")
    import uuid

    from fibsem.ai.agent import manager, sources

    monkeypatch.setattr(harness_mod.Path, "home", lambda: tmp_path)
    monkeypatch.delenv("FIBSEM_SERVER_TOKEN", raising=False)

    def assemble(*args, **kwargs):
        raise FileExistsError("run/.claude/skills was made meanwhile")

    monkeypatch.setattr(sources, "assemble", assemble)
    session_id = str(uuid.uuid4())
    code = _main_kept_apart(
        ["--url", "http://127.0.0.1:9", "--server-wait", "0"]
        + ["--session-id", session_id]
    )
    assert code == 1
    why = "fibsem-agent failed: FileExistsError: run/.claude/skills was made meanwhile"
    assert why in capsys.readouterr().out
    folder = manager.find_folder(session_id)
    logged = (folder / manager.LOG).read_text(encoding="utf-8")
    assert why in logged and "Traceback" in logged
    assert f"--resume {session_id} carries the session on." in logged
    (run,) = manager.read_json(folder / manager.RECORD)["runs"]
    assert run["exit_code"] == 1 and run["reason"] == why


def test_a_session_running_already_is_not_run_again(tmp_path, monkeypatch, capsys):
    """A second resume of a session still running says so, and leaves the
    running one's record alone."""
    pytest.importorskip("claude_agent_sdk")
    import subprocess
    import sys
    import uuid

    from fibsem.ai.agent import manager

    monkeypatch.setattr(harness_mod.Path, "home", lambda: tmp_path)
    monkeypatch.delenv("FIBSEM_SERVER_TOKEN", raising=False)
    session_id = str(uuid.uuid4())
    session = manager.SessionFolder.open(session_id, {})
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        record = session.record()
        record["runs"][-1]["pid"] = other.pid
        manager.write_json(session.folder / manager.RECORD, record)
        code = _main_kept_apart(
            ["--url", "http://127.0.0.1:9", "--server-wait", "0"]
            + ["--resume", session_id]
        )
    finally:
        other.kill()
        other.wait(10)
    assert code == 1
    err = capsys.readouterr().err
    assert f"Session {session_id} is already running (process {other.pid})" in err
    assert manager.read_json(session.folder / manager.RECORD) == record


def test_a_specialist_copied_for_a_session_gets_the_extra_servers_tools(
    tmp_path, monkeypatch
):
    """In run/ and in a work folder; the checkout's file stays as shipped."""
    pytest.importorskip("claude_agent_sdk")
    import uuid

    from fibsem.ai.agent import manager, sources, subagents

    monkeypatch.setattr(harness_mod.Path, "home", lambda: tmp_path)
    monkeypatch.delenv("FIBSEM_SERVER_TOKEN", raising=False)
    monkeypatch.setenv("FIBSEM_HOME", str(tmp_path / "home"))
    (tmp_path / "home" / "agent").mkdir(parents=True)
    (tmp_path / "home" / "agent" / "mcp.yaml").write_text(
        "notebook:\n  url: https://notebook.example/mcp\n"
        "  tools: [search]\n  subagents: [search]\n",
        encoding="utf-8",
    )
    # No file tools: the session runs in its run/.
    (tmp_path / "home" / "knowledge" / "playbooks").mkdir(parents=True)
    (tmp_path / "home" / "knowledge" / "playbooks" / "plain.yaml").write_text(
        "agents: [autolamella-verifier]\nmcp: [notebook]\nsession:\n  mode: none\n",
        encoding="utf-8",
    )
    shipped = sources.agent_file("autolamella-verifier", harness_mod.find_checkout())
    before = shipped.read_bytes()
    handlers = list(harness_mod.log.handlers)
    run_id = str(uuid.uuid4())
    plain = ["--session-id", run_id, "--playbook", "plain"]
    try:
        # The playbook names the notebook: its specialists get its tools.
        named = ["--playbook", "plain", "--files", str(tmp_path / "w")]
        for more in (plain, named):
            code = harness_mod.main(
                ["--url", "http://127.0.0.1:9", "--server-wait", "0"] + more
            )
            assert code == 1
    finally:
        for handler in harness_mod.log.handlers[len(handlers) :]:
            handler.close()
            harness_mod.log.removeHandler(handler)
    for folder in (manager.find_folder(run_id) / manager.RUN, tmp_path / "w"):
        copied = folder / ".claude" / "agents" / "autolamella-verifier.md"
        data, _ = subagents.front_matter(copied.read_text(encoding="utf-8"))
        assert subagents.names(data["tools"])[-1] == "mcp__notebook__search"
    assert shipped.read_bytes() == before


def test_a_session_id_is_a_new_session_and_a_uuid(capsys):
    with pytest.raises(SystemExit):
        harness_mod.main(["--session-id", str(__import__("uuid").uuid4()), "--resume"])
    assert "--resume carries one on" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        harness_mod.main(["--session-id", "not-a-uuid"])
    assert "not a UUID" in capsys.readouterr().err


def test_a_bare_resume_with_no_earlier_session_says_so(tmp_path, monkeypatch, capsys):
    pytest.importorskip("claude_agent_sdk")
    monkeypatch.setattr(harness_mod.Path, "home", lambda: tmp_path)
    assert harness_mod.main(["--resume", "--files", str(tmp_path / "w")]) == 1
    assert "No earlier fibsem-agent session" in capsys.readouterr().err


def test_a_given_folder_is_resolved(tmp_path, monkeypatch):
    from fibsem.ai.agent import code_mode

    monkeypatch.chdir(tmp_path)
    assert code_mode.workdir(".") == tmp_path.resolve()


# --- the specialists ------------------------------------------------------------

# Read-only, but each saves its result, figure or drawing beside the image it
# measured or drew on.
MEASURES = {
    "item_measure_thickness",
    "analysis_draw_image",
    "analysis_draw_pair",
    "analysis_measure_profile",
    "analysis_find_grid_squares",
    "analysis_measure_coincidence_offset",
    "grid_draw_milling",
}
# Only the Run Lead decides, starts, stops, mills, edits and keeps the record.
LEAD_ONLY = {
    "proposal_decide",
    "proposal_suggest",
    "request_create",
    "experiment_add_note",
    "experiment_message_operator",
    "item_update_task_config",
    "item_remove",
    "milling_run_stage",
    "milling_run_task",
    "milling_stop",
    "milling_end",
}
# The Verifier puts its own answer on the record, so the review needn't trust the
# Run Lead's copy of it.
NOTES_ITS_ANSWER = {"autolamella-verifier": {"experiment_add_note"}}


def _specialists():
    """Each agent file's frontmatter, by its file name."""
    import yaml

    checkout = harness_mod.find_checkout()
    agents = {}
    for path in sorted((checkout / ".claude" / "agents").glob("autolamella-*.md")):
        _, front, _ = path.read_text(encoding="utf-8").split("---\n", 2)
        agents[path.stem] = yaml.safe_load(front)
    return agents


def _tools(front):
    return [tool.strip() for tool in front["tools"].split(",")]


def test_the_harness_allows_the_specialists_the_checkout_defines():
    agents = _specialists()
    assert set(agents) == set(_all().agents)
    for name, front in agents.items():
        assert front["name"] == name
        assert front["description"]


def test_a_specialist_has_only_tools_that_exist_and_the_harness_can_refuse():
    from fibsem.ai.agent.catalog import CATALOG

    catalog = {spec.name for spec in CATALOG}
    for name, front in _specialists().items():
        for tool in _tools(front):
            if tool.startswith("mcp__fibsem__"):
                assert tool.removeprefix("mcp__fibsem__") in catalog, (name, tool)
            else:
                # An agent gets a tool it lists unless the harness denies it.
                assert tool in harness_mod.DENIED_TOOLS, (name, tool)


def test_only_the_run_lead_decides_and_its_agents_only_read():
    from fibsem.ai.agent.catalog import CATALOG

    read_only = {spec.name for spec in CATALOG if spec.read_only}
    for name, front in _specialists().items():
        fibsem = {
            tool.removeprefix("mcp__fibsem__")
            for tool in _tools(front)
            if tool.startswith("mcp__fibsem__")
        } - NOTES_ITS_ANSWER.get(name, set())
        assert not fibsem & LEAD_ONLY, (name, fibsem & LEAD_ONLY)
        assert not {t for t in fibsem if t.startswith(("workflow_", "protocol_"))} - (
            read_only
        ), name
        assert fibsem <= read_only | MEASURES, (name, fibsem - read_only)


def test_the_site_finder_only_looks_and_the_lead_keeps_the_record():
    finder = _specialists()["autolamella-site-finder"]
    fibsem = {
        tool.removeprefix("mcp__fibsem__")
        for tool in _tools(finder)
        if tool.startswith("mcp__fibsem__")
    }
    specs = {spec.name: spec for spec in CATALOG if spec.name in fibsem}
    assert set(specs) == fibsem
    kind = instruments.devices("fibsem", harness_mod.find_checkout())
    assert all(kind.offers(spec) for spec in specs.values())
    # It saves drawings and figures beside images, and nothing else.
    assert {n for n, spec in specs.items() if not spec.read_only} <= MEASURES
    assert "autolamella-site-finder" not in NOTES_ITS_ANSWER
    assert not {t for t in fibsem if t.startswith(("proposal_", "plan_"))}
    assert not {"grid_set_note", "grid_set_quality", "knowledge_write"} & fibsem


# --- turning lines into turns -------------------------------------------------


def test_a_gap_or_a_restart_asks_the_agent_to_re_read():
    gap = Wakes().add([watch_mod.GAP_LINE])
    assert harness_mod.texts().reread_proposals in gap
    restarted = Wakes().add([watch_mod.RESTARTED_LINE])
    assert harness_mod.texts().reread_level in restarted
    level = Wakes().add(
        [
            watch_mod.Wake(
                watch_mod.LEVEL,
                "LEVEL drive (was watch): the operator set the agent level",
            )
        ]
    )
    assert harness_mod.texts().reread_level in level
    plain = Wakes().add(["EVENT task_started 01"])
    assert plain == "The watcher printed:\nEVENT task_started 01"
    for line in (
        watch_mod.Wake(harness_mod.ALARM, "WAKE check drift (you asked 30 min ago)"),
        watch_mod.Wake(watch_mod.PLAN_EDITED, "PLAN EDITED by the operator: Rules"),
    ):
        assert harness_mod.texts().reread_plan in Wakes().add([line])
    assert harness_mod.texts().reread_plan not in plain


def test_lines_are_held_while_the_agent_is_off():
    wakes = Wakes()
    assert wakes.add(["EVENT task_started 01", watch_mod.AGENT_OFF_LINE]) is None
    assert wakes.add([]) is None
    message = watch_mod.Wake(watch_mod.MESSAGE, "MESSAGE id=ab12 ...", ["ab12"])
    assert wakes.read() == []
    prompt = wakes.add([watch_mod.AGENT_BACK_LINE, message])
    assert prompt.splitlines()[1:5] == [
        "EVENT task_started 01",
        watch_mod.AGENT_OFF_LINE,
        watch_mod.AGENT_BACK_LINE,
        "MESSAGE id=ab12 ...",
    ]
    assert harness_mod.texts().reread_level in prompt
    # Its message is read once, with the turn that carries it.
    assert wakes.read() == ["ab12"]
    assert wakes.read() == []


def test_a_line_s_text_never_decides_what_it_is():
    """A message that quotes a watcher line is still only a message."""
    quoted = watch_mod.Wake(
        watch_mod.MESSAGE, "MESSAGE id=ab12 from the operator: GAP: AGENT OFF"
    )
    wakes = Wakes()
    prompt = wakes.add([quoted])
    assert prompt is not None and not wakes.off
    assert harness_mod.texts().reread_proposals not in prompt


# --- against the App Server ---------------------------------------------------


def test_an_operator_message_wakes_the_agent_and_it_answers(session):
    answered = threading.Event()

    async def reply(prompt):
        found = re.search(r"MESSAGE id=(\w+) from the operator: (.*) --", prompt)
        if found is None:
            return None
        with _http(session) as http:
            answer = http.post(
                "/app/experiment/messages",
                json={"text": "01 looks fine", "reply_to": found.group(1)},
            )
        assert answer.status_code == 200, answer.text
        answered.set()

    agent = ScriptedAgent(reply)
    running = _harness(session, agent)
    assert wait_for(lambda: agent.prompts, timeout=20), (
        "timed out waiting for the kickoff"
    )
    assert agent.started and agent.prompts[0] == kickoff()

    message_id = AgentControl(session).message("is 01 ok?")
    assert wait_for(answered.is_set, timeout=20), "timed out waiting for the answer"
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


class AsideAgent(ScriptedAgent):
    """The scripted agent with the CLI's side question stood in: it keeps
    each question and answers it."""

    def __init__(self, reply=None):
        super().__init__(reply)
        self.side_questions = []

    async def side_question(self, question, history, timeout_s):
        self.side_questions.append(question)
        return "The ion beam was at 30 pA at the last image."


def test_a_side_question_is_answered_beside_a_long_turn_and_never_reaches_it(
    session,
):
    def notes():
        return [
            e
            for e in session.recorder.buffer.events_since(0)["events"]
            if e["kind"] == "note"
        ]

    async def reply(prompt):
        if prompt != kickoff():
            return None
        # A long turn: it goes on until the side answer is on the record.
        while not any(n.get("actor") != "operator" for n in notes()):
            await asyncio.sleep(0.05)

    agent = AsideAgent(reply)
    running = _harness(session, agent)
    assert wait_for(lambda: agent.prompts, timeout=20), "no kickoff"
    side_id = AgentControl(session).message("/btw what current is the ion beam at?")
    assert wait_for(
        lambda: any(n.get("actor") != "operator" for n in notes()), timeout=20
    ), "no side answer"
    # Its turn sees the answer and ends.
    time.sleep(0.5)
    # An ordinary message after it still reaches the agent, as its next turn.
    message_id = AgentControl(session).message("is 01 ok?")
    assert wait_for(lambda: len(agent.prompts) >= 2, timeout=20), "no second turn"
    time.sleep(2 * POLL_S)
    assert running.stop() == 0

    assert agent.interrupts == 0, "the side question cut the turn short"
    assert agent.side_questions == [
        harness_mod.aside.FRAME + "what current is the ion beam at?"
    ]
    assert len(agent.prompts) == 2
    assert agent.prompts[1].startswith(
        f"The watcher printed:\nMESSAGE id={message_id} from the operator: is 01 ok?"
    )
    for prompt in agent.prompts:
        assert side_id not in prompt and "ion beam" not in prompt
    # Both on the record: the question marked, the answer replying to it.
    question, ordinary = [n for n in notes() if n.get("actor") == "operator"]
    (answer,) = [n for n in notes() if n.get("actor") != "operator"]
    assert question["payload"]["id"] == side_id
    assert question["payload"][harness_mod.aside.MARK] is True
    assert harness_mod.aside.MARK not in ordinary["payload"]
    assert answer["payload"]["reply_to"] == side_id
    assert answer["payload"]["text"] == (
        harness_mod.aside.LEAD + "The ion beam was at 30 pA at the last image."
    )


def test_a_side_question_sent_before_the_agent_started_is_answered_beside_it(
    session,
):
    side_id = AgentControl(session).message("/btw where is the report?")
    agent = AsideAgent()
    running = _harness(session, agent)
    assert wait_for(lambda: agent.side_questions, timeout=20), "not answered"
    time.sleep(2 * POLL_S)
    assert running.stop() == 0
    assert agent.prompts == [kickoff()]
    assert all(side_id not in prompt for prompt in agent.prompts)


def test_an_event_id_after_lines_already_taken_is_the_current_turn_s():
    # The watch says a poll's lines, then the id of its newest event: two
    # items. A turn can take the lines before the id arrives.
    async def main():
        harness = Harness(ScriptedAgent(), lambda: None, keep_cursor=kept.append)
        harness._items, harness._urgent = asyncio.Queue(), asyncio.Event()
        harness._watcher = watcher = object()
        harness._put(watcher, watch_mod.Wake(watch_mod.MESSAGE, "MESSAGE id=ab"))
        assert len(harness._drain()) == 1  # the turn starts with the line
        harness._put(watcher, harness_mod._Cursor("s-5"))
        harness._keep_given_cursor()  # the turn is over
        # With lines still waiting, the id waits behind them.
        harness._put(watcher, watch_mod.Wake(watch_mod.MESSAGE, "MESSAGE id=cd"))
        harness._put(watcher, harness_mod._Cursor("s-6"))
        harness._keep_given_cursor()

    kept = []
    asyncio.run(main())
    assert kept == ["s-5"]


def test_a_resumed_session_hears_what_it_missed_while_it_was_down(session):
    kept = []

    async def reply(prompt):
        return None

    agent = ScriptedAgent(reply)
    running = _harness(session, agent, keep_cursor=kept.append)
    assert wait_for(lambda: agent.prompts, timeout=20), "no kickoff"
    first = AgentControl(session).message("is 01 ok?")
    assert wait_for(lambda: len(agent.prompts) >= 2, timeout=20), "no second turn"
    # The id is kept once the turn carrying the message is over, not as it
    # starts: an id kept before then (the kickoff's) is from before the message.
    (sent,) = [
        e
        for e in session.recorder.buffer.events_since(0)["events"]
        if (e.get("payload") or {}).get("id") == first
    ]
    assert wait_for(
        lambda: any(split_event_id(k)[1] >= sent["seq"] for k in kept), timeout=20
    ), "the message's event id was not kept after its turn"
    assert running.stop() == 0
    assert first in agent.prompts[1]

    # Sent while no agent watched.
    missed = AgentControl(session).message("and 02?")

    again = ScriptedAgent(reply)
    running = _harness(session, again, resumed=True, after=kept[-1])
    assert wait_for(lambda: len(again.prompts) >= 2, timeout=20), (
        "the resumed session did not hear the message it missed"
    )
    assert running.stop() == 0
    assert missed in again.prompts[1]
    assert first not in again.prompts[1], "a message it had already heard came again"


def test_lines_that_arrive_during_a_turn_are_the_next_turn(session, caplog):
    async def reply(prompt):
        if prompt.startswith("You are"):
            # Raising the level is news, but not urgent.
            for level in (AgentLevel.ASSIST, AgentLevel.DRIVE):
                assert AgentControl(session).set_level(level) is None
            # The turn goes on while the watcher hears both.
            while len(_heard(caplog, "LEVEL")) < 2:
                await asyncio.sleep(0.05)

    agent = ScriptedAgent(reply)
    running = _harness(session, agent)
    assert wait_for(lambda: len(agent.prompts) >= 2, timeout=20), (
        "timed out waiting for the second turn"
    )
    time.sleep(2 * POLL_S)
    assert running.stop() == 0

    assert len(agent.prompts) == 2, "both lines came as one turn"
    assert agent.interrupts == 0, "a line that is not urgent interrupted"
    lines = agent.prompts[1].splitlines()
    assert lines[:3] == [
        "The watcher printed:",
        "LEVEL assist (was watch): the operator set the agent level",
        "LEVEL drive (was assist): the operator set the agent level",
    ]


def test_no_turns_while_the_agent_is_off(session, caplog):
    agent = ScriptedAgent()
    running = _harness(session, agent)
    assert wait_for(lambda: agent.prompts, timeout=20), (
        "timed out waiting for the kickoff"
    )
    operator = AgentControl(session)

    assert operator.set_level(AgentLevel.OFF) is None
    assert wait_for(lambda: _heard(caplog, "AGENT OFF"), timeout=20), (
        "timed out waiting for AGENT OFF"
    )
    message_id = operator.message("still there?")
    time.sleep(5 * POLL_S)
    assert len(agent.prompts) == 1, "a turn while the agent was Off"

    assert operator.set_level(AgentLevel.WATCH) is None
    # The message may come with AGENT BACK or as the turn after it.
    assert wait_for(
        lambda: any(f"MESSAGE id={message_id}" in p for p in agent.prompts), timeout=20
    ), "timed out waiting for the message after AGENT BACK"
    assert running.stop() == 0

    back = agent.prompts[1]
    assert watch_mod.AGENT_OFF_LINE in back and watch_mod.AGENT_BACK_LINE in back
    assert harness_mod.texts().reread_level in back


def test_the_agent_answers_a_supervised_run_through_the_harness(session):
    assert AgentControl(session).set_level(AgentLevel.DRIVE) is None
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
    running = _harness(session, agent)
    assert wait_for(ended.is_set, timeout=90), "timed out waiting for the run's end"
    assert running.stop() == 0

    assert session.runner.join(30), "the run did not end"
    assert answered, "the run asked nothing, or the agent never heard it"
    asked = [p for p in lamella.proposals_from(SETUP) if p.id in answered]
    assert all(str(p.current.actor).startswith("agent") for p in asked)
    assert not [p for p in lamella.proposals_from(SETUP) if p.pending]


def test_the_session_is_reported_to_the_app_after_each_turn_and_at_the_end(session):
    async def reply(prompt):
        return TurnEnd(
            cost_usd=0.25,
            session_id="scripted",
            tokens={"input": 1, "output": 2, "cache_read": 300, "cache_write": 4},
            context={"used": 305, "window": 1000},
        )

    agent = ScriptedAgent(reply)
    agent.model = "a-model"
    running = _harness(session, agent, goal="Mill 01 and 02", budget_usd=5.0)
    assert wait_for(lambda: agent.prompts, timeout=20), (
        "timed out waiting for the kickoff"
    )
    assert "Mill 01 and 02" in agent.prompts[0]
    assert wait_for(
        lambda: (AgentControl(session).reported_session() or {}).get("turns") == 1,
        timeout=20,
    ), "timed out waiting for the report"
    reported = AgentControl(session).reported_session()
    assert reported["session_id"] == "scripted"
    assert reported["model"] == "a-model"
    assert reported["goal"] == "Mill 01 and 02"
    assert (reported["budget_usd"], reported["spent_usd"]) == (5.0, 0.25)
    assert reported["tokens"]["cache_read"] == 300
    assert reported["context"] == {"used": 305, "window": 1000}
    assert not reported.get("ended")

    assert running.stop() == 0
    ended = AgentControl(session).reported_session()
    assert ended["ended"] and ended["reason"] == "stopped at fibsem-agent"
    kinds = [e["kind"] for e in session.recorder.buffer.events_since(0)["events"]]
    assert kinds.count("agent_session_started") == 1
    assert kinds.count("agent_session_ended") == 1


def test_a_long_turn_reports_its_tokens_and_the_context_while_it_goes_on(session):
    """The spend is known only as a turn ends, but its tokens and how full
    the context is are reported while it runs: a turn can last an hour."""
    finish = threading.Event()

    async def reply(prompt):
        if len(agent.prompts) > 1:
            return TurnEnd(cost_usd=0.5, session_id="scripted")
        agent.calls.seen(
            "call-1", "a-model", {"input_tokens": 10, "output_tokens": 5}, []
        )
        agent.calls.seen(
            "call-2",
            "a-model",
            {"input_tokens": 20, "cache_read_input_tokens": 4000},
            [],
        )
        while not finish.is_set():
            await asyncio.sleep(0.05)
        agent.calls.end_turn()
        return TurnEnd(
            cost_usd=0.25,
            session_id="scripted",
            tokens={"input": 30, "output": 5, "cache_read": 4000, "cache_write": 0},
            context={"used": 4100, "window": 1000000},
        )

    agent = ScriptedAgent(reply)
    agent.calls = usage.Calls()
    running = _harness(session, agent, budget_usd=5.0, report_every_s=0.1)
    control = AgentControl(session)
    assert wait_for(
        lambda: (
            ((control.reported_session() or {}).get("context") or {}).get("used")
            == 4020
        ),
        timeout=20,
    ), "timed out waiting for a report from inside the turn"
    reported = control.reported_session()
    assert reported["tokens"] == {
        "input": 30,
        "output": 5,
        "cache_read": 4000,
        "cache_write": 0,
    }
    assert reported.get("turns", 0) == 0 and "spent_usd" not in reported

    finish.set()
    assert wait_for(
        lambda: (control.reported_session() or {}).get("turns") == 1, timeout=20
    ), "timed out waiting for the turn's end"
    reported = control.reported_session()
    assert reported["spent_usd"] == 0.25
    assert reported["context"] == {"used": 4100, "window": 1000000}
    assert reported["tokens"]["cache_read"] == 4000
    assert running.stop() == 0


def test_the_harness_reports_the_goal_once_and_leaves_later_ones_to_the_agent():
    class Client:
        def __init__(self):
            self.bodies = []

        def request(self, method, path, json=None, timeout=None):
            self.bodies.append(json)

    harness = Harness(ScriptedAgent(), lambda: None, goal="Mill 01")
    harness._client = client = Client()
    asyncio.run(harness._report())
    asyncio.run(harness._report())
    asyncio.run(harness._report(ended=True))
    assert client.bodies[0]["goal"] == "Mill 01"
    # A goal the operator explicitly gives later is the agent's to report.
    assert all("goal" not in body for body in client.bodies[1:])
    assert "explicitly" in harness_mod.texts().prompt


def test_the_harness_says_what_the_agent_can_do_with_its_first_report():
    class Client:
        def __init__(self):
            self.bodies = []

        def request(self, method, path, json=None, timeout=None):
            self.bodies.append(json)

    about = {"harness": "fibsem-agent", "code": True, "workdir": "/w", "skills": ["a"]}
    harness = Harness(ScriptedAgent(), lambda: None, about=about)
    harness._client = client = Client()
    asyncio.run(harness._report())
    asyncio.run(harness._report())
    assert about.items() <= client.bodies[0].items()
    assert "code" not in client.bodies[1] and "skills" not in client.bodies[1]


def test_a_tool_call_is_one_plain_line(tmp_path):
    step = harness_mod.step
    assert step("Write", {"file_path": str(tmp_path / "a.py")}, tmp_path) == (
        "Wrote",
        "a.py",
    )
    assert step("Edit", {"file_path": "/elsewhere/b.py"}, tmp_path) == (
        "Edited",
        "/elsewhere/b.py",
    )
    assert step("Bash", {"command": "python a.py"}) == ("Ran", "python a.py")
    assert step("Bash", {"command": "python - <<EOF\nprint(1)\nEOF"}) == (
        "Ran",
        "python - <<EOF ...",
    )
    assert step("mcp__fibsem__experiment_fetch_file", {"path": "02/ref.tif"}) == (
        "Fetched",
        "02/ref.tif",
    )
    verb, text = step("mcp__fibsem__stage_move_relative", {"dx": 1e-6})
    assert verb == "Called" and text.startswith("stage_move_relative {")


def test_the_agent_keeps_what_it_prints_and_a_failed_step_names_it():
    pytest.importorskip("claude_agent_sdk")
    from claude_agent_sdk import (
        AssistantMessage,
        TextBlock,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
    )

    from fibsem.ai.agent.harness import SdkAgent, _show

    agent = SdkAgent(harness_mod.build_options(harness_mod.find_checkout()))
    _show(
        AssistantMessage(
            content=[
                TextBlock("Measuring it."),
                ToolUseBlock("t1", "Bash", {"command": "python a.py"}),
            ],
            model="m",
        ),
        agent,
    )
    _show(
        UserMessage(content=[ToolResultBlock("t1", "no module named cv2", True)]),
        agent,
    )
    said, ran, failed = agent.transcript
    assert (said["kind"], said["text"]) == ("said", "Measuring it.")
    assert (ran["verb"], ran["text"]) == ("Ran", "python a.py")
    assert failed["kind"] == "failed" and failed["step"] == ran["seq"]
    assert "cv2" in failed["text"]
    assert failed["took_s"] >= 0


def test_a_step_that_answers_is_done_and_says_how_long_it_took():
    pytest.importorskip("claude_agent_sdk")
    from claude_agent_sdk import (
        AssistantMessage,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
    )

    from fibsem.ai.agent.harness import SdkAgent, _show

    agent = SdkAgent(harness_mod.build_options(harness_mod.find_checkout()))
    printed = []
    agent.on_line = lambda: printed.append(len(agent.transcript))
    call = ToolUseBlock("t1", "mcp__fibsem__milling_run", {"stage": "Rough"})
    _show(AssistantMessage(content=[call], model="m"), agent)
    _show(UserMessage(content=[ToolResultBlock("t1", "done", False)]), agent)
    # An answer to a call it never saw makes no line.
    _show(UserMessage(content=[ToolResultBlock("t9", "done", False)]), agent)
    ran, done = agent.transcript
    assert ran["kind"] == "step" and ran["verb"] == "Called"
    assert done["kind"] == "done" and done["step"] == ran["seq"]
    assert done["took_s"] >= 0 and done["text"] == ""
    assert printed == [1, 2], "each line tells the harness at once"


def test_a_subagent_s_lines_are_kept_under_the_call_that_started_it():
    pytest.importorskip("claude_agent_sdk")
    from claude_agent_sdk import (
        AssistantMessage,
        TextBlock,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
    )

    from fibsem.ai.agent.harness import SdkAgent, _show

    agent = SdkAgent(harness_mod.build_options(harness_mod.find_checkout()))
    asked = {
        "subagent_type": "autolamella-verifier",
        "description": "Check lamella 02's polishing pass",
        "prompt": "...",
    }
    _show(
        AssistantMessage(content=[ToolUseBlock("a1", "Agent", asked)], model="m"),
        agent,
    )
    _show(
        AssistantMessage(
            content=[
                TextBlock("Reading the last image."),
                ToolUseBlock("v1", "mcp__fibsem__item_get", {"name": "02"}),
            ],
            model="m",
            parent_tool_use_id="a1",
        ),
        agent,
    )
    _show(
        UserMessage(
            content=[ToolResultBlock("v1", "ok", False)], parent_tool_use_id="a1"
        ),
        agent,
    )
    answer = [{"type": "text", "text": "Concern: the left edge.\n\nagentId: x1"}]
    _show(UserMessage(content=[ToolResultBlock("a1", answer, False)]), agent)

    call, said, read, read_done, done = agent.transcript
    assert (call["verb"], call["text"]) == ("Asked", asked["description"])
    assert call["subagent"] == "autolamella-verifier" and "run" not in call
    for line in (said, read, read_done):
        assert line["run"] == call["seq"], line
    assert said["kind"] == "said" and read_done["step"] == read["seq"]
    # The run's answer stays on its call's done line, without the CLI's footer.
    assert done["step"] == call["seq"] and "run" not in done
    assert done["text"] == "Concern: the left edge."


def test_a_subagent_s_json_answer_gives_its_card_a_verdict_or_attention():
    from fibsem.ai.agent.harness import _answer

    def said(body):
        return [{"type": "text", "text": body + "\n\nagentId: x1"}]

    no = '{"decision": "polishing pass", "agree": false, "concern": "left edge"}'
    text, verdict = _answer(said(f"Checked it.\n\n```json\n{no}\n```"))
    assert (text, verdict) == ("left edge", {"verdict": "concern"})
    yes = '{"agree": true, "concern": ""}'
    text, verdict = _answer(said(f"The pass is safe.\n```json\n{yes}\n```"))
    assert (text, verdict) == ("The pass is safe.", {"verdict": "agree"})
    looked = '{"cause": "x", "summary": "Move the area and retry.", "attention": true}'
    text, flags = _answer(said(f"Found it.\n```json\n{looked}\n```"))
    assert (text, flags) == ("Move the area and retry.", {"attention": True})
    # Any other answer, a JSON block without agree or attention included,
    # stays as it is.
    assert _answer(said("Six sites.")) == ("Six sites.", {})
    plain = 'Sites:\n```json\n{"sites": [1]}\n```'
    assert _answer(said(plain)) == (plain, {})


def test_the_session_asks_for_the_subagents_words_too():
    pytest.importorskip("claude_agent_sdk")
    options = harness_mod.build_options(harness_mod.find_checkout())
    assert options.forward_subagent_text is True


def test_the_skills_the_agent_reads_are_reported_when_it_reads_one():
    pytest.importorskip("claude_agent_sdk")
    from claude_agent_sdk import AssistantMessage, ToolUseBlock

    from fibsem.ai.agent.harness import SdkAgent, _show

    class Client:
        def __init__(self):
            self.bodies = []

        def request(self, method, path, json=None, timeout=None):
            if path == "/app/agent/session":
                self.bodies.append(json)

    def read(skill):
        call = ToolUseBlock(f"t-{skill}", "Skill", {"skill": skill})
        _show(AssistantMessage(content=[call], model="m"), agent)

    agent = SdkAgent(harness_mod.build_options(harness_mod.find_checkout()))
    harness = Harness(agent, lambda: None)
    harness._client = client = Client()
    read("fibsem-agent")
    asyncio.run(harness._report())
    asyncio.run(harness._report())
    read("fibsem-experiment")
    read("fibsem-agent")
    asyncio.run(harness._report())
    assert client.bodies[0]["skills"] == ["fibsem-agent"]
    assert "skills" not in client.bodies[1], "sent again only when it changes"
    assert client.bodies[2]["skills"] == ["fibsem-agent", "fibsem-experiment"]


def test_the_transcript_is_sent_and_kept_while_the_server_cant_take_it():
    class Client:
        def __init__(self, fail=False):
            self.bodies, self.fail = [], fail

        def request(self, method, path, json=None, timeout=None):
            if self.fail:
                raise RuntimeError("409")
            self.bodies.append((path, json))

    agent = ScriptedAgent()
    agent.transcript = [
        {"seq": 1, "kind": "step", "verb": "Ran", "text": "x", "at": time.monotonic()}
    ]
    harness = Harness(agent, lambda: None)
    harness._client = Client(fail=True)
    asyncio.run(harness._send_transcript())
    assert len(agent.transcript) == 1
    harness._client = client = Client()
    asyncio.run(harness._send_transcript())
    assert agent.transcript == []
    ((path, body),) = client.bodies
    assert path == "/app/agent/transcript" and body["session_id"] == "scripted"
    (entry,) = body["entries"]
    assert entry["verb"] == "Ran" and "at" not in entry and entry["age_s"] >= 0


def test_the_transcript_goes_soon_after_a_line_not_on_the_ten_second_clock(
    monkeypatch,
):
    monkeypatch.setattr(harness_mod, "TRANSCRIPT_SOON_S", 0.01)
    monkeypatch.setattr(harness_mod, "TRANSCRIPT_EVERY_S", 3600)

    class Client:
        def __init__(self):
            self.bodies = []

        def request(self, method, path, json=None, timeout=None):
            self.bodies.append(json)

    class Agent(ScriptedAgent):
        on_line = None
        transcript: list = []

    agent = Agent()
    agent.transcript = []
    harness = Harness(agent, lambda: None)
    harness._client = client = Client()

    async def main():
        sender = asyncio.ensure_future(harness._send_transcript_every())
        await asyncio.sleep(0)
        assert agent.on_line is not None
        agent.transcript.append(
            {
                "seq": 1,
                "kind": "step",
                "verb": "Ran",
                "text": "x",
                "at": time.monotonic(),
            }
        )
        agent.on_line()
        await wait_until(lambda: client.bodies)
        sender.cancel()
        await asyncio.gather(sender, return_exceptions=True)

    async def wait_until(done, timeout=2.0):
        end = time.monotonic() + timeout
        while not done() and time.monotonic() < end:
            await asyncio.sleep(0.01)

    asyncio.run(main())
    ((body),) = client.bodies
    assert [e["seq"] for e in body["entries"]] == [1]
    assert agent.transcript == [] and agent.on_line is None


# --- urgent lines --------------------------------------------------------------


def test_the_urgent_lines_are_the_four_the_operator_chose():
    assert harness_mod.URGENT == {
        # an operator message
        watch_mod.MESSAGE,
        # the level lowered, or control taken back
        watch_mod.LEVEL_LOWERED,
        watch_mod.AGENT_OFF,
        watch_mod.OPERATOR_STOP,
        # a run that stops or fails, or a hardware error
        watch_mod.WORKFLOW_FAILED,
        "task_failed",
        watch_mod.COMMAND_FAILED,
        fibsem_lines.MILL_FAILED,
        # a stop rule firing
        fibsem_lines.STOP_RULE,
    }
    assert harness_mod.is_urgent(watch_mod.NO_END_EVENT_LINE)
    for line in (
        watch_mod.Wake(watch_mod.LEVEL, "LEVEL drive (was watch)"),
        watch_mod.Wake(watch_mod.WORKFLOW_ENDED, "WORKFLOW ENDED (completed)"),
        watch_mod.Wake(watch_mod.WORKFLOW_ENDED, "WORKFLOW ENDED (cancelled)"),
        watch_mod.Wake(watch_mod.PROPOSAL, "PROPOSAL item=01"),
        watch_mod.Wake(harness_mod.ALARM, "WAKE check drift"),
        "MESSAGE id=ab12 from the operator: a plain string is never urgent",
    ):
        assert not harness_mod.is_urgent(line), line


def test_a_lowered_level_or_a_stop_asks_the_agent_to_re_read_its_level():
    for kind in (watch_mod.LEVEL_LOWERED, watch_mod.OPERATOR_STOP):
        prompt = Wakes().add([watch_mod.Wake(kind, "...")])
        assert harness_mod.texts().reread_level in prompt


def test_an_urgent_line_cuts_the_turn_short_and_others_wait(session, caplog):
    operator = AgentControl(session)

    async def reply(prompt):
        if prompt.startswith("You are"):
            # Not urgent: the turn goes on through it.
            assert operator.set_level(AgentLevel.ASSIST) is None
            while not _heard(caplog, "LEVEL assist"):
                await asyncio.sleep(0.05)
            await asyncio.sleep(3 * POLL_S)
            assert agent.interrupts == 0
            operator.message("stop and look at 01")
            return await agent.until_interrupted()

    agent = ScriptedAgent(reply)
    running = _harness(session, agent)
    assert wait_for(lambda: len(agent.prompts) >= 2, timeout=20), (
        "timed out waiting for the turn after the interrupt"
    )
    assert running.stop() == 0

    assert agent.interrupts == 1
    after = agent.prompts[1]
    assert after.startswith(harness_mod.INTERRUPTED + "\n\nThe watcher printed:\n")
    assert "LEVEL assist (was watch)" in after
    assert "from the operator: stop and look at 01" in after


def test_a_stage_move_cut_short_finishes_once_and_lets_go_of_control(session):
    """The operator takes control back during the agent's stage move: the turn
    is cut short and the agent stops waiting, but the move is not undone or
    left half done. It finishes, once, and control is free; the next turn
    says to read where the stage is rather than move again."""
    pytest.importorskip("mcp")
    from fibsem.ai.agent.mcp.sidecar import build_sidecar

    operator = AgentControl(session)
    assert operator.set_level(AgentLevel.DRIVE) is None
    microscope = session.microscope
    stage = microscope.stage
    before = microscope.get_stage_position()
    moving, release = threading.Event(), threading.Event()
    move_relative = stage.move_relative

    def slow_move(delta):
        moving.set()
        assert release.wait(30), "the move was never let go"
        return move_relative(delta)

    stage.move_relative = slow_move
    answers = []

    def call_the_sidecar():
        # Its own process in fibsem-agent: the CLI stops waiting for it, but
        # its request runs on.
        with _http(session) as http:
            sidecar = build_sidecar(http, http.get("/capabilities").json())
            answers.append(
                asyncio.run(sidecar.call_tool("stage_move_relative", {"dx": 2e-6}))
            )

    async def reply(prompt):
        if prompt.startswith("You are"):
            threading.Thread(target=call_the_sidecar, daemon=True).start()
            while not moving.is_set():
                await asyncio.sleep(0.05)
            assert microscope.control.current.by == "agent"
            operator.stop()
            return await agent.until_interrupted()

    agent = ScriptedAgent(reply)
    running = _harness(session, agent)
    try:
        assert wait_for(lambda: len(agent.prompts) >= 2, timeout=20), (
            "timed out waiting for the turn after the stop"
        )
        assert not answers, "the move answered before it was let go"
    finally:
        release.set()
    assert wait_for(lambda: answers, timeout=20), (
        "timed out waiting for the move's answer"
    )
    assert running.stop() == 0
    assert not answers[0].is_error, answers[0]

    after = agent.prompts[1]
    assert after.startswith(harness_mod.INTERRUPTED)
    assert "STOPPED by the operator: Stop the agent; your level is now watch" in after
    assert harness_mod.texts().reread_level in after
    moved = microscope.get_stage_position()
    assert moved.x == pytest.approx(before.x + 2e-6)
    assert moved.y == pytest.approx(before.y)
    control = microscope.control
    assert control.current is None, "the move kept control"
    assert session.agent_auth.level is AgentLevel.WATCH


# --- ending -------------------------------------------------------------------


def test_stopping_interrupts_the_turn_and_closes_the_session(session):
    agent = ScriptedAgent(lambda prompt: agent.until_interrupted())
    running = _harness(session, agent)
    assert wait_for(lambda: agent.prompts, timeout=20), (
        "timed out waiting for the kickoff"
    )

    assert running.stop() == 0
    assert agent.interrupts == 1
    assert agent.closed


def test_a_spent_budget_ends_the_run_at_the_end_of_its_turn(session, caplog):
    async def reply(prompt):
        return TurnEnd(cost_usd=5.01, session_id="scripted")

    agent = ScriptedAgent(reply)
    running = _harness(session, agent, budget_usd=5.0)
    assert running.ended() == 2
    assert agent.closed and len(agent.prompts) == 1
    assert running.harness.outcome()["reason"] == "its budget is spent"
    assert any("The budget is spent ($5.01 of $5 " in m for m in caplog.messages)
    ended = AgentControl(session).reported_session()
    assert ended["ended"] and ended["reason"] == "its budget is spent"
    assert (ended["budget_usd"], ended["spent_usd"]) == (5.0, 5.01)


def test_a_budget_the_operator_raises_keeps_the_run_going(session):
    session.agent_auth.operator_token = "op-token"
    spent = iter([1.0, 4.5, 5.5, 12.0])

    async def reply(prompt):
        cost = next(spent)
        if cost == 5.5:
            # The operator raises it while the turn that spends it goes on.
            with httpx.Client(
                base_url=session.app_server.url,
                headers={"Authorization": "Bearer op-token"},
                timeout=30,
            ) as http:
                raised = http.put("/app/agent/budget", json={"budget_usd": 10})
            assert raised.status_code == 200, raised.text
        return TurnEnd(cost_usd=cost, session_id="scripted")

    agent = ScriptedAgent(reply)
    running = _harness(session, agent, budget_usd=5.0)
    control = AgentControl(session)
    for turn in range(1, 4):
        assert wait_for(lambda: len(agent.prompts) == turn, timeout=20)
        control.message(f"carry on {turn}")
    assert running.ended() == 2
    assert len(agent.prompts) == 4
    # Said once, in the turn after the spend first reached 80% of the budget.
    said = [harness_mod.BUDGET_LOW[:8] in prompt for prompt in agent.prompts]
    assert said == [False, False, True, False]
    assert "$4.50 of this run's $5 is spent (90%)" in agent.prompts[2]
    ended = control.reported_session()
    assert (ended["budget_usd"], ended["budget_set_usd"]) == (10.0, 10.0)


def test_a_resumed_run_counts_its_own_spend_against_the_budget():
    def resumed(before=None):
        return Harness(
            ScriptedAgent(),
            lambda: None,
            resumed=True,
            budget_usd=5.0,
            spent_before_usd=before,
        )

    # The CLI's running total carries the earlier runs on.
    harness = resumed(40.0)
    harness._count_spend(44.0)
    assert harness._run_spent() == 4.0 and not harness._over_budget()
    harness._count_spend(45.5)
    assert harness._over_budget()
    # A total that did not carry them on is all this run's.
    harness = resumed(40.0)
    harness._count_spend(3.0)
    assert harness._run_spent() == 3.0
    # Not known: the run's first turn is not counted.
    harness = resumed()
    harness._count_spend(44.0)
    harness._count_spend(46.0)
    assert harness._run_spent() == 2.0
    record = {"runs": [{"spent_usd": 12.5}, {"spent_usd": 40.0}, {}]}
    assert harness_mod._spent_before(record) == 40.0
    assert harness_mod._spent_before({"runs": [{"spent_usd": 3.0}]}) is None


def test_a_model_that_cannot_sign_in_ends_the_run(session):
    # Not waiting for a wake-up the model could never answer.
    async def reply(prompt):
        return TurnEnd(error="authentication_failed")

    agent = ScriptedAgent(reply)
    assert _harness(session, agent).ended() == 1
    assert agent.closed and len(agent.prompts) == 1


def test_a_turn_with_no_result_ends_the_run(session):
    # The CLI died: the next wake-up would find no one to answer.
    async def reply(prompt):
        return TurnEnd("no_result")

    agent = ScriptedAgent(reply)
    assert _harness(session, agent).ended() == 1
    assert agent.closed


def test_a_cli_that_dies_on_the_ctrl_c_still_stops_cleanly(session):
    # A Ctrl-C at the terminal reaches the CLI too, which may end mid-turn.
    async def reply(prompt):
        await agent.until_interrupted()
        raise RuntimeError("the CLI exited")

    agent = ScriptedAgent(reply)
    running = _harness(session, agent)
    assert wait_for(lambda: agent.prompts, timeout=20), (
        "timed out waiting for the kickoff"
    )
    assert running.stop() == 0
    assert agent.closed


def test_the_kickoff_waits_while_the_agent_is_off(session, caplog):
    operator = AgentControl(session)
    assert operator.set_level(AgentLevel.OFF) is None
    agent = ScriptedAgent()
    running = _harness(session, agent)
    assert wait_for(lambda: _heard(caplog, "AGENT OFF"), timeout=20), (
        "timed out waiting for AGENT OFF"
    )
    time.sleep(5 * POLL_S)
    assert not agent.prompts, "a kickoff the server would refuse"

    assert operator.set_level(AgentLevel.WATCH) is None
    assert wait_for(lambda: agent.prompts, timeout=20), (
        "timed out waiting for the kickoff"
    )
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
        Harness(
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
    running = Running(Harness(agent, lambda: None, server_wait_s=0.3, retry_s=0.05))
    assert running.ended() == 1
    assert not agent.started and not agent.prompts


def test_a_server_of_another_api_version_ends_the_agent_at_once(caplog):
    class OtherVersion:
        closed = False

        def request(self, method, path, **kwargs):
            return SimpleNamespace(
                status_code=200, json=lambda: {"api_version": "99.0.0"}
            )

        def close(self):
            self.closed = True

    client, agent = OtherVersion(), ScriptedAgent()
    # A long wait: it ends on the version, not on the deadline.
    running = Running(Harness(agent, lambda: client, server_wait_s=600, retry_s=0.05))
    assert running.ended(timeout=10.0) == 1
    assert not agent.started and client.closed
    assert any("routes are version 99.0.0" in r.getMessage() for r in caplog.records)
    assert running.harness._resumable is False


def test_compacting_shows_as_a_running_step_until_it_ends():
    """The CLI's "compacting" status opens a step, so the page shows the
    agent at work rather than frozen; the boundary closes it and says so."""
    pytest.importorskip("claude_agent_sdk")
    from claude_agent_sdk import SystemMessage

    from fibsem.ai.agent.harness import SdkAgent, _show

    agent = SdkAgent(harness_mod.build_options(harness_mod.find_checkout()))
    status = SystemMessage(subtype="status", data={"status": "compacting"})
    _show(status, agent)
    _show(status, agent)  # said twice, one step
    (step,) = agent.transcript
    assert (step["kind"], step["verb"], step["text"]) == (
        "step",
        "Compacted",
        "the context",
    )
    _show(
        SystemMessage(
            subtype="compact_boundary",
            data={"compact_metadata": {"trigger": "auto", "pre_tokens": 812000}},
        ),
        agent,
    )
    _show(SystemMessage(subtype="status", data={"status": None}), agent)
    done, said = agent.transcript[1:]
    assert done["kind"] == "done" and done["step"] == step["seq"]
    assert said["kind"] == "context"
    assert said["text"] == "Context compacted: 812k tokens before, automatically"
    assert agent._compacting is None


def test_a_compacted_context_asks_for_the_plan_on_the_next_turn():
    pytest.importorskip("claude_agent_sdk")
    from claude_agent_sdk import SystemMessage

    from fibsem.ai.agent.harness import SdkAgent, _show

    agent = SdkAgent(harness_mod.build_options(harness_mod.find_checkout()))
    assert not agent.compacted
    _show(
        SystemMessage(
            subtype="compact_boundary",
            data={"compact_metadata": {"trigger": "auto", "pre_tokens": 900000}},
        ),
        agent,
    )
    assert agent.compacted

    class Turns(ScriptedAgent):
        compacted = True

    scripted = Turns()
    harness = Harness(scripted, lambda: None)

    async def main():
        harness._stopping = asyncio.Event()
        harness._report_read = lambda: asyncio.sleep(0)
        harness._send_transcript_every = lambda: asyncio.sleep(3600)
        await scripted.start()
        await harness._turn("The watcher printed:\nEVENT x")
        await harness._turn("The watcher printed:\nEVENT y")

    asyncio.run(main())
    first, second = scripted.prompts
    assert first.endswith(
        "Your context was compacted. " + harness_mod.texts().reread_plan
    )
    assert "compacted" not in second


def test_a_lab_tool_s_picture_goes_on_the_transcript_under_its_step():
    pytest.importorskip("claude_agent_sdk")
    from claude_agent_sdk import (
        AssistantMessage,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
    )

    from fibsem.ai.agent.harness import (
        PICTURES_WAITING,
        SdkAgent,
        _one_picture,
        _show,
    )

    agent = SdkAgent(harness_mod.build_options(harness_mod.find_checkout()))
    image = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw=="},
    }
    for call, tool in (("t1", "mcp__notebook__read_page"), ("t2", "mcp__fibsem__x")):
        _show(
            AssistantMessage(content=[ToolUseBlock(call, tool, {})], model="m"), agent
        )
        result = [{"type": "text", "text": "a page"}, image]
        _show(UserMessage(content=[ToolResultBlock(call, result, False)]), agent)
    lab, picture = agent.transcript[0], agent.transcript[2]
    # The lab's picture follows its step's done line; ours shows none.
    assert [line["kind"] for line in agent.transcript] == [
        "step",
        "done",
        "picture",
        "step",
        "done",
    ]
    assert picture["step"] == lab["seq"]
    assert (picture["media_type"], picture["image"]) == ("image/png", "iVBORw==")

    # An image as MCP gives it, a kind the server won't keep, and one too many.
    agent.transcript.clear()
    mcp = {"type": "image", "data": "R0lG", "mimeType": "image/gif"}
    svg = {"type": "image", "data": "PHN2Zz4=", "mimeType": "image/svg+xml"}
    for n in range(PICTURES_WAITING + 1):
        call = f"m{n}"
        _show(
            AssistantMessage(
                content=[ToolUseBlock(call, "mcp__notebook__plot", {})], model="m"
            ),
            agent,
        )
        _show(UserMessage(content=[ToolResultBlock(call, [mcp, svg], False)]), agent)
    pictures = [line for line in agent.transcript if line["kind"] == "picture"]
    assert len(pictures) == PICTURES_WAITING
    assert {line["media_type"] for line in pictures} == {"image/gif"}
    # A send carries one picture at most.
    sent = _one_picture(agent.transcript)
    assert sum(line["kind"] == "picture" for line in sent) == 1


# --- what wakes the agent (wake_on) ---------------------------------------------


def test_at_watch_routine_lines_wait_for_a_line_that_wakes_the_agent(session, caplog):
    """The real watch on the real App Server: at Watch, under digest, a task's
    start and end wait, and come at the top of the turn an operator's message
    starts. Raised to Drive, every line wakes the agent again."""
    policy = wake_policy.WakeOn(
        wake_policy.Settings(watch="digest", every_min=60, after_lines=0)
    )
    agent = ScriptedAgent()
    running = _harness(session, agent, wake_on=policy)
    assert wait_for(lambda: agent.prompts, timeout=20), "no kickoff"
    # The level, read as the watch started.
    assert policy.level == "watch" and policy.mode == "digest"

    buffer = session.recorder.buffer
    buffer.append("task_started", {"item_name": "01-a", "task_name": "Setup"})
    buffer.append("task_completed", {"item_name": "01-a", "task_name": "Setup"})
    assert wait_for(lambda: len(_heard(caplog, "EVENT")) >= 2, timeout=20)
    time.sleep(4 * POLL_S)
    assert len(agent.prompts) == 1, "a routine line woke the agent at Watch"
    assert len(policy.held) == 2

    message_id = AgentControl(session).message("how is it going?")
    assert wait_for(lambda: len(agent.prompts) >= 2, timeout=20), "no turn"
    digest, lines = agent.prompts[1].split("\n\nThe watcher printed:\n")
    assert digest.splitlines()[1:] == [
        "EVENT task_started 01-a Setup",
        "EVENT task_completed 01-a Setup",
    ]
    assert lines.startswith(f"MESSAGE id={message_id} from the operator")

    assert AgentControl(session).set_level(AgentLevel.DRIVE) is None
    assert wait_for(lambda: len(agent.prompts) >= 3, timeout=20), "no level turn"
    assert (
        "What wakes you now: wake_on all, the default at drive." in (agent.prompts[2])
    )
    buffer.append("task_started", {"item_name": "02-b", "task_name": "Setup"})
    assert wait_for(lambda: len(agent.prompts) >= 4, timeout=20), "no task turn"
    assert agent.prompts[3] == "The watcher printed:\nEVENT task_started 02-b Setup"
    assert running.stop() == 0
