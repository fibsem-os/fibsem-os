"""``fibsem-agent``: an AutoLamella agent with nobody at a terminal.

The Claude Agent SDK runs one long session with the ``fibsem-mcp`` sidecar as
its only MCP server, and the runner runs ``fibsem-mcp watch`` itself: each
batch of the watcher's lines is the agent's next turn. The App Server still
decides what the agent may do, and the runner never sets the level. Needs the
[agent] extra; nothing else in fibsem imports ``claude_agent_sdk``.

    fibsem-agent
    fibsem-agent --resume <session id>
"""

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import signal
import sys
import sysconfig
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from fibsem.ai.agent.levels import LEVEL_SKILLS
from fibsem.ai.agent.mcp import watch as watch_mod
from fibsem.util import timestamps

BASE_SKILL = "autolamella-agent"
SKILLS = (BASE_SKILL,) + tuple(LEVEL_SKILLS.values())

SERVER = "fibsem"
# The server's level gate decides what the fibsem tools may do, so they need no
# prompt; ToolSearch only finds them.
ALLOWED_TOOLS = [f"mcp__{SERVER}__*", "ToolSearch"]
# The only built-in tools the agent has: reading its skills, and finding the
# sidecar's tools when the CLI defers them.
BUILT_IN_TOOLS = ["Skill", "ToolSearch"]
# Named as well, so a CLI that adds one back still refuses it.
DENIED_TOOLS = [
    "Bash",
    "Read",
    "Write",
    "Edit",
    "MultiEdit",
    "NotebookEdit",
    "WebFetch",
    "WebSearch",
    "Agent",
]
# Nothing waits on a permission prompt: what is not allowed above is denied.
PERMISSION_MODE = "dontAsk"

# The model can't answer until someone fixes the sign-in or the account, so
# waiting for the next wake-up would wait for ever.
FATAL_ERRORS = {"authentication_failed", "billing_error"}

# The CLI's environment: no CLAUDE.md (the checkout's is developer guidance)
# and no auto memory from a developer's sessions in the same folder.
CLI_ENV = {
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
}

# What never goes in the log: a bearer token or an Anthropic API key.
SECRETS = re.compile(r"(Bearer\s+)\S+|sk-ant-[\w-]+", re.IGNORECASE)

MAX_TURNS = 40  # tool calls per wake-up
MAX_BUDGET_USD = 5.0  # what one run spends
SERVER_WAIT_S = 600.0
RETRY_MAX_S = 60.0

RUNNER_PROMPT = """\
You are running under fibsem-agent, a program on the microscope PC, not an \
interactive Claude Code session. Nobody is at a terminal.

- fibsem-agent runs `fibsem-mcp watch` for you. Each batch of its lines \
arrives as your next message, starting "The watcher printed:". Don't start a \
watcher, don't loop app_get_events, and don't wait inside a turn: do what the \
lines call for, then end your turn.
- Your tools are the mcp__fibsem__* tools and Skill. There is no shell, no \
file access and no web. Where a skill shows curl, python or a script, use the \
matching mcp__fibsem__ tool, or say what you can't do.
- Your replies go to fibsem-agent's console and log, which the operator may \
not be reading. Anything the operator must see goes on the record with \
experiment_add_note; answer a MESSAGE with its reply_to.
- The operator sets your level in the app. Work by the skill for your level: \
observe-autolamella at Watch, assist-autolamella at Assist, \
supervise-autolamella at Drive. A LEVEL line means it changed: read that \
level's skill if you haven't. A refusal is the server's answer: handle it as \
the skills say."""

REREAD_PROPOSALS = (
    "Events were lost: re-read proposal_list and app_get_status before acting "
    "on the lines above."
)
REREAD_LEVEL = (
    "Your level may have changed: re-read server_get_capabilities and work by "
    "the skill for the level it gives, then re-read app_get_status and "
    "proposal_list."
)

log = logging.getLogger("fibsem.agent.runner")


def kickoff(resumed: bool = False) -> str:
    """The first turn: the skills, the level, and where things stand."""
    if resumed:
        return (
            "fibsem-agent restarted and resumed this session. Its watcher started "
            "again from now, so whatever happened while it was down is not in its "
            "lines. Call server_get_capabilities, read the skill for your level "
            f"again if it is no longer in your context ({BASE_SKILL} first), tell "
            "the operator in a sentence what your level means now, re-read "
            "app_get_status and proposal_list, then end your turn."
        )
    return (
        "You are the AutoLamella agent, run by fibsem-agent.\n"
        f"1. Read the {BASE_SKILL} skill with the Skill tool.\n"
        "2. Call server_get_capabilities, then read the skill for the level it "
        "gives. Tell the operator in a sentence what your level means for this "
        "session and whether the microscope is served. If an experiment is "
        "open, put that sentence on the record with experiment_add_note too, so "
        "the operator sees it in the app. The level is the operator's.\n"
        "3. Read where things stand, as the base skill's Session setup says. "
        "The watcher is already running: don't start one.\n"
        "Then end your turn. The runner wakes you with the watcher's lines."
    )


def turn_prompt(lines: List[str]) -> str:
    """The watcher's lines as one turn, with what a GAP or a level change
    asks of the agent first."""
    prompt = "The watcher printed:\n" + "\n".join(lines)
    if any(line.startswith("GAP") for line in lines):
        prompt += "\n\n" + REREAD_PROPOSALS
    if any(line.startswith(("AGENT BACK", "RESTARTED", "LEVEL")) for line in lines):
        prompt += "\n\n" + REREAD_LEVEL
    return prompt


class Wakes:
    """The watcher's lines, held while the operator has the agent at Off and
    given to the agent as one turn otherwise."""

    def __init__(self) -> None:
        self.off = False
        self._held: List[str] = []

    def add(self, lines: List[str]) -> Optional[str]:
        """The next turn's prompt, or None when there is nothing to say yet."""
        for line in lines:
            if line.startswith("AGENT OFF"):
                self.off = True
            elif line.startswith("AGENT BACK"):
                if not self.off:
                    # The level came back before the agent heard it go.
                    continue
                self.off = False
            self._held.append(line)
        if self.off or not self._held:
            return None
        lines, self._held = self._held, []
        return turn_prompt(lines)


@dataclass
class TurnEnd:
    """How one turn ended: the SDK's result subtype and what it reported."""

    subtype: str = "success"
    cost_usd: Optional[float] = None
    session_id: Optional[str] = None
    # Why the model did not answer (the SDK's AssistantMessage.error), if so.
    error: Optional[str] = None


class _Stopped(Exception):
    """The runner is stopping: raised inside ``watch()`` to end it."""


class _Seeded:
    """The watch has read where the event stream stands: from here on, no
    line is missed."""


@dataclass
class _WatchEnded:
    code: int


# What the watch prints just before it gives up.
LAST_LINES = ("SERVER GONE", "SERVER UNAVAILABLE", "NO EVENTS")


def _last_line(batch) -> bool:
    return any(isinstance(item, str) and item.startswith(LAST_LINES) for item in batch)


def _has_events(response) -> bool:
    if response.status_code != 200:
        return False
    try:
        return bool(response.json().get("available"))
    except (ValueError, AttributeError):
        return False


class _WatchClient:
    """The watch's client: it refuses another request once stopped, and
    calls ``on_seeded`` when the event stream first answers."""

    def __init__(self, client, stopping: threading.Event, on_seeded) -> None:
        self._client = client
        self._stopping = stopping
        self._on_seeded = on_seeded
        self._seeded = False

    def request(self, method, path, *args, **kwargs):
        if self._stopping.is_set():
            raise _Stopped
        response = self._client.request(method, path, *args, **kwargs)
        if not self._seeded and path.startswith("/app/events"):
            if _has_events(response):
                self._seeded = True
                self._on_seeded()
        return response


class Watcher:
    """``watch()`` on a daemon thread. ``deliver`` gets each line, a
    ``_Seeded`` once the watch holds its place, and a ``_WatchEnded`` when it
    gives up."""

    def __init__(
        self,
        client,
        deliver: Callable[[object], None],
        poll_s: float = watch_mod.POLL_S,
        retry_s: Optional[float] = None,
    ) -> None:
        self._client = client
        self._deliver = deliver
        self._poll_s = poll_s
        self._retry_s = retry_s
        self._stopping = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="fibsem-agent-watch", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Ends the watch at its next request; a parked poll is not waited for."""
        self._stopping.set()

    def _out(self, line: str) -> None:
        if self._stopping.is_set():
            raise _Stopped
        self._deliver(line)
        log.info("watch: %s", line)

    def _sleep(self, seconds: float) -> None:
        if self._retry_s is not None:
            seconds = min(seconds, self._retry_s)
        self._stopping.wait(seconds)

    def _run(self) -> None:
        client = _WatchClient(
            self._client, self._stopping, lambda: self._deliver(_Seeded())
        )
        try:
            code = watch_mod.watch(
                client, out=self._out, sleep_fn=self._sleep, poll_s=self._poll_s
            )
        except _Stopped:
            return
        except Exception:
            log.exception("The watcher failed")
            code = 1
        if not self._stopping.is_set():
            try:
                self._deliver(_WatchEnded(code))
            except _Stopped:
                pass


def _reachable(client) -> bool:
    """Whether the App Server answers this client, at any level."""
    try:
        response = client.request("GET", "/capabilities", timeout=10.0)
    except Exception:
        return False
    return response.status_code != 401 and response.status_code < 500


class Runner:
    """The wake loop: the kickoff, then one turn per batch of watcher lines,
    until the operator stops it, the budget is spent or the app is gone.

    ``agent`` is the seam: ``start()``, ``turn(prompt) -> TurnEnd``,
    ``interrupt()`` and ``close()``. ``connect`` gives the watch's client,
    or None while no server is known. ``run()`` returns the exit code: 0
    when stopped, 1 when the app or the agent is gone, 2 when the budget is.
    """

    def __init__(
        self,
        agent,
        connect: Callable[[], Optional[object]],
        resumed: bool = False,
        server_wait_s: float = SERVER_WAIT_S,
        poll_s: float = watch_mod.POLL_S,
        retry_s: Optional[float] = None,
    ) -> None:
        self._agent = agent
        self._connect = connect
        self._resumed = resumed
        self._server_wait_s = server_wait_s
        self._poll_s = poll_s
        self._retry_s = retry_s
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stopping: Optional[asyncio.Event] = None
        # (watch, item): an item from a watch that has been replaced is dropped.
        self._items: Optional[asyncio.Queue] = None
        self._watcher: Optional[Watcher] = None
        self._wakes = Wakes()
        self._code = 0
        self._stop_requested = False

    def stop(self) -> None:
        """Interrupt the agent's turn and end the run. Safe from a signal
        handler or another thread."""
        self._stop_requested = True
        if self._loop is not None and self._stopping is not None:
            try:
                self._loop.call_soon_threadsafe(self._stopping.set)
            except RuntimeError:  # the loop has closed
                pass

    async def run(self) -> int:
        self._loop = asyncio.get_running_loop()
        self._stopping = asyncio.Event()
        self._items = asyncio.Queue()
        if self._stop_requested:
            return 0
        # The kickoff's own actions are news: the watch starts at the tail, so
        # it has to hold its place before the agent acts.
        if not await self._watch():
            return self._code
        try:
            await self._agent.start()
            prompt = kickoff(self._resumed)
            while prompt is not None and await self._turn(prompt):
                prompt = await self._next_prompt()
        except Exception:
            # Ctrl-C reaches the CLI as well, which may end before its turn.
            if not self._stopping.is_set():
                raise
            log.debug("The agent's process ended as it stopped.", exc_info=True)
        finally:
            if self._watcher is not None:
                self._watcher.stop()
            try:
                await self._agent.close()
            except Exception:
                log.debug("Closing the session failed.", exc_info=True)
        return self._code

    def _start_watcher(self, client) -> None:
        loop, items = self._loop, self._items
        if self._watcher is not None:
            self._watcher.stop()
        watcher = None

        def deliver(item):
            try:
                loop.call_soon_threadsafe(items.put_nowait, (watcher, item))
            except RuntimeError:  # the loop has closed
                raise _Stopped from None

        watcher = Watcher(client, deliver, poll_s=self._poll_s, retry_s=self._retry_s)
        self._watcher = watcher
        watcher.start()

    async def _watch(self) -> bool:
        """Watch the app until the watch holds its place: True then, False
        when that hasn't happened within ``server_wait_s`` or the run was
        stopped. While the level is Off it waits for the level, not a time."""
        deadline = time.monotonic() + self._server_wait_s
        delay = self._retry_s or watch_mod.RETRY_DELAY_S
        while True:
            client = await self._wait_for_server(deadline)
            if client is None:
                return False
            self._start_watcher(client)
            while True:
                item = await self._get()
                if item is None:
                    return False
                if isinstance(item, _Seeded):
                    return True
                if isinstance(item, _WatchEnded):
                    break
            # The server answers, but its events can't be read yet.
            self._watcher.stop()
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not await self._pause(min(delay, remaining)):
                if not self._stopping.is_set():
                    log.error(
                        "The App Server's events could not be read within "
                        "%.0f s; fibsem-agent stops.",
                        self._server_wait_s,
                    )
                    self._code = 1
                return False
            delay = min(delay * 2, RETRY_MAX_S)

    async def _turn(self, prompt: str) -> bool:
        """One turn, interrupted if the run is stopped; False when the run is
        over."""
        if self._stopping.is_set():
            return False
        interrupter = asyncio.ensure_future(self._interrupt_on_stop())
        try:
            end = await self._agent.turn(prompt)
        finally:
            interrupter.cancel()
        if self._stopping.is_set():
            return False
        session = end.session_id or getattr(self._agent, "session_id", None)
        if end.error in FATAL_ERRORS:
            log.error(
                "The model can't answer (%s); fibsem-agent stops. Set "
                "ANTHROPIC_API_KEY to a Claude API key (or configure Bedrock or "
                "Vertex), then start it again.",
                end.error,
            )
            self._code = 1
            return False
        if end.subtype == "error_max_budget_usd":
            log.error(
                "The budget is spent (--max-budget-usd); fibsem-agent stops. "
                "Raise it and --resume %s to carry on.",
                session,
            )
            self._code = 2
            return False
        if end.subtype == "no_result":
            log.error(
                "The agent's process ended without finishing its turn; "
                "fibsem-agent stops. --resume %s carries the session on.",
                session,
            )
            self._code = 1
            return False
        if end.subtype == "error_max_turns":
            log.warning(
                "The turn hit --max-turns before it finished; the next wake-up "
                "starts a new one."
            )
        elif end.subtype != "success":
            log.warning("The turn ended with %s.", end.subtype)
        return True

    async def _interrupt_on_stop(self) -> None:
        await self._stopping.wait()
        log.info("Stopping: interrupting the agent's turn.")
        try:
            await self._agent.interrupt()
        except Exception:
            # The CLI may have had the Ctrl-C too, and be gone.
            log.debug("Interrupting the turn failed.", exc_info=True)

    async def _pause(self, seconds: float) -> bool:
        """Wait, unless the run is stopped; False when it was."""
        try:
            await asyncio.wait_for(self._stopping.wait(), timeout=max(seconds, 0))
        except asyncio.TimeoutError:
            return True
        return False

    async def _get(self):
        """The current watch's next item, or None once the run is stopped."""
        while not self._stopping.is_set():
            get = asyncio.ensure_future(self._items.get())
            stop = asyncio.ensure_future(self._stopping.wait())
            done, _ = await asyncio.wait(
                {get, stop}, return_when=asyncio.FIRST_COMPLETED
            )
            stop.cancel()
            if get not in done:
                get.cancel()
                return None
            watcher, item = get.result()
            if watcher is self._watcher:
                return item
        return None

    def _drain(self) -> list:
        items = []
        while not self._items.empty():
            watcher, item = self._items.get_nowait()
            if watcher is self._watcher:
                items.append(item)
        return items

    async def _next_prompt(self) -> Optional[str]:
        """The next turn: every line that arrived since the last one, held
        while the agent is Off. None when the run is over."""
        while True:
            item = await self._get()
            if item is None:
                return None
            batch = [item] + self._drain()
            ended = any(isinstance(line, _WatchEnded) for line in batch)
            if not ended and _last_line(batch):
                # Its end comes right after: one turn, not two.
                item = await self._get()
                if item is None:
                    return None
                batch.append(item)
                ended = isinstance(item, _WatchEnded)
            lines = [line for line in batch if isinstance(line, str)]
            prompt = self._wakes.add(lines)
            if ended:
                prompt = await self._rewatch(prompt)
                if prompt is None:
                    return None
            if prompt is not None:
                return prompt

    async def _rewatch(self, prompt: Optional[str]) -> Optional[str]:
        """The watch gave up: wait for the app to come back and watch again,
        or end the run. The turn says what the agent missed."""
        self._watcher.stop()
        log.warning("The watcher stopped: the app can't be watched.")
        started = time.monotonic()
        if not await self._watch():
            return None
        # A restarted app starts its own session, at Watch.
        self._wakes = Wakes()
        back = (
            f"The app could not be watched for {time.monotonic() - started:.0f} s "
            "and the runner is watching it again: events in between were not "
            "seen. " + REREAD_LEVEL
        )
        return f"{prompt}\n\n{back}" if prompt else back

    async def _wait_for_server(self, deadline: float):
        """The watch's client once the App Server answers; None when it has
        not by ``deadline`` or the run was stopped."""
        delay = self._retry_s or watch_mod.RETRY_DELAY_S
        said = False
        while not self._stopping.is_set():
            client = self._connect()
            if client is not None and await asyncio.get_running_loop().run_in_executor(
                None, _reachable, client
            ):
                if said:
                    log.info("The App Server answers.")
                return client
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                log.error(
                    "No App Server answered within %.0f s; fibsem-agent stops. "
                    "Start it again once the app's App Server is on.",
                    self._server_wait_s,
                )
                self._code = 1
                return None
            if not said:
                log.info(
                    "Waiting up to %.0f s for the App Server (Tools -> App "
                    "Server in AutoLamella).",
                    remaining,
                )
                said = True
            if not await self._pause(min(delay, remaining)):
                return None
            delay = min(delay * 2, RETRY_MAX_S)
        return None


class SdkAgent:
    """The agent: one ``ClaudeSDKClient`` session, its messages logged."""

    def __init__(self, options) -> None:
        self._options = options
        self._client = None
        self.session_id: Optional[str] = options.resume
        self.introduced = False

    async def start(self) -> None:
        from claude_agent_sdk import ClaudeSDKClient

        self._client = ClaudeSDKClient(self._options)
        await self._client.connect()

    async def turn(self, prompt: str) -> TurnEnd:
        from claude_agent_sdk import AssistantMessage, ResultMessage

        await self._client.query(prompt)
        end, error = TurnEnd(subtype="no_result"), None
        async for message in self._client.receive_response():
            _show(message, self)
            if isinstance(message, AssistantMessage) and message.error:
                error = message.error
            if isinstance(message, ResultMessage):
                end = TurnEnd(
                    message.subtype, message.total_cost_usd, message.session_id
                )
        end.error = error
        return end

    async def interrupt(self) -> None:
        if self._client is not None:
            await self._client.interrupt()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.disconnect()
            self._client = None
        if self.session_id:
            log.info(
                "Session %s: --resume %s carries it on.",
                self.session_id,
                self.session_id,
            )


def _short(name: str) -> str:
    return name.replace(f"mcp__{SERVER}__", "")


def _clip(text: str, limit: int = 300) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _result_text(content) -> str:
    if isinstance(content, list):
        parts = [
            item.get("text", f"[{item.get('type')}]")
            for item in content
            if isinstance(item, dict)
        ]
        return " ".join(parts)
    return str(content or "")


def _show(message, agent: SdkAgent) -> None:
    """One SDK message on the console and in the log, readably."""
    from claude_agent_sdk import (
        AssistantMessage,
        ResultMessage,
        SystemMessage,
        TextBlock,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
    )

    if isinstance(message, SystemMessage):
        data = message.data or {}
        # Every turn starts with one; the first says it all.
        if message.subtype == "init" and not agent.introduced:
            agent.introduced = True
            agent.session_id = data.get("session_id") or agent.session_id
            log.info("Session %s", agent.session_id)
            tools = data.get("tools") or []
            ours = [tool for tool in tools if tool.startswith(f"mcp__{SERVER}__")]
            others = [tool for tool in tools if tool not in ours]
            log.info(
                "Tools: %d from %s, and %s. Model %s.",
                len(ours),
                SERVER,
                ", ".join(others) or "nothing else",
                data.get("model"),
            )
            for server in data.get("mcp_servers") or []:
                if server.get("status") != "connected":
                    log.warning(
                        "MCP server %s is %s.", server.get("name"), server.get("status")
                    )
        elif message.subtype == "compact_boundary":
            meta = data.get("compact_metadata") or {}
            log.info(
                "Context compacted (%s, %s tokens before).",
                meta.get("trigger"),
                meta.get("pre_tokens"),
            )
    elif isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, TextBlock) and block.text.strip():
                log.info("agent: %s", block.text.strip())
            elif isinstance(block, ToolUseBlock):
                arguments = json.dumps(block.input, ensure_ascii=False)
                log.info("tool: %s %s", _short(block.name), _clip(arguments, 200))
    elif isinstance(message, UserMessage) and isinstance(message.content, list):
        for block in message.content:
            if isinstance(block, ToolResultBlock) and block.is_error:
                log.warning("tool refused: %s", _clip(_result_text(block.content)))
    elif isinstance(message, ResultMessage):
        cost = message.total_cost_usd
        log.info(
            "Turn ended: %s, %d steps%s.",
            message.subtype,
            message.num_turns,
            f", ${cost:.4f} so far" if cost is not None else "",
        )


def find_checkout(given: Optional[str] = None) -> Optional[Path]:
    """The fibsem-os checkout whose ``.claude/skills`` the agent reads: the
    one given, else the one this module runs from, else the current folder."""
    if given:
        candidates = [Path(given).expanduser()]
    else:
        candidates = [Path(__file__).resolve().parents[3], Path.cwd()]
    for candidate in candidates:
        if (candidate / ".claude" / "skills" / BASE_SKILL / "SKILL.md").is_file():
            return candidate.resolve()
    return None


def sidecar_command() -> List[str]:
    """``fibsem-mcp`` of this environment, by absolute path. Failing that,
    ``python -I -m``, which keeps the current folder off the path, so another
    checkout there cannot shadow the installed fibsem."""
    scripts = sysconfig.get_path("scripts")
    found = shutil.which("fibsem-mcp", path=scripts) if scripts else None
    if found:
        return [str(Path(found).resolve())]
    return [sys.executable, "-I", "-m", "fibsem.ai.agent.mcp.sidecar"]


def build_options(
    checkout: Path,
    url: Optional[str] = None,
    token: Optional[str] = None,
    model: Optional[str] = None,
    max_turns: Optional[int] = MAX_TURNS,
    max_budget_usd: Optional[float] = MAX_BUDGET_USD,
    resume: Optional[str] = None,
):
    """The SDK session: the sidecar as the only MCP server, the project's
    skills and nothing of the user's Claude Code config."""
    from claude_agent_sdk import ClaudeAgentOptions

    command = sidecar_command()
    server = {"type": "stdio", "command": command[0], "args": command[1:]}
    env = {}
    if url:
        env["FIBSEM_SERVER_URL"] = url
    if token:
        env["FIBSEM_SERVER_TOKEN"] = token
    if env:
        server["env"] = env
    return ClaudeAgentOptions(
        cwd=str(checkout),
        # The checkout's .claude/skills and CLAUDE.md, and none of
        # ~/.claude: the user's own settings, hooks and skills stay out.
        setting_sources=["project"],
        skills=list(SKILLS),
        tools=list(BUILT_IN_TOOLS),
        allowed_tools=list(ALLOWED_TOOLS),
        disallowed_tools=list(DENIED_TOOLS),
        permission_mode=PERMISSION_MODE,
        mcp_servers={SERVER: server},
        env=dict(CLI_ENV),
        # Only the server above: not a `fibsem` registered in ~/.claude.json.
        strict_mcp_config=True,
        system_prompt={
            "type": "preset",
            "preset": "claude_code",
            "append": RUNNER_PROMPT,
        },
        # Watcher lines carry operator-typed text: never expand an @path in it.
        verbatim_prompts=True,
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
        model=model,
        resume=resume,
        stderr=lambda line: log.debug("claude: %s", line.rstrip()),
    )


class _Formatter(logging.Formatter):
    """A log line's time as an aware ISO 8601 instant, with its offset."""

    def formatTime(self, record, datefmt=None):
        return timestamps.from_posix(record.created).isoformat(timespec="seconds")


def redact(text: str, *known: Optional[str]) -> str:
    """*text* with bearer tokens, API keys and the *known* secrets cut out."""
    text = SECRETS.sub(lambda m: f"{m.group(1) or ''}[redacted]", text)
    for secret in known:
        if secret:
            text = text.replace(secret, "[redacted]")
    return text


class _Redact(logging.Filter):
    """Keeps secrets out of the console and the log file."""

    def __init__(self, *known: Optional[str]) -> None:
        super().__init__()
        self._known = known

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage(), *self._known)
        record.args = None
        return True


def _log_to(path: Path, *secrets: Optional[str]) -> None:
    log.setLevel(logging.DEBUG)
    log.addFilter(_Redact(*secrets))
    log.propagate = False
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    log.addHandler(console)
    path.parent.mkdir(parents=True, exist_ok=True)
    file = logging.FileHandler(path, encoding="utf-8")
    file.setFormatter(_Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(file)


def _default_log() -> Path:
    stamp = timestamps.now().strftime("%Y-%m-%d-%H-%M-%S")
    return Path.home() / ".fibsem" / "agent" / f"fibsem-agent-{stamp}.log"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="fibsem-agent",
        description="Run an AutoLamella agent through the App Server, with no "
        "terminal session: it wakes on the app's events.",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="App Server URL, for a server on another machine; its token goes "
        "in FIBSEM_SERVER_TOKEN, never on the command line",
    )
    parser.add_argument("--model", default=None, help="model (default: the CLI's)")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=MAX_TURNS,
        help=f"tool calls the agent may make per wake-up (default {MAX_TURNS})",
    )
    parser.add_argument(
        "--max-budget-usd",
        type=float,
        default=MAX_BUDGET_USD,
        help="what this run may spend, after which fibsem-agent stops "
        f"(default {MAX_BUDGET_USD:g})",
    )
    parser.add_argument(
        "--resume", default=None, metavar="SESSION_ID", help="carry on a session"
    )
    parser.add_argument(
        "--server-wait",
        type=float,
        default=SERVER_WAIT_S,
        help="seconds to wait for an App Server that isn't answering before "
        f"stopping (default {SERVER_WAIT_S:.0f})",
    )
    parser.add_argument(
        "--log", default=None, help="log file (default ~/.fibsem/agent/...)"
    )
    parser.add_argument(
        "--checkout",
        default=None,
        help="the fibsem-os checkout whose .claude/skills the agent reads",
    )
    args = parser.parse_args(argv)

    try:
        import claude_agent_sdk  # noqa: F401
    except ImportError:
        print(
            "fibsem-agent needs the [agent] extra: pip install fibsem[agent]",
            file=sys.stderr,
        )
        return 1
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError, ValueError):
        pass
    checkout = find_checkout(args.checkout)
    if checkout is None:
        print(
            "fibsem-agent reads the agent's skills from a fibsem-os checkout "
            "(.claude/skills); pass --checkout <path to the checkout>.",
            file=sys.stderr,
        )
        return 1
    # From the environment, so it is not in the process list or shell history.
    token = os.environ.get("FIBSEM_SERVER_TOKEN")
    log_path = Path(args.log).expanduser() if args.log else _default_log()
    _log_to(log_path, token)
    log.info("fibsem-agent: skills from %s, log %s", checkout, log_path)

    options = build_options(
        checkout,
        url=args.url,
        token=token,
        model=args.model,
        max_turns=args.max_turns,
        max_budget_usd=args.max_budget_usd,
        resume=args.resume,
    )
    agent = SdkAgent(options)
    runner = Runner(
        agent,
        lambda: watch_mod.connect(args.url, token),
        resumed=args.resume is not None,
        server_wait_s=args.server_wait,
    )

    def on_interrupt(signum, frame):
        # A second Ctrl-C ends it at once.
        signal.signal(signal.SIGINT, signal.default_int_handler)
        runner.stop()

    previous = signal.signal(signal.SIGINT, on_interrupt)
    try:
        return asyncio.run(runner.run())
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        log.exception("The agent failed: %s", error)
        if agent.session_id:
            log.error("--resume %s carries the session on.", agent.session_id)
        return 1
    finally:
        signal.signal(signal.SIGINT, previous)


if __name__ == "__main__":
    sys.exit(main())
