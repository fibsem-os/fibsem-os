"""``fibsem-agent``: an AutoLamella agent with nobody at a terminal.

The Claude Agent SDK runs one long session with the ``fibsem-mcp`` sidecar as
its only MCP server, and the runner runs ``fibsem-mcp watch`` itself: each
batch of the watcher's lines is the agent's next turn. The App Server still
decides what the agent may do, and the runner never sets the level. Needs the
[agent] extra; nothing else in fibsem imports ``claude_agent_sdk``.

    fibsem-agent
    fibsem-agent --resume <session id>
    fibsem-agent --resume                     # the last session run in the same folder
    cd ~/fibsem-work && fibsem-agent --code   # it writes and runs code, sandboxed
    fibsem-agent --files ~/fibsem-work         # it reads and writes files there, runs nothing
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import signal
import sys
import sysconfig
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from fibsem.ai.agent import code_mode
from fibsem.ai.agent.levels import LEVEL_SKILLS
from fibsem.ai.agent.mcp import watch as watch_mod
from fibsem.logo import print_fibsem_logo
from fibsem.util import timestamps

BASE_SKILL = "autolamella-agent"
SKILLS = (BASE_SKILL,) + tuple(LEVEL_SKILLS.values())

SERVER = "fibsem"
# The specialists a Run Lead calls (supervise-autolamella, lead mode): the
# checkout's .claude/agents, each with its own tool list.
SPECIALISTS = (
    "autolamella-verifier",
    "autolamella-investigator",
    "autolamella-auditor",
)
# The agent's own todo list: TodoWrite, or the Task* tools in newer CLIs. They
# touch nothing outside the session.
TODO_TOOLS = ["TodoWrite", "TaskCreate", "TaskUpdate", "TaskList", "TaskGet"]
# The server's level gate decides what the fibsem tools may do, so they need no
# prompt; ToolSearch only finds them.
ALLOWED_TOOLS = (
    [f"mcp__{SERVER}__*", "ToolSearch", "SendMessage"]
    + [f"Agent({name})" for name in SPECIALISTS]
    + TODO_TOOLS
)
# The only built-in tools the agent has: reading its skills, finding the
# sidecar's tools when the CLI defers them, calling a specialist or carrying
# one on (a lamella's later polishing passes, one Verifier), and its todo list.
BUILT_IN_TOOLS = ["Skill", "ToolSearch", "Agent", "SendMessage"] + TODO_TOOLS
# Named as well, so a CLI that adds one back still refuses it: a specialist
# that lists a tool gets it unless it is denied here. The CLI's own agents
# would carry every fibsem tool, without a specialist's limits.
DENIED_TOOLS = [
    "Bash",
    "Read",
    "Grep",
    "Glob",
    "Write",
    "Edit",
    "MultiEdit",
    "NotebookEdit",
    "WebFetch",
    "WebSearch",
    "Agent(claude)",
    "Agent(general-purpose)",
    "Agent(Explore)",
    "Agent(Plan)",
    "Agent(statusline-setup)",
    "Agent(claude-code-guide)",
]
# Nothing waits on a permission prompt: what is not allowed above is denied.
PERMISSION_MODE = "dontAsk"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
# The agent's own reasoning effort; each specialist sets its own.
EFFORT = "high"

# The model can't answer until someone fixes the sign-in or the account, so
# waiting for the next wake-up would wait for ever.
FATAL_ERRORS = {"authentication_failed", "billing_error"}

# The CLI's environment: no CLAUDE.md (the checkout's is developer guidance),
# no auto memory from a developer's sessions in the same folder, and no
# background tasks: a specialist answers inside the turn that called it, and
# one the CLI ran in the background would answer after the turn had ended.
CLI_ENV = {
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
}

# One message from the CLI: a tool result carrying an image (the Read tool on
# a plot, in code mode) is more than the SDK's 1 MB default.
MAX_BUFFER_SIZE = 32 * 1024 * 1024
MAX_TURNS = 40  # tool calls per wake-up
MAX_BUDGET_USD = 5.0  # what one run spends
SERVER_WAIT_S = 600.0
# The transcript (`Runner._send_transcript`): sent this often while a turn
# runs and as it ends, at most this many lines at once, at most this many kept
# while the server can't take them.
TRANSCRIPT_EVERY_S = 10.0
TRANSCRIPT_BATCH = 200
TRANSCRIPT_KEEP = 2000
RETRY_MAX_S = 60.0

RUNNER_PROMPT = """\
You are running under fibsem-agent, a program on the microscope PC, not an \
interactive Claude Code session. Nobody is at a terminal.

- fibsem-agent runs `fibsem-mcp watch` for you. Each batch of its lines \
arrives as your next message, starting "The watcher printed:". Don't start a \
watcher, don't loop app_get_events, and don't wait inside a turn: do what the \
lines call for, then end your turn.
{tools}
- Your replies go to fibsem-agent's console and log, which the operator may \
not be reading. Anything the operator must read now (you have stopped and \
need them, or you answer their MESSAGE with its reply_to) goes to them with \
experiment_message_operator; a judgment worth keeping goes on the record with \
experiment_add_note.
- Keep the operator's instructions on your todo list (TodoWrite, or \
TaskCreate and TaskUpdate): the goal and each instruction a MESSAGE gives, in \
their words, as it arrives. Read the list before each step and tick an item \
only when it is done. A new instruction adds to the list; it never replaces \
what is there unless the operator says so.
- Your goal changes only when the operator's MESSAGE explicitly gives you a \
new one. Then call server_report_session with just goal, so the app shows \
it, and say back the goal you took.
- The operator sets your level in the app. Work by the skill for your level: \
observe-autolamella at Watch, assist-autolamella at Assist, \
supervise-autolamella at Drive. A LEVEL line means it changed: read that \
level's skill if you haven't. A refusal is the server's answer: handle it as \
the skills say."""
NO_CODE = """\
- Your tools are the mcp__fibsem__* tools, Skill, and Agent for the \
autolamella-* agents only. There is no shell, no file access and no web, for \
you or for them. Where a skill shows curl, python or a script, use the \
matching mcp__fibsem__ tool, or say what you can't do."""


def runner_prompt(work: Optional[Path] = None, code: bool = True) -> str:
    """What the runner tells the agent about itself; with a work folder (code
    mode, or files only), about it too."""
    if work is None:
        return RUNNER_PROMPT.format(tools=NO_CODE)
    return RUNNER_PROMPT.format(tools=code_mode.prompt(work, code))


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


def kickoff(resumed: bool = False, goal: Optional[str] = None) -> str:
    """The first turn: the skills, the level, and where things stand, and the
    goal the operator gave, if any."""
    prompt = _kickoff(resumed)
    if goal:
        prompt += (
            f"\n\nYour goal, from the operator who started you: {goal}\n"
            "Work toward it within your level, and say so when it is done or "
            "can't be done."
        )
    return prompt


def _kickoff(resumed: bool) -> str:
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
        "open, send that sentence with experiment_message_operator too, so "
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
        """Ends the watch: closing its client ends a parked poll as well."""
        self._stopping.set()
        _close(self._client)

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
        finally:
            _close(self._client)
        if not self._stopping.is_set():
            try:
                self._deliver(_WatchEnded(code))
            except _Stopped:
                pass


def _close(client) -> None:
    """Close a client the runner is done with, and its pooled connections."""
    try:
        client.close()
    except Exception:
        log.debug("Closing a client failed.", exc_info=True)


def _reachable(client) -> bool:
    """Whether the App Server answers this client, at any level; a warning
    when it runs other fibsem code than fibsem-agent."""
    from fibsem.server.discovery import code_mismatch

    try:
        response = client.request("GET", "/capabilities", timeout=10.0)
    except Exception:
        return False
    if response.status_code == 200:
        try:
            warning = code_mismatch(response.json(), "fibsem-agent")
        except Exception:  # noqa: BLE001 - a warning never costs the connection
            warning = None
        if warning:
            log.warning(warning)
    return response.status_code != 401 and response.status_code < 500


class Runner:
    """The wake loop: the kickoff, then one turn per batch of watcher lines,
    until the operator stops it, the budget is spent or the app is gone.

    ``agent`` is the seam: ``start()``, ``turn(prompt) -> TurnEnd``,
    ``interrupt()`` and ``close()``. ``connect`` gives the watch's client,
    or None while no server is known. ``run()`` returns the exit code: 0
    when stopped, 1 when the app or the agent is gone, 2 when the budget is.

    After every turn, and as it ends, the runner reports the session to the
    App Server (``POST /app/agent/session``): its id, model, goal, budget and
    spend, so the operator sees them in the app.
    """

    def __init__(
        self,
        agent,
        connect: Callable[[], Optional[object]],
        resumed: bool = False,
        server_wait_s: float = SERVER_WAIT_S,
        poll_s: float = watch_mod.POLL_S,
        retry_s: Optional[float] = None,
        goal: Optional[str] = None,
        budget_usd: Optional[float] = None,
        about: Optional[dict] = None,
    ) -> None:
        self._agent = agent
        self._goal = goal
        self._goal_sent = False
        # What it can do (code mode, effort, skills), sent with the first report.
        self._about = dict(about or {})
        self._budget_usd = budget_usd
        # The App Server's client, once one answers: the session is reported on it.
        self._client = None
        self._turns = 0
        self._spent_usd: Optional[float] = None
        self._end_reason = "stopped at fibsem-agent"
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
            prompt = kickoff(self._resumed, self._goal)
            while prompt is not None and await self._turn(prompt):
                prompt = await self._next_prompt()
        except Exception:
            # Ctrl-C reaches the CLI as well, which may end before its turn.
            if not self._stopping.is_set():
                raise
            log.debug("The agent's process ended as it stopped.", exc_info=True)
        finally:
            # Before the watcher stops: stopping it closes the client.
            await self._report(ended=True)
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
            self._client = client
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
        sender = asyncio.ensure_future(self._send_transcript_every())
        try:
            end = await self._agent.turn(prompt)
        finally:
            interrupter.cancel()
            sender.cancel()
        self._turns += 1
        if end.cost_usd is not None:
            self._spent_usd = end.cost_usd
        if self._stopping.is_set():
            return False
        session = end.session_id or getattr(self._agent, "session_id", None)
        if end.error in FATAL_ERRORS:
            self._end_reason = f"the model can't answer ({end.error})"
            log.error(
                "The model can't answer (%s); fibsem-agent stops. Set "
                "ANTHROPIC_API_KEY to a Claude API key (or configure Bedrock or "
                "Vertex), then start it again.",
                end.error,
            )
            self._code = 1
            return False
        if end.subtype == "error_max_budget_usd":
            self._end_reason = "its budget is spent"
            log.error(
                "The budget is spent (--max-budget-usd); fibsem-agent stops. "
                "Raise it and --resume %s to carry on.",
                session,
            )
            self._code = 2
            return False
        if end.subtype == "no_result":
            self._end_reason = "the agent's process ended"
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
        await self._report()
        return True

    async def _report(self, ended: bool = False) -> None:
        """The session as it stands, to the App Server; a refusal or a server
        that is gone costs nothing."""
        client = self._client
        if client is None:
            return
        body = {
            "session_id": getattr(self._agent, "session_id", None),
            "model": getattr(self._agent, "model", None),
            # Once: after that the goal is the agent's to report, when the
            # operator explicitly gives it a new one.
            "goal": None if self._goal_sent else self._goal,
            "budget_usd": self._budget_usd,
            "spent_usd": self._spent_usd,
            "turns": self._turns,
            **({} if self._goal_sent else self._about),
        }
        body = {key: value for key, value in body.items() if value is not None}
        if ended:
            body.update(ended=True, reason=self._end_reason)
        self._goal_sent = True

        def post():
            try:
                client.request("POST", "/app/agent/session", json=body, timeout=10.0)
            except Exception:
                log.debug("Reporting the session failed.", exc_info=True)

        await asyncio.get_running_loop().run_in_executor(None, post)
        await self._send_transcript()

    async def _send_transcript(self) -> None:
        """The lines printed since the last send, to the App Server for the web
        page (``POST /app/agent/transcript``). Kept for the next send when it
        can't take them (no experiment open); the oldest go past
        `TRANSCRIPT_KEEP`."""
        client, lines = self._client, getattr(self._agent, "transcript", None)
        session_id = getattr(self._agent, "session_id", None)
        if client is None or not lines or not session_id:
            return
        now = time.monotonic()
        batch = lines[:TRANSCRIPT_BATCH]
        entries = [
            {**{k: v for k, v in line.items() if k != "at"}, "age_s": now - line["at"]}
            for line in batch
        ]

        def post():
            try:
                client.request(
                    "POST",
                    "/app/agent/transcript",
                    json={"session_id": session_id, "entries": entries},
                    timeout=10.0,
                )
                return True
            except Exception:
                log.debug("Sending the transcript failed.", exc_info=True)
                return False

        if await asyncio.get_running_loop().run_in_executor(None, post):
            del lines[: len(batch)]
        del lines[:-TRANSCRIPT_KEEP]

    async def _send_transcript_every(self) -> None:
        while True:
            await asyncio.sleep(TRANSCRIPT_EVERY_S)
            await self._send_transcript()

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
        self._end_reason = "the app could not be watched"
        log.warning("The watcher stopped: the app can't be watched.")
        started = time.monotonic()
        if not await self._watch():
            return None
        self._end_reason = "stopped at fibsem-agent"
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
            if client is not None:
                if await asyncio.get_running_loop().run_in_executor(
                    None, _reachable, client
                ):
                    if said:
                        log.info("The App Server answers.")
                    return client
                _close(client)
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
        self.session_id: Optional[str] = options.resume or options.session_id
        self.model: Optional[str] = options.model
        self.introduced = False
        # The lines printed, until the runner sends them (`Runner._send_transcript`).
        self.transcript: List[dict] = []
        self._seq = 0
        # A tool call's line, by the call's id: its result may say it failed.
        self._steps: dict = {}
        self.cwd = Path(options.cwd) if options.cwd else None

    def line(self, kind: str, text: str, **extra) -> int:
        """Keep a printed line for the transcript; its number."""
        self._seq += 1
        self.transcript.append(
            {
                "seq": self._seq,
                "kind": kind,
                "text": text,
                "at": time.monotonic(),
                **extra,
            }
        )
        return self._seq

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


# The code tools' lines: what was done, to which file.
_FILE_VERBS = {
    "Write": "Wrote",
    "Edit": "Edited",
    "MultiEdit": "Edited",
    "Read": "Read",
}


def step(name: str, arguments: dict, cwd: Optional[Path] = None) -> tuple:
    """A tool call as the transcript says it, (verb, text): ("Wrote",
    "analyse.py"), ("Ran", "python analyse.py"), ("Fetched", "02/ref.tif");
    any other tool ("Called", "<name> <its arguments, clipped>")."""
    arguments = arguments if isinstance(arguments, dict) else {}
    short = _short(name)
    if name in _FILE_VERBS:
        path = str(arguments.get("file_path", ""))
        if cwd is not None:
            try:
                path = str(Path(path).relative_to(cwd))
            except ValueError:
                pass
        return _FILE_VERBS[name], path
    if name == "Bash":
        command = str(arguments.get("command", "")).strip().splitlines() or [""]
        more = " ..." if len(command) > 1 else ""
        return "Ran", _clip(command[0], 160) + more
    if name in ("Glob", "Grep"):
        return "Searched", _clip(arguments.get("pattern", ""), 160)
    if name == "Skill":
        return "Read", f"the {arguments.get('skill', '')} skill"
    if short == "experiment_fetch_file":
        return "Fetched", _clip(arguments.get("path", ""), 160)
    text = json.dumps(arguments, ensure_ascii=False) if arguments else ""
    return "Called", _clip(f"{short} {text}".strip(), 200)


def _line(agent, kind: str, text: str, **extra) -> Optional[int]:
    """Keep a printed line on *agent*'s transcript, if it keeps one."""
    keep = getattr(agent, "line", None)
    return keep(kind, text, **extra) if keep is not None else None


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
            agent.model = data.get("model") or agent.model
            log.info("Session %s", agent.session_id)
            if agent.session_id and agent.cwd is not None:
                keep_session(agent.cwd, agent.session_id)
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
                _line(agent, "said", block.text.strip())
            elif isinstance(block, ToolUseBlock):
                verb, text = step(block.name, block.input, getattr(agent, "cwd", None))
                log.info("%s %s", verb, text)
                seq = _line(agent, "step", text, verb=verb)
                if seq is not None:
                    agent._steps[block.id] = seq
    elif isinstance(message, UserMessage) and isinstance(message.content, list):
        for block in message.content:
            if isinstance(block, ToolResultBlock) and block.is_error:
                why = _clip(_result_text(block.content))
                log.warning("failed: %s", why)
                seq = getattr(agent, "_steps", {}).get(block.tool_use_id)
                _line(agent, "failed", _clip(why, 160), step=seq)
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
    effort: Optional[str] = EFFORT,
    work: Optional[Path] = None,
    session_id: Optional[str] = None,
    code: bool = True,
):
    """The SDK session: the sidecar as the only MCP server, the project's
    skills and nothing of the user's Claude Code config. With ``work``, code
    mode: the code tools, sandboxed, in ``work`` (``code_mode.prepare`` has
    copied the skills there); without ``code``, the file tools only."""
    from claude_agent_sdk import ClaudeAgentOptions

    command = sidecar_command()
    server = {"type": "stdio", "command": command[0], "args": command[1:]}
    # In the CLI's environment, which the sidecar inherits: the server entry
    # goes on the CLI's command line, where anyone can list it.
    env = dict(CLI_ENV)
    if url:
        env["FIBSEM_SERVER_URL"] = url
    if token:
        env["FIBSEM_SERVER_TOKEN"] = token
    tools, allowed, denied = BUILT_IN_TOOLS, ALLOWED_TOOLS, DENIED_TOOLS
    settings = None
    if work is not None:
        env.update(code_mode.env(work))
        given = code_mode.tools(code)
        tools = BUILT_IN_TOOLS + given
        allowed = ALLOWED_TOOLS + code_mode.allowed_tools(work, code)
        denied = [tool for tool in DENIED_TOOLS if tool not in given]
        settings = code_mode.settings_json(work, {**os.environ, **env}, code)
    return ClaudeAgentOptions(
        cwd=str(work or checkout),
        # The checkout's .claude/skills (CLAUDE.md is off, see CLI_ENV), and none of
        # ~/.claude: the user's own settings, hooks and skills stay out.
        setting_sources=["project"],
        skills=list(SKILLS),
        tools=list(tools),
        allowed_tools=list(allowed),
        disallowed_tools=list(denied),
        permission_mode=PERMISSION_MODE,
        settings=settings,
        mcp_servers={SERVER: server},
        env=env,
        # Only the server above: not a `fibsem` registered in ~/.claude.json.
        strict_mcp_config=True,
        system_prompt={
            "type": "preset",
            "preset": "claude_code",
            "append": runner_prompt(work, code),
        },
        # Watcher lines carry operator-typed text: never expand an @path in it.
        verbatim_prompts=True,
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
        model=model,
        effort=effort,
        resume=resume,
        session_id=session_id,
        max_buffer_size=MAX_BUFFER_SIZE,
        stderr=lambda line: log.debug("claude: %s", line.rstrip()),
    )


class _Formatter(logging.Formatter):
    """A log line's time as an aware ISO 8601 instant, with its offset."""

    def formatTime(self, record, datefmt=None):
        return timestamps.from_posix(record.created).isoformat(timespec="seconds")


def _log_to(path: Path) -> None:
    log.setLevel(logging.DEBUG)
    log.propagate = False
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    log.addHandler(console)
    path.parent.mkdir(parents=True, exist_ok=True)
    file = logging.FileHandler(path, encoding="utf-8")
    file.setFormatter(_Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(file)


# A bare --resume: the last session run in the same folder.
LAST = "last"


def _sessions_file() -> Path:
    return Path.home() / ".fibsem" / "agent" / "sessions.json"


def _sessions() -> dict:
    try:
        sessions = json.loads(_sessions_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return sessions if isinstance(sessions, dict) else {}


def last_session(folder: Path) -> Optional[str]:
    """The last session run in *folder* (the CLI's: the work folder, else the
    checkout), or None. The CLI keeps a session under its folder, so one run
    elsewhere can't be resumed here."""
    session = _sessions().get(str(folder))
    return session if isinstance(session, str) else None


def keep_session(folder: Path, session_id: str) -> None:
    """Remember *session_id* as the last one run in *folder*."""
    sessions = _sessions()
    sessions[str(folder)] = session_id
    path = _sessions_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sessions, indent=2), encoding="utf-8")
    except OSError as error:
        log.warning("Couldn't keep the session id in %s: %s", path, error)


def _default_log() -> Path:
    stamp = timestamps.now().strftime("%Y-%m-%d-%H-%M-%S")
    return Path.home() / ".fibsem" / "agent" / f"fibsem-agent-{stamp}.log"


def status(
    args,
    checkout: Path,
    log_path: Path,
    work: Optional[Path],
    session_id: str,
    code: bool = True,
) -> List[str]:
    """What this run is, at its start: one line each."""
    from fibsem import __version__

    rows = [
        ("fibsem", __version__),
        ("Server", args.url or "this computer's App Server"),
        ("Model", args.model or "Claude Code's default"),
        ("Goal", args.goal or "none: the skills decide"),
        (
            "Budget",
            f"${args.max_budget_usd:g} per run, {args.max_turns} tool calls "
            "per wake-up",
        ),
        ("Session", f"{session_id} ({'resumed' if args.resume else 'new'})"),
        ("Skills", str(checkout)),
        ("Code", code_mode.describe(work, code)),
        ("Log", str(log_path)),
    ]
    return [f"{label + ':':<9}{value}" for label, value in rows]


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
        "--goal",
        default=None,
        help="what the agent is to do, in a sentence; the app shows it",
    )
    parser.add_argument(
        "--effort",
        choices=EFFORTS,
        default=EFFORT,
        help=f"the agent's reasoning effort (default {EFFORT}); each "
        "specialist sets its own",
    )
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
        "--resume",
        nargs="?",
        const=LAST,
        default=None,
        metavar="SESSION_ID",
        help="carry on a session; with no id, the last one run in the same "
        "folder (the --code or --files folder, else the checkout)",
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
    parser.add_argument(
        "--code",
        nargs="?",
        const=".",
        default=None,
        metavar="PATH",
        help="code mode: the agent writes and runs its own code in PATH (the "
        "current folder if none), sandboxed (macOS and Linux only)",
    )
    parser.add_argument(
        "--files",
        nargs="?",
        const=".",
        default=None,
        metavar="PATH",
        help="the agent reads and writes files in PATH (the current folder if "
        "none) and runs nothing: no shell",
    )
    args = parser.parse_args(argv)
    if args.code and args.files:
        parser.error("--code already gives the agent its files: use one of them")

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
    work = None
    code = bool(args.code)
    if args.code or args.files:
        work = code_mode.workdir(args.code or args.files)
        refused = code_mode.refusal(work, checkout, code)
        if refused:
            print(refused, file=sys.stderr)
            return 1
    if args.resume == LAST:
        args.resume = last_session(work or checkout)
        if args.resume is None:
            print(
                f"No earlier fibsem-agent session in {work or checkout} to "
                "resume: pass its id (the Session line of its log).",
                file=sys.stderr,
            )
            return 1
    # From the environment, so it is not in the process list or shell history.
    token = os.environ.get("FIBSEM_SERVER_TOKEN")
    log_path = Path(args.log).expanduser() if args.log else _default_log()
    print_fibsem_logo()
    _log_to(log_path)
    # A new session's id is chosen here, so the status can name it.
    session_id = args.resume or str(uuid.uuid4())
    for line in status(args, checkout, log_path, work, session_id, code):
        log.info(line)
    if work is not None:
        code_mode.prepare(work, checkout, list(SKILLS), list(SPECIALISTS))
        for line in code_mode.details():
            log.debug(line)

    options = build_options(
        checkout,
        url=args.url,
        token=token,
        model=args.model,
        max_turns=args.max_turns,
        max_budget_usd=args.max_budget_usd,
        resume=args.resume,
        effort=args.effort,
        work=work,
        session_id=None if args.resume else session_id,
        code=code,
    )
    agent = SdkAgent(options)
    runner = Runner(
        agent,
        lambda: watch_mod.connect(args.url, token),
        resumed=args.resume is not None,
        server_wait_s=args.server_wait,
        goal=args.goal,
        budget_usd=args.max_budget_usd,
        about={
            "harness": "fibsem-agent",
            "code": work is not None and code,
            "files": work is not None and not code,
            "workdir": str(work) if work is not None else None,
            "effort": args.effort,
            "skills": list(SKILLS),
            "resumed": args.resume is not None,
        },
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
