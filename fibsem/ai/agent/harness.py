"""``fibsem-agent``: the fibsem agent, with nobody at a terminal.

The Claude Agent SDK runs one long session with the ``fibsem-mcp`` sidecar as
its only MCP server, and the harness runs ``fibsem-mcp watch`` itself: each
batch of the watcher's lines is the agent's next turn. The App Server still
decides what the agent may do, and the harness never sets the level. Needs the
[agent] extra; nothing else in fibsem imports ``claude_agent_sdk``.

What it runs with comes from its playbook (`playbooks`): skills, subagents,
session settings and the instrument kind whose words it is told
(`instruments`). The shipped base playbook holds the defaults; the code keeps
no second copy.

    fibsem-agent
    fibsem-agent --resume <session id>
    fibsem-agent --resume                     # the last session run in the same folder
    fibsem-agent --session-id <id>            # a new session with this id (a manager's)
    fibsem-agent --code                       # it writes and runs code, sandboxed,
                                              # in work/ in the session's folder
    fibsem-agent --files ~/fibsem-work        # it reads and writes files there, runs nothing
    fibsem-agent --playbook autolamella       # what it runs with, from a playbook
    fibsem-agent --playbook autolamella --show-config
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
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from fibsem.ai.agent import (
    code_mode,
    instruments,
    mirror,
    playbooks,
    reminders,
    sources,
    subagents,
    usage,
)
from fibsem.ai.agent import manager as sessions
from fibsem.ai.agent import watch as watch_mod
from fibsem.ai.agent.catalog import INSTRUMENT_ENV
from fibsem.client.transport import SERVER_TOKEN_ENV, SERVER_URL_ENV, server_env
from fibsem.extras import missing_extra_message
from fibsem.keystore import get_secret
from fibsem.logo import print_fibsem_logo
from fibsem.unreleased import refuse
from fibsem.util import timestamps

# The skill that marks a folder as holding the shipped .claude (`find_checkout`).
# Which skills and subagents a session loads is its playbooks' (`session_names`).
BASE_SKILL = "fibsem-agent"

SERVER = "fibsem"
# The harness's own tools, in-process: the wake-up timer (`Alarm`).
HARNESS_SERVER = "fibsem_agent"
WAKE_TOOL = f"mcp__{HARNESS_SERVER}__wake_after"
# The playbook tools (`PlaybookControl`), on the same in-process server.
PLAYBOOK_TOOLS = [
    f"mcp__{HARNESS_SERVER}__{name}"
    for name in ("playbook_list", "playbook_set", "skill_done")
]
# The server's level gate decides what the fibsem tools may do, so they need no
# prompt; ToolSearch only finds them. The session's subagents are added
# (`subagent_tools`).
ALLOWED_TOOLS = (
    [f"mcp__{SERVER}__*", "ToolSearch", "SendMessage"] + [WAKE_TOOL] + PLAYBOOK_TOOLS
)
# The only built-in tools the agent has: reading its skills, finding the
# sidecar's tools when the CLI defers them, calling a specialist or carrying
# one on (a lamella's later polishing passes, one Verifier).
BUILT_IN_TOOLS = ["Skill", "ToolSearch", "Agent", "SendMessage"]
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

# The model can't answer until someone fixes the sign-in or the account, so
# waiting for the next wake-up would wait for ever.
FATAL_ERRORS = {"authentication_failed", "billing_error"}
API_KEY = "ANTHROPIC_API_KEY"

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
SERVER_WAIT_S = 600.0
# How long the CLI has to say how full the context is, after a turn.
CONTEXT_WAIT_S = 10.0
# The transcript (`Harness._send_transcript`): sent this often while a turn
# runs and as it ends, at most this many lines at once, at most this many kept
# while the server can't take them.
TRANSCRIPT_EVERY_S = 10.0
TRANSCRIPT_BATCH = 200
TRANSCRIPT_KEEP = 2000
RETRY_MAX_S = 60.0

NO_CODE = """\
- Your tools are the mcp__fibsem__* tools, Skill, and Agent for the subagents \
the playbook names. There is no shell, no file access and no web, for \
you or for them. Where a skill shows curl, python or a script, use the \
matching mcp__fibsem__ tool, or say what you can't do."""


def defaults() -> playbooks.Session:
    """The shipped base playbook's session settings: what a timer or a
    watcher built without them uses."""
    base = playbooks.shipped_base(find_checkout())
    return base.session if base is not None else playbooks.Session()


def texts(kind: Optional[str] = None) -> instruments.Texts:
    """What the agent is told for instrument *kind* (else the base
    playbook's), from the shipped files."""
    checkout = find_checkout()
    if kind is None:
        base = playbooks.shipped_base(checkout)
        kind = base.instrument if base is not None else None
    return instruments.shipped(kind or instruments.DEFAULT, checkout)


def harness_prompt(
    work: Optional[Path] = None,
    code: bool = True,
    words: Optional[instruments.Texts] = None,
    deadline_warn_min: Optional[float] = None,
) -> str:
    """What the harness tells the agent about itself; with a work folder (code
    mode, or files only), about it too."""
    words = words or texts()
    tools = NO_CODE if work is None else code_mode.prompt(work, code)
    warn = deadline_warn_min or defaults().deadline_warn_min
    return words.prompt.format(tools=tools, deadline_warn_min=f"{warn:g}")


def system_prompt(
    work: Optional[Path] = None,
    code: bool = True,
    playbook: Optional[playbooks.Playbook] = None,
    words: Optional[instruments.Texts] = None,
) -> str:
    """The harness's prompt, and the rules of the playbook the session
    started on after it, so they are read every turn."""
    warn = playbook.session.deadline_warn_min if playbook is not None else None
    prompt = harness_prompt(work, code, words, warn)
    if playbook is not None and playbook.rules:
        prompt += (
            f"\n\nRules of the {playbook.name} playbook you started on:\n"
            + "\n".join(f"- {rule}" for rule in playbook.rules)
        )
    return prompt


@dataclass
class Loaded:
    """The skills and subagents a session loads, fixed at its start: the CLI
    reads them from its folder once."""

    skills: List[str]
    agents: List[str]

    def missing(self, playbook: playbooks.Playbook) -> List[str]:
        """What *playbook* names that the session didn't load."""
        return [n for n in playbook.all_skills() if n not in self.skills] + [
            n for n in playbook.all_agents() if n not in self.agents
        ]


def session_names(
    checkout: Optional[Path], playbook: Optional[playbooks.Playbook] = None
) -> Loaded:
    """What a session loads. Started on a playbook other than the base one:
    that playbook's skills and subagents (with what it extends). Else every
    playbook's of its instrument kind, as intake may set any of them. The lab's
    own skills and subagents always, a lab copy of a shipped one replacing it."""
    if playbook is not None and playbook.name != playbooks.DEFAULT:
        found = [playbook]
    else:
        kind = _kind(playbook)
        found = []
        for name in playbooks.available(checkout):
            try:
                each = playbooks.load(name, checkout)
            except playbooks.PlaybookError as error:
                log.warning("Playbook %s is left out: %s", name, error)
                continue
            if _kind(each) == kind:
                found.append(each)
    skills: List[str] = []
    agents: List[str] = []
    for each in found:
        skills += [n for n in each.all_skills() if n not in skills]
        agents += [n for n in each.all_agents() if n not in agents]
    skills += [n for n in sources.lab_skills() if n not in skills]
    agents += [n for n in sources.lab_agents() if n not in agents]
    return Loaded(skills, agents)


def _kind(playbook: Optional[playbooks.Playbook]) -> str:
    """*playbook*'s instrument kind; the default kind's for none."""
    kind = playbook.instrument if playbook is not None else None
    return kind or instruments.DEFAULT


def subagent_tools(names: List[str]) -> List[str]:
    """The ``Agent(<name>)`` rules for the session's subagents."""
    return [f"Agent({name})" for name in names]


log = logging.getLogger("fibsem.agent.harness")


def kickoff(
    resumed: bool = False,
    goal: Optional[str] = None,
    words: Optional[instruments.Texts] = None,
) -> str:
    """The first turn: the skills, the level, and where things stand, and the
    goal the operator gave, if any."""
    words = words or texts()
    prompt = words.resumed if resumed else words.start
    if goal:
        prompt += "\n\n" + words.goal.format(goal=goal)
    return prompt


def turn_prompt(
    lines: List[str],
    remind: Optional[List[str]] = None,
    words: Optional[instruments.Texts] = None,
) -> str:
    """The watcher's lines as one turn, with what a GAP or a level change
    asks of the agent first, and the reminders *remind* the lines call for."""
    words = words or texts()
    prompt = "The watcher printed:\n" + "\n".join(lines)
    if remind:
        prompt += "\n\n" + reminders.reminder(remind)
    if any(line.startswith("GAP") for line in lines):
        prompt += "\n\n" + words.reread_proposals
    if any(line.startswith(("AGENT BACK", "RESTARTED", "LEVEL")) for line in lines):
        prompt += "\n\n" + words.reread_level
    if any(line.startswith(("WAKE ", "PLAN EDITED")) for line in lines):
        prompt += "\n\n" + words.reread_plan
    return prompt


class Wakes:
    """The watcher's lines, held while the operator has the agent at Off and
    given to the agent as one turn otherwise."""

    def __init__(
        self,
        remind: Optional[Callable[[List[str]], List[str]]] = None,
        words: Optional[instruments.Texts] = None,
    ) -> None:
        self.off = False
        self._held: List[str] = []
        # The reminders a turn's lines call for (`playbooks.Reminders.for_lines`).
        self._remind = remind
        self._words = words

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
        remind = self._remind(lines) if self._remind else None
        return turn_prompt(lines, remind, self._words)


class Alarm:
    """The agent's wake-up timer (the ``wake_after`` tool): one at a time, a
    new one replacing it. When it rings, a ``WAKE`` line joins the watcher's
    lines, so it is a turn like any other and is held while the agent is Off."""

    def __init__(self, max_min: Optional[float] = None) -> None:
        # The longest a wake-up can be set for, in minutes.
        self.max_min = max_min or defaults().wake_max_min
        self._ring: Optional[Callable[[str], None]] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._timer: Optional[asyncio.TimerHandle] = None
        self.reason: Optional[str] = None

    def bind(self, loop: asyncio.AbstractEventLoop, ring: Callable[[str], None]):
        """Ring through *ring* on *loop* (`Harness.run`)."""
        self._loop, self._ring = loop, ring

    def set(self, minutes: float, reason: str) -> str:
        """Wake the agent after *minutes*; 0 cancels. What to tell the agent."""
        self.cancel()
        if minutes <= 0:
            return "No wake-up is set."
        if self._loop is None:
            return "The harness can't wake you yet; the watcher's lines still will."
        minutes = min(float(minutes), self.max_min)
        self.reason = reason.strip() or "you asked to be woken"
        line = f"WAKE {self.reason} (you asked {minutes:g} min ago)"
        self._timer = self._loop.call_later(minutes * 60, self._rang, line)
        log.info("Wake-up in %g min: %s", minutes, self.reason)
        return (
            f"You will be woken in {minutes:g} min with: {self.reason}. A new "
            "wake_after replaces it; 0 cancels it."
        )

    def cancel(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer, self.reason = None, None

    def _rang(self, line: str) -> None:
        self._timer, self.reason = None, None
        if self._ring is not None:
            self._ring(line)


class Deadlines:
    """The plan's deadlines (Rules with a time): each wakes the agent *warn_s*
    before it (the playbook's ``deadline_warn_min``) and at it, with a
    ``DEADLINE`` line, like the watcher's lines. A deadline kept from one
    report to the next keeps its timers."""

    def __init__(self, warn_s: Optional[float] = None) -> None:
        self.warn_s = warn_s or defaults().deadline_warn_min * 60
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ring: Optional[Callable[[str], None]] = None
        self._timers: Dict[Tuple[str, str], List[asyncio.TimerHandle]] = {}

    def bind(self, loop: asyncio.AbstractEventLoop, ring: Callable[[str], None]):
        self._loop, self._ring = loop, ring

    def set(self, deadlines: Optional[List[Dict[str, Any]]]) -> None:
        """The session's deadlines as the App Server reports them."""
        wanted = {
            (c["text"], c["by"])
            for c in deadlines or []
            if isinstance(c, dict) and c.get("text") and c.get("by")
        }
        for key in set(self._timers) - wanted:
            for timer in self._timers.pop(key):
                timer.cancel()
        if self._loop is None:
            return
        now = datetime.now().astimezone()
        for text, by in wanted - set(self._timers):
            try:
                when = datetime.fromisoformat(by)
            except ValueError:
                continue
            at = when.strftime("%H:%M")
            lines = (
                (
                    self.warn_s,
                    f"DEADLINE in {self.warn_s / 60:g} min: {text} (by {at})",
                ),
                (0.0, f"DEADLINE now: {text} (by {at})"),
            )
            self._timers[(text, by)] = [
                self._loop.call_later(delay, self._ring, line)
                for before, line in lines
                if (delay := (when - now).total_seconds() - before) > 0
            ]

    def cancel(self) -> None:
        self.set(None)

    @property
    def times(self) -> List[str]:
        return sorted(by for _, by in self._timers)


class PlaybookControl:
    """The session's playbook, as its tools set it: which one runs, who chose
    it, and which of its optional skills are loaded (`playbooks.Reminders`).
    Each change is said on the transcript (*say*) and kept (*keep*)."""

    def __init__(
        self,
        reminders: playbooks.Reminders,
        checkout: Optional[Path],
        say: Optional[Callable[[str], None]] = None,
        keep: Optional[Callable[[dict], None]] = None,
        loaded: Optional[Loaded] = None,
    ) -> None:
        self.reminders = reminders
        self.checkout = checkout
        # What the session loaded at its start; None takes any playbook.
        self.loaded_names = loaded
        self.say = say or (lambda text: None)
        self.keep = keep or (lambda state: None)

    def _changed(self, line: str) -> None:
        log.info(line)
        self.say(line)
        self.keep(self.reminders.state())

    def list(self) -> str:
        """The playbooks of the session's instrument kind: its tools and words
        are fixed at its start."""
        rows = []
        for name in playbooks.available(self.checkout):
            try:
                found = playbooks.load(name, self.checkout)
            except playbooks.PlaybookError as error:
                rows.append({"name": name, "error": str(error)})
                continue
            if _kind(found) == _kind(self.reminders.playbook):
                rows.append(found.summary())
        current = self.reminders.playbook
        return json.dumps(
            {"current": current.name if current else None, "playbooks": rows},
            ensure_ascii=False,
        )

    def set(self, name: str, chosen_by: str) -> str:
        """Run playbook *name* from now on; an empty name, the base one. Its
        session settings count only at a session's start, so are not given."""
        try:
            found = playbooks.load(name or playbooks.DEFAULT, self.checkout)
        except playbooks.PlaybookError as error:
            return f"Refused: {error}"
        if _kind(found) != _kind(self.reminders.playbook):
            return (
                f"Refused: the {found.name} playbook is for a {_kind(found)} "
                f"instrument, and this session runs a "
                f"{_kind(self.reminders.playbook)}'s tools."
            )
        missing = self.loaded_names.missing(found) if self.loaded_names else []
        if missing:
            return (
                f"Refused: the {found.name} playbook needs {', '.join(missing)}, "
                "which this session didn't load. Tell the operator to restart the "
                f"agent with the {found.name} playbook."
            )
        self.reminders.set(found, chosen_by)
        self._changed(f"Playbook set: {found.name}, {chosen_by}")
        self._subagents([])
        summary = found.summary()
        summary.pop("session", None)
        return (
            f"Running the {found.name} playbook. Its reminders are on now and its "
            "rules hold. The model, budget and mode stay as this session started. "
            "Read its skills, and an optional one when its when holds:\n"
            + json.dumps(summary, ensure_ascii=False)
        )

    def loaded(self, skill: str) -> None:
        """The agent read *skill* with the Skill tool."""
        before = self.reminders.subagents()
        entry = self.reminders.loaded(skill)
        if entry is not None:
            self._changed(f"Skill loaded: {skill}, because {entry.when}")
            self._subagents(before)

    def _subagents(self, before: List[str]) -> None:
        """Say the subagents on now, when they changed."""
        now = self.reminders.subagents()
        if now != before and now:
            self._changed(f"Subagents on: {', '.join(now)}")

    def done(self, skill: str, summary: str) -> str:
        if not self.reminders.finished(skill, summary):
            return (
                f"Refused: {skill} is not an optional skill of this session's "
                "playbook that you have read."
            )
        self._changed(f"Skill done: {skill}, {summary}".rstrip(", "))
        return f"{skill} is done: its reminders are off."

    def hooks(self) -> List[dict]:
        """The reminders on now, as the Agent page shows them."""
        rows = []
        for when, option in reminders.HOOK_OPTIONS.items():
            for say in self.reminders.texts(when):
                rows.append({"when": when, "label": option.label, "say": say})
        return rows


def wake_server(alarm: Alarm, control: Optional[PlaybookControl] = None):
    """The harness's in-process MCP server: the ``wake_after`` tool on
    *alarm*, and with *control* the playbook tools."""
    from claude_agent_sdk import create_sdk_mcp_server, tool

    @tool(
        "wake_after",
        "Wake yourself after some minutes to look again: after a long mill, to "
        "check drift, before a deadline. When it rings you get a WAKE line with "
        f"your reason, like the watcher's lines. One at a time, at most "
        f"{alarm.max_min:g} min: a new one replaces it and minutes 0 cancels it. "
        "The watcher still wakes you for anything that happens before then.",
        {"minutes": float, "reason": str},
    )
    async def wake_after(args):
        said = alarm.set(float(args.get("minutes", 0)), str(args.get("reason", "")))
        return {"content": [{"type": "text", "text": said}]}

    if control is None:
        return create_sdk_mcp_server(HARNESS_SERVER, tools=[wake_after])

    def text(said: str) -> dict:
        return {"content": [{"type": "text", "text": said}]}

    @tool(
        "playbook_list",
        "The playbooks you can run: each one's description, skills, optional "
        "skills with when to read them, reminders and rules, and the one "
        "running now.",
        {},
    )
    async def playbook_list(args):
        return text(control.list())

    @tool(
        "playbook_set",
        "Run a playbook from now on: its reminders go on at once and its rules "
        "hold. Set it once the operator has agreed to it, or when they name "
        "one. chosen_by says who chose it, as the operator reads it: 'named by "
        "the operator', or 'proposed by the agent, confirmed by the "
        "operator'. An empty name runs the base fibsem-agent playbook.",
        {"name": str, "chosen_by": str},
    )
    async def playbook_set(args):
        return text(
            control.set(
                str(args.get("name", "")).strip(),
                str(args.get("chosen_by", "")).strip() or "by the agent",
            )
        )

    @tool(
        "skill_done",
        "An optional skill of the playbook is done for this session: its "
        "reminders go off. summary says in a few words what came of it, "
        "('9 usable squares, 6 sites chosen').",
        {"skill": str, "summary": str},
    )
    async def skill_done(args):
        return text(
            control.done(
                str(args.get("skill", "")).strip(), str(args.get("summary", "")).strip()
            )
        )

    return create_sdk_mcp_server(
        HARNESS_SERVER, tools=[wake_after, playbook_list, playbook_set, skill_done]
    )


@dataclass
class TurnEnd:
    """How one turn ended: the SDK's result subtype and what it reported."""

    subtype: str = "success"
    cost_usd: Optional[float] = None
    session_id: Optional[str] = None
    # Why the model did not answer (the SDK's AssistantMessage.error), if so.
    error: Optional[str] = None
    # The turn's tokens ({input, output, cache_read, cache_write}).
    tokens: Optional[Dict[str, int]] = None
    # How full the context is: {used, window, compact_at}, in tokens.
    context: Optional[Dict[str, int]] = None


class _Stopped(Exception):
    """The harness is stopping: raised inside ``watch()`` to end it."""


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
        quiet_s: Optional[float] = None,
    ) -> None:
        self._client = client
        self._deliver = deliver
        self._poll_s = poll_s
        self._retry_s = retry_s
        # A running workflow quiet this long wakes the agent, once.
        self._quiet_s = quiet_s or defaults().quiet_min * 60
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
            # The harness marks a message read when the turn carrying it starts.
            code = watch_mod.watch(
                client,
                out=self._out,
                sleep_fn=self._sleep,
                poll_s=self._poll_s,
                quiet_s=self._quiet_s,
                report_reads=False,
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
    """Close a client the harness is done with, and its pooled connections."""
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


class Harness:
    """The wake loop: the kickoff, then one turn per batch of watcher lines,
    until the operator stops it, the budget is spent or the app is gone.

    ``agent`` is the seam: ``start()``, ``turn(prompt) -> TurnEnd``,
    ``interrupt()`` and ``close()``. ``connect`` gives the watch's client,
    or None while no server is known. ``run()`` returns the exit code: 0
    when stopped, 1 when the app or the agent is gone, 2 when the budget is.

    After every turn, and as it ends, the harness reports the session to the
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
        alarm: Optional[Alarm] = None,
        playbook: Optional[PlaybookControl] = None,
        words: Optional[instruments.Texts] = None,
        deadline_warn_s: Optional[float] = None,
        quiet_s: Optional[float] = None,
    ) -> None:
        self._agent = agent
        # What the agent is told, for the session's instrument kind.
        self._words = words or texts()
        self._quiet_s = quiet_s
        # The session's playbook: sent again when it or its skills change.
        self._playbook = playbook
        self._playbook_sent: Optional[dict] = None
        # The agent's wake-up timer: it rings into the watcher's lines.
        self._alarm = alarm
        self._deadlines = Deadlines(deadline_warn_s)
        self._goal = goal
        self._goal_sent = False
        # The skills read, as last reported: sent again when the agent reads one.
        self._skills_sent: List[str] = []
        # What it can do (code mode, effort, skills), sent with the first report.
        self._about = dict(about or {})
        self._budget_usd = budget_usd
        # The App Server's client, once one answers: the session is reported on it.
        self._client = None
        self._turns = 0
        self._spent_usd: Optional[float] = None
        self._tokens: Optional[Dict[str, int]] = None
        self._context: Optional[Dict[str, int]] = None
        self._end_reason = "stopped at fibsem-agent"
        # False when resuming can't help: the model can't answer.
        self._resumable = True
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
        self._wakes = self._new_wakes()
        self._code = 0
        self._stop_requested = False

    def _new_wakes(self) -> Wakes:
        playbook = self._playbook
        remind = playbook.reminders.for_lines if playbook is not None else None
        return Wakes(remind, self._words)

    def outcome(self) -> dict:
        """How the run ended, for its session's record."""
        return {
            "reason": self._end_reason,
            "spent_usd": self._spent_usd,
            "tokens": self._tokens,
            "turns": self._turns,
            "resumable": self._resumable,
        }

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

        def ring(line: str) -> None:
            self._items.put_nowait((self._watcher, line))

        if self._alarm is not None:
            self._alarm.bind(self._loop, ring)
        self._deadlines.bind(self._loop, ring)
        if self._stop_requested:
            return 0
        # The kickoff's own actions are news: the watch starts at the tail, so
        # it has to hold its place before the agent acts.
        if not await self._watch():
            return self._code
        try:
            await self._agent.start()
            prompt = kickoff(self._resumed, self._goal, self._words)
            while prompt is not None and await self._turn(prompt):
                prompt = await self._next_prompt()
        except Exception:
            # Ctrl-C reaches the CLI as well, which may end before its turn.
            if not self._stopping.is_set():
                raise
            log.debug("The agent's process ended as it stopped.", exc_info=True)
        finally:
            if self._alarm is not None:
                self._alarm.cancel()
            self._deadlines.cancel()
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

        watcher = Watcher(
            client,
            deliver,
            poll_s=self._poll_s,
            retry_s=self._retry_s,
            quiet_s=self._quiet_s,
        )
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
        await self._report_read(prompt)
        if getattr(self._agent, "compacted", False):
            # The context was compacted during the last turn: the plan and the
            # playbook's rules may be all that is left of the intake.
            self._agent.compacted = False
            prompt += "\n\nYour context was compacted. " + self._words.reread_plan
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
        if end.tokens is not None:
            self._tokens = usage.add(self._tokens or usage.zero(), end.tokens)
        if end.context is not None:
            self._context = end.context
        if self._stopping.is_set():
            return False
        session = end.session_id or getattr(self._agent, "session_id", None)
        if end.error in FATAL_ERRORS:
            self._end_reason = f"the model can't answer ({end.error})"
            self._resumable = False
            log.error(
                "The model can't answer (%s); fibsem-agent stops. Store a Claude "
                "API key with fibsem-cli key set %s (or configure Bedrock or "
                "Vertex), then start it again.",
                end.error,
                API_KEY,
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
            "tokens": self._tokens,
            "context": self._context,
            "turns": self._turns,
            **({} if self._goal_sent else self._about),
        }
        skills = list(getattr(self._agent, "skills_read", None) or [])
        if skills != self._skills_sent:
            body["skills"] = skills
            self._skills_sent = skills
        if self._playbook is not None:
            state = self._playbook.reminders.state()
            if state != self._playbook_sent:
                body["playbook"] = state["playbook"] or {}
                body["hooks"] = self._playbook.hooks()
                self._playbook_sent = state
        body = {key: value for key, value in body.items() if value is not None}
        if ended:
            body.update(ended=True, reason=self._end_reason)
        self._goal_sent = True

        def post():
            try:
                response = client.request(
                    "POST", "/app/agent/session", json=body, timeout=10.0
                )
                if response.status_code == 200:
                    return response.json().get("session")
            except Exception:
                log.debug("Reporting the session failed.", exc_info=True)
            return None

        reported = await asyncio.get_running_loop().run_in_executor(None, post)
        if isinstance(reported, dict) and not ended:
            self._deadlines.set(reported.get("deadlines"))
        await self._send_transcript()

    async def _report_read(self, prompt: str) -> None:
        """The operator's messages this turn carries are read: the web page
        stops showing them as waiting."""
        ids, client = watch_mod.message_ids(prompt), self._client
        if ids and client is not None:
            await asyncio.get_running_loop().run_in_executor(
                None, watch_mod.report_read, client, ids
            )

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
        self._wakes = self._new_wakes()
        back = (
            f"The app could not be watched for {time.monotonic() - started:.0f} s "
            "and the harness is watching it again: events in between were not "
            "seen. " + self._words.reread_level
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
                    "No fibsem host answered within %.0f s; fibsem-agent stops. "
                    "Start it again once the host is serving.",
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

    def __init__(self, options, usage_path: Optional[Path] = None) -> None:
        self._options = options
        self._client = None
        # Each model call's tokens, a line each in the session's usage file.
        self.calls = usage.Calls(usage_path)
        self._cost_usd: Optional[float] = None
        self.session_id: Optional[str] = options.resume or options.session_id
        self.model: Optional[str] = options.model
        self.introduced = False
        # The context was compacted: the next turn asks for the plan again.
        self.compacted = False
        # The skills it has read, in order: the session reports them.
        self.skills_read: List[str] = []
        # The lines printed, until the harness sends them (`Harness._send_transcript`).
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
        from claude_agent_sdk import AssistantMessage, ResultMessage, ToolUseBlock

        await self._client.query(prompt)
        end, error, result = TurnEnd(subtype="no_result"), None, None
        async for message in self._client.receive_response():
            _show(message, self)
            if isinstance(message, AssistantMessage):
                error = message.error or error
                self.calls.seen(
                    message.message_id,
                    message.model,
                    message.usage,
                    [
                        _short(block.name)
                        for block in message.content
                        if isinstance(block, ToolUseBlock)
                    ],
                    message.parent_tool_use_id,
                )
            if isinstance(message, ResultMessage):
                result = message
                end = TurnEnd(
                    message.subtype, message.total_cost_usd, message.session_id
                )
        end.error = error
        end.tokens = self.calls.end_turn()
        end.context = await self._context()
        if result is not None:
            log.info("%s", self._turn_line(result, end))
            if result.total_cost_usd is not None:
                self._cost_usd = result.total_cost_usd
        return end

    async def _context(self) -> Optional[Dict[str, int]]:
        """How full the context is, from the CLI; the last call's size, if
        the CLI can't say."""
        context = {"used": self.calls.context} if self.calls.context else None
        try:
            reply = await asyncio.wait_for(
                self._client.get_context_usage(), CONTEXT_WAIT_S
            )
        except Exception:  # noqa: BLE001 - the figure is a nicety
            log.debug("The context's size couldn't be read.", exc_info=True)
            return context
        found = {
            "used": reply.get("totalTokens"),
            "window": reply.get("rawMaxTokens") or reply.get("maxTokens"),
            "compact_at": reply.get("autoCompactThreshold"),
        }
        found = {key: int(value) for key, value in found.items() if value}
        return found if "used" in found else context

    def _turn_line(self, result, end: TurnEnd) -> str:
        """ "Turn ended: success, 12 steps, $0.41 ($12.30 so far). ..." """
        line = f"Turn ended: {result.subtype}, {result.num_turns} steps"
        cost = result.total_cost_usd
        if cost is not None and self._cost_usd is not None:
            line += f", ${cost - self._cost_usd:.4f} (${cost:.4f} so far)"
        elif cost is not None:
            line += f", ${cost:.4f} so far"
        counts = end.tokens or usage.zero()
        line += (
            f". Tokens: {usage.short(counts['input'])} in, "
            f"{usage.short(counts['cache_read'])} cached, "
            f"{usage.short(counts['cache_write'])} written, "
            f"{usage.short(counts['output'])} out"
        )
        if result.duration_ms:
            line += (
                f". {result.duration_ms / 1000:.0f} s, "
                f"{result.duration_api_ms / 1000:.0f} s of it the model"
            )
        context = usage.context_text(end.context)
        return f"{line}. {context}." if context else f"{line}."

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
    if name == WAKE_TOOL:
        minutes = arguments.get("minutes", "")
        return "Set", _clip(
            f"a wake-up in {minutes} min: {arguments.get('reason', '')}", 200
        )
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
            agent.compacted = True
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
                skill = block.input.get("skill") if block.name == "Skill" else None
                if skill and skill not in agent.skills_read:
                    agent.skills_read.append(skill)
                control = getattr(agent, "playbook", None)
                if skill and control is not None:
                    control.loaded(skill)
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


def find_checkout(given: Optional[str] = None) -> Optional[Path]:
    """The folder whose ``.claude`` holds the shipped skills: the checkout
    given, else the one this module runs from, else the current folder, else
    the copy in the package (`sources.find_root`)."""
    return sources.find_root(given, BASE_SKILL)


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
    max_turns: Optional[int] = None,
    max_budget_usd: Optional[float] = None,
    resume: Optional[str] = None,
    effort: Optional[str] = None,
    work: Optional[Path] = None,
    session_id: Optional[str] = None,
    code: bool = True,
    alarm: Optional[Alarm] = None,
    playbook: Optional[playbooks.Playbook] = None,
    transcript: Optional[Path] = None,
    cwd: Optional[Path] = None,
    control: Optional[PlaybookControl] = None,
    loaded: Optional[Loaded] = None,
    words: Optional[instruments.Texts] = None,
):
    """The SDK session: the sidecar as the only MCP server, the *loaded*
    skills and subagents (`session_names`) and nothing of the user's Claude
    Code config. Settings left None are the base playbook's. With ``work``, code
    mode: the code tools, sandboxed, in ``work`` (``code_mode.prepare`` has
    copied the skills there); without ``code``, the file tools only. The
    *playbook* the session starts on puts its rules in the prompt, and its
    reminders hold when there is no *control* to ask. With
    *transcript*, a folder, the CLI's transcript is copied there as it is
    written (`mirror.TranscriptMirror`)."""
    from claude_agent_sdk import ClaudeAgentOptions

    loaded = loaded or session_names(checkout, playbook)
    base = defaults()
    command = sidecar_command()
    server = {"type": "stdio", "command": command[0], "args": command[1:]}
    # In the CLI's environment, which the sidecar inherits: the server entry
    # goes on the CLI's command line, where anyone can list it.
    env = dict(CLI_ENV)
    if url:
        env[SERVER_URL_ENV] = url
    if token:
        env[SERVER_TOKEN_ENV] = token
    # The sidecar offers this instrument kind's tools only.
    kind = playbook.instrument if playbook is not None else None
    if kind:
        env[INSTRUMENT_ENV] = kind
    granted = ALLOWED_TOOLS + subagent_tools(loaded.agents)
    tools, allowed, denied = BUILT_IN_TOOLS, granted, DENIED_TOOLS
    settings = None
    if work is not None:
        env.update(code_mode.env(work))
        given = code_mode.tools(code)
        tools = BUILT_IN_TOOLS + given
        allowed = granted + code_mode.allowed_tools(work, code)
        denied = [tool for tool in DENIED_TOOLS if tool not in given]
        settings = code_mode.settings_json(work, {**os.environ, **env}, code)
    servers = {SERVER: server}
    if alarm is not None:
        servers[HARNESS_SERVER] = wake_server(alarm, control)
    return ClaudeAgentOptions(
        # Its .claude holds the session's skills (`sources.assemble`; CLAUDE.md
        # is off, see CLI_ENV), and none of ~/.claude: the user's own
        # settings, hooks and skills stay out.
        cwd=str(work or cwd or checkout),
        setting_sources=["project"],
        skills=list(loaded.skills),
        tools=list(tools),
        allowed_tools=list(allowed),
        disallowed_tools=list(denied),
        permission_mode=PERMISSION_MODE,
        settings=settings,
        mcp_servers=servers,
        env=env,
        # Only the server above: not a `fibsem` registered in ~/.claude.json.
        strict_mcp_config=True,
        system_prompt={
            "type": "preset",
            "preset": "claude_code",
            "append": system_prompt(work, code, playbook, words),
        },
        # With a playbook control, the reminders on now, asked at each call.
        hooks=(
            control.reminders.sdk_hooks(SERVER)
            if control is not None
            else reminders.sdk_hooks(playbook.hooks, SERVER)
            if playbook
            else None
        ),
        # Watcher lines carry operator-typed text: never expand an @path in it.
        verbatim_prompts=True,
        max_turns=max_turns or base.max_turns,
        max_budget_usd=max_budget_usd or base.max_budget_usd,
        model=model,
        effort=effort or base.effort,
        resume=resume,
        session_id=session_id,
        session_store=(
            mirror.TranscriptMirror(transcript, resume or session_id)
            if transcript is not None
            else None
        ),
        session_store_flush="eager",
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
# --code or --files without a folder: work/ in the session's folder.
IN_SESSION = ":session:"


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
        ("Playbook", getattr(args, "playbook", None) or "none"),
        ("Model", args.model or "Claude Code's default"),
        ("Goal", args.goal or "none: the skills decide"),
        (
            "Budget",
            f"${args.max_budget_usd:g} per run, {args.max_turns} tool calls "
            "per wake-up",
        ),
        ("Session", f"{session_id} ({'resumed' if args.resume else 'new'})"),
        ("Skills", _skills_line(checkout)),
        ("Code", code_mode.describe(work, code)),
        ("Log", str(log_path)),
    ]
    return [f"{label + ':':<10}{value}" for label, value in rows]


def _skills_line(checkout: Path) -> str:
    lab = sources.lab_skills()
    if not lab:
        return str(checkout)
    return f"{checkout}; the lab's own: {', '.join(lab)}"


def main(argv=None) -> int:
    refuse("fibsem-agent")
    parser = argparse.ArgumentParser(
        prog="fibsem-agent",
        description="Run the fibsem agent through the App Server, with no "
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
        default=None,
        help="the agent's reasoning effort (default: the playbook's); each "
        "subagent sets its own",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="tool calls the agent may make per wake-up (default: the playbook's)",
    )
    parser.add_argument(
        "--max-budget-usd",
        type=float,
        default=None,
        help="what this run may spend, after which fibsem-agent stops "
        "(default: the playbook's)",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=LAST,
        default=None,
        metavar="SESSION_ID",
        help="carry on a session; with no id, the last one run in the same "
        "folder (the --code or --files folder, else the checkout; with no "
        "folder given, the last one that worked in its own)",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        type=_session_id,
        metavar="SESSION_ID",
        help="start a new session with this id (a UUID), as a manager does so "
        "it can find the session's folder",
    )
    parser.add_argument(
        "--server-wait",
        type=float,
        default=SERVER_WAIT_S,
        help="seconds to wait for an App Server that isn't answering before "
        f"stopping (default {SERVER_WAIT_S:.0f})",
    )
    parser.add_argument(
        "--log",
        default=None,
        help="log file (default: in the session's folder, "
        "~/.fibsem/agent/sessions/<date>-<session id>/)",
    )
    parser.add_argument(
        "--checkout",
        default=None,
        help="the fibsem-os checkout whose .claude/skills the agent reads",
    )
    parser.add_argument(
        "--code",
        nargs="?",
        const=IN_SESSION,
        default=None,
        metavar="PATH",
        help="code mode: the agent writes and runs its own code in PATH (if "
        "none, work/ in the session's folder), sandboxed (macOS and Linux only)",
    )
    parser.add_argument(
        "--files",
        nargs="?",
        const=IN_SESSION,
        default=None,
        metavar="PATH",
        help="the agent reads and writes files in PATH (if none, work/ in the "
        "session's folder) and runs nothing: no shell",
    )
    parser.add_argument(
        "--playbook",
        default=None,
        metavar="NAME",
        help="start on the playbook NAME.yaml, from ~/.fibsem/knowledge/playbooks/ "
        "or the checkout's .claude/fibsem-agent/playbooks/ (default: "
        f"{playbooks.DEFAULT}); its session settings apply, and flags given here "
        "win over them",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="print what the agent would run with, and start nothing",
    )
    parser.add_argument(
        "--list-playbooks", action="store_true", help="list the playbooks found"
    )
    parser.add_argument(
        "--list-hooks",
        action="store_true",
        help="list when a playbook's reminders can fire",
    )
    parser.add_argument(
        "--json", action="store_true", help="with --show-config or --list-*: JSON"
    )
    args = parser.parse_args(argv)
    if args.code and args.files:
        parser.error("--code already gives the agent its files: use one of them")
    if args.session_id and args.resume:
        parser.error("--session-id starts a new session: --resume carries one on")
    if args.list_hooks:
        return _print(reminders.hook_rows(), args.json, _hook_text)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError, ValueError):
        pass
    checkout = find_checkout(args.checkout)
    if checkout is None:
        print(
            "fibsem-agent found no skills: this fibsem has none shipped and "
            "isn't a checkout; pass --checkout <path to a fibsem-os checkout>.",
            file=sys.stderr,
        )
        return 1
    if args.list_playbooks:
        return _print(_playbook_rows(checkout), args.json, _playbook_text)
    try:
        playbook = playbooks.load(args.playbook or playbooks.DEFAULT, checkout, EFFORTS)
    except playbooks.PlaybookError as error:
        print(f"{error} (fibsem-agent --list-playbooks).", file=sys.stderr)
        return 1
    _resolve(args, playbook.session)
    unset = [key for key in REQUIRED if getattr(playbook.session, key) is None]
    if unset:
        print(
            f"The {playbook.name} playbook has no {', '.join(unset)}, and the "
            f"shipped {playbooks.DEFAULT} playbook gives none: is it missing?",
            file=sys.stderr,
        )
        return 1
    try:
        words = instruments.load(playbook.instrument, checkout)
    except instruments.InstrumentError as error:
        print(error, file=sys.stderr)
        return 1
    # What it loads is fixed for the session: a playbook named at launch
    # narrows it (`session_names`).
    loaded = session_names(checkout, playbook if args.playbook else None)
    try:
        for name in loaded.agents:
            subagents.check_file(name, checkout, EFFORTS)
    except subagents.SubagentError as error:
        print(error, file=sys.stderr)
        return 1
    work = None
    code = bool(args.code)
    given = args.code or args.files
    in_session = given == IN_SESSION
    if given and not in_session:
        work = code_mode.workdir(given)
    if args.resume == LAST:
        own = sessions.WORK if in_session else sessions.RUN
        args.resume = sessions.last_session(work, own=own)
        if args.resume is None and work is None and not in_session:
            # A session from before run/: the CLI ran it in the checkout.
            args.resume = sessions.last_session(checkout)
        if args.resume is None:
            where = work or "its own folder"
            print(
                f"No earlier fibsem-agent session in {where} to resume: pass "
                "its id (the Session line of its log).",
                file=sys.stderr,
            )
            return 1
    # A new session's id is chosen here, so the status can name it.
    session_id = args.resume or args.session_id or str(uuid.uuid4())
    folder = sessions.session_folder(session_id)
    if in_session:
        work = folder / sessions.WORK
    # Without a work folder the CLI runs in the session's run/, which holds
    # only the session's skills; a session from before it keeps its folder.
    run = None if work is not None else _run_folder(args.resume, folder, checkout)
    if work is not None:
        refused = code_mode.refusal(work, checkout, code)
        if refused:
            print(refused, file=sys.stderr)
            return 1
    if args.show_config:
        config = show_config(args, checkout, work, code, playbook, loaded, words)
        return _print(config, args.json, _config_text)
    missing = missing_extra_message("agent", "fibsem-agent")
    if missing:
        print(missing, file=sys.stderr)
        return 1
    # From the environment, so it is not in the process list or shell history.
    token = server_env()[1]
    # The CLI reads the key from its environment, which it inherits from here.
    key = get_secret(API_KEY)
    if key:
        os.environ[API_KEY] = key
    session = sessions.SessionFolder.open(
        session_id, _about(args, checkout, work, code, playbook, run), folder=folder
    )
    log_path = Path(args.log).expanduser() if args.log else session.log_path
    print_fibsem_logo()
    _log_to(log_path)
    for line in status(args, checkout, log_path, work, session_id, code):
        log.info(line)
    if work is not None:
        code_mode.prepare(work, checkout, loaded.skills, loaded.agents)
        for line in code_mode.details():
            log.debug(line)
    elif run != checkout:
        sources.assemble(run, checkout, loaded.skills, loaded.agents)

    alarm = Alarm(playbook.session.wake_max_min)
    control = PlaybookControl(
        playbooks.Reminders(),
        checkout,
        keep=session.keep_playbook,
        loaded=loaded,
    )
    record = session.record() if args.resume else {}
    if record:
        why = control.reminders.restore(record, checkout)
        if why:
            log.warning("The session's playbook can't be picked up again: %s", why)
    # A resumed session keeps the playbook it had, or none if one was cleared.
    named = "playbook" not in record
    if named:
        control.reminders.set(
            playbook, "named at launch" if args.playbook else "the default"
        )
        session.keep_playbook(control.reminders.state())
    if control.reminders.playbook is not None:
        log.info("Playbook: %s", control.reminders.playbook.name)
    options = build_options(
        checkout,
        alarm=alarm,
        control=control,
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
        playbook=playbook,
        transcript=session.folder,
        cwd=run,
        loaded=loaded,
        words=words,
    )
    agent = SdkAgent(options, usage_path=session.folder / usage.FILE)
    agent.playbook = control
    control.say = lambda text: _line(agent, "context", text)
    if named:
        control.say(f"Playbook set: {playbook.name}, {control.reminders.chosen_by}")
    harness = Harness(
        agent,
        lambda: watch_mod.connect(args.url, token),
        resumed=args.resume is not None,
        server_wait_s=args.server_wait,
        goal=args.goal,
        budget_usd=args.max_budget_usd,
        playbook=control,
        words=words,
        deadline_warn_s=playbook.session.deadline_warn_min * 60,
        quiet_s=playbook.session.quiet_min * 60,
        about={
            "harness": "fibsem-agent",
            "code": work is not None and code,
            "files": work is not None and not code,
            "workdir": str(work) if work is not None else None,
            "effort": args.effort,
            "resumed": args.resume is not None,
        },
    )

    def on_interrupt(signum, frame):
        # A second Ctrl-C ends it at once.
        signal.signal(signal.SIGINT, signal.default_int_handler)
        harness.stop()

    previous = signal.signal(signal.SIGINT, on_interrupt)
    # A manager stops it with a file, which works where a signal can't reach.
    sessions.watch_for_stop(session, harness.stop)
    outcome = None
    try:
        code = asyncio.run(harness.run())
    except KeyboardInterrupt:
        code = 130
    except Exception as error:
        log.exception("The agent failed: %s", error)
        if agent.session_id:
            log.error("--resume %s carries the session on.", agent.session_id)
        code, outcome = 1, {"reason": f"the agent failed: {error}"}
    finally:
        signal.signal(signal.SIGINT, previous)
    session.end(code, **(outcome or harness.outcome()))
    return code


def _session_id(text: str) -> str:
    try:
        return str(uuid.UUID(text))
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a UUID: {text}") from None


def _run_folder(resume: Optional[str], folder: Path, checkout: Path) -> Path:
    """Where a session without a work folder runs: its folder's run/, or for
    one resumed from before run/ the folder its record says it ran in."""
    if resume:
        ran = sessions.read_json(folder / sessions.RECORD).get("cwd")
        if ran and Path(ran) == checkout:
            return checkout
    return folder / sessions.RUN


def _about(
    args,
    checkout: Path,
    work: Optional[Path],
    code: bool,
    playbook: Optional[playbooks.Playbook] = None,
    run: Optional[Path] = None,
) -> dict:
    """What a session runs with, for its record."""
    from fibsem import __version__

    return {
        "fibsem": __version__,
        "cwd": str(work or run or checkout),
        "url": args.url,
        "goal": args.goal,
        "model": args.model,
        "effort": args.effort,
        "max_budget_usd": args.max_budget_usd,
        "max_turns": args.max_turns,
        "mode": ("code" if code else "files") if work is not None else "none",
        "checkout": str(checkout),
        "playbook": playbook.name if playbook else None,
        "playbook_hash": playbook.hash if playbook else None,
    }


# What a session can't start without; the shipped base playbook sets each.
REQUIRED = (
    "effort",
    "max_budget_usd",
    "max_turns",
    "wake_max_min",
    "deadline_warn_min",
    "quiet_min",
)


def _resolve(args, given: playbooks.Session) -> None:
    """Fill what the command line left out from the playbook's session
    settings, which carry the base playbook's defaults."""
    if args.model is None:
        args.model = given.model
    if args.effort is None:
        args.effort = given.effort
    if args.max_turns is None:
        args.max_turns = given.max_turns
    if args.max_budget_usd is None:
        args.max_budget_usd = given.max_budget_usd
    if not (args.code or args.files) and given.mode not in (None, "none"):
        folder = str(Path(given.folder).expanduser()) if given.folder else IN_SESSION
        if given.mode == "code":
            args.code = folder
        else:
            args.files = folder


def show_config(
    args,
    checkout: Path,
    work: Optional[Path],
    code: bool,
    playbook: playbooks.Playbook,
    loaded: Optional[Loaded] = None,
    words: Optional[instruments.Texts] = None,
) -> dict:
    """``--show-config``: everything the agent would run with, from the same
    pieces `build_options` uses, without the SDK and without starting."""
    loaded = loaded or session_names(checkout, playbook)
    granted = ALLOWED_TOOLS + subagent_tools(loaded.agents)
    tools, allowed, denied = BUILT_IN_TOOLS, granted, DENIED_TOOLS
    if work is not None:
        given = code_mode.tools(code)
        tools = BUILT_IN_TOOLS + given
        allowed = granted + code_mode.allowed_tools(work, code)
        denied = [tool for tool in DENIED_TOOLS if tool not in given]
    return {
        "level": playbook.session.level,
        "description": playbook.description,
        "model": args.model,
        "effort": args.effort,
        "max_budget_usd": args.max_budget_usd,
        "max_turns": args.max_turns,
        "wake_max_min": playbook.session.wake_max_min,
        "deadline_warn_min": playbook.session.deadline_warn_min,
        "quiet_min": playbook.session.quiet_min,
        "instrument": words.kind if words else playbook.instrument,
        "mode": ("code" if code else "files") if work is not None else "none",
        "folder": str(work) if work is not None else None,
        "checkout": str(checkout),
        "skills": list(loaded.skills),
        "specialists": list(loaded.agents),
        "tools": list(tools),
        "allowed_tools": list(allowed),
        "denied_tools": list(denied),
        # Its reminders are on from the start; the agent may propose another.
        "playbook": {
            **playbook.summary(),
            "path": str(playbook.path),
            "hash": playbook.hash,
        },
        "system_prompt": system_prompt(work, code, playbook, words),
    }


def _playbook_rows(checkout: Path) -> List[dict]:
    rows = []
    for name in playbooks.available(checkout):
        try:
            found = playbooks.load(name, checkout, EFFORTS)
        except playbooks.PlaybookError as error:
            rows.append({"name": name, "error": str(error)})
            continue
        rows.append(
            {
                "name": found.name,
                "path": str(found.path),
                "hash": found.hash,
                "description": found.description,
                "level": found.session.level,
                "mode": found.session.mode or "none",
            }
        )
    return rows


def _print(data, as_json: bool, as_text) -> int:
    print(json.dumps(data, indent=2) if as_json else as_text(data))
    return 0


def _hook_text(rows: List[dict]) -> str:
    return "\n".join(
        f"{row['when']:<18}{row['label']}, {row['fires']}: {row['tools']}"
        f" ({row['instrument']} instrument"
        f"{', on when a playbook names it' if row['by_playbook'] else ''})"
        for row in rows
    )


def _playbook_text(rows: List[dict]) -> str:
    if not rows:
        return "No playbooks."
    return "\n".join(
        f"{row['name']:<26}"
        + (f"error: {row['error']}" if "error" in row else f"{row['mode']:<6}")
        + (f"  {row['path']}" if "path" in row else "")
        for row in rows
    )


def _config_text(config: dict) -> str:
    playbook = config["playbook"]
    lines = [
        f"Playbook: {' <- '.join(playbook['extends'] + [playbook['name']])} "
        f"({playbook['path']}, {playbook['hash']})",
        f"Model:    {config['model'] or 'the CLI default'}, {config['effort']} effort",
        f"Budget:   ${config['max_budget_usd']:g} per run, "
        f"{config['max_turns']} tool calls per wake-up",
        f"Timers:   wake-ups up to {config['wake_max_min']:g} min, deadlines "
        f"warned {config['deadline_warn_min']:g} min before, a quiet workflow "
        f"looked at after {config['quiet_min']:g} min",
        f"Instrument: {config['instrument']}",
        f"Mode:     {config['mode']}"
        + (f" in {config['folder']}" if config["folder"] else ""),
        f"Skills:   {', '.join(config['skills'])}",
        f"Agents:   {', '.join(config['specialists'])}",
        f"Tools:    {', '.join(config['tools'])}",
        f"Allowed:  {', '.join(config['allowed_tools'])}",
        f"Denied:   {', '.join(config['denied_tools'])}",
        f"Playbook skills: {', '.join(playbook['skills']) or 'none'}",
    ]
    lines += [f"  When needed: {o['skill']}, {o['when']}" for o in playbook["optional"]]
    lines.append("Reminders:")
    lines += [f"  {r['label']}: {r['say']}" for r in playbook["reminders"]] or [
        "  none"
    ]
    lines += [f"Rule: {rule}" for rule in playbook["rules"]]
    return "\n".join(lines + ["", "System prompt (appended):", config["system_prompt"]])


if __name__ == "__main__":
    sys.exit(main())
