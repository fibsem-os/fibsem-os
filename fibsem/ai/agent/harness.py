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
import collections
import functools
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
import traceback
import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from fibsem import diagnostics
from fibsem.ai.agent import (
    aside,
    code_mode,
    extra_mcp,
    host_link,
    instruments,
    mirror,
    playbooks,
    reminders,
    sources,
    subagents,
    usage,
    wake_lines,
    wake_policy,
)
from fibsem.ai.agent import handoff as handoff_mod
from fibsem.ai.agent import manager as sessions
from fibsem.ai.agent import report as report_mod
from fibsem.ai.agent import watch as watch_mod
from fibsem.ai.agent.catalog import INSTRUMENT_ENV
from fibsem.ai.agent.report import SessionReport, plain
from fibsem.ai.agent.session import BUDGET_WARNING
from fibsem.ai.agent.wake_lines import fibsem as fibsem_lines
from fibsem.client.transport import (
    HOST_TOKEN_ENV,
    SERVER_TOKEN_ENV,
    SERVER_URL_ENV,
    server_env,
    unreachable_why,
)
from fibsem.codec import decode
from fibsem.extras import missing_extra_message
from fibsem.keystore import get_secret
from fibsem.logo import print_fibsem_logo
from fibsem.server.discovery import ApiMismatch, check_api, code_mismatch
from fibsem.unreleased import refuse
from fibsem.util import timestamps

# The skill that marks a folder as holding the shipped .claude (`find_checkout`).
# Which skills and subagents a session loads is its playbooks' (`session_names`).
BASE_SKILL = "fibsem-agent"

SERVER = "fibsem"
# With an agent host (--host), the agent's record tools on the host: its plan,
# notes and messages (`catalog.RECORD_TOOLS`), from a second sidecar.
RECORD_SERVER = "fibsem_record"
# The harness's own tools, in-process: the wake-up timer (`Alarm`).
HARNESS_SERVER = "fibsem_agent"
WAKE_TOOL = f"mcp__{HARNESS_SERVER}__wake_after"
# What wakes the agent (`wake_policy.WakeOn`), on the same in-process server.
WAKE_ON_TOOL = f"mcp__{HARNESS_SERVER}__wake_on"
# The playbook tools (`PlaybookControl`), on the same in-process server.
PLAYBOOK_TOOLS = [
    f"mcp__{HARNESS_SERVER}__{name}"
    for name in ("playbook_list", "playbook_set", "skill_done")
]
# The fresh-context tool (`handoff`), on the same in-process server.
HANDOFF_TOOL = f"mcp__{HARNESS_SERVER}__{handoff_mod.TOOL}"
# The server's level gate decides what the fibsem tools may do, so they need no
# prompt; ToolSearch only finds them. The session's subagents are added
# (`subagent_tools`).
ALLOWED_TOOLS = (
    [f"mcp__{SERVER}__*", "ToolSearch", "SendMessage"]
    + [WAKE_TOOL, WAKE_ON_TOOL]
    + PLAYBOOK_TOOLS
    + [HANDOFF_TOOL]
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
    # The agent's prompt cache lasts an hour, not the CLI's 5 min: a Watch
    # session waits longer than that between wakes (`wake_policy`).
    "CLAUDE_CODE_PROMPT_CACHE_TTL": wake_policy.PROMPT_CACHE_TTL,
}

# One message from the CLI: a tool result carrying an image (the Read tool on
# a plot, in code mode) is more than the SDK's 1 MB default.
MAX_BUFFER_SIZE = 32 * 1024 * 1024
SERVER_WAIT_S = 600.0
# How long the CLI has to say how full the context is, after a turn.
CONTEXT_WAIT_S = 10.0
# The transcript (`Harness._send_transcript`): sent this soon after a line is
# printed, so a burst goes together, and again this often while the server
# can't take it; at most this many lines at once, at most this many kept while
# the server can't take them.
TRANSCRIPT_SOON_S = 1.0
TRANSCRIPT_EVERY_S = 10.0
TRANSCRIPT_BATCH = 200
TRANSCRIPT_KEEP = 2000
# A lab tool's pictures on the transcript (`_show_pictures`): at most this many
# from one result, this many waiting to be sent, and this big (base64, under
# the server's 5 MB). A send carries one picture at most.
PICTURES_PER_RESULT = 4
PICTURES_WAITING = 20
PICTURE_MAX_B64 = 6_800_000
# The kinds the server keeps (`fibsem.server.agent.transcript.PICTURE_TYPES`).
PICTURE_TYPES = ("image/png", "image/jpeg", "image/gif", "image/webp")
# The tools that start a subagent: its run's lines carry the call's number.
SUBAGENT_TOOLS = ("Agent", "Task")
# How much of a subagent's answer its run keeps.
SUBAGENT_ANSWER = 1500
# While a turn goes on, its tokens and the context's fill are reported this
# often (when they changed), not only as it ends: a turn can run for an hour.
REPORT_EVERY_S = 30.0
RETRY_MAX_S = 60.0

NO_CODE = """\
- Your tools are the mcp__fibsem__* tools, Skill, and Agent for the subagents \
the playbook names. There is no shell, no file access and no web, for \
you or for them. Where a skill shows curl, python or a script, use the \
matching mcp__fibsem__ tool, or say what you can't do."""


def defaults() -> playbooks.SessionSettings:
    """The shipped base playbook's session settings: what a timer or a
    watcher built without them uses."""
    base = playbooks.shipped_base(find_checkout())
    return base.session if base is not None else playbooks.SessionSettings()


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
    extras: Optional[Dict[str, extra_mcp.Server]] = None,
    servers: Optional[List[str]] = None,
) -> str:
    """What the harness tells the agent about itself; with a work folder (code
    mode, or files only), about it too; and about the lab's extra MCP servers
    (*extras*, else read), if any. *servers*, the session's MCP servers, are
    named in place of the fibsem sidecar's when they are others."""
    words = words or texts()
    fibsem = not servers or SERVER in servers
    tools = NO_CODE if work is None else code_mode.prompt(work, code, fibsem)
    tools = _name_servers(tools, servers)
    more = extra_mcp.prompt_lines(extra_mcp.load() if extras is None else extras)
    if more:
        tools += "\n" + more
    warn = deadline_warn_min or defaults().deadline_warn_min
    return words.prompt.format(tools=tools, deadline_warn_min=f"{warn:g}")


def _name_servers(tools: str, servers: Optional[List[str]]) -> str:
    """*tools*, the prompt's tool line, naming *servers* where it names the
    fibsem sidecar, when the session's servers are others (an agent host)."""
    if not servers or servers == [SERVER]:
        return tools
    named = ", ".join(f"mcp__{name}__*" for name in servers)
    return (
        tools.replace("the mcp__fibsem__* tools", f"the {named} tools")
        .replace("the matching mcp__fibsem__ tool", "the matching tool")
        .replace("so the mcp__fibsem__* tools", "so those tools")
        .replace("can't reach the App Server or", "can't reach your servers or")
    )


def system_prompt(
    work: Optional[Path] = None,
    code: bool = True,
    playbook: Optional[playbooks.Playbook] = None,
    words: Optional[instruments.Texts] = None,
    extras: Optional[Dict[str, extra_mcp.Server]] = None,
    servers: Optional[List[str]] = None,
) -> str:
    """The harness's prompt, and the rules of the playbook the session
    started on after it, so they are read every turn."""
    warn = playbook.session.deadline_warn_min if playbook is not None else None
    prompt = harness_prompt(work, code, words, warn, extras, servers)
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


# The CLI's last lines on stderr: why a resumed conversation would not load.
CLI_SAID: "collections.deque[str]" = collections.deque(maxlen=50)
# What the CLI says when it can't load a conversation: none by that id, one too
# large for it, or a file it can't read. Only these start a fresh context on a
# resume; any other failure stops as before.
CANNOT_LOAD = re.compile(
    r"no conversation found|session\b.*\bnot found|invalid string length"
    r"|heap out of memory|allocation failed|unexpected (token|end of json)"
    r"|failed to (load|parse|read|resume)",
    re.IGNORECASE,
)


def _cli_said(line: str) -> None:
    log.debug("claude: %s", line.rstrip())
    CLI_SAID.append(line.rstrip())


def _last_said() -> str:
    """The CLI's last line that says why it can't load, shortened."""
    for line in reversed(CLI_SAID):
        if CANNOT_LOAD.search(line):
            return line.strip()[:200]
    return ""


def cannot_load(*texts: str) -> bool:
    """Whether *texts* (an error, the CLI's last lines) say the conversation
    itself can't be loaded."""
    return any(CANNOT_LOAD.search(text or "") for text in texts)


def _reload_failed(error: BaseException) -> str:
    why = str(error).strip().splitlines()[0] if str(error).strip() else ""
    return handoff_mod.RELOAD_FAILED.format(why=why[:200] or type(error).__name__)


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
    digest: Optional[str] = None,
    notes: Optional[List[str]] = None,
) -> str:
    """The watcher's lines as one turn, with what a GAP or a level change
    asks of the agent first, and the reminders *remind* the lines call for.
    The *digest* of held lines (`wake_policy`) comes first, and *notes* about
    what wakes the agent right after the lines."""
    words = words or texts()
    parts = [digest] if digest else []
    if lines:
        parts.append("The watcher printed:\n" + "\n".join(lines))
    parts += [note for note in notes or [] if note]
    prompt = "\n\n".join(parts)
    if remind:
        prompt += "\n\n" + reminders.reminder(remind)
    said = watch_mod.kinds(lines)
    if watch_mod.GAP in said:
        prompt += "\n\n" + words.reread_proposals
    if said & {
        watch_mod.AGENT_BACK,
        watch_mod.RESTARTED,
        watch_mod.LEVEL,
        watch_mod.LEVEL_LOWERED,
        watch_mod.OPERATOR_STOP,
    }:
        prompt += "\n\n" + words.reread_level
    if said & {ALARM, watch_mod.PLAN_EDITED}:
        prompt += "\n\n" + words.reread_plan
    return prompt


def context_parts(categories) -> List[Dict[str, Any]]:
    """The CLI's ``/context`` categories as the session reports them: each
    part's name and tokens, without the free space and the tools only named
    (deferred), at most `report_mod.MAX_PARTS`."""
    parts = []
    for category in categories if isinstance(categories, list) else []:
        if not isinstance(category, dict) or category.get("isDeferred"):
            continue
        name = " ".join(str(category.get("name") or "").split())[: report_mod.MAX_ITEM]
        tokens = category.get("tokens")
        if (
            not name
            or name.lower() == "free space"
            or isinstance(tokens, bool)
            or not isinstance(tokens, (int, float))
            or tokens <= 0
        ):
            continue
        parts.append({"name": name, "tokens": int(tokens)})
    return parts[: report_mod.MAX_PARTS]


# The kinds of the harness's own wake-ups: the agent's timer, and a deadline.
ALARM = "alarm"
DEADLINE = "deadline"

# The wake-ups that cut the agent's turn short: news that makes the turn's
# work wrong, not news that is only new. Everything else waits for the turn
# to end.
URGENT = frozenset(
    {
        # The operator wrote to the agent.
        watch_mod.MESSAGE,
        # The level went down, or the operator took control back.
        watch_mod.LEVEL_LOWERED,
        watch_mod.AGENT_OFF,
        watch_mod.OPERATOR_STOP,
        # A run stopped or failed, or the hardware failed. The operator's
        # stop is OPERATOR_STOP above.
        watch_mod.WORKFLOW_FAILED,
        "task_failed",
        watch_mod.COMMAND_FAILED,
        fibsem_lines.MILL_FAILED,
        # A milling monitor's stop rule fired.
        fibsem_lines.STOP_RULE,
    }
)

# Said first in the turn after one cut short. A call cut short is not undone:
# the sidecar finishes its request and the App Server the action, but the
# agent never sees the answer.
INTERRUPTED = (
    "Your last turn was cut short by a line below. A call you were waiting on "
    "then may have finished on the instrument after you stopped waiting, and "
    "its answer was lost: read where things stand (the stage's position, "
    "app_get_status, service_wait_for_run with wait=0 for a long command you "
    "started) before you act. Never repeat a relative move, or any call that "
    "changes something by an amount, until you have checked it did not "
    "already happen."
)


CARRY_ON = (
    "WAKE your last turn reached its step limit before you finished: carry on "
    "from where you were."
)

# Said once, in the next turn, when the run's spend first reaches
# `BUDGET_WARNING` of its budget: the CLI is given no budget, so it says none.
BUDGET_LOW = (
    "Budget: ${spent:.2f} of this run's ${budget:g} is spent ({used:.0f}%). "
    "fibsem-agent stops at the end of the turn that spends the rest: tell the "
    "operator now if the work needs more."
)


def is_urgent(line) -> bool:
    """Whether *line* cuts the agent's turn short (`URGENT`)."""
    return getattr(line, "kind", None) in URGENT


class Wakes:
    """The watcher's lines, held while the operator has the agent at Off and
    given to the agent as one turn otherwise. The wake policy (*wake_on*,
    `wake_policy.WakeOn`) holds routine lines for a digest."""

    def __init__(
        self,
        remind: Optional[Callable[[List[str]], List[str]]] = None,
        words: Optional[instruments.Texts] = None,
        waking: Optional[Callable[[], Optional[frozenset]]] = None,
        wake_on: Optional[wake_policy.WakeOn] = None,
        sizes_said: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.off = False
        # Whether the next turn already says the context's size (a mark line,
        # `handoff`): the policy's note about it then waits.
        self._sizes_said = sizes_said
        self._wake_on = wake_on if wake_on is not None else wake_policy.WakeOn()
        # What the policy says about a level line, for the next turn.
        self._notes: List[str] = []
        # The line kinds the playbook lets wake the agent (`wake_lines.waking`),
        # read at each line: None for every line.
        self._waking = waking
        self._held: List[str] = []
        # The operator's messages in the turns given, until they are read.
        self.unread: List[str] = []
        # The reminders a turn's lines call for (`playbooks.Reminders.for_lines`).
        self._remind = remind
        self._words = words

    def add(self, lines: List[str]) -> Optional[str]:
        """The next turn's prompt, or None when there is nothing to say yet."""
        policy = self._wake_on
        for line in lines:
            kind = getattr(line, "kind", None)
            if kind == wake_policy.DIGEST_DUE:
                continue  # the policy knows the digest is due
            if kind == watch_mod.AGENT_OFF:
                self.off = True
            elif kind == watch_mod.AGENT_BACK:
                if not self.off:
                    # The level came back before the agent heard it go.
                    continue
                self.off = False
            level = (getattr(line, "facts", None) or {}).get("level")
            if level:
                note = policy.at_level(level)
                if note:
                    self._notes.append(note)
            if policy.holds(line):
                policy.keep_held(line)
                continue
            self._held.append(line)
        if self.off:
            return None
        due = policy.due()
        if not due and (not self._held or not self._wakes_now()):
            return None
        lines, self._held = self._held, []
        held, digest = policy.take(due or "with this wake")
        for line in lines:
            self.unread += getattr(line, "message_ids", ())
        remind = self._remind(held + lines) if self._remind else None
        notes, self._notes = self._notes, []
        if not (self._sizes_said and self._sizes_said()):
            notes.append(policy.idle_note())
        return turn_prompt(lines, remind, self._words, digest, notes)

    def _wakes_now(self) -> bool:
        """Whether a held line wakes the agent: any, unless the playbook names
        its ``wakes``; then one of those, an urgent one or one that always does
        (`wake_lines.ALWAYS`). The rest wait for the next turn."""
        kinds = self._waking() if self._waking is not None else None
        if kinds is None:
            return True
        for line in self._held:
            kind = getattr(line, "kind", None)
            if kind is None or kind in kinds or kind in wake_lines.ALWAYS:
                return True
            if kind in URGENT:
                return True
        return False

    def read(self) -> List[str]:
        """The ids of the operator's messages in the turns given since the
        last call: read, once their turn starts."""
        ids, self.unread = self.unread, []
        return ids


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
        line = watch_mod.Wake(
            ALARM, f"WAKE {self.reason} (you asked {minutes:g} min ago)"
        )
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
                    watch_mod.Wake(
                        DEADLINE,
                        f"DEADLINE in {self.warn_s / 60:g} min: {text} (by {at})",
                    ),
                ),
                (0.0, watch_mod.Wake(DEADLINE, f"DEADLINE now: {text} (by {at})")),
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


def wake_server(
    alarm: Alarm,
    control: Optional[PlaybookControl] = None,
    handoff: Optional[handoff_mod.Handoff] = None,
    wake_on: Optional[wake_policy.WakeOn] = None,
):
    """The harness's in-process MCP server: the ``wake_after`` tool on
    *alarm*, with *control* the playbook tools, with *handoff* the
    ``context_reset`` tool, and with *wake_on* the ``wake_on`` tool."""
    from claude_agent_sdk import create_sdk_mcp_server, tool

    more = []
    if handoff is not None:

        @tool(handoff_mod.TOOL, handoff_mod.DESCRIPTION, handoff_mod.SCHEMA)
        async def context_reset(args):
            said = handoff.request(args or {})
            return {"content": [{"type": "text", "text": said}]}

        more.append(context_reset)

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

    own = [wake_after]
    if wake_on is not None:
        own.append(wake_on_tool(wake_on))
    if control is None:
        return create_sdk_mcp_server(HARNESS_SERVER, tools=own + more)

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
        HARNESS_SERVER,
        tools=own + [playbook_list, playbook_set, skill_done] + more,
    )


def wake_on_tool(policy: wake_policy.WakeOn):
    """The ``wake_on`` tool on *policy*: called with nothing, it answers what
    wakes the agent now."""
    from claude_agent_sdk import tool

    @tool(
        "wake_on",
        "What wakes you, and what waits for a digest. Called with nothing, it "
        "says the policy now and how many lines are held. mode: 'all' (every "
        "watcher line wakes you), 'digest' (routine lines, such as a task "
        "finishing normally or an autofocus that passed, wait and come "
        "together at the top of a later wake), or 'default' (the playbook's "
        "for your level). every_min and after_lines: a digest comes this long "
        "after the first line was held, or once this many are held (0: never "
        "on that ground). wake / hold: routine line kinds, comma separated, "
        "that wake you at once under digest, or wait under all ('none' "
        "clears). Messages, the level, proposals, failures and your own "
        "wake-ups always wake you. It holds until you change it.",
        {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["all", "digest", "default"]},
                "every_min": {"type": "number", "minimum": 0},
                "after_lines": {"type": "integer", "minimum": 0},
                "wake": {"type": "string"},
                "hold": {"type": "string"},
            },
        },
    )
    async def wake_on(args):
        given = {k: v for k, v in args.items() if v not in (None, "")}
        if not given:
            said = policy.describe()
        else:
            said = policy.set(
                mode=str(given.get("mode", "")),
                every_min=given.get("every_min"),
                after_lines=given.get("after_lines"),
                wake=str(given.get("wake", "")),
                hold=str(given.get("hold", "")),
            )
        return {"content": [{"type": "text", "text": said}]}

    return wake_on


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


@dataclass
class _Cursor:
    """The id of the newest event the watch has read: its lines came before
    this. Kept once the turn carrying them is over, so a resumed session
    watches from there (`watch.watch` ``after``)."""

    event_id: str


def _last_line(batch) -> bool:
    """Whether the watch said it is giving up (`watch.LAST_KINDS`)."""
    return bool(watch_mod.kinds(batch) & watch_mod.LAST_KINDS)


# how long stopping the watch waits for a poll in flight (s)
STOP_JOIN_S = 5.0


class _WatchClient:
    """The watch's client: it refuses another request once stopped."""

    def __init__(self, client, stopping: threading.Event) -> None:
        self._client = client
        self._stopping = stopping

    def request(self, method, path, *args, **kwargs):
        if self._stopping.is_set():
            raise _Stopped
        return self._client.request(method, path, *args, **kwargs)


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
        after: Optional[str] = None,
        lines: Optional[wake_lines.Lines] = None,
    ) -> None:
        self._client = client
        self._deliver = deliver
        # The instrument kind's own lines (`wake_lines`); the FIB-SEM's for None.
        self._lines = lines
        # An event id to watch from, for a resumed session; None starts at the tail.
        self._after = after
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
        # wait for a poll in flight to come back, so its connection is closed
        # with the client instead of left open after it
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(STOP_JOIN_S)

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
        client = _WatchClient(self._client, self._stopping)
        try:
            # The harness marks a message read when the turn carrying it starts.
            code = watch_mod.watch(
                client,
                out=self._out,
                sleep_fn=self._sleep,
                poll_s=self._poll_s,
                quiet_s=self._quiet_s,
                report_reads=False,
                on_seeded=lambda: self._deliver(_Seeded()),
                after=self._after,
                on_cursor=lambda event_id: self._deliver(_Cursor(event_id)),
                lines=self._lines,
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


def _level_of(client) -> Optional[str]:
    """The agent level the App Server reports (``/capabilities``), or None."""
    try:
        response = client.request("GET", "/capabilities", timeout=10.0)
        level = response.json().get("level") if response.status_code == 200 else None
    except Exception:  # noqa: BLE001 - an unknown level takes the policy for none
        return None
    return level if isinstance(level, str) else None


def _close(client) -> None:
    """Close a client the harness is done with, and its pooled connections."""
    try:
        client.close()
    except Exception:
        log.debug("Closing a client failed.", exc_info=True)


def _reachable(client) -> bool:
    """Whether the App Server answers this client, at any level; a warning
    when it runs other fibsem code than fibsem-agent. Raises `ApiMismatch`
    when its routes are another version's."""
    try:
        response = client.request("GET", "/capabilities", timeout=10.0)
    except Exception:
        return False
    if response.status_code == 200:
        try:
            capabilities = response.json()
        except Exception:  # noqa: BLE001 - not an App Server's answer
            return False
        check_api(capabilities, "fibsem-agent")
        try:
            warning = code_mismatch(capabilities, "fibsem-agent")
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
    spend, so the operator sees them in the app. While a turn goes on, its
    tokens and the context's fill are reported every ``report_every_s``; the
    spend is known only as a turn ends.

    The budget is the harness's to keep, not the CLI's: a turn that ends with
    this run's spend at or over it ends the run, so that last turn can go
    over. The operator may set a new budget at the App Server
    (``PUT /app/agent/budget``); the harness takes it from the answer to a
    report, and asks again before it ends the run. *spent_before_usd* is the
    session's spend when this run started (its record's, on a resume): the
    CLI's running total carries earlier runs on, the budget counts this one.

    With *handoff*, the agent may start a fresh context (`handoff`): when the
    turn that asked for one ends, the agent's ``new_context()`` starts a new
    CLI conversation under the same session id, *keep_context* records it,
    and the next turn starts with the handoff prompt. Lines that came in
    meanwhile wait in the queue for that turn. The CLI's running total starts
    again with each conversation: *spent_carried_usd* is the session's spend
    before the current one, added to it.
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
        spent_before_usd: Optional[float] = None,
        about: Optional[dict] = None,
        alarm: Optional[Alarm] = None,
        playbook: Optional[PlaybookControl] = None,
        words: Optional[instruments.Texts] = None,
        deadline_warn_s: Optional[float] = None,
        quiet_s: Optional[float] = None,
        url: Optional[str] = None,
        after: Optional[str] = None,
        keep_cursor: Optional[Callable[[str], None]] = None,
        keep_where: Optional[Callable[[dict], None]] = None,
        report_every_s: float = REPORT_EVERY_S,
        handoff: Optional[handoff_mod.Handoff] = None,
        keep_context: Optional[Callable[[str, float], None]] = None,
        spent_carried_usd: float = 0.0,
        wake_on: Optional[wake_policy.WakeOn] = None,
        recover: Optional[str] = None,
        latest_note: Optional[Callable[[], Optional[str]]] = None,
    ) -> None:
        self._agent = agent
        # Fresh contexts (`handoff`); None for a session that can't start one.
        self._handoff = handoff
        # A resume that starts in a fresh context, and why (`_recover`); a
        # resume whose reload fails starts one too, once.
        self._recover_why = recover
        self._recovery_tried = recover is not None
        # Reads the session's latest handoff note, for a fresh context
        # fibsem-agent starts itself.
        self._latest_note = latest_note or (lambda: None)
        self._keep_context = keep_context
        self._carried_usd = spent_carried_usd
        # What wakes the agent; with none given, every line does.
        self._wake_on = wake_on if wake_on is not None else wake_policy.WakeOn()
        if self._wake_on.context is None:
            self._wake_on.context = self._idle_and_context
        # When the agent's last turn ended (monotonic), for the wake policy.
        self._turn_ended_at: Optional[float] = None
        self._report_every_s = report_every_s
        # A turn is going on: its tokens so far are the agent's to say.
        self._in_turn = False
        # Reports go one at a time, in order: (its loop, the lock).
        self._reporting: Optional[Tuple[Any, asyncio.Lock]] = None
        # The tokens and context last reported while a turn went on.
        self._live_sent: Optional[tuple] = None
        # Stores the instrument and experiment the App Server names in its
        # answer to a report, in the session's own record.
        self._keep_where = keep_where
        # The App Server's address when given (--url): a waiting line says
        # what to check there, such as an SSH tunnel.
        self._url = url
        # A resumed session's first watch starts after the last event its
        # agent was given (*after*); *keep_cursor* stores that id after each
        # turn, for the next resume.
        self._after = after
        self._keep_cursor = keep_cursor
        self._cursor_given: Optional[str] = None
        # The newest cursor given before the wake policy's first held line:
        # what a resume may start after while lines wait for a digest.
        self._cursor_unheld: Optional[str] = None
        self._cursor_kept: Optional[str] = after
        # What the agent is told, for the session's instrument kind.
        self._words = words or texts()
        self._quiet_s = quiet_s
        # The session's playbook: sent again when it or its skills change.
        self._playbook = playbook
        self._playbook_sent: Optional[dict] = None
        # The agent's wake-up timer: it rings into the watcher's lines.
        self._alarm = alarm
        self._deadlines = Deadlines(deadline_warn_s)
        # The transcript's sends go one at a time: (its loop, the lock).
        self._sending: Optional[Tuple[Any, asyncio.Lock]] = None
        self._goal = goal
        self._goal_sent = False
        # The skills read, as last reported: sent again when the agent reads one.
        self._skills_sent: List[str] = []
        # What it can do (code mode, effort, skills), sent with the first report.
        self._about = dict(about or {})
        self._budget_usd = budget_usd
        # The CLI's running total when this run started: 0 for a new session;
        # None for a resume whose record doesn't say, and then this run's
        # first turn is not counted (`_count_spend`).
        self._run_from_usd: Optional[float] = spent_before_usd if resumed else 0.0
        # `BUDGET_LOW` is said once per crossing; it waits here for a turn.
        self._budget_warned = False
        self._budget_note: Optional[str] = None
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
        # Set when an urgent line arrives; cleared as the lines are taken.
        self._urgent: Optional[asyncio.Event] = None
        # The last turn was cut short by an urgent line.
        self._cut_short = False
        self._wakes = self._new_wakes()
        # The operator's side questions (/btw): answered beside the turns.
        self._asides = aside.Asides(agent, lambda: self._client)
        self._code = 0
        self._stop_requested = False

    def _new_wakes(self) -> Wakes:
        playbook = self._playbook
        remind = playbook.reminders.for_lines if playbook is not None else None

        def waking():
            # The playbook now: playbook_set may change it mid session.
            current = playbook.reminders.playbook if playbook is not None else None
            return wake_lines.waking(current.wakes if current is not None else None)

        def sizes_said():
            return self._handoff is not None and self._handoff.has_lines

        return Wakes(remind, self._words, waking, self._wake_on, sizes_said)

    def _idle_and_context(self) -> Tuple[Optional[float], Optional[int]]:
        """Seconds since the agent's last turn ended, and its context's size."""
        idle = (
            None
            if self._in_turn or self._turn_ended_at is None
            else time.monotonic() - self._turn_ended_at
        )
        return idle, (self._context or {}).get("used")

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
        self._urgent = asyncio.Event()
        self._items = asyncio.Queue()

        def ring(line: str) -> None:
            self._put(self._watcher, line)

        if self._alarm is not None:
            self._alarm.bind(self._loop, ring)
        self._deadlines.bind(self._loop, ring)
        self._wake_on.bind(self._loop, ring)
        if self._stop_requested:
            return 0
        # The kickoff's own actions are news: the watch starts at the tail, so
        # it has to hold its place before the agent acts.
        if not await self._watch():
            return self._code
        asides = None
        try:
            recovered = False
            try:
                await self._agent.start()
            except Exception as error:
                said = "\n".join(CLI_SAID)
                if not cannot_load(str(error), said) or not self._may_recover():
                    raise
                # The reload failed: a fresh context in the same session.
                await self._recover(_reload_failed(error), reconnect=True)
                recovered = True
            if not recovered and self._recover_why is not None:
                # `main` chose a fresh context before the start.
                await self._recover(self._recover_why, reconnect=False)
                recovered = True
            asides = asyncio.ensure_future(self._asides.run())
            # The session's id is known before its first turn: reporting it
            # now lets that turn already write the plan.
            await self._report()
            if recovered:
                prompt = self._handoff.into(handoff_mod.RECOVERED_WAKE)
            else:
                prompt = kickoff(self._resumed, self._goal, self._words)
            while prompt is not None and await self._turn(prompt):
                prompt = await self._next_prompt()
        except Exception:
            # Ctrl-C reaches the CLI as well, which may end before its turn.
            if not self._stopping.is_set():
                raise
            log.debug("The agent's process ended as it stopped.", exc_info=True)
        finally:
            if asides is not None:
                asides.cancel()
            if self._alarm is not None:
                self._alarm.cancel()
            self._deadlines.cancel()
            # Before the watcher stops: stopping it closes the client.
            await self._keep_unstarted_handoff()
            await self._report(ended=True)
            if self._watcher is not None:
                self._watcher.stop()
            try:
                await self._agent.close()
            except Exception:
                log.debug("Closing the session failed.", exc_info=True)
        return self._code

    def _put(self, watcher, item) -> None:
        """An item for the next turn, on the loop; an urgent line from the
        current watch cuts the turn going on short. A side question is never
        a turn's: it is answered beside them (`aside`)."""
        if isinstance(item, watch_mod.Aside):
            if watcher is self._watcher:
                self._asides.add(item)
            return
        if isinstance(item, _Cursor) and self._items.empty():
            # Its lines come before it and have all been taken, by the turn
            # going on or an earlier one: it is that turn's to keep. Queued, it
            # would wait for the next turn, and a stop before then would give
            # those lines again on a resume.
            if watcher is self._watcher:
                self._cursor_given = item.event_id
            return
        self._items.put_nowait((watcher, item))
        if watcher is self._watcher and is_urgent(item):
            self._urgent.set()

    def _start_watcher(self, client) -> None:
        loop = self._loop
        if self._watcher is not None:
            self._watcher.stop()
        watcher = None

        def deliver(item):
            try:
                loop.call_soon_threadsafe(self._put, watcher, item)
            except RuntimeError:  # the loop has closed
                raise _Stopped from None

        watcher = Watcher(
            client,
            deliver,
            poll_s=self._poll_s,
            retry_s=self._retry_s,
            quiet_s=self._quiet_s,
            after=self._after,
            # The session's instrument kind's own lines: it can't switch kind.
            lines=wake_lines.for_kind(self._words.kind),
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
                    # Only the first watch resumes: a later one starts at the tail.
                    self._after = None
                    level = await asyncio.get_running_loop().run_in_executor(
                        None, _level_of, client
                    )
                    self._wake_on.at_level(level, initial=True)
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
        await self._report_read()
        if getattr(self._agent, "compacted", False):
            # The context was compacted during the last turn: the plan and the
            # playbook's rules may be all that is left of the intake.
            self._agent.compacted = False
            prompt += "\n\nYour context was compacted. " + self._words.reread_plan
        interrupter = asyncio.ensure_future(self._interrupt_on_stop())
        cutter = asyncio.ensure_future(self._interrupt_on_urgent())
        sender = asyncio.ensure_future(self._send_transcript_every())
        self._in_turn, self._live_sent = True, None
        reporter = asyncio.ensure_future(self._report_every())
        try:
            end = await self._agent.turn(prompt)
        finally:
            self._in_turn = False
            self._turn_ended_at = time.monotonic()
            interrupter.cancel()
            cutter.cancel()
            sender.cancel()
            reporter.cancel()
        self._turns += 1
        self._keep_given_cursor()
        spent_before = self._spent_usd
        if end.cost_usd is not None:
            self._count_spend(end.cost_usd + self._carried_usd)
        if end.tokens is not None:
            self._tokens = usage.add(self._tokens or usage.zero(), end.tokens)
        if end.context is not None:
            self._context = end.context
        if self._handoff is not None:
            known = end.cost_usd is not None and spent_before is not None
            cost = self._spent_usd - spent_before if known else None
            self._handoff.measure(end.context, cost)
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
        if self._over_budget():
            # The operator may have raised it since the last report.
            await self._reread_budget()
        if self._over_budget():
            self._end_reason = "its budget is spent"
            log.error(
                "The budget is spent ($%.2f of $%g this run, --max-budget-usd); "
                "fibsem-agent stops. Raise it and --resume %s to carry on.",
                self._run_spent(),
                self._budget_usd,
                session,
            )
            self._code = 2
            return False
        self._warn_budget()
        if (
            end.subtype == "no_result"
            and self._turns == 1
            and cannot_load(end.error or "", "\n".join(CLI_SAID))
            and self._may_recover()
        ):
            # The resumed conversation died on its first turn: carry the
            # session on in a fresh context instead of stopping.
            try:
                await self._recover(
                    handoff_mod.RELOAD_FAILED.format(
                        why=_last_said()
                        or "the agent's process ended on its first turn"
                    ),
                    reconnect=True,
                )
            except Exception as error:  # noqa: BLE001 - it stops as before
                log.warning("A fresh context could not start: %s", error)
            else:
                self._put(
                    self._watcher, watch_mod.Wake(ALARM, handoff_mod.RECOVERED_WAKE)
                )
                return True
        if end.subtype == "no_result":
            self._end_reason = "the agent's process ended"
            log.error(
                "The agent's process ended without finishing its turn; "
                "fibsem-agent stops. --resume %s carries the session on.",
                session,
            )
            self._code = 1
            return False
        fresh = self._handoff is not None and self._handoff.pending is not None
        if end.subtype == "error_max_turns" and not fresh:
            # The limit is a checkpoint, not a stop: the next turn starts at
            # once, with any lines that came in. Waiting for a line instead
            # stalls a run nobody is watching.
            log.warning(
                "The turn hit --max-turns before it finished; a new one starts."
            )
            self._put(self._watcher, watch_mod.Wake(ALARM, CARRY_ON))
        elif end.subtype != "success":
            log.warning("The turn ended with %s.", end.subtype)
        if fresh:
            await self._fresh_context()
        await self._report()
        return True

    async def _keep_unstarted_handoff(self) -> None:
        """A handoff note the agent wrote in the turn the run ended on: kept
        with the others, though no fresh context started from it."""
        note = self._handoff.take() if self._handoff is not None else None
        if note is None:
            return
        self._handoff.keep(note)
        session_id = getattr(self._agent, "session_id", None)
        if self._client is not None and session_id:
            await asyncio.get_running_loop().run_in_executor(
                None, handoff_mod.post, self._client, session_id, note.markdown()
            )

    def _skills_to_read(self) -> Tuple[List[str], List[str]]:
        """(The skills a fresh context reads, the others read before.)"""
        playbook = self._playbook.reminders if self._playbook is not None else None
        return handoff_mod.skills_to_read(
            playbook.playbook if playbook is not None else None,
            playbook.skills if playbook is not None else {},
            list(getattr(self._agent, "skills_read", None) or []),
            # Every instrument's session reads the base skill first.
            None if self._words.kind == instruments.NONE else BASE_SKILL,
        )

    def _may_recover(self) -> bool:
        """Whether a resume whose reload failed may start a fresh context:
        once, and only with the words for one."""
        if (
            not self._resumed
            or self._recovery_tried
            or self._handoff is None
            or not self._words.handoff_recovered
            or (self._stopping is not None and self._stopping.is_set())
        ):
            return False
        self._recovery_tried = True
        return True

    async def _recover(self, why: str, reconnect: bool) -> None:
        """Carry the session on in a fresh context fibsem-agent starts itself,
        *why* (a resume that can't reload the last conversation): with *reconnect*, start a new
        conversation now; without, the agent started in one `main` chose.
        Its first turn rebuilds from the plan, the latest handoff note and
        the record."""
        self._recover_why = why
        # Its spend so far: the new conversation's running total starts at 0.
        if self._spent_usd is not None:
            spent = self._spent_usd
        elif self._run_from_usd is not None:
            spent = self._run_from_usd
        else:
            spent = self._carried_usd
        if reconnect:
            conversation = await self._agent.new_context()
        else:
            conversation = getattr(self._agent, "conversation_id", None)
        self._carried_usd = spent
        self._run_from_usd = spent
        if self._keep_context is not None and conversation:
            try:
                self._keep_context(conversation, spent)
            except Exception:  # noqa: BLE001 - the session goes on
                log.debug("Keeping the new context failed.", exc_info=True)
        skills, _ = self._skills_to_read()
        try:
            note = self._latest_note()
        except Exception:  # noqa: BLE001 - the record is read without it
            log.debug("Reading the latest handoff note failed.", exc_info=True)
            note = None
        self._handoff.recovered(why, skills, note)
        calls = getattr(self._agent, "calls", None)
        if calls is not None and not reconnect:
            # `new_context` counts its own; one chosen before the start is here.
            calls.segment = self._handoff.contexts
        log.warning(
            "Fresh context %d: %s The agent rebuilds from its plan, the latest "
            "handoff note and the experiment's record, and asks before it acts.",
            self._handoff.contexts,
            why[:1].upper() + why[1:],
        )
        _line(
            self._agent,
            "context",
            f"Fresh context {self._handoff.contexts}, started by fibsem-agent: {why}",
        )

    async def _fresh_context(self) -> None:
        """The agent asked for a fresh context this turn: keep its note, start
        a new conversation under the same session, and wake it with the note.
        If the new conversation can't start, the agent carries on in this one."""
        handoff = self._handoff
        note = handoff.take()
        skills, read = self._skills_to_read()
        handoff.keep(note)
        session_id = getattr(self._agent, "session_id", None)
        kept = None
        if self._client is not None and session_id:
            kept = await asyncio.get_running_loop().run_in_executor(
                None, handoff_mod.post, self._client, session_id, note.markdown()
            )
        spent = self._spent_usd or 0.0
        try:
            conversation = await self._agent.new_context()
        except Exception as error:  # noqa: BLE001 - the old context carries on
            log.warning("The fresh context could not start: %s", error, exc_info=True)
            handoff.failed(str(error) or type(error).__name__)
            return
        # The new conversation's running total starts at 0.
        self._carried_usd = spent
        if self._keep_context is not None:
            try:
                self._keep_context(conversation, spent)
            except Exception:  # noqa: BLE001 - the session goes on
                log.debug("Keeping the new context failed.", exc_info=True)
        where = (
            f"{kept} in the experiment"
            if kept
            else "handoff.md in the session's folder on the agent's PC"
        )
        handoff.started(note, skills, read, where)
        log.info(
            "Fresh context %d (conversation %s): the agent carries on from its "
            "handoff note.",
            handoff.contexts,
            conversation,
        )
        _line(
            self._agent,
            "context",
            f"Fresh context: the agent started context {handoff.contexts} of "
            f"this session here, from its plan and its handoff note ({where})",
        )
        self._put(self._watcher, watch_mod.Wake(ALARM, handoff_mod.WAKE))

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
            **self._usage_now(),
            "turns": self._turns,
            **({} if self._goal_sent else self._about),
        }
        if self._handoff is not None and self._handoff.contexts > 1:
            body["contexts"] = self._handoff.contexts
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
        if ended:
            body.update(ended=True, reason=self._end_reason)
        try:
            body = plain(decode(SessionReport, body, strict=True))
        except ValueError:
            log.warning("The session report did not read.", exc_info=True)
            return
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

        loop = asyncio.get_running_loop()
        if self._reporting is None or self._reporting[0] is not loop:
            self._reporting = (loop, asyncio.Lock())
        lock = self._reporting[1]

        async def send():
            # In order: a report made while a turn went on never lands after
            # the one made as it ended. Shielded, so a report that has started
            # holds the lock until its answer, even when its turn ends.
            async with lock:
                return await loop.run_in_executor(None, post)

        reported = await asyncio.shield(send())
        self._adopt_budget(reported)
        if isinstance(reported, dict) and self._keep_where is not None:
            self._keep_where(reported)
        if isinstance(reported, dict) and not ended:
            self._deadlines.set(reported.get("deadlines"))
        await self._send_transcript()

    def _count_spend(self, total_usd: float) -> None:
        """The CLI's running total after a turn: the session's spend."""
        if self._run_from_usd is None:
            # A resume whose earlier spend is unknown counts from here.
            self._run_from_usd = total_usd
        elif total_usd < self._run_from_usd:
            # The CLI's total did not carry the earlier runs on: it starts
            # from the spend carried into the current context.
            self._run_from_usd = self._carried_usd
        self._spent_usd = total_usd

    def _run_spent(self) -> Optional[float]:
        """What this run has spent, as the last turn ended."""
        if self._spent_usd is None:
            return None
        return self._spent_usd - (self._run_from_usd or 0.0)

    def _over_budget(self) -> bool:
        spent = self._run_spent()
        return (
            bool(self._budget_usd) and spent is not None and spent >= self._budget_usd
        )

    def _warn_budget(self) -> None:
        """`BUDGET_LOW` for the next turn, once the spend first reaches
        `BUDGET_WARNING` of the budget; again after a new budget takes it
        back under."""
        spent, budget = self._run_spent(), self._budget_usd
        if spent is None or not budget:
            return
        low = spent >= BUDGET_WARNING * budget
        if low and not self._budget_warned:
            self._budget_note = BUDGET_LOW.format(
                spent=spent, budget=budget, used=100 * spent / budget
            )
        elif not low:
            self._budget_note = None
        self._budget_warned = low

    def _adopt_budget(self, reported) -> None:
        """The budget the operator set at the App Server for this session
        (``budget_set_usd``), if it set one."""
        if not isinstance(reported, dict):
            return
        session_id = getattr(self._agent, "session_id", None)
        if reported.get("session_id") not in (None, session_id):
            return
        budget = reported.get("budget_set_usd")
        if isinstance(budget, (int, float)) and budget > 0:
            if budget != self._budget_usd:
                log.info(
                    "The operator set this run's budget to $%g (it was $%g).",
                    budget,
                    self._budget_usd or 0,
                )
                self._budget_usd = float(budget)

    async def _reread_budget(self) -> None:
        """Ask the App Server for the session's budget, in case the operator
        set a new one; a server that can't say leaves it as it is."""
        client = self._client
        if client is None:
            return

        def get():
            try:
                response = client.request("GET", "/app/agent/session", timeout=10.0)
                if response.status_code == 200:
                    return response.json().get("session")
            except Exception:
                log.debug("Reading the session's budget failed.", exc_info=True)
            return None

        self._adopt_budget(await asyncio.get_running_loop().run_in_executor(None, get))

    def _usage_now(self) -> dict:
        """The run's tokens and the context's fill: as the last turn ended,
        or, while a turn goes on, with what it has used so far."""
        tokens, context = self._tokens, self._context
        calls = getattr(self._agent, "calls", None)
        if self._in_turn and calls is not None:
            tokens = usage.add(tokens or usage.zero(), calls.so_far())
            used = calls.reading()
            if used:
                # The window and where it compacts, as the CLI last said.
                context = {**(context or {}), "used": used}
        return {"tokens": tokens, "context": context}

    async def _report_every(self) -> None:
        """While a turn goes on: the session, every `REPORT_EVERY_S` that its
        tokens or the context's fill changed."""
        while True:
            await asyncio.sleep(self._report_every_s)
            now = self._usage_now()
            seen = (now["tokens"], now["context"])
            if seen == self._live_sent or seen == (self._tokens, self._context):
                continue
            self._live_sent = seen
            await self._report()

    async def _report_read(self) -> None:
        """The operator's messages this turn carries are read: the web page
        stops showing them as waiting."""
        ids, client = self._wakes.read(), self._client
        if ids and client is not None:
            await asyncio.get_running_loop().run_in_executor(
                None, watch_mod.report_read, client, ids
            )

    async def _send_transcript(self) -> None:
        """The lines printed since the last send, to the App Server for the web
        page (``POST /app/agent/transcript``). Kept for the next send when it
        can't take them (no experiment open); the oldest go past
        `TRANSCRIPT_KEEP`. One send at a time, and a send that has started
        finishes even when the turn ends under it: the lines it took are not
        sent twice."""
        loop = asyncio.get_running_loop()
        if self._sending is None or self._sending[0] is not loop:
            self._sending = (loop, asyncio.Lock())
        lock = self._sending[1]

        async def send() -> None:
            async with lock:
                await self._send_transcript_now()

        await asyncio.shield(send())

    async def _send_transcript_now(self) -> None:
        # A send carries one picture at most: the rest go straight after.
        for _ in range(PICTURES_WAITING + 1):
            if not await self._send_some_transcript():
                return

    async def _send_some_transcript(self) -> bool:
        """One send of the waiting lines; whether it sent and more wait."""
        client, lines = self._client, getattr(self._agent, "transcript", None)
        session_id = getattr(self._agent, "session_id", None)
        if client is None or not lines or not session_id:
            return False
        now = time.monotonic()
        batch = _one_picture(lines[:TRANSCRIPT_BATCH])
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

        sent = await asyncio.get_running_loop().run_in_executor(None, post)
        if sent:
            del lines[: len(batch)]
        del lines[:-TRANSCRIPT_KEEP]
        return sent and bool(lines)

    async def _send_transcript_every(self) -> None:
        """While a turn runs: the transcript `TRANSCRIPT_SOON_S` after a line
        is printed, and every `TRANSCRIPT_EVERY_S` for what the server could not
        take, so the web page shows a step while it runs."""
        printed = asyncio.Event()
        agent = self._agent
        listens = hasattr(agent, "on_line")
        if listens:
            agent.on_line = printed.set
        try:
            while True:
                try:
                    await asyncio.wait_for(printed.wait(), TRANSCRIPT_EVERY_S)
                    await asyncio.sleep(TRANSCRIPT_SOON_S)
                except asyncio.TimeoutError:
                    pass
                printed.clear()
                await self._send_transcript()
        finally:
            if listens:
                agent.on_line = None

    async def _interrupt_on_stop(self) -> None:
        await self._stopping.wait()
        log.info("Stopping: interrupting the agent's turn.")
        try:
            await self._agent.interrupt()
        except Exception:
            # The CLI may have had the Ctrl-C too, and be gone.
            log.debug("Interrupting the turn failed.", exc_info=True)

    async def _interrupt_on_urgent(self) -> None:
        await self._urgent.wait()
        if self._stopping.is_set():
            return
        log.info("An urgent line: interrupting the agent's turn.")
        self._cut_short = True
        try:
            await self._agent.interrupt()
        except Exception:
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

    def _keep_given_cursor(self) -> None:
        """Store the newest event id whose lines a finished turn carried."""
        given = self._cursor_given
        if self._keep_cursor is None or given is None or given == self._cursor_kept:
            return
        if self._wake_on.held:
            # Lines before it wait for a digest: a resume must give them
            # again, so it keeps the newest cursor before the first was held.
            given = self._cursor_unheld
            if given is None or given == self._cursor_kept:
                return
        try:
            self._keep_cursor(given)
        except Exception:
            log.debug("Keeping the event cursor failed.", exc_info=True)
            return
        self._cursor_kept = given

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
            # Everything waiting is taken now: an urgent line after this
            # cuts the next turn short.
            self._urgent.clear()
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
            held_before, before = bool(self._wake_on.held), self._cursor_given
            for item in batch:
                if isinstance(item, _Cursor):
                    self._cursor_given = item.event_id
            prompt = self._wakes.add(lines)
            if not self._wake_on.held:
                self._cursor_unheld = None
            elif not held_before:
                # This batch's lines are the first held: a resume starts
                # no later than before them.
                self._cursor_unheld = before
            if ended:
                prompt = await self._rewatch(prompt)
                if prompt is None:
                    return None
            if prompt is not None:
                if self._cut_short:
                    self._cut_short = False
                    prompt = f"{INTERRUPTED}\n\n{prompt}"
                if self._budget_note:
                    prompt, self._budget_note = (
                        f"{prompt}\n\n{self._budget_note}",
                        None,
                    )
                if self._handoff is not None:
                    prompt = self._handoff.into(prompt)
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
        # A restarted app starts its own session, at Watch. The messages in
        # *prompt* are still to be read.
        unread = self._wakes.read()
        self._wakes = self._new_wakes()
        self._wakes.unread = unread
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
                try:
                    answers = await asyncio.get_running_loop().run_in_executor(
                        None, _reachable, client
                    )
                except ApiMismatch as mismatch:
                    # Waiting can't help: the two installs differ.
                    _close(client)
                    log.error("%s fibsem-agent stops.", mismatch)
                    self._end_reason = "the App Server is another version"
                    self._resumable = False
                    self._code = 1
                    return None
                if answers:
                    if said:
                        log.info("The App Server answers.")
                    return client
                _close(client)
            remaining = deadline - time.monotonic()
            check = (
                unreachable_why(self._url)
                if self._url
                else "Start the host (Tools -> Host in AutoLamella)."
            )
            if remaining <= 0:
                log.error(
                    "No fibsem host answered within %.0f s; fibsem-agent stops. %s",
                    self._server_wait_s,
                    check,
                )
                self._code = 1
                return None
            if not said:
                log.warning(
                    "Waiting up to %.0f s for the App Server. %s", remaining, check
                )
                said = True
            if not await self._pause(min(delay, remaining)):
                return None
            delay = min(delay * 2, RETRY_MAX_S)
        return None


class SdkAgent:
    """The agent: one ``ClaudeSDKClient`` session, its messages logged."""

    def __init__(
        self,
        options,
        usage_path: Optional[Path] = None,
        session_id: Optional[str] = None,
        client_factory: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self._options = options
        self._client = None
        # Makes the ClaudeSDKClient for *options*; the SDK's own when None.
        self._client_factory = client_factory
        # Each model call's tokens, a line each in the session's usage file.
        self.calls = usage.Calls(usage_path)
        self._cost_usd: Optional[float] = None
        # The CLI conversation now: the session's id until a fresh context
        # (`new_context`) starts another.
        self.conversation_id: Optional[str] = options.resume or options.session_id
        # The fibsem session's id, which a fresh context keeps.
        self.session_id: Optional[str] = session_id or self.conversation_id
        self.model: Optional[str] = options.model
        self.introduced = False
        # The context was compacted: the next turn asks for the plan again.
        self.compacted = False
        # The skills it has read, in order: the session reports them.
        self.skills_read: List[str] = []
        # The lines printed, until the harness sends them (`Harness._send_transcript`).
        self.transcript: List[dict] = []
        self._seq = 0
        # A tool call's line and when it was made, by the call's id: its
        # result says it is done, or that it failed.
        self._steps: dict = {}
        # While the CLI compacts the context: its line and when it began.
        self._compacting: Optional[tuple] = None
        # Called after each line is kept (`Harness._send_transcript_every`).
        self.on_line: Optional[Callable[[], None]] = None
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
        if self.on_line is not None:
            self.on_line()
        return self._seq

    async def start(self) -> None:
        self._client = await self._connect(self._options)

    async def _connect(self, options):
        if self._client_factory is not None:
            client = self._client_factory(options)
        else:
            from claude_agent_sdk import ClaudeSDKClient

            client = ClaudeSDKClient(options)
        await client.connect()
        return client

    async def new_context(self) -> str:
        """A new CLI conversation under the same session, between turns: the
        new client is connected before the old one goes, so a side question
        always has one, and a new one that can't start leaves the old one.
        Its id."""
        conversation = str(uuid.uuid4())
        options = replace(
            self._options,
            resume=None,
            continue_conversation=False,
            session_id=conversation,
        )
        store = options.session_store
        mirrored = isinstance(store, mirror.TranscriptMirror)
        if mirrored:
            # Its lines go on in transcript.jsonl, after the earlier context's.
            before, store.session_id = store.session_id, conversation
        try:
            client = await self._connect(options)
        except BaseException:
            if mirrored:
                store.session_id = before
            raise
        old, self._client, self._options = self._client, client, options
        self.conversation_id = conversation
        self.introduced = False
        self.compacted = False
        self.skills_read = []
        self._steps = {}
        self._compacting = None
        self._cost_usd = None
        self.calls.new_context()
        if old is not None:
            try:
                await old.disconnect()
            except Exception:  # noqa: BLE001 - the old CLI is done with
                log.debug("Closing the earlier context failed.", exc_info=True)
        return conversation

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
        # The context first: while the CLI is asked, the turn's tokens are
        # still its own, for a report made meanwhile (`Harness._usage_now`).
        end.context = await self._context()
        end.tokens = self.calls.end_turn()
        if result is not None:
            log.info("%s", self._turn_line(result, end))
            if result.total_cost_usd is not None:
                self._cost_usd = result.total_cost_usd
        return end

    async def _context(self) -> Optional[Dict[str, int]]:
        """How full the context is, from the CLI; the last call's size, if
        the CLI can't say."""
        used = self.calls.reading()
        context = {"used": used} if used else None
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
        if "used" not in found:
            return context
        parts = context_parts(reply.get("categories"))
        return {**found, "parts": parts} if parts else found

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

    async def side_question(
        self, question: str, history: List[dict], timeout_s: float
    ) -> Optional[str]:
        """The CLI's answer to a side question on this session, mid-turn or
        not: no tools, and nothing added to the conversation (`aside.ask`)."""
        if self._client is None:
            raise aside.Unavailable("the session hasn't started")
        return await aside.ask(self._client, question, history, timeout_s)

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
    if name in SUBAGENT_TOOLS:
        return "Asked", _clip(arguments.get("description", ""), 160)
    if name == WAKE_TOOL:
        minutes = arguments.get("minutes", "")
        return "Set", _clip(
            f"a wake-up in {minutes} min: {arguments.get('reason', '')}", 200
        )
    if name == HANDOFF_TOOL:
        return "Asked", _clip(
            f"for a fresh context, to carry on with: {arguments.get('next', '')}", 200
        )
    if short == "experiment_fetch_file":
        return "Fetched", _clip(arguments.get("path", ""), 160)
    text = json.dumps(arguments, ensure_ascii=False) if arguments else ""
    return "Called", _clip(f"{short} {text}".strip(), 200)


def _one_picture(lines: List[dict]) -> List[dict]:
    """*lines* up to the second picture: a send carries one picture at most."""
    pictures = [i for i, line in enumerate(lines) if line["kind"] == "picture"]
    return lines[: pictures[1]] if len(pictures) > 1 else lines


def _lab_tool(name: str) -> bool:
    """Whether *name* is a tool of a lab's MCP server (`extra_mcp`), not one
    of the harness's own."""
    parts = name.split("__")
    return len(parts) >= 3 and parts[0] == "mcp" and parts[1] not in extra_mcp.RESERVED


def _images(content) -> List[tuple]:
    """The images in a tool result's *content*: (media type, base64) each,
    as the CLI gives them (``source``) or as MCP does (``data``, ``mimeType``)."""
    found = []
    for item in content if isinstance(content, list) else []:
        if not isinstance(item, dict) or item.get("type") != "image":
            continue
        source = item.get("source") if isinstance(item.get("source"), dict) else item
        data = source.get("data")
        media_type = source.get("media_type") or source.get("mimeType")
        if isinstance(data, str) and isinstance(media_type, str):
            found.append((media_type, data))
    return found


def _show_pictures(agent, content, step: int) -> None:
    """The pictures a lab tool answered with, on the transcript under its
    step, so the web page shows what the agent saw. At most
    `PICTURES_WAITING` wait to be sent; one past that is left off."""
    for media_type, data in _images(content)[:PICTURES_PER_RESULT]:
        waiting = sum(
            1 for line in getattr(agent, "transcript", []) if line["kind"] == "picture"
        )
        if (
            waiting >= PICTURES_WAITING
            or len(data) > PICTURE_MAX_B64
            or media_type not in PICTURE_TYPES
        ):
            log.info("A picture from step %s is left off the transcript.", step)
            continue
        _line(agent, "picture", "", step=step, image=data, media_type=media_type)


def _line(agent, kind: str, text: str, **extra) -> Optional[int]:
    """Keep a printed line on *agent*'s transcript, if it keeps one."""
    keep = getattr(agent, "line", None)
    return keep(kind, text, **extra) if keep is not None else None


def _compacting(agent) -> None:
    """The CLI began compacting the context: a step that runs until it ends."""
    if getattr(agent, "_compacting", None) is not None:
        return
    log.info("Compacting the context.")
    seq = _line(agent, "step", "the context", verb="Compacted")
    if seq is not None:
        agent._compacting = (seq, time.monotonic())


def _compacted(agent) -> None:
    """The compaction ended: its step is done."""
    began = getattr(agent, "_compacting", None)
    if began is None:
        return
    agent._compacting = None
    _line(
        agent, "done", "", step=began[0], took_s=round(time.monotonic() - began[1], 1)
    )


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
            conversation = data.get("session_id")
            # The session's id stays as it started; a fresh context is a new
            # conversation in it.
            agent.session_id = agent.session_id or conversation
            if hasattr(agent, "conversation_id"):
                agent.conversation_id = conversation or agent.conversation_id
            agent.model = data.get("model") or agent.model
            if conversation and conversation != agent.session_id:
                log.info("Session %s, conversation %s", agent.session_id, conversation)
            else:
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
        elif message.subtype == "status":
            # The CLI says "compacting" as it starts and clears it when done:
            # the page shows a running step meanwhile, not a frozen agent.
            if data.get("status") == "compacting":
                _compacting(agent)
            else:
                _compacted(agent)
        elif message.subtype == "compact_boundary":
            agent.compacted = True
            meta = data.get("compact_metadata") or {}
            log.info(
                "Context compacted (%s, %s tokens before).",
                meta.get("trigger"),
                meta.get("pre_tokens"),
            )
            _compacted(agent)
            before = meta.get("pre_tokens")
            _line(
                agent,
                "context",
                "Context compacted: "
                + (f"{before / 1000:.0f}k tokens before, " if before else "")
                + (
                    "by the agent"
                    if meta.get("trigger") == "manual"
                    else "automatically"
                ),
            )
    elif isinstance(message, AssistantMessage):
        run = _run_of(agent, message)
        for block in message.content:
            if isinstance(block, TextBlock) and block.text.strip():
                log.info("%s: %s", "subagent" if run else "agent", block.text.strip())
                _line(agent, "said", block.text.strip(), **run)
            elif isinstance(block, ToolUseBlock):
                skill = block.input.get("skill") if block.name == "Skill" else None
                if skill and skill not in agent.skills_read:
                    agent.skills_read.append(skill)
                control = getattr(agent, "playbook", None)
                if skill and control is not None:
                    control.loaded(skill)
                verb, text = step(block.name, block.input, getattr(agent, "cwd", None))
                log.info("%s %s", verb, text)
                extra = dict(run)
                if block.name in SUBAGENT_TOOLS:
                    extra["subagent"] = _clip(block.input.get("subagent_type", ""), 60)
                seq = _line(agent, "step", text, verb=verb, **extra)
                if seq is not None:
                    agent._steps[block.id] = (seq, time.monotonic(), block.name)
    elif isinstance(message, UserMessage) and isinstance(message.content, list):
        run = _run_of(agent, message)
        # A subagent's results are its own, not the agent's.
        parent = getattr(message, "parent_tool_use_id", None)
        for block in message.content:
            if not isinstance(block, ToolResultBlock):
                continue
            seq, took, name = None, None, ""
            made = getattr(agent, "_steps", {}).pop(block.tool_use_id, None)
            if made is not None:
                seq, took = made[0], round(time.monotonic() - made[1], 1)
                name = made[2]
            if block.is_error:
                why = _clip(_result_text(block.content))
                log.warning("failed: %s", why)
                _line(agent, "failed", _clip(why, 160), step=seq, took_s=took, **run)
            elif seq is not None:
                # The step is done: the web page stops showing it as running.
                # A subagent's answer is kept on it, for its run to show.
                answer, verdict = "", {}
                if name in SUBAGENT_TOOLS:
                    answer, verdict = _answer(block.content)
                    answer = _clip(answer, SUBAGENT_ANSWER)
                _line(agent, "done", answer, step=seq, took_s=took, **run, **verdict)
                if _lab_tool(name):
                    _show_pictures(agent, block.content, seq)
            policy = getattr(agent, "wake_on", None)
            if policy is not None and not block.is_error and not parent:
                # A long command's end the agent read itself wakes no one.
                for run_id in wake_policy.ended_runs(name, block.content):
                    policy.saw_run_end(run_id)


def _run_of(agent, message) -> dict:
    """``{"run": <seq>}`` for a subagent's message: the line number of the
    Agent call that started it, so its lines show under that run; ``{}`` for
    the agent's own."""
    parent = getattr(message, "parent_tool_use_id", None)
    made = getattr(agent, "_steps", {}).get(parent) if parent else None
    return {"run": made[0]} if made is not None else {}


_JSON_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def _answer(content) -> tuple:
    """A subagent's answer, without the CLI's agentId footer, and what its
    closing JSON block says for the run's card: ``verdict`` (agree or concern)
    from a boolean ``agree``, the Verifier's, and ``attention`` from a boolean
    ``attention``. The answer is then the block's ``summary``, else its
    ``concern``, else the words before the block; any other answer is kept as
    it is."""
    text = re.split(r"\s*agentId: ", _result_text(content))[0].strip()
    blocks = _JSON_BLOCK.findall(text)
    try:
        said = json.loads(blocks[-1]) if blocks else None
    except ValueError:
        said = None
    if not isinstance(said, dict):
        return text, {}
    flags = {}
    if isinstance(said.get("agree"), bool):
        flags["verdict"] = "agree" if said["agree"] else "concern"
    if isinstance(said.get("attention"), bool):
        flags["attention"] = said["attention"]
    if not flags:
        return text, {}
    for key in ("summary", "concern"):
        words = said.get(key)
        if isinstance(words, str) and words.strip():
            return words.strip(), flags
    return text[: text.rfind("```json")].strip(), flags


def find_checkout(given: Optional[str] = None) -> Optional[Path]:
    """The folder whose ``.claude`` holds the shipped skills: the checkout
    given, else the one this module runs from, else the current folder, else
    the copy in the package (`sources.find_root`)."""
    return sources.find_root(given, BASE_SKILL)


def mcp_names(
    playbook: Optional[playbooks.Playbook],
    extras: Dict[str, extra_mcp.Server],
    host: bool = False,
) -> List[str]:
    """The MCP servers *playbook* names (``mcp``): ``fibsem``, the sidecar,
    and any of the lab's (*extras*). Without a playbook, the sidecar alone.
    Raises `SetupRefused` for a server that isn't there, and, without an
    agent *host*, for a playbook without ``fibsem``: the agent's wake-ups and
    its record go through the App Server then. A playbook with no instrument
    (``instrument: none``) has no ``fibsem`` and runs through a host."""
    if playbook is not None and playbook.instrument == instruments.NONE:
        if not host:
            raise SetupRefused(
                f"The {playbook.name} playbook has no instrument: it keeps its "
                "record on an agent host, so start it with --host."
            )
        if SERVER in playbook.mcp:
            raise SetupRefused(
                f"The {playbook.name} playbook has no instrument, so no "
                f"{SERVER} server: take {SERVER} out of its mcp."
            )
        names = list(playbook.mcp)
    else:
        names = (
            list(playbook.mcp) if playbook is not None and playbook.mcp else [SERVER]
        )
    missing = [n for n in names if n != SERVER and n not in extras]
    if missing:
        raise SetupRefused(
            f"The {playbook.name} playbook uses the MCP server(s) "
            f"{', '.join(missing)}, which {extra_mcp.path()} doesn't set up."
        )
    if SERVER not in names and not host:
        raise SetupRefused(
            f"The {playbook.name} playbook's mcp has no {SERVER}: without "
            "--host the agent hears from the app and keeps its record through "
            "it, so it needs it."
        )
    return names


def used_servers(
    extras: Dict[str, extra_mcp.Server], names: List[str]
) -> Dict[str, extra_mcp.Server]:
    """The lab's servers (*extras*) the session's *names* name: a lab server
    is on only for a playbook that names it in ``mcp``."""
    return {name: extras[name] for name in names if name in extras}


def granted_tools(
    loaded: Loaded, extras, host: bool = False, fibsem: bool = True
) -> List[str]:
    """The tools allowed without asking: the fibsem tools (with *fibsem*, the
    sidecar among the session's servers), the harness's, the subagents', the
    lab servers' and, with an agent host, the record tools."""
    sidecar = f"mcp__{SERVER}__*"
    own = [t for t in ALLOWED_TOOLS if fibsem or t != sidecar]
    granted = own + subagent_tools(loaded.agents) + extra_mcp.allowed(extras)
    if host:
        granted = granted + [f"mcp__{RECORD_SERVER}__*"]
    return granted


def _hidden(extras, secrets: Dict[str, str]) -> List[str]:
    """The lab servers' tools that mcp.yaml doesn't list, kept from the
    model. A server that can't be listed is warned about: its other tools stay
    in sight, and the level gate still refuses them."""
    if not extras:
        return []
    try:
        return extra_mcp.hidden(extras, secrets)
    except extra_mcp.ExtraMcpError as error:
        log.warning(f"{error}; the agent may see tools mcp.yaml doesn't list")
        return []


def level_gate(extras, url: Optional[str], token: Optional[str]):
    """The PreToolUse hook that refuses a lab server's tool the agent level
    doesn't reach (`extra_mcp.refusal`), reading the level from the App
    Server at each call, so the operator's change holds at once."""

    async def hook(hook_input, tool_use_id, context):
        tool = str(hook_input.get("tool_name") or "")
        why = await asyncio.to_thread(
            extra_mcp.refusal, extras, tool, lambda: extra_mcp.read_level(url, token)
        )
        if why is None:
            return {}
        log.info(f"refused {tool}: {why}")
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": why,
            }
        }

    return hook


def _with_level_gate(hooks: Optional[dict], extras, url, token) -> Optional[dict]:
    """*hooks* with the level gate on every lab server's tools."""
    if not extras:
        return hooks
    from claude_agent_sdk import HookMatcher

    names = "|".join(re.escape(name) for name in extras)
    hooks = dict(hooks or {})
    hooks["PreToolUse"] = [
        HookMatcher(
            matcher=f"mcp__({names})__.*", hooks=[level_gate(extras, url, token)]
        )
    ] + list(hooks.get("PreToolUse", []))
    return hooks


def sidecar_command() -> List[str]:
    """``fibsem-mcp`` of this environment, by absolute path. Failing that,
    ``python -I -m``, which keeps the current folder off the path, so another
    checkout there cannot shadow the installed fibsem."""
    scripts = sysconfig.get_path("scripts")
    found = shutil.which("fibsem-mcp", path=scripts) if scripts else None
    if found:
        return [str(Path(found).resolve())]
    return [sys.executable, "-I", "-m", "fibsem.ai.agent.mcp.sidecar"]


def always_loaded(playbook: Optional[playbooks.Playbook]) -> List[str]:
    """The sidecar's arguments for the playbook's ``tools``: none without."""
    if playbook is None or not playbook.tools:
        return []
    return ["--always-load", ",".join(playbook.tools)]


def build_options(
    checkout: Path,
    url: Optional[str] = None,
    token: Optional[str] = None,
    model: Optional[str] = None,
    max_turns: Optional[int] = None,
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
    extras: Optional[Dict[str, extra_mcp.Server]] = None,
    host: Optional[str] = None,
    host_token: Optional[str] = None,
    handoff: Optional[handoff_mod.Handoff] = None,
    wake_on: Optional[wake_policy.WakeOn] = None,
):
    """The SDK session: the MCP servers the *playbook* names (``mcp``; the
    sidecar alone without one), among them the lab's (*extras*, else read;
    `extra_mcp`) and no other of the lab's, the *loaded* skills and
    subagents (`session_names`) and nothing else of the user's Claude Code
    config. With an agent *host*, its own sidecar offers the record tools
    there, and the instrument's sidecar offers the rest. Settings left None are the base playbook's. With ``work``, code
    mode: the code tools, sandboxed, in ``work`` (``code_mode.prepare`` has
    copied the skills there); without ``code``, the file tools only. The
    *playbook* the session starts on puts its rules in the prompt, and its
    reminders hold when there is no *control* to ask. With
    *transcript*, a folder, the CLI's transcript is copied there as it is
    written (`mirror.TranscriptMirror`). With *handoff*, the agent has the
    ``context_reset`` tool. The CLI's compaction is always asked to keep what
    the instrument kind's ``handoff.md`` says, and the playbook's
    ``compact_window`` sets where it compacts."""
    from claude_agent_sdk import ClaudeAgentOptions

    loaded = loaded or session_names(checkout, playbook)
    extras = extra_mcp.load() if extras is None else extras
    base = defaults()
    compact = (
        words or texts(playbook.instrument if playbook is not None else None)
    ).handoff_compact
    command = sidecar_command()
    # The playbook's tools, which the CLI loads from the first turn; each
    # sidecar marks those it offers.
    always = always_loaded(playbook)
    server = {
        "type": "stdio",
        "command": command[0],
        "args": command[1:] + (["--no-record"] if host else []) + always,
    }
    # In the CLI's environment, which the sidecar inherits: the server entry
    # goes on the CLI's command line, where anyone can list it.
    env = dict(CLI_ENV)
    settings_now = playbook.session if playbook is not None else base
    env.update(handoff_mod.compact_env(settings_now.compact_window))
    if url:
        env[SERVER_URL_ENV] = url
    if token:
        env[SERVER_TOKEN_ENV] = token
    # The sidecar offers this instrument kind's tools only.
    kind = playbook.instrument if playbook is not None else None
    if kind:
        env[INSTRUMENT_ENV] = kind
    # The playbook's servers, in its order: the sidecar and the lab's it
    # names. The lab's others are off.
    names = mcp_names(playbook, extras, host is not None)
    extras = used_servers(extras, names)
    # The extra servers' tokens: their entries name them, ${NAME}, and the
    # CLI fills them in from here (`extra_mcp.definitions`).
    secrets = extra_mcp.tokens(extras)
    env.update(secrets)
    if host_token:
        env[HOST_TOKEN_ENV] = host_token
    granted = granted_tools(loaded, extras, host is not None, SERVER in names)
    tools, allowed, denied = BUILT_IN_TOOLS, granted, DENIED_TOOLS
    settings = None
    if work is not None:
        env.update(code_mode.env(work))
        given = code_mode.tools(code)
        tools = BUILT_IN_TOOLS + given
        allowed = granted + code_mode.allowed_tools(work, code)
        denied = [tool for tool in DENIED_TOOLS if tool not in given]
        settings = code_mode.settings_json(work, {**os.environ, **env}, code)
    denied = denied + _hidden(extras, secrets)
    servers = {}
    lab = extra_mcp.definitions(extras)
    for name in names:
        servers[name] = server if name == SERVER else lab[name]
    if host:
        # The address goes on the command line; its token, by name only.
        servers[RECORD_SERVER] = {
            "type": "stdio",
            "command": command[0],
            "args": command[1:]
            + ["--url", host, "--token-env", HOST_TOKEN_ENV, "--record-only"]
            + always,
        }
    if alarm is not None:
        servers[HARNESS_SERVER] = wake_server(alarm, control, handoff, wake_on)
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
        # Only the servers above: not a `fibsem` registered in ~/.claude.json.
        strict_mcp_config=True,
        system_prompt={
            "type": "preset",
            "preset": "claude_code",
            "append": system_prompt(
                work,
                code,
                playbook,
                words,
                extras,
                names + [RECORD_SERVER] if host else None,
            ),
        },
        # With a playbook control, the reminders on now, asked at each call;
        # and the lab servers' tools gated by the agent level.
        # And what the CLI's compaction keeps, should it run.
        hooks=handoff_mod.compact_hooks(
            _with_level_gate(
                control.reminders.sdk_hooks(SERVER)
                if control is not None
                else reminders.sdk_hooks(playbook.hooks, SERVER)
                if playbook
                else None,
                extras,
                url,
                token,
            ),
            compact,
        ),
        # Watcher lines carry operator-typed text: never expand an @path in it.
        verbatim_prompts=True,
        # Subagents' words too, so the transcript shows each run in full.
        forward_subagent_text=True,
        # No max_budget_usd: given one, the CLI puts the spend in a reminder
        # on every model call. The harness keeps the budget (`Harness`).
        max_turns=max_turns or base.max_turns,
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
        stderr=_cli_said,
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
        *(
            [
                ("Host", args.host),
                (
                    "Server",
                    args.url
                    or "this computer's App Server, if the playbook names fibsem",
                ),
            ]
            if getattr(args, "host", None)
            else [("Server", args.url or "this computer's App Server")]
        ),
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


NO_CHECKOUT = (
    "fibsem-agent found no skills: this fibsem has none shipped and "
    "isn't a checkout; pass --checkout <path to a fibsem-os checkout>."
)


def _parser() -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--host",
        nargs="?",
        const="",
        default=None,
        metavar="URL",
        help="an agent host (fibsem-agent-host) to hear from and keep the "
        "agent's record through, in place of the App Server: the one running "
        "on this PC, or at URL. It asks the operator to allow this agent, and "
        "keeps the token it is given; or set the token in "
        f"{HOST_TOKEN_ENV} (environment or key store). The instrument's "
        "server is then one of the playbook's mcp servers, or none",
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
        help="what this run may spend: fibsem-agent stops at the end of the "
        "turn that spends it, so that turn can go over (by at most its "
        "--max-turns model calls and the subagents it called); the operator "
        "can set a new budget while it runs, PUT /app/agent/budget "
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
    return parser


class SetupRefused(Exception):
    """Why fibsem-agent can't run with what it was given."""


@dataclass
class Setup:
    """What a session runs with, from its command line and its playbook."""

    playbook: playbooks.Playbook
    words: instruments.Texts
    loaded: Loaded
    work: Optional[Path]
    code: bool
    session_id: str
    folder: Path
    run: Optional[Path]
    extras: Dict[str, extra_mcp.Server]


def _setup(args, checkout: Path) -> Setup:
    """Read the playbook and settle the session's folders; fills what *args*
    left out from the playbook. Raises `SetupRefused`."""
    try:
        playbook = playbooks.load(args.playbook or playbooks.DEFAULT, checkout, EFFORTS)
    except playbooks.PlaybookError as error:
        raise SetupRefused(f"{error} (fibsem-agent --list-playbooks).") from None
    _resolve(args, playbook.session)
    unset = [key for key in REQUIRED if getattr(playbook.session, key) is None]
    if unset:
        raise SetupRefused(
            f"The {playbook.name} playbook has no {', '.join(unset)}, and the "
            f"shipped {playbooks.DEFAULT} playbook gives none: is it missing?"
        )
    try:
        words = instruments.load(playbook.instrument, checkout)
    except instruments.InstrumentError as error:
        raise SetupRefused(str(error)) from None
    # What it loads is fixed for the session: a playbook named at launch
    # narrows it (`session_names`).
    loaded = session_names(checkout, playbook if args.playbook else None)
    try:
        extras = extra_mcp.load()
        # Each token stored, or the agent would start without what it was
        # told it has.
        extra_mcp.tokens(extras)
    except extra_mcp.ExtraMcpError as error:
        raise SetupRefused(str(error)) from None
    names = mcp_names(playbook, extras, getattr(args, "host", None) is not None)
    # A specialist gets only the lab servers its session has.
    used = used_servers(extras, names)
    try:
        for name in loaded.agents:
            subagents.check_file(name, checkout, EFFORTS, used)
    except subagents.SubagentError as error:
        raise SetupRefused(str(error)) from None
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
            raise SetupRefused(
                f"No earlier fibsem-agent session in {where} to resume: pass "
                "its id (the Session line of its log)."
            )
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
            raise SetupRefused(refused)
    return Setup(playbook, words, loaded, work, code, session_id, folder, run, extras)


def playbook_list() -> List[dict]:
    """What ``fibsem-agent --list-playbooks`` prints. Raises `SetupRefused`."""
    checkout = find_checkout(None)
    if checkout is None:
        raise SetupRefused(NO_CHECKOUT)
    return _playbook_rows(checkout)


def playbook_configuration(name: str) -> dict:
    """What ``fibsem-agent --playbook NAME --show-config`` prints. Raises
    `SetupRefused`."""
    args = _parser().parse_args(["--playbook", name])
    checkout = find_checkout(None)
    if checkout is None:
        raise SetupRefused(NO_CHECKOUT)
    setup = _setup(args, checkout)
    return show_config(
        args,
        checkout,
        setup.work,
        setup.code,
        setup.playbook,
        setup.loaded,
        setup.words,
        setup.extras,
    )


def main(argv=None) -> int:
    """``fibsem-agent``. Every exit but a clean one says why: on the console,
    and in the session's log once it is open."""
    opened: List[sessions.SessionFolder] = []
    try:
        return _main(argv, opened)
    except sessions.SessionRunning as busy:
        print(f"fibsem-agent: {busy}", file=sys.stderr)
        return 1
    except Exception as error:  # noqa: BLE001 - said, then the exit code
        _failed(error, opened)
        return 1


def _failed(error: Exception, opened: List["sessions.SessionFolder"]) -> None:
    """Say why fibsem-agent failed outside its run: in its log, with the
    traceback, once the log is open, else on the console."""
    why = f"fibsem-agent failed: {type(error).__name__}: {error}"
    if any(isinstance(h, logging.FileHandler) for h in log.handlers):
        log.error(why, exc_info=error)
        say = log.error
    else:
        print(why, file=sys.stderr)
        traceback.print_exception(error)
        say = functools.partial(print, file=sys.stderr)
    for session in opened:
        session_id = session.record().get("session_id")
        if session_id:
            say(f"--resume {session_id} carries the session on.")
        try:
            session.end(1, why)
        except Exception:  # noqa: BLE001 - what was said stands
            log.debug("Could not record the run's end.", exc_info=True)


def _main(argv, opened: List["sessions.SessionFolder"]) -> int:
    refuse("fibsem-agent")
    diagnostics.install("fibsem-agent")
    parser = _parser()
    args = parser.parse_args(argv)
    if args.code and args.files:
        parser.error("--code already gives the agent its files: use one of them")
    if args.session_id and args.resume:
        parser.error("--session-id starts a new session: --resume carries one on")
    if args.list_hooks:
        return _print(reminders.hook_rows(), args.json, _hook_text)
    if args.host is not None:
        # --host alone: the host running on this PC, from its discovery file.
        url = host_link.host_url(args.host)
        if url is None and not args.show_config:
            print(host_link.NO_HOST, file=sys.stderr)
            return 1
        args.host = url or host_link.ON_THIS_PC

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError, ValueError):
        pass
    checkout = find_checkout(args.checkout)
    if checkout is None:
        print(NO_CHECKOUT, file=sys.stderr)
        return 1
    if args.list_playbooks:
        return _print(_playbook_rows(checkout), args.json, _playbook_text)

    try:
        setup = _setup(args, checkout)
    except SetupRefused as refused:
        print(refused, file=sys.stderr)
        return 1
    playbook, words, loaded = setup.playbook, setup.words, setup.loaded
    work, code, session_id = setup.work, setup.code, setup.session_id
    folder, run, extras = setup.folder, setup.run, setup.extras
    if args.show_config:
        config = show_config(
            args, checkout, work, code, playbook, loaded, words, extras
        )
        return _print(config, args.json, _config_text)
    missing = missing_extra_message("agent", "fibsem-agent")
    if missing:
        print(missing, file=sys.stderr)
        return 1
    # From the environment, so it is not in the process list or shell history.
    token = server_env()[1]
    # The agent's token at the host: set, kept from an Allow there, or asked for.
    host_token = None
    if args.host:
        try:
            host_token = host_link.host_token(args.host)
        except (LookupError, PermissionError) as why:
            print(f"fibsem-agent: {why}", file=sys.stderr)
            return 1
    # The CLI reads the key from its environment, which it inherits from here.
    key = get_secret(API_KEY)
    if key:
        os.environ[API_KEY] = key
    session = sessions.SessionFolder.open(
        session_id, _about(args, checkout, work, code, playbook, run), folder=folder
    )
    opened.append(session)
    log_path = Path(args.log).expanduser() if args.log else session.log_path
    print_fibsem_logo()
    _log_to(log_path)
    for line in status(args, checkout, log_path, work, session_id, code):
        log.info(line)
    # The specialists' tools on the extra servers go on their copies in the
    # folder just assembled; with nothing assembled (run is the checkout),
    # the files are the shipped ones and stay as they are.
    extra_tools = extra_mcp.subagent_allowed(
        used_servers(
            extras, mcp_names(playbook, extras, getattr(args, "host", None) is not None)
        )
    )
    if work is not None:
        code_mode.prepare(work, checkout, loaded.skills, loaded.agents)
        subagents.add_tools(work, loaded.agents, extra_tools)
        for line in code_mode.details():
            log.debug(line)
    elif run != checkout:
        sources.assemble(run, checkout, loaded.skills, loaded.agents)
        subagents.add_tools(run, loaded.agents, extra_tools)

    alarm = Alarm(playbook.session.wake_max_min)
    wake_on = wake_policy.WakeOn(
        wake_policy.Settings.of(playbook.session), keep=session.keep_wake_on
    )
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
    # Fresh contexts: a resume carries on the newest one's conversation.
    handoff = handoff_mod.Handoff(
        words,
        playbook.session.context_marks,
        keep=session.keep_handoff,
        contexts=1 + len(record.get(sessions.CONTEXTS) or []),
    )
    conversation = sessions.conversation(record) if args.resume else None
    # A resume whose last conversation is too large to be worth reloading
    # carries on in a fresh context, rebuilt from the record (`Harness._recover`).
    recover = _too_large_to_resume(
        record, session.folder, playbook.session.resume_max_context, words
    )
    fresh_conversation = str(uuid.uuid4()) if recover else None
    # The agent's own wake policy, if it set one before a resume.
    wake_on.restore(record.get(sessions.WAKE_ON))
    options = build_options(
        checkout,
        alarm=alarm,
        wake_on=wake_on,
        control=control,
        url=args.url,
        token=token,
        model=args.model,
        max_turns=args.max_turns,
        resume=None if recover else conversation or args.resume,
        effort=args.effort,
        work=work,
        session_id=fresh_conversation or (None if args.resume else session_id),
        code=code,
        playbook=playbook,
        transcript=session.folder,
        cwd=run,
        loaded=loaded,
        words=words,
        extras=extras,
        host=args.host,
        host_token=host_token,
        handoff=handoff,
    )
    agent = SdkAgent(
        options, usage_path=session.folder / usage.FILE, session_id=session_id
    )
    agent.calls.segment = handoff.contexts
    agent.playbook = control
    agent.wake_on = wake_on
    control.say = lambda text: _line(agent, "context", text)
    wake_on.say = control.say
    if named:
        control.say(f"Playbook set: {playbook.name}, {control.reminders.chosen_by}")
    harness = Harness(
        agent,
        # Its record and wake-ups: the agent host's, else the App Server's.
        (
            (lambda: watch_mod.connect(args.host, host_token))
            if args.host
            else (lambda: watch_mod.connect(args.url, token))
        ),
        url=args.host or args.url,
        resumed=args.resume is not None,
        server_wait_s=args.server_wait,
        goal=args.goal,
        budget_usd=args.max_budget_usd,
        spent_before_usd=_spent_before(record),
        playbook=control,
        words=words,
        deadline_warn_s=playbook.session.deadline_warn_min * 60,
        quiet_s=playbook.session.quiet_min * 60,
        after=record.get(sessions.EVENT_AFTER),
        keep_cursor=session.keep_event_cursor,
        keep_where=session.keep_where,
        handoff=handoff,
        keep_context=session.keep_context,
        spent_carried_usd=sessions.context_spend(record),
        wake_on=wake_on,
        recover=recover,
        latest_note=lambda: handoff_mod.latest_note(session.folder / sessions.HANDOFF),
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


def _too_large_to_resume(
    record: dict,
    folder: Path,
    limit: Optional[int],
    words: instruments.Texts,
) -> Optional[str]:
    """Why a resumed session's last conversation is not reloaded, or None to
    reload it: its context was over *limit* tokens as it stopped (its usage
    file's last call)."""
    if not record or not limit or not words.handoff_recovered:
        return None
    used = usage.last_context(folder / usage.FILE)
    if used is None or used < limit:
        return None
    return handoff_mod.TOO_LARGE.format(
        used=usage.short(used), limit=usage.short(limit)
    )


def _spent_before(record: dict) -> Optional[float]:
    """What a resumed session had spent when its last run ended (the CLI's
    running total, from its record); None when no run says."""
    spent = [
        run["spent_usd"]
        for run in (record.get("runs") or [])[:-1]
        if isinstance(run.get("spent_usd"), (int, float))
    ]
    return max(spent) if spent else None


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
        "host": getattr(args, "host", None),
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
    "wake_on_watch",
    "wake_on_assist",
    "wake_on_drive",
    "digest_min",
    "digest_lines",
)


def _resolve(args, given: playbooks.SessionSettings) -> None:
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
    extras: Optional[Dict[str, extra_mcp.Server]] = None,
) -> dict:
    """``--show-config``: everything the agent would run with, from the same
    pieces `build_options` uses, without the SDK and without starting. The
    extra servers with their tokens' names, never the tokens."""
    loaded = loaded or session_names(checkout, playbook)
    extras = extra_mcp.load() if extras is None else extras
    host = getattr(args, "host", None)
    named = mcp_names(playbook, extras, host is not None)
    # The lab's servers this playbook leaves off, so none goes unnoticed.
    unused = sorted(set(extras) - set(named))
    extras = used_servers(extras, named)
    servers = named + ([RECORD_SERVER] if host else [])
    granted = granted_tools(loaded, extras, host is not None, SERVER in servers)
    tools, allowed, denied = BUILT_IN_TOOLS, granted, DENIED_TOOLS
    if work is not None:
        given = code_mode.tools(code)
        tools = BUILT_IN_TOOLS + given
        allowed = granted + code_mode.allowed_tools(work, code)
        denied = [tool for tool in DENIED_TOOLS if tool not in given]
    config = {
        "level": playbook.session.level,
        "description": playbook.description,
        "model": args.model,
        "effort": args.effort,
        "max_budget_usd": args.max_budget_usd,
        "max_turns": args.max_turns,
        "wake_max_min": playbook.session.wake_max_min,
        "deadline_warn_min": playbook.session.deadline_warn_min,
        "quiet_min": playbook.session.quiet_min,
        # What wakes the agent at each level, until it sets its own
        # (`wake_policy`): every line, or routine ones held for a digest.
        "wake_on": {
            "watch": playbook.session.wake_on_watch,
            "assist": playbook.session.wake_on_assist,
            "drive": playbook.session.wake_on_drive,
            "digest_min": playbook.session.digest_min,
            "digest_lines": playbook.session.digest_lines,
        },
        "instrument": words.kind if words else playbook.instrument,
        # The playbook's servers, and the lab's it doesn't name (off).
        "mcp": servers,
        "mcp_unused": unused,
        # The lines that wake the agent, by group; None for every line.
        "wakes": playbook.wakes,
        # The agent host its record and wake-ups go through; None for the
        # App Server.
        "host": host,
        "mode": ("code" if code else "files") if work is not None else "none",
        "folder": str(work) if work is not None else None,
        "checkout": str(checkout),
        "skills": list(loaded.skills),
        "specialists": list(loaded.agents),
        "tools": list(tools),
        # The fibsem tools loaded from the first turn; the CLI defers the rest.
        "always_loaded": list(playbook.tools),
        "allowed_tools": list(allowed),
        "denied_tools": list(denied),
        # Its reminders are on from the start; the agent may propose another.
        "playbook": {
            **playbook.summary(),
            "path": str(playbook.path),
            "hash": playbook.hash,
        },
        "system_prompt": system_prompt(
            work, code, playbook, words, extras, servers if host else None
        ),
    }
    if extras:
        config["extra_mcp"] = {
            name: {
                "where": s.url or " ".join(s.command),
                "token": s.token,
                "about": s.about,
                "tools": [extra_mcp.ALL] if s.every else s.tools,
                "watch": ([extra_mcp.ALL] if s.every and s.at_watch("") else s.watch),
                "subagents": s.subagents,
            }
            for name, s in extras.items()
        }
    return config


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


def _wake_on_text(wake_on: dict) -> str:
    """``Wakes:    every line at Drive and Assist; at Watch, routine lines in a
    digest every 30 min or 20 lines``."""
    order = ("drive", "assist", "watch")
    modes = {level: wake_on.get(level) or "all" for level in order}
    every = [level.title() for level in order if modes[level] == "all"]
    held = [level.title() for level in order if modes[level] != "all"]
    parts = []
    if every:
        parts.append(f"every line at {' and '.join(every)}")
    if held:
        due = []
        if wake_on.get("digest_min"):
            due.append(f"every {wake_on['digest_min']:g} min")
        if wake_on.get("digest_lines"):
            due.append(f"{wake_on['digest_lines']} lines")
        parts.append(
            f"at {' and '.join(held)}, routine lines in a digest "
            f"{' or '.join(due) or 'with the next line that wakes it'}"
        )
    return f"Wakes:    {'; '.join(parts)} (the agent changes it with wake_on)"


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
        _wake_on_text(config["wake_on"]),
        f"Instrument: {config['instrument']}",
        f"Mode:     {config['mode']}"
        + (f" in {config['folder']}" if config["folder"] else ""),
        f"Skills:   {', '.join(config['skills'])}",
        f"Agents:   {', '.join(config['specialists'])}",
        f"Tools:    {', '.join(config['tools'])}",
        f"Loaded:   {', '.join(config['always_loaded']) or 'none'} (the other "
        "fibsem tools wait for ToolSearch)",
        f"Allowed:  {', '.join(config['allowed_tools'])}",
        f"Denied:   {', '.join(config['denied_tools'])}",
    ]
    lines += [
        f"MCP {name} ({each['where']}"
        + (f", token {each['token']}" if each["token"] else "")
        + f"): {', '.join(each['tools'])}; at Watch: "
        f"{', '.join(each.get('watch', [])) or 'none'}, the rest at Drive; specialists: "
        f"{', '.join(each['subagents']) or 'none'}"
        for name, each in config.get("extra_mcp", {}).items()
    ]
    if config.get("mcp_unused"):
        lines.append(
            f"MCP off (the lab's, not named in this playbook's mcp): "
            f"{', '.join(config['mcp_unused'])}"
        )
    if config.get("wakes") is not None:
        lines.append(
            "Wakes on: "
            + (", ".join(config["wakes"]) or "nothing")
            + " (and always messages, the level, Stop and its own wake-ups)"
        )
    lines.append(f"Playbook skills: {', '.join(playbook['skills']) or 'none'}")
    lines += [f"  When needed: {o['skill']}, {o['when']}" for o in playbook["optional"]]
    lines.append("Reminders:")
    lines += [f"  {r['label']}: {r['say']}" for r in playbook["reminders"]] or [
        "  none"
    ]
    lines += [f"Rule: {rule}" for rule in playbook["rules"]]
    return "\n".join(lines + ["", "System prompt (appended):", config["system_prompt"]])


if __name__ == "__main__":
    sys.exit(main())
