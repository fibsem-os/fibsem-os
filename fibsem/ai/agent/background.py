"""Subagents in the background: which may run there, and what they may do.

The lead may start a subagent in the background (``run_in_background``) and
carry on; its answer comes back on its own, as a turn of the lead's. Only a
subagent that never acts on the microscope may: every fibsem tool its file
lists is a Watch tool, it lists no lab server's tool, and it starts no
subagent of its own. At most
`MAX_RUNNING` run in the background at once, and a subagent's own calls run
in its turn. The microscope stays one agent's at a time: a background run's
call to anything but a Watch tool is refused (`lease_hook`), whatever its file
lists.

The operator may write to a subagent while it runs, blocking or not. The
message waits here and goes in after its next tool call (`deliver_hook`),
without a turn of the lead's; the lead hears of it in its next wake. A message
its run never read, because it ended first, goes to the lead instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fibsem.ai.agent import subagents
from fibsem.ai.agent.catalog import CATALOG

log = logging.getLogger("fibsem.agent.harness")

MAX_RUNNING = 2
# The tools that start a subagent (`harness.SUBAGENT_TOOLS`).
AGENT_TOOLS = ("Agent", "Task")
# The fibsem tools that never act on the microscope: the Watch level's.
WATCH_TOOLS = frozenset(spec.name for spec in CATALOG if spec.level == "watch")
# The servers whose tools are fibsem's own: the instrument's sidecar and,
# with an agent host, the record's (`harness.SERVER`, `harness.RECORD_SERVER`).
FIBSEM_SERVERS = ("fibsem", "fibsem_record")

NOT_HERE = (
    "Only a subagent that can't act on the microscope may run in the "
    "background, and {name} can. Call it again without run_in_background: the "
    "microscope is one agent's at a time."
)
FULL = (
    "{count} subagents run in the background already, the most at once. "
    "Wait for one to answer, or call this one without run_in_background."
)
SHELL = (
    "Commands run in the foreground here: call Bash again without run_in_background."
)
NESTED = (
    "A subagent's own calls run in its turn: call {name} without run_in_background."
)
LEASE = (
    "{tool} is refused: a subagent in the background doesn't act on the "
    "microscope, which is the lead's or its blocking subagent's. Say what you "
    "would do in your answer instead."
)
DELIVERED = "The operator wrote to you while you work (message id={id}): {text}"


def off_microscope(tool: str) -> bool:
    """Whether *tool* never acts on the microscope: a fibsem Watch tool, or one
    of the CLI's own (code mode's shell has no network, so the fibsem tools
    are its only way there, `code_mode`). A lab server's tool may act."""
    parts = tool.split("__")
    if parts[0] != "mcp":
        return True
    if len(parts) == 3 and parts[1] in FIBSEM_SERVERS:
        return parts[2] in WATCH_TOOLS
    return False


def may_run(text: str) -> bool:
    """Whether the subagent file *text* lists no tool that acts on the
    microscope: no tools line (every tool), a subagent of its own or one
    acting tool says no."""
    data, _ = subagents.front_matter(text)
    if not data or not data.get("tools"):
        return False
    try:
        tools = subagents.names(data.get("tools"))
    except subagents.SubagentError:
        return False
    starts = [tool for tool in tools if tool.split("(")[0] in AGENT_TOOLS]
    return not starts and all(off_microscope(tool) for tool in tools)


@dataclass
class Run:
    """A subagent's run: the Agent call that started it."""

    call_id: str
    subagent: str
    background: bool
    task_id: Optional[str] = None
    # The transcript line of its Agent call, which the page names it by.
    seq: Optional[int] = None
    # The operator's messages it hasn't read yet: (message id, text).
    waiting: List[tuple] = field(default_factory=list)


class Runs:
    """The session's subagent runs while they last, for the hooks and the
    harness. *folder*: the session's working folder, whose
    ``.claude/agents/`` holds the subagents it started with."""

    def __init__(self, folder: Optional[Path] = None) -> None:
        self.folder = folder
        self._runs: Dict[str, Run] = {}
        self._may: Dict[str, bool] = {}
        # Told of a message delivered to a run, and of one it never read:
        # (run, message id, text).
        self.on_delivered: Optional[Callable[[Run, str, str], None]] = None
        self.on_unread: Optional[Callable[[Run, str, str], None]] = None

    def may_run_in_background(self, name: str) -> bool:
        if name not in self._may:
            path = (
                self.folder / ".claude" / "agents" / f"{name}.md"
                if self.folder is not None
                else None
            )
            try:
                text = path.read_text(encoding="utf-8") if path else ""
            except OSError:
                text = ""
            self._may[name] = bool(text) and may_run(text)
        return self._may[name]

    def running_in_background(self) -> int:
        return sum(1 for run in self._runs.values() if run.background)

    def called(self, call_id: str, subagent: str, background: bool) -> Run:
        """An Agent call allowed (`agent_hook`), or shown (`shown`), whichever
        comes first: the CLI asks the hook and prints the call side by side."""
        run = self._runs.get(call_id)
        if run is None:
            run = self._runs[call_id] = Run(call_id, subagent, background)
        else:
            run.subagent, run.background = subagent or run.subagent, background
        return run

    def shown(self, call_id: str, seq: Optional[int]) -> None:
        """The call's transcript line, which the operator's messages name."""
        run = self._runs.get(call_id)
        if run is None:
            run = self._runs[call_id] = Run(call_id, "", False)
        run.seq = seq

    def get(self, call_id: Optional[str]) -> Optional[Run]:
        return self._runs.get(call_id) if call_id else None

    def started(self, call_id: Optional[str], task_id: Optional[str]) -> None:
        run = self.get(call_id)
        if run is not None and task_id:
            run.task_id = task_id

    def of_task(self, task_id: Optional[str]) -> Optional[Run]:
        for run in self._runs.values():
            if task_id and run.task_id == task_id:
                return run
        return None

    def ended(self, call_id: Optional[str]) -> Optional[Run]:
        """The run is over: its unread messages go to the lead."""
        run = self._runs.pop(call_id, None) if call_id else None
        if run is not None and run.waiting and self.on_unread is not None:
            for message_id, text in run.waiting:
                self.on_unread(run, message_id, text)
            run.waiting = []
        return run

    def write(self, call_id: Any, message_id: str, text: str) -> bool:
        """The operator's message to the run of the Agent call *call_id*:
        kept for its next tool call. False when no such run is going."""
        run = self.get(str(call_id)) if call_id else None
        if run is None:
            return False
        run.waiting.append((message_id, text))
        return True

    def take(self, task_id: Optional[str]) -> List[tuple]:
        run = self.of_task(task_id)
        if run is None or not run.waiting:
            return []
        waiting, run.waiting = run.waiting, []
        for message_id, text in waiting:
            if self.on_delivered is not None:
                self.on_delivered(run, message_id, text)
        return waiting


def _deny(event: str, why: str) -> dict:
    return {
        "hookSpecificOutput": {
            "hookEventName": event,
            "permissionDecision": "deny",
            "permissionDecisionReason": why,
        }
    }


def agent_hook(runs: Runs):
    """PreToolUse on the Agent tool: a call in the background only for a
    subagent that reads, from the lead, and while fewer than `MAX_RUNNING`
    run there; every allowed call is kept. On Bash: never in the background."""

    async def hook(hook_input, tool_use_id, context):
        arguments = hook_input.get("tool_input") or {}
        background = bool(arguments.get("run_in_background"))
        if hook_input.get("tool_name") not in AGENT_TOOLS:
            # Code mode's shell: only subagents run in the background.
            return _deny("PreToolUse", SHELL) if background else {}
        name = str(arguments.get("subagent_type") or "")
        if background:
            why = None
            if hook_input.get("agent_id"):
                why = NESTED.format(name=name)
            elif not runs.may_run_in_background(name):
                why = NOT_HERE.format(name=name)
            elif runs.running_in_background() >= MAX_RUNNING:
                why = FULL.format(count=runs.running_in_background())
            if why is not None:
                log.info("refused %s in the background: %s", name, why)
                return _deny("PreToolUse", why)
        if tool_use_id:
            runs.called(tool_use_id, name, background)
        return {}

    return hook


def lease_hook(runs: Runs):
    """PreToolUse on every tool: a background run's call that may act on the
    microscope is refused."""

    async def hook(hook_input, tool_use_id, context):
        task_id = hook_input.get("agent_id")
        run = runs.of_task(task_id) if task_id else None
        if run is None or not run.background:
            return {}
        tool = str(hook_input.get("tool_name") or "")
        if off_microscope(tool):
            return {}
        log.info("refused %s from %s in the background", tool, run.subagent)
        return _deny("PreToolUse", LEASE.format(tool=tool.split("__")[-1]))

    return hook


def deliver_hook(runs: Runs):
    """PostToolUse on every tool: the operator's waiting messages to the run
    that made the call, after its result."""

    async def hook(hook_input, tool_use_id, context):
        waiting = runs.take(hook_input.get("agent_id"))
        if not waiting:
            return {}
        said = "\n\n".join(
            DELIVERED.format(id=message_id, text=text) for message_id, text in waiting
        )
        return {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": said,
            }
        }

    return hook


def with_hooks(hooks: Optional[dict], runs: Optional[Runs]) -> Optional[dict]:
    """*hooks* with the background hooks first."""
    if runs is None:
        return hooks
    from claude_agent_sdk import HookMatcher

    hooks = dict(hooks or {})
    hooks["PreToolUse"] = [
        HookMatcher(matcher="Agent|Task|Bash", hooks=[agent_hook(runs)]),
        HookMatcher(hooks=[lease_hook(runs)]),
    ] + list(hooks.get("PreToolUse", []))
    hooks["PostToolUse"] = [HookMatcher(hooks=[deliver_hook(runs)])] + list(
        hooks.get("PostToolUse", [])
    )
    return hooks
