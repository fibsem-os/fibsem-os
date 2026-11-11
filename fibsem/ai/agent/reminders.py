"""Reminders: what is said to the agent at the moment it makes a matching
tool call, so it doesn't rely on remembering a skill. A playbook lists its own
(``reminders: {when: say}``); ``HOOK_OPTIONS`` is the one list of when a
reminder can fire: which tools, before or after the call. An option on one
instrument kind's tools is for playbooks of that kind only (`fits`).

    fibsem-agent --list-hooks
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class HookOption:
    """When a reminder fires: *event* is the SDK's hook event, *tools* a
    pattern on the fibsem tool's name."""

    label: str
    event: str
    tools: str
    # For a failure: only a server error (5xx), never a refusal.
    server_errors_only: bool = False
    # After the call: only when its answer has this in it.
    answer_has: str = ""
    # Only when these arguments match (a pattern, searched, any case).
    input_matches: Tuple[Tuple[str, str], ...] = ()
    # For event WATCHER: a pattern on the watcher's lines, said with that turn.
    line: str = ""
    # The instrument kind whose tools it fires on, or "" for any kind.
    instrument: str = ""

    def fits(self, kind: Optional[str]) -> bool:
        """Whether a playbook of instrument *kind* can use it."""
        return not self.instrument or self.instrument == kind

    def on_lines(self, lines: List[str]) -> bool:
        return bool(self.line) and any(re.search(self.line, x) for x in lines)

    def applies(self, hook_input: dict, server: str) -> bool:
        name = str(hook_input.get("tool_name", ""))
        prefix = f"mcp__{server}__"
        if not name.startswith(prefix):
            return False
        if not re.fullmatch(self.tools, name[len(prefix) :]):
            return False
        if self.server_errors_only:
            return str(hook_input.get("error", "")).startswith("failed (5")
        given = hook_input.get("tool_input") or {}
        for key, pattern in self.input_matches:
            if not re.search(pattern, str(given.get(key, "")), re.IGNORECASE):
                return False
        if self.answer_has:
            return self.answer_has in str(hook_input.get("tool_response", ""))
        return True


# An option said with a turn of watcher lines, not by an SDK hook on a call.
WATCHER = "Watcher"

# The FIB-SEM's kind, for the options on its tools (milling, beams, alignment)
# and AutoLamella's tasks.
FIBSEM = "fibsem"

HOOK_OPTIONS: Dict[str, HookOption] = {
    "site": HookOption(
        "At a site",
        "PreToolUse",
        "item_add|item_set_position|item_set_pose|item_move_to_pose",
    ),
    "before_milling": HookOption(
        "Before milling", "PreToolUse", "milling_run_.*", instrument=FIBSEM
    ),
    "after_auto": HookOption(
        "After an auto step",
        "PostToolUse",
        "beam_auto_.*|fm_auto_focus|alignment_.*",
        instrument=FIBSEM,
    ),
    # An Iterative Polishing pass has ended: only its facts carry "Pass data".
    "after_polish_pass": HookOption(
        "After a polishing pass",
        "PostToolUse",
        "service_wait_for_run",
        answer_has="Pass data",
        instrument=FIBSEM,
    ),
    # Deciding or suggesting one of Setup's proposals: its position, point of
    # interest or alignment area. By the task's name in the protocol.
    "setup_placement": HookOption(
        "Placing a lamella at Setup",
        "PreToolUse",
        "proposal_decide|proposal_suggest",
        input_matches=(("task_name", r"setup|select.*position"),),
        instrument=FIBSEM,
    ),
    # Deciding or suggesting a pass of the Polishing task's mill.
    "polishing_pass": HookOption(
        "Deciding a polishing pass",
        "PreToolUse",
        "proposal_decide|proposal_suggest",
        input_matches=(("task_name", r"polish"),),
        instrument=FIBSEM,
    ),
    # A task finished: the watcher's line, not a tool call.
    "after_task": HookOption(
        "After a task finishes", WATCHER, "", line=r"^EVENT task_completed\b"
    ),
    "hardware_error": HookOption(
        "On a hardware error", "PostToolUseFailure", ".*", server_errors_only=True
    ),
    "workflow_start": HookOption(
        "When a workflow starts", "PreToolUse", "workflow_start|workflow_start_grids"
    ),
}


@dataclass
class Hook:
    when: str
    say: str


def reminder(texts: List[str]) -> str:
    return "Reminder: " + " ".join(texts)


def callback(option: HookOption, texts: List[str], server: str):
    """The SDK hook for one option: its reminders when the call matches."""

    async def hook(hook_input, tool_use_id, context):
        if not option.applies(hook_input, server):
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": option.event,
                "additionalContext": reminder(texts),
            }
        }

    return hook


def sdk_hooks(hooks: List[Hook], server: str) -> Optional[dict]:
    """*hooks* as the SDK's ``hooks`` option, or None."""
    if not hooks:
        return None
    from claude_agent_sdk import HookMatcher

    said: Dict[str, List[str]] = {}
    for hook in hooks:
        said.setdefault(hook.when, []).append(hook.say)
    matchers: Dict[str, list] = {}
    for when, texts in said.items():
        option = HOOK_OPTIONS[when]
        if option.event == WATCHER:
            continue
        matchers.setdefault(option.event, []).append(
            HookMatcher(
                matcher=f"mcp__{server}__({option.tools})",
                hooks=[callback(option, texts, server)],
            )
        )
    return matchers


def hook_rows() -> List[dict]:
    """``--list-hooks``: each option, when it fires and on which tools."""
    rows = []
    for key, option in HOOK_OPTIONS.items():
        timing = {
            "PreToolUse": "before the call",
            "PostToolUse": "after the call",
            "PostToolUseFailure": "when the call fails with a server error",
            WATCHER: "on the watcher's line",
        }[option.event]
        rows.append(
            {
                "when": key,
                "label": option.label,
                "fires": timing,
                "tools": option.tools or option.line,
                "instrument": option.instrument or "any",
            }
        )
    return rows
