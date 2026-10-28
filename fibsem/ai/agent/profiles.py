"""``fibsem-agent`` profiles: a YAML file that names what an agent runs with.

    fibsem-agent --profile overnight-lamella
    fibsem-agent --profile overnight-lamella --show-config
    fibsem-agent --list-profiles
    fibsem-agent --list-hooks

A profile is ``<name>.yaml`` in this machine's ``~/.fibsem/agent/profiles/``
or in the checkout's ``.claude/fibsem-agent/`` (the machine's wins). It sets
the model, effort, budget and turns, the mode and its folder, extra skills,
instructions added to the prompt, and reminders (hooks). Flags on the command
line win over it. The level and any secret never go in a profile: the
operator sets the level, and the API key and server token come from the
environment.

A reminder is said to the agent at the moment it makes a matching tool call,
so it doesn't rely on remembering a skill. ``HOOK_OPTIONS`` is the one list of
when a reminder can fire: which tools, before or after the call.
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from fibsem import paths

PROFILE_DIR = Path(".claude") / "fibsem-agent"
# The profile fibsem-agent and the gateway run with when none is named.
DEFAULT_PROFILE = "fibsem-agent"
MODES = ("none", "files", "code")
FIELDS = (
    "name",
    "model",
    "effort",
    "max_budget_usd",
    "max_turns",
    "mode",
    "folder",
    "skills",
    "prompt",
    "hooks",
)


def machine_dir() -> Path:
    return paths.user_dir() / "agent" / "profiles"


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

    def applies(self, hook_input: dict, server: str) -> bool:
        name = str(hook_input.get("tool_name", ""))
        prefix = f"mcp__{server}__"
        if not name.startswith(prefix):
            return False
        if not re.fullmatch(self.tools, name[len(prefix) :]):
            return False
        if self.server_errors_only:
            return str(hook_input.get("error", "")).startswith("failed (5")
        if self.answer_has:
            return self.answer_has in str(hook_input.get("tool_response", ""))
        return True


HOOK_OPTIONS: Dict[str, HookOption] = {
    "site": HookOption(
        "At a site",
        "PreToolUse",
        "item_add|item_set_position|item_set_pose|item_move_to_pose",
    ),
    "before_milling": HookOption("Before milling", "PreToolUse", "milling_run_.*"),
    "after_auto": HookOption(
        "After an auto step",
        "PostToolUse",
        "beam_auto_.*|fm_auto_focus|alignment_.*",
    ),
    # An Iterative Polishing pass has ended: only its facts carry "Pass data".
    "after_polish_pass": HookOption(
        "After a polishing pass",
        "PostToolUse",
        "service_wait_for_run",
        answer_has="Pass data",
    ),
    "hardware_error": HookOption(
        "On a hardware error", "PostToolUseFailure", ".*", server_errors_only=True
    ),
    "workflow_start": HookOption(
        "When a workflow starts", "PreToolUse", "workflow_start|workflow_start_grids"
    ),
}


class ProfileError(ValueError):
    """A profile that can't be used, with the reason in words."""


@dataclass
class Hook:
    when: str
    say: str


@dataclass
class Profile:
    name: str
    path: Path
    hash: str
    model: Optional[str] = None
    effort: Optional[str] = None
    max_budget_usd: Optional[float] = None
    max_turns: Optional[int] = None
    mode: str = "none"
    folder: Optional[str] = None
    skills: List[str] = field(default_factory=list)
    prompt: Optional[str] = None
    hooks: List[Hook] = field(default_factory=list)


def dirs(checkout: Optional[Path]) -> List[Path]:
    """Where profiles are looked for, the first found winning."""
    found = [machine_dir()]
    if checkout is not None:
        found.append(checkout / PROFILE_DIR)
    return found


def find(name: str, checkout: Optional[Path]) -> Optional[Path]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        return None
    for folder in dirs(checkout):
        path = folder / f"{name}.yaml"
        if path.is_file():
            return path
    return None


def available(checkout: Optional[Path]) -> List[Path]:
    """Every profile's file, one per name, by name."""
    by_name: Dict[str, Path] = {}
    for folder in dirs(checkout):
        if folder.is_dir():
            for path in sorted(folder.glob("*.yaml")):
                by_name.setdefault(path.stem, path)
    return [by_name[name] for name in sorted(by_name)]


def load(path: Path, efforts=(), checkout: Optional[Path] = None) -> Profile:
    """The profile in *path*; a `ProfileError` says what is wrong with it."""
    raw = path.read_bytes()
    try:
        data = yaml.safe_load(raw) or {}
    except yaml.YAMLError as error:
        raise ProfileError(f"{path} isn't YAML: {error}") from None
    if not isinstance(data, dict):
        raise ProfileError(f"{path} must be a mapping of settings")
    unknown = sorted(set(data) - set(FIELDS))
    if unknown:
        raise ProfileError(f"{path}: unknown setting(s) {', '.join(unknown)}")
    profile = Profile(
        name=str(data.get("name") or path.stem),
        path=path,
        hash=hashlib.sha256(raw).hexdigest()[:12],
        model=_text(data, "model", path),
        effort=_text(data, "effort", path),
        max_budget_usd=_number(data, "max_budget_usd", float, path),
        max_turns=_number(data, "max_turns", int, path),
        mode=_text(data, "mode", path) or "none",
        folder=_text(data, "folder", path),
        skills=_names(data.get("skills"), path),
        prompt=_text(data, "prompt", path),
        hooks=_hooks(data.get("hooks"), path),
    )
    if profile.mode not in MODES:
        raise ProfileError(f"{path}: mode is one of {', '.join(MODES)}")
    if profile.mode != "none" and not profile.folder:
        raise ProfileError(f"{path}: mode {profile.mode} needs a folder")
    if efforts and profile.effort is not None and profile.effort not in efforts:
        raise ProfileError(f"{path}: effort is one of {', '.join(efforts)}")
    if checkout is not None:
        missing = [
            name
            for name in profile.skills
            if not (checkout / ".claude" / "skills" / name / "SKILL.md").is_file()
        ]
        if missing:
            raise ProfileError(f"{path}: no skill {', '.join(missing)} in {checkout}")
    return profile


def _text(data: dict, key: str, path: Path) -> Optional[str]:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ProfileError(f"{path}: {key} must be text")
    return value.strip() or None


def _number(data: dict, key: str, kind, path: Path):
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ProfileError(f"{path}: {key} must be a number above 0")
    return kind(value)


def _names(value: Any, path: Path) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ProfileError(f"{path}: skills must be a list of names")
    return list(value)


def _hooks(value: Any, path: Path) -> List[Hook]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ProfileError(f"{path}: hooks must be a list")
    hooks = []
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != {"when", "say"}:
            raise ProfileError(f"{path}: each hook is a when and a say")
        when, say = entry["when"], entry["say"]
        if when not in HOOK_OPTIONS:
            raise ProfileError(
                f"{path}: no hook option {when!r} "
                f"(one of {', '.join(HOOK_OPTIONS)}; fibsem-agent --list-hooks)"
            )
        if not isinstance(say, str) or not say.strip():
            raise ProfileError(f"{path}: the {when} hook says nothing")
        hooks.append(Hook(when, say.strip()))
    return hooks


def reminder(texts: List[str]) -> str:
    return "Reminder from your profile: " + " ".join(texts)


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
    """The profile's reminders as the SDK's ``hooks`` option, or None."""
    if not hooks:
        return None
    from claude_agent_sdk import HookMatcher

    said: Dict[str, List[str]] = {}
    for hook in hooks:
        said.setdefault(hook.when, []).append(hook.say)
    matchers: Dict[str, list] = {}
    for when, texts in said.items():
        option = HOOK_OPTIONS[when]
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
        }[option.event]
        rows.append(
            {"when": key, "label": option.label, "fires": timing, "tools": option.tools}
        )
    return rows
