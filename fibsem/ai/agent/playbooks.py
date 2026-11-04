"""Playbooks: how a kind of job is run, as a YAML file the agent runs with.

    # playbooks/autolamella.yaml
    description: On-grid lamellae, not correlative.
    extends: other-playbook          # optional
    skills: [autolamella]            # read at the start of the job
    optional:                        # read only when the job needs them
      - skill: autolamella-polishing
        when: before the first polishing pass
        reminders:                   # on only while that skill is loaded
          polishing_pass: Measure the band before you decide.
    reminders:
      before_milling: Confirm the current and depth.
    rules:
      - Stop at the operator's thickness target, 200 nm by default.

A playbook is ``<name>.yaml`` in the lab's ``~/.fibsem/knowledge/playbooks/``
or the shipped ``.claude/fibsem-agent/playbooks/``, the lab's winning. Skills,
optional skills, reminders and rules add up along ``extends``, the parent's
first. A profile can name one (``playbook:``); else the agent proposes one at
intake and sets it with ``playbook_set`` once the operator agrees. No
playbook is a valid state: the profile's reminders still hold.

`Reminders` is what is on now: the profile's, the playbook's, and those of the
optional skills the agent has loaded and not yet finished. The SDK's hooks ask
it at each call, so setting a playbook or loading a skill needs no restart.
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from fibsem.ai.agent import profiles, sources

PLAYBOOK_DIR = Path(sources.CLAUDE) / "fibsem-agent" / "playbooks"
FIELDS = ("description", "extends", "skills", "optional", "reminders", "rules")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


class PlaybookError(ValueError):
    """A playbook that can't be used, with the reason in words."""


@dataclass
class Optional_:
    """An optional skill: read *when* the job needs it, its reminders on while
    it is loaded."""

    skill: str
    when: str
    hooks: List[profiles.Hook] = field(default_factory=list)


@dataclass
class Playbook:
    name: str
    path: Path
    hash: str
    description: Optional[str] = None
    # The chain, this playbook last: ["autolamella", "autolamella-correlative"].
    chain: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    optional: List[Optional_] = field(default_factory=list)
    hooks: List[profiles.Hook] = field(default_factory=list)
    rules: List[str] = field(default_factory=list)

    def optional_skill(self, name: str) -> Optional[Optional_]:
        return next((o for o in self.optional if o.skill == name), None)

    def summary(self) -> dict:
        """The playbook as the agent and the Agent page read it."""
        return {
            "name": self.name,
            "description": self.description,
            "extends": self.chain[:-1],
            "skills": list(self.skills),
            "optional": [{"skill": o.skill, "when": o.when} for o in self.optional],
            "reminders": [
                {"when": h.when, "label": _label(h.when), "say": h.say}
                for h in self.hooks
            ],
            "rules": list(self.rules),
        }


def _label(when: str) -> str:
    option = profiles.HOOK_OPTIONS.get(when)
    return option.label if option else when


def dirs(checkout: Optional[Path]) -> List[Path]:
    """Where playbooks are looked for, the first found winning."""
    found = [sources.lab_folder("playbooks")]
    if checkout is not None:
        found.append(checkout / PLAYBOOK_DIR)
    return found


def find(name: str, checkout: Optional[Path]) -> Optional[Path]:
    if not _NAME.fullmatch(name or ""):
        return None
    for folder in dirs(checkout):
        path = folder / f"{name}.yaml"
        if path.is_file():
            return path
    return None


def available(checkout: Optional[Path]) -> List[str]:
    """Every playbook's name, sorted."""
    names = set()
    for folder in dirs(checkout):
        if folder.is_dir():
            names.update(path.stem for path in folder.glob("*.yaml"))
    return sorted(names)


def load(name: str, checkout: Optional[Path]) -> Playbook:
    """Playbook *name* with what it extends added in; a `PlaybookError` says
    what is wrong with it."""
    return _load(name, checkout, [])


def _load(name: str, checkout: Optional[Path], seen: List[str]) -> Playbook:
    if name in seen:
        raise PlaybookError(
            f"playbook {name} extends itself: {' -> '.join(seen + [name])}"
        )
    path = find(name, checkout)
    if path is None:
        where = " or ".join(str(d) for d in dirs(checkout))
        if seen:
            raise PlaybookError(f"{seen[-1]} extends {name}, which isn't in {where}")
        raise PlaybookError(f"No playbook {name!r} in {where}")
    raw = path.read_bytes()
    try:
        data = yaml.safe_load(raw) or {}
    except yaml.YAMLError as error:
        raise PlaybookError(f"{path} isn't YAML: {error}") from None
    if not isinstance(data, dict):
        raise PlaybookError(f"{path} must be a mapping")
    unknown = sorted(set(data) - set(FIELDS))
    if unknown:
        raise PlaybookError(f"{path}: unknown field(s) {', '.join(unknown)}")
    parent = None
    if data.get("extends") is not None:
        extends = data["extends"]
        if not isinstance(extends, str):
            raise PlaybookError(f"{path}: extends is one playbook's name")
        parent = _load(extends, checkout, seen + [name])
    skills = _names(data.get("skills"), "skills", path)
    optional = _optional(data.get("optional"), path)
    hooks = _reminders(data.get("reminders"), path)
    rules = _names(data.get("rules"), "rules", path)
    if checkout is not None:
        missing = [
            skill
            for skill in skills + [o.skill for o in optional]
            if sources.skill_dir(skill, checkout) is None
        ]
        if missing:
            raise PlaybookError(f"{path}: no skill {', '.join(missing)}")
    digest = hashlib.sha256(raw)
    if parent is not None:
        digest.update(parent.hash.encode())
    description = data.get("description")
    playbook = Playbook(
        name=name,
        path=path,
        hash=digest.hexdigest()[:12],
        description=str(description).strip() if description else None,
        chain=(parent.chain if parent else []) + [name],
        skills=_merge(parent.skills if parent else [], skills),
        optional=_merge_optional(parent.optional if parent else [], optional),
        hooks=(parent.hooks if parent else []) + hooks,
        rules=_merge(parent.rules if parent else [], rules),
    )
    return playbook


def _names(value: Any, key: str, path: Path) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(
        isinstance(v, str) and v.strip() for v in value
    ):
        raise PlaybookError(f"{path}: {key} must be a list of text")
    return [v.strip() for v in value]


def _reminders(value: Any, path: Path) -> List[profiles.Hook]:
    """``{when: say}`` as hooks, in the file's order."""
    if value is None:
        return []
    if not isinstance(value, dict):
        raise PlaybookError(f"{path}: reminders is a mapping of when to what to say")
    hooks = []
    for when, say in value.items():
        if when not in profiles.HOOK_OPTIONS:
            raise PlaybookError(
                f"{path}: no hook option {when!r} "
                f"(one of {', '.join(profiles.HOOK_OPTIONS)}; fibsem-agent --list-hooks)"
            )
        if not isinstance(say, str) or not say.strip():
            raise PlaybookError(f"{path}: the {when} reminder says nothing")
        hooks.append(profiles.Hook(when, " ".join(say.split())))
    return hooks


def _optional(value: Any, path: Path) -> List[Optional_]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise PlaybookError(f"{path}: optional is a list of skills")
    found = []
    for entry in value:
        if not isinstance(entry, dict) or not {"skill", "when"} <= set(entry):
            raise PlaybookError(f"{path}: each optional skill has a skill and a when")
        extra = sorted(set(entry) - {"skill", "when", "reminders"})
        if extra:
            raise PlaybookError(f"{path}: unknown field(s) {', '.join(extra)}")
        skill, when = entry["skill"], entry["when"]
        if not isinstance(skill, str) or not isinstance(when, str) or not when.strip():
            raise PlaybookError(f"{path}: an optional skill's skill and when are text")
        found.append(
            Optional_(
                skill, " ".join(when.split()), _reminders(entry.get("reminders"), path)
            )
        )
    return found


def _merge(first: List[str], then: List[str]) -> List[str]:
    return first + [item for item in then if item not in first]


def _merge_optional(first: List[Optional_], then: List[Optional_]) -> List[Optional_]:
    """The parent's, then the child's; a child's entry for the same skill
    replaces its when and adds its reminders."""
    merged = {o.skill: Optional_(o.skill, o.when, list(o.hooks)) for o in first}
    for o in then:
        if o.skill in merged:
            merged[o.skill].when = o.when
            merged[o.skill].hooks += o.hooks
        else:
            merged[o.skill] = Optional_(o.skill, o.when, list(o.hooks))
    return list(merged.values())


class Reminders:
    """What is said at each moment now: the profile's reminders, the
    playbook's, and those of its optional skills while they are loaded. All
    that apply are said together, profile first, duplicates dropped."""

    def __init__(
        self,
        base: Optional[List[profiles.Hook]] = None,
        on_change: Optional[Callable[[], None]] = None,
    ) -> None:
        self.base = list(base or [])
        self.playbook: Optional[Playbook] = None
        self.chosen_by: Optional[str] = None
        # Optional skills: name -> "loaded" | "done".
        self.skills: Dict[str, str] = {}
        self.done: Dict[str, str] = {}
        self._on_change = on_change

    def set(self, playbook: Optional[Playbook], chosen_by: str) -> None:
        self.playbook, self.chosen_by = playbook, chosen_by
        self.skills, self.done = {}, {}
        self._changed()

    def loaded(self, skill: str) -> Optional[Optional_]:
        """*skill* was read: an optional one's reminders go on. Answers the
        playbook's entry for it, else None."""
        entry = self.playbook.optional_skill(skill) if self.playbook else None
        if entry is not None and self.skills.get(skill) != "loaded":
            self.skills[skill] = "loaded"
            self._changed()
        return entry

    def finished(self, skill: str, summary: str) -> bool:
        """An optional skill is done: its reminders go off. False if it
        isn't one the playbook has loaded."""
        if self.skills.get(skill) != "loaded":
            return False
        self.skills[skill] = "done"
        self.done[skill] = summary
        self._changed()
        return True

    def texts(self, when: str) -> List[str]:
        said: List[str] = []
        hooks = list(self.base)
        if self.playbook is not None:
            hooks += self.playbook.hooks
            for entry in self.playbook.optional:
                if self.skills.get(entry.skill) == "loaded":
                    hooks += entry.hooks
        for hook in hooks:
            if hook.when == when and hook.say not in said:
                said.append(hook.say)
        return said

    def for_lines(self, lines: List[str]) -> List[str]:
        """What is said with a turn of watcher *lines*: the reminders of the
        watcher options a line matches."""
        said: List[str] = []
        for when, option in profiles.HOOK_OPTIONS.items():
            if option.event == profiles.WATCHER and option.on_lines(lines):
                said += [t for t in self.texts(when) if t not in said]
        return said

    def state(self) -> dict:
        """For the session record and the Agent page."""
        if self.playbook is None:
            return {"playbook": None}
        return {
            "playbook": {
                "name": self.playbook.name,
                "hash": self.playbook.hash,
                "chosen_by": self.chosen_by,
                "skills": [
                    {
                        "skill": o.skill,
                        "state": self.skills.get(o.skill, "not loaded"),
                        **(
                            {"summary": self.done[o.skill]}
                            if o.skill in self.done
                            else {}
                        ),
                    }
                    for o in self.playbook.optional
                ],
            }
        }

    def restore(self, state: dict, checkout: Optional[Path]) -> Optional[str]:
        """Pick up a resumed session's playbook; why not, if it can't be."""
        given = (state or {}).get("playbook")
        if not given:
            return None
        try:
            playbook = load(given["name"], checkout)
        except PlaybookError as error:
            return str(error)
        self.playbook, self.chosen_by = playbook, given.get("chosen_by")
        for row in given.get("skills") or []:
            if row.get("state") in ("loaded", "done"):
                self.skills[row["skill"]] = row["state"]
            if row.get("summary"):
                self.done[row["skill"]] = row["summary"]
        return None

    def _changed(self) -> None:
        if self._on_change is not None:
            self._on_change()

    def sdk_hooks(self, server: str) -> dict:
        """One SDK matcher per hook option, each asking what is on now."""
        from claude_agent_sdk import HookMatcher

        matchers: Dict[str, list] = {}
        for when, option in profiles.HOOK_OPTIONS.items():
            if option.event == profiles.WATCHER:
                continue
            matchers.setdefault(option.event, []).append(
                HookMatcher(
                    matcher=f"mcp__{server}__({option.tools})",
                    hooks=[self.callback(when, server)],
                )
            )
        return matchers

    def callback(self, when: str, server: str):
        option = profiles.HOOK_OPTIONS[when]

        async def hook(hook_input, tool_use_id, context):
            texts = self.texts(when)
            if not texts or not option.applies(hook_input, server):
                return {}
            return {
                "hookSpecificOutput": {
                    "hookEventName": option.event,
                    "additionalContext": profiles.reminder(texts),
                }
            }

        return hook
