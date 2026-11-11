"""Playbooks: how a kind of job is run, as a YAML file the agent runs with.

    # playbooks/autolamella.yaml
    description: On-grid lamellae, not correlative.
    extends: fibsem-agent            # optional
    instrument: fibsem               # the instrument kind its prompts are for
    session:                         # what a session STARTED on it runs with
      level: watch                   # the level the gateway's card picks first
      model: opus
      effort: high
      max_budget_usd: 500
      max_turns: 40
      mode: files                    # none, files or code
      wake_max_min: 240              # the longest wake-up the agent can set
      deadline_warn_min: 30          # a deadline wakes it this long before
      quiet_min: 20                  # a quiet running workflow wakes it after
    skills: [autolamella]            # read at the start of the job
    optional:                        # read only when the job needs them
      - skill: autolamella-polishing
        when: before the first polishing pass
        reminders:                   # on only while that skill is loaded
          polishing_pass: Measure the band before you decide.
        agents: [autolamella-verifier]   # subagents, on while it is loaded
    agents: [my-checker]             # subagents, on with the playbook
    reminders:
      before_milling: Confirm the current and depth.
    rules:
      - Stop at the operator's thickness target, 200 nm by default.
    plan: [Brief, Playbook, Rules, Steps, Progress, Summary]   # plan.md's headings

A playbook is ``<name>.yaml`` in the lab's ``~/.fibsem/knowledge/playbooks/``
or the shipped ``.claude/fibsem-agent/playbooks/``, the lab's winning. Skills,
optional skills, subagents, reminders and rules add up along ``extends``, the parent's
first; ``session`` settings and ``instrument`` too, the child's winning.

The shipped base playbook is the one place the agent's defaults live: it sits
under every playbook, a lab's copy of the base included, so its session
settings and its instrument kind hold wherever nothing overrides them, and its
skills under every playbook of its instrument kind (`load`). A session starts on the
playbook its launch names, else on ``fibsem-agent``, the base the shipped ones
extend; the agent proposes a better fit at intake and sets it with
``playbook_set`` once the operator agrees. ``session`` counts only at the
start: a running session keeps its model, budget and mode.

`Reminders` is what is on now: the playbook's, and those of the
optional skills the agent has loaded and not yet finished. The SDK's hooks ask
it at each call, so setting a playbook or loading a skill needs no restart.
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from fibsem.ai.agent import instruments, reminders, sources

PLAYBOOK_DIR = Path(sources.CLAUDE) / "fibsem-agent" / "playbooks"
# The playbook a session starts on when its launch names none.
DEFAULT = "fibsem-agent"
FIELDS = (
    "description",
    "extends",
    "instrument",
    "session",
    "skills",
    "optional",
    "agents",
    "reminders",
    "rules",
    "plan",
)
# plan.md's headings when a playbook names none (fibsem.server.agent.plan).
PLAN = ["Brief", "Playbook", "Rules", "Steps", "Progress", "Summary"]
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
SESSION = (
    "level",
    "model",
    "effort",
    "max_budget_usd",
    "max_turns",
    "mode",
    "folder",
    "wake_max_min",
    "deadline_warn_min",
    "quiet_min",
)
# Numbers above 0, and the type each is read as.
NUMBERS = {
    "max_budget_usd": float,
    "max_turns": int,
    "wake_max_min": float,
    "deadline_warn_min": float,
    "quiet_min": float,
}
MODES = ("none", "files", "code")
LEVELS = ("watch", "assist", "drive")


class PlaybookError(ValueError):
    """A playbook that can't be used, with the reason in words."""


@dataclass
class Session:
    """What a session started on the playbook runs with; None is the default."""

    level: Optional[str] = None
    """The level the gateway's Agent card picks first; the operator still starts it."""
    model: Optional[str] = None
    effort: Optional[str] = None
    max_budget_usd: Optional[float] = None
    max_turns: Optional[int] = None
    mode: Optional[str] = None
    folder: Optional[str] = None
    wake_max_min: Optional[float] = None
    """The longest wake-up the agent can set, in minutes."""
    deadline_warn_min: Optional[float] = None
    """A deadline wakes the agent this many minutes before it, and at it."""
    quiet_min: Optional[float] = None
    """A running workflow quiet this many minutes wakes the agent, once."""

    def merged(self, child: "Session") -> "Session":
        """This, with *child*'s settings winning where it has them."""
        return Session(
            **{
                key: getattr(child, key)
                if getattr(child, key) is not None
                else getattr(self, key)
                for key in SESSION
            }
        )

    def as_dict(self) -> dict:
        return {
            key: getattr(self, key) for key in SESSION if getattr(self, key) is not None
        }


@dataclass
class Optional_:
    """An optional skill: read *when* the job needs it, its reminders on while
    it is loaded."""

    skill: str
    when: str
    hooks: List[reminders.Hook] = field(default_factory=list)
    # Subagents on while it is loaded.
    agents: List[str] = field(default_factory=list)


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
    hooks: List[reminders.Hook] = field(default_factory=list)
    rules: List[str] = field(default_factory=list)
    # The subagents its work calls (`subagents`), by name.
    agents: List[str] = field(default_factory=list)
    # plan.md's headings, the nearest playbook's that names them.
    plan: List[str] = field(default_factory=lambda: list(PLAN))
    session: Session = field(default_factory=Session)
    # The instrument kind its prompts are for (`instruments`).
    instrument: Optional[str] = None

    def optional_skill(self, name: str) -> Optional[Optional_]:
        return next((o for o in self.optional if o.skill == name), None)

    def all_skills(self) -> List[str]:
        """Its skills, the optional ones too."""
        return _merge(self.skills, [o.skill for o in self.optional])

    def all_agents(self) -> List[str]:
        """Its subagents, those of its optional skills too."""
        return _merge(self.agents, [a for o in self.optional for a in o.agents])

    def summary(self) -> dict:
        """The playbook as the agent and the Agent page read it."""
        return {
            "name": self.name,
            "description": self.description,
            "extends": self.chain[:-1],
            "instrument": self.instrument,
            "skills": list(self.skills),
            "optional": [
                {
                    "skill": o.skill,
                    "when": o.when,
                    **({"agents": o.agents} if o.agents else {}),
                }
                for o in self.optional
            ],
            "reminders": [
                {"when": h.when, "label": _label(h.when), "say": h.say}
                for h in self.hooks
            ],
            "rules": list(self.rules),
            "agents": list(self.agents),
            "plan": list(self.plan),
            "session": self.session.as_dict(),
        }


def _label(when: str) -> str:
    option = reminders.HOOK_OPTIONS.get(when)
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


def load(name: str, checkout: Optional[Path], efforts=()) -> Playbook:
    """Playbook *name* with what it extends added in, on the shipped base
    playbook's session settings and instrument kind, and its skills for a
    playbook of the base's instrument kind; a
    `PlaybookError` says what is wrong with it. *efforts*, when given, are
    the efforts allowed."""
    playbook = _load(name, checkout, [], tuple(efforts))
    base = shipped_base(checkout)
    if base is not None:
        playbook.instrument = playbook.instrument or base.instrument
        # The base's skills are for its instrument kind: a playbook for
        # another kind names its own.
        if playbook.instrument == base.instrument:
            playbook.skills = _merge(base.skills, playbook.skills)
        playbook.session = base.session.merged(playbook.session)
    if checkout is not None:
        kinds = instruments.kinds(checkout)
        if (playbook.instrument or instruments.DEFAULT) not in kinds:
            raise PlaybookError(
                f"{playbook.path}: no instrument kind {playbook.instrument!r} "
                f"(one of {', '.join(kinds) or 'none'})"
            )
    return playbook


def shipped_base(checkout: Optional[Path]) -> Optional[Playbook]:
    """The base playbook as fibsemOS ships it, never the lab's copy: where
    the agent's defaults live. None without a checkout or the file."""
    if checkout is None:
        return None
    path = checkout / PLAYBOOK_DIR / f"{DEFAULT}.yaml"
    if not path.is_file():
        return None
    return _read(DEFAULT, path, checkout, None, ())


def shipped_names(checkout: Optional[Path]) -> Dict[str, List[str]]:
    """Every skill (the optional ones too) and subagent the shipped playbooks
    name, in file order: what fibsem-agent may load of the checkout's
    ``.claude``, whose other skills are for developers."""
    found: Dict[str, List[str]] = {"skills": [], "agents": []}
    folder = checkout / PLAYBOOK_DIR if checkout is not None else None
    if folder is None or not folder.is_dir():
        return found
    for path in sorted(folder.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_bytes()) or {}
        except yaml.YAMLError:
            continue
        if not isinstance(data, dict):
            continue
        optional = [o for o in data.get("optional") or [] if isinstance(o, dict)]
        skills = list(data.get("skills") or []) + [o.get("skill") for o in optional]
        agents = list(data.get("agents") or []) + [
            a for o in optional for a in o.get("agents") or []
        ]
        for kind, names in (("skills", skills), ("agents", agents)):
            found[kind] = _merge(
                found[kind], [n for n in names if isinstance(n, str) and n]
            )
    return found


def _load(name: str, checkout: Optional[Path], seen: List[str], efforts=()) -> Playbook:
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
    return _read(name, path, checkout, seen, efforts)


def _read(
    name: str,
    path: Path,
    checkout: Optional[Path],
    seen: Optional[List[str]],
    efforts=(),
) -> Playbook:
    """The playbook in *path*; with *seen* (the chain so far), what it
    extends added in, else on its own."""
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
        if seen is not None:
            parent = _load(extends, checkout, seen + [name], efforts)
    instrument = data.get("instrument")
    if instrument is not None and (
        not isinstance(instrument, str) or not _NAME.fullmatch(instrument)
    ):
        raise PlaybookError(f"{path}: instrument is one instrument kind's name")
    skills = _names(data.get("skills"), "skills", path)
    optional = _optional(data.get("optional"), path)
    hooks = _reminders(data.get("reminders"), path)
    rules = _names(data.get("rules"), "rules", path)
    agents = _names(data.get("agents"), "agents", path)
    plan = _names(data.get("plan"), "plan", path)
    session = _session(data.get("session"), path, efforts)
    if any("#" in h or len(h) > 60 for h in plan):
        raise PlaybookError(
            f"{path}: plan lists headings, no # and at most 60 characters"
        )
    if checkout is not None:
        missing = [
            skill
            for skill in skills + [o.skill for o in optional]
            if sources.skill_dir(skill, checkout) is None
        ]
        if missing:
            raise PlaybookError(f"{path}: no skill {', '.join(missing)}")
        named = agents + [a for o in optional for a in o.agents]
        missing = [a for a in named if sources.agent_file(a, checkout) is None]
        if missing:
            raise PlaybookError(f"{path}: no subagent {', '.join(missing)}")
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
        agents=_merge(parent.agents if parent else [], agents),
        plan=plan or (parent.plan if parent else list(PLAN)),
        session=(parent.session if parent else Session()).merged(session),
        instrument=instrument or (parent.instrument if parent else None),
    )
    return playbook


def _session(value: Any, path: Path, efforts=()) -> Session:
    if value is None:
        return Session()
    if not isinstance(value, dict):
        raise PlaybookError(f"{path}: session is a mapping of settings")
    unknown = sorted(set(value) - set(SESSION))
    if unknown:
        raise PlaybookError(f"{path}: unknown session setting(s) {', '.join(unknown)}")
    found = {}
    for key in ("level", "model", "effort", "mode", "folder"):
        given = value.get(key)
        if given is None:
            continue
        if not isinstance(given, str) or not given.strip():
            raise PlaybookError(f"{path}: session {key} must be text")
        found[key] = given.strip()
    for key, kind in NUMBERS.items():
        given = value.get(key)
        if given is None:
            continue
        if isinstance(given, bool) or not isinstance(given, (int, float)) or given <= 0:
            raise PlaybookError(f"{path}: session {key} must be a number above 0")
        found[key] = kind(given)
    if found.get("level", "watch") not in LEVELS:
        raise PlaybookError(f"{path}: session level is one of {', '.join(LEVELS)}")
    if found.get("mode", "none") not in MODES:
        raise PlaybookError(f"{path}: session mode is one of {', '.join(MODES)}")
    if efforts and found.get("effort", efforts[0]) not in efforts:
        raise PlaybookError(f"{path}: session effort is one of {', '.join(efforts)}")
    return Session(**found)


def _names(value: Any, key: str, path: Path) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(
        isinstance(v, str) and v.strip() for v in value
    ):
        raise PlaybookError(f"{path}: {key} must be a list of text")
    return [v.strip() for v in value]


def _reminders(value: Any, path: Path) -> List[reminders.Hook]:
    """``{when: say}`` as hooks, in the file's order."""
    if value is None:
        return []
    if not isinstance(value, dict):
        raise PlaybookError(f"{path}: reminders is a mapping of when to what to say")
    hooks = []
    for when, say in value.items():
        if when not in reminders.HOOK_OPTIONS:
            raise PlaybookError(
                f"{path}: no hook option {when!r} "
                f"(one of {', '.join(reminders.HOOK_OPTIONS)}; fibsem-agent --list-hooks)"
            )
        if not isinstance(say, str) or not say.strip():
            raise PlaybookError(f"{path}: the {when} reminder says nothing")
        hooks.append(reminders.Hook(when, " ".join(say.split())))
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
        extra = sorted(set(entry) - {"skill", "when", "reminders", "agents"})
        if extra:
            raise PlaybookError(f"{path}: unknown field(s) {', '.join(extra)}")
        skill, when = entry["skill"], entry["when"]
        if not isinstance(skill, str) or not isinstance(when, str) or not when.strip():
            raise PlaybookError(f"{path}: an optional skill's skill and when are text")
        found.append(
            Optional_(
                skill,
                " ".join(when.split()),
                _reminders(entry.get("reminders"), path),
                _names(entry.get("agents"), "agents", path),
            )
        )
    return found


def _merge(first: List[str], then: List[str]) -> List[str]:
    return first + [item for item in then if item not in first]


def _merge_optional(first: List[Optional_], then: List[Optional_]) -> List[Optional_]:
    """The parent's, then the child's; a child's entry for the same skill
    replaces its when and adds its reminders."""
    merged = {
        o.skill: Optional_(o.skill, o.when, list(o.hooks), list(o.agents))
        for o in first
    }
    for o in then:
        if o.skill in merged:
            merged[o.skill].when = o.when
            merged[o.skill].hooks += o.hooks
            merged[o.skill].agents = _merge(merged[o.skill].agents, o.agents)
        else:
            merged[o.skill] = Optional_(o.skill, o.when, list(o.hooks), list(o.agents))
    return list(merged.values())


class Reminders:
    """What is said at each moment now: the playbook's reminders, and those
    of its optional skills while they are loaded. All that apply are said
    together, duplicates dropped."""

    def __init__(self, on_change: Optional[Callable[[], None]] = None) -> None:
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

    def subagents(self) -> List[str]:
        """The subagents on now: the playbook's, and those of its optional
        skills while they are loaded. None without a playbook."""
        if self.playbook is None:
            return []
        on = list(self.playbook.agents)
        for entry in self.playbook.optional:
            if self.skills.get(entry.skill) == "loaded":
                on += [a for a in entry.agents if a not in on]
        return on

    def texts(self, when: str) -> List[str]:
        said: List[str] = []
        hooks: List[reminders.Hook] = []
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
        for when, option in reminders.HOOK_OPTIONS.items():
            if option.event == reminders.WATCHER and option.on_lines(lines):
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
        for when, option in reminders.HOOK_OPTIONS.items():
            if option.event == reminders.WATCHER:
                continue
            matchers.setdefault(option.event, []).append(
                HookMatcher(
                    matcher=f"mcp__{server}__({option.tools})",
                    hooks=[self.callback(when, server)],
                )
            )
        return matchers

    def callback(self, when: str, server: str):
        option = reminders.HOOK_OPTIONS[when]

        async def hook(hook_input, tool_use_id, context):
            texts = self.texts(when)
            if not texts or not option.applies(hook_input, server):
                return {}
            return {
                "hookSpecificOutput": {
                    "hookEventName": option.event,
                    "additionalContext": reminders.reminder(texts),
                }
            }

        return hook
