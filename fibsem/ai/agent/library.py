"""The agent's playbooks, skills and subagents, as the operator sees them: what
fibsemOS ships and the lab's own, side by side.

The lab's own sit in the knowledge base (``sources.lab_folder``) and win by
name over the shipped ones. Saving an edit to a
shipped one makes the lab's copy there; removing the copy goes back to the
shipped one. The shipped files are never written.
"""

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from fibsem.ai.agent import playbooks, reminders, sources, subagents

KINDS = ("playbooks", "skills", "agents")
LAB = "lab"
SHIPPED = "shipped"
MAX_TEXT = 100_000
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}")


class LibraryError(ValueError):
    """Why a call is refused: ``error_type`` is ``invalid``, ``not_found`` or
    ``exists``."""

    def __init__(self, error_type: str, message: str):
        super().__init__(message)
        self.error_type = error_type
        self.message = message


@dataclass
class Item:
    """One playbook, skill or subagent in the list."""

    name: str
    description: Optional[str] = None
    # The lab has its own; it is the one used.
    lab: bool = False
    # fibsemOS ships one by that name.
    shipped: bool = False


@dataclass
class File:
    """One playbook, skill or subagent's file, from the lab or as shipped."""

    name: str
    source: str
    text: str
    path: str
    lab: bool = False
    shipped: bool = False
    # A playbook's or subagent's settings, for the window's form; None for a
    # skill, or a file that isn't a mapping.
    data: Optional[Dict[str, Any]] = None
    # A playbook's session settings with what it extends added in: each
    # setting's {value, source}, the playbook that sets it.
    session: Optional[Dict[str, Dict[str, Any]]] = None


@dataclass
class Options:
    """What the playbook and subagent forms offer to pick from."""

    levels: List[str]
    modes: List[str]
    efforts: List[str]
    # Each reminder's ``when`` and its label, in `reminders.HOOK_OPTIONS` order.
    reminders: List[Dict[str, str]]
    skills: List[str]
    playbooks: List[str]
    # The subagents a playbook can name, and the tools a subagent can list.
    agents: List[str]
    tools: List[str]


# A new one's file, before the operator writes it.
STARTING = {
    "playbooks": (
        f"description: What this playbook is for.\nextends: {playbooks.DEFAULT}\n"
    ),
    "skills": (
        "---\nname: {name}\ndescription: When the agent should read this skill.\n"
        "---\n\nWhat the agent should know.\n"
    ),
    "agents": (
        "---\nname: {name}\ndescription: When the agent should call this subagent.\n"
        "tools: mcp__fibsem__app_get_status\n---\n\nWhat the subagent does and answers.\n"
    ),
}


def _kind(kind: str) -> str:
    if kind not in KINDS:
        raise LibraryError("invalid", f"kind is one of {list(KINDS)}.")
    return kind


def _name(name: str) -> str:
    if not isinstance(name, str) or not _NAME.fullmatch(name) or name.endswith("."):
        raise LibraryError(
            "invalid", f"a name is letters, digits, '.', '_' and '-', not {name!r}."
        )
    return name


def _shipped_dir(kind: str) -> Optional[Path]:
    root = sources.find_root()
    if root is None:
        return None
    if kind == "playbooks":
        return root / playbooks.PLAYBOOK_DIR
    return root / sources.CLAUDE / kind


def _agent_skills() -> List[str]:
    """The skills fibsem-agent loads; the checkout's others are for developers."""
    # Here, not at the top: the server's answers import this module.
    from fibsem.ai.agent import harness

    return list(harness.SKILLS)


def _shipped_names(kind: str) -> List[str]:
    names = _names(_shipped_dir(kind), kind)
    if kind == "skills":
        names = [n for n in names if n in _agent_skills()]
    return names


def _shipped_path(kind: str, name: str) -> Optional[Path]:
    folder = _shipped_dir(kind)
    if folder is None or name not in _shipped_names(kind):
        return None
    return _path(folder, kind, name)


def _lab_dir(kind: str) -> Path:
    return sources.lab_folder(kind)


def _path(folder: Path, kind: str, name: str) -> Path:
    if kind == "skills":
        return folder / name / sources.SKILL_FILE
    if kind == "agents":
        return folder / f"{name}.md"
    return folder / f"{name}.yaml"


def _names(folder: Optional[Path], kind: str) -> List[str]:
    if folder is None or not folder.is_dir():
        return []
    if kind == "skills":
        return sorted(p.parent.name for p in folder.glob(f"*/{sources.SKILL_FILE}"))
    return sorted(p.stem for p in folder.glob("*.md" if kind == "agents" else "*.yaml"))


def _front_matter(text: str) -> Optional[dict]:
    """A skill's or subagent's front matter, or None without one."""
    return subagents.front_matter(text)[0]


def _description(path: Path, kind: str) -> Optional[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if kind in ("skills", "agents"):
        data = _front_matter(text)
    else:
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError:
            return None
    value = data.get("description") if isinstance(data, dict) else None
    return " ".join(str(value).split()) if value else None


def items(kind: str) -> List[Item]:
    """Every playbook, skill or subagent, by name, the lab's described when it has one."""
    kind = _kind(kind)
    shipped_dir = _shipped_dir(kind)
    lab = set(_names(_lab_dir(kind), kind))
    shipped = set(_shipped_names(kind))
    found = []
    for name in sorted(lab | shipped):
        folder = _lab_dir(kind) if name in lab else shipped_dir
        found.append(
            Item(
                name=name,
                description=_description(_path(folder, kind, name), kind),
                lab=name in lab,
                shipped=name in shipped,
            )
        )
    return found


def read(kind: str, name: str, source: Optional[str] = None) -> File:
    """One file: the lab's when it has one (or *source* says so), else the shipped one."""
    kind, name = _kind(kind), _name(name)
    lab_path = _path(_lab_dir(kind), kind, name)
    shipped_path = _shipped_path(kind, name)
    lab, shipped = lab_path.is_file(), shipped_path is not None
    if source not in (None, LAB, SHIPPED):
        raise LibraryError("invalid", f"source is {LAB} or {SHIPPED}.")
    if source is None:
        source = LAB if lab else SHIPPED
    path = lab_path if source == LAB else shipped_path
    if path is None or not path.is_file():
        raise LibraryError("not_found", f"No {source} {kind[:-1]} {name!r}.")
    text = path.read_text(encoding="utf-8")
    data = _data(kind, text)
    session = _session(name, data) if kind == "playbooks" else None
    return File(name, source, text, str(path), lab, shipped, data, session)


def _session(name: str, data: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """*name*'s session settings and the playbook each comes from, walking
    ``extends`` (the one used of each: the lab's, else the shipped)."""
    found: Dict[str, Dict[str, Any]] = {}
    seen = set()
    while isinstance(data, dict) and name not in seen:
        seen.add(name)
        own = data.get("session")
        if isinstance(own, dict):
            for key in playbooks.SESSION:
                if own.get(key) not in (None, "") and key not in found:
                    found[key] = {"value": own[key], "source": name}
        parent = data.get("extends")
        if not isinstance(parent, str):
            break
        try:
            name, data = parent, read("playbooks", parent).data
        except LibraryError:
            break
    return found


def _data(kind: str, text: str) -> Optional[Dict[str, Any]]:
    if kind == "skills":
        return None
    if kind == "agents":
        # Its settings, its lists as lists, and its instructions as `prompt`.
        front, body = subagents.front_matter(text)
        if front is None:
            return None
        data = dict(front)
        for key in ("tools", "skills"):
            try:
                data[key] = subagents.names(data.get(key))
            except subagents.SubagentError:
                return None
        data["prompt"] = body.strip()
        return data
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    return data if isinstance(data, dict) else None


class _Dumper(yaml.SafeDumper):
    """Text over one line as a block, so a prompt reads as it was typed."""


_Dumper.add_representer(
    str,
    lambda dumper, value: dumper.represent_scalar(
        "tag:yaml.org,2002:str", value, style="|" if "\n" in value else None
    ),
)


def _empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _clean(value: Any) -> Any:
    """*value* with empty settings left out, and text trimmed."""
    if isinstance(value, dict):
        cleaned = {k: _clean(v) for k, v in value.items()}
        return {k: v for k, v in cleaned.items() if not _empty(v)}
    if isinstance(value, list):
        cleaned = [_clean(v) for v in value]
        return [v for v in cleaned if not _empty(v)]
    if isinstance(value, str):
        return value.strip()
    return value


def dump(kind: str, data: Dict[str, Any]) -> str:
    """A form's settings as the file's text, in the order the settings are
    documented, without the ones left empty."""
    if kind == "skills" or not isinstance(data, dict):
        raise LibraryError("invalid", "data is a playbook's or subagent's settings.")
    if kind == "agents":
        data = _clean(data)
        prompt = data.pop("prompt", "")
        if not isinstance(prompt, str):
            raise LibraryError("invalid", "a subagent's prompt is text.")
        try:
            return subagents.render(data, prompt)
        except subagents.SubagentError as error:
            raise LibraryError("invalid", str(error)) from None
    data = _clean(data)
    if isinstance(data.get("session"), dict):
        session = data["session"]
        data["session"] = {k: session[k] for k in playbooks.SESSION if k in session}
    ordered = {k: data[k] for k in playbooks.FIELDS if k in data}
    ordered.update({k: v for k, v in data.items() if k not in ordered})
    return yaml.dump(
        ordered, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=88
    )


def options() -> Options:
    # Here, not at the top: the server's answers import this module.
    from fibsem.ai.agent import harness

    return Options(
        levels=list(playbooks.LEVELS),
        modes=list(playbooks.MODES),
        efforts=list(harness.EFFORTS),
        reminders=[
            {"when": when, "label": option.label}
            for when, option in reminders.HOOK_OPTIONS.items()
        ],
        skills=[item.name for item in items("skills")],
        playbooks=[item.name for item in items("playbooks")],
        agents=[item.name for item in items("agents")],
        tools=subagents.known_tools(),
    )


def create(kind: str, name: str) -> File:
    """A new one of the lab's own, from a short starting file."""
    kind, name = _kind(kind), _name(name)
    path = _path(_lab_dir(kind), kind, name)
    if path.exists() or _shipped_path(kind, name) is not None:
        raise LibraryError("exists", f"There is already a {kind[:-1]} {name!r}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    _write(path, STARTING[kind].format(name=name))
    return read(kind, name, LAB)


def _copy(kind: str, name: str) -> None:
    """Copy a shipped one into the lab's folder, its first edit made there."""
    source = _shipped_path(kind, name)
    if source is None:
        raise LibraryError("not_found", f"fibsemOS ships no {kind[:-1]} {name!r}.")
    target = _path(_lab_dir(kind), kind, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    if kind == "skills":
        shutil.copytree(source.parent, target.parent, dirs_exist_ok=True)
    else:
        shutil.copyfile(source, target)


def _check(kind: str, name: str, text: str) -> None:
    """Refuse *text* that would not load as a *kind*."""
    if kind == "agents":
        # Here, not at the top: the server's answers import this module.
        from fibsem.ai.agent import harness

        try:
            subagents.check(text, harness.EFFORTS)
        except subagents.SubagentError as error:
            raise LibraryError("invalid", str(error)) from None
        return
    if kind == "skills":
        data = _front_matter(text)
        if not data or not data.get("description"):
            raise LibraryError(
                "invalid",
                "A skill starts with front matter between '---' lines, with its name and description.",
            )
        return
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise LibraryError("invalid", str(error)) from None
    if not isinstance(data, dict):
        raise LibraryError("invalid", "a playbook is a mapping of settings")


def _write(path: Path, text: str) -> None:
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as f:
            f.write(text if text.endswith("\n") else text + "\n")
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise


def save(kind: str, name: str, text: str) -> File:
    """Save *text* as the lab's own; editing a shipped one makes the lab's copy."""
    kind, name = _kind(kind), _name(name)
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT:
        raise LibraryError("invalid", f"text is 1 to {MAX_TEXT} characters.")
    path = _path(_lab_dir(kind), kind, name)
    _check(kind, name, text)
    # Editing a shipped one makes the lab's copy; the shipped file is never written.
    copied = not path.is_file()
    if copied:
        _copy(kind, name)
    previous = path.read_text(encoding="utf-8")
    _write(path, text)
    if kind == "playbooks":
        # It must load, and what it extends must still resolve.
        from fibsem.ai.agent import harness

        try:
            playbooks.load(name, sources.find_root(), harness.EFFORTS)
        except playbooks.PlaybookError as error:
            if copied:
                remove(kind, name)
            else:
                _write(path, previous)
            raise LibraryError("invalid", str(error)) from None
    return read(kind, name, LAB)


def remove(kind: str, name: str) -> Item:
    """Remove the lab's own copy: the shipped one, if any, is used again."""
    kind, name = _kind(kind), _name(name)
    path = _path(_lab_dir(kind), kind, name)
    if not path.is_file():
        raise LibraryError("not_found", f"The lab has no {name!r} of its own.")
    if kind == "skills":
        shutil.rmtree(path.parent)
    else:
        path.unlink()
    return Item(name=name, lab=False, shipped=_shipped_path(kind, name) is not None)
