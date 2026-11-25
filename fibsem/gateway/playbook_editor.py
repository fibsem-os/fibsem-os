"""The gateway page's playbook editor (Customise): read a playbook's file,
check a draft as the agent would load it, save it as the lab's copy.

Any operator edits a playbook, at this PC or paired; a hub or an agent never
does. Every save is the lab's file (``~/.fibsem/knowledge/playbooks/``), so a
shipped playbook edited here becomes the lab's copy, which wins over it, and
removing that copy brings the shipped one back. A save is checked first as
the agent loads it, so a playbook that wouldn't load can't be saved; the file
before is kept as ``<name>.yaml.bak``. A running session keeps the playbook
it started on: an edit counts from the next session.
"""

from __future__ import annotations

import difflib
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from fibsem.ai.agent import extra_mcp, instruments, playbooks, reminders, sources
from fibsem.ai.agent.catalog import CATALOG
from fibsem.gateway.settings import KeyRefused

__all__ = [
    "MAX_TEXT",
    "PlaybookRefused",
    "check",
    "options",
    "read",
    "remove",
    "save",
]

# Far above any playbook; short enough that a pasted file isn't one.
MAX_TEXT = 200_000


class PlaybookRefused(KeyRefused):
    """Why a playbook can't be read, checked, saved or removed."""


def _checkout() -> Optional[Path]:
    from fibsem.ai.agent.harness import find_checkout

    return find_checkout(None)


def _efforts() -> tuple:
    from fibsem.ai.agent.harness import EFFORTS

    return tuple(EFFORTS)


def _lab_file(name: str) -> Path:
    if not playbooks._NAME.fullmatch(name or ""):
        raise PlaybookRefused(
            422, "bad_playbook_name", f"{name!r} isn't a playbook name."
        )
    return sources.lab_folder("playbooks") / f"{name}.yaml"


def _skills(checkout: Optional[Path]) -> List[str]:
    names = set(sources.lab_skills())
    for folder in sources.claude_dirs(checkout, "skills"):
        if folder.is_dir():
            names.update(p.parent.name for p in folder.glob(f"*/{sources.SKILL_FILE}"))
    return sorted(names - set(sources.DEVELOPER_SKILLS))


def _agents(checkout: Optional[Path]) -> List[str]:
    names = set(sources.lab_agents())
    for folder in sources.claude_dirs(checkout, "agents"):
        if folder.is_dir():
            names.update(p.stem for p in folder.glob("*.md"))
    return sorted(names)


def options() -> dict:
    """What the editor's pickers offer: every playbook, instrument kind,
    skill, subagent, MCP server, fibsem tool, wake group, hook, effort,
    mode and level, and the session settings that are numbers."""
    from fibsem.ai.agent import wake_lines

    checkout = _checkout()
    try:
        servers = sorted(extra_mcp.load())
    except extra_mcp.ExtraMcpError:
        servers = []
    return {
        "fields": list(playbooks.FIELDS),
        "playbooks": playbooks.available(checkout),
        "instruments": instruments.kinds(checkout),
        "skills": _skills(checkout),
        "agents": _agents(checkout),
        "mcp": ["fibsem", *servers],
        "tools": sorted(spec.name for spec in CATALOG),
        "wakes": list(wake_lines.GROUPS),
        "hooks": [
            {
                "when": when,
                "label": option.label,
                "instrument": option.instrument or None,
                "by_playbook": option.by_playbook,
            }
            for when, option in reminders.HOOK_OPTIONS.items()
        ],
        "efforts": list(_efforts()),
        "modes": list(playbooks.MODES),
        "levels": list(playbooks.LEVELS),
        "session": list(playbooks.SESSION),
        "numbers": {key: kind.__name__ for key, kind in playbooks.NUMBERS.items()},
        "plan": list(playbooks.PLAN),
    }


def _inherited(name: str, data: Dict[str, Any], checkout: Optional[Path]):
    """What playbook *name* gets from below it: what it extends, as loaded,
    else the shipped base. None for the shipped base itself."""
    parent = data.get("extends")
    try:
        if isinstance(parent, str) and parent:
            return playbooks.load(parent, checkout, _efforts()).summary()
        if name == playbooks.DEFAULT:
            return None
        base = playbooks.shipped_base(checkout)
        return base.summary() if base is not None else None
    except playbooks.PlaybookError:
        return None


def read(name: str) -> dict:
    """``{name, source, path, text, data, lab, shipped, inherited, error}``:
    playbook *name*'s file as it is, where it is from, whether a lab copy
    and a shipped one exist, what it gets from below it, and why it
    wouldn't load, if it wouldn't."""
    checkout = _checkout()
    lab = _lab_file(name)
    path = playbooks.find(name, checkout)
    if path is None:
        raise PlaybookRefused(404, "no_playbook", f"No playbook {name}.")
    text = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    try:
        playbooks.load(name, checkout, _efforts())
        error = None
    except playbooks.PlaybookError as refused:
        error = str(refused)
    below = [
        folder / f"{name}.yaml"
        for folder in playbooks.dirs(checkout)[1:]
        if (folder / f"{name}.yaml").is_file()
    ]
    return {
        "name": name,
        "source": playbooks.source(path),
        "path": str(path),
        "text": text,
        "data": data,
        "lab": lab.is_file(),
        "shipped": bool(below),
        "inherited": _inherited(name, data, checkout),
        "error": error,
    }


def _text(body: Dict[str, Any]) -> str:
    """The draft's YAML: ``text`` as typed, else ``data`` (the form's)
    written in the playbook's field order."""
    text = body.get("text")
    if text is None:
        data = body.get("data")
        if not isinstance(data, dict):
            raise PlaybookRefused(
                422, "no_playbook", "Give the playbook's text or data."
            )
        ordered = {k: data[k] for k in playbooks.FIELDS if k in data}
        ordered.update({k: v for k, v in data.items() if k not in ordered})
        empty = (None, "", [], {})
        ordered = {k: v for k, v in ordered.items() if not _is(v, empty)}
        if isinstance(ordered.get("session"), dict):
            ordered["session"] = {
                k: v for k, v in ordered["session"].items() if not _is(v, empty)
            }
            if not ordered["session"]:
                del ordered["session"]
        text = yaml.dump(
            ordered, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=88
        )
    if not isinstance(text, str):
        raise PlaybookRefused(422, "no_playbook", "The playbook's text is text.")
    if len(text) > MAX_TEXT:
        raise PlaybookRefused(422, "too_long", "That playbook is too long to be one.")
    return text


class _Dumper(yaml.SafeDumper):
    """Short lists of names (no spaces) on one line, as the shipped playbooks write
    them (``skills: [autolamella]``); anything longer one item a line,
    indented under its key."""

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


def _list(dumper: yaml.SafeDumper, items: list):
    short = (
        all(
            isinstance(i, str) and i and not any(c in i for c in " :#,[]{}'\"")
            for i in items
        )
        and len(", ".join(items)) <= 60
    )
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq", items, flow_style=bool(items) and short
    )


_Dumper.add_representer(list, _list)


def _is(value: Any, empty: tuple) -> bool:
    return any(value == e and type(value) is type(e) for e in empty)


def check(name: str, body: Dict[str, Any]) -> dict:
    """``{text, data, inherited, playbook, error, diff}``: the draft's YAML
    and its fields (None when it isn't YAML), what it gets from below, the
    playbook it loads as (what the agent would run) or why it wouldn't
    load, and its change from the file it would replace. From the form's
    fields, the file's opening comments are kept."""
    lab = _lab_file(name)
    checkout = _checkout()
    before_path = lab if lab.is_file() else playbooks.find(name, checkout)
    before = before_path.read_text(encoding="utf-8") if before_path else ""
    text = _text(body)
    if body.get("text") is None:
        # The form's fields carry no comments: keep the file's opening ones.
        header = []
        for line in before.splitlines(keepends=True):
            if not line.startswith("#"):
                break
            header.append(line)
        text = "".join(header) + text
    try:
        loaded = playbooks.check(name, text, checkout, _efforts())
        found, error = loaded.summary(), None
    except playbooks.PlaybookError as refused:
        found, error = None, str(refused)
    diff = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            text.splitlines(keepends=True),
            fromfile=str(before_path) if before_path else "(new)",
            tofile=str(lab),
        )
    )
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        data = None
    data = data if isinstance(data, dict) else None
    return {
        "text": text,
        "data": data,
        "inherited": _inherited(name, data or {}, checkout),
        "playbook": found,
        "error": error,
        "diff": diff,
    }


def save(name: str, body: Dict[str, Any]) -> dict:
    """Save the draft as the lab's copy of *name*, once it loads; the file
    before is kept as ``<name>.yaml.bak``. Raises `PlaybookRefused`."""
    checked = check(name, body)
    if checked["error"]:
        raise PlaybookRefused(422, "bad_playbook", checked["error"])
    lab = _lab_file(name)
    lab.parent.mkdir(parents=True, exist_ok=True)
    if lab.is_file():
        shutil.copyfile(lab, lab.with_name(lab.name + ".bak"))
    staged = lab.with_name(lab.name + ".new")
    staged.write_text(checked["text"], encoding="utf-8")
    os.replace(staged, lab)
    return read(name)


def remove(name: str) -> Optional[dict]:
    """Remove the lab's copy of *name* (kept as ``.bak``): the shipped one,
    if any, is used again; the answer is it, or None."""
    lab = _lab_file(name)
    if not lab.is_file():
        raise PlaybookRefused(404, "no_lab_playbook", f"The lab has no copy of {name}.")
    shutil.copyfile(lab, lab.with_name(lab.name + ".bak"))
    lab.unlink()
    try:
        return read(name)
    except PlaybookRefused:
        return None
