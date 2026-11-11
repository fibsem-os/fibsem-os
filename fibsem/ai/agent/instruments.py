"""What the agent is told for a kind of instrument: the harness prompt, the
intake steps and the re-read lines, as files beside the playbooks.

    .claude/fibsem-agent/instruments/<kind>/
        prompt.md     the harness prompt
        intake.md     ## start, ## resumed, ## goal
        reread.md     ## proposals, ## plan, ## level
        devices.yaml  what it has, so the agent is offered the tools on them:
                      ``microscope: true`` for a FIB-SEM (every tool), else
                      ``devices:`` and ``services:``, each a list of names;
                      none gets only the tools for any instrument

A playbook names its kind (``instrument:``); the shipped base playbook sets
``fibsem``. The files are fibsemOS's own, or an installed plugin's for its own
kind (`sources.plugin_roots`), never the lab's: they carry the safety wording.
A plugin can't replace a shipped kind. A leading ``<!-- -->`` comment is a
note for whoever edits the file, not part of the text. ``{name}`` fills in a
value; a file that is missing, lacks a section or names a value nobody gives
fails when the session starts (`load`).
"""

import re
import string
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from fibsem.ai.agent import sources
from fibsem.ai.agent.catalog import Devices

INSTRUMENT_DIR = Path(sources.CLAUDE) / "fibsem-agent" / "instruments"
# The kind a playbook that names none is for: the only one there is.
DEFAULT = "fibsem"
# Each file's sections ("" for a file that is one text), and the values each
# may fill in.
FILES: Dict[str, Dict[str, set]] = {
    "prompt.md": {"": {"tools", "deadline_warn_min"}},
    "intake.md": {"start": set(), "resumed": set(), "goal": {"goal"}},
    "reread.md": {"proposals": set(), "plan": set(), "level": set()},
}
_NOTE = re.compile(r"\A\s*<!--.*?-->\s*", re.S)
_SECTION = re.compile(r"^## +(\S.*?)\s*$", re.M)


class InstrumentError(ValueError):
    """An instrument kind's files that can't be used, with the reason."""


@dataclass(frozen=True)
class Texts:
    """One instrument kind's words, each a template for `str.format`."""

    kind: str
    prompt: str
    start: str
    resumed: str
    goal: str
    reread_proposals: str
    reread_plan: str
    reread_level: str


def _folders(checkout: Optional[Path]) -> Dict[str, Path]:
    """Each instrument kind's folder: the checkout's, then each plugin's for a
    kind the checkout doesn't ship."""
    found: Dict[str, Path] = {}
    relative = str(INSTRUMENT_DIR.relative_to(sources.CLAUDE))
    for folder in sources.claude_dirs(checkout, relative):
        if folder.is_dir():
            for kind in folder.iterdir():
                if (kind / "prompt.md").is_file():
                    found.setdefault(kind.name, kind)
    return found


def kinds(checkout: Optional[Path]) -> List[str]:
    """The instrument kinds the checkout ships and the plugins add, sorted."""
    return sorted(_folders(checkout))


def load(kind: Optional[str], checkout: Path) -> Texts:
    """Instrument *kind*'s words; an `InstrumentError` says what is wrong."""
    kind = kind or DEFAULT
    folders = _folders(checkout)
    if kind not in folders:
        raise InstrumentError(
            f"No instrument kind {kind!r} in {checkout / INSTRUMENT_DIR} "
            f"or a plugin (one of {', '.join(sorted(folders)) or 'none'})"
        )
    folder = folders[kind]
    found: Dict[str, Dict[str, str]] = {}
    for name, sections in FILES.items():
        path = folder / name
        if not path.is_file():
            raise InstrumentError(f"{path} is missing")
        texts = _sections(path.read_text(encoding="utf-8"), set(sections) != {""})
        for section, allowed in sections.items():
            if section not in texts:
                raise InstrumentError(f"{path} has no '## {section}' section")
            try:
                unknown = _fields(texts[section]) - allowed
            except ValueError as error:
                raise InstrumentError(f"{path}: a stray brace: {error}") from None
            if unknown:
                raise InstrumentError(
                    f"{path}: {section or 'the text'} fills in "
                    f"{', '.join(sorted(unknown))}, which nothing gives"
                )
        found[name] = texts
    intake, reread = found["intake.md"], found["reread.md"]
    return Texts(
        kind=kind,
        prompt=found["prompt.md"][""],
        start=intake["start"],
        resumed=intake["resumed"],
        goal=intake["goal"],
        reread_proposals=reread["proposals"],
        reread_plan=reread["plan"],
        reread_level=reread["level"],
    )


DEVICES_FILE = "devices.yaml"
_DEVICES_KEYS = {"microscope", "devices", "services"}


def devices(kind: Optional[str], checkout: Optional[Path]) -> Devices:
    """What instrument *kind* has (its ``devices.yaml``); an
    `InstrumentError` says what is wrong."""
    kind = kind or DEFAULT
    folders = _folders(checkout)
    if kind not in folders:
        raise InstrumentError(
            f"No instrument kind {kind!r} (one of {', '.join(sorted(folders)) or 'none'})"
        )
    path = folders[kind] / DEVICES_FILE
    if not path.is_file():
        return Devices()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict) or set(data) - _DEVICES_KEYS:
        raise InstrumentError(
            f"{path}: give {', '.join(sorted(_DEVICES_KEYS))}, not {data!r}"
        )
    names = {}
    for key in ("devices", "services"):
        value = data.get(key) or []
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise InstrumentError(f"{path}: {key} is a list of names")
        names[key] = frozenset(value)
    return Devices(microscope=data.get("microscope") is True, **names)


@cache
def shipped(kind: str, checkout: Path) -> Texts:
    """`load`, once per kind and checkout: the files don't change while the
    agent runs."""
    return load(kind, checkout)


def _sections(text: str, split: bool) -> Dict[str, str]:
    text = _NOTE.sub("", text, count=1)
    if not split:
        return {"": text.strip()}
    parts = _SECTION.split(text)
    # [before the first heading, heading, body, heading, body, ...]
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def _fields(text: str) -> set:
    """The values *text* fills in; a `ValueError` on a stray brace."""
    return {
        field.split(".")[0].split("[")[0]
        for _, field, _, _ in string.Formatter().parse(text)
        if field is not None
    }
