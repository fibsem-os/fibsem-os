"""The agent's profiles, playbooks and skills, as the operator sees them: what
fibsemOS ships and the lab's own, side by side.

The lab's own sit in the knowledge base (``sources.lab_folder``) and win by
name over the shipped ones. The operator copies a shipped one there to edit it,
edits the lab's copy, or removes it to go back to the shipped one. The shipped
files are never written.
"""

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml

from fibsem.ai.agent import playbooks, profiles, sources

KINDS = ("profiles", "playbooks", "skills")
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
    """One profile, playbook or skill in the list."""

    name: str
    description: Optional[str] = None
    # The lab has its own; it is the one used.
    lab: bool = False
    # fibsemOS ships one by that name.
    shipped: bool = False


@dataclass
class File:
    """One profile, playbook or skill's file, from the lab or as shipped."""

    name: str
    source: str
    text: str
    path: str
    lab: bool = False
    shipped: bool = False


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
    if kind == "profiles":
        return root / profiles.PROFILE_DIR
    if kind == "playbooks":
        return root / playbooks.PLAYBOOK_DIR
    return root / sources.CLAUDE / "skills"


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
    return folder / f"{name}.yaml"


def _names(folder: Optional[Path], kind: str) -> List[str]:
    if folder is None or not folder.is_dir():
        return []
    if kind == "skills":
        return sorted(p.parent.name for p in folder.glob(f"*/{sources.SKILL_FILE}"))
    return sorted(p.stem for p in folder.glob("*.yaml"))


def _front_matter(text: str) -> Optional[dict]:
    """A skill's front matter, or None without one. Claude Code reads it a
    line at a time, so a value with a colon in it, not quoted, still counts."""
    front, sep, _ = (
        text[4:].partition("\n---") if text.startswith("---\n") else ("", "", "")
    )
    if not sep:
        return None
    try:
        data = yaml.safe_load(front)
    except yaml.YAMLError:
        data = None
    if isinstance(data, dict):
        return data
    found = re.findall(r"^([A-Za-z][\w-]*):[ \t]*(.*)$", front, re.MULTILINE)
    return dict(found) if found else None


def _description(path: Path, kind: str) -> Optional[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if kind == "skills":
        data = _front_matter(text)
    else:
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError:
            return None
    value = data.get("description") if isinstance(data, dict) else None
    return " ".join(str(value).split()) if value else None


def items(kind: str) -> List[Item]:
    """Every profile, playbook or skill, by name, the lab's described when it has one."""
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
    return File(name, source, path.read_text(encoding="utf-8"), str(path), lab, shipped)


def copy(kind: str, name: str) -> File:
    """Copy a shipped one into the lab's folder, to edit there."""
    kind, name = _kind(kind), _name(name)
    source = _shipped_path(kind, name)
    if source is None:
        raise LibraryError("not_found", f"fibsemOS ships no {kind[:-1]} {name!r}.")
    target = _path(_lab_dir(kind), kind, name)
    if target.exists():
        raise LibraryError("exists", f"The lab already has its own {name!r}.")
    target.parent.mkdir(parents=True, exist_ok=True)
    if kind == "skills":
        shutil.copytree(source.parent, target.parent, dirs_exist_ok=True)
    else:
        shutil.copyfile(source, target)
    return read(kind, name, LAB)


def _check(kind: str, name: str, text: str) -> None:
    """Refuse *text* that would not load as a *kind*."""
    if kind == "skills":
        data = _front_matter(text)
        if not data or not data.get("description"):
            raise LibraryError(
                "invalid",
                "A skill starts with front matter between '---' lines, with its name and description.",
            )
        return
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / f"{name}.yaml"
        path.write_text(text, encoding="utf-8")
        try:
            if kind == "profiles":
                profiles.load(path, checkout=sources.find_root())
            else:
                data = yaml.safe_load(text)
                if not isinstance(data, dict):
                    raise playbooks.PlaybookError("a playbook is a mapping of settings")
        except (
            profiles.ProfileError,
            playbooks.PlaybookError,
            yaml.YAMLError,
        ) as error:
            raise LibraryError("invalid", str(error).replace(str(path), name)) from None


def save(kind: str, name: str, text: str) -> File:
    """Replace the lab's own copy's text; only the lab's are edited."""
    kind, name = _kind(kind), _name(name)
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT:
        raise LibraryError("invalid", f"text is 1 to {MAX_TEXT} characters.")
    path = _path(_lab_dir(kind), kind, name)
    if not path.is_file():
        raise LibraryError(
            "not_found",
            f"The lab has no {name!r} of its own: copy the shipped one first.",
        )
    _check(kind, name, text)
    previous = path.read_text(encoding="utf-8")
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as f:
            f.write(text if text.endswith("\n") else text + "\n")
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise
    if kind == "playbooks":
        # what it extends must still resolve
        try:
            playbooks.load(name, sources.find_root())
        except playbooks.PlaybookError as error:
            path.write_text(previous, encoding="utf-8")
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
