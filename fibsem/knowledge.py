"""The knowledge base: what was learned on this microscope, kept as Markdown files.

Under ``<state root>/knowledge``:

- ``entries/``: one folder per entry, ``instrument/<id>``,
  ``sample/<sample>/<id>``, ``playbook/<playbook>/<id>`` and
  ``experiment/<experiment>`` (a past run's record), each holding ``entry.md``
  and the images it cites.
- ``profiles/``, ``playbooks/`` and ``skills/``: the lab's own, read beside
  the ones fibsem ships.

An entry is self-contained: its images are copied in (at most ``IMAGE_PX`` on
their long side), so it still reads after the experiment is moved or gone. Where
it came from is kept as metadata, never a path: the experiment's name (its
session tag) and id, the instrument, the agent session, the item and each
image's file name in the experiment. Finding that experiment again is a lookup
that may fail; nothing here depends on it.

``entry.md`` is YAML front matter and then the text. The front matter says
who wrote it and when; ``status`` (the agent's entries are ``to review`` until
the operator keeps them, the operator's are ``kept``); ``lasts`` (``until
fixed``, ``ongoing`` or ``record``); ``tags`` (instrument, sample, playbook,
session); its ``source``; an optional ``remind`` line; and, for an ``until fixed`` entry, when
it was last ``checked`` and when it was ``resolved``. The App Server's routes,
and so the agent's tools, read and write through these functions.
"""

import os
import re
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml

from fibsem import paths
from fibsem.util.timestamps import now, now_iso, to_datetime

INSTRUMENT = "instrument"
SAMPLE = "sample"
PLAYBOOK = "playbook"
EXPERIMENT = "experiment"
SECTIONS = (INSTRUMENT, SAMPLE, PLAYBOOK, EXPERIMENT)
AGENT = "agent"
OPERATOR = "operator"
WRITERS = (AGENT, OPERATOR)
TO_REVIEW = "to review"
KEPT = "kept"
STATUSES = (TO_REVIEW, KEPT)
UNTIL_FIXED = "until fixed"
ONGOING = "ongoing"
RECORD = "record"
LASTS = (UNTIL_FIXED, ONGOING, RECORD)
TAGS = ("instrument", "sample", "playbook", "session")
# Where an entry came from, besides its tags: ids and names, never a path.
SOURCE = ("experiment_id", "instrument_serial", "agent_session", "item", "evidence_id")
# The lab's own profiles, playbooks and skills, beside the shipped ones.
LAB_FOLDERS = ("profiles", "playbooks", "skills")
# An "until fixed" entry not checked for this long is asked about again.
ASK_AGAIN = timedelta(days=14)

IMAGE_PX = 1024
MAX_TITLE = 200
MAX_TEXT = 20000
MAX_REMIND = 200
MAX_IMAGES = 8
# An id, an image name or a tag: one folder or file name, never a path.
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._-]{0,119}")
_FILE = "entry.md"
_lock = threading.Lock()


class KnowledgeError(ValueError):
    """Why a call is refused: ``error_type`` is ``invalid`` or ``not_found``."""

    def __init__(self, error_type: str, message: str):
        super().__init__(message)
        self.error_type = error_type
        self.message = message


@dataclass
class Row:
    """One entry in the index."""

    section: str
    id: str
    title: str
    written: str = ""
    by: str = ""
    status: str = KEPT
    lasts: str = ONGOING
    tags: Dict[str, str] = field(default_factory=dict)
    remind: Optional[str] = None
    images: int = 0
    edited: bool = False
    resolved: bool = False
    ask_again: bool = False


@dataclass
class Entry:
    """One entry: its front matter, its text, its image names and its file."""

    section: str
    id: str
    title: str
    text: str = ""
    written: str = ""
    by: str = ""
    status: str = KEPT
    lasts: str = ONGOING
    tags: Dict[str, str] = field(default_factory=dict)
    remind: Optional[str] = None
    images: List[str] = field(default_factory=list)
    # SOURCE's keys, and ``images``: each copied image's file name in the experiment
    source: Dict[str, Any] = field(default_factory=dict)
    edited: bool = False
    edited_by: Optional[str] = None
    checked: Optional[str] = None
    resolved: Optional[str] = None
    path: str = ""

    def ask_again(self) -> bool:
        """An unresolved "until fixed" entry not checked for `ASK_AGAIN`."""
        if self.lasts != UNTIL_FIXED or self.resolved:
            return False
        at = to_datetime(self.checked or self.written)
        return at is not None and now() - at > ASK_AGAIN

    def row(self) -> Row:
        return Row(
            section=self.section,
            id=self.id,
            title=self.title,
            written=self.written,
            by=self.by,
            status=self.status,
            lasts=self.lasts,
            tags=dict(self.tags),
            remind=self.remind,
            images=len(self.images),
            edited=self.edited,
            resolved=bool(self.resolved),
            ask_again=self.ask_again(),
        )


def root() -> Path:
    return paths.state_root() / "knowledge"


def lab_folder(kind: str) -> Path:
    """The folder of the lab's own profiles, playbooks or skills, made if missing."""
    if kind not in LAB_FOLDERS:
        raise KnowledgeError("invalid", f"kind is one of {list(LAB_FOLDERS)}.")
    folder = root() / kind
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def make_folders() -> None:
    """The lab's folders, so an operator finds where their own files go."""
    for kind in LAB_FOLDERS:
        lab_folder(kind)


def slug(text: str, limit: int = 48) -> str:
    """*text* as lower-case letters, digits and hyphens: "HeLa GFP" -> "hela-gfp"."""
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower())[:limit].strip("-")


def _name(value: Optional[str], what: str) -> str:
    if (
        not isinstance(value, str)
        or not _NAME.fullmatch(value)
        or value.endswith((".", " "))
    ):
        raise KnowledgeError(
            "invalid",
            f"{what} is a name of letters, digits, spaces, '.', '_' and '-', "
            f"not {value!r}.",
        )
    return value


def _section(section: str) -> str:
    if section not in SECTIONS:
        raise KnowledgeError("invalid", f"section is one of {list(SECTIONS)}.")
    return section


def _group(section: str, sample: Optional[str], playbook: Optional[str]):
    """The folder a sample or playbook entry sits in: its sample's or its
    playbook's slug. None for an instrument or experiment entry."""
    if section == SAMPLE:
        named = slug(sample or "", 40)
        if not named:
            raise KnowledgeError(
                "invalid", "A sample entry names its sample: a short slug, e.g. yeast."
            )
        return named
    if section == PLAYBOOK:
        named = slug(playbook or "", 40)
        if not named:
            raise KnowledgeError(
                "invalid",
                "A playbook entry names its playbook, e.g. autolamella-correlative.",
            )
        return named
    return None


def _folder(section: str, id: str, group: Optional[str]) -> Path:
    if group:
        return root() / "entries" / section / group / id
    return root() / "entries" / section / id


def _text(meta: dict, key: str) -> Optional[str]:
    value = meta.get(key)
    # an unquoted time or date is read by YAML as one
    value = value.isoformat() if hasattr(value, "isoformat") else value
    return str(value) if value not in (None, "") else None


def _parse(path: Path, section: str, id: str, group: Optional[str]) -> Entry:
    raw = path.read_text(encoding="utf-8")
    meta, text = {}, raw
    if raw.startswith("---\n"):
        front, sep, rest = raw[4:].partition("\n---\n")
        if sep:
            try:
                loaded = yaml.safe_load(front)
            except yaml.YAMLError:
                loaded = None
            if isinstance(loaded, dict):
                meta, text = loaded, rest
    images = meta.get("images")
    by = _text(meta, "by") or ""
    tags = meta.get("tags") if isinstance(meta.get("tags"), dict) else {}
    tags = {key: str(tags[key]) for key in TAGS if tags.get(key)}
    # the folder says which sample, playbook or run it is, whatever the file says
    if section == SAMPLE:
        tags["sample"] = group
    elif section == PLAYBOOK:
        tags["playbook"] = group
    elif section == EXPERIMENT:
        tags["session"] = id
    status = _text(meta, "status")
    lasts = _text(meta, "lasts")
    found = meta.get("source") if isinstance(meta.get("source"), dict) else {}
    source: Dict[str, Any] = {key: str(found[key]) for key in SOURCE if found.get(key)}
    if isinstance(found.get("images"), dict):
        source["images"] = {str(k): str(v) for k, v in found["images"].items()}
    return Entry(
        section=section,
        id=id,
        title=_text(meta, "title") or id,
        text=text.strip("\n"),
        written=_text(meta, "written") or "",
        by=by,
        # a hand-written entry is the operator's
        status=status if status in STATUSES else (TO_REVIEW if by == AGENT else KEPT),
        lasts=RECORD
        if section == EXPERIMENT
        else (lasts if lasts in LASTS else ONGOING),
        tags=tags,
        remind=_text(meta, "remind"),
        images=[str(name) for name in images] if isinstance(images, list) else [],
        source=source,
        edited=bool(meta.get("edited")),
        edited_by=_text(meta, "edited_by"),
        checked=_text(meta, "checked"),
        resolved=_text(meta, "resolved"),
        path=str(path),
    )


def _found(folder: Path, section: str, group: Optional[str]) -> List[Entry]:
    if not folder.is_dir():
        return []
    return [
        _parse(child / _FILE, section, child.name, group)
        for child in sorted(folder.iterdir())
        if (child / _FILE).is_file()
    ]


def _newest_first(entry: Entry):
    at = to_datetime(entry.written)
    return at.timestamp() if at is not None else 0.0


def _groups(section: str) -> List[str]:
    folder = root() / "entries" / section
    if not folder.is_dir():
        return []
    return sorted(child.name for child in folder.iterdir() if child.is_dir())


def all_entries(
    section: Optional[str] = None,
    sample: Optional[str] = None,
    playbook: Optional[str] = None,
    status: Optional[str] = None,
    lasts: Optional[str] = None,
    resolved: bool = False,
) -> List[Entry]:
    """Every entry, newest first, or only a section's; only those tagged with
    *sample* or *playbook*; only those with *status* or that last *lasts*.
    A resolved entry is left out unless *resolved*."""
    sections = SECTIONS if section is None else (_section(section),)
    if status is not None and status not in STATUSES:
        raise KnowledgeError("invalid", f"status is one of {list(STATUSES)}.")
    if lasts is not None and lasts not in LASTS:
        raise KnowledgeError("invalid", f"lasts is one of {list(LASTS)}.")
    found: List[Entry] = []
    for name in sections:
        if name in (SAMPLE, PLAYBOOK):
            for group in _groups(name):
                found += _found(root() / "entries" / name / group, name, group)
        else:
            found += _found(root() / "entries" / name, name, None)
    wanted = {"sample": slug(sample or "", 40), "playbook": slug(playbook or "", 40)}
    return sorted(
        (
            entry
            for entry in found
            if all(
                not value or slug(entry.tags.get(key, ""), 40) == value
                for key, value in wanted.items()
            )
            and (status is None or entry.status == status)
            and (lasts is None or entry.lasts == lasts)
            and (resolved or not entry.resolved)
        ),
        key=_newest_first,
        reverse=True,
    )


def entries(
    section: Optional[str] = None,
    sample: Optional[str] = None,
    playbook: Optional[str] = None,
    status: Optional[str] = None,
    lasts: Optional[str] = None,
    resolved: bool = False,
) -> List[Row]:
    """The index: one row per entry, newest first (`all_entries`' filters)."""
    return [
        entry.row()
        for entry in all_entries(section, sample, playbook, status, lasts, resolved)
    ]


def samples() -> List[str]:
    """The samples with sample entries, by slug."""
    return _groups(SAMPLE)


def playbooks() -> List[str]:
    """The playbooks with playbook entries, by name."""
    return _groups(PLAYBOOK)


def read(
    section: str,
    id: str,
    sample: Optional[str] = None,
    playbook: Optional[str] = None,
) -> Entry:
    section = _section(section)
    group = _group(section, sample, playbook)
    path = _folder(section, _name(id, "id"), group) / _FILE
    if not path.is_file():
        where = f" for {section} {group}" if group else ""
        raise KnowledgeError(
            "not_found", f"No {section} entry {id!r}{where}: knowledge_list lists them."
        )
    return _parse(path, section, id, group)


def image(
    section: str,
    id: str,
    name: str,
    sample: Optional[str] = None,
    playbook: Optional[str] = None,
) -> Path:
    """The file of one image an entry cites."""
    entry = read(section, id, sample, playbook)
    if _name(name, "the image") not in entry.images:
        raise KnowledgeError("not_found", f"{id} has no image {name!r}.")
    path = Path(entry.path).parent / name
    if not path.is_file():
        raise KnowledgeError("not_found", f"{id}'s image {name!r} is missing.")
    return path


def _picture(source: Path):
    """*source* as a PIL image at most `IMAGE_PX` on its long side."""
    import numpy as np
    from PIL import Image

    try:
        if source.suffix.lower() in (".tif", ".tiff"):
            import tifffile

            data = np.squeeze(np.asarray(tifffile.imread(source)))
            # stacks project on their leading axes; a trailing RGB(A) axis stays
            while data.ndim > 2 and data.shape[-1] not in (3, 4):
                data = data.max(axis=0)
            if data.dtype != np.uint8:
                low, high = float(data.min()), float(data.max())
                scale = 255 / (high - low) if high > low else 0.0
                data = ((data.astype(np.float32) - low) * scale).astype(np.uint8)
            picture = Image.fromarray(data)
        else:
            with Image.open(source) as opened:
                picture = opened.copy()
    except Exception as exc:  # noqa: BLE001 - any unreadable file is refused alike
        raise KnowledgeError(
            "invalid", f"{source.name} can't be read as an image: {exc}"
        ) from exc
    if picture.mode not in ("L", "RGB", "RGBA"):
        picture = picture.convert("RGB")
    picture.thumbnail((IMAGE_PX, IMAGE_PX), Image.Resampling.LANCZOS)
    return picture


def _image_name(source: Path, taken: Sequence[str]) -> str:
    stem = slug(source.name.split(".")[0]) or "image"
    name, n = f"{stem}.png", 2
    while name in taken:
        name, n = f"{stem}-{n}.png", n + 1
    return name


def _save(path: Path, meta: dict, text: str) -> None:
    """entry.md written whole, or not at all."""
    front = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True)
    content = f"---\n{front}---\n{text.strip()}\n"
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise


def _meta(entry: Entry) -> dict:
    """*entry*'s front matter, in the order a reader wants it."""
    meta: dict = {
        "title": entry.title,
        "written": entry.written,
        "by": entry.by,
        "status": entry.status,
        "lasts": entry.lasts,
    }
    if entry.tags:
        meta["tags"] = {key: entry.tags[key] for key in TAGS if key in entry.tags}
    if entry.remind:
        meta["remind"] = entry.remind
    if entry.checked:
        meta["checked"] = entry.checked
    if entry.resolved:
        meta["resolved"] = entry.resolved
    meta["images"] = list(entry.images)
    if entry.source:
        meta["source"] = entry.source
    if entry.edited:
        meta["edited"] = True
        meta["edited_by"] = entry.edited_by
    return meta


def _tag(value: Optional[str], key: str) -> Optional[str]:
    if value in (None, ""):
        return None
    if key in ("sample", "playbook"):
        return slug(value, 40) or None
    return _name(value.strip(), key)


def write(
    section: str,
    title: str,
    text: str,
    by: str,
    sample: Optional[str] = None,
    playbook: Optional[str] = None,
    id: Optional[str] = None,
    lasts: Optional[str] = None,
    remind: Optional[str] = None,
    instrument: Optional[str] = None,
    session: Optional[str] = None,
    source: Optional[Dict[str, Optional[str]]] = None,
    images: Sequence[Union[str, Path]] = (),
    image_names: Optional[Sequence[str]] = None,
) -> Entry:
    """Write a new entry, or rewrite entry *id*: its title and text replaced,
    *images* (files to copy in) added to those it has.

    The agent's writes are ``to review`` and the operator's ``kept``, a
    rewrite's too. *lasts* defaults to ``ongoing``; an experiment entry is a
    ``record``, its id the experiment's name, and a rewrite of one that isn't
    there writes it. *sample*, *playbook*, *instrument* and *session* (the
    experiment's name) are its tags; a rewrite keeps the ones it has, and its
    first author and time, and is marked edited by *by*. *source* (`SOURCE`'s
    keys) says where it came from, kept from the first write; *image_names* are
    *images*' file names in the experiment, their own names when left out.
    A rewrite of an
    ``until fixed`` entry says it was checked now. *remind* is one line said
    before the steps it concerns; an empty one removes it."""
    section = _section(section)
    group = _group(section, sample, playbook)
    if by not in WRITERS:
        raise KnowledgeError("invalid", f"by is one of {list(WRITERS)}.")
    title = title.strip() if isinstance(title, str) else ""
    if not title or len(title) > MAX_TITLE:
        raise KnowledgeError("invalid", f"title is 1 to {MAX_TITLE} characters.")
    if not isinstance(text, str) or len(text) > MAX_TEXT:
        raise KnowledgeError("invalid", f"text is at most {MAX_TEXT} characters.")
    if lasts is not None and lasts not in LASTS:
        raise KnowledgeError("invalid", f"lasts is one of {list(LASTS)}.")
    if section == EXPERIMENT:
        if lasts not in (None, RECORD):
            raise KnowledgeError("invalid", "An experiment entry is a record.")
        if not id:
            raise KnowledgeError(
                "invalid", "An experiment entry's id is the experiment's name."
            )
        lasts, session = RECORD, id
    elif lasts == RECORD:
        raise KnowledgeError(
            "invalid", "Only an experiment entry is a record: a past run's."
        )
    if remind is not None:
        remind = " ".join(str(remind).split())
        if len(remind) > MAX_REMIND:
            raise KnowledgeError(
                "invalid", f"remind is one line of at most {MAX_REMIND} characters."
            )
    given = {
        "instrument": _tag(instrument, "instrument"),
        "sample": _tag(sample, "sample"),
        "playbook": _tag(playbook, "playbook"),
        "session": _tag(session, "session"),
    }
    sources = [Path(each) for each in images or ()]
    origins = list(image_names) if image_names is not None else []
    if origins and len(origins) != len(sources):
        raise KnowledgeError("invalid", "image_names names each image, in order.")
    origins = origins or [each.name for each in sources]
    given_source = {
        key: str(value)
        for key, value in (source or {}).items()
        if key in SOURCE and value not in (None, "")
    }
    pictures = [_picture(each) for each in sources]
    with _lock:
        existing = None
        if id is not None:
            folder = _folder(section, _name(id, "id"), group)
            path = folder / _FILE
            if path.is_file():
                existing = _parse(path, section, id, group)
            elif section != EXPERIMENT:
                raise KnowledgeError(
                    "not_found",
                    f"No {section} entry {id!r}: leave id out to write a new one.",
                )
        else:
            base = f"{now().date().isoformat()}-{slug(title) or 'entry'}"
            id, n = base, 2
            while _folder(section, id, group).exists():
                id, n = f"{base}-{n}", n + 1
            folder = _folder(section, id, group)
            path = folder / _FILE
        names = list(existing.images) if existing else []
        if len(names) + len(pictures) > MAX_IMAGES:
            raise KnowledgeError(
                "invalid", f"An entry cites at most {MAX_IMAGES} images."
            )
        tags = dict(existing.tags) if existing else {}
        for key, value in given.items():
            if value and key not in tags:
                tags[key] = value
        at = now_iso()
        entry = Entry(
            section=section,
            id=id,
            title=title,
            text=text,
            written=existing.written if existing else at,
            by=existing.by if existing else by,
            status=TO_REVIEW if by == AGENT else KEPT,
            lasts=lasts or (existing.lasts if existing else ONGOING),
            tags=tags,
            remind=(remind or None)
            if remind is not None
            else (existing.remind if existing else None),
            edited=existing is not None,
            edited_by=by if existing else None,
            resolved=existing.resolved if existing else None,
        )
        if entry.lasts == UNTIL_FIXED:
            entry.checked = at
        entry.source = dict(existing.source) if existing else given_source
        copied = dict(entry.source.get("images") or {})
        folder.mkdir(parents=True, exist_ok=True)
        for each, origin, picture in zip(sources, origins, pictures):
            name = _image_name(each, names)
            picture.save(folder / name, format="PNG")
            names.append(name)
            copied[name] = origin
        entry.images = names
        if copied:
            entry.source["images"] = copied
        _save(path, _meta(entry), text)
        return _parse(path, section, id, group)


def _change(
    section: str,
    id: str,
    sample: Optional[str],
    playbook: Optional[str],
    change,
) -> Entry:
    with _lock:
        entry = read(section, id, sample, playbook)
        change(entry)
        _save(Path(entry.path), _meta(entry), entry.text)
        return read(section, id, sample, playbook)


def keep(
    section: str,
    id: str,
    sample: Optional[str] = None,
    playbook: Optional[str] = None,
) -> Entry:
    """The operator keeps an entry the agent wrote: ``to review`` -> ``kept``."""

    def kept(entry: Entry) -> None:
        entry.status = KEPT

    return _change(section, id, sample, playbook, kept)


def resolve(
    section: str,
    id: str,
    sample: Optional[str] = None,
    playbook: Optional[str] = None,
) -> Entry:
    """The operator says an ``until fixed`` entry is fixed: it stops applying,
    and is kept on file."""

    def resolved(entry: Entry) -> None:
        if entry.lasts != UNTIL_FIXED:
            raise KnowledgeError(
                "invalid", f"Only an 'until fixed' entry is resolved, not {id!r}."
            )
        entry.resolved = now_iso()
        entry.status = KEPT

    return _change(section, id, sample, playbook, resolved)


def delete(
    section: str,
    id: str,
    sample: Optional[str] = None,
    playbook: Optional[str] = None,
) -> Entry:
    """Remove an entry and its images: the entry as it was."""
    entry = read(section, id, sample, playbook)
    with _lock:
        folder = Path(entry.path).parent
        shutil.rmtree(folder)
        if section in (SAMPLE, PLAYBOOK) and not any(folder.parent.iterdir()):
            folder.parent.rmdir()
    return entry
