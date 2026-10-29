"""The knowledge base: what was learned on this microscope, kept as Markdown files.

Under ``<state root>/knowledge``: ``instrument/<id>/entry.md``,
``sample/<sample>/<id>/entry.md`` and ``experiment/<experiment>/summary.md``.
Each file is YAML front matter (title, written, by, experiment, images, and
edited, edited_by once rewritten) and then the text; the images it cites sit
beside it, copied in at most ``IMAGE_PX`` on their long side. The App Server's
routes, and so the agent's tools, read and write through these functions.
"""

import os
import re
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Union

import yaml

from fibsem import paths
from fibsem.util.timestamps import now, now_iso, to_datetime

INSTRUMENT = "instrument"
SAMPLE = "sample"
EXPERIMENT = "experiment"
SECTIONS = (INSTRUMENT, SAMPLE, EXPERIMENT)
AGENT = "agent"
OPERATOR = "operator"
WRITERS = (AGENT, OPERATOR)

IMAGE_PX = 1024
MAX_TITLE = 200
MAX_TEXT = 20000
MAX_IMAGES = 8
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
# An id, a sample or an image name: one folder or file name, never a path.
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._-]{0,119}")
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
    sample: Optional[str] = None
    images: int = 0
    edited: bool = False


@dataclass
class Entry:
    """One entry: its front matter, its text, its image names and its file."""

    section: str
    id: str
    title: str
    text: str = ""
    written: str = ""
    by: str = ""
    sample: Optional[str] = None
    experiment: Optional[str] = None
    images: List[str] = field(default_factory=list)
    edited: bool = False
    edited_by: Optional[str] = None
    path: str = ""

    def row(self) -> Row:
        return Row(
            self.section,
            self.id,
            self.title,
            self.written,
            self.by,
            self.sample,
            len(self.images),
            self.edited,
        )


def root() -> Path:
    return paths.state_root() / "knowledge"


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


def _sample(section: str, sample: Optional[str]) -> Optional[str]:
    """The sample's slug for a sample entry; None for any other."""
    if section != SAMPLE:
        if sample:
            raise KnowledgeError("invalid", "Only a sample entry takes a sample.")
        return None
    named = slug(sample or "", 40)
    if not named:
        raise KnowledgeError(
            "invalid", "A sample entry names its sample: a short slug, e.g. yeast."
        )
    return named


def _folder(section: str, id: str, sample: Optional[str]) -> Path:
    if section == SAMPLE:
        return root() / SAMPLE / sample / id
    return root() / section / id


def _file(section: str) -> str:
    return "summary.md" if section == EXPERIMENT else "entry.md"


def _parse(path: Path, section: str, id: str, sample: Optional[str]) -> Entry:
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
    # an unquoted time or date is read by YAML as one
    written = meta.get("written")
    written = written.isoformat() if hasattr(written, "isoformat") else written
    return Entry(
        section=section,
        id=id,
        title=str(meta.get("title") or id),
        text=text.strip("\n"),
        written=str(written or ""),
        by=str(meta.get("by") or ""),
        sample=sample,
        experiment=str(meta["experiment"]) if meta.get("experiment") else None,
        images=[str(name) for name in images] if isinstance(images, list) else [],
        edited=bool(meta.get("edited")),
        edited_by=str(meta["edited_by"]) if meta.get("edited_by") else None,
        path=str(path),
    )


def _found(folder: Path, section: str, sample: Optional[str]) -> List[Entry]:
    file = _file(section)
    if not folder.is_dir():
        return []
    return [
        _parse(child / file, section, child.name, sample)
        for child in sorted(folder.iterdir())
        if (child / file).is_file()
    ]


def _newest_first(entry: Entry):
    at = to_datetime(entry.written)
    return at.timestamp() if at is not None else 0.0


def all_entries(
    section: Optional[str] = None, sample: Optional[str] = None
) -> List[Entry]:
    """Every entry, or a section's (and a sample's) only; newest first."""
    sections = SECTIONS if section is None else (_section(section),)
    found: List[Entry] = []
    for name in sections:
        if name == SAMPLE:
            wanted = [slug(sample, 40)] if sample else samples()
            for each in wanted:
                found += _found(root() / SAMPLE / each, SAMPLE, each)
        elif not sample:
            found += _found(root() / name, name, None)
    return sorted(found, key=_newest_first, reverse=True)


def entries(section: Optional[str] = None, sample: Optional[str] = None) -> List[Row]:
    """The index: one row per entry, newest first."""
    return [entry.row() for entry in all_entries(section, sample)]


def samples() -> List[str]:
    """The samples with entries, by slug."""
    folder = root() / SAMPLE
    if not folder.is_dir():
        return []
    return sorted(child.name for child in folder.iterdir() if child.is_dir())


def read(section: str, id: str, sample: Optional[str] = None) -> Entry:
    section = _section(section)
    sample = _sample(section, sample)
    path = _folder(section, _name(id, "id"), sample) / _file(section)
    if not path.is_file():
        where = f" for sample {sample}" if sample else ""
        raise KnowledgeError(
            "not_found", f"No {section} entry {id!r}{where}: knowledge_list lists them."
        )
    return _parse(path, section, id, sample)


def image(section: str, id: str, name: str, sample: Optional[str] = None) -> Path:
    """The file of one image an entry cites."""
    entry = read(section, id, sample)
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


def write(
    section: str,
    title: str,
    text: str,
    by: str,
    sample: Optional[str] = None,
    id: Optional[str] = None,
    experiment: Optional[str] = None,
    images: Sequence[Union[str, Path]] = (),
) -> Entry:
    """Write a new entry, or rewrite entry *id*: its title and text replaced,
    *images* (files to copy in) added to those it has. A new entry's id is the
    date and its title's slug; an experiment summary's is the experiment's name,
    and a rewrite of one that isn't there writes it. A rewrite keeps who wrote
    it and when, and is marked edited by *by*."""
    section = _section(section)
    sample = _sample(section, sample)
    if by not in WRITERS:
        raise KnowledgeError("invalid", f"by is one of {list(WRITERS)}.")
    title = title.strip() if isinstance(title, str) else ""
    if not title or len(title) > MAX_TITLE:
        raise KnowledgeError("invalid", f"title is 1 to {MAX_TITLE} characters.")
    if not isinstance(text, str) or len(text) > MAX_TEXT:
        raise KnowledgeError("invalid", f"text is at most {MAX_TEXT} characters.")
    if section == EXPERIMENT:
        if not id:
            raise KnowledgeError(
                "invalid", "An experiment summary's id is the experiment's name."
            )
        experiment = id
    sources = [Path(source) for source in images or ()]
    pictures = [_picture(source) for source in sources]
    with _lock:
        existing = None
        if id is not None:
            folder = _folder(section, _name(id, "id"), sample)
            path = folder / _file(section)
            if path.is_file():
                existing = _parse(path, section, id, sample)
            elif section != EXPERIMENT:
                raise KnowledgeError(
                    "not_found",
                    f"No {section} entry {id!r}: leave id out to write a new one.",
                )
        else:
            base = f"{now().date().isoformat()}-{slug(title) or 'entry'}"
            id, n = base, 2
            while _folder(section, id, sample).exists():
                id, n = f"{base}-{n}", n + 1
            folder = _folder(section, id, sample)
            path = folder / _file(section)
        names = list(existing.images) if existing else []
        if len(names) + len(pictures) > MAX_IMAGES:
            raise KnowledgeError(
                "invalid", f"An entry cites at most {MAX_IMAGES} images."
            )
        folder.mkdir(parents=True, exist_ok=True)
        for source, picture in zip(sources, pictures):
            name = _image_name(source, names)
            picture.save(folder / name, format="PNG")
            names.append(name)
        meta = {
            "title": title,
            "written": existing.written if existing else now_iso(),
            "by": existing.by if existing else by,
        }
        kept = existing.experiment if existing and existing.experiment else experiment
        if kept:
            meta["experiment"] = kept
        meta["images"] = names
        if existing:
            meta["edited"] = True
            meta["edited_by"] = by
        _save(path, meta, text)
        return _parse(path, section, id, sample)


def delete(section: str, id: str, sample: Optional[str] = None) -> Entry:
    """Remove an entry and its images: the entry as it was."""
    entry = read(section, id, sample)
    with _lock:
        folder = Path(entry.path).parent
        shutil.rmtree(folder)
        if entry.sample and not any(folder.parent.iterdir()):
            folder.parent.rmdir()
    return entry
