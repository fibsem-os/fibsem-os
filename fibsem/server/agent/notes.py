"""The latest notes by tag, from the whole record.

An agent keeps a long run's state in tagged notes (``experiment_add_note``
with text starting ``[intake]``, ``[run_state v3]``, ...), and a fresh
context rebuilds from the latest of each. The event long-poll serves only
the live buffer, which a long run's progress records fill, and which an app
restart empties: a note written early in the run is no longer in it. This
reads the buffer, then the record's file backwards until it has what was
asked, so it costs what it finds, not the whole file.

A tag is the note's first bracketed word, lower case: ``[run_state v3]`` is
``run_state``. Messages between the operator and the agent are not notes.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from fibsem.workflows.actions import NOTE, is_message
from fibsem.workflows.events.kinds import LIVE_KINDS
from fibsem.workflows.events.recording import read_events_before

# How many tags one read asks for, and how many notes of each.
MAX_TAGS = 10
MAX_EACH = 10
_TAG = re.compile(r"^\s*\[([A-Za-z][\w-]*)")
_ASKED = re.compile(r"^[A-Za-z][\w-]*$")


class NotesRefused(ValueError):
    """The tags or the count asked for aren't a read this answers."""


def tag_of(record: Dict[str, Any]) -> Optional[str]:
    """The tag of a note *record*, lower case; None for anything else."""
    if record.get("kind") != NOTE or is_message(record):
        return None
    text = (record.get("payload") or {}).get("text")
    found = _TAG.match(text) if isinstance(text, str) else None
    return found.group(1).lower() if found else None


def asked_tags(tags: Sequence[str]) -> List[str]:
    """*tags* as asked (``run_state``, ``[intake]``, "intake, sites"), lower
    case and in order, or `NotesRefused`."""
    found: List[str] = []
    for part in tags:
        for tag in str(part).split(","):
            tag = tag.strip().strip("[]").strip().lower()
            if not tag:
                continue
            if not _ASKED.match(tag):
                raise NotesRefused(f"{tag!r} isn't a tag: a word, as in [run_state].")
            if tag not in found:
                found.append(tag)
    if not found:
        raise NotesRefused("Name at least one tag, as in tags=intake,run_state.")
    if len(found) > MAX_TAGS:
        raise NotesRefused(f"At most {MAX_TAGS} tags at once.")
    return found


def latest_notes(
    held: Sequence[Dict[str, Any]],
    path: Optional[Path],
    tags: Sequence[str],
    each: int = 1,
) -> Dict[str, Any]:
    """The newest *each* notes of every tag in *tags*: from *held* (the live
    buffer, oldest first) and then the record's file at *path*, read
    backwards. Answers ``notes`` (by tag as asked, newest first) and
    ``missing``, the tags with no note on the record."""
    tags = asked_tags(tags)
    if not 1 <= each <= MAX_EACH:
        raise NotesRefused(f"each is 1 to {MAX_EACH}.")
    wanted = {tag: each for tag in tags}
    seen: set = set()
    found: Dict[str, List[Dict[str, Any]]] = {tag: [] for tag in tags}

    def take(record: Dict[str, Any]) -> bool:
        tag = tag_of(record)
        if tag is None or not wanted.get(tag):
            return False
        key = record.get("id") or (record.get("timestamp"), record.get("seq"))
        if key in seen:
            return False
        seen.add(key)
        wanted[tag] -= 1
        found[tag].append(record)
        return True

    for record in reversed(list(held)):
        take(record)
    still = sum(wanted.values())
    if still and path is not None and Path(path).is_file():
        read_events_before(Path(path), limit=still, skip_kinds=LIVE_KINDS, keep=take)
    notes = [_row(tag, record) for tag in tags for record in found[tag]]
    return {"notes": notes, "missing": [tag for tag in tags if not found[tag]]}


def _row(tag: str, record: Dict[str, Any]) -> Dict[str, Any]:
    payload = record.get("payload") or {}
    row: Dict[str, Any] = {
        "tag": tag,
        "id": record.get("id"),
        "timestamp": record.get("timestamp"),
        "text": payload.get("text", ""),
    }
    if payload.get("item_name"):
        row["item_name"] = payload["item_name"]
    if payload.get("evidence"):
        row["evidence"] = payload["evidence"]
    return row
