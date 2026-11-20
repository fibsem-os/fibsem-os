"""Segments: an episode cut into the decisions an eval case can score.

The segmenter (an agent, offline) writes ``segments.yaml`` in the episode
folder; a person keeps, changes or drops each one; the cutter builds a case
from each kept one. A segment names its task (or, for a decision no task
scores yet, a ``new_task`` with its question and answer shape), the item it
is about, the events it starts and ends at, the files the case is built
from, what the run decided and whether the operator overrode it.

The start event is the cut: a case shows only what was saved before it, so
:func:`check` refuses an input the record shows was saved later.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import yaml

from fibsem.ai.evals import tasks
from fibsem.ai.evals.core.catalogue import ANSWER_SHAPES
from fibsem.ai.evals.core.episode import Episode

FILE = "segments.yaml"
FORMAT = 0
STATUSES = ("proposed", "kept", "dropped")


@dataclass
class Check:
    problems: List[str] = field(default_factory=list)
    # What the record can't confirm, such as when an input was saved.
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def path(episode: Episode) -> Path:
    return episode.path / FILE


def load(episode: Episode) -> Dict[str, Any]:
    try:
        return yaml.safe_load(path(episode).read_text()) or {}
    except OSError:
        return {}


def save(episode: Episode, data: Dict[str, Any]) -> Path:
    target = path(episode)
    target.write_text(yaml.safe_dump(data, sort_keys=False, width=100))
    return target


def _files(value: Any, where: str = "") -> Iterator[Tuple[str, str]]:
    """Each (where, path) among an input's values: a string with a file
    extension, at any depth of lists and mappings."""
    if isinstance(value, str):
        if "." in Path(value.replace("\\", "/")).name:
            yield where, value
    elif isinstance(value, dict):
        for key, inner in value.items():
            yield from _files(inner, f"{where}.{key}" if where else str(key))
    elif isinstance(value, list):
        for i, inner in enumerate(value):
            yield from _files(inner, f"{where}[{i}]")


def _first_named(episode: Episode, events: List[Dict[str, Any]]) -> Dict[Path, float]:
    """When the record first names each file in the experiment: the earliest
    event whose payload holds its path."""
    found: Dict[Path, float] = {}
    for event in events:
        for _, recorded in _files(event.get("payload") or {}):
            local = episode.local(recorded)
            if local is not None and local not in found:
                found[local] = float(event.get("timestamp") or 0)
    return found


def check(episode: Episode, data: Optional[Dict[str, Any]] = None) -> Check:
    """What is wrong with the episode's segments: unknown tasks or events,
    inputs missing from the copy or saved after the segment's start."""
    data = load(episode) if data is None else data
    result = Check()
    if data.get("format") != FORMAT:
        result.problems.append(f"format is {data.get('format')!r}, not {FORMAT}")
    segments = data.get("segments")
    if not isinstance(segments, list):
        result.problems.append("no segments list")
        return result
    events = episode.events()
    by_id = {str(e.get("id")): e for e in events if e.get("id") is not None}
    named = _first_named(episode, events)
    seen = set()
    for n, segment in enumerate(segments, start=1):
        sid = str(segment.get("id") or f"#{n}")
        say = result.problems.append
        if sid in seen:
            say(f"{sid}: id used twice")
        seen.add(sid)
        status = segment.get("status", "proposed")
        if status not in STATUSES:
            say(f"{sid}: status {status!r} is not one of {', '.join(STATUSES)}")
        if not str(segment.get("decision") or "").strip():
            say(f"{sid}: no decision (what the run decided)")

        name = segment.get("task")
        wanted: Dict[str, str] = {}
        if name:
            task = tasks.get(name)
            if task is None or task.from_run is None:
                say(f"{sid}: task {name!r} is not in the catalogue")
            else:
                wanted = task.from_run.inputs
        else:
            new = segment.get("new_task") or {}
            if not new.get("name") or not new.get("question"):
                say(f"{sid}: no task, and new_task has no name or question")
            if new.get("answer_shape") not in ANSWER_SHAPES:
                say(
                    f"{sid}: new_task answer_shape {new.get('answer_shape')!r} is "
                    f"not one of {', '.join(ANSWER_SHAPES)}"
                )

        start, end = segment.get("start_event"), segment.get("end_event")
        cut = None
        for label, event_id in (("start_event", start), ("end_event", end)):
            if event_id is None:
                say(f"{sid}: no {label}")
            elif str(event_id) not in by_id:
                say(f"{sid}: {label} {event_id!r} is not an event of this episode")
        if str(start) in by_id:
            cut = float(by_id[str(start)].get("timestamp") or 0)
        if cut is not None and str(end) in by_id:
            if float(by_id[str(end)].get("timestamp") or 0) < cut:
                say(f"{sid}: end_event comes before start_event")

        if not isinstance(segment.get("facts") or {}, dict):
            say(f"{sid}: facts is not a mapping")
        inputs = segment.get("inputs") or {}
        if not isinstance(inputs, dict):
            say(f"{sid}: inputs is not a mapping")
            continue
        for missing in sorted(set(wanted) - set(inputs)):
            say(f"{sid}: no input {missing!r} ({wanted[missing]})")
        for extra in sorted(set(inputs) - set(wanted)) if wanted else ():
            say(f"{sid}: input {extra!r} is not one the task takes")
        for where, recorded in _files(inputs):
            local = episode.local(recorded)
            if local is None:
                say(f"{sid}: input {where} {recorded!r} is not in the episode")
                continue
            when = named.get(local)
            if when is None:
                result.notes.append(
                    f"{sid}: no event names {recorded!r}, so when it was saved "
                    "is not on the record"
                )
            elif cut is not None and when > cut:
                say(
                    f"{sid}: input {where} {recorded!r} was saved after the "
                    "segment starts; a case may only show what came before"
                )
    return result


def kept(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [s for s in data.get("segments") or [] if s.get("status") == "kept"]
