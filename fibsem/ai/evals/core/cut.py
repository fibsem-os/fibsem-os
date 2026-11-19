"""Cutting cases from an episode: one case per kept segment.

A cut case is an ordinary case (format 0, tier ``real``): ``experiment/``
holds copies of the segment's inputs, all saved before its start event;
``reference/`` holds what the run saved about the item afterwards, which
the agent never sees; ``live:`` holds what the run decided and any override;
``source:`` names the episode, segment and events. Two files give the
context modes: ``context.yaml`` (what came before on this item, as a short
summary) and ``reference/transcript-to-cut.jsonl`` (the agent's own
transcript up to the cut). The key starts as ``todo``: it is set on the
keying page, from the override or from what happened next.

Each task builds its own fields (its inputs, its brief) with its
``from_run.cut``, ``cut(segment, inputs, folder) -> dict``, where *inputs* are the
segment's inputs with each file already copied and named by its path in
the case.
"""

import importlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from fibsem.ai.evals import tasks
from fibsem.ai.evals.core import segments
from fibsem.ai.evals.core.cases import FILE, FORMAT
from fibsem.ai.evals.core.episode import Episode

log = logging.getLogger(__name__)

CONTEXT = "context.yaml"
TRANSCRIPT = "transcript-to-cut.jsonl"


class CutError(RuntimeError):
    pass


def _function(spec: str) -> Callable[..., Dict[str, Any]]:
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name)


def _copy_inputs(value: Any, episode: Episode, into: Path) -> Any:
    """The inputs with each file copied into *into* (``experiment/``) and
    named by its path in the case."""
    if isinstance(value, dict):
        return {k: _copy_inputs(v, episode, into) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy_inputs(v, episode, into) for v in value]
    if isinstance(value, str):
        local = episode.local(value)
        if local is None or not local.is_file():
            return value
        rel = local.relative_to(episode.experiment)
        target = into / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, target)
        return f"{into.name}/{rel.as_posix()}"
    return value


def _time(event: Dict[str, Any]) -> float:
    return float(event.get("timestamp") or 0)


def _later_files(
    episode: Episode, events: List[Dict[str, Any]], item: Optional[str], cut: float
) -> List[Path]:
    """Files the record names after the cut for the segment's item (for a
    segment with no item, none): what happened next."""
    if not item:
        return []
    found: Dict[Path, None] = {}
    for event in events:
        if _time(event) <= cut or event.get("item") not in (item, None):
            continue
        for _, recorded in segments._files(event.get("payload") or {}):
            local = episode.local(recorded)
            if local is None or not local.is_file():
                continue
            if item in local.relative_to(episode.experiment).parts:
                found.setdefault(local, None)
    return list(found)


def _transcript_to_cut(episode: Episode, cut: float) -> List[str]:
    """Every session's transcript lines up to the cut, by their timestamp."""
    kept: List[str] = []
    for entry in episode.data.get("sessions") or []:
        folder = episode.session(str(entry.get("id")))
        path = folder / "transcript.jsonl" if folder else None
        if path is None or not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                stamp = json.loads(line).get("timestamp")
            except ValueError:
                continue
            if not stamp:
                continue
            when = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
            if when.timestamp() <= cut:
                kept.append(line)
    return kept


def _context(data: Dict[str, Any], segment: Dict[str, Any]) -> Dict[str, Any]:
    """The summary context: earlier kept segments on the same item, what
    was decided and whether it was overridden."""
    earlier = []
    for other in segments.kept(data):
        if other is segment:
            break
        if segment.get("item") and other.get("item") == segment.get("item"):
            entry = {"decision": other.get("decision")}
            if other.get("override"):
                entry["override"] = other["override"]
            earlier.append(entry)
    return {"item": segment.get("item"), "earlier_decisions": earlier}


def case_id(episode: Episode, segment: Dict[str, Any]) -> str:
    return f"{episode.name}-{segment['id']}"


def cut_segment(
    episode: Episode, data: Dict[str, Any], segment: Dict[str, Any], out: Path
) -> Path:
    """The case for one kept segment, under *out*/<task>/; its folder."""
    task = tasks.get(segment.get("task"))
    if task is None or task.from_run is None:
        raise CutError(f"{segment.get('id')}: no task to cut a case for")
    if task.from_run.cut is None:
        raise CutError(f"{segment.get('id')}: no cutter yet for {task.name}")
    events = episode.events()
    by_id = {str(e.get("id")): e for e in events}
    start = by_id[str(segment["start_event"])]
    cut = _time(start)

    folder = Path(out) / task.name.replace("_", "-") / case_id(episode, segment)
    if folder.exists():
        raise CutError(f"{folder} exists")
    experiment, reference = folder / "experiment", folder / "reference"
    experiment.mkdir(parents=True)
    reference.mkdir()
    inputs = _copy_inputs(segment.get("inputs") or {}, episode, experiment)
    later = reference / "later"
    for path in _later_files(episode, events, segment.get("item"), cut):
        target = later / path.relative_to(episode.experiment)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    lines = _transcript_to_cut(episode, cut)
    if lines:
        (reference / TRANSCRIPT).write_text("\n".join(lines) + "\n")

    sessions = [
        {k: s[k] for k in ("id", "model", "fibsem", "playbook_hash") if k in s}
        for s in episode.data.get("sessions") or []
    ]
    case: Dict[str, Any] = {
        "id": case_id(episode, segment),
        "task": task.name,
        "tier": "real",
        "format": FORMAT,
        "playbook": None,
        "source": {
            "episode": episode.name,
            "segment": segment["id"],
            "item": segment.get("item"),
            "start_event": segment["start_event"],
            "end_event": segment.get("end_event"),
            "start_time": start.get("t"),
            "sessions": sessions,
        },
    }
    case.update(_function(task.from_run.cut)(segment, inputs, folder))
    live: Dict[str, Any] = {"decision": segment.get("decision")}
    if segment.get("answer") is not None:
        live["answer"] = segment["answer"]
    if segment.get("override"):
        live["override"] = segment["override"]
    case["live"] = live
    case["key"] = {"status": "todo: key it from the override or what happened next"}
    if case.get("playbook") is None:
        case.pop("playbook")
    (folder / CONTEXT).write_text(
        yaml.safe_dump(_context(data, segment), sort_keys=False, width=100)
    )
    with open(folder / FILE, "w") as f:
        yaml.safe_dump(case, f, sort_keys=False, width=100)
    return folder


def cut_episode(episode: Episode, out: Path) -> Dict[str, Any]:
    """A case for each kept segment that checks; what was cut and what was
    skipped, and why."""
    data = segments.load(episode)
    check = segments.check(episode, data)
    if not check.ok:
        raise CutError(
            "segments.yaml has problems; fix them first:\n- "
            + "\n- ".join(check.problems)
        )
    made, skipped = [], []
    for segment in segments.kept(data):
        try:
            made.append(cut_segment(episode, data, segment, out))
        except CutError as e:
            skipped.append(str(e))
    return {"made": made, "skipped": skipped}
