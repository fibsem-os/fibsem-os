"""Read an experiment's recorded actions back as a timeline, for replay.

An experiment recorded with the event stream has an ``events.jsonl`` beside its
log (FIB-455), and that is read when it is there: every event already says which
lamella or grid and task it belongs to, and names the file an image was saved
to. Experiments recorded before it are read from the log, as follows.

Everything an experiment did is already in its ``logfile.log``: every image
acquisition (with its full metadata and where it was saved), every stage move
and stage read-back, every milling stage (its patterns and how long it ran),
beam shifts, answers to the workflow's prompts, and the task step each of them
happened in. Fluorescence acquisitions are the exception -- the log has no
record of them -- so they are placed from the ``.ome.tiff`` files' own
metadata instead. This module turns all of it into an ordered list of
:class:`ReplayEvent` and answers "what did the instrument look like at event
*i*" (:meth:`ExperimentReplay.scene`).

It reads; it never writes. It does not call ``Experiment.load()``, which has
side effects on the directory it opens, and it touches no Qt.

The records are the ``logging.debug({...})`` dicts the microscope and task
code emit, so they are Python reprs rather than JSON: ``np.float64(2e-09)``,
``FibsemStagePosition(x=...)``, ``array([...])``. They are read with a small
AST walk that accepts literals and those call shapes and nothing else -- no
``eval`` -- so a record that is not data is skipped, never executed. The
stream exists only at DEBUG level.
"""

from __future__ import annotations

import ast
import bisect
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, Iterator, List, Optional, Tuple

from fibsem.applications.autolamella.event_recording import EVENTS_FILENAME, read_events

logger = logging.getLogger(__name__)

LOGFILE_NAME = "logfile.log"

# `2026-09-13 20:02:40,521 — root — DEBUG — get:748 — {...}`
_HEADER = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) — (.*?) — ([A-Z]+) — (.*?):(\d+) — (.*)$"
)
_MSG_KEY = re.compile(r"^\{'msg': '([^']+)'")
# `burning spot 3: Point(x=0.08, y=0.56, name=None), exposure time: 10.0, milling current: 1e-10`
# -- free text, not a record; x and y are fractions of the beam's scan field.
_SPOT = re.compile(
    r"burning spot (\d+): Point\(x=([-+\d.e]+), y=([-+\d.e]+).*?"
    r"exposure time: ([-+\d.e]+), milling current: ([-+\d.e]+)"
)
# `prompt answered: PickPOI response=True by=operator adjusted=False`
_PROMPT = re.compile(
    r"^prompt answered: (\w+) response=(\w+) by=(\w+)(?: adjusted=(\w+))?"
)

# How long after a move the stage read-back that reports where it ended up
# may arrive and still count as that move's result.
_MOVE_SETTLE = timedelta(seconds=10)

# The file names acquisitions save under, as `acquire.take_reference_images`
# suffixes them. The recorded `filename` is the stem before this suffix.
_BEAM_SUFFIX = {"ELECTRON": "_eb", "ION": "_ib"}
_BEAM_LABEL = {"ELECTRON": "SEM", "ION": "FIB"}

# The steps that end a task's context. The log records only FINISHED; the event
# stream also says when a task failed, was cancelled or was skipped.
_TASK_ENDS = {"FINISHED", "FAILED", "CANCELLED", "SKIPPED"}


class EventKind:
    """What an event records. Plain strings, so a filter can be a set of them."""

    TASK = "task"
    PROMPT = "prompt"
    IMAGE = "image"
    FLUORESCENCE = "fluorescence"
    STAGE = "stage"
    MILLING = "milling"
    ALIGNMENT = "alignment"
    CORRELATION = "correlation"  # an accepted correlation: event stream only
    EDIT = "edit"  # a change to a lamella's plan: recorded by the event stream only
    MESSAGE = "message"

    ALL = (
        TASK,
        PROMPT,
        IMAGE,
        FLUORESCENCE,
        STAGE,
        MILLING,
        ALIGNMENT,
        CORRELATION,
        EDIT,
        MESSAGE,
    )


# ── reading the log ──────────────────────────────────────────────────────────


@dataclass
class LogRecord:
    """One line of the log that has the standard header."""

    time: datetime
    level: str
    function: str
    message: str


def _decode(raw: bytes) -> str:
    # The field separator is an em dash, so the encoding decides whether a line
    # parses at all: older Windows installs wrote cp1252.
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def read_log_records(path: Path) -> Iterator[LogRecord]:
    """Every line of *path* with the standard header, in file order.

    Lines without a header (the rest of a traceback, a multi-line message) are
    skipped: nothing replayed is carried on them.
    """
    text = _decode(Path(path).read_bytes())
    for line in text.splitlines():
        m = _HEADER.match(line)
        if m is None:
            continue
        try:
            time = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f")
        except ValueError:
            continue
        yield LogRecord(time, m.group(3), m.group(4), m.group(6))


class _NotData(ValueError):
    pass


def _is_dotted_name(node: ast.AST) -> bool:
    """`name` or `a.b.name` -- what a repr of a value calls, and nothing else."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name)


def _literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Dict):
        return {_literal(k): _literal(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [_literal(e) for e in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _literal(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.Call) and _is_dotted_name(node.func):
        # `np.float64(2e-09)`, `array([1, 2])`: the value is the argument.
        if len(node.args) == 1 and not node.keywords:
            return _literal(node.args[0])
        # `FibsemStagePosition(x=..., y=...)`: the value is its fields.
        if not node.args and node.keywords and all(k.arg for k in node.keywords):
            return {k.arg: _literal(k.value) for k in node.keywords}
    if isinstance(node, ast.Attribute) and node.attr in ("True_", "False_"):
        return node.attr == "True_"  # np.True_
    if isinstance(node, ast.Name) and node.id in ("nan", "inf"):
        return float(node.id)
    raise _NotData(type(node).__name__)


def parse_record(message: str) -> Optional[Dict[str, Any]]:
    """The dict a ``logging.debug({...})`` call wrote, or None if it is not one."""
    try:
        value = _literal(ast.parse(message, mode="eval").body)
    except (SyntaxError, _NotData, TypeError, ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


# ── resolving where an image went ────────────────────────────────────────────


def _pure_path(recorded: str):
    if "\\" in recorded or re.match(r"^[A-Za-z]:", recorded):
        return PureWindowsPath(recorded)
    return PurePosixPath(recorded)


def _path_parts(recorded: str) -> Tuple[str, ...]:
    return _pure_path(recorded).parts


class _ImageResolver:
    """Finds a recorded save path under the experiment directory as it is now.

    The recorded path is where the file was written on the instrument PC; the
    directory may since have been copied, renamed or moved to another OS. The
    longest tail of the recorded directory that exists under *root* is taken as
    the same directory.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self._dirs: Dict[str, Optional[Path]] = {}

    def _directory(self, recorded_dir: str) -> Optional[Path]:
        if recorded_dir in self._dirs:
            return self._dirs[recorded_dir]
        parts = _path_parts(recorded_dir)
        found = None
        # From the longest tail to the empty one (the root itself); never the
        # whole recorded path, which could name a directory outside the root.
        for i in range(1, len(parts) + 1):
            candidate = self.root.joinpath(*parts[i:])
            if candidate.is_dir():
                found = candidate
                break
        self._dirs[recorded_dir] = found
        return found

    def resolve(
        self, recorded_dir: Optional[str], filename: Optional[str], beam: Optional[str]
    ) -> Optional[Path]:
        if not recorded_dir or not filename:
            return None
        directory = self._directory(recorded_dir)
        if directory is None:
            return None
        names = [f"{filename}{_BEAM_SUFFIX.get(beam or '', '')}.tif", f"{filename}.tif"]
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate
        return None

    def resolve_path(self, recorded: Optional[str]) -> Optional[Path]:
        """A recorded file path -- the file actually written -- as it is now."""
        if not recorded:
            return None
        path = _pure_path(recorded)
        directory = self._directory(str(path.parent))
        if directory is None or not (directory / path.name).is_file():
            return None
        return directory / path.name


# ── events ───────────────────────────────────────────────────────────────────


@dataclass
class ReplayEvent:
    """One recorded action, in the context it happened in.

    ``item`` is what the workflow was working on -- a lamella, or a grid in a
    grid workflow, as ``item_type`` says -- and ``task``/``step`` the task
    step that was running; all None for an action taken outside a workflow.
    """

    time: datetime
    kind: str
    summary: str
    item: Optional[str] = None
    task: Optional[str] = None
    step: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    # IMAGE
    beam: Optional[str] = None
    image_path: Optional[Path] = None
    saved: bool = False
    # MILLING
    duration: Optional[float] = None
    # STAGE, and IMAGE (where the stage was when it was taken)
    position: Optional[Dict[str, Any]] = None
    item_type: Optional[str] = None  # "lamella" or "grid"

    @property
    def image_on_disk(self) -> bool:
        """This acquisition's own pixels can be shown: saved, found, not overwritten."""
        return self.image_path is not None

    @property
    def _image_settings(self) -> Dict[str, Any]:
        # The log nests an acquisition's settings in its metadata; the event
        # stream records them flat.
        return (self.data.get("metadata") or {}).get("image") or self.data

    @property
    def field_size(self) -> Optional[Tuple[float, float]]:
        """The (width, height) in metres an image covers, if recorded."""
        settings = self._image_settings
        try:
            hfw = float(settings["hfw"])
            w, h = (float(v) for v in settings["resolution"])
        except (KeyError, TypeError, ValueError):
            return None
        return (hfw, hfw * h / w) if w else None

    @property
    def is_full_frame(self) -> bool:
        """An image of the beam's whole field, not a reduced-area crop."""
        return not self._image_settings.get("reduced_area")


@dataclass
class ReplayScene:
    """What the instrument looked like at one event.

    The images are the latest acquisitions *that can be shown*, which may be
    older than the latest acquisitions -- ``sem_unsaved_since`` and
    ``fib_unsaved_since`` count the ones in between, so a viewer can say so
    rather than pass an old frame off as the current one.

    While milling (or a spot burn) is being shown, ``fib`` is the latest
    full-frame FIB image instead: patterns are placed relative to the beam's
    whole field, so drawing them on a reduced-area crop would misplace them.
    Both last until the FIB next acquires a full frame, saved or not.
    """

    index: int
    event: ReplayEvent
    sem: Optional[ReplayEvent]
    fib: Optional[ReplayEvent]
    fm: Optional[ReplayEvent]
    sem_unsaved_since: int
    fib_unsaved_since: int
    stage_position: Optional[Dict[str, Any]]
    milling: Optional[ReplayEvent]
    milling_stages: List[Dict[str, Any]]
    # Spots burnt since the FIB last acquired a full frame, in metres from the
    # centre of the field (+y down), so they land right on an image taken at
    # another field width. Empty when that frame's field width is unknown.
    spots: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class ExperimentReplay:
    root: Path
    events: List[ReplayEvent]
    stage_track: List[Tuple[datetime, Dict[str, Any]]]
    records_read: int = 0
    records_unreadable: int = 0
    # The file the timeline was read from: events.jsonl, or the log.
    source: str = LOGFILE_NAME
    # The scene lookups, per item they are scoped to (None: the whole run).
    _indexes: Dict[Optional[str], Dict[str, List[int]]] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        self._stages_by_task: Dict[Any, List[Dict[str, Any]]] = {}
        for e in self.events:
            stage = e.data.get("stage")
            if e.kind == EventKind.MILLING and isinstance(stage, dict):
                self._stages_by_task.setdefault(
                    e.data.get("milling_task_id"), []
                ).append(stage)
        self._track_times = [t for t, _ in self.stage_track]

    @property
    def start(self) -> Optional[datetime]:
        return self.events[0].time if self.events else None

    @property
    def end(self) -> Optional[datetime]:
        return self.events[-1].time if self.events else None

    def _index(self, item: Optional[str]) -> Dict[str, List[int]]:
        if item not in self._indexes:
            self._indexes[item] = self._build_index(item)
        return self._indexes[item]

    def _build_index(self, item: Optional[str]) -> Dict[str, List[int]]:
        # For each event: the latest showable SEM/FIB image, the number of
        # unshowable ones since, and the latest milling event -- all at or
        # before it, and all of *item* when one is given. Precomputed so a
        # seek is a lookup, not a scan.
        sem = fib = fib_full = fib_acquired = fm = mill = -1
        sem_n = fib_n = 0
        cols: Dict[str, List[int]] = {
            k: []
            for k in (
                "sem",
                "fib",
                "fib_full",
                "fib_acquired",
                "fm",
                "sem_n",
                "fib_n",
                "mill",
            )
        }
        for i, e in enumerate(self.events):
            if item is not None and e.item != item:
                pass  # another item's: changes nothing in this item's view
            elif e.kind == EventKind.IMAGE:
                if e.beam == "ELECTRON":
                    if e.image_on_disk:
                        sem, sem_n = i, 0
                    else:
                        sem_n += 1
                elif e.beam == "ION":
                    if e.is_full_frame:
                        fib_acquired = i
                    if e.image_on_disk:
                        fib, fib_n = i, 0
                        if e.is_full_frame:
                            fib_full = i
                    else:
                        fib_n += 1
            elif e.kind == EventKind.FLUORESCENCE and e.image_on_disk:
                fm = i
            elif e.kind == EventKind.MILLING and "stage" in e.data:
                mill = i
            cols["sem"].append(sem)
            cols["fib"].append(fib)
            cols["fib_full"].append(fib_full)
            cols["fib_acquired"].append(fib_acquired)
            cols["fm"].append(fm)
            cols["sem_n"].append(sem_n)
            cols["fib_n"].append(fib_n)
            cols["mill"].append(mill)
        return cols

    def stage_position_at(self, time: datetime) -> Optional[Dict[str, Any]]:
        """The last stage position the log reported at or before *time*."""
        i = bisect.bisect_right(self._track_times, time) - 1
        return self.stage_track[i][1] if i >= 0 else None

    def scene(self, index: int, item: Optional[str] = None) -> ReplayScene:
        """The instrument at event *index*.

        With *item*, only that item's images, milling and FM are shown: the
        latest of *its* acquisitions, not whatever another lamella left on
        screen. The stage position is the instrument's either way.
        """
        event = self.events[index]
        idx = self._index(item)

        def at(col: str) -> Optional[ReplayEvent]:
            j = idx[col][index]
            return self.events[j] if j >= 0 else None

        # Milling stays drawn until the FIB acquires a new full frame: the
        # patterns sit over the frame they were placed on, and the next frame
        # shows the result. Reduced-area frames taken while milling (drift
        # correction) do not end it.
        acquired_i = idx["fib_acquired"][index]
        milling = at("mill")
        if milling is not None and idx["mill"][index] < acquired_i:
            milling = None
        stages: List[Dict[str, Any]] = []
        if milling is not None:
            stages = list(
                self._stages_by_task.get(milling.data.get("milling_task_id"), [])
            )

        # A spot is a fraction of the field the beam was scanning, which is the
        # field of the last full frame it acquired -- not necessarily the image
        # shown, if that one was not saved.
        spots: List[Tuple[float, float]] = []
        field_size = self.events[acquired_i].field_size if acquired_i >= 0 else None
        if field_size is not None:
            width, height = field_size
            for j in range(acquired_i + 1, index + 1):
                spot = self.events[j].data.get("spot")
                if item is not None and self.events[j].item != item:
                    continue
                if self.events[j].kind == EventKind.MILLING and spot is not None:
                    # The event stream records the field the burn scanned; the
                    # log leaves it to be taken from the last full frame.
                    fov = self.events[j].data.get("field_of_view")
                    w, h = (fov, fov * height / width) if fov else (width, height)
                    spots.append(((spot[0] - 0.5) * w, (spot[1] - 0.5) * h))

        fib = at("fib_full") if (milling is not None or spots) else at("fib")

        if event.kind == EventKind.STAGE and event.position is not None:
            position = event.position
        else:
            position = self.stage_position_at(event.time)
        return ReplayScene(
            index=index,
            event=event,
            sem=at("sem"),
            fib=fib,
            fm=at("fm"),
            sem_unsaved_since=idx["sem_n"][index],
            fib_unsaved_since=idx["fib_n"][index],
            stage_position=position,
            milling=milling,
            milling_stages=stages,
            spots=spots,
        )

    def counts(self) -> Dict[str, int]:
        out = {k: 0 for k in EventKind.ALL}
        for e in self.events:
            out[e.kind] = out.get(e.kind, 0) + 1
        return out


# ── building the timeline ────────────────────────────────────────────────────


def _um(value: Any) -> str:
    try:
        return f"{float(value) * 1e6:.1f}"
    except (TypeError, ValueError):
        return "?"


def _deg(value: Any) -> str:
    try:
        return f"{math.degrees(float(value)):.1f}"
    except (TypeError, ValueError):
        return "?"


def _describe_position(pos: Dict[str, Any]) -> str:
    parts = [
        f"{axis}={_um(pos.get(axis))} µm"
        for axis in ("x", "y", "z")
        if pos.get(axis) is not None
    ]
    parts += [
        f"{axis}={_deg(pos.get(axis))}°"
        for axis in ("r", "t")
        if pos.get(axis) is not None
    ]
    return ", ".join(parts)


def _mm(value: Any) -> str:
    try:
        return f"{float(value) * 1e3:.3f}"
    except (TypeError, ValueError):
        return "?"


def _shift(value: Any) -> str:
    """A beam shift or alignment step: tens of nanometres, so nm below a µm."""
    try:
        metres = float(value)
    except (TypeError, ValueError):
        return "?"
    if abs(metres) < 1e-6:
        return f"{round(metres * 1e9)} nm"
    return f"{metres * 1e6:.1f} µm"


def _beam(value: Any) -> str:
    return _BEAM_LABEL.get(value, str(value))


# The wording both readers use, so a move reads the same from either file.


def _beam_move_summary(move: Any, d: Dict[str, Any]) -> str:
    words = ["Stable move" if move == "stable_move" else "Vertical move"]
    if d.get("beam_type") in _BEAM_LABEL:
        words.append(f"in the {_BEAM_LABEL[d['beam_type']]}")
    words.append(f"dx={_um(d.get('dx'))} µm, dy={_um(d.get('dy'))} µm")
    return " ".join(words)


def _beam_shift_summary(d: Dict[str, Any]) -> str:
    return (
        f"{_beam(d.get('beam_type'))} beam shift "
        f"dx={_shift(d.get('dx'))}, dy={_shift(d.get('dy'))}"
    )


def _coincidence_summary(d: Dict[str, Any]) -> str:
    ok = "reliable" if d.get("is_reliable") else f"refused ({d.get('refusal_reason')})"
    return (
        f"Coincidence measured dx={_um(d.get('dx'))} µm, "
        f"dy={_um(d.get('dy'))} µm — {ok}"
    )


def _image_event(record: LogRecord, d: Dict[str, Any]) -> Optional[ReplayEvent]:
    metadata = d.get("metadata")
    if not isinstance(metadata, dict):
        return None
    settings = metadata.get("image") or {}
    state = metadata.get("microscope_state") or {}
    beam = settings.get("beam_type")
    label = _BEAM_LABEL.get(beam, str(beam))
    resolution = settings.get("resolution") or []
    size = "×".join(str(v) for v in resolution) if resolution else "?"
    summary = f"{label} image {size}, HFW {_um(settings.get('hfw'))} µm"
    if settings.get("save") and settings.get("filename"):
        summary += f" — {settings.get('filename')}"
    return ReplayEvent(
        time=record.time,
        kind=EventKind.IMAGE,
        summary=summary,
        data=d,
        beam=beam,
        saved=bool(settings.get("save")),
        position=state.get("stage_position"),
    )


def _milling_summary(
    task_name: Any, stage: Dict[str, Any], duration: Optional[float]
) -> str:
    milling = stage.get("milling") or {}
    pattern = stage.get("pattern") or {}
    current = milling.get("milling_current")
    current_txt = (
        f", {float(current) * 1e9:.2f} nA" if isinstance(current, (int, float)) else ""
    )
    took = f", {duration:.0f} s" if duration else ""
    return (
        f"Mill {task_name or '?'} / {stage.get('name', '?')}"
        f" ({pattern.get('name', '?')}{current_txt}{took})"
    )


def _milling_event(
    record: LogRecord, d: Dict[str, Any], not_before: Optional[datetime] = None
) -> ReplayEvent:
    """``not_before``: when the task's previous stage finished, by the log's clock."""
    stage = d.get("stage") if isinstance(d.get("stage"), dict) else {}
    duration = None
    try:
        duration = float(d["end_time"]) - float(d["start_time"])
    except (KeyError, TypeError, ValueError):
        pass
    # Logged when the stage *finishes*. Placed at its start by the log's own
    # clock -- not by `start_time`, an epoch that would be read in this
    # machine's time zone rather than the instrument's.
    time = record.time - timedelta(seconds=duration) if duration else record.time
    # The duration is measured by a clock that can be coarser than the log's
    # (about 16 ms on Windows before Python 3.13): a stage milled inside one
    # tick measures 0 s, the next a whole tick. Worked back from its end, that
    # next stage would start before the one before it finished, which it
    # cannot have.
    if not_before is not None and time < not_before:
        time = not_before
    summary = _milling_summary(d.get("milling_task_name"), stage, duration)
    return ReplayEvent(
        time=time, kind=EventKind.MILLING, summary=summary, data=d, duration=duration
    )


def _spot_event(record: LogRecord) -> Optional[ReplayEvent]:
    m = _SPOT.search(record.message)
    if m is None:
        return None
    try:
        n = int(m.group(1))
        x, y, exposure, current = (float(m.group(i)) for i in range(2, 6))
    except ValueError:
        return None
    return ReplayEvent(
        time=record.time,
        kind=EventKind.MILLING,
        summary=f"Burn spot {n} ({exposure:g} s, {current * 1e9:.2f} nA)",
        data={"spot": (x, y), "exposure_time": exposure, "milling_current": current},
        duration=exposure,
    )


def _answer_summary(kind: Any, response: Any, by: Any, adjusted: bool) -> str:
    answer = {True: "Yes", False: "No"}.get(response, response)
    summary = f"{kind} answered {answer} by the {by}"
    if adjusted:
        summary += ", after adjusting it"
    return summary


def _task_summary(task: Any, step: Any) -> str:
    return f"{task or 'Task'} — {str(step).replace('_', ' ').title()}"


def _prompt_event(record: LogRecord, m: "re.Match", item, task, step) -> ReplayEvent:
    kind, response, by, adjusted = m.groups()
    summary = _answer_summary(kind, response == "True", by, adjusted == "True")
    return ReplayEvent(
        record.time,
        EventKind.PROMPT,
        summary,
        item,
        task,
        step,
        data={
            "prompt": kind,
            "response": response == "True",
            "answered_by": by,
            "adjusted": adjusted == "True",
        },
    )


def find_fluorescence_images(root: Path) -> List[Path]:
    """The fluorescence images an experiment holds: z-stacks and overviews.

    The tiles an overview was stitched from (``tile-*.ome.tiff``) are left
    out; the stitched overview stands for them.
    """
    return sorted(
        p for p in Path(root).rglob("*.ome.tiff") if not p.name.startswith("tile-")
    )


def _fluorescence_event(path: Path) -> Optional[ReplayEvent]:
    """An FM acquisition, placed by the time its own metadata records.

    Reads the OME header only, not the pixels. ``acquisition_date`` is when
    the acquisition started, on the instrument's clock -- the same clock as
    the log -- so it is read as naive local time, like the log's timestamps.
    """
    import json

    from fibsem.fm.structures import safe_ome_from_tiff

    try:
        ome = safe_ome_from_tiff(str(path))
    except Exception as e:
        logger.debug(f"Replay could not read the metadata of {path}: {e}")
        return None
    metadata: Dict[str, Any] = {}
    annotations = ome.structured_annotations
    for annotation in (annotations.map_annotations if annotations else None) or []:
        value = (annotation.value or {}).get("FluorescenceImageMetadata")
        if value:
            try:
                metadata = json.loads(value)
            except ValueError:
                pass
    started = _acquisition_time(metadata.get("acquisition_date"))
    if started is None and ome.images and ome.images[0].acquisition_date is not None:
        started = _acquisition_time(ome.images[0].acquisition_date.isoformat())
    if started is None:
        return None

    channels = [c.get("name", "?") for c in metadata.get("channels") or []]
    planes = len(metadata.get("z_positions") or []) or 1
    what = _fluorescence_what(channels, planes)
    return ReplayEvent(
        time=started,
        kind=EventKind.FLUORESCENCE,
        summary=f"{what} — {path.name}",
        data={"channels": channels, "planes": planes},
        image_path=path,
        saved=True,
        position=metadata.get("stage_position"),
    )


def _acquisition_time(value: Any) -> Optional[datetime]:
    """When an FM acquisition started, as its metadata records it."""
    try:
        started = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if started.tzinfo is not None:
        started = started.astimezone().replace(tzinfo=None)
    return started


def _fluorescence_what(channels: List[str], planes: int, overview: bool = False) -> str:
    if overview:
        what = "FM overview"
    else:
        what = f"FM z-stack, {planes} planes" if planes > 1 else "FM image"
    if channels:
        what += f", {', '.join(channels)}"
    return what


def _stamp_from_task_steps(
    events: List[ReplayEvent], kinds: Tuple[str, ...] = (EventKind.FLUORESCENCE,)
) -> None:
    """Give events placed from outside the log the task step they fell in.

    The same rule the log parse applies: a step's context holds from its
    status record until the task finishes.
    """
    context: Tuple[Any, Any, Any] = (None, None, None)
    for e in events:
        if e.kind == EventKind.TASK:
            finished = e.step in _TASK_ENDS
            context = (None, None, None) if finished else (e.item, e.task, e.step)
        elif e.kind in kinds and e.item is None:
            e.item, e.task, e.step = context


def _stamp_item_types(events: List[ReplayEvent]) -> None:
    """Say whether each event's item is a lamella or a grid.

    Only a task's status record knows: it names the item under the key of
    its kind (``lamella`` or ``grid``). Every event on the same item shares it.
    """
    types: Dict[str, str] = {}
    for e in events:
        if e.kind == EventKind.TASK and e.item:
            for kind in ("lamella", "grid"):
                if e.data.get(kind) == e.item:
                    types.setdefault(e.item, kind)
    for e in events:
        if e.item:
            e.item_type = types.get(e.item)


def load_replay(root: Path) -> ExperimentReplay:
    """Read the experiment at *root* into a replay timeline.

    From its ``events.jsonl`` when it has one, otherwise from its log. Raises
    FileNotFoundError when *root* has neither; anything inside them that cannot
    be read is skipped and counted, never raised.
    """
    root = Path(root)
    if (root / EVENTS_FILENAME).is_file():
        return _load_from_events(root)
    return _load_from_log(root)


def _load_from_log(root: Path) -> ExperimentReplay:
    logfile = root / LOGFILE_NAME
    if not logfile.is_file():
        raise FileNotFoundError(f"No {EVENTS_FILENAME} or {LOGFILE_NAME} in {root}")

    resolver = _ImageResolver(root)
    events: List[ReplayEvent] = []
    track: List[Tuple[datetime, Dict[str, Any]]] = []
    item = task = step = None
    read = unreadable = 0
    # Milling task id -> when its latest stage finished, by the log's clock.
    stage_finished: Dict[Any, datetime] = {}

    for record in read_log_records(logfile):
        if record.level in ("WARNING", "ERROR", "CRITICAL"):
            events.append(
                ReplayEvent(
                    record.time,
                    EventKind.MESSAGE,
                    f"{record.level.title()}: {record.message}",
                    item,
                    task,
                    step,
                    data={"level": record.level},
                )
            )
            continue
        if record.function == "_workflow_finished":
            item = task = step = None
            continue
        m = _PROMPT.match(record.message)
        if m is not None:
            events.append(_prompt_event(record, m, item, task, step))
            continue
        if record.function == "run_spot_burn":
            event = _spot_event(record)
            if event is not None:
                event.item, event.task, event.step = item, task, step
                events.append(event)
            continue

        m = _MSG_KEY.match(record.message)
        if m is None:
            continue
        msg = m.group(1)
        if msg not in _WANTED:
            continue
        d = parse_record(record.message)
        if d is None:
            unreadable += 1
            continue
        read += 1

        event: Optional[ReplayEvent] = None
        if msg == "status":
            item = d.get("lamella") or d.get("grid")
            task, step = d.get("task_name"), d.get("task_step")
            event = ReplayEvent(
                record.time, EventKind.TASK, _task_summary(task, step), data=d
            )
        elif msg == "get_stage_position":
            if isinstance(d.get("pos"), dict):
                track.append((record.time, d["pos"]))
        elif msg == "acquire_image":
            event = _image_event(record, d)
            if event is not None and isinstance(event.position, dict):
                track.append((record.time, event.position))
        elif msg == "move_stage_absolute":
            pos = d.get("position") or {}
            event = ReplayEvent(
                record.time,
                EventKind.STAGE,
                f"Move stage to {_describe_position(pos)}",
                data=d,
            )
        elif msg == "move_stage_relative":
            pos = d.get("position") or {}
            event = ReplayEvent(
                record.time,
                EventKind.STAGE,
                f"Move stage by {_describe_position(pos)}",
                data=d,
            )
        elif msg in ("stable_move", "vertical_move"):
            summary = _beam_move_summary(msg, d)
            event = ReplayEvent(record.time, EventKind.STAGE, summary, data=d)
        elif msg == "milling_task":
            task_id = d.get("milling_task_id")
            event = _milling_event(record, d, stage_finished.get(task_id))
            stage_finished[task_id] = record.time
        elif msg == "beam_shift":
            summary = _beam_shift_summary(d)
            event = ReplayEvent(record.time, EventKind.ALIGNMENT, summary, data=d)
        elif msg == "measure_coincidence":
            summary = _coincidence_summary(d)
            event = ReplayEvent(record.time, EventKind.ALIGNMENT, summary, data=d)

        if event is not None:
            event.item, event.task, event.step = item, task, step
            events.append(event)
        if msg == "status" and step == "FINISHED":
            item = task = step = None

    for path in find_fluorescence_images(root):
        event = _fluorescence_event(path)
        if event is not None:
            events.append(event)

    # Milling and fluorescence events are placed at their start, so the order
    # is by time. Stable: records logged in the same millisecond keep their
    # file order.
    events.sort(key=lambda e: e.time)
    _stamp_from_task_steps(events)
    _stamp_item_types(events)
    track.sort(key=lambda p: p[0])
    _attach_images(events, resolver)
    _attach_stage_results(events, track)
    return ExperimentReplay(
        root,
        events,
        track,
        records_read=read,
        records_unreadable=unreadable,
        source=LOGFILE_NAME,
    )


_WANTED = {
    "status",
    "get_stage_position",
    "acquire_image",
    "move_stage_absolute",
    "move_stage_relative",
    "stable_move",
    "vertical_move",
    "milling_task",
    "beam_shift",
    "measure_coincidence",
}


def _attach_images(events: List[ReplayEvent], resolver: _ImageResolver) -> None:
    """Point each saved acquisition at its file -- unless a later one overwrote it.

    Several acquisitions save to the same name (``ref_alignment`` is retaken by
    every task). Only the last write is on disk, so the earlier ones are shown
    as not saved rather than with pixels from later in the run.
    """
    last_writer: Dict[Path, ReplayEvent] = {}
    for e in events:
        if e.kind != EventKind.IMAGE or not e.saved:
            continue
        if e.data.get("path"):  # the event stream: the file actually written
            path = resolver.resolve_path(e.data["path"])
        else:  # the log: a directory and a stem, the suffix guessed
            settings = e._image_settings
            path = resolver.resolve(
                settings.get("path"), settings.get("filename"), e.beam
            )
        if path is not None:
            last_writer[path] = e
    for path, e in last_writer.items():
        e.image_path = path


def _attach_stage_results(
    events: List[ReplayEvent], track: List[Tuple[datetime, Dict[str, Any]]]
) -> None:
    """Give each move the position the stage reported after it.

    A relative or beam-frame move records only the step, so where the stage
    went comes from the next read-back. An absolute move with no read-back in
    time keeps its target.
    """
    times = [t for t, _ in track]
    for e in events:
        if e.kind != EventKind.STAGE:
            continue
        i = bisect.bisect_left(times, e.time)
        if i < len(track) and track[i][0] - e.time <= _MOVE_SETTLE:
            e.position = track[i][1]
        elif e.data.get("msg") == "move_stage_absolute":
            e.position = e.data.get("position")


# ── reading events.jsonl ─────────────────────────────────────────────────────

# The lifecycle events that end a task, as the step the timeline shows.
_TASK_END_STEPS = {
    "task_completed": "FINISHED",
    "task_failed": "FAILED",
    "task_cancelled": "CANCELLED",
    "task_skipped": "SKIPPED",
}
_SPOT_BURN_ENDS = ("finished", "cancelled", "failed")


def _record_time(record: Dict[str, Any]) -> Optional[datetime]:
    """When an event happened, on the instrument's wall clock.

    ``t`` carries its UTC offset. The offset is dropped rather than converted
    to this machine's zone, so the time reads as the log and the FM files
    record theirs: naive, on the instrument's clock.
    """
    try:
        return datetime.fromisoformat(record["t"]).replace(tzinfo=None)
    except (KeyError, TypeError, ValueError):
        return None


def _recorded_image(time: datetime, payload: Dict[str, Any]) -> ReplayEvent:
    beam = payload.get("beam_type")
    shape = payload.get("shape") or []
    data = dict(payload)
    if len(shape) >= 2:
        data["resolution"] = [shape[1], shape[0]]  # (width, height), as the log has it
    size = f"{shape[1]}×{shape[0]}" if len(shape) >= 2 else "?"
    summary = f"{_BEAM_LABEL.get(beam, str(beam))} image {size}, HFW {_um(payload.get('hfw'))} µm"
    path = payload.get("path")
    if path:
        summary += f" — {_pure_path(path).name}"
    return ReplayEvent(
        time=time,
        kind=EventKind.IMAGE,
        summary=summary,
        data=data,
        beam=beam,
        saved=bool(path),
        position=payload.get("stage_position"),
    )


def _failed(payload: Dict[str, Any]) -> str:
    return f" — failed: {payload['error']}" if payload.get("error") else ""


def _move_summary(payload: Dict[str, Any]) -> str:
    """A stage move, in the words the log's reader uses where it has them."""
    move = payload.get("move")
    request = payload.get("request") or {}
    if move == "move_stage_absolute":
        text = f"Move stage to {_describe_position(request.get('position') or {})}"
    elif move == "move_stage_relative":
        text = f"Move stage by {_describe_position(request.get('position') or {})}"
    elif move in ("stable_move", "vertical_move"):
        text = _beam_move_summary(move, request)
    elif move == "safe_absolute_stage_movement":
        # the parameter's name differs between the drivers
        target = request.get("stage_position") or request.get("position") or {}
        text = f"Move stage safely to {_describe_position(target)}"
    elif move == "move_to_orientation":
        text = f"Move to the {request.get('orientation')} orientation"
    elif move == "move_to_milling_angle":
        text = f"Move to a milling angle of {_deg(request.get('milling_angle'))}°"
    elif move == "move_to_device":
        text = f"Move to the {request.get('device')}"
        if request.get("orientation"):
            text += f", in the {request['orientation']} orientation"
    else:
        text = f"Stage move ({move})"
    return text + _failed(payload)


def _alignment_summary(payload: Dict[str, Any]) -> str:
    steps = payload.get("results") or []
    subsystem = str(payload.get("subsystem") or "?").replace("-", " ")
    text = (
        f"{_beam(payload.get('beam_type'))} alignment by {subsystem}, "
        f"{len(steps)} step{'' if len(steps) == 1 else 's'}"
    )
    if steps and isinstance(steps[-1], dict):
        shift = steps[-1].get("shift") or {}
        text += f", last shift dx={_shift(shift.get('x'))}, dy={_shift(shift.get('y'))}"
    validation = payload.get("validation")
    if isinstance(validation, dict):
        if validation.get("agreement"):
            text += " — the methods agree"
        else:
            try:
                apart = f" by {float(validation['max_disagreement_px']):.0f} px"
            except (KeyError, TypeError, ValueError):
                apart = ""
            text += f" — the methods disagree{apart}"
    if payload.get("aborted"):
        text += " (stopped)"
    return text


# How many of an edit's changed values its row names; the rest are counted.
_EDIT_CHANGES_SHOWN = 3
_ABSENT = object()  # a value one side of an edit does not have


def _edit_summary(payload: Dict[str, Any], actor: Any) -> str:
    """What an edit changed, from what to what, and who made it from where.

    ``before`` and ``after`` are the whole object edited, so the values that
    differ are found by walking both.
    """
    changes = _changed_values(payload.get("before"), payload.get("after"))
    shown = [
        f"{path} {_edit_value(old)} → {_edit_value(new)}".lstrip()
        for path, old, new in changes[:_EDIT_CHANGES_SHOWN]
    ]
    if len(changes) > _EDIT_CHANGES_SHOWN:
        shown.append(f"{len(changes) - _EDIT_CHANGES_SHOWN} more")
    text = f"{payload.get('target')}: {', '.join(shown) or 'changed'}"
    if actor:
        text += f" — by the {actor}"
    if payload.get("via"):
        text += f" ({payload['via']})"
    return text


def _changed_values(
    before: Any, after: Any, path: str = ""
) -> List[Tuple[str, Any, Any]]:
    """``(path, before, after)`` for each value that differs, in order."""
    if isinstance(before, dict) and isinstance(after, dict):
        keys = list(before) + [k for k in after if k not in before]
        pairs = [(k, before.get(k, _ABSENT), after.get(k, _ABSENT)) for k in keys]
    elif isinstance(before, list) and isinstance(after, list):
        pairs = [
            (
                i,
                before[i] if i < len(before) else _ABSENT,
                after[i] if i < len(after) else _ABSENT,
            )
            for i in range(max(len(before), len(after)))
        ]
    else:
        return [] if before == after else [(path, before, after)]
    return [
        change
        for key, old, new in pairs
        for change in _changed_values(old, new, f"{path}.{key}" if path else str(key))
    ]


def _correlation_summary(payload: Dict[str, Any], actor: Any) -> str:
    """An accepted correlation: the point of interest it gave, and how well it fits."""
    poi = payload.get("poi") or {}
    text = (
        f"Correlation: point of interest x={_um(poi.get('x'))} µm, "
        f"y={_um(poi.get('y'))} µm"
    )
    fit = []
    nm, px = (
        _number(payload.get("rms_nm"), ".0f"),
        _number(payload.get("rms_px"), ".1f"),
    )
    if nm or px:
        rms = f"RMS {nm} nm" if nm else f"RMS {px} px"
        if payload.get("fiducials"):
            rms += f" over {payload['fiducials']} fiducials"
        fit.append(rms)
    if payload.get("verdict"):
        fit.append(f"{payload['verdict']} fit")
    elif payload.get("seeded") is False:
        fit.append("unseeded")
    ri = payload.get("refractive_index") or {}
    factor = _number(ri.get("factor"), ".2f")
    if factor:
        when = "before" if ri.get("mode") == "pre" else "after"
        fit.append(f"refractive index ×{factor} {when} the fit")
    if fit:
        text += " — " + ", ".join(fit)
    if actor:
        text += f" — by the {actor}"
    return text


def _number(value: Any, spec: str) -> Optional[str]:
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return None


def _edit_value(value: Any) -> str:
    if value is _ABSENT:
        return "(none)"
    if isinstance(value, float):
        return f"{value:.4g}"
    text = str(value)
    return text if len(text) <= 32 else "…" + text[-31:]


def _recorded_fluorescence(
    time: datetime, payload: Dict[str, Any], resolver: _ImageResolver
) -> ReplayEvent:
    """An FM acquisition as recorded: its file, and when it started."""
    channels = [
        c.get("name", "?") for c in payload.get("channels") or [] if isinstance(c, dict)
    ]
    planes = len(payload.get("z_positions") or []) or 1
    what = _fluorescence_what(channels, planes, bool(payload.get("overview")))
    recorded = payload.get("path")
    if recorded:
        what += f" — {_pure_path(recorded).name}"
    return ReplayEvent(
        time=_acquisition_time(payload.get("acquired_at")) or time,
        kind=EventKind.FLUORESCENCE,
        summary=what,
        data={
            "channels": channels,
            "planes": planes,
            "overview": payload.get("overview"),
            "path": recorded,
        },
        image_path=resolver.resolve_path(recorded) if recorded else None,
        saved=bool(recorded),
        position=payload.get("stage_position"),
    )


def _spot_events(burn: Dict[str, Any], burned: int) -> List[ReplayEvent]:
    """One event per spot burned, placed at when its exposure started.

    The burn records its points once, at the start, and each runs for the
    exposure time in order; a cancelled burn stops at the point it was on.
    """
    exposure = burn["exposure_time"] or 0.0
    current = burn["milling_current"]
    current_txt = f", {current * 1e9:.2f} nA" if current else ""
    events = []
    for i, (x, y) in enumerate(burn["coordinates"][:burned]):
        events.append(
            ReplayEvent(
                burn["time"] + timedelta(seconds=i * exposure),
                EventKind.MILLING,
                f"Burn spot {i + 1} ({exposure:g} s{current_txt})",
                burn["item"],
                burn["task"],
                burn["step"],
                data={
                    "spot": (x, y),
                    "field_of_view": burn["field_of_view"],
                    "exposure_time": exposure,
                    "milling_current": current,
                },
                duration=exposure,
            )
        )
    return events


def _load_from_events(root: Path) -> ExperimentReplay:
    """The timeline from ``events.jsonl``.

    Each event names its lamella or grid and task, so nothing is inferred from
    order; an image names the file it was saved to. Warnings and errors are
    human messages and stay in the log, so they are read from there, for the
    span the events cover.

    A stage move is one row however many moves it was made of, and shows where
    it ended; a position read is only the stage track. An edit to a lamella's
    plan, and an accepted correlation, are on the lamella and task they were
    about. Live view is not recorded. An
    FM file the stream did not record is found on disk and placed by its own
    metadata, as the log's reader places every FM image.
    """
    path = root / EVENTS_FILENAME
    records = list(read_events(path))
    with open(path, encoding="utf-8") as f:
        lines = sum(1 for line in f if line.strip())

    resolver = _ImageResolver(root)
    events: List[ReplayEvent] = []
    track: List[Tuple[datetime, Dict[str, Any]]] = []
    steps: Dict[Any, Any] = {}  # task id -> the step it is on
    item_types: Dict[str, str] = {}
    mills: Dict[Tuple[Any, Any], ReplayEvent] = {}  # (task id, stage) -> started
    # FM acquisitions as recorded. They carry their own item and task, so they
    # are kept out of the stamping by time below.
    recorded_fm: List[ReplayEvent] = []
    burn: Optional[Dict[str, Any]] = None
    burned = 0

    for record in records:
        time = _record_time(record)
        if time is None:
            continue
        kind = record.get("kind")
        payload = record.get("payload") or {}
        item = (record.get("item") or {}).get("name")
        task_ref = record.get("task") or {}
        task, task_id = task_ref.get("name"), task_ref.get("id")
        event: Optional[ReplayEvent] = None

        if kind == "task_started" or kind in _TASK_END_STEPS or kind == "task_step":
            if kind == "task_step":
                step = payload.get("step")
                if item and payload.get("item_type"):
                    item_types.setdefault(item, payload["item_type"])
            else:
                step = _TASK_END_STEPS.get(kind, "STARTED")
            summary = _task_summary(task, step)
            if kind == "task_failed" and payload.get("error"):
                summary += f": {payload['error']}"
            event = ReplayEvent(time, EventKind.TASK, summary, data=payload)
            steps[task_id] = step
        elif kind == "image_acquired":
            event = _recorded_image(time, payload)
            if isinstance(event.position, dict):
                track.append((time, event.position))
        elif kind == "stage_position_changed":
            # A read, not a move: the track the scene's stage position comes
            # from, as the log's reader treats its read-backs.
            position = payload.get("position")
            if isinstance(position, dict):
                track.append((time, position))
        elif kind == "stage_moved":
            end = payload.get("end") if isinstance(payload.get("end"), dict) else None
            if end is not None:
                track.append((time, end))
            event = ReplayEvent(
                time,
                EventKind.STAGE,
                _move_summary(payload),
                data=payload,
                position=end,
            )
        elif kind == "beam_shifted":
            summary = _beam_shift_summary(payload) + _failed(payload)
            event = ReplayEvent(time, EventKind.ALIGNMENT, summary, data=payload)
        elif kind == "coincidence_measured":
            summary = _coincidence_summary(payload)
            event = ReplayEvent(time, EventKind.ALIGNMENT, summary, data=payload)
        elif kind == "alignment":
            summary = _alignment_summary(payload)
            event = ReplayEvent(time, EventKind.ALIGNMENT, summary, data=payload)
        elif kind == "autofocus":
            summary = (
                f"{_beam(payload.get('beam_type'))} autofocus: working distance "
                f"{_mm(payload.get('initial_working_distance'))} → "
                f"{_mm(payload.get('working_distance'))} mm"
            )
            event = ReplayEvent(time, EventKind.ALIGNMENT, summary, data=payload)
        elif kind == "fm_image_acquired":
            fm = _recorded_fluorescence(time, payload, resolver)
            fm.item, fm.task = item, task
            fm.step = steps.get(task_id) if task_id is not None else None
            recorded_fm.append(fm)
        elif kind == "fm_autofocus":
            summary = (
                f"FM autofocus: objective {_um(payload.get('initial_position'))} → "
                f"{_um(payload.get('position'))} µm"
            )
            event = ReplayEvent(time, EventKind.FLUORESCENCE, summary, data=payload)
        elif kind == "objective_state_changed":
            state = str(payload.get("state") or "?").lower()
            event = ReplayEvent(
                time, EventKind.FLUORESCENCE, f"FM objective {state}", data=payload
            )
        elif kind == "milling_stage_started":
            stage = payload.get("stage") or {}
            event = ReplayEvent(
                time,
                EventKind.MILLING,
                _milling_summary(payload.get("task_name"), stage, None),
                data={
                    "stage": stage,
                    "milling_task_id": payload.get("task_id"),
                    "milling_task_name": payload.get("task_name"),
                },
            )
            mills[(payload.get("task_id"), stage.get("name"))] = event
        elif kind == "milling_progress" and payload.get("status") == "stage-finished":
            started = mills.pop(
                (payload.get("task_id"), payload.get("stage_name")), None
            )
            if started is not None:
                started.duration = (time - started.time).total_seconds()
                started.summary = _milling_summary(
                    started.data["milling_task_name"],
                    started.data["stage"],
                    started.duration,
                )
        elif kind == "spot_burn_started":
            if burn is not None:  # a burn that never reported its end
                events.extend(_spot_events(burn, burned))
            burn = {
                "time": time,
                "item": item,
                "task": task,
                "step": steps.get(task_id),
                "coordinates": [tuple(p) for p in payload.get("coordinates") or []],
                "field_of_view": payload.get("field_of_view"),
                "exposure_time": payload.get("exposure_time"),
                "milling_current": payload.get("milling_current"),
            }
            burned = 0
        elif kind == "spot_burn_progress" and burn is not None:
            status = payload.get("status")
            if status == "burning":
                burned = max(burned, payload.get("current_point") or 0)
            elif status in _SPOT_BURN_ENDS:
                if status == "finished":
                    burned = len(burn["coordinates"])
                events.extend(_spot_events(burn, burned))
                burn = None
        elif kind in ("edit", "correlation"):
            # On the lamella (and task) it was about, which need not be the
            # ones a workflow was running when it was made.
            item = (payload.get("item") or {}).get("name")
            task, task_id = payload.get("task"), None
            actor = record.get("actor")
            if kind == "edit":
                summary = _edit_summary(payload, actor)
                event = ReplayEvent(time, EventKind.EDIT, summary, data=payload)
            else:
                summary = _correlation_summary(payload, actor)
                event = ReplayEvent(time, EventKind.CORRELATION, summary, data=payload)
        elif kind in ("prompt_raised", "prompt_answered", "prompt_cancelled"):
            prompt = payload.get("type", "Prompt")
            if kind == "prompt_answered":
                summary = _answer_summary(
                    prompt,
                    payload.get("response"),
                    payload.get("answered_by"),
                    bool(payload.get("adjusted")),
                )
            elif kind == "prompt_raised":
                message = payload.get("message")
                summary = f"{prompt} asked: {message}" if message else f"{prompt} asked"
            else:
                summary = f"{prompt} withdrawn"
            event = ReplayEvent(time, EventKind.PROMPT, summary, data=payload)

        if event is not None:
            event.item, event.task = item, task
            event.step = steps.get(task_id) if task_id is not None else None
            events.append(event)
        if kind in _TASK_END_STEPS:
            steps.pop(task_id, None)

    if burn is not None:
        events.extend(_spot_events(burn, burned))

    start = min((e.time for e in events), default=None)
    logfile = root / LOGFILE_NAME
    if start is not None and logfile.is_file():
        for line in read_log_records(logfile):
            if line.level in ("WARNING", "ERROR", "CRITICAL") and line.time >= start:
                events.append(
                    ReplayEvent(
                        line.time,
                        EventKind.MESSAGE,
                        f"{line.level.title()}: {line.message}",
                        data={"level": line.level},
                    )
                )

    # The files the stream did not record -- an FM image saved outside the
    # acquisition functions -- are still found, and placed as the log's
    # reader places them. A file recorded twice shows only for its last write.
    written: Dict[Path, ReplayEvent] = {}
    for fm in recorded_fm:
        if fm.image_path is not None:
            earlier = written.get(fm.image_path.resolve())
            if earlier is not None:
                earlier.image_path = None
            written[fm.image_path.resolve()] = fm
    for fm_path in find_fluorescence_images(root):
        if fm_path.resolve() in written:
            continue
        event = _fluorescence_event(fm_path)
        if event is not None:
            events.append(event)

    events.sort(key=lambda e: e.time)
    _stamp_from_task_steps(events, (EventKind.FLUORESCENCE, EventKind.MESSAGE))
    events = sorted(events + recorded_fm, key=lambda e: e.time)
    for e in events:
        if e.item:
            e.item_type = item_types.get(e.item)
    track.sort(key=lambda p: p[0])
    _attach_images(events, resolver)
    return ExperimentReplay(
        root,
        events,
        track,
        records_read=len(records),
        records_unreadable=lines - len(records),
        source=EVENTS_FILENAME,
    )
