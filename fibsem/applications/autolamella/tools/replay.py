"""Read an experiment's recorded actions back as a timeline, for replay.

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
stream exists only at DEBUG level; a machine-readable event file (FIB-455)
would replace :func:`read_log_records` and leave the rest unchanged.
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


class EventKind:
    """What an event records. Plain strings, so a filter can be a set of them."""

    TASK = "task"
    PROMPT = "prompt"
    IMAGE = "image"
    FLUORESCENCE = "fluorescence"
    STAGE = "stage"
    MILLING = "milling"
    ALIGNMENT = "alignment"
    MESSAGE = "message"

    ALL = (TASK, PROMPT, IMAGE, FLUORESCENCE, STAGE, MILLING, ALIGNMENT, MESSAGE)


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


def _path_parts(recorded: str) -> Tuple[str, ...]:
    if "\\" in recorded or re.match(r"^[A-Za-z]:", recorded):
        return PureWindowsPath(recorded).parts
    return PurePosixPath(recorded).parts


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
    def field_size(self) -> Optional[Tuple[float, float]]:
        """The (width, height) in metres an image covers, if recorded."""
        settings = (self.data.get("metadata") or {}).get("image") or {}
        try:
            hfw = float(settings["hfw"])
            w, h = (float(v) for v in settings["resolution"])
        except (KeyError, TypeError, ValueError):
            return None
        return (hfw, hfw * h / w) if w else None

    @property
    def is_full_frame(self) -> bool:
        """An image of the beam's whole field, not a reduced-area crop."""
        settings = (self.data.get("metadata") or {}).get("image") or {}
        return not settings.get("reduced_area")


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
                    spots.append(((spot[0] - 0.5) * width, (spot[1] - 0.5) * height))

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


def _milling_event(record: LogRecord, d: Dict[str, Any]) -> ReplayEvent:
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
    milling = stage.get("milling") or {}
    pattern = stage.get("pattern") or {}
    current = milling.get("milling_current")
    current_txt = (
        f", {float(current) * 1e9:.2f} nA" if isinstance(current, (int, float)) else ""
    )
    took = f", {duration:.0f} s" if duration else ""
    summary = (
        f"Mill {d.get('milling_task_name', '?')} / {stage.get('name', '?')}"
        f" ({pattern.get('name', '?')}{current_txt}{took})"
    )
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


def _prompt_event(record: LogRecord, m: "re.Match", item, task, step) -> ReplayEvent:
    kind, response, by, adjusted = m.groups()
    answer = {"True": "Yes", "False": "No"}.get(response, response)
    summary = f"{kind} answered {answer} by the {by}"
    if adjusted == "True":
        summary += ", after adjusting it"
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
    started: Optional[datetime] = None
    try:
        started = datetime.fromisoformat(str(metadata["acquisition_date"]))
    except (KeyError, ValueError):
        if ome.images and ome.images[0].acquisition_date is not None:
            started = ome.images[0].acquisition_date
    if started is None:
        return None
    if started.tzinfo is not None:
        started = started.astimezone().replace(tzinfo=None)

    channels = [c.get("name", "?") for c in metadata.get("channels") or []]
    planes = len(metadata.get("z_positions") or []) or 1
    what = f"FM z-stack, {planes} planes" if planes > 1 else "FM image"
    if channels:
        what += f", {', '.join(channels)}"
    return ReplayEvent(
        time=started,
        kind=EventKind.FLUORESCENCE,
        summary=f"{what} — {path.name}",
        data={"channels": channels, "planes": planes},
        image_path=path,
        saved=True,
        position=metadata.get("stage_position"),
    )


def _stamp_from_task_steps(events: List[ReplayEvent]) -> None:
    """Give events placed from outside the log the task step they fell in.

    The same rule the log parse applies: a step's context holds from its
    status record until the task finishes.
    """
    context: Tuple[Any, Any, Any] = (None, None, None)
    for e in events:
        if e.kind == EventKind.TASK:
            finished = e.step == "FINISHED"
            context = (None, None, None) if finished else (e.item, e.task, e.step)
        elif e.kind == EventKind.FLUORESCENCE and e.item is None:
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

    Raises FileNotFoundError when *root* has no logfile; anything inside the
    log that cannot be read is skipped and counted, never raised.
    """
    root = Path(root)
    logfile = root / LOGFILE_NAME
    if not logfile.is_file():
        raise FileNotFoundError(f"No {LOGFILE_NAME} in {root}")

    resolver = _ImageResolver(root)
    events: List[ReplayEvent] = []
    track: List[Tuple[datetime, Dict[str, Any]]] = []
    item = task = step = None
    read = unreadable = 0

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
                record.time,
                EventKind.TASK,
                f"{task or 'Task'} — {str(step).replace('_', ' ').title()}",
                data=d,
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
            words = ["Stable move" if msg == "stable_move" else "Vertical move"]
            if d.get("beam_type") in _BEAM_LABEL:
                words.append(f"in the {_BEAM_LABEL[d['beam_type']]}")
            words.append(f"dx={_um(d.get('dx'))} µm, dy={_um(d.get('dy'))} µm")
            event = ReplayEvent(record.time, EventKind.STAGE, " ".join(words), data=d)
        elif msg == "milling_task":
            event = _milling_event(record, d)
        elif msg == "beam_shift":
            beam = _BEAM_LABEL.get(d.get("beam_type"), str(d.get("beam_type")))
            event = ReplayEvent(
                record.time,
                EventKind.ALIGNMENT,
                f"{beam} beam shift dx={_um(d.get('dx'))} µm, dy={_um(d.get('dy'))} µm",
                data=d,
            )
        elif msg == "measure_coincidence":
            ok = (
                "reliable"
                if d.get("is_reliable")
                else f"refused ({d.get('refusal_reason')})"
            )
            event = ReplayEvent(
                record.time,
                EventKind.ALIGNMENT,
                f"Coincidence measured dx={_um(d.get('dx'))} µm, "
                f"dy={_um(d.get('dy'))} µm — {ok}",
                data=d,
            )

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
        root, events, track, records_read=read, records_unreadable=unreadable
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
        settings = (e.data.get("metadata") or {}).get("image") or {}
        path = resolver.resolve(settings.get("path"), settings.get("filename"), e.beam)
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
