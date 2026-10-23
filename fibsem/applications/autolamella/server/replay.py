"""The experiment replay, as the web page reads it: the timeline of recorded
actions, and the images and overlays at any one of them
(``tools/replay.py`` builds both)."""

import logging
import math
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from fibsem.applications.autolamella.tools.replay import (
    LOGFILE_NAME,
    EventKind,
    ExperimentReplay,
    ReplayEvent,
    ReplayScene,
    load_replay,
    proposal_marks,
)
from fibsem.server.responses import (
    Refusal,
    ReplayImage,
    ReplayMillingStage,
    ReplayRow,
    ReplayStage,
    ReplayTimeline,
    ReplayView,
    refused,
)

NOTHING_TO_REPLAY = "This experiment has no events.jsonl or log to replay."
from fibsem.structures import FibsemStagePosition
from fibsem.workflows.event_recording import EVENTS_FILENAME

logger = logging.getLogger(__name__)


class ReplayReader:
    """One experiment's replay, read again when its record changes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key: Optional[Tuple[Any, ...]] = None
        self._replay: Optional[ExperimentReplay] = None
        self._rows: List[ReplayRow] = []

    def _load(self, root: Path) -> Tuple[ExperimentReplay, List[ReplayRow]]:
        key: Tuple[Any, ...] = (str(root),)
        for name in (EVENTS_FILENAME, LOGFILE_NAME):
            try:
                stat = (root / name).stat()
                key += (name, stat.st_mtime_ns, stat.st_size)
            except OSError:
                key += (name, None)
        with self._lock:
            if key != self._key:
                replay = load_replay(root)
                self._rows = [_row(i, e) for i, e in enumerate(replay.events)]
                self._replay, self._key = replay, key
            return self._replay, self._rows

    def timeline(self, root: Path) -> Union[ReplayTimeline, Refusal]:
        try:
            replay, rows = self._load(root)
        except FileNotFoundError:
            return refused("not_found", NOTHING_TO_REPLAY)
        return ReplayTimeline(
            source=replay.source,
            start=_iso(replay.start),
            end=_iso(replay.end),
            counts=replay.counts(),
            events=rows,
        )

    def view(
        self,
        root: Path,
        index: int,
        item: Optional[str] = None,
        microscope=None,
        positions: Sequence[FibsemStagePosition] = (),
    ):
        """The scene at *index*; with *microscope*, the stage drawn from its
        configuration, with *positions* (the lamellae) on its map."""
        try:
            replay, rows = self._load(root)
        except FileNotFoundError:
            return refused("not_found", NOTHING_TO_REPLAY)
        if not 0 <= index < len(replay.events):
            return refused(
                "not_found", f"No event {index}: the replay has {len(replay.events)}."
            )
        scene = replay.scene(index, item)
        return ReplayView(
            index=index,
            event=rows[index],
            sem=_image(root, scene, scene.sem, scene.sem_unsaved_since),
            fib=_image(root, scene, scene.fib, scene.fib_unsaved_since),
            fm=_relative(root, scene.fm.image_path) if scene.fm else None,
            stage_position=scene.stage_position,
            stage=_stage(microscope, scene.stage_position, replay, index, positions),
        )


def _stage(
    microscope,
    recorded: Optional[Dict[str, Any]],
    replay: ExperimentReplay,
    index: int,
    positions: Sequence[FibsemStagePosition],
) -> Optional[ReplayStage]:
    """The stage at a recorded position, as the Qt chamber view draws it
    (``QuadView._update_chamber_view``); None without a microscope or a position."""
    from fibsem.geometry.stage_places import is_half_turn, stage_map

    if microscope is None or not recorded:
        return None
    position = FibsemStagePosition(
        **{k: recorded.get(k) for k in ("x", "y", "z", "r", "t")},
        name=recorded.get("name") or "",
    )
    out = ReplayStage()
    try:
        stage = microscope.system.stage
        out.pre_tilt = stage.shuttle_pre_tilt
        out.column_tilt = microscope.system.ion.column_tilt
        if position.r is not None:
            out.mirrored = is_half_turn(position.r, stage.rotation_reference)
        out.objective_below = microscope.fm is not None and microscope._fm_is_a_pose()
    except Exception:
        logger.debug("Replay could not read the stage's geometry", exc_info=True)
    if position.t is not None:
        out.stage_tilt = math.degrees(position.t)
    if None not in (position.r, position.t):
        try:
            orientation = microscope.get_stage_orientation(position)
            out.orientation = None if orientation == "NONE" else orientation
            out.milling_angle = microscope.get_current_milling_angle(position)
        except Exception:
            logger.debug("Replay could not place the stage's pose", exc_info=True)
    out.objective_inserted = _objective_inserted(replay, index)
    if None not in (position.x, position.y):
        out.map = stage_map(microscope, position, positions)
    loaded = _grid_loaded(replay, index)
    if loaded is not None:
        out.grid = loaded or None
    elif out.map is not None:
        out.grid = out.map.slot
    return out


def _grid_loaded(replay: ExperimentReplay, index: int) -> Optional[str]:
    """The grid the last recorded load at or before *index* put on the stage; ""
    after an unload; None when no load is recorded before it."""
    for event in reversed(replay.events[: index + 1]):
        msg = event.data.get("msg") if event.kind == EventKind.STAGE else None
        if msg == "grid_loaded" and not event.data.get("error"):
            return event.data.get("grid") or ""
        if msg == "grid_unloaded" and not event.data.get("error"):
            return ""
    return None


def _objective_inserted(replay: ExperimentReplay, index: int) -> bool:
    """Whether the objective was last recorded inserted, at or before *index*."""
    for event in reversed(replay.events[: index + 1]):
        if event.kind != EventKind.FLUORESCENCE:
            continue
        state = str(event.data.get("state") or "").lower()
        if state in ("inserted", "retracted"):
            return state == "inserted"
    return False


def _iso(time) -> Optional[str]:
    return time.isoformat() if time is not None else None


def _row(index: int, event: ReplayEvent) -> ReplayRow:
    return ReplayRow(
        index=index,
        time=event.time.isoformat(),
        kind=event.kind,
        summary=event.summary,
        item=event.item,
        item_type=event.item_type,
        task=event.task,
        step=event.step,
        actor=event.actor,
        duration=event.duration,
    )


def _relative(root: Path, path: Optional[Path]) -> Optional[str]:
    """*path* relative to the experiment folder; None outside it, where the
    files route can't read it."""
    if path is None:
        return None
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def _image(
    root: Path, scene: ReplayScene, shown: Optional[ReplayEvent], unsaved: int
) -> Optional[ReplayImage]:
    path = _relative(root, shown.image_path) if shown is not None else None
    if path is None:
        return None
    image = ReplayImage(
        path=path, time=shown.time.isoformat(), item=shown.item, unsaved_since=unsaved
    )
    field_size = shown.field_size
    if shown is scene.fib and field_size is not None:
        width, height = field_size
        image.milling = _milling(scene, width, height)
        image.spots = [[0.5 + x / width, 0.5 + y / height] for x, y in scene.spots]
    if scene.proposal is not None and scene.proposal.shown_on is shown:
        _proposal(image, scene.proposal, shown)
    return image


def _milling(scene: ReplayScene, width: float, height: float):
    """The milling stages' pattern outlines as fractions of a field
    *width* by *height* metres."""
    from fibsem.milling.base import FibsemMillingStage
    from fibsem.milling.patterning.utils import pattern_outline

    if scene.milling is None:
        return []
    current = (scene.milling.data.get("stage") or {}).get("name")
    stages = []
    for d in scene.milling_stages:
        try:
            stage = FibsemMillingStage.from_dict(d)
            patterns = stage.define_patterns()
        except Exception as e:  # noqa: BLE001 - a stage the record can't rebuild isn't drawn
            logger.debug(f"Replay could not rebuild milling stage {d.get('name')}: {e}")
            continue
        outlines = [
            [[0.5 + x / width, 0.5 - y / height] for x, y in ring]
            for pattern in patterns
            for ring in pattern_outline(pattern)
        ]
        stages.append(
            ReplayMillingStage(
                name=stage.name, current=stage.name == current, outlines=outlines
            )
        )
    return stages


def _proposal(image: ReplayImage, proposal: ReplayEvent, shown: ReplayEvent) -> None:
    """A question's or decision's values (``proposal_marks``) on the image they
    sit on, as fractions of it."""
    marks = proposal_marks(proposal.data)
    if marks is None:
        return
    unit, proposed, decided = marks
    if unit == "rect":
        image.areas_proposed = [list(r) for r in proposed]
        image.areas_decided = [list(r) for r in decided]
        return
    if unit == "m":
        size = shown.field_size
        if size is None:
            return
        width, height = size

        def place(p):
            return [0.5 + p[0] / width, 0.5 - p[1] / height]

    else:
        if shown.resolution is None:
            return
        width, height = shown.resolution

        def place(p):
            return [p[0] / width, p[1] / height]

    image.proposed = [place(p) for p in proposed]
    image.decided = [place(p) for p in decided]
