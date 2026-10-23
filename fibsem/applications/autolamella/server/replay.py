"""The experiment replay, as the web page reads it: the timeline of recorded
actions, and the images and overlays at any one of them
(``tools/replay.py`` builds both)."""

import logging
import threading
from pathlib import Path
from typing import Any, List, Optional, Tuple

from fibsem.applications.autolamella.tools.replay import (
    LOGFILE_NAME,
    ExperimentReplay,
    ReplayEvent,
    ReplayScene,
    load_replay,
    proposal_marks,
)
from fibsem.server.responses import (
    ReplayImage,
    ReplayMillingStage,
    ReplayRow,
    ReplayTimeline,
    ReplayView,
)
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

    def timeline(self, root: Path) -> ReplayTimeline:
        try:
            replay, rows = self._load(root)
        except FileNotFoundError:
            return ReplayTimeline(
                available=False,
                error="This experiment has no events.jsonl or log to replay.",
            )
        return ReplayTimeline(
            available=True,
            source=replay.source,
            start=_iso(replay.start),
            end=_iso(replay.end),
            counts=replay.counts(),
            events=rows,
        )

    def view(self, root: Path, index: int, item: Optional[str] = None):
        try:
            replay, rows = self._load(root)
        except FileNotFoundError:
            return ReplayView(
                available=False,
                error="This experiment has no events.jsonl or log to replay.",
            )
        if not 0 <= index < len(replay.events):
            return {
                "error_type": "not_found",
                "error": f"No event {index}: the replay has {len(replay.events)}.",
            }
        scene = replay.scene(index, item)
        return ReplayView(
            available=True,
            index=index,
            event=rows[index],
            sem=_image(root, scene, scene.sem, scene.sem_unsaved_since),
            fib=_image(root, scene, scene.fib, scene.fib_unsaved_since),
            fm=_relative(root, scene.fm.image_path) if scene.fm else None,
            stage_position=scene.stage_position,
        )


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
