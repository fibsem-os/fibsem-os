"""AppContext: what every app's context serves the /app core reads from.

An app's context is the facade an embedded server sees the app through: it
resolves the session's experiment, runner and microscope at call time, on
every call, and returns plain JSON-able data, never live objects or Qt. Reads
are point-in-time snapshots, safe from a server thread. An app subclasses it
and adds what its own experiment and protocol need.
"""

import logging
import os
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from fibsem.acting import OPERATOR, WORKFLOW, Actor, current_actor, signer
from fibsem.codec import decode, encode
from fibsem.config import get_recent_experiments
from fibsem.geometry.saved_images import image_point_to_stage_position, stage_projector
from fibsem.imaging.cells import from_crop
from fibsem.imaging.marks import (
    drawn_pair,
    drawn_path,
    drawn_picture,
    image_stem,
    save_drawn,
)
from fibsem.server import log_tail
from fibsem.server.files import FileRefusal, resolve
from fibsem.server.image_payloads import image_info, image_payload
from fibsem.server.responses import (
    AppStatus,
    Asking,
    DisplayImages,
    ExperimentRef,
    ExperimentSummary,
    HistoryRow,
    ImageDrawn,
    ImagePairDrawn,
    ImagePayload,
    ItemDetail,
    ItemStack,
    PixelPoint,
    Queue,
    QueueEntry,
    Refusal,
    RunFailed,
    SavedImageMetadata,
    SavedImagePixels,
    SavedImagePosition,
    SummaryRow,
    TaskHistory,
    TaskOutputs,
    WorkflowState,
    no_experiment,
    refusal_of,
    refused,
)
from fibsem.structures.image import FibsemImage
from fibsem.structures.settings import BeamType, FibsemStagePosition
from fibsem.util.records import dataframe_records
from fibsem.util.timestamps import to_iso, wall_time_of
from fibsem.workflows import workflow_edits
from fibsem.workflows.actions import NOTE, note_record
from fibsem.workflows.event_recording import (
    EVENTS_FILENAME,
    LIVE_KINDS,
    read_events_before,
    read_new_events,
)
from fibsem.workflows.events import buffer_or_unavailable, iso_time, to_plain
from fibsem.workflows.evidence import (
    EVIDENCE,
    EvidenceRefused,
    checked_marks,
    frame_path,
)
from fibsem.workflows.experiment import EXPERIMENT_FILENAME
from fibsem.workflows.experiment import find_item as find_experiment_item
from fibsem.workflows.experiments import (
    DEFAULT_MAX_DEPTH,
    default_experiment_directory,
    discover_experiments,
    filter_experiments,
)
from fibsem.workflows.history_rows import (
    HISTORY_KINDS,
    HistoryFilter,
    filter_runs,
    history_rows,
    run_record,
)
from fibsem.workflows.item import item_poses
from fibsem.workflows.outputs import files_by_run, recorded_outputs, stack_slices
from fibsem.workflows.progress_line import ProgressLine, ProgressLines
from fibsem.workflows.proposals import (
    Decision,
    DecisionOutcome,
    Suggestion,
    ValueRefused,
    _decode_values,
    patch_values,
)
from fibsem.workflows.protocol_files import (
    adopt_protocol,
    list_protocols,
    protocol_file_for,
    record_recent,
)
from fibsem.workflows.reference_images import load_reference_image, preview_of
from fibsem.workflows.runner import ITEMS, RunRequest, stop_message
from fibsem.workflows.session import CONFIG, PROTOCOL
from fibsem.workflows.status import TaskStatus
from fibsem.workflows.task_configs import agent_patch_task_config, config_version
from fibsem.workflows.workflow_edits import set_supervision

__all__ = [
    "AppContext",
]


def decision_answer(result) -> Any:
    """A ``DecisionResult`` as a route answers it: its dict when it applied,
    else its refusal (a task running under it is ``task_running``)."""
    if result.applied:
        return result.to_dict()
    error_type = result.error_type or (
        "task_running" if result.running else "not_pending"
    )
    return refused(error_type, result.reason)


def find_item(experiment, item_name: str):
    """The item named *item_name*, and None; or None and the refusal every
    route answers an unknown item with, naming the items there are."""
    if experiment is None:
        return None, no_experiment()
    item, why = find_experiment_item(experiment, item_name)
    if item is not None:
        return item, None
    return None, refused("not_found", why["error"], item_names=why["item_names"])


def is_failure(item) -> bool:
    """Whether *item* failed: its own flag where it keeps one (a lamella), else
    whether a run of it failed."""
    flag = getattr(item, "is_failure", None)
    if flag is not None:
        return bool(flag)
    return any(s.status is TaskStatus.Failed for s in item.task_history)


def history_row(item, state) -> HistoryRow:
    """One run of *item* (*state*, a ``TaskState``) as task_history lists it."""
    return HistoryRow(
        item_name=item.name,
        item_id=item.id,
        task_name=state.name,
        task_id=state.task_id,
        task_type=state.task_type,
        task_status=state.status.name,
        task_status_message=state.status_message,
        iteration=state.iteration,
        start_timestamp=to_iso(state.start_timestamp),
        end_timestamp=to_iso(state.end_timestamp),
        completed_at=state.completed_at,
        duration=state.duration,
    )


class AppContext:
    """The reads every app serves, over its ``AppSession``."""

    # The app the operator works in, as the App Server names it to the agent.
    app_name = "fibsemOS"

    def __init__(self, session, event_buffer=None):
        self._session = session
        self._event_buffer = event_buffer
        # The microscope's progress, followed from the first status read.
        self._progress = ProgressLines()
        self._progress_microscope = None
        self._stop_progress: Optional[Callable[[], None]] = None

    # --- call-time resolution: never cache what the app rebinds ---------------

    @property
    def _experiment(self):
        return self._session.experiment

    @property
    def experiment_path(self) -> Optional[str]:
        """The open experiment's folder, or None: where an agent may save."""
        experiment = self._experiment
        return str(experiment.path) if experiment is not None else None

    def experiment_root(self, experiment: Optional[str] = None) -> Optional[str]:
        """The folder an experiment's files are read from: the open one's, or
        *experiment*, a past one by its path from the recents or the
        experiments folder (its folder or experiment.yaml). Raises
        ``FileRefusal`` for any other."""
        if experiment is None:
            return self.experiment_path
        wanted = os.path.normcase(os.path.realpath(experiment))
        for entry in self.recent_experiments():
            yaml = os.path.normcase(os.path.realpath(entry["path"]))
            if wanted in (yaml, os.path.dirname(yaml)):
                return os.path.dirname(entry["path"])
        found = self._under_experiments_folder(experiment)
        if found is not None and os.path.basename(found) == "experiment.yaml":
            found = os.path.dirname(found)
        if found is None or not os.path.isfile(os.path.join(found, "experiment.yaml")):
            raise FileRefusal(
                404,
                "not_found",
                f"No experiment at {experiment}: give a path from "
                "experiment_list_recent or experiment_list_all.",
            )
        return found

    def picture_sources(self) -> Optional[Any]:
        """What the agent's pictures read from the experiment (its items'
        images, configs and poses; see ``ai.agent.pictures``), or None."""
        return None

    def recording_experiment(self):
        """The experiment what the agent asks is recorded on: the open one, or
        the session's plain one. Raises OSError when it cannot be made."""
        return self._session.recording_experiment()

    def image_metadata(self, path: str) -> Any:
        """A saved beam image's full metadata, as its file records it, for a
        path inside the open experiment only (the agent's drill-down behind
        the short block every image answer carries)."""
        target, refusal = self._experiment_image(path)
        if refusal:
            return refusal
        image = FibsemImage.load(target)
        if image.metadata is None:
            return refused("not_found", f"{path} carries no beam image metadata.")
        return SavedImageMetadata(path, image.metadata)

    def _experiment_image(self, path: str):
        """``(real path, None)`` for a file inside the open experiment, else
        ``(None, refusal)``: the agent reads saved images there only."""
        import os

        try:
            target = resolve(self.experiment_path, path)
        except FileRefusal as refusal:
            return None, refused(refusal.error_type, refusal.message)
        if not os.path.isfile(target):
            return None, refused("not_found", f"{path} is a folder.")
        return target, None

    def image_stage_position(self, path: str, px=None, fraction=None, crop=None) -> Any:
        """The stage position under a point of any saved image in the
        experiment, beam or fluorescence, through the same projection the
        overview canvases use, built from the image's own metadata: it converts
        as the image was taken, wherever the stage is now. ``px`` is [x, y] in
        source pixels (origin top-left, +y down); ``fraction`` is [x, y] from 0
        to 1 across the image. With *crop* (a cell's name or a box), both are
        read off that crop's picture instead (`fibsem.imaging.cells`)."""
        target, refusal = self._experiment_image(path)
        if refusal:
            return refusal
        try:
            if crop is not None:
                px, fraction = _uncropped_point(target, crop, px, fraction)
            position = image_point_to_stage_position(target, px=px, fraction=fraction)
        except ValueError as exc:
            return refused("invalid", str(exc))
        return SavedImagePosition(path, position)

    def image_pixels(self, path: str, stage_positions: List[Dict[str, Any]]) -> Any:
        """Where each stage position falls on a saved image in the experiment,
        in source pixels, with the image's scale block. A position the image
        cannot place (no geometry for it) answers its reason instead."""
        target, refusal = self._experiment_image(path)
        if refusal:
            return refusal
        try:
            positions = [FibsemStagePosition.from_dict(p) for p in stage_positions]
        except (TypeError, KeyError, AttributeError) as exc:
            return refused("invalid", f"Not a stage position: {exc}")
        block, project = stage_projector(target)
        points = []
        for position in positions:
            try:
                point = project(position)
            except Exception as exc:  # noqa: BLE001 - per-position answer
                points.append(PixelPoint(reason=str(exc)))
                continue
            x, y = float(point.x), float(point.y)
            inside = 0 <= x <= block["width"] and 0 <= y <= block["height"]
            points.append(PixelPoint(px=[x, y], inside=inside))
        return SavedImagePixels(path, block, points)

    def draw_image(
        self,
        path: str,
        marks: Any = None,
        crosshair: bool = False,
        metadata_bar: bool = True,
        channels: Optional[List[str]] = None,
    ) -> Any:
        """A saved image in the experiment (beam, or a fluorescence stack's max
        projection) with *marks* (the one marks vocabulary, its pixels) drawn
        on it, as evidence draws them, the centre *crosshair*, the scale bar
        and the metadata bar. Saved as ``drawn/<image>.png`` beside the image
        (replaced when drawn again), a path to cite as evidence."""
        target, refusal = self._experiment_image(path)
        if refusal:
            return refusal
        checked, refusal = _checked_marks(marks, target)
        if refusal:
            return refusal
        try:
            rgb = drawn_picture(
                target,
                checked,
                crosshair=crosshair,
                metadata_bar=metadata_bar,
                channels=channels,
            )
        except ValueError as exc:
            return {"error_type": "invalid", "message": str(exc)}
        saved = drawn_path(target, image_stem(target))
        return ImageDrawn(path, saved, checked, save_drawn(rgb, saved))

    def draw_pair(
        self,
        before: str,
        after: str,
        marks: Any = None,
        metadata_bar: bool = True,
        channels: Optional[List[str]] = None,
    ) -> Any:
        """Two saved images of the experiment side by side at one scale, each
        with its own bars, *marks* (in *before*'s pixels) drawn on both: pre and
        post mill, before and after an alignment, SEM next to FIB. Saved as
        ``drawn/<before>-<after>.png`` beside *before*."""
        first, refusal = self._experiment_image(before)
        if refusal:
            return refusal
        second, refusal = self._experiment_image(after)
        if refusal:
            return refusal
        checked, refusal = _checked_marks(marks, first)
        if refusal:
            return refusal
        try:
            rgb = drawn_pair(
                first,
                second,
                checked,
                metadata_bar=metadata_bar,
                channels=channels,
            )
        except ValueError as exc:
            return {"error_type": "invalid", "message": str(exc)}
        saved = drawn_path(first, f"{image_stem(first)}-{image_stem(second)}")
        return ImagePairDrawn(before, after, saved, checked, save_drawn(rgb, saved))

    @property
    def _runner(self):
        return self._session.runner

    @property
    def _manager(self):
        return self._runner.manager

    @property
    def _running(self) -> bool:
        return self._runner.is_running

    @property
    def _microscope(self):
        return self._session.microscope

    # --- status ---------------------------------------------------------------

    def status(self) -> AppStatus:
        """What the app is doing right now, at a glance."""
        experiment = self._experiment
        status = AppStatus(
            microscope_connected=self._microscope is not None,
            experiment=None,
            workflow=WorkflowState(running=self._running),
        )
        if experiment is not None:
            status.experiment = ExperimentRef(
                experiment.name, str(experiment.path), len(experiment.positions)
            )
            # The proposals a task is stopped on, for an agent that has just
            # connected: their events are behind its cursor.
            status.workflow.asking = [
                Asking(item.id, item.name, task_name, proposal.id)
                for item, task_name, proposal in experiment.pending_proposals()
                if proposal.asking
            ]
        manager = self._manager
        if manager is not None:
            items = manager.queue.items  # thread-safe copy under the queue's lock
            status.workflow.queue_total = len(items)
            status.workflow.queue_version = manager.queue.version
            active = next(
                (i for i in items if i.status is TaskStatus.InProgress),
                None,
            )
            if active is not None:
                status.workflow.current_task = active.task_name
                status.workflow.current_item = active.item_name
        self._run_line(status.workflow)
        status.progress = self._progress_now()
        control = getattr(self._microscope, "control", None)
        held = getattr(control, "current", None)
        if held is not None and held.by != WORKFLOW:
            status.in_control = held
        return status

    def _run_line(self, workflow: WorkflowState) -> None:
        """What the run last said (`RunLine`): its place, step, hold and failures."""
        runner = self._runner
        line = runner.line
        report = line.report
        if self._running:
            if report is not None:
                workflow.queue_position = report.queue_position
                workflow.iteration = report.iteration
            workflow.step = line.step or None
            if line.hold is not None:
                workflow.waiting_on = line.hold.releases
                workflow.waiting_since = iso_time(line.held_since)
        if line.failures:
            last = line.failures[-1]
            workflow.failed = RunFailed(
                last.what, last.message, iso_time(last.at), len(line.failures)
            )
        elif not self._running and runner.last_note:
            workflow.failed = RunFailed("Run stalled", runner.last_note)

    def _progress_now(self) -> Optional[ProgressLine]:
        """The microscope's progress line now, following the microscope there is."""
        microscope = self._microscope
        if microscope is not self._progress_microscope:
            if self._stop_progress is not None:
                self._stop_progress()
            self._progress = ProgressLines()
            self._stop_progress = (
                self._progress.follow(microscope) if microscope is not None else None
            )
            self._progress_microscope = microscope
        return self._progress.current

    def queue(self) -> Queue:
        """The live work queue: id-anchored items plus the mutation version;
        empty, with no version, when nothing runs."""
        manager = self._manager
        if manager is None:
            return Queue()
        return Queue(
            version=manager.queue.version,
            items=[
                QueueEntry(item.id, item.item_name, item.task_name, item.status.name)
                for item in manager.queue.items
            ],
        )

    # --- the experiment ---------------------------------------------------------

    # What each app's reads answer: an app with facts of its own maps the base
    # class to its subclass.
    answer_types: Dict[type, type] = {}

    def answer_type(self, base: type) -> type:
        """The class this app answers for *base*."""
        return self.answer_types.get(base, base)

    def summary_row(self, experiment, item) -> SummaryRow:
        """*item*'s row of the experiment summary; an app adds its facts."""
        last = item.last_completed_task
        return SummaryRow(
            experiment_name=experiment.name,
            item_name=item.name,
            item_id=item.id,
            is_failure=is_failure(item),
            last_completed_task=last.name if last is not None else None,
            last_completed_at=last.completed_at if last is not None else None,
        )

    def experiment_summary(self) -> ExperimentSummary:
        experiment = self._experiment
        if experiment is None:
            return no_experiment()
        return self.answer_type(ExperimentSummary)(
            items=[self.summary_row(experiment, p) for p in experiment.positions],
        )

    def task_history(self) -> TaskHistory:
        """Every run of every item, in the order they started."""
        experiment = self._experiment
        if experiment is None:
            return no_experiment()
        rows = [
            (s.start_timestamp, history_row(p, s))
            for p in experiment.positions
            for s in p.task_history
        ]
        # Through POSIX time, so old naive stamps and aware ones still compare.
        rows.sort(key=lambda r: r[0].timestamp() if r[0] else float("inf"))
        return TaskHistory(items=[row for _, row in rows])

    def item_read(self, item_name: str, read: Callable[[Any, Any], Any], type_):
        """*read*(experiment, item) for the item named *item_name*, or the
        refusal naming the items there are."""
        item, refusal = find_item(self._experiment, item_name)
        if refusal:
            return refusal
        return read(self._experiment, item)

    def item_detail(self, item_name: str) -> ItemDetail:
        """The durable facts about one item, as a curated snapshot: every
        field is wire contract. Files are task_outputs; history task_history."""
        return self.item_read(item_name, self.detail_of, self.answer_type(ItemDetail))

    def detail_of(self, experiment, item) -> ItemDetail:
        """*item*'s detail; an app adds its facts."""
        return self.answer_type(ItemDetail)(
            item_name=item.name,
            id=item.id,
            description=item.description,
            poi=getattr(item, "poi", None),
            alignment_area=getattr(item, "alignment_area", None),
            poses=item_poses(item),
            version=self.item_version(item),
        )

    def item_version(self, item) -> str:
        """The item document's version: patches echo it."""
        raise NotImplementedError

    def task_outputs(self, item_name: str) -> TaskOutputs:
        """The files an item's tasks produced, existence-checked, per run in
        run order: which task produced which image."""
        return self.item_read(item_name, self.outputs_of, self.answer_type(TaskOutputs))

    def outputs_of(self, experiment, item) -> TaskOutputs:
        """*item*'s outputs; an app adds its own."""
        history = list(item.task_history)
        return self.answer_type(TaskOutputs)(
            item_name=item.name,
            completed_tasks=[
                s.name for s in history if s.status is TaskStatus.Completed
            ],
            tasks=files_by_run(experiment.item_path(item), history),
        )

    def item_stack(self, item_name: str, role: str) -> ItemStack:
        """The images an item's runs recorded under *role*, in iteration
        order: a volume position's slices, scrubbed as one stack."""

        def read(experiment, item) -> ItemStack:
            slices = stack_slices(
                experiment.item_path(item), item.task_history, role, experiment.path
            )
            for entry in slices:
                entry.z = self.slice_depth(item, entry.iteration)
            return ItemStack(item_name=item.name, role=role, slices=slices)

        return self.item_read(item_name, read, ItemStack)

    def slice_depth(self, item, iteration: Optional[int]) -> Optional[float]:
        """How far into *item* its *iteration* cut, in metres; None when the
        app doesn't slice."""
        return None

    def item_history(
        self, item_name: str, flt: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """The item's History, as the Qt tab builds it: one record per task run
        (and per session's operations outside a task), with its shown images
        and operations, through the tab's filter. ``tasks`` names every task
        the item ran, for a task filter; ``total`` counts the runs unfiltered."""
        from pathlib import Path

        experiment = self._experiment
        item, refusal = find_item(experiment, item_name)
        if refusal:
            return refusal
        path = Path(experiment.path) / EVENTS_FILENAME
        events: List[Dict[str, Any]] = []
        if path.is_file():
            events, _ = read_new_events(path, 0, HISTORY_KINDS)
        runs = history_rows(
            item,
            events,
            self.history_images(experiment, item),
            folder=str(experiment.item_path(item)),
        )
        shown = HistoryFilter.from_dict(
            {key: value for key, value in flt.items() if value not in (None, "")}
        )
        return {
            "item_name": item.name,
            "filter": shown.to_dict(),
            "tasks": list(dict.fromkeys(run.task_name for run in runs)),
            "total": len(runs),
            "runs": [
                run_record(run, str(experiment.path))
                for run in filter_runs(runs, shown)
            ],
        }

    def history_images(self, experiment, item):
        """``history_rows``' ``run_images``: what a History row shows for a
        run. Every file the run recorded; an app narrows it."""
        root = experiment.item_path(item)
        return lambda run, fallback: recorded_outputs(root, [run], *run.outputs)

    def run_summary(self) -> Dict[str, Any]:
        """The most recent run's outcome table, if a run has happened."""
        manager = self._manager
        if manager is not None:
            return {"items": dataframe_records(manager.build_run_summary_dataframe())}
        runner = self._runner
        last = runner.last_summary if runner is not None else None
        if last is not None:
            return {"items": dataframe_records(last)}
        return refused("not_found", "No run has finished in this session.")

    def log(
        self,
        since: Optional[str] = None,
        until: Optional[str] = None,
        task_id: Optional[str] = None,
        level: Optional[str] = None,
        lines: int = 200,
    ) -> Dict[str, Any]:
        """The session's log in a time window: ``since``/``until``, or the run
        ``task_id`` (from its start to its end, a few seconds either side), or
        the tail when neither is given. See ``log_tail`` for what is served."""
        from pathlib import Path

        experiment = self._experiment
        if experiment is None:
            return no_experiment()
        if level is not None:
            level = level.upper()
            if level not in log_tail.LEVELS:
                return refused(
                    "invalid", f"No log level {level!r}.", levels=list(log_tail.LEVELS)
                )
        window = {}
        for name, value in (("since", since), ("until", until)):
            if value is None:
                continue
            window[name] = wall_time_of(value)
            if window[name] is None:
                return refused("invalid", f"{name}={value!r} is not a time.")
        if task_id is not None:
            if window:
                return refused("invalid", "Give task_id or since/until, not both.")
            found = log_tail.run_window(experiment.path, task_id)
            if found is None:
                return refused(
                    "not_found", f"No run {task_id!r} in this experiment's record."
                )
            window = {"since": found[0], "until": found[1]}
            # A run's window at DEBUG is mostly noise, and the cap keeps its
            # end: the start of the run would be the part dropped.
            level = level or "INFO"
        path = Path(experiment.path) / log_tail.LOG_FILENAME
        if not path.is_file():
            return {"records": []}
        result = log_tail.read_log(path, level=level, lines=lines, **window)
        start, end = window.get("since"), window.get("until")
        return {
            "since": start.isoformat() if start is not None else None,
            "until": end.isoformat() if end is not None else None,
            "level": level or "DEBUG",
            **result,
        }

    # --- instrument-adjacent (cached only, never a hardware call) --------------

    def display_images(self) -> DisplayImages:
        """The SEM/FIB images the GUI is displaying right now, and the FM's
        latest frame, as previews.

        This is the display cache, not an acquisition: nothing touches
        hardware, and the workflow's post-mill images appear here the moment
        the GUI shows them — long before task completion writes them to disk —
        which is what makes inline inspection possible.

        Thread note: grabs the widget's current image references (replaced
        whole on acquisition, never mutated in place) and encodes previews on
        the server thread; the GUI thread is neither entered nor blocked.
        """
        images = self._latest_images
        if images is None:
            return refused("not_found", "No microscope is connected.")

        def preview(image):
            payload = image_payload(image)
            if payload is None:
                return None
            return decode(ImagePayload, {**payload, "info": image_info(image)})

        return DisplayImages(
            sem=preview(images.get(BeamType.ELECTRON)),
            fib=preview(images.get(BeamType.ION)),
            fm=preview(images.get_fm()),
            fm_available=getattr(self._microscope, "fm", None) is not None,
        )

    @property
    def _latest_images(self):
        """The session's latest image of each beam, window or not; None
        until a microscope connects."""
        if self._microscope is None:
            return None
        return self._session.images

    # --- events -----------------------------------------------------------------

    def events(self, since: int = 0, timeout: float = 0.0) -> Dict[str, Any]:
        """Events after ``since``, parking up to ``timeout`` seconds for news.

        Unavailable (rather than empty) when the hosting wired no buffer, so a
        client can tell "no events yet" from "this server has no event stream".
        """
        return buffer_or_unavailable(self._event_buffer, since=since, timeout=timeout)

    def event_history(
        self, before: Optional[int] = None, limit: int = 500
    ) -> Dict[str, Any]:
        """The experiment's recorded events from before the live buffer, newest
        page first: everything ``events`` no longer holds, read from the
        experiment's events file, without the live progress records.

        ``start`` is the file offset to pass back as ``before`` for the page
        before this one; 0 when this page reaches the file's start.
        """
        from pathlib import Path

        experiment = self._experiment
        if experiment is None:
            return no_experiment()
        path = Path(experiment.path) / EVENTS_FILENAME
        if not path.is_file():
            return {"events": [], "start": 0}
        # What the buffer still holds, `events` serves; leave it out here.
        held = None
        if self._event_buffer is not None:
            first = self._event_buffer.events_since(0).get("events") or []
            if first:
                held = (first[0].get("session"), first[0].get("seq"))

        def keep(record: Dict[str, Any]) -> bool:
            if held is None or record.get("session") != held[0]:
                return True
            seq = record.get("seq")
            return not isinstance(seq, int) or seq < held[1]

        events, start = read_events_before(
            path,
            before=before,
            limit=max(1, min(limit, 2000)),
            skip_kinds=LIVE_KINDS,
            keep=keep,
        )
        return {"events": events, "start": start}

    # --- discovery --------------------------------------------------------------

    def _experiments_folder(self) -> str:
        """Where experiments are created and found: the session's folder, else
        the default experiments folder."""
        return self._session.experiments_folder or default_experiment_directory()

    def _under_experiments_folder(self, path: str) -> Optional[str]:
        """*path* (relative to the experiments folder, or absolute) when it is
        inside that folder, links followed; else None."""
        folder = os.path.realpath(self._experiments_folder())
        target = os.path.realpath(os.path.join(folder, path))
        try:
            inside = os.path.commonpath([target, folder]) == folder
        except ValueError:  # another drive
            return None
        return target if inside and target != folder else None

    def experiment_entry(self, summary) -> Dict[str, Any]:
        """One experiment in a listing, from its ``ExperimentSummary``.
        ``relative_path`` is its folder relative to the experiments folder, as
        loads and reads take it; None for one outside that folder."""
        folder = os.path.dirname(summary.path)
        relative = None
        if self._under_experiments_folder(folder) is not None:
            relative = os.path.relpath(
                os.path.realpath(folder), os.path.realpath(self._experiments_folder())
            ).replace(os.sep, "/")
        return {
            "name": summary.name,
            "path": summary.path,
            "relative_path": relative,
            "app": summary.app,
            "created_at": to_iso(summary.created_at),
            "num_items": summary.num_items,
            "available": summary.available,
            "instrument_model": summary.instrument_model,
            "instrument_serial": summary.instrument_serial,
            "operator": summary.operator,
            "software_version": summary.software_version,
        }

    def recent_experiments(self) -> List[Dict[str, Any]]:
        """The recents list, via the never-raising peek layer."""
        return [self.experiment_entry(s) for s in get_recent_experiments()]

    def saved_experiments(
        self,
        instrument: Optional[str] = None,
        operator: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Every experiment under the experiments folder, newest first, narrowed
        by `filter_experiments` (each filter left None does not narrow)."""
        folder = self._experiments_folder()
        found = filter_experiments(
            discover_experiments(folder),
            instrument=instrument,
            operator=operator,
            since=since,
            until=until,
        )
        return {"folder": folder, "items": [self.experiment_entry(s) for s in found]}

    def _not_while_running(self, what: str) -> Optional[Refusal]:
        """The refusal while a workflow runs (*what* to do when it has
        stopped), or None."""
        why = self._runner.run_refusal(what)
        return refused("task_running", why) if why else None

    def _busy(self, why: str) -> Optional[Refusal]:
        """A run_edits refusal as the routes answer it, or None."""
        return refused("task_running", why) if why else None

    def _no_microscope(self) -> Optional[Refusal]:
        if self._microscope is None:
            return refused("not_found", "No microscope is connected.")
        return None

    def _session_refusal(self) -> Optional[Refusal]:
        """Why the open experiment cannot be replaced now, or None."""
        return self._not_while_running("change experiments") or self._no_microscope()

    def _agent_call(self, call, timeout: float = 30.0):
        """``call()`` where the session edits the experiment."""
        return self._session.run_edit(call).result(timeout=timeout)

    def proposals(self) -> Dict[str, Any]:
        """Every pending proposal, derived from the experiment, with the
        reference image its values sit on as an agent-sized preview."""
        experiment = self._experiment
        if experiment is None:
            return no_experiment()

        def describe(item, task_name, proposal):
            doc = to_plain(proposal.to_dict())
            doc.update(
                {
                    "item_id": item.id,
                    "item_name": item.name,
                    "task_name": task_name,
                    # what this is; proposal_decide passes it back. The run
                    # is here as a fact about it, and still names it for a
                    # caller that has not learned the id.
                    "proposal_id": proposal.id,
                    "task_id": proposal.task_id,
                }
            )
            image = load_reference_image(experiment, item, proposal)
            # a fluorescence result is shown as its channel composite
            preview = preview_of(image) if image is not None else None
            doc["reference_image"] = (
                image_payload(preview) if preview is not None else None
            )
            return doc

        proposals = [describe(*entry) for entry in experiment.pending_proposals()]
        return {"proposals": proposals}

    def decide(
        self,
        item_id: str,
        task_name: str,
        outcome: str,
        values: Optional[Dict[str, Any]] = None,
        reason: str = "",
        actor: str = "",
        task_id: str = "",
        proposal_id: str = "",
        action: str = "",
        patch: Optional[Dict[str, Any]] = None,
        evidence: str = "",
    ) -> Dict[str, Any]:
        """Decide a pending proposal, exactly as the operator's answer is: the same
        Experiment.decide, on the main thread, blocking until applied. A call
        with the operator token is recorded as the operator named on the
        experiment; any other as the agent (``actor`` names its model).

        ``action`` is what a loop's pass is decided as (run, edit, skip,
        accept). ``patch`` sets fields of its value by dotted path, on the
        value given or else the one proposed: a milling config's
        ``stages.Polish 1.pattern.depth``.

        ``proposal_id`` is the proposal the agent was shown (from
        proposal_list), and what the decision is checked against: a task
        may ask several proposals in one run, and only the id tells the one
        that was shown from the one that replaced it. ``task_id``, the run, is
        accepted in its place from a caller that has not learned the id, and
        is checked the way it always was. One of them is needed; a stale one
        is refused."""
        experiment = self._experiment
        if experiment is None:
            return no_experiment()

        try:
            decided = DecisionOutcome[outcome]
        except KeyError:
            return refused(
                "invalid_value",
                f"outcome must be Confirmed or Rejected, not {outcome!r}",
            )
        try:
            decoded = _decode_values(values or {})
        except Exception as exc:  # noqa: BLE001 - any bad document is refused
            return refused("invalid_value", f"values could not be read: {exc}")
        if patch:
            item = experiment.get_item_by_id(item_id)
            shown = next(
                (
                    p
                    for p in (item.current_proposals(task_name) if item else [])
                    if p.id == proposal_id or (not proposal_id and p.pending)
                ),
                None,
            )
            if shown is None:
                return refused(
                    "invalid_value",
                    "patch needs the proposal_id of a current proposal.",
                )
            try:
                decoded = patch_values(shown.kind, decoded or dict(shown.values), patch)
            except ValueRefused as exc:
                return refused("invalid_value", str(exc))
        decision = Decision(
            outcome=decided,
            actor=signer(experiment.actor(), actor or "remote"),
            values=decoded,
            reason=reason,
            via="server",
            task_id=str(task_id or ""),
            proposal_id=str(proposal_id or ""),
            action=str(action or ""),
            evidence=evidence,
        )
        return decision_answer(experiment.decide(item_id, task_name, decision))

    def suggest(
        self,
        item_id: str,
        task_name: str,
        proposal_id: str,
        reason: str,
        outcome: str = "Confirmed",
        values: Optional[Dict[str, Any]] = None,
        patch: Optional[Dict[str, Any]] = None,
        action: str = "",
        evidence: Optional[Dict[str, Any]] = None,
        actor: str = "",
    ) -> Dict[str, Any]:
        """Suggest a decision on a pending proposal, for the operator to
        approve or decline in the window. Nothing is decided. Checked as the
        decision would be, so one that could not be approved is refused now."""
        experiment = self._experiment
        if experiment is None:
            return no_experiment()

        if outcome not in ("Confirmed", "Rejected"):
            return refused(
                "invalid_value", f"outcome is Confirmed or Rejected, not {outcome!r}"
            )
        try:
            decoded = _decode_values(values or {})
        except Exception as exc:  # noqa: BLE001 - any bad document is refused
            return refused("invalid_value", f"values could not be read: {exc}")
        # The beam image of the cited evidence, to draw the value on.
        image = frame_path(evidence)
        if image:
            root = self.experiment_path
            image = os.path.relpath(image, root) if root else image
        suggestion = Suggestion(
            actor=Actor.agent(actor or "remote"),
            reason=reason,
            outcome=DecisionOutcome[outcome],
            values=decoded,
            patch=dict(patch or {}),
            action=action,
            image=image or "",
            evidence=(evidence or {}).get("evidence_id", ""),
        )
        result = experiment.suggest(item_id, task_name, proposal_id, suggestion)
        if not result.applied:
            return decision_answer(result)
        # Nothing was decided, so nothing ran or synced.
        doc = {"applied": True, "suggestion_id": suggestion.id}
        if result.save_error:
            doc["save_error"] = result.save_error
        return doc

    def add_note(
        self,
        text: str,
        item_name: Optional[str] = None,
        evidence: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Put an agent observation on the record, citing the *evidence* (its
        id) it rests on.

        The agent's judgments otherwise live only in its chat: this writes
        them where the run's story is told — the event stream (live consumers:
        timeline, dashboard, other agents) and the experiment log (durable).
        Notes are observations, not actions: nothing changes state.
        """
        return self._note(text, item_name, evidence=evidence)

    def add_message(
        self,
        text: str,
        item_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        evidence: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Put the agent's message to the operator on the record: something they
        should read now, or its answer to their message (``reply_to``, the
        message's id). It changes nothing."""
        message_id = uuid.uuid4().hex[:8]
        result = self._note(
            text,
            item_name,
            evidence=evidence,
            to=OPERATOR,
            id=message_id,
            reply_to=reply_to,
        )
        if not isinstance(result, Refusal):
            result["id"] = message_id
        return result

    def _note(
        self, text: str, item_name: Optional[str], **fields: Any
    ) -> Dict[str, Any]:
        text = str(text).strip()
        if not text:
            return refused("missing_field", "A note needs its text.")
        item_id = None
        if item_name is not None:
            item, refusal = find_item(self._experiment, item_name)
            if refusal:
                return refusal
            item_id = item.id
        logging.info(
            "agent %s%s: %s",
            "message" if fields.get("to") else "note",
            f" [{item_name}]" if item_name else "",
            text,
        )
        # With the item's id, so the record files the note under the item
        # it is about, not whichever one is running.
        self.record_event(
            NOTE,
            note_record(text, item_id=item_id, item_name=item_name, **fields),
        )
        return {"recorded": True}

    def add_evidence(
        self, item_name: Optional[str], save: Callable[[], Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Keep the agent's evidence (*save* keeps it in the experiment and
        answers its record) and put it on the record under the item it names.
        What cites it names its id."""
        item_id = None
        if item_name is not None:
            item, refusal = find_item(self._experiment, item_name)
            if refusal:
                return refusal
            item_id = item.id
        record = save()
        logging.info(
            "agent evidence %s%s: %s",
            record["evidence_id"],
            f" [{item_name}]" if item_name else "",
            record["image"],
        )
        self.record_event(EVIDENCE, {**record, "item_id": item_id})
        return {"recorded": True, **record}

    def record_event(
        self, kind: str, payload: Dict[str, Any], actor: Optional[str] = None
    ) -> None:
        """Put an event on the experiment's record: everything the server's
        side writes, the agent's requests and their outcomes
        among them. *actor* as for ``EventBuffer.append``."""
        if self._event_buffer is not None:
            self._event_buffer.append(kind, payload, actor=actor)

    def stop_workflow(self, reason: str = "") -> Dict[str, Any]:
        """Stop the running workflow — the same path as the GUI's Stop button.

        Everything on that path is thread-safe from here: the manager's stop is
        an event plus the milling service's stop, the call the core server's
        ``stop_milling`` route already issues from this thread. Stopping is the
        safety action, so like ``stop_milling`` it never waits on the command
        lock.
        """
        if not self._running:
            return {"stopped": False, "reason": "no workflow is running"}
        self._runner.stop(message=stop_message(current_actor(), reason))
        return {"stopped": True}

    def stop_task(
        self, item_name: str, task_name: str, reason: str = ""
    ) -> Dict[str, Any]:
        """Stop the task now running and carry on with the rest of the queue,
        as the timeline's Stop Task does. *item_name* and *task_name* name the
        task meant: a stop that lands after it finished is refused rather than
        stopping the next one."""
        manager = self._manager
        if manager is None:
            return {"stopped": False, "reason": "no workflow is running"}
        active = manager.queue.active
        if active is None or (active.item_name, active.task_name) != (
            item_name,
            task_name,
        ):
            return refused(
                "not_running",
                f"{task_name} for {item_name} is not the task running.",
                running=(
                    {"item_name": active.item_name, "task_name": active.task_name}
                    if active is not None
                    else None
                ),
            )
        self._runner.stop_task(message=stop_message(current_actor(), reason))
        return {"stopped": True, "item_name": item_name, "task_name": task_name}

    def _start_run(self, request: RunRequest, timeout: float) -> Dict[str, Any]:
        """Start ``request`` through the session's runner, where the
        experiment is edited, as every start goes."""
        runner = self._runner
        outcome = self._session.run_edit(
            lambda: runner.start(self._experiment, request)
        )
        started = outcome.result(timeout=timeout)
        if not started.started:
            return refused(started.error_type, started.reason, **started.details)
        return started.as_dict()

    def start_workflow(
        self,
        task_names: List[str],
        item_names: Optional[List[str]] = None,
        timeout: float = 15.0,
    ) -> Dict[str, Any]:
        """Start a workflow as the Run button would — the batch-review opener.

        Started where the experiment is edited (the GUI thread, with a window,
        which draws its run chrome there); validation refusals come back
        structured with the valid names. ``item_names`` of None means every item in the experiment.
        The agent at Drive is the consent that stands in for the Run click's
        confirm dialog.
        """
        result = self._start_run(
            RunRequest(
                kind=ITEMS,
                task_names=tuple(task_names),
                names=tuple(item_names) if item_names is not None else None,
            ),
            timeout,
        )
        if not isinstance(result, Refusal):
            # Who started it: the run's own workflow_started fires on its
            # worker thread, which carries no one's mark.
            self.record_event(
                "workflow_start",
                {"task_names": list(task_names), "item_names": item_names},
            )
        return result

    def requeue_task(
        self, item_name: str, task_name: str, front: bool = False
    ) -> Dict[str, Any]:
        """Queue ``task_name`` for ``item_name`` (again) — the batch-review verb.

        The queue exists only while a workflow runs, so a re-run is queued into
        the live run: "run 03 again" while the batch continues. Adding a
        duplicate of a completed pair is the queue's own re-run mechanism; the
        item lands at the back (or the front with ``front=True``). A pair
        already waiting is refused (``already_queued``): it runs once. The
        timeline's Run again queues the same way (``session.runner.requeue``).
        """
        edit = self._agent_call(
            lambda: self._session.runner.requeue(
                item_name, task_name, front=bool(front)
            )
        )
        if not edit.applied:
            return refused(edit.error_type, edit.reason, **edit.details)
        (item,) = edit.items
        return {
            "queued": True,
            "item_id": item.id,
            "item_name": item_name,
            "task_name": task_name,
            "queue_version": edit.version,
        }

    def edit_queue(self, item_name: str, task_name: str, action: str) -> Dict[str, Any]:
        """Remove or move a waiting (item, task) pair in the running workflow's
        queue, as the timeline's menu does.

        ``action`` is one of :data:`QUEUE_ACTIONS`: ``remove`` (it will not run;
        kept in the run record as Removed), ``move_up``, ``move_down`` or
        ``run_next``. Only a pair still waiting can be edited: the running one
        is refused (``not_pending``) -- stopping it is ``stop_workflow`` -- and
        so is one that already ran.
        """
        edit = self._session.runner.edit(item_name, task_name, action)
        if not edit.applied:
            return refused(edit.error_type, edit.reason, **edit.details)
        (item,) = edit.items
        return {
            "applied": True,
            # Already first or last: nothing moved.
            "changed": edit.changed,
            "item_id": item.id,
            "item_name": item_name,
            "task_name": task_name,
            "action": action,
            "queue_version": edit.version,
        }

    # --- experiments and protocols: the File menu, for an agent at Drive ---------
    #
    # Kept to known places: a new
    # experiment goes anywhere under the experiments folder, a load is a recent
    # experiment or one under that folder, and a protocol is one of the files in
    # the app's protocol folder, by name (discovery, above, lists the
    # experiments). An app says where its protocols are and how its experiments
    # are made and loaded; one with no protocol_type has none.

    protocol_type = None

    def _protocol_folder(self) -> Optional[str]:
        """Where the app's protocols are saved; None for an app with none."""
        return None

    def _protocol_folders(self) -> List[str]:
        """Where the app's protocols are listed from, its saving folder first."""
        folder = self._protocol_folder()
        return [folder] if folder else []

    def _default_protocol_path(self) -> Optional[str]:
        """The protocol a new experiment starts with; None for the protocol
        type's default."""
        return None

    def _new_experiment(self, directory: str, name: str, protocol, metadata):
        """A new experiment in *directory* with *protocol*, saved and recorded
        as the most recent."""
        raise NotImplementedError

    def _load_for_adoption(self, path: str):
        """The experiment at *path* to open, and None; or None and why not."""
        raise NotImplementedError

    def _no_files(self) -> Optional[Refusal]:
        if self.protocol_type is None:
            return refused(
                "invalid", f"{self.app_name} has no experiment or protocol files."
            )
        return None

    def _listed_protocols(self) -> List[Dict[str, Any]]:
        folders = self._protocol_folders()
        return list_protocols(folders, self._default_protocol_path()) if folders else []

    def protocol(self) -> Dict[str, Any]:
        """The workflow: each task's plan entry, in order."""
        experiment = self._experiment
        protocol = getattr(experiment, "task_protocol", None) if experiment else None
        if protocol is None:
            return no_experiment()
        return {
            "name": protocol.name,
            "tasks": [task.to_dict() for task in protocol.workflow_config.tasks],
        }

    def protocols(self) -> Dict[str, Any]:
        """The protocol files an agent may load or create an experiment with."""
        return {
            "folder": self._protocol_folder(),
            "protocols": self._listed_protocols(),
        }

    def _protocol_or_refusal(self, name: Optional[str]):
        if name is None:
            path = self._default_protocol_path()
            if path is None:
                return self.protocol_type.default(), None
        else:
            path = next(
                (p["path"] for p in self._listed_protocols() if p["name"] == name),
                None,
            )
        if path is None:
            return None, refused(
                "not_found",
                f"No protocol named {name!r} in the protocol folder.",
                protocol_names=[p["name"] for p in self._listed_protocols()],
            )
        try:
            return self.protocol_type.load(path), None
        except Exception as exc:  # noqa: BLE001 - a broken file is a data fact
            return None, refused(
                "invalid", f"Could not load the protocol {path}: {exc}"
            )

    def _adopt(self, experiment) -> None:
        """Make *experiment* the open one, as the dialogs' OK does."""
        self._agent_call(lambda: self._session.adopt(experiment))

    def create_experiment(
        self,
        name: str,
        protocol: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create an experiment at *name*, a folder path under the experiments
        folder (made as needed), and open it, with *protocol* from the protocol
        folder (the default protocol when omitted). An existing folder is
        refused, never overwritten."""
        refusal = self._no_files() or self._session_refusal()
        if refusal:
            return refusal
        parts = (name or "").replace("\\", "/").split("/")
        path = self._under_experiments_folder(name or "")
        if any(p in ("", ".", "..") for p in parts) or path is None:
            return refused(
                "invalid",
                "name is a folder, or a path of folders, under the "
                f"experiments folder {self._experiments_folder()}.",
            )
        if len(parts) > DEFAULT_MAX_DEPTH:
            return refused(
                "invalid",
                f"At most {DEFAULT_MAX_DEPTH} folders deep, so "
                "experiment_list_all finds it again.",
            )
        if os.path.exists(path):
            return refused("invalid", f"{path} exists.")
        task_protocol, refusal = self._protocol_or_refusal(protocol)
        if refusal:
            return refusal
        metadata = {"description": description} if description else None
        experiment = self._new_experiment(
            os.path.dirname(path), os.path.basename(path), task_protocol, metadata
        )
        self._adopt(experiment)
        return {"applied": True, "experiment": encode(self.experiment_summary())}

    def load_experiment(self, path: str) -> Dict[str, Any]:
        """Open a recent experiment (``path`` as the recent list gives it) or
        one under the experiments folder (its folder or experiment.yaml,
        relative to that folder or absolute), as the Load dialog does. One the
        app cannot open is refused."""
        refusal = self._no_files() or self._session_refusal()
        if refusal:
            return refusal
        recent = [entry["path"] for entry in self.recent_experiments()]
        if path not in recent:
            found = self._under_experiments_folder(path)
            if found is not None and os.path.isdir(found):
                found = os.path.join(found, EXPERIMENT_FILENAME)
            if (
                found is None
                or os.path.basename(found) != EXPERIMENT_FILENAME
                or (not os.path.isfile(found))
            ):
                return refused(
                    "not_found",
                    "Not a recent experiment or one under the "
                    f"experiments folder {self._experiments_folder()}; the "
                    "operator opens any other from the Load dialog.",
                    recent_paths=recent,
                )
            path = found
        experiment, why = self._load_for_adoption(path)
        if experiment is None:
            return refused("invalid", f"Not loaded: {why}.")
        self._adopt(experiment)
        record_recent(experiment)
        return {"applied": True, "experiment": encode(self.experiment_summary())}

    def load_protocol(self, protocol: str) -> Dict[str, Any]:
        """Load a protocol from the protocol folder into the open experiment, as
        the Load Protocol dialog does: it replaces the experiment's protocol and
        is saved beside it."""
        refusal = self._no_files() or self._session_refusal()
        if refusal:
            return refusal
        experiment = self._experiment
        if experiment is None:
            return no_experiment()
        task_protocol, refusal = self._protocol_or_refusal(protocol)
        if refusal:
            return refusal
        self._agent_call(lambda: adopt_protocol(self._session, task_protocol))
        return {"applied": True, "protocol": self.protocol()}

    def save_protocol_as(self, name: str, overwrite: bool = False) -> Dict[str, Any]:
        """Save the experiment's protocol to the protocol folder as *name*, as
        Export Protocol does, so a later experiment can be created with it or
        load it. Refused when the file exists, unless *overwrite*."""
        refusal = self._no_files()
        if refusal:
            return refusal
        experiment = self._experiment
        protocol = getattr(experiment, "task_protocol", None) if experiment else None
        if protocol is None:
            return no_experiment()
        try:
            path = protocol_file_for(self._protocol_folder(), name)
        except ValueError as exc:
            return refused("invalid", str(exc))
        if os.path.exists(path) and not overwrite:
            return refused(
                "invalid",
                f"{os.path.basename(path)} already exists in the protocol "
                "folder; pass overwrite to replace it.",
                protocol_names=[p["name"] for p in self._listed_protocols()],
            )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        protocol.save(path)
        self.record_event("workflow_changed", {"field": "protocol_saved", "path": path})
        return {"saved": True, "name": os.path.basename(path), "path": path}

    # --- the protocol: its tasks, their order, requirements, schedule and
    # supervision, and each task's settings. Saved with the protocol at once.

    def protocol_task_config(self, task_name: str) -> Dict[str, Any]:
        """One task's protocol-level defaults document — what new items copy."""
        experiment = self._experiment
        protocol = getattr(experiment, "task_protocol", None) if experiment else None
        config_map = getattr(protocol, "task_config", None) if protocol else None
        if config_map is None:
            return no_experiment()
        config = dict(config_map).get(task_name)
        if config is None:
            return refused(
                "not_found",
                f"No task named {task_name!r} in the protocol.",
                task_names=list(config_map.keys()),
            )
        return self._config_document(task_name, config, level="protocol")

    @staticmethod
    def _config_document(task_name: str, config, level: str) -> Dict[str, Any]:
        """One config as a wire document with its version token: the full
        ``to_dict``, the contract for editing (you cannot patch what you
        cannot read)."""
        return {
            "task_name": task_name,
            "level": level,
            "version": config_version(config),
            "config": to_plain(config.to_dict()),
        }

    def apply_protocol_task_config_patch(
        self,
        task_name: str,
        patch: Dict[str, Any],
        version: str,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Patch a task's protocol-level settings, what each run of it reads,
        when *version* names the state the patch was written against. Recorded
        as the agent's. A running task holds its item's copy, so the edit
        reaches items created (or re-copied) after it lands."""
        session = self._session

        def edit():
            result = agent_patch_task_config(
                session.experiment,
                lambda: session.microscope,
                "protocol",
                "",
                task_name,
                dict(patch),
                str(version),
            )
            result = refusal_of(result)
            if isinstance(result, Refusal):
                return result
            if result["applied"]:
                session.changed.emit(
                    CONFIG,
                    {
                        "level": "protocol",
                        "item_name": "",
                        "task_name": task_name,
                        "result": result,
                    },
                )
            return result

        return self._agent_call(edit, timeout)

    def set_supervision(self, task_name: str, supervise: bool) -> Dict[str, Any]:
        """Set whether ``task_name`` asks for supervision
        (``workflow_edits.set_supervision``), as the window's chips do."""
        experiment = self._experiment
        if experiment is None or experiment.task_protocol is None:
            return no_experiment()
        return refusal_of(
            set_supervision(experiment, task_name, supervise, self.record_event)
        )

    def set_task_schedule(
        self, task_name: str, scheduled_at: Optional[str]
    ) -> Dict[str, Any]:
        """Set (or clear) when ``task_name`` may start, in the live workflow,
        as the Workflow tab's Edit Task does
        (``RunPlan.set_schedule``): up to two days ahead.

        The workflow reads the schedule at each task start (never a snapshot),
        so a change takes effect at the next start, and like ``set_supervision``
        it is saved at once.

        ``scheduled_at`` is ISO-8601 (naive = local time; an offset is
        normalized by the workflow) or ``None`` to clear.
        """
        parsed = None
        if scheduled_at is not None:
            try:
                parsed = datetime.fromisoformat(scheduled_at)
            except ValueError:
                return refused(
                    "invalid_value",
                    f"{scheduled_at!r} is not an ISO-8601 timestamp "
                    "(e.g. '2026-09-04T06:00:00'); null clears the schedule.",
                )
        experiment = self._experiment
        if experiment is None or experiment.task_protocol is None:
            return no_experiment()
        return refusal_of(
            workflow_edits.set_schedule(
                experiment, task_name, parsed, self.record_event
            )
        )

    def _workflow_edit(self, what: str):
        """The experiment and its workflow config for an edit, or a refusal."""
        experiment = self._experiment
        protocol = getattr(experiment, "task_protocol", None) if experiment else None
        if protocol is None:
            return None, None, no_experiment()
        refusal = self._not_while_running(what)
        if refusal:
            return None, None, refusal
        return experiment, protocol.workflow_config, None

    def _protocol_edit(self, edit, experiment, *args):
        """*edit* (one of ``workflow_edits``) on *experiment*, to run where the
        experiment is edited: recorded as this context's, and said to the
        window when it applied, so no editor writes back a list it drew
        before the change."""

        def run():
            result = edit(experiment, *args, self.record_event)
            if result["applied"]:
                self._session.changed.emit(PROTOCOL)
            return result

        return run

    def _protocol_changed(self, experiment, change: Dict[str, Any]) -> bool:
        """Save the protocol after an edit, say it changed, so no editor writes
        back a list it drew before the change, and put it on the record.
        Whether it saved."""
        saved = self._save_protocol(experiment)
        self._session.changed.emit(PROTOCOL)
        self.record_event("workflow_changed", change)
        return saved

    @staticmethod
    def _save_protocol(experiment) -> bool:
        """Save the experiment with its protocol; whether it saved. A failed
        save leaves the change in memory."""
        try:
            experiment.save(save_protocol=True)
        except Exception:
            logging.exception("save after a protocol change failed; applied in memory")
            return False
        return True

    def set_workflow_options(self, **options: Any) -> Dict[str, Any]:
        """Set the workflow's run options, as its options panel does, and like
        the operator's it persists with the protocol. Any of:

        - ``show_run_summary``: whether the window's run summary dialog opens
          when a run ends. Off leaves the run-end toast and its Show summary link: an agent
          starting many short runs turns it off so no dialog stands over the
          window.
        - ``turn_beams_off``: whether the beams are turned off when a run ends.

        Each is read when it applies (a run's end), so a change
        takes effect from then, mid-run included; plain rebinds, so no GUI
        marshal, like ``set_task_schedule``.
        """
        experiment = self._experiment
        protocol = getattr(experiment, "task_protocol", None) if experiment else None
        if getattr(protocol, "options", None) is None:
            return no_experiment()
        for name, value in options.items():
            setattr(protocol.options, name, value)
        saved = self._save_protocol(experiment)
        for name, value in options.items():
            self.record_event("workflow_changed", {"field": name, "value": value})
        return {"applied": True, "saved": saved, **options}

    def add_task(self, task_type: str, task_name: str) -> Dict[str, Any]:
        """Add a task of a type the protocol registers
        (``experiment.add_task``), as the protocol editor's Add Task does: its
        default settings, at the end of the workflow."""
        experiment, _, refusal = self._workflow_edit("add tasks")
        if refusal:
            return refusal
        types = sorted(experiment.task_protocol.task_classes())
        if task_type not in types:
            return refused(
                "not_found", f"No task type {task_type!r}.", task_type_names=types
            )

        result = self._agent_call(
            self._protocol_edit(
                workflow_edits.add_task, experiment, task_type, task_name
            )
        )
        if not result["applied"]:
            return refused("invalid", result["error"])
        return {
            **result,
            "task_names": experiment.task_protocol.workflow_config.workflow,
        }

    def remove_task(self, task_name: str) -> Dict[str, Any]:
        """Remove a task from the protocol and the workflow
        (``experiment.remove_task``), as the protocol editor's Remove does.
        Task history stays: it is the record of what ran."""
        experiment, config, refusal = self._workflow_edit("remove tasks")
        if refusal:
            return refusal
        if task_name not in experiment.task_protocol.task_config:
            return refused(
                "not_found",
                f"No task named {task_name!r} in the protocol.",
                task_names=list(experiment.task_protocol.task_config),
            )

        result = self._agent_call(
            self._protocol_edit(workflow_edits.remove_task, experiment, task_name)
        )
        return {**result, "task_names": config.workflow}

    def reorder_tasks(self, order: List[str]) -> Dict[str, Any]:
        """Put the workflow's tasks in *order*, as dragging them on the Workflow
        tab does: every task, each once. Refused when a task would come before
        a task it requires."""
        experiment, config, refusal = self._workflow_edit("reorder tasks")
        if refusal:
            return refusal

        result = self._agent_call(
            self._protocol_edit(workflow_edits.reorder_tasks, experiment, order)
        )
        if not result["applied"]:
            return refused("invalid", result["error"], task_names=config.workflow)
        return {**result, "task_names": config.workflow}

    def set_task_requirements(
        self,
        task_name: str,
        requires: Optional[List[str]] = None,
        required: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Set what a workflow task requires (the tasks an item must have
        completed first) and whether it is required or optional, as the
        Workflow tab's Edit Task does. Refused when a requirement is not an
        earlier task."""
        experiment, config, refusal = self._workflow_edit("edit tasks")
        if refusal:
            return refusal
        protocol = experiment.task_protocol
        task = config.get(task_name)
        if task is None or task_name not in protocol.task_config:
            return refused(
                "not_found",
                f"No task named {task_name!r} in the workflow.",
                task_names=config.workflow,
            )

        result = self._agent_call(
            self._protocol_edit(
                workflow_edits.set_requirements,
                experiment,
                task_name,
                requires,
                required,
            )
        )
        if not result["applied"]:
            return refused("invalid", result["error"])
        return result


def _uncropped_point(target: str, crop: Any, px, fraction):
    """``(px, None)``: the image's pixels under a point read off *crop*'s
    picture, *px* in its pixels or *fraction* of it. Raises ValueError."""
    scale, x0, y0, box = from_crop(target, crop)
    if (px is None) == (fraction is None):
        raise ValueError("Give exactly one of px ([x, y] source pixels) or fraction.")
    if px is not None:
        return [px[0] * scale + x0, px[1] * scale + y0], None
    return [x0 + fraction[0] * box[2], y0 + fraction[1] * box[3]], None


def _checked_marks(marks: Any, image_path: str):
    """``(marks, None)`` checked and placed on the stage, or ``(None, refusal)``."""
    try:
        return checked_marks(marks, image_path), None
    except EvidenceRefused as refusal:
        return None, {"error_type": refusal.error_type, "message": refusal.message}
