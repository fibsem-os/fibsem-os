from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from fibsem import acquire
from fibsem import config as fcfg
from fibsem.cancellation import (
    CANCELLED,
    COMPLETED,
    FAILED,
    SKIPPED,
    AnyStopEvent,
    OperationCancelledError,
    ended_by,
    raise_if_cancelled,
)
from fibsem.devices.runs import Runs, current_run_id
from fibsem.microscope import FibsemMicroscope
from fibsem.milling.base import FibsemMillingStage, MillingStrategyResult
from fibsem.milling.progress import MillingProgress, MillingProgressStatus
from fibsem.milling.steps import MillingStageRun
from fibsem.services.milling import report_progress, stage_running
from fibsem.services.milling import task_running as _task_running
from fibsem.structures import (
    BeamType,
    Configurable,
    Display,
    FibsemImage,
    FibsemRectangle,
    ImageSettings,
    MillingAlignment,
    Point,
    distance_param,
    param,
)
from fibsem.util.timestamps import now_iso
from fibsem.utils import current_timestamp_v3
from fibsem.workflows.config_patch import PatchError, apply_patch


@dataclass
class MillingTaskAcquisitionSettings(Configurable):
    """Settings for the acquisition of images during a milling task."""

    acquire_sem: bool = param(
        False,
        doc="Whether to acquire SEM images between the milling task stages.",
        display=Display("Acquire SEM Image"),
    )
    acquire_fib: bool = param(
        False,
        doc="Whether to acquire FIB images between the milling task stages.",
        display=Display("Acquire FIB Image"),
    )
    acquire_final_image: bool = param(
        True,
        doc="Refresh the FIB view with a single image once the task finishes. "
        "Disable for low-kV polishing, where imaging the lamella with a "
        "higher-voltage beam undoes the polish.",
        display=Display("Acquire Final Image"),
    )
    imaging: ImageSettings = field(default_factory=ImageSettings)

    @property
    def enabled(self) -> bool:
        return self.acquire_sem or self.acquire_fib

    @property
    def estimated_time(self) -> float:
        if not self.enabled:
            return 0.0
        n = sum([self.acquire_sem, self.acquire_fib])
        return self.imaging.estimated_time * n

    def to_dict(self) -> Dict[str, Any]:
        return {
            "acquire_sem": self.acquire_sem,
            "acquire_fib": self.acquire_fib,
            "acquire_final_image": self.acquire_final_image,
            "imaging": self.imaging.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MillingTaskAcquisitionSettings":
        imaging = data.get("imaging", {})
        if imaging == {} or imaging.get("path", None) is None:
            imaging["path"] = None
        return cls(
            acquire_sem=data.get("acquire_sem", False),
            acquire_fib=data.get("acquire_fib", False),
            # default True so protocols written before this flag existed keep the
            # post-task FIB refresh they have always had
            acquire_final_image=data.get("acquire_final_image", True),
            imaging=ImageSettings.from_dict(imaging),
        )


# What a milling task sets on each stage before the stage runs, as stage path: task
# path. A stage's own value there is never used inside a task, so an edit to it at
# the milling question is refused, naming the task's field to set instead.
TASK_SETS_ON_STAGE: Dict[str, str] = {
    "milling.hfw": "field_of_view",
    "milling.milling_channel": "channel",
    "milling.acquire_images": "acquisition.enabled",
    "imaging": "acquisition.imaging",
    "alignment": "alignment",
}


def _get_path(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _set_path(obj: Any, path: str, value: Any) -> None:
    *parents, leaf = path.split(".")
    for part in parents:
        obj = getattr(obj, part)
    setattr(obj, leaf, value)


def task_overridden_path(path: str) -> Optional[str]:
    """Why a dotted path into a milling task config sets nothing, when it is a
    stage field the task sets itself (``stages.0.milling.hfw``); else None."""
    parts = path.split(".")
    if len(parts) < 3 or parts[0] != "stages":
        return None
    within = ".".join(parts[2:])
    for stage_path, task_path in TASK_SETS_ON_STAGE.items():
        if within == stage_path or within.startswith(stage_path + "."):
            return (
                f"the task sets the stage's {stage_path} from its own "
                f"{task_path}; set {task_path} instead"
            )
    return None


class InvalidMillingEdit(ValueError):
    """An edit to a milling config that cannot be made: the message says why."""


def resolve_stage_names(
    config: "FibsemMillingTaskConfig", patch: Dict[str, Any]
) -> Dict[str, Any]:
    """``patch`` with each ``stages.<name>.`` segment as the stage's index, which
    is what :func:`fibsem.workflows.config_patch.apply_patch` resolves.

    A stage is named by its name or its index. A name no stage has, or more
    than one has, is refused with the names there are."""
    names = [stage.name for stage in config.stages]
    resolved: Dict[str, Any] = {}
    for path, value in patch.items():
        head, sep, rest = str(path).partition(".")
        if head != "stages" or not sep:
            resolved[path] = value
            continue
        segment, sep, tail = rest.partition(".")
        if segment.isdigit():
            resolved[path] = value
            continue
        # A stage name may hold dots: match the longest name the path starts with.
        matches = [
            name for name in names if rest == name or rest.startswith(name + ".")
        ]
        if not matches:
            raise InvalidMillingEdit(
                f"{path!r}: no stage named {segment!r}; the stages are {names}."
            )
        name = max(matches, key=len)
        if names.count(name) > 1:
            raise InvalidMillingEdit(
                f"{path!r}: more than one stage is named {name!r}; name it by "
                "its index instead."
            )
        tail = rest[len(name) :]
        resolved[f"stages.{names.index(name)}{tail}"] = value
    return resolved


def patch_milling_config(
    config: "FibsemMillingTaskConfig", fields: Dict[str, Any]
) -> "FibsemMillingTaskConfig":
    """``config`` with ``fields`` set by dotted path, as
    ``item_update_task_config``'s patch sets them, a stage named by its name or
    its index (``stages.Polish 1.pattern.depth``). A new config; ``config``
    itself is not touched. Raises :class:`InvalidMillingEdit` naming what is
    wrong, with nothing changed."""
    from copy import deepcopy

    edited = deepcopy(config)
    patch = resolve_stage_names(edited, fields)
    for path in patch:
        note = task_overridden_path(path)
        if note is not None:
            raise InvalidMillingEdit(f"{path!r}: {note}")
    try:
        apply_patch(edited, patch)
    except PatchError as exc:
        raise InvalidMillingEdit(f"{exc.path!r}: {exc}") from None
    return edited


def describe_config_fields(config: "FibsemMillingTaskConfig") -> Dict[str, Any]:
    """What each field of ``config`` means, for an agent editing it: the task's
    fields and the stage's by dotted path, and each pattern's and strategy's
    fields by its name. Each entry has the field's doc, its SI unit, and
    ``not_used`` (why setting it does nothing) where that applies; the stage
    fields the task sets itself say so. Only fields that say something."""

    def walk(cls: type, prefix: str, out: Dict[str, Any], seen: tuple) -> None:
        from dataclasses import is_dataclass

        from fibsem.structures import field_specs

        if cls in seen:
            return
        for name, spec in field_specs(cls).items():
            path = f"{prefix}{name}"
            entry = {
                key: value
                for key, value in (
                    ("doc", spec.doc),
                    ("unit", spec.unit),
                    ("not_used", spec.not_used),
                )
                if value
            }
            if entry:
                out[path] = entry
            hint = spec.type
            if isinstance(hint, type) and is_dataclass(hint) and not spec.not_used:
                walk(hint, f"{path}.", out, seen + (cls,))

    from fibsem.milling.base import FibsemMillingStage

    task: Dict[str, Any] = {}
    walk(type(config), "", task, ())
    task.pop("stages", None)
    stage: Dict[str, Any] = {}
    walk(FibsemMillingStage, "", stage, ())
    for path in list(stage):
        # Inside a section the task sets: the section says so, below.
        if path not in TASK_SETS_ON_STAGE and task_overridden_path(f"stages.0.{path}"):
            del stage[path]
    for path in TASK_SETS_ON_STAGE:
        note = task_overridden_path(f"stages.0.{path}")
        stage[path] = {**stage.get(path, {}), "not_used": note}
    patterns: Dict[str, Any] = {}
    strategies: Dict[str, Any] = {}
    for each in config.stages:
        pattern, strategy = each.pattern, each.strategy
        if pattern.name not in patterns:
            walk(type(pattern), "", patterns.setdefault(pattern.name, {}), ())
        if strategy.strategy_name not in strategies:
            walk(
                type(strategy),
                "",
                strategies.setdefault(strategy.strategy_name, {}),
                (),
            )
    return {
        "task": task,
        "stage": stage,
        "pattern": patterns,
        "strategy": strategies,
    }


@dataclass
class FibsemMillingTaskConfig(Configurable):
    """Configuration for a milling task."""

    name: str = "Milling Task"
    field_of_view: float = param(
        150e-6,
        distance_param(
            "Field of view", "The FIB field of view the patterns mill in.", decimals=1
        ),
    )
    channel: BeamType = BeamType.ION
    alignment: MillingAlignment = field(default_factory=MillingAlignment)
    acquisition: MillingTaskAcquisitionSettings = field(
        default_factory=MillingTaskAcquisitionSettings
    )
    stages: List[FibsemMillingStage] = field(default_factory=list)

    @property
    def enabled_stages(self) -> List[FibsemMillingStage]:
        """Return only stages where enabled is True."""
        return [s for s in self.stages if s.enabled]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "field_of_view": self.field_of_view,
            "channel": self.channel.name,
            "alignment": self.alignment.to_dict(),
            "acquisition": self.acquisition.to_dict(),
            "stages": [stage.to_dict(short=True) for stage in self.stages],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FibsemMillingTaskConfig":
        alignment = data.get("alignment", {})
        acquisition = data.get("acquisition", {})
        return cls(
            name=data.get("name", "Milling Task"),
            field_of_view=data.get("field_of_view", 150e-6),
            channel=BeamType[data.get("channel", BeamType.ION.name)],
            alignment=MillingAlignment.from_dict(alignment),
            acquisition=MillingTaskAcquisitionSettings.from_dict(acquisition),
            stages=[
                FibsemMillingStage.from_dict(stage) for stage in data.get("stages", [])
            ],
        )

    @classmethod
    def from_stages(
        cls, stages: List[FibsemMillingStage], name: str = "Milling Task"
    ) -> "FibsemMillingTaskConfig":
        """Create a FibsemMillingTaskConfig from a list of FibsemMillingStage."""

        if not stages:
            raise ValueError("No milling stages provided to create task config.")

        # Use the first stage's properties as defaults
        first_stage = stages[0]
        return FibsemMillingTaskConfig(
            name=name,
            field_of_view=first_stage.milling.hfw,
            channel=first_stage.milling.milling_channel,
            alignment=first_stage.alignment,
            acquisition=MillingTaskAcquisitionSettings(
                acquire_sem=first_stage.milling.acquire_images,
                acquire_fib=first_stage.milling.acquire_images,
                imaging=first_stage.imaging,
            ),
            stages=stages,
        )

    @property
    def estimated_time(self) -> float:
        """Estimate the total milling time for a list of milling stages.

        Over the enabled stages only, matching what run() will actually mill.
        Summing every stage counted work that was switched off: a real task with
        two of its three stages disabled estimated 845 s against ~102 s milled.
        """
        milling_time = sum(stage.estimated_time for stage in self.enabled_stages)
        return milling_time + self.acquisition.estimated_time

    def compatible_stages(
        self, reference_idx: int = 0
    ) -> List[Tuple[int, FibsemMillingStage]]:
        """Return stages whose milling settings & strategy match the stage at reference_idx."""
        if not self.stages:
            return []

        if reference_idx < 0 or reference_idx >= len(self.stages):
            raise IndexError(
                f"reference_idx {reference_idx} out of range for {len(self.stages)} stages."
            )

        reference_stage = self.stages[reference_idx]
        compatible: List[Tuple[int, FibsemMillingStage]] = []

        for idx, stage in enumerate(self.stages):
            if idx != reference_idx and stage.is_compatible_with(reference_stage):
                compatible.append((idx, stage))

        return compatible

    def merge_compatible_stages(self, reference_idx: int = 0) -> "FibsemMillingStage":
        compat_stages = self.compatible_stages(reference_idx=reference_idx)
        logging.info(
            f"Compatible Stages: {[(idx, stage.name) for idx, stage in compat_stages]}"
        )

        reference_stage = self.stages[reference_idx]
        if compat_stages:
            reference_stage.patterns = [reference_stage.pattern]
            for idx, stage in compat_stages:
                reference_stage.patterns.append(self.stages[idx].pattern)

        return reference_stage


# TODO: restore current to initial current, rather than system.ion.beam.beam_current
# TODO: support customising alignment imaging parameters?
@dataclass
class FibsemMillingTask:
    config: FibsemMillingTaskConfig = field(default_factory=FibsemMillingTaskConfig)
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    reference_image: Optional[FibsemImage] = None
    reference_image_full_frame: Optional[FibsemImage] = None

    def __init__(
        self,
        microscope: FibsemMicroscope,
        config: FibsemMillingTaskConfig,
        stop_event: Optional[threading.Event] = None,
        output_dir: Optional[str] = None,
    ):
        self.config = config
        self.microscope = microscope
        self.task_id = str(uuid.uuid4())
        # The folder the caller gave this run; the task writes under
        # <output_dir>/Milling/<name>, resolved by _configure_path into
        # self.output_dir. None: the config's imaging path, then DATA_CC_PATH.
        self._output_root: Optional[str] = (
            str(output_dir) if output_dir is not None else None
        )
        self.output_dir: Optional[str] = None
        self.initial_beam_shift: Optional[Point] = None
        # Set to cancel: checked between stages, and passed to the strategy, whose
        # run_milling stops the beam when it is set.
        self._stop_event: Optional[threading.Event] = stop_event
        # Set by the milling service's ``stop`` while this task runs, so a stop of
        # the service ends the task, not only the stage milling then (FIB-1398).
        self._stopped = threading.Event()
        # How the last run() ended, for callers that never see the progress signal.
        # None until run() has finished.
        self.outcome: Optional[MillingProgressStatus] = None
        self.error: Optional[BaseException] = None
        # How each stage the last run() reached ended, for its recorded event.
        self._stage_records: List[Dict[str, Any]] = []
        # The images the last run() saved, in the order it took them.
        self.images: List[FibsemImage] = []

    @property
    def _stop_token(self) -> Union[threading.Event, AnyStopEvent]:
        """What stops the task: the caller's stop event, or the service's stop."""
        if self._stop_event is None:
            return self._stopped
        return AnyStopEvent(self._stopped, self._stop_event)

    @property
    def name(self) -> str:
        """Return the name of the milling task."""
        return self.config.name

    @property
    def stages(self) -> List[FibsemMillingStage]:
        """Return the list of milling stages."""
        return self.config.enabled_stages

    def _emit(self, status: MillingProgressStatus, **fields) -> None:
        """Report where this task has got to, stamped with the task, on the milling
        service's ``progress``, which the microscope's milling progress signal sends on.
        """
        report_progress(
            self.microscope,
            MillingProgress(
                status=status,
                milling_run_id=self.task_id,
                milling_task_name=self.name,
                **fields,
            ),
        )

    def _configure_path(self) -> None:
        """Configure the acquisition path for the milling task.

        Everything the task saves hangs off this path: the alignment reference image,
        the drift-correction run, and the per-stage images. The path is unset by
        default — a config built from stages, or loaded from a protocol that doesn't
        name one, carries ``None`` — so fall back to the cross-correlation data
        directory, as every other acquisition in this path already does. Stringifying
        the ``None`` instead put the whole task under a directory literally named
        "None", beside the current working directory.

        A folder passed to the task wins over the imaging path: the imaging path is
        an imaging setting that happened to also mean "where to save". The result is
        ``self.output_dir``, which each stage is handed before its strategy runs. The
        imaging path is still set to it, for strategies written before
        ``stage.output_dir`` existed.
        """
        path = self._output_root
        if path is None:
            path = self.config.acquisition.imaging.path
        if path is None:
            path = fcfg.DATA_CC_PATH
        self.output_dir = os.path.join(
            str(path),
            "Milling",
            self.name.replace(" ", "-"),
        )
        self.config.acquisition.imaging.path = self.output_dir

    def run(self) -> None:
        """Run a list of milling stages, with a progress bar and notifications.

        The task is one run on the milling service (``microscope.milling.runs``)
        under its own id, whoever runs it: a workflow task, the operator, the
        agent. Inside a run already going on there (the App Server's, the
        service's ``run_task``) it is part of that run, and takes its id.
        """
        milling = getattr(self.microscope, "milling", None)
        runs = getattr(milling, "runs", None)
        if not isinstance(runs, Runs):
            self._run()
            return
        with runs.tracking("run_task", run_id=self.task_id) as run:
            self.task_id = current_run_id(milling.name) or self.task_id
            try:
                self._run()
            finally:
                if run is not None:
                    # The task catches its own failure: its run says it.
                    run.result, run.exception = self.ended(), self.error

    def _run(self) -> None:
        logging.info(f"Running milling task: {self.name} with ID: {self.task_id}")

        # Read by the `finally` below, which runs whichever way the task ends. Seeded
        # with the failure case because `except Exception` does not catch everything:
        # a KeyboardInterrupt unwinds straight through to the `finally`, and reporting
        # that as a completed mill is the defect this whole block exists to fix.
        outcome = MillingProgressStatus.TASK_FAILED
        error: Optional[str] = None
        self.outcome, self.error = None, None
        self._stage_records = []
        self.images = []
        started_at = now_iso()

        self._emit(MillingProgressStatus.TASK_STARTED, total_stages=len(self.stages))

        try:
            self.initial_beam_shift = self.microscope.get_beam_shift(
                beam_type=self.config.channel
            )

            # configure acquisition filepaths
            self._configure_path()

            # acquire a reference image for alignment
            if self.config.alignment.enabled:
                self._acquire_reference_image()

            # acquire an image before starting the task
            if self.config.acquisition.enabled:
                self._acquire_milling_task_images(stage_name=self.name, tag="pre-task")

            # A stop of the milling service stops the whole task, not only the
            # pattern milling now.
            with _task_running(self.microscope, self._stopped):
                for idx, stage in enumerate(self.stages):
                    self._mill_stage(stage, idx)

            outcome = MillingProgressStatus.TASK_FINISHED
        except OperationCancelledError as e:
            logging.info(f"Milling task '{self.name}' cancelled by user: {e}")
            outcome = MillingProgressStatus.TASK_CANCELLED
        except Exception as e:
            logging.error(f"Milling task '{self.name}' failed: {e}", exc_info=True)
            outcome = MillingProgressStatus.TASK_FAILED
            error = str(e)
            self.error = e
        finally:
            self.outcome = outcome
            # The outcome, not just "it stopped". Both `except` blocks used to log and
            # fall through to a `finally` that emitted `finished` regardless, so a mill
            # the user cancelled and a mill that crashed both told the UI they had
            # finished -- the status bar rendered "Done" either way and the exception
            # text reached only the logfile.
            self._emit(outcome, error=error)
            # Before the cleanup, which can raise: the record of how the run ended
            # must not depend on restoring the beam afterwards.
            self._record_run(started_at)
            # put the beam back as the first setup_milling found it, even when cancelled
            self.microscope.finish_milling()
            # restore initial beam shift
            if self.initial_beam_shift is not None:
                self.microscope.set_beam_shift(
                    self.initial_beam_shift, beam_type=self.config.channel
                )

            # No beam after a stop or a failure (FIB-1430, FIB-1508).
            if outcome is MillingProgressStatus.TASK_FINISHED:
                self._post_task_acquisition()

    def _post_task_acquisition(self) -> None:
        """Acquire an image after finishing the milling task."""
        try:
            # refresh the view with a single image if the task didn't already acquire one.
            # NB: acquisition.enabled is (acquire_sem or acquire_fib), so it subsumes the
            # acquire_fib check this condition used to carry.
            if (
                self.config.acquisition.acquire_final_image
                and not self.config.acquisition.enabled
            ):
                self.microscope.autocontrast(beam_type=self.config.channel)
                fib_image = acquire.acquire_current_image(
                    self.microscope, self.config.channel
                )
                if self.microscope.imaging is not None:
                    self.microscope.imaging.show(fib_image)
        except Exception as e:
            logging.error(f"Error acquiring image after milling task: {e}")

    def _mill_stage(self, stage: FibsemMillingStage, idx: int) -> None:
        """Run a single milling stage with progress updates.

        Raises whatever the stage raises. run() ends the task there, failed or
        cancelled, and skips the later stages: they build on this one's geometry
        (rough then polish). Catching here used to log a stage the microscope
        rejected and report the task finished with nothing milled (FIB-1112).

        Args:
            stage (FibsemMillingStage): The milling stage to run.
            idx (int): The index of the milling stage.
        """

        start_time = time.time()
        raise_if_cancelled(self._stop_token)
        # Reached: from here the stage ends completed, failed or cancelled, never
        # skipped (FIB-1257).
        record: Dict[str, Any] = {
            "name": stage.name,
            "index": idx,
            "milling_current": stage.milling.milling_current,
            "started_at": now_iso(),
            # which strategy milled it, and what it said about the run
            "strategy": {"name": stage.strategy.strategy_name},
        }
        self._stage_records.append(record)
        with stage_running(self.microscope, stage) as stage_run:
            error: Optional[BaseException] = None
            try:
                result = self._run_stage(stage, idx, start_time)
                # Before the cancel check: a stopped stage keeps what its
                # strategy said.
                record["strategy"].update(_strategy_result(stage, result))
                # A stop of the service while the stage milled: the run answered
                # cancelled, which a strategy need not pass on.
                raise_if_cancelled(self._stopped, "Milling stopped.")
            except BaseException as e:
                error = e
                record.update(ended_by(e), ended_at=now_iso())
                raise
            finally:
                record.update(_stage_run_ended(stage_run, error))
                record.update(_current_used(self.microscope))
        record.update(status=COMPLETED, ended_at=now_iso())

        # optionally acquire images after milling
        if self.config.acquisition.enabled:
            self._acquire_milling_task_images(
                stage_name=f"{self.name}-{stage.name}", tag="finished"
            )

        self._emit(
            MillingProgressStatus.STAGE_FINISHED,
            start_time=start_time,
            current_stage=idx,
            total_stages=len(self.stages),
            stage_name=stage.name,
        )

    def _run_stage(
        self, stage: FibsemMillingStage, idx: int, start_time: float
    ) -> Optional[MillingStrategyResult]:
        """Set the stage up from the task and mill it; what the strategy said."""
        self._emit(
            MillingProgressStatus.STAGE_STARTED,
            start_time=start_time,
            current_stage=idx,
            total_stages=len(self.stages),
            stage_name=stage.name,
        )

        # if self.config.acquisition.enabled:
        # self._acquire_milling_task_images(stage_name=stage.name, tag="start")

        # Set up the stage with the task configuration
        stage.reference_image = self.reference_image
        stage.reference_image_full_frame = self.reference_image_full_frame
        stage.output_dir = self.output_dir
        for stage_path, task_path in TASK_SETS_ON_STAGE.items():
            _set_path(stage, stage_path, _get_path(self.config, task_path))
        self._record_stage_started(stage, idx)
        result = stage.get_strategy().run(
            microscope=self.microscope,
            stage=stage,
            stop_event=self._stop_token,
        )
        # TODO: pass task as parent into strategy.run()?, allow logging from strategy?
        # performance logging
        msgd = {
            "msg": "milling_task",
            "milling_task_id": self.task_id,
            "milling_task_name": self.name,
            "idx": idx,
            "stage": stage.to_dict(),
            "start_time": start_time,
            "end_time": time.time(),
            "timestamp": now_iso(),
        }
        logging.debug(f"{msgd}")
        return result

    def _record_run(self, started_at: str) -> None:
        """Record how this run ended: one ``milling`` event, with how each stage
        ended (FIB-1257).

        Each run is its own event, so a task the operator ran twice records two.
        Stages are in the protocol's order, each with a status: a stage the run
        never reached, or one turned off in the protocol, is ``skipped`` with
        why, and a run with every stage turned off is ``skipped`` too. Never
        raises: a run that cannot be described still milled.
        """
        try:
            payload = {
                "milling_run_id": self.task_id,
                "milling_task_name": self.name,
                "started_at": started_at,
                **self.ended(),
            }
        except Exception:  # noqa: BLE001 - recording must not matter
            logging.debug(f"could not record the {self.name} run", exc_info=True)
            return
        self.microscope.record_event("milling", payload)

    def ended(self) -> Dict[str, Any]:
        """How the run ended, as its ``milling`` event says: its status (and
        the error's type and message when it failed), the folder it saved
        into, and how each stage ended."""
        if self.error is not None:
            ended = ended_by(self.error)
        else:
            ended = {"status": _RUN_STATUS.get(self.outcome, FAILED)}
        if ended["status"] == COMPLETED and not self.stages:
            ended = {"status": SKIPPED, "reason": _TURNED_OFF}
        return {
            **ended,
            "path": self.output_dir,
            "stages": self._stages_as_ended(ended["status"]),
        }

    def _stages_as_ended(self, status: str) -> List[Dict[str, Any]]:
        """Every stage in the protocol, as the run left it. ``index`` is the
        stage's place among the enabled ones, as ``milling_stage_started`` has it."""
        reached = {record["index"]: record for record in self._stage_records}
        why = (
            "the run was stopped before this stage"
            if status == CANCELLED
            else "the run failed before this stage"
        )
        stages, index = [], 0
        for stage in self.config.stages:
            if not stage.enabled:
                stages.append(
                    {"name": stage.name, "status": SKIPPED, "reason": _TURNED_OFF}
                )
                continue
            record = reached.get(index)
            if record is not None:
                stages.append(dict(record))
            else:
                stages.append(
                    {
                        "name": stage.name,
                        "index": index,
                        "status": SKIPPED,
                        "reason": why,
                    }
                )
            index += 1
        return stages

    def _record_stage_started(self, stage: FibsemMillingStage, idx: int) -> None:
        """Record what a stage is about to mill, for the experiment's record.

        The stage as this task configured it, patterns included. The log records a
        stage only once it has finished, so a reader had to subtract the duration
        to place the start; the end is already on the progress signal
        (STAGE_FINISHED). Never raises: a stage that cannot be described still mills.
        """
        try:
            payload = {
                "milling_run_id": self.task_id,
                "milling_task_name": self.name,
                "stage_index": idx,
                "total_stages": len(self.stages),
                "stage": stage.to_dict(),
            }
        except Exception:  # noqa: BLE001 - recording must not matter
            logging.debug(
                f"could not describe {stage.name} for the record", exc_info=True
            )
            return
        self.microscope.record_event("milling_stage_started", payload)

    def _acquire_reference_image(self) -> Optional[FibsemImage]:
        """Acquire the reference images for the milling task: the alignment
        area, and the whole frame too when a stage may align full frame."""

        path = self.output_dir
        if path is None:
            path = Path(fcfg.DATA_CC_PATH)

        if self.reference_image is None:
            self.reference_image = self._acquire_alignment_reference(
                path, reduced_area=self.config.alignment.rect
            )
        if (
            self.config.alignment.full_frame_below > 0
            and self.reference_image_full_frame is None
        ):
            self.reference_image_full_frame = self._acquire_alignment_reference(
                path, reduced_area=None, tag="full-frame"
            )
        return self.reference_image

    def _acquire_alignment_reference(
        self,
        path,
        reduced_area: Optional[FibsemRectangle],
        tag: str = "",
    ) -> FibsemImage:
        """One alignment reference at the task's field of view, at the current
        the beam is at now (the imaging current, before any stage sets its own)."""
        filename = f"{self.name}_{fcfg.REFERENCE_FILENAME}"
        if tag:
            filename = f"{filename}_{tag}"
        filename = f"{filename}_{current_timestamp_v3(timeonly=True)}".replace(" ", "-")
        image_settings = ImageSettings(
            hfw=self.config.field_of_view,
            dwell_time=self.config.alignment.imaging.dwell_time,
            resolution=self.config.alignment.imaging.resolution,
            beam_type=self.config.channel,
            reduced_area=reduced_area,
            save=True,
            path=path,
            filename=filename,
        )
        return acquire.acquire_image(
            microscope=self.microscope, settings=image_settings
        )

    def _acquire_milling_task_images(
        self,
        stage_name: str,
        tag: str = "finished",
    ) -> None:
        """Acquire images after milling for reference.
        Args:
            stage_name (str): Name of the milling stage
            tag (str): Tag to append to the filename
        """
        self.microscope.finish_milling()

        acq_date = current_timestamp_v3(timeonly=True)
        self.config.acquisition.imaging.filename = (
            f"{stage_name}_{tag}_{acq_date}".replace(" ", "-")
        )
        self.config.acquisition.imaging.save = True
        self.config.acquisition.imaging.hfw = (
            self.config.field_of_view
        )  # force field of view to match task

        if self.config.acquisition.imaging.path is None:
            # Where this run saves, not wherever the microscope's last image went.
            self._configure_path()

        # support specifying acquiring sem and/or fib images only
        sem_image, fib_image = acquire.acquire_channels(
            microscope=self.microscope,
            image_settings=self.config.acquisition.imaging,
            acquire_sem=self.config.acquisition.acquire_sem,
            acquire_fib=self.config.acquisition.acquire_fib,
        )
        self.images.extend(
            image for image in (sem_image, fib_image) if image is not None
        )

        try:
            imaging = self.microscope.imaging
            for image in (sem_image, fib_image):
                if image is not None and imaging is not None:
                    imaging.show(image)
        except Exception as e:
            logging.error(f"Error showing the images after milling: {e}")


def _strategy_result(stage: FibsemMillingStage, result: Any) -> Dict[str, Any]:
    """What the stage's strategy said about its run, as its record holds it.
    Anything but a MillingStrategyResult or None is a plugin's mistake: logged
    and left out, never a reason to fail a mill that finished."""
    if result is None:
        return {}
    if not isinstance(result, MillingStrategyResult):
        logging.warning(
            f"{stage.strategy.strategy_name} returned {type(result).__name__} for "
            f"{stage.name}, not a MillingStrategyResult; it is not recorded"
        )
        return {}
    try:
        return result.to_dict()
    except Exception:  # noqa: BLE001 - recording must not matter
        logging.debug(f"could not record {stage.name}'s strategy result", exc_info=True)
        return {}


def _stage_run_ended(
    stage_run: Optional[MillingStageRun], error: Optional[BaseException]
) -> Dict[str, Any]:
    """How a stage run ended and its mill steps, as its record holds them:
    ``end`` beside its status, and ``steps``. Never raises."""
    if not isinstance(stage_run, MillingStageRun):
        stage_run = MillingStageRun()
    try:
        return {
            "end": stage_run.ended(error),
            "steps": [step.to_dict() for step in stage_run.steps],
        }
    except Exception:  # noqa: BLE001 - recording must not matter
        logging.debug("could not record how a stage run ended", exc_info=True)
        return {}


def _current_used(microscope: FibsemMicroscope) -> Dict[str, Any]:
    """The current a stage milled at, as the beam read back after its setup
    (``milling_current_used``), and ``current_warning`` when the beam snapped the
    current asked for to the nearest it lists. Never raises."""
    try:
        answer = microscope.milling.current_used()
    except Exception:  # noqa: BLE001 - recording must not matter
        logging.debug("could not read the milling current used", exc_info=True)
        return {}
    used = {}
    if answer.get("milling_current") is not None:
        used["milling_current_used"] = answer["milling_current"]
    if answer.get("warning"):
        used["current_warning"] = answer["warning"]
        logging.warning(answer["warning"])
    return used


# How a run ended, as its recorded event says it.
_RUN_STATUS = {
    MillingProgressStatus.TASK_FINISHED: COMPLETED,
    MillingProgressStatus.TASK_CANCELLED: CANCELLED,
    MillingProgressStatus.TASK_FAILED: FAILED,
}
# Why a stage, or a whole task, did not mill when the protocol said not to.
_TURNED_OFF = "turned off in the protocol"


def record_milling_skipped(
    microscope: FibsemMicroscope, config: FibsemMillingTaskConfig, reason: str
) -> None:
    """Record a milling task that did not run, with why: a ``milling`` event like
    a run's, every stage ``skipped`` (FIB-1257). Never raises."""
    try:
        payload = {
            "milling_task_name": config.name,
            "status": SKIPPED,
            "reason": reason,
            "stages": [
                {"name": stage.name, "status": SKIPPED} for stage in config.stages
            ],
        }
    except Exception:  # noqa: BLE001 - recording must not matter
        logging.debug("could not record a skipped milling task", exc_info=True)
        return
    microscope.record_event("milling", payload)


def run_milling_task(
    microscope: FibsemMicroscope,
    config: FibsemMillingTaskConfig,
    stop_event: Optional[threading.Event] = None,
    raise_on_failure: bool = False,
    output_dir: Optional[str] = None,
) -> FibsemMillingTask:
    """Run a milling task with the given configuration.
    Args:
        microscope (FibsemMicroscope): The microscope to use for milling.
        config (FibsemMillingTaskConfig): The configuration for the milling task.
        stop_event (Optional[threading.Event]): Set to cancel the run.
        raise_on_failure (bool): Re-raise a failed task's error, after its cleanup
            has run. For callers that don't listen to the progress signal, which
            is the only other place a failure is reported. A cancelled task
            does not raise.
        output_dir (Optional[str]): The folder this run writes under, as
            ``<output_dir>/Milling/<name>``. An argument rather than a config
            field: the config is saved with the protocol, and a run's folder is
            not. None falls back to the config's imaging path.
    Returns:
        FibsemMillingTask: The milling task that was run; ``outcome`` and
            ``error`` say how it ended.
    """
    task = FibsemMillingTask(
        microscope=microscope,
        config=config,
        stop_event=stop_event,
        output_dir=output_dir,
    )
    task.run()
    if raise_on_failure and task.error is not None:
        raise task.error
    return task
