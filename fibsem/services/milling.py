"""Milling as a service: a recipe and patterns handed to the instrument, run, and the
beam put back as it was found.

`Milling` is what every backend shares. A driver subclasses it and implements the
hooks (``_setup``, ``_start``, ``_stop``, ``_pause``, ``_resume``, ``_estimate``,
``_clear`` and ``read_state``) the way the instrument mills: an instrument queue on
ThermoFisher and Odemis, a DrawBeam layer on Tescan, a list on the Demo.

Drawing is dispatched here, by pattern type, to one hook per type the driver draws
(``_draw_rectangle``, ``_draw_line``, ``_draw_circle``, ``_draw_bitmap``,
``_draw_polygon``). The types a driver draws are the hooks it implements, and
``supported_pattern_types`` lists them. ``draw`` checks every pattern before drawing
any: a type the driver doesn't draw raises `UnsupportedPattern` with nothing drawn,
rather than milling without it.

The driver also chooses the per-pattern application file and applies the recipe's
beam conditions, since which recipe fields a backend uses differs (Tescan mills with
a preset, the others with a current and voltage).

What it does the same everywhere is put the beam back. The first ``setup`` saves the
milling beam's conditions (its preset where it has one, voltage, current and field of
view), and ``restore`` writes them back; ``finish_milling`` restores, so the beam ends
as milling found it on every backend, not as the caller guessed it was.

Which recipe fields an instrument mills with differs too, and ``supported_settings``
says: the fields its driver's ``setup`` reads, each with its choices and limits. A
field that is a beam parameter (current, voltage, preset, hfw) has the beam's own
metadata; the application file and the rest have what the driver reports. The milling
form shows those fields and no others.

``run`` is the one run loop every backend shares: start, look at the instrument about
once a second, report ``progress``, and clear the patterns at the end. What differs is
small and is a driver hook: what one look reads (``_poll``: the state, and the total
and remaining time where the instrument reports them, as Tescan's DrawBeam does),
and what happens around the run (``_before_run``, ``_after_run``: Tescan loads its
layer and shows a progress bar). Elapsed time the instrument doesn't report is the wall
clock while running, paused time left out. A set ``stop_event`` stops the beam; a
failure stops it and clears the patterns before it is raised.

`FibsemMicroscope`'s milling methods (``setup_milling``, ``draw_patterns``,
``start_milling``, ``run_milling`` and the rest) go to the microscope's service,
``microscope.milling``, so they keep their signatures on every backend.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import threading
import time
import uuid
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from psygnal import Signal

# At import, not under TYPE_CHECKING: the App Server decodes a command's arguments
# from its type hints (a FibsemMillingSettings sent as a dict).
from fibsem.cancellation import (
    CANCELLED,
    COMPLETED,
    AnyStopEvent,
    NothingToStop,
    ended_by,
    raise_if_cancelled,
)
from fibsem.devices.beam import Beam
from fibsem.devices.core import DeviceParameter, ParameterMetadata, Role, command, query
from fibsem.devices.runs import current_run_id
from fibsem.milling.base import FibsemMillingStage, estimate_milling_time
from fibsem.milling.progress import MillingProgress, MillingProgressStatus
from fibsem.milling.steps import (
    ENDED,
    FINISHED,
    STEP_FAILED,
    STOPPED,
    MillingStageRun,
    MillingStep,
    MillingStepEnd,
)
from fibsem.services.core import Service, save_beam_conditions
from fibsem.structures import (
    ACTIVE_MILLING_STATES,
    BeamType,
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemImage,
    FibsemLineSettings,
    FibsemMillingSettings,
    FibsemPatternSettings,
    FibsemPolygonSettings,
    FibsemRectangleSettings,
    MillingState,
    field_specs,
)
from fibsem.util.timestamps import now_iso

# The beam conditions milling changes and `restore` puts back, in the order they are
# written back: a preset first, since it sets the others on a column that has one,
# then the voltage before the current, as the instruments want.
SAVED_BEAM_CONDITIONS = ("preset", "voltage", "current", "hfw")

# What a stage update leaves out and the run's own reports said: which run, which
# task config, which stage (FIB-1356). A backend's poll loop and a strategy's tick
# know only the countdown.
_RUN_CONTEXT = (
    "milling_run_id",
    "milling_task_name",
    "stage_name",
    "current_stage",
    "total_stages",
)

# How a run outside a task ended, as the terminal report that ends its progress.
_ENDED_AS = {
    COMPLETED: MillingProgressStatus.TASK_FINISHED,
    CANCELLED: MillingProgressStatus.TASK_CANCELLED,
}

_M = TypeVar("_M", bound="Milling")

# The hook that draws each pattern type. A driver draws the types whose hook it
# implements.
DRAW_HOOKS: Dict[Type[FibsemPatternSettings], str] = {
    FibsemRectangleSettings: "_draw_rectangle",
    FibsemLineSettings: "_draw_line",
    FibsemCircleSettings: "_draw_circle",
    FibsemBitmapSettings: "_draw_bitmap",
    FibsemPolygonSettings: "_draw_polygon",
}


class UnsupportedPattern(NotImplementedError):
    """A pattern of a type this instrument's driver doesn't draw."""


class MillingStepFailed(RuntimeError):
    """A monitor failed the mill step: a required monitor's error, or a pause
    longer than its ``max_pause``."""


class Milling(Service):
    """Pattern milling with the ion beam, or the electron beam when a recipe asks.

    ``setup`` takes the recipe (`FibsemMillingSettings`), ``draw`` the patterns, and
    ``prepare`` both. Then ``start``, ``pause``, ``resume`` and ``stop`` run it, and
    ``state`` says where it is. ``restore`` puts the milling beam back as the first
    ``setup`` found it.
    """

    ion = Role(Beam, doc="The ion beam, which mills unless a recipe asks otherwise.")
    electron = Role(Beam, required=False, doc="The electron beam, on a dual beam.")

    # Read only when asked: on ThermoFisher a read selects the milling view, which a
    # caller holding the view for something else (coincidence milling) must not have
    # done behind its back, so the commands don't read it after themselves.
    state = DeviceParameter(MillingState, doc="Idle, running, paused, ...; read-only.")
    # Reported by `run` as it goes; reading it doesn't touch the instrument.
    # A stage update (`MillingProgress`) with what a run knows: its state, start,
    # total and remaining time. The task and stage fields are a task's to fill.
    progress = DeviceParameter(
        MillingProgress, doc="The run's state and times; read-only."
    )

    # The patterns the next run mills, as drawn: the list so far after each
    # ``draw``, and empty when setup, ``clear`` or the end of a run removes them.
    # The app draws an agent's mill from it (FIB-1375), which it has no config of.
    patterns_changed = Signal(object)

    # How often `run` looks at the instrument, in seconds.
    poll_interval: float = 1.0
    # How long after the start an idle instrument still means "not running yet"
    # rather than "already finished", in seconds.
    start_timeout: float = 5.0
    # How long `run` waits for the beam to stop after its stop event is set.
    stop_timeout: float = 30.0

    # The `FibsemMillingSettings` fields this driver's ``setup`` reads; it ignores
    # the rest. A driver sets it.
    setting_names: Tuple[str, ...] = ()
    # The directions this driver can scan a pattern in. A driver sets it.
    scan_directions: Tuple[str, ...] = ()

    def __init__(self, name: str = "milling", **kwargs: Any):
        super().__init__(name, **kwargs)
        # The beam conditions the first setup found, until restore puts them back.
        self._saved: Optional[Dict[str, Any]] = None
        self._saved_beam: Optional[Beam] = None
        # What the driver says about its own fields, asked once.
        self._driver_settings: Dict[str, ParameterMetadata] = {}
        self._progress = progress_update(state=MillingState.IDLE)
        # The run and stage the last task or stage start named, for the stage
        # updates that follow it; empty outside a milling task.
        self._run_context: Dict[str, Any] = {}
        self._drawn: List[FibsemPatternSettings] = []
        # The mill step running now, and the stage run it is part of, if any.
        self._step: Optional[MillingStep] = None
        self._stage_run: Optional[MillingStageRun] = None
        # The stop of the milling task running on this service, if one is: ``stop``
        # sets it, so the task starts no later stage (FIB-1398).
        self._task_stop: Optional[threading.Event] = None
        # A run is in progress, for ``stop`` to know there is something to stop.
        self._running = False
        # The name the last setup was given, for a run outside a task to go by.
        self._name: Optional[str] = None
        # The recipe the last setup applied, for the answers to say what it mills at.
        self._set_up: Optional[FibsemMillingSettings] = None
        # Inside ``part_of``: a run there is part of something with its own record.
        self._part_of = threading.local()

    @property
    def drawn_patterns(self) -> List[FibsemPatternSettings]:
        """What ``draw`` has drawn since the patterns were last removed."""
        return list(self._drawn)

    def _set_drawn(self, patterns: Sequence[FibsemPatternSettings]) -> None:
        self._drawn = list(patterns)
        try:
            self.patterns_changed.emit(list(self._drawn))
        except Exception:  # noqa: BLE001 - showing them must not cost the mill
            logging.debug("could not announce the drawn patterns", exc_info=True)

    def connect(self: _M) -> _M:
        super().connect()
        # Read once, so that every report from a run is a change and signals.
        self.progress.get_value()
        return self

    def read_progress(self) -> MillingProgress:
        return self._progress

    def _report(self, progress: MillingProgress) -> None:
        progress = self._in_run_context(progress)
        report = getattr(self._part_of, "report", None)
        if report is not None:
            # A run that is part of something else reports to it, not as a mill.
            report(progress)
            return
        self._progress = progress
        self.progress.report(progress)

    def _in_run_context(self, progress: MillingProgress) -> MillingProgress:
        """``progress``, a stage update given the run and stage it belongs to.

        A task's start and each stage's start name them; the ticks in between,
        from a backend or a strategy, do not, so a reader of the event stream
        saw a stage update with every name null. A milling run outside a task
        (a bare ``run``) has nothing to fill in, and its updates stay as they are.
        """
        status = progress.status
        if status in (
            MillingProgressStatus.TASK_STARTED,
            MillingProgressStatus.STAGE_STARTED,
        ):
            self._run_context = {
                name: getattr(progress, name)
                for name in _RUN_CONTEXT
                if getattr(progress, name) is not None
            }
        elif status.is_terminal:
            self._run_context = {}
        elif status is MillingProgressStatus.STAGE_UPDATE and self._run_context:
            missing = {
                name: value
                for name, value in self._run_context.items()
                if getattr(progress, name) is None
            }
            if missing:
                progress = dataclasses.replace(progress, **missing)
        return progress

    def report(self, progress: MillingProgress) -> None:
        """Report ``progress`` for whoever is driving the mill: a strategy or a task
        with words of its own, or its own countdown (coincidence milling). The
        app's progress bars listen on ``progress``, so this is how a report reaches
        them."""
        self._report(progress)

    def beam(self, channel: BeamType = BeamType.ION) -> Beam:
        """The beam that mills on ``channel``."""
        return self.electron if channel is BeamType.ELECTRON else self.ion

    def supported_settings(
        self, channel: BeamType = BeamType.ION
    ) -> Dict[str, ParameterMetadata]:
        """The recipe (`FibsemMillingSettings`) fields this instrument mills with on
        ``channel``, each with its choices and limits; the fields left out it ignores.

        A field bound to one of the milling beam's parameters
        (milling current and voltage, preset, hfw) has that parameter's own metadata, and is left out when that beam doesn't
        have it or can't set it. The others (application file, patterning mode,
        Tescan's rate and dwell, ...) have what the driver reports. A channel this
        instrument doesn't mill on (no beam, or not one the driver mills with) has
        no settings.
        """
        if channel not in (self._metadata_of("milling_channel").choices or ()):
            return {}
        beam = self.beam(channel)
        available: Dict[str, ParameterMetadata] = {}
        specs = FibsemMillingSettings.params
        for name in self.setting_names:
            beam_parameter = specs[name].bound_name
            if beam_parameter not in beam.parameters:
                available[name] = self._metadata_of(name)
            elif _settable(beam, beam_parameter):
                available[name] = beam.parameters[beam_parameter].metadata
        return available

    def supported_pattern_settings(self) -> Dict[str, ParameterMetadata]:
        """The pattern fields whose choices this instrument decides: the scan
        direction. The other pattern fields keep the choices and limits their own
        metadata gives."""
        if not self.scan_directions:
            return {}
        return {"scan_direction": ParameterMetadata(choices=self.scan_directions)}

    @classmethod
    def supported_pattern_types(cls) -> Tuple[Type[FibsemPatternSettings], ...]:
        """The pattern types this instrument draws: those whose draw hook its driver
        implements."""
        return tuple(
            kind
            for kind, hook in DRAW_HOOKS.items()
            if getattr(cls, hook) is not getattr(Milling, hook)
        )

    def _metadata_of(self, name: str) -> ParameterMetadata:
        """The driver's metadata for its field *name*, asked for once."""
        if name not in self._driver_settings:
            self._driver_settings[name] = self._setting_metadata(name)
        return self._driver_settings[name]

    def _setting_metadata(self, name: str) -> ParameterMetadata:
        """What the driver allows for its recipe field *name*. The base knows the
        channel and the patterning mode; a driver adds its own fields."""
        if name == "milling_channel":
            channels = [BeamType.ION]
            if getattr(self, "electron", None) is not None:
                channels.append(BeamType.ELECTRON)
            return ParameterMetadata(choices=tuple(channels))
        if name == "patterning_mode":
            return ParameterMetadata(choices=("Serial", "Parallel"))
        return ParameterMetadata()

    # -- commands ----------------------------------------------------------------

    @command
    def setup(
        self, settings: FibsemMillingSettings, name: Optional[str] = None
    ) -> None:
        """Get the instrument ready to mill ``settings``, with no patterns yet.

        The first setup since the last ``restore`` saves the milling beam's conditions.
        """
        if self._saved is None:
            self._save(self.beam(settings.milling_channel))
        self._name = name
        self._set_up = settings
        self._setup(settings, name)
        if self._drawn:
            self._set_drawn([])

    @command
    def draw(self, patterns: Sequence[FibsemPatternSettings]) -> None:
        """Add ``patterns`` to what the next run mills.

        Every pattern is checked before any is drawn, so a pattern this instrument
        can't draw raises `UnsupportedPattern` with nothing drawn.
        """
        patterns = list(patterns)
        hooks = [self._hook_for(pattern) for pattern in patterns]
        try:
            for pattern, hook in zip(patterns, hooks):
                self._draw(pattern, hook)
                self._drawn.append(pattern)
        finally:
            self._set_drawn(self._drawn)

    def _hook_for(self, pattern: FibsemPatternSettings) -> str:
        """The draw hook for *pattern*, or a raise when this instrument can't draw it."""
        if not isinstance(pattern, FibsemPatternSettings):
            raise TypeError(f"Expected FibsemPatternSettings, got {type(pattern)}")
        kind = type(pattern)
        drawn = next((base for base in kind.__mro__ if base in DRAW_HOOKS), None)
        if drawn is None or drawn not in self.supported_pattern_types():
            raise UnsupportedPattern(
                f"{self._driver_name()} cannot draw {kind.__name__} patterns."
            )
        return DRAW_HOOKS[drawn]

    def _driver_name(self) -> str:
        parent = self.parent
        return type(parent).__name__ if parent is not None else type(self).__name__

    @command
    def prepare(
        self,
        settings: FibsemMillingSettings,
        patterns: Sequence[FibsemPatternSettings],
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """``setup`` with ``settings``, then ``draw`` ``patterns``. Answers the
        current the milling beam reads after the setup, in amps, and a warning when
        the recipe asked for a current the beam doesn't list (`milling_current`)."""
        self.setup(settings, name)
        self.draw(patterns)
        return self._milling_current(live=True)

    def _milling_current(self, live: bool = False) -> Dict[str, Any]:
        """The current what is drawn mills at: ``milling_current``, the milling
        beam's current in amps (read now when *live*, else the last one read or
        written; None before there is one), and ``warning``, set when the last
        setup asked for a current that is not one of those the beam lists. A beam
        that lists no currents, or a driver that sets the current another way
        (Tescan's preset), has no warning."""
        settings = self._set_up
        beam = self.beam(settings.milling_channel if settings else BeamType.ION)
        if beam is None or "current" not in beam.parameters:
            return {"milling_current": None, "warning": None}
        parameter = beam.parameters["current"]
        used = parameter.last_known
        if live:
            try:
                used = parameter.get_value()
            except Exception as error:  # the answer says what it can
                logging.debug(f"milling current: could not read it: {error!r}")
        asked = (
            settings.milling_current
            if settings is not None and "milling_current" in self.setting_names
            else None
        )
        return {"milling_current": used, "warning": _unlisted(asked, used, parameter)}

    @command
    def start(self) -> None:
        """Start milling what is drawn, and return at once. Nothing follows it:
        no ``progress`` and no ``milling`` event, so an agent mills with ``run``."""
        self._start()

    @command
    def stop(self) -> None:
        """Stop milling: the pattern now, and the milling task it is part of, which
        starts no later stage. A ``run`` in progress ends and answers ``cancelled``.

        Refused with `NothingToStop` when nothing is milling and no task is running.
        """
        task_stop = self._task_stop
        if task_stop is None and not self._is_milling():
            raise NothingToStop("Nothing is milling.")
        if task_stop is not None:
            task_stop.set()
        self._end_step(MillingStepEnd(STOPPED, reason="Milling stopped."))

    def end(self, reason: str, by: str) -> None:
        """End this stage, completed: the mill step now stops, and its stage run
        starts no further step. A run in progress answers ``completed``; the
        milling task goes on to its next stage. Only a stage that mills
        through ``run`` has steps to end: coincidence milling, which starts
        its own mill, ignores it until it moves to monitors (FIB-1484).

        Refused with `NothingToStop` when nothing is milling and no stage runs.
        """
        stage_run = self._stage_run
        if stage_run is None and not self._is_milling():
            raise NothingToStop("Nothing is milling.")
        end = MillingStepEnd(ENDED, by=by, reason=reason)
        if stage_run is not None:
            stage_run.end_as(end)
        self._end_step(end)

    def stop_pattern(self) -> None:
        """Stop the pattern milling now, and nothing more: the mill step ends
        ``ended`` and a milling task goes on. For a strategy that ends a mill on
        purpose (coincidence milling); ``stop`` is the way to stop a mill."""
        self._end_step(
            MillingStepEnd(ENDED, by="strategy", reason="The pattern was stopped.")
        )

    def _end_step(self, end: MillingStepEnd) -> None:
        """End the mill step running now with *end*, and stop the beam at once."""
        step = self._step
        if step is None:
            self._stop()
            return
        step.end_as(end)
        step.stop_beam(self._stop)

    @contextmanager
    def stage_run(self, stage: Any = None) -> Iterator[MillingStageRun]:
        """A strategy carries out *stage* in the block: the runs in it are its
        mill steps, each with the stage's monitors, and once ``end`` is called
        none starts."""
        previous, self._stage_run = self._stage_run, MillingStageRun(stage)
        try:
            yield self._stage_run
        finally:
            self._stage_run = previous

    @contextmanager
    def task_running(self, stop_event: threading.Event) -> Iterator[None]:
        """A milling task runs on this service in the block: ``stop`` sets
        ``stop_event`` as well as stopping the pattern, so the whole task ends."""
        previous, self._task_stop = self._task_stop, stop_event
        try:
            yield
        finally:
            self._task_stop = previous

    def _is_milling(self) -> bool:
        """Whether a run is in progress, or the instrument says it is milling (a
        ``start`` returns at once)."""
        if self._running:
            return True
        try:
            return self._poll().milling_state in ACTIVE_MILLING_STATES
        except Exception:  # noqa: BLE001 - unsure counts as milling: stop it
            return True

    @command
    def pause(self) -> None:
        """Pause milling."""
        self._pause()

    @command
    def resume(self) -> None:
        """Resume paused milling."""
        self._resume()

    @command(long=True)
    def run(
        self, stop_event: Optional[Union[threading.Event, AnyStopEvent]] = None
    ) -> str:
        """Mill what is drawn and return how it ended, reporting ``progress``.

        The patterns are cleared at the end, however the run ends. A ``stop_event``
        set while it runs stops the beam, and the run then raises
        `OperationCancelledError`; ``stop`` from another thread ends it, and it
        answers ``cancelled`` rather than ``completed`` (FIB-1377). A failure stops
        the beam before it is raised.

        Outside a milling task, the run is on the record as one ``milling`` event,
        with the name ``setup`` was given and how it ended.
        """
        if self._run_context or getattr(self._part_of, "depth", 0):
            # A task's run, or a spot burn's: that records the whole.
            return self._run(stop_event)
        return self._run_on_the_record(stop_event)

    @command(long=True)
    def run_task(self, config: Any, path: Optional[str] = None) -> Dict[str, Any]:
        """Run a milling task, as the app runs a protocol's: each enabled stage
        in turn with its strategy, aligning to the task's reference image
        between stages, and the beam put back at the end.

        *config* is a ``FibsemMillingTaskConfig`` (or its dict). The task saves
        its reference and stage images under ``<path>/Milling/<name>``; with
        no *path*, under the app's data folder, never under the imaging path
        the config carries. Answers the task's status (``completed``,
        ``cancelled``, ``skipped`` when no stage is enabled), the folder it
        saved into, and how each stage ended, as its ``milling`` event says.
        The folder is answered however the task ended, made empty when nothing
        was saved in it (a task stopped before its first image). A failed task
        raises its error, after the beam is put back. ``stop`` ends it
        ``cancelled``.
        """
        return self._run_task(config, path, answer_empty_folder=True)

    def _run_task(
        self, config: Any, path: Optional[str], answer_empty_folder: bool
    ) -> Dict[str, Any]:
        # Imported here: fibsem.milling.tasks imports this module.
        from fibsem.milling.tasks import FibsemMillingTaskConfig, run_milling_task

        if isinstance(config, dict):
            config = FibsemMillingTaskConfig.from_dict(config)
        if not isinstance(config, FibsemMillingTaskConfig):
            raise TypeError("config must be a FibsemMillingTaskConfig or its dict")
        # Where it saves is the path argument's to say, so the App Server can
        # keep it inside the experiment; the config's own path is not checked.
        config.acquisition.imaging.path = None
        # Registered for the whole call, so a stop before the first stage mills
        # (while the reference image is taken) ends the task too.
        stop_event = threading.Event()
        with self.task_running(stop_event):
            task = run_milling_task(
                self.parent,
                config,
                stop_event=stop_event,
                raise_on_failure=True,
                output_dir=path,
            )
        ended = task.ended()
        if ended.get("path") and not os.path.isdir(ended["path"]):
            if answer_empty_folder:
                # The task's folder, whether or not it got to save anything: a
                # stopped task answered none (FIB-1430).
                os.makedirs(ended["path"], exist_ok=True)
            else:
                # Nothing was saved (a stage that acquires no images, as
                # run_stage's usually is): answer no folder rather than one
                # that isn't there.
                ended["path"] = None
        return ended

    @command(long=True)
    def run_stage(
        self, stage: FibsemMillingStage, path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run one milling stage as a task of its own, named after the stage:
        ``run_task`` with that one stage, answering the same. It saves images
        only when the stage's ``milling.acquire_images`` is set; otherwise its
        answer's ``path`` is None."""
        from fibsem.milling.tasks import FibsemMillingTaskConfig

        return self._run_task(
            FibsemMillingTaskConfig.from_stages([stage], name=stage.name),
            path,
            answer_empty_folder=False,
        )

    @query
    def preview_patterns(
        self,
        image_path: str,
        stages: Optional[List[FibsemMillingStage]] = None,
    ) -> Any:
        """*stages* drawn over the saved image at *image_path* (an ion image, to
        see where they would mill), or with no stages the patterns ``prepare``
        drew. An RGB image the saved image's shape, scaled down to 1024 px wide;
        reads the file and moves nothing."""
        from fibsem.devices.wire import Frame
        from fibsem.milling.patterning.plotting import render_milling_patterns

        image = FibsemImage.load(image_path)
        if stages is None:
            drawn = [_Drawn(self.drawn_patterns)] if self.drawn_patterns else []
        else:
            drawn = list(stages)
        data = render_milling_patterns(image, drawn)
        return Frame(
            data,
            {
                "image_path": image_path,
                "stages": [stage.name for stage in drawn],
                "full_width": int(image.data.shape[1]),
                "full_height": int(image.data.shape[0]),
            },
        )

    @query
    def check_stages(
        self, stages: List[FibsemMillingStage], aspect: float = 2 / 3
    ) -> List[Dict[str, Any]]:
        """Each of *stages* checked before anything is drawn: how long it would
        mill, in seconds (``estimate_stage``), and its patterns that reach past
        the field, its milling hfw wide and *aspect* of that high (an image's
        height over width), which a real instrument refuses to mill. A stage
        that cannot define its patterns answers why. Moves nothing."""
        from fibsem.milling.patterning.utils import pattern_extent, stage_outside_field

        checked = []
        for stage in stages:
            try:
                outside = stage_outside_field(stage, aspect)
                seconds = self.estimate_stage(stage)
            except Exception as exc:  # noqa: BLE001 - a stage's own trouble
                checked.append({"name": stage.name, "error": str(exc)})
                continue
            checked.append(
                {
                    "name": stage.name,
                    "estimated_seconds": float(seconds),
                    "field_m": [stage.milling.hfw, stage.milling.hfw * aspect],
                    "outside_field": [
                        {
                            "pattern": type(pattern).__name__,
                            "extent_m": list(pattern_extent(pattern) or ()),
                        }
                        for pattern in outside
                    ],
                }
            )
        return checked

    @query
    def list_options(self) -> Dict[str, List[str]]:
        """The names a stage's ``pattern.name`` and ``strategy.name`` take."""
        from fibsem.milling.patterning import get_pattern_names
        from fibsem.milling.strategy import get_strategy_names

        return {"patterns": get_pattern_names(), "strategies": get_strategy_names()}

    @contextmanager
    def part_of(
        self, report: Optional[Callable[[MillingProgress], None]] = None
    ) -> Iterator[None]:
        """Runs in the block, on this thread, are part of something that records
        itself (a spot burn), so they record no ``milling`` event of their own.

        With *report*, their progress goes to it instead of ``progress``: what
        listens for milling progress does not see a spot burn's mill as a mill."""
        self._part_of.depth = getattr(self._part_of, "depth", 0) + 1
        outer = getattr(self._part_of, "report", None)
        if report is not None:
            self._part_of.report = report
        try:
            yield
        finally:
            self._part_of.depth -= 1
            self._part_of.report = outer

    def _run_on_the_record(
        self, stop_event: Optional[Union[threading.Event, AnyStopEvent]]
    ) -> str:
        """A run outside a milling task (an agent's ``prepare`` and ``run``, or a
        script's): its stage updates carry the name it was set up with, and it
        records one ``milling`` event with how it ended, as a task's run does
        (FIB-1257, FIB-1377)."""
        payload: Dict[str, Any] = {
            # The id of its run on this service (`fibsem.devices.runs`), so
            # its events and its run share one id.
            "milling_run_id": current_run_id(self.name) or str(uuid.uuid4()),
            "milling_task_name": self._name,
            "started_at": now_iso(),
        }
        self._run_context = {
            name: value for name, value in payload.items() if name in _RUN_CONTEXT
        }
        try:
            status = self._run(stop_event)
        except BaseException as error:
            payload.update(ended_by(error))
            raise
        else:
            payload["status"] = status
        finally:
            self._run_context = {}
            payload["ended_at"] = now_iso()
            self.session.record("milling", payload)
            # No task ends it, so the run does: the progress bars clear on a
            # terminal report, and otherwise stay at the last tick (FIB-1401).
            self._report(
                MillingProgress(
                    status=_ENDED_AS.get(
                        payload.get("status"), MillingProgressStatus.TASK_FAILED
                    ),
                    milling_run_id=payload["milling_run_id"],
                    milling_task_name=payload["milling_task_name"],
                    error=payload.get("error"),
                )
            )
        return status

    def _run(self, stop_event: Optional[Union[threading.Event, AnyStopEvent]]) -> str:
        stage_run = self._stage_run
        if stage_run is not None and stage_run.ending.is_set():
            # The stage was ended: it mills no further step.
            try:
                self._clear()
            finally:
                self._set_drawn([])
            return COMPLETED
        stage = stage_run.stage if stage_run is not None else None
        step = MillingStep(
            index=len(stage_run.steps) if stage_run else 0, milling=self, stage=stage
        )
        if stage_run is not None:
            stage_run.steps.append(step)
        monitors = _monitors_of(stage)
        self._step = step
        self._before_run()
        self._running = True
        try:
            total = float(self._estimate() or 0.0)
            for monitor in monitors:
                monitor.start(step)
            if not step.ending.is_set():
                self._start()
                MillingStateMonitor(self, step, stop_event).watch(total)
        except BaseException as error:
            step.end_as(MillingStepEnd.from_error(error))
            self._abort()
            raise
        finally:
            self._running = False
            # However the step ended: its end is set, so monitors leave their loops.
            step.end_as(MillingStepEnd(FINISHED, by="instrument"))
            step.monitors = [monitor.stop() for monitor in monitors]
            try:
                self._after_run()
            finally:
                self._clear()
                self._set_drawn([])
                step.ended_at = now_iso()
                self._step = None
        raise_if_cancelled(stop_event, "Milling stopped.")
        if step.end.kind == STEP_FAILED:
            # A monitor failed it (a required one, or a pause too long).
            raise MillingStepFailed(step.end.reason or "The mill step failed.")
        return step.end.status

    def _abort(self) -> None:
        """Stop the beam after a failure, without hiding the failure."""
        try:
            self._stop()
        except Exception as e:
            logging.warning(f"Error stopping milling after a failure: {e}")

    @query
    def estimate_time(self) -> Dict[str, Any]:
        """How long milling what is drawn takes: ``seconds``, with the current it
        mills at and any warning about it (`_milling_current`, from what the
        setup wrote, not read again). Moves nothing, so a query."""
        return {"seconds": self._estimate(), **self._milling_current()}

    @classmethod
    def estimate_stage(cls, stage: FibsemMillingStage) -> float:
        """How long milling *stage* would take on this instrument, in seconds,
        before anything is drawn: for planning (task and workflow ETAs).

        By default the shared sputter-rate table, keyed on the recipe's milling
        current. A driver with its own model overrides it. A model of the driver,
        not of one connection: the session keeps it after a disconnect, so it
        holds no reference to the microscope.
        """
        return estimate_milling_time(stage.pattern, stage.milling.milling_current)

    @command
    def clear(self) -> None:
        """Remove every drawn pattern."""
        self._clear()
        self._set_drawn([])

    @command
    def restore(self) -> None:
        """Put the milling beam back as the first ``setup`` found it.

        The beam is left alone when nothing was saved, so it is safe to call twice;
        what else the driver resets (``_restore``) is reset every time.
        """
        saved, beam = self._saved, self._saved_beam
        self._saved = self._saved_beam = None
        if saved is not None and beam is not None:
            for name in SAVED_BEAM_CONDITIONS:
                if name in saved:
                    self._write_back(beam, name, saved[name])
        self._restore()

    @command
    def finish(self) -> None:
        """Clear the patterns and put the milling beam back as ``setup`` found it."""
        self.clear()
        self.restore()

    def _save(self, beam: Beam) -> None:
        """Keep the beam's conditions that it has, can set, and reports."""
        self._saved_beam = beam
        self._saved = save_beam_conditions(beam, SAVED_BEAM_CONDITIONS)

    def _write_back(self, beam: Beam, name: str, value: Any) -> None:
        """Write one saved condition back to the beam, as the old API would."""
        getattr(beam, name).write_through(value)

    # -- what a driver implements -------------------------------------------------

    def _setup(self, settings: FibsemMillingSettings, name: Optional[str]) -> None:
        """Apply the recipe (beam conditions, channel, patterning mode, application
        file) and clear the patterns."""
        raise NotImplementedError

    def _draw(self, pattern: FibsemPatternSettings, hook: str) -> None:
        """Draw one checked pattern with its *hook*. A driver extends it for what goes
        around every pattern, such as an application file of its own."""
        getattr(self, hook)(pattern)

    def _draw_rectangle(self, pattern: FibsemRectangleSettings) -> None:
        raise NotImplementedError

    def _draw_line(self, pattern: FibsemLineSettings) -> None:
        raise NotImplementedError

    def _draw_circle(self, pattern: FibsemCircleSettings) -> None:
        raise NotImplementedError

    def _draw_bitmap(self, pattern: FibsemBitmapSettings) -> None:
        raise NotImplementedError

    def _draw_polygon(self, pattern: FibsemPolygonSettings) -> None:
        raise NotImplementedError

    def _start(self) -> None:
        raise NotImplementedError

    def _stop(self) -> None:
        raise NotImplementedError

    def _pause(self) -> None:
        raise NotImplementedError

    def _resume(self) -> None:
        raise NotImplementedError

    def _estimate(self) -> float:
        raise NotImplementedError

    def _clear(self) -> None:
        raise NotImplementedError

    def _restore(self) -> None:
        """Put back anything else milling changed, after the beam; by default nothing."""

    def _poll(self) -> MillingProgress:
        """One look at the instrument during a run: a stage update with its state,
        and its total (``estimated_time``) and ``remaining_time`` where the instrument
        reports them. By default only the state: the loop counts the time itself."""
        return progress_update(state=self.read_state())

    def _before_run(self) -> None:
        """Get a drawn run ready to start, before its estimate; by default nothing."""

    def _after_run(self) -> None:
        """Tidy up after a run, however it ended, before the patterns are cleared;
        by default nothing."""

    def _wait(self, seconds: float) -> None:
        """Wait between looks at the instrument; an end of the step wakes it.
        A driver that overrides it (the Demo's simulated clock) sees the end
        at its next look instead."""
        step = self._step
        if step is not None and not step.ending.is_set():
            step.ending.wait(seconds)
        else:
            time.sleep(seconds)


class MillingStateMonitor:
    """The monitor every mill step runs: it looks at the instrument every
    ``poll_interval``, reports ``progress``, and ends the step ``finished`` when
    the pattern goes idle. Once something else ends the step, it stops the beam
    and waits up to ``stop_timeout`` for it to go idle."""

    name = "state"

    def __init__(
        self,
        milling: Milling,
        step: MillingStep,
        stop_event: Optional[Union[threading.Event, AnyStopEvent]],
    ):
        self.milling = milling
        self.step = step
        self.stop_event = stop_event

    def watch(self, total: float) -> None:
        """Look at the instrument until the step is over."""
        milling, step = self.milling, self.step
        start_time = time.time()
        began = last = time.monotonic()
        elapsed = 0.0
        started = False
        stop_deadline: Optional[float] = None
        while True:
            if self.stop_event is not None and self.stop_event.is_set():
                step.end_as(MillingStepEnd(STOPPED, reason="Milling stopped."))
            if step.pausing.is_set() and not step.ending.is_set():
                # A monitor holds a pause: no look, and no milled time.
                last = time.monotonic()
                milling._wait(milling.poll_interval)
                continue
            poll = milling._poll()
            now = time.monotonic()
            state = poll.milling_state
            if state is MillingState.RUNNING:
                elapsed += now - last
            last = now
            started = started or state in ACTIVE_MILLING_STATES
            if poll.estimated_time is not None:
                total = float(poll.estimated_time)
            if poll.remaining_time is not None:
                elapsed = max(0.0, total - float(poll.remaining_time))
            step.set_milled_time(elapsed)
            milling._report(
                progress_update(
                    state=state,
                    start_time=start_time,
                    total=total,
                    remaining=max(0.0, total - elapsed),
                )
            )
            if state not in ACTIVE_MILLING_STATES:
                if started or now - began >= milling.start_timeout:
                    step.end_as(MillingStepEnd(FINISHED, by="instrument"))
                    break
            if self.stop_event is not None and self.stop_event.is_set():
                step.end_as(MillingStepEnd(STOPPED, reason="Milling stopped."))
            if step.ending.is_set():
                if stop_deadline is None:
                    if step.stop_beam(milling._stop):
                        logging.info("Milling stop requested; stopping the beam.")
                    stop_deadline = now + milling.stop_timeout
                elif now >= stop_deadline:
                    logging.warning(
                        f"Milling did not stop within {milling.stop_timeout} s."
                    )
                    break
            milling._wait(milling.poll_interval)


class _Drawn:
    """What ``prepare`` drew, in the shape the pattern drawing reads a stage in."""

    name = "drawn"

    def __init__(self, patterns: Sequence[FibsemPatternSettings]):
        self._patterns = list(patterns)
        self.pattern = self

    def define_patterns(self) -> List[FibsemPatternSettings]:
        return self._patterns


def _settable(beam: Optional[Beam], name: str) -> bool:
    return beam is not None and name in beam.parameters and getattr(beam, name).settable


def _unlisted(asked: Any, used: Any, current: Any) -> Optional[str]:
    """The warning for a recipe's milling current, *asked*, that the beam's
    *current* parameter does not list among its choices, naming the listed
    currents either side of it; None when it is listed, or nothing is listed."""
    from fibsem.utils import format_current

    if not isinstance(asked, (int, float)) or isinstance(asked, bool):
        return None
    listed = sorted(
        c
        for c in (current.choices or ())
        if isinstance(c, (int, float)) and not isinstance(c, bool)
    )
    if not listed or any(math.isclose(asked, c, rel_tol=1e-3) for c in listed):
        return None
    below = [c for c in listed if c < asked][-1:]
    above = [c for c in listed if c > asked][:1]
    nearest = " and ".join(format_current(c) for c in below + above)
    warning = (
        f"{format_current(asked)} is not a current the milling beam lists; "
        f"the nearest it lists: {nearest}."
    )
    if isinstance(used, (int, float)) and not math.isclose(used, asked, rel_tol=1e-3):
        warning += f" The beam is at {format_current(used)}."
    return warning


def bind_milling(service: Type[_M], microscope: Any, **kwargs: Any) -> Optional[_M]:
    """Build a microscope's milling service of class *service* over its beams, or
    None when it has no ion beam (the column is disabled), which leaves the
    microscope's own milling code in charge. *kwargs* go to the service's
    constructor, such as the driver's connection."""
    beams = microscope.beams
    if BeamType.ION not in beams:
        return None
    milling = service(parent=microscope, **kwargs)
    milling.fill_roles(ion=beams[BeamType.ION])
    if BeamType.ELECTRON in beams:
        milling.fill_roles(electron=beams[BeamType.ELECTRON])
    milling.connect()
    return milling


def report_progress(microscope: Any, progress: MillingProgress) -> None:
    """Report ``progress`` on *microscope*'s milling service; without one (no ion
    beam) nothing mills, so there is nothing to report."""
    milling = getattr(microscope, "milling", None)
    if milling is not None:
        milling.report(progress)


def _monitors_of(stage: Any) -> List[Any]:
    """The monitors a step of *stage* runs, made afresh from its configs."""
    configs = getattr(stage, "monitors", None) or []
    return [config.monitor_class(config=config) for config in configs]


@contextmanager
def stage_running(microscope: Any, stage: Any = None) -> Iterator[Any]:
    """A strategy carries out a stage on *microscope*'s milling service in the
    block (`Milling.stage_run`); yields the stage run, or None without a milling
    service."""
    milling = getattr(microscope, "milling", None)
    if milling is None:
        yield None
        return
    with milling.stage_run(stage) as stage_run:
        yield stage_run


@contextmanager
def task_running(microscope: Any, stop_event: threading.Event) -> Iterator[None]:
    """A milling task runs on *microscope*'s milling service in the block, so the
    service's ``stop`` sets ``stop_event`` and the whole task ends (FIB-1398).
    Without a milling service nothing mills, and there is nothing to register."""
    milling = getattr(microscope, "milling", None)
    if milling is None:
        yield
        return
    with milling.task_running(stop_event):
        yield


def progress_update(
    state: MillingState,
    start_time: Optional[float] = None,
    total: Optional[float] = None,
    remaining: Optional[float] = None,
) -> MillingProgress:
    """A stage update: what a run, or one look at the instrument, reports."""
    return MillingProgress(
        status=MillingProgressStatus.STAGE_UPDATE,
        milling_state=state,
        start_time=start_time,
        estimated_time=total,
        remaining_time=remaining,
    )
