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
from fibsem.milling.base import FibsemMillingStage, estimate_milling_time
from fibsem.milling.progress import MillingProgress, MillingProgressStatus
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
        # Set by ``stop`` while a run is on, so the run can say it was stopped.
        self._stop_requested = threading.Event()
        # The stop of the milling task running on this service, if one is: ``stop``
        # sets it, so the task starts no later stage (FIB-1398).
        self._task_stop: Optional[threading.Event] = None
        # A run is in progress, for ``stop`` to know there is something to stop.
        self._running = False
        # The name the last setup was given, for a run outside a task to go by.
        self._name: Optional[str] = None
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
        specs = field_specs(FibsemMillingSettings)
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
    ) -> None:
        """``setup`` with ``settings``, then ``draw`` ``patterns``."""
        self.setup(settings, name)
        self.draw(patterns)

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
        self.stop_pattern()

    def stop_pattern(self) -> None:
        """Stop the pattern milling now, and nothing more: a milling task goes on
        to its next stage. For a strategy that ends a mill on purpose (coincidence
        milling); ``stop`` is the way to stop a mill."""
        self._stop_requested.set()
        self._stop()

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
        self._stop_requested.clear()
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
            "milling_run_id": str(uuid.uuid4()),
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
        self._before_run()
        self._running = True
        try:
            total = float(self._estimate() or 0.0)
            self._start()
            self._monitor(total, stop_event)
        except BaseException:
            self._abort()
            raise
        finally:
            self._running = False
            try:
                self._after_run()
            finally:
                self._clear()
                self._set_drawn([])
        raise_if_cancelled(stop_event, "Milling stopped.")
        return CANCELLED if self._stop_requested.is_set() else COMPLETED

    def _monitor(
        self,
        total: float,
        stop_event: Optional[Union[threading.Event, AnyStopEvent]],
    ) -> None:
        """Look at the instrument every ``poll_interval`` until the run is over."""
        start_time = time.time()
        began = last = time.monotonic()
        elapsed = 0.0
        started = False
        stop_deadline: Optional[float] = None
        while True:
            poll = self._poll()
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
            self._report(
                progress_update(
                    state=state,
                    start_time=start_time,
                    total=total,
                    remaining=max(0.0, total - elapsed),
                )
            )
            if state not in ACTIVE_MILLING_STATES:
                if started or now - began >= self.start_timeout:
                    break
            if stop_event is not None and stop_event.is_set():
                if stop_deadline is None:
                    logging.info("Milling stop requested; stopping the beam.")
                    self._stop()
                    stop_deadline = now + self.stop_timeout
                elif now >= stop_deadline:
                    logging.warning(
                        f"Milling did not stop within {self.stop_timeout} s."
                    )
                    break
            self._wait(self.poll_interval)

    def _abort(self) -> None:
        """Stop the beam after a failure, without hiding the failure."""
        try:
            self._stop()
        except Exception as e:
            logging.warning(f"Error stopping milling after a failure: {e}")

    @query
    def estimate_time(self) -> float:
        """How long milling what is drawn takes, in seconds. Moves nothing, so a
        query."""
        return self._estimate()

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
        """Wait between looks at the instrument."""
        time.sleep(seconds)


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
