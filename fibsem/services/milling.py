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
from contextlib import contextmanager, nullcontext
from typing import (
    Any,
    Callable,
    ContextManager,
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
from fibsem.acting import current_actor
from fibsem.cancellation import (
    CANCELLED,
    COMPLETED,
    AnyStopEvent,
    NothingToStop,
    ended_by,
    raise_if_cancelled,
)
from fibsem.devices.beam import Beam
from fibsem.devices.core import (
    DeviceParameter,
    ParameterMetadata,
    Role,
    RoleUnfilled,
    command,
    query,
)
from fibsem.devices.runs import RUNNING, current_run, current_run_id
from fibsem.milling.base import FibsemMillingStage, estimate_milling_time
from fibsem.milling.progress import MillingProgress, MillingProgressStatus
from fibsem.milling.readings import MillingReadings
from fibsem.milling.rtm import RtmFrame, is_monitored
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
from fibsem.structures.image import FibsemImage
from fibsem.structures.params import Parameter, field_specs
from fibsem.structures.settings import (
    ACTIVE_MILLING_STATES,
    BeamType,
    DriverMillingSettings,
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemLineSettings,
    FibsemMillingSettings,
    FibsemPatternSettings,
    FibsemPolygonSettings,
    FibsemRectangleSettings,
    MillingState,
)
from fibsem.util import wording
from fibsem.util.timestamps import now_iso

# The beam conditions milling changes and `restore` puts back, in the order they are
# written back: a preset first, since it sets the others on a column that has one,
# then the voltage before the current, as the instruments want.
SAVED_BEAM_CONDITIONS = ("preset", "voltage", "current", "hfw")

# The patterning modes a recipe mills in.
PATTERNING_MODES = ("Serial", "Parallel")

# What a stage update leaves out and the run's own reports said: which run, which
# task config, which stage. A backend's poll loop and a strategy's tick
# know only the countdown.
_RUN_CONTEXT = (
    "milling_run_id",
    "milling_task_name",
    "stage_name",
    "current_stage",
    "total_stages",
)

# A run outside a task that ``setup`` gave no name.
UNNAMED_RUN = "Prepared patterns"
# The commands that mill, which a stop ends even before they start milling.
_RUNS = ("run", "run_task", "run_stage")

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


class MixedMonitorViews(ValueError):
    """A stage's live monitors need different views; on one instrument view
    only one can have it."""


class MillingStepFailed(RuntimeError):
    """A monitor failed the mill step: a required monitor's error, or a pause
    longer than its ``max_pause``."""


# --- A command as the operator reads it, in an agent's request (wording) ---


def parse_patterns(values: Any) -> List[Any]:
    """Pattern dicts as the milling service takes them, as pattern settings;
    one that does not parse is left out."""
    patterns = []
    for value in values or []:
        try:
            patterns.append(FibsemPatternSettings.from_dict(dict(value)))
        except Exception:  # noqa: BLE001 - the route refuses it; describe the rest
            continue
    return patterns


def _pattern_kind(pattern: Any) -> str:
    return type(pattern).__name__.replace("Fibsem", "").replace("Settings", "").lower()


def _size(pattern: Any) -> str:
    width, height = getattr(pattern, "width", None), getattr(pattern, "height", None)
    if width is not None and height is not None:
        return f"{wording.number(width * 1e6)} × {wording.number(height * 1e6)} µm"
    radius = getattr(pattern, "radius", None)
    if radius is not None:
        return f"radius {wording.distance(radius)}"
    return ""


def _centre(pattern: Any) -> str:
    x, y = getattr(pattern, "centre_x", None), getattr(pattern, "centre_y", None)
    if x is None or y is None:
        return ""
    return f"({x * 1e6:+.1f}, {y * 1e6:+.1f}) µm"


def _patterns_time(patterns: List[Any], current: Optional[float]) -> Optional[float]:
    """Seconds to mill ``patterns`` at ``current``, from the sputter-rate table."""
    if not patterns or not current:
        return None
    try:
        return sum(estimate_milling_time(p, current) for p in patterns)
    except Exception:  # noqa: BLE001 - no estimate, nothing more
        return None


def _past_field(patterns: List[Any], settings: Any) -> str:
    """ "1 of them past the 150 µm field": the patterns a real instrument would
    refuse, on the step's line and not only in the picture's legend. The field
    is an image's 2:3, as the picture checks it."""
    from fibsem.milling.patterning.utils import outside_field

    try:
        hfw = float(settings["hfw"])
    except (KeyError, TypeError, ValueError):
        return ""
    past = [p for p in patterns if outside_field(p, hfw, hfw * 2 / 3)]
    if not past:
        return ""
    which = "past" if len(patterns) == 1 else f"{len(past)} of them past"
    return f"{which} the {wording.distance(hfw)} field"


def _prepare_line(kwargs: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    patterns = parse_patterns(kwargs.get("patterns"))
    settings = kwargs.get("settings") or {}
    current = settings.get("milling_current") if isinstance(settings, dict) else None
    voltage = settings.get("milling_voltage") if isinstance(settings, dict) else None
    kinds = {_pattern_kind(p) for p in patterns}
    if len(kinds) == 1:
        parts = [wording.plural(len(patterns), kinds.pop())]
    else:
        parts = [wording.plural(len(patterns), "pattern")]
    sizes = {_size(p) for p in patterns}
    if len(sizes) == 1 and "" not in sizes:
        parts.append(sizes.pop())
    depths = {getattr(p, "depth", None) for p in patterns}
    if len(depths) == 1 and None not in depths:
        parts.append(f"{wording.distance(depths.pop())} deep")
    if current is not None:
        beam = wording.current(current)
        if voltage:
            beam += f" at {wording.number(float(voltage) / 1e3)} kV"
        parts.append(beam)
    past = _past_field(patterns, settings)
    if past:
        parts.append(past)
    name = kwargs.get("name")
    change = (f"Prepare '{name}'" if name else "Prepare") + ": " + ", ".join(parts)
    rows = [
        (
            f"{_pattern_kind(p).capitalize()} {i + 1}",
            _size(p),
            wording.distance(getattr(p, "depth", None)) if hasattr(p, "depth") else "",
            _centre(p),
        )
        for i, p in enumerate(patterns)
    ]
    return {
        "what": "Milling",
        "change": change,
        "details": wording.table(("Pattern", "Size", "Depth", "Centre"), rows),
        "_estimate": (_patterns_time(patterns, current), current),
    }


# Whose estimate a mill's time is: the sputter-rate table's, at the mill's
# current, not the instrument's.
_ESTIMATED = "estimated from the sputter rate"


def _parsed_stages(values: Any) -> List[Any]:
    """Milling stage dicts as stages; one that does not parse is left out."""
    stages = []
    for value in values or []:
        try:
            stages.append(FibsemMillingStage.from_dict(dict(value)))
        except Exception:  # noqa: BLE001 - the route refuses it; describe the rest
            continue
    return stages


def _stage_rows(stages: List[Any]) -> Dict[str, Any]:
    rows = []
    for stage in stages:
        pattern = stage.pattern
        point = getattr(pattern, "point", None)
        depth = getattr(pattern, "depth", None)
        rows.append(
            (
                stage.name,
                str(getattr(pattern, "name", "") or type(pattern).__name__),
                wording.current(stage.milling.milling_current),
                wording.distance(depth) if depth is not None else "",
                (
                    f"({point.x * 1e6:+.1f}, {point.y * 1e6:+.1f}) µm"
                    if point is not None
                    else ""
                ),
            )
        )
    return wording.table(("Stage", "Pattern", "Current", "Depth", "Position"), rows)


def _stages_estimate(stages: List[Any]) -> Optional[float]:
    try:
        return sum(stage.estimated_time for stage in stages)
    except Exception:  # noqa: BLE001 - no estimate, nothing more
        return None


def _run_task_line(
    kwargs: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    """A milling task, by its stages: what each mills, at what current, where."""
    config = kwargs.get("config")
    config = config if isinstance(config, dict) else {}
    name = config.get("name") or "milling task"
    stages = [s for s in _parsed_stages(config.get("stages")) if s.enabled]
    if not stages:
        raw = [s.get("name") for s in config.get("stages") or [] if isinstance(s, dict)]
        listed = f": {wording.plural(len(raw), 'stage')}" if raw else ""
        return {"what": "Milling", "change": f"Run task '{name}'{listed}"}
    names = [stage.name for stage in stages]
    change = f"Run task '{name}': {wording.plural(len(stages), 'stage')} ({wording.names_cut(names)})"
    estimate = _stages_estimate(stages)
    if estimate:
        change += f", about {wording.seconds(estimate)} ({_ESTIMATED})"
    return {"what": "Milling", "change": change, "details": _stage_rows(stages)}


def _run_stage_line(
    kwargs: Dict[str, Any], ctx: wording.ActionContext
) -> Dict[str, Any]:
    stages = _parsed_stages([kwargs.get("stage")])
    if not stages:
        stage = kwargs.get("stage")
        name = stage.get("name") if isinstance(stage, dict) else None
        return {"what": "Milling", "change": f"Run stage '{name or 'milling stage'}'"}
    stage = stages[0]
    pattern = stage.pattern
    change = (
        f"Run stage '{stage.name}': "
        f"{str(getattr(pattern, 'name', '') or 'pattern').lower()}, "
        f"{wording.current(stage.milling.milling_current)}"
    )
    estimate = _stages_estimate(stages)
    if estimate:
        change += f", about {wording.seconds(estimate)} ({_ESTIMATED})"
    return {"what": "Milling", "change": change, "details": _stage_rows(stages)}


def _run_line(kwargs: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    """A run of what an earlier action of the request prepared, with its time."""
    seconds, current = ctx.estimate or (None, None)
    about = ""
    if seconds:
        at = f" at {wording.current(current)}" if current else ""
        about = f", about {wording.seconds(seconds)} ({_ESTIMATED}{at})"
    return {"what": "Milling", "change": f"Run{about}"}


def _said(change: str) -> wording.Line:
    """The line of a command that takes nothing: always the same words."""
    return lambda kwargs, ctx: {"what": "Milling", "change": change}


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
    # The readings the monitors of the latest mill step reported: per monitor,
    # the latest and the last minute of milled time. `follow_readings` gives
    # every reading, as it comes.
    readings = DeviceParameter(
        dict, doc="The mill step's monitor readings, latest and recent; read-only."
    )
    # Each notable reading (a drop, a trigger, a pause, an error, the end of a
    # step with monitors), for the event log.
    notable_reading = Signal(dict)

    # The patterns the next run mills, as drawn: the list so far after each
    # ``draw``, and empty when setup, ``clear`` or the end of a run removes them.
    # The app draws an agent's mill from it, which it has no config of.
    patterns_changed = Signal(object)

    # Each RTM read (`RtmFrame`), on the reader's thread: a subscriber returns
    # quickly and does heavy work on its own thread. Only a step with the RTM
    # monitor reads.
    rtm_frame = Signal(object)

    remote_signals = ("notable_reading", "patterns_changed")

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
    # The driver's own settings, which a recipe keeps in its ``driver`` block under
    # the driver's name; its registry entry names the same class
    # (`DriverEntry.milling_settings`). None for a driver with none.
    settings_class: Optional[Type[DriverMillingSettings]] = None
    # The directions this driver can scan a pattern in. A driver sets it.
    scan_directions: Tuple[str, ...] = ()
    # The RTM modes this driver reads in (`fibsem.milling.rtm.RTM_MODES`); none
    # for a driver without RTM. A driver with the `_*_rtm` hooks sets it.
    rtm_modes: Tuple[str, ...] = ()

    def __init__(self, name: str = "milling", **kwargs: Any):
        super().__init__(name, **kwargs)
        # The beam conditions the first setup found, until restore puts them back.
        self._saved: Optional[Dict[str, Any]] = None
        self._saved_beam: Optional[Beam] = None
        # What the driver says about its own fields, asked once.
        self._driver_settings: Dict[str, ParameterMetadata] = {}
        self._progress = progress_update(state=MillingState.IDLE)
        self._readings = MillingReadings()
        # The run and stage the last task or stage start named, for the stage
        # updates that follow it; empty outside a milling task.
        self._run_context: Dict[str, Any] = {}
        self._drawn: List[FibsemPatternSettings] = []
        # The mill step running now, and the stage run it is part of, if any.
        self._step: Optional[MillingStep] = None
        self._stage_run: Optional[MillingStageRun] = None
        # The stop of the milling task running on this service, if one is: ``stop``
        # sets it, so the task starts no later stage.
        self._task_stop: Optional[threading.Event] = None
        # A run is in progress, for ``stop`` to know there is something to stop.
        self._running = False
        # The name the last setup was given, for a run outside a task to go by.
        self._name: Optional[str] = None
        # The recipe the last setup applied, for the answers to say what it mills at.
        self._set_up: Optional[FibsemMillingSettings] = None
        # Inside ``part_of``: a run there is part of something with its own record.
        self._part_of = threading.local()
        # The latest RTM read and how many reads this RTM run has made.
        self.last_rtm_frame: Optional[RtmFrame] = None
        self._rtm_sequence = 0

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

    @property
    def has_rtm(self) -> bool:
        """Whether this driver reads RTM."""
        return bool(self.rtm_modes)

    def rtm_patterns(self) -> Dict[int, FibsemPatternSettings]:
        """The drawn patterns RTM reads, by draw index."""
        return {i: p for i, p in enumerate(self._drawn) if is_monitored(p)}

    def rtm_start(self, mode: str) -> None:
        """Start reading RTM in *mode*; the reads come from ``rtm_read``."""
        if mode not in self.rtm_modes:
            raise ValueError(
                f"{self._driver_name()} reads no RTM in {mode!r}; "
                f"its modes: {list(self.rtm_modes) or 'none'}."
            )
        self._rtm_sequence = 0
        self._start_rtm(mode)

    def rtm_read(self, interval: float = 0.0) -> Optional[RtmFrame]:
        """Read RTM once, keep it as ``last_rtm_frame`` and emit it on
        ``rtm_frame``; None when the instrument had nothing new."""
        read = self._read_rtm()
        if read is None:
            return None
        resolution, values, positions = read
        self._rtm_sequence += 1
        frame = RtmFrame(
            sequence=self._rtm_sequence,
            timestamp=time.time(),
            interval=interval,
            resolution=tuple(resolution),
            values=values,
            positions=positions,
        )
        self.last_rtm_frame = frame
        self.rtm_frame.emit(frame)
        return frame

    def rtm_stop(self) -> None:
        """Stop reading RTM."""
        self._stop_rtm()

    def connect(self: _M) -> _M:
        super().connect()
        # Read once, so that every report from a run is a change and signals.
        self.progress.get_value()
        return self

    def read_progress(self) -> MillingProgress:
        return self._progress

    def read_readings(self) -> Dict[str, Any]:
        return self._readings.snapshot()

    def _on_reading(self, entry: Dict[str, Any]) -> None:
        """A step reported *entry*: keep it, and announce it when notable."""
        entry = self._readings.add(entry)
        try:
            self.readings.report(self._readings.snapshot())
            if entry["notable"]:
                self.notable_reading.emit(entry)
        except Exception:  # noqa: BLE001 - announcing must not cost the mill
            logging.debug("could not announce a milling reading", exc_info=True)

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

    @classmethod
    def own_name(cls) -> Optional[str]:
        """The name of the block a recipe keeps this driver's own settings in: the
        driver's registry name. None for a driver with none."""
        if cls.settings_class is None:
            return None
        from fibsem.drivers.registry import milling_settings_name

        return milling_settings_name(cls.settings_class)

    @classmethod
    def own_settings(cls, settings: FibsemMillingSettings) -> Any:
        """This driver's own settings in *settings*, an instance of its
        ``settings_class``."""
        block = settings.driver.get(cls.own_name())
        return block if isinstance(block, cls.settings_class) else cls.settings_class()

    def ignored_settings(self, settings: FibsemMillingSettings) -> Optional[str]:
        """The warning for the fields *settings* sets, away from their defaults,
        that this driver does not mill with; None when there are none."""
        default = FibsemMillingSettings()
        ignored = [
            f"{name}: {getattr(settings, name)!r}"
            for name in field_specs(FibsemMillingSettings)
            if name not in self.setting_names
            and name not in _NOT_A_SETTING
            and getattr(settings, name) != getattr(default, name)
        ]
        if not ignored:
            return None
        return (
            f"{', '.join(ignored)} {'is' if len(ignored) == 1 else 'are'} ignored "
            f"on this instrument ({type(self).__name__})."
        )

    def supported_settings(
        self, channel: BeamType = BeamType.ION
    ) -> Dict[str, ParameterMetadata]:
        """The recipe (`FibsemMillingSettings`) fields this instrument mills with on
        ``channel``, each with its choices and limits; the fields left out it ignores.
        The driver's own are named by their path, ``driver.Tescan.preset``.

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
        for name, spec in self._setting_specs().items():
            beam_parameter = spec.bound_name
            if beam_parameter not in beam.parameters:
                available[name] = self._metadata_of(name)
            elif _settable(beam, beam_parameter):
                available[name] = beam.parameters[beam_parameter].metadata
        return available

    def _setting_specs(self) -> Dict[str, Parameter]:
        """The declared field of each recipe setting this driver lists, by its path:
        its ``setting_names``, then its own (``driver.Tescan.preset``)."""
        specs = FibsemMillingSettings.params
        fields = {name: specs[name] for name in self.setting_names}
        if self.settings_class is not None:
            prefix = f"driver.{self.own_name()}."
            fields.update(
                (prefix + name, spec)
                for name, spec in self.settings_class.params.items()
            )
        return fields

    def settings_schema(self, channel: BeamType = BeamType.ION) -> Dict[str, Any]:
        """``supported_settings`` as one JSON Schema object, a property per setting
        by its path, in the shape a monitor config's ``describe()`` has. A number the
        beam lists values for (the milling current) keeps any value, so its listed
        ones are its ``examples``."""
        from fibsem.schema import _plain, parameter_schema

        specs = self._setting_specs()
        properties: Dict[str, Any] = {}
        for name, reported in self.supported_settings(channel).items():
            spec = specs[name]
            numeric = spec.choices is None and spec.type in (int, float)
            listed = reported.choices if numeric else None
            if listed is not None:
                reported = dataclasses.replace(reported, choices=None)
            schema = parameter_schema(spec.with_reported(reported), {})
            if listed:
                schema["examples"] = [_plain(choice) for choice in listed]
            properties[name] = schema
        return {
            "type": "object",
            "title": f"{type(self).__name__} settings",
            "properties": properties,
        }

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
            return ParameterMetadata(choices=PATTERNING_MODES)
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
        ignored = self.ignored_settings(settings)
        if ignored is not None:
            logging.warning(ignored)
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
            with self._milling_channel():
                for pattern, hook in zip(patterns, hooks):
                    self._draw(pattern, hook)
                    self._drawn.append(pattern)
        finally:
            self._set_drawn(self._drawn)

    @contextmanager
    def _milling_channel(self) -> Iterator[None]:
        """Hold the imaging channel with the milling beam selected on it, for a
        patterning call that acts on whatever view is active (AutoScript's): an FM
        acquisition beside a coincidence mill can't move it in between."""
        channel = getattr(self.parent, "milling_channel", BeamType.ION)
        try:
            beam = self.beam(channel)
        except RoleUnfilled:
            beam = None
        if beam is None:
            yield
            return
        with beam.claim_channel():
            yield

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

    @command(line=_prepare_line)
    def prepare(
        self,
        settings: FibsemMillingSettings,
        patterns: Sequence[FibsemPatternSettings],
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """``setup`` with ``settings``, then ``draw`` ``patterns``. Answers the
        current the milling beam reads after the setup, in amps, and a warning when
        the recipe asked for a current the beam doesn't list or set fields this
        instrument ignores (`_milling_current`)."""
        self.setup(settings, name)
        self.draw(patterns)
        return self._milling_current(live=True)

    def current_used(self) -> Dict[str, Any]:
        """The current the last setup left the milling beam at, and the warning
        when it is not the one asked for (`_milling_current`, not read again)."""
        return self._milling_current()

    def _milling_current(self, live: bool = False) -> Dict[str, Any]:
        """The current what is drawn mills at: ``milling_current``, the milling
        beam's current in amps (read now when *live*, else the last one read or
        written; None before there is one), and ``warning``, set when the last
        setup asked for a current that is not one of those the beam lists (a beam
        that lists no currents, or a driver that sets the current another way,
        Tescan's preset, has none), or set fields this instrument ignores
        (`ignored_settings`)."""
        settings = self._set_up
        ignored = self.ignored_settings(settings) if settings is not None else None
        beam = self.beam(settings.milling_channel if settings else BeamType.ION)
        if beam is None or "current" not in beam.parameters:
            return {"milling_current": None, "warning": ignored}
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
        warnings = [w for w in (_unlisted(asked, used, parameter), ignored) if w]
        return {"milling_current": used, "warning": " ".join(warnings) or None}

    @command(line=_said("Start milling what is drawn"))
    def start(self) -> None:
        """Start milling what is drawn, and return at once. Nothing follows it:
        no ``progress`` and no ``milling`` event, so an agent mills with ``run``."""
        self._start()

    @command(line=_said("Stop the mill now running"))
    def stop(self) -> None:
        """Stop milling: the pattern now, and the milling task it is part of, which
        starts no later stage. A ``run`` in progress ends and answers ``cancelled``.

        Refused with `NothingToStop` when nothing is milling and no task is running.
        """
        # The run first: one that has yet to register its task or start its
        # step takes the stop from it.
        run = self.runs.latest
        starting = run is not None and run.state == RUNNING and run.command in _RUNS
        if starting:
            run.stop.set()
        task_stop = self._task_stop
        if task_stop is None and not starting and not self._is_milling():
            raise NothingToStop("Nothing is milling.")
        if task_stop is not None:
            task_stop.set()
        self._end_step(MillingStepEnd(STOPPED, reason="Milling stopped."))

    @command(line=_said("End the milling stage now running, keeping what it milled"))
    def end(self, reason: str, by: Optional[str] = None) -> None:
        """End this stage, completed: the mill step now stops, and its stage run
        starts no further step. A run in progress answers ``completed``; the
        milling task goes on to its next stage. Only a stage that mills
        through ``run`` has steps to end: coincidence milling, which starts
        its own mill, ignores it until it moves to monitors.

        Refused with `NothingToStop` when nothing is milling and no stage runs.
        *by* is who the call is marked as, when it is (a server request), not
        what a client says.
        """
        stage_run = self._stage_run
        if stage_run is None and not self._is_milling():
            raise NothingToStop("Nothing is milling.")
        actor = current_actor()
        if actor is not None:
            by = actor.value
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

    def _stopped_before_it_mills(self) -> bool:
        """Whether a stop came for the run going on here before it could
        take one: before its task or its step was there to stop."""
        run = current_run()
        return run is not None and run.service == self.name and run.stop.is_set()

    def _is_milling(self) -> bool:
        """Whether a run is in progress, or the instrument says it is milling (a
        ``start`` returns at once)."""
        if self._running:
            return True
        try:
            return self._poll().milling_state in ACTIVE_MILLING_STATES
        except Exception:  # noqa: BLE001 - unsure counts as milling: stop it
            return True

    @command(line=_said("Pause the mill"))
    def pause(self) -> None:
        """Pause milling. While a monitor has the pattern paused (an SEM
        peek), it stays paused when the monitor is done."""
        step = self._step
        if step is None:
            self._pause()
        else:
            step.hold(self._pause)

    @command(line=_said("Resume the paused mill"))
    def resume(self) -> None:
        """Resume paused milling. While a monitor has the pattern paused, it
        resumes when the monitor is done."""
        step = self._step
        if step is None:
            self._resume()
        else:
            step.release(self._resume)

    @query
    def follow_readings(self, since: int = 0, timeout: float = 0.0) -> Dict[str, Any]:
        """Every monitor reading numbered after *since*, across steps, waiting up
        to *timeout* seconds (at most 30) for one: ``{seq, readings, missed}``.
        Pass the answer's ``seq`` as the next *since* to follow along."""
        return self._readings.follow(since, timeout)

    @command(long=True, line=_run_line)
    def run(
        self, stop_event: Optional[Union[threading.Event, AnyStopEvent]] = None
    ) -> str:
        """Mill what is drawn and return how it ended, reporting ``progress``.

        The patterns are cleared at the end, however the run ends. A ``stop_event``
        set while it runs stops the beam, and the run then raises
        `OperationCancelledError`; ``stop`` from another thread ends it, and it
        answers ``cancelled`` rather than ``completed``. A failure stops
        the beam before it is raised.

        Outside a milling task, the run is on the record as one ``milling`` event,
        with the name ``setup`` was given and how it ended.
        """
        if self._run_context or getattr(self._part_of, "depth", 0):
            # A task's run, or a spot burn's: that records the whole.
            return self._run(stop_event)
        return self._run_on_the_record(stop_event)

    @command(long=True, line=_run_task_line)
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
            if self._stopped_before_it_mills():
                stop_event.set()
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
                # stopped task answered none.
                os.makedirs(ended["path"], exist_ok=True)
            else:
                # Nothing was saved (a stage that acquires no images, as
                # run_stage's usually is): answer no folder rather than one
                # that isn't there.
                ended["path"] = None
        return ended

    @command(long=True, line=_run_stage_line)
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
        drew. An RGB image the saved image's shape, scaled down to 1024 px wide,
        drawn as every milling picture is (`milling_picture`); reads the file
        and moves nothing."""
        from fibsem.devices.wire import Frame
        from fibsem.milling.patterning.picture import milling_picture

        image = FibsemImage.load(image_path)
        if stages is None:
            drawn = [_Drawn(self.drawn_patterns)] if self.drawn_patterns else []
        else:
            drawn = list(stages)
        data = milling_picture(image, drawn, metadata_bar=False)
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
    def list_options(self) -> Dict[str, Any]:
        """The names a stage's ``pattern.name`` and ``strategy.name`` take, the
        monitors this instrument can run (each name with its config's fields), and
        the recipe settings it mills with on each channel it mills on
        (`settings_schema`)."""
        from fibsem.milling.monitors import get_monitors
        from fibsem.milling.patterning import get_pattern_names
        from fibsem.milling.strategy import get_strategy_names

        return {
            "patterns": get_pattern_names(),
            "strategies": get_strategy_names(),
            "monitors": {
                name: monitor.config_class.describe()
                for name, monitor in get_monitors().items()
                if monitor.available(self)
            },
            "settings": {
                channel.name: schema
                for channel in (BeamType.ION, BeamType.ELECTRON)
                if (schema := self.settings_schema(channel))["properties"]
            },
        }

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
        records one ``milling`` event with how it ended, as a task's run does.
        It starts as a task does, so the milling panel's
        Stop and the progress bars follow it."""
        payload: Dict[str, Any] = {
            # The id of its run on this service (`fibsem.devices.runs`), so
            # its events and its run share one id.
            "milling_run_id": current_run_id(self.name) or str(uuid.uuid4()),
            "milling_task_name": self._name or UNNAMED_RUN,
            "started_at": now_iso(),
        }
        self._report(
            MillingProgress(
                status=MillingProgressStatus.TASK_STARTED,
                milling_run_id=payload["milling_run_id"],
                milling_task_name=payload["milling_task_name"],
            )
        )
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
            # terminal report, and otherwise stay at the last tick.
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
        monitors = _monitors_of(stage)
        views = sorted({m.view for m in monitors if m.view})
        if len(views) > 1:
            raise MixedMonitorViews(
                f"The monitors of {getattr(stage, 'name', 'this stage')} need "
                f"different views ({', '.join(views)}); the instrument has one."
            )
        index = len(stage_run.steps) if stage_run else 0
        step = MillingStep(
            index=index, milling=self, stage=stage, seq=self._readings.next_seq
        )
        step.live = [m for m in monitors if m.view]
        step.view = views[0] if views else None
        if stage_run is not None:
            stage_run.steps.append(step)
        # The `readings` parameter shows this step's, none for a plain step.
        self._readings.start_step(getattr(stage, "name", None), index)
        if monitors:
            step.readings.connect(self._on_reading)
        self._step = step
        self._before_run()
        self._running = True
        try:
            total = float(self._estimate() or 0.0)
            if self._stopped_before_it_mills():
                step.end_as(MillingStepEnd(STOPPED, reason="Milling stopped."))
            for monitor in monitors:
                monitor.start(step)
            if not step.ending.is_set():
                self._start()
                # Starting may select the milling view (AutoScript).
                step.restore_view()
                milled = 0.0
                if step.view not in (None, "fib"):
                    # A live monitor holds another view: no looks at the
                    # instrument until the milling time is milled.
                    milled = MillingClock(self, step, stop_event).watch(total)
                MillingStateMonitor(self, step, stop_event).watch(total, milled)
        except BaseException as error:
            step.end_as(MillingStepEnd.from_error(error))
            self._abort()
            raise
        finally:
            self._running = False
            # However the step ended: its end is set, so monitors leave their loops.
            step.end_as(MillingStepEnd(FINISHED, by="instrument"))
            step.monitors = [monitor.stop() for monitor in monitors]
            if monitors:
                step.report(
                    {"end": step.end.to_dict()},
                    notable=True,
                    monitor="step",
                    keep=False,
                )
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
    def estimate_time(
        self, stages: Optional[List[FibsemMillingStage]] = None
    ) -> Dict[str, Any]:
        """How long milling what is drawn takes: ``seconds``, with the current it
        mills at and any warning about it (`_milling_current`, from what the
        setup wrote, not read again). With *stages*, how long they would take
        before anything is drawn (``estimate_stage``): ``seconds`` in all and each
        stage's. Moves nothing, so a query."""
        if stages is None:
            return {"seconds": self._estimate(), **self._milling_current()}
        each = [
            {"name": stage.name, "seconds": float(self.estimate_stage(stage))}
            for stage in stages
        ]
        return {"seconds": sum(s["seconds"] for s in each), "stages": each}

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

    @command(line=_said("Clear the drawn patterns"))
    def clear(self) -> None:
        """Remove every drawn pattern."""
        with self._milling_channel():
            self._clear()
        self._set_drawn([])

    @command(line=_said("Put the beam back as it was before milling"))
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

    @command(line=_said("Finish milling: clear the patterns and put the beam back"))
    def finish(self) -> None:
        """Clear the patterns and put the milling beam back as ``setup`` found it."""
        self.clear()
        self.restore()

    def _save(self, beam: Beam) -> None:
        """Keep the beam's conditions that it has, can set, and reports."""
        self._saved_beam = beam
        self._saved = save_beam_conditions(beam, SAVED_BEAM_CONDITIONS)

    def _write_back(self, beam: Beam, name: str, value: Any) -> None:
        """Write one saved condition back to the beam, with no new checks."""
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

    def _start_rtm(self, mode: str) -> None:
        raise NotImplementedError(f"{self._driver_name()} has no RTM.")

    def _read_rtm(
        self,
    ) -> Optional[Tuple[Tuple[int, int], Dict[int, Any], Dict[int, Any]]]:
        """One read: the image resolution (width, height), and by draw index each
        monitored pattern's values and their positions (n x 2 pixels); None
        when there is nothing new."""
        raise NotImplementedError(f"{self._driver_name()} has no RTM.")

    def _stop_rtm(self) -> None:
        pass

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


class BeamMilling(Milling):
    """Milling where the recipe sets the milling beam's own conditions (field of
    view, voltage, current) and the instrument patterns with an application file
    and a patterning mode: ThermoFisher's, through AutoScript or Odemis, and the
    Demo's.

    The sequence and its state checks are here; a driver implements the vendor
    calls: ``_select`` (the milling view, the application file and the patterning
    mode), ``_set_patterning_mode``, ``_start_patterning``, ``_stop_patterning``,
    ``_pause_patterning`` and ``_resume_patterning``, and ``_locked`` when its calls
    go under a lock."""

    setting_names = (
        "milling_channel",
        "hfw",
        "milling_current",
        "milling_voltage",
        "application_file",
        "patterning_mode",
    )

    def _locked(self) -> ContextManager[Any]:
        """The lock the vendor calls go under, held across a state check and the
        call it decides; none by default."""
        return nullcontext()

    def _setup(self, settings: FibsemMillingSettings, name: Optional[str]) -> None:
        if settings.patterning_mode not in PATTERNING_MODES:
            raise ValueError(
                f"Patterning mode {settings.patterning_mode} not supported. "
                f"Supported modes: {', '.join(PATTERNING_MODES)}"
            )
        channel = settings.milling_channel
        self.parent.milling_channel = channel
        with self._locked():
            self._select(settings)
            self._clear()
        beam = self.beam(channel)
        beam.hfw.write_through(settings.hfw)
        # voltage before current: the available ion currents are calibrated per voltage
        beam.voltage.write_through(settings.milling_voltage)
        beam.current.write_through(settings.milling_current)
        logging.debug({"msg": "setup_milling", "mill_settings": settings.to_dict()})

    def _select(self, settings: FibsemMillingSettings) -> None:
        """Select the milling view on ``settings.milling_channel``, and set the
        application file and the patterning mode new patterns are made with."""
        raise NotImplementedError

    def _start(self) -> None:
        with self._locked():
            if self.read_state() is MillingState.IDLE:
                self._start_patterning()
                logging.info("Starting milling...")

    def _stop(self) -> None:
        with self._locked():
            if self.read_state() in ACTIVE_MILLING_STATES:
                logging.info("Stopping milling...")
                self._stop_patterning()
                logging.info("Milling stopped.")

    def _pause(self) -> None:
        with self._locked():
            if self.read_state() is MillingState.RUNNING:
                logging.info("Pausing milling...")
                self._pause_patterning()
                logging.info("Milling paused.")

    def _resume(self) -> None:
        with self._locked():
            if self.read_state() is MillingState.PAUSED:
                logging.info("Resuming milling...")
                self._resume_patterning()
                logging.info("Milling resumed.")

    def _restore(self) -> None:
        # The patterning mode persists on the instrument (xT), so a stage left in
        # Parallel would carry over to the next one unless reset.
        self._set_patterning_mode("Serial")

    def _set_patterning_mode(self, mode: str) -> None:
        raise NotImplementedError

    def _start_patterning(self) -> None:
        raise NotImplementedError

    def _stop_patterning(self) -> None:
        raise NotImplementedError

    def _pause_patterning(self) -> None:
        raise NotImplementedError

    def _resume_patterning(self) -> None:
        raise NotImplementedError


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

    def watch(self, total: float, milled: float = 0.0) -> None:
        """Look at the instrument until the step is over; *milled* seconds are
        milled already."""
        milling, step = self.milling, self.step
        start_time = time.time() - milled
        began = last = time.monotonic()
        elapsed = milled
        # Milled time counted already means the pattern was running.
        started = milled > 0
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


class MillingClock:
    """Measures the milled time while a live monitor holds a view other than
    the milling one, so the instrument is not looked at: the time the pattern
    mills, on the step's own watch, its pauses left out. Ends when the milling
    time is milled, or the step ends; the state monitor then sees the pattern
    out (a pattern is never cut off). Pauses made outside fibsem are not seen."""

    def __init__(
        self,
        milling: Milling,
        step: MillingStep,
        stop_event: Optional[Union[threading.Event, AnyStopEvent]],
    ):
        self.milling = milling
        self.step = step
        self.stop_event = stop_event

    def watch(self, total: float) -> float:
        """Measure until the step is over or *total* seconds are milled; the
        seconds milled."""
        milling, step = self.milling, self.step
        start_time = time.time()
        step.count_milled_time(True)
        try:
            while not step.ending.is_set():
                if self.stop_event is not None and self.stop_event.is_set():
                    step.end_as(MillingStepEnd(STOPPED, reason="Milling stopped."))
                    break
                milled = step.milled_time()
                held = step.held or step.pausing.is_set()
                milling._report(
                    progress_update(
                        state=MillingState.PAUSED if held else MillingState.RUNNING,
                        start_time=start_time,
                        total=total,
                        remaining=max(0.0, total - milled),
                    )
                )
                if milled >= total:
                    break
                milling._wait(milling.poll_interval)
        finally:
            step.count_milled_time(False)
        return step.milled_time()


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


# The recipe fields no driver lists: what the recipe is for, not how it mills.
_NOT_A_SETTING = ("milling_channel", "acquire_images", "driver")


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
    service's ``stop`` sets ``stop_event`` and the whole task ends.
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
