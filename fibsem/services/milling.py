"""Milling as a service: a recipe and patterns handed to the instrument, run, and the
beam put back as it was found.

`Milling` is what every backend shares. A driver subclasses it and implements the
hooks (``_setup``, ``_draw``, ``_start``, ``_stop``, ``_pause``, ``_resume``,
``_estimate``, ``_clear`` and ``read_state``) the way the instrument mills: an
instrument queue on ThermoFisher and Odemis, a DrawBeam layer on Tescan, a list on the
Demo. The driver also chooses the per-pattern application file and applies the
recipe's beam conditions, since which recipe fields a backend uses differs (Tescan
mills with a preset, the others with a current and voltage).

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

import logging
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from fibsem.cancellation import raise_if_cancelled
from fibsem.devices.beam import Beam
from fibsem.devices.core import Parameter, ParameterMetadata, Role, command
from fibsem.milling.progress import MillingProgress, MillingProgressStatus
from fibsem.services.core import Service, save_beam_conditions
from fibsem.structures import (
    ACTIVE_MILLING_STATES,
    BeamType,
    FibsemMillingSettings,
    FibsemPatternSettings,
    MillingState,
    get_fields_with_metadata,
)

if TYPE_CHECKING:
    import threading

    from fibsem.cancellation import AnyStopEvent

# The beam conditions milling changes and `restore` puts back, in the order they are
# written back: a preset first, since it sets the others on a column that has one,
# then the voltage before the current, as the instruments want.
SAVED_BEAM_CONDITIONS = ("preset", "voltage", "current", "hfw")

_M = TypeVar("_M", bound="Milling")


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
    state = Parameter(MillingState, doc="Idle, running, paused, ...; read-only.")
    # Reported by `run` as it goes; reading it doesn't touch the instrument.
    # A stage update (`MillingProgress`) with what a run knows: its state, start,
    # total and remaining time. The task and stage fields are a task's to fill.
    progress = Parameter(MillingProgress, doc="The run's state and times; read-only.")

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

    def connect(self: _M) -> _M:
        super().connect()
        # Read once, so that every report from a run is a change and signals.
        self.progress.get_value()
        return self

    def read_progress(self) -> MillingProgress:
        return self._progress

    def _report(self, progress: MillingProgress) -> None:
        self._progress = progress
        self.progress.report(progress)

    def beam(self, channel: BeamType = BeamType.ION) -> Beam:
        """The beam that mills on ``channel``."""
        return self.electron if channel is BeamType.ELECTRON else self.ion

    def supported_settings(
        self, channel: BeamType = BeamType.ION
    ) -> Dict[str, ParameterMetadata]:
        """The recipe (`FibsemMillingSettings`) fields this instrument mills with on
        ``channel``, each with its choices and limits; the fields left out it ignores.

        A field whose ``microscope_parameter`` is one of the milling beam's parameters
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
        fields = get_fields_with_metadata(FibsemMillingSettings)
        for name in self.setting_names:
            beam_parameter = fields[name].get("microscope_parameter")
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
        self._setup(settings, name)

    @command
    def draw(self, patterns: Sequence[FibsemPatternSettings]) -> None:
        """Add ``patterns`` to what the next run mills."""
        for pattern in patterns:
            if not isinstance(pattern, FibsemPatternSettings):
                raise TypeError(f"Expected FibsemPatternSettings, got {type(pattern)}")
            self._draw(pattern)

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
        """Start milling what is drawn, and return."""
        self._start()

    @command
    def stop(self) -> None:
        """Stop milling."""
        self._stop()

    @command
    def pause(self) -> None:
        """Pause milling."""
        self._pause()

    @command
    def resume(self) -> None:
        """Resume paused milling."""
        self._resume()

    @command
    def run(
        self, stop_event: Optional[Union[threading.Event, AnyStopEvent]] = None
    ) -> None:
        """Mill what is drawn and return when it is done, reporting ``progress``.

        The patterns are cleared at the end, however the run ends. A ``stop_event``
        set while it runs stops the beam, and the run then raises
        `OperationCancelledError`; ``stop`` from another thread just ends it. A
        failure stops the beam before it is raised.
        """
        self._before_run()
        try:
            total = float(self._estimate() or 0.0)
            self._start()
            self._monitor(total, stop_event)
        except BaseException:
            self._abort()
            raise
        finally:
            try:
                self._after_run()
            finally:
                self._clear()
        raise_if_cancelled(stop_event, "Milling stopped.")

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

    @command
    def estimate(self) -> float:
        """How long milling what is drawn takes, in seconds."""
        return self._estimate()

    @command
    def clear(self) -> None:
        """Remove every drawn pattern."""
        self._clear()

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

    def _draw(self, pattern: FibsemPatternSettings) -> None:
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


def _settable(beam: Optional[Beam], name: str) -> bool:
    return beam is not None and name in beam.parameters and getattr(beam, name).settable


def bind_milling(service: Type[_M], microscope: Any) -> Optional[_M]:
    """Build a microscope's milling service of class *service* over its beams, or
    None when it has no ion beam (the column is disabled), which leaves the
    microscope's own milling code in charge."""
    beams = microscope.beams
    if BeamType.ION not in beams:
        return None
    milling = service(parent=microscope)
    milling.fill_roles(ion=beams[BeamType.ION])
    if BeamType.ELECTRON in beams:
        milling.fill_roles(electron=beams[BeamType.ELECTRON])
    milling.connect()
    signal = getattr(microscope, "milling_progress_signal", None)
    if signal is not None:
        milling.progress.changed.connect(signal.emit)
    return milling


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
