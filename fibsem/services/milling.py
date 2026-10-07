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
small and is a driver hook: what one look reads (``_poll``: the state, and the time
elapsed, total or remaining where the instrument reports it, as Tescan's DrawBeam does),
and what happens around the run (``_before_run``, ``_after_run``: Tescan loads its
layer and shows a progress bar). Elapsed time the instrument doesn't report is the wall
clock while running, paused time left out. A set ``stop_event`` stops the beam; a
failure stops it and clears the patterns before it is raised.

`ServiceMilling` gives a microscope the old milling methods over its service, so
``setup_milling``, ``draw_patterns``, ``start_milling`` and the rest keep their
signatures.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
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
from fibsem.services.core import Service
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
    from fibsem.structures import (
        FibsemBitmapSettings,
        FibsemCircleSettings,
        FibsemLineSettings,
        FibsemPolygonSettings,
        FibsemRectangleSettings,
    )

# The beam conditions milling changes and `restore` puts back, in the order they are
# written back: a preset first, since it sets the others on a column that has one,
# then the voltage before the current, as the instruments want.
SAVED_BEAM_CONDITIONS = ("preset", "voltage", "current", "hfw")

_M = TypeVar("_M", bound="Milling")


@dataclass(frozen=True)
class MillingRunProgress:
    """Where one run (`Milling.run`) has got to: ``progress``'s value.

    Times are in seconds. ``total`` is the instrument's figure where it reports one,
    otherwise the estimate taken when the run started; ``elapsed`` is the
    instrument's, otherwise the time spent running, paused time left out.
    """

    state: MillingState = MillingState.IDLE
    elapsed: float = 0.0
    total: float = 0.0
    remaining: float = 0.0
    start_time: Optional[float] = None  # when the run started, time.time()


@dataclass(frozen=True)
class MillingPoll:
    """What one look at the instrument found (`Milling._poll`): the state, and the
    times it reports itself; None for each it doesn't."""

    state: MillingState
    elapsed: Optional[float] = None
    total: Optional[float] = None
    remaining: Optional[float] = None


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
    progress = Parameter(
        MillingRunProgress, doc="The run's state and times; read-only."
    )

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

    def __init__(self, name: str = "milling", **kwargs: Any):
        super().__init__(name, **kwargs)
        # The beam conditions the first setup found, until restore puts them back.
        self._saved: Optional[Dict[str, Any]] = None
        self._saved_beam: Optional[Beam] = None
        # What the driver says about its own fields, asked once.
        self._driver_settings: Dict[str, ParameterMetadata] = {}
        self._progress = MillingRunProgress()

    def connect(self: _M) -> _M:
        super().connect()
        # Read once, so that every report from a run is a change and signals.
        self.progress.get_value()
        return self

    def read_progress(self) -> MillingRunProgress:
        return self._progress

    def _report(self, progress: MillingRunProgress) -> None:
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
            if poll.state is MillingState.RUNNING:
                elapsed += now - last
            last = now
            started = started or poll.state in ACTIVE_MILLING_STATES
            if poll.total is not None:
                total = float(poll.total)
            if poll.elapsed is not None:
                elapsed = float(poll.elapsed)
            elif poll.remaining is not None:
                elapsed = max(0.0, total - float(poll.remaining))
            remaining = (
                float(poll.remaining)
                if poll.remaining is not None
                else max(0.0, total - elapsed)
            )
            self._report(
                MillingRunProgress(
                    state=poll.state,
                    elapsed=elapsed,
                    total=total,
                    remaining=remaining,
                    start_time=start_time,
                )
            )
            if poll.state not in ACTIVE_MILLING_STATES:
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
        """Keep the beam's conditions that it has, can set, and reports.

        A condition the beam can't set (a Tescan ion column's current and voltage come
        with its preset) is left out, and so is one it reads as None.
        """
        self._saved_beam = beam
        saved = {}
        for name in SAVED_BEAM_CONDITIONS:
            if name in beam.parameters and getattr(beam, name).settable:
                value = getattr(beam, name).get_value()
                if value is not None:
                    saved[name] = value
        self._saved = saved
        logging.debug({"msg": "milling saved the beam", "saved": self._saved})

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

    def _poll(self) -> MillingPoll:
        """One look at the instrument during a run. By default only the state: the
        loop counts the time itself."""
        return MillingPoll(state=self.read_state())

    def _before_run(self) -> None:
        """Get a drawn run ready to start, before its estimate; by default nothing."""

    def _after_run(self) -> None:
        """Tidy up after a run, however it ended, before the patterns are cleared;
        by default nothing."""

    def _wait(self, seconds: float) -> None:
        """Wait between looks at the instrument."""
        time.sleep(seconds)


class ServiceMilling:
    """The microscope's milling methods, over its milling service (``self.milling``).

    A backend lists this before its own milling code. With a service, each method
    goes to it; without one (no ion beam was built), each falls through to that code.
    """

    milling: Optional[Milling] = None
    milling_channel: BeamType
    # and FibsemMicroscope's set_beam_voltage and set_beam_current

    def setup_milling(self, mill_settings: FibsemMillingSettings) -> None:
        if self.milling is None:
            return super().setup_milling(mill_settings)
        self.milling.setup(mill_settings)

    def draw_rectangle(self, pattern_settings: FibsemRectangleSettings) -> None:
        if self.milling is None:
            return super().draw_rectangle(pattern_settings)
        self.milling.draw([pattern_settings])

    def draw_line(self, pattern_settings: FibsemLineSettings) -> None:
        if self.milling is None:
            return super().draw_line(pattern_settings)
        self.milling.draw([pattern_settings])

    def draw_circle(self, pattern_settings: FibsemCircleSettings) -> None:
        if self.milling is None:
            return super().draw_circle(pattern_settings)
        self.milling.draw([pattern_settings])

    def draw_polygon(self, pattern_settings: FibsemPolygonSettings) -> None:
        if self.milling is None:
            return super().draw_polygon(pattern_settings)
        self.milling.draw([pattern_settings])

    def draw_bitmap_pattern(self, pattern_settings: FibsemBitmapSettings) -> None:
        if self.milling is None:
            return super().draw_bitmap_pattern(pattern_settings)
        self.milling.draw([pattern_settings])

    def start_milling(self) -> None:
        if self.milling is None:
            return super().start_milling()
        self.milling.start()

    def stop_milling(self) -> None:
        if self.milling is None:
            return super().stop_milling()
        self.milling.stop()

    def pause_milling(self) -> None:
        if self.milling is None:
            return super().pause_milling()
        self.milling.pause()

    def resume_milling(self) -> None:
        if self.milling is None:
            return super().resume_milling()
        self.milling.resume()

    def run_milling(
        self,
        milling_current: Optional[float] = None,
        milling_voltage: Optional[float] = None,
        asynch: bool = False,
        stop_event: Optional[Union[threading.Event, AnyStopEvent]] = None,
    ) -> None:
        """Mill what is drawn with the service's run loop (`Milling.run`), which
        reports progress on ``milling_progress_signal``. A current or voltage given
        is set first, where the beam can set it, as the old loop did. ``asynch``, on
        its way out, still starts it the backend's old way and returns."""
        if self.milling is None or asynch:
            return super().run_milling(milling_current, milling_voltage, asynch)
        beam = self.milling.beam(self.milling_channel)
        try:
            if milling_voltage is not None and _settable(beam, "voltage"):
                if beam.voltage.get_value() != milling_voltage:
                    self.set_beam_voltage(
                        voltage=milling_voltage, beam_type=self.milling_channel
                    )
            if milling_current is not None and _settable(beam, "current"):
                if beam.current.get_value() != milling_current:
                    self.set_beam_current(
                        current=milling_current, beam_type=self.milling_channel
                    )
        except Exception as e:
            logging.warning(
                f"Failed to set voltage or current: {e}, voltage={milling_voltage}, "
                f"current={milling_current}"
            )
        logging.info("running milling now...")
        self.milling.run(stop_event=stop_event)

    def get_milling_state(self) -> MillingState:
        if self.milling is None:
            return super().get_milling_state()
        return self.milling.state.get_value()

    def estimate_milling_time(self) -> float:
        if self.milling is None:
            return super().estimate_milling_time()
        return self.milling.estimate()

    def clear_patterns(self) -> None:
        if self.milling is None:
            return super().clear_patterns()
        self.milling.clear()

    def finish_milling(
        self,
        imaging_current: Optional[float] = None,
        imaging_voltage: Optional[float] = None,
    ) -> None:
        """Clear the patterns and put the milling beam back as ``setup_milling`` found
        it. An imaging current or voltage given wins over what was saved."""
        if self.milling is None:
            return super().finish_milling(imaging_current, imaging_voltage)
        self.milling.clear()
        self.milling.restore()
        # only what the beam can set: a Tescan ion column takes both from its preset
        beam = self.milling.beam(self.milling_channel)
        if imaging_voltage is not None and _settable(beam, "voltage"):
            self.set_beam_voltage(
                voltage=imaging_voltage, beam_type=self.milling_channel
            )
        if imaging_current is not None and _settable(beam, "current"):
            self.set_beam_current(
                current=imaging_current, beam_type=self.milling_channel
            )
        logging.debug(
            {
                "msg": "finish_milling",
                "imaging_current": imaging_current,
                "imaging_voltage": imaging_voltage,
            }
        )


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
        milling.progress.changed.connect(_stage_update(signal.emit))
    return milling


def _stage_update(emit: Callable[[Any], None]) -> Callable[[MillingRunProgress], None]:
    """Each change of ``progress`` as the stage update `milling_progress_signal`
    has always carried from a run."""
    from fibsem.milling.progress import MillingProgress, MillingProgressStatus

    def forward(progress: MillingRunProgress) -> None:
        emit(
            MillingProgress(
                status=MillingProgressStatus.STAGE_UPDATE,
                start_time=progress.start_time,
                milling_state=progress.state,
                estimated_time=progress.total,
                remaining_time=progress.remaining,
            )
        )

    return forward
