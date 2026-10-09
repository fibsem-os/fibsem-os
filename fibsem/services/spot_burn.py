"""Spot burn as a service: points on the sample exposed one after another with a beam
held still, and the beam put back as it was found.

`SpotBurn` is what every backend shares. It burns with the ion beam only. ``run`` takes a `SpotBurnSettings`: it drops
the points outside the image (0 to 1), records what it is about to burn, reports
``progress`` as it goes and once more at the end (finished, cancelled or failed), and
puts the beam back afterwards however the run ends. A restore that fails is logged
and never hides the error that ended the run.

The burn itself is the point-by-point one any beam with the scan commands can do:
blank, park the beam on the point (the ``spot`` command), unblank, wait the exposure
time, then back to full frame at the end. Demo, ThermoFisher and Odemis burn this way.
A driver whose instrument burns another way overrides ``_burn``.

What a run changes is the burn current, so that is what it saves and writes back
(``saved_conditions``); a driver that burns with other conditions names them.

`FibsemMicroscope.run_spot_burn` goes to the microscope's service,
``microscope.spot_burn``, and keeps its signature on every backend.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from fibsem.devices.beam import Beam
from fibsem.devices.core import Parameter, ParameterMetadata, Role, command
from fibsem.imaging.spot import SpotBurnProgress, SpotBurnSettings, SpotBurnStatus
from fibsem.services.core import Service, forward_to, save_beam_conditions
from fibsem.structures import BeamType, Point

if TYPE_CHECKING:
    from fibsem.cancellation import AnyStopEvent

_S = TypeVar("_S", bound="SpotBurn")


class SpotBurn(Service):
    """Spot burns with the ion beam.

    ``run`` burns the points of a `SpotBurnSettings` and returns how it ended;
    ``stop`` ends a run from another thread; ``estimate`` says how long one takes.
    """

    ion = Role(Beam, doc="The ion beam, which burns.")

    # Reported by `run` as it goes; reading it doesn't touch the instrument.
    progress = Parameter(SpotBurnProgress, doc="The run's point and times; read-only.")

    # How often a run reports while a point exposes, in seconds.
    poll_interval: float = 1.0

    # The beam conditions a run changes and puts back, in the order they are written
    # back.
    saved_conditions: Tuple[str, ...] = ("current",)

    # The `SpotBurnSettings` fields this driver's ``run`` reads; it ignores the rest.
    setting_names: Tuple[str, ...] = ("coordinates", "exposure_time", "milling_current")

    def __init__(self, name: str = "spot_burn", **kwargs: Any):
        super().__init__(name, **kwargs)
        self._progress: Optional[SpotBurnProgress] = None
        self._stop_requested = threading.Event()

    def connect(self: _S) -> _S:
        super().connect()
        # Read once, so that every report from a run is a change and signals.
        self.progress.get_value()
        return self

    def read_progress(self) -> Optional[SpotBurnProgress]:
        return self._progress

    def _report(self, progress: SpotBurnProgress) -> None:
        self._progress = progress
        self.progress.report(progress)

    def supported_settings(self) -> Dict[str, ParameterMetadata]:
        """The `SpotBurnSettings` fields this instrument burns with, each with its
        choices and limits; the fields left out it ignores.

        The burn current has the ion beam's own current metadata, and is left out
        when the beam can't set its current.
        """
        beam = self.ion
        supported: Dict[str, ParameterMetadata] = {}
        for name in self.setting_names:
            if name != "milling_current":
                supported[name] = ParameterMetadata()
            elif "current" in beam.parameters and beam.current.settable:
                supported[name] = beam.current.metadata
        return supported

    # -- commands ----------------------------------------------------------------

    @command
    def run(
        self,
        settings: SpotBurnSettings,
        stop_event: Optional[Union[threading.Event, AnyStopEvent]] = None,
    ) -> SpotBurnStatus:
        """Burn the points of ``settings`` and return how the run ended.

        A ``stop_event`` set while it runs, or ``stop`` from another thread, blanks
        the beam and ends the run as cancelled, without raising. A failure is
        reported and raised. The beam is put back either way.
        """
        # Protocol-editor fields can arrive as strings ("3e-11"). Read into locals:
        # the settings belong to the caller.
        exposure_time = float(settings.exposure_time)
        current = float(settings.milling_current)
        points, dropped = in_bounds(settings.coordinates)
        beam = self.ion
        self._stop_requested.clear()

        self._record_started(points, exposure_time, current, len(dropped))
        total = len(points) * exposure_time
        self._report(
            SpotBurnProgress(
                status=SpotBurnStatus.BURNING,
                current_point=0,
                total_points=len(points),
                remaining_time=exposure_time,
                total_remaining_time=total,
                total_estimated_time=total,
            )
        )
        if not points:
            logging.warning("No spot burn coordinates to burn.")
            self._report_end(SpotBurnStatus.FINISHED, 0)
            return SpotBurnStatus.FINISHED

        saved = save_beam_conditions(beam, self.saved_conditions)
        try:
            self._setup(beam, current)
            status = self._burn(beam, points, exposure_time, current, stop_event)
            self._report_end(status, len(points))
            return status
        except Exception as e:
            logging.error(f"Error in run_spot_burn: {e}")
            self._report(SpotBurnProgress(status=SpotBurnStatus.FAILED, error=str(e)))
            raise
        finally:
            self._finish(beam)
            for name in self.saved_conditions:
                if name in saved:
                    try:
                        getattr(beam, name).write_through(saved[name])
                    except Exception:
                        logging.exception(
                            f"Failed to restore the beam {name} after the spot burn"
                        )

    @command
    def stop(self) -> None:
        """End the run in progress as cancelled."""
        self._stop_requested.set()

    @command
    def estimate(self, settings: SpotBurnSettings) -> float:
        """How long burning ``settings`` takes, in seconds."""
        points, _ = in_bounds(settings.coordinates)
        return len(points) * float(settings.exposure_time)

    # -- the run's parts ----------------------------------------------------------

    def _stopped(
        self, stop_event: Optional[Union[threading.Event, AnyStopEvent]]
    ) -> bool:
        if self._stop_requested.is_set():
            return True
        return stop_event is not None and stop_event.is_set()

    def _record_started(
        self,
        points: List[Point],
        exposure_time: float,
        current: float,
        dropped: int,
    ) -> None:
        """Record what the run is about to burn, on the microscope's record."""
        record = getattr(self.parent, "_record_spot_burn_started", None)
        if record is None:
            return
        # A driver that ignores the requested current records none.
        if "milling_current" not in self.setting_names:
            current = None
        record(points, BeamType.ION, exposure_time, current, dropped)

    def _report_end(self, status: SpotBurnStatus, points: int) -> None:
        self._report(
            SpotBurnProgress(status=status, current_point=points, total_points=points)
        )

    # -- what a driver may override ---------------------------------------------

    def _setup(self, beam: Beam, current: float) -> None:
        """Set the beam up to burn: the burn current."""
        beam.current.write_through(current)

    def _burn(
        self,
        beam: Beam,
        points: Sequence[Point],
        exposure_time: float,
        current: float,
        stop_event: Optional[Union[threading.Event, AnyStopEvent]],
    ) -> SpotBurnStatus:
        """Expose each point in turn, reporting as it goes: blank, park the beam on
        the point, unblank, wait. Returns FINISHED, or CANCELLED when stopped."""
        n = len(points)
        total_remaining = n * exposure_time
        for i, point in enumerate(points, 1):
            if self._stopped(stop_event):
                logging.info(f"Spot burn cancelled before point {i}/{n}.")
                return SpotBurnStatus.CANCELLED
            # The experiment replay reads this line from the log (its `_SPOT`), so
            # keep its wording and this method's name.
            logging.info(
                f"burning spot {i}: {point}, exposure time: {exposure_time}, "
                f"milling current: {current}"
            )
            beam.blank()
            beam.spot(point)
            beam.unblank()

            remaining = exposure_time
            while remaining > 0:
                if self._stopped(stop_event):
                    beam.blank()
                    logging.info(f"Spot burn cancelled during point {i}/{n}.")
                    return SpotBurnStatus.CANCELLED
                # The last wait is only what is left, so an exposure that isn't a
                # whole number of polls is timed exactly.
                step = min(self.poll_interval, remaining)
                self._wait(step)
                remaining -= step
                total_remaining -= step
                self._report(
                    SpotBurnProgress(
                        status=SpotBurnStatus.BURNING,
                        current_point=i,
                        total_points=n,
                        remaining_time=max(0.0, remaining),
                        total_remaining_time=max(0.0, total_remaining),
                        total_estimated_time=n * exposure_time,
                    )
                )
        return SpotBurnStatus.FINISHED

    def _finish(self, beam: Beam) -> None:
        """Put the scan back after a run, however it ended: full frame."""
        try:
            beam.full_frame()
        except Exception:
            logging.exception(
                "Failed to restore full-frame scanning after the spot burn"
            )

    def _wait(self, seconds: float) -> None:
        """Wait while a point exposes."""
        time.sleep(seconds)

    @classmethod
    def _can_burn(cls, beam: Optional[Beam]) -> bool:
        """Whether this driver can burn with *beam*: the point-by-point burn needs
        a beam that blanks and parks."""
        return (
            beam is not None
            and beam.commands["spot"].available
            and beam.commands["blank"].available
        )


def in_bounds(points: Sequence[Point]) -> Tuple[List[Point], List[Point]]:
    """The points inside the image (0 to 1 on both axes), and the ones dropped, with
    a warning: the instruments refuse a spot outside the scan field."""
    kept: List[Point] = []
    dropped: List[Point] = []
    for point in points:
        inside = 0 <= point.x <= 1 and 0 <= point.y <= 1
        (kept if inside else dropped).append(point)
    if dropped:
        logging.warning(
            f"Skipping {len(dropped)} spot burn coordinate(s) outside image bounds "
            f"(0-1): {dropped}"
        )
    return kept, dropped


def bind_spot_burn(service: Type[_S], microscope: Any) -> Optional[_S]:
    """Build a microscope's spot burn service of class *service* over its ion beam,
    or None when that beam can't burn (no ion beam, or one that can't park), and
    ``run_spot_burn`` then raises."""
    ion = microscope.beams.get(BeamType.ION)
    if not service._can_burn(ion):
        return None
    spot_burn = service(parent=microscope)
    spot_burn.fill_roles(ion=ion)
    spot_burn.connect()
    signal = getattr(microscope, "spot_burn_progress_signal", None)
    if signal is not None:
        spot_burn.progress.changed.connect(forward_to(signal))
    return spot_burn
