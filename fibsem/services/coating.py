"""Coating as a service: a GIS deposition and a sputter coat, each one command.

`Coating` runs over the ``gis`` and ``sputter_coater`` devices, whichever the
instrument has. The devices keep their single commands (insert, open, run, ...); the
service runs the sequences the operator and the agent ask for:

``deposit(seconds)`` moves the stage to the deposition position, when the driver
names one, inserts the needle, warms the gas, opens the valve for ``seconds``,
closes it, retracts the needle and moves the stage back to where it started. The
valve is closed, the needle retracted and the stage moved back however the run
ends, a stop and a failure included. The heater is left on, as the GIS driver
leaves it between grids.

``sputter(seconds)`` coats with the sputter coater, at ``current`` when given. Where
the instrument moves the stage to the coater and back itself (the Arctis), that is
the whole procedure.

The deposition position is the driver's: on an Arctis it is fixed
(``ARCTIS_DEPOSITION_POSITION``), and elsewhere there is none and the deposition
runs where the stage is.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Type, TypeVar, Union

from fibsem.cancellation import AnyStopEvent
from fibsem.devices.core import DeviceParameter, Role, command
from fibsem.devices.gis import GasInjector
from fibsem.devices.sputter_coater import SputterCoater
from fibsem.services.core import Service
from fibsem.structures import FibsemStagePosition, InsertableDeviceState
from fibsem.util import wording

_C = TypeVar("_C", bound="Coating")

ARCTIS_DEPOSITION_POSITION = FibsemStagePosition(
    x=0.0, y=0.0, z=-990e-6, t=math.radians(10), coordinate_system="RAW"
)
"""Where an Arctis deposits from: x and y 0, z -990 µm, tilted 10°."""


class CoatingStatus(str, Enum):
    """What a coating run is doing, or how it ended."""

    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self is not CoatingStatus.RUNNING


@dataclass(frozen=True)
class CoatingProgress:
    """One report on the coating service's ``progress``: the step a run is on,
    and while the valve is open or the coater runs, the seconds left of it."""

    status: CoatingStatus
    step: str = ""
    remaining_time: Optional[float] = None
    total_time: Optional[float] = None
    error: Optional[str] = None


def position_text(position: Optional[FibsemStagePosition]) -> str:
    """ "x 0 µm, y 0 µm, z −990 µm, tilt 10°", or "where the stage is"."""
    if position is None:
        return "where the stage is"
    parts = [
        f"{axis} {wording.number(value * 1e6)} µm"
        for axis, value in (("x", position.x), ("y", position.y), ("z", position.z))
        if value is not None
    ]
    if position.t is not None:
        parts.append(f"tilt {wording.number(math.degrees(position.t))}°")
    return ", ".join(parts)


# --- A command as the operator reads it, in an agent's request (wording) ---


def _deposit_line(kwargs: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    """ "Deposit for 30 s at the deposition position, then move back"."""
    change = "Deposit"
    if kwargs.get("seconds") is not None:
        change += f" for {wording.seconds(float(kwargs['seconds']))}"
    return {
        "what": "GIS deposition",
        "change": change + ": insert, warm, open, close, retract, then move back",
    }


def _sputter_line(kwargs: Dict[str, Any], ctx: wording.ActionContext) -> Dict[str, Any]:
    """ "Sputter for 30 s at 120 nA"."""
    change = "Sputter"
    if kwargs.get("seconds") is not None:
        change += f" for {wording.seconds(float(kwargs['seconds']))}"
    if kwargs.get("current") is not None:
        change += f" at {wording.current(kwargs['current'])}"
    if kwargs.get("grids"):
        change += f" on grids {', '.join(str(n) for n in kwargs['grids'])}"
    return {"what": "Sputter coat", "change": change}


class Coating(Service):
    """GIS deposition and sputter coating.

    ``deposit`` and ``sputter`` each run one coat and return how it ended; ``stop``
    ends a deposition from another thread (a sputter run can't be stopped).
    """

    gis = Role(GasInjector, required=False, doc="The gas injection system.")
    sputter_coater = Role(SputterCoater, required=False, doc="The sputter coater.")

    progress = DeviceParameter(
        CoatingProgress, doc="The run's step and time left; read-only."
    )
    deposition_position = DeviceParameter(
        FibsemStagePosition,
        doc="Where a deposition moves the stage to, fixed by the instrument; "
        "absent where it deposits where the stage is.",
    )

    # How often a deposition reports while the valve is open, in seconds.
    poll_interval: float = 1.0

    def __init__(
        self,
        name: str = "coating",
        position: Optional[FibsemStagePosition] = None,
        **kwargs: Any,
    ):
        super().__init__(name, **kwargs)
        self._position = position
        self._progress: Optional[CoatingProgress] = None
        self._stop_requested = threading.Event()

    def connect(self: _C) -> _C:
        super().connect()
        self.progress.get_value()
        return self

    def read_progress(self) -> Optional[CoatingProgress]:
        return self._progress

    def available_deposition_position(self) -> bool:
        return self._position is not None

    def read_deposition_position(self) -> FibsemStagePosition:
        return self._position

    def _report(self, progress: CoatingProgress) -> None:
        self._progress = progress
        self.progress.report(progress)

    def _has_gis(self) -> bool:
        return "gis" in self._roles

    def _has_coater(self) -> bool:
        return "sputter_coater" in self._roles

    # -- commands ----------------------------------------------------------------

    @command(available=_has_gis, long=True, line=_deposit_line)
    def deposit(
        self,
        seconds: float,
        stop_event: Optional[Union[threading.Event, AnyStopEvent]] = None,
    ) -> CoatingStatus:
        """Deposit with the GIS for *seconds* with the valve open: move to the
        deposition position, insert, warm, open, wait, close, retract, and move
        back. A stop closes the valve early and ends the run as cancelled; the
        valve, needle and stage are put back however it ends."""
        seconds = float(seconds)
        if seconds <= 0:
            raise ValueError(f"A deposition cannot last {seconds} s.")
        self._stop_requested.clear()
        self.session.record(
            "coating_deposit_started",
            {"seconds": seconds, "position": position_text(self._position)},
        )
        gis = self.gis
        start = None
        try:
            if self._position is not None:
                start = self.parent.get_stage_position()
                self._step("Moving to the deposition position")
                self._move_to(self._position)
            self._step("Inserting the needle")
            gis.insert()
            self._step("Warming the gas")
            gis.heater_on()
            if self._stopped(stop_event):
                return self._end(CoatingStatus.CANCELLED)
            self._step("Depositing", seconds, seconds)
            gis.open()
            status = self._hold_open(seconds, stop_event)
            self._close_and_retract(gis)
            if start is not None:
                self._step("Moving back")
                self._move_to(start)
                start = None
            return self._end(status)
        except Exception as e:
            logging.error(f"The deposition failed: {e}")
            self._report(CoatingProgress(status=CoatingStatus.FAILED, error=str(e)))
            raise
        finally:
            self._put_back(gis, start)

    @command(available=_has_coater, long=True, line=_sputter_line)
    def sputter(
        self,
        seconds: float,
        current: Optional[float] = None,
        grids: Optional[Sequence[int]] = None,
    ) -> CoatingStatus:
        """Sputter for *seconds*, at *current* (amperes) when given, on the
        numbered *grids* where the coater moves to each itself."""
        seconds = float(seconds)
        coater = self.sputter_coater
        self.session.record(
            "coating_sputter_started",
            {
                "seconds": seconds,
                "current": current,
                "grids": None if grids is None else [int(n) for n in grids],
            },
        )
        try:
            if current is not None:
                coater.current.set_value(float(current))
            self._step("Sputtering", seconds, seconds)
            coater.run(seconds, grids)
            return self._end(CoatingStatus.FINISHED)
        except Exception as e:
            logging.error(f"The sputter coat failed: {e}")
            self._report(CoatingProgress(status=CoatingStatus.FAILED, error=str(e)))
            raise

    @command
    def stop(self) -> None:
        """End the deposition in progress as cancelled: the valve closes, the needle
        retracts and the stage moves back."""
        self._stop_requested.set()

    # -- the run's parts ----------------------------------------------------------

    def _step(
        self,
        step: str,
        remaining: Optional[float] = None,
        total: Optional[float] = None,
    ) -> None:
        logging.info(f"Coating: {step}")
        self._report(
            CoatingProgress(
                status=CoatingStatus.RUNNING,
                step=step,
                remaining_time=remaining,
                total_time=total,
            )
        )

    def _end(self, status: CoatingStatus) -> CoatingStatus:
        self._report(CoatingProgress(status=status))
        return status

    def _stopped(
        self, stop_event: Optional[Union[threading.Event, AnyStopEvent]]
    ) -> bool:
        if self._stop_requested.is_set():
            return True
        return stop_event is not None and stop_event.is_set()

    def _hold_open(
        self,
        seconds: float,
        stop_event: Optional[Union[threading.Event, AnyStopEvent]],
    ) -> CoatingStatus:
        """Wait *seconds* with the valve open, reporting as it goes. CANCELLED when
        stopped."""
        remaining = seconds
        while remaining > 0:
            if self._stopped(stop_event):
                logging.info("Deposition stopped with the valve open.")
                return CoatingStatus.CANCELLED
            step = min(self.poll_interval, remaining)
            self._wait(step)
            remaining -= step
            self._step("Depositing", max(0.0, remaining), seconds)
        return CoatingStatus.FINISHED

    def _close_and_retract(self, gis: GasInjector) -> None:
        self._step("Closing the valve")
        gis.close()
        self._step("Retracting the needle")
        gis.retract()

    def _put_back(self, gis: GasInjector, start: Optional[FibsemStagePosition]) -> None:
        """After a run that didn't finish cleanly: close the valve, retract the
        needle and move the stage back, each tried whatever the others do. A failure
        here is logged, never raised over the error that ended the run."""
        for what, done, undo in (
            ("close the valve", lambda: not gis.opened.get_value(), gis.close),
            (
                "retract the needle",
                lambda: gis.state.get_value() is InsertableDeviceState.RETRACTED,
                gis.retract,
            ),
        ):
            try:
                if not done():
                    undo()
            except Exception:
                logging.exception(f"After the deposition, failed to {what}")
        if start is not None:
            try:
                self._move_to(start)
            except Exception:
                logging.exception("After the deposition, failed to move the stage back")

    def _move_to(self, position: FibsemStagePosition) -> None:
        """The safe move, with the FM objective out first: it must not be over the
        sample while the stage moves."""
        fm = getattr(self.parent, "fm", None)
        if fm is not None:
            fm.objective.retract()
        self.parent.stage_movement.safe_absolute_stage_movement(position)

    def _wait(self, seconds: float) -> None:
        """Wait while the valve is open."""
        time.sleep(seconds)


def bind_coating(
    microscope: Any,
    service: Type[_C] = Coating,
    position: Optional[FibsemStagePosition] = None,
) -> Optional[_C]:
    """Build a microscope's coating service over its gas injector and sputter
    coater, or None when it has neither. *position* is where a
    deposition moves the stage to; None deposits where the stage is."""
    roles: Dict[str, Any] = {}
    for device in microscope.devices.values():
        for role, interface in (
            ("gis", GasInjector),
            ("sputter_coater", SputterCoater),
        ):
            if isinstance(device, interface) and role not in roles:
                roles[role] = device
    if not roles:
        return None
    coating = service(parent=microscope, position=position)
    coating.fill_roles(**roles)
    coating.connect()
    return coating
