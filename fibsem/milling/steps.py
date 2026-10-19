"""Mill steps and how they end (FIB-1480).

A strategy carries out a stage (a ``MillingStageRun``) as one or more mill steps:
Standard mills once, Overtilt once per pattern. A ``MillingStep`` is one mill on
the milling service: draw, start, wait, one teardown. Every step ends once, with a
``MillingStepEnd``; the first end set wins.

| kind       | from                                      | the step  |
|------------|-------------------------------------------|-----------|
| finished   | the pattern finished                      | completed |
| ended      | "end this stage", or a strategy's stop    | completed |
| stopped    | a stop                                    | cancelled |
| failed     | an error                                  | failed    |
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional

from psygnal import Signal

from fibsem.cancellation import CANCELLED, COMPLETED, FAILED, ended_by
from fibsem.util.timestamps import now_iso

FINISHED, ENDED, STOPPED, STEP_FAILED = "finished", "ended", "stopped", "failed"

# The status a step, or a stage run, ends with, by how it ended.
_STATUS = {
    FINISHED: COMPLETED,
    ENDED: COMPLETED,
    STOPPED: CANCELLED,
    STEP_FAILED: FAILED,
}


@dataclass(frozen=True)
class MillingStepEnd:
    """How a mill step ended: its kind, who ended it (``instrument``,
    ``strategy``, ``operator``, ``agent``, ``workflow``; None when not known)
    and why."""

    kind: str
    by: Optional[str] = None
    reason: Optional[str] = None

    @classmethod
    def from_error(cls, error: BaseException) -> "MillingStepEnd":
        """A stop's end, or a failure's, from what was raised."""
        said = ended_by(error)
        if said["status"] == CANCELLED:
            return cls(STOPPED, reason=str(error) or None)
        return cls(STEP_FAILED, reason=said["error"])

    @property
    def status(self) -> str:
        return _STATUS[self.kind]

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "by": self.by, "reason": self.reason}


class MillingStep:
    """One mill on the milling service. ``ending`` is set once something ends it;
    the step waits on it between looks at the instrument.

    It is all a monitor may do to the mill: read the milled time, ``report`` a
    reading onto ``readings``, ``end`` the step, or pause it (``paused``).
    """

    # Every reading reported in the step: {monitor, seq, milled_time, notable,
    # reading}. The strategy, other monitors and the viewer listen here.
    readings = Signal(dict)

    def __init__(
        self,
        index: int = 0,
        milling: Any = None,
        stage: Any = None,
    ):
        self.index = index
        # The milling service milling it, and the stage it is a step of (None
        # for a run outside a stage).
        self.milling = milling
        self.stage = stage
        self.started_at = now_iso()
        self.ended_at: Optional[str] = None
        self.ending = threading.Event()
        self.monitors: List[Dict[str, Any]] = []
        self.pauses: List[Dict[str, Any]] = []
        self.notable: List[Dict[str, Any]] = []
        self._milled = 0.0
        self._seq = 0
        self._end: Optional[MillingStepEnd] = None
        self._beam_stopped = False
        self._lock = threading.Lock()
        self._pause_lock = threading.Lock()
        # A monitor holds a pause: the state monitor does not look meanwhile.
        self.pausing = threading.Event()

    @property
    def microscope(self) -> Any:
        return getattr(self.milling, "parent", None)

    @property
    def output_dir(self) -> Optional[str]:
        return getattr(self.stage, "output_dir", None)

    @property
    def end(self) -> Optional[MillingStepEnd]:
        return self._end

    def milled_time(self) -> float:
        """Seconds milled so far in this step, paused time left out."""
        return self._milled

    def set_milled_time(self, seconds: float) -> None:
        self._milled = float(seconds)

    def report(
        self, reading: Dict[str, Any], notable: bool = False, monitor: str = ""
    ) -> None:
        """Put *reading* onto ``readings``. A notable one (a drop, a threshold
        crossed, an error) is also kept for the step's record."""
        with self._lock:
            self._seq += 1
            seq = self._seq
        entry = {
            "monitor": monitor,
            "seq": seq,
            "milled_time": round(self._milled, 3),
            "notable": notable,
            "reading": reading,
        }
        if notable:
            self.notable.append(entry)
        self.readings.emit(entry)

    def end_step(self, reason: str, by: Optional[str] = None) -> None:
        """End this step, completed; the strategy decides whether another
        follows."""
        self._end_and_stop(MillingStepEnd(ENDED, by=by, reason=reason))

    def fail(self, reason: str, by: Optional[str] = None) -> None:
        """End this step failed: the stage fails with it."""
        self._end_and_stop(MillingStepEnd(STEP_FAILED, by=by, reason=reason))

    def _end_and_stop(self, end: MillingStepEnd) -> None:
        self.end_as(end)
        if self.milling is not None:
            self.stop_beam(self.milling._stop)

    @contextmanager
    def paused(
        self,
        reason: str,
        by: Optional[str] = None,
        max_pause: Optional[float] = None,
    ) -> Iterator[None]:
        """Pause the pattern, act, resume. One pause at a time; milled time stops
        meanwhile. A pause longer than *max_pause* seconds fails the step (it
        never resumes on a timer); an end during the pause ends the step
        without resuming."""
        with self._pause_lock:
            if self.ending.is_set():
                yield
                return
            self.milling._pause()
            self.pausing.set()
            began = time.monotonic()
            timer = None
            if max_pause is not None:
                timer = threading.Timer(
                    max_pause,
                    self.fail,
                    args=(f"paused for longer than {max_pause:g} s", by),
                )
                timer.daemon = True
                timer.start()
            try:
                yield
            finally:
                if timer is not None:
                    timer.cancel()
                self.pauses.append(
                    {
                        "reason": reason,
                        "by": by,
                        "seconds": round(time.monotonic() - began, 3),
                    }
                )
                try:
                    if not self.ending.is_set():
                        self.milling._resume()
                finally:
                    self.pausing.clear()

    def end_as(self, end: MillingStepEnd) -> bool:
        """End the step, unless it has already ended; whether this end won."""
        with self._lock:
            if self._end is not None:
                return False
            self._end = end
        self.ending.set()
        return True

    def stop_beam(self, stop: Callable[[], None]) -> bool:
        """Stop the pattern with *stop*, once per step; whether it was called.
        A stop that raises is not counted, so the next caller tries again."""
        with self._lock:
            if self._beam_stopped:
                return False
            self._beam_stopped = True
        try:
            stop()
        except BaseException:
            with self._lock:
                self._beam_stopped = False
            raise
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "end": self._end.to_dict() if self._end is not None else None,
            "milled_time": round(self._milled, 3),
            "monitors": list(self.monitors),
            "pauses": list(self.pauses),
            "notable": list(self.notable),
        }


class MillingStageRun:
    """A strategy carrying out one stage, as one or more mill steps. Once
    ``ending`` is set ("end this stage"), the milling service starts no more
    steps in it."""

    def __init__(self, stage: Any = None):
        # The stage it carries out; a bare name where there is no stage object.
        self.stage = stage
        self.name = getattr(stage, "name", stage)
        self.ending = threading.Event()
        self.end: Optional[MillingStepEnd] = None
        self.steps: List[MillingStep] = []

    def end_as(self, end: MillingStepEnd) -> None:
        if self.end is None:
            self.end = end
        self.ending.set()

    def ended(self, error: Optional[BaseException] = None) -> Dict[str, Any]:
        """How the stage run ended, for its record, beside its status: its
        error's, else the stage's own end ("end this stage"), else its last
        step's."""
        if error is not None:
            return MillingStepEnd.from_error(error).to_dict()
        if self.end is not None:
            return self.end.to_dict()
        for step in reversed(self.steps):
            if step.end is not None:
                return step.end.to_dict()
        return MillingStepEnd(FINISHED).to_dict()
