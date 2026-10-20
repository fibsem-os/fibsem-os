"""Milling monitors: code that runs during a mill step (FIB-1481).

A strategy is how a stage is milled, between steps; a monitor is what happens
during one step. A stage lists its monitors (``stage.monitors``, each a
``MillingMonitorConfig``); the milling service makes them afresh for every
step, starts them before the pattern and stops them however the step ends.

A monitor reports readings (``report``), ends the step (``end``), or pauses the
pattern to act (``paused``). A live monitor reads or images while the pattern
mills and names the ``view`` it needs; the live monitors of a step share one. It fails quietly: an error is reported and the
monitor stops, and the step carries on, unless the monitor is ``required``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    ContextManager,
    Dict,
    Generic,
    Optional,
    Type,
    TypeVar,
)

from fibsem.milling.base import MillingStrategyResult
from fibsem.milling.monitor_config import MillingMonitorConfig
from fibsem.structures import param

if TYPE_CHECKING:
    from fibsem.milling.steps import MillingStep

TMillingMonitorConfig = TypeVar("TMillingMonitorConfig", bound=MillingMonitorConfig)

# What a deciding monitor does when its rule triggers: end the step itself, or
# report it as notable and keep milling, for the operator or agent to decide.
ON_TRIGGER_END, ON_TRIGGER_SUGGEST = "end", "suggest"


@dataclass
class MillingMonitorResult(MillingStrategyResult):
    """What a monitor says about its step, recorded on the step: facts, a
    figure, and the folder of its files."""

    files: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        record = super().to_dict()
        if self.files:
            record["files"] = str(self.files)
        return record


class MillingMonitor(Generic[TMillingMonitorConfig]):
    """Runs during one mill step. ``on_start`` runs before the pattern starts and
    returns at once; ``on_stop`` runs however the step ended; ``result`` says
    what it saw."""

    name: ClassVar[str] = "Milling Monitor"
    config_class: ClassVar[Type[Any]] = MillingMonitorConfig
    # If it fails, the step fails.
    required: ClassVar[bool] = False
    # The view a live monitor needs while the pattern mills: "fib", "sem" or
    # "fm". None for one that only acts while paused, or reads no image.
    view: ClassVar[Optional[str]] = None
    # The shortest time between two readings that are not notable, in seconds:
    # one coming sooner waits, and only the latest is reported.
    report_interval: ClassVar[float] = 0.5

    def __init__(self, config: Optional[TMillingMonitorConfig] = None) -> None:
        self.config: TMillingMonitorConfig = (config or self.config_class()).made_for(
            type(self)
        )
        self.step: Optional["MillingStep"] = None
        self.error: Optional[BaseException] = None
        self._reported_at: Optional[float] = None
        self._held_reading: Optional[Dict[str, Any]] = None
        self._report_lock = threading.Lock()

    @classmethod
    def available(cls, milling: Any) -> bool:
        """Whether this monitor can run on *milling*'s instrument; one that
        can't is not offered."""
        return True

    @property
    def by(self) -> str:
        """Who this monitor is, in a step's end or pause."""
        return f"monitor:{self.name}"

    @property
    def folder(self) -> Optional[str]:
        """Where this monitor saves: ``<stage output>/monitors/<name>``."""
        output_dir = self.step.output_dir if self.step is not None else None
        if output_dir is None:
            return None
        return os.path.join(output_dir, "monitors", self.name.replace(" ", "-"))

    # ── what the service calls ────────────────────────────────────────────

    def start(self, step: "MillingStep") -> None:
        self.step = step
        self.guarded(self.on_start)

    def stop(self) -> Dict[str, Any]:
        """Stop, and say what it saw, as the step's record holds it. Its
        ``on_stop`` runs even after it failed, to let go of what it holds."""
        try:
            self.on_stop()
        except Exception as e:  # noqa: BLE001 - a monitor's own failure
            logging.warning(f"Milling monitor {self.name} failed to stop: {e}")
            self.error = self.error or e
        self._report_held()
        record: Dict[str, Any] = {"name": self.name}
        try:
            record.update(self.result().to_dict())
        except Exception as e:  # noqa: BLE001 - recording must not matter
            logging.debug(f"could not record {self.name}'s result", exc_info=True)
            self.error = self.error or e
        if self.error is not None:
            record["error"] = f"{type(self.error).__name__}: {self.error}"
        return record

    def guarded(self, call: Callable[[], Any]) -> None:
        """Run *call*, one of this monitor's own; an error stops the monitor,
        and the step too when it is ``required``."""
        if self.error is not None:
            return
        try:
            call()
        except Exception as e:  # noqa: BLE001 - a monitor's own failure
            self.error = e
            logging.warning(f"Milling monitor {self.name} failed: {e}", exc_info=True)
            if self.step is not None:
                self.step.report(
                    {"error": f"{type(e).__name__}: {e}"},
                    notable=True,
                    monitor=self.name,
                )
                if self.required:
                    self.step.fail(f"{self.name} failed: {e}", by=self.by)

    # ── what a monitor does to the step ───────────────────────────────────

    def report(self, reading: Dict[str, Any], notable: bool = False) -> None:
        """Report *reading*. A notable one goes at once; others at most every
        ``report_interval``, the latest kept meanwhile."""
        with self._report_lock:
            now = time.monotonic()
            due = (
                self._reported_at is None
                or now - self._reported_at >= self.report_interval
            )
            if not notable and not due:
                self._held_reading = reading
                return
            if not notable:
                self._reported_at = now
                self._held_reading = None
        self.step.report(reading, notable=notable, monitor=self.name)

    def _report_held(self) -> None:
        """Report the reading the interval held back, at the end of the step."""
        with self._report_lock:
            reading, self._held_reading = self._held_reading, None
        if reading is not None and self.step is not None:
            self.step.report(reading, monitor=self.name)

    def end(self, reason: str) -> None:
        """End this step, completed."""
        self.step.end_step(reason, by=self.by)

    def paused(self, reason: str) -> ContextManager[None]:
        """Pause the pattern, act, resume, within ``config.max_pause``."""
        return self.step.paused(reason, by=self.by, max_pause=self.config.max_pause)

    def trigger(self, reason: str, reading: Dict[str, Any]) -> None:
        """The monitor's rule triggered: a notable reading, and the step ends
        when ``config.on_trigger`` is ``end`` (the default)."""
        on_trigger = getattr(self.config, "on_trigger", ON_TRIGGER_END)
        self.report({**reading, "trigger": reason, "on_trigger": on_trigger}, True)
        if on_trigger == ON_TRIGGER_END:
            self.end(reason)

    # ── what a monitor implements ─────────────────────────────────────────

    def on_start(self) -> None:
        """Before the pattern starts; returns at once."""

    def on_stop(self) -> None:
        """However the step ended."""

    def restore_view(self) -> None:
        """Take this live monitor's view back after the pattern was paused or
        resumed, which selects the milling view on some instruments."""

    def result(self) -> MillingMonitorResult:
        return MillingMonitorResult()


@dataclass
class MillingPeriodicMonitorConfig(MillingMonitorConfig):
    interval: float = param(30.0, unit="s", doc="Milled time between ticks.")


TPeriodic = TypeVar("TPeriodic", bound=MillingPeriodicMonitorConfig)


class MillingPeriodicMonitor(MillingMonitor[TPeriodic]):
    """Calls ``on_tick`` on its own thread every ``config.interval`` seconds of
    milled time; the clock starts again each step."""

    config_class: ClassVar[Type[Any]] = MillingPeriodicMonitorConfig
    # How often the thread looks at the milled time, in seconds.
    look_interval: ClassVar[float] = 0.1

    def __init__(self, config: Optional[TPeriodic] = None) -> None:
        super().__init__(config)
        self.ticks = 0
        self._thread: Optional[threading.Thread] = None

    def on_start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name=f"monitor-{self.name}", daemon=True
        )
        self._thread.start()

    def on_stop(self) -> None:
        if self._thread is not None:
            self._thread.join(timeout=self.config.max_pause + 5.0)

    def _run(self) -> None:
        step = self.step
        due = float(self.config.interval)
        while not step.ending.is_set() and self.error is None:
            if step.milled_time() >= due:
                due += float(self.config.interval)
                self.ticks += 1
                self.guarded(self.on_tick)
                continue
            step.ending.wait(self.look_interval)

    def on_tick(self) -> None:
        """Every ``config.interval`` seconds of milled time."""


class MillingStreamMonitor(MillingMonitor[TMillingMonitorConfig]):
    """Reads frames from a source as they come (FM live imaging, an RTM
    reader): ``on_frame`` on the source's thread, which must return quickly.
    ``source`` names the signal to listen to; frames stop with the step and
    are skipped while the pattern is paused."""

    def __init__(self, config: Optional[TMillingMonitorConfig] = None) -> None:
        super().__init__(config)
        self.frames = 0
        self._source: Any = None

    def source(self) -> Any:
        """The signal the frames come on (a psygnal signal instance)."""
        raise NotImplementedError

    def on_start(self) -> None:
        self._source = self.source()
        self._source.connect(self._frame)

    def on_stop(self) -> None:
        if self._source is not None:
            self._source.disconnect(self._frame)
            self._source = None

    def _frame(self, frame: Any) -> None:
        step = self.step
        if step is None or step.ending.is_set() or step.pausing.is_set():
            return
        self.frames += 1
        self.guarded(lambda: self.on_frame(frame))

    def on_frame(self, frame: Any) -> None:
        """One frame from the source."""
