"""RTM: read what the ion beam reads back from the patterns as they mill.

A live monitor on the FIB view. Its reader reads RTM at most once per
``interval`` while the pattern mills, and once more at the end; each read is
an `RtmFrame` on ``Milling.rtm_frame`` (for a canvas, a recorder or endpoint
detection) and a reading with each pattern's mean value. Offered only where
the driver has RTM.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from fibsem.milling.monitors.base import (
    MillingMonitor,
    MillingMonitorConfig,
    MillingMonitorResult,
)
from fibsem.milling.rtm import RTM_MODES, RtmFrame
from fibsem.structures import param


@dataclass
class RtmConfig(MillingMonitorConfig):
    mode: str = param(
        RTM_MODES[0],
        choices=RTM_MODES,
        doc="How densely to read: low or high.",
    )
    interval: float = param(1.0, unit="s", doc="The shortest time between reads.")


class RtmMonitor(MillingMonitor[RtmConfig]):
    name = "RTM"
    config_class = RtmConfig
    view = "fib"

    def __init__(self, config: Optional[RtmConfig] = None) -> None:
        super().__init__(config)
        self.reads = 0
        self.patterns = 0
        self._thread: Optional[threading.Thread] = None
        self._warned = False

    @classmethod
    def available(cls, milling: Any) -> bool:
        return bool(getattr(milling, "has_rtm", False))

    @property
    def milling(self) -> Any:
        return self.step.milling

    def on_start(self) -> None:
        if not self.available(self.milling):
            raise RuntimeError("This instrument has no RTM.")
        self.milling.rtm_start(self.config.mode)
        self.patterns = len(self.milling.rtm_patterns())
        self._thread = threading.Thread(
            target=self._run, name=f"monitor-{self.name}", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        step = self.step
        interval = max(float(self.config.interval), 0.01)
        while not step.ending.wait(interval) and self.error is None:
            if not step.pausing.is_set():
                self.guarded(self._read)

    def _read(self) -> None:
        interval = float(self.config.interval)
        started = time.monotonic()
        frame = self.milling.rtm_read(interval)
        took = time.monotonic() - started
        if took > interval and not self._warned:
            self._warned = True
            logging.warning(
                f"An RTM read and its subscribers took {took:.2f} s, longer than "
                f"the {interval:.2f} s interval; a subscriber should return quickly."
            )
        if frame is not None:
            self.on_frame(frame)

    def on_frame(self, frame: RtmFrame) -> None:
        self.reads += 1
        means = {str(i): round(v, 4) for i, v in frame.means().items()}
        self.report(
            {
                "sequence": frame.sequence,
                "milled_time": round(self.step.milled_time(), 1),
                "means": means,
            }
        )

    def on_stop(self) -> None:
        try:
            if self._thread is not None:
                self._thread.join(timeout=float(self.config.interval) + 5.0)
                # The last read, so a short mill has one.
                self.guarded(self._read)
        finally:
            self.milling.rtm_stop()

    def result(self) -> MillingMonitorResult:
        return MillingMonitorResult(
            facts={"Reads": self.reads, "Patterns read": self.patterns}
        )
