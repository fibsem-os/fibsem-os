"""FM intensity: watch the fluorescence under the mill, and end or flag the
step when it drops.

A live monitor on the FM view: the FM streams frames while the pattern mills,
and each frame's mean (inside ``bbox`` when set) goes through the same drop
rule as the coincidence milling strategy, on milled time. Its files: every
frame's numbers (``intensity.csv``), a frame every ``save_interval`` seconds,
and the trace (``intensity.png``).
"""

from __future__ import annotations

import csv
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional

import numpy as np

from fibsem.milling.monitors.base import (
    ON_TRIGGER_SUGGEST,
    MillingMonitorConfig,
    MillingMonitorResult,
    MillingStreamMonitor,
)
from fibsem.milling.monitors.intensity import IntensityDropRule
from fibsem.structures.params import Display, RangeLimit, param
from fibsem.structures.settings import FibsemRectangle

# The numbers each reading carries, from the drop rule.
_READING = (
    "value",
    "rolling_mean",
    "peak_rolling_mean",
    "threshold_value",
    "drop_fraction",
    "warmup_complete",
)


@dataclass
class FmIntensityConfig(MillingMonitorConfig):
    drop_fraction: float = param(
        0.4,
        limits=RangeLimit(0.05, 0.95),
        doc="A drop is the rolling mean this fraction below its peak.",
        display=Display("Drop Fraction"),
    )
    rolling_window: int = param(10, doc="Frames in the rolling mean.")
    warmup_duration: float = param(
        30.0, unit="s", doc="Milled time before the peak is tracked."
    )
    consecutive_triggers: int = param(
        10, doc="Frames in a row below the threshold that make a drop."
    )
    on_trigger: str = param(
        ON_TRIGGER_SUGGEST,
        doc="On a drop: end the step, or suggest it and keep milling.",
        display=Display("On Drop"),
    )
    save_interval: float = param(
        2.0, unit="s", doc="Save a frame this often; 0 saves none."
    )
    bbox: Optional[FibsemRectangle] = param(
        None, display=Display(hidden=True)
    )  # the area whose mean is watched, in the frame's fractions


class FmIntensityMonitor(MillingStreamMonitor[FmIntensityConfig]):
    name = "FM Intensity"
    config_class = FmIntensityConfig
    view = "fm"
    # Without the FM watching, the mill would run blind to the drop.
    required = True
    # Milled time with no frame, or a stream gone, fails the step.
    frame_timeout: ClassVar[float] = 10.0
    look_interval: ClassVar[float] = 0.2

    def __init__(self, config: Optional[FmIntensityConfig] = None) -> None:
        super().__init__(config)
        self.rows: List[Dict[str, Any]] = []
        self.rule: Optional[IntensityDropRule] = None
        self.drop_at: Optional[float] = None
        self._saved_at: Optional[float] = None
        self._figure: Optional[str] = None
        self._own_stream = False
        self._frame_at = 0.0
        self._done = threading.Event()
        self._watchdog: Optional[threading.Thread] = None

    @property
    def fm(self) -> Any:
        fm = getattr(self.step.microscope, "fm", None)
        if fm is None:
            raise RuntimeError("FM Intensity needs a fluorescence microscope.")
        return fm

    def source(self) -> Any:
        return self.fm.acquisition_signal

    def on_start(self) -> None:
        c = self.config
        self.rule = IntensityDropRule(
            c.drop_fraction, c.rolling_window, c.warmup_duration, c.consecutive_triggers
        )
        if self.folder is not None:
            os.makedirs(self.folder, exist_ok=True)
        super().on_start()
        fm = self.fm
        # The operator's live view, when on, is ridden and left on.
        self._own_stream = not fm.is_streaming
        if self._own_stream:
            fm.start_acquisition()
        self._watchdog = threading.Thread(
            target=self._watch_frames, name=f"monitor-{self.name}", daemon=True
        )
        self._watchdog.start()

    def _watch_frames(self) -> None:
        while not self._done.wait(self.look_interval):
            if self.step.ending.is_set() or self.error is not None:
                return
            self.guarded(self._check_frames)

    def _check_frames(self) -> None:
        if not self.fm.is_streaming:
            raise RuntimeError("the FM live view stopped")
        if self.step.milled_time() - self._frame_at >= self.frame_timeout:
            raise RuntimeError(f"no FM frame in {self.frame_timeout:g} s of milling")

    def restore_view(self) -> None:
        self.fm.set_active_channel()

    def on_frame(self, frame: Any) -> None:
        data = frame.data
        if self.config.bbox is not None:
            x, y, w, h = self.config.bbox.to_pixel_coordinates(
                (data.shape[-1], data.shape[-2])
            )
            data = data[..., y : y + h, x : x + w]
        milled = self.step.milled_time()
        self._frame_at = milled
        mean = float(np.mean(data))
        if np.isnan(mean):
            return
        stats = self.rule.update(mean, milled)
        reading = {k: _plain(stats[k]) for k in _READING}
        self.rows.append({"milled_time": round(milled, 3), **reading})
        self._save_frame(frame, milled)
        if stats["drop_detected"]:
            if self.drop_at is None:
                self.drop_at = milled
            self.trigger("intensity drop", reading)
        else:
            self.report(reading)

    def _save_frame(self, frame: Any, milled: float) -> None:
        interval = self.config.save_interval
        if self.folder is None or interval <= 0:
            return
        now = time.monotonic()
        if self._saved_at is not None and now - self._saved_at < interval:
            return
        self._saved_at = now
        frame.save(os.path.join(self.folder, f"fm-{milled:07.1f}s.ome.tiff"))

    def on_stop(self) -> None:
        self._done.set()
        if self._watchdog is not None:
            self._watchdog.join(timeout=self.look_interval + 5.0)
        try:
            fm = getattr(self.step.microscope, "fm", None)
            if fm is not None and self._own_stream:
                fm.stop_acquisition()
        finally:
            super().on_stop()
        if self.folder is not None and self.rows:
            self._write_csv()
            self._figure = self._plot()

    def _write_csv(self) -> None:
        with open(os.path.join(self.folder, "intensity.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def _plot(self) -> Optional[str]:
        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
        except ImportError:
            return None
        t = [r["milled_time"] for r in self.rows]
        fig = Figure(figsize=(8, 3.5))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot()
        ax.plot(t, [r["value"] for r in self.rows], lw=0.8, alpha=0.5, label="mean")
        ax.plot(t, [r["rolling_mean"] for r in self.rows], lw=1.5, label="rolling mean")
        warm = [
            (x, r["threshold_value"])
            for x, r in zip(t, self.rows)
            if r["warmup_complete"]
        ]
        if warm:
            ax.plot(*zip(*warm), "--", lw=1, label="threshold")
        if self.drop_at is not None:
            ax.axvline(self.drop_at, color="red", lw=1, label="drop")
        ax.set_xlabel("milled time (s)")
        ax.set_ylabel("FM intensity")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        path = os.path.join(self.folder, "intensity.png")
        try:
            fig.savefig(path, dpi=100)
        except Exception:  # noqa: BLE001 - the figure must not cost the record
            logging.debug("could not save the intensity figure", exc_info=True)
            path = None
        return path

    def result(self) -> MillingMonitorResult:
        rule = self.rule
        facts: Dict[str, Any] = {"Frames": len(self.rows)}
        if rule is not None and rule.peak > 0:
            facts["Peak rolling mean"] = round(rule.peak, 1)
            lowest = min(r["drop_fraction"] for r in self.rows if r["warmup_complete"])
            facts["Lowest (fraction of peak)"] = round(lowest, 2)
        facts["Drop seen"] = self.drop_at is not None
        if self.drop_at is not None:
            facts["Drop at (s)"] = round(self.drop_at, 1)
        return MillingMonitorResult(facts=facts, figure=self._figure, files=self.folder)


def _plain(value: Any) -> Any:
    return round(float(value), 3) if isinstance(value, (float, np.floating)) else value
