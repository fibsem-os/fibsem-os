"""The intensity drop rule: when the fluorescence under a mill has dropped.

One rule for the coincidence milling strategy and the FM intensity monitor:
after a warmup, track the peak of the rolling mean, and call it a drop once the
rolling mean has stayed below ``(1 - drop_fraction) * peak`` for
``consecutive_triggers`` frames in a row. It fires once per drop.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


class IntensityDropRule:
    def __init__(
        self,
        drop_fraction: float,
        rolling_window: int,
        warmup_duration: float,
        consecutive_triggers: int,
    ):
        self.drop_fraction = float(drop_fraction)
        self.rolling_window = int(rolling_window)
        self.warmup_duration = float(warmup_duration)
        self.consecutive_triggers = int(consecutive_triggers)
        self.values: List[float] = []
        self.peak = 0.0
        self.warmup_complete = False
        self.drop_seen = False
        self._below = 0

    def update(self, value: float, elapsed: float) -> Dict[str, Any]:
        """Take one frame's mean *value*, *elapsed* seconds in; the rule's
        numbers for it, with ``drop_detected`` True on the frame a drop fires."""
        self.values.append(value)
        rolling_mean = float(np.mean(self.values[-self.rolling_window :]))
        threshold_fraction = 1.0 - self.drop_fraction
        if not self.warmup_complete:
            if elapsed >= self.warmup_duration:
                self.warmup_complete = True
            return {
                "value": value,
                "rolling_mean": rolling_mean,
                "peak_rolling_mean": self.peak,
                "threshold_value": 0.0,
                "warmup_complete": False,
                "drop_detected": False,
                "drop_fraction": 1.0,
                "threshold_fraction": threshold_fraction,
                "consecutive_count": 0,
            }

        self.peak = max(self.peak, rolling_mean)
        threshold_value = threshold_fraction * self.peak
        self._below = self._below + 1 if rolling_mean < threshold_value else 0
        drop_detected = self._below >= self.consecutive_triggers
        if drop_detected:
            self._below = 0  # so it fires once per drop
            self.drop_seen = True
        return {
            "value": value,
            "rolling_mean": rolling_mean,
            "peak_rolling_mean": self.peak,
            "threshold_value": threshold_value,
            "warmup_complete": True,
            "drop_detected": drop_detected,
            "drop_fraction": rolling_mean / self.peak if self.peak > 0 else 1.0,
            "threshold_fraction": threshold_fraction,
            "consecutive_count": self.consecutive_triggers,
        }
