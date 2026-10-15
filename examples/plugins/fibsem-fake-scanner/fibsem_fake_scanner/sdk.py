"""A stand-in for a scan generator vendor's SDK: a controller on its own cable.

Nothing here knows about fibsem. It is what a real unit's library looks like from the
driver's side: open a connection by address, set analogue gain and offset per input,
pick an input, scan a frame, abort it. A frame is a test pattern, so a test can tell
which input and settings made it.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Tuple

import numpy as np

INPUTS = ("adc0", "adc1", "adc2", "adc3")
PATTERNS = ("gradient", "checker", "noise")


class ControllerError(RuntimeError):
    pass


class FakeScanController:
    """One connection to the unit at *address*."""

    serial = "FS-0001"
    firmware = "1.4.2"
    max_resolution = (8192, 8192)
    min_dwell_time = 25e-9
    max_dwell_time = 100e-6

    def __init__(self, address: str, *, simulate_delay: bool = False):
        if not address:
            raise ControllerError("no address")
        self.address = address
        self.simulate_delay = simulate_delay
        self.open = True
        self.input = INPUTS[0]
        self.gain: Dict[str, float] = {name: 0.5 for name in INPUTS}
        self.offset: Dict[str, float] = {name: 0.0 for name in INPUTS}
        self.pattern = PATTERNS[0]
        self.frames = 0
        self._abort = threading.Event()

    def close(self) -> None:
        self.open = False

    def _check(self) -> None:
        if not self.open:
            raise ControllerError(f"controller at {self.address} is closed")

    def scan(self, width: int, height: int, dwell_time: float) -> np.ndarray:
        self._check()
        if width > self.max_resolution[0] or height > self.max_resolution[1]:
            raise ControllerError(f"{width}x{height} is over the unit's maximum")
        if dwell_time < self.min_dwell_time:
            raise ControllerError(f"dwell {dwell_time} is under the unit's minimum")
        self._abort.clear()
        if self.simulate_delay:
            self._abort.wait(width * height * dwell_time)
        frame = _pattern(self.pattern, (height, width), INPUTS.index(self.input))
        gain, offset = self.gain[self.input], self.offset[self.input]
        frame = np.clip(frame * (2 * gain) + offset * 255, 0, 255)
        self.frames += 1
        return frame.astype(np.uint8)

    def abort(self) -> None:
        self._abort.set()

    def self_test(self) -> Dict[str, object]:
        self._check()
        started = time.monotonic()
        return {
            "ok": True,
            "serial": self.serial,
            "inputs": list(INPUTS),
            "took": time.monotonic() - started,
        }


def _pattern(name: str, shape: Tuple[int, int], channel: int) -> np.ndarray:
    height, width = shape
    if name == "gradient":
        row = np.linspace(0, 255, width)
        return np.tile(row, (height, 1))
    if name == "checker":
        size = max(1, width // (8 * (channel + 1)))
        y, x = np.indices(shape)
        return (((x // size) + (y // size)) % 2) * 255.0
    rng = np.random.default_rng(channel)
    return rng.integers(0, 255, size=shape).astype(float)
