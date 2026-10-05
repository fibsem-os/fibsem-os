"""The fluorescence microscope's parts as devices, and the FM that groups them.

Each part is its own device with its own parameters: the camera, the light source,
the filter set and the objective. ``FM`` holds no parameters of its own; it is where
sequences that use several parts live, such as acquiring one channel (filter, light,
exposure, camera). A driver runs those sequences next to the hardware, which matters
when the FM is on another computer: one call over the network, not one per step, and
a dropped connection can't leave the light on.

Commands call a hook the driver implements (``_acquire``, ``_insert``, ...), the way
the stage's moves do, so the command's name, arguments and checks are the same on
every driver.

Live view is pulled rather than pushed: `FM.start_live` keeps the hardware acquiring,
each `FM.acquire_frame` returns the next frame, and live view stops by itself when
nobody asks for one (FIB-1096).
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from fibsem.devices.core import Device, Parameter, command
from fibsem.devices.wire import Frame, to_wire
from fibsem.fm.structures import EmissionFilter
from fibsem.structures import InsertableDeviceState


class Camera(Device):
    exposure_time = Parameter(float, unit="s")
    binning = Parameter(int)
    gain = Parameter(float)
    offset = Parameter(float)
    pixel_size = Parameter(tuple, unit="m", doc="(x, y), after binning.")
    resolution = Parameter(tuple, doc="(width, height) in pixels, after binning.")

    @command
    def acquire(self) -> np.ndarray:
        """One frame with the current settings, as the camera gives it."""
        return self._acquire()

    def _acquire(self) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} can't acquire")


class LightSource(Device):
    """The excitation light.

    ``power`` is a setting for the next exposure, not the light's live output: a
    driver applies it when it acquires (the light is off in between), so reading it
    says what the next frame will use.
    """

    power = Parameter(float, doc="A fraction of the source's maximum, 0 to 1.")


class FilterSet(Device):
    """Which light reaches the sample and the camera.

    Both parameters are settings for the next exposure, on every driver: what a
    channel asks for, applied when the driver acquires. They are not a reading of the
    hardware's position while idle: Odemis pushes them to the filter wheel and light
    only while its stream runs, and Thermo applies the excitation when it grabs a
    frame. Camera and objective parameters, by contrast, are live hardware state.
    """

    excitation_wavelength = Parameter(
        float,
        unit="nm",
        nearest=True,
        doc="The excitation band, by its centre in nm: a wavelength between bands "
        "selects the nearest band.",
    )
    emission_filter = Parameter(
        EmissionFilter,
        doc="The emission filter in the light path; the choices are this filter set's.",
    )


class Objective(Device):
    """Where the objective is and what it's doing. It moves only through commands."""

    position = Parameter(float, unit="m")
    state = Parameter(InsertableDeviceState)
    magnification = Parameter(float)
    numerical_aperture = Parameter(float)
    limit_position = Parameter(float, unit="m", doc="The furthest a move may go in.")

    @command
    def insert(self) -> None:
        """Move the objective in, to its imaging position."""
        self._insert()

    @command
    def retract(self) -> None:
        """Move the objective out, clear of the sample."""
        self._retract()

    @command
    def move_absolute(self, position: float) -> None:
        """Move the objective to a position, in metres."""
        self._move_absolute(position)

    @command
    def move_relative(self, delta: float) -> None:
        """Move the objective by a distance, in metres (positive is towards the sample)."""
        self._move_relative(delta)

    def _insert(self) -> None:
        raise NotImplementedError

    def _retract(self) -> None:
        raise NotImplementedError

    def _move_absolute(self, position: float) -> None:
        raise NotImplementedError

    def _move_relative(self, delta: float) -> None:
        raise NotImplementedError


# What `FM.acquire_frame` reports with a frame: (part, parameter) -> metadata key.
FRAME_METADATA: Dict[Tuple[str, str], str] = {
    ("camera", "exposure_time"): "exposure_time",
    ("camera", "gain"): "gain",
    ("camera", "offset"): "offset",
    ("camera", "binning"): "binning",
    ("camera", "pixel_size"): "pixel_size",
    ("camera", "resolution"): "resolution",
    ("light_source", "power"): "power",
    ("filter_set", "excitation_wavelength"): "excitation_wavelength",
    ("filter_set", "emission_filter"): "emission_filter",
    ("objective", "position"): "objective_position",
    ("objective", "magnification"): "objective_magnification",
    ("objective", "numerical_aperture"): "objective_numerical_aperture",
}


class FM(Device):
    """The group: sequences over several parts, run by the driver next to them."""

    parts: Mapping[str, Device] = MappingProxyType({})
    """The parts this group drives, by device name. A driver sets it."""

    live_timeout: Optional[float] = 5.0
    """Seconds live view may go without a frame being asked for before it stops by
    itself, switching the light off: the viewer has gone (a closed window, a crashed
    or disconnected client). None never stops it."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._live_lock = threading.Lock()
        self._live = False
        self._last_pull = 0.0
        self._pulling = 0  # frames being acquired right now, so a long exposure counts

    # -- live view --------------------------------------------------------------------
    #
    # Live view is pulled: `start_live` keeps the hardware acquiring continuously
    # (odemis: the stream active, light on), and each `acquire_frame` returns the next
    # frame from it. The viewer asks only when it is ready, so it never falls behind.

    @command
    def start_live(self, channel: Optional[Dict[str, Any]] = None) -> None:
        """Set up ``channel`` (or keep the current settings) and keep acquiring until
        `stop_live`, or until no frame has been asked for in ``live_timeout``."""
        with self._live_lock:
            already = self._live
            self._live = True
            self._last_pull = time.monotonic()
        if already:
            self._stop_live()
        self._start_live(channel)
        if not already and self.live_timeout is not None:
            threading.Thread(
                target=self._watch_live, name=f"{self.name}-live-watchdog", daemon=True
            ).start()

    @command
    def stop_live(self) -> None:
        """Stop live view: the light off and the hardware idle. Safe when not live."""
        with self._live_lock:
            was_live, self._live = self._live, False
        if was_live:
            self._stop_live()

    @property
    def is_live(self) -> bool:
        return self._live

    def _start_live(self, channel: Optional[Dict[str, Any]]) -> None:
        """Start acquiring continuously. Without a hook a driver takes each frame on
        demand, which is still live view, only slower."""

    def _stop_live(self) -> None:
        """Stop what `_start_live` started; called once per start."""

    def _watch_live(self) -> None:
        while True:
            time.sleep(min(0.5, self.live_timeout or 0.5))
            with self._live_lock:
                if not self._live:
                    return
                idle = time.monotonic() - self._last_pull
                if self._pulling or self.live_timeout is None:
                    continue
                if idle <= self.live_timeout:
                    continue
            logging.warning(
                f"{self.name}: no live frame asked for in {idle:.1f} s, stopping live "
                "view"
            )
            self.stop_live()
            return

    def _pulled(self, delta: int) -> None:
        with self._live_lock:
            self._pulling += delta
            self._last_pull = time.monotonic()

    @command
    def acquire_channel(self, channel: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Set up one channel (``ChannelSettings.to_dict()``), then acquire a frame.

        Without a channel, acquires with the current settings.
        """
        return self._acquire_channel(channel)

    @command
    def acquire_frame(self, channel: Optional[Dict[str, Any]] = None) -> Frame:
        """``acquire_channel``, with what the frame was taken with.

        The metadata is read here, next to the hardware, so building the image needs
        no reads after the frame. It holds ``acquisition_date`` (ISO, when the
        acquisition started) and the keys of `FRAME_METADATA` the parts have, as plain
        JSON-ready values: tuples are lists and the emission filter is its
        ``to_dict()``.
        """
        self._pulled(1)
        try:
            return self._acquire_frame(channel)
        finally:
            self._pulled(-1)

    def _acquire_channel(self, channel: Optional[Dict[str, Any]]) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} can't acquire")

    def _acquire_frame(self, channel: Optional[Dict[str, Any]]) -> Frame:
        acquisition_date = datetime.now().isoformat()
        data = self._acquire_channel(channel)
        metadata = {"acquisition_date": acquisition_date, **self._frame_metadata()}
        return Frame(data, metadata)

    def _frame_metadata(self) -> Dict[str, Any]:
        """What the parts say now, through their parameters. A part or parameter the
        driver doesn't have is left out."""
        found: Dict[str, Any] = {}
        for (part_name, parameter), key in FRAME_METADATA.items():
            part = self.parts.get(part_name)
            if part is not None and parameter in part.parameters:
                found[key] = to_wire(part.parameters[parameter].get_value())
        return found
