"""The fluorescence microscope's parts as devices, and the FM that groups them.

Each part is its own device with its own parameters: the camera, the light source,
the filter set and the objective. ``FM`` holds no hardware parameters of its own (only
``progress``, where a running z-stack is); it is where sequences that use several parts
live, such as acquiring one channel (filter, light, exposure, camera) or a z-stack. A driver runs those sequences next to the hardware, which matters
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
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from fibsem.devices.core import Device, Parameter, command
from fibsem.devices.wire import Frame, to_wire
from fibsem.fm.structures import EmissionFilter
from fibsem.structures import CameraImageTransform, InsertableDeviceState


def mount_transform_name(transform: CameraImageTransform) -> str:
    """How a mount transform is written in a configuration and on the wire."""
    return transform.value or "none"


def mount_transform_from_name(name: Optional[str]) -> CameraImageTransform:
    """The mount transform a configuration or a camera names; absent is none."""
    if name is None or name == "none":
        return CameraImageTransform.NONE
    try:
        return CameraImageTransform(name)
    except ValueError:
        names = ", ".join(mount_transform_name(t) for t in CameraImageTransform)
        raise ValueError(
            f"Unknown mount transform {name!r}; it is one of {names}."
        ) from None


class Camera(Device):
    exposure_time = Parameter(float, unit="s")
    binning = Parameter(int)
    gain = Parameter(float, doc="A fraction of the camera's gain range, 0 to 1.")
    offset = Parameter(float)
    pixel_size = Parameter(tuple, unit="m", doc="(x, y), after binning.")
    resolution = Parameter(tuple, doc="(width, height) in pixels, after binning.")
    mount_transform = Parameter(
        str,
        doc="The flip that puts the camera's frames into the stage's axes, from how "
        "it is mounted: none, flip-x, flip-y or flip-xy. Applied to every frame "
        "before the user's own image transform.",
    )

    _mount_transform: CameraImageTransform = CameraImageTransform.NONE
    """Set by the driver's binder from the site's configuration; a fact about the
    mount, not something the camera can measure."""

    @command
    def acquire(self) -> np.ndarray:
        """One frame with the current settings, as the camera gives it. ``acquire``
        returns the sensor's frame; ``mount_transform`` is applied by whoever builds
        the image."""
        return self._acquire()

    def read_mount_transform(self) -> str:
        return mount_transform_name(self._mount_transform)

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
    def insert(self) -> Optional[bool]:
        """Move the objective in, to its imaging position.

        Returns False when it was already in and nothing moved, where the driver
        knows; otherwise None."""
        return self._insert()

    @command
    def retract(self) -> Optional[bool]:
        """Move the objective out, clear of the sample.

        Returns False when it was already out and nothing moved, where the driver
        knows; otherwise None."""
        return self._retract()

    @command
    def move_absolute(self, position: float) -> None:
        """Move the objective to a position, in metres."""
        self._move_absolute(position)

    @command
    def move_relative(self, delta: float) -> None:
        """Move the objective by a distance, in metres (positive is towards the sample)."""
        self._move_relative(delta)

    def _insert(self) -> Optional[bool]:
        raise NotImplementedError

    def _retract(self) -> Optional[bool]:
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


Z_STACK_ORDERS = ("channel", "z")
"""A z-stack's order: each channel through every z position, or every channel at each
z position."""


class FM(Device):
    """The group: sequences over several parts, run by the driver next to them."""

    progress = Parameter(
        dict,
        doc="Where a running z-stack is: channel, channel_index, total_channels, "
        "zlevel and total_zlevels (1-based), reported before each frame; empty when "
        "none is running.",
    )

    parts: Mapping[str, Device] = MappingProxyType({})
    """The parts this group drives, by device name. A driver sets it."""

    runs_elsewhere: bool = False
    """Whether the group's commands run on another computer, where one call for a
    whole sequence saves a round trip per step."""

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
        self._cancel = threading.Event()
        self._progress: Dict[str, Any] = {}

    def read_progress(self) -> Dict[str, Any]:
        return dict(self._progress)

    def _report_progress(self, progress: Dict[str, Any]) -> None:
        param = self.parameters.get("progress")
        if param is not None:
            _ = param.cached  # a first report signals only once something was read
        self._progress = progress
        if param is not None:
            param.report(dict(progress))

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

    # -- z-stack ------------------------------------------------------------------------

    @command
    def acquire_z_stack(
        self,
        channels: List[Dict[str, Any]],
        positions: List[float],
        order: str = "channel",
        restore_position: Optional[float] = None,
    ) -> List[Frame]:
        """A z-stack, next to the hardware: one frame per channel (each a
        ``ChannelSettings.to_dict()``) at each objective position, in metres.

        ``order`` is "channel" (each channel through every position) or "z" (every
        channel at each position). The frames come back channel by channel, each
        channel's in position order, whichever the order. The objective goes back to
        ``restore_position`` afterwards, when given. `cancel` stops it between frames:
        the objective goes back, and the result is empty.
        """
        if order not in Z_STACK_ORDERS:
            raise ValueError(f"order must be one of {Z_STACK_ORDERS}, not {order!r}")
        self._cancel.clear()
        try:
            return self._acquire_z_stack(channels, positions, order, restore_position)
        finally:
            self._report_progress({})

    @command
    def cancel(self) -> None:
        """Stop a running z-stack before its next frame. Safe when none is running."""
        self._cancel.set()
        self._cancelled()

    def _cancelled(self) -> None:
        """Pass the cancel on, for a driver whose z-stack runs elsewhere."""

    def _acquire_z_stack(
        self,
        channels: List[Dict[str, Any]],
        positions: List[float],
        order: str,
        restore_position: Optional[float],
    ) -> List[Frame]:
        """Moves the objective and takes each frame with `acquire_frame`, so it runs
        on any driver; the same steps, in the same order, as the FM API's z-stack."""
        objective = self.parts["objective"]
        n_channels, n_positions = len(channels), len(positions)
        frames: List[List[Optional[Frame]]] = [[None] * n_positions for _ in channels]

        def frame(i: int, j: int) -> bool:
            channel = channels[i]
            self._report_progress(
                {
                    "channel": channel.get("name"),
                    "channel_index": i + 1,
                    "total_channels": n_channels,
                    "zlevel": j + 1,
                    "total_zlevels": n_positions,
                }
            )
            if self._cancel.is_set():
                return False
            if order == "channel":
                objective.move_absolute(positions[j])
            frames[i][j] = self.acquire_frame(channel)
            return True

        def cancelled() -> List[Frame]:
            logging.info(f"{self.name}: z-stack cancelled")
            if restore_position is not None:
                objective.move_absolute(restore_position)
            return []

        if order == "z":
            for j, z in enumerate(positions):
                if self._cancel.is_set():
                    return cancelled()
                objective.move_absolute(z)
                for i in range(n_channels):
                    if not frame(i, j):
                        return cancelled()
        else:
            for i in range(n_channels):
                if self._cancel.is_set():
                    return cancelled()
                for j in range(n_positions):
                    if not frame(i, j):
                        return cancelled()

        if restore_position is not None:
            objective.move_absolute(restore_position)
        return [f for channel_frames in frames for f in channel_frames]
