"""The Beam device, and the key router that keeps today's get/set working.

One ``Beam`` class serves both columns; a parameter one column lacks is simply not
bound on it. ``KeyRouter`` is what ``FibsemMicroscope.get``/``set`` would
become: a key that has moved to a device is routed to its parameter, and every other
key falls through to the backend's untouched if/elif chain.
"""

from __future__ import annotations

import logging
import threading
from enum import Enum
from math import pi
from typing import Any, Dict, List, Mapping, Optional

from psygnal import Signal

from fibsem.devices.core import BoundParameter, Device, Parameter, command
from fibsem.devices.stage import Stage
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemRectangle,
    ImageSettings,
    Point,
    RangeLimit,
    ScanMode,
)


class Beam(Device):
    voltage = Parameter(float, unit="V")
    current = Parameter(float, unit="A", depends_on=("plasma_gas",))
    plasma_gas = Parameter(str)
    working_distance = Parameter(float, unit="m")
    hfw = Parameter(float, unit="m")
    scan_rotation = Parameter(float, unit="rad", limits=RangeLimit(min=0.0, max=2 * pi))
    blanked = Parameter(bool)
    preset = Parameter(str)
    detector_type = Parameter(str)
    detector_mode = Parameter(str)
    detector_contrast = Parameter(float, limits=RangeLimit(min=0.0, max=1.0))
    detector_brightness = Parameter(float, limits=RangeLimit(min=0.0, max=1.0))
    resolution = Parameter(tuple, unit="px", doc="(width, height)")
    dwell_time = Parameter(float, unit="s")
    stigmation = Parameter(Point)
    shift = Parameter(Point, unit="m", doc="Beam shift.")
    on = Parameter(bool, doc="The beam is switched on.")
    scanning_mode = Parameter(
        ScanMode, doc="What the beam scans; the scan commands set it."
    )
    angular_correction = Parameter(
        float, unit="rad", doc="The tilt the angular correction corrects the image for."
    )
    tilt_correction = Parameter(
        bool, doc="The angular correction's tilt correction is on."
    )

    live_frame = Signal(object)
    """Each image live view acquires (a FibsemImage), from the live view's thread."""

    def __init__(self, beam_type: BeamType, parent: Any = None, **kwargs: Any):
        super().__init__(name=beam_type.name.lower(), parent=parent, **kwargs)
        self.beam_type = beam_type
        self._live_lock = threading.Lock()
        self._live_stop = threading.Event()
        self._live_thread: Optional[threading.Thread] = None

    @command(available=lambda beam: "blanked" in beam.parameters)
    def blank(self) -> None:
        """Blank the beam."""
        self.blanked.set_value(True)

    @command(available=lambda beam: "blanked" in beam.parameters)
    def unblank(self) -> None:
        """Unblank the beam."""
        self.blanked.set_value(False)

    # The scan area. Each command changes scanning_mode; a backend implements
    # _spot, _reduced_area and _full_frame, and a beam that can't set the scan area
    # has no scanning_mode, so the commands are unavailable. A driver that can set
    # the scan area but not read it back says so in _scans (Odemis); its commands
    # then read nothing back.

    @command(available=lambda beam: beam._scans())
    def spot(self, point: Point) -> None:
        """Park the beam on a point, in image coordinates (0 to 1)."""
        self._spot(point)
        self._read_scanning_mode()

    @command(available=lambda beam: beam._scans())
    def reduced_area(self, area: FibsemRectangle) -> None:
        """Scan only a rectangle of the frame, in image coordinates (0 to 1)."""
        self._reduced_area(area)
        self._read_scanning_mode()

    @command(available=lambda beam: beam._scans())
    def full_frame(self) -> None:
        """Scan the whole frame."""
        self._full_frame()
        self._read_scanning_mode()

    def _scans(self) -> bool:
        """Whether this beam has the scan commands."""
        return "scanning_mode" in self.parameters

    def _read_scanning_mode(self) -> None:
        if "scanning_mode" in self.parameters:
            self.scanning_mode.get_value()

    def _spot(self, point: Point) -> None:
        raise NotImplementedError

    def _reduced_area(self, area: FibsemRectangle) -> None:
        raise NotImplementedError

    def _full_frame(self) -> None:
        raise NotImplementedError

    # Imaging and the autofunctions. A driver implements _acquire, _last_image,
    # _autocontrast and _auto_focus, claiming the imaging channel for the vendor call
    # (claim_channel), and builds the FibsemImage as its backend does today. A beam
    # whose driver lacks a hook doesn't have that command. auto_focus is the
    # instrument's own routine only: the software sweep stays microscope.auto_focus's.

    @command
    def acquire(self, image_settings: Optional[ImageSettings] = None) -> FibsemImage:
        """Acquire an image with this beam: with the given settings, or with the beam's
        current ones. Imaging is a beam command, not a device."""
        if (
            image_settings is not None
            and image_settings.beam_type is not self.beam_type
        ):
            raise ValueError(
                f"{self.name} can't acquire an image for the "
                f"{image_settings.beam_type.name} beam"
            )
        return self._acquire(image_settings)

    @command(available=lambda beam: implements(beam, "_last_image"))
    def last_image(self) -> FibsemImage:
        """The last image this beam acquired, read back from the instrument."""
        return self._last_image()

    @command(available=lambda beam: implements(beam, "_autocontrast"))
    def autocontrast(self, reduced_area: Optional[FibsemRectangle] = None) -> None:
        """Run the instrument's brightness and contrast routine, optionally on a
        rectangle of the frame (0 to 1); the beam scans the full frame after."""
        self._autocontrast(reduced_area)

    @command(available=lambda beam: implements(beam, "_auto_focus"))
    def auto_focus(self, reduced_area: Optional[FibsemRectangle] = None) -> None:
        """Run the instrument's autofocus routine, optionally on a rectangle of the
        frame (0 to 1); the beam scans the full frame after."""
        self._auto_focus(reduced_area)

    def _acquire(self, image_settings: Optional[ImageSettings]) -> FibsemImage:
        # Until a driver implements it: the microscope's own acquire_image.
        if image_settings is None:
            return self.parent.acquire_image(beam_type=self.beam_type)
        return self.parent.acquire_image(image_settings)

    def _last_image(self) -> FibsemImage:
        raise NotImplementedError

    def _autocontrast(self, reduced_area: Optional[FibsemRectangle]) -> None:
        raise NotImplementedError

    def _auto_focus(self, reduced_area: Optional[FibsemRectangle]) -> None:
        raise NotImplementedError

    # Live view: the driver's _live runs on a thread of its own, acquiring with the
    # beam's current settings and emitting each image on live_frame, until stop_live
    # sets the event it is given. Frames are pushed, as the SEM and FIB viewers take
    # them today.

    @command(available=lambda beam: implements(beam, "_live"))
    def start_live(self) -> None:
        """Acquire continuously with the current settings, each image on
        ``live_frame``, until `stop_live`. Warns and does nothing when already live."""
        with self._live_lock:
            if self.is_live:
                logging.warning(f"{self.name} live view is already running.")
                return
            self._live_stop.clear()
            self._live_thread = threading.Thread(
                target=self._run_live, name=f"{self.name}-live", daemon=True
            )
            self._live_thread.start()

    @command(available=lambda beam: implements(beam, "_live"))
    def stop_live(self) -> None:
        """Stop live view, waiting briefly for its last frame. Safe when not live."""
        thread = self._live_thread
        if thread is None or self._live_stop.is_set():
            return
        self._live_stop.set()
        if thread is not threading.current_thread():
            thread.join(timeout=2)

    @property
    def is_live(self) -> bool:
        thread = self._live_thread
        return thread is not None and thread.is_alive()

    def _run_live(self) -> None:
        try:
            self._live(self._live_stop)
        except Exception as e:
            logging.error(f"{self.name} live view stopped: {e}")

    def _live(self, stop: threading.Event) -> None:
        raise NotImplementedError


def implements(beam: Beam, hook: str) -> bool:
    """Whether the beam's driver overrides *hook*."""
    return getattr(type(beam), hook) is not getattr(Beam, hook)


# Old key -> parameter name. Every beam key keeps its old name here, so the table is
# also the list of what has moved. A key missing from it has not moved yet.
BEAM_ROUTES: Dict[str, str] = {
    "voltage": "voltage",
    "current": "current",
    "plasma_gas": "plasma_gas",
    "working_distance": "working_distance",
    "hfw": "hfw",
    "scan_rotation": "scan_rotation",
    "blanked": "blanked",
    "preset": "preset",
    "detector_type": "detector_type",
    "detector_mode": "detector_mode",
    "detector_contrast": "detector_contrast",
    "detector_brightness": "detector_brightness",
    "resolution": "resolution",
    "dwell_time": "dwell_time",
    "stigmation": "stigmation",
    "shift": "shift",
    "on": "on",
    "scanning_mode": "scanning_mode",
    "angular_correction_angle": "angular_correction",
    "angular_correction_tilt_correction": "tilt_correction",
}


# Old set keys that are verbs, moved to the beam's scan commands: key -> command. The
# value is the command's argument (none for full_frame, which ignores it).
BEAM_COMMAND_ROUTES: Dict[str, str] = {
    "spot_mode": "spot",
    "reduced_area": "reduced_area",
    "full_frame": "full_frame",
}


# Stage keys take no beam type. A get key routes to a parameter; a set key that is a
# verb ("home", "link") routes to a command, which ignores the value as the old
# branches do.
STAGE_ROUTES: Dict[str, str] = {
    "stage_position": "position",
    "stage_homed": "homed",
    "stage_linked": "linked",
}
STAGE_COMMAND_ROUTES: Dict[str, str] = {
    "stage_home": "home",
    "stage_link": "link",
}


class KeyRouter:
    """Today's ``get``/``set``/``get_available_values``, routed where a key has moved.

    Routed calls make the same instrument call the old branch made and skip the new
    API's validation, so a half-migrated backend behaves exactly like an unmigrated one.
    Logging matches ``FibsemMicroscope.get`` and ``set``.
    """

    def __init__(
        self,
        microscope: Any,
        beams: Mapping[BeamType, Beam],
        routes: Optional[Mapping[str, str]] = None,
        stage: Optional[Stage] = None,
    ):
        self.microscope = microscope
        self.beams = dict(beams)
        self.routes = dict(BEAM_ROUTES if routes is None else routes)
        self.stage = stage

    def route(
        self, key: str, beam_type: Optional[BeamType]
    ) -> Optional[BoundParameter]:
        if self.stage is not None and key in STAGE_ROUTES:
            return self.stage.parameters.get(STAGE_ROUTES[key])
        name = self.routes.get(key)
        beam = self.beams.get(beam_type) if beam_type is not None else None
        if name is None or beam is None:
            return None
        return beam.parameters.get(name)

    def get(self, key: str, beam_type: Optional[BeamType] = None) -> Any:
        param = self.route(key, beam_type)
        if param is not None:
            value = param.get_value()
            if isinstance(value, Enum):
                value = value.value  # old keys return the plain value ("spot")
        else:
            value = self.microscope._get(key, beam_type)
        beam_name = "None" if beam_type is None else beam_type.name
        logging.debug(
            {"msg": "get", "key": key, "beam_type": beam_name, "value": value}
        )
        return value

    def route_command(self, key: str) -> Optional[Any]:
        """The device command an old set key has become, if it has moved and is available."""
        name = STAGE_COMMAND_ROUTES.get(key)
        if self.stage is None or name is None:
            return None
        info = self.stage.commands.get(name)
        if info is None or not info.available:
            return None
        return getattr(self.stage, name)

    def set(self, key: str, value: Any, beam_type: Optional[BeamType] = None) -> None:
        param = self.route(key, beam_type)
        run = self.route_command(key)
        if run is not None:
            run()
        elif param is not None and param.writable:
            param.write_through(value)
        else:
            # Unmoved keys, and read-only ones (set("stage_position") warns there).
            self.microscope._set(key, value, beam_type)
        beam_name = "None" if beam_type is None else beam_type.name
        logging.debug(
            {"msg": "set", "key": key, "beam_type": beam_name, "value": value}
        )

    def get_available_values(
        self, key: str, beam_type: Optional[BeamType] = None
    ) -> List[Any]:
        param = self.route(key, beam_type)
        if param is not None and param.choices is not None:
            return list(param.choices)
        return self.microscope.get_available_values(key, beam_type)
