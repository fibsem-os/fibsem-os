"""The Beam device.

One ``Beam`` class serves both columns; a parameter one column lacks is simply not
bound on it.
"""

from __future__ import annotations

import copy
import logging
import threading
from dataclasses import replace
from math import pi
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from psygnal import Signal

from fibsem.cancellation import COMPLETED, ended_by
from fibsem.constants import DEGREE_SYMBOL
from fibsem.devices.core import (
    UNLIMITED,
    BoundParameter,
    Device,
    DeviceParameter,
    MetadataSource,
    ParameterMetadata,
    ResourceBusy,
    Role,
    command,
)
from fibsem.devices.scanner import SCANNER_RESOURCE, Scanner, resolution_within
from fibsem.devices.wire import to_wire
from fibsem.structures.image import FibsemImage, FibsemImageMetadata
from fibsem.structures.params import Display, RangeLimit
from fibsem.structures.settings import (
    BeamType,
    FibsemRectangle,
    ImageSettings,
    Point,
    ScanMode,
)
from fibsem.util.timestamps import now

# The resolutions a scan is set to on a driver whose instrument can't list its own:
# the standard 3:2 frames. A beam's resolution choices are what its scan can be set
# to; square frames are left out until acquisition can ask for them on its own.
STANDARD_RESOLUTIONS: Tuple[Tuple[int, int], ...] = (
    (384, 256),
    (768, 512),
    (1536, 1024),
    (3072, 2048),
    (6144, 4096),
)


def nearest_resolution(
    resolution: Tuple[int, int], choices: Optional[Sequence[Tuple[int, int]]]
) -> Tuple[int, int]:
    """The listed resolution nearest ``resolution`` (least width plus height
    difference), for a driver that can only scan the ones it lists."""
    if not choices:
        return tuple(resolution)
    return min(
        (tuple(c) for c in choices),
        key=lambda c: abs(c[0] - resolution[0]) + abs(c[1] - resolution[1]),
    )


def scanned_resolution(
    microscope: Any, beam_type: BeamType, resolution: Tuple[int, int]
) -> Tuple[int, int]:
    """The resolution *microscope*'s *beam_type* beam scans an image asked for at
    *resolution* (`Beam.scan_resolution`); as asked when it has no such beam."""
    beam = getattr(microscope, "beams", {}).get(beam_type)
    return (
        tuple(resolution) if beam is None else tuple(beam.scan_resolution(resolution))
    )


# Field of view and dwell time limits for a driver that can't report its instrument's:
# wide enough not to refuse a real setting. A driver reports its own in metadata.
DEFAULT_HFW_LIMITS = RangeLimit(min=1e-9, max=1e-2)
DEFAULT_DWELL_TIME_LIMITS = RangeLimit(min=1e-9, max=1e-3)

# What the beam panel has always allowed for the parameters no driver reports a range
# for yet. A relative shift is clamped to the shift's; stigmation is per field, so a
# write isn't clipped by it.
DEFAULT_WORKING_DISTANCE_LIMITS = RangeLimit(min=1e-3, max=30e-3)
DEFAULT_SHIFT_LIMITS = {
    "x": RangeLimit(min=-50e-6, max=50e-6),
    "y": RangeLimit(min=-50e-6, max=50e-6),
}
# For a driver that doesn't know its instrument's beam-shift range: a relative shift is
# clamped to the shift's limits, and these clamp nothing.
UNBOUNDED_SHIFT_LIMITS = {"x": UNLIMITED, "y": UNLIMITED}
DEFAULT_STIGMATION_LIMITS = {
    "x": RangeLimit(min=-1.0, max=1.0),
    "y": RangeLimit(min=-1.0, max=1.0),
}


class Beam(Device):
    voltage = DeviceParameter(
        float, unit="V", display=Display("Beam Voltage", advanced=True)
    )
    # the currents an instrument offers change with the gas and the voltage (AutoScript)
    current = DeviceParameter(
        float,
        unit="A",
        depends_on=("plasma_gas", "voltage"),
        display=Display("Beam Current"),
    )
    plasma_gas = DeviceParameter(str, display=Display("Plasma Gas"))
    working_distance = DeviceParameter(
        float,
        unit="m",
        limits=DEFAULT_WORKING_DISTANCE_LIMITS,
        display=Display("Working Distance", scale=1e3, step=0.01, decimals=3),
    )
    hfw = DeviceParameter(
        float,
        unit="m",
        limits=DEFAULT_HFW_LIMITS,
        display=Display("Field of View", scale=1e6, step=50.0, decimals=1),
    )
    scan_rotation = DeviceParameter(
        float,
        unit="rad",
        limits=RangeLimit(min=0.0, max=2 * pi),
        display=Display(
            "Scan Rotation",
            scale=180 / pi,
            unit=DEGREE_SYMBOL,
            step=180.0,
            decimals=0,
            advanced=True,
        ),
    )
    blanked = DeviceParameter(bool, display=Display("Blanked"))
    live = DeviceParameter(bool, doc="Whether live view is running. Read only.")
    preset = DeviceParameter(str, display=Display("Preset"))
    detector_type = DeviceParameter(str, display=Display("Detector Type"))
    detector_mode = DeviceParameter(
        str, depends_on=("detector_type",), display=Display("Detector Mode")
    )
    detector_contrast = DeviceParameter(
        float,
        limits=RangeLimit(min=0.0, max=1.0),
        display=Display("Contrast", step=0.01, decimals=3),
    )
    detector_brightness = DeviceParameter(
        float,
        limits=RangeLimit(min=0.0, max=1.0),
        display=Display("Brightness", step=0.01, decimals=3),
    )
    resolution = DeviceParameter(
        tuple,
        unit="px",
        choices=STANDARD_RESOLUTIONS,
        doc="(width, height). The choices are what the scan can be set to.",
        display=Display("Resolution"),
    )
    dwell_time = DeviceParameter(
        float,
        unit="s",
        limits=DEFAULT_DWELL_TIME_LIMITS,
        display=Display("Dwell Time", scale=1e6, step=0.01, decimals=3),
    )
    stigmation = DeviceParameter(
        Point,
        limits=DEFAULT_STIGMATION_LIMITS,
        display=Display("Stigmation", step=0.001, decimals=4, advanced=True),
    )
    shift = DeviceParameter(
        Point,
        unit="m",
        limits=DEFAULT_SHIFT_LIMITS,
        doc="Beam shift.",
        display=Display("Shift", scale=1e6, step=0.01, decimals=3, advanced=True),
    )
    source_tilt = DeviceParameter(
        Point,
        unit="rad",
        doc="Electron source tilt: the angle the beam enters the column from the gun "
        "at. Corrects an illumination drop.",
        display=Display("Source Tilt", scale=1e3, step=0.01, decimals=3, advanced=True),
    )
    lens_alignment = DeviceParameter(
        Point,
        unit="m",
        doc="Objective lens alignment: the beam's position relative to the final lens "
        "aperture. Minimises the image shift while focusing.",
        display=Display(
            "Lens Alignment", scale=1e6, step=0.01, decimals=3, advanced=True
        ),
    )
    on = DeviceParameter(bool, doc="The beam is switched on.")
    scanning_mode = DeviceParameter(
        ScanMode, doc="What the beam scans; the scan commands set it."
    )
    angular_correction = DeviceParameter(
        float, unit="rad", doc="The tilt the angular correction corrects the image for."
    )
    tilt_correction = DeviceParameter(
        bool, doc="The angular correction's tilt correction is on."
    )

    scanner = Role(
        Scanner,
        required=False,
        doc="An external scan generator that images in place of the vendor's scan, "
        "when the configuration binds one (``roles: {scanner: <entry>}``).",
    )

    live_frame = Signal(object)
    """Each image live view acquires (a FibsemImage), from the live view's thread."""

    image_acquired = Signal(object)
    """Every image this beam produces (a FibsemImage): each `acquire` and each live
    frame, once, from the thread that acquired it. Nothing in the app listens yet;
    the viewers still show what is sent to them."""

    def __init__(self, beam_type: BeamType, parent: Any = None, **kwargs: Any):
        super().__init__(name=beam_type.name.lower(), parent=parent, **kwargs)
        self.beam_type = beam_type
        self._live_lock = threading.Lock()
        # Live frames are images too. A driver's live view may acquire each frame
        # through `acquire`, which then leaves the frame to this.
        self.live_frame.connect(self._live_frame_acquired)
        self._live_stop = threading.Event()
        self._live_thread: Optional[threading.Thread] = None
        # The settings this beam last acquired with, when it was given some: where
        # that image was saved, which `image_settings` carries on.
        self._last_image_settings: Optional[ImageSettings] = None

    # -- a bound scanner's limits ------------------------------------------------
    #
    # While a scanner is bound, the beam's dwell time and resolution offer only what
    # both the column and the scanner can do: the limits are intersected, and the
    # resolution choices filtered. The values stay the beam's.

    _SCANNER_LIMITED = ("dwell_time", "resolution")

    def bind(
        self,
        name: str,
        read: Callable[[], Any],
        write: Optional[Callable[[Any], None]] = None,
        metadata: MetadataSource = None,
        needs_channel: bool = False,
    ) -> BoundParameter:
        if name in self._SCANNER_LIMITED:
            metadata = self._within_scanner(name, metadata)
        return super().bind(name, read, write, metadata, needs_channel)

    def fill_roles(self, **devices: Device) -> "Beam":
        super().fill_roles(**devices)
        if "scanner" in devices:
            for name in self._SCANNER_LIMITED:
                param = self.parameters.get(name)
                if param is not None:
                    param.refresh_metadata()
        return self

    def _within_scanner(
        self, name: str, source: MetadataSource
    ) -> Callable[[], ParameterMetadata]:
        spec = self.declared_parameters()[name]

        def metadata() -> ParameterMetadata:
            own = source() if callable(source) else (source or ParameterMetadata())
            scanner = self._roles.get("scanner")
            if scanner is None:
                return own
            if name == "dwell_time":
                return self._dwell_within(own, spec.limits, scanner.dwell_time_limits)
            return self._resolutions_within(
                own, spec.choices, scanner.resolution_limits
            )

        return metadata

    def _dwell_within(
        self,
        own: ParameterMetadata,
        default: Optional[RangeLimit],
        scanner: Optional[RangeLimit],
    ) -> ParameterMetadata:
        limits = own.limits if own.limits is not None else default
        if scanner is None:
            return own
        if limits is None:
            return replace(own, limits=scanner)
        both = RangeLimit(
            min=max(limits.min, scanner.min), max=min(limits.max, scanner.max)
        )
        if both.min > both.max:
            logging.warning(
                f"{self.name}: its dwell times ({limits.min}-{limits.max} s) and its "
                f"scanner's ({scanner.min}-{scanner.max} s) don't overlap; offering "
                "the scanner's"
            )
            both = scanner
        return replace(own, limits=both)

    def _resolutions_within(
        self,
        own: ParameterMetadata,
        default: Optional[Sequence[Any]],
        scanner: Optional[Tuple[Tuple[int, int], Tuple[int, int]]],
    ) -> ParameterMetadata:
        choices = own.choices if own.choices is not None else default
        if scanner is None or choices is None:
            return own
        within = [c for c in choices if resolution_within(tuple(c), scanner)]
        if not within:
            logging.warning(
                f"{self.name}: none of its resolutions is within its scanner's "
                f"{scanner}; offering them all, and the scanner refuses what it "
                "can't scan"
            )
            return own
        return replace(own, choices=within)

    def facts(self) -> Dict[str, Any]:
        """The gas the column's source reports running, if any."""
        return {"installed_plasma_gas": self._installed_gas_or_none()}

    def _installed_gas_or_none(self) -> Optional[str]:
        # A Ga column refuses the question the same way a sick connection does.
        try:
            return self.installed_plasma_gas() or None
        except Exception as error:
            logging.debug(f"{self.name}: installed_plasma_gas: {error}")
            return None

    def installed_plasma_gas(self) -> Optional[str]:
        """The gas the column's source reports running, read from the instrument
        whatever the configuration says, or None if it cannot say (a Ga column, an
        electron beam, a backend that cannot ask).

        It is how connect finds the gas when the configuration names none
        (`FibsemMicroscope._read_plasma_source`), which then binds the ``plasma_gas``
        parameter the column did not have.
        """
        return None

    @command(available=lambda beam: "on" in beam.parameters)
    def turn_on(self) -> bool:
        """Switch the beam on. Answers whether it is on."""
        self.on.write_through(True)
        return bool(self.on.get_value())

    @command(available=lambda beam: "on" in beam.parameters)
    def turn_off(self) -> bool:
        """Switch the beam off. Answers whether it is on."""
        self.on.write_through(False)
        return bool(self.on.get_value())

    @command(available=lambda beam: "blanked" in beam.parameters)
    def blank(self) -> None:
        """Blank the beam."""
        self.blanked.set_value(True)

    @command(available=lambda beam: "blanked" in beam.parameters)
    def unblank(self) -> None:
        """Unblank the beam."""
        self.blanked.set_value(False)

    @command(available=lambda beam: "shift" in beam.parameters)
    def shift_by(self, dx: float, dy: float) -> Point:
        """Shift the beam by (dx, dy) metres from where it is, as a beam image
        measured it. Returns the shift afterwards, which a driver may have clipped to
        its limits. Recorded as one ``beam_shifted`` event, with dx and dy as asked."""
        with self.session.recording(
            "beam_shifted", dx=dx, dy=dy, beam_type=self.beam_type.name, shift=None
        ) as entry:
            self._shift_by(float(dx), float(dy))
            shift = self.shift.get_value()
            entry["shift"] = shift.to_dict() if isinstance(shift, Point) else None
            return shift

    def _shift_by(self, dx: float, dy: float) -> None:
        """Add (dx, dy) to the shift, unvalidated, clamped to the shift's limits: the
        range the driver reports, or the declared default where it reports none. A
        driver whose instrument needs more (a flipped scan) overrides this."""
        shift = self.shift
        shift.refresh_metadata(emit=False)  # the limits as the instrument has them now
        target = shift.get_value() + Point(dx, dy)
        limits = shift.limits or {}
        x, y = limits.get("x"), limits.get("y")
        target = Point(
            x=x.clamp(target.x) if x is not None else target.x,
            y=y.clamp(target.y) if y is not None else target.y,
        )
        shift.write_through(target)

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

    # Imaging and the autofunctions. A driver implements _acquire,
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
        # When the image was acquired, taken just before the driver runs, so every
        # backend's image carries it. A driver that reads a time off the
        # instrument has already set its own, and keeps it.
        started = now()
        if "scanner" in self.roles:
            image = self._acquire_with_scanner(image_settings)
        else:
            image = self._acquire(image_settings)
        metadata = getattr(image, "metadata", None)
        if (
            hasattr(metadata, "acquisition_datetime")
            and not metadata.acquisition_datetime
        ):
            metadata.acquisition_datetime = started
        self._stamp_roles(image)
        if image_settings is not None:
            self._last_image_settings = image_settings
        self.session.stamp(image)
        if threading.current_thread() is not getattr(self, "_live_thread", None):
            self.image_acquired.emit(image)
        return image

    def scan_resolution(self, resolution: Tuple[int, int]) -> Tuple[int, int]:
        """The resolution an image asked for at *resolution* comes back at.

        As asked, unless the driver can only scan the sizes it lists and snaps to the
        nearest (AutoScript). Asked before acquiring by what lays images out from the
        size, so an overview's grid and stitch use the tiles' real size."""
        return tuple(resolution)

    def _stamp_roles(self, image: Any) -> None:
        """Record on *image* the device in each role this beam has filled, such as
        the external scanner that made the frame (``FibsemImageMetadata.roles``)."""
        metadata = getattr(image, "metadata", None)
        if self._roles and hasattr(metadata, "roles"):
            metadata.roles = {
                name: _role_record(device) for name, device in self._roles.items()
            }

    def _live_frame_acquired(self, image: Any) -> None:
        self.image_acquired.emit(image)

    def image_settings(self) -> ImageSettings:
        """The settings this beam would image with now: its resolution, dwell time and
        field of view, with the path and filename it last saved to. What an acquire
        given no settings uses, so the driver needn't ask the microscope."""
        last = self._last_image_settings or ImageSettings()
        return ImageSettings(
            beam_type=self.beam_type,
            resolution=self._value_or_none("resolution"),
            dwell_time=self._value_or_none("dwell_time"),
            hfw=self._value_or_none("hfw"),
            path=last.path,
            filename=last.filename,
        )

    def _value_or_none(self, name: str) -> Any:
        param = self.parameters.get(name)
        return None if param is None else param.get_value()

    @command(available=lambda beam: implements(beam, "_autocontrast"))
    def autocontrast(self, reduced_area: Optional[FibsemRectangle] = None) -> None:
        """Run the instrument's brightness and contrast routine, optionally on a
        rectangle of the frame (0 to 1); the beam scans the full frame after."""
        self._autocontrast(reduced_area)

    @command(
        available=lambda beam: (
            implements(beam, "_auto_focus") and beam._has_auto_focus()
        )
    )
    def auto_focus(self, reduced_area: Optional[FibsemRectangle] = None) -> None:
        """Run the instrument's autofocus routine, optionally on a rectangle of the
        frame (0 to 1); the beam scans the full frame after.

        Recorded as one ``autofocus`` event, as the focus sweep is: it moves the
        working distance too. ``method`` is ``instrument``. The working distance
        it left is not read for the record, which would be an instrument call the
        focus did not make.
        """
        from fibsem.util.timestamps import now_iso

        record: Dict[str, Any] = {
            "beam_type": self.beam_type.name,
            "method": "instrument",
            "reduced_area": None if reduced_area is None else reduced_area.to_dict(),
            "started_at": now_iso(),
        }
        try:
            self._auto_focus(reduced_area)
        except Exception as error:
            self.session.record("autofocus", {**record, **ended_by(error)})
            raise
        self.session.record("autofocus", {**record, "status": COMPLETED})

    def _acquire(self, image_settings: Optional[ImageSettings]) -> FibsemImage:
        # Not the microscope's acquire_image: that goes through this beam, so it
        # would recurse for a driver without the hook.
        raise NotImplementedError(
            f"{type(self).__name__} can't acquire an image: its driver has no "
            "_acquire and no scanner is bound"
        )

    # A bound scanner images instead of the driver: the beam's settings (or the
    # given ones) say what to scan, the scanner scans it, and the beam builds the
    # image. Only the frame comes from the scanner; hfw is still the column's.

    def _acquire_with_scanner(
        self, image_settings: Optional[ImageSettings]
    ) -> FibsemImage:
        self._check_scanner_free()
        if image_settings is None:
            settings = ImageSettings(
                resolution=tuple(self.resolution.get_value()),
                dwell_time=self.dwell_time.get_value(),
                hfw=self.hfw.get_value(),
                beam_type=self.beam_type,
            )
        else:
            settings = copy.deepcopy(image_settings)
            if "hfw" in self.parameters:
                self.hfw.set_value(settings.hfw)
        frame = self.scanner.acquire(settings.resolution, settings.dwell_time)
        width = settings.resolution[0]
        state = None
        get_state = getattr(self.parent, "get_microscope_state", None)
        if get_state is not None:
            state = get_state(beam_type=self.beam_type)
        return FibsemImage(
            data=frame,
            metadata=FibsemImageMetadata(
                image_settings=settings,
                microscope_state=state,
                pixel_size=Point(settings.hfw / width, settings.hfw / width),
            ),
        )

    # One scanner can be bound to both beams (a unit that switches columns), but it
    # scans for one at a time: a live view holds it for its whole session, and the
    # other beam is refused meanwhile rather than interleaving its frames.

    def _check_scanner_free(self) -> None:
        holder = self.scanner.resources.holder(SCANNER_RESOURCE)
        if holder is not None and holder != self.name:
            raise ResourceBusy(SCANNER_RESOURCE, holder, self._busy(holder))

    def _hold_scanner(self) -> None:
        try:
            self.scanner.resources.hold(SCANNER_RESOURCE, self.name)
        except ResourceBusy as busy:
            raise ResourceBusy(
                SCANNER_RESOURCE, busy.holder, self._busy(busy.holder)
            ) from None

    def _busy(self, holder: str) -> str:
        return (
            f"{self.scanner.name} is live on {holder}: stop live view there before "
            f"imaging with {self.name}"
        )

    def _autocontrast(self, reduced_area: Optional[FibsemRectangle]) -> None:
        raise NotImplementedError

    def _auto_focus(self, reduced_area: Optional[FibsemRectangle]) -> None:
        raise NotImplementedError

    def _has_auto_focus(self) -> bool:
        """Whether this column has the driver's focus routine: a driver whose columns
        differ (one with no focus control) says no for that one."""
        return True

    # Live view: the driver's _live runs on a thread of its own, acquiring with the
    # beam's current settings and emitting each image on live_frame, until stop_live
    # sets the event it is given. Frames are pushed, as the SEM and FIB viewers take
    # them today.

    @command(
        available=lambda beam: implements(beam, "_live") or "scanner" in beam.roles
    )
    def start_live(self) -> None:
        """Acquire continuously with the current settings, each image on
        ``live_frame``, until `stop_live`. Warns and does nothing when already live."""
        with self._live_lock:
            if self.is_live:
                logging.warning(f"{self.name} live view is already running.")
                return
            scanned = "scanner" in self.roles
            if scanned:
                self._hold_scanner()  # refused here if the other beam is live on it
            self._live_stop.clear()
            self._live_thread = threading.Thread(
                target=self._run_live, name=f"{self.name}-live", daemon=True
            )
            self._live_thread.start()

    @command(
        available=lambda beam: implements(beam, "_live") or "scanner" in beam.roles
    )
    def stop_live(self) -> None:
        """Stop live view, waiting briefly for its last frame. Safe when not live."""
        thread = self._live_thread
        if thread is None or self._live_stop.is_set():
            return
        self._live_stop.set()
        if "scanner" in self.roles:
            self.scanner.stop()
        if thread is not threading.current_thread():
            thread.join(timeout=2)

    @property
    def is_live(self) -> bool:
        thread = self._live_thread
        return thread is not None and thread.is_alive()

    def read_live(self) -> bool:
        return self.is_live

    def _run_live(self) -> None:
        scanned = "scanner" in self.roles
        live = self._live_with_scanner if scanned else self._live
        try:
            live(self._live_stop)
        except Exception as e:
            logging.error(f"{self.name} live view stopped: {e}")
        finally:
            if scanned:
                self.scanner.resources.release(SCANNER_RESOURCE, self.name)

    def _live(self, stop: threading.Event) -> None:
        raise NotImplementedError

    def _live_with_scanner(self, stop: threading.Event) -> None:
        while not stop.is_set():
            image = self._acquire_with_scanner(None)
            self._stamp_roles(image)
            if not stop.is_set():
                self.live_frame.emit(image)


def implements(beam: Beam, hook: str) -> bool:
    """Whether the beam's driver overrides *hook*."""
    return getattr(type(beam), hook) is not getattr(Beam, hook)


def _role_record(device: Device) -> Dict[str, Any]:
    """What an image records of the device in one of its beam's roles
    (``FibsemImageMetadata.roles``): its name, class and driver, its parameters'
    cached values and its facts. A value is read from the device only the first
    time, before anything was cached; one that can't be read is None."""
    try:
        facts = device.facts()
    except Exception as error:
        logging.debug(f"{device.name}: facts: {error}")
        facts = {}
    return {
        "name": device.name,
        "class": type(device).__name__,
        "driver": device.driver,
        "values": {
            name: to_wire(_cached_or_none(param))
            for name, param in device.parameters.items()
        },
        "facts": facts,
    }


def _cached_or_none(param: BoundParameter) -> Any:
    try:
        return param.cached
    except Exception as error:
        logging.debug(f"{param.device.name}.{param.name}: {error}")
        return param.last_known
