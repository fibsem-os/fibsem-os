"""The FM API: what the FM UI, acquisition and workflows call, over the FM's devices
(``fibsem.devices.fm``), wherever the devices are.

``FluorescenceMicroscope``'s parts forward to the FM's devices: the camera, light
source, filter set and objective, and the ``fm`` group that runs a channel. It doesn't
know which driver built them, or whether they run in this process or on another
computer:

- the Demo's FM is this over the Demo FM devices (``fibsem.drivers.demo.devices``);
- the Thermo and Odemis FMs add their hardware's extras (``fibsem.fm.autoscript``,
  ``fibsem.fm.odemis``);
- ``RemoteFluorescenceMicroscope`` (``fibsem.fm.remote``) is this over remote devices,
  plus connecting to their server.

Every property is a live read, or a write through the old API path
(``write_through``): nothing is cached behind the caller's back, so a guard reading
``fm.objective.state`` asks the device, and a remote one fails closed with
``RemoteDeviceUnreachable`` when its server can't be reached. Acquiring a channel is
one command on the ``fm`` group, run next to the hardware.

What belongs to the session rather than the hardware stays here: the objective's
saved focus position, the channel name and colour, and the image transform.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from datetime import datetime, timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from psygnal import Signal

from fibsem.fm.progress import (
    FluorescenceAcquisitionProgress,
    FluorescenceAcquisitionStatus,
)
from fibsem.fm.structures import (
    REFLECTION,
    CameraImageTransform,
    ChannelSettings,
    EmissionFilter,
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
    ObjectiveStateName,
    ZParameters,
    ZStackOrder,
    emission_filter_for,
    objective_state_name,
    same_emission_value,
)
from fibsem.util.timestamps import now, now_iso, to_aware

if TYPE_CHECKING:
    from fibsem.devices.core import BoundParameter, Device
    from fibsem.microscope import FibsemMicroscope

RATE_LIMIT_DEFAULT = 0.05  # seconds between updates

FM_DEVICE_NAMES = ("fm", "camera", "light_source", "filter_set", "objective")
"""The devices an FM is made of: the four parts and the group."""


def _param(device: Device, name: str) -> BoundParameter:
    """A part's parameter, failing closed while a remote part's server has never
    answered.

    A remote part built offline has no parameters yet, so a guard reading
    ``fm.objective.state`` gets the same ``RemoteDeviceUnreachable`` as one whose
    server went away later, not an ``AttributeError`` that reads like a missing
    feature. A local part is always online.
    """
    if not getattr(device, "online", True):
        from fibsem.drivers.remote.devices import RemoteDeviceUnreachable

        raise RemoteDeviceUnreachable(
            f"{device.name} at {device.client.base_url} has not connected yet"
        )
    return getattr(device, name)


class ObjectiveLens:
    # Raised after the objective moves, carrying position (metres) and state, so a
    # display can refresh instead of polling the device for numbers it mostly does not
    # need (FIB-534). On a TFS system each such read takes the shared imaging channel,
    # so a label refreshed on every stage poll moves the microscope's own active view to
    # the FM and back to fetch a millimetre reading -- which is how FIB-517 happened.
    #
    # Carries the values rather than being a bare "something moved", for the reason
    # `stage_position_changed` does: subscribers that each re-read would turn one move
    # into a read per subscriber, and there are at least three displays. Read once at
    # the source instead -- see `_notify_moved` for what that costs.
    #
    # State travels with position because every display that wants one wants both: the
    # overview info bar renders "objective 10.061 mm (inserted)" from a single line.
    #
    # On the objective rather than the microscope, following `Stage.position_changed`
    # (`fibsem/microscopes/_stage.py`). It also keeps the name unambiguous --
    # `objective_position_changed` already means something else on the lamella widget --
    # and lets an objective built without a parent announce its own moves.
    #
    # For displays only. Guards read the device, every time -- one of them suppresses z
    # and t from a stage move while the objective is inserted, and a stale "Retracted"
    # there moves the stage with the objective in the chamber.
    #
    # Emitted from whichever thread made the move -- connect with `@ensure_main_thread`
    # if the slot touches widgets, and disconnect on teardown: this is psygnal, so
    # nothing is severed for you when the C++ object goes (FIB-550).
    position_changed = Signal(float, str)

    """The FM API's objective lens: what the FM UI and workflows call.

    Its position, magnification and numerical aperture, and moving it, inserting it and
    retracting it; each implementation answers them from its own hardware (the FM
    devices, ``fibsem.fm.microscope``).
    The saved focus position is the session's, not the hardware's, so it is kept here.

    Attributes:
        parent: Reference to the parent fluorescence microscope
    """

    blocked_axes: Tuple[str, ...] = ("z", "t")
    """The stage axes an absolute move leaves alone while this objective is inserted.

    z and t, as measured on a compustage (FIB-640): with the objective in, the
    microscope refuses a height or tilt change, and a move that sends them anyway
    half-succeeds. A driver whose objective differs overrides this."""

    def __init__(
        self,
        device: Optional[Device] = None,
        parent: Optional["FluorescenceMicroscope"] = None,
    ):
        """Args:
        device: The objective device; a subclass that keeps its own state (the legacy
            simulator) passes none and overrides what reads it.
        parent: Optional parent fluorescence microscope instance
        """
        self.parent = parent
        self._device = device
        self._focus_position: Optional[float] = None  # a session setting, kept here

    def _notify_moved(self) -> None:
        """Announce where the objective ended up, for displays to refresh on (FIB-534).

        Called by every write method that actually moves the device, in every
        implementation -- there is no single chokepoint to put it behind, and the shape
        differs per driver: the simulated `move_relative` adjusts its own field rather
        than delegating, the TFS one delegates to `move_absolute`, and `insert`/`retract`
        delegate on the simulator but not on TFS or odemis. `tests/fm/
        test_objective_signal.py` pins that every one of them calls this, since a driver
        that silently stops announcing is a display that silently goes stale.

        Deliberately not called when a method returns early without moving -- an
        `insert` on an already-inserted objective has nothing to announce.

        Both reads share one channel scope, so on a TFS system a move costs **one** extra
        take of the shared imaging channel rather than two. Not zero: every caller
        invokes this *after* its own scope has closed, deliberately, because psygnal is
        synchronous -- inside, every subscriber's slot would run while the channel was
        held, and a display refresh is not something to hold the microscope for.

        One take per move is still the point of the exercise. What this replaces is a
        read per *poll tick* by each display, and there are at least three of them.

        A read that fails must not turn a move that worked into a raised exception, so a
        failure here is logged and swallowed. The cost of missing one notification is a
        stale label until the next move; the cost of propagating would be a move that
        reports failure after succeeding.
        """
        # Parentless objectives are constructed directly in several tests, and there is
        # no shared channel to take when there is no microscope holding it.
        scope = (
            self.parent.active_channel() if self.parent is not None else nullcontext()
        )
        try:
            with scope:
                position, state = self.position, self.state
        except Exception as e:
            logging.debug(f"Could not read the objective after moving it: {e}")
            return
        self.position_changed.emit(position, state)

    @property
    def magnification(self) -> float:
        """The magnification of the objective lens (e.g. 100.0 for 100x)."""
        return _param(self._device, "magnification").get_value()

    @property
    def numerical_aperture(self) -> float:
        """The numerical aperture of the objective lens."""
        return _param(self._device, "numerical_aperture").get_value()

    @property
    def position(self) -> float:
        """The objective's z position, in metres (negative = retracted)."""
        return _param(self._device, "position").get_value()

    @property
    def focus_position(self) -> Optional[float]:
        """Get the focus position of the objective lens.

        Returns:
            The focus position in meters, or None if not set
        """
        return self._focus_position

    @focus_position.setter
    def focus_position(self, position: Optional[float]):
        """Set the focus position of the objective lens, or None to clear it.

        The getter has always been `Optional[float]` -- an objective has no saved focus
        until someone saves one -- but the setter took a bare float and formatted it
        into its own log line, so clearing raised a TypeError from inside logging.

        Args:
            position: The focus position in metres, or None to forget it.
        """
        self._focus_position = position
        if position is None:
            logging.info("Objective focus position cleared.")
            return
        logging.info(
            f"Objective focus position set to: {self._focus_position * 1e3:.3f} mm"
        )

    @property
    def limit_position(self) -> float:
        """The user-defined z position limit of the objective lens, in metres."""
        return _param(self._device, "limit_position").get_value()

    @limit_position.setter
    def limit_position(self, position: float) -> None:
        _param(self._device, "limit_position").write_through(position)

    @property
    def limits(self) -> Tuple[float, float]:
        """The objective's (minimum, maximum) z positions, in metres."""
        limits = _param(self._device, "position").limits
        return (limits.min, limits.max)

    @property
    def state(self) -> ObjectiveStateName:
        """The objective's state ('Inserted', 'Retracted', 'Busy', 'Error', ...)."""
        return objective_state_name(_param(self._device, "state").get_value())

    # Every move announces itself (`_notify_moved`); insert and retract do unless the
    # driver says nothing moved.

    def move_relative(self, delta: float) -> None:
        """Move the objective by ``delta`` metres (positive = towards the sample)."""
        self._device.move_relative(delta)
        self._notify_moved()

    def move_absolute(self, position: float) -> None:
        """Move the objective to ``position`` metres."""
        self._device.move_absolute(position)
        self._notify_moved()

    def insert(self) -> None:
        """Insert the objective into its working position, for imaging."""
        if self._device.insert() is not False:
            self._notify_moved()

    def retract(self) -> None:
        """Retract the objective to a safe position away from the sample."""
        if self._device.retract() is not False:
            self._notify_moved()


class Camera:
    """The FM API's camera, over the camera device: acquiring a frame, and its
    exposure, binning, gain and offset. Pixel size and resolution are as binned.

    A camera without a gain control (a driver that offers no ``gain``) reads its gain
    as None and ignores a write, warning once."""

    def __init__(
        self,
        device: Optional[Device] = None,
        parent: Optional["FluorescenceMicroscope"] = None,
    ):
        self.parent = parent
        self._device = device
        self._gain_warning_logged = False

    def _has_no_gain(self) -> bool:
        # An offline remote camera has no parameters yet; reading it fails closed.
        return getattr(self._device, "online", True) and (
            "gain" not in self._device.parameters
        )

    def acquire_image(self) -> np.ndarray:
        return self._device.acquire()

    @property
    def exposure_time(self) -> float:
        return _param(self._device, "exposure_time").get_value()

    @exposure_time.setter
    def exposure_time(self, value: float) -> None:
        _param(self._device, "exposure_time").write_through(value)

    @property
    def binning(self) -> int:
        return _param(self._device, "binning").get_value()

    @binning.setter
    def binning(self, value: int) -> None:
        _param(self._device, "binning").write_through(value)

    @property
    def available_binnings(self) -> Tuple[int, ...]:
        return tuple(_param(self._device, "binning").choices or ())

    @property
    def exposure_time_limits(self) -> Tuple[float, float]:
        limits = _param(self._device, "exposure_time").limits
        return (limits.min, limits.max)

    @property
    def gain(self) -> Optional[float]:
        if self._has_no_gain():
            return None
        return _param(self._device, "gain").get_value()

    @gain.setter
    def gain(self, value: float) -> None:
        if self._has_no_gain():
            if not self._gain_warning_logged:
                logging.warning("Camera has no gain control; ignoring gain settings.")
                self._gain_warning_logged = True
            return
        _param(self._device, "gain").write_through(value)

    @property
    def gain_native_scale(self) -> Optional[Tuple[float, Optional[str]]]:
        return _native_scale(self._device, "gain")

    @property
    def offset(self) -> float:
        return _param(self._device, "offset").get_value()

    @offset.setter
    def offset(self, value: float) -> None:
        _param(self._device, "offset").write_through(value)

    @property
    def pixel_size(self) -> Tuple[float, float]:
        return tuple(_param(self._device, "pixel_size").get_value())

    @property
    def resolution(self) -> Tuple[int, int]:
        return tuple(_param(self._device, "resolution").get_value())

    @property
    def field_of_view(self) -> Tuple[float, float]:
        """The (width, height) field of view in metres, as binned."""
        return (
            self.pixel_size[0] * self.resolution[0],
            self.pixel_size[1] * self.resolution[1],
        )


def _native_scale(device: Device, name: str) -> Optional[Tuple[float, Optional[str]]]:
    """A fraction parameter's full scale in hardware units, when the driver gives it."""
    if name not in device.parameters:
        return None
    metadata = _param(device, name).metadata
    if metadata.native_max is None:
        return None
    return (metadata.native_max, metadata.native_unit)


class LightSource:
    """The FM API's light source, over its device: the power, as a fraction of
    full power on every driver; hardware units are the driver's to convert."""

    def __init__(
        self,
        device: Optional[Device] = None,
        parent: Optional["FluorescenceMicroscope"] = None,
    ):
        self.parent = parent
        self._device = device

    @property
    def power(self) -> float:
        return _param(self._device, "power").get_value()

    @power.setter
    def power(self, value: float) -> None:
        _param(self._device, "power").write_through(value)

    @property
    def power_limits(self) -> Tuple[float, float]:
        limits = _param(self._device, "power").limits
        return (limits.min, limits.max)

    @property
    def power_native_scale(self) -> Optional[Tuple[float, Optional[str]]]:
        return _native_scale(self._device, "power")


def _old_emission_value(found: EmissionFilter) -> Optional[Union[float, str]]:
    if found == REFLECTION:
        return None
    return found.name if found.low is None else found.low


def emission_filter_named(
    value: Optional[Union[float, str]], filters: Sequence[EmissionFilter]
) -> EmissionFilter:
    """The filter among ``filters`` that today's emission value names.

    ``None`` is reflection. A label is the filter of that name, or else the multi-band
    filter, as the FM classes take any label to mean fluorescence. A number is the
    band whose bottom edge is closest, as on Odemis; a filter set with no bands (Thermo,
    the simulator) has only its multi-band filter, which is what Thermo reports a
    number as.
    """
    multi_band = [f for f in filters if f != REFLECTION and f.low is None]
    if value is None:
        matches = [f for f in filters if f == REFLECTION]
    elif isinstance(value, str):
        matches = [f for f in filters if f.name == value] or multi_band
    else:
        banded = [f for f in filters if f.low is not None]
        matches = sorted(banded, key=lambda f: abs(f.low - value))[:1] or multi_band
    if not matches:
        raise ValueError(f"No emission filter for {value!r}")
    return matches[0]


class FilterSet:
    """The FM API's filter set, over its device: the excitation wavelength and
    the emission filter. An emission filter is named by one value: None for
    reflection, a label for a multi-band filter, or the band's bottom edge in nm."""

    def __init__(
        self,
        device: Optional[Device] = None,
        parent: Optional["FluorescenceMicroscope"] = None,
    ):
        self.parent = parent
        self._device = device

    @property
    def available_excitation_wavelengths(self) -> Tuple[float, ...]:
        return tuple(_param(self._device, "excitation_wavelength").choices or ())

    # Today's API names a filter by one value: None for reflection, a label for a
    # multi-band filter, or the band's bottom edge in nm.

    @property
    def available_emission_wavelengths(self) -> Tuple[Union[None, str, float], ...]:
        return tuple(_old_emission_value(f) for f in self._emission_filters())

    def _emission_filters(self) -> Tuple[EmissionFilter, ...]:
        return tuple(_param(self._device, "emission_filter").choices or ())

    def emission_filter(self, value: Optional[Union[float, str]]) -> EmissionFilter:
        for found in self._emission_filters():
            if same_emission_value(_old_emission_value(found), value):
                return found
        return emission_filter_for(value, {})

    @property
    def excitation_wavelength(self) -> float:
        return _param(self._device, "excitation_wavelength").get_value()

    @excitation_wavelength.setter
    def excitation_wavelength(self, value: float) -> None:
        _param(self._device, "excitation_wavelength").write_through(value)

    @property
    def emission_wavelength(self) -> Optional[Union[float, str]]:
        return _old_emission_value(_param(self._device, "emission_filter").get_value())

    @emission_wavelength.setter
    def emission_wavelength(self, value: Optional[Union[float, str]]) -> None:
        found = emission_filter_named(value, self._emission_filters())
        _param(self._device, "emission_filter").write_through(found)


class FluorescenceMicroscope:
    """The FM API, over the FM's devices (``fibsem.devices.fm``), local or remote.

    What the FM UI, acquisition and workflows call: the objective, filter set, camera
    and light source, single images, z-stacks and live view. Each part forwards to its
    device, and acquiring a channel is one command on the ``fm`` group, run next to the
    hardware. Backends add only what their hardware needs on top (the Thermo FM's
    shared channel, Odemis's filter bands, a remote FM's connection).

    Attributes:
        objective: The objective lens controller
        filter_set: The filter set controller
        camera: The camera controller
        light_source: The light source controller
        acquisition_signal: Signal emitted when new images are acquired

    Signals:
        acquisition_signal(FluorescenceImage): Emitted during live acquisition
    """

    objective: ObjectiveLens
    filter_set: FilterSet
    camera: Camera
    light_source: LightSource

    # live acquisition signals (psygnal binds these per-instance on access)
    acquisition_signal = Signal(FluorescenceImage)
    # How far along an acquisition *routine* is. Still a bare dict, pending the typed
    # payload it should carry (FIB-401); until then the contract is here rather than
    # only in the emit sites.
    #
    #   operation: which routine -- "channels" | "z-stack" | "autofocus". A closed set,
    #             one member per function in `fm.acquisition` / `fm.calibration` that
    #             reports progress. Absent on events that are not about a routine.
    #   state:    "acquiring" | "moving" | "finished" | "autofocus"
    #
    # Plus, per operation: `channel`, `channel_index`, `total_channels`; `zlevel` and
    # `total_zlevels` for a z-stack or a focus sweep; `pass_index` and `total_passes`
    # for a sweep.
    #
    # `operation`, not `task`: an AutoLamella *task* is a different and larger thing,
    # and while this field was called `task` the workflow tasks filled it in with their
    # own names -- six emit sites whose value no consumer could ever match, since every
    # read compares against the closed set above. Removed in #521, renamed here so the
    # mistake is not available to make again.
    #
    # What a *task* is doing belongs on the task's own signals (`step_update_signal`,
    # `workflow_status_signal`), and tile progress belongs on `tiled_acquisition_signal`
    # (FIB-725). This signal is the acquisition functions', and nothing else's.
    acquisition_progress_signal = Signal(FluorescenceAcquisitionProgress)
    # Raised when something starts or stops driving the FM, so a widget that did not
    # start it can grey its controls out. Emitted from whichever thread set the flag --
    # connect with `@ensure_main_thread` if the slot touches widgets.
    acquiring_changed = Signal(bool)
    # Raised when the user's image transform changes. It is part of
    # `fm_image_geometry()`, so it is an input to every projection between stage space
    # and a canvas -- a display that keeps one has to be told, and had no way to know
    # (FIB-521). Same threading contract as the signals above.
    transform_changed = Signal(object)  # CameraImageTransform

    def __init__(
        self,
        devices: Optional[Dict[str, Device]] = None,
        parent: Optional["FibsemMicroscope"] = None,
    ):
        """Args:
        devices: The FM's devices by name (``FM_DEVICE_NAMES``). Without them the
            parts are left for a subclass to set.
        parent: Optional parent FibsemMicroscope instance for stage access
        """
        super().__init__()

        self.parent = parent
        self.devices = devices

        # per-instance acquisition state (previously shared class attributes)
        self._stop_acquisition_event = threading.Event()
        self._acquisition_thread: Optional[threading.Thread] = None
        # Set while the live stream runs: by `start_acquisition`, and cleared by the
        # worker as its very last act, which then announces the stop itself. Not the
        # thread's `is_alive()`: the worker is still alive while it announces, so the
        # announcement would read "streaming", and `stop_acquisition` gives up waiting
        # after 2 s -- a frame slower than that left the stream announced as running
        # with nothing left to say otherwise.
        self._streaming = threading.Event()
        # Something other than the live stream is driving the FM: an overview tileset,
        # a z-stack, an autofocus sweep. Set by whoever is driving it. The stream is not
        # in here -- it reports itself through `_streaming`. See `is_acquiring`.
        self._acquiring: bool = False
        self._acquiring_reason: str = ""

        self.channel_name: str = "channel-01"
        self.channel_color: str = "gray"
        if devices is not None:
            self.objective = ObjectiveLens(devices["objective"], parent=self)
            self.camera = Camera(devices["camera"], parent=self)
            self.light_source = LightSource(devices["light_source"], parent=self)
            self.filter_set = FilterSet(devices["filter_set"], parent=self)
        self._last_updated_at: Optional[datetime] = datetime.now()
        self._rate_limit = RATE_LIMIT_DEFAULT  # seconds between updates
        self._transform: Optional[CameraImageTransform] = (
            CameraImageTransform.NONE
        )  # image transformation
        self.default_orientation: str = (
            "FM"  # orientation used when computing fluorescence pose for new lamellas
        )
        # Orientations the objective can actually image the sample from. Not a control
        # gate: whether the user may *operate* the FM somewhere is answered by hardware
        # interlocks (the objective's own z/t restrictions, the no-rotation-at-the-FM
        # guard), and whether an *acquisition* may start is
        # `get_device_imaging_state(...).allows_acquisition` on the parent microscope
        # -- the stage owns stage questions.
        #
        # Read from the device declaration (`stage.devices.FM.available_orientations`)
        # so there is exactly one source of truth. On a compustage that is `["FM"]` --
        # what this attribute always held. On an offset mount it is the beam pose the
        # sample is held in at the FM (`["FIB"]` on the iFLM simulator), which the old
        # hardcoded `[default_orientation]` got wrong: `"FM"` is a pose the classifier
        # never returns there. This attribute survives only until its readers move onto
        # `get_device_imaging_state` (FIB-839), then goes with them.
        self.acquisition_orientations: list[str] = (
            self._configured_acquisition_orientations()
        )

    def _configured_acquisition_orientations(self) -> list[str]:
        """The FM device's declared imaging orientations, from the stage configuration.

        Falls back to `[default_orientation]` for an FM constructed without a parent
        microscope (widget tests do this), where there is no configuration to read.
        """
        try:
            devices = self.parent.system.stage.devices
            return list(devices["FM"].available_orientations)
        except (AttributeError, KeyError, TypeError):
            return [self.default_orientation]

    def __repr__(self):
        """Return a string representation of the fluorescence microscope.

        Returns:
            A string showing the microscope class and component status
        """
        return f"{self.__class__.__name__}(objective={self.objective}, filter_set={self.filter_set}, camera={self.camera}, light_source={self.light_source})"

    # ── what is running ──────────────────────────────────────────────────
    #
    # Three questions, because widgets ask three. Assembling them at each call site is
    # what produced six overlapping flags and one wrong answer (FIB-513):
    #
    #   is_streaming    the live view is on
    #   is_acquiring    anything is running: the stream, or a claimed operation
    #   is_interactive  the user may drive the hardware

    @property
    def is_streaming(self) -> bool:
        """Whether the live view is on -- a stream of frames off the camera.

        Narrower than `is_acquiring`, and the difference matters twice: this is what
        labels the start/stop button, since stopping is the one thing that must stay
        possible while it runs, and it is what makes `is_interactive` true.
        """
        return self._streaming.is_set()

    @property
    def is_acquiring(self) -> bool:
        """Whether anything is driving the FM: the live stream, or some other operation.

        The question to ask before starting new work. A tileset moves the stage between
        every pair of tiles with no stream running, so `is_streaming` reads idle for most
        of somebody else's run -- which is how a live stream came to be startable
        mid-tileset (FIB-441).
        """
        return self.is_streaming or self._acquiring

    @property
    def is_interactive(self) -> bool:
        """Whether the user may drive the hardware -- the objective, channel parameters.

        True while idle, and true while streaming: focusing during live view is what the
        objective control is *for*. False while a tileset, z-stack or autofocus sweep is
        running, because those drive the objective themselves and a second hand on it
        corrupts the run.

        Widgets used to derive this per call site from two or three flags, and the
        objective panel got it wrong -- it asked only whether *this* widget was busy, so
        another tab's tileset left it live (FIB-513).
        """
        return not self.is_acquiring or self.is_streaming

    @property
    def acquiring_reason(self) -> str:
        """What is driving the FM, phrased for a warning, or "" if nothing is."""
        if self._acquiring:
            return self._acquiring_reason
        if self.is_streaming:
            return "live acquisition"
        return ""

    def set_acquiring(self, acquiring: bool, reason: str = "") -> None:
        """Mark an operation as running, or finished. Call in a `finally` when it ends.

        Not for the live stream, which reports itself through its thread: a flag that
        outlived a dead thread would strand the instrument.

        Args:
            acquiring: Whether something is now driving the microscope.
            reason: What it is, e.g. "overview acquisition". Shown to the user by
                whatever gets refused, so it should read as a noun phrase.
        """
        self._acquiring = acquiring
        self._acquiring_reason = reason if acquiring else ""
        self.acquiring_changed.emit(self.is_acquiring)

    def refusal_to_start(self, what: str) -> Optional[str]:
        """Why *what* may not start right now, as one actionable sentence -- or None.

        The two questions every entry point has to ask, in the order that matters
        (FIB-839): *when* first -- is something already driving the instrument --
        then *where* -- can it image the sample from here. Asked together so no
        caller assembles them its own way; the control widget used to be the only
        site that composed both, as a private method returning a bare bool, and
        every other gate either asked half the question or logged a message about
        the wrong half.

        Returns the message rather than raising or logging, because the right
        delivery differs per caller: a workflow raises, a widget toasts, a canvas
        handler logs. `None` means go.
        """
        if self.is_acquiring:
            reason = self.acquiring_reason or "another acquisition"
            return (
                f"Cannot start {what}: the fluorescence microscope is in use "
                f"({reason})."
            )
        state = self.parent.get_device_imaging_state("FM")
        if not state.allows_acquisition:
            return (
                f"Cannot start {what}. "
                f"{self.parent.describe_device_imaging_state('FM', state)}"
            )
        return None

    def set_active_channel(self) -> None:
        """Point the microscope's imaging channel at the FM, and leave it there.

        A no-op here, for the same reason `active_channel` is. Part of the contract
        rather than a driver detail so that code which needs the unscoped form -- taking
        the channel for a live stream that owns it until it stops -- can call it on any
        FM without asking what it is talking to.

        Prefer :meth:`active_channel`. This is the form that caused FIB-517: a read that
        takes the channel and walks away leaves the microscope pointed at the FM, and
        the next beam operation to read a buffer reads the FM's.
        """
        return

    @contextmanager
    def active_channel(self):
        """Hold the microscope's imaging channel on the FM for the length of the block.

        A no-op here, and on any system where the FM has a connection of its own. On a
        TFS system the FM and the beams share one -- one active view, one active device,
        last writer wins -- so the driver overrides this to point it at the FM and put it
        back afterwards (FIB-517).

        Wrap whole operations rather than individual reads. Each frame of a tileset
        taking and returning the channel would flick the microscope's own UI between
        views per tile; the run takes it once instead. Nesting is safe, so an inner
        acquisition inside an outer run costs nothing.
        """
        yield

    def set_channel(self, channel_settings: ChannelSettings):
        """Configure the microscope for a specific fluorescence channel.

        Args:
            channel_settings: Complete channel configuration including wavelengths,
                            power, exposure time, and channel name

        Raises:
            ValueError: If required components are not available
        """
        if not self.filter_set:
            raise ValueError("No filter sets available.")
        if not self.light_source:
            raise ValueError("Light source is not set.")
        if not self.camera:
            raise ValueError("Camera is not set.")

        # Configure filter wavelengths
        self.filter_set.excitation_wavelength = channel_settings.excitation_wavelength
        self.filter_set.emission_wavelength = channel_settings.emission_wavelength

        # Set light source power
        self.set_power(channel_settings.power)

        # Set camera settings
        self.set_exposure_time(channel_settings.exposure_time)

        # set gain (optional)
        if channel_settings.gain is not None:
            self.set_gain(channel_settings.gain)

        # set channel name
        self.set_channel_name(channel_settings.name)
        self.set_channel_color(channel_settings.color)

    def set_channel_name(self, name: str):
        """Set the name identifier for the current imaging channel.

        Args:
            name: The channel name (e.g., 'DAPI', 'GFP', 'channel-01')
        """
        self.channel_name = name

    def set_channel_color(self, color: str):
        """Set the color for the current channel.

        Args:
            color: The color name or hex code (e.g., 'red', '#FF0000')
        """
        self.channel_color = color

    def set_binning(self, binning: int):
        """Set the camera binning factor.

        Args:
            binning: The binning factor (1, 2, 4, 8, etc.)

        Raises:
            ValueError: If camera is not available
        """
        if not self.camera:
            raise ValueError("Camera is not set.")
        self.camera.binning = binning

    def set_exposure_time(self, exposure_time: float):
        """Set the camera exposure time.

        Args:
            exposure_time: The exposure time in seconds

        Raises:
            ValueError: If camera is not available
        """
        if not self.camera:
            raise ValueError("Camera is not set.")
        self.camera.exposure_time = exposure_time

    def set_power(self, power: float):
        """Set the light source power level.

        Args:
            power: The power level in watts

        Raises:
            ValueError: If light source is not available
        """
        if not self.light_source:
            raise ValueError("Light source is not set.")
        self.light_source.power = power

    def set_rate_limit(self, limit: float):
        """Set the rate limit for image acquisition.
        Args:
            limit: The rate limit in seconds
        """
        self._rate_limit = limit

    def set_gain(self, gain: float):
        """Set the camera gain.

        Args:
            gain: The gain value (amplification factor)

        Raises:
            ValueError: If camera is not available
        """
        if not self.camera:
            raise ValueError("Camera is not set.")
        self.camera.gain = gain

    def set_image_transform(self, transform: Optional[CameraImageTransform]):
        """Set the image transformation to align fluorescence images with SEM/FIB images.

        Args:
            transform: The transform to apply (CameraImageTransform enum value or None).
                - CameraImageTransform.NONE or None: No transformation
                - CameraImageTransform.FLIP_X: Horizontal flip
                - CameraImageTransform.FLIP_Y: Vertical flip
                - CameraImageTransform.FLIP_XY: Both flips (equivalent to a 180° rotation)

            Rotations are not offered here: a fixed rotation between the sensor and
            the stage describes the mount, and is corrected by `mount_transform`
            inside the driver before this preference is applied.

        Raises:
            ValueError: If transform is not a valid CameraImageTransform enum value
        """
        if transform is not None and not isinstance(transform, CameraImageTransform):
            raise ValueError(
                f"Invalid transform '{transform}'. Must be a CameraImageTransform enum value or None"
            )
        previous = self._transform
        self._transform = (
            transform if transform is not None else CameraImageTransform.NONE
        )
        logging.info(f"Image transform set to: {transform}")
        # Announced, because this is not only a display preference: `fm_image_geometry()`
        # carries it, so it is an input to every projection between stage space and a
        # canvas. A view holding a projection has no other way to learn it moved
        # (FIB-521). Only on a real change -- the camera widget re-applies the saved
        # transform on load, and a redraw for a value that did not move is noise.
        if self._transform is not previous:
            self.transform_changed.emit(self._transform)

    @property
    def camera_tilt(self) -> float:
        """Tilt of the FM optical axis from the SEM column, in degrees.

        The camera's analogue of a beam column's ``column_tilt``; used to project
        in-image displacements onto the tilted sample plane.

        Derived from the mount geometry rather than configured:

        - **Under-grid mounts (Arctis / compustage)** look up at the grid from the
          opposite side to the SEM: a half turn, 180 degrees.
        - **Offset mounts (METEOR, iFLM)** sit parallel to the FIB column, displaced
          along x, so they share the ion column's tilt.

        Drivers override if a system disagrees. This needs to become configurable for
        systems whose mount is neither of the two known cases -- see FIB-335.
        """
        if self.parent is None:
            return 0.0  # simulator without a parent microscope
        if self.parent._fm_is_a_pose():
            return 180.0
        return self.parent.system.ion.column_tilt

    @property
    def mount_transform(self) -> CameraImageTransform:
        """How the camera is mounted, as its device reports it, so an FM on another
        computer brings its own. A camera without the parameter (a server from before
        it) is mounted straight, as every FM was. Read once and kept: every frame
        uses it, and a mount does not change."""
        # Here rather than at the top: fibsem.devices.fm imports fibsem.fm.
        from fibsem.devices.fm import mount_transform_from_name

        camera = self.devices["camera"]
        if "mount_transform" not in camera.parameters:
            return CameraImageTransform.NONE
        return mount_transform_from_name(_param(camera, "mount_transform").cached)

    @staticmethod
    def _transform_array(
        data: np.ndarray, transform: Optional[CameraImageTransform]
    ) -> np.ndarray:
        """Apply a single CameraImageTransform to an array."""
        if transform is CameraImageTransform.FLIP_X:
            return np.fliplr(data)
        elif transform is CameraImageTransform.FLIP_Y:
            return np.flipud(data)
        elif transform is CameraImageTransform.FLIP_XY:
            return np.fliplr(np.flipud(data))  # a half turn is both flips
        else:
            return data

    def _apply_image_transform(self, data: np.ndarray) -> np.ndarray:
        """Apply the configured image transformation to align with SEM/FIB coordinate system.

        Two stages: the fixed mount correction puts the raw sensor into stage-aligned
        axes, then the user's transform applies their display preference on top.

        Args:
            data: The image data to transform

        Returns:
            The transformed image data
        """
        data = self._transform_array(data, self.mount_transform)
        return self._transform_array(data, self._transform)

    def acquire_image(
        self, channel_settings: Optional[ChannelSettings] = None
    ) -> FluorescenceImage:
        """One command on the ``fm`` group sets up the channel and takes the frame,
        with what it was taken with, so building the image needs no further reads."""
        with self.active_channel():
            channel = None
            if channel_settings is not None:
                # The name and colour are this session's labels, not hardware.
                self.channel_name = channel_settings.name
                self.channel_color = channel_settings.color
                channel = channel_settings.to_dict()
            frame = self.devices["fm"].acquire_frame(channel)
            return self._construct_image(frame.data, frame.metadata)

    def _construct_image(
        self, data: np.ndarray, frame_metadata: Optional[dict] = None
    ) -> FluorescenceImage:
        """Construct a FluorescenceImage from raw data with associated metadata.

        Applies the configured image transformation to align the fluorescence image
        with the SEM/FIB coordinate system.

        Args:
            data: The raw image data.
            frame_metadata: Optional per-frame metadata captured at exposure
                time by the driver; where present these values override the
                state snapshot from get_metadata(). Supported keys:
                'pixel_size' ((x, y) in metres), 'acquisition_date'
                (ISO string), 'exposure_time' (seconds).
        """
        md = self._metadata_for_frame(frame_metadata)

        # Apply image transformation to align with SEM/FIB images
        data = self._apply_image_transform(data)

        img = FluorescenceImage(data=data, metadata=md)

        now = datetime.now()
        if self._last_updated_at is not None:
            time_since_last_update = now - self._last_updated_at
            if time_since_last_update >= timedelta(seconds=self._rate_limit):
                self.acquisition_signal.emit(img)  # Emit the acquired image signal
                self._last_updated_at = now

        return img

    def _metadata_for_frame(
        self, frame_metadata: Optional[dict]
    ) -> FluorescenceImageMetadata:
        """Built from what the ``fm`` group reported with the frame. Anything it didn't
        report (a server from before ``acquire_frame``) is read live, as before."""
        frame = frame_metadata or {}

        def reported(key: str, read: Callable[[], Any]) -> Any:
            return frame[key] if key in frame else read()

        if "emission_filter" in frame:
            emission = _old_emission_value(
                EmissionFilter.from_dict(frame["emission_filter"])
            )
        else:
            emission = self.filter_set.emission_wavelength
        camera, objective = self.camera, self.objective
        channel = FluorescenceChannelMetadata(
            name=self.channel_name,
            color=self.channel_color,
            excitation_wavelength=reported(
                "excitation_wavelength",
                lambda: self.filter_set.excitation_wavelength,
            ),
            emission_wavelength=emission,
            power=reported("power", lambda: self.light_source.power),
            exposure_time=reported("exposure_time", lambda: camera.exposure_time),
            gain=reported("gain", lambda: camera.gain),
            offset=reported("offset", lambda: camera.offset),
            binning=reported("binning", lambda: camera.binning),
            objective_position=reported(
                "objective_position", lambda: objective.position
            ),
            objective_magnification=reported(
                "objective_magnification", lambda: objective.magnification
            ),
            objective_numerical_aperture=reported(
                "objective_numerical_aperture", lambda: objective.numerical_aperture
            ),
        )
        pixel_size = reported("pixel_size", lambda: camera.pixel_size)
        resolution = reported("resolution", lambda: camera.resolution)
        # With its offset from `FM.acquire_frame`; a frame from a server older than
        # that may have none, and then it is the acquiring machine's clock time alone.
        acquired = reported("acquisition_date", now_iso)
        parent = self.parent
        # The coordinator's own state, as `get_metadata` stamps it.
        return FluorescenceImageMetadata(
            acquisition_date=acquired,
            acquisition_datetime=to_aware(acquired),
            pixel_size_x=pixel_size[0],
            pixel_size_y=pixel_size[1],
            resolution=(resolution[0], resolution[1]),
            stage_position=parent.get_stage_position() if parent else None,
            geometry=parent.fm_image_geometry() if parent else None,
            experiment=deepcopy(parent.experiment) if parent else None,
            channels=[channel],
        )

    def get_metadata(self) -> FluorescenceImageMetadata:
        """Generate comprehensive metadata for the current microscope state.

        Collects settings from all microscope components and stage position
        to create complete acquisition metadata.

        Returns:
            Structured metadata including all relevant acquisition parameters
        """
        stage_position = self.parent.get_stage_position() if self.parent else None

        # Create channel metadata from current microscope state
        channel_metadata = FluorescenceChannelMetadata(
            name=self.channel_name,
            color=self.channel_color,
            excitation_wavelength=self.filter_set.excitation_wavelength,
            emission_wavelength=self.filter_set.emission_wavelength,
            power=self.light_source.power,
            exposure_time=self.camera.exposure_time,
            gain=self.camera.gain,
            offset=self.camera.offset,
            binning=self.camera.binning,
            objective_position=self.objective.position,
            objective_magnification=self.objective.magnification,
            objective_numerical_aperture=self.objective.numerical_aperture,
        )

        # Stamp the geometry the image is being captured under, so a stage position can
        # be projected onto it later without assuming the microscope still matches --
        # by then the stage has usually moved, and the display transform may have been
        # flipped. Needs the parent for the stage and system settings, so an FM with no
        # parent records nothing rather than recording a default that looks valid.
        geometry = self.parent.fm_image_geometry() if self.parent else None

        # Which experiment, item and task, from the same place the beam path reads it.
        # Copied rather than referenced: the workflow half is rewritten as the run
        # moves on, and an image should keep saying where it was taken (FIB-466).
        experiment = deepcopy(self.parent.experiment) if self.parent else None

        # Create complete image metadata
        acquired = now()
        return FluorescenceImageMetadata(
            acquisition_date=acquired.isoformat(),
            acquisition_datetime=acquired,
            pixel_size_x=self.camera.pixel_size[0],
            pixel_size_y=self.camera.pixel_size[1],
            resolution=(self.camera.resolution[0], self.camera.resolution[1]),
            stage_position=stage_position,
            geometry=geometry,
            experiment=experiment,
            channels=[channel_metadata],
        )

    def start_acquisition(
        self, channel_settings: Optional[ChannelSettings] = None
    ) -> None:
        """Start continuous live image acquisition in a separate thread.

        Begins continuous image acquisition and emits acquisition_signal for each
        captured image. Useful for live preview and real-time monitoring.

        Args:
            channel_settings: Optional channel configuration to apply before
                            starting acquisition

        Note:
            Images are emitted via the acquisition_signal. Connect to this signal
            to receive live images. Call stop_acquisition() to end the process.
        """
        if self.is_streaming or (
            self._acquisition_thread and self._acquisition_thread.is_alive()
        ):
            logging.warning("Acquisition thread is already running.")
            return

        # reset stop event if needed
        self._stop_acquisition_event.clear()
        self._streaming.set()

        # start acquisition thread
        self._acquisition_thread = threading.Thread(
            target=self._acquisition_worker, args=(channel_settings,), daemon=True
        )
        self._acquisition_thread.start()
        # A live stream counts as acquiring just as anything else does, so it announces
        # itself the same way -- a widget deriving its controls from `is_acquiring`
        # would otherwise notice tilesets and never notice streaming.
        self.acquiring_changed.emit(self.is_acquiring)

    def stop_acquisition(self) -> None:
        """Stop the continuous live image acquisition.

        Signals the acquisition thread to stop and waits for it to complete.
        Safe to call even if acquisition is not currently running.

        Note:
            Will wait up to 2 seconds for the acquisition thread to terminate.
            The acquisition_signal will stop emitting new images.
        """
        if self._stop_acquisition_event and not self._stop_acquisition_event.is_set():
            self._stop_acquisition_event.set()
            if self._acquisition_thread:
                self._acquisition_thread.join(timeout=2)
            # Asked rather than asserted: the join has a timeout, so a thread that has
            # not stopped yet must not be announced as stopped.
            self.acquiring_changed.emit(self.is_acquiring)

    @property
    def runs_z_stack_on_device(self) -> bool:
        """Whether a z-stack is one command on the ``fm`` group: when the group runs
        on another computer and has the command. A local FM runs it step by step, so
        each slice is shown as it arrives."""
        group = self.devices["fm"]
        return getattr(group, "runs_elsewhere", False) and (
            "acquire_z_stack" in getattr(group, "server_commands", group.commands)
        )

    def acquire_z_stack_on_device(
        self,
        channel_settings: Union[ChannelSettings, List[ChannelSettings]],
        zparams: ZParameters,
        stop_event: Optional[threading.Event] = None,
    ) -> Optional[FluorescenceImage]:
        """``fibsem.fm.acquisition.acquire_z_stack`` as one ``fm`` group command:
        the same positions, order, progress and cancelling, with the frames coming
        back together at the end."""
        group = self.devices["fm"]
        channels = (
            channel_settings
            if isinstance(channel_settings, list)
            else [channel_settings]
        )
        with self.active_channel():
            z_init = self.objective.position
            positions = [float(z) for z in zparams.generate_positions(z_init=z_init)]
            order = "z" if zparams.order == ZStackOrder.Z_LEVEL else "channel"

            def on_changed(name: str, value: Any) -> None:
                if name != "progress" or not value:
                    return
                self.acquisition_progress_signal.emit(
                    FluorescenceAcquisitionProgress(
                        status=FluorescenceAcquisitionStatus.ACQUIRING_ZSTACK,
                        **value,
                    )
                )

            done = threading.Event()

            def watch_for_stop() -> None:
                while not done.wait(0.1):
                    if stop_event.is_set():
                        group.cancel()
                        return

            group.changed.connect(on_changed)
            if stop_event is not None:
                threading.Thread(
                    target=watch_for_stop, name="fm-z-stack-stop", daemon=True
                ).start()
            try:
                frames = group.acquire_z_stack(
                    channels=[ch.to_dict() for ch in channels],
                    positions=positions,
                    order=order,
                    restore_position=z_init,
                )
            finally:
                done.set()
                group.changed.disconnect(on_changed)
            # The objective moved on the FM's side; announce where it ended up.
            self.objective._notify_moved()
            if not frames:
                logging.info("Z-stack acquisition cancelled")
                return None

            images: List[FluorescenceImage] = []
            n = len(positions)
            for i, ch in enumerate(channels):
                self.channel_name, self.channel_color = ch.name, ch.color
                planes = [
                    self._construct_image(frame.data, frame.metadata)
                    for frame in frames[i * n : (i + 1) * n]
                ]
                images.append(FluorescenceImage.create_z_stack(planes))
            return FluorescenceImage.create_multi_channel_image(images)

    def _acquisition_worker(
        self, channel_settings: Optional[ChannelSettings] = None
    ) -> None:
        """Live view, pulled: the ``fm`` group keeps the hardware acquiring, and each
        frame is one `acquire_image` with the current settings. Stopping, or this
        process going away, ends it; the group stops by itself if no frame is asked
        for in its ``live_timeout``."""
        try:
            group = self.devices["fm"]
            if channel_settings is not None:
                self.set_channel(channel_settings)
            group.start_live()
            try:
                while not self._stop_acquisition_event.is_set():
                    self.acquire_image()
            finally:
                group.stop_live()
        except Exception as e:
            logging.error(f"Error in acquisition worker: {e}")
        finally:
            self._streaming.clear()
            self.acquiring_changed.emit(self.is_acquiring)
