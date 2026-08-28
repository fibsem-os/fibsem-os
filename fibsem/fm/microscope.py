from __future__ import annotations

import logging
import threading
from abc import ABC
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Literal, Optional, Tuple, Union

import numpy as np
from psygnal import Signal

from fibsem._timing import sim_sleep
from fibsem.fm.progress import FluorescenceAcquisitionProgress
from fibsem.fm.structures import (
    CameraImageTransform,
    ChannelSettings,
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)
from fibsem.util.draw_numbers import draw_text

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope
    from fibsem.structures import FibsemStagePosition

EXCITATION_WAVELENGTHS = (365, 450, 550, 635)  # in nm, example wavelengths
EMISSION_WAVELENGTHS = (None, "Fluorescence")  # in nm, example wavelengths

SIM_OBJECTIVE_MAGNIFICATION = 100.0  # placeholder for simulation
SIM_OBJECTIVE_NA = 0.8
SIM_OBJECTIVE_INSERT_POSITION = 6.0e-3  # z-axis
SIM_OBJECTIVE_RETRACT_POSITION = -10e-3  # z-axis
SIM_OBJECTIVE_POSITION_LIMITS = (-12e-3, 10e-3)  # z-axis limits for the objective lens
SIM_OBJECTIVE_USER_POSITION_LIMIT = 8.6e-3  # user-defined limits for the objective lens
SIM_OBJECTIVE_FOCUS_POSITION = 8.0e-3
# Insertion and retraction traverse the objective's whole range -- 16 mm here -- where a
# focus nudge moves it by microns, so they are seconds of travel on a real system rather
# than the fraction of one `move_absolute` simulates. Modelled because the difference is
# what makes a second command *during* one reachable by hand (FIB-628); a delegating
# `insert` that returned as fast as a nudge made that window too small to click into.
# `sim_sleep`, so `FIBSEM_SIM_NO_DELAY=1` keeps it out of the test suite.
SIM_OBJECTIVE_TRAVEL_SECONDS = 2.0

SIM_CAMERA_EXPOSURE_TIME = 0.1  # seconds
SIM_CAMERA_EXPOSURE_LIMITS = (1e-6, 60.0)  # seconds
SIM_CAMERA_BINNING = 4
SIM_CAMERA_GAIN = 0.01  # 1%
SIM_CAMERA_OFFSET = 0.0
SIM_CAMERA_PIXEL_SIZE = (
    0.25 * 100e-9,
    0.25 * 100e-9,
)  # in meters (100 nm -> 0.25 um with 4x binning)
SIM_CAMERA_RESOLUTION = (4 * 1024, 4 * 1024)  # default resolution

UINT16_MIN = np.iinfo(np.uint16).min  # 0 for uint16
UINT16_MAX = np.iinfo(np.uint16).max  # 65535 for uint16
BINNING_VALUES = [1, 2, 4, 8]  # typical binning values

RATE_LIMIT_DEFAULT = 0.05  # seconds between updates
ALLOW_UNKNOWN_ORIENTATIONS = True  # allow fm control at any orientation


class ObjectiveLens(ABC):
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
    # and r from a stage move while the objective is inserted, and a stale "Retracted"
    # there moves the stage with the objective in the chamber.
    #
    # Emitted from whichever thread made the move -- connect with `@ensure_main_thread`
    # if the slot touches widgets, and disconnect on teardown: this is psygnal, so
    # nothing is severed for you when the C++ object goes (FIB-550).
    position_changed = Signal(float, str)

    """Abstract base class for objective lens control in fluorescence microscopy.

    Provides a standardized interface for controlling objective lens positioning,
    magnification, and numerical aperture across different microscope implementations.
    Supports insertion/retraction operations for automated workflows.

    Attributes:
        parent: Reference to the parent fluorescence microscope
    """

    def __init__(self, parent: Optional["FluorescenceMicroscope"] = None):
        """Initialize the objective lens with default simulation parameters.

        Args:
            parent: Optional parent fluorescence microscope instance
        """
        self.parent = parent
        self._position: float = SIM_OBJECTIVE_RETRACT_POSITION  # initial position
        self._magnification: float = SIM_OBJECTIVE_MAGNIFICATION
        self._numerical_aperture = SIM_OBJECTIVE_NA
        self._insert_position = SIM_OBJECTIVE_INSERT_POSITION
        self._retract_position = SIM_OBJECTIVE_RETRACT_POSITION
        self._focus_position: Optional[float] = SIM_OBJECTIVE_FOCUS_POSITION
        self._limit_position: float = SIM_OBJECTIVE_USER_POSITION_LIMIT

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
        """Get the magnification of the objective lens.

        Returns:
            The objective lens magnification (e.g., 100.0 for 100x)
        """
        return self._magnification

    @property
    def numerical_aperture(self) -> float:
        """Get the numerical aperture of the objective lens.

        Returns:
            The numerical aperture value (typically 0.1 to 1.4)
        """
        return self._numerical_aperture

    @property
    def position(self) -> float:
        """Get the current z-axis position of the objective lens.

        Returns:
            The current position in meters (negative values = retracted)
        """
        sim_sleep(0.1)
        # logging.info(f"Objective position read: {self._position * 1e3:.3f} mm")
        return self._position

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
        """Get the user-defined z-axis position limit of the objective lens.

        Returns:
            The maximum allowable position in meters
        """
        return self._limit_position

    @limit_position.setter
    def limit_position(self, position: float):
        """Set the user-defined z-axis position limit of the objective lens.

        Args:
            position: The maximum allowable position in meters
        """
        self._limit_position = position
        logging.info(
            f"Objective user-defined position limit set to: {self._limit_position * 1e3:.3f} mm"
        )

    def move_relative(self, delta: float):
        """Move the objective lens by a relative distance.

        Args:
            delta: The distance to move in meters (positive = towards sample)
        """
        self._position += delta
        logging.info(
            f"Objective moved to new position: {self._position * 1e3:.3f} mm (delta: {delta * 1e3:.3f} mm)"
        )
        # Announced here rather than relying on `move_absolute`: this implementation
        # adjusts the field itself instead of delegating.
        self._notify_moved()

    def move_absolute(self, position: float):
        """Move the objective lens to an absolute z-axis position.

        Args:
            position: The target position in meters
        """
        # clip to user-defined limits
        if not position <= self._limit_position:
            logging.warning(
                f"Clipping position {position} to user-defined limits {self._limit_position}"
            )
            position = np.clip(position, 0, self._limit_position)

        sim_sleep(0.5)  # Simulate time taken to move the objective
        self._position = position
        logging.info(
            f"Objective moved to absolute position: {self._position * 1e3:.3f} mm"
        )
        self._notify_moved()

    def insert(self):
        """Insert the objective lens into the working position for imaging.

        Moves the objective lens to the predefined insertion position,
        typically at or near the sample focal plane.
        """
        sim_sleep(SIM_OBJECTIVE_TRAVEL_SECONDS)  # the traverse, on top of the move
        self.move_absolute(self._insert_position)
        logging.info(
            f"Objective lens inserted to position: {self._insert_position:.3f} mm"
        )

    def retract(self):
        """Retract the objective lens to a safe position away from the sample.

        Moves the objective lens to the predefined retraction position
        to prevent damage during stage movements or sample changes.
        """
        sim_sleep(SIM_OBJECTIVE_TRAVEL_SECONDS)
        self.move_absolute(self._retract_position)
        logging.info(
            f"Objective lens retracted to position: {self._retract_position:.3f} mm"
        )

    @property
    def limits(self) -> Tuple[float, float]:
        """Get the z-axis position limits of the objective lens.

        Returns:
            A tuple of (minimum, maximum) positions in meters
        """
        return SIM_OBJECTIVE_POSITION_LIMITS

    @property
    def state(self) -> Literal["Inserted", "Retracted", "Busy", "Error", "Other"]:
        """Get the current state of the objective lens.

        Returns:
            The objective lens state (RetractableDeviceState) (e.g., 'Inserted', 'Retracted', 'Busy', 'Error', 'Other')
        """
        state = "Inserted" if self.position >= self._insert_position else "Retracted"
        return state


class Camera(ABC):
    """Abstract base class for camera control in fluorescence microscopy.

    Provides a standardized interface for camera operations including image acquisition,
    exposure control, binning, gain, and offset adjustments. Handles pixel size and
    resolution calculations with binning compensation.

    Attributes:
        parent: Reference to the parent fluorescence microscope
    """

    def __init__(self, parent: Optional["FluorescenceMicroscope"] = None):
        """Initialize the camera with default simulation parameters.

        Args:
            parent: Optional parent fluorescence microscope instance
        """
        self.parent = parent
        self._index: int = 0  # Image index for simulating sequential images
        self._use_counter: bool = True
        self._exposure_time: float = SIM_CAMERA_EXPOSURE_TIME
        self._binning: int = SIM_CAMERA_BINNING
        self._gain: float = SIM_CAMERA_GAIN
        self._offset: float = SIM_CAMERA_OFFSET
        self._pixel_size: Tuple[float, float] = SIM_CAMERA_PIXEL_SIZE
        self._resolution: Tuple[int, int] = SIM_CAMERA_RESOLUTION
        self._number_cache: dict = {}  # Cache for draw_number images by mod
        super().__init__()

    def acquire_image(self) -> np.ndarray:
        """Acquire a single image from the camera.

        Simulates camera acquisition by generating random noise with realistic
        timing based on exposure time settings. Accounts for current binning
        settings in the output resolution.

        Returns:
            A 16-bit numpy array representing the acquired image
        """
        sim_sleep(self.exposure_time)  # Simulate exposure time in seconds

        # get min and max values for the image
        noise = np.random.randint(
            UINT16_MIN, UINT16_MAX, size=self.resolution[::-1], dtype=np.uint16
        )
        if not self._use_counter:
            return noise

        # Simulate a simple image with a number drawn in the center
        mod = self._index % 10  # cycle through digits 0-9

        # Cache the draw_number image by mod and resolution
        cache_key = (mod, self.resolution)
        if cache_key not in self._number_cache:
            self._number_cache[cache_key] = draw_text(
                f"FM{mod}",
                size=(self.resolution[0] // 4, self.resolution[1] // 4),
                thickness=min(64, self.resolution[0] // 16),
                image_shape=self.resolution[::-1],
            )

        image = self._number_cache[cache_key]
        self._index += 1  # increment index for next image
        # use the image as an inverse mask for the noise
        data = np.where(image > 0, image, noise)
        return data

    @property
    def exposure_time(self) -> float:
        """Get the current exposure time of the camera.

        Returns:
            The exposure time in seconds
        """
        return self._exposure_time

    @exposure_time.setter
    def exposure_time(self, value: float):
        """Set the exposure time of the camera.

        Args:
            value: The exposure time in seconds (must be positive)
        """
        self._exposure_time = value

    @property
    def binning(self) -> int:
        """Get the current binning setting of the camera.

        Returns:
            The binning factor (1 = no binning, 2 = 2x2 binning, etc.)
        """
        return self._binning

    @binning.setter
    def binning(self, value: int):
        """Set the binning of the camera.

        Args:
            value: The binning factor (must be in available_binnings)

        Raises:
            ValueError: If the binning value is not supported
        """
        if value not in self.available_binnings:
            raise ValueError(
                f"Binning must be one of {self.available_binnings}, got {value}"
            )
        self._binning = value

    @property
    def available_binnings(self) -> Tuple[int, ...]:
        """Get the supported binning values for the camera.

        Returns:
            A tuple of supported binning factors (e.g., (1, 2, 4, 8))
        """
        return tuple(BINNING_VALUES)

    @property
    def exposure_time_limits(self) -> Tuple[float, float]:
        """Get the valid exposure time range for the camera.

        Returns:
            A tuple of (minimum, maximum) exposure times in seconds
        """
        return SIM_CAMERA_EXPOSURE_LIMITS

    @property
    def gain(self) -> float:
        """Get the current gain setting of the camera.

        Returns:
            The gain value (amplification factor)
        """
        return self._gain

    @gain.setter
    def gain(self, value: float):
        """Set the gain of the camera.

        Args:
            value: The gain value (must be non-negative)

        Raises:
            ValueError: If gain is negative
        """
        if value < 0:
            raise ValueError("Gain must be non-negative.")
        self._gain = value

    @property
    def offset(self) -> float:
        """Get the current offset setting of the camera.

        Returns:
            The offset value (baseline signal level)
        """
        return self._offset

    @offset.setter
    def offset(self, value: float):
        """Set the offset of the camera.

        Args:
            value: The offset value (must be non-negative)

        Raises:
            ValueError: If offset is negative
        """
        if value < 0:
            raise ValueError("Offset must be non-negative.")
        self._offset = value

    @property
    def pixel_size(self) -> Tuple[float, float]:
        """Get the effective pixel size accounting for current binning.

        Returns:
            A tuple of (x, y) pixel sizes in meters, scaled by binning factor
        """
        return (
            self._pixel_size[0] * self.binning,
            self._pixel_size[1] * self.binning,
        )

    @property
    def resolution(self) -> Tuple[int, int]:
        """Get the effective image resolution accounting for current binning.

        Returns:
            A tuple of (width, height) in pixels, reduced by binning factor
        """
        return self._resolution[0] // self.binning, self._resolution[1] // self.binning

    @property
    def field_of_view(self) -> Tuple[float, float]:
        """Get the effective field of view in meters accounting for binning.

        Returns:
            A tuple of (width, height) in meters
        """
        return (
            self.pixel_size[0] * self.resolution[0],
            self.pixel_size[1] * self.resolution[1],
        )


class LightSource(ABC):
    """Abstract base class for light source control in fluorescence microscopy.

    Provides a standardized interface for controlling illumination power across
    different light source implementations (LEDs, lasers, arc lamps, etc.).

    Attributes:
        parent: Reference to the parent fluorescence microscope
    """

    def __init__(self, parent: Optional["FluorescenceMicroscope"] = None):
        """Initialize the light source with default simulation parameters.

        Args:
            parent: Optional parent fluorescence microscope instance
        """
        self.parent = parent
        self._power: float = 0.1  # W
        super().__init__()

    @property
    def power(self) -> float:
        """Get the current power output of the light source.

        Returns:
            The power level in watts
        """
        return self._power

    @power.setter
    def power(self, value: float):
        """Set the power output of the light source.

        Args:
            value: The power level in watts (should be non-negative)
        """
        self._power = value

    @property
    def power_limits(self) -> Tuple[float, float]:
        """Get the valid power range for the light source.

        Power is expressed as a fraction of maximum power (0-1) across all
        drivers; hardware units are normalised inside each driver.

        Returns:
            A tuple of (minimum, maximum) power levels
        """
        return (0.0, 1.0)


class FilterSet(ABC):
    """Abstract base class for filter set control in fluorescence microscopy.

    Manages excitation and emission wavelength selection for fluorescence imaging.
    Provides standardized wavelength options and supports reflection mode (None emission).

    Attributes:
        parent: Reference to the parent fluorescence microscope
    """

    def __init__(self, parent: Optional["FluorescenceMicroscope"] = None):
        """Initialize the filter set with default wavelength settings.

        Args:
            parent: Optional parent fluorescence microscope instance
        """
        self.parent = parent
        self._excitation_wavelength: float = EXCITATION_WAVELENGTHS[0]
        self._emission_wavelength: Optional[Union[float, str]] = None
        super().__init__()

    @property
    def available_excitation_wavelengths(self) -> Tuple[float, ...]:
        """Get the available excitation wavelengths for the filter set.

        Returns:
            A tuple of supported excitation wavelengths in nanometers
        """
        return EXCITATION_WAVELENGTHS

    @property
    def available_emission_wavelengths(self) -> Tuple[Union[None, str, float], ...]:
        """Get the available emission wavelengths for the filter set.

        Returns:
            A tuple of supported emission wavelengths in nanometers.
            Includes None for reflection/pass-through mode.
        """
        return EMISSION_WAVELENGTHS

    @property
    def excitation_wavelength(self) -> float:
        """Get the current excitation wavelength of the filter set.

        Returns:
            The excitation wavelength in nanometers
        """
        return self._excitation_wavelength

    @excitation_wavelength.setter
    def excitation_wavelength(self, value: float):
        """Set the excitation wavelength of the filter set.

        Args:
            value: The desired excitation wavelength in nanometers
        """
        self._excitation_wavelength = value

    @property
    def emission_wavelength(self) -> Optional[Union[float, str]]:
        """Get the current emission wavelength of the filter set.

        Returns:
            The emission wavelength in nanometers, or None for reflection mode
        """
        return self._emission_wavelength

    @emission_wavelength.setter
    def emission_wavelength(self, value: Optional[Union[float, str]]):
        """Set the emission wavelength of the filter set.

        Args:
            value: The desired emission wavelength in nanometers, or None
                   for reflection/pass-through mode
        """
        self._emission_wavelength = value


class FluorescenceMicroscope(ABC):
    """Abstract base class for fluorescence microscope control.

    Provides a unified interface for controlling all aspects of fluorescence microscopy
    including objective lens, camera, light source, and filter sets. Supports both
    single image acquisition and live/continuous acquisition modes.

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
    # `workflow_update_signal`), and tile progress belongs on `tiled_acquisition_signal`
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

    def __init__(self, parent: Optional["FibsemMicroscope"] = None):
        """Initialize the fluorescence microscope with default components.

        Args:
            parent: Optional parent FibsemMicroscope instance for stage access
        """
        super().__init__()

        self.parent = parent

        # per-instance acquisition state (previously shared class attributes)
        self._stop_acquisition_event = threading.Event()
        self._acquisition_thread: Optional[threading.Thread] = None
        # Something other than the live stream is driving the FM: an overview tileset,
        # a z-stack, an autofocus sweep. Set by whoever is driving it. The stream is not
        # in here -- it reports itself through `_acquisition_thread`. See `is_acquiring`.
        self._acquiring: bool = False
        self._acquiring_reason: str = ""

        self.channel_name: str = "channel-01"
        self.channel_color: str = "gray"
        self.objective = ObjectiveLens(parent=self)
        self.filter_set = FilterSet(parent=self)
        self.camera = Camera(parent=self)
        self.light_source = LightSource(parent=self)
        self._last_updated_at: Optional[datetime] = datetime.now()
        self._rate_limit = RATE_LIMIT_DEFAULT  # seconds between updates
        self._transform: Optional[CameraImageTransform] = (
            CameraImageTransform.NONE
        )  # image transformation
        self.valid_orientations: list[str] = [
            "FM",
            "SEM",
            "MILLING",
        ]  # valid orientations for fluorescence acquisition
        self._allow_unknown_orientations: bool = ALLOW_UNKNOWN_ORIENTATIONS
        self.default_orientation: str = (
            "FM"  # orientation used when computing fluorescence pose for new lamellas
        )
        # Orientations the objective can actually image the sample from -- a stricter
        # question than `valid_orientations`, which asks only whether FM control is
        # allowed. Turning the light on and watching the camera from a beam pose is
        # harmless and sometimes useful, so `valid_orientations` includes SEM and
        # MILLING; driving the stage across a grid and stitching the result from one is
        # not, since on a compustage the sample is flipped away from the objective there
        # and on an offset mount it is translated out from under it.
        #
        # A list rather than `default_orientation` alone: a system whose objective sees
        # the sample from more than one pose says so here, and has somewhere to say it.
        self.acquisition_orientations: list[str] = [self.default_orientation]

    def has_valid_orientation(
        self, stage_position: Optional["FibsemStagePosition"] = None
    ) -> bool:
        """Return True if the current (or given) stage orientation is allowed for FM acquisition."""
        if self._allow_unknown_orientations:
            return True
        orientation = self.parent.get_stage_orientation(stage_position)
        return orientation in self.valid_orientations

    def is_acquisition_orientation(
        self, stage_position: Optional["FibsemStagePosition"] = None
    ) -> bool:
        """Return True if the objective can image the sample at the current (or given) pose.

        The question anything that drives the stage should ask, and stricter than
        `has_valid_orientation` in two ways: it asks the narrower
        `acquisition_orientations`, and it has no `ALLOW_UNKNOWN_ORIENTATIONS` escape
        hatch. The hatch exists so live FM control works from anywhere while a system is
        being set up; an unrecognised pose is not an inconvenience to a tileset, which
        would walk the stage across a grid and stitch the result against a frame nobody
        has checked.

        Answers True unconditionally on an offset mount, where the question cannot be
        asked: `_update_orientations` gives a non-compustage system an FM orientation
        copied from its FIB one, so `get_stage_orientation` at the fluorescence position
        returns a beam orientation -- measured, "MILLING", since the milling tilt band
        matches first. It cannot tell a fluorescence position from a beam one at all,
        which is the same limitation `build_lamella_poses` refuses over (FIB-93).
        Refusing on a question with no answer is not a guard, it is a lockout: it would
        leave the overview tab permanently dead on every METEOR.
        """
        if not self.parent.stage_is_compustage:
            return True
        orientation = self.parent.get_stage_orientation(stage_position)
        return orientation in self.acquisition_orientations

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
        if not self._acquisition_thread:
            return False
        return self._acquisition_thread and self._acquisition_thread.is_alive()

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
        if self.parent.stage_is_compustage:
            return 180.0
        return self.parent.system.ion.column_tilt

    @property
    def mount_transform(self) -> CameraImageTransform:
        """Fixed correction from raw sensor axes to stage-aligned axes.

        Hardware truth about how the camera is mounted, not a user preference: it
        is applied before the user's ``CameraImageTransform`` so that every
        consumer (display, correlation, saved data, movement) sees one consistently
        oriented image, and so that movement needs only the user transform.

        Defaults to no correction; drivers override per system. The value is
        determined by observing which stage axis a feature travels along in the
        FM view (see docs/design/fm-stable-move.md).
        """
        return CameraImageTransform.NONE

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
        """Acquire a single fluorescence image.

        Args:
            channel_settings: Optional channel configuration. If provided,
                            the microscope will be reconfigured before acquisition.

        Returns:
            A FluorescenceImage object containing the image data and metadata
        """
        with self.active_channel():
            if channel_settings is not None:
                self.set_channel(channel_settings)
            data = self.camera.acquire_image()
            # Inside the scope, not after it. `_construct_image` looks like formatting
            # but calls `get_metadata`, which reads 14 device properties that each take
            # the channel themselves -- outside, that is 56 round trips and 28 changes
            # of the microscope's active view per image, and the metadata would then
            # describe the state *after* the channel had been handed back rather than
            # the one the frame was taken under.
            return self._construct_image(data)

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
        md = self.get_metadata()

        if frame_metadata:
            pixel_size = frame_metadata.get("pixel_size")
            if pixel_size is not None:
                md.pixel_size_x, md.pixel_size_y = pixel_size[0], pixel_size[1]
            acquisition_date = frame_metadata.get("acquisition_date")
            if acquisition_date is not None:
                md.acquisition_date = acquisition_date
            exposure_time = frame_metadata.get("exposure_time")
            if exposure_time is not None and md.channels:
                md.channels[0].exposure_time = exposure_time

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
        return FluorescenceImageMetadata(
            acquisition_date=datetime.now().isoformat(),
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
        if self.is_streaming:
            logging.warning("Acquisition thread is already running.")
            return

        # reset stop event if needed
        self._stop_acquisition_event.clear()

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

    def _acquisition_worker(self, channel_settings: Optional[ChannelSettings] = None):
        """Internal worker thread for continuous image acquisition.

        Runs in a separate thread to continuously acquire images and emit them
        via the acquisition_signal until stop_acquisition() is called.

        Args:
            channel_settings: Optional channel configuration to apply

        Note:
            This is an internal method and should not be called directly.
            Use start_acquisition() instead.
        """
        # TODO: add thread lock for thread safety
        try:
            if channel_settings is not None:
                self.set_channel(channel_settings)
            logging.info("Starting acquisition worker thread.")
            while True:
                if self._stop_acquisition_event.is_set():
                    break

                if hasattr(self.camera, "_start_fast_acquisition"):
                    self.camera._start_fast_acquisition()  # type: ignore
                    break

                # acquire and emit image using current settings
                self.acquire_image()

        except Exception as e:
            logging.error(f"Error in acquisition worker: {e}")
