from __future__ import annotations

import copy
import glob
import logging
import os
import random
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import cycle
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from skimage.transform import resize

from fibsem._timing import sim_sleep
from fibsem.fm.microscope import (
    Camera,
    FilterSet,
    FluorescenceMicroscope,
    LightSource,
    ObjectiveLens,
)
from fibsem.fm.structures import (
    CameraImageTransform,
    ChannelSettings,
    EmissionFilter,
    FluorescenceImage,
    FluorescenceImageMetadata,
    ObjectiveStateName,
    emission_filter_for,
)
from fibsem.microscope import (
    FibsemMicroscope,
    _records_beam_shift,
    _records_stage_move,
)
from fibsem.microscopes.autoscript import match_application_file
from fibsem.microscopes.sim_scene import fm_channel_weights
from fibsem.milling.progress import MillingProgress, MillingProgressStatus
from fibsem.projection import FMStageProjection
from fibsem.structures import (
    ACTIVE_MILLING_STATES,
    BeamSettings,
    BeamType,
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemDetectorSettings,
    FibsemExperimentRef,
    FibsemImage,
    FibsemImageMetadata,
    FibsemLineSettings,
    FibsemManipulatorPosition,
    FibsemMillingSettings,
    FibsemPatternSettings,
    FibsemPolygonSettings,
    FibsemRectangle,
    FibsemRectangleSettings,
    FibsemStagePosition,
    FibsemUser,
    ImageSettings,
    MicroscopeState,
    MillingState,
    Point,
    RangeLimit,
    SystemSettings,
)
from fibsem.util.draw_numbers import draw_text

if TYPE_CHECKING:
    from fibsem.microscopes._stage import DemoSampleLoader

######################## SIMULATOR ########################

SIMULATOR_KNOWN_UNKNOWN_KEYS = ["preset"]

# simulator constants
SIMULATOR_PLASMA_GASES = ["Oxygen", "Argon", "Nitrogen", "Xenon"]
SIMULATOR_SCAN_DIRECTIONS = ["BottomToTop", "LeftToRight", "RightToLeft", "TopToBottom"]
SIMULATOR_APPLICATION_FILES = [
    "Si",
    "Si-multipass",
    "Si-ccs",
    "autolamella",
    "cryo_Pt_dep",
]
SIMULATOR_BEAM_CURRENTS = {
    BeamType.ELECTRON: [
        1.0e-12,
        3.0e-12,
        10e-12,
        30e-12,
        0.1e-9,
        0.3e-9,
        1e-9,
        4e-9,
        15e-9,
        60e-9,
    ],
    BeamType.ION: {
        "Xenon": [
            1.0e-12,
            3.0e-12,
            10e-12,
            30e-12,
            0.1e-9,
            0.3e-9,
            1.00005e-9,
            4e-9,
            15e-9,
            60e-9,
        ],
        "Argon": [
            1.0e-12,
            6.0e-12,
            20e-12,
            60e-12,
            0.2e-9,
            0.74e-9,
            2.0e-9,
            7.4e-9,
            28.0e-9,
            120.0e-9,
        ],
        None: [
            1.0e-12,
            3.0e-12,
            20e-12,
            41e-12,
            90e-12,
            0.2e-9,
            0.4e-9,
            1.000005e-9,
            2.0e-9,
            4.0e-9,
            15e-9,
        ],  # None = Gallium
    },
}

# How long a stage move is pretended to take, in seconds.
#
# The simulator already models how long an *acquisition* takes -- `acquire_image` sleeps
# the dwell time over the frame -- and modelled a stage move as instantaneous, which is
# not a smaller error than it sounds. Anything the UI says while the stage is moving was
# unreachable here: the message went up and came down inside one event loop iteration,
# so click-to-move feedback, the moving/arrived transition and any control disabled for
# the duration could not be seen at all, let alone judged (FIB-765).
#
# One second is at the short end of a real move and deliberately so. It goes through
# `sim_sleep`, so it is a no-op under `FIBSEM_SIM_NO_DELAY=1`, which the test suite sets.
#
# It does lengthen a simulated tileset: every tile is a `safe_absolute_stage_movement`
# (`imaging/tiled.py`), and on a stage that rotates that is up to three calls to the
# primitives below. That is the honest shape -- a real safe move really does tilt flat,
# rotate, then move -- but it is the thing to turn down here if the simulator starts
# feeling slow for grid work.
STAGE_MOVEMENT_SLEEP_TIME = 1.0

# An asynchronous mill (`start_milling`) has no end on the simulator: it runs until
# stopped. Its estimate says so, long enough to watch a run that is timed by the
# estimate -- coincidence milling stops when its estimate runs out (FIB-1119).
SIM_ASYNC_MILLING_EXTRA_TIME = 300  # seconds

STAGE_LIMITS_DEFAULT = {
    "x": RangeLimit(min=-100.0e-3, max=100.0e-3),
    "y": RangeLimit(min=-100.0e-3, max=100.0e-3),
    "z": RangeLimit(min=0.0e-3, max=40.0e-3),
    "r": RangeLimit(min=-360.0, max=360.0),
    "t": RangeLimit(min=-10.0, max=90.0),
}
STAGE_LIMITS_COMPUSTAGE = {
    "x": RangeLimit(min=-999.9e-6, max=999.9e-6),
    "y": RangeLimit(min=-377.8e-6, max=377.8e-6),
    "z": RangeLimit(min=-999.9e-6, max=999.9e-6),
    "t": RangeLimit(min=-195.0, max=15.0),
}
# hack, do this properly @patrick


@dataclass
class DemoMicroscopeClient:
    connected: bool = False

    def connect(self, ip_address: str, port: int = 8080):
        logging.debug(f"Connecting to microscope at {ip_address}:{port}")
        self.connected = True
        logging.debug(f"Connected to microscope at {ip_address}:{port}")

    def disconnect(self):
        self.connected = False


@dataclass
class BeamSystem:
    on: bool
    blanked: bool
    beam: BeamSettings
    detector: FibsemDetectorSettings
    scanning_mode: str
    scanning_mode_value: Union[None, Point, FibsemRectangle] = None


@dataclass
class ChamberSystem:
    state: str
    pressure: float


@dataclass
class StageSystem:
    is_homed: bool
    is_linked: bool
    position: FibsemStagePosition


@dataclass
class ManipulatorSystem:
    inserted: bool
    position: FibsemManipulatorPosition


@dataclass
class MillingSystem:
    state: MillingState = MillingState.IDLE
    patterns: List[FibsemPatternSettings] = field(default_factory=list)
    patterning_mode: str = "Serial"
    default_beam_type: BeamType = BeamType.ION
    default_application_file: str = "Si"
    application_files: List[str] = field(
        default_factory=lambda: SIMULATOR_APPLICATION_FILES
    )


@dataclass
class ImagingSystem:
    active_view: int = BeamType.ELECTRON.value
    active_device: int = BeamType.ELECTRON.value
    last_image: Dict[BeamType, Optional[FibsemImage]] = field(default_factory=dict)
    image_iterators: Dict[BeamType, Iterator[str]] = field(
        default_factory=dict
    )  # Image filename iterators


# The FM's place on the shared imaging channel. The view number is the driver's
# (`FM_ACTIVE_VIEW = 3` in `fibsem.devices.drivers.autoscript_fm`, quadrant 3 on an
# Arctis); all that matters here is that it is neither beam's, so a beam operation can
# tell that the FM took the channel out from under it.
FM_ACTIVE_VIEW = 3
FM_ACTIVE_DEVICE = 3

# The simulated FM: what its parts offer and where they start. The Demo FM devices
# (``fibsem.devices.drivers.demo``) start from the same values.
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
SIM_LIGHT_SOURCE_POWER = 0.1  # a fraction of full power

UINT16_MIN = np.iinfo(np.uint16).min  # 0 for uint16
UINT16_MAX = np.iinfo(np.uint16).max  # 65535 for uint16
BINNING_VALUES = [1, 2, 4, 8]  # typical binning values

# The chamber camera's place on the same channel, matching the numbers
# `ThermoMicroscope.acquire_chamber_image` writes.
CHAMBER_ACTIVE_VIEW = 4
CHAMBER_ACTIVE_DEVICE = 3

# The detector keys, which read and write whichever detector is on the active device.
DETECTOR_KEYS = (
    "detector_type",
    "detector_mode",
    "detector_brightness",
    "detector_contrast",
)


def render_fm_scene(
    fm: FluorescenceMicroscope,
    exposure_time: float,
    pixel_size: float,
    resolution: Tuple[int, int],
) -> Optional[np.ndarray]:
    """A simulated FM camera frame of the sample scene, or None without a scene.

    Renders the same synthetic sample the beams image, through the FM's projection,
    for the channel ``fm`` is set to and with the objective's defocus, at the binned
    ``resolution`` (width, height) and ``pixel_size`` the camera has. Shared by
    ``SceneCamera`` and the Demo FM camera device.
    """
    microscope = getattr(fm, "parent", None)
    scene = getattr(microscope, "_sample_scene", None)
    if scene is None:
        return None
    projection = FMStageProjection.from_microscope(microscope)
    if projection is None:
        return None
    sim_sleep(exposure_time)
    weights = fm_channel_weights(
        fm.filter_set.emission_wavelength, fm.filter_set.excitation_wavelength
    )
    focus = fm.objective.focus_position
    defocus = 0.0 if focus is None else fm.objective.position - focus
    # the projection carries the unbinned camera shape; render at the
    # binned resolution with the matching pixel size
    projection = FMStageProjection(
        geometry=projection.geometry,
        pixel_size=pixel_size,
        shape=(resolution[1], resolution[0]),
    )
    scene.holder_slots = microscope._scene_holder_slots()
    frame = scene.render_fm(
        microscope.get_stage_position(),
        resolution,
        projection,
        weights=weights,
        defocus=defocus,
    )
    # the projection speaks in displayed-image coordinates, but a camera
    # frame goes through the mount and user transforms before display:
    # pre-apply their inverse (flips are self-inverse; reversed order)
    frame = fm._transform_array(frame, fm._transform)
    return fm._transform_array(frame, fm.mount_transform)


class SimulatedObjectiveLens(ObjectiveLens):
    """The simulated FM's objective: a position that moves when told, clipped to the
    user-defined limit, with seconds of travel to insert or retract."""

    def __init__(self, parent: Optional[FluorescenceMicroscope] = None):
        super().__init__(parent=parent)
        self._position: float = SIM_OBJECTIVE_RETRACT_POSITION  # initial position
        self._magnification: float = SIM_OBJECTIVE_MAGNIFICATION
        self._numerical_aperture = SIM_OBJECTIVE_NA
        self._insert_position = SIM_OBJECTIVE_INSERT_POSITION
        self._retract_position = SIM_OBJECTIVE_RETRACT_POSITION
        self._focus_position: Optional[float] = SIM_OBJECTIVE_FOCUS_POSITION
        self._limit_position: float = SIM_OBJECTIVE_USER_POSITION_LIMIT

    @property
    def magnification(self) -> float:
        return self._magnification

    @property
    def numerical_aperture(self) -> float:
        return self._numerical_aperture

    @property
    def position(self) -> float:
        sim_sleep(0.1)
        return self._position

    @property
    def limit_position(self) -> float:
        return self._limit_position

    @limit_position.setter
    def limit_position(self, position: float):
        self._limit_position = position
        logging.info(
            f"Objective user-defined position limit set to: {self._limit_position * 1e3:.3f} mm"
        )

    def move_relative(self, delta: float):
        self._position += delta
        logging.info(
            f"Objective moved to new position: {self._position * 1e3:.3f} mm (delta: {delta * 1e3:.3f} mm)"
        )
        # Announced here rather than relying on `move_absolute`: this implementation
        # adjusts the field itself instead of delegating.
        self._notify_moved()

    def move_absolute(self, position: float):
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
        sim_sleep(SIM_OBJECTIVE_TRAVEL_SECONDS)  # the traverse, on top of the move
        self.move_absolute(self._insert_position)
        logging.info(
            f"Objective lens inserted to position: {self._insert_position:.3f} mm"
        )

    def retract(self):
        sim_sleep(SIM_OBJECTIVE_TRAVEL_SECONDS)
        self.move_absolute(self._retract_position)
        logging.info(
            f"Objective lens retracted to position: {self._retract_position:.3f} mm"
        )

    @property
    def limits(self) -> Tuple[float, float]:
        return SIM_OBJECTIVE_POSITION_LIMITS

    @property
    def state(self) -> ObjectiveStateName:
        return "Inserted" if self.position >= self._insert_position else "Retracted"


class SimulatedCamera(Camera):
    """The simulated FM's camera: a noise frame with a numbered "FM<n>" in it, taking
    the exposure time to arrive."""

    def __init__(self, parent: Optional[FluorescenceMicroscope] = None):
        super().__init__(parent=parent)
        self._index: int = 0  # Image index for simulating sequential images
        self._use_counter: bool = True
        self._exposure_time: float = SIM_CAMERA_EXPOSURE_TIME
        self._binning: int = SIM_CAMERA_BINNING
        self._gain: float = SIM_CAMERA_GAIN
        self._offset: float = SIM_CAMERA_OFFSET
        self._pixel_size: Tuple[float, float] = SIM_CAMERA_PIXEL_SIZE
        self._resolution: Tuple[int, int] = SIM_CAMERA_RESOLUTION
        self._number_cache: dict = {}  # Cache for draw_number images by mod

    def acquire_image(self) -> np.ndarray:
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
        return self._exposure_time

    @exposure_time.setter
    def exposure_time(self, value: float):
        self._exposure_time = value

    @property
    def binning(self) -> int:
        return self._binning

    @binning.setter
    def binning(self, value: int):
        if value not in self.available_binnings:
            raise ValueError(
                f"Binning must be one of {self.available_binnings}, got {value}"
            )
        self._binning = value

    @property
    def available_binnings(self) -> Tuple[int, ...]:
        return tuple(BINNING_VALUES)

    @property
    def exposure_time_limits(self) -> Tuple[float, float]:
        return SIM_CAMERA_EXPOSURE_LIMITS

    @property
    def gain(self) -> float:
        return self._gain

    @gain.setter
    def gain(self, value: float):
        if value < 0:
            raise ValueError("Gain must be non-negative.")
        self._gain = value

    @property
    def gain_native_scale(self) -> Optional[Tuple[float, Optional[str]]]:
        return None

    @property
    def offset(self) -> float:
        return self._offset

    @offset.setter
    def offset(self, value: float):
        if value < 0:
            raise ValueError("Offset must be non-negative.")
        self._offset = value

    @property
    def pixel_size(self) -> Tuple[float, float]:
        return (
            self._pixel_size[0] * self.binning,
            self._pixel_size[1] * self.binning,
        )

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._resolution[0] // self.binning, self._resolution[1] // self.binning


class SimulatedLightSource(LightSource):
    """The simulated FM's light source: one power, unchecked."""

    def __init__(self, parent: Optional[FluorescenceMicroscope] = None):
        super().__init__(parent=parent)
        self._power: float = SIM_LIGHT_SOURCE_POWER

    @property
    def power(self) -> float:
        return self._power

    @power.setter
    def power(self, value: float):
        self._power = value

    @property
    def power_limits(self) -> Tuple[float, float]:
        return (0.0, 1.0)

    @property
    def power_native_scale(self) -> Optional[Tuple[float, Optional[str]]]:
        return None


class SimulatedFilterSet(FilterSet):
    """The simulated FM's filter set: ``EXCITATION_WAVELENGTHS``, and reflection or
    one multi-band emission filter (``EMISSION_WAVELENGTHS``)."""

    def __init__(self, parent: Optional[FluorescenceMicroscope] = None):
        super().__init__(parent=parent)
        self._excitation_wavelength: float = EXCITATION_WAVELENGTHS[0]
        self._emission_wavelength: Optional[Union[float, str]] = None

    @property
    def available_excitation_wavelengths(self) -> Tuple[float, ...]:
        return EXCITATION_WAVELENGTHS

    @property
    def available_emission_wavelengths(self) -> Tuple[Union[None, str, float], ...]:
        return EMISSION_WAVELENGTHS

    @property
    def excitation_wavelength(self) -> float:
        return self._excitation_wavelength

    @excitation_wavelength.setter
    def excitation_wavelength(self, value: float):
        self._excitation_wavelength = value

    @property
    def emission_wavelength(self) -> Optional[Union[float, str]]:
        return self._emission_wavelength

    @emission_wavelength.setter
    def emission_wavelength(self, value: Optional[Union[float, str]]):
        self._emission_wavelength = value

    def emission_filter(self, value: Optional[Union[float, str]]) -> EmissionFilter:
        """The filter an emission value names, with its band's edges when this filter
        set knows them (``emission_bands``), for showing it by name and band."""
        return emission_filter_for(value, getattr(self, "emission_bands", {}))


class SceneCamera(SimulatedCamera):
    """The simulated FM camera, imaging the sample scene when there is one.

    Renders the same synthetic sample the beams image (``render_fm_scene``);
    falls back to the stock noise/counter frames when no scene is enabled.
    """

    def acquire_image(self) -> np.ndarray:
        frame = render_fm_scene(
            self.parent, self.exposure_time, self.pixel_size[0], self.resolution
        )
        if frame is None:
            return super().acquire_image()
        self._index += 1
        return frame


class SimulatedFluorescenceMicroscope(FluorescenceMicroscope):
    """The simulated FM: the legacy Demo's, and a stand-in FM wherever one is needed
    without hardware (tests, the FM widgets run on their own). Its parts are the
    ``Simulated*`` ones above, and the camera images the sample scene when its
    microscope has one (``SceneCamera``).

    It is also the FM half of the one imaging channel a TFS system shares (FIB-518).

    `DemoMicroscope` simulates a TFS system, where the FM and the beams are one
    connection with one active view and one active device: whoever writes last owns the
    microscope. The base class's `active_channel()` is a no-op -- right for a system
    whose FM has a connection of its own, and the reason the simulator could show none
    of FIB-517/542/544/545. Every one of those was found on hardware instead, one of
    them by a workflow task stopping.

    So this participates in `parent.imaging_system` the way the Thermo FM devices'
    channel (`AutoscriptFMChannel`) participates in the shared connection: same depth
    count, same lock, same restore. Deliberately a mirror rather than an
    approximation, so a test written against the simulator says something about the
    hardware.
    """

    def __init__(self, parent: Optional["FibsemMicroscope"] = None):
        super().__init__(parent=parent)
        self.objective = SimulatedObjectiveLens(parent=self)
        self.filter_set = SimulatedFilterSet(parent=self)
        self.camera = SceneCamera(parent=self)
        self.light_source = SimulatedLightSource(parent=self)
        self._active_view = FM_ACTIVE_VIEW
        self._active_device = FM_ACTIVE_DEVICE
        # The parent's lock, as the driver takes it: an FM scope and a beam acquisition
        # then queue against each other rather than interleaving, which is what makes
        # the scoped and unscoped paths behave differently here as they do on hardware.
        self._channel_lock = (
            getattr(parent, "_threading_lock", None) or threading.RLock()
        )
        self._channel_depth = 0
        self._restore_view: Optional[int] = None
        self._restore_device: Optional[int] = None

    # The simulated FM has no devices: it takes frames from its own camera, as the FM
    # API did before it ran over devices.

    @property
    def runs_z_stack_on_device(self) -> bool:
        return False

    @property
    def mount_transform(self) -> CameraImageTransform:
        """Fixed correction from raw sensor axes to stage-aligned axes.

        Hardware truth about how the camera is mounted, not a user preference: it
        is applied before the user's ``CameraImageTransform`` so that every
        consumer (display, correlation, saved data, movement) sees one consistently
        oriented image, and so that movement needs only the user transform.

        Defaults to no correction; drivers override per system. The value is
        determined by observing which stage axis a feature travels along in the
        FM view.
        """
        return CameraImageTransform.NONE

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

    def _metadata_for_frame(
        self, frame_metadata: Optional[dict]
    ) -> FluorescenceImageMetadata:
        """The image's metadata: the current state, with what the driver reported for
        the frame itself in place of it."""
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
        return md

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

    def _shares_a_channel(self) -> bool:
        """Whether there is a channel to share: a microscope with an imaging system.
        On its own, or on a microscope that isn't a demo, there is none."""
        return getattr(self.parent, "imaging_system", None) is not None

    def set_active_channel(self) -> None:
        """Point the shared channel at the FM and leave it there.

        The unscoped form, and the shape of the bug: a property getter that calls this
        and walks away leaves the microscope on the FM, and the next beam operation to
        read a buffer reads the FM's. Mirrors
        `AutoscriptFMChannel.set_active_channel`.
        """
        if not self._shares_a_channel():
            return
        self.parent.imaging_system.active_view = self._active_view
        self.parent.imaging_system.active_device = self._active_device

    def _channel_is_ours(self) -> bool:
        """Whether the shared channel is already pointed at the FM.

        The view alone answers it, matching the driver, where the device follows the
        view. Read outside the lock on purpose: taking the lock to find out whether the
        lock is needed would defeat the point of asking.
        """
        return self.parent.imaging_system.active_view == self._active_view

    @contextmanager
    def active_channel(self):
        """Hold the channel on the FM for the length of the block, then put it back.

        Depth counted, so a tileset that holds it for a whole run is not undone by each
        tile's acquisition restoring between frames; the lock covers the bookkeeping and
        never the body, since the body can be that whole run. Both rules are the
        driver's -- see `AutoscriptFMChannel.scope` for why.

        Restores the device alongside the view, where the driver restores the view
        alone. Not a divergence: on hardware `set_active_device` changes the device *in
        the active view*, so the view owns it and it comes back with it. Here the two
        are independent fields, and putting only the view back would leave
        `active_device` reporting the FM for the rest of the session.

        Skips both the lock and the bookkeeping when the channel is already the FM, as
        the driver does -- there is nothing to set, so nothing to put back. Modelled
        here because that fast path is what makes the objective usable while streaming,
        and a simulator that always took the lock would let a change reintroduce the
        contention with every test still green.
        """
        if not self._shares_a_channel():
            yield
            return

        if self._channel_depth == 0 and self._channel_is_ours():
            yield
            return

        imaging = self.parent.imaging_system
        with self._channel_lock:
            if self._channel_depth == 0:
                self._restore_view = imaging.active_view
                self._restore_device = imaging.active_device
            self.set_active_channel()
            self._channel_depth += 1
        try:
            yield
        finally:
            with self._channel_lock:
                self._channel_depth -= 1
                if self._channel_depth == 0:
                    imaging.active_view = self._restore_view
                    imaging.active_device = self._restore_device


def _grid_stage_position(grid_position) -> FibsemStagePosition:
    """Where the simulated autoloader puts a grid, as a stage position at the
    working slot's pose (the SEM orientation, r = t = 0)."""
    x, y, z = (float(v) for v in grid_position)
    return FibsemStagePosition(name="Slot-01", x=x, y=y, z=z, r=0.0, t=0.0)


class DemoConfiguration:
    """What a demo configuration says the instrument has, shared by both demos.

    Everything here reads only ``system`` (its ``sim:`` block and ``ion``), never a
    simulated part (the beam keys' values go through ``get``), so it is the same whether
    the parts are Demo's or devices.
    """

    system: SystemSettings
    stage_is_compustage: bool

    # ---- fitted subsystems, as the simulated instrument reports them ---------
    #
    # The `sim:` block is where a simulated configuration stands in for a hardware
    # probe (`has_fm`, `is_compustage`), so that is where these come from too. Absent
    # means the Demo default -- fitted.

    def _probe_manipulator_installed(self) -> Optional[bool]:
        return self.system.sim.get("has_manipulator")

    def _probe_plasma_gas(self) -> Optional[str]:
        return self.system.sim.get("plasma_gas")

    def _get_axis_limits(self) -> Dict[str, RangeLimit]:
        """Get the axis limits for the stage."""
        if self.stage_is_compustage:
            return STAGE_LIMITS_COMPUSTAGE
        return STAGE_LIMITS_DEFAULT

    def _create_grid_loader(self) -> "DemoSampleLoader":
        """An in-memory autoloader, populated from the ``sim.loader`` block.

        Only reached on a compustage configuration (the Arctis simulator). Keys:
        ``capacity`` (default 12), ``occupied`` (1-based slot numbers), ``names``
        (slot number -> grid name), ``exchange_delay`` (seconds, default 0),
        ``start_unscanned`` (default false), ``scan_delay`` (seconds, default 0),
        ``grid_position`` ([x, y, z] metres from the stage origin where a loaded
        grid really sits; default none, the origin).
        """
        from fibsem.microscopes._stage import DemoSampleLoader

        cfg = self.system.sim.get("loader") or {}
        return DemoSampleLoader(
            parent=self,
            capacity=int(cfg.get("capacity", 12)),
            occupied=cfg.get("occupied") or (),
            names=cfg.get("names") or {},
            exchange_delay=float(cfg.get("exchange_delay", 0.0)),
            start_unscanned=bool(cfg.get("start_unscanned", False)),
            scan_delay=float(cfg.get("scan_delay", 0.0)),
            grid_position=cfg.get("grid_position") or None,
        )

    def _read_plasma(self, beam_type: Optional[BeamType]) -> bool:
        """Whether the ion column is a plasma one; an electron beam never is."""
        if beam_type is BeamType.ION:
            return self.system.ion.plasma
        return False

    def _configured_values(self, key: str) -> Optional[List[str]]:
        """The values of a key that come from the simulator's constants alone."""
        if key == "scan_direction":
            return SIMULATOR_SCAN_DIRECTIONS
        if key == "plasma_gas":
            return SIMULATOR_PLASMA_GASES
        return None

    def get_available_values(
        self, key: str, beam_type: Optional[BeamType] = None
    ) -> List[Union[str, int, float]]:
        """Get the available values for a given key."""
        values = []
        if key == "current":
            # return values based on beam type, and plasma gas
            if beam_type is BeamType.ION:
                plasma_gas = self.get("plasma_gas", beam_type)
                values = SIMULATOR_BEAM_CURRENTS[beam_type][plasma_gas]
            else:
                values = SIMULATOR_BEAM_CURRENTS[beam_type]

        if key == "voltage":
            if beam_type is BeamType.ELECTRON:
                # SEM: [1000, 2000, 3000, 5000, 10000, 20000, 30000]
                values = [2000, 5000, 10000, 20000, 30000]
            elif beam_type is BeamType.ION:
                values = [500, 1000, 2000, 8000, 16000, 30000]
                # FIB: [500, 1000, 2000, 8000, 1600, 30000]

        milling = self._milling_values(key)
        if milling is not None:
            values = milling

        if key == "detector_type":
            values = ["ETD", "TLD", "EDS"]
        if key == "detector_mode":
            values = ["SecondaryElectrons", "BackscatteredElectrons", "EDS"]

        configured = self._configured_values(key)
        if configured is not None:
            values = configured

        return values


class DemoImaging:
    """Imaging on a demo: the beams' frames, the chamber camera and the shared channel.

    Shared by both demos. It reads and changes the beams only through
    ``get``/``set``, so on the device-built Demo it images through the beam devices.
    Its own state is the imaging channel and last images (``imaging_system``), the
    image sequence and the sample scene, which the demo sets up at construction.
    """

    imaging_system: ImagingSystem

    def set_channel(self, beam_type: BeamType) -> None:
        self.imaging_system.active_view = beam_type.value
        self.imaging_system.active_device = beam_type.value

    def _warn_if_channel_moved(self, expected: BeamType, operation: str) -> bool:
        """Report a beam operation whose imaging channel was taken out from under it.

        Warns rather than raises. What this models is a *silent* hardware failure -- the
        grab returns whoever owns the view now, and the metadata is built from the
        settings that were asked for rather than from what came back -- so the value
        here is making it visible in a log and catchable in a test, not changing what
        the simulator returns. Raising would also take down the very callers this exists
        to help debug, and it would turn a diagnostic into a new failure mode that
        hardware does not have.

        Returns True when the channel was still the operation's own.
        """
        view = self.imaging_system.active_view
        if view == expected.value:
            return True
        logging.warning(
            "Imaging channel changed during %s: set to %s (view %d), found view %d. "
            "The FM and the beams share one channel on a TFS system, so the grab here "
            "would return the other view's buffer (FIB-517).",
            operation,
            expected.name,
            expected.value,
            view,
        )
        return False

    def acquire_image(
        self,
        image_settings: Optional[ImageSettings] = None,
        beam_type: Optional[BeamType] = None,
    ) -> FibsemImage:
        """
        Acquire a new image with the specified settings or current settings for the given beam type.

        Args:
            image_settings (ImageSettings, optional): The settings for the new image.
                Takes precedence if both parameters are provided.
            beam_type (BeamType, optional): The beam type to use with current settings.
                Used only if image_settings is not provided.

        Returns:
            FibsemImage: A new FibsemImage representing the acquired image.

        Raises:
            ValueError: If neither image_settings nor beam_type is provided.

        Examples:
            # Acquire with specific settings
            settings = ImageSettings(beam_type=BeamType.ELECTRON, hfw=1e-6, resolution=(1024, 1024))
            image = microscope.acquire_image(image_settings=settings)

            # Acquire with current settings for a specific beam type
            image = microscope.acquire_image(beam_type=BeamType.ION)

            # If both provided, image_settings takes precedence
            image = microscope.acquire_image(image_settings=settings, beam_type=BeamType.ION)  # Uses settings
        """
        # Validate parameters - at least one must be provided
        if image_settings is None and beam_type is None:
            raise ValueError(
                "Must provide either image_settings (to acquire with specific settings) or beam_type (to acquire with current microscope settings for that beam type)."
            )
        # The beam's acquire command, on a demo whose beams have it; settings win.
        target = image_settings.beam_type if image_settings is not None else beam_type
        beam = self._imaging_beam(target, "_acquire")
        if beam is not None:
            return beam.acquire(image_settings)
        return self._demo_acquire(image_settings, beam_type)

    def _imaging_beam(self, beam_type: Optional[BeamType], hook: str):
        """The beam device whose driver implements *hook*, or None for the code here.

        The device-built Demo's beams run imaging as commands, which call back into
        the ``_demo_*`` methods below; the legacy Demo has no beam devices.
        """
        from fibsem.devices.beam import implements

        beam = self.beams.get(beam_type) if beam_type is not None else None
        return beam if beam is not None and implements(beam, hook) else None

    def _demo_acquire(
        self,
        image_settings: Optional[ImageSettings],
        beam_type: Optional[BeamType],
    ) -> FibsemImage:
        """``acquire_image``'s frame: what both demos acquire, through the beam's
        command on the device-built one."""
        # Determine which beam type and settings to use (image_settings takes precedence)
        if image_settings is not None:
            # Use provided image settings
            effective_beam_type = image_settings.beam_type
            effective_image_settings = image_settings
        elif beam_type is not None:
            # Use current settings for the specified beam type
            effective_beam_type = beam_type
            effective_image_settings = self.get_imaging_settings(
                beam_type=effective_beam_type
            )

        logging.info(f"acquiring new {effective_beam_type.name} image.")

        # set the imaging hfw, as the hardware drivers do: an acquisition leaves
        # the beam at the field it imaged, so anything that follows in image
        # coordinates - a spot burn parked at a normalised point - lands where
        # the reference image says (FIB-954)
        self.set("hfw", effective_image_settings.hfw, effective_beam_type)

        # get state for image metadata
        microscope_state = self.get_microscope_state(beam_type=effective_beam_type)

        # One lock over the channel and the frame, as `ThermoMicroscope.acquire_image`
        # holds it over `set_channel` + `grab_frame` (FIB-542): the grab reads the
        # active view's buffer, so a channel that is not still ours when the frame lands
        # returns whoever took it in between. Deliberately just that pair --
        # `_threading_lock` is shared by every caller on this microscope.
        with self._threading_lock:
            self.set_channel(effective_beam_type)
            # The frame. On hardware this is the one `grab_frame` RPC; here the sleep is
            # the only part of it with any duration, so it is the window an unguarded FM
            # read would land in.
            sim_sleep(
                effective_image_settings.dwell_time
                * effective_image_settings.resolution[0]
                * effective_image_settings.resolution[1]
            )  # simulate acquisition time
            self._warn_if_channel_moved(effective_beam_type, "acquire_image")

        # construct image (random noise)
        image = FibsemImage.generate_blank_image(
            resolution=effective_image_settings.resolution,
            hfw=effective_image_settings.hfw,
            random=True,
        )

        # shared-scene projection takes precedence when enabled (FIB-874)
        if self._sample_scene is not None:
            from fibsem.projection import BeamStageProjection

            projection = BeamStageProjection.from_microscope(
                self, beam_type=effective_beam_type
            )
            shift = self.get_beam_shift(effective_beam_type)
            try:
                current = float(self.get("current", effective_beam_type))
            except Exception:
                current = None
            self._sample_scene.holder_slots = self._scene_holder_slots()
            image.data = self._sample_scene.render(
                beam_type=effective_beam_type,
                stage_position=self.get_stage_position(),
                hfw=effective_image_settings.hfw,
                resolution=effective_image_settings.resolution,
                projection=projection,
                beam_shift=(float(shift.x), float(shift.y)),
                beam_current=current,
            )
        # generate the next image from the sequence iterator
        elif self.use_image_sequence:
            image.data = self._generate_next_image(
                beam_type=effective_beam_type,
                output_shape=image.data.shape,
                dtype=image.data.dtype,
            )
        else:
            char = "SEM" if effective_beam_type == BeamType.ELECTRON else "FIB"
            resolution = (
                effective_image_settings.resolution[1],
                effective_image_settings.resolution[0],
            )
            cache_key = (char, resolution)
            if cache_key not in self._image_cache:
                self._image_cache[cache_key] = draw_text(
                    char, size=(256, 256), thickness=48, image_shape=resolution
                )

            num_image = self._image_cache[cache_key]
            # use the image as an inverse mask for the noise
            image.data = np.where(num_image > 0, num_image, image.data)

        # add additional metadata
        image.metadata.image_settings = copy.deepcopy(effective_image_settings)
        image.metadata.microscope_state = microscope_state
        self._set_additional_metadata(image)

        # crop the image data if reduced area is set
        if effective_image_settings.reduced_area is not None:
            rect = effective_image_settings.reduced_area
            width = int(rect.width * image.data.shape[1])
            height = int(rect.height * image.data.shape[0])

            x0 = int(rect.left * image.data.shape[1])
            y0 = int(rect.top * image.data.shape[0])
            x1, y1 = x0 + width, y0 + height
            image.data = image.data[y0:y1, x0:x1]

        # store last image, and imaging settings (only if image_settings was provided)
        self.imaging_system.last_image[effective_beam_type] = image
        if image_settings is not None:
            self._last_imaging_settings = image_settings

        logging.debug({"msg": "acquire_image", "metadata": image.metadata.to_dict()})

        return image

    def _generate_next_image(
        self,
        beam_type: BeamType,
        output_shape: Tuple[int, int],
        dtype: np.dtype = np.uint8,
    ) -> np.ndarray:
        """Generate the next image in the sequence for the specified beam type,
            formatted for the current acquisition settings.
        Args:
            beam_type: The type of beam (electron or ion).
            output_shape: The shape of the output image (height, width).
            dtype: The data type of the output image.
        Returns:
            np.ndarray: The generated image data.
        """
        try:
            # get the next filename from the imaging system
            image_iterator = self.imaging_system.image_iterators.get(beam_type, None)
            if image_iterator is None:
                raise ValueError(
                    f"No image iterator found for beam type {beam_type.name}"
                )

            filename = next(image_iterator)

            # check if file still exists
            if not os.path.exists(filename):
                logging.warning(
                    f"Image file not found: {filename}, falling back to random noise"
                )
                return np.random.randint(0, 256, output_shape, dtype=dtype)

            # load and process the image
            logging.debug(
                f"Generating image from {filename} for beam type {beam_type.name}"
            )
            img = FibsemImage.load(filename)

            # resize the image data to the specified resolution
            image_data = resize(
                img.data,
                output_shape=output_shape,
                anti_aliasing=True,
                preserve_range=True,
            )
            return image_data.astype(dtype)
        except StopIteration as e:
            logging.debug(
                f"Image sequence for {beam_type.name} exhausted, falling back to random noise: {e}"
            )
            return np.random.randint(0, 256, output_shape, dtype=dtype)

        except (FileNotFoundError, OSError, ValueError) as e:
            logging.warning(
                f"Failed to load image for {beam_type.name}: {e}, falling back to random noise"
            )
            return np.random.randint(0, 256, output_shape, dtype=dtype)
        except Exception as e:
            logging.error(
                f"Unexpected error loading image for {beam_type.name}: {e}, falling back to random noise"
            )
            return np.random.randint(0, 256, output_shape, dtype=dtype)

    def _setup_image_iterators(self) -> None:
        """Setup image iterators for simulator image sequences.

        Initializes image sequence iterators from configured SEM and FIB data paths.
        Falls back to random noise generation if no simulator configuration is provided.
        """
        self.use_image_sequence = False

        if self.system.sim is None:
            logging.debug(
                "No simulator configuration found, using random noise generation"
            )
            return

        sem_data_path = self.system.sim.get("sem", None)
        fib_data_path = self.system.sim.get("fib", None)

        if sem_data_path is None or fib_data_path is None:
            logging.info(
                "SEM or FIB data path not configured in simulator settings, using random noise generation"
            )
            return

        if not os.path.exists(sem_data_path) or not os.path.exists(fib_data_path):
            raise ValueError(
                f"SEM data path {sem_data_path} or FIB data path {fib_data_path} does not exist."
            )

        # find all .tif files in the directories
        self._sem_filenames = sorted(glob.glob(os.path.join(sem_data_path, "*.tif*")))
        self._fib_filenames = sorted(glob.glob(os.path.join(fib_data_path, "*.tif*")))

        # validate that files were found
        if len(self._sem_filenames) == 0:
            raise ValueError(f"No .tif files found in SEM data path: {sem_data_path}")
        if len(self._fib_filenames) == 0:
            raise ValueError(f"No .tif files found in FIB data path: {fib_data_path}")

        # create cycling iterators for continuous image sequences
        use_cycle = self.system.sim.get("use_cycle", False)
        if use_cycle:
            self.imaging_system.image_iterators[BeamType.ELECTRON] = cycle(
                self._sem_filenames
            )
            self.imaging_system.image_iterators[BeamType.ION] = cycle(
                self._fib_filenames
            )
        else:
            self.imaging_system.image_iterators[BeamType.ELECTRON] = iter(
                self._sem_filenames
            )
            self.imaging_system.image_iterators[BeamType.ION] = iter(
                self._fib_filenames
            )

        self.use_image_sequence = True
        logging.info(
            f"Image iterators initialized: {len(self._sem_filenames)} SEM images, {len(self._fib_filenames)} FIB images"
        )

    def last_image(self, beam_type: BeamType) -> Optional[FibsemImage]:
        """Get the last acquired image of the specified beam type.
        Args:
            beam_type: The type of beam (electron or ion).
        Returns:
            FibsemImage: The last acquired image.
        """
        beam = self._imaging_beam(beam_type, "_last_image")
        if beam is not None:
            return beam.last_image()
        return self._demo_last_image(beam_type)

    def _demo_last_image(self, beam_type: BeamType) -> Optional[FibsemImage]:
        # `ThermoMicroscope.last_image` sets the channel and then reads the *active
        # view's* buffer -- `imaging.get_image` "retrieves a microscope image currently
        # present in the active view" -- so it is FIB-542's pair on the retrieval path
        # (FIB-569). Claimed and checked here for the same reason, though the simulator
        # has no window between the two: it reads a dict rather than making a second
        # round trip, so only a caller that already lost the channel can trip the check.
        with self._threading_lock:
            self.set_channel(beam_type)
            self._warn_if_channel_moved(beam_type, "last_image")
            image = self.imaging_system.last_image.get(beam_type)
        logging.debug(
            {
                "msg": "last_image",
                "beam_type": beam_type.name,
                "metadata": image.metadata.to_dict(),
            }
        )
        return image

    def acquire_chamber_image(self) -> FibsemImage:
        """Acquire an image of the chamber inside."""
        # The chamber camera is a third device on the shared channel, so looking at it
        # takes the microscope away from whichever beam had it. Captured and put back in
        # a `finally`, because a glance at the chamber that strands a running beam
        # operation is FIB-517 with a different thief (FIB-545).
        with self._threading_lock:
            restore_view = self.imaging_system.active_view
            restore_device = self.imaging_system.active_device
            self.imaging_system.active_view = CHAMBER_ACTIVE_VIEW
            self.imaging_system.active_device = CHAMBER_ACTIVE_DEVICE
            try:
                image = FibsemImage(
                    data=np.random.randint(
                        low=0, high=256, size=(1024, 1536), dtype=np.uint8
                    ),
                    metadata=None,
                )
            finally:
                self.imaging_system.active_view = restore_view
                self.imaging_system.active_device = restore_device
        logging.debug({"msg": "acquire_chamber_image"})
        return image

    def _acquisition_worker(self, beam_type: BeamType):
        """Worker thread for image acquisition."""
        signal = (
            self.sem_acquisition_signal
            if beam_type is BeamType.ELECTRON
            else self.fib_acquisition_signal
        )
        self._demo_live(beam_type, self._stop_acquisition_event, signal.emit)

    def _demo_live(
        self,
        beam_type: BeamType,
        stop: threading.Event,
        emit: Callable[[FibsemImage], None],
    ) -> None:
        """Live view: acquire with the current settings and emit each image until
        *stop* is set. The legacy Demo's worker, and the device-built Demo's beam's."""
        # TODO: add lock

        self.set_channel(beam_type)

        try:
            while True:
                if stop.is_set():
                    break

                # "acquire" image
                image = self.acquire_image(beam_type=beam_type)

                # simulate acquisition time
                dwell_time = image.metadata.image_settings.dwell_time
                resolution = image.metadata.image_settings.resolution
                estimated_time = dwell_time * resolution[0] * resolution[1]
                sim_sleep(estimated_time)

                # emit the acquired image
                emit(image)

        except Exception as e:
            logging.error(f"Error in acquisition worker: {e}")

    def autocontrast(
        self, beam_type: BeamType, reduced_area: Optional[FibsemRectangle] = None
    ) -> None:
        beam = self._imaging_beam(beam_type, "_autocontrast")
        if beam is not None:
            beam.autocontrast(reduced_area)
            return
        self._demo_autocontrast(beam_type, reduced_area)

    def _demo_autocontrast(
        self, beam_type: BeamType, reduced_area: Optional[FibsemRectangle]
    ) -> None:
        # Claims the channel and holds it for the routine, as `ThermoMicroscope`'s does:
        # `run_auto_cb` optimises "the active detector in the active view", so a channel
        # that is not still ours when it runs tunes the other column -- and unlike a
        # stolen grab, that damage persists (FIB-569).
        with self._threading_lock:
            self.set_channel(beam_type)
            if reduced_area is not None:
                self.set_reduced_area_scanning_mode(reduced_area, beam_type)
            # TODO: implement auto-contrast
            logging.info(f"Autocontrasting {beam_type.name} beam.")
            sim_sleep(
                random.uniform(0.5, 1.0)
            )  # simulate time taken to calculate auto-contrast
            self._warn_if_channel_moved(beam_type, "autocontrast")
            self.set_detector_brightness(random.uniform(0.4, 0.6), beam_type)
            self.set_detector_contrast(random.uniform(0.4, 0.6), beam_type)

        if reduced_area:
            self.set_full_frame_scanning_mode(beam_type)
        logging.debug({"msg": "autocontrast", "beam_type": beam_type.name})

    def auto_focus(
        self, beam_type: BeamType, reduced_area: Optional[FibsemRectangle] = None
    ) -> None:
        beam = self._imaging_beam(beam_type, "_auto_focus")
        if beam is not None:
            beam.auto_focus(reduced_area)
            return
        self._demo_auto_focus(beam_type, reduced_area)

    def _demo_auto_focus(
        self, beam_type: BeamType, reduced_area: Optional[FibsemRectangle]
    ) -> None:
        # Same claim as `autocontrast`, for the same reason: `run_auto_focus` runs "in
        # the active view", so losing the channel focuses the other column and leaves it
        # that way. `imaging/tiled.py` calls this once per tile of an unattended run.
        with self._threading_lock:
            self.set_channel(beam_type)
            if reduced_area is not None:
                self.set_reduced_area_scanning_mode(reduced_area, beam_type)
            # TODO: implement auto-focus
            logging.info(f"Auto-focusing {beam_type.name} beam.")
            wd: float = self.get("eucentric_height", beam_type=beam_type)  # type: ignore
            sim_sleep(
                random.uniform(0.5, 1.0)
            )  # simulate time taken to calculate auto-focus
            self._warn_if_channel_moved(beam_type, "auto_focus")
            focus_adjustment = random.uniform(-100e-6, 100e-6)
            new_wd = wd + focus_adjustment
            self.set_working_distance(new_wd, beam_type)

        if reduced_area:
            self.set_full_frame_scanning_mode(beam_type)
        logging.debug({"msg": "auto_focus", "beam_type": beam_type.name})

    def _set_imaging_key(self, key: str, value) -> bool:
        """Set the imaging channel's view or device; False for any other key."""
        if key == "active_view":
            self.imaging_system.active_view = value.value
        elif key == "active_device":
            self.imaging_system.active_device = value.value
        else:
            return False
        return True


class DemoScene:
    """The demo's synthetic sample (FIB-874), shared by both demos.

    The scene is set up from the ``sim: sample`` block; milling and spot burns
    mark it. It reads the beams and stage only through the microscope's API,
    and the parked spot through ``_spot_and_beam``, which each demo answers from
    its own parts.
    """

    def _setup_sample_scene(self) -> None:
        """Opt-in synthetic-sample imaging (FIB-874), default off.

        `sim: sample: {enabled: true, ...}` makes both beams (and the FM,
        where present) image one synthetic cryo-grid through their
        projections, so geometry between the views - coincidence above all -
        is measurable and correctable on the simulator. The block's other
        keys are the scene's options (see SampleScene.CONFIG_KEYS). The
        older flat keys `coincidence_projection`, `coincidence_offset` and
        `tilt_axis_offset` are still honoured when there is no `sample` block.
        """
        from fibsem.microscopes.sim_scene import SampleScene

        self._sample_scene: Optional[SampleScene] = None
        sim = self.system.sim
        config = sim.get("sample")
        if config is None:
            if not sim.get("coincidence_projection", False):
                return
            config = {
                "coincidence_offset": sim.get("coincidence_offset", 10e-6),
                "tilt_axis_offset": sim.get("tilt_axis_offset", 0.0),
            }
        elif not config.get("enabled", False):
            return
        self._sample_scene = SampleScene.from_config(
            {k: v for k, v in config.items() if k != "enabled"}
        )
        try:
            # anchor the world NOW, at the connect pose - so moving straight
            # to a saved position and acquiring shows that position's
            # surroundings rather than anchoring the world there
            self._sample_scene.anchor(self.get_stage_position())
        except Exception as e:
            logging.warning(
                "Could not anchor the sample scene at connect (%s); "
                "it will anchor at the first acquisition instead.",
                e,
            )
        logging.info(
            "Simulator sample scene enabled (coincidence offset %.2f um, "
            "tilt axis offset %.1f um)",
            self._sample_scene.coincidence_offset * 1e6,
            self._sample_scene.tilt_axis_offset * 1e6,
        )

    def _scene_holder_slots(self) -> list:
        """The holder's occupied slots with a position, for the scene to put
        a grid at each: (grid name, grid radius, stage position)."""
        scene = getattr(self, "_sample_scene", None)
        if scene is None or not scene.grids_from_holder:
            return []
        stage = getattr(self, "_stage", None)
        holder = getattr(stage, "holder", None)
        if holder is None:
            return []
        # Where the simulated autoloader really puts a grid, if it is told: the
        # scene draws the grid there whatever the working slot is calibrated to,
        # as on a real Arctis, where a loaded grid sits off the origin (FIB-1144).
        grid_position = getattr(getattr(stage, "loader", None), "grid_position", None)
        try:
            return [
                (
                    slot.loaded_grid.name,
                    slot.loaded_grid.radius,
                    _grid_stage_position(grid_position)
                    if grid_position
                    else slot.position,
                )
                for slot in holder.occupied_slots
                if slot.position is not None
            ]
        except Exception:
            return []

    def _mill_into_sample_scene(self, milling_current: float) -> None:
        """Commit the drawn patterns to the sample scene, when there is one:
        from now on every view shows them as trenches (FIB-877). Done when
        milling starts, so an asynchronous run stamps too."""
        scene = getattr(self, "_sample_scene", None)
        if scene is None or not self.milling_system.patterns:
            return
        from fibsem.projection import BeamStageProjection

        beam = self.milling_channel
        projection = BeamStageProjection.from_microscope(self, beam_type=beam)
        if projection is None:
            return
        shift = self.get_beam_shift(beam)
        scene.mill(
            list(self.milling_system.patterns),
            beam,
            self.get_stage_position(),
            projection,
            beam_shift=(float(shift.x), float(shift.y)),
            beam_current=float(milling_current) if milling_current else None,
        )

    def _burn_into_sample_scene(self, beam_type: BeamType) -> None:
        """Commit the parked beam's spot to the sample scene, when there is
        one: from now on every view shows a small mark there (FIB-954). The
        spot is the 0-1 image coordinate run_spot_burn parked the beam on;
        it becomes metres from the view centre (y up) at the beam's current
        field of view, the same convention the milling patterns use."""
        scene = getattr(self, "_sample_scene", None)
        if scene is None:
            return
        point, beam = self._spot_and_beam(beam_type)
        if point is None:
            return
        from fibsem.projection import BeamStageProjection

        projection = BeamStageProjection.from_microscope(self, beam_type=beam_type)
        if projection is None:
            return
        width, height = beam.resolution
        hfw = float(beam.hfw)
        dx = (float(point.x) - 0.5) * hfw
        dy = (0.5 - float(point.y)) * hfw * (height / width)
        shift = self.get_beam_shift(beam_type)
        logging.info(
            {
                "msg": "sim_spot_burn",
                "point": (float(point.x), float(point.y)),
                "hfw": hfw,
                "resolution": (width, height),
                "view_offset_m": (dx, dy),
                "beam_shift": (float(shift.x), float(shift.y)),
            }
        )
        scene.burn(
            [(dx, dy)],
            beam_type,
            self.get_stage_position(),
            projection,
            beam_shift=(float(shift.x), float(shift.y)),
            beam_current=float(beam.beam_current) if beam.beam_current else None,
        )


class DemoMilling:
    """Simulated milling, shared by both demos.

    The patterns, the milling state and the application files are
    ``milling_system``'s, which the demo sets up at construction; the beams
    change only through the microscope's API.
    """

    milling_system: MillingSystem

    # Whether the mill running now was started by `start_milling`, which never ends
    # on its own here.
    _async_milling: bool = False

    def setup_milling(self, mill_settings: FibsemMillingSettings):
        """Setup the milling parameters."""

        self.milling_system.default_application_file = mill_settings.application_file
        self.milling_channel = mill_settings.milling_channel
        self.set_milling_settings(mill_settings=mill_settings)
        self.clear_patterns()

        logging.debug(
            {"msg": "setup_milling", "mill_settings": mill_settings.to_dict()}
        )

    def run_milling(
        self, milling_current: float, milling_voltage: float, asynch: bool = False
    ) -> None:
        """Run milling with the specified current and voltage."""

        MILLING_SLEEP_TIME = 1
        self._mill_into_sample_scene(milling_current)

        # start milling: this mill is timed by its estimate, not open-ended
        self._async_milling = False
        start_time = time.time()
        estimated_time = self.estimate_milling_time()
        remaining_time = estimated_time
        self.milling_system.state = MillingState.RUNNING

        if asynch:
            return  # up to the caller to handle

        while remaining_time > 0 or self.get_milling_state() in ACTIVE_MILLING_STATES:
            logging.debug(f"Running milling: {remaining_time} s remaining.")
            if self.get_milling_state() == MillingState.PAUSED:
                logging.info("Milling paused.")
                sim_sleep(MILLING_SLEEP_TIME)
                continue
            if self.get_milling_state() == MillingState.IDLE:
                logging.info("Milling stopped.")
                break
            sim_sleep(MILLING_SLEEP_TIME)
            remaining_time -= MILLING_SLEEP_TIME

            # update milling progress via signal
            self.milling_progress_signal.emit(
                MillingProgress(
                    status=MillingProgressStatus.STAGE_UPDATE,
                    start_time=start_time,
                    milling_state=self.get_milling_state(),
                    estimated_time=estimated_time,
                    remaining_time=remaining_time,
                )
            )

            if remaining_time <= 0:  # milling complete
                self.milling_system.state = MillingState.IDLE

        # stop milling and clear patterns
        self.milling_system.state = MillingState.IDLE
        self.clear_patterns()
        logging.debug(
            {
                "msg": "run_milling",
                "milling_current": milling_current,
                "milling_voltage": milling_voltage,
                "asynch": asynch,
            }
        )

    def finish_milling(self, imaging_current: float, imaging_voltage: float) -> None:
        """Finish milling by restoring the imaging current and voltage."""
        self.set_beam_current(current=imaging_current, beam_type=self.milling_channel)
        self.set_beam_voltage(voltage=imaging_voltage, beam_type=self.milling_channel)
        self.clear_patterns()

    def clear_patterns(self) -> None:
        self.milling_system.patterns = []

    def start_milling(self) -> None:
        """Start milling by setting the state to RUNNING."""
        # TODO: support this by properly estimating the end time
        if self.get_milling_state() is MillingState.IDLE:
            self.milling_system.state = MillingState.RUNNING
            self._async_milling = True
            logging.info("Milling started.")

    def stop_milling(self) -> None:
        self.milling_system.state = MillingState.IDLE
        self._async_milling = False

    def pause_milling(self) -> None:
        self.milling_system.state = MillingState.PAUSED

    def resume_milling(self) -> None:
        self.milling_system.state = MillingState.RUNNING

    def get_milling_state(self) -> MillingState:
        return self.milling_system.state

    def estimate_milling_time(self) -> float:
        """Estimate the milling time for the specified patterns.

        While an asynchronous mill is running, which only a stop ends here, the
        estimate adds `SIM_ASYNC_MILLING_EXTRA_TIME`.
        """
        PATTERN_SLEEP_TIME = 5
        estimate = PATTERN_SLEEP_TIME * len(self.milling_system.patterns)
        if self._async_milling and self.get_milling_state() in ACTIVE_MILLING_STATES:
            estimate += SIM_ASYNC_MILLING_EXTRA_TIME
        return estimate

    def set_default_application_file(
        self, application_file: str, strict: bool = True
    ) -> str:
        # Demo models a ThermoFisher system, so it matches application files as one.
        application_file = match_application_file(
            application_file, self.get_available_values("application_file"), strict
        )
        self.milling_system.default_application_file = application_file
        return application_file

    def set_patterning_mode(self, patterning_mode: str) -> None:
        """Set the patterning mode for milling."""
        if patterning_mode not in ["Serial", "Parallel"]:
            raise ValueError(
                f"Invalid patterning mode: {patterning_mode}. Must be 'Serial' or 'Parallel'."
            )
        self.milling_system.patterning_mode = patterning_mode
        logging.debug(
            {"msg": "set_patterning_mode", "patterning_mode": patterning_mode}
        )

    def draw_rectangle(self, pattern_settings: FibsemRectangleSettings) -> None:
        logging.debug(
            {"msg": "draw_rectangle", "pattern_settings": pattern_settings.to_dict()}
        )
        if pattern_settings.time != 0:
            logging.info(f"Setting pattern time to {pattern_settings.time}.")
        self.milling_system.patterns.append(pattern_settings)

    def draw_line(self, pattern_settings: FibsemLineSettings) -> None:
        logging.debug(
            {"msg": "draw_line", "pattern_settings": pattern_settings.to_dict()}
        )
        self.milling_system.patterns.append(pattern_settings)

    def draw_circle(self, pattern_settings: FibsemCircleSettings) -> None:
        logging.debug(
            {"msg": "draw_circle", "pattern_settings": pattern_settings.to_dict()}
        )
        self.milling_system.patterns.append(pattern_settings)

    def draw_polygon(self, pattern_settings: FibsemPolygonSettings) -> None:
        logging.debug(
            {"msg": "draw_polygon", "pattern_settings": pattern_settings.to_dict()}
        )
        self.milling_system.patterns.append(pattern_settings)

    def draw_bitmap_pattern(self, pattern_settings: FibsemBitmapSettings) -> None:
        logging.debug(
            {
                "msg": "draw_bitmap_pattern",
                "pattern_settings": pattern_settings.to_dict(),
            }
        )
        self.milling_system.patterns.append(pattern_settings)

    def _milling_values(self, key: str) -> Optional[List[str]]:
        """The values of a milling key, or None for any other key."""
        if key == "application_file":
            return self.milling_system.application_files
        return None

    def _set_milling_key(self, key: str, value) -> bool:
        """Set a milling key; False for any other key."""
        if key == "patterning_mode":
            self.milling_system.patterning_mode = value
        elif key == "application_file":
            self.milling_system.default_application_file = value
        elif key == "milling_channel":
            self.milling_channel = value
        elif key == "default_patterning_beam_type":
            self.milling_system.default_beam_type = value
        else:
            return False
        return True


@dataclass
class DemoParts:
    """The simulated parts a demo starts with, before anything has moved them."""

    chamber: ChamberSystem
    stage_system: StageSystem
    manipulator_system: ManipulatorSystem
    electron_system: BeamSystem
    ion_system: BeamSystem


def initial_demo_parts(system: SystemSettings) -> DemoParts:
    """A demo's parts at construction, for its configuration."""
    chamber = ChamberSystem(state="Pumped", pressure=1e-6)
    stage_system = StageSystem(
        is_homed=True,
        is_linked=True,
        position=FibsemStagePosition(x=0, y=0, z=0, r=0, t=0, coordinate_system="RAW"),
    )

    manipulator_system = ManipulatorSystem(
        inserted=False,
        position=FibsemManipulatorPosition(
            x=0, y=0, z=0, r=0, t=0, coordinate_system="RAW"
        ),
    )

    electron_system = BeamSystem(
        on=True,
        blanked=False,
        beam=BeamSettings(
            beam_type=BeamType.ELECTRON,
            working_distance=4.0e-3,
            beam_current=100e-12,
            voltage=2000,
            hfw=150e-6,
            resolution=(1536, 1024),
            dwell_time=1e-6,
            stigmation=Point(0, 0),
            shift=Point(0, 0),
            scan_rotation=0,
        ),
        detector=FibsemDetectorSettings(
            type="ETD",
            mode="SecondaryElectrons",
            brightness=0.5,
            contrast=0.5,
        ),
        scanning_mode="full_frame",
    )

    ion_system = BeamSystem(
        on=True,
        blanked=False,
        beam=BeamSettings(
            beam_type=BeamType.ION,
            working_distance=16.5e-3,
            beam_current=20e-12,
            voltage=30000,
            hfw=150e-6,
            resolution=(1536, 1024),
            dwell_time=1e-6,
            stigmation=Point(0, 0),
            shift=Point(0, 0),
            scan_rotation=0,
        ),
        detector=FibsemDetectorSettings(
            type="ETD",
            mode="SecondaryElectrons",
            brightness=0.5,
            contrast=0.5,
        ),
        scanning_mode="full_frame",
        scanning_mode_value=None,
    )
    compustage = system.sim.get("is_compustage", False)
    # A compustage can't link (`set("stage_link")` refuses), so it is never linked.
    stage_system.is_linked = not compustage
    if not compustage:
        # boot at the SEM orientation, as a loaded shuttle sits: at t=0 a
        # pre-tilted shuttle presents the FIB a grazing 3 deg view, a pose
        # no real session starts in. A compustage is flat at t=0 already
        stage_system.position.r = np.radians(system.stage.rotation_reference)
        stage_system.position.t = np.radians(system.stage.shuttle_pre_tilt)
    return DemoParts(
        chamber=chamber,
        stage_system=stage_system,
        manipulator_system=manipulator_system,
        electron_system=electron_system,
        ion_system=ion_system,
    )


class DemoSession:
    """Building and connecting a demo, shared by both demos.

    A demo's ``__init__`` is ``_start_session``, then its parts (Demo's simulated
    parts, or devices), then ``_setup_fluorescence`` and ``_finish_session``.
    """

    def _start_session(self, system_settings: SystemSettings) -> None:
        # initialise system
        self.connection = DemoMicroscopeClient()
        self.system = system_settings
        self.stage_is_compustage: bool = self.system.sim.get("is_compustage", False)
        self.milling_system = MillingSystem(patterns=[])
        self.imaging_system = ImagingSystem()

        # setup image iterators
        try:
            self._setup_image_iterators()
        except ValueError as e:
            logging.error("Failed to set up sim image iterators: %s", str(e))

    def _setup_fluorescence(self) -> None:
        # fluorescence microscope
        #
        # `has_fm` stands in for a capability read, not for configuration. A real
        # Thermo system has no `is_installed` for the FM -- every other subsystem has
        # one -- so the only way to know is to try selecting it and see whether the
        # microscope refuses, which `ThermoMicroscope.__init__` already does. The
        # simulator has nothing to ask, so it is told what the pretend hardware would
        # have answered, and that belongs in `sim:` rather than in a configuration
        # block describing the instrument.
        #
        # Deliberately separate from the fluorescence *geometry*, so that "an FM is
        # present but nothing is configured for it" stays representable -- that is the
        # state an existing site hits on upgrade, and the one worth testing (FIB-830).
        #
        # Defaults to `stage_is_compustage`, which is what this branched on before, so
        # every simulator configuration keeps its current behaviour without the key.
        has_fm = bool(self.system.sim.get("has_fm", self.stage_is_compustage))

        # Two independent questions, and the simulator is the only place both can be
        # posed. `_fluorescence_is_configured` is whether the site said its instrument
        # has an FM; `has_fm` is what the hardware probe would have answered. Both are
        # required, which is what makes the middle row of the table below the sim
        # configuration representable: an FM detected on a system nothing is
        # configured for -- a site upgrading -- gets no FM, and that is the case worth
        # being able to test.
        if (
            has_fm
            and self._fluorescence_is_configured()
            and self._fluorescence_uses_own_driver()
        ):
            self.fm = self._local_fluorescence()
            # Bringing the FM up leaves the shared channel on it, as
            # `ThermoMicroscope.__init__` does; taking it back is the next beam
            # operation's job.
            self.fm.set_active_channel()
        else:
            self.fm = self._connect_remote_fluorescence()
            if self.fm is None:
                logging.info("No fluorescence microscope in this simulated system.")

        self._apply_fluorescence_calibration()
        self._warn_on_fluorescence_geometry()

    def _local_fluorescence(self) -> FluorescenceMicroscope:
        """The FM this demo simulates itself: the simulated FM, sharing the imaging
        channel with the beams."""
        return SimulatedFluorescenceMicroscope(self)

    def _finish_session(self) -> None:
        # user, experiment metadata
        # TODO: remove once db integrated
        self.user = FibsemUser.from_environment()
        self.experiment = FibsemExperimentRef()

        self._last_imaging_settings: ImageSettings = ImageSettings()
        self.milling_channel: BeamType = BeamType.ION
        self._image_cache: dict = {}
        self._setup_sample_scene()
        logging.debug(
            {
                "msg": "create_microscope_client",
                "system_settings": self.system.to_dict(),
            }
        )

    def connect_to_microscope(
        self, ip_address: str, port: int = 8080, reset_beam_shift: bool = True
    ) -> None:
        """Connect to the microscope server.
        Args:
            ip_address: The IP address of the microscope server.
            port: The port number of the microscope server.
            reset_beam_shift: Whether to reset beam shifts on connect (default: True).
        """
        # connect to microscope
        self.connection.connect(ip_address=ip_address, port=port)

        # system information
        self.system.info.model = "DemoMicroscope"
        self.system.info.serial_number = "123456"
        self.system.info.software_version = "0.1"
        self.system.info.hardware_version = "v0.23"
        self.system.info.ip_address = ip_address

        # reset beam shifts
        if reset_beam_shift:
            self.reset_beam_shifts()

        # user logging
        info = self.system.info
        logging.info(
            f"Microscope client connected to {info.model} with serial number {info.serial_number} and software version {info.software_version}"
        )

        # logging
        logging.debug(
            {
                "msg": "connect_to_microscope",
                "ip_address": ip_address,
                "port": port,
                "system_info": info.to_dict(),
            }
        )

        try:
            self._create_sample_stage()
        except Exception as e:
            logging.warning(f"Could not create sample stage: {e}")

        return

    def disconnect(self) -> None:
        """Disconnect from the microscope server."""
        self.connection.disconnect()
        logging.info("Disconnected from Demo Microscope")


class LegacyDemoMicroscope(
    DemoSession,
    DemoConfiguration,
    DemoImaging,
    DemoScene,
    DemoMilling,
    FibsemMicroscope,
):
    """The Demo backend before devices: simulated parts behind ``_get``/``_set``.

    ``DemoMicroscope`` (``fibsem.microscopes.device_demo``) replaced it, built from
    devices. This class is kept frozen as the reference the contract suite compares
    the device-built Demo against (``tests/test_microscope_contract.py``); a
    deliberate behaviour change to the Demo lands here in the same PR. No
    configuration selects it.
    """

    vertical_move_views = (BeamType.ION, BeamType.ELECTRON)

    def __init__(self, system_settings: SystemSettings):
        self._start_session(system_settings)
        parts = initial_demo_parts(self.system)
        self.chamber = parts.chamber
        self.stage_system = parts.stage_system
        self.manipulator_system = parts.manipulator_system
        self.electron_system = parts.electron_system
        self.ion_system = parts.ion_system
        self._setup_fluorescence()
        self._finish_session()

    # The milling beam's conditions the first setup_milling found, until
    # finish_milling puts them back: what the Demo's milling service does
    # (`fibsem.services.milling.Milling`), by key.
    _milling_saved: Optional[Tuple[BeamType, Dict[str, Any]]] = None

    def setup_milling(self, mill_settings: FibsemMillingSettings):
        if self._milling_saved is None:
            channel = mill_settings.milling_channel
            saved = {
                key: self.get(key, channel) for key in ("voltage", "current", "hfw")
            }
            self._milling_saved = (channel, saved)
        super().setup_milling(mill_settings)

    def finish_milling(
        self,
        imaging_current: Optional[float] = None,
        imaging_voltage: Optional[float] = None,
    ) -> None:
        self.clear_patterns()
        if self._milling_saved is not None:
            channel, saved = self._milling_saved
            self._milling_saved = None
            for key, value in saved.items():
                self.set(key, value, channel)
        if imaging_voltage is not None:
            self.set_beam_voltage(
                voltage=imaging_voltage, beam_type=self.milling_channel
            )
        if imaging_current is not None:
            self.set_beam_current(
                current=imaging_current, beam_type=self.milling_channel
            )

    @_records_beam_shift
    def beam_shift(self, dx: float, dy: float, beam_type: BeamType) -> None:

        logging.debug(
            {"msg": "beam_shift", "dx": dx, "dy": dy, "beam_type": beam_type.name}
        )

        if beam_type == BeamType.ELECTRON:
            self.electron_system.beam.shift += Point(float(dx), float(dy))
        elif beam_type == BeamType.ION:
            self.ion_system.beam.shift += Point(float(dx), float(dy))

    @_records_stage_move
    def move_stage_absolute(self, position: FibsemStagePosition) -> FibsemStagePosition:
        """Move the stage to the specified position."""
        # Before the position is assigned, not after: a stage that is moving has not
        # arrived, and anything reading the position during the move should see where it
        # set off from. The read happens on the GUI thread while the move runs on a
        # worker, so the two really can overlap.
        sim_sleep(STAGE_MOVEMENT_SLEEP_TIME)

        # only assign if not None
        if position.x is not None:
            self.stage_system.position.x = position.x
        if position.y is not None:
            self.stage_system.position.y = position.y
        if position.z is not None:
            self.stage_system.position.z = position.z
        if position.r is not None:
            self.stage_system.position.r = position.r
        if position.t is not None:
            self.stage_system.position.t = position.t

        logging.debug({"msg": "move_stage_absolute", "position": position.to_dict()})

        return self.get_stage_position()

    @_records_stage_move
    def move_stage_relative(self, position: FibsemStagePosition) -> FibsemStagePosition:
        """Move the stage by the specified amount."""
        sim_sleep(STAGE_MOVEMENT_SLEEP_TIME)  # see `move_stage_absolute`

        self.stage_system.position += position

        logging.debug({"msg": "move_stage_relative", "position": position.to_dict()})

        return self.get_stage_position()

    def insert_manipulator(self, name: str = "PARK") -> FibsemManipulatorPosition:
        """Insert the manipulator to the specified position."""

        logging.info(f"Inserting manipulator to {name}...")
        self.move_manipulator_absolute(
            FibsemManipulatorPosition(x=0, y=0, z=180e-6, r=0, t=0)
        )
        self.manipulator_system.inserted = True
        logging.debug({"msg": "insert_manipulator", "name": name})

        return self.get_manipulator_position()

    def retract_manipulator(self) -> FibsemManipulatorPosition:
        """Retract the manipulator."""
        logging.info("Retracting manipulator...")
        self.move_manipulator_absolute(
            FibsemManipulatorPosition(x=0, y=0, z=0, r=0, t=0)
        )
        self.manipulator_system.inserted = False
        logging.debug({"msg": "retract_manipulator"})
        return self.get_manipulator_position()

    def move_manipulator_relative(
        self, position: FibsemManipulatorPosition
    ) -> FibsemManipulatorPosition:
        logging.info(f"Moving manipulator: {position} (Relative)")
        self.manipulator_system.position += position
        logging.debug(
            {"msg": "move_manipulator_relative", "position": position.to_dict()}
        )
        return self.get_manipulator_position()

    def move_manipulator_absolute(
        self, position: FibsemManipulatorPosition
    ) -> FibsemManipulatorPosition:
        logging.info(f"Moving manipulator: {position} (Absolute)")
        self.manipulator_system.position = position
        logging.debug(
            {"msg": "move_manipulator_absolute", "position": position.to_dict()}
        )
        return self.get_manipulator_position()

    def move_manipulator_corrected(
        self, dx: float, dy: float, beam_type: BeamType
    ) -> FibsemManipulatorPosition:
        logging.info(
            f"Moving manipulator: dx={dx:.2e}, dy={dy:.2e}, beam_type = {beam_type.name} (Corrected)"
        )
        self.manipulator_system.position.x += dx
        self.manipulator_system.position.y += dy
        logging.debug(
            {
                "msg": "move_manipulator_corrected",
                "dx": dx,
                "dy": dy,
                "beam_type": beam_type.name,
            }
        )
        return self.get_manipulator_position()

    def move_manipulator_to_position_offset(
        self, offset: FibsemManipulatorPosition, name: Optional[str] = None
    ) -> FibsemManipulatorPosition:
        if name is None:
            name = "EUCENTRIC"

        position = self._get_saved_manipulator_position(name)

        logging.info(f"Moving manipulator: {offset} to {name}")
        self.move_manipulator_absolute(position + offset)
        logging.debug(
            {
                "msg": "move_manipulator_to_position_offset",
                "offset": offset.to_dict(),
                "name": name,
            }
        )
        return self.get_manipulator_position()

    manipulator_move_types = ("relative", "corrected")

    def manipulator_named_positions(self) -> List[str]:
        return ["PARK", "EUCENTRIC"]

    def _get_saved_manipulator_position(
        self, name: str = "PARK"
    ) -> FibsemManipulatorPosition:

        if name not in ["PARK", "EUCENTRIC"]:
            raise ValueError(f"Unknown manipulator position: {name}")
        if name == "PARK":
            return FibsemManipulatorPosition(x=0, y=0, z=180e-6, r=0, t=0)
        if name == "EUCENTRIC":
            return FibsemManipulatorPosition(x=0, y=0, z=0, r=0, t=0)

    def _spot_and_beam(
        self, beam_type: BeamType
    ) -> Tuple[Union[None, Point, FibsemRectangle], BeamSettings]:
        """The point a beam is parked on (its scan target) and its settings."""
        beam_system = (
            self.electron_system if beam_type is BeamType.ELECTRON else self.ion_system
        )
        return beam_system.scanning_mode_value, beam_system.beam

    def _get(
        self, key, beam_type: Optional[BeamType] = None
    ) -> Union[float, int, bool, str, list, FibsemStagePosition]:
        """Get a value from the microscope."""
        # get beam
        if beam_type is not None:
            beam_system = (
                self.electron_system
                if beam_type is BeamType.ELECTRON
                else self.ion_system
            )
            beam, detector = beam_system.beam, beam_system.detector

        # TODO: change this so value is returned, so we can log the return value

        # beam properties
        if key == "on":
            return beam_system.on
        if key == "blanked":
            return beam_system.blanked
        if key == "voltage":
            return beam.voltage
        if key == "current":
            return beam.beam_current
        if key == "working_distance":
            return beam.working_distance
        if key == "hfw":
            return beam.hfw
        if key == "resolution":
            return beam.resolution
        if key == "dwell_time":
            return beam.dwell_time
        if key == "stigmation":
            return Point(beam.stigmation.x, beam.stigmation.y)
        if key == "shift":
            return Point(beam.shift.x, beam.shift.y)
        if key == "scan_rotation":
            return float(beam.scan_rotation)

        # ion beam properties
        if key == "plasma":
            return self._read_plasma(beam_type)

        if key == "plasma_gas":
            if beam_type is BeamType.ION and self.system.ion.plasma:
                return (
                    self.system.ion.plasma_gas
                )  # might need to check if this is available?
            else:
                return None

        # stage
        if key == "stage_position":
            sim_sleep(0.1)
            return self.stage_system.position
        if key == "stage_homed":
            return self.stage_system.is_homed
        if key == "stage_linked":
            return self.stage_system.is_linked

        # detector properties
        #
        # Set, then read against the shared channel, as on hardware, where
        # `connection.detector` resolves against the active device: a read whose channel
        # has moved answers from the other column's detector, silently. Warned here
        # rather than answered wrongly (FIB-544).
        if key in DETECTOR_KEYS:
            self.set_channel(beam_type)
            self._warn_if_channel_moved(beam_type, f"reading {key}")
            if key == "detector_type":
                return detector.type
            if key == "detector_mode":
                return detector.mode
            if key == "detector_brightness":
                return detector.brightness
            if key == "detector_contrast":
                return detector.contrast

        # manipulator properties
        if key == "manipulator_position":
            return self.manipulator_system.position
        if key == "manipulator_state":
            return self.manipulator_system.inserted

        # chamber properties
        if key == "chamber_state":
            return self.chamber.state
        if key == "chamber_pressure":
            return self.chamber.pressure

        # scanning mode
        if key == "scanning_mode":
            return beam_system.scanning_mode

        if key in SIMULATOR_KNOWN_UNKNOWN_KEYS:
            logging.debug(f"Skipping unknown key: {key} for {beam_type}")
            return None

        logging.warning(f"Unknown key: {key} ({beam_type})")
        return None

    def _set(self, key: str, value, beam_type: Optional[BeamType] = None) -> None:
        """Set a property of the microscope."""

        # get beam
        if beam_type is not None:
            beam_system = (
                self.electron_system
                if beam_type is BeamType.ELECTRON
                else self.ion_system
            )
            beam = beam_system.beam
            detector = beam_system.detector

        # voltage
        if key == "voltage":
            beam.voltage = value
            return
        # current
        if key == "current":
            beam.beam_current = value
            return

        if key == "working_distance":
            beam.working_distance = value
            return

        if key == "stigmation":
            beam.stigmation = value
            return
        if key == "shift":
            beam.shift = value
            return
        if key == "scan_rotation":
            beam.scan_rotation = float(value)
            return
        if key == "hfw":
            beam.hfw = value
            return
        if key == "resolution":
            beam.resolution = value
            return
        if key == "dwell_time":
            beam.dwell_time = value
            return

        # beam control
        if key == "on":
            beam_system.on = value
            return

        if key == "blanked":
            beam_system.blanked = value
            # the beam parked on a point and let through is a spot burn: the
            # base run_spot_burn does blank -> spot -> unblank per point
            if not value and beam_system.scanning_mode == "spot":
                self._burn_into_sample_scene(beam_type)
            return

        # detector: the write half of the same pair, which on hardware would land on
        # the other column's detector and stay there (FIB-544)
        if key in DETECTOR_KEYS:
            self.set_channel(beam_type)
            self._warn_if_channel_moved(beam_type, f"writing {key}")
            if key == "detector_type":
                detector.type = value
                return
            if key == "detector_mode":
                detector.mode = value
                return
            if key == "detector_contrast":
                detector.contrast = value
                return
            if key == "detector_brightness":
                detector.brightness = value
                return

        if beam_type is BeamType.ION:
            if key == "plasma_gas":
                if not self.system.ion.plasma:
                    logging.debug("Plasma gas cannot be set on this microscope.")
                    return
                if value not in self.get_available_values("plasma_gas", beam_type):
                    logging.warning(
                        f"Plasma gas {value} not available. Available values: {self.get_available_values('plasma_gas', beam_type)}"
                    )
                    return
                logging.info(
                    f"Setting plasma gas to {value}... this may take some time..."
                )
                self.system.ion.plasma_gas = value
                logging.info(f"Plasma gas set to {value}.")

                return

        if key == "spot_mode":
            # value: Point, image pixels
            beam_system.scanning_mode = "spot"
            beam_system.scanning_mode_value = value
            return

        if key == "reduced_area":
            beam_system.scanning_mode = "reduced_area"
            beam_system.scanning_mode_value = value
            return

        if key == "full_frame":
            beam_system.scanning_mode = "full_frame"
            beam_system.scanning_mode_value = value
            return

        if self._set_imaging_key(key, value) or self._set_milling_key(key, value):
            return

        # stage properties
        if key == "stage_home":
            logging.info("Homing stage...")
            self.stage_system.is_homed = True
            logging.info("Stage homed.")
            return

        if key == "stage_link":
            if self.stage_is_compustage:
                logging.debug("Compustage does not support linking.")
                return
            logging.info("Linking stage...")
            self.stage_system.is_linked = True
            logging.info("Stage linked.")
            return

        # chamber properties
        if key == "pump_chamber":
            if value:
                logging.info("Pumping chamber...")
                self.chamber.state = "Pumped"
                self.chamber.pressure = 1e-6  # 1 uTorr
                logging.info("Chamber pumped.")
            else:
                logging.info(f"Invalid value for pump_chamber: {value}")
            return
        if key == "vent_chamber":
            if value:
                logging.info("Venting chamber...")
                self.chamber.state = "Vented"
                self.chamber.pressure = 1e5
                logging.info("Chamber vented.")
            else:
                logging.info(f"Invalid value for vent_chamber: {value}")
            return

        if key in SIMULATOR_KNOWN_UNKNOWN_KEYS:
            logging.debug(f"Skipping unknown key: {key} for {beam_type}")
            return

        logging.warning(f"Unknown key: {key} ({beam_type})")
        return None

    def home(self) -> bool:
        self.stage_system.is_homed = True
        return self.get("stage_homed")


def __getattr__(name: str):
    # `DemoMicroscope` is the device-built Demo, which imports this module, so it is
    # looked up when asked for rather than imported at the top.
    if name == "DemoMicroscope":
        from fibsem.microscopes.device_demo import DemoMicroscope

        return DemoMicroscope
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
