from __future__ import annotations

import copy
import glob
import logging
import os
import random
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import cycle
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from skimage.transform import resize

from fibsem import manufacturers
from fibsem._timing import sim_sleep
from fibsem.drivers.demo.sim_scene import fm_channel_weights
from fibsem.fm.microscope import (
    FluorescenceMicroscope,
)
from fibsem.projection import FMStageProjection
from fibsem.structures import (
    BeamSettings,
    BeamType,
    FibsemDetectorSettings,
    FibsemExperimentRef,
    FibsemImage,
    FibsemManipulatorPosition,
    FibsemPatternSettings,
    FibsemRectangle,
    FibsemStagePosition,
    FibsemUser,
    ImageSettings,
    MillingState,
    Point,
    RangeLimit,
    SystemSettings,
)
from fibsem.util.draw_numbers import draw_text

######################## SIMULATOR ########################


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


def sim_is_compustage(system: SystemSettings) -> bool:
    """Whether a simulated configuration describes a compustage.

    ``sim.is_compustage`` stands in for the hardware probe a real backend makes at
    connect (``specimen.compustage.is_installed`` on Thermo), so it is read from the
    configuration, never from a flag set on the microscope afterwards.
    """
    return bool(system.sim.get("is_compustage", False))


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
# (`FM_ACTIVE_VIEW = 3` in `fibsem.drivers.autoscript.devices`, quadrant 3 on an
# Arctis); all that matters here is that it is neither beam's, so a beam operation can
# tell that the FM took the channel out from under it.
FM_ACTIVE_VIEW = 3
FM_ACTIVE_DEVICE = 3

# The simulated FM: what its parts offer and where they start. The Demo FM devices
# (``fibsem.drivers.demo.devices``) start from the same values.
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
    the Demo FM camera device.
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


def _grid_stage_position(grid_position) -> FibsemStagePosition:
    """Where the simulated autoloader puts a grid, as a stage position at the
    working slot's pose (the SEM orientation, r = t = 0)."""
    x, y, z = (float(v) for v in grid_position)
    return FibsemStagePosition(name="Slot-01", x=x, y=y, z=z, r=0.0, t=0.0)


class DemoConfiguration:
    """What a demo configuration says the instrument has.

    Everything here reads only ``system`` (its ``sim:`` block and ``ion``), never a
    simulated part (the beam keys' values go through ``get``), so it is the same whether
    the parts are Demo's or devices.
    """

    system: SystemSettings

    # ---- fitted subsystems, as the simulated instrument reports them ---------
    #
    # The `sim:` block is where a simulated configuration stands in for a hardware
    # probe (`has_fm`, `is_compustage`), so that is where these come from too. Absent
    # means the Demo default -- fitted.

    def _probe_manipulator_installed(self) -> Optional[bool]:
        return self.system.sim.get("has_manipulator")

    def _probe_plasma_gas(self) -> Optional[str]:
        return self.system.sim.get("plasma_gas")


class DemoImaging:
    """Imaging on a demo: the beams' frames, the chamber camera and the shared channel.

    It reads and changes the beams only through the microscope's beam methods, so it
    images through the beam devices.
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
        return self._imaging_beam(target).acquire(image_settings)

    def _imaging_beam(self, beam_type: BeamType):
        """The beam device that images, as on the other backends. The Demo's beams
        run imaging as commands, which call back into the ``_demo_*`` methods below;
        a column disabled in the config has none."""
        beam = self.beams.get(beam_type)
        if beam is None:
            raise ValueError(f"The {beam_type.name} beam is not enabled.")
        return beam

    def _demo_acquire(
        self,
        image_settings: Optional[ImageSettings],
        beam_type: Optional[BeamType],
    ) -> FibsemImage:
        """``acquire_image``'s frame, through the beam's acquire command."""
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
        self._write_beam("hfw", effective_image_settings.hfw, effective_beam_type)

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
                current = float(self.get_beam_current(effective_beam_type))
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
        return self._imaging_beam(beam_type).last_image()

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

    def _demo_live(
        self,
        beam_type: BeamType,
        stop: threading.Event,
        emit: Callable[[FibsemImage], None],
    ) -> None:
        """Live view: acquire with the current settings and emit each image until
        *stop* is set. The Demo beam's live view."""
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
        self._imaging_beam(beam_type).autocontrast(reduced_area)

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
        self._imaging_beam(beam_type).auto_focus(reduced_area)

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
            wd: float = self._beam_config(
                "eucentric_height", beam_type
            ).eucentric_height
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
    """The demo's synthetic sample (FIB-874).

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
        from fibsem.drivers.demo.sim_scene import SampleScene

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
        device = self.devices.get("sample_loader")
        grid_position = getattr(device, "sim_grid_position", None)
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
    compustage = sim_is_compustage(system)
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
    """Building and connecting a demo.

    A demo's ``__init__`` is ``_start_session``, then its devices, then
    ``_setup_fluorescence`` (which builds its FM from the configuration's ``fm``
    entry) and ``_finish_session``.
    """

    def _start_session(self, system_settings: SystemSettings) -> None:
        # initialise system
        self.connection = DemoMicroscopeClient()
        self.system = system_settings
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
        # Defaults to `sim.is_compustage`, which is what this branched on before, so
        # every simulator configuration keeps its current behaviour without the key.
        has_fm = bool(self.system.sim.get("has_fm", sim_is_compustage(self.system)))

        # Two independent questions, and the simulator is the only place both can be
        # posed. `_fluorescence_is_configured` is whether the site said its instrument
        # has an FM; `has_fm` is what the hardware probe would have answered. Both are
        # required, which is what makes the middle row of the table below the sim
        # configuration representable: an FM detected on a system nothing is
        # configured for -- a site upgrading -- gets no FM, and that is the case worth
        # being able to test.
        # The probe only answers for the Demo's own FM; one on another driver (an FM
        # on its own PC) is built as the configuration says.
        if has_fm or not self._fluorescence_uses_own_driver():
            self.fm = self._build_fluorescence(manufacturers.DEMO)
        else:
            self.fm = None
        if self.fm is None:
            logging.info("No fluorescence microscope in this simulated system.")

        self._apply_fluorescence_calibration()
        self._warn_on_fluorescence_geometry()

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


def __getattr__(name: str):
    # `DemoMicroscope` is the device-built Demo, which imports this module, so it is
    # looked up when asked for rather than imported at the top.
    if name == "DemoMicroscope":
        from fibsem.drivers.demo.microscope import DemoMicroscope

        return DemoMicroscope
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
