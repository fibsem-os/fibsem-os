import logging
import threading
from contextlib import contextmanager
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
from autoscript_sdb_microscope_client.enumerations import (
    CameraEmissionType,
    CameraFilterType,
    ImagingDevice,
    ImagingState,
    RetractableDeviceState,
)
from autoscript_sdb_microscope_client.structures import (
    GetImageSettings,
    GrabFrameSettings,
)

from fibsem.fm.microscope import (
    Camera,
    FilterSet,
    FluorescenceMicroscope,
    LightSource,
    ObjectiveLens,
)
from fibsem.microscope import SdbMicroscopeClient

COLOR_TO_WAVELENGTH = {
    CameraEmissionType.BLUE: 365,
    CameraEmissionType.GREEN_YELLOW: 450,
    CameraEmissionType.RED: 550,
    CameraEmissionType.VIOLET: 635,
}
# TODO: migrate to using the enumeration from autoscript_sdb_microscope_client
WAVELENGTH_TO_COLOR = {v: k for k, v in COLOR_TO_WAVELENGTH.items()}
AVAILABLE_FM_COLORS = list(COLOR_TO_WAVELENGTH.keys())
AVAILABLE_FM_WAVELENGTHS = list(COLOR_TO_WAVELENGTH.values())

# specs:
# arctis: https://assets.thermofisher.com/TFS-Assets/MSD/Datasheets/arctis-cryo-plasma-fib-ds0384-en.pdf
# - 100x magnification
# - 0.75 NA
# - 150 um fov
# - 4mm working distance
# - light source: 365 nm, 450 nm, 550 nm, 635 nm
# iflm: https://assets.thermofisher.com/TFS-Assets/MSD/Datasheets/iflm-correlative-system-ds0499.pdf
# - 20x magnification
# - 0.7 NA
# - 500 um fov
# - 1.3 mm working distance
# - light source: 365 nm, 450 nm, 550 nm, 635 nm

ARCTIS_CONFIGURATION = {
    "name": "ARCTIS",
    "magnification": 100.0,
    "numerical_aperture": 0.75,
    "working_distance": 4e-3,
    "pixel_size": (2.74822695035461e-08, 2.74822695035461e-08),
    "resolution": (4512, 4512),
    "focus_position": 8.0e-3,  # 8000 microns
    "limit_position": 8.6e-3,  # 8600 microns
}

ARCTIS_LWD_CONFIGURATION = {
    "name": "ARCTIS_LWD",
    "magnification": 50.0,
    "numerical_aperture": 0.75,
    "working_distance": 4e-3,
    "pixel_size": (2.74822695035461e-08 * 2, 2.74822695035461e-08 * 2),
    "resolution": (4512, 4512),
    "focus_position": 7.6e-3,  # 7600 microns
    "limit_position": 8.6e-3,  # 8600 microns
}

IFLM_CONFIGURATION = {
    "magnification": 20.0,
    "numerical_aperture": 0.7,
    "working_distance": 1.3e-3,
    "pixel_size": (2.74822695035461e-08, 2.74822695035461e-08),
    "resolution": (4512, 4512),
}

# 2.74822695035461e-08
DEFAULT_CONFIGURATION = ARCTIS_CONFIGURATION  # Default to ARCTIS configuration

HFW = 150e-6  # Horizontal field width for ARCTIS (diagonal)
IFLM_HFW = 500e-6  # Horizontal field width for iFlm


class ThermoFisherObjectiveLens(ObjectiveLens):
    """Thermo Fisher objective lens implementation for fluorescence microscopy.

    Provides control over objective lens positioning, state management, and configuration
    for Thermo Fisher FLM systems (Arctis, iFlm).
    """

    def __init__(self, parent: "ThermoFisherFluorescenceMicroscope"):
        """Initialize the objective lens with default configuration.

        Args:
            parent: The parent fluorescence microscope instance
        """
        super().__init__(parent)
        self.parent = parent
        self._magnification = DEFAULT_CONFIGURATION["magnification"]
        self._numerical_aperture = DEFAULT_CONFIGURATION["numerical_aperture"]
        self._working_distance = DEFAULT_CONFIGURATION["working_distance"]
        self._focus_position = DEFAULT_CONFIGURATION["focus_position"]
        self._limit_position: float = DEFAULT_CONFIGURATION.get(
            "limit_position", 8.6e-3
        )

    @property
    def magnification(self) -> float:
        """Get the magnification of the objective lens.

        Returns:
            The current magnification value
        """
        return self._magnification

    @magnification.setter
    def magnification(self, value: float):
        """Set the magnification of the objective lens.

        Args:
            value: The magnification value to set
        """
        self._magnification = value

    @property
    def position(self) -> float:
        """Get the current focus position of the objective lens.

        Returns:
            The current focus position in metres
        """
        with self.parent.active_channel():
            return self.parent.fm_settings.focus.value

    @property
    def limits(self) -> Tuple[float, float]:
        """Get the focus position limits of the objective lens.

        Returns:
            A tuple of (minimum, maximum) focus position limits in metres
        """
        with self.parent.active_channel():
            limits = self.parent.fm_settings.focus.limits
            return (limits.min, limits.max)

    def move_relative(self, delta: float):
        """Move the objective lens by a relative distance.

        Args:
            delta: The distance to move in metres (positive = towards sample)
        """
        current_position = self.position
        new_position = current_position + delta
        # Delegates, so the announcement comes from `move_absolute`. Not every driver
        # does -- see `_notify_moved`.
        self.move_absolute(new_position)

    def move_absolute(self, position: float):
        """Move the objective lens to an absolute focus position.

        Args:
            position: The target focus position in metres
        """
        # check hardware limits
        if not self.limits[0] <= position <= self.limits[1]:
            raise ValueError(f"Position {position} out of limits {self.limits}")

        # clip to user-defined limits
        if not position <= self._limit_position:
            logging.warning(
                f"Clipping position {position} to user-defined limits {self._limit_position}"
            )
            position = np.clip(position, 0, self._limit_position)
        with self.parent.active_channel():
            self.parent.fm_settings.focus.value = position
        self._notify_moved()

    def insert(self) -> None:
        """Insert the objective lens into the working position.

        Moves the objective lens to the active/inserted position for imaging.
        """
        with self.parent.active_channel():
            if self.state == "Inserted":
                logging.warning("Objective lens is already inserted.")
                return
            self.parent.connection.detector.insert()
        # Outside the scope, and after the early return: nothing moved in that case, so
        # there is nothing to announce.
        self._notify_moved()

    def retract(self) -> None:
        """Retract the objective lens from the working position.

        Moves the objective lens to the inactive/retracted position for safety.
        """
        with self.parent.active_channel():
            if self.state == "Retracted":
                logging.warning("Objective lens is already retracted.")
                return
            self.parent.connection.detector.retract()
        self._notify_moved()

    @property
    def state(self) -> Literal["Inserted", "Retracted", "Busy", "Error", "Other"]:
        """Get the current state of the objective lens.

        Returns:
            The objective lens state (RetractableDeviceState) (e.g., 'Inserted', 'Retracted', 'Busy', 'Error', 'Other')
        """
        # TODO: migrate to standard enum, rather than string
        with self.parent.active_channel():
            return self.parent.connection.detector.state

    def is_homed(self) -> bool:
        """Check if the objective lens is in the homed position.

        Returns:
            True if the objective lens is homed, False otherwise
        """
        with self.parent.active_channel():
            return self.parent.connection.detector.is_homed

    def home(self) -> None:
        """Home the objective lens to its reference position.

        Moves the objective lens to its home/reference position for calibration.
        """
        with self.parent.active_channel():
            self.parent.connection.detector.home()
        self._notify_moved()


class ThermoFisherCamera(Camera):
    """Thermo Fisher camera implementation for fluorescence microscopy.

    Provides control over camera settings including exposure time, binning,
    and image acquisition for Thermo Fisher FLM systems.
    """

    def __init__(self, parent: "ThermoFisherFluorescenceMicroscope"):
        """Initialize the camera with parent microscope.

        Args:
            parent: The parent fluorescence microscope instance
        """
        super().__init__(parent)
        self.parent = parent
        self._pixel_size = DEFAULT_CONFIGURATION["pixel_size"]
        self._resolution = DEFAULT_CONFIGURATION["resolution"]

    def acquire_image(self) -> np.ndarray:
        """Acquire a single image from the camera.

        Uses the current camera settings (exposure time, binning) to capture
        an image from the fluorescence microscope.

        Returns:
            A numpy array containing the image data
        """
        # get the internal excitation wavelength (emission type)
        emission_type = self.parent.filter_set._emission_type
        frame_settings = GrabFrameSettings(emission_type=emission_type)
        # Uses current camera settings for binning and exposure time

        # Deliberately unscoped: every path here comes through
        # `FluorescenceMicroscope.acquire_image`, which holds the channel around the
        # whole acquisition, and a tileset holds it around the whole run. Taking and
        # returning it per frame would flick the microscope's own UI between views once
        # per tile (FIB-517).
        self.parent.set_active_channel()
        image = self.parent.connection.imaging.grab_frame(frame_settings)
        return image.data  # AdornedImage.data -> np.ndarray

    def _start_fast_acquisition(self):
        # The one place that deliberately keeps the channel rather than handing it back.
        # A live stream owns the instrument for as long as it runs -- that is what it is
        # -- and re-forces the channel each frame because something else may have taken
        # it. Everything discrete goes through `active_channel()` instead (FIB-517).
        try:
            with self.parent.parent._threading_lock:
                self.parent.set_active_channel()
                emission_color = (
                    self.parent.connection.detector.camera_settings.emission.type.value
                )
                self.parent.light_source.start_emission(emission_type=emission_color)
                self.start_acquisition()

            while self.parent.connection.imaging.state == ImagingState.ACQUIRING:
                if self.parent._stop_acquisition_event.is_set():
                    self.parent.light_source.stop_emission()
                    self.parent.stop_acquisition()
                    break
                with self.parent.parent._threading_lock:
                    self.parent.set_active_channel()  # re-force active channel...?
                    image = self.parent.connection.imaging.get_image()
                    # if image.data.shape != self.resolution: # if shape doesn't match expected resolution, skip (likely acquired wrong channel)
                    # logging.warning(f"Acquired image shape {image.shape} does not match expected resolution {self.resolution}")
                    # continue
                    self.parent._construct_image(image.data)

        except Exception as e:
            logging.error(f"Exception occurred during fast acquisition: {e}")
        finally:
            self.parent.light_source.stop_emission()
            self.stop_acquisition()

    def start_acquisition(self) -> None:
        """Start the camera acquisition process.

        Begins image acquisition using the current camera settings.
        This method is typically used for continuous imaging or live view.
        """
        self.parent.set_active_channel()
        self.parent.connection.imaging.start_acquisition()

    def stop_acquisition(self) -> None:
        """Stop the camera acquisition process.

        Halts any ongoing image acquisition and releases resources.
        This is typically used to end continuous imaging or live view.
        """
        self.parent.set_active_channel()
        self.parent.connection.imaging.stop_acquisition()

    @property
    def exposure_time(self) -> float:
        """Get the current exposure time of the camera.

        Returns:
            The exposure time in seconds
        """
        with self.parent.active_channel():
            return self.parent.fm_settings.exposure_time.value

    @exposure_time.setter
    def exposure_time(self, value: float) -> None:
        """Set the exposure time of the camera.

        Args:
            value: The exposure time in seconds

        Raises:
            ValueError: If the exposure time is outside the valid range
        """
        limits = self.exposure_time_limits
        if not limits[0] <= value <= limits[1]:
            raise ValueError(
                f"Exposure time must be between {limits[0]} and {limits[1]}, got {value}"
            )
        with self.parent.active_channel():
            self.parent.fm_settings.exposure_time.value = value

    @property
    def exposure_time_limits(self) -> Tuple[float, float]:
        """Get the valid exposure time range for the camera.

        Returns:
            A tuple of (minimum, maximum) exposure times in seconds
        """
        with self.parent.active_channel():
            limits = self.parent.fm_settings.exposure_time.limits
            return (limits.min, limits.max)

    @property
    def binning(self) -> int:
        """Get the current binning setting of the camera.

        Returns:
            The current binning value (e.g., 1, 2, 4, 8)
        """
        with self.parent.active_channel():
            return self.parent.fm_settings.binning.value

    @binning.setter
    def binning(self, value: int) -> None:
        """Set the binning of the camera.

        Args:
            value: The binning value (must be in available_binnings)

        Raises:
            ValueError: If the binning value is not supported
        """
        if value not in self.available_binnings:
            raise ValueError(
                f"Binning must be one of {self.available_binnings}, got {value}"
            )
        with self.parent.active_channel():
            self.parent.fm_settings.binning.value = value

    @property
    def available_binnings(self) -> Tuple[int, ...]:
        """Get the available binning options for the camera.

        Returns:
            A tuple of supported binning values (e.g., (1, 2, 4, 8))
        """
        with self.parent.active_channel():
            return self.parent.fm_settings.binning.available_values

    @property
    def gain(self) -> float:
        """Get the current gain setting of the camera.

        Returns:
            The current gain value
        """
        with self.parent.active_channel():
            return self.parent.connection.detector.contrast.value

    @gain.setter
    def gain(self, value: float) -> None:
        """Set the gain of the camera.

        Args:
            value: The gain value to set
        """
        with self.parent.active_channel():
            self.parent.connection.detector.contrast.value = value


class ThermoFisherLightSource(LightSource):
    """Thermo Fisher light source implementation for fluorescence microscopy.

    Provides control over light source power, emission control, and power limits
    for Thermo Fisher FLM systems.
    """

    def __init__(self, parent: "ThermoFisherFluorescenceMicroscope"):
        """Initialize the light source with parent microscope.

        Args:
            parent: The parent fluorescence microscope instance
        """
        super().__init__(parent)
        self.parent = parent

    @property
    def power(self) -> float:
        """Get the current power of the light source.

        The brightness is expressed as a percentage (0-1).

        Returns:
            The current light source power as a percentage (0.0-1.0)
        """
        with self.parent.active_channel():
            return self.parent.connection.detector.brightness.value

    @power.setter
    def power(self, value: float) -> None:
        """Set the power of the light source.

        Args:
            value: The power level as a percentage (0.0-1.0)
        """
        with self.parent.active_channel():
            # to set the power for a specific channel, we need to change to that emission type
            # to change to that emission type we need to start emission...
            # so we start, change power, stop
            emission_color = self.parent.filter_set._emission_type
            self.parent.light_source.start_emission(emission_type=emission_color)
            self.parent.connection.detector.brightness.value = value
            self.parent.light_source.stop_emission()

    @property
    def power_limits(self) -> Tuple[float, float]:
        """Get the valid power range for the light source.

        Returns:
            A tuple of (minimum, maximum) power levels as percentages (0.0-1.0)
        """
        with self.parent.active_channel():
            limits = self.parent.connection.detector.brightness.limits
            return (limits.min, limits.max)

    @property
    def is_emitting(self) -> bool:
        """Check if the light source is currently emitting light.

        Returns:
            True if the light source is actively emitting, False otherwise
        """
        with self.parent.active_channel():
            return self.parent.connection.detector.camera_settings.emission.is_on

    def start_emission(self, emission_type: CameraEmissionType) -> None:
        """Start the light source emission.

        Begins light emission from the active light source for imaging.
        """
        with self.parent.active_channel():
            self.parent.connection.detector.camera_settings.emission.start(
                emission_type=emission_type
            )

    def stop_emission(self) -> None:
        """Stop the light source emission.

        Stops light emission from the active light source.
        """
        with self.parent.active_channel():
            self.parent.connection.detector.camera_settings.emission.stop()


class ThermoFisherFilterSet(FilterSet):
    """Thermo Fisher filter set implementation for fluorescence microscopy.

    Manages excitation and emission wavelength selection for Thermo Fisher FLM systems.
    Supports reflection mode and fluorescence mode with multiple wavelength options.
    """

    def __init__(self, parent: "ThermoFisherFluorescenceMicroscope"):
        """Initialize the filter set with parent microscope.

        Args:
            parent: The parent fluorescence microscope instance
        """
        super().__init__(parent)
        self.parent = parent
        self._emission_type: CameraEmissionType = CameraEmissionType.RED

    @property
    def available_excitation_wavelengths(self) -> Tuple[float, ...]:
        """Get the available excitation wavelengths for the filter set.

        Returns:
            A tuple of available excitation wavelengths in nanometers
            (e.g., (365, 450, 550, 635))
        """
        return tuple(sorted(AVAILABLE_FM_WAVELENGTHS))

    @property
    def available_emission_wavelengths(self) -> Tuple[Union[None, str, float], ...]:
        """Get the available emission wavelengths for the filter set.

        Returns:
            A tuple of available emission wavelengths in nanometers
            Same as excitation wavelengths for this system
        """
        return (None, "Fluorescence")  # Reflection mode or fluorescence mode

    @property
    def excitation_wavelength(self) -> float:
        """Get the current excitation wavelength of the filter set.

        Returns:
            The current excitation wavelength in nanometers

        Raises:
            ValueError: If the current color setting is invalid
        """
        # This is read-only, doesn't seem to change?
        # color: str = self.parent.fm_settings.emission.type.value
        color = self._emission_type  # Use internal storage for wavelength
        if color not in COLOR_TO_WAVELENGTH:
            raise ValueError(
                f"Invalid excitation color: {color}: must be one of {list(COLOR_TO_WAVELENGTH.keys())}"
            )
        return COLOR_TO_WAVELENGTH[color]  # map to excitation wavelength

    @excitation_wavelength.setter
    def excitation_wavelength(self, value: float) -> None:
        """Set the excitation wavelength of the filter set.

        Args:
            value: The desired excitation wavelength in nanometers.
                   If an exact match is not available, the closest available
                   wavelength will be selected automatically.
        """
        # Try exact match first
        color = WAVELENGTH_TO_COLOR.get(value, None)

        # If no exact match, find the closest wavelength
        if color is None:
            available_wavelengths = list(COLOR_TO_WAVELENGTH.values())
            closest_wavelength = min(
                available_wavelengths, key=lambda x: abs(x - value)
            )
            color = WAVELENGTH_TO_COLOR[closest_wavelength]

        # .emission.type.value is read-only, so we just store it internally
        self._emission_type = color  # Store the wavelength internally
        # self.parent.fm_settings.emission.type.value = color

    @property
    def emission_wavelength(self) -> Optional[float]:
        """Get the current emission wavelength of the filter set.

        Thermo Fisher FLM does not support specific emission filters, only
        reflection or fluorescence modes with multi-band filters.

        Returns:
            None for reflection mode, or the excitation wavelength for
            fluorescence mode (as the system uses multi-band filters)
        """
        with self.parent.active_channel():
            mode = self.parent.fm_settings.filter.type.value
        if mode is CameraFilterType.REFLECTION:
            return None
        elif mode is CameraFilterType.FLUORESCENCE:
            return self.excitation_wavelength  # Multi-band filter system

    @emission_wavelength.setter
    def emission_wavelength(self, value: Optional[float]) -> None:
        """Set the emission wavelength mode of the filter set.

        Thermo Fisher FLM only supports reflection or fluorescence modes,
        not specific emission wavelength selection.

        Args:
            value: None for reflection mode, any non-None value for
                   fluorescence mode
        """
        with self.parent.active_channel():
            if value is None:
                self.parent.fm_settings.filter.type.value = CameraFilterType.REFLECTION
            else:
                self.parent.fm_settings.filter.type.value = (
                    CameraFilterType.FLUORESCENCE
                )


class ThermoFisherFluorescenceMicroscope(FluorescenceMicroscope):
    """Thermo Fisher fluorescence microscope implementation.

    Provides integrated control over Thermo Fisher FLM systems including
    Arctis and iFlm microscopes. Manages objective lens, camera, light source,
    and filter set components for fluorescence imaging.

    Attributes:
        objective: The objective lens controller
        filter_set: The filter set controller
        camera: The camera controller
        light_source: The light source controller
    """

    objective: ThermoFisherObjectiveLens
    filter_set: ThermoFisherFilterSet
    camera: ThermoFisherCamera
    light_source: ThermoFisherLightSource

    def __init__(
        self,
        parent: Optional["FibsemMicroscope"] = None,
        connection: Optional[SdbMicroscopeClient] = None,
    ):
        """Initialize the Thermo Fisher fluorescence microscope.

        Args:
            connection: Optional SDB microscope client connection.
                       If None, a new connection will be created.
        """
        super().__init__(parent=parent)

        if connection is None:
            connection = SdbMicroscopeClient()
            connection.connect()
        # TODO: Identify microscope type (Arctis vs iFlm) automatically

        self.connection = connection
        self.objective = ThermoFisherObjectiveLens(self)
        self.camera = ThermoFisherCamera(self)
        self.light_source = ThermoFisherLightSource(self)
        self.filter_set = ThermoFisherFilterSet(self)

        self._active_view = 3  # default active view for FLM (Arctis)
        self._active_device = ImagingDevice.FLUORESCENCE_LIGHT_MICROSCOPE
        # Serialises `active_channel`. The parent's lock when there is one -- the live
        # stream already takes it, so FM channel scopes and beam operations queue
        # against each other rather than interleaving. `parent` is optional, and a
        # parentless FM still needs a lock of its own to be internally consistent.
        self._channel_lock = (
            getattr(parent, "_threading_lock", None) or threading.RLock()
        )
        # How many scopes are open. Only the outermost captures and restores, so an
        # acquisition inside a run does not put the view back mid-run.
        self._channel_depth = 0
        self._restore_view: Optional[int] = None

    def set_active_channel(self):
        """Set the active imaging channel for the fluorescence microscope.

        Configures the microscope to use the fluorescence light microscope
        device and the appropriate view for FLM operations.

        Leaves the connection pointed at the FM. For anything that should hand it back
        -- which is everything except an acquisition that owns it for its duration --
        use :meth:`active_channel` instead.
        """
        self.connection.imaging.set_active_view(self._active_view)
        self.connection.imaging.set_active_device(self._active_device)

    def _channel_is_ours(self) -> bool:
        """Whether the connection is already pointed at the FM.

        The view alone answers it, for the same reason only the view is captured and
        restored: AutoScript documents `set_active_device` as changing the device *in
        the active view*, so the device follows the view rather than varying under it.

        One read, and deliberately not under the lock -- taking the lock to find out
        whether we need the lock would defeat the whole point.
        """
        return self.connection.imaging.get_active_view() == self._active_view

    @contextmanager
    def active_channel(self):
        """Point the shared connection at the FM for the length of the block, then put
        it back where it was.

        The FM and the beams are one connection with one active view and one active
        device, so whoever sets it last owns it. A property getter that sets it and
        walks away steals the microscope from whatever else is using it -- which is what
        `objective.state` did to a running workflow task: the stage is polled constantly,
        every poll read the objective for the overview tab's info bar, and each read left
        the connection on the FM. A beam acquisition that had set its own channel then
        found the FM there instead (FIB-517).

        The view is what is captured and put back, and that is sufficient: AutoScript
        documents `set_active_device` as changing the device *in the active view*, so the
        device belongs to the view and comes back with it.

        The lock covers the bookkeeping only, and deliberately **not** the body. A scope
        can span a whole tileset, and this is `FibsemMicroscope._threading_lock` -- a
        class attribute every caller in the process shares. Holding it for minutes would
        block all of them, and while the callers today are ones that should not overlap
        an FM run anyway, a shared lock held that long makes a hostage of whoever takes
        it next.

        A depth count rather than a captured local, so the view is put back once, by the
        outermost scope. A tileset holds the channel for the whole run; each tile's
        acquisition opens a scope inside it and must not restore the beam view between
        tiles.

        **Nothing to change means nothing to lock.** When the connection is already on
        the FM there is no view to set and therefore none to put back, so the scope does
        no work and takes no lock. That is not a micro-optimisation: the live stream
        re-takes this lock every frame in a loop with nothing between iterations, so a
        waiter is *starved* rather than delayed -- Python locks are not fair, and a
        thread that releases and immediately re-acquires usually wins against one that
        was already waiting. Moving the objective while streaming was unusable for
        exactly this reason, and it was unaffected by halving the number of acquisitions,
        which is what a queueing problem would have responded to.

        The fast path is safe against FIB-517 by construction: it performs **no writes**
        to the channel, so it cannot leave the connection somewhere the next beam
        operation does not expect. What it does accept is reading a value that another
        thread has since invalidated -- but the body has always run unlocked, so that
        race is not new, and a live stream re-forces the FM every frame anyway.
        """
        if self._channel_depth == 0 and self._channel_is_ours():
            yield
            return

        with self._channel_lock:
            if self._channel_depth == 0:
                self._restore_view = self.connection.imaging.get_active_view()
            self.set_active_channel()
            # Counted only once the channel is actually ours. Raising here comes out of
            # `__enter__`, so the caller's block never runs and the `finally` below never
            # runs either -- a depth left too high would mean no later scope ever
            # restored the view again. That is FIB-517 back, silently, for the rest of
            # the session, and a dropped connection is enough to cause it.
            self._channel_depth += 1
        try:
            yield
        finally:
            with self._channel_lock:
                self._channel_depth -= 1
                if self._channel_depth == 0:
                    self.connection.imaging.set_active_view(self._restore_view)

    @property
    def fm_settings(self) -> "CameraSettings":
        """Get the camera settings for the fluorescence microscope.

        Ensures the active channel is set correctly and returns the
        camera settings object for the FLM detector.

        Returns:
            The camera settings object for the active FLM channel
        """
        self.set_active_channel()
        return self.connection.detector.camera_settings
