import logging
import threading
import time
from abc import ABC
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from psygnal import Signal

from fibsem.fm.structures import (
    ChannelSettings,
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)
from fibsem.util.draw_numbers import draw_text

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope

EXCITATION_WAVELENGTHS = [365, 450, 550, 635]  # in nm, example wavelengths
EMISSION_WAVELENGTHS = [None, "Fluorescence"]  # in nm, example wavelengths

SIM_OBJECTIVE_MAGNIFICATION = 100.0  # placeholder for simulation
SIM_OBJECTIVE_NA = 0.8
SIM_OBJECTIVE_INSERT_POSITION = 6.0e-3  # z-axis position for inserting the objective lens
SIM_OBJECTIVE_RETRACT_POSITION = -10e-3  # z-axis position for retracting the objective lens
SIM_OBJECTIVE_POSITION_LIMITS = (-12e-3, 10e-3)  # z-axis limits for the objective lens
SIM_OBJECTIVE_FOCUS_POSITION = 8.0e-3

SIM_CAMERA_EXPOSURE_TIME = 0.1  # seconds
SIM_CAMERA_BINNING = 1
SIM_CAMERA_GAIN = 1.0
SIM_CAMERA_OFFSET = 0.0
SIM_CAMERA_PIXEL_SIZE = (100e-9, 100e-9)  # in meters (100 nm)
SIM_CAMERA_RESOLUTION = (1024, 1024)  # default resolution

class ObjectiveLens(ABC):
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
        self._position: float =  SIM_OBJECTIVE_RETRACT_POSITION  # start at retracted position
        self._magnification: float = SIM_OBJECTIVE_MAGNIFICATION
        self._numerical_aperture = SIM_OBJECTIVE_NA
        self._insert_position = SIM_OBJECTIVE_INSERT_POSITION
        self._retract_position = SIM_OBJECTIVE_RETRACT_POSITION
        self._focus_position: Optional[float] = SIM_OBJECTIVE_FOCUS_POSITION

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
        return self._position

    @property
    def focus_position(self) -> Optional[float]:
        """Get the focus position of the objective lens.
        
        Returns:
            The focus position in meters, or None if not set
        """
        return self._focus_position
    
    @focus_position.setter
    def focus_position(self, position: float):
        """Set the focus position of the objective lens.
        
        Args:
            position: The focus position in meters
        """
        self._focus_position = position
        logging.info(f"Objective focus position set to: {self._focus_position * 1e3} mm")

    def move_relative(self, delta: float):
        """Move the objective lens by a relative distance.
        
        Args:
            delta: The distance to move in meters (positive = towards sample)
        """
        self._position += delta
        logging.info(f"Objective moved to new position: {self._position * 1e3} mm (delta: {delta * 1e3} mm)")

    def move_absolute(self, position: float):
        """Move the objective lens to an absolute z-axis position.
        
        Args:
            position: The target position in meters
        """
        self._position = position
        logging.info(f"Objective moved to absolute position: {self._position * 1e3} mm")

    def insert(self):
        """Insert the objective lens into the working position for imaging.
        
        Moves the objective lens to the predefined insertion position,
        typically at or near the sample focal plane.
        """
        self.move_absolute(self._insert_position)
        logging.info(f"Objective lens inserted to position: {self._insert_position}")

    def retract(self):
        """Retract the objective lens to a safe position away from the sample.
        
        Moves the objective lens to the predefined retraction position
        to prevent damage during stage movements or sample changes.
        """
        self.move_absolute(self._retract_position)
        logging.info(f"Objective lens retracted to position: {self._retract_position}")

    @property
    def limits(self) -> Tuple[float, float]:
        """Get the z-axis position limits of the objective lens.
        
        Returns:
            A tuple of (minimum, maximum) positions in meters
        """
        return SIM_OBJECTIVE_POSITION_LIMITS

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
        time.sleep(self.exposure_time)  # Simulate exposure time in seconds

        # get min and max values for the image
        min_value = np.iinfo(np.uint16).min  # 0 for uint16
        max_value = np.iinfo(np.uint16).max  # 65535 for uint16
        noise = np.random.randint(
            min_value, max_value, size=self.resolution[::-1], dtype=np.uint16
        )
        if not self._use_counter:
            return noise

        # Simulate a simple image with a number drawn in the center
        mod = self._index % 10  # cycle through digits 0-9
        
        # Cache the draw_number image by mod and resolution
        cache_key = (mod, self.resolution)
        if cache_key not in self._number_cache:
            self._number_cache[cache_key] = draw_text(f"FM{mod}", size=(256, 256), thickness=64, image_shape=self.resolution[::-1])

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
            value: The binning factor (must be >= 1)
            
        Raises:
            ValueError: If binning is less than 1
        """
        if value < 1:
            raise ValueError("Binning must be at least 1.")
        self._binning = value

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
        self._power: float = 0.1 # W
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
        self._excitation_wavelength: float = EXCITATION_WAVELENGTHS[0]  # default to first wavelength
        self._emission_wavelength: Optional[float] = None  # None = reflection
        super().__init__()

    @property
    def available_excitation_wavelengths(self) -> Tuple[float, ...]:
        """Get the available excitation wavelengths for the filter set.
        
        Returns:
            A tuple of supported excitation wavelengths in nanometers
        """
        return tuple(EXCITATION_WAVELENGTHS)

    @property
    def available_emission_wavelengths(self) -> Tuple[float, ...]:
        """Get the available emission wavelengths for the filter set.
        
        Returns:
            A tuple of supported emission wavelengths in nanometers.
            Includes None for reflection/pass-through mode.
        """
        return tuple(EMISSION_WAVELENGTHS)

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
    def emission_wavelength(self) -> Optional[float]:
        """Get the current emission wavelength of the filter set.
        
        Returns:
            The emission wavelength in nanometers, or None for reflection mode
        """
        return self._emission_wavelength

    @emission_wavelength.setter
    def emission_wavelength(self, value: Optional[float]):
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

    # live acquisition
    acquisition_signal = Signal(FluorescenceImage)
    _stop_acquisition_event = threading.Event()
    _acquisition_thread: Optional[threading.Thread] = None

    def __init__(self, parent: Optional['FibsemMicroscope'] = None):
        """Initialize the fluorescence microscope with default components.
        
        Args:
            parent: Optional parent FibsemMicroscope instance for stage access
        """
        super().__init__()

        self.parent = parent

        self.channel_name: str = "channel-01"
        self.objective = ObjectiveLens(parent=self)
        self.filter_set = FilterSet(parent=self)
        self.camera = Camera(parent=self)
        self.light_source = LightSource(parent=self)
        self._last_updated_at: Optional[float] = datetime.now()

    def __repr__(self):
        """Return a string representation of the fluorescence microscope.
        
        Returns:
            A string showing the microscope class and component status
        """
        return f"{self.__class__.__name__}(objective={self.objective}, filter_set={self.filter_set}, camera={self.camera}, light_source={self.light_source})"

    @property
    def is_acquiring(self) -> bool:
        """Check if the microscope is currently in live acquisition mode.
        
        Returns:
            True if live acquisition is active, False otherwise
        """
        if not self._acquisition_thread:
            return False
        return self._acquisition_thread and self._acquisition_thread.is_alive()

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

        # set channel name
        self.set_channel_name(channel_settings.name)

    def set_channel_name(self, name: str):
        """Set the name identifier for the current imaging channel.
        
        Args:
            name: The channel name (e.g., 'DAPI', 'GFP', 'channel-01')
        """
        self.channel_name = name

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
        if channel_settings is not None:
            self.set_channel(channel_settings)
        data = self.camera.acquire_image()
        img = self._construct_image(data)
        return img

    def _construct_image(self, data: np.ndarray) -> FluorescenceImage:
        """Construct a FluorescenceImage from raw data with associated metadata."""
        md = self.get_metadata()
        img = FluorescenceImage(data=data, metadata=md)

        now = datetime.now()
        if self._last_updated_at is not None:
            time_since_last_update = now - self._last_updated_at
            if time_since_last_update > timedelta(seconds=1):
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
            excitation_wavelength=self.filter_set.excitation_wavelength,
            emission_wavelength=self.filter_set.emission_wavelength,
            power=self.light_source.power,
            exposure_time=self.camera.exposure_time,
            gain=self.camera.gain,
            offset=self.camera.offset,
            binning=self.camera.binning,
            objective_position=self.objective.position,
            objective_magnification=self.objective.magnification,
            objective_numerical_aperture=self.objective.numerical_aperture
        )
        
        # Create complete image metadata
        return FluorescenceImageMetadata(
            acquisition_date=datetime.now().isoformat(),
            pixel_size_x=self.camera.pixel_size[0],
            pixel_size_y=self.camera.pixel_size[1],
            resolution=(self.camera.resolution[0], self.camera.resolution[1]),
            stage_position=stage_position,
            channels=[channel_metadata]
        )

    def start_acquisition(self, channel_settings: Optional[ChannelSettings] = None) -> None:
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
        if self.is_acquiring:
            logging.warning("Acquisition thread is already running.")
            return

        # reset stop event if needed
        self._stop_acquisition_event.clear()

        # start acquisition thread
        self._acquisition_thread = threading.Thread(
            target=self._acquisition_worker, args=(channel_settings, ), daemon=True
        )
        self._acquisition_thread.start()

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

                # acquire and emit image using current settings
                self.acquire_image()

        except Exception as e:
            logging.error(f"Error in acquisition worker: {e}")
