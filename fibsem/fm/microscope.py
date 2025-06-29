import logging
import threading
import time
from abc import ABC
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from psygnal import Signal

from fibsem.fm.structures import (
    ChannelSettings,
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope

EXCITATION_WAVELENGTHS = [365, 450, 550, 635]  # in nm, example wavelengths
EMISSION_WAVELENGTHS = [365, 450, 550, 635, None]  # in nm, example wavelengths

SIM_OBJECTIVE_MAGNIFICATION = 100.0  # placeholder for simulation
SIM_OBJECTIVE_NA = 0.8
SIM_OBJECTIVE_INSERT_POSITION = 0.0  # z-axis position for inserting the objective lens
SIM_OBJECTIVE_RETRACT_POSITION = -10e-3  # z-axis position for retracting the objective lens
SIM_OBJECTIVE_POSITION_LIMITS = (-12e-3, 5e-3)  # z-axis limits for the objective lens


SIM_CAMERA_EXPOSURE_TIME = 0.1  # seconds
SIM_CAMERA_BINNING = 1
SIM_CAMERA_GAIN = 1.0
SIM_CAMERA_OFFSET = 0.0
SIM_CAMERA_PIXEL_SIZE = (100e-9, 100e-9)  # in meters (100 nm)
SIM_CAMERA_RESOLUTION = (512, 512)  # default resolution


class ObjectiveLens(ABC):
    def __init__(self, parent: Optional["FluorescenceMicroscope"] = None):
        self.parent = parent
        self._position: float =  SIM_OBJECTIVE_RETRACT_POSITION  # start at retracted position
        self._magnification: float = SIM_OBJECTIVE_MAGNIFICATION
        self._numerical_aperture = SIM_OBJECTIVE_NA
        self._insert_position = SIM_OBJECTIVE_INSERT_POSITION
        self._retract_position = SIM_OBJECTIVE_RETRACT_POSITION

    @property
    def magnification(self) -> float:
        """Return the magnification of the objective lens."""
        return self._magnification

    @property
    def numerical_aperture(self) -> float:
        """Return the numerical aperture of the objective lens."""
        return self._numerical_aperture

    @property
    def position(self) -> float:
        """Return the current position of the objective lens. (z-axis)"""
        return self._position

    def move_relative(self, delta: float):
        """Move the objective lens relative to its current position."""
        self._position += delta
        logging.info(f"Objective moved to new position: {self._position} (delta: {delta})")

    def move_absolute(self, position: float):
        """Move the objective lens to an absolute position."""
        self._position = position
        logging.info(f"Objective moved to absolute position: {self._position}")

    def insert(self):
        """Insert the objective lens into the optical path."""
        # Implementation for inserting the lens
        self.move_absolute(self._insert_position)
        logging.info(f"Objective lens inserted to position: {self._insert_position}")

    def retract(self):
        """Retract the objective lens from the optical path."""
        # Implementation for retracting the lens
        self.move_absolute(self._retract_position)
        logging.info(f"Objective lens retracted to position: {self._retract_position}")

    @property
    def limits(self) -> Tuple[float, float]:
        """Return the limits of the objective lens position."""
        return SIM_OBJECTIVE_POSITION_LIMITS

class Camera(ABC):
    def __init__(self, parent: Optional["FluorescenceMicroscope"] = None):
        self.parent = parent
        self._exposure_time: float = SIM_CAMERA_EXPOSURE_TIME
        self._binning: int = SIM_CAMERA_BINNING
        self._gain: float = SIM_CAMERA_GAIN
        self._offset: float = SIM_CAMERA_OFFSET
        self._pixel_size: Tuple[float, float] = SIM_CAMERA_PIXEL_SIZE
        self._resolution: Tuple[int, int] = SIM_CAMERA_RESOLUTION
        super().__init__()

    def acquire_image(self) -> np.ndarray:
        """Acquire an image with the current channel settings."""

        time.sleep(self.exposure_time)  # Simulate exposure time in seconds

        # get min and max values for the image
        min_value = np.iinfo(np.uint16).min  # 0 for uint16
        max_value = np.iinfo(np.uint16).max  # 65535 for uint16
        return np.random.randint(
            min_value, max_value, size=self.resolution, dtype=np.uint16
        )

    @property
    def exposure_time(self) -> float:
        """Return the current exposure time of the camera."""
        return self._exposure_time

    @exposure_time.setter
    def exposure_time(self, value: float):
        """Set the exposure time of the camera."""
        self._exposure_time = value

    @property
    def binning(self) -> int:
        """Return the current binning of the camera."""
        return self._binning

    @binning.setter
    def binning(self, value: int):
        """Set the binning of the camera."""
        if value < 1:
            raise ValueError("Binning must be at least 1.")
        self._binning = value

    @property
    def gain(self) -> float:
        """Return the current gain of the camera."""
        return self._gain

    @gain.setter
    def gain(self, value: float):
        """Set the gain of the camera."""
        if value < 0:
            raise ValueError("Gain must be non-negative.")
        self._gain = value

    @property
    def offset(self) -> float:
        """Return the current offset of the camera."""
        return self._offset

    @offset.setter
    def offset(self, value: float):
        """Set the offset of the camera."""
        if value < 0:
            raise ValueError("Offset must be non-negative.")
        self._offset = value

    @property
    def pixel_size(self) -> Tuple[float, float]:
        """Return the current pixel size of the camera accounting for binning."""
        return (
            self._pixel_size[0] * self.binning,
            self._pixel_size[1] * self.binning,
        )

    @property
    def resolution(self) -> Tuple[int, int]:
        """Return the current resolution of the camera accounting for binning."""
        return self._resolution[0] // self.binning, self._resolution[1] // self.binning


class LightSource(ABC):
    def __init__(self, parent: Optional["FluorescenceMicroscope"] = None):
        self.parent = parent
        self._power: float = 0.1 # W
        super().__init__()

    @property
    def power(self) -> float:
        """Return the current power of the light source."""
        return self._power

    @power.setter
    def power(self, value: float):
        """Set the power of the light source."""
        self._power = value


class FilterSet(ABC):
    def __init__(self, parent: Optional["FluorescenceMicroscope"] = None):
        self.parent = parent
        self._excitation_wavelength: float = EXCITATION_WAVELENGTHS[0]  # default to first wavelength
        self._emission_wavelength: Optional[float] = None  # None = reflection
        super().__init__()

    @property
    def available_excitation_wavelengths(self) -> Tuple[float, ...]:
        """Return a tuple of available excitation wavelengths."""
        return tuple(EXCITATION_WAVELENGTHS)

    @property
    def available_emission_wavelengths(self) -> Tuple[float, ...]:
        """Return a tuple of available emission wavelengths."""
        # For simplicity, we assume emission wavelengths are the same as excitation
        return tuple(EMISSION_WAVELENGTHS)

    @property
    def excitation_wavelength(self) -> float:
        """Return the current excitation wavelength."""
        return self._excitation_wavelength

    @excitation_wavelength.setter
    def excitation_wavelength(self, value: float):
        """Set the excitation wavelength."""
        self._excitation_wavelength = value

    @property
    def emission_wavelength(self) -> Optional[float]:
        """Return the current emission wavelength."""
        return self._emission_wavelength

    @emission_wavelength.setter
    def emission_wavelength(self, value: Optional[float]):
        """Set the emission wavelength."""
        self._emission_wavelength = value


class FluorescenceMicroscope(ABC):
    objective: ObjectiveLens
    filter_set: FilterSet
    camera: Camera
    light_source: LightSource

    # live acquisition
    acquisition_signal = Signal(FluorescenceImage)
    _stop_acquisition_event = threading.Event()
    _acquisition_thread: threading.Thread = None

    def __init__(self, parent: Optional['FibsemMicroscope'] = None):
        super().__init__()

        self.parent = parent

        self.channel_name: str = "channel-01"
        self.objective = ObjectiveLens(parent=self)
        self.filter_set = FilterSet(parent=self)
        self.camera = Camera(parent=self)
        self.light_source = LightSource(parent=self)

    def __repr__(self):
        return f"{self.__class__.__name__}(objective={self.objective}, filter_set={self.filter_set}, camera={self.camera}, light_source={self.light_source})"

    @property
    def is_acquiring(self) -> bool:
        """Check if the microscope is currently acquiring an image."""
        return self._acquisition_thread and self._acquisition_thread.is_alive()

    def set_channel(self, channel_settings: ChannelSettings):
        """Set the channel settings for the fluorescence microscope."""
        if not self.filter_set:
            raise ValueError("No filter sets available.")
        if not self.light_source:
            raise ValueError("Light source is not set.")
        if not self.camera:
            raise ValueError("Camera is not set.")

        # set the filter wheel to the correct settings? QUERY: better way to do this?
        self.filter_set.excitation_wavelength = channel_settings.excitation_wavelength
        self.filter_set.emission_wavelength = channel_settings.emission_wavelength

        # Set light source power
        self.set_power(channel_settings.power)

        # Set camera settings
        self.set_exposure_time(channel_settings.exposure_time)

        # set channel name
        self.set_channel_name(channel_settings.name)

    def set_channel_name(self, name: str):
        """Set the name of the current channel."""
        self.channel_name = name

    def set_binning(self, binning: int):
        """Set the binning of the camera."""
        if not self.camera:
            raise ValueError("Camera is not set.")
        self.camera.binning = binning

    def set_exposure_time(self, exposure_time: float):
        """Set the exposure time of the camera."""
        if not self.camera:
            raise ValueError("Camera is not set.")
        self.camera.exposure_time = exposure_time

    def set_power(self, power: float):
        """Set the power of the light source."""
        if not self.light_source:
            raise ValueError("Light source is not set.")
        self.light_source.power = power

    def acquire_image(
        self, channel_settings: Optional[ChannelSettings] = None
    ) -> FluorescenceImage:
        """Acquire an image using the channel settings."""
        if channel_settings is not None:
            self.set_channel(channel_settings)
        image = self.camera.acquire_image()
        md = self.get_metadata()
        img = FluorescenceImage(data=image, metadata=md)
        return img
    
    def get_metadata(self) -> FluorescenceImageMetadata:
        """Get structured metadata for the current microscope configuration."""

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
            resolution=tuple(self.camera.resolution),
            stage_position=stage_position,
            channels=[channel_metadata]
        )

    def start_acquisition(self, channel_settings: Optional[ChannelSettings] = None) -> None:
        """Start the image acquisition process."""
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
        """Stop the image acquisition process."""
        if self._stop_acquisition_event and not self._stop_acquisition_event.is_set():
            self._stop_acquisition_event.set()
            if self._acquisition_thread:
                self._acquisition_thread.join(timeout=2)
            # Disconnect signal handler
            # self.acquisition_signal.disconnect()

    def _acquisition_worker(self, channel_settings: Optional[ChannelSettings] = None):
        """Worker thread for image acquisition."""

        # TODO: add lock
        try:
            if channel_settings is not None:
                self.set_channel(channel_settings)
            logging.info("Starting acquisition worker thread.")
            while True:
                if self._stop_acquisition_event.is_set():
                    break

                # acquire image using current beam settings
                image = self.acquire_image()
                # emit the acquired image
                self.acquisition_signal.emit(image)

        except Exception as e:
            logging.error(f"Error in acquisition worker: {e}")
