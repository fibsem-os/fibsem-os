import logging
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
from psygnal import Signal

from fibsem.fm.structures import ChannelSettings, FluorescenceImage


class ObjectiveLens(ABC):
    def __init__(self, parent: Optional["FluorescenceMicroscope"] = None):
        self.parent = parent
        self._position: float = 0.0
        self._magnification: float = 100.0 # placeholder
        self._numerical_aperture = 0.8     # placeholder
        self._insert_position = 0.0 # z-axis
        self._retract_position = -10e-3

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
        print(f"Objective moved to new position: {self._position} (delta: {delta})")

    def move_absolute(self, position: float):
        """Move the objective lens to an absolute position."""
        self._position = position
        print(f"Objective moved to absolute position: {self._position}")

    def insert(self):
        """Insert the objective lens into the optical path."""
        # Implementation for inserting the lens
        self.move_absolute(self._insert_position)
        print(f"Objective lens inserted to position: {self._insert_position}")
        
    def retract(self):
        """Retract the objective lens from the optical path."""
        # Implementation for retracting the lens
        self.move_absolute(self._retract_position)
        print(f"Objective lens retracted to position: {self._retract_position}")


class Camera(ABC):
    def __init__(self, parent: Optional["FluorescenceMicroscope"] = None):
        self.parent = parent
        self._exposure_time: float = 0.1  # seconds
        self._binning: int = 1
        self._gain: float = 1.0
        self._offset: float = 0.0
        self._pixel_size: Tuple[float, float] = (100e-9, 100e-9)  # in meters
        self._resolution: Tuple[int, int] = (1024, 1024)  # default resolution
        super().__init__()

    def acquire_image(self, channel_settings: ChannelSettings) -> np.ndarray:
        """Acquire an image with the specified channel settings."""

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
        self._power: float = 100  # Power level of the light source
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
        self._excitation_wavelength: float = 488.0
        self._emission_wavelength: Optional[float] = None  # None = reflection
        super().__init__()

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

    def __init__(self):
        super().__init__()

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
        # move excitation filter to the correct position
        self.filter_set.excitation_wavelength = channel_settings.excitation_wavelength
        # move emission filter to the correct position
        self.filter_set.emission_wavelength = channel_settings.emission_wavelength

        # Set light source power
        self.set_power(channel_settings.power)

        # Set camera settings
        # self.set_binning(channel_settings.binning)
        self.set_exposure_time(channel_settings.exposure_time)

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

    def acquire_image(self, channel_settings: Optional[ChannelSettings] = None) -> FluorescenceImage:
        """Acquire an image using the channel settings."""
        if channel_settings is not None:
            self.set_channel(channel_settings)
        image = self.camera.acquire_image(channel_settings)
        md = self.get_metadata()
        img = FluorescenceImage(data=image, metadata=md)
        return img

    def get_metadata(self) -> dict:
        """Get metadata for the current configuration."""
        metadata = {
            "objective": {
                "position": self.objective.position,
                "magnification": self.objective.magnification,
                "numerical_aperture": self.objective.numerical_aperture,
            },
            "filter_set": {
                "excitation_wavelength": self.filter_set.excitation_wavelength,
                "emission_wavelength": self.filter_set.emission_wavelength,
                },
            "camera": {
                "exposure_time": self.camera.exposure_time,
                "binning": self.camera.binning,
                "gain": self.camera.gain,
                "offset": self.camera.offset,
                "pixel_size": self.camera.pixel_size,
                "resolution": self.camera.resolution,
                "sensor_pixel_size": self.camera._pixel_size,  # Original pixel size before binning
                "sensor_resolution": self.camera._resolution,  # Full resolution before binning
            },
            "light_source": {"power": self.light_source.power},
            "acquisition_date": datetime.now().isoformat(),
        }

        # TODO: microscope-state metadata
        return metadata

    def start_acquisition(self) -> None:
        """Start the image acquisition process."""
        if self.is_acquiring:
            logging.warning("Acquisition thread is already running.")
            return

        # reset stop event if needed
        self._stop_acquisition_event.clear()

        # start acquisition thread
        self._acquisition_thread = threading.Thread(
            target=self._acquisition_worker,
            args=(),
            daemon=True
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

    def _acquisition_worker(self):
        """Worker thread for image acquisition."""

        # TODO: add lock
        try:
            while True:
                if self._stop_acquisition_event.is_set():
                    break

                # acquire image using current beam settings
                image = self.acquire_image()
                # emit the acquired image
                self.acquisition_signal.emit(image)

        except Exception as e:
            logging.error(f"Error in acquisition worker: {e}")
