from typing import Optional
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from pprint import pprint

from fibsem import utils
from dataclasses import dataclass, field
from fibsem.fm.structures import ChannelSettings, FluorescenceImage


class ObjectiveLens(ABC):
    def __init__(self, parent: Optional["FluorescenceMicroscope"] = None):
        self.parent = parent
        self._position: float = 0.0
        self._magnification: float = (
            100.0  # placeholder value, should be overridden by subclasses
        )
        self._na = 1.4  # Numerical Aperture, placeholder value

    @property
    def magnification(self) -> float:
        """Return the magnification of the objective lens."""
        return self._magnification

    @magnification.setter
    def magnification(self, value: float):
        """Set the magnification of the objective lens."""
        self._magnification = value

    @property
    def position(self) -> float:
        """Return the current position of the objective lens. (z-axis)"""
        return self._position

    def move_relative(self, delta: float):
        """Move the objective lens relative to its current position."""
        self._position += delta
        print(
            f"Objective lens moved to new position: {self._position} microns (delta: {delta} microns)"
        )

    def move_absolute(self, position: float):
        """Move the objective lens to an absolute position."""
        self._position = position
        print(f"Objective lens moved to absolute position: {self._position} microns")

    def insert(self):
        """Insert the objective lens into the optical path."""
        # Implementation for inserting the lens
        pass

    def retract(self):
        """Retract the objective lens from the optical path."""
        # Implementation for retracting the lens
        pass


class Camera(ABC):
    def __init__(self, parent: Optional["FluorescenceMicroscope"] = None):
        self.parent = parent
        self._exposure_time: float = 100.0  # Default exposure time in milliseconds
        self._binning: int = 1
        self._gain: float = 1.0
        self._offset: float = 0.0
        self._resolution: Tuple[int, int] = (1024, 1024)  # default resolution
        super().__init__()

    # @abstractmethod
    def acquire_image(self, channel_settings: ChannelSettings) -> np.ndarray:
        """Acquire an image with the specified channel settings."""

        # get min and max values for the image
        min_value = np.iinfo(np.uint16).min  # 0 for uint16
        max_value = np.iinfo(np.uint16).max  # 65535 for uint16
        return np.random.randint(
            min_value, max_value, size=self._resolution, dtype=np.uint16
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
    filter_sets: List[FilterSet]
    camera: "Camera"
    light_source: "LightSource"

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return f"{self.__class__.__name__}(objective={self.objective}, filter_sets={self.filter_sets}, camera={self.camera}, light_source={self.light_source})"

    def set_channel(self, channel_settings: ChannelSettings):
        """Set the channel settings for the fluorescence microscope."""
        if not self.filter_sets:
            raise ValueError("No filter sets available.")
        if not self.light_source:
            raise ValueError("Light source is not set.")
        if not self.camera:
            raise ValueError("Camera is not set.")

        # set the filter wheel to the correct settings? QUERY: better way to do this?
        # move excitation filter to the correct position
        self.filter_sets[
            0
        ].excitation_wavelength = channel_settings.excitation_wavelength
        # move emission filter to the correct position
        self.filter_sets[0].emission_wavelength = channel_settings.emission_wavelength

        # Set light source power
        self.set_power(channel_settings.power)

        # Set camera settings
        self.set_binning(channel_settings.binning)
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

    def acquire_image(self, channel_settings: ChannelSettings) -> FluorescenceImage:
        """Acquire an image using the channel settings."""
        if not self.camera:
            raise ValueError("Camera is not set.")
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
            },
            "filter_sets": [
                {
                    "excitation_wavelength": fs.excitation_wavelength,
                    "emission_wavelength": fs.emission_wavelength,
                }
                for fs in self.filter_sets
            ],
            "camera": {
                "exposure_time": self.camera.exposure_time,
                "binning": self.camera.binning,
                "gain": self.camera.gain,
                "offset": self.camera.offset,
            },
            "light_source": {"power": self.light_source.power},
        }
        # QUERY: FOV, PIXELSIZE, RESOLUTION
        return metadata
