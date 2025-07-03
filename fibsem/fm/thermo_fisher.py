from typing import Tuple, Optional, Union, Dict, Any

import numpy as np
from autoscript_sdb_microscope_client.structures import (
    GrabFrameSettings,
)
from autoscript_sdb_microscope_client.enumerations import CameraEmissionType, CameraFilterType, ImagingDevice

from fibsem.fm.microscope import (
    Camera,
    FilterSet,
    FluorescenceMicroscope,
    LightSource,
    ObjectiveLens,
)
from fibsem.microscope import SdbMicroscopeClient

COLOR_TO_WAVELENGTH = {
    CameraEmissionType.VIOLET: 365,
    CameraEmissionType.BLUE: 450,
    CameraEmissionType.GREEN_YELLOW: 550,
    CameraEmissionType.RED: 635,
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
    "magnification": 100.0,
    "numerical_aperture": 0.75,
    "working_distance": 4e-3,
    "pixel_size": (100e-9, 100e-9),
    "resolution": (1024, 1024),
}

IFLM_CONFIGURATION = {
    "magnification": 20.0,
    "numerical_aperture": 0.7,
    "working_distance": 1.3e-3,
    "pixel_size": (100e-9, 100e-9),
    "resolution": (1024, 1024),
}
DEFAULT_CONFIGURATION = ARCTIS_CONFIGURATION  # Default to ARCTIS configuration

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
        self._pixel_size = DEFAULT_CONFIGURATION["pixel_size"]
        self._resolution = DEFAULT_CONFIGURATION["resolution"]

    @property
    def magnification(self) -> float:
        """Get the magnification of the objective lens.
        
        Returns:
            The current magnification value
        """
        self.parent.set_active_channel()
        return self._magnification

    @magnification.setter
    def magnification(self, value: float):
        """Set the magnification of the objective lens.
        
        Args:
            value: The magnification value to set
        """
        self.parent.set_active_channel()
        self._magnification = value

    @property
    def position(self) -> float:
        """Get the current focus position of the objective lens.
        
        Returns:
            The current focus position in metres
        """
        position = self.parent.fm_settings.focus.value
        return position

    @property
    def limits(self) -> Tuple[float, float]:
        """Get the focus position limits of the objective lens.
        
        Returns:
            A tuple of (minimum, maximum) focus position limits in metres
        """
        self.parent.set_active_channel()
        limits = self.parent.fm_settings.focus.limits
        return (limits.min, limits.max)

    def move_relative(self, delta: float):
        """Move the objective lens by a relative distance.
        
        Args:
            delta: The distance to move in metres (positive = towards sample)
        """
        self.parent.set_active_channel()
        current_position = self.position
        new_position = current_position + delta
        self.move_absolute(new_position)

    def move_absolute(self, position: float):
        """Move the objective lens to an absolute focus position.
        
        Args:
            position: The target focus position in metres
        """
        self.parent.fm_settings.focus.value = position

    def insert(self) -> None:
        """Insert the objective lens into the working position.
        
        Moves the objective lens to the active/inserted position for imaging.
        """
        self.parent.set_active_channel()
        self.parent.connection.detector.insert()

    def retract(self) -> None:
        """Retract the objective lens from the working position.
        
        Moves the objective lens to the inactive/retracted position for safety.
        """
        self.parent.set_active_channel()
        self.parent.connection.detector.retract()

    def state(self) -> str:
        """Get the current state of the objective lens.
        
        Returns:
            The objective lens state (e.g., 'INSERTED', 'RETRACTED', 'MOVING')
        """
        self.parent.set_active_channel()
        return self.parent.connection.detector.state.value
    
    def is_homed(self) -> bool:
        """Check if the objective lens is in the homed position.
        
        Returns:
            True if the objective lens is homed, False otherwise
        """
        self.parent.set_active_channel()
        return self.parent.connection.detector.is_homed

    def home(self) -> None:
        """Home the objective lens to its reference position.
        
        Moves the objective lens to its home/reference position for calibration.
        """
        self.parent.set_active_channel()
        self.parent.connection.detector.home()


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

    def acquire_image(self) -> np.ndarray:
        """Acquire a single image from the camera.
        
        Uses the current camera settings (exposure time, binning) to capture
        an image from the fluorescence microscope.
        
        Returns:
            A numpy array containing the image data
        """
        frame_settings = GrabFrameSettings()
        # Uses current camera settings for binning and exposure time

        self.parent.set_active_channel()
        image = self.parent.connection.imaging.grab_frame(frame_settings)

        return image.data  # AdornedImage.data -> np.ndarray

    @property
    def exposure_time(self)  -> float:
        """Get the current exposure time of the camera.
        
        Returns:
            The exposure time in seconds
        """
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
            raise ValueError(f"Exposure time must be between {limits[0]} and {limits[1]}, got {value}")
        self.parent.fm_settings.exposure_time.value = value

    @property
    def exposure_time_limits(self) -> Tuple[float, float]:
        """Get the valid exposure time range for the camera.
        
        Returns:
            A tuple of (minimum, maximum) exposure times in seconds
        """
        limits = self.parent.fm_settings.exposure_time.limits
        return (limits.min, limits.max)

    @property
    def binning(self) -> int:
        """Get the current binning setting of the camera.
        
        Returns:
            The current binning value (e.g., 1, 2, 4, 8)
        """
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
            raise ValueError(f"Binning must be one of {self.available_binnings}, got {value}")
        self.parent.fm_settings.binning.value = value

    @property
    def available_binnings(self) -> Tuple[int, ...]:
        """Get the available binning options for the camera.
        
        Returns:
            A tuple of supported binning values (e.g., (1, 2, 4, 8))
        """
        return self.parent.fm_settings.binning.available_values

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
    def power(self)  -> float:
        """Get the current power of the light source.
        
        The brightness is expressed as a percentage (0-1).
        
        Returns:
            The current light source power as a percentage (0.0-1.0)
        """
        self.parent.set_active_channel()
        return self.parent.connection.detector.brightness.value

    @power.setter
    def power(self, value: float) -> None:
        """Set the power of the light source.
        
        Args:
            value: The power level as a percentage (0.0-1.0)
        """
        self.parent.set_active_channel()
        self.parent.connection.detector.brightness.value = value

    @property
    def power_limits(self) -> Tuple[float, float]:
        """Get the valid power range for the light source.
        
        Returns:
            A tuple of (minimum, maximum) power levels as percentages (0.0-1.0)
        """
        self.parent.set_active_channel()
        limits = self.parent.connection.detector.brightness.limits
        return (limits.min, limits.max)

    @property
    def is_emitting(self) -> bool:
        """Check if the light source is currently emitting light.
        
        Returns:
            True if the light source is actively emitting, False otherwise
        """
        self.parent.set_active_channel()
        return self.parent.connection.detector.camera_settings.emission.is_on

    def start_emission(self) -> None:
        """Start the light source emission.
        
        Begins light emission from the active light source for imaging.
        """
        self.parent.set_active_channel()
        self.parent.connection.detector.camera_settings.emission.start()

    def stop_emission(self) -> None:
        """Stop the light source emission.
        
        Stops light emission from the active light source.
        """
        self.parent.set_active_channel()
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

    @property
    def available_excitation_wavelengths(self) -> Tuple[float, ...]:
        """Get the available excitation wavelengths for the filter set.
        
        Returns:
            A tuple of available excitation wavelengths in nanometers
            (e.g., (365, 450, 550, 635))
        """
        return tuple(sorted(AVAILABLE_FM_WAVELENGTHS))

    @property
    def available_emission_wavelengths(self) -> Tuple[float, ...]:
        """Get the available emission wavelengths for the filter set.
        
        Returns:
            A tuple of available emission wavelengths in nanometers
            Same as excitation wavelengths for this system
        """
        return tuple(sorted(AVAILABLE_FM_WAVELENGTHS))

    @property
    def excitation_wavelength(self) -> float:
        """Get the current excitation wavelength of the filter set.
        
        Returns:
            The current excitation wavelength in nanometers
            
        Raises:
            ValueError: If the current color setting is invalid
        """
        color: str = self.parent.fm_settings.emission.type.value
        if color not in COLOR_TO_WAVELENGTH:
            raise ValueError(
                f"Invalid excitation color: {color}: must be one of {list(COLOR_TO_WAVELENGTH.keys())}"
            )
        return COLOR_TO_WAVELENGTH[color] # map to excitation wavelength

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
            closest_wavelength = min(available_wavelengths, key=lambda x: abs(x - value))
            color = WAVELENGTH_TO_COLOR[closest_wavelength]
            
        self.parent.fm_settings.emission.type.value = color

    @property
    def emission_wavelength(self) -> Optional[float]:
        """Get the current emission wavelength of the filter set.
        
        Thermo Fisher FLM does not support specific emission filters, only
        reflection or fluorescence modes with multi-band filters.
        
        Returns:
            None for reflection mode, or the excitation wavelength for 
            fluorescence mode (as the system uses multi-band filters)
        """
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
        if value is None:
            self.parent.fm_settings.filter.type.value = CameraFilterType.REFLECTION
        else:
            self.parent.fm_settings.filter.type.value = CameraFilterType.FLUORESCENCE

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

    def __init__(self, connection: Optional[SdbMicroscopeClient] = None):
        """Initialize the Thermo Fisher fluorescence microscope.
        
        Args:
            connection: Optional SDB microscope client connection.
                       If None, a new connection will be created.
        """
        super().__init__()

        if connection is None:
            connection = SdbMicroscopeClient()
        # TODO: Identify microscope type (Arctis vs iFlm) automatically

        self.connection = connection
        self.objective = ThermoFisherObjectiveLens(self)
        self.camera = ThermoFisherCamera(self)
        self.light_source = ThermoFisherLightSource(self)
        self.filter_set = ThermoFisherFilterSet(self)

        self._active_view = 3  # default active view for FLM (Arctis)
        self._active_device = ImagingDevice.FLUORESCENCE_LIGHT_MICROSCOPE

    def set_active_channel(self):
        """Set the active imaging channel for the fluorescence microscope.
        
        Configures the microscope to use the fluorescence light microscope
        device and the appropriate view for FLM operations.
        """
        self.connection.imaging.set_active_view(self._active_view)
        self.connection.imaging.set_active_device(self._active_device)

    @property
    def fm_settings(self) -> 'CameraSettings':
        """Get the camera settings for the fluorescence microscope.
        
        Ensures the active channel is set correctly and returns the
        camera settings object for the FLM detector.
        
        Returns:
            The camera settings object for the active FLM channel
        """
        self.set_active_channel()
        return self.connection.detector.camera_settings

