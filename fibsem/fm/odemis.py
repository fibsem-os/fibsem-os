import numpy as np
import logging
import time

from typing import List, Optional, Tuple
from fibsem.fm.microscope import (
    Camera,
    FilterSet,
    FluorescenceMicroscope,
    LightSource,
    ObjectiveLens,
)
from fibsem.microscopes.odemis_microscope import add_odemis_path

add_odemis_path()

from odemis import model
from odemis.acq.acqmng import acquire
from odemis.acq.stream import FluoStream

# NOTES: needed to install shapely, pylibtiff, and odemis

class OdemisObjectiveLens(ObjectiveLens):
    """Odemis objective lens implementation for fluorescence microscopy.
    
    Provides control over objective lens positioning, state management, and configuration
    for Odemis-compatible fluorescence microscope systems. Implements metadata caching
    for improved performance when accessing frequently used position values.
    """
    
    def __init__(self, parent: "Client", focuser: model.Actuator = None):
        """Initialize the objective lens with optional focuser component.
        
        Args:
            parent: The parent client instance
            focuser: Optional focuser actuator component. If None, will be
                    automatically detected using the 'focus' role.
        """
        super().__init__(parent)
        self.parent = parent
        if focuser is None:
            focuser = model.getComponent(role="focus")
        self._focuser = focuser
        self._lens = model.getComponent(role="lens")
        
        # Cache metadata on initialization
        self._metadata_cache = {}
        self._cache_metadata()
        self._focus_position = self.active_position

    def _cache_metadata(self):
        """Cache frequently accessed metadata for improved performance.
        
        Caches the active and deactive positions from the focuser metadata
        to reduce hardware communication overhead during repeated access.
        
        Note:
            If caching fails, the cache will be empty and methods will
            fall back to direct hardware access.
        """
        try:
            metadata = self._focuser.getMetadata()
            self._metadata_cache = {
                'active_position': metadata[model.MD_FAV_POS_ACTIVE],
                'deactive_position': metadata[model.MD_FAV_POS_DEACTIVE],
                'last_updated': time.time()
            }
        except Exception as e:
            logging.warning(f"Failed to cache metadata: {e}")
            self._metadata_cache = {}

    @property
    def active_position(self):
        """Get the active (inserted) position of the objective lens.
        
        Uses cached metadata when available for improved performance,
        with fallback to direct hardware access if needed.
        
        Returns:
            The active position coordinates as a dictionary with 'z' key,
            or None if the position cannot be determined.
        """
        try:
            return self._metadata_cache['active_position']
        except (KeyError, TypeError):
            # Fallback to direct hardware call if cache fails
            try:
                return self._focuser.getMetadata()[model.MD_FAV_POS_ACTIVE]
            except Exception as e:
                logging.error(f"Failed to get active position: {e}")
                return None

    @property
    def deactive_position(self):
        """Get the deactive (retracted) position of the objective lens.
        
        Uses cached metadata when available for improved performance,
        with fallback to direct hardware access if needed.
        
        Returns:
            The deactive position coordinates as a dictionary with 'z' key,
            or None if the position cannot be determined.
        """
        try:
            return self._metadata_cache['deactive_position']
        except (KeyError, TypeError):
            # Fallback to direct hardware call if cache fails
            try:
                return self._focuser.getMetadata()[model.MD_FAV_POS_DEACTIVE]
            except Exception as e:
                logging.error(f"Failed to get deactive position: {e}")
                return None

    @property
    def magnification(self):
        """Get the current magnification of the objective lens.
        
        Returns:
            The magnification value as a float
        """
        return self._lens.magnification.value 

    @magnification.setter
    def magnification(self, value: float):
        """Set the magnification of the objective lens.
        
        Args:
            value: The desired magnification value
        """
        self._lens.magnification.value = value

    @property
    def numerical_aperture(self) -> float:
        """Get the numerical aperture of the objective lens.
        
        Returns:
            The numerical aperture value as a float
        """
        return self._lens.numericalAperture.value
    
    @numerical_aperture.setter
    def numerical_aperture(self, value: float):
        """Set the numerical aperture of the objective lens.
        
        Args:
            value: The numerical aperture value (must be numeric)
            
        Raises:
            TypeError: If the value is not a number
        """
        if not isinstance(value, (int, float)):
            raise TypeError("Numerical aperture must be an integer or float.")
        self._lens.numericalAperture.value = value

    @property
    def position(self) -> float:
        """Get the current focus position of the objective lens.
        
        Returns:
            The current z-axis position of the focuser in meters
        """
        pos = self._focuser.position.value
        return pos["z"]

    def move_relative(self, delta):
        """Move the objective lens by a relative distance.
        
        Args:
            delta: The distance to move in meters (positive = towards sample)
            
        Raises:
            TypeError: If delta is not a number
        """
        if not isinstance(delta, (int, float)):
            raise TypeError("Delta must be an integer or float.")
        f = self._focuser.moveRel({"z": delta})
        f.result()

    def move_absolute(self, position: float):
        """Move the objective lens to an absolute focus position.
        
        Args:
            position: The target z-axis position in meters
            
        Raises:
            TypeError: If position is not a number
        """
        if not isinstance(position, (int, float)):
            raise TypeError("Position must be an integer or float.")

        f = self._focuser.moveAbs({"z": position})
        f.result()

    def insert(self):
        """Move the objective lens to the active (inserted) position.
        
        Moves the objective lens to the predefined active position for imaging.
        Uses cached position data when available for improved performance.
        """
        active_position = self.active_position
        if active_position is not None:
            f = self._focuser.moveAbs(active_position)
            f.result()

    def retract(self):
        """Move the objective lens to the deactive (retracted) position.
        
        Moves the objective lens to the predefined deactive position for safety.
        Uses cached position data when available for improved performance.
        """
        deactive_position = self.deactive_position
        if deactive_position is not None:
            f = self._focuser.moveAbs(deactive_position)
            f.result()

    def state(self) -> str:
        """Get the current state of the objective lens.
        
        Determines the lens state by comparing the current position with
        the cached active and deactive positions.
        
        Returns:
            The lens state: 'INSERTED', 'RETRACTED', or 'UNKNOWN'
            
        Note:
            Returns 'UNKNOWN' if position data is unavailable or if the
            lens is between the active and deactive positions.
        """
        try:
            position = self._focuser.position.value
            active_position = self.active_position
            deactive_position = self.deactive_position
            
            if active_position is None or deactive_position is None:
                return "UNKNOWN"
                
            if position["z"] > active_position["z"]:
                return "INSERTED"
            elif position["z"] < deactive_position["z"]:
                return "RETRACTED"
            else:
                return "UNKNOWN"
        except Exception as e:
            logging.error(f"Failed to get objective lens state: {e}")
            return "UNKNOWN"

class OdemisCamera(Camera):
    """Odemis camera implementation for fluorescence microscopy.
    
    Provides control over camera settings and image acquisition for Odemis-compatible
    fluorescence microscope systems. Manages exposure time, binning, and image capture
    through the Odemis streaming interface.
    """
    
    def __init__(self, parent: "Client", camera: Optional[model.DigitalCamera] = None):
        """Initialize the camera with optional camera component.
        
        Args:
            parent: The parent client instance
            camera: Optional camera component. If None, will be automatically
                   detected using the 'ccd' role.
        """
        super().__init__(parent)
        self.parent = parent
        self._stream: FluoStream = self.parent._stream
        if camera is None:
            camera: model.DigitalCamera = model.getComponent(role="ccd")
        self._camera = camera
        camera_md = self._camera.getMetadata()
        self._resolution = self._camera.resolution.value # (width, height)
        self._pixel_size = camera_md[model.MD_PIXEL_SIZE]  # (x, y) sensor pixel size
        self._offset = camera_md[model.MD_BASELINE]  # offset for the camera

    # other attributes: depthOfField, readoutRate, pointSpreadFunctionSize

    def acquire_image(self) -> np.ndarray:
        """Acquire a single image from the camera.
        
        Uses the Odemis acquisition manager to capture an image through
        the configured fluorescence stream with current camera settings.
        
        Returns:
            A numpy array containing the image data, or None if acquisition fails
            
        Note:
            May show timing warnings if rapid successive acquisitions are attempted.
            Consider checking if acquisition is already in progress before calling.
        """
        da: List[model.DataArray]
        try:
            # May show timing warnings for rapid successive acquisitions
            # TODO: Check if acquisition is already in progress
            f = acquire([self._stream])
            da, err = f.result()
            if err:
                raise RuntimeError(f"Error acquiring image: {err}")
        except Exception as e:
            logging.warning(f"Error acquiring image: {e}")
            return None
        return da[0] # model.DataArray -> np.ndarray

    # QUERY: migrate to using the camera dataflow interface?
    # .camera._camera.data.get()

    @property
    def exposure_time(self) -> float:
        """Get the current exposure time of the camera.
        
        Returns:
            The exposure time in seconds
        """
        return self._camera.exposureTime.value

    @exposure_time.setter
    def exposure_time(self, value: float):
        """Set the exposure time of the camera.
        
        Args:
            value: The exposure time in seconds (must be numeric)
            
        Raises:
            TypeError: If the value is not a number
        """
        if not isinstance(value, (int, float)):
            raise TypeError("Exposure time must be an integer or float.")
        self._camera.exposureTime.value = value

    @property
    def binning(self) -> int:
        """Get the current binning setting of the camera.
        
        Returns:
            The binning value (assumes symmetric binning, returns x-axis value)
        """
        return self._camera.binning.value[0]  # Assuming binning is a tuple (x, y)

    @binning.setter
    def binning(self, value: int):
        """Set the binning of the camera.
        
        Args:
            value: The binning value (must be 1, 2, 4, or 8)
            
        Raises:
            ValueError: If the binning value is not supported
            
        Note:
            Sets symmetric binning (same value for both x and y axes)
        """
        if value not in [1, 2, 4, 8]:
            raise ValueError(f"Binning must be one of [1, 2, 4, 8], got {value}")
        self._camera.binning.value = (value, value)

    @property
    def offset(self) -> float:
        """Get the baseline offset of the camera.
        
        Returns:
            The camera's baseline offset value from metadata
        """
        return self._offset

class OdemisLightSource(LightSource):
    """Odemis light source implementation for fluorescence microscopy.
    
    Provides control over light source power and intensity for Odemis-compatible
    fluorescence microscope systems. Manages power control through the fluorescence
    stream interface.
    """
    
    def __init__(self, parent: "OdemisFluorescenceMicroscope", 
                 light_source: model.Emitter = None):
        """Initialize the light source with optional emitter component.
        
        Args:
            parent: The parent fluorescence microscope instance
            light_source: Optional light source emitter component. If None,
                         will be automatically detected using the 'light' role.
        """
        super().__init__(parent)
        self.parent = parent
        self._stream = self.parent._stream
        if light_source is None:
            light_source = model.getComponent(role="light")
        self._light_source = light_source

    @property
    def power(self) -> float:
        """Get the current power of the light source.
        
        Returns:
            The power level as a float (units depend on light source configuration)
        """
        return self._stream.power.value

    @power.setter
    def power(self, value: float):
        """Set the power of the light source.
        
        Args:
            value: The power level to set (must be numeric)
            
        Raises:
            TypeError: If the value is not a number
        """
        if not isinstance(value, (int, float)):
            raise TypeError("Power must be an integer or float.")
        self._stream.power.value = value

    def power_limits(self) -> Tuple[float, float]:
        """Get the valid power range for the light source.
        
        Returns:
            A tuple of (minimum, maximum) power levels
        """
        return self._stream.power.range

    # spectra property?

# NOTE: we also need to determine which filter set is active, as the light source power returns the power for each emitter...
# power, excitation, and emission should represent the 'currentlly selected' rather than the full available?
# QUERY: should excitiation be a property of the filter set or the light source?

class OdemisFilterSet(FilterSet):
    """Odemis filter set implementation for fluorescence microscopy.
    
    Manages excitation and emission wavelength selection for Odemis-compatible
    fluorescence microscope systems. Handles wavelength conversion between
    nanometers (external interface) and meters (internal Odemis units).
    
    Note:
        Wavelengths are stored internally in meters but exposed in nanometers
        for user convenience. None values represent reflection/pass-through mode.
    """
    
    def __init__(self, parent = None):
        """Initialize the filter set with parent microscope.
        
        Args:
            parent: The parent fluorescence microscope instance
        """
        super().__init__(parent)
        self.parent = parent
        self._stream: FluoStream = self.parent._stream
        self._excitation_wavelength = None
        self._emission_wavelength = None

    # NOTE on internal representation:
    # images are acquired via the public stream interface, as it contains the necessary components

    # excitation wavelength is 5D tuple: (bottom 99%, bottom 25%, centre, top 25%, top 99%)
    # emission wavelength is a 2D tuple: (bottom, top)
    # to set the exciation/emission wavelngth you need to set this tuple

    # externally, we pass in the centre wavelength for excitation and the bottom wavelength for emission
    # this may change to use a str or enum for the excitation and emission wavelengths in the future, to be more structured. 
    # None -> reflection

    @property
    def available_excitation_wavelengths(self) -> Tuple[float, ...]:
        """Get the available excitation wavelengths for the filter set.
        
        Returns:
            A tuple of available excitation wavelengths in nanometers
            (converted from Odemis internal meter units)
        """
        return [c[2] * 1e9 for c in self._stream.excitation.choices] # centre wavelengths

    @property
    def available_emission_wavelengths(self) -> Tuple[float, ...]:
        """Get the available emission wavelengths for the filter set.
        
        Returns:
            A tuple of available emission wavelengths in nanometers.
            Includes None for reflection/pass-through mode.
            
        Note:
            Returns the bottom wavelength of each emission filter range.
        """
        choices = []
        # return the bottom wavelength of each emission choice
        for c in self._stream.emission.choices:
            # special case for pass-through (reflection -> None)
            if isinstance(c, str) and c == model.BAND_PASS_THROUGH:
                choices.append(None)
                continue
            choices.append(c[0]* 1e9)  # convert from m to nm
        return choices

    @property
    def excitation_wavelength(self) -> float:
        """Get the current excitation wavelength of the filter set.
        
        Returns:
            The center excitation wavelength in nanometers
            (converted from Odemis internal meter units)
        """
        return self._stream.excitation.value[2] * 1e9 # centre wavelength (nm)

    @excitation_wavelength.setter
    def excitation_wavelength(self, value: float) -> None:
        """Set the excitation wavelength of the filter set.
        
        Args:
            value: The desired excitation wavelength in nanometers
            
        Raises:
            TypeError: If the value is not a number
            ValueError: If no suitable wavelength is available
            
        Note:
            Automatically selects the closest available wavelength if an
            exact match is not found.
        """
        if not isinstance(value, (int, float)):
            raise TypeError("Excitation wavelength must be an integer or float.")

        value *= 1e-9
        closest_excitation = min(self.available_excitation_wavelengths, key=lambda x: abs(x - value))
        if closest_excitation is None:
            raise ValueError(f"Excitation wavelength {value} is not available. Available wavelengths: {self.available_excitation_wavelengths}")

        # find the index of the closest excitation wavelength
        idx = self.available_excitation_wavelengths.index(closest_excitation)
        choices = tuple(self._stream.excitation.choices)

        logging.info(f"Setting excitation wavelength to index: {idx}, value: {choices[idx]}, requested: {value}")
        self._stream.excitation.value = choices[idx]

    @property
    def emission_wavelength(self) -> Optional[float]:
        """Get the current emission wavelength of the filter set.
        
        Returns:
            The bottom emission wavelength in nanometers, or None for
            reflection/pass-through mode
            
        Note:
            Converts from Odemis internal meter units to nanometers.
        """
        value = self._stream.emission.value
        if isinstance(value, str) and value == model.BAND_PASS_THROUGH:
            return None  # pass-through means reflection, so we return None
        return self._stream.emission.value[0] * 1e9 # convert from m to nm
    
    @emission_wavelength.setter
    def emission_wavelength(self, value: Optional[float]) -> None:
        """Set the emission wavelength of the filter set.
        
        Args:
            value: The desired emission wavelength in nanometers, or None
                  for reflection/pass-through mode
                  
        Raises:
            ValueError: If the requested wavelength or pass-through mode
                       is not available
                       
        Note:
            Setting to None enables reflection/pass-through mode.
            Automatically selects the closest available wavelength.
            Performance note: Setting emission wavelength can be slow.
        """

        # None means reflection, so we set the emission to pass-through (reflection)
        if value is None:
            choices = tuple(self._stream.emission.choices)
            if model.BAND_PASS_THROUGH not in choices:
                raise ValueError(f"Pass-through (reflection) is not available in the current filter set. Available choices: {choices}")
            self._stream.emission.value = model.BAND_PASS_THROUGH
            return

        # filter out None values from available wavelengths
        value *= 1e-9  # convert from nm to m
        available_wavelengths = [wl for wl in self.available_emission_wavelengths if wl is not None]
        closest_emission = min(available_wavelengths, key=lambda x: abs(x - value))
        if closest_emission is None:
            raise ValueError(f"Emission wavelength {value} is not available. Available wavelengths: {self.available_emission_wavelengths}")

        # TODO: only set the emission wavelength if it is different from the current value, it's slow to set it

        # find the index of the closest emission wavelength
        idx = self.available_emission_wavelengths.index(closest_emission)
        choices = tuple(self._stream.emission.choices)
        logging.info(f"Setting emission wavelength to index: {idx}, value: {choices[idx]}, requested: {value}")
        self._stream.emission.value = choices[idx]


class OdemisFluorescenceMicroscope(FluorescenceMicroscope):
    """Odemis fluorescence microscope implementation.
    
    Provides integrated control over Odemis-compatible fluorescence microscope systems.
    Manages objective lens, camera, light source, and filter set components through
    the Odemis framework using a unified fluorescence stream interface.
    
    Attributes:
        objective: The objective lens controller
        camera: The camera controller
        light_source: The light source controller  
        filter_set: The filter set controller
        
    Note:
        Requires Odemis to be properly installed and configured with the
        appropriate hardware components (ccd, light, filter, focus).
    """
    
    objective: OdemisObjectiveLens
    camera: OdemisCamera
    light_source: OdemisLightSource
    filter_set: OdemisFilterSet

    def __init__(self, parent: "FibsemMicroscope"):
        """Initialize the Odemis fluorescence microscope.
        
        Args:
            parent: The parent FibsemMicroscope instance
            
        Note:
            Automatically detects and configures hardware components using
            Odemis role-based component discovery (ccd, light, filter, focus).
        """
        super().__init__()

        self.parent = parent
        camera = model.getComponent(role="ccd")
        light_source = model.getComponent(role="light")
        light_filter = model.getComponent(role="filter")
        focuser = model.getComponent(role="focus")

        self._stream = FluoStream(
            name="fm-stream",
            detector=camera,
            dataflow=camera.data,
            emitter=light_source,
            em_filter=light_filter,
            focuser=focuser,
        )

        self.objective = OdemisObjectiveLens(self, focuser=focuser)
        self.camera = OdemisCamera(self, camera=camera)
        self.light_source = OdemisLightSource(self, light_source=light_source)
        self.filter_set = OdemisFilterSet(self)

