import numpy as np
import logging

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
    def __init__(self, parent: "Client", focuser: model.Actuator = None):
        super().__init__(parent)
        self.parent = parent
        if focuser is None:
            focuser = model.getComponent(role="focus")
        self._focuser = focuser
        self._lens = model.getComponent(role="lens")

    @property
    def magnification(self):
        """Get the magnification of the objective lens."""
        return self._lens.magnification.value 

    @magnification.setter
    def magnification(self, value: float):
        """Set the magnification of the objective lens."""
        self._lens.magnification.value = value

    @property
    def numerical_aperture(self) -> float:
        """Get the numerical aperture of the objective lens."""
        return self._lens.numericalAperture.value
    
    @numerical_aperture.setter
    def numerical_aperture(self, value: float):
        """Set the numerical aperture of the objective lens."""
        if not isinstance(value, (int, float)):
            raise TypeError("Numerical aperture must be an integer or float.")
        self._lens.numericalAperture.value = value

    @property
    def position(self) -> float:
        pos = self._focuser.position.value
        return pos["z"]

    def move_relative(self, delta):
        """Move the objective lens by a relative distance."""
        if not isinstance(delta, (int, float)):
            raise TypeError("Delta must be an integer or float.")
        f = self._focuser.moveRel({"z": delta})
        f.result()

    def move_absolute(self, position: float):
        """Move the objective lens to an absolute position."""
        if not isinstance(position, (int, float)):
            raise TypeError("Position must be an integer or float.")

        f = self._focuser.moveAbs({"z": position})
        f.result()

    def insert(self):
        """Move the objective lens to the active position (inserted position)."""
        active_position = self._focuser.getMetadata()[model.MD_FAV_POS_ACTIVE]
        f = self._focuser.moveAbs(active_position)
        f.result()

    def retract(self):
        """Move the objective lens to the deactive position (retracted position)."""
        deactive_position = self._focuser.getMetadata()[model.MD_FAV_POS_DEACTIVE]
        f = self._focuser.moveAbs(deactive_position)
        f.result()

class OdemisCamera(Camera):
    def __init__(self, parent: "Client", camera: model.DigitalCamera = None):
        super().__init__(parent)
        self.parent = parent
        self._stream: FluoStream = self.parent._stream
        if camera is None:
            camera = model.getComponent(role="ccd")
        self._camera = camera
        self._resolution = self._camera.resolution.value # (width, height)
        self._pixel_size = self._camera.pixelSize.value  # (x, y) sensor pixel size
        self._offset = self._camera.getMetadata()[model.MD_BASELINE]  # offset for the camera

    # other attributes: depthOfField, readoutRate, pointSpreadFunctionSize

    def acquire_image(self, channel_settings) -> np.ndarray:
        da: List[model.DataArray]
        try:
            # QUERY: WARNING:root:Re-acquiring an image, as the one received appears 0.519948 s too early
            # We should check if anything is acquiring, and stop it?
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
        """Get the exposure time of the camera."""
        return self._camera.exposureTime.value

    @exposure_time.setter
    def exposure_time(self, value: float):
        """Set the exposure time of the camera."""
        if not isinstance(value, (int, float)):
            raise TypeError("Exposure time must be an integer or float.")
        self._camera.exposureTime.value = value

    @property
    def binning(self) -> int:
        """Get the binning of the camera."""
        return self._camera.binning.value[0]  # Assuming binning is a tuple (x, y)

    @binning.setter
    def binning(self, value: int):
        """Set the binning of the camera."""
        if value not in [1, 2, 4, 8]:
            raise ValueError(f"Binning must be one of [1, 2, 4, 8], got {value}")
        self._camera.binning.value = (value, value)

    @property
    def offset(self) -> float:
        """Get the offset of the camera."""
        return self._offset

class OdemisLightSource(LightSource):
    def __init__(self, parent: "OdemisFluorescenceMicroscope", 
                 light_source: model.Emitter = None):
        super().__init__(parent)
        self.parent = parent
        self._stream = self.parent._stream
        if light_source is None:
            light_source = model.getComponent(role="light")
        self._light_source = light_source

    @property
    def power(self) -> float:
        """Get the power of the light source."""
        return self._stream.power.value

    @power.setter
    def power(self, value: float):
        """Set the power of the light source."""
        if not isinstance(value, (int, float)):
            raise TypeError("Power must be an integer or float.")
        self._stream.power.value = value

    # spectra property?

# NOTE: we also need to determine which filter set is active, as the light source power returns the power for each emitter...
# power, excitation, and emission should represent the 'currentlly selected' rather than the full available?
# QUERY: should excitiation be a property of the filter set or the light source?

class OdemisFilterSet(FilterSet):
    def __init__(self, parent = None):
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
        """Get the available excitation wavelengths."""
        return [c[2] for c in self._stream.excitation.choices] # centre wavelengths

    @property
    def available_emission_wavelengths(self) -> Tuple[float, ...]:
        """Get the available emission wavelengths."""
        choices = []
        # return the bottom wavelength of each emission choice
        for c in self._stream.emission.choices:
            # special case for pass-through (reflection -> None)
            if isinstance(c, str) and c == model.BAND_PASS_THROUGH:
                choices.append(None)
                continue
            choices.append(c[0])
        return choices

    @property
    def excitation_wavelength(self) -> float:
        """Get the excitation wavelength of the filter set."""
        return self._stream.excitation.value[2] * 1e9 # centre wavelength (nm)

    @excitation_wavelength.setter
    def excitation_wavelength(self, value: float) -> None:
        """Set the excitation wavelength of the filter set."""
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
        """Get the emission wavelength of the filter set."""
        value = self._stream.emission.value
        if isinstance(value, str) and value == model.BAND_PASS_THROUGH:
            return None  # pass-through means reflection, so we return None
        return self._stream.emission.value[0] * 1e9 # convert from m to nm
    
    @emission_wavelength.setter
    def emission_wavelength(self, value: Optional[float]) -> None:
        """Set the emission wavelength of the filter set."""

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
    objective: OdemisObjectiveLens
    camera: OdemisCamera
    light_source: OdemisLightSource
    filter_set: OdemisFilterSet

    def __init__(self, parent: "Client"):
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

