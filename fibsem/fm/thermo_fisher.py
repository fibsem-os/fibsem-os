from typing import Tuple

import numpy as np
from autoscript_sdb_microscope_client.structures import (
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

REFLECTION_MODE = "Reflection"
FLUORESCENCE_MODE = "Fluorescence"


AVAILABLE_FM_MODES = [REFLECTION_MODE, FLUORESCENCE_MODE]
COLOR_TO_WAVELENGTH = {
    "Violet": 365,
    "Blue": 450,
    "GreenYellow": 550,
    "Red": 635,
}
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
    def __init__(self, parent: "ThermoFisherFluorescenceMicroscope"):
        super().__init__(parent)
        self.parent = parent
        self._magnification = DEFAULT_CONFIGURATION["magnification"]
        self._numerical_aperture = DEFAULT_CONFIGURATION["numerical_aperture"]
        self._pixel_size = DEFAULT_CONFIGURATION["pixel_size"]
        self._resolution = DEFAULT_CONFIGURATION["resolution"]

    @property
    def magnification(self) -> float:
        self.parent.set_active_channel()
        return self._magnification

    @magnification.setter
    def magnification(self, value: float):
        self.parent.set_active_channel()
        self._magnification = value

    @property
    def position(self) -> float:
        position = self.parent.fm_settings.focus.value
        return position

    def move_relative(self, delta: float):
        self.parent.set_active_channel()
        current_position = self.position
        new_position = current_position + delta
        self.move_absolute(new_position)

    def move_absolute(self, position: float):
        self.parent.fm_settings.focus.value = position

    def insert(self) -> None:
        self.parent.set_active_channel()
        self.parent.connection.detector.insert()

    def retract(self) -> None:
        self.parent.set_active_channel()
        self.parent.connection.detector.retract()


class ThermoFisherCamera(Camera):
    def __init__(self, parent: "ThermoFisherFluorescenceMicroscope"):
        super().__init__(parent)
        self.parent = parent

    # QUERY: other properties like pixel size, resolution, etc.?
    def acquire_image(self) -> np.ndarray:
        frame_settings = GrabFrameSettings()

        self.parent.set_active_channel()
        image = self.parent.connection.imaging.grab_frame(frame_settings)

        return image.data  # AdornedImage.data -> np.ndarray

    @property
    def exposure_time(self)  -> float:
        return self.parent.fm_settings.exposure_time.value

    @exposure_time.setter
    def exposure_time(self, value: float) -> None:
        self.parent.fm_settings.exposure_time.value = value

    @property
    def binning(self) -> int:
        return self.parent.fm_settings.binning.value

    @binning.setter
    def binning(self, value: int) -> None:
        if value not in [1, 2, 4, 8]:
            raise ValueError(f"Binning must be one of [1, 2, 4, 8], got {value}")
        self.parent.fm_settings.binning.value = value


class ThermoFisherLightSource(LightSource):
    def __init__(self, parent: "ThermoFisherFluorescenceMicroscope"):
        super().__init__(parent)
        self.parent = parent

    @property
    def power(self)  -> float:
        # QUERY: is this in W or percentage?
        self.parent.set_active_channel()
        return self.parent.connection.detector.brightness.value

    @power.setter
    def power(self, value: float) -> None:
        self.parent.set_active_channel()
        self.parent.connection.detector.brightness.value = value


class ThermoFisherFilterSet(FilterSet):
    def __init__(self, parent: "ThermoFisherFluorescenceMicroscope"):
        super().__init__(parent)
        self.parent = parent

    @property
    def available_excitation_wavelengths(self) -> Tuple[float, ...]:
        return tuple(sorted(AVAILABLE_FM_WAVELENGTHS))

    @property
    def available_emission_wavelengths(self) -> Tuple[float, ...]:
        return tuple(sorted(AVAILABLE_FM_WAVELENGTHS))

    @property
    def excitation_wavelength(self) -> float:
        color: str = self.parent.fm_settings.emission.type
        return COLOR_TO_WAVELENGTH[color] # map to excitation wavelength

    @excitation_wavelength.setter
    def excitation_wavelength(self, value: float) -> None:
        color = WAVELENGTH_TO_COLOR.get(value, None)  # TODO: support closest match?
        if color is None:
            raise ValueError(
                f"Invalid excitation wavelength: {value}: must be one of {list(COLOR_TO_WAVELENGTH.keys())}"
            )
        self.parent.fm_settings.emission.type = color

    @property
    def emission_wavelength(self) -> float:
        # Thermo Fisher FLM does not support specific emission filters, only reflection or fluorescence
        # uses a multi-band filter, so we should return the map of excitation -> emission filter
        # not sure what these values are for now, so for now we just return the excitation wavelength
        mode = self.parent.fm_settings.filter.type.value
        if mode == REFLECTION_MODE:
            return None
        elif mode == FLUORESCENCE_MODE:
            return self.excitation_wavelength  # This should probably be different?

    @emission_wavelength.setter
    def emission_wavelength(self, value: float) -> None:
        # Thermo Fisher FLM does not support specific emission filters, only reflection or fluorescence
        if value is None:
            self.parent.fm_settings.filter.type.value = REFLECTION_MODE
        else:
            self.parent.fm_settings.filter.type.value = FLUORESCENCE_MODE

class ThermoFisherFluorescenceMicroscope(FluorescenceMicroscope):
    objective: ThermoFisherObjectiveLens
    filter_set: ThermoFisherFilterSet
    camera: ThermoFisherCamera
    light_source: ThermoFisherLightSource

    def __init__(self, connection: SdbMicroscopeClient = None):
        super().__init__()

        if connection is None:
            connection = SdbMicroscopeClient()
        # possible to identify the microscope type? e.g. arctis or iflm?

        self.connection = connection
        self.objective = ThermoFisherObjectiveLens(self)
        self.camera = ThermoFisherCamera(self)
        self.light_source = ThermoFisherLightSource(self)
        self.filter_set = ThermoFisherFilterSet(self)

        self._active_view = 3  # default active view for FLM (arctis)
        # self._active_device = 8

    def set_active_channel(self):
        self.connection.imaging.set_active_view(self._active_view)
        # set active device to FLM?

    @property
    def fm_settings(self) -> 'CameraSettings':
        """Return the camera settings for the current active channel."""
        self.set_active_channel()
        return self.connection.detector.camera_settings

