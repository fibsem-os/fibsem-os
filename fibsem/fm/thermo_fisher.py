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

AVAILABLE_FLM_MODES = ["Fluorescence", "Reflection"]
AVAILABLE_FLM_COLORS = [
    "Blue",
    "GreenYellow",
    "Red",
    "Violet",
]  # map to excitation wavelength -> 440

COLOR_TO_WAVELENGTH = {
    "Blue": 440,
    "GreenYellow": 525,
    "Red": 590,
    "Violet": 405,
}
WAVELENGTH_TO_COLOR = {v: k for k, v in COLOR_TO_WAVELENGTH.items()}


class ThermoFisherObjectiveLens(ObjectiveLens):
    def __init__(self, parent: "ThermoFisherFluorescenceMicroscope"):
        super().__init__(parent)
        self.parent = parent
        self._magnification = 40.0

    @property
    def magnification(self):
        self.parent.set_active_channel()
        return self._magnification

    @magnification.setter
    def magnification(self, value: float):
        self.parent.set_active_channel()
        self._magnification = value

    @property
    def position(self) -> float:
        self.parent.set_active_channel()
        position = self.parent.connection.detector.camera_settings.focus.value
        return position

    def move_relative(self, delta):
        self.parent.set_active_channel()
        current_position = self.position
        new_position = current_position + delta
        self.move_absolute(new_position)

    def move_absolute(self, position: float):
        self.parent.set_active_channel()
        self.parent.connection.detector.camera_settings.focus.value = position

    def insert(self):
        self.parent.set_active_channel()
        self.parent.connection.detector.insert()

    def retract(self):
        self.parent.set_active_channel()
        self.parent.detector.retract()


class ThermoFisherCamera(Camera):
    def __init__(self, parent: "ThermoFisherFluorescenceMicroscope"):
        super().__init__(parent)
        self._binning = 1
        self.parent = parent

    # QUERY: other properties like pixel size, resolution, etc.?
    def acquire_image(self, channel_settings):
        frame_settings = GrabFrameSettings()

        self.parent.set_active_channel()
        image = self.parent.connection.imaging.grab_frame(frame_settings)

        return image.data  # AdornedImage.data -> np.ndarray

    @property
    def exposure_time(self):
        self.parent.set_active_channel()
        exposure_time = self.parent.connection.detector.camera_settings.exposure_time
        return exposure_time

    @exposure_time.setter
    def exposure_time(self, value: float):
        self.parent.set_active_channel()
        self.parent.connection.detector.camera_settings.exposure_time = value

    @property
    def binning(self):
        self.parent.set_active_channel()
        return self.parent.connection.detector.camera_settings.binning.value

    @binning.setter
    def binning(self, value: int):
        self.parent.set_active_channel()
        if value not in [1, 2, 4, 8]:
            raise ValueError(f"Binning must be one of [1, 2, 4, 8], got {value}")
        self.parent.connection.detector.camera_settings.binning.value = value


class ThermoFisherLightSource(LightSource):
    def __init__(self, parent: "ThermoFisherFluorescenceMicroscope"):
        super().__init__(parent)
        self.parent = parent
        self._power = 0.0

    @property
    def power(self):
        self.parent.set_active_channel()
        return self.parent.connection.detector.brightness.value

    @power.setter
    def power(self, value: float):
        self.parent.set_active_channel()
        self.parent.connection.detector.brightness.value = value


class ThermoFisherFilterSet(FilterSet):
    def __init__(self, parent: "ThermoFisherFluorescenceMicroscope"):
        super().__init__(parent)
        self.parent = parent
        self._excitation_wavelength = None
        self._emission_wavelength = None

    @property
    def excitation_wavelength(self) -> float:
        self.parent.set_active_channel()
        color: str = self.parent.connection.detector.camera_settings.color
        return COLOR_TO_WAVELENGTH[color]  # map to excitation wavelength

    @excitation_wavelength.setter
    def excitation_wavelength(self, value: float):
        self.parent.set_active_channel()
        color = WAVELENGTH_TO_COLOR.get(value, None)  # TODO: support closest match?
        if color is None:
            raise ValueError(
                f"Invalid excitation wavelength: {value}: must be one of {list(COLOR_TO_WAVELENGTH.keys())}"
            )
        self.parent.connection.detector.camera_settings.color = color

    @property
    def emission_wavelength(self) -> float:
        self.parent.set_active_channel()
        # Thermo Fisher FLM does not support specific emission filters, only reflection or fluorescence
        mode = self.parent.connection.detector.camera_settings.filter.type.value
        if mode == "Reflection":
            return None
        elif mode == "Fluorescence":
            return self.excitation_wavelength  # This should probably be different?

    @emission_wavelength.setter
    def emission_wavelength(self, value: float):
        self.parent.set_active_channel()
        # Thermo Fisher FLM does not support specific emission filters, only reflection or fluorescence
        if value is None:
            self.parent.connection.detector.camera_settings.filter.type.value = (
                "Reflection"
            )
        else:
            self.parent.connection.detector.camera_settings.filter.type.value = (
                "Fluorescence"
            )
            # don't think I wanna set it like this
            # self.excitation_wavelength = value  # Set excitation wavelength to match emission


class ThermoFisherFluorescenceMicroscope(FluorescenceMicroscope):
    objective: ThermoFisherObjectiveLens
    filter_sets: list[ThermoFisherFilterSet]
    camera: ThermoFisherCamera
    light_source: ThermoFisherLightSource

    def __init__(self, connection: SdbMicroscopeClient = None):
        super().__init__()

        if connection is None:
            connection = SdbMicroscopeClient()

        self.connection = connection
        self.objective = ThermoFisherObjectiveLens(self)
        self.camera = ThermoFisherCamera(self)
        self.light_source = ThermoFisherLightSource(self)
        self.filter_sets = [ThermoFisherFilterSet(self)]

        self._active_view = 3  # default active view for FLM (arctis)
        # self._active_device = 8

    def set_active_channel(self):
        self.connection.imaging.set_active_view(self._active_view)
        # set active device to FLM?
