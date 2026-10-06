"""The FM API over the Thermo FM devices (Arctis, Hydra, iFLM), and the AutoScript
names and objective configurations the drivers take from here.

The devices are ``fibsem.devices.drivers.autoscript_fm``.
"""

from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional, Union

from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.enumerations import (  # noqa: F401 (the drivers')
    CameraEmissionType,
    CameraFilterType,
    ImagingDevice,
    ImagingState,
)
from autoscript_sdb_microscope_client.structures import (  # noqa: F401 (the drivers')
    GrabFrameSettings,
)

from fibsem.fm.microscope import (
    FilterSet,
    FluorescenceMicroscope,
    ObjectiveLens,
)
from fibsem.fm.structures import REFLECTION

if TYPE_CHECKING:
    from fibsem.fm.structures import CameraSettings
    from fibsem.microscope import FibsemMicroscope

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


class DeviceThermoFisherObjectiveLens(ObjectiveLens):
    """The FM API's objective over the Thermo objective device, with what the old
    Thermo objective added: its configured focus position, and homing."""

    def __init__(self, device, channel, parent=None):
        super().__init__(device, parent=parent)
        self._channel = channel
        self._focus_position = DEFAULT_CONFIGURATION["focus_position"]

    def is_homed(self) -> bool:
        with self._channel.scope():
            return self._channel.connection.detector.is_homed

    def home(self) -> None:
        with self._channel.scope():
            self._channel.connection.detector.home()
        self._notify_moved()


class DeviceThermoFisherFilterSet(FilterSet):
    """The FM API's filter set over the Thermo filter set device. Thermo names its
    multi-band fluorescence filter by the excitation wavelength, as the old filter set
    did, where the other drivers name it "Fluorescence"."""

    @property
    def emission_wavelength(self) -> Optional[float]:
        if self._device.emission_filter.get_value() == REFLECTION:
            return None
        return self.excitation_wavelength

    @emission_wavelength.setter
    def emission_wavelength(self, value: Optional[Union[float, str]]) -> None:
        FilterSet.emission_wavelength.fset(self, value)


class DeviceThermoFisherFluorescenceMicroscope(FluorescenceMicroscope):
    """The FM API over the Thermo FM devices (``fibsem.devices.drivers.autoscript_fm``).

    What the old ``ThermoFisherFluorescenceMicroscope`` did, through its devices: the channel
    scope is the devices' own (``AutoscriptFMChannel``), so a tileset that holds the
    FM's view holds it for every device inside it, and live view is the old fast
    acquisition, pulled.
    """

    objective: DeviceThermoFisherObjectiveLens
    filter_set: DeviceThermoFisherFilterSet

    def __init__(self, devices, parent: Optional["FibsemMicroscope"] = None):
        super().__init__(devices, parent=parent)
        self._channel = devices["fm"]._channel
        self.objective = DeviceThermoFisherObjectiveLens(
            devices["objective"], self._channel, parent=self
        )
        self.filter_set = DeviceThermoFisherFilterSet(
            devices["filter_set"], parent=self
        )

    @property
    def connection(self) -> SdbMicroscopeClient:
        return self._channel.connection

    def set_active_channel(self) -> None:
        self._channel.set_active_channel()

    @contextmanager
    def active_channel(self):
        with self._channel.scope():
            yield

    @property
    def fm_settings(self) -> "CameraSettings":
        return self._channel.settings()

    def _metadata_for_frame(self, frame_metadata):
        md = super()._metadata_for_frame(frame_metadata)
        for channel in md.channels:
            if isinstance(channel.emission_wavelength, str):
                # The multi-band filter, named by the excitation as the old one is.
                channel.emission_wavelength = channel.excitation_wavelength
        return md
