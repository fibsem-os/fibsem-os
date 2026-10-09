"""The FM API over the Odemis FM devices, as a METEOR runs it.

The devices are ``fibsem.drivers.odemis.devices``; this module is the FM API's parts
over them, and the odemis imports the drivers take from here, after the odemis path is
set up.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

from fibsem.drivers.odemis import add_odemis_path
from fibsem.fm.microscope import (
    FilterSet,
    FluorescenceMicroscope,
    ObjectiveLens,
)
from fibsem.util.timestamps import iso_from_posix

add_odemis_path()

# The drivers import odemis from here.
from odemis import model  # noqa: E402
from odemis.util import fluo  # noqa: E402, F401

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope

# NOTES: needed to install shapely, pylibtiff, and odemis

# position tolerance for deciding the objective is at the active/deactive position
OBJECTIVE_POSITION_ATOL = 100e-6  # m


def _frame_metadata_from_data(data) -> Optional[dict]:
    """Translate odemis DataArray metadata to the neutral frame-metadata keys
    consumed by FluorescenceMicroscope._construct_image.

    Odemis DataArrays carry metadata captured at exposure time (pixel size,
    acquisition date, exposure time), which is authoritative over a state
    snapshot taken when the frame is processed.
    """
    md = getattr(data, "metadata", None)
    if not md:
        return None

    frame_metadata = {}
    pixel_size = md.get(model.MD_PIXEL_SIZE)
    if pixel_size is not None:
        frame_metadata["pixel_size"] = tuple(pixel_size)
    acquisition_date = md.get(model.MD_ACQ_DATE)
    if acquisition_date is not None:
        # POSIX, so the instant is known: written with the offset (FIB-1190).
        frame_metadata["acquisition_date"] = iso_from_posix(acquisition_date)
    exposure_time = md.get(model.MD_EXP_TIME)
    if exposure_time is not None:
        frame_metadata["exposure_time"] = exposure_time
    return frame_metadata or None


def _odemis_bands_nm(choice: Any) -> Tuple[Tuple[float, float], ...]:
    """The bands of an Odemis emission choice in nm, lowest first. A single band is
    a tuple of edges in metres (2 or 5 values: its first and last are the edges); a
    multi-band filter is a tuple of such bands."""
    bands = choice if isinstance(choice[0], (tuple, list)) else (choice,)
    return tuple(sorted((band[0] * 1e9, band[-1] * 1e9) for band in bands))


class DeviceOdemisObjectiveLens(ObjectiveLens):
    """The FM API's objective over the Odemis objective device, focusing where the old
    objective does: odemis's favourite inserted position."""

    def __init__(self, device, parent=None):
        super().__init__(device, parent=parent)
        self._focus_position = device.focus_position


class DeviceOdemisFilterSet(FilterSet):
    """The FM API's filter set over the Odemis filter set device. A "Fluorescence"
    emission (TFS-style channel settings) is the band odemis matches to the current
    excitation, as on the old filter set."""

    @property
    def emission_bands(self) -> Dict[float, Tuple[Tuple[float, float], ...]]:
        """Each emission filter's bands in nm, keyed by its bottom edge."""
        return {f.low: f.bands for f in self._emission_filters() if f.low is not None}

    @property
    def emission_wavelength(self) -> Optional[float]:
        return FilterSet.emission_wavelength.fget(self)

    @emission_wavelength.setter
    def emission_wavelength(self, value: Optional[Union[float, str]]) -> None:
        if isinstance(value, str):
            self._device.select_fluorescence()
            return
        FilterSet.emission_wavelength.fset(self, value)


class DeviceOdemisFluorescenceMicroscope(FluorescenceMicroscope):
    """The FM API over the Odemis FM devices (``fibsem.drivers.odemis.devices``).
    Live view is the light on and the camera acquiring, with each frame pulled."""

    objective: DeviceOdemisObjectiveLens
    filter_set: DeviceOdemisFilterSet

    def __init__(self, devices, parent: Optional["FibsemMicroscope"] = None):
        super().__init__(devices, parent=parent)
        self.objective = DeviceOdemisObjectiveLens(devices["objective"], parent=self)
        self.filter_set = DeviceOdemisFilterSet(devices["filter_set"], parent=self)
