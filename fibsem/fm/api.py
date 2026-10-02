"""Today's FM API over FM devices, wherever the devices are.

``DeviceFluorescenceMicroscope`` is a ``FluorescenceMicroscope`` whose parts forward
to the FM's devices (``fibsem.devices.fm``): the camera, light source, filter set
and objective, and the ``fm`` group that runs a channel. It doesn't know which driver
built them, or whether they run in this process or on another computer, so the FM
UI, acquisition and workflows use it unchanged either way:

- DeviceDemo's FM is this over the Demo FM devices (``fibsem.devices.drivers.demo``);
- ``RemoteFluorescenceMicroscope`` (``fibsem.fm.remote``) is this over remote devices,
  plus connecting to their server.

Every property is a live read, or a write through the old API path
(``write_through``): nothing is cached behind the caller's back, so a guard reading
``fm.objective.state`` asks the device, and a remote one fails closed with
``RemoteDeviceUnreachable`` when its server can't be reached. Acquiring a channel is
one command on the ``fm`` group, run next to the hardware.

What belongs to the session rather than the hardware stays here: the objective's
saved focus position, the channel name and colour, and the image transform.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from fibsem.fm.microscope import (
    Camera,
    FilterSet,
    FluorescenceMicroscope,
    LightSource,
    ObjectiveLens,
)
from fibsem.fm.structures import (
    REFLECTION,
    ChannelSettings,
    EmissionFilter,
    FluorescenceImage,
    emission_filter_for,
    objective_state_name,
    same_emission_value,
)

if TYPE_CHECKING:
    from fibsem.devices.core import BoundParameter, Device
    from fibsem.microscope import FibsemMicroscope

FM_DEVICE_NAMES = ("fm", "camera", "light_source", "filter_set", "objective")
"""The devices an FM is made of: the four parts and the group."""


def _param(device: Device, name: str) -> BoundParameter:
    """A part's parameter, failing closed while a remote part's server has never
    answered.

    A remote part built offline has no parameters yet, so a guard reading
    ``fm.objective.state`` gets the same ``RemoteDeviceUnreachable`` as one whose
    server went away later, not an ``AttributeError`` that reads like a missing
    feature. A local part is always online.
    """
    if not getattr(device, "online", True):
        from fibsem.devices.drivers.remote import RemoteDeviceUnreachable

        raise RemoteDeviceUnreachable(
            f"{device.name} at {device.client.base_url} has not connected yet"
        )
    return getattr(device, name)


class DeviceObjectiveLens(ObjectiveLens):
    def __init__(self, device: Device, parent: Optional[FluorescenceMicroscope] = None):
        super().__init__(parent=parent)
        self._device = device
        self._focus_position = None  # a session setting, kept here

    @property
    def magnification(self) -> float:
        return _param(self._device, "magnification").get_value()

    @property
    def numerical_aperture(self) -> float:
        return _param(self._device, "numerical_aperture").get_value()

    @property
    def position(self) -> float:
        return _param(self._device, "position").get_value()

    @property
    def limit_position(self) -> float:
        return _param(self._device, "limit_position").get_value()

    @limit_position.setter
    def limit_position(self, position: float) -> None:
        _param(self._device, "limit_position").write_through(position)

    @property
    def limits(self) -> Tuple[float, float]:
        limits = _param(self._device, "position").limits
        return (limits.min, limits.max)

    @property
    def state(self) -> str:
        return objective_state_name(_param(self._device, "state").get_value())

    def move_relative(self, delta: float) -> None:
        self._device.move_relative(delta)
        self._notify_moved()

    def move_absolute(self, position: float) -> None:
        self._device.move_absolute(position)
        self._notify_moved()

    def insert(self) -> None:
        self._device.insert()
        self._notify_moved()

    def retract(self) -> None:
        self._device.retract()
        self._notify_moved()


class DeviceCamera(Camera):
    def __init__(self, device: Device, parent: Optional[FluorescenceMicroscope] = None):
        super().__init__(parent=parent)
        self._device = device

    def acquire_image(self) -> np.ndarray:
        return self._device.acquire()

    @property
    def exposure_time(self) -> float:
        return _param(self._device, "exposure_time").get_value()

    @exposure_time.setter
    def exposure_time(self, value: float) -> None:
        _param(self._device, "exposure_time").write_through(value)

    @property
    def binning(self) -> int:
        return _param(self._device, "binning").get_value()

    @binning.setter
    def binning(self, value: int) -> None:
        _param(self._device, "binning").write_through(value)

    @property
    def available_binnings(self) -> Tuple[int, ...]:
        return tuple(_param(self._device, "binning").choices or ())

    @property
    def exposure_time_limits(self) -> Tuple[float, float]:
        limits = _param(self._device, "exposure_time").limits
        return (limits.min, limits.max)

    @property
    def gain(self) -> float:
        return _param(self._device, "gain").get_value()

    @gain.setter
    def gain(self, value: float) -> None:
        _param(self._device, "gain").write_through(value)

    @property
    def offset(self) -> float:
        return _param(self._device, "offset").get_value()

    @offset.setter
    def offset(self, value: float) -> None:
        _param(self._device, "offset").write_through(value)

    @property
    def pixel_size(self) -> Tuple[float, float]:
        return tuple(_param(self._device, "pixel_size").get_value())

    @property
    def resolution(self) -> Tuple[int, int]:
        return tuple(_param(self._device, "resolution").get_value())


class DeviceLightSource(LightSource):
    def __init__(self, device: Device, parent: Optional[FluorescenceMicroscope] = None):
        super().__init__(parent=parent)
        self._device = device

    @property
    def power(self) -> float:
        return _param(self._device, "power").get_value()

    @power.setter
    def power(self, value: float) -> None:
        _param(self._device, "power").write_through(value)

    @property
    def power_limits(self) -> Tuple[float, float]:
        limits = _param(self._device, "power").limits
        return (limits.min, limits.max)


def _old_emission_value(found: EmissionFilter) -> Optional[Union[float, str]]:
    if found == REFLECTION:
        return None
    return found.name if found.low is None else found.low


def emission_filter_named(
    value: Optional[Union[float, str]], filters: Sequence[EmissionFilter]
) -> EmissionFilter:
    """The filter among ``filters`` that today's emission value names.

    ``None`` is reflection. A label is the filter of that name, or else the multi-band
    filter, as the FM classes take any label to mean fluorescence. A number is the
    band whose bottom edge is closest, as on Odemis; a filter set with no bands (Thermo,
    the simulator) has only its multi-band filter, which is what Thermo reports a
    number as.
    """
    multi_band = [f for f in filters if f != REFLECTION and f.low is None]
    if value is None:
        matches = [f for f in filters if f == REFLECTION]
    elif isinstance(value, str):
        matches = [f for f in filters if f.name == value] or multi_band
    else:
        banded = [f for f in filters if f.low is not None]
        matches = sorted(banded, key=lambda f: abs(f.low - value))[:1] or multi_band
    if not matches:
        raise ValueError(f"No emission filter for {value!r}")
    return matches[0]


class DeviceFilterSet(FilterSet):
    def __init__(self, device: Device, parent: Optional[FluorescenceMicroscope] = None):
        super().__init__(parent=parent)
        self._device = device

    @property
    def available_excitation_wavelengths(self) -> Tuple[float, ...]:
        return tuple(_param(self._device, "excitation_wavelength").choices or ())

    # Today's API names a filter by one value: None for reflection, a label for a
    # multi-band filter, or the band's bottom edge in nm.

    @property
    def available_emission_wavelengths(self) -> Tuple[Union[None, str, float], ...]:
        return tuple(_old_emission_value(f) for f in self._emission_filters())

    def _emission_filters(self) -> Tuple[EmissionFilter, ...]:
        return tuple(_param(self._device, "emission_filter").choices or ())

    def emission_filter(self, value: Optional[Union[float, str]]) -> EmissionFilter:
        for found in self._emission_filters():
            if same_emission_value(_old_emission_value(found), value):
                return found
        return emission_filter_for(value, {})

    @property
    def excitation_wavelength(self) -> float:
        return _param(self._device, "excitation_wavelength").get_value()

    @excitation_wavelength.setter
    def excitation_wavelength(self, value: float) -> None:
        _param(self._device, "excitation_wavelength").write_through(value)

    @property
    def emission_wavelength(self) -> Optional[Union[float, str]]:
        return _old_emission_value(_param(self._device, "emission_filter").get_value())

    @emission_wavelength.setter
    def emission_wavelength(self, value: Optional[Union[float, str]]) -> None:
        found = emission_filter_named(value, self._emission_filters())
        _param(self._device, "emission_filter").write_through(found)


class DeviceFluorescenceMicroscope(FluorescenceMicroscope):
    """Today's FM API over FM devices, local or remote."""

    objective: DeviceObjectiveLens
    camera: DeviceCamera
    light_source: DeviceLightSource
    filter_set: DeviceFilterSet

    def __init__(
        self,
        devices: Dict[str, Device],
        parent: Optional[FibsemMicroscope] = None,
    ):
        super().__init__(parent=parent)
        self.devices = devices
        self.objective = DeviceObjectiveLens(devices["objective"], parent=self)
        self.camera = DeviceCamera(devices["camera"], parent=self)
        self.light_source = DeviceLightSource(devices["light_source"], parent=self)
        self.filter_set = DeviceFilterSet(devices["filter_set"], parent=self)

    def acquire_image(
        self, channel_settings: Optional[ChannelSettings] = None
    ) -> FluorescenceImage:
        """One command on the ``fm`` group sets up the channel and takes the frame."""
        with self.active_channel():
            if channel_settings is None:
                data = self.devices["camera"].acquire()
            else:
                # The name and colour are this session's labels, not hardware.
                self.channel_name = channel_settings.name
                self.channel_color = channel_settings.color
                data = self.devices["fm"].acquire_channel(channel_settings.to_dict())
            return self._construct_image(data)
